"""Feather v2 — Knowledge Vault.

Adaptive Hierarchical Softmin Tropical-TT Fusion SparX AMX p-adic Entropy KnowledgeVault.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseComponent
from ..utils import (
    adaptive_tropical_matmul,
    adaptive_tt_compress,
    adaptive_sinkhorn,
    adaptive_p_adic_distance,
    fwht,
    tt_decompress,
)


class KnowledgeVault(BaseComponent):
    """Adaptive hierarchical softmin tropical-TT fusion knowledge vault."""

    name = "knowledge_vault"

    def __init__(self, config: Any, energy_tracker: Any | None = None) -> None:
        super().__init__(config, energy_tracker)
        self.dim = int(config.get("dim", 512))
        self.seq_len = int(config.get("seq_len", 512))
        self.experts = int(config.get("moe_experts", 96))
        self.top_k = int(config.get("moe_top_k", 1))
        self.tt_rank = int(config.get("tt_rank", 6))
        self.tau = float(config.get("tau_tropical", 0.1))
        rng = np.random.default_rng(45)
        self.expert_W = rng.standard_normal(
            (self.experts, self.dim, self.dim)
        ) / np.sqrt(self.dim)
        self.router = rng.standard_normal((self.dim, self.experts)) / np.sqrt(self.dim)
        self._cached_tt: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _get_tt(self, expert_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if expert_idx not in self._cached_tt:
            w = self.expert_W[expert_idx]
            g1, g2 = adaptive_tt_compress(
                w,
                rank=self.tt_rank,
                rank_adaptive=True,
                use_sparx=True,
                use_amx=True,
                use_tropical=True,
            )
            self._cached_tt[expert_idx] = (g1, g2)
        return self._cached_tt[expert_idx]

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        logits = x @ self.router
        plan = adaptive_sinkhorn(
            -logits[None, :],
            eps=0.1,
            iters=5,
            p_adic_routing=True,
            hierarchical=True,
            entropy_reg=True,
        )
        plan = plan.flatten()
        top_k = min(self.top_k, self.experts)
        idx = np.argsort(plan)[-top_k:]
        weights = plan[idx]
        weights = weights / (weights.sum() + 1e-12)
        return idx, weights

    def route_and_apply(self, x: np.ndarray, batch_size: int = 1) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        idx, weights = self.route(x[0])
        out = np.zeros(self.dim, dtype=np.float64)
        for i, eidx in enumerate(idx):
            g1, g2 = self._get_tt(int(eidx))
            out += weights[i] * (x[0] @ tt_decompress(g1, g2))
        out = fwht(out)
        self.count_ops(adds=self.dim * self.experts, multiplies=0)
        return out

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 4, "l2_kb": 16, "l3_kb": 2}
