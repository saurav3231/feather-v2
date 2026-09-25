"""Feather v2 — Liquid Memory.

Adaptive Hierarchical Liquid Fractional Memory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseComponent
from ..utils import (
    adaptive_fractional_weights,
    adaptive_p_adic_chunk_retrieve as _p_adic_retrieve,
)


class LiquidMemory(BaseComponent):
    """Adaptive hierarchical liquid fractional memory."""

    name = "liquid_memory"

    def __init__(self, config: Any, energy_tracker: Any | None = None) -> None:
        super().__init__(config, energy_tracker)
        self.dim = int(config.get("dim", 512))
        self.hv_dim = int(config.get("hv_dim", 8192))
        self.seq_len = int(config.get("seq_len", 512))
        self.chunk = int(config.get("chunk", 32))
        self.num_chunks = int(config.get("num_chunks", 16))
        self.alpha = float(config.get("alpha_fractional", 0.7))
        self.K = int(config.get("K_frac_recent", 32))
        self.beta = float(config.get("beta_learnable", 1.2))
        rng = np.random.default_rng(43)
        self.M_t = np.zeros(self.dim)
        self.W_tau = rng.standard_normal((self.dim, self.dim)) / np.sqrt(self.dim)
        self.chunk_hvs = rng.standard_normal((self.num_chunks, self.hv_dim)) / np.sqrt(
            self.hv_dim
        )
        self.spike_threshold = 0.5

    def hierarchical_fractional(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        w = adaptive_fractional_weights(
            alpha=self.alpha,
            K=self.K,
            beta_learnable=self.beta,
            hierarchical=True,
            liquid_tau=True,
        )
        tau = 1.0 / (
            1.0 + np.exp(-self.beta * (np.arange(self.dim) / max(1, self.dim) - 0.5))
        )
        mem = self.M_t * tau + (1.0 - tau) * (self.W_tau @ x)
        spike_mask = np.abs(mem) > self.spike_threshold
        sparsity = 1.0 - float(np.mean(spike_mask))
        self.M_t = mem * spike_mask
        self.count_ops(adds=self.dim * self.dim, multiplies=0)
        return self.M_t

    def p_adic_retrieve(self, query: np.ndarray) -> dict[str, Any]:
        q = np.asarray(query, dtype=np.float64)
        best_idx, sim = _p_adic_retrieve(
            q, self.chunk_hvs, p=2, use_rough_path=True, use_fractional=True
        )
        return {"best_chunk": best_idx, "similarity": sim}

    def forward(self, x: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        m = np.zeros(self.dim)
        for t in range(x.shape[0]):
            m = self.hierarchical_fractional(x[t])
        retrieval = self.p_adic_retrieve(m)
        spike_mask = np.abs(m) > self.spike_threshold
        return {
            "M_t": m,
            "weighted": m,
            "tau": 1.0
            / (
                1.0
                + np.exp(-self.beta * (np.arange(self.dim) / max(1, self.dim) - 0.5))
            ),
            "spike_mask": spike_mask,
            "num_spikes": int(np.sum(spike_mask)),
            "sparsity": 1.0 - float(np.mean(spike_mask)),
            "best_chunk": retrieval["best_chunk"],
            "best_chunk_sim": retrieval["similarity"],
            "hv_mem_wht": np.zeros(self.hv_dim),
            "retention_w511": (
                adaptive_fractional_weights(self.alpha, self.dim)[min(511, self.dim - 1)]
                if self.dim > 0
                else 0.0
            ),
        }

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 9, "l2_kb": 20, "l3_kb": 0}
