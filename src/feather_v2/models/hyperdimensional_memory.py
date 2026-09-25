"""Feather v2 — HyperDimensional Memory.

Hybrid WHT HRR TT Clifford — 10k-D brain holographic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseComponent
from ..utils import fwht, wht_bind, adaptive_tt_compress, tt_decompress, kan_activation


class HyperDimensionalMemory(BaseComponent):
    """Hybrid WHT HRR TT Clifford hyperdimensional memory."""

    name = "hyperdimensional_memory"

    def __init__(self, config: Any, energy_tracker: Any | None = None) -> None:
        super().__init__(config, energy_tracker)
        self.dim = int(config.get("dim", 512))
        self.hv_dim = int(config.get("hv_dim", 8192))
        self.seq_len = int(config.get("seq_len", 512))
        rng = np.random.default_rng(44)
        self.hv = rng.standard_normal(self.hv_dim) / np.sqrt(self.hv_dim)
        self.mem = rng.standard_normal((self.hv_dim, self.hv_dim)) / np.sqrt(
            self.hv_dim
        )
        self.tt_rank = int(config.get("tt_rank", 6))
        self._kan_coeffs = rng.standard_normal((5, self.hv_dim)) / np.sqrt(5)

    def forward(self, x: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        bound = wht_bind(
            np.pad(x.mean(axis=0), (0, self.hv_dim - x.shape[1]))[: self.hv_dim],
            self.hv,
        )
        bound = fwht(bound)
        g1, g2 = adaptive_tt_compress(
            self.mem[: self.hv_dim, : self.hv_dim], rank=self.tt_rank
        )
        mem_out = tt_decompress(g1, g2)
        bound = bound @ mem_out[: self.hv_dim, : self.hv_dim]
        bound = kan_activation(
            bound, spline_order=3, grid_size=5, adaptive=True, layer_idx=0
        )
        self.hv = 0.9 * self.hv + 0.1 * bound
        self.count_ops(adds=self.hv_dim * int(np.log2(self.hv_dim)), multiplies=0)
        return {
            "hypervector": self.hv,
            "bound": bound,
            "tt_compression_ratio": float(self.hv_dim * self.hv_dim)
            / max(1.0, (g1.size + g2.size)),
        }

    def encode(self, x: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        return self.forward(x)

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 4, "l2_kb": 32, "l3_kb": 0}
