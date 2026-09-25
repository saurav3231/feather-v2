"""Feather v2 — Sensory Encoder.

Adaptive Multi-Scale Fractional p-adic Rough Path Encoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseComponent
from ..utils import (
    adaptive_rough_path_signature,
    adaptive_fractional_weights,
    fwht,
    wht_bind,
)


class SensoryEncoder(BaseComponent):
    """Adaptive multi-scale fractional p-adic rough path encoder."""

    name = "sensory_encoder"

    def __init__(self, config: Any, energy_tracker: Any | None = None) -> None:
        super().__init__(config, energy_tracker)
        self.dim = int(config.get("dim", 512))
        self.hv_dim = int(config.get("hv_dim", 8192))
        self.seq_len = int(config.get("seq_len", 512))
        self.chunk = int(config.get("chunk", 32))
        self.num_chunks = int(config.get("num_chunks", 16))
        self.alpha = float(config.get("alpha_fractional", 0.7))
        self.K = int(config.get("K_frac_recent", 32))
        rng = np.random.default_rng(42)
        self.random_proj = rng.standard_normal((self.dim, 3)) / np.sqrt(self.dim)
        self.learned_proj = rng.standard_normal((self.dim, 8)) / np.sqrt(self.dim)
        self.hv_dim_pow2 = 1 << (self.hv_dim - 1).bit_length()

    def encode(self, x: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        seq_len, dim = x.shape
        projected = x @ self.random_proj
        sig = adaptive_rough_path_signature(
            projected,
            level=2,
            vals_adaptive=True,
            path_adaptive=True,
            use_fractional=True,
            use_p_adic=True,
        )
        pooled = x.mean(axis=0) if seq_len > 0 else np.zeros(self.dim)
        encoded = pooled @ self.learned_proj
        hv = np.zeros(self.hv_dim_pow2, dtype=np.float64)
        hv[: self.hv_dim] = wht_bind(
            np.pad(encoded, (0, self.hv_dim - encoded.shape[0])),
            np.random.default_rng(0).standard_normal(self.hv_dim_pow2),
        )[: self.hv_dim]
        compression_ratio = float(seq_len * dim) / max(1.0, sig.shape[0])
        self.count_ops(adds=seq_len * dim * 3, multiplies=0)
        return {
            "hypervector": hv,
            "signature": sig,
            "pooled": pooled,
            "compression_ratio": compression_ratio,
        }

    def cache_report(self) -> dict[str, int]:
        return {
            "l1_kb": 1,
            "l2_kb": self.hv_memory_kb(),
            "l3_kb": 0,
        }

    def hv_memory_kb(self) -> int:
        return int(self.hv_dim * 4 / 1024)
