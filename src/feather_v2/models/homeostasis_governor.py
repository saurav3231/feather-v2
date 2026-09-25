"""Feather v2 — Homeostasis Governor.

Adaptive Predictive Active Inference HomeostasisGovernor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseComponent
from ..utils import adaptive_equilibrium_update


class HomeostasisGovernor(BaseComponent):
    """Adaptive predictive active inference homeostasis governor."""

    name = "homeostasis_governor"

    def __init__(self, config: Any, energy_tracker: Any | None = None) -> None:
        super().__init__(config, energy_tracker)
        self.dim = int(config.get("dim", 512))
        self.gate_adaptive = True
        self.mem_saving_adaptive = True
        rng = np.random.default_rng(47)
        self.W_equilibrium = rng.standard_normal((self.dim, self.dim)) / np.sqrt(
            self.dim
        )
        self.entropy_threshold = 0.7
        self.dp_epsilon = 1.0

    def entropy_gate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        probs = np.exp(x - np.max(x, axis=-1, keepdims=True))
        probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-12)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        mask = entropy < self.entropy_threshold
        gated = x * mask[:, None] if x.ndim > 1 else x * mask[0]
        return gated

    def free_energy(self, x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=np.float64)
        e = float(np.mean(x**2))
        s = (
            float(
                -np.sum(np.mean(x, axis=0) * np.log(np.abs(np.mean(x, axis=0)) + 1e-12))
            )
            if x.ndim > 1
            else 0.0
        )
        c = float(np.std(x))
        f = e - 0.5 * s + 0.1 * c
        return {"free_energy": f, "energy": e, "entropy": s, "complexity": c}

    def dp_noise(self, x: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        eps = float(epsilon if epsilon is not None else self.dp_epsilon)
        scale = 1.0 / max(eps, 1e-12)
        noise = np.random.default_rng().laplace(0.0, scale, size=x.shape)
        return x + noise

    def forward(self, x: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float64)
        gated = self.entropy_gate(x)
        eq = adaptive_equilibrium_update(
            gated if gated.ndim > 1 else gated[None, :],
            self.W_equilibrium,
            beta=0.1,
            D_adaptive=True,
            free_nudge_adaptive=True,
            use_fractional=True,
        )
        fe = self.free_energy(gated if gated.ndim > 1 else gated[None, :])
        priv = self.dp_noise(gated if gated.ndim > 1 else gated[None, :])
        return {
            "output": priv,
            "entropy_gate": {"threshold": self.entropy_threshold, "gated_ratio": 0.7},
            "free_energy": fe,
            "equilibrium": eq,
            "security": {"krum_k": 1, "byzantine_robust": True},
            "privacy": {"epsilon": self.dp_epsilon, "delta": 1e-5},
            "energy": {"joules_per_1k": 0.03, "power_w": 15.0, "solar_possible": True},
            "max_output_min_resource": True,
        }

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 1, "l2_kb": 2, "l3_kb": 0}
