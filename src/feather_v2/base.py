"""Feather v2 — optional base helper.

``BaseComponent`` provides small shared helpers: adaptive kernel lookup, energy
accounting, op counting, cache reporting and a cosine-similarity shortcut.

**The shipped components do not inherit from it.** ``SensoryEncoder``,
``LiquidMemory``, ``HyperDimensionalMemory``, ``TTExpert``, ``KnowledgeVault``,
``CognitiveWeaver``, ``HomeostasisGovernor`` and ``GenerativeEvolution`` are all
``torch.nn.Module`` subclasses and get their behaviour from ``nn_math`` and
``nn.Module``. An earlier version of this docstring claimed every component
inherited ``BaseComponent``; none of them do.

It is kept because it is a usable mixin for new non-``nn.Module`` code, and it is
importable, but nothing in the model depends on it.
"""

from __future__ import annotations

from typing import Any

from .hardware import get_best_kernel
from .utils import cos_sim


class BaseComponent:
    """Common behaviour shared by all Feather v2 components."""

    name = "base"

    def __init__(
        self,
        config: Any,
        energy_tracker: Any | None = None,
    ) -> None:
        self.config = config
        self.kernel: dict[str, Any] = get_best_kernel()
        self.energy_tracker = energy_tracker
        self._joules: float = 0.0
        self.total_ops: int = 0
        self.total_multiplies: int = 0

    def encode(self, x: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def record_energy(self, joules: float) -> None:
        self._joules += float(joules)
        if self.energy_tracker is not None:
            try:
                self.energy_tracker.record(self.name, joules)
            except Exception:
                pass

    @property
    def joules(self) -> float:
        return self._joules

    def reset_energy(self) -> None:
        self._joules = 0.0

    def count_ops(self, adds: int = 0, multiplies: int = 0) -> None:
        self.total_ops += int(adds) + int(multiplies)
        self.total_multiplies += int(multiplies)

    @property
    def multiplies(self) -> int:
        return self.total_multiplies

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 0, "l2_kb": 0, "l3_kb": 0}

    @staticmethod
    def similarity(a: Any, b: Any) -> float:
        return cos_sim(a, b)
