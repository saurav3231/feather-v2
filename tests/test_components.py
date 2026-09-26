"""Feather v2 — tests for the NumPy reference components.

The trainable PyTorch components are covered by ``test_trainable.py``. These
exercises the NumPy implementations that remain in ``feather_v2.models`` as a
cross-check reference for the inference path. They are instantiated directly
rather than through ``FeatherV2Model`` so that a change to the model cannot
silently stop exercising them.
"""

from __future__ import annotations

import numpy as np

from feather_v2.models.cognitive_weaver import CognitiveWeaver
from feather_v2.models.generative_evolution import GenerativeEvolution
from feather_v2.models.homeostasis_governor import HomeostasisGovernor
from feather_v2.models.hyperdimensional_memory import HyperDimensionalMemory
from feather_v2.models.knowledge_vault import KnowledgeVault
from feather_v2.models.liquid_memory import LiquidMemory
from feather_v2.models.sensory_encoder import SensoryEncoder

CONFIG = {
    "dim": 64,
    "hv_dim": 1024,
    "seq_len": 64,
    "chunk": 16,
    "vocab": 256,
    "tt_rank": 2,
    "moe_experts": 4,
    "seed": 0,
}


def _x(shape: tuple[int, ...] = (64, 64)) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(shape)


def test_sensory_encoder_returns_descriptor() -> None:
    out = SensoryEncoder(CONFIG).encode(_x())
    assert "hypervector" in out
    assert "signature" in out
    assert "compression_ratio" in out
    assert out["compression_ratio"] > 0.0


def test_liquid_memory_reports_sparsity() -> None:
    out = LiquidMemory(CONFIG).forward(_x())
    assert 0.0 <= out["sparsity"] <= 1.0


def test_hyperdimensional_memory_shape() -> None:
    module = HyperDimensionalMemory(CONFIG)
    out = module.forward(_x())
    assert out["hypervector"].shape[0] == CONFIG["hv_dim"]


def test_knowledge_vault_output_shape() -> None:
    module = KnowledgeVault(CONFIG)
    out = module.route_and_apply(_x(), batch_size=1)
    assert out.shape[0] == CONFIG["dim"]


def test_cognitive_weaver_preserves_shape() -> None:
    module = CognitiveWeaver(CONFIG)
    x = _x((1, 64))
    out = module.reasoning_loop(x, np.full(6, 0.62))
    assert out.shape == x.shape


def test_homeostasis_governor_preserves_shape() -> None:
    module = HomeostasisGovernor(CONFIG)
    x = _x((1, 64))
    out = module.entropy_gate(x)
    assert out.shape == x.shape


def test_generative_evolution_preserves_shape() -> None:
    module = GenerativeEvolution(CONFIG)
    x = _x((8, 64))
    out = module.speculative_generate(lambda k, c: x, entropy=0.5)
    assert out.shape[0] == 8


def test_components_are_finite() -> None:
    """No reference component may emit NaN or inf on well-formed input."""
    x = _x((1, 64))
    outputs = [
        SensoryEncoder(CONFIG).encode(x)["hypervector"],
        LiquidMemory(CONFIG).forward(x).get("output", x),
        HyperDimensionalMemory(CONFIG).forward(x)["hypervector"],
        KnowledgeVault(CONFIG).route_and_apply(x, batch_size=1),
        CognitiveWeaver(CONFIG).reasoning_loop(x, np.full(6, 0.62)),
        HomeostasisGovernor(CONFIG).entropy_gate(x),
        GenerativeEvolution(CONFIG).speculative_generate(lambda k, c: x, 0.5),
    ]
    for value in outputs:
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), value
