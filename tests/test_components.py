"""Feather v2 — component tests."""

from __future__ import annotations

import numpy as np

from feather_v2 import FeatherV2Model


def _small_config() -> dict:
    return {
        "dim": 64,
        "hv_dim": 1024,
        "seq_len": 64,
        "chunk": 16,
        "vocab": 256,
        "tt_rank": 2,
        "moe_experts": 4,
        "seed": 0,
    }


def test_sensory_encoder_returns_dict():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((64, 64))
    out = model.sensory.encode(x)
    assert "hypervector" in out
    assert "signature" in out
    assert "compression_ratio" in out


def test_liquid_memory_sparsity():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((64, 64))
    out = model.memory.forward(x)
    assert "sparsity" in out
    assert out["sparsity"] >= 0.0
    assert out["sparsity"] <= 1.0


def test_hyperdimensional_memory_shape():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((64, 64))
    out = model.hyper.forward(x)
    assert "hypervector" in out
    assert out["hypervector"].shape[0] == model.hyper.hv_dim


def test_knowledge_vault_output_shape():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((64, 64))
    out = model.knowledge.route_and_apply(x, batch_size=1)
    assert out.shape[0] == model.config["dim"]


def test_cognitive_weaver_loops():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((1, 64))
    entropies = np.full(6, 0.62)
    out = model.reasoning.reasoning_loop(x, entropies)
    assert out.shape == x.shape


def test_homeostasis_governor_gate():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((1, 64))
    out = model.governor.entropy_gate(x)
    assert out.shape == x.shape


def test_generative_evolution_shape():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((8, 64))
    out = model.generation.speculative_generate(lambda k, c: x, entropy=0.5)
    assert out.shape[0] == 8


def test_model_encode_runs():
    model = FeatherV2Model(config=_small_config())
    x = np.random.default_rng(0).standard_normal((64, 64))
    out = model.encode(x)
    assert "final_output" in out
    assert out["final_output"].shape[0] >= 1
