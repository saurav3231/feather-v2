"""Feather v2 — model integration tests."""

from __future__ import annotations

import time

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
        "seed": 42,
    }


def test_model_forward_loss_trend():
    model = FeatherV2Model(config=_small_config())
    rng = np.random.default_rng(0)
    losses = []
    start = time.perf_counter()
    for step in range(5):
        x = rng.standard_normal((64, 64))
        out = model.forward(x)
        logits = (
            np.asarray(out["final_output"], dtype=np.float64).flatten()
            @ model._logit_projection
        )
        target = int(rng.integers(0, 256))
        target_vec = np.eye(256)[target]
        loss = float(np.mean((logits - target_vec) ** 2))
        losses.append(loss)
    elapsed = time.perf_counter() - start
    assert losses[0] > losses[-1]
    assert elapsed < 60.0


def test_model_weights_nonzero():
    model = FeatherV2Model(config=_small_config())
    assert not np.allclose(model._logit_projection, 0.0)
    assert np.std(model._logit_projection) > 0.01


def test_model_hardware_summary():
    model = FeatherV2Model(config=_small_config())
    s = model.hardware_summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_model_generate_length():
    model = FeatherV2Model(config=_small_config())
    prompt = np.zeros((1, 64))
    out = model.generate(prompt, steps=4)
    assert out.shape[0] >= 1


def test_model_save_load_weights(tmp_path):
    model = FeatherV2Model(config=_small_config())
    p = tmp_path / "weights.npz"
    model.save_weights(str(p))
    loaded = FeatherV2Model.from_weights(str(p))
    assert loaded._logit_projection.shape == model._logit_projection.shape
    np.testing.assert_allclose(loaded._logit_projection, model._logit_projection)
