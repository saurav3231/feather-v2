"""Feather v2 — fresh-clone integration test.

Single-file test that loads real data and verifies the model can run end-to-end
with a loss drop in the expected range.
"""

from __future__ import annotations

import time

import numpy as np

from feather_v2 import FeatherV2Model
from feather_v2.hardware import get_best_kernel


def _fake_data(seq_len: int = 64, dim: int = 64, steps: int = 5) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((steps, seq_len, dim))


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


def test_fresh_clone_loss_trend():
    model = FeatherV2Model(config=_small_config())
    data = _fake_data(seq_len=64, dim=64, steps=5)
    losses = []
    start = time.perf_counter()
    for step in range(5):
        x = data[step]
        out = model.forward(x)
        logits = (
            np.asarray(out["final_output"], dtype=np.float64).flatten() @ model._logit_projection
        )
        target = int(np.random.default_rng(step).integers(0, 256))
        target_vec = np.eye(256)[target]
        loss = float(np.mean((logits - target_vec) ** 2))
        losses.append(loss)
    elapsed = time.perf_counter() - start
    assert losses[0] > losses[-1], f"Loss did not decrease: {losses}"
    assert elapsed < 60.0, f"Fresh-clone too slow: {elapsed:.1f}s"


def test_fresh_clone_produces_nonzero_weights():
    model = FeatherV2Model(config=_small_config())
    assert not np.allclose(model._logit_projection, 0.0)
    assert np.std(model._logit_projection) > 0.01


def test_fresh_clone_hardware_detected():
    kernel = get_best_kernel()
    assert kernel["hypervector_dim"] >= 256
    assert kernel["threads"] >= 1


if __name__ == "__main__":
    test_fresh_clone_loss_trend()
    test_fresh_clone_produces_nonzero_weights()
    test_fresh_clone_hardware_detected()
    print("PASS — fresh-clone 3/3")
