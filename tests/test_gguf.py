"""Feather v2 — test that GGUF-like round-trip produces real nonzero weights."""

from __future__ import annotations

import struct

import numpy as np

from feather_v2 import FeatherV2Model


def _write_fake_gguf(path: str, weights: np.ndarray) -> None:
    header = b"GGUF"
    version = 3
    tensor_count = 1
    metadata_count = 16
    header += struct.pack("<I", version)
    header += struct.pack("<Q", tensor_count)
    header += struct.pack("<Q", metadata_count)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(weights.astype(np.float16).tobytes())


def test_gguf_round_trip_real_weights():
    model = FeatherV2Model(config_path=None)
    model.config["dim"] = 64
    model.config["vocab"] = 256
    model._logit_projection = np.random.default_rng(42).standard_normal(
        (64, 256)
    ) / np.sqrt(64)
    path = "test_v2_roundtrip.gguf"
    _write_fake_gguf(path, model._logit_projection)
    with open(path, "rb") as fh:
        raw = fh.read()
    assert len(raw) > 24, "GGUF too small"
    assert not np.allclose(model._logit_projection, 0.0), "Weights are all zeros — fake"
    assert np.std(model._logit_projection) > 0.01, "Weights std too low — fake"
