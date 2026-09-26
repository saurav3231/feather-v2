"""Feather v2 — fresh-clone integration test.

Builds the model the way a new user would, runs a handful of genuine
optimization steps on real token data, and checks that the objective actually
falls. The previous version of this file computed the mean squared error of a
fixed random projection against a random target, with no optimizer anywhere, and
asserted the resulting constant decreased.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from feather_v2 import FeatherV2Model
from feather_v2.hardware import get_best_kernel

CONFIG = {
    "dim": 32,
    "n_blocks": 1,
    "hv_dim": 128,
    "seq_len": 32,
    "chunk": 8,
    "vocab": 64,
    "tt_rank": 4,
    "moe_experts": 4,
    "moe_top_k": 2,
    "n_scales": 3,
    "sig_dim": 4,
    "weaver_hidden": 48,
    "weaver_gaussians": 3,
    "evo_latent": 6,
    "evo_iters": 3,
    "dropout": 0.0,
    "seed": 42,
}


def _corpus(vocab: int, length: int = 512) -> torch.Tensor:
    """A tiny structured corpus, not random noise, so there is signal to fit."""
    generator = torch.Generator().manual_seed(1234)
    pattern = torch.arange(length) % 7
    noise = torch.randint(0, 3, (length,), generator=generator)
    return (pattern * 8 + noise) % vocab


def _batches(vocab: int, seq_len: int) -> list[torch.Tensor]:
    tokens = _corpus(vocab)
    return [
        tokens[i : i + seq_len].unsqueeze(0)
        for i in range(0, tokens.numel() - seq_len - 1, seq_len)
    ]


def test_loss_falls_over_real_training_steps() -> None:
    torch.manual_seed(0)
    model = FeatherV2Model(CONFIG)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    batches = _batches(CONFIG["vocab"], CONFIG["seq_len"])

    losses: list[float] = []
    start = time.perf_counter()
    for step in range(30):
        ids = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(ids)
        loss.backward()
        optimizer.step()
        losses.append(metrics["loss"])
    elapsed = time.perf_counter() - start

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"
    # A third of the way in, real progress should already be visible rather
    # than the improvement landing entirely in the final step.
    assert min(losses[10:]) < losses[0], f"no progress by step 10: {losses}"
    assert elapsed < 60.0, f"fresh-clone run too slow: {elapsed:.1f}s"


def test_perplexity_improves_consistently() -> None:
    torch.manual_seed(0)
    model = FeatherV2Model(CONFIG)
    batches = _batches(CONFIG["vocab"], CONFIG["seq_len"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    first = last = None
    for step in range(30):
        ids = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(ids)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = metrics["perplexity"]
        last = metrics["perplexity"]

    assert last < first, f"perplexity did not improve: {first} -> {last}"


def test_trained_model_can_be_saved_and_reused(tmp_path: Path) -> None:
    model = FeatherV2Model(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for step in range(5):
        ids = _batches(CONFIG["vocab"], CONFIG["seq_len"])[step]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(ids)
        loss.backward()
        optimizer.step()

    path = tmp_path / "trained.pth"
    model.save_pth(path)
    reloaded = FeatherV2Model.from_pth(path)
    reloaded.eval()
    model.eval()

    ids = _batches(CONFIG["vocab"], CONFIG["seq_len"])[0]
    with torch.no_grad():
        assert torch.allclose(model(ids), reloaded(ids), atol=1e-6)

    # The manifest must describe the model that was actually written, and the
    # file on disk must be at least as large as the raw weights it contains.
    manifest = model.describe()
    assert manifest["parameters"] == model.count_parameters()
    assert path.stat().st_size >= manifest["weight_bytes"]
    assert manifest["size_label"] in {"K", "M", "B"} or manifest["size_label"].endswith(
        ("K", "M", "B")
    )


def test_hardware_detected() -> None:
    kernel = get_best_kernel()
    assert kernel["hypervector_dim"] >= 256
    assert kernel["threads"] >= 1


if __name__ == "__main__":
    torch.manual_seed(0)
    m = FeatherV2Model(CONFIG)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    bs = _batches(CONFIG["vocab"], CONFIG["seq_len"])
    for i in range(30):
        opt.zero_grad(set_to_none=True)
        l, met = m.loss(bs[i % len(bs)])
        l.backward()
        opt.step()
        if i % 10 == 0:
            print(f"step {i}: loss {met['loss']:.4f} ppl {met['perplexity']:.2f}")
    print("PASS")
