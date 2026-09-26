"""Feather v2 — model integration tests against the real trainable model.

These previously exercised a NumPy scaffold whose "loss" was the mean squared
error of a fixed random projection against a random target, with no optimizer
anywhere in the loop. The quantity was constant, so asserting it decreased was
asserting that the RNG was cooperative. Everything below now measures the real
model with a real optimizer and a real cross-entropy objective.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from feather_v2 import FeatherV2Model

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


@pytest.fixture(scope="module")
def model() -> FeatherV2Model:
    return FeatherV2Model(CONFIG)


def _ids(batch: int = 2, time: int = 8) -> torch.Tensor:
    return torch.randint(0, CONFIG["vocab"], (batch, time))


def test_forward_returns_real_logits(model: FeatherV2Model) -> None:
    logits = model(_ids())
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 8, CONFIG["vocab"])
    assert torch.isfinite(logits).all()


def test_optimizer_step_reduces_loss_on_repeated_batch() -> None:
    """The real test the old suite only pretended to run."""
    torch.manual_seed(0)
    net = FeatherV2Model(CONFIG)
    net.train()
    ids = _ids(batch=1, time=8)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)

    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = net.loss(ids)
        loss.backward()
        optimizer.step()
        losses.append(metrics["loss"])

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert all(np.isfinite(losses))


def test_parameters_change_after_a_step(model: FeatherV2Model) -> None:
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model.loss(_ids())
    loss.backward()
    optimizer.step()
    changed = [
        name
        for name, param in model.named_parameters()
        if not torch.equal(param.detach(), before[name])
    ]
    assert changed, "no parameter moved during an optimizer step"


def test_weights_are_not_all_zero(model: FeatherV2Model) -> None:
    total = model.count_parameters()
    assert total > 0
    assert any(
        float(p.detach().abs().sum()) > 0 for p in model.parameters()
    ), "every parameter is exactly zero"


def test_hardware_summary(model: FeatherV2Model) -> None:
    summary = model.hardware_summary()
    assert isinstance(summary, str)
    assert summary.strip()


def test_generate_extends_prompt(model: FeatherV2Model) -> None:
    model.eval()
    prompt = _ids(batch=1, time=4)
    out = model.generate(prompt, max_new_tokens=6, greedy=True)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)
    model.train()


def test_checkpoint_round_trip_preserves_weights(
    model: FeatherV2Model, tmp_path
) -> None:
    path = tmp_path / "weights.pth"
    model.save_pth(path)
    loaded = FeatherV2Model.from_pth(path)
    model.eval()
    loaded.eval()
    ids = _ids()
    with torch.no_grad():
        assert torch.allclose(model(ids), loaded(ids), atol=1e-6)
