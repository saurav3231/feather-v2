"""Feather v2 — export integrity tests.

The previous version of this file wrote a header of ``b"GGUF"`` followed by raw
float32 bytes from a random projection matrix, and then asserted that the result
was "not fake". Those two 524,312-byte files shipped in ``release/v1.0.0`` as
``feather-v2-40M-Q4_K_M.gguf`` and ``feather-v2-40M-f16.gguf``.

There is no GGUF exporter in this project. Rather than keep a test that
manufactures a file to then congratulates itself on, this module tests the
export path that actually exists -- the real PyTorch checkpoint -- and fails if
a mislabeled GGUF artifact reappears.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from feather_v2 import FeatherV2Model

REPO = Path(__file__).resolve().parents[1]

CONFIG = {
    "dim": 32,
    "n_blocks": 1,
    "hv_dim": 128,
    "seq_len": 16,
    "chunk": 4,
    "vocab": 64,
    "tt_rank": 4,
    "moe_experts": 2,
    "moe_top_k": 1,
    "n_scales": 2,
    "sig_dim": 2,
    "weaver_hidden": 32,
    "weaver_gaussians": 3,
    "evo_latent": 4,
    "evo_iters": 2,
    "dropout": 0.0,
    "seed": 7,
}


def _ids() -> torch.Tensor:
    return torch.randint(0, CONFIG["vocab"], (1, 8))


def test_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    model = FeatherV2Model(CONFIG)
    model.eval()
    path = tmp_path / "model.pth"
    model.save_pth(path)
    loaded = FeatherV2Model.from_pth(path)
    loaded.eval()

    ids = _ids()
    with torch.no_grad():
        assert torch.equal(model(ids), loaded(ids))

    for (name, a), (_, b) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        assert torch.equal(a, b), name


def test_checkpoint_preserves_config(tmp_path: Path) -> None:
    model = FeatherV2Model(CONFIG)
    path = tmp_path / "model.pth"
    model.save_pth(path)
    loaded = FeatherV2Model.from_pth(path)
    assert loaded.config == model.config


def test_no_mislabeled_gguf_artifacts_are_shipped() -> None:
    """A ``.gguf`` file may only exist if a real exporter ships with it.

    There is no ggml-compatible writer or quantizer in this project, so any
    ``.gguf`` currently in the tree is, by construction, fabricated.
    """
    import feather_v2.nn_components as components
    import feather_v2.nn_math as math_module
    import feather_v2.model as model_module

    has_exporter = any(
        hasattr(module, attr)
        for module in (components, math_module, model_module)
        for attr in ("save_gguf", "export_gguf", "write_gguf")
    )
    stray = list(REPO.rglob("*.gguf"))
    if not has_exporter:
        assert (
            not stray
        ), f"found GGUF artifacts with no exporter to produce them: {stray}"


def test_saved_size_matches_reported_parameter_count(tmp_path: Path) -> None:
    """A checkpoint's real byte size must agree with its real parameter count.

    This is the check that the fabricated 40M/"80 MB" pairing could never
    pass, since the shipped file was a fraction of a megabyte.
    """
    model = FeatherV2Model(CONFIG)
    path = tmp_path / "model.pth"
    model.save_pth(path)

    parameters = model.count_parameters()
    actual_bytes = path.stat().st_size
    f16_lower_bound = parameters * 2

    assert actual_bytes >= f16_lower_bound, (
        f"checkpoint of {parameters} parameters is only {actual_bytes} bytes, "
        f"below the {f16_lower_bound}-byte f16 floor"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_saved_dtype_round_trips(tmp_path: Path, dtype: torch.dtype) -> None:
    model = FeatherV2Model(CONFIG).to(dtype)
    model.eval()
    path = tmp_path / f"model-{dtype}.pth"
    model.save_pth(path)
    loaded = FeatherV2Model.from_pth(path).to(dtype)
    loaded.eval()
    ids = _ids()
    with torch.no_grad():
        assert torch.allclose(model(ids), loaded(ids))
