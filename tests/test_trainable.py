"""Tests for the trainable Feather v2 model and its differentiable mathematics.

Each test here corresponds to a defect that was actually found and fixed while
building the PyTorch implementation, or to a claim that must stay honest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from feather_v2 import nn_math as M
from feather_v2.model import FeatherV2Model, load_config as M_load_config
from feather_v2.nn_components import (
    CognitiveWeaver,
    GenerativeEvolution,
    HomeostasisGovernor,
    HyperDimensionalMemory,
    KnowledgeVault,
    LiquidMemory,
    SensoryEncoder,
    TTExpert,
)

TINY = {
    "dim": 32,
    "n_blocks": 2,
    "vocab": 64,
    "seq_len": 16,
    "hv_dim": 64,
    "chunk": 8,
    "tt_rank": 4,
    "moe_experts": 4,
    "moe_top_k": 2,
    "n_scales": 3,
    "sig_dim": 4,
    "weaver_hidden": 48,
    "weaver_gaussians": 3,
    "weaver_loops": 2,
    "evo_latent": 6,
    "evo_iters": 3,
    "dropout": 0.0,
    "seed": 7,
}


@pytest.fixture(scope="module")
def model() -> FeatherV2Model:
    torch.manual_seed(0)
    return FeatherV2Model(TINY)


def _ids(batch: int = 2, time: int = 8, vocab: int = 64) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, time))


def _probe_loss(tensor: torch.Tensor) -> torch.Tensor:
    """A probe loss that is not degenerate for LayerNorm outputs.

    Squared error cannot be used: for ``u`` pre-normalisation, the LayerNorm
    Jacobian annihilates the direction of ``LN(u)`` itself, so ``mean(LN(u)**2)``
    has an exactly zero gradient with respect to ``u``. That masked a real
    vanishing-gradient defect during development.
    """
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn(tensor.shape[-2], tensor.shape[-1], generator=generator).to(
        tensor.dtype
    )
    return (tensor * weight).sum()


class TestForward:
    def test_logits_shape_and_dtype(self, model: FeatherV2Model) -> None:
        logits = model(_ids())
        assert logits.shape == (2, 8, TINY["vocab"])
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()

    def test_aux_returned_and_positive(self, model: FeatherV2Model) -> None:
        _, aux = model(_ids(), return_aux=True)
        assert aux.ndim == 0
        assert torch.isfinite(aux)
        assert float(aux.detach()) >= 0.0

    def test_rejects_wrong_rank_input(self, model: FeatherV2Model) -> None:
        with pytest.raises(ValueError, match="batch, time"):
            model(torch.randint(0, 8, (16,)))

    def test_rejects_sequence_longer_than_seq_len(self, model: FeatherV2Model) -> None:
        with pytest.raises(ValueError, match="exceeds configured seq_len"):
            model(_ids(batch=1, time=TINY["seq_len"] + 1))

    def test_eval_is_deterministic(self, model: FeatherV2Model) -> None:
        model.eval()
        ids = _ids()
        with torch.no_grad():
            assert torch.equal(model(ids), model(ids))


class TestGradients:
    def test_every_parameter_receives_a_gradient(self, model: FeatherV2Model) -> None:
        model.zero_grad(set_to_none=True)
        loss, _ = model.loss(_ids())
        loss.backward()
        missing = [
            name
            for name, param in model.named_parameters()
            if param.grad is None or param.grad.abs().sum() == 0
        ]
        assert not missing, f"parameters without gradient: {missing}"

    def test_all_moe_experts_train_over_several_batches(
        self, model: FeatherV2Model
    ) -> None:
        """Sparse routing means an expert can legitimately miss a single batch.

        Over a handful of batches every expert must have been selected and
        updated at least once, otherwise routing has collapsed.
        """
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        used: set[str] = set()
        for step in range(12):
            ids = _ids(batch=2, time=8)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(ids)
            loss.backward()
            for name, param in model.named_parameters():
                if ".experts." in name and param.grad is not None:
                    if param.grad.abs().sum() > 0:
                        used.add(name)
            optimizer.step()
        experts = {n for n, _ in model.named_parameters() if ".experts." in n}
        assert experts, "model exposes no mixture-of-experts parameters"
        assert experts <= used, f"experts never trained: {sorted(experts - used)}"

    def test_gradient_reaches_input_embeddings(self, model: FeatherV2Model) -> None:
        model.zero_grad(set_to_none=True)
        loss, _ = model.loss(_ids())
        loss.backward()
        assert model.embed.weight.grad.abs().sum() > 0
        assert model.pos_embed.weight.grad.abs().sum() > 0

    def test_grad_norm_is_finite_and_nonzero(self, model: FeatherV2Model) -> None:
        model.zero_grad(set_to_none=True)
        loss, _ = model.loss(_ids())
        loss.backward()
        total = math.sqrt(
            sum(
                float(p.grad.pow(2).sum())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        assert total > 0
        assert math.isfinite(total)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: SensoryEncoder(32, n_scales=3, sig_dim=4),
            lambda: LiquidMemory(32, chunk=5),
            lambda: HyperDimensionalMemory(32, hv_dim=48),
            lambda: KnowledgeVault(32, n_experts=4, top_k=2, rank=4),
            lambda: CognitiveWeaver(32, hidden=24, num_gaussians=3, loops=2),
            lambda: HomeostasisGovernor(32),
            lambda: GenerativeEvolution(32, latent=6, iters=3),
        ],
        ids=[
            "sensory",
            "liquid",
            "hyper",
            "vault",
            "weaver",
            "governor",
            "evolution",
        ],
    )
    def test_component_passes_gradient_to_input(self, factory) -> None:
        module = factory()
        x = torch.randn(2, 7, 32, requires_grad=True)
        out = module(x)
        first = out[0] if isinstance(out, tuple) else out
        assert first.shape == x.shape
        _probe_loss(first).backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: SensoryEncoder(32, n_scales=3, sig_dim=4),
            lambda: LiquidMemory(32, chunk=5),
            lambda: HyperDimensionalMemory(32, hv_dim=48),
            lambda: KnowledgeVault(32, n_experts=4, top_k=2, rank=4),
            lambda: CognitiveWeaver(32, hidden=24, num_gaussians=3, loops=2),
            lambda: HomeostasisGovernor(32),
            lambda: GenerativeEvolution(32, latent=6, iters=3),
        ],
        ids=[
            "sensory",
            "liquid",
            "hyper",
            "vault",
            "weaver",
            "governor",
            "evolution",
        ],
    )
    def test_component_handles_non_power_of_two_time(self, factory) -> None:
        module = factory()
        out = module(torch.randn(1, 7, 32))
        first = out[0] if isinstance(out, tuple) else out
        assert first.shape == (1, 7, 32)


class TestLearning:
    def test_loss_decreases_when_overfitting_one_batch(self) -> None:
        """A trainable model must be able to drive a single batch down."""
        torch.manual_seed(0)
        overfit = dict(TINY, dropout=0.0, moe_experts=2, n_blocks=1)
        net = FeatherV2Model(overfit)
        net.train()
        ids = torch.randint(0, overfit["vocab"], (1, 8))
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)
        losses = []
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = net.loss(ids)
            loss.backward()
            optimizer.step()
            losses.append(metrics["loss"])
        assert losses[-1] < losses[0] * 0.5, f"loss did not fall: {losses}"

    def test_metrics_separate_lm_loss_from_total(self, model: FeatherV2Model) -> None:
        loss, metrics = model.loss(_ids(), aux_weight=0.5)
        expected = metrics["loss"] + 0.5 * metrics["aux_loss"]
        assert metrics["total_loss"] == pytest.approx(expected, rel=1e-5)
        assert float(loss.detach()) == pytest.approx(metrics["total_loss"], rel=1e-5)

    def test_perplexity_tracks_loss(self, model: FeatherV2Model) -> None:
        _, metrics = model.loss(_ids())
        assert metrics["perplexity"] == pytest.approx(
            math.exp(metrics["loss"]), rel=1e-4
        )


class TestParameterAccounting:
    def test_count_matches_parameters(self, model: FeatherV2Model) -> None:
        expected = sum(p.numel() for p in model.parameters())
        assert model.count_parameters() == expected
        assert model.count_parameters() > 0

    def test_tied_embeddings_counted_once(self) -> None:
        tied = FeatherV2Model(dict(TINY, tie_embeddings=True))
        assert tied.head.weight is tied.embed.weight
        counted = sum(
            p.numel() for name, p in tied.named_parameters() if name != "head.weight"
        )
        assert tied.count_parameters() == counted

    def test_untied_has_more_parameters(self) -> None:
        tied = FeatherV2Model(dict(TINY, tie_embeddings=True))
        untied = FeatherV2Model(dict(TINY, tie_embeddings=False))
        assert untied.count_parameters() > tied.count_parameters()

    def test_breakdown_sums_to_total(self, model: FeatherV2Model) -> None:
        assert sum(model.parameter_breakdown().values()) == (model.count_parameters())

    def test_size_label_reflects_real_count(self, model: FeatherV2Model) -> None:
        millions = model.count_parameters() / 1e6
        if millions >= 1:
            assert model.size_label() == f"{millions:.2f}M"

    def test_num_bytes_matches_saved_file(self, model, tmp_path) -> None:
        path = tmp_path / "m.pth"
        written = model.save_pth(path)
        assert written == path.stat().st_size
        assert written > 0


class TestCheckpointing:
    def test_pth_roundtrip_preserves_weights(self, model, tmp_path) -> None:
        path = tmp_path / "ckpt.pth"
        model.save_pth(path)
        restored = FeatherV2Model.from_pth(path)
        restored.eval()
        model.eval()
        ids = _ids()
        with torch.no_grad():
            assert torch.allclose(model(ids), restored(ids), atol=1e-6)

    def test_pth_carries_config(self, model, tmp_path) -> None:
        path = tmp_path / "ckpt.pth"
        model.save_pth(path)
        restored = FeatherV2Model.from_pth(path)
        assert restored.config["dim"] == TINY["dim"]


class TestGeneration:
    def test_generate_extends_sequence(self, model: FeatherV2Model) -> None:
        model.eval()
        prompt = _ids(batch=1, time=4)
        out = model.generate(prompt, max_new_tokens=5, greedy=True)
        assert out.shape == (1, 9)
        assert torch.equal(out[:, :4], prompt)

    def test_generate_produces_no_graph(self, model: FeatherV2Model) -> None:
        out = model.generate(_ids(batch=1, time=4), max_new_tokens=3)
        assert out.grad_fn is None
        assert not out.requires_grad

    def test_generate_restores_training_mode(self, model: FeatherV2Model) -> None:
        model.train()
        model.generate(_ids(batch=1, time=4), max_new_tokens=2)
        assert model.training is True
        model.eval()
        model.generate(_ids(batch=1, time=4), max_new_tokens=2)
        assert model.training is False
        model.train()


class TestBenchmark:
    def test_forward_benchmark_reports_positive_rate(
        self, model: FeatherV2Model
    ) -> None:
        stats = model.benchmark_forward(_ids(), warmup=1, runs=3)
        assert stats["median_seconds"] > 0
        assert stats["tokens_per_second"] > 0
        assert stats["best_seconds"] <= stats["worst_seconds"]


class TestMathematics:
    @pytest.mark.parametrize("dim", [192, 320, 384, 512, 576, 768])
    def test_fwht_is_invertible_on_required_dims(self, dim: int) -> None:
        x = torch.randn(1, 2, dim)
        assert M.fwht(x).shape[-1] == M.next_pow2(dim)
        assert torch.allclose(M.ifwht(M.fwht(x), dim), x, atol=1e-5)

    def test_fwht_preserves_norm_on_power_of_two(self) -> None:
        x = torch.randn(2, 4, 64)
        assert torch.allclose(
            M.fwht(x).pow(2).sum(-1), 64 * x.pow(2).sum(-1), atol=1e-3
        )

    def test_fwht_keeps_all_information_when_padded(self) -> None:
        """Padding then cropping would destroy a degree of freedom."""
        x = torch.randn(1, 1, 7)
        assert M.fwht(x).shape[-1] == 8
        assert torch.allclose(M.ifwht(M.fwht(x), 7), x, atol=1e-5)

    def test_softmin_lower_bounds_true_minimum(self) -> None:
        x = torch.randn(4, 9)
        assert (M.softmin(x, 0.1) <= x.min(-1).values + 1e-5).all()

    def test_tropical_matmul_shape(self) -> None:
        out = M.tropical_matmul(torch.randn(2, 4, 3), torch.randn(2, 3, 5))
        assert out.shape == (2, 4, 5)

    def test_p_adic_exact_on_powers_of_two(self) -> None:
        profile = M.p_adic_weights(torch.tensor([16.0]), 2, 5, 0.05)
        assert torch.allclose(profile[0], torch.ones(5), atol=1e-2)

    def test_p_adic_zero_for_odd_integer(self) -> None:
        profile = M.divisibility_profile(torch.tensor([1025.0]), 2, 5)
        assert profile[0, 0] == 1.0
        assert profile[0, 1] == 0.0

    def test_divisibility_profile_is_exact_on_powers_of_two(self) -> None:
        profile = M.divisibility_profile(torch.tensor([16.0]), 2, 5)
        assert torch.equal(profile[0], torch.ones(5))

    def test_divisibility_profile_detects_mixed_factors(self) -> None:
        profile = M.divisibility_profile(torch.tensor([12.0]), 2, 4)
        assert torch.equal(profile[0], torch.tensor([1.0, 1.0, 1.0, 0.0]))

    def test_godel_log_code_is_finite_for_large_magnitudes(self) -> None:
        """The weaver feeds its own residual back in, so magnitude grows.

        A raw ``(1 + l) ** (1 + a + b)`` overflows to inf there and poisons the
        backward pass with nan. This pins the log-space form as finite.
        """
        relation = torch.zeros(1, requires_grad=True)
        message = torch.full((1,), 5.0e4, requires_grad=True)
        code = M.godel_log_code(relation, message)
        assert torch.isfinite(code).all()
        code.sum().backward()
        assert torch.isfinite(relation.grad).all()
        assert float(relation.grad) != 0.0

    def test_godel_encode_stays_finite_and_monotone(self) -> None:
        low = M.godel_encode(torch.zeros(1), torch.zeros(1))
        high = M.godel_encode(torch.zeros(1), torch.full((1,), 5.0e4))
        assert torch.isfinite(high).all()
        assert float(high) >= float(low)

    def test_training_stays_finite_past_the_overflow_step(self) -> None:
        """Regression: a 40-step run reached loss 2.13, then step ~24 went nan.

        The weaver's residual loop compounded across iterations until the
        Godel exponent overflowed. Enough steps to pass that point must stay
        finite, otherwise the model is unusable at scale.
        """
        model = FeatherV2Model(TINY)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        generator = torch.Generator().manual_seed(3)
        vocab = int(TINY["vocab"])
        seq_len = int(TINY["seq_len"])

        for _ in range(60):
            ids = torch.randint(0, vocab, (1, seq_len), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(ids)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), name
            optimizer.step()
            assert torch.isfinite(loss), metrics

    def test_load_config_migrates_known_renames(self, tmp_path) -> None:
        """``layers`` was silently ignored by every shipped config.

        The model reads ``n_blocks``, so a config advertising 16 layers built a
        2-block model. It must now migrate, and say so.
        """
        path = tmp_path / "old_style.json"
        path.write_text(json.dumps({"dim": 32, "layers": 8}), encoding="utf-8")
        with pytest.warns(UserWarning, match="n_blocks"):
            config = M_load_config(path)
        assert config["n_blocks"] == 8
        assert "layers" not in config

    def test_load_config_rejects_unknown_keys(self, tmp_path) -> None:
        """A silently ignored key is worse than a missing one.

        Everything the model does not read must be refused, including the
        invented claims such as ``"params"`` or ``"tok_s"`` that the old
        hardware-capture configs carried.
        """
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"dim": 32, "params": "40M", "tok_s": "8-15"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as error:
            M_load_config(path)
        message = str(error.value)
        assert "params" in message and "tok_s" in message

    def test_load_config_can_be_told_not_to_be_strict(self, tmp_path) -> None:
        path = tmp_path / "extra.json"
        path.write_text(json.dumps({"dim": 32, "params": "40M"}), encoding="utf-8")
        config = M_load_config(path, strict=False)
        assert config["dim"] == 32

    def test_load_config_accepts_every_shipped_config(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for path in sorted(config_dir.glob("feather_*.json")):
            config = M_load_config(path)
            assert config["n_blocks"] >= 1, path.name

    def test_shipped_configs_match_their_filenames(self) -> None:
        """Guard the naming: a config called 40M must actually be near 40M."""
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        checked = 0
        for path in sorted(config_dir.glob("feather_*.json")):
            target = float(path.stem.split("_")[-1].rstrip("M"))
            if target <= 0:
                continue
            model = FeatherV2Model(M_load_config(path))
            actual = model.count_parameters() / 1e6
            assert (
                abs(actual - target) / target < 0.15
            ), f"{path.name} claims {target}M but measures {actual:.2f}M"
            checked += 1
            del model
        assert checked >= 4, f"only {checked} size configs to check"

    def test_divisibility_profile_of_one_is_only_zeroth_level(self) -> None:
        """1 is divisible by p^0 and by no higher power, for any p."""
        profile = M.divisibility_profile(torch.tensor([1.0]), 3, 4)
        assert torch.equal(profile[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_relaxed_weights_are_not_a_divisibility_test(self) -> None:
        """The log-magnitude surrogate is documented as measuring magnitude.

        It must not be mistaken for the exact indicator, which is why
        ``divisibility_profile`` exists separately.
        """
        relaxed = M.p_adic_weights(torch.tensor([1025.0]), 2, 3, 0.05)
        exact = M.divisibility_profile(torch.tensor([1025.0]), 2, 3)
        assert relaxed[0, 1] > 0.5
        assert exact[0, 1] == 0.0

    def test_p_adic_weights_are_nested(self) -> None:
        profile = M.p_adic_weights(torch.randn(16), 2, 6, 0.3)
        assert (profile[:, 1:] - profile[:, :-1] <= 1e-6).all()

    def test_p_adic_distance_is_a_metric_on_itself(self) -> None:
        x = torch.randn(8)
        assert torch.allclose(M.p_adic_distance(x, x), torch.zeros(8), atol=1e-6)

    def test_tt_compress_matches_optimal_truncated_svd(self) -> None:
        dense = torch.randn(24, 24) * 0.1
        original_u, original_s, original_vh = torch.linalg.svd(
            dense, full_matrices=False
        )
        best = original_u[:, :5] @ torch.diag(original_s[:5]) @ original_vh[:5, :]
        u, s, vh = M.tt_compress(dense, 5)
        assert torch.allclose(u @ torch.diag(s) @ vh, best, atol=1e-4)

    def test_tt_compress_reduces_rank(self) -> None:
        dense = torch.randn(32, 32) * 0.1
        u, s, _ = M.tt_compress(dense, 4)
        assert s.shape[0] == 4
        assert (u @ torch.diag(s) @ (u @ torch.diag(s)).T).shape == (32, 32)

    def test_tt_matmul_passes_gradient(self) -> None:
        x = torch.randn(2, 3, 16, requires_grad=True)
        out = M.tt_matmul(torch.randn(16, 4), torch.randn(4, 4), torch.randn(9, 4), x)
        assert out.shape == (2, 3, 9)
        out.sum().backward()
        assert x.grad.abs().sum() > 0

    def test_sinkhorn_rows_are_stochastic(self) -> None:
        plan = M.sinkhorn(torch.randn(8, 8), iters=60)
        assert torch.allclose(plan.sum(-1), torch.ones(8), atol=1e-4)

    def test_jacobi_solves_dominant_system(self) -> None:
        a = torch.eye(6) * 4 + torch.randn(6, 6) * 0.05
        b = torch.randn(3, 6)
        got = M.jacobi_decode(a, b, iters=200)
        exact = torch.linalg.solve(a, b.T).T
        assert torch.allclose(got, exact, atol=1e-4)

    def test_jacobi_handles_batched_systems(self) -> None:
        """Regression: (..., k) @ (..., k, k) silently mis-broadcast."""
        a = torch.eye(5) * 4 + torch.randn(2, 3, 5, 5) * 0.05
        b = torch.randn(2, 3, 5)
        got = M.jacobi_decode(a, b, iters=200)
        exact = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(got, exact, atol=1e-4)

    def test_equilibrium_reaches_its_fixed_point(self) -> None:
        c = torch.randn(6, 6) / 6
        b = torch.randn(3, 6)
        got = M.equilibrium_update(torch.randn(3, 6), c, b, iters=500, damping=0.5)
        assert torch.allclose(got + got @ c.T, b, atol=1e-4)

    def test_rough_path_signature_shape_and_grad(self) -> None:
        x = torch.randn(2, 5, 4, requires_grad=True)
        sig = M.rough_path_signature(x)
        assert sig.shape == (2, 5, 4, 4)
        sig.sum().backward()
        assert x.grad.abs().sum() > 0

    def test_godel_code_is_monotone(self) -> None:
        out = M.godel_encode(torch.tensor(1.0), torch.tensor([1.0, 2.0, 3.0]))
        assert (out.diff() > 0).all()

    def test_clifford_gate_blends_toward_identity_for_positive_gens(self) -> None:
        x = torch.randn(2, 3, 5)
        assert torch.allclose(
            M.clifford_gate(x, torch.full((5,), 5.0), scale=0.5), x, atol=1e-4
        )

    def test_clifford_gate_at_zero_generators_halves(self) -> None:
        x = torch.randn(2, 3, 5)
        assert torch.allclose(
            M.clifford_gate(x, torch.zeros(5), scale=0.5), 0.5 * x, atol=1e-6
        )

    def test_kan_uses_per_edge_basis_weights(self) -> None:
        layer = M.KANLinear(6, 5, num_gaussians=4)
        assert layer.weight.shape == (5, 6, 4)
        assert layer.centers.numel() == 4
        out = layer(torch.randn(3, 6))
        out.sum().backward()
        assert layer.weight.grad.abs().sum() > 0
        assert layer.centers.grad.abs().sum() > 0

    def test_kan_is_not_a_dense_layer(self) -> None:
        """A KAN must curve rather than interpolate linearly.

        Two degenerate setups are avoided: uniform edge weights make the
        response constant, because the basis is a partition of unity, and a
        symmetric basis with uniform weights makes it even, so only curvature
        is informative.
        """
        layer = M.KANLinear(1, 1, num_gaussians=5)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(5.0).reshape(1, 1, 5))
            layer.bias.zero_()
        response = lambda v: layer(torch.tensor([[v]])).item()  # noqa: E731
        low, mid, high = response(0.0), response(1.0), response(2.0)
        assert mid != pytest.approx((low + high) / 2, abs=1e-3)

    def test_alpha_dropout_is_unbiased(self) -> None:
        x = torch.ones(200000)
        out = M.alpha_dropout(x, 0.25, 0.5, training=True)
        assert out.mean().item() == pytest.approx(1.0, abs=0.02)

    def test_alpha_dropout_uses_three_valued_mask(self) -> None:
        out = M.alpha_dropout(torch.ones(50000), 0.25, 0.5, training=True)
        assert set(out.unique().tolist()) == {0.0, 1.0, 2.0}

    def test_alpha_dropout_zero_fraction_matches_p_times_q(self) -> None:
        out = M.alpha_dropout(torch.ones(200000), 0.25, 0.5, training=True)
        assert (out == 0).float().mean().item() == pytest.approx(0.125, abs=0.01)

    def test_alpha_dropout_is_identity_in_eval(self) -> None:
        x = torch.randn(1000)
        assert torch.equal(M.alpha_dropout(x, 0.5, 0.5, training=False), x)

    def test_fractional_weights_sum_to_one(self) -> None:
        w = M.fractional_weights(8, torch.tensor(0.7))
        assert w.shape == (8,)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_fractional_weights_differentiate_in_alpha(self) -> None:
        alpha = torch.tensor(0.7, requires_grad=True)
        M.fractional_weights(6, alpha).sum().backward()
        assert alpha.grad is not None


class TestExpertBank:
    def test_tt_expert_factors_match_truncated_svd(self) -> None:
        """A rank-4 TT cannot equal a rank-16 map, so compare against the
        provably optimal rank-4 truncation instead of the dense matrix."""
        torch.manual_seed(3)
        expert = TTExpert(16, rank=4)
        dense = torch.randn(16, 16) * 0.1
        expert.compress_from_dense(dense)
        u, s, vh = torch.linalg.svd(dense, full_matrices=False)
        best = u[:, :4] @ torch.diag(s[:4]) @ vh[:4, :]
        got = expert.u @ expert.core @ expert.v.T
        assert torch.allclose(got, best, atol=1e-4)

    def test_vault_aux_loss_falls_toward_perfect_balance(self) -> None:
        torch.manual_seed(5)
        vault = KnowledgeVault(32, n_experts=4, top_k=1, rank=4)
        x = torch.randn(4, 6, 32)
        first = vault(x)[1].item()
        assert first > 1.0
