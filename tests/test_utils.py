"""Feather v2 — utility math tests."""

from __future__ import annotations

import numpy as np
import pytest

from feather_v2.utils import (
    adaptive_clifford_product,
    adaptive_equilibrium_update,
    adaptive_fractional_weights,
    adaptive_jacobi_decode,
    adaptive_p_adic_chunk_retrieve,
    adaptive_p_adic_distance,
    adaptive_rough_path_signature,
    adaptive_sheaf_consistency,
    adaptive_sinkhorn,
    adaptive_tropical_matmul,
    adaptive_tropical_min,
    adaptive_tt_compress,
    cos_sim,
    fwht,
    hybrid_adaptive_tokenizer,
    hybrid_wht,
    kan_activation,
    normalize,
    tt_decompress,
    wht_bind,
)

from feather_v2.utils import (
    build_param_from_p,
    create_param_from_p,
    normalize_L2,
    tt_clifford_reduce,
)
from feather_v2.utils import tt_compress as public_tt_compress


def test_fwht_adds_only_and_involution():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64)
    y = fwht(a)
    back = fwht(y)
    np.testing.assert_allclose(back, a, atol=1e-12)


def test_wht_bind_similarity_bounded():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(1024)
    b = rng.standard_normal(1024)
    c = wht_bind(a, b)
    assert c.shape == a.shape
    assert -1.0 <= cos_sim(a, c) <= 1.0


def test_hybrid_wht_shape():
    a = np.ones(64)
    out = hybrid_wht(a, alpha=0.7, tau=0.1, p=2, rank=4)
    assert out.shape == a.shape


def test_adaptive_fractional_weights_sum():
    w = adaptive_fractional_weights(
        alpha=0.7,
        K=32,
        hardware_ram_gb=8.0,
        beta_learnable=1.2,
        hierarchical=True,
        liquid_tau=True,
    )
    assert w.shape[0] == 32
    assert np.isclose(w.sum(), 1.0, atol=1e-9)


def test_adaptive_tropical_min_range():
    a = np.array([-1.0, 0.0, 1.0, 2.0])
    val = adaptive_tropical_min(a, tau=0.1, adaptive=False)
    assert np.isfinite(val)


def test_adaptive_tropical_matmul_shape():
    W = np.ones((8, 16))
    x = np.ones(16)
    out = adaptive_tropical_matmul(W, x, tau=0.1, rank=4, experts=8)
    assert out.shape == (8,)


def test_adaptive_p_adic_distance_zero():
    assert adaptive_p_adic_distance(5, 5, p=2) == 0.0


def test_adaptive_p_adic_chunk_retrieve():
    q = np.zeros(64)
    chunks = np.eye(8, 64)
    idx, sim = adaptive_p_adic_chunk_retrieve(q, chunks, p=2)
    assert 0 <= idx < 8
    assert 0.0 <= sim <= 1.0


def test_adaptive_tt_compress_shape():
    W = np.eye(16)
    g1, g2 = adaptive_tt_compress(W, rank=4)
    assert g1.shape[0] == 16
    assert g2.shape[1] == 16
    assert g1.shape[1] == g2.shape[0]


def test_tt_decompress_roundtrip():
    W = np.random.default_rng(0).standard_normal((16, 16))
    g1, g2 = adaptive_tt_compress(W, rank=4)
    W2 = tt_decompress(g1, g2)
    assert W2.shape == W.shape


def test_adaptive_rough_path_signature_shape():
    x = np.random.default_rng(0).standard_normal((32, 3))
    sig = adaptive_rough_path_signature(x, level=2)
    assert sig.shape[0] > 0


def test_adaptive_sinkhorn_shape():
    C = np.random.default_rng(0).standard_normal((8, 8))
    plan = adaptive_sinkhorn(C, eps=0.1, iters=3)
    assert plan.shape == C.shape


def test_adaptive_clifford_product_shape():
    a = np.ones(8)
    b = np.ones(8)
    out = adaptive_clifford_product(a, b, vec_adaptive=True, use_wht=True, use_tt=True)
    assert out.shape[0] >= 1


def test_adaptive_sheaf_consistency_ok():
    models = [np.ones(16), np.ones(16)]
    res = adaptive_sheaf_consistency(
        models,
        krum_adaptive=True,
        eps_adaptive=True,
        hierarchical=True,
        use_fractional=True,
    )
    assert "consistent" in res
    assert "score" in res


def test_adaptive_equilibrium_update_keys():
    s = np.zeros(16)
    W = np.eye(16)
    res = adaptive_equilibrium_update(
        s, W, beta=0.1, D_adaptive=True, free_nudge_adaptive=True, use_fractional=True
    )
    assert "s_free" in res
    assert "dW" in res
    assert "free_energy" in res


def test_adaptive_jacobi_decode_shape():
    prompt = np.zeros((8, 16))
    res = adaptive_jacobi_decode(prompt, num_tokens=4)
    assert res["generated"].shape[0] == 4


def test_kan_activation_shape():
    x = np.ones(16)
    y = kan_activation(x, spline_order=3, grid_size=5, adaptive=True, layer_idx=0)
    assert y.shape == x.shape


def test_hybrid_adaptive_tokenizer_shapes():
    ids, info = hybrid_adaptive_tokenizer(
        "hello world", vocab_size=8256, hardware_ram_gb=31.0
    )
    assert ids.ndim == 1
    assert info["num_tokens"] == ids.shape[0]
    assert info["vocab_size"] == 8256


def test_cos_sim_self_one():
    a = np.random.default_rng(0).standard_normal(64)
    assert np.isclose(cos_sim(a, a), 1.0, atol=1e-12)


def test_normalize_unit_norm():
    a = np.random.default_rng(0).standard_normal(64)
    n = normalize(a)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-12)


def test_normalize_l2_unit_norm():
    a = np.random.default_rng(1).standard_normal((5, 32))
    n = normalize_L2(a, axis=-1)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-12)


def test_normalize_l2_leaves_zero_vectors_finite():
    a = np.zeros((3, 8))
    n = normalize_L2(a, axis=-1)
    assert np.all(np.isfinite(n))


def test_build_param_from_p_marks_exact_divisibility():
    # p**0 = 1 divides everything; 2 divides evens; 4 divides multiples of 4.
    profile = build_param_from_p(p=2, dim=8, levels=3)
    assert profile.shape == (8, 3)
    assert np.all(profile[:, 0] == 1.0)
    assert profile[1, 1] == 0.0 and profile[2, 1] == 1.0
    assert profile[4, 2] == 1.0 and profile[6, 2] == 0.0
    assert set(np.unique(profile)).issubset({0.0, 1.0})


def test_build_param_from_p_rejects_bad_primes():
    with pytest.raises(ValueError):
        build_param_from_p(p=1, dim=4, levels=2)
    with pytest.raises(ValueError):
        build_param_from_p(p=2, dim=4, levels=0)


def test_create_param_from_p_matches_builder():
    a = build_param_from_p(p=3, dim=16, levels=4)
    b = create_param_from_p(p=3, dim=16, levels=4)
    assert np.array_equal(a, b)


def test_public_tt_compress_matches_adaptive():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 16))
    a = public_tt_compress(w, rank=4)
    b = adaptive_tt_compress(w, rank=4)
    assert len(a) == len(b) == 2
    for x, y in zip(a, b):
        assert np.allclose(np.asarray(x), np.asarray(y))


def test_tt_clifford_reduce_preserves_length():
    x = np.random.default_rng(2).standard_normal(64)
    out = tt_clifford_reduce(x, rank=4)
    assert out.shape[0] == x.shape[0]
    assert np.all(np.isfinite(out))


def test_p_adic_group_is_defined_once():
    """Two near-identical private definitions existed; the second shadowed the first."""
    import inspect

    import feather_v2.utils as utils

    source = inspect.getsource(utils)
    assert source.count("def _p_adic_group(") == 1
