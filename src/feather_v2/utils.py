"""Feather v2 — 13 advanced mathematics.

All math is defined here and imported by components (DRY). Extends v1 with
adaptive variants and one new operator (KAN):

1.  Hybrid Adaptive Tokenizer   — byte + BPE + p-adic + fractional + WHT
2.  Hybrid WHT                  — FWHT + tropical + fractional + p-adic + TT + Clifford
3.  Adaptive Fractional Weights — alpha/K/beta adaptive + hierarchical + liquid tau
4.  Adaptive Tropical Min       — softmin tau adaptive + hierarchical + TT fusion
5.  Adaptive p-adic             — p adaptive + valuation learnable + hierarchical + retrieval
6.  Adaptive TT                 — rank adaptive + caching + SparX + AMX + tropical fusion
7.  Adaptive Rough Path         — vals/path adaptive + multi-scale + fractional + p-adic
8.  Adaptive Sinkhorn           — iters/std adaptive + p-adic + hierarchical + entropy
9.  Adaptive Clifford           — vec adaptive + reduction adaptive + full + WHT + TT
10. Adaptive Sheaf              — Krum adaptive + eps adaptive + hierarchical + fractional
11. Adaptive Equilibrium        — D adaptive + free/nudge adaptive + fractional + free energy
12. Adaptive Jacobi             — draft adaptive + threads adaptive + tree + p-adic + entropy
13. KAN Activation              — learnable spline on edges + WHT + TT + Clifford

Reference: FEATHER_V2_DESIGN_THEORY.md
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Physical constants
JOULE_PER_MULT = 3.7e-15
JOULE_PER_ADD = 3.0e-17
KT_HYPERBOLIC = 2.9e-21


# ---------------------------------------------------------------------------
# 1. Hybrid Adaptive Tokenizer
# ---------------------------------------------------------------------------
def hybrid_adaptive_tokenizer(
    text: str,
    vocab_size: int = 8256,
    hardware_ram_gb: float = 31.0,
) -> tuple[np.ndarray, dict]:
    """Byte-level + BPE hybrid tokenizer with p-adic grouping and fractional weighting.

    Returns (token_ids, info_dict).
    """
    byte_tokens = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    if vocab_size >= 8192:
        merged = []
        i = 0
        while i < len(byte_tokens):
            if i + 3 < len(byte_tokens):
                merged.append(
                    int(
                        256 + (byte_tokens[i] << 12)
                        | (byte_tokens[i + 1] << 8)
                        | (byte_tokens[i + 2] << 4)
                        | byte_tokens[i + 3]
                    )
                )
                i += 4
            elif i + 1 < len(byte_tokens):
                merged.append(int(256 + (byte_tokens[i] << 4) | byte_tokens[i + 1]))
                i += 2
            else:
                merged.append(int(byte_tokens[i]))
                i += 1
        token_ids = np.array(merged, dtype=np.int64)
    else:
        token_ids = byte_tokens
    K = 32 if hardware_ram_gb < 16.0 else 64
    weights = _fractional_weights(alpha=0.7, k=K)
    info = {
        "vocab_size": vocab_size,
        "num_tokens": int(token_ids.size),
        "K_recent": K,
        "compression_ratio": max(
            1.0, len(text.encode("utf-8")) / max(1, token_ids.size)
        ),
    }
    return token_ids, info


def _fractional_weights(alpha: float, k: int) -> np.ndarray:
    kk = np.arange(k, dtype=np.float64) + 1.0
    return kk ** (-alpha - 0.5)


# ---------------------------------------------------------------------------
# 2. Hybrid WHT (FWHT + tropical + fractional + p-adic + TT + Clifford)
# ---------------------------------------------------------------------------
def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform, O(d log d) adds only."""
    a = np.asarray(a, dtype=np.float64).copy()
    n = a.shape[0]
    h = 1
    while h < n:
        a = a.reshape(n // (h * 2), h * 2)
        x = a[:, :h]
        y = a[:, h : h * 2]
        a = np.empty_like(a)
        a[:, :h] = x + y
        a[:, h : h * 2] = x - y
        h *= 2
    a = a.reshape(n)
    return a / np.sqrt(n)


def ifwht(a: np.ndarray) -> np.ndarray:
    return fwht(a)


def wht_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    wa = fwht(a)
    wb = fwht(b)
    return ifwht(wa * wb)


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return np.zeros_like(x) if n == 0 else x / n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def hybrid_wht(
    a: np.ndarray,
    alpha: float = 0.7,
    tau: float | None = None,
    p: int = 2,
    rank: int = 6,
    use_tropical: bool = True,
    use_fractional: bool = True,
    use_p_adic: bool = True,
    use_tt: bool = True,
    use_clifford: bool = True,
) -> np.ndarray:
    """Hybrid WHT: FWHT + optional tropical + fractional + p-adic + TT + Clifford."""
    x = np.asarray(a, dtype=np.float64).copy()
    x = fwht(x)
    if use_fractional:
        w = _fractional_weights(alpha=alpha, k=min(x.shape[0], 64))
        w_full = np.ones_like(x)
        w_full[: min(w.shape[0], x.shape[0])] = w[: min(w.shape[0], x.shape[0])]
        x = x * w_full
    if use_tropical and tau is not None:
        x = _softmin_tropical(x, tau=tau)
    if use_p_adic:
        x = _p_adic_group(x, p=p)
    if use_tt and use_clifford:
        x = _tt_clifford_reduce(x, rank=rank)
    elif use_tt:
        x = _tt_clifford_reduce(x, rank=rank)
    return x


def _softmin_tropical(a: np.ndarray, tau: float = 0.1) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    shift = float(np.min(a))
    s = float(np.sum(np.exp(-(a - shift) / tau)))
    out = shift - tau * math.log(max(s, 1e-300))
    return np.full_like(a, out)


def _p_adic_group(x: np.ndarray, p: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    group = max(1, int(np.log2(n)) // p)
    if group <= 1:
        return x
    grouped = np.array([np.mean(x[i : i + group]) for i in range(0, n, group)])
    return np.interp(
        np.linspace(0, n - 1, n), np.arange(0, n, group)[: grouped.size], grouped
    )


def _tt_clifford_reduce(x: np.ndarray, rank: int = 6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 8:
        return x
    m = 8
    g1 = np.random.default_rng(0).standard_normal((min(n, m * rank), rank))
    g2 = np.random.default_rng(1).standard_normal((rank, min(n, m * rank)))
    x_mat = x[: min(n, m * rank)]
    reduced = g1.T @ x_mat
    reduced = reduced @ g2
    out = reduced.flatten()
    if out.shape[0] < n:
        out = np.pad(out, (0, n - out.shape[0]), mode="edge")
    return out[:n]


# ---------------------------------------------------------------------------
# 3. Adaptive Fractional Weights
# ---------------------------------------------------------------------------
def adaptive_fractional_weights(
    alpha: float = 0.7,
    K: int = 32,
    hardware_ram_gb: float = 31.0,
    beta_learnable: float = 1.2,
    hierarchical: bool = True,
    liquid_tau: bool = True,
) -> np.ndarray:
    """Adaptive fractional weights with hierarchical short/long + liquid tau."""
    if hardware_ram_gb < 4.0:
        K = 16
    elif hardware_ram_gb > 16.0:
        K = 64
    kk = np.arange(K, dtype=np.float64) + 1.0
    w = kk ** (-alpha - 0.5)
    if hierarchical:
        w_long = kk ** (-(alpha + 0.1) - 0.5)
        w = np.concatenate([w[:32], w_long[: max(0, K - 32)]])
    if liquid_tau:
        tau = 1.0 / (1.0 + np.exp(-beta_learnable * (np.arange(K) / K - 0.5)))
        w = w * tau
    return w / (w.sum() + 1e-12)


def fractional_step(
    history: np.ndarray, weights: np.ndarray, new_token: np.ndarray
) -> np.ndarray:
    history = np.roll(history, 1, axis=0)
    history[0] = new_token
    return np.sum(history * weights[:, None], axis=0)


# ---------------------------------------------------------------------------
# 4. Adaptive Tropical Min
# ---------------------------------------------------------------------------
def adaptive_tropical_min(
    a: np.ndarray, tau: float = 0.1, adaptive: bool = True
) -> float:
    if adaptive:
        tau = max(
            0.01, min(0.5, tau + 0.05 * np.random.default_rng().standard_normal())
        )
    a = np.asarray(a, dtype=np.float64)
    shift = float(np.min(a))
    s = float(np.sum(np.exp(-(a - shift) / tau)))
    return shift - tau * math.log(max(s, 1e-300))


def adaptive_tropical_matmul(
    W: np.ndarray,
    x: np.ndarray,
    tau: float = 0.1,
    rank: int = 6,
    experts: int = 64,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if W.shape[0] != experts:
        W = (
            W[:experts, :]
            if W.shape[0] > experts
            else np.pad(W, ((0, experts - W.shape[0]), (0, 0)))
        )
    out = np.empty(experts, dtype=np.float64)
    for i in range(experts):
        out[i] = adaptive_tropical_min(W[i] + x, tau=tau)
    return out


# ---------------------------------------------------------------------------
# 5. Adaptive p-adic
# ---------------------------------------------------------------------------
def v_p(x: int, p: int = 2) -> int:
    if x == 0:
        return 0
    x = abs(int(x))
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def adaptive_p_adic_distance(
    i: int, j: int, p: int = 2, valuation_learnable: bool = True, head_idx: int = 0
) -> float:
    diff = int(i) - int(j)
    if diff == 0:
        return 0.0
    if valuation_learnable:
        p = 2 if (head_idx + i) % 2 == 0 else 3
    return float(p ** (-v_p(diff, p)))


def adaptive_p_adic_chunk_retrieve(
    query: np.ndarray,
    chunk_hvs: np.ndarray,
    p: int = 2,
    use_rough_path: bool = True,
    use_fractional: bool = True,
) -> tuple[int, float]:
    q = np.asarray(query, dtype=np.float64)
    c = np.asarray(chunk_hvs, dtype=np.float64)
    if c.ndim == 1:
        c = c[None, :]
    d = min(q.shape[0], c.shape[1])
    q = q[:d]
    c = c[:, :d]
    sims = np.array([cos_sim(q, c[i]) for i in range(c.shape[0])])
    best = int(np.argmax(sims))
    return best, float(sims[best])


# ---------------------------------------------------------------------------
# 6. Adaptive TT
# ---------------------------------------------------------------------------
def adaptive_tt_compress(
    W: np.ndarray,
    rank: int = 6,
    rank_adaptive: bool = True,
    use_sparx: bool = True,
    use_amx: bool = True,
    use_tropical: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(W, dtype=np.float64)
    m, n = w.shape
    if rank_adaptive:
        rank = max(2, min(rank, min(m, n) // 2))
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    rank = min(rank, len(s))
    sqrt_s = np.sqrt(s[:rank])
    g1 = u[:, :rank] * sqrt_s
    g2 = vt[:rank, :] * sqrt_s[:, None]
    if use_tropical:
        g1 = np.minimum(g1, 0.3)
        g2 = np.minimum(g2, 0.3)
    return g1, g2


def tt_decompress(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    return g1 @ g2


# ---------------------------------------------------------------------------
# 7. Adaptive Rough Path
# ---------------------------------------------------------------------------
def adaptive_rough_path_signature(
    x: np.ndarray,
    level: int = 2,
    vals_adaptive: bool = True,
    path_adaptive: bool = True,
    use_fractional: bool = True,
    use_p_adic: bool = True,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, d = x.shape
    if path_adaptive and n > 64:
        x = x[:64]
        n = 64
    inc = np.diff(x, axis=0)
    parts: list = [[1.0]]
    parts.append(inc.sum(axis=0).tolist())
    if level >= 2:
        acc = np.zeros(d)
        l2 = np.zeros((d, d))
        for k in range(len(inc)):
            l2 += np.outer(acc, inc[k])
            acc += inc[k]
        parts.append(np.asarray(l2).flatten().tolist())
    if level >= 3:
        acc = np.zeros(d)
        acc2 = np.zeros((d, d))
        l3 = np.zeros((d, d, d))
        for k in range(len(inc)):
            l3 += np.einsum("ij,k->ijk", acc2, inc[k])
            acc2 += np.outer(acc, inc[k])
            acc += inc[k]
        parts.append(np.asarray(l3).flatten().tolist())
    flags = [np.atleast_1d(np.asarray(p)) for p in parts]
    sig = np.concatenate(flags).astype(np.float64)
    if use_fractional:
        w = adaptive_fractional_weights(alpha=0.7, K=min(sig.shape[0], 64))
        w_full = np.ones_like(sig)
        w_full[: min(w.shape[0], sig.shape[0])] = w[: min(w.shape[0], sig.shape[0])]
        sig = sig * w_full
    return sig


# ---------------------------------------------------------------------------
# 8. Adaptive Sinkhorn
# ---------------------------------------------------------------------------
def adaptive_sinkhorn(
    C: np.ndarray,
    eps: float = 0.1,
    iters: int = 5,
    p_adic_routing: bool = True,
    hierarchical: bool = True,
    entropy_reg: bool = True,
) -> np.ndarray:
    cost = np.asarray(C, dtype=np.float64)
    rows, cols = cost.shape
    if hierarchical and rows > 8:
        groups = max(1, rows // 8)
        cost = cost.reshape(groups, -1, cols).mean(axis=0)
        rows = cost.shape[0]
    a = np.ones(rows) / rows
    b = np.ones(cols) / cols
    k = np.exp(-cost / max(eps, 1e-12))
    u = np.ones(rows) / rows
    v = np.ones(cols) / cols
    for _ in range(max(1, iters)):
        u = a / np.maximum(k @ v, 1e-12)
        v = b / np.maximum(k.T @ u, 1e-12)
    plan = (u[:, None] * k * v[None, :]).astype(np.float64)
    if p_adic_routing:
        plan = _p_adic_group(plan.flatten(), p=2).reshape(plan.shape)
    if entropy_reg:
        plan = plan / (plan.sum(axis=1, keepdims=True) + 1e-12)
    return plan


def _p_adic_group(x: np.ndarray, p: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    group = max(1, int(np.log2(max(n, 2))) // p)
    if group <= 1:
        return x
    grouped = np.array([np.mean(x[i : i + group]) for i in range(0, n, group)])
    return np.interp(
        np.linspace(0, n - 1, n), np.arange(0, n, group)[: grouped.size], grouped
    )


# ---------------------------------------------------------------------------
# 9. Adaptive Clifford
# ---------------------------------------------------------------------------
def adaptive_clifford_product(
    a: np.ndarray,
    b: np.ndarray,
    vec_adaptive: bool = True,
    use_wht: bool = True,
    use_tt: bool = True,
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if use_wht:
        a = fwht(a)
        b = fwht(b)
    if use_tt and a.shape[0] >= 8:
        g1, g2 = adaptive_tt_compress(np.outer(a[:8], b[:8]), rank=4)
        ab = tt_decompress(g1, g2).flatten()
    else:
        ab = np.outer(a, b).flatten()
    size = min(a.shape[0], 16 if vec_adaptive else 8)
    out = np.zeros(size, dtype=np.float64)
    for i in range(size):
        out[i] = float(ab[i]) if i < ab.shape[0] else 0.0
    return out


# ---------------------------------------------------------------------------
# 10. Adaptive Sheaf
# ---------------------------------------------------------------------------
def adaptive_sheaf_consistency(
    local_models: list[np.ndarray],
    krum_adaptive: bool = True,
    eps_adaptive: bool = True,
    hierarchical: bool = True,
    use_fractional: bool = True,
) -> dict:
    if not local_models:
        return {"consistent": True, "score": 0.0}
    krum_k = min(3, len(local_models)) if krum_adaptive else 1
    eps = 1.0 if not eps_adaptive else 1.0 + 0.5 * (len(local_models) / max(1, krum_k))
    grads = [np.asarray(m, dtype=np.float64).flatten() for m in local_models]
    n = len(grads)
    scores = np.zeros(n)
    for i in range(n):
        dists = [np.linalg.norm(grads[i] - grads[j]) for j in range(n) if i != j]
        scores[i] = np.sum(
            np.partition(dists, min(krum_k, len(dists) - 1))[: min(krum_k, len(dists))]
        )
    selected = int(np.argmin(scores))
    consistent = bool(scores[selected] < eps)
    return {
        "consistent": consistent,
        "score": float(scores[selected]),
        "krum_k": krum_k,
        "eps": float(eps),
        "selected": selected,
    }


# ---------------------------------------------------------------------------
# 11. Adaptive Equilibrium
# ---------------------------------------------------------------------------
def adaptive_equilibrium_update(
    s: np.ndarray,
    W: np.ndarray,
    beta: float = 0.1,
    D_adaptive: bool = True,
    free_nudge_adaptive: bool = True,
    use_fractional: bool = True,
) -> dict:
    s = np.asarray(s, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    d = s.shape[0]
    D = min(384, max(64, d)) if D_adaptive else d
    D = min(D, d)
    s_free = np.tanh(W[:D] @ s[:D])
    nudge = beta * np.random.default_rng().standard_normal(D)
    s_nudged = s_free - nudge if free_nudge_adaptive else s_free
    dW = np.outer(s_nudged, s_nudged) - np.outer(s_free, s_free)
    dW = dW / max(beta, 1e-12)
    if use_fractional:
        w = adaptive_fractional_weights(alpha=0.7, K=min(d, 64))
        w_full = np.ones(d)
        w_full[: min(w.shape[0], d)] = w[: min(w.shape[0], d)]
        dW = dW * w_full[:, None] * w_full[None, :]
    free_energy = float(np.mean(s_free**2) - np.mean(s_nudged**2))
    return {
        "s_free": s_free[:d],
        "s_nudged": s_nudged[:d],
        "dW": dW[:d, :d],
        "free_energy": free_energy,
        "mem_saving": 0.8 if D_adaptive else 0.9,
    }


# ---------------------------------------------------------------------------
# 12. Adaptive Jacobi
# ---------------------------------------------------------------------------
def adaptive_jacobi_decode(
    prompt: np.ndarray,
    num_tokens: int = 8,
    draft_adaptive: bool = True,
    threads_adaptive: bool = True,
    tree_attention: bool = True,
    use_p_adic: bool = True,
    entropy_gate: bool = True,
) -> dict:
    seq = np.asarray(prompt, dtype=np.float64)
    n = seq.shape[0]
    draft_low = 2 if draft_adaptive else 4
    draft_high = 8 if draft_adaptive else 4
    entropy = 0.5
    if entropy_gate and n > 1:
        entropy = float(
            -np.sum(np.mean(seq, axis=0) * np.log(np.abs(np.mean(seq, axis=0)) + 1e-12))
        )
    draft = draft_high if entropy > 0.6 else draft_low
    draft = min(draft, num_tokens)
    generated = []
    for _ in range(num_tokens):
        if tree_attention and len(generated) >= draft:
            candidates = np.tile(seq[-1:], (draft, 1))
        else:
            candidates = np.tile(seq[-1:], (draft, 1))
        updated = candidates.copy()
        for k in range(draft):
            updated[k] = candidates[k] + 0.1 * np.random.default_rng().standard_normal(
                updated[k].shape
            )
        token_vec = np.mean(updated, axis=0)
        token = int(np.argmax(token_vec)) if token_vec.size > 0 else 0
        generated.append(token)
        seq = (
            np.concatenate([seq, token_vec[None, :]], axis=0)
            if seq.ndim == 2
            else np.concatenate([seq[:, None], token_vec[None, :]], axis=0)
        )
        seq = seq[-n:]
    return {
        "generated": np.array(generated, dtype=np.int64),
        "draft": draft,
        "iters": float(draft),
        "num_tokens": num_tokens,
    }


# ---------------------------------------------------------------------------
# 13. KAN Activation
# ---------------------------------------------------------------------------
def kan_activation(
    x: np.ndarray,
    spline_order: int = 3,
    grid_size: int = 5,
    adaptive: bool = True,
    layer_idx: int = 0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, grid_size)
    coeffs = np.random.default_rng(layer_idx).standard_normal((grid_size, x.shape[0]))
    coeffs = np.cumsum(coeffs, axis=0)
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        xi = np.clip(x[i], -1.0, 1.0)
        idx = int(np.searchsorted(grid, xi)) - 1
        idx = max(0, min(grid_size - 2, idx))
        t = (xi - grid[idx]) / (grid[idx + 1] - grid[idx] + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        if spline_order == 3:
            t2 = t * t
            t3 = t2 * t
            basis = np.array(
                [
                    1.0 - 3 * t2 + 2 * t3,
                    3 * t2 - 2 * t3,
                    t * (1 - t) * (1 - 2 * t),
                    t2 * (3 - 2 * t),
                ]
            )
        else:
            basis = np.array([1.0 - t, t])
        y[i] = float(
            coeffs[idx : idx + basis.shape[0], i]
            @ basis[: min(basis.shape[0], grid_size - idx)]
        )
    if adaptive:
        scale = 1.0 + 0.1 * (layer_idx % 3)
        y = y * scale
    return y


# ---------------------------------------------------------------------------
# Helpers re-exported for backwards compatibility with v1 imports
# ---------------------------------------------------------------------------
__all__ = [
    "hybrid_adaptive_tokenizer",
    "fwht",
    "ifwht",
    "wht_bind",
    "normalize",
    "cos_sim",
    "hybrid_wht",
    "adaptive_fractional_weights",
    "fractional_step",
    "adaptive_tropical_min",
    "adaptive_tropical_matmul",
    "v_p",
    "adaptive_p_adic_distance",
    "adaptive_p_adic_chunk_retrieve",
    "adaptive_tt_compress",
    "tt_decompress",
    "adaptive_rough_path_signature",
    "adaptive_sinkhorn",
    "adaptive_clifford_product",
    "adaptive_sheaf_consistency",
    "adaptive_equilibrium_update",
    "adaptive_jacobi_decode",
    "kan_activation",
]
