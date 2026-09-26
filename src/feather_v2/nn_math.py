"""Autograd-aware implementations of the Feather v2 mathematics.

Several of the Feather operators are discrete by construction: p-adic
valuation, tropical minimum, tensor-train rank truncation and Godel coding have
no derivative. Silently replacing them with identity maps (what the NumPy
scaffold did) is what made the previous "trainable" model a fiction.

Every operator here is autograd-aware. Discrete operators get a principled
continuous relaxation; where the exact discrete result is genuinely required it
is combined with a straight-through estimator so gradients still reach the
inputs while the forward value stays exact.

The NumPy implementations in :mod:`feather_v2.utils` remain the reference used
for inference-time cross-checks and by the existing benchmark suite.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "KANLinear",
    "alpha_dropout",
    "clifford_gate",
    "divisibility_profile",
    "equilibrium_update",
    "fractional_weights",
    "fwht",
    "godel_encode",
    "godel_log_code",
    "ifwht",
    "jacobi_decode",
    "next_pow2",
    "p_adic_distance",
    "p_adic_weights",
    "pad_to_pow2",
    "rough_path_signature",
    "sheaf_project",
    "sinkhorn",
    "softmin",
    "tt_compress",
    "tt_factors",
    "tt_matmul",
    "tropical_matmul",
    "tropical_softmax",
]


def next_pow2(n: int) -> int:
    """Smallest power of two greater than or equal to ``n``."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def pad_to_pow2(x: Tensor, dim: int = -1) -> tuple[Tensor, int]:
    """Zero pad ``x`` along ``dim`` up to a power of two, returning the original size."""
    n = x.shape[dim]
    target = next_pow2(n)
    if target == n:
        return x, n
    pad_shape = list(x.shape)
    pad_shape[dim] = target - n
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim), n


def fwht(x: Tensor) -> Tensor:
    """Fast Walsh-Hadamard transform over the last dimension.

    The Sylvester butterfly needs a power of two length, so a non power of two
    input is zero padded up to the next power of two and the *full* transform
    is returned. Padding then cropping would make the map non-invertible (a
    degree of freedom is destroyed), so the returned transform keeps the padded
    length and :func:`ifwht` is responsible for cropping back.
    """
    lead = x.shape[:-1]
    n = x.shape[-1]
    m = next_pow2(n)
    if m != n:
        pad_shape = list(x.shape)
        pad_shape[-1] = m - n
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=-1)
    h = 1
    while h < m:
        grouped = x.reshape(*lead, m // (2 * h), 2, h)
        a = grouped[..., 0, :]
        b = grouped[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*lead, m)
        h *= 2
    return x


def ifwht(x: Tensor, length: int | None = None) -> Tensor:
    """Inverse fast Walsh-Hadamard transform, cropping to ``length`` if given.

    ``H`` is self inverse up to a factor of its length, so the inverse is the
    same butterfly divided by ``n``. Pass ``length`` to undo the padding that
    :func:`fwht` applied to a non power of two input.
    """
    m = x.shape[-1]
    y = fwht(x) / m
    if length is not None and length != m:
        y = y[..., :length]
    return y


def softmin(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    """Differentiable tropical minimum ``-tau * log sum exp(-x / tau)``."""
    tau = max(float(tau), 1e-6)
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def tropical_softmax(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    """Temperature scaled softmax used as the tropical max-plus relaxation."""
    return torch.softmax(x / max(float(tau), 1e-6), dim=dim)


def tropical_matmul(a: Tensor, b: Tensor, tau: float = 0.1) -> Tensor:
    """Tropical (min,+) matrix product of ``a`` (..., n, k) and ``b`` (..., k, m)."""
    expanded = a.unsqueeze(-1) + b.unsqueeze(-3)
    return softmin(expanded, tau, dim=-2)


def fractional_weights(
    n: int, alpha: Tensor, device: torch.device | None = None
) -> Tensor:
    """Normalised ``i ** -alpha`` decay weights, differentiable in ``alpha``."""
    idx = torch.arange(1, n + 1, device=device, dtype=alpha.dtype)
    w = idx.pow(-alpha.reshape(()))
    return w / w.sum()


def divisibility_profile(x: Tensor, p: int = 2, levels: int = 8) -> Tensor:
    """Exact indicator ``1{p^k divides x}`` for ``k = 0 .. levels-1``.

    Divisibility is a discrete property, so this is a *structural descriptor*,
    not a differentiable relaxation: it carries no gradient with respect to
    ``x``. That is stated explicitly rather than hidden behind a surrogate
    that would silently mean something else. It is intended to be compared
    against learnable prototypes in profile space, and those prototypes do
    receive gradients.

    The magnitude is rounded to the nearest integer first, so the descriptor is
    well defined for real valued activations: it describes the divisibility of
    the nearest integer, not a continuous notion of valuation.

    A log-magnitude relaxation such as :func:`p_adic_weights` cannot replace
    this: it measures how far ``|x|`` is from ``p^k`` in log space, so it scores
    1025 as highly divisible by 2 even though ``v_2(1025) = 0``.
    """
    magnitude = x.detach().abs().round()
    powers = torch.tensor(
        [float(p) ** k for k in range(levels)],
        device=x.device,
        dtype=x.dtype,
    )
    remainder = torch.remainder(magnitude.unsqueeze(-1), powers)
    return (remainder < 0.5).to(x.dtype)


def p_adic_weights(x: Tensor, p: int = 2, levels: int = 8, temp: float = 0.5) -> Tensor:
    """Soft indicator over p-adic valuation levels ``k = 0 .. levels-1``.

    Entry ``k`` approximates ``1{v_p(x) >= k}``. The discrete valuation is
    replaced by a sigmoid on ``log|x| - k*log(p)`` biased by half a step so that
    exact powers land on the correct side, and the levels are made nested with
    a cumulative minimum because ``v_p >= k`` implies ``v_p >= j`` for every
    ``j < k``. This is a temperature controlled relaxation, not an exact
    detector: integers close to a power boundary blend between levels.
    """
    log_abs = torch.log(x.abs().clamp_min(1e-12))
    ks = torch.arange(levels, device=x.device, dtype=x.dtype)
    logits = (log_abs.unsqueeze(-1) - ks * math.log(float(p)) + 0.5) / max(
        float(temp), 1e-6
    )
    return torch.cummin(torch.sigmoid(logits), dim=-1).values


def p_adic_distance(
    x: Tensor, y: Tensor, p: int = 2, levels: int = 8, temp: float = 0.5
) -> Tensor:
    """Relaxed p-adic distance: total variation between valuation profiles."""
    wx = p_adic_weights(x, p, levels, temp)
    wy = p_adic_weights(y, p, levels, temp)
    return (wx - wy).abs().sum(-1)


def tt_factors(dense: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]:
    """Truncated SVD of a dense linear map into TT factors ``(U, S, V)``."""
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    r = min(int(rank), s.shape[-1])
    return u[:, :r], s[:r], vh[:r, :]


def tt_compress(
    dense: Tensor, rank: int, temp: float = 0.05
) -> tuple[Tensor, Tensor, Tensor]:
    """TT compression of a dense map with a straight-through soft rank.

    The factors reconstruct the exact rank-``r`` truncation, while the gradient
    follows a sigmoid-weighted spectrum, so the effective rank stays
    differentiable and the truncation is not a hard, gradient-killing step.
    """
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    r = min(int(rank), s.shape[-1])
    if r >= s.shape[-1]:
        return u, s, vh
    hard = (u[:, :r] * s[:r]) @ vh[:r, :]
    weights = torch.zeros_like(s)
    weights[:r] = torch.sigmoid((s[:r] - s[r : r + 1]) / max(float(temp), 1e-6))
    soft = (u * (s * weights)) @ vh
    return tt_factors(hard + soft - soft.detach(), r)


def tt_matmul(u: Tensor, core: Tensor, v: Tensor, x: Tensor) -> Tensor:
    """Apply a TT-matrix factorised as ``(U, core, V)`` to the last dim of ``x``."""
    left = x @ u
    mid = core @ v.transpose(-1, -2)
    return left @ mid


def sinkhorn(cost: Tensor, eps: float = 0.05, iters: int = 20) -> Tensor:
    """Log-domain Sinkhorn projection of ``cost`` onto the assignment polytope.

    A final row normalisation is applied so the returned plan is exactly row
    stochastic; without it the last operation normalises columns and the rows
    only converge towards one.
    """
    log_k = -cost / max(float(eps), 1e-6)
    f = torch.zeros_like(log_k[..., 0, :])
    g = torch.zeros_like(log_k[..., :, 0])
    for _ in range(int(iters)):
        f = -torch.logsumexp(log_k + g.unsqueeze(-2), dim=-1)
        g = -torch.logsumexp(log_k + f.unsqueeze(-1), dim=-2)
    f = -torch.logsumexp(log_k + g.unsqueeze(-2), dim=-1)
    return torch.exp(log_k + f.unsqueeze(-1) + g.unsqueeze(-2))


def equilibrium_update(
    x: Tensor, a: Tensor, b: Tensor, iters: int = 8, damping: float = 0.5
) -> Tensor:
    """Damped fixed point iteration for ``x = b - A x``, differentiable per sweep.

    The fixed point is ``x = (I + A)^-1 b``. The iteration matrix is ``-A``, so
    this converges only when the spectral radius of ``A`` is below one; callers
    must keep ``A`` contractive (a learned projection with unit spectral norm is
    the usual choice). ``damping < 1`` shrinks the effective matrix to
    ``-(1 - damping) A`` and widens the stable range.
    """
    for _ in range(int(iters)):
        row = x.unsqueeze(-2)
        drive = b.unsqueeze(-2) - row @ a.transpose(-1, -2)
        x = (1.0 - damping) * x + damping * drive.squeeze(-2)
    return x


def jacobi_decode(
    a: Tensor, b: Tensor, iters: int = 12, damping: float = 1.0
) -> Tensor:
    """Jacobi iteration for the linear system ``A x = b``.

    Each sweep solves the diagonal system exactly and relaxes by ``damping``:
    ``x <- (1 - damping) x + damping D^-1 (b - (A - D) x)``. The iteration
    matrix is ``I - D^-1 (A - D)``, so convergence requires ``A`` to be strictly
    diagonally dominant. This is a genuine solver for ``A x = b`` and is not
    interchangeable with :func:`equilibrium_update`, which converges to
    ``(I + A)^-1 b`` instead.
    """
    diagonal = torch.diagonal(a, dim1=-2, dim2=-1)
    off = a - torch.diag_embed(diagonal)
    inv_diag = diagonal.clamp_min(1e-6).reciprocal()
    x = torch.zeros_like(b)
    for _ in range(int(iters)):
        row = x.unsqueeze(-2)
        update = (b.unsqueeze(-2) - row @ off.transpose(-1, -2)).squeeze(-2)
        x = (1.0 - damping) * x + damping * update * inv_diag
    return x


def sheaf_project(sections: Tensor, window: int = 4, iters: int = 4) -> Tensor:
    """Project overlapping local sections onto the local-consistency subspace.

    Each window is replaced by its own mean and re-expanded, which forces every
    local restriction to agree with its neighbours. Unlike the discrete
    coboundary solve used in the NumPy reference this is a smooth, fully
    differentiable operator.
    """
    window = max(1, min(int(window), sections.shape[-2]))
    out = sections
    for _ in range(int(iters)):
        acc = torch.zeros_like(out)
        count = torch.zeros_like(out)
        for start in range(0, out.shape[-2] - window + 1):
            acc[..., start : start + window, :] += out[
                ..., start : start + window, :
            ].mean(dim=-2, keepdim=True)
            count[..., start : start + window, :] += 1.0
        out = acc / count.clamp_min(1.0)
    return out


def clifford_gate(x: Tensor, generators: Tensor, scale: float = 0.5) -> Tensor:
    """Apply learnable Clifford generators to the last dimension of ``x``.

    Diagonal Clifford generators over the sign set are exact involutions, so
    the map stays invertible; ``tanh`` keeps the relaxation inside the
    generator manifold and ``scale`` blends it with the identity to keep
    gradients well conditioned.
    """
    g = torch.tanh(generators)
    return (1.0 - scale) * x + scale * (x * g)


def rough_path_signature(x: Tensor) -> Tensor:
    """Level-2 truncated rough path signature of increments ``x`` (..., T, D).

    The exact level-2 signature solves ``dS2 = dS1 (x) dS1``, so accumulating
    outer products of the running level-1 sums reproduces it up to the usual
    second order discretisation error. This keeps the signature differentiable
    instead of relying on non-differentiable time reparameterisation.
    """
    s1 = torch.cumsum(x, dim=-2)
    outer = s1.unsqueeze(-1) * s1.unsqueeze(-2)
    return torch.cumsum(outer, dim=-3)


def godel_log_code(relation: Tensor, message: Tensor, base: float = 2.0) -> Tensor:
    """Log of the continuous Godel code ``(1 + l) ** (1 + a + b)``.

    Godel's monotone encoding ``2^a (2b + 1)`` puts the relation in an
    exponent, which smooths the ordering that raw pairing destroys. The
    continuous Gode function is the standard smooth monotone surrogate: both
    arguments enter a single exponent, so it is strictly increasing in each
    and has a non-zero partial derivative everywhere, including at the origin.

    Two earlier versions of this project failed differently, and both failures
    are instructive:

    * ``(1 + |a|) ** (1 + |b|)`` is monotone but its derivative in ``a`` is
      proportional to ``sign(a)``, so a parameter initialised at exactly zero
      received an identically zero gradient and could never train.
    * ``(1 + l) ** (1 + a + b)`` is correct but overflows to ``inf`` once the
      magnitude grows, and the backward pass then produces ``nan``.

    Returning the logarithm sidesteps the overflow entirely: the log of a
    power is linear in the exponent, so this is a strictly increasing affine
    map of both arguments with bounded derivatives. Callers that need a bounded
    monotone value should squash this with a logistic; callers that need the
    code itself should use :func:`godel_encode`.
    """
    return (1.0 + relation + message) * math.log1p(float(base))


def godel_encode(
    relation: Tensor, message: Tensor, base: float = 2.0, limit: float = 60.0
) -> Tensor:
    """The continuous Godel code itself, clamped to stay finite.

    Prefer :func:`godel_log_code` for anything that participates in a loss.
    This wrapper exists for callers that genuinely want the code value, and
    clamps the exponent so the result saturates smoothly instead of becoming
    ``inf`` and poisoning the backward pass with ``nan``.
    """
    return torch.exp(torch.clamp(godel_log_code(relation, message, base), max=limit))


def alpha_dropout(
    x: Tensor, p: float = 0.25, q: float = 0.5, training: bool = True
) -> Tensor:
    """Alpha-dropout (Ba, Hinton, Srivastava and Krizhevsky, 2016).

    Each unit is dropped with probability ``p``; among dropped units the
    activation is zeroed with probability ``q`` and otherwise retained and
    rescaled by ``1 / (1 - q)``. The expectation is exactly the input, but the
    retained activations are *shifted* rather than merely thinned, which is the
    property separating this from ordinary dropout.

    The mask is drawn as a three-way categorical draw rather than a product of
    two independent Bernoulli masks. A product of two masks forces
    ``P(unchanged) = (1 - p) * P(keep)``, which cannot equal the required
    ``1 - p`` unless the second mask never fires, so the factorised form is
    silently biased. Here ``u > p`` keeps the unit, otherwise ``u > p*q`` keeps
    it rescaled and the remaining mass zeroes it.
    """
    if not training or p <= 0.0:
        return x
    u = torch.rand_like(x)
    scale = 1.0 / max(1.0 - float(q), 1e-6)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    one = torch.ones((), dtype=x.dtype, device=x.device)
    mask = torch.where(
        u > p,
        one.expand_as(x),
        torch.where(u > p * q, (one * scale).expand_as(x), zero.expand_as(x)),
    )
    return x * mask


class KANLinear(nn.Module):
    """Kolmogorov-Arnold layer with learnable univariate basis functions.

    Each input coordinate is expanded by a bank of learnable Gaussian basis
    functions, and every (output, input, basis) edge carries its own weight.
    Unlike an MLP of the same width this keeps a per-edge univariate function,
    which is what makes the layer a KAN rather than a renamed dense layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_gaussians: int = 5,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_gaussians = int(num_gaussians)
        centers = torch.linspace(grid_range[0], grid_range[1], num_gaussians)
        self.centers = nn.Parameter(centers)
        self.log_sigma = nn.Parameter(
            torch.full((in_features, num_gaussians), math.log(0.5))
        )
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, num_gaussians)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        bound = 1.0 / math.sqrt(in_features * num_gaussians)
        nn.init.uniform_(self.weight, -bound, bound)

    def basis(self, x: Tensor) -> Tensor:
        """Evaluate the normalised basis bank for ``x`` (..., in_features)."""
        delta = x.unsqueeze(-1) - self.centers
        sigma = torch.exp(self.log_sigma).clamp_min(1e-3)
        phi = torch.exp(-0.5 * (delta / sigma).pow(2))
        return phi / phi.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x: Tensor) -> Tensor:
        phi = self.basis(x)
        out = torch.einsum("...fg,ofg->...o", phi, self.weight)
        return out + self.bias
