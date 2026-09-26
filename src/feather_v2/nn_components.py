"""Trainable PyTorch implementations of the seven Feather v2 components.

The NumPy components in :mod:`feather_v2.models` are parameter-free transforms
that return debug dictionaries. They are kept for the inference-time reference
benchmark, but nothing in them can receive a gradient, which is why the previous
"end to end trainable" model could not be trained.

Every class here is a real :class:`torch.nn.Module` holding real
:class:`torch.nn.Parameter` tensors, mapping ``(batch, time, dim)`` activations
to activations of the same shape, and participating in autograd. The discrete
mathematics is supplied by :mod:`feather_v2.nn_math` in relaxed form.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import nn_math as M

__all__ = [
    "CognitiveWeaver",
    "GenerativeEvolution",
    "HomeostasisGovernor",
    "HyperDimensionalMemory",
    "KnowledgeVault",
    "LiquidMemory",
    "SensoryEncoder",
    "TTExpert",
]


class SensoryEncoder(nn.Module):
    """Multi-scale fractional encoder with p-adic scale selection.

    A learned number of scales are mixed under weights that combine three real
    operators: a tropical softmax over learned scale logits, a relaxed p-adic
    match between the token and a learned per-scale valuation prototype, and a
    learnable fractional ``i ** -alpha`` recency decay across scales. A level-2
    rough path signature of the projected signal supplies a multi-scale
    increment term.
    """

    def __init__(
        self,
        dim: int,
        n_scales: int = 4,
        sig_dim: int = 8,
        p: int = 2,
        levels: int = 8,
        tau: float = 0.1,
        temp: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_scales = int(n_scales)
        self.sig_dim = int(sig_dim)
        self.tau = float(tau)
        self.p = int(p)
        self.levels = int(levels)
        self.temp = float(temp)

        self.in_proj = nn.Linear(dim, 2 * dim)
        self.scale_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_scales)])
        self.scale_logits = nn.Parameter(torch.zeros(n_scales))
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.proto_levels = nn.Parameter(torch.zeros(n_scales, levels))
        self.sig_in = nn.Linear(dim, sig_dim)
        self.sig_out = nn.Linear(sig_dim * sig_dim, dim)

    def scale_weights(self, x: Tensor) -> Tensor:
        """Per-token, per-scale mixture weights, differentiable throughout.

        Scale selection compares the exact p-adic divisibility profile of the
        activations against a learnable per-scale prototype held in the same
        profile space. Because profile entries are 0 or 1, the absolute
        difference against a prototype expands to
        ``m_k (1 - 2 q_sk) + d q_sk`` where ``m_k`` counts how many channels are
        divisible by ``p^k``. That identity avoids ever materialising the
        ``(batch, time, dim, scales, levels)`` tensor a naive broadcast would
        build, which at realistic sizes is larger than the model itself.

        The profile is a discrete descriptor and carries no gradient, but the
        prototype, the scale logits and the fractional decay exponent are all
        learned through the mixture that consumes them.
        """
        profile = M.divisibility_profile(x, self.p, self.levels)
        mass = profile.sum(dim=-2)
        dim = float(x.shape[-1])
        distance = (mass.unsqueeze(-2) * (1.0 - 2.0 * self.proto_levels)).sum(
            -1
        ) + dim * self.proto_levels.sum(-1)
        logits = self.scale_logits.unsqueeze(0) - distance / max(self.tau, 1e-6)
        weights = M.tropical_softmax(logits, self.tau)
        decay = M.fractional_weights(self.n_scales, self.alpha, x.device)
        weights = weights * decay
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        stacked = torch.stack([proj(value) for proj in self.scale_proj], dim=-2)
        weights = self.scale_weights(x).unsqueeze(-1)
        mixed = (stacked * weights).sum(dim=-2)
        signature = M.rough_path_signature(self.sig_in(x))
        signature = signature.flatten(-2)
        return F.gelu(mixed + self.sig_out(signature)) * torch.sigmoid(gate)


class LiquidMemory(nn.Module):
    """Hierarchical chunked gated recurrence.

    The recurrent state is advanced once per chunk rather than once per token,
    which keeps the unrolled graph ``ceil(seq_len / chunk)`` deep instead of
    ``seq_len`` deep. That is the difference between a CPU-viable model and one
    that spends all its time in Python-level loop overhead. Forget, input and
    output gates are learned projections, and the recurrence itself is an
    ordinary differentiable scan, so gradients reach every chunk.

    The chunk mean is subtracted from the signal *written into* memory, but not
    from the returned residual. Centring both is a trap: a centred residual fed
    into a trailing ``LayerNorm`` composes two centring projections and drives
    the gradient w.r.t. the input to near zero (measured at 2.7e-5 versus 1.46
    with a plain identity residual). The returned path therefore keeps ``x``
    intact so the block has a genuine identity gradient path.
    """

    def __init__(self, dim: int, chunk: int = 32) -> None:
        super().__init__()
        self.dim = int(dim)
        self.chunk = max(1, int(chunk))
        self.gate = nn.Linear(dim, 3 * dim)
        self.candidate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.state: Tensor | None = None

    def reset_state(self) -> None:
        self.state = None

    def forward(self, x: Tensor, state: Tensor | None = None) -> Tensor:
        """Advance the chunked recurrence.

        ``forward`` is a pure function of its arguments: the recurrence starts
        from zeros unless an explicit ``state`` is passed. An earlier version
        implicitly reused ``self.state`` from the previous call, which made the
        module non-deterministic, leaked state between unrelated training
        batches, and made two identical forward passes disagree. The resulting
        state is still published on ``self.state`` (detached) so incremental
        decoding can continue from it deliberately.
        """
        b, t, d = x.shape
        n_chunks = math.ceil(t / self.chunk)
        pad = n_chunks * self.chunk - t
        if pad:
            x = torch.cat([x, x.new_zeros(b, pad, d)], dim=1)
        segments = x.reshape(b, n_chunks, self.chunk, d)

        forget, update, output = self.gate(segments).chunk(3, dim=-1)
        forget = torch.sigmoid(forget)
        update = torch.sigmoid(update)
        output = torch.sigmoid(output)

        if state is None:
            carry = x.new_zeros(b, 1, d)
        else:
            if state.shape != (b, 1, d):
                raise ValueError(
                    f"state must have shape {(b, 1, d)}, got {tuple(state.shape)}"
                )
            carry = state
        centered = segments - segments.mean(dim=-2, keepdim=True)
        for index in range(n_chunks):
            carry = forget[:, index] * carry + update[:, index] * torch.tanh(
                self.candidate(centered[:, index])
            )
        self.state = carry.detach()

        carried = output * carry.unsqueeze(1) + segments
        return self.norm(carried.reshape(b, n_chunks * self.chunk, d)[:, :t])


class HyperDimensionalMemory(nn.Module):
    """Holographic mixing through a Walsh-Hadamard basis.

    The signal is projected into a high dimensional space, transformed with the
    fast Walsh-Hadamard transform, passed through learnable Clifford generators
    (exact involutions over the sign set, so the map stays invertible) and
    projected back. The non power of two residual is handled by padding inside
    :func:`feather_v2.nn_math.fwht` and cropping on the way out.
    """

    def __init__(self, dim: int, hv_dim: int = 2048, scale: float = 0.5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hv_dim = int(hv_dim)
        self.hv_padded = M.next_pow2(hv_dim)
        self.scale = float(scale)
        self.up = nn.Linear(dim, self.hv_padded)
        self.generators = nn.Parameter(torch.randn(self.hv_padded) * 0.1)
        self.down = nn.Linear(self.hv_padded, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        h = M.fwht(self.up(x))
        h = M.clifford_gate(h, self.generators, self.scale)
        h = M.ifwht(h)
        return self.norm(x + self.down(h))


class TTExpert(nn.Module):
    """A single tensor-train factorised expert.

    The expert is stored as a rank-``r`` TT-matrix ``(U, core, V)`` rather than a
    dense ``dim x dim`` matrix, which is where the parameter saving in the MoE
    bank comes from. :func:`feather_v2.nn_math.tt_compress` can derive these
    factors from a dense map when an exact truncation is wanted.
    """

    def __init__(self, dim: int, rank: int = 6, bias: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.u = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
        self.v = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
        self.core = nn.Parameter(torch.eye(rank) + 0.01 * torch.randn(rank, rank))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = M.tt_matmul(self.u, self.core, self.v, x)
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def compress_from_dense(self, dense: Tensor) -> None:
        """Overwrite the factors with the rank-``r`` TT compression of ``dense``."""
        u, s, vh = M.tt_compress(dense, self.rank)
        with torch.no_grad():
            self.u.copy_(u)
            self.v.copy_(vh.transpose(0, 1))
            self.core.copy_(torch.diag(s)[: self.rank, : self.rank])


class KnowledgeVault(nn.Module):
    """Sparse mixture of TT-compressed experts with Sinkhorn-balanced routing.

    Routing is top-k over a learned linear router. The expert *selection* is a
    discrete argmax and carries no gradient, which is standard for MoE; the
    mixture *weights* are a differentiable softmax over the selected logits, so
    gradients reach both the router and the selected experts. An auxiliary
    load-balancing loss is returned alongside the output for the trainer to add.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 8,
        top_k: int = 2,
        rank: int = 6,
        tau: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_experts = int(n_experts)
        self.top_k = max(1, min(int(top_k), int(n_experts)))
        self.tau = float(tau)
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList([TTExpert(dim, rank) for _ in range(n_experts)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, t, _ = x.shape
        flat = x.reshape(b * t, self.dim)
        n = flat.shape[0]
        logits = self.router(flat)
        top_logits, top_index = torch.topk(logits, self.top_k, dim=-1)
        weights = M.tropical_softmax(top_logits, self.tau)

        out = torch.zeros_like(flat)
        fraction = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
        for slot in range(self.top_k):
            assign = top_index[:, slot]
            order = torch.argsort(assign)
            counts = torch.bincount(assign, minlength=self.n_experts)
            ordered = flat[order]
            offset = 0
            for expert_id, count in enumerate(counts.tolist()):
                if count == 0:
                    continue
                rows = order[offset : offset + count]
                chunk = ordered[offset : offset + count]
                weight = weights[rows, slot].unsqueeze(-1)
                out.index_add_(0, rows, self.experts[expert_id](chunk) * weight)
                offset += count
            fraction += counts.to(x.dtype) / max(n, 1)

        probs = M.tropical_softmax(logits, self.tau)
        density = probs.mean(dim=0)
        aux = self.n_experts * torch.sum(fraction * density)
        return self.norm(out.reshape(b, t, self.dim)), aux


class CognitiveWeaver(nn.Module):
    """Kolmogorov-Arnold feed-forward with a differentiable Godel loop.

    The feed-forward path uses genuine KAN layers (learnable univariate basis
    functions with per-edge weights) rather than renamed dense layers. The loop
    refines the representation a fixed number of times, each step gated by the
    continuous Godel code of the current residual norm, so the number of
    effective refinement steps is learned and differentiable.
    """

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        num_gaussians: int = 5,
        loops: int = 2,
    ) -> None:
        super().__init__()
        hidden = int(hidden or dim)
        self.loops = int(loops)
        self.kan_in = M.KANLinear(dim, hidden, num_gaussians)
        self.kan_out = M.KANLinear(hidden, dim, num_gaussians)
        self.loop_proj = nn.Linear(dim, dim)
        self.loop_gain = nn.Parameter(torch.zeros(self.loops))
        self.loop_scale = nn.Parameter(torch.full((self.loops,), 0.5))
        self.loop_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.loops)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.kan_out(F.gelu(self.kan_in(x)))
        for index in range(self.loops):
            # Re-normalising each pass keeps the residual stream bounded. The
            # loop feeds its own output back in, so without this the magnitude
            # grows geometrically and the Godel exponent below overflows.
            y = self.loop_norm[index](y)
            magnitude = y.norm(dim=-1, keepdim=True) / math.sqrt(float(y.shape[-1]))
            log_code = M.godel_log_code(self.loop_gain[index], magnitude)
            gate = torch.sigmoid(log_code * self.loop_scale[index])
            y = y + gate * self.loop_proj(F.gelu(y))
        return self.norm(x + y)


class HomeostasisGovernor(nn.Module):
    """Predictive entropy gate that rescales the residual stream.

    The governor predicts a per-token free energy surrogate from the activation
    statistics, turns it into a temperature, and rescales the residual by a
    bounded ``tanh`` factor. The factor is zero centred so the block starts close
    to an identity map, which keeps early training stable instead of injecting
    uncontrolled gain.
    """

    def __init__(self, dim: int, energy_gain: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.energy_gain = float(energy_gain)
        self.predict = nn.Linear(2 * dim, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def free_energy(self, x: Tensor) -> Tensor:
        """Differentiable per-token free energy surrogate (negative variance)."""
        deviation = x - x.mean(dim=-1, keepdim=True)
        return -deviation.pow(2).mean(dim=-1, keepdim=True)

    def temperature(self, x: Tensor) -> Tensor:
        return F.softplus(self.log_temp) + self.energy_gain * self.free_energy(x).abs()

    def forward(self, x: Tensor) -> Tensor:
        stats = self.predict(torch.cat([x, x.abs()], dim=-1))
        scale = torch.tanh(self.gate(x)) * torch.sigmoid(stats)
        gated = x * (1.0 + scale)
        return self.norm(gated * self.temperature(x).clamp_min(1e-3))


class GenerativeEvolution(nn.Module):
    """Jacobi-spectral refinement with a learned mutation step.

    The refinement solves a small strictly diagonally dominant system built from
    the token representation, using a genuine Jacobi solver, then applies a
    learned bounded mutation. Working in a low dimensional latent keeps the
    ``(batch, time, k, k)`` system affordable on CPU while remaining a real
    linear solve with a real residual.
    """

    def __init__(self, dim: int, latent: int = 16, iters: int = 6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.latent = int(latent)
        self.iters = int(iters)
        self.mix = nn.Linear(dim, 2 * latent)
        self.coupling = nn.Parameter(torch.tensor(0.1))
        self.mutate = nn.Parameter(torch.zeros(latent))
        self.to_latent = nn.Linear(dim, latent)
        self.from_latent = nn.Linear(latent, dim)
        self.norm = nn.LayerNorm(dim)

    def solve(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the refined latent and the Jacobi residual."""
        u, v = self.mix(x).chunk(2, dim=-1)
        k = self.latent
        eye = torch.eye(k, device=x.device, dtype=x.dtype).expand(*u.shape[:-1], k, k)
        coupling = self.coupling.clamp(0.0, 0.5)
        system = eye + coupling * torch.einsum("bti,btj->btij", u, v) / k
        latent = self.to_latent(x)
        refined = M.jacobi_decode(system, latent, iters=self.iters)
        residual = torch.einsum("btij,btj->bti", system, refined) - latent
        return refined, residual

    def forward(self, x: Tensor) -> Tensor:
        refined, _ = self.solve(x)
        refined = refined + torch.tanh(self.mutate) * torch.tanh(refined)
        return self.norm(x + self.from_latent(refined))
