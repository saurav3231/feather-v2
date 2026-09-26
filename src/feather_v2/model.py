"""Feather v2 — the trainable end-to-end model.

This replaces the previous NumPy scaffold, which returned debug dictionaries
from ``encode()`` and had no parameters, no ``backward()`` and no logits, so the
"trainable end to end" description could not be true of it.

:class:`FeatherV2Model` is a real :class:`torch.nn.Module`. It maps integer
token ids to next-token logits, has a real cross-entropy objective, a real
parameter count taken from ``parameters()``, and trains end to end with
``loss.backward()``.

Each :class:`CognitiveBlock` wires all seven components as pre-norm residual
sublayers. Pre-norm with an explicit identity residual is deliberate: it gives
every sublayer a clean gradient path, which the earlier centred-residual
variant lacked.
"""

from __future__ import annotations

import difflib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import nn_math as M
from .hardware import get_best_kernel, summary
from .nn_components import (
    CognitiveWeaver,
    GenerativeEvolution,
    HomeostasisGovernor,
    HyperDimensionalMemory,
    KnowledgeVault,
    LiquidMemory,
    SensoryEncoder,
)

__all__ = ["CognitiveBlock", "FeatherV2Model", "load_config"]

DEFAULT_CONFIG: dict[str, Any] = {
    "dim": 512,
    "n_blocks": 2,
    "vocab": 8256,
    "seq_len": 512,
    "hv_dim": 8192,
    "chunk": 32,
    "tt_rank": 6,
    "moe_experts": 96,
    "moe_top_k": 2,
    "n_scales": 4,
    "sig_dim": 8,
    "p_adic_p": 2,
    "p_adic_levels": 8,
    "tau": 0.1,
    "alpha_fractional": 0.7,
    "weaver_hidden": 0,
    "weaver_gaussians": 5,
    "weaver_loops": 2,
    "evo_latent": 16,
    "evo_iters": 6,
    "dropout": 0.25,
    "dropout_q": 0.5,
    "tie_embeddings": True,
    "seed": 42,
}


CONFIG_ALIASES: dict[str, str] = {
    "layers": "n_blocks",
    "num_layers": "n_blocks",
    "n_layer": "n_blocks",
    "hidden": "dim",
    "hidden_dim": "dim",
    "d_model": "dim",
    "hypervector_dim": "hv_dim",
    "experts": "moe_experts",
    "top_k": "moe_top_k",
    "rank": "tt_rank",
    "tau_tropical": "tau",
    "p_adic_levels_count": "p_adic_levels",
}


def resolve_config_aliases(
    supplied: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Rewrite known former key names, reporting what was changed.

    ``layers`` -> ``n_blocks`` is the important one: every shipped config used
    ``layers``, the model reads ``n_blocks``, and the mismatch was silent, so a
    config advertising 16 layers quietly built 2.
    """
    notes: list[str] = []
    resolved: dict[str, Any] = {}
    for key, value in supplied.items():
        if key in CONFIG_ALIASES:
            target = CONFIG_ALIASES[key]
            if target in supplied:
                notes.append(f"dropped {key!r} (conflicts with {target!r})")
                continue
            notes.append(f"renamed {key!r} -> {target!r}")
            resolved[target] = value
        else:
            resolved[key] = value
    return resolved, notes


def load_config(path: str | Path, strict: bool = True) -> dict[str, Any]:
    """Read a JSON config and fill in any missing defaults.

    Known former key names are rewritten with a note. Genuinely unknown keys
    are rejected by default, because a key that is never read is worse than a
    missing one: it looks like it took effect.
    """
    with open(path, encoding="utf-8") as handle:
        supplied = json.load(handle)
    if not isinstance(supplied, dict):
        raise ValueError(f"{path}: config must be a JSON object")

    supplied, notes = resolve_config_aliases(supplied)
    unknown = sorted(set(supplied) - set(DEFAULT_CONFIG))
    if unknown and strict:
        suggestions = {
            key: difflib.get_close_matches(key, DEFAULT_CONFIG, n=1, cutoff=0.6)
            for key in unknown
        }
        hints = ", ".join(
            f"{key} -> {match[0]!r}" for key, match in suggestions.items() if match
        )
        raise ValueError(
            f"{path}: unknown config key(s) {unknown}."
            + (f" Did you mean {hints}?" if hints else "")
            + f" Valid keys: {sorted(DEFAULT_CONFIG)}"
        )

    config = dict(DEFAULT_CONFIG)
    config.update(supplied)
    for note in notes:
        warnings.warn(f"{path}: {note}", stacklevel=2)
    return config


class CognitiveBlock(nn.Module):
    """One pre-norm block containing all seven trainable components.

    Every sublayer is wrapped in its own ``LayerNorm`` and added through an
    identity residual, so no sublayer can block the gradient path of the ones
    around it. The mixture-of-experts auxiliary load-balancing loss is returned
    alongside the activations for the trainer to fold into the objective.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dim = int(config["dim"])
        self.norm_sensory = nn.LayerNorm(dim)
        self.sensory = SensoryEncoder(
            dim,
            n_scales=int(config["n_scales"]),
            sig_dim=int(config["sig_dim"]),
            p=int(config["p_adic_p"]),
            levels=int(config["p_adic_levels"]),
            tau=float(config["tau"]),
        )
        self.norm_liquid = nn.LayerNorm(dim)
        self.liquid = LiquidMemory(dim, chunk=int(config["chunk"]))
        self.norm_hyper = nn.LayerNorm(dim)
        self.hyper = HyperDimensionalMemory(dim, hv_dim=int(config["hv_dim"]))
        self.norm_vault = nn.LayerNorm(dim)
        self.vault = KnowledgeVault(
            dim,
            n_experts=int(config["moe_experts"]),
            top_k=int(config["moe_top_k"]),
            rank=int(config["tt_rank"]),
            tau=float(config["tau"]),
        )
        self.norm_weaver = nn.LayerNorm(dim)
        self.weaver = CognitiveWeaver(
            dim,
            hidden=int(config["weaver_hidden"]) or dim,
            num_gaussians=int(config["weaver_gaussians"]),
            loops=int(config["weaver_loops"]),
        )
        self.norm_governor = nn.LayerNorm(dim)
        self.governor = HomeostasisGovernor(dim)
        self.norm_evolution = nn.LayerNorm(dim)
        self.evolution = GenerativeEvolution(
            dim, latent=int(config["evo_latent"]), iters=int(config["evo_iters"])
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x + self.sensory(self.norm_sensory(x))
        x = x + self.liquid(self.norm_liquid(x))
        x = x + self.hyper(self.norm_hyper(x))
        vaulted, aux = self.vault(self.norm_vault(x))
        x = x + vaulted
        x = x + self.weaver(self.norm_weaver(x))
        x = x + self.governor(self.norm_governor(x))
        x = x + self.evolution(self.norm_evolution(x))
        return x, aux


class FeatherV2Model(nn.Module):
    """Trainable Feather v2 language model.

    Parameters
    ----------
    config:
        Configuration mapping. Missing keys fall back to
        :data:`DEFAULT_CONFIG`. A config path may be passed instead and is
        loaded with :func:`load_config`.
    """

    def __init__(self, config: dict[str, Any] | str | Path | None = None) -> None:
        super().__init__()
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config: dict[str, Any] = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        torch.manual_seed(int(self.config["seed"]))

        dim = int(self.config["dim"])
        vocab = int(self.config["vocab"])
        seq_len = int(self.config["seq_len"])

        self.embed = nn.Embedding(vocab, dim)
        self.pos_embed = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [CognitiveBlock(self.config) for _ in range(int(self.config["n_blocks"]))]
        )
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if bool(self.config["tie_embeddings"]):
            self.head.weight = self.embed.weight
        self.dropout_p = float(self.config["dropout"])
        self.dropout_q = float(self.config["dropout_q"])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Normal (0.02) initialisation, with residual projections scaled down."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if not bool(self.config["tie_embeddings"]):
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.head:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, return_aux: bool = False) -> Tensor | tuple:
        """Map token ids to next-token logits.

        Parameters
        ----------
        input_ids:
            ``(batch, time)`` integer tensor of token ids.
        return_aux:
            Also return the summed mixture-of-experts balancing loss.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be (batch, time), got {tuple(input_ids.shape)}"
            )
        batch, time = input_ids.shape
        seq_len = int(self.config["seq_len"])
        if time > seq_len:
            raise ValueError(
                f"sequence length {time} exceeds configured seq_len {seq_len}"
            )

        positions = torch.arange(time, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions)
        x = M.alpha_dropout(x, self.dropout_p, self.dropout_q, self.training)

        aux = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x)
            aux = aux + block_aux

        x = self.norm_f(x)
        x = M.alpha_dropout(x, self.dropout_p, self.dropout_q, self.training)
        logits = self.head(x)
        if return_aux:
            return logits, aux
        return logits

    def loss(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        aux_weight: float = 0.01,
    ) -> tuple[Tensor, dict[str, float]]:
        """Next-token cross-entropy plus the MoE balancing term.

        Returns the scalar loss and a metrics dictionary containing the plain
        language-modelling loss, the perplexity and the auxiliary term, so a
        training log can never confuse the total with the language loss.
        """
        logits, aux = self.forward(input_ids, return_aux=True)
        if labels is None:
            labels = input_ids
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        total = lm_loss + aux_weight * aux
        with torch.no_grad():
            metrics = {
                "loss": float(lm_loss.item()),
                "total_loss": float(total.item()),
                "perplexity": float(math.exp(min(lm_loss.item(), 20.0))),
                "aux_loss": float(aux.item()),
            }
        return total, metrics

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Actual parameter count from ``parameters()``.

        Shared tensors (tied embeddings) are counted once because
        ``parameters()`` de-duplicates by default. The previous implementation
        hand-summed NumPy array sizes, which cannot detect an untied or
        accidentally duplicated tensor.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def parameter_breakdown(self) -> dict[str, int]:
        """Parameter count per top-level submodule, largest first.

        Shared tensors are attributed to the first submodule that reports them,
        so the breakdown sums to exactly :meth:`count_parameters` even when the
        embeddings are tied.
        """
        counts: dict[str, int] = {}
        seen: set[int] = set()
        for name, module in self.named_children():
            total = 0
            for param in module.parameters():
                if id(param) in seen:
                    continue
                seen.add(id(param))
                total += param.numel()
            counts[name] = total
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def size_label(self) -> str:
        """Human readable size derived from the real parameter count."""
        millions = self.count_parameters() / 1e6
        if millions < 1:
            return f"{self.count_parameters() / 1e3:.0f}K"
        return f"{millions:.2f}M"

    def num_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        """Serialized weight size for ``dtype``, counting shared tensors once.

        ``dtype`` is honoured: passing ``torch.float16`` returns half the float32
        size. The previous implementation always used each parameter's current
        ``element_size()`` and ignored the argument, so ``num_bytes(torch.float16)``
        and ``num_bytes()`` returned the same number.

        This is an arithmetic size, not a file. A real float16 checkpoint has not
        been produced or measured here.
        """
        seen: set[int] = set()
        itemsize = torch.empty(0, dtype=dtype).element_size()
        total = 0
        for param in self.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel() * itemsize
        return total

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> Tensor:
        """Autoregressively sample continuations.

        Runs under ``no_grad`` and uses discrete sampling, which is correct for
        inference and deliberately kept out of the training graph. Alpha-dropout
        is disabled because the model is put in eval mode.
        """
        was_training = self.training
        self.eval()
        try:
            generated = input_ids
            for _ in range(int(max_new_tokens)):
                window = generated[:, -int(self.config["seq_len"]) :]
                logits = self.forward(window)[:, -1, :]
                if not greedy and temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
                else:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, nxt], dim=1)
            return generated
        finally:
            self.train(was_training)

    def save_pth(self, path: str | Path) -> int:
        """Save weights plus config with ``torch.save``; returns bytes written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "feather-v2-pth-1",
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)
        return path.stat().st_size

    @classmethod
    def from_pth(
        cls, path: str | Path, config: dict[str, Any] | None = None
    ) -> "FeatherV2Model":
        """Rebuild a model from :meth:`save_pth` output."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(config or payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model

    def describe(self) -> dict[str, Any]:
        """Measured facts about this model, with no estimates or targets.

        Every number here is read back off the live module. The point is that
        a caller can print this and get the truth, rather than a name that was
        chosen before the weights existed.
        """
        breakdown = self.parameter_breakdown()
        dtypes = {p.dtype for p in self.parameters()}
        return {
            "parameters": self.count_parameters(),
            "parameters_trainable": self.count_parameters(trainable_only=True),
            "parameter_breakdown": breakdown,
            "largest_group": max(breakdown, key=breakdown.get) if breakdown else None,
            "size_label": self.size_label(),
            "weight_bytes": self.num_bytes(),
            "dtypes": sorted(str(d) for d in dtypes),
            "tied_embeddings": bool(self.config["tie_embeddings"]),
            "n_blocks": len(self.blocks),
            "config": dict(self.config),
        }

    def hardware_summary(self) -> str:
        return summary()

    def best_kernel(self) -> dict[str, Any]:
        return get_best_kernel()

    def benchmark_forward(
        self, input_ids: Tensor, warmup: int = 2, runs: int = 10
    ) -> dict[str, float]:
        """Measure real forward throughput with warmup and median timing."""
        self.eval()
        with torch.no_grad():
            for _ in range(warmup):
                self.forward(input_ids)
            samples = []
            for _ in range(runs):
                start = time.perf_counter()
                self.forward(input_ids)
                samples.append(time.perf_counter() - start)
        samples.sort()
        tokens = int(input_ids.numel())
        median = samples[len(samples) // 2]
        return {
            "median_seconds": median,
            "tokens": tokens,
            "tokens_per_second": tokens / median if median > 0 else 0.0,
            "best_seconds": samples[0],
            "worst_seconds": samples[-1],
        }

    @torch.no_grad()
    def state_stability(self, seq_len: int | None = None) -> dict[str, Any]:
        """Measure how much a suffix perturbs an earlier representation.

        The same token prefix is encoded twice: once on its own, and once as the
        start of a longer sequence. The hidden state at the *same position* of the
        prefix is captured in both cases via a forward hook, and the cosine
        similarity between them is returned.

        This is deliberately not called recall, and it is not a long-context
        measurement. It says how similar one position's representation is when
        later tokens are present versus absent.

        Comparing final *logits* instead would mix a vocab-wide vector into a
        representation-similarity claim, and comparing the last position of a short
        prefix against the last position of a long sequence would compare two
        different positions. An earlier version of this check did both and
        reported a vocab width as if it were a representation width.

        Returns ``{"similarity": None, "hidden_dim": 0}`` when no comparable
        hidden state can be captured.
        """
        empty: dict[str, Any] = {"similarity": None, "hidden_dim": 0}
        try:
            vocab = int(self.config["vocab"])
            seq = min(
                int(seq_len or self.config["seq_len"]), int(self.config["seq_len"])
            )
            if seq < 4:
                return empty
        except Exception:
            return empty

        captured: dict[str, Tensor] = {}

        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured["h"] = tensor.detach()

        was_training = self.training
        handle = self.norm_f.register_forward_hook(_hook)
        try:
            generator = torch.Generator().manual_seed(99)
            ids = torch.randint(0, vocab, (1, seq), generator=generator)
            third = max(1, seq // 3)
            self.eval()
            self(ids)  # long: the prefix followed by a suffix
            long_state = captured["h"][:, third - 1]
            self(ids[:, :third])  # short: the prefix alone
            short_state = captured["h"][:, third - 1]
        except Exception:
            return empty
        finally:
            handle.remove()
            if was_training:
                self.train()

        similarity = float(
            F.cosine_similarity(
                long_state.flatten(), short_state.flatten(), dim=-1
            ).mean()
        )
        return {
            "similarity": similarity,
            "hidden_dim": int(short_state.shape[-1]),
            "prefix_positions": third,
        }


def _main() -> int:
    """Count a config's real parameters: ``python -m feather_v2.model --config ...``.

    The count is a sum over ``p.numel()``, so it is the number of scalars the
    model actually allocates. No estimate, no label attached by hand.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m feather_v2.model")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--count-params",
        action="store_true",
        help="print the measured parameter count and exit",
    )
    args = parser.parse_args()

    model = FeatherV2Model(load_config(args.config))
    count = model.count_parameters()
    if args.count_params:
        print(f"{count:,} params ({count / 1e6:.2f}M)")
        print(f"dim={model.config['dim']} n_blocks={model.config['n_blocks']}")
        return 0
    print(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
