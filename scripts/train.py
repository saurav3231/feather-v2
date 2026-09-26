"""Real CPU training loop for Feather v2.

This trains the model end to end with next-token cross-entropy on real token
ids, which the previous benchmark did not do: it fitted only the logit
projection against a target vector with mean squared error and passed whenever
the loss failed to get 5% worse.

Everything reported here is measured. The loss curve comes from
``loss.item()`` on the actual objective, the gradient norm from
``parameters().grad``, and the parameter count from ``parameters()``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from feather_v2.model import DEFAULT_CONFIG, FeatherV2Model

SMOKE_CONFIG: dict[str, object] = {
    "dim": 128,
    "n_blocks": 2,
    "vocab": 256,
    "seq_len": 64,
    "hv_dim": 512,
    "chunk": 16,
    "tt_rank": 4,
    "moe_experts": 8,
    "moe_top_k": 2,
    "n_scales": 4,
    "sig_dim": 6,
    "weaver_hidden": 256,
    "weaver_gaussians": 4,
    "weaver_loops": 2,
    "evo_latent": 12,
    "evo_iters": 4,
    "dropout": 0.1,
    "tie_embeddings": True,
    "seed": 42,
}

CORPUS = (
    "the feather engine learns mathematics by gradient descent and by "
    "holographic memory. a walsh hadamard transform mixes hyperdimensional "
    "vectors, a tensor train compresses experts, and a jacobi sweep refines "
    "the representation. the model is trained on cpu with real cross entropy "
    "loss over byte level tokens. "
) * 24


def byte_tokens(text: str) -> torch.Tensor:
    """Encode text as a 1-D tensor of byte values (vocab 256, no dependencies)."""
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def batches(
    tokens: torch.Tensor, batch_size: int, seq_len: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sample fixed-length windows; inputs and labels are shifted by one token."""
    generator = torch.Generator().manual_seed(seed)
    span = seq_len + 1
    usable = tokens.numel() - span
    if usable <= 0:
        raise ValueError(
            f"corpus of {tokens.numel()} tokens is shorter than seq_len {seq_len}"
        )
    starts = torch.randint(0, usable, (batch_size * 8,), generator=generator)
    windows = [tokens[s : s + span] for s in starts.tolist()]
    return [(w[:-1].unsqueeze(0), w[1:].unsqueeze(0)) for w in windows]


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def train(
    config: dict, steps: int, lr: float, seed: int
) -> tuple[dict, FeatherV2Model]:
    """Run real optimization and return the report plus the trained model."""
    torch.manual_seed(seed)
    model = FeatherV2Model(config)
    model.train()

    tokens = byte_tokens(CORPUS)
    data = batches(tokens, batch_size=1, seq_len=int(config["seq_len"]), seed=seed)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.dim() < 2 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )

    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    for step in range(steps):
        inputs, labels = data[step % len(data)]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(inputs, labels)
        loss.backward()
        norm = grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        record = {"step": step, "grad_norm": norm, **metrics}
        history.append(record)
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(
                f"step {step:4d}  loss {metrics['loss']:.4f}  "
                f"ppl {metrics['perplexity']:8.2f}  "
                f"aux {metrics['aux_loss']:.4f}  |g| {norm:8.3f}"
            )

    elapsed = time.perf_counter() - start_time
    first = history[0]["loss"]
    final = history[-1]["loss"]
    return {
        "config": config,
        "parameters": model.count_parameters(),
        "size_label": model.size_label(),
        "steps": steps,
        "learning_rate": lr,
        "seconds": elapsed,
        "first_loss": first,
        "final_loss": final,
        "loss_decreased": final < first,
        "loss_reduction": first - final,
        "initial_perplexity": history[0]["perplexity"],
        "final_perplexity": history[-1]["perplexity"],
        "all_grads_nonzero": all(h["grad_norm"] > 0 for h in history),
        "history": history,
    }, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--out", type=str, default="train_report.json")
    parser.add_argument("--pth", type=str, default="")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    config = dict(SMOKE_CONFIG)
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))

    result, model = train(config, args.steps, args.lr, args.seed)

    print()
    print(f"parameters      : {result['parameters']:,} ({result['size_label']})")
    print(f"loss            : {result['first_loss']:.4f} -> {result['final_loss']:.4f}")
    print(
        f"perplexity      : {result['initial_perplexity']:.2f} -> "
        f"{result['final_perplexity']:.2f}"
    )
    print(f"loss decreased  : {result['loss_decreased']}")
    print(f"grads nonzero   : {result['all_grads_nonzero']}")
    print(f"wall clock      : {result['seconds']:.1f}s")

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    if args.pth:
        # Save the model that was just trained. This previously constructed a
        # fresh untrained model here and wrote that instead, so the flag
        # produced a plausible-looking checkpoint containing random weights.
        size = model.save_pth(args.pth)
        print(f"wrote {args.pth} ({size / 1024:.1f} KiB)")

        manifest = Path(args.pth).with_suffix(".json")
        manifest.write_text(
            json.dumps(
                {
                    **model.describe(),
                    "trained_steps": args.steps,
                    "learning_rate": args.lr,
                    "final_loss": result["final_loss"],
                    "final_perplexity": result["final_perplexity"],
                    "checkpoint_bytes": size,
                    "corpus": "scripts/train.py built-in seed corpus",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {manifest}")

    if not result["loss_decreased"]:
        raise SystemExit("FAIL: loss did not decrease")
    if not result["all_grads_nonzero"]:
        raise SystemExit("FAIL: some step had a zero gradient norm")


if __name__ == "__main__":
    main()
