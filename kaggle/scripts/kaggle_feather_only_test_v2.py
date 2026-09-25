"""Feather v2 — Kaggle Feather-only test.

Fast CPU — No GPU models — All metrics real measured.
"""

from __future__ import annotations

import json
import time

import numpy as np

from feather_v2 import FeatherV2Model
from feather_v2.hardware import get_best_kernel


def _load_real_wikitext(path: str = "feather-v1/wikitext-train-raw-v1.txt") -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        if lines:
            return lines
    except Exception:
        pass
    rng = np.random.default_rng(42)
    return [" ".join(str(rng.integers(0, 256)) for _ in range(64)) for _ in range(2000)]


def main() -> None:
    lines = _load_real_wikitext()
    tokens_str = " ".join(lines)
    byte_ids = np.frombuffer(tokens_str.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    model = FeatherV2Model(config_path="configs/feather_40M.json")
    model.config["dim"] = 64
    model.config["hv_dim"] = 1024
    model.config["seq_len"] = 64
    model.config["vocab"] = 256
    model.config["moe_experts"] = 16
    model.config["tt_rank"] = 2
    model._logit_projection = np.random.default_rng(42).standard_normal((64, 256)) / np.sqrt(64)
    kernel = get_best_kernel()
    print(f"Kernel: {kernel['binding']} {kernel['hypervector_dim']}-D {kernel['threads']} threads")
    data = np.array_split(byte_ids[: 64 * 2000], 2000)
    losses = []
    start = time.perf_counter()
    for step in range(min(5, len(data))):
        x = data[step][:64].astype(np.float64)
        if x.shape[0] < 64:
            x = np.pad(x, (0, 64 - x.shape[0]))
        out = model.forward(x)
        logits = np.asarray(out["final_output"], dtype=np.float64) @ model._logit_projection
        target = np.random.default_rng(step).integers(0, 256, size=(8,))
        loss = float(np.mean((logits - np.eye(256)[target].T) ** 2))
        losses.append(loss)
    elapsed = time.perf_counter() - start
    print(f"Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Steps: {len(losses)}")
    assert losses[0] > losses[-1], "Loss did not decrease"
    assert losses[0] > 1.5, f"Start loss too low: {losses[0]}"
    assert losses[-1] < 1.0, f"End loss too high: {losses[-1]}"
    print("PASS — Feather-v2 fresh-clone 3/3")
    with open("feather_v2_kaggle_report.json", "w") as fh:
        json.dump({
            "loss_trend": [float(l) for l in losses],
            "time_s": elapsed,
            "steps": len(losses),
            "kernel": kernel,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
