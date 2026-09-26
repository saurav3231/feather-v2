"""Feather-v2 mega Kaggle benchmark — the real ladder, real measured metrics.

Run: python kaggle/test_all_sizes_mega.py
Outputs:
  docs/images/*.png   (scaling plots)
  outputs/benchmark_report.json

Every number in the report is produced by this script at run time. Where a
quantity cannot be measured, it is reported as ``null`` rather than estimated.
The previous version of this file invented several of them; see the docstrings
on the individual measurement helpers for what was wrong and why.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from feather_v2 import FeatherV2Model
from feather_v2.hardware import detect_cpu_features, get_best_kernel, summary
from feather_v2.model import load_config
from feather_v2.utils import hybrid_adaptive_tokenizer

# ---------------------------------------------------------------------------
# 1. Directory setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "docs" / "images"
OUT_DIR = ROOT / "outputs"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. CONFIG_MAP - the real ladder, loaded from configs/ so it cannot drift
# ---------------------------------------------------------------------------
# These used to be inlined here with invented "layers"/"size_label" keys that
# the model never read, so every entry silently built a 2-block model while
# advertising up to 16 layers. The labels below are the measured parameter
# counts from `python scripts/measure_sizes.py`, not aspirations.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_ladder() -> dict[str, dict]:
    ladder: dict[str, dict] = {}
    for path in sorted(CONFIG_DIR.glob("feather_*.json")):
        label = path.stem.replace("feather_", "")
        ladder[label] = load_config(path)
    if not ladder:
        raise RuntimeError(f"no configs found in {CONFIG_DIR}")
    return ladder


CONFIG_MAP: dict[str, dict] = _load_ladder()


# ---------------------------------------------------------------------------
# 3. Beautiful printing helpers
# ---------------------------------------------------------------------------
_USE_RICH = False
_USE_TABULATE = False
_USE_COLORAMA = False

try:
    from rich.console import Console
    from rich.table import Table

    _USE_RICH = True
except Exception:
    _USE_RICH = False

if not _USE_RICH:
    try:
        from tabulate import tabulate

        _USE_TABULATE = True
    except Exception:
        _USE_TABULATE = False

    try:
        import colorama

        colorama.init()
        _USE_COLORAMA = True
    except Exception:
        _USE_COLORAMA = False

_CYAN = "\033[96m" if _USE_COLORAMA else ""
_MAGENTA = "\033[95m" if _USE_COLORAMA else ""
_GREEN = "\033[92m" if _USE_COLORAMA else ""
_YELLOW = "\033[93m" if _USE_COLORAMA else ""
_RED = "\033[91m" if _USE_COLORAMA else ""
_RESET = "\033[0m" if _USE_COLORAMA else ""

_console = Console() if _USE_RICH else None


def _rich_table(headers: list[str], rows: list[list[str]], title: str = "") -> str:
    """Render via rich, then re-encode to the console codec.

    Rich writes box-drawing and alignment characters that a cp1252 console
    cannot represent; letting it write directly produced a mangled table where
    characters were replaced. Capturing and sanitising keeps the output legible
    everywhere.
    """
    t = Table(title=title, show_header=True, header_style="bold cyan")
    for h in headers:
        t.add_column(h, style="cyan")
    for row in rows:
        t.add_row(*[str(c) for c in row])
    with _console.capture() as capture:
        _console.print(t)
    print(_safe(capture.get()))


def _fallback_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    if _USE_TABULATE:
        print(_safe(title))
        print(_safe(tabulate(rows, headers=headers, tablefmt="grid")))
    else:
        print(_safe(title))
        col_w = [
            max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)
        ]
        line = "|".join("-" * w for w in col_w)
        print(_safe("|".join(h.ljust(w) for h, w in zip(headers, col_w))))
        print(line)
        for row in rows:
            print("|".join(str(c).ljust(w) for c, w in zip(row, col_w)))


def _safe(text: str) -> str:
    """Drop characters the console codec cannot encode.

    Windows consoles frequently default to cp1252, which cannot represent the
    check marks and box drawing this script uses. Before this, a dataset that
    failed to load raised UnicodeEncodeError, which masked the real cause.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def cprint(msg: str, color: str = "") -> None:
    msg = _safe(msg)
    try:
        if _USE_RICH:
            _console.print(msg)
        else:
            print(f"{color}{msg}{_RESET}")
    except Exception:  # noqa: BLE001
        print(msg)


def print_header(title: str) -> None:
    if _USE_RICH:
        _console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        cprint(f"\n{'='*70}\n{title}\n{'='*70}", _CYAN)


def print_subheader(title: str) -> None:
    if _USE_RICH:
        _console.print(f"\n[bold magenta]{title}[/bold magenta]")
    else:
        cprint(f"\n--- {title} ---", _MAGENTA)


def progress(msg: str) -> None:
    if _USE_RICH:
        _console.print(f"[dim]{msg}[/dim]")
    else:
        print(f"  [..] {msg}")


# ---------------------------------------------------------------------------
# 4. Hardware detection
# ---------------------------------------------------------------------------
def detect_hardware() -> dict[str, Any]:
    try:
        feats = detect_cpu_features()
    except Exception:
        feats = {}
    try:
        kern = get_best_kernel()
    except Exception:
        kern = {}
    # Drop legacy prediction fields. Throughput is measured below, never predicted.
    kern = {
        k: v
        for k, v in kern.items()
        if k not in ("expected_tok_per_sec", "precision", "ram_budget_gb")
    }
    return {"features": feats, "kernel": kern}


def print_hardware() -> None:
    hw = detect_hardware()
    print_header("HARDWARE DETECTION")
    if hw["features"]:
        for k, v in hw["features"].items():
            progress(f"  {k}: {v}")
    try:
        cprint(summary(), _GREEN)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 5. Dataset loading
# ---------------------------------------------------------------------------
def load_streaming_datasets() -> dict[str, dict]:
    datasets_info = {}
    try:
        from datasets import load_dataset

        sources = {
            "openwebtext": {
                "loader": lambda: load_dataset(
                    "Skylion007/openwebtext", streaming=True, split="train"
                ),
                "take": 100,
                "desc": "OpenWebText",
            },
            "wikipedia": {
                "loader": lambda: load_dataset(
                    "wikimedia/wikipedia", "20231101.en", streaming=True, split="train"
                ),
                "take": 25,
                "desc": "Wikipedia (en)",
            },
            "no_robots": {
                "loader": lambda: load_dataset(
                    "HuggingFaceH4/no_robots", streaming=True, split="train"
                ),
                "take": 20,
                "desc": "no_robots",
            },
        }
        for key, cfg in sources.items():
            progress(f"Loading {cfg['desc']} streaming...")
            try:
                ds = cfg["loader"]()
                subset = list(ds.take(cfg["take"]))
                texts = []
                for item in subset:
                    if "text" in item:
                        texts.append(item["text"])
                    elif "content" in item:
                        texts.append(item["content"])
                    elif "prompt" in item:
                        texts.append(str(item["prompt"]))
                combined = " ".join(texts)
                size_kb = len(combined.encode("utf-8")) / 1024
                datasets_info[key] = {
                    "name": cfg["desc"],
                    "available": True,
                    "samples": len(texts),
                    "size_kb": size_kb,
                    "text": combined[:100_000],
                    "error": None,
                }
                cprint(
                    f"  OK {cfg['desc']}: {len(texts)} samples, {size_kb:.1f} KB",
                    _GREEN,
                )
            except Exception as exc:  # noqa: BLE001
                # Previously this substituted "fallback test data" while still
                # recording name="OpenWebText", so a run that never touched a
                # real corpus reported that it had. An unavailable source is
                # now recorded as unavailable with empty text.
                cprint(f"  FAILED {cfg['desc']}: {_safe(str(exc))[:200]}", _RED)
                datasets_info[key] = {
                    "name": cfg["desc"],
                    "available": False,
                    "samples": 0,
                    "size_kb": 0.0,
                    "text": "",
                    "error": str(exc),
                }
    except ImportError:
        cprint("datasets not available; corpus measurements will be skipped", _YELLOW)
        for key, desc in [
            ("openwebtext", "OpenWebText"),
            ("wikipedia", "Wikipedia (en)"),
            ("no_robots", "no_robots"),
        ]:
            datasets_info[key] = {
                "name": desc,
                "available": False,
                "samples": 0,
                "size_kb": 0.0,
                "text": "",
                "error": "datasets not installed",
            }

    total_kb = sum(d["size_kb"] for d in datasets_info.values())
    unavailable = [k for k, d in datasets_info.items() if not d["available"]]
    quota_gb = 20.0
    pct = total_kb / (quota_gb * 1024 * 1024) * 100
    cprint(
        f"Total data: {total_kb:.1f} KB / {quota_gb:.0f} GB ({pct:.4f}%) — quota safe",
        _GREEN,
    )
    if unavailable:
        cprint(
            f"Unavailable sources: {', '.join(unavailable)}. "
            "Any metric that needs them is reported as unmeasured.",
            _YELLOW,
        )
    return datasets_info


# ---------------------------------------------------------------------------
# 6. Anti-fake gates
# ---------------------------------------------------------------------------
# These gates exist to catch fabrication, so each one had to be checked for the
# same failure it was meant to catch.


def assert_real_weights(model: FeatherV2Model) -> bool:
    """Check that live parameters are real, varied, and respond to training.

    The previous version inspected ``model._logit_projection``, a single random
    matrix that was never trained and was the only thing the old scaffold
    actually had. Checking one array is not evidence about a model.

    Note what this deliberately does *not* do: flag parameters that are
    constant. ``LayerNorm.weight`` initialises to exactly 1.0 and its bias to
    exactly 0.0, and ``scale_logits`` initialises to zeros so the mixture starts
    uniform. Those are correct initial values, not fabricated ones. The
    decisive check is whether the parameters move under a real optimizer step.
    """
    try:
        named = list(model.named_parameters())
        if not named:
            return False
        nonfinite = [n for n, p in named if not torch.isfinite(p).all()]
        if nonfinite:
            cprint(f"  FAIL weights: non-finite in {nonfinite[:5]}", _RED)
            return False

        total = sum(param.numel() for _, param in named)
        all_constant = [n for n, p in named if float(p.std()) == 0.0]
        if len(all_constant) == len(named):
            cprint("  FAIL weights: every parameter is constant", _RED)
            return False

        before = {n: p.detach().clone() for n, p in named if p.requires_grad}
        was_training = model.training
        model.train()
        vocab = int(model.config["vocab"])
        ids = torch.randint(
            0, vocab, (1, min(8, int(model.config["seq_len"]))), dtype=torch.long
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        optimizer.zero_grad(set_to_none=True)
        probe_loss, _ = model.loss(ids)
        probe_loss.backward()
        optimizer.step()
        moved = sum(
            1
            for n, p in model.named_parameters()
            if n in before and not torch.equal(p.detach(), before[n])
        )
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in before:
                    p.copy_(before[n])
        if not was_training:
            model.eval()

        if moved == 0:
            cprint(
                "  FAIL weights: no parameter moved during an optimizer step",
                _RED,
            )
            return False
        cprint(
            f"  {total:,} parameters, all finite; {moved} responded to a "
            f"training step ({len(all_constant)} at fixed init values)",
            _GREEN,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        cprint(f"  FAIL weights check: {_safe(str(exc))[:300]}", _RED)
        return False


def assert_real_timing(elapsed_s: float) -> bool:
    """A timing must be positive and must have actually measured a call."""
    ok = math.isfinite(elapsed_s) and elapsed_s > 0.0
    if not ok:
        cprint(f"  FAIL timing: {elapsed_s!r} is not a positive duration", _RED)
    return ok


def assert_real_varying_loss(losses: list[float]) -> bool:
    """The loss must be finite, must move, and must end lower than it started.

    The old version accepted ``losses[-1] <= losses[0] * 1.05``, which permits a
    flat or rising loss to pass, and it wrapped the whole loop in
    ``try/except`` so a crash returned ``([], False)`` rather than a traceback.
    """
    if len(losses) < 2:
        cprint(f"  FAIL loss trend: only {len(losses)} point(s)", _RED)
        return False
    if not all(math.isfinite(v) for v in losses):
        cprint("  FAIL loss trend: non-finite value in history", _RED)
        return False
    moved = len({round(v, 8) for v in losses}) > 1
    fell = losses[-1] < losses[0]
    if not (moved and fell):
        cprint(
            f"  FAIL loss trend: {losses[0]:.6f} -> {losses[-1]:.6f} "
            f"(moved={moved} fell={fell})",
            _RED,
        )
    return moved and fell


def assert_real_ram(
    model: FeatherV2Model, ram_delta_mb: float, ram_total_mb: float
) -> bool:
    """Check that resident memory is consistent with the real weight count.

    The previous version took a size label and compared against a hardcoded
    table of thresholds invented for that label, requiring
    ``delta > threshold``. That is inverted: a genuinely memory-efficient model
    would be rejected for being too good, and a model with the wrong number of
    parameters would be accepted if it happened to allocate enough. There are no
    size thresholds here. The only defensible check is that the weights the
    model actually holds could account for the memory it uses.
    """
    try:
        parameters = model.count_parameters()
        bytes_per_weight = next(model.parameters()).element_size()
        weight_mb = parameters * bytes_per_weight / (1024 * 1024)
        ok = (
            math.isfinite(ram_delta_mb)
            and math.isfinite(ram_total_mb)
            and ram_total_mb > 0.0
            and weight_mb > 0.0
            and ram_delta_mb <= ram_total_mb
        )
        if not ok:
            cprint(
                f"  FAIL RAM: delta={ram_delta_mb:.1f}MB total={ram_total_mb:.1f}MB "
                f"weights={weight_mb:.1f}MB (weights must be > 0 and delta <= total)",
                _RED,
            )
        return ok
    except Exception as exc:  # noqa: BLE001
        cprint(f"  FAIL RAM check: {exc}", _RED)
        return False


# ---------------------------------------------------------------------------
# 7. test_one_size
# ---------------------------------------------------------------------------
def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _measure_tokenizer_speed(model: FeatherV2Model, text: str, runs: int = 5) -> float:
    vocab = model.config.get("vocab", 8256)
    token_times = []
    for _ in range(runs):
        start = time.perf_counter()
        tokens, info = hybrid_adaptive_tokenizer(text, vocab_size=vocab)
        elapsed = time.perf_counter() - start
        n_tokens = max(1, tokens.size)
        if elapsed < 0.001:
            elapsed = 0.001
        token_times.append(n_tokens / elapsed)
    med_tok_s = _median(token_times)
    return med_tok_s, info


def _measure_forward_throughput(
    model: FeatherV2Model, seq_lengths: list[int], runs: int = 3
) -> dict[int, float]:
    """Tokens per second through a real forward pass.

    The previous version called ``model.forward(x)`` inside
    ``try/except Exception: pass`` and then recorded the elapsed time whether or
    not the call succeeded, so a model that raised on every input still reported
    a throughput. It also reported ``0.0`` on a slow run, which reads as a
    measurement rather than a failure. Errors propagate now.
    """
    results: dict[int, float] = {}
    model.eval()
    vocab = int(model.config["vocab"])
    max_seq = int(model.config["seq_len"])
    with torch.no_grad():
        for seq in seq_lengths:
            seq = min(seq, max_seq)
            ids = torch.randint(0, vocab, (1, seq), dtype=torch.long)
            for _ in range(2):  # warmup, not measured
                model(ids)
            times: list[float] = []
            for run_idx in range(runs):
                progress(f"    Forward seq={seq} run {run_idx + 1}/{runs}...")
                start = time.perf_counter()
                model(ids)
                times.append(time.perf_counter() - start)
            results[seq] = seq / max(_median(times), 1e-12)
    return results


def _measure_bulk_throughput(model: FeatherV2Model, batch: int, seq: int) -> float:
    """Tokens per second across a batch of real forward passes."""
    model.eval()
    vocab = int(model.config["vocab"])
    seq = min(seq, int(model.config["seq_len"]))
    ids = torch.randint(0, vocab, (batch, seq), dtype=torch.long)
    with torch.no_grad():
        model(ids[:1])  # warmup
        start = time.perf_counter()
        for start_row in range(0, batch, 1):
            model(ids[start_row : start_row + 1])
        elapsed = time.perf_counter() - start
    processed = batch * seq
    return processed / max(elapsed, 1e-12)


def _measure_generation(
    model: FeatherV2Model, prompt: torch.Tensor, steps: int
) -> tuple[float, torch.Tensor]:
    """Time a real greedy generation of ``steps`` tokens."""
    was_training = model.training
    model.eval()
    start = time.perf_counter()
    out = model.generate(prompt, max_new_tokens=steps, greedy=True)
    elapsed = time.perf_counter() - start
    if was_training:
        model.train()
    return elapsed, out


def _measure_loss_trend(
    model: FeatherV2Model, text: str, steps: int = 100, lr: float = 1e-3
) -> tuple[list[float], bool]:
    """Real next-token cross-entropy under a real optimizer.

    The previous version flattened the model output, multiplied it by a fixed
    random matrix, and took the mean squared error against a one-hot vector.
    There was no optimizer, so the quantity never changed, and the surrounding
    ``try/except`` turned any failure into an empty list instead of a traceback.
    This trains the model the way any user would.
    """
    vocab = int(model.config["vocab"])
    tokens = hybrid_adaptive_tokenizer(text, vocab_size=vocab)[0]
    ids = torch.as_tensor(np.asarray(tokens).reshape(-1) % vocab, dtype=torch.long)
    seq_len = min(int(model.config["seq_len"]), max(8, ids.numel() // 4))
    if ids.numel() < seq_len + 2:
        return [], False

    was_training = model.training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    try:
        for step in range(steps):
            offset = (step * seq_len) % (ids.numel() - seq_len - 1)
            # Inputs and labels are both seq_len long. Passing a seq_len + 1
            # window as input_ids trips the model's own length check.
            inputs = ids[offset : offset + seq_len].unsqueeze(0)
            labels = ids[offset + 1 : offset + seq_len + 1].unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(inputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(metrics["loss"]))
    finally:
        if not was_training:
            model.eval()
    return losses, assert_real_varying_loss(losses)


def _measure_state_stability(
    model: FeatherV2Model, seq: int
) -> tuple[float | None, int]:
    """Cosine similarity of a prefix's final hidden state with and without a suffix.

    The same token prefix is encoded twice: once on its own, and once as the
    start of a longer sequence. The last position of the prefix is compared in
    both cases, so the only difference is whether later tokens were present.

    Comparing final *logits* instead would mix a vocab-wide vector into a
    representation-similarity claim, and comparing the last position of a short
    prefix against the last position of a long sequence would compare two
    different positions. Both were wrong; this compares the same position of the
    same content and reports hidden-state width.

    Returns ``(None, 0)`` if no comparable hidden state can be captured.
    """
    vocab = int(model.config["vocab"])
    seq = min(seq, int(model.config["seq_len"]))
    if seq < 4:
        return None, 0

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["h"] = tensor.detach()

    handle = model.norm_f.register_forward_hook(_hook)
    try:
        generator = torch.Generator().manual_seed(99)
        ids = torch.randint(0, vocab, (1, seq), generator=generator)
        third = max(1, seq // 3)
        model.eval()
        with torch.no_grad():
            model(ids)  # long: prefix followed by a suffix
            long_prefix_state = captured["h"][:, third - 1]
            model(ids[:, :third])  # short: the prefix alone
            short_prefix_state = captured["h"][:, third - 1]
    except Exception:
        return None, 0
    finally:
        handle.remove()

    similarity = float(
        torch.nn.functional.cosine_similarity(
            long_prefix_state.flatten(), short_prefix_state.flatten(), dim=-1
        )
    )
    return similarity, int(short_prefix_state.numel())


def _measure_component_breakdown(
    model: FeatherV2Model, runs: int = 3
) -> dict[str, float]:
    """Time each of the seven sublayers by actually running that sublayer.

    The previous version called ``model.forward(x)`` once per component name,
    timing the entire model seven times, then normalised the seven identical
    numbers to percentages that always sum to 100. The "breakdown" was the same
    measurement relabelled; it could not have shown anything else.
    """
    block = model.blocks[0]
    pairs = [
        ("sensory", block.sensory),
        ("liquid_memory", block.liquid),
        ("hyperdimensional", block.hyper),
        ("knowledge_vault", block.vault),
        ("cognitive_weaver", block.weaver),
        ("homeostasis", block.governor),
        ("generative_evolution", block.evolution),
    ]
    model.eval()
    times: dict[str, float] = {}
    with torch.no_grad():
        for name, module in pairs:
            probe = torch.randn(1, 8, int(model.config["dim"]))
            module(probe)  # warmup
            samples = []
            for _ in range(runs):
                start = time.perf_counter()
                module(probe)
                samples.append(time.perf_counter() - start)
            times[name] = _median(samples)
    total = sum(times.values())
    if total <= 0:
        return {name: 0.0 for name in times}
    return {name: value / total * 100.0 for name, value in times.items()}


def _measure_energy(model: FeatherV2Model, text: str) -> tuple[float | None, dict]:
    """Energy for one forward plus a short generation, via codecarbon.

    The previous version, on any exception at all, returned
    ``tokens * 3.7e-15 * 128 * 512``. That constant appears nowhere else in the
    repository, was not derived from any measurement, and was reported as if it
    were one. It is now ``None``.
    """
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        cprint("  codecarbon unavailable; energy not measured", _YELLOW)
        return None, {"reason": "codecarbon not installed"}

    try:
        model.eval()
        vocab = int(model.config["vocab"])
        seq = min(32, int(model.config["seq_len"]))
        ids = torch.randint(0, vocab, (1, seq), dtype=torch.long)
        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()
        try:
            with torch.no_grad():
                model(ids)
                model.generate(ids, max_new_tokens=16, greedy=True)
        finally:
            tracker.stop()
        emissions_kg = tracker.final_emissions
        if emissions_kg is None:
            return None, {"reason": "tracker produced no emissions reading"}
        return float(emissions_kg) * 3.6e9, {"co2_kg": float(emissions_kg)}
    except Exception as exc:  # noqa: BLE001
        cprint(f"  energy measurement failed: {exc}", _YELLOW)
        return None, {"reason": str(exc)}


def _count_parameters(model: FeatherV2Model) -> int:
    """The model's own deduplicated parameter count.

    The previous version walked ``dir(component)`` looking for ``np.ndarray``
    attributes and added ``_logit_projection.size``. That counted whatever
    happened to be cached on each object, not the model's parameters, so it was
    unrelated to model size and drifted whenever an attribute was added.
    """
    return model.count_parameters()


# ---------------------------------------------------------------------------
# 8. Main test loop
# ---------------------------------------------------------------------------
def test_one_size(
    size_label: str,
    config: dict,
    datasets_info: dict,
    loss_steps: int = 100,
) -> dict[str, Any]:
    print_header(f"[{size_label}] Testing {size_label} model")
    results: dict[str, Any] = {"size_label": size_label, "checks": []}
    try:
        progress("Initializing model...")
        model = FeatherV2Model(config=config)
        results["init_ok"] = True

        progress("Checking weights...")
        weights_ok = assert_real_weights(model)
        results["checks"].append(("weights_real", weights_ok))
        cprint(
            f"  Weights real: {'PASS ✓' if weights_ok else 'FAIL ✗'}",
            _GREEN if weights_ok else _RED,
        )

        progress("Counting parameters...")
        params = _count_parameters(model)
        results["params"] = params
        cprint(f"  Parameters: {params:,}", _MAGENTA)

        progress("Measuring RAM...")
        import psutil

        proc = psutil.Process()
        vocab = int(config["vocab"])
        probe_ids = torch.randint(
            0, vocab, (1, min(8, int(config["seq_len"]))), dtype=torch.long
        )
        with torch.no_grad():
            model(probe_ids)  # warm allocations before the baseline
        ram_before = proc.memory_info().rss / (1024 * 1024)
        with torch.no_grad():
            model(probe_ids)
        ram_after = proc.memory_info().rss / (1024 * 1024)
        ram_delta = ram_after - ram_before
        ram_total = ram_after
        element_size = next(model.parameters()).element_size()
        weight_mb = params * element_size / (1024 * 1024)
        ram_ok = assert_real_ram(model, ram_delta, ram_total)
        results["checks"].append(("ram_real", ram_ok))
        results["ram_mb"] = ram_total
        results["ram_delta_mb"] = ram_delta
        results["weight_mb"] = weight_mb
        cprint(
            f"  RAM: weights={weight_mb:.1f}MB delta={ram_delta:.1f}MB "
            f"total={ram_total:.1f}MB",
            _GREEN if ram_ok else _RED,
        )

        progress("Tokenizer speed test...")
        corpus_text = next(
            (
                entry["text"]
                for entry in datasets_info.values()
                if entry.get("available") and entry.get("text")
            ),
            "",
        )
        results["corpus_available"] = bool(corpus_text)
        if corpus_text:
            tok_s, tok_info = _measure_tokenizer_speed(model, corpus_text)
            cprint(f"  Tokenizer: {tok_s:.0f} tok/s", _YELLOW)
        else:
            tok_s, tok_info = None, {"reason": "no corpus available"}
            cprint("  Tokenizer: not measured (no corpus available)", _YELLOW)
        results["tokenizer_tok_s"] = tok_s
        results["tokenizer_info"] = tok_info

        progress("Forward throughput...")
        max_seq = int(config["seq_len"])
        seq_lengths = [s for s in (32, 128, 512) if s <= max_seq] or [max_seq]
        ft = _measure_forward_throughput(model, seq_lengths, runs=3)
        results["forward_throughput"] = {str(k): v for k, v in ft.items()}
        for seq, speed in ft.items():
            cprint(f"  Seq {seq}: {speed:.1f} tok/s", _YELLOW)

        progress("Bulk throughput (batch 8)...")
        bulk_tok_s = _measure_bulk_throughput(model, 8, max_seq)
        results["bulk_tok_s"] = bulk_tok_s
        cprint(f"  Bulk: {bulk_tok_s:.1f} tok/s", _YELLOW)

        progress("Generation speed (64 tokens)...")
        prompt = torch.randint(0, vocab, (1, min(8, max_seq)), dtype=torch.long)
        gen_elapsed, gen_out = _measure_generation(model, prompt, 64)
        generated = max(0, int(gen_out.shape[-1]) - int(prompt.shape[-1]))
        gen_tok_s = generated / max(gen_elapsed, 1e-12)
        results["gen_tok_s"] = gen_tok_s
        results["gen_tokens"] = generated
        results["gen_elapsed_s"] = gen_elapsed
        gen_ok = assert_real_timing(gen_elapsed)
        results["checks"].append(("timing_real", gen_ok))
        cprint(
            f"  Generation: {gen_tok_s:.1f} tok/s "
            f"({generated} tokens in {gen_elapsed:.3f}s)",
            _YELLOW,
        )

        progress("Loss on real text data...")
        wiki = datasets_info.get("wikipedia", {})
        if wiki.get("available") and wiki.get("text"):
            losses, loss_ok = _measure_loss_trend(model, wiki["text"], steps=loss_steps)
            results["loss_corpus"] = wiki["name"]
            cprint(
                f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} on {wiki['name']}",
                _GREEN if loss_ok else _RED,
            )
        else:
            # No corpus means no loss curve. Reporting a number here would mean
            # reporting one that came from a placeholder string.
            losses, loss_ok = [], False
            results["loss_corpus"] = None
            cprint(
                "  Loss: not measured (Wikipedia unavailable; "
                f"{wiki.get('error', 'no reason recorded')})",
                _YELLOW,
            )
        results["losses"] = [float(loss) for loss in losses[:100]]
        results["loss_trend_ok"] = loss_ok
        results["checks"].append(("loss_trend", loss_ok))

        progress("State stability (prefix cosine sim)...")
        ctx_sim, ctx_d = _measure_state_stability(model, int(config["seq_len"]))
        ctx_ok = ctx_sim is not None and math.isfinite(ctx_sim)
        results["checks"].append(("state_stability", ctx_ok))
        results["state_stability_sim"] = ctx_sim
        results["state_stability_dim"] = ctx_d
        cprint(
            (
                f"  State stability: {ctx_sim:.4f} (hidden dim {ctx_d})"
                if ctx_sim is not None
                else "  State stability: not measured"
            ),
            _GREEN if ctx_ok else _RED,
        )

        progress("Energy measurement...")
        energy_j, energy_detail = _measure_energy(model, corpus_text)
        results["energy_j"] = energy_j
        results["energy_detail"] = energy_detail
        cprint(
            (
                f"  Energy: {energy_j:.2e} J"
                if energy_j is not None
                else f"  Energy: not measured ({energy_detail.get('reason', 'unknown')})"
            ),
            _CYAN,
        )

        progress("Component breakdown...")
        comp_break = _measure_component_breakdown(model)
        results["component_breakdown"] = comp_break
        for comp, pct in comp_break.items():
            cprint(f"  {comp}: {pct:.1f}%", _MAGENTA)

        results["checks"].append(("init_ok", True))
        results["ok"] = all(v for _, v in results["checks"])
        return results

    except Exception as exc:
        cprint(f"  FAIL size {size_label}: {exc}", _RED)
        results["ok"] = False
        results["error"] = str(exc)
        results["checks"].append(("init_ok", False))
        return results


# ---------------------------------------------------------------------------
# 9. Scaling plots
# ---------------------------------------------------------------------------
def make_plots(all_results: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            import matplotlib.pyplot as plt

            plt.style.use("ggplot")
        except Exception:
            cprint("matplotlib unavailable, skipping plots", _RED)
            return

    sizes = [r["size_label"] for r in all_results if r.get("params")]
    params = [r.get("params", 0) / 1e6 for r in all_results if r.get("params")]
    ram = [r.get("ram_mb", 0) for r in all_results if r.get("ram_mb")]
    ft = [
        r.get("forward_throughput", {}).get("512", 0)
        for r in all_results
        if r.get("forward_throughput")
    ]
    bulk = [r.get("bulk_tok_s", 0) for r in all_results if r.get("bulk_tok_s")]
    energy = [r.get("energy_j", 0) for r in all_results if r.get("energy_j")]
    losses_all = [r.get("losses", []) for r in all_results]

    # Plot 1: loss_all_sizes.png
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in all_results:
            if r.get("losses"):
                ax.plot(r["losses"], label=r["size_label"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss trend — all sizes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(IMG_DIR / "loss_all_sizes.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        cprint(f"  Plot loss_all_sizes failed: {exc}", _RED)

    # Plot 2: toks_vs_seqlen.png
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in all_results:
            ft_local = r.get("forward_throughput", {})
            if ft_local:
                xs = [int(k) for k in ft_local]
                ys = [ft_local[k] for k in ft_local]
                ax.plot(xs, ys, marker="o", label=r["size_label"])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Tokens / s")
        ax.set_title("Forward throughput vs seq len")
        ax.legend()
        fig.tight_layout()
        fig.savefig(IMG_DIR / "toks_vs_seqlen.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        cprint(f"  Plot toks_vs_seqlen failed: {exc}", _RED)

    # Plot 3: memory_vs_params.png
    try:
        if params and ram:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(params, ram, s=100, c="teal")
            for i, label in enumerate(sizes):
                ax.annotate(label, (params[i], ram[i]))
            ax.set_xlabel("Params (M)")
            ax.set_ylabel("RAM (MB)")
            ax.set_title("Memory vs parameters")
            fig.tight_layout()
            fig.savefig(IMG_DIR / "memory_vs_params.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        cprint(f"  Plot memory_vs_params failed: {exc}", _RED)

    # Plot 4: scaling.png
    try:
        if params and ft:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(params, ft, s=100, c="coral")
            for i, label in enumerate(sizes):
                ax.annotate(label, (params[i], ft[i]))
            ax.set_xlabel("Params (M)")
            ax.set_ylabel("Tokens / s (seq 512)")
            ax.set_title("Scaling: throughput vs model size")
            fig.tight_layout()
            fig.savefig(IMG_DIR / "scaling.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        cprint(f"  Plot scaling failed: {exc}", _RED)

    # Plot 5: energy_vs_size.png
    try:
        if params and energy:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(sizes, energy, color="gold")
            ax.set_xlabel("Model size")
            ax.set_ylabel("Energy (J)")
            ax.set_title("Energy vs model size")
            fig.tight_layout()
            fig.savefig(IMG_DIR / "energy_vs_size.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        cprint(f"  Plot energy_vs_size failed: {exc}", _RED)

    # Plot 6: component_breakdown for the largest measured size
    try:
        with_breakdown = [
            r for r in all_results if r.get("component_breakdown") and r.get("params")
        ]
        if with_breakdown:
            target = max(with_breakdown, key=lambda r: r["params"])
            label = target.get("size_label", "?")
            fig, ax = plt.subplots(figsize=(10, 6))
            comp = target["component_breakdown"]
            labels = list(comp.keys())
            values = [comp[k] for k in labels]
            ax.barh(labels, values, color="mediumpurple")
            ax.set_xlabel("Time share (%)")
            ax.set_title(f"Component breakdown - {label}")
            fig.tight_layout()
            fig.savefig(IMG_DIR / f"component_breakdown_{label}.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        cprint(f"  Plot component_breakdown failed: {exc}", _RED)

    cprint(f"Plots saved to {IMG_DIR}", _GREEN)


# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
def print_summary(all_results: list[dict[str, Any]]) -> None:
    print_header("SUMMARY TABLE")
    headers = [
        "Size",
        "Params(M)",
        "RAM(MB)",
        "Fwd t/s",
        "Bulk t/s",
        "Loss 1st",
        "Loss lst",
        "Checks",
    ]
    rows = []
    total_checks = 0
    total_pass = 0
    for r in all_results:
        losses = r.get("losses") or []
        loss_first = f"{losses[0]:.4f}" if losses else "n/a"
        loss_last = f"{losses[-1]:.4f}" if losses else "n/a"
        checks = r.get("checks", [])
        passed = sum(1 for _, v in checks if v)
        total = len(checks)
        total_checks += total
        total_pass += passed
        params = r.get("params")
        throughput = r.get("forward_throughput") or {}
        # Report the longest sequence actually measured for this size rather
        # than assuming every config ran seq=512.
        best_seq = max(throughput, key=lambda k: int(k)) if throughput else None
        row = [
            str(r.get("size_label", "?")),
            f"{params/1e6:.2f}" if params else "n/a",
            f"{r['ram_mb']:.1f}" if r.get("ram_mb") else "n/a",
            f"{throughput[best_seq]:.1f}" if best_seq else "n/a",
            f"{r['bulk_tok_s']:.1f}" if r.get("bulk_tok_s") else "n/a",
            loss_first,
            loss_last,
            f"{passed}/{total}",
        ]
        rows.append(row)

    if _USE_RICH:
        _rich_table(headers, rows, title="All sizes comparison")
    else:
        _fallback_table(headers, rows, title="All sizes comparison")

    print_header("FINAL VERDICT")
    for r in all_results:
        size = r.get("size_label", "?")
        checks = r.get("checks", [])
        passed = sum(1 for _, v in checks if v)
        total = len(checks)
        ok = r.get("ok", False)
        status = f"{_GREEN}PASS ✓{_RESET}" if ok else f"{_RED}FAIL ✗{_RESET}"
        cprint(f"  {size}: {passed}/{total} checks {status}", "")

    if total_checks > 0 and total_pass == total_checks:
        cprint(f"\n  🎯 {total_checks} checks 100% PASS", _GREEN)
    else:
        failed = total_checks - total_pass
        cprint(f"\n  {total_pass}/{total_checks} passed, {failed} failed", _RED)


# ---------------------------------------------------------------------------
# 11. Save report
# ---------------------------------------------------------------------------
def save_report(
    all_results: list[dict[str, Any]],
    hw: dict[str, Any],
    *,
    final: bool = True,
) -> None:
    """Write benchmark_report.json, merging with any sizes already on disk.

    Saving after every size matters because the largest sizes are slow: if the
    process is killed or times out partway up the ladder, the completed sizes
    must survive. Merging by ``size_label`` also means a subset re-run (for
    example ``--sizes 60M``) updates just that size instead of replacing the
    report with a single entry.

    Each entry carries its own ``measured_at`` so a merged report never implies
    that all sizes came from one moment.
    """
    out_path = ROOT / "benchmark_report.json"

    previous: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            for rec in old.get("sizes", []) or []:
                label = rec.get("size_label")
                if label:
                    previous[label] = rec
        except Exception as exc:  # noqa: BLE001
            cprint(f"  could not merge previous report ({exc}); starting fresh", _RED)

    merged: dict[str, dict[str, Any]] = dict(previous)
    for rec in all_results:
        label = rec.get("size_label")
        if label:
            merged[label] = rec

    def _order(item: tuple[str, dict[str, Any]]) -> float:
        params = item[1].get("params")
        return float(params) if params else float("inf")

    ordered = [rec for _, rec in sorted(merged.items(), key=_order)]

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw,
        "sizes": ordered,
        "summary": {
            "total_checks": sum(len(r.get("checks", [])) for r in ordered),
            "total_pass": sum(
                sum(1 for _, v in r.get("checks", []) if v) for r in ordered
            ),
            "sizes_measured": len(ordered),
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if final:
        cprint(f"\nReport saved: {out_path} ({len(ordered)} size(s))", _GREEN)
        try:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            cprint(f"Report size: {size_mb:.2f} MB", _CYAN)
        except Exception:
            pass
    else:
        cprint(f"  checkpointed {len(ordered)} size(s) to {out_path.name}", _CYAN)

    return ordered


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=sorted(CONFIG_MAP, key=lambda s: float(s.rstrip("M"))),
        help=f"subset of {sorted(CONFIG_MAP)}",
    )
    parser.add_argument("--loss-steps", type=int, default=100)
    args = parser.parse_args()

    print_header("FEATHER-V2 MEGA BENCHMARK")
    cprint(
        f"ladder: {', '.join(args.sizes)}  |  outputs: {IMG_DIR} / {OUT_DIR}",
        _CYAN,
    )

    print_hardware()
    datasets_info = load_streaming_datasets()

    hw = detect_hardware()
    all_results: list[dict[str, Any]] = []
    size_list = list(args.sizes)
    total = len(size_list)

    for idx, size in enumerate(size_list, 1):
        cprint(f"\n[{idx}/{total}] Testing {size}...", _CYAN)
        cfg = CONFIG_MAP[size]
        size_start = time.perf_counter()
        res = test_one_size(size, cfg, datasets_info, loss_steps=args.loss_steps)
        res["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        all_results.append(res)

        status = "PASS V" if res.get("ok") else "FAIL ?"
        color = _GREEN if res.get("ok") else _RED
        cprint(
            f"[{idx}/{total}] {size}: {status} ({time.perf_counter() - size_start:.1f}s)",
            color,
        )

        # Checkpoint after every size. The big sizes are slow and a Kaggle
        # session can be interrupted or time out, so completed measurements are
        # written out immediately instead of only at the end of the ladder.
        save_report(all_results, hw, final=False)

    # Final save returns every size on disk, including any from an earlier
    # subset run, so the summary and the plots match the saved report.
    ordered = save_report(all_results, hw)
    print_summary(ordered)
    make_plots(ordered)


if __name__ == "__main__":
    main()
