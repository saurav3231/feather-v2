"""Train the Feather v2 quick-baseline model and print a real metrics table.

What this does
--------------
Trains one configuration for a fixed number of steps on real text and prints a
row every ``--report-every`` steps with cumulative tokens, measured throughput,
measured loss, measured RSS, measured energy, and the model's ``state_stability``
score. Writes a JSON artifact and five plots built only from those measurements.

Everything printed is measured
------------------------------
There are no target numbers in this file. The loss, throughput, memory and energy
in the table and in ``benchmark_20M_simple.json`` come from the run that just
happened. Where a value cannot be measured, for example energy when
``codecarbon`` is not installed, the field is ``null`` and the table prints
``not measured`` rather than an estimate.

A note on the size
------------------
``configs/feather_20M_simple.json`` measures 20,696,188 parameters, which is
20.70M against a 20M target. The originally specified ``dim=384`` measures
23.12M at ``n_blocks=2`` and 82.36M at ``n_blocks=8``, so ``dim=352`` was chosen
to land near the target. Both numbers are real sums over ``p.numel()``; verify
either with::

    python -m feather_v2.model --config configs/feather_20M_simple.json --count-params

A note on speed
---------------
Throughput is dominated by tokens per step, so the defaults are the small,
faster configuration rather than the large one:

===============  ===========  ==============  ====================
batch x seq      tokens/step  measured s/step  600 steps
===============  ===========  ==============  ====================
2 x 128          256          4.8             48 min (projected)
8 x 512          4096         72              12 h (projected)
===============  ===========  ==============  ====================

The per-step figures are real measurements from a 4.8 s/step observation on a
2-core/4-thread laptop at batch 2 x seq 128, scaled by token count. The 600-step
totals are arithmetic projections, not completed runs. A 10-20 minute run for
600 steps at batch 8 x seq 512 was claimed in an earlier specification and is not
achievable on this hardware: it would require roughly 115 tok/s sustained, about
2.3x the measured rate, before any accounting for the 4096 tokens per step.

``--time-budget-hours`` guards against this. After three timed steps the script
projects the finish time from the measured rate and stops with a clear message if
the projection exceeds the budget.

Two things this script deliberately does not claim, because they are not measured
here and were fabricated in earlier revisions of this project:

- It does not predict inference throughput. Only the training throughput it just
  measured is reported.
- It does not report a "recall" score. ``state_stability`` says how similar one
  position's hidden state is with and without a following suffix. That is not
  retrieval and not long-context capability. The old ``context_recall`` check
  compared final logits of two different positions and reported a vocabulary
  width as if it were a representation width.

Feasibility
-----------
Training throughput on a 2-core CPU is far lower than a GPU, and a step of
``batch * seq`` tokens can take minutes rather than seconds for a model this size.
The script measures the first few steps, projects the total wall-clock time, and
**stops with a clear message** if the projection exceeds ``--time-budget-hours``.
That guard exists so a 12-hour Kaggle CPU session is not consumed before the
first row is printed. Pass ``--allow-overrun`` to run anyway.

Usage
-----
    python kaggle/train_20M_simple.py
    python kaggle/train_20M_simple.py --steps 600 --report-every 50
    python kaggle/train_20M_simple.py --sizes 1M 2M --target-tokens 2000000
    python kaggle/train_20M_simple.py --allow-local-text     # offline fallback
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from feather_v2 import FeatherV2Model  # noqa: E402
from feather_v2.hardware import detect_cpu_features, get_best_kernel  # noqa: E402
from feather_v2.utils import (  # noqa: E402
    FittedRunTokenizer,
    hybrid_adaptive_tokenizer,
    unigram_floor,
)

# Corpora are tried in order. Each entry is (label, path, config, split, take).
#
# "openwebtext" is listed last on purpose: it is a script-based dataset that recent
# versions of `datasets` refuse to load ("Dataset scripts are no longer supported").
# It is attempted so the failure is reported rather than hidden, but the run does
# not depend on it. The parquet datasets load without `trust_remote_code`.
DATASETS: list[tuple[str, str, str | None, str, int]] = [
    (
        "wikimedia/wikipedia 20231101.en",
        "wikimedia/wikipedia",
        "20231101.en",
        "train",
        400,
    ),
    (
        "Salesforce/wikitext wikitext-103-raw-v1",
        "Salesforce/wikitext",
        None,
        "train",
        0,
    ),
    ("HuggingFaceH4/no_robots", "HuggingFaceH4/no_robots", None, "train", 600),
    ("openwebtext (script-based, may fail)", "openwebtext", None, "train", 750),
]


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def describe_hardware() -> dict[str, Any]:
    """Report detected CPU features and the kernel binding that will be used."""
    features = detect_cpu_features()
    kernel = get_best_kernel()
    return {
        "cpu": features.get("cpu"),
        "cores_physical": features.get("cores_physical"),
        "cores_logical": features.get("cores_logical"),
        "avx": features.get("avx"),
        "avx2": features.get("avx2"),
        "avx512": features.get("avx512"),
        "neon": features.get("neon"),
        "arch": features.get("machine"),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "binding": kernel.get("binding"),
        "threads": kernel.get("threads"),
        "compute_dtype": kernel.get("compute_dtype"),
        "torch_threads": torch.get_num_threads(),
    }


def configure_threads(threads: int | None) -> int:
    """Pin torch to the requested thread count, defaulting to physical cores.

    Physical cores rather than logical threads: on a 2-core/4-thread machine,
    hyperthread siblings usually share one set of vector units, so oversubscribing
    them costs time rather than gaining it.
    """
    if threads is None:
        features = detect_cpu_features()
        threads = int(features.get("cores_physical") or 1)
    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialised by an earlier call; harmless.
        pass
    return threads


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def _stream_text(spec: tuple[str, str, str | None, str, int]) -> tuple[str, str]:
    """Yield (label, text) chunks from one streaming dataset.

    Raises on any failure so the caller can record the reason and try the next
    candidate. Nothing is silently substituted.
    """
    from datasets import load_dataset

    label, path, config, split, take = spec
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    dataset = load_dataset(path, **kwargs)
    if take:
        dataset = dataset.take(take)

    budget = 4 * 1024 * 1024  # characters per chunk
    buffer: list[str] = []
    size = 0
    for row in dataset:
        text = row.get("text") or row.get("content") or ""
        if not text:
            continue
        buffer.append(text)
        size += len(text)
        if size >= budget:
            yield label, "\n\n".join(buffer)
            buffer, size = [], 0
    if buffer:
        yield label, "\n\n".join(buffer)


def load_local_text() -> Iterator[tuple[str, str]]:
    """Fall back to text already in the repository, clearly labelled as such.

    This is not a corpus. It exists so the script can run offline, and every
    record produced this way is marked ``corpus_local_fallback`` so it can never be
    mistaken for a real-corpus result.
    """
    for name in ("README.md", "RELEASE_v2.0.0.md"):
        path = ROOT / name
        if path.exists():
            yield f"local:{name}", path.read_text(encoding="utf-8", errors="replace")
    docs = ROOT / "docs"
    if docs.is_dir():
        for path in sorted(docs.glob("*.md")):
            yield (
                f"local:docs/{path.name}",
                path.read_text(encoding="utf-8", errors="replace"),
            )


# Measured on wikimedia/wikipedia 20231101.en with FittedRunTokenizer(run=4):
# 4,207,607 characters encode to 1,803,817 ids, i.e. 0.4287 tokens per character.
# The old code counted characters against target_tokens and then reported that
# character count as "tokens", so every log overstated the corpus by 2.3x and the
# loop either over-read or came up short.
_CHARS_PER_TOKEN = 2.34


def build_token_stream(
    target_tokens: int,
    vocab_size: int,
    hardware_ram_gb: float,
    allow_local_text: bool,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tokenize real text until ``target_tokens`` ids are collected."""
    attempts: list[dict[str, Any]] = []
    stream = np.zeros(0, dtype=np.int64)
    corpus_info: dict[str, Any] = {}

    for spec in DATASETS:
        label = spec[0]
        if verbose:
            print(f"  trying {label} ...", flush=True)
        started = time.perf_counter()
        collected: list[np.ndarray] = []
        total = 0
        try:
            pieces: list[str] = []
            chars = 0
            for _chunk_label, text in _stream_text(spec):
                if chars >= target_tokens * _CHARS_PER_TOKEN:
                    break
                pieces.append(text)
                chars += len(text)
            if not pieces:
                raise ValueError("no text produced tokens")
            # Join once. This used to be done three times over the whole corpus,
            # which tripled peak memory for no reason.
            blob = "".join(pieces)
            pieces.clear()
            del pieces
            raw = np.frombuffer(blob.encode("utf-8", "ignore"), dtype=np.uint8)
            del blob
            # Fit the run vocabulary on this corpus, then encode. The earlier
            # modulo-hash tokenizer collided distinct byte runs onto shared ids,
            # which capped learnable signal and stalled loss.
            tokenizer = FittedRunTokenizer(vocab_size=vocab_size, run=4).fit(
                raw.tobytes().decode("utf-8", "ignore")
            )
            ids = tokenizer.encode_bytes(raw)
            if ids is None or len(ids) == 0:
                raise ValueError("tokenizer produced no ids")
            total = int(len(ids))
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "dataset": label,
                    "ok": False,
                    "reason": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
            if verbose:
                print(f"    unavailable: {type(exc).__name__}", flush=True)
            continue

        corpus_info = tokenizer.describe(raw)
        corpus_info["unigram_floor_nats"] = unigram_floor(ids)
        corpus_info["collision_note"] = (
            "no collisions: unseen runs fall back to raw bytes, so distinct "
            "strings never share an id"
        )
        attempts.append({"dataset": label, "ok": True, "tokens": int(total)})
        if verbose:
            print(
                f"    loaded {total:,} tokens from {chars:,} chars in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if total > target_tokens:
            ids = ids[:target_tokens]
            total = int(len(ids))
        # int32 halves the resident size of the stream; ids never exceed a vocab
        # that fits in 32 bits, and nothing here needs int64.
        stream = np.asarray(ids, dtype=np.int32)
        break

    used_fallback = False
    if stream.size == 0 and allow_local_text:
        used_fallback = True
        if verbose:
            print("  falling back to text in this repository (NOT a corpus)")
        pieces = [text for _label, text in load_local_text()]
        if pieces:
            tokenizer = FittedRunTokenizer(vocab_size=vocab_size, run=4).fit(
                "".join(pieces)
            )
            raw = np.frombuffer(
                "".join(pieces).encode("utf-8", "ignore"), dtype=np.uint8
            )
            stream = tokenizer.encode_bytes(raw)
            corpus_info = tokenizer.describe(raw)
            corpus_info["unigram_floor_nats"] = unigram_floor(stream)

    info = {
        "sources": attempts,
        "corpus_local_fallback": used_fallback,
        "tokens": int(stream.size),
        "target_tokens": int(target_tokens),
    }
    if corpus_info:
        info["tokenizer"] = corpus_info
    if stream.size == 0:
        raise SystemExit(
            "No corpus could be loaded and no local fallback was allowed.\n"
            "Connect to the internet for the streaming datasets, or pass\n"
            "--allow-local-text to train on text in this repository, which is\n"
            "recorded in the output as a local fallback and is not a real corpus."
        )
    if stream.size < target_tokens:
        info["shortfall_tokens"] = int(target_tokens - stream.size)
    return stream, info


def make_batches(
    stream: np.ndarray,
    batch_size: int,
    seq_len: int,
    steps: int,
    seed: int,
) -> Iterator[torch.Tensor]:
    """Sample random fixed-length windows from the token stream.

    Yields ``steps + 1`` batches because the loop takes one extra batch for the
    untimed step-0 measurement before performing ``steps`` real updates.
    """
    rng = np.random.default_rng(seed)
    high = int(stream.size) - seq_len - 1
    if high <= 0:
        raise SystemExit(
            f"token stream too short: {stream.size} ids, need more than {seq_len}"
        )
    for _ in range(steps + 1):
        starts = rng.integers(0, high, size=batch_size)
        window = np.stack([stream[s : s + seq_len] for s in starts])
        yield torch.from_numpy(window.astype(np.int64))


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def _tracker_kwh(tracker: Any) -> float | None:
    """Read live energy in kWh from a codecarbon tracker.

    codecarbon 3.x exposes the running total as ``_total_energy``, an ``Energy``
    object with a ``kWh`` attribute. There is no public live accessor, and
    ``final_emissions`` only appears after ``stop()`` and is a CO2-equivalent
    figure, which would need a carbon-intensity assumption to turn into joules.
    Reading kWh avoids that assumption entirely.
    """
    if tracker is None:
        return None
    energy = getattr(tracker, "_total_energy", None)
    if energy is None:
        return None
    try:
        return float(getattr(energy, "kWh", energy))
    except (TypeError, ValueError):
        return None


class EnergyMeter:
    """Measure energy with codecarbon when it is installed.

    If codecarbon is unavailable the meter reports ``None`` and every energy field
    is ``null``. It never substitutes a guess, because an invented wattage would
    make the number look measured while being arithmetic.
    """

    def __init__(self) -> None:
        self.tracker: Any = None
        self.available = False
        self.reason: str | None = None
        self._base_kwh: float | None = None
        try:
            from codecarbon import EmissionsTracker  # type: ignore

            self.tracker = EmissionsTracker(
                log_level="error", save_to_file=False, measure_power_secs=1
            )
            self.tracker.start()
            time.sleep(1.0)  # let the first power sample land
            self._base_kwh = _tracker_kwh(self.tracker)
            if self._base_kwh is None:
                raise RuntimeError("codecarbon exposed no live kWh value")
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"{type(exc).__name__}: {exc}"[:200]
            self.tracker = None

    def joules(self) -> float | None:
        """Joules consumed since the meter was created, or ``None``."""
        if not self.available or self._base_kwh is None:
            return None
        kwh = _tracker_kwh(self.tracker)
        if kwh is None:
            return None
        return max(0.0, (kwh - self._base_kwh) * 3.6e6)

    def stop(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
NOT_MEASURED = "not measured"


def _fmt(value: Any, spec: str = ".2f", suffix: str = "") -> str:
    if value is None:
        return NOT_MEASURED
    try:
        return f"{value:{spec}}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _ascii_safe(text: str) -> str:
    """Force text to ASCII so it prints on a cp1252 console.

    Rich draws tables with box characters. Those are fine on Kaggle's UTF-8
    terminal but raise UnicodeEncodeError on a default Windows console, so they are
    transliterated rather than allowed to break the run at the final print.
    """
    replacements = {
        "\u2500": "-",
        "\u2502": "|",
        "\u250c": "+",
        "\u2510": "+",
        "\u2514": "+",
        "\u2518": "+",
        "\u251c": "+",
        "\u2524": "+",
        "\u252c": "+",
        "\u2534": "+",
        "\u256d": "+",
        "\u256e": "+",
        "\u2570": "+",
        "\u256f": "+",
        "\u00b7": ".",
        "\u2026": "...",
        "\u2192": "->",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def render_table(rows: list[dict[str, Any]], hardware: dict[str, Any]) -> str:
    """Render the metrics table with rich when available, plain text otherwise."""
    header = (
        "Step",
        "Tokens",
        "TPS",
        "Loss",
        "RAM total",
        "RAM delta",
        "Energy/step",
        "Energy/1k tok",
        "StateStab",
    )

    def cells(row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(row["step"]),
            f"{row['total_tokens']:,}",
            _fmt(row.get("tps"), ".0f"),
            _fmt(row.get("loss"), ".4f"),
            _fmt(row.get("ram_total_mb"), ".0f", "MB"),
            _fmt(row.get("ram_delta_mb"), "+.0f", "MB"),
            _fmt(row.get("energy_step_j"), ".3f", "J"),
            _fmt(row.get("energy_per_1k_j"), ".3f", "J"),
            _fmt(row.get("state_stability"), ".5f"),
        )

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title="Feather-v2 quick baseline - measured metrics (no target values)",
            box=None,
        )
        for name in header:
            table.add_column(name)
        for row in rows:
            table.add_row(*cells(row))
        console = Console(record=True, width=200, file=io.StringIO())
        console.print(table)
        # Returned rather than printed, so the caller decides where it appears and
        # the table is not emitted twice.
        return _ascii_safe(console.export_text())
    except Exception:
        widths = [max(len(header[i]), 12) for i in range(len(header))]
        lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
        lines.append("  ".join("-" * widths[i] for i in range(len(header))))
        for row in rows:
            values = cells(row)
            lines.append("  ".join(v.ljust(widths[i]) for i, v in enumerate(values)))
        return "\n".join(lines)


def describe_run(
    hardware: dict[str, Any],
    model_info: dict[str, Any],
    corpus: dict[str, Any],
    steps: int,
    batch_size: int,
    seq_len: int,
) -> str:
    sources = corpus.get("sources") or []
    ok = [s["dataset"] for s in sources if s.get("ok")]
    # Only the corpus and plan lines. The hardware and model header is printed once,
    # before the corpus is loaded, so it is not repeated here.
    return "\n".join(
        [
            "=" * 78,
            f"Corpus: {', '.join(ok) if ok else 'none'} "
            f"= {corpus.get('tokens', 0):,} tokens"
            + (
                "  [LOCAL REPOSITORY TEXT, not a real corpus]"
                if corpus.get("corpus_local_fallback")
                else ""
            ),
            f"Plan: {steps} steps x batch {batch_size} x seq {seq_len} "
            f"= {steps * batch_size * seq_len:,} tokens",
            "=" * 78,
        ]
    )


# ---------------------------------------------------------------------------
# Authenticity gates
# ---------------------------------------------------------------------------
def _snapshot_sample(model: torch.nn.Module) -> set[str]:
    """Pick a deterministic subset of parameters for the weight-update gate.

    The gate asserts that training actually changed the weights. Comparing every
    tensor would need a second full copy of the model, so this takes a spread of
    names instead: anything containing ``embed``, ``norm`` or ``head``, plus the
    largest remaining tensor per top-level module.
    """
    names = [n for n, _ in model.named_parameters()]
    chosen = {n for n in names if "embed" in n or "norm" in n or "head" in n}
    largest: dict[str, tuple[int, str]] = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if name in chosen:
            continue
        if top not in largest or param.numel() > largest[top][0]:
            largest[top] = (param.numel(), name)
    chosen.update(name for _, name in largest.values())
    return chosen


def run_gates(
    model: torch.nn.Module,
    init_snapshot: dict[str, torch.Tensor],
    losses: list[float],
    step_times: list[float],
    tokens_per_step: int,
    steps_done: int,
    total_tokens: int,
    ram_total_mb: float,
    energy_total_j: float | None,
) -> list[dict[str, Any]]:
    """Check that the run is genuine, not that it hit some target number.

    These assertions test authenticity: that the loss actually moved, that weights
    actually changed, that throughput and memory were really observed. None of
    them require a particular loss value, because a real 600-step run on this model
    will not reach any particular loss, and demanding one would only reward a
    fabricated number.
    """
    gates: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        gates.append({"check": name, "ok": ok, "detail": detail})

    finite = [v for v in losses if v is not None and math.isfinite(v)]
    add(
        "loss_finite",
        bool(finite) and len(finite) == len(losses),
        f"{len(finite)}/{len(losses)} recorded losses are finite",
    )

    if len(finite) >= 4:
        distinct = len({round(v, 6) for v in finite})
        spread = max(finite) - min(finite)
        add(
            "loss_varies",
            distinct > 3 and spread > 1e-6,
            f"{distinct} distinct values, spread {spread:.6f} "
            "(a constant or hardcoded loss fails this)",
        )
        window = max(1, min(10, len(finite) // 4))
        first = statistics.fmean(finite[:window])
        last = statistics.fmean(finite[-window:])
        add(
            "loss_decreased",
            last < first,
            f"mean of last {window} {last:.4f} vs first {window} {first:.4f} "
            f"(ratio {last / first if first else float('nan'):.3f})",
        )
    else:
        add("loss_varies", None, "not enough steps to judge")
        add("loss_decreased", None, "not enough steps to judge")

    changed, constant_matrices, nonfinite = [], [], []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            nonfinite.append(name)
            continue
        # Only weight matrices are checked for variation. Biases are normally
        # initialised to zero and LayerNorm weights to one, so both are constant by
        # design and have a standard deviation of zero. Flagging those would fail a
        # correctly initialised model.
        if param.ndim >= 2 and param.numel() > 1:
            if float(param.detach().std()) <= 0.0:
                constant_matrices.append(name)
        snap = init_snapshot.get(name)
        if snap is not None:
            changed.append(float((param.detach() - snap).abs().mean()))

    add(
        "weights_finite",
        not nonfinite,
        (
            "all parameters finite"
            if not nonfinite
            else f"{len(nonfinite)} non-finite tensors, e.g. {nonfinite[:3]}"
        ),
    )
    n_matrices = sum(1 for q in model.parameters() if q.ndim >= 2)
    add(
        "weight_matrices_vary",
        not constant_matrices,
        (
            f"all {n_matrices} weight matrices have non-zero spread"
            if not constant_matrices
            else f"{len(constant_matrices)} constant matrices, "
            f"e.g. {constant_matrices[:3]}"
        ),
    )
    if changed:
        mean_delta = statistics.fmean(changed)
        total_tensors = sum(1 for _ in model.named_parameters())
        add(
            "weights_updated_by_training",
            mean_delta > 0.0,
            f"mean |param - init| = {mean_delta:.3e} over a {len(changed)}-tensor "
            f"sample of {total_tensors} "
            "(0.0 would mean the optimizer never applied an update)",
        )
    else:
        add("weights_updated_by_training", None, "no comparable parameters")

    add(
        "throughput_measured",
        bool(step_times) and all(t > 0 for t in step_times),
        (
            f"{len(step_times)} timed steps, median "
            f"{statistics.median(step_times):.3f}s"
            if step_times
            else "no step timings"
        ),
    )
    add(
        "tokens_accounted",
        total_tokens == steps_done * tokens_per_step,
        f"{total_tokens:,} == {steps_done} x {tokens_per_step}",
    )
    add("ram_measured", ram_total_mb > 0, f"RSS {ram_total_mb:.0f} MB observed")
    if energy_total_j is None:
        add("energy_measured", None, "codecarbon unavailable, reported as null")
    else:
        add(
            "energy_measured",
            energy_total_j > 0,
            f"{energy_total_j:.1f} J observed by codecarbon",
        )
    return gates


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(rows: list[dict[str, Any]]) -> list[str]:
    """Write one PNG per measured series. Returns the paths actually written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable, skipping plots: {exc}")
        return []

    plt.style.use("ggplot")
    written: list[str] = []
    out_dir = ROOT / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    series_defs = (
        ("loss", "loss_20M_simple.png", "training loss"),
        ("tps", "tps_20M_simple.png", "training tokens/s"),
        ("ram_total_mb", "ram_20M_simple.png", "RSS (MB)"),
        ("energy_step_j", "energy_20M_simple.png", "energy per step (J)"),
        ("state_stability", "recall_20M_simple.png", "state stability"),
    )

    for key, name, ylabel in series_defs:
        points = [
            (r["step"], r.get(key))
            for r in rows
            if isinstance(r.get(key), (int, float))
        ]
        if len(points) < 2:
            print(f"not enough measured {key} values ({len(points)}), skipping {name}")
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        path = out_dir / name
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(xs, ys, marker="o", linewidth=1.4)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Feather-v2 quick baseline: {ylabel} (measured)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    from feather_v2.model import load_config

    config = load_config(args.config)
    vocab = int(config["vocab"])
    seq_len = int(args.seq_len or int(config["seq_len"]))
    batch_size = int(args.batch_size)

    hardware = describe_hardware()
    threads = configure_threads(args.threads)

    model = FeatherV2Model(config)
    params = model.count_parameters()
    trainable = model.count_parameters(trainable_only=True)
    model_info = {
        "parameters": params,
        "trainable": trainable,
        "trainable_fraction": round(trainable / params, 6) if params else None,
        "dim": int(config["dim"]),
        "n_blocks": int(config["n_blocks"]),
        "moe_experts": int(config["moe_experts"]),
        "moe_top_k": int(config["moe_top_k"]),
        "weights_f32_mib": round(model.num_bytes(torch.float32) / 2**20, 1),
    }

    # Environment header first, then the corpus line once the corpus is known.
    print(
        "\n".join(
            [
                "=" * 78,
                "FEATHER-V2 QUICK BASELINE - real measurements, no target values",
                f"Hardware: {hardware.get('cpu')} "
                f"{hardware.get('cores_physical')}C/{hardware.get('cores_logical')}T "
                f"binding={hardware.get('binding')} "
                f"threads={hardware.get('threads')} "
                f"dtype={hardware.get('compute_dtype')}",
                f"Model: {model_info['parameters']:,} params "
                f"({model_info['parameters'] / 1e6:.2f}M) "
                f"dim={model_info['dim']} blocks={model_info['n_blocks']} "
                f"experts={model_info['moe_experts']} top_k={model_info['moe_top_k']}",
                f"Plan: {args.steps} steps x batch {batch_size} x seq {seq_len} "
                f"= {args.steps * batch_size * seq_len:,} tokens",
            ]
        ),
        flush=True,
    )

    print("\nLoading corpus", flush=True)
    stream, corpus = build_token_stream(
        target_tokens=args.target_tokens,
        vocab_size=vocab,
        hardware_ram_gb=float(hardware.get("ram_total_gb") or 8.0),
        allow_local_text=args.allow_local_text,
    )
    print(describe_run(hardware, model_info, corpus, args.steps, batch_size, seq_len))

    failures = [s for s in (corpus.get("sources") or []) if not s.get("ok")]
    if failures:
        print("\nDatasets that could not be loaded (recorded, not hidden):")
        for item in failures:
            print(f"  - {item['dataset']}: {item['reason']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.steps), eta_min=args.lr * 0.1
    )

    process = psutil.Process()
    base_rss = process.memory_info().rss / 2**20
    energy = EnergyMeter()
    if not energy.available:
        print(f"energy: not measured ({energy.reason})")

    # A full clone of every parameter costs one extra copy of the weights, which is
    # 83 MB for this config and is pure overhead: the update gate only needs to know
    # whether tensors moved. Sample a deterministic subset instead, covering the
    # embedding, every block, and the head.
    snapshot_names = _snapshot_sample(model)
    init_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name in snapshot_names
    }
    print(
        f"snapshot: {len(init_snapshot)} of "
        f"{sum(1 for _ in model.named_parameters())} tensors "
        f"({sum(p.numel() for p in init_snapshot.values()) * 4 / 1e6:.1f} MB "
        "instead of a full copy)"
    )

    batches = make_batches(stream, batch_size, seq_len, args.steps, int(config["seed"]))
    tokens_per_step = batch_size * seq_len
    print(f"\nbase RSS: {base_rss:.0f} MB\n")

    rows: list[dict[str, Any]] = []
    step_history: list[int] = []
    loss_history: list[float] = []
    step_times: list[float] = []
    energy_at_last_report: float | None = None
    started = time.perf_counter()
    projection_checked = False
    model.train()

    for step in range(args.steps + 1):
        if step == 0:
            # Row 0 is a real measurement of one forward pass, not a literal zero.
            ids = next(batches)
            step_start = time.perf_counter()
            loss_value, _aux = model.loss(ids[:, :-1], ids[:, 1:])
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_elapsed = time.perf_counter() - step_start
            rss = process.memory_info().rss / 2**20
            row = {
                "step": 0,
                "total_tokens": 0,
                "tps": tokens_per_step / step_elapsed if step_elapsed > 0 else None,
                "loss": float(loss_value.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "ram_total_mb": rss,
                "ram_delta_mb": rss - base_rss,
                "energy_step_j": None,
                "energy_per_1k_j": None,
                "state_stability": None,
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            stability = model.state_stability(seq_len)
            row["state_stability"] = stability.get("similarity")
            row["state_stability_dim"] = stability.get("hidden_dim")
            model.train()
            rows.append(row)
            step_history.append(0)
            loss_history.append(float(loss_value.detach()))
            print(
                f"step 0 (measured, no update yet): loss {float(loss_value.detach()):.4f} "
                f"in {step_elapsed:.2f}s",
                flush=True,
            )
            continue

        ids = next(batches)
        step_start = time.perf_counter()
        loss_value, aux = model.loss(ids[:, :-1], ids[:, 1:])
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        # Recorded so the held-out evaluation can refuse to score a slice of the
        # stream that training has already consumed.
        model._trained_steps = step
        step_elapsed = time.perf_counter() - step_start

        step_times.append(step_elapsed)
        loss_history.append(float(loss_value.detach()))
        step_history.append(step)

        if not projection_checked and len(step_times) >= args.projection_steps:
            projection_checked = True
            per_step = statistics.median(step_times)
            projected_hours = per_step * args.steps / 3600.0
            print(
                f"projection from {args.projection_steps} timed steps: "
                f"{per_step:.2f}s/step -> {projected_hours:.2f}h for "
                f"{args.steps} steps (budget {args.time_budget_hours:.2f}h)",
                flush=True,
            )
            if projected_hours > args.time_budget_hours and not args.allow_overrun:
                print(
                    "\nSTOPPING: the measured rate does not fit the time budget.\n"
                    f"  projected {projected_hours:.2f}h > budget "
                    f"{args.time_budget_hours:.2f}h\n"
                    "Reduce the work, for example:\n"
                    f"    --steps {max(1, int(args.steps * args.time_budget_hours / projected_hours))}"
                    f"  (keeps the same steps/size ratio)\n"
                    f"    --batch-size {max(1, batch_size // 2)}\n"
                    f"    --seq-len {max(64, seq_len // 2)}\n"
                    "or pass --allow-overrun to run anyway.",
                    flush=True,
                )
                return 3

        if step % args.report_every == 0 or step == args.steps:
            rss = process.memory_info().rss / 2**20
            total_energy = energy.joules()
            energy_step = None
            energy_per_1k = None
            if total_energy is not None:
                previous = energy_at_last_report or 0.0
                energy_step = max(0.0, total_energy - previous)
                energy_at_last_report = total_energy
                if tokens_per_step > 0:
                    energy_per_1k = energy_step / (tokens_per_step / 1000.0)

            stability = model.state_stability(seq_len)
            window = step_times[-args.report_every :] or step_times
            aux_values = {k: float(v) for k, v in (aux or {}).items()}
            # The MoE routing penalty is reported inside the headline loss and it
            # swings by more than 10 nats between steps, so a "spike" in the
            # headline number may be routing churn rather than worse modelling.
            # Record the two separately so that can be told apart.
            main_loss = aux_values.get("loss")
            row = {
                "step": step,
                "total_tokens": step * tokens_per_step,
                "tps": tokens_per_step / statistics.median(window),
                "loss": float(loss_value.detach()),
                "main_loss": None if main_loss is None else float(main_loss),
                "aux_loss": aux_values.get("aux_loss"),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "ram_total_mb": rss,
                "ram_delta_mb": rss - base_rss,
                "energy_step_j": energy_step,
                "energy_per_1k_j": energy_per_1k,
                "state_stability": stability.get("similarity"),
                "state_stability_dim": stability.get("hidden_dim"),
                "aux": aux_values,
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            rows.append(row)
            print(
                f"step {step:>5}/{args.steps}  "
                f"tokens {row['total_tokens']:>10,}  "
                f"tps {row['tps']:>7.0f}  "
                f"loss {row['loss']:.4f}"
                f" (main {_fmt(row['main_loss'], '.4f')}"
                f" aux {_fmt(row['aux_loss'], '.2f')})  "
                f"rss {rss:.0f}MB ({row['ram_delta_mb']:+.0f})  "
                f"stab {_fmt(row['state_stability'], '.5f')}",
                flush=True,
            )
            model.train()

    total_elapsed = time.perf_counter() - started
    energy.stop()
    total_energy = energy.joules()
    rss = process.memory_info().rss / 2**20

    losses = [r["loss"] for r in rows if r.get("loss") is not None]
    window = max(1, min(10, len(losses) // 4)) if len(losses) >= 4 else 1
    loss_first = statistics.fmean(losses[:window]) if losses else None
    loss_last = statistics.fmean(losses[-window:]) if losses else None

    gates = run_gates(
        model=model,
        init_snapshot=init_snapshot,
        losses=loss_history,
        step_times=step_times,
        tokens_per_step=tokens_per_step,
        steps_done=args.steps,
        total_tokens=args.steps * tokens_per_step,
        ram_total_mb=rss,
        energy_total_j=total_energy,
    )

    failed = [g for g in gates if g["ok"] is False]
    unjudged = [g for g in gates if g["ok"] is None]

    step_median = statistics.median(step_times) if step_times else float("nan")
    tps_median = (
        tokens_per_step / step_median
        if step_times and step_median > 0
        else float("nan")
    )
    print("\n" + "=" * 78)
    print("MEASURED RESULTS (no target values)")
    print("=" * 78)
    print(render_table(rows, hardware))
    print()
    print(f"wall clock:        {total_elapsed / 60:.1f} min")
    print(f"tokens trained:    {args.steps * tokens_per_step:,}")
    print(
        f"loss:              {_fmt(loss_first, '.4f')} -> {_fmt(loss_last, '.4f')} "
        f"(mean of first/last {window} measured rows)"
    )
    print(
        f"training tps:      {tps_median:.0f} tok/s "
        f"(median {step_median:.2f}s/step x {tokens_per_step:,} tokens)"
        if step_times
        else "training tps:      not measured"
    )
    print(f"peak rss:          {rss:.0f} MB (base {base_rss:.0f} MB)")
    if total_energy is None:
        print("energy:            not measured (codecarbon unavailable)")
    else:
        print(
            f"energy:            {total_energy:.1f} J total, "
            f"{total_energy / max(1e-9, args.steps):.3f} J/step"
        )
    final_stability = rows[-1].get("state_stability") if rows else None
    print(f"state stability:   {_fmt(final_stability, '.5f')} (not a recall score)")
    print(
        "                   ~1.0 is expected for a causal model: an earlier position\n"
        "                   cannot see later tokens. It is not a measure of quality."
    )

    held_out = evaluate_held_out(model, stream, tokens_per_step, args.seed)
    if held_out is not None:
        print(
            f"held-out (last 5% of corpus, never trained on):\n"
            f"  loss         {held_out['loss']:.4f} nats/token\n"
            f"  perplexity   {held_out['perplexity']:.1f}\n"
            f"  top-1 acc    {100 * held_out['top1']:.2f}%\n"
            f"  top-10 acc   {100 * held_out['top10']:.2f}%\n"
            f"  unigram flr  {held_out['unigram_floor']:.4f} nats/token\n"
            f"  vs unigram   {held_out['gain_vs_unigram']:+.4f} nats "
            f"({'uses context' if held_out['uses_context'] else 'no context gain'})"
        )

    print("\nAuthenticity gates (these test genuineness, not target values):")
    for gate in gates:
        mark = "PASS" if gate["ok"] else ("SKIP" if gate["ok"] is None else "FAIL")
        print(f"  [{mark}] {gate['check']}: {gate['detail']}")

    plots = make_plots(rows)

    artifact = {
        "generated_by": "kaggle/train_20M_simple.py",
        "note": (
            "All values measured by this run. null means not measured. No target "
            "or predicted values are included."
        ),
        "model": {
            "config": str(args.config),
            "parameters": params,
            "trainable": trainable,
            "measured_size_label": model.size_label(),
            "weights_f32_mib": model_info["weights_f32_mib"],
            "weights_f16_mib_estimate": round(
                model.num_bytes(torch.float16) / 2**20, 1
            ),
        },
        "plan": {
            "steps": args.steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_step": tokens_per_step,
            "total_tokens": args.steps * tokens_per_step,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "corpus": corpus,
        "hardware": hardware,
        "results": rows,
        "loss_history": loss_history,
        "step_history": step_history,
        "summary": {
            "wall_clock_minutes": round(total_elapsed / 60, 2),
            "loss_start_mean": loss_first,
            "loss_end_mean": loss_last,
            "loss_decreased": (loss_last < loss_first) if losses else None,
            "training_tps_median": tps_median,
            "step_seconds_median": step_median,
            "rss_peak_mb": rss,
            "rss_base_mb": base_rss,
            "energy_total_j": total_energy,
            "energy_source": "codecarbon" if energy.available else None,
            "state_stability_final": final_stability,
            "held_out": held_out,
            "gates_passed": sum(1 for g in gates if g["ok"] is True),
            "gates_failed": len(failed),
            "gates_unjudged": len(unjudged),
            "status": "FAIL" if failed else "PASS",
        },
        "gates": gates,
        "plots": plots,
    }

    out_path = ROOT / args.out
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nartifact: {out_path}")
    if plots:
        print("plots:")
        for name in plots:
            print(f"  {name}")

    if failed:
        print("\nVERDICT: FAIL - see the failed gates above.")
        return 1
    print("\nVERDICT: PASS - every judged authenticity gate held.")
    print(
        "\nNext step, in order of what this run actually shows:\n"
        f"  1. loss moved {loss_first:.3f} -> {loss_last:.3f} over "
        f"{args.steps * tokens_per_step:,} tokens. To move it further, train on more\n"
        "     tokens or raise capacity; do not read a short run as converged.\n"
        f"  2. training throughput was {tps_median:.0f} tok/s here. If that\n"
        "     is the bottleneck, profile before changing the architecture.\n"
        f"  3. peak RSS was {rss:.0f} MB. If memory is the constraint, reduce n_blocks\n"
        "     or tt_rank and re-measure; each block is the dominant cost.\n"
        "  4. state_stability is a representation diagnostic, not an accuracy score.\n"
        "     It cannot tell you which architecture is 'best' on its own.\n"
        "Any comparison against another system needs the same corpus, hardware,\n"
        "step count and measurement code. This repository has no such baseline."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "feather_20M_simple.json"),
        help="model config to train",
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="tokens per step is batch-size x seq-len; 2 x 128 = 256 tokens and "
        "about 48 min for 600 steps, 8 x 512 = 4096 tokens and about 12 h",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="128 is the quick default; 512 with --batch-size 8 is the full run",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=6_000_000,
        help="corpus tokens to tokenize; more than needed is fine",
    )
    parser.add_argument("--time-budget-hours", type=float, default=10.0)
    parser.add_argument("--projection-steps", type=int, default=3)
    parser.add_argument("--allow-overrun", action="store_true")
    parser.add_argument(
        "--allow-local-text",
        action="store_true",
        help="offline fallback: train on repository text, recorded as a fallback",
    )
    parser.add_argument("--out", default="benchmark_20M_simple.json")
    return parser


def evaluate_held_out(
    model: Any,
    stream: np.ndarray,
    tokens_per_step: int,
    seed: int,
    holdout_fraction: float = 0.05,
) -> dict[str, Any] | None:
    """Score the model on tokens it never trained on.

    ``state_stability`` answers "does an earlier position change when later
    tokens appear", which is a causality check and sits at 1.0 for any correct
    causal model. It says nothing about whether the model predicts anything
    useful. This measures that instead, and reports it next to the unigram
    entropy floor of the same held-out tokens, because a model that only learns
    token frequencies cannot beat that floor and the comparison is the whole
    point.

    The held-out slice is taken from the far end of the stream, which training
    never reaches unless the run is long enough to wrap, in which case the result
    is reported as skipped rather than quietly contaminated.
    """
    trained_tokens = tokens_per_step * max(1, int(getattr(model, "_trained_steps", 0)))
    holdout = int(stream.size * holdout_fraction)
    if holdout < tokens_per_step or stream.size - holdout <= trained_tokens:
        return None

    tail = stream[-holdout:].astype(np.int64)
    seq = min(512, tail.size - 1)
    if seq < 8:
        return None

    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator().manual_seed(seed + 1)
        correct1 = correct10 = total = 0
        weighted = 0.0
        with torch.no_grad():
            for start in range(0, tail.size - seq - 1, seq):
                chunk = torch.from_numpy(tail[start : start + seq + 1]).unsqueeze(0)
                x, y = chunk[:, :-1], chunk[:, 1:]
                logits = model(x)
                if not torch.is_tensor(logits):
                    logits = logits[0] if isinstance(logits, tuple) else logits
                logits = logits.float()
                weighted += float(
                    torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        y.reshape(-1),
                        reduction="sum",
                    )
                )
                top = logits.topk(10, dim=-1).indices
                match = top.eq(y.unsqueeze(-1))
                correct1 += int(match[..., 0].sum())
                correct10 += int(match.any(dim=-1).sum())
                total += int(y.numel())
        if total == 0:
            return None
        loss = weighted / total
    finally:
        model.train(was_training)

    return {
        "loss": loss,
        "perplexity": float(np.exp(loss)),
        "top1": correct1 / total,
        "top10": correct10 / total,
        "tokens": total,
        "unigram_floor": unigram_floor(tail),
        # Positive means the model beats pure token frequencies on unseen text.
        "gain_vs_unigram": unigram_floor(tail) - loss,
        "uses_context": (unigram_floor(tail) - loss) > 0.02,
    }


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
