"""Generate kaggle/notebook_train_20M_simple.ipynb.

The notebook is produced from a template rather than hand-written JSON so that the
code cells stay readable and diffable. Run this after changing the training script:

    python scripts/make_train_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "notebook_train_20M_simple.ipynb"

MD_INTRO = """# Feather v2 quick baseline - real training run

Trains the shipped `configs/feather_23M_simple.json` configuration and reports only
values measured during the run.

**What this notebook does not do:** it contains no target loss, no target
throughput, no predicted hardware numbers, and no recall score. The metric labelled
`StateStab` is `state_stability`, a cosine similarity between a position's hidden
state with and without a later suffix. It is not recall and not accuracy.

**Size note:** the config measures 23.12M parameters. The output filenames keep the
requested `20M_simple` label, but a configuration that genuinely measures 20.0M
needs `dim=352` (20.70M)."""

MD_SETUP = """## 1. Environment

Checks the interpreter, installs the local package, and reports what codecarbon can
actually measure on this machine. GPU is not used; this is a CPU baseline."""

MD_RUN = """## 2. Run the training script

The defaults match the requested baseline: 600 steps, batch 8, sequence length 512.

`--time-budget-hours` is a guard, not a target. The script measures the first three
timed steps, projects the finish time, and stops early with a clear message if the
projection exceeds the budget, so a long run cannot silently occupy a Kaggle session
for a day. Raise or lower it to match the session you are on.

The first run downloads a Wikipedia slice through `datasets`, which needs network
access. Without it, pass `--allow-local-text` to fall back to repository text; the
artifact is then flagged as a local fallback rather than a real corpus."""

MD_READ = """## 3. Read the results

`benchmark_20M_simple.json` holds the measurements, the per-step history, and the
authenticity gates. The gates test whether the run is genuine (finite and varying
loss, weights actually updated, tokens accounted for, energy actually observed).
They are not target checks: a short run can fail `loss_decreased` honestly, because
4 steps cannot show a trend.

A `VERDICT: FAIL` line with only `loss_decreased` failing is the expected result of
a short smoke run, not a broken model."""

MD_PLOTS = """## 4. Plots

All five figures are generated from the measured run only."""

CODE_SETUP = """import json
import os
import platform
import subprocess
import sys

!pip -q install -e . codecarbon datasets matplotlib psutil

print("python  :", sys.version.split()[0])
print("platform:", platform.platform())
try:
    import torch

    print("torch   :", torch.__version__, "| threads:", torch.get_num_threads())
except Exception as exc:  # noqa: BLE001
    print("torch   : MISSING", exc)

from pathlib import Path

ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
print("root    :", ROOT)"""

CODE_VERIFY = """# Confirm the package under test is the local one, not a stale wheel.
import feather_v2

print("feather_v2 :", feather_v2.__file__)
assert Path(feather_v2.__file__).resolve().is_relative_to(ROOT), (
    "importing feather_v2 from outside the working directory"
)
print("version    :", getattr(feather_v2, "__version__", "n/a"))"""

CODE_RUN = """STEPS = 600
BATCH = 8
SEQ = 512
BUDGET_HOURS = 10.0

cmd = [
    sys.executable,
    "kaggle/train_20M_simple.py",
    "--steps", str(STEPS),
    "--batch-size", str(BATCH),
    "--seq-len", str(SEQ),
    "--report-every", "50",
    "--out", "benchmark_20M_simple.json",
    "--time-budget-hours", str(BUDGET_HOURS),
]
print(" ".join(cmd), flush=True)
result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
print(result.stdout[-6000:])
if result.returncode != 0:
    print(result.stderr[-4000:])
    raise SystemExit(f"training script failed with code {result.returncode}")"""

CODE_READ = """import json as _json
from pathlib import Path as _Path

report_path = _Path("benchmark_20M_simple.json")
if not report_path.exists():
    raise SystemExit(
        "benchmark_20M_simple.json was not written, so there is nothing to report. "
        "Read the output of the previous cell first."
    )

report = _json.loads(report_path.read_text(encoding="utf-8"))
summary = report["summary"]

print("model      :", report["model"]["parameters"], "parameters")
print("corpus     :", report["corpus"].get("tokens"), "tokens from",
      [s["dataset"] for s in report["corpus"]["sources"] if s.get("ok")])
print("tokens     :", summary["tokens_trained"])
print("loss       :", summary.get("loss_start_mean"), "->", summary.get("loss_end_mean"))
print("tps        :", summary.get("training_tps_median"))
print("step secs  :", summary.get("step_seconds_median"))
print("peak rss   :", summary.get("rss_peak_mb"), "MB")
print("energy     :", summary.get("energy_total_j"), "J from", summary.get("energy_source"))
print("stability  :", summary.get("state_stability_final"), "(not a recall score)")
print()
for gate in report["gates"]:
    mark = "PASS" if gate["ok"] else ("SKIP" if gate["ok"] is None else "FAIL")
    print(f"[{mark}] {gate['check']}: {gate['detail']}")
print()
print("gates passed:", summary["gates_passed"], " failed:", summary["gates_failed"])"""

CODE_PLOTS = """from IPython.display import Image, display
from pathlib import Path as _P

figures = [
    "loss_20M_simple.png",
    "tps_20M_simple.png",
    "ram_20M_simple.png",
    "energy_20M_simple.png",
    "recall_20M_simple.png",
]
image_dir = _P("docs/images")
for name in figures:
    path = image_dir / name
    if path.exists():
        print(name)
        display(Image(filename=str(path)))
    else:
        print(f"{name}: not generated by this run")"""


def cell(kind: str, source: str) -> dict:
    lines = source.strip("\n").split("\n")
    body = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": body}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": body,
    }


def main() -> None:
    notebook = {
        "cells": [
            cell("markdown", MD_INTRO),
            cell("markdown", MD_SETUP),
            cell("code", CODE_SETUP),
            cell("code", CODE_VERIFY),
            cell("markdown", MD_RUN),
            cell("code", CODE_RUN),
            cell("markdown", MD_READ),
            cell("code", CODE_READ),
            cell("markdown", MD_PLOTS),
            cell("code", CODE_PLOTS),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
