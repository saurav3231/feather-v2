"""One-cell Kaggle runner for the Feather v2 size-ladder benchmark.

Runs the real benchmark, then prints the verdict read from the JSON artifact it
produced. It does not assert any result of its own.

This is a plain Python script and is also valid as a single notebook cell. It
self-times its steps with ``time.perf_counter`` and prints the elapsed seconds, so
it needs no ``%%time`` magic.

Where the code comes from
-------------------------
By default this script does **not** clone anything. It expects the project to already
be present at ``WORK`` (the usual Kaggle route is to upload the repository as a dataset
and point ``WORK`` at the extracted copy). That is deliberate: cloning a public remote
would silently benchmark whatever is on that remote's default branch rather than the
tree you are actually trying to measure.

To clone explicitly, set the environment variable to the URL you want, for example::

    FEATHER_REPO=https://github.com/<you>/<fork>.git python onecell_kaggle_run.py

Point it at a commit or branch that contains the code you intend to measure.
"""

import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Where the code comes from. Empty means "use what is already on disk".
REPO = os.environ.get("FEATHER_REPO", "").strip()
WORK = os.environ.get("FEATHER_WORK", "/kaggle/working/feather-v2")
REPO_ROOT = WORK  # the project directory is the repository root


def run(cmd, desc=""):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"[{status}] {desc} ({elapsed:.1f}s)")
    if r.returncode != 0:
        print("STDOUT:", r.stdout[-800:])
        print("STDERR:", r.stderr[-800:])
    return r


print("=" * 60)
print("FEATHER-V2 SIZE LADDER BENCHMARK - KAGGLE CPU")
print("Work dir:", WORK)
print("Source:", REPO if REPO else "(using the copy already on disk)")
print("=" * 60)

if not os.path.exists(WORK):
    if REPO:
        run(f"git clone --depth 1 {REPO} {WORK}", "Clone repo")
    else:
        print(
            "\nERROR: no project found at",
            WORK,
            "\n"
            "Upload the repository as a Kaggle dataset and set FEATHER_WORK to the\n"
            "extracted path, or set FEATHER_REPO to clone a specific URL/branch.\n"
            "This script will not guess a remote, because benchmarking the wrong\n"
            "revision is worse than not running at all.",
        )
        raise SystemExit(2)
else:
    print("-> Project already present, using it as-is")

os.chdir(WORK)
run(
    "pip install -q torch psutil codecarbon rich matplotlib tabulate "
    "colorama tokenizers datasets numpy",
    "Install deps",
)
run("pip install -e . -q", "Install feather-v2")

# Make the package importable even if the editable install misses the src layout.
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import importlib.util

mega_path = os.path.join(REPO_ROOT, "kaggle", "test_all_sizes_mega.py")
spec = importlib.util.spec_from_file_location("test_all_sizes_mega", mega_path)
mega = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mega)

# Available ladder labels, read from configs/ rather than hardcoded.
print("\nLadder:", sorted(mega.CONFIG_MAP, key=lambda s: float(s.rstrip("M"))))

print("\n" + "=" * 60)
print("RUNNING BENCHMARK")
print("=" * 60 + "\n")

# main() writes benchmark_report.json itself and returns None.
mega.main()

print("\n" + "=" * 60)
print("VERDICT (read from benchmark_report.json)")
print("=" * 60)

report_path = os.path.join(REPO_ROOT, "benchmark_report.json")
if not os.path.exists(report_path):
    print("No report was produced. The benchmark did not complete.")
else:
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)

    # `sizes` is a list of per-size records.
    for rec in report.get("sizes", []):
        checks = rec.get("checks") or []
        passed = sum(1 for _, v in checks if v)
        failed = [name for name, v in checks if not v]
        print(
            f"  {rec.get('size_label', '?'):>5}  "
            f"params={rec.get('params', 0):>11,}  "
            f"checks={passed}/{len(checks)}"
            + (f"  failed={failed}" if failed else "")
            + (f"  error={rec['error']}" if rec.get("error") else "")
        )

    summary = report.get("summary") or {}
    print(
        f"\n  totals: {summary.get('total_pass', 0)}/{summary.get('total_checks', 0)} "
        "checks passed"
    )
    print(f"  report: {report_path}")

    print("\nPlots generated:")
    img_dir = os.path.join(REPO_ROOT, "docs", "images")
    if os.path.exists(img_dir):
        found = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".png")]
        for name in found:
            print("  ->", name)
        if not found:
            print("  (none)")
    else:
        print("  (no images dir found)")

print("\nRegenerate the results document with: python scripts/make_report.py")
