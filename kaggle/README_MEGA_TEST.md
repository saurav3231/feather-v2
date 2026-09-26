# Feather v2 — size ladder benchmark on Kaggle CPU

Runs the real model across the size ladder and records what it measures.

The benchmark performs actual forward passes, real backprop, and real autoregressive
generation. It does not estimate throughput, derive RAM from parameter counts, or
substitute a placeholder corpus when a dataset is unavailable.

---

## Quick start

1. Open <https://www.kaggle.com/code> and create a notebook.
2. Select the **CPU** accelerator, not GPU.
3. Internet on for the install cell.
4. **Upload this repository as a Kaggle dataset** (or otherwise place the project files
   in the notebook's working directory), then run the cells below.

Uploading the tree you want to measure is the recommended route. Cloning a public remote
is deliberately not automated, because it would quietly benchmark whatever is on that
remote's default branch instead of the code you are trying to test. If you do want a
clone, set `FEATHER_REPO` to an explicit URL and pin the branch or commit.

---

## Cell 1 — install

```python
!pip install -q torch psutil codecarbon rich matplotlib tabulate colorama tokenizers datasets numpy
!pip install -e . -q
```

`torch` is a real dependency of this package and is declared in `pyproject.toml`.

---

## Cell 2 — locate the repository

Adjust the path to wherever the repository was uploaded or extracted.

```python
import sys, pathlib
REPO = pathlib.Path("/kaggle/working/feather-v2")
sys.path.insert(0, str(REPO))
print(REPO.exists(), sorted(p.name for p in REPO.iterdir())[:10])
```

---

## Cell 3 — hardware detection

```python
from feather_v2.hardware import detect_cpu_features, summary

print(summary())
feats = detect_cpu_features()
print(feats)
```

`summary()` reports the detected features and the selected kernel binding. It does not
predict throughput; there is no per-tier tok/s estimate in this project.

---

## Cell 4 — run the benchmark

```python
from kaggle.test_all_sizes_mega import main

main()
```

Or from a shell:

```bash
python kaggle/test_all_sizes_mega.py --loss-steps 60
python kaggle/test_all_sizes_mega.py --sizes 5M --loss-steps 20   # quick check
```

`main()` writes `benchmark_report.json` and the plots. It returns `None`; read the JSON for
results.

---

## Cell 5 — read the results

The report's `sizes` field is a **list** of per-size records.

```python
import json
from rich.console import Console
from rich.table import Table

report = json.load(open("benchmark_report.json"))
console = Console()

table = Table(title="Feather v2 — measured, by size")
for col in ("Size", "Params", "RAM MB", "Fwd@512", "Bulk t/s", "Gen t/s", "Loss", "Energy J", "Checks"):
    table.add_column(col)

for r in report["sizes"]:
    ft = r.get("forward_throughput") or {}
    losses = r.get("losses") or []
    checks = r.get("checks") or []
    table.add_row(
        r.get("size_label", "?"),
        f"{r.get('params', 0):,}",
        f"{r.get('ram_mb', 0):.1f}",
        f"{ft.get('512', 0):.1f}",
        f"{r.get('bulk_tok_s', 0):.1f}",
        f"{r.get('gen_tok_s', 0):.2f}",
        f"{losses[0]:.3f} -> {losses[-1]:.3f}" if losses else "not measured",
        f"{r.get('energy_j', 0):.4g}",
        f"{sum(1 for _, v in checks if v)}/{len(checks)}",
    )

console.print(table)
print(report["summary"])
```

Check names are `weights_real`, `ram_real`, `timing_real`, `loss_trend`, `state_stability`,
and `init_ok`. Each entry in `checks` is a `(name, passed)` pair, and the value that
justified it is stored alongside in the same record.

---

## Cell 6 — view the plots

```python
from IPython.display import Image, display
import pathlib

for png in sorted((REPO / "docs" / "images").glob("*.png")):
    print(f"\n## {png.name}")
    display(Image(filename=str(png)))
```

---

## Cell 7 — regenerate the results document

```python
!python {REPO}/scripts/make_report.py
```

Writes `docs/RESULTS_v2.0.md` from the JSON artifacts, so the documented numbers cannot
drift from the measurements.

---

## Running one size at a time

```python
from kaggle.test_all_sizes_mega import test_one_size, CONFIG_MAP, load_streaming_datasets

datasets_info = load_streaming_datasets()
result = test_one_size("5M", CONFIG_MAP["5M"], datasets_info, loss_steps=20)
print(result["checks"])
```

`CONFIG_MAP` is loaded from `configs/`, so the ladder in the benchmark and the ladder in
the repository cannot diverge. Available labels are `sorted(CONFIG_MAP)`.

---

## Outputs

- `benchmark_report.json` — every measurement and check, per size
- `docs/images/scaling.png` — parameters against RAM
- `docs/images/memory_vs_params.png`
- `docs/images/loss_all_sizes.png`
- `docs/images/toks_vs_seqlen.png`
- `docs/images/energy_vs_size.png`

---

## Ladder

Five configurations, each named for its measured parameter count:

| Label | Parameters | Config |
| --- | ---: | --- |
| 5M | 5,055,020 | `configs/feather_5M.json` |
| 10M | 9,604,412 | `configs/feather_10M.json` |
| 20M | 19,520,508 | `configs/feather_20M.json` |
| 40M | 40,337,084 | `configs/feather_40M.json` |
| 60M | 58,368,714 | `configs/feather_60M.json` |

There are no 50M or 100M configurations. Those labels were removed because no config
produced those counts.

---

## Runtime expectations

Measured on a 2-core i5-3337U, the smallest config took roughly 10 minutes for 60 loss
steps. The larger configs are substantially slower, because the measurement is real
backprop on a CPU rather than a cached or synthesised number. On a faster machine expect
less; there is no fixed per-size estimate in this repository because none has been
measured across machines.

If a session runs short, run one size at a time with `--sizes`.

### Interruption and subset runs

`benchmark_report.json` is written **after every size**, not only when the whole ladder
finishes, so a session that is interrupted or times out keeps the sizes it already
measured.

Re-running merges by size rather than replacing the file, so a subset run such as
`--sizes 60M` updates only the 60M entry and leaves the others intact. Re-measuring a
size overwrites that entry in place instead of appending a duplicate. Each entry records
its own `measured_at` timestamp, so a report assembled from more than one session is
honest about that.

Practical recovery pattern after an interruption:

```bash
python kaggle/test_all_sizes_mega.py --sizes 5M 10M 20M 40M 60M --loss-steps 60
# interrupted during 60M? nothing is lost, and this replaces just that entry:
python kaggle/test_all_sizes_mega.py --sizes 60M --loss-steps 60
```

---

## Corpus availability

The benchmark attempts three corpora:

| Role | Identifier |
| --- | --- |
| Tokenisation | `Skylion007/openwebtext` |
| Training loss | `wikimedia/wikipedia`, config `20231101.en` |
| Generation text | `HuggingFaceH4/no_robots` |

If one is unavailable it is recorded as unavailable and the dependent measurement is
marked as such. No placeholder text is substituted.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'feather_v2'`**

The repository root must be on `sys.path`; see Cell 2.

**Out of memory on the largest config**

Run fewer sizes at once. The 58M configuration needs roughly 1.3 GB of weights plus the
PyTorch runtime.

**A check fails**

Read the value stored next to the check in `benchmark_report.json`. Every check records
its measurement, so a failure can be diagnosed rather than guessed at.

---

## Author

Saurav Bhandari, Pokhara, Nepal
