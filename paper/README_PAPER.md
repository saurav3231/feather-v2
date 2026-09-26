# Feather v2 paper

**Title:** Feather v2: a trainable CPU-native language model with measured results

## What this is

`main.py` builds `main.pdf` with reportlab. Every number in the document is read from
measurement artifacts at build time:

- `configs/size_report.json` — parameter counts and byte sizes
- `benchmark_report.json` — CPU runtime, RAM, throughput, loss, energy

Nothing is hardcoded in the paper source. If a measurement is missing, the document prints
"not measured" rather than a placeholder number. This means the paper cannot drift away
from what the code actually does.

## Build

Run the measurements first, then build:

```bash
python scripts/measure_sizes.py
python kaggle/test_all_sizes_mega.py --loss-steps 60
python paper/main.py
```

Output:

- `paper/main.pdf` — the paper
- `docs/images/paper_f16_size.pdf`
- `docs/images/paper_generation.pdf`
- `docs/images/paper_forward512.pdf`

Charts are written as PDF vector graphics directly from the measured values. Earlier
versions of this script wrote blank 400x200 placeholder PNGs and plotted invented numbers;
that code path has been removed.

## What the paper does not contain

- No accuracy benchmark. No MMLU or any other score.
- No comparison against other models or hardware. No baseline was measured under identical
  conditions.
- No energy-per-token figure. The energy number is a single measurement covering one
  forward pass plus a short generation.
- No composite efficiency metric. The "MOMR" figure that appeared in earlier drafts has no
  definition or implementation in this repository and has been dropped.
- No throughput predictions. `hardware.py` no longer emits per-tier tok/s estimates.

## Files

- `main.py` — paper generator
- `references.bib` — references

## Related documents

- `docs/ARCHITECTURE_FINAL_v2.0.md` — component structure and the exact-versus-relaxed
  breakdown
- `docs/MODEL_DESIGN_v2.0.md` — Python API
- `docs/BENCHMARK_v2.0.md` — benchmark methodology
- `docs/RESULTS_v2.0.md` — generated measured results
