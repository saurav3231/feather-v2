# Feather v2

A small, CPU-trainable language model built from seven components and a library of
differentiable mathematical operators. Everything in this repository is either
implemented and tested, or explicitly marked as not implemented.

**Status: trainable core complete. Runtime measured. Quantised export not implemented.**

---

## What is real here

| Area | State |
| --- | --- |
| Trainable model | Implemented in PyTorch, 136 tests passing |
| Parameter counts | Measured from instantiated models |
| Checkpointing | Real PTH save/load via `torch.save` |
| Training loop | Real backprop; measured loss decrease |
| Runtime benchmark | Measured on one CPU, recorded in `benchmark_report.json` |
| GGUF / Q4_K_M export | **Not implemented.** No llama.cpp writer exists |
| Benchmark accuracy (MMLU etc.) | **Not measured.** Not claimed |
| GPU / BitNet comparisons | **Not measured.** Not claimed |

Read [`docs/RESULTS_v2.0.md`](docs/RESULTS_v2.0.md) for the current numbers. It is
generated from `configs/size_report.json` and `benchmark_report.json` by
`scripts/make_report.py`, so it cannot drift from the artifacts.

---

## Measured model sizes

All parameters are trainable. F16 size is `parameters x 2` bytes; no quantisation is
applied.

| Config | Measured label | Parameters | F32 | F16 |
| --- | --- | ---: | ---: | ---: |
| `configs/feather_5M.json` | 5.06M | 5,055,020 | 19.28 MiB | 9.64 MiB |
| `configs/feather_10M.json` | 9.60M | 9,604,412 | 36.64 MiB | 18.32 MiB |
| `configs/feather_20M.json` | 19.52M | 19,520,508 | 74.46 MiB | 37.23 MiB |
| `configs/feather_40M.json` | 40.34M | 40,337,084 | 153.87 MiB | 76.94 MiB |
| `configs/feather_60M.json` | 58.37M | 58,368,714 | 222.66 MiB | 111.33 MiB |

The labels in each filename are the *measured* counts, not targets. All five
configurations fit under 128 MiB in F16. Reproduce with:

```bash
python scripts/measure_sizes.py
```

---

## Quick start

From the root of this repository (no clone step needed):

```bash
pip install -e .

# Report detected CPU features and the kernel binding that will be used.
python -m feather_v2.hardware

# Build a model and print its measured manifest.
python -c "from feather_v2 import FeatherV2Model; import json; print(json.dumps(FeatherV2Model().describe(), indent=2, default=str))"
```

Train a small model and save a real checkpoint:

```bash
python scripts/train.py --config configs/feather_5M.json --steps 40 --pth runs/feather.pth
```

This writes the trained weights to `runs/feather.pth` and a measured manifest to
`runs/feather.json`. Use `--config ""` for the built-in default configuration.

---

## Architecture

Seven components, each a real `nn.Module` with real parameters:

1. **`SensoryEncoder`** — multi-scale fractional encoder with p-adic scale selection.
2. **`LiquidMemory`** — hierarchical chunked gated recurrence.
3. **`HyperDimensionalMemory`** — holographic mixing through a Walsh-Hadamard basis.
4. **`KnowledgeVault`** — sparse mixture of TT-compressed experts with Sinkhorn-balanced
   routing, plus a load-balancing penalty.
5. **`CognitiveWeaver`** — Kolmogorov-Arnold feed-forward with a differentiable Godel
   loop.
6. **`HomeostasisGovernor`** — predictive entropy gate that rescales the residual stream.
7. **`GenerativeEvolution`** — Jacobi-spectral refinement with a learned mutation step.

Each component also instantiates `TTExpert`, a tensor-train factorised expert used by the
mixture-of-experts path.

The differentiable operator library in `src/feather_v2/nn_math.py` includes:
fast Walsh-Hadamard transform and its inverse, tropical (softmin) matmul, fractional
weights, p-adic weight profiles, TT factor construction and matmul, Sinkhorn projection,
sheaf consistency projection, Clifford-style gating, rough-path signatures, Godel log
coding, alpha dropout, and `KANLinear`.

### On the "discrete" operators

Several components are named after discrete mathematics but cannot be implemented
differentiably on a GPU. Where that is the case, the repository states the relaxation
explicitly:

- **p-adic divisibility** is an exact integer descriptor, so it is **detached** from the
  graph. It informs gating but receives no gradient.
- **tropical / min-plus** operations use a softmin relaxation with a temperature, so they
  are differentiable but approximate.
- **Godel coding** uses `log`-based coding to stay finite on large messages. An earlier
  absolute-value form produced zero gradients and was fixed.
- **Fractional, TT, rough-path, Sinkhorn, sheaf and Clifford** operations are smooth
  relaxations, not exact discrete algorithms.

The divisibility descriptor is the only genuinely exact operator; it is also the only one
that is not trainable end-to-end.

---

## Hardware adaptation

`feather_v2.hardware` detects CPU features and selects a kernel binding, falling back
without failing:

```
AVX-512 -> AVX2 -> AVX -> NEON -> scalar
```

The reference measurements in this repository were taken on an Intel i5-3337U
(2 physical / 4 logical cores, ~3 GB RAM, AVX, no AVX2). Throughput on other CPUs will
differ and has not been measured here.

---

## Export

`FeatherV2Model.save_pth(path)` writes a PyTorch PTH checkpoint. There is **no GGUF
exporter**. A file named `.gguf` will not be produced by this project, and a test
(`tests/test_gguf.py`) fails if a `.gguf` file appears in the tree without a real
exporter being present.

The earlier `feather-v2-40M-f16.gguf` and `feather-v2-40M-Q4_K_M.gguf` files that were
committed to this repository were not real GGUF files. They have been deleted.

---

## Repository layout

```
src/feather_v2/
  nn_math.py         differentiable operator library
  nn_components.py   the seven trainable components
  model.py           FeatherV2Model, config loading, training/loss/generation
  hardware.py        CPU feature detection and kernel selection
  utils.py           NumPy reference implementations and v1 compatibility helpers
  base.py            component base classes
  config.py          config schema
configs/             measured size ladder + size_report.json
scripts/
  train.py           real training loop
  measure_sizes.py   parameter and byte-size measurement
  make_report.py     generates docs/RESULTS_v2.0.md
kaggle/
  test_all_sizes_mega.py   real CPU benchmark across the size ladder
docs/                architecture, design, benchmark, measured results
paper/               paper source
```

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the operator library, component gradients, numerical stability of the
Godel coder, real model training, checkpoint round trips, config validation, and a guard
against mislabelled GGUF artifacts.

---

## License

MIT License plus an additional "No Big Tech Clause". See [`LICENSE`](LICENSE).

---

**Author:** Saurav Bhandari, Pokhara, Nepal
