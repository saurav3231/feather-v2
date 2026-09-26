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

### Training baseline on Kaggle or Colab

`configs/feather_20M_simple.json` measures **20,696,188 parameters (20.70M)**. Check
that yourself rather than trusting the filename:

```bash
python -m feather_v2.model --config configs/feather_20M_simple.json --count-params
```

#### Two configs at the same width: pick on measurement

`hyper.up` and `hyper.down` are `dim x hv_dim` each, so `hv_dim` alone decides
55.8% of the parameter count and, because the Walsh-Hadamard transform is
`O(hv_dim log hv_dim)`, most of the step time as well. Both configs below use
`dim 352` and 2 blocks; they differ only in `hv_dim`.

Measured on this machine, 2 threads, `batch 2 x seq 128`, same seed, same
Wikipedia text, 40 steps:

| config | `hv_dim` | params | main loss @40 | tok/s | RSS | RAM delta |
| --- | --- | --- | --- | --- | --- | --- |
| `feather_20M_simple.json` | 6144 | 20,696,188 | 8.0949 | 59 | 831 MB | +256 MB |
| `feather_10M_fast.json` | 1024 | 10,574,972 | 8.0916 | 126 | 675 MB | +139 MB |

At 40 steps the two configs are indistinguishable, which is what an earlier
version of this README reported as "same loss to four decimal places". **That
was wrong and it does not survive contact with a longer run.** Comparing at
matched token counts instead of matched steps:

| tokens | 20.70M, `seq 128` | 10.57M, `seq 512` |
| ---: | ---: | ---: |
| 25,600 | 6.6401 | 7.9906 |
| 51,200 | 6.2151 | 7.0308 |
| 76,800 | 6.0823 | 6.8481 |

The narrow config is ~2.1x faster per step and about **0.8 nats behind per
token**. It is a trade, not a free win: `hv_dim` buys speed and costs loss per
token. This comparison varies width and sequence length together, so it does not
separate the two; a 20.70M run at `seq 512` would. Pick by wall-clock budget,
and do not cite the 40-step row as evidence of equivalence.

```python
# fast: 10.57M params, ~2x throughput, ~0.8 nats/token behind
!python feather-v2/kaggle/train_20M_simple.py --config feather-v2/configs/feather_10M_fast.json --steps 600 --report-every 50 --out benchmark_10M_fast.json
```

See `docs/DIAGNOSIS_v2.0.md` for the full set of measurements, including which
operators actually dominate the step and which four common claims the numbers
contradict.

Note that the headline loss in the log includes the MoE routing penalty, which
swings by more than 10 nats between reports. The training line prints both parts,
and the artifact stores `main_loss` and `aux_loss` separately, so a rise in the
headline number is not automatically worse modelling.

### Token budget is the binding constraint

The default 600-step run consumes 600 x 2 x 128 = 153,600 tokens. Measured, a
12x increase in tokens from 12,800 to 153,600 buys only 0.25 nats, and the
unigram entropy floor of this corpus is 6.10-6.73 nats. A loss near 6.4 at
153,600 tokens means the model has learned token frequencies and little else.
This is a data limit, not a capacity or architecture limit: the same model
overfits 32 tokens to 0.14 loss in 25 steps, so the gradient path is fine.

Raising `--seq-len` to 512 is the cheapest way to buy more, because the
hyperdimensional matmuls are per-token and longer sequences amortise the fixed
costs. Measured at `batch 2` on `feather_10M_fast.json`:

| `seq_len` | tokens/step | tok/s | us/token |
| ---: | ---: | ---: | ---: |
| 128 | 254 | 145 | 6901.7 |
| 256 | 510 | 164 | 6096.9 |
| 512 | 1022 | 192 | 5219.6 |

`seq 512` is 1.32x the tokens per second and 4x the tokens per step, so roughly
5.3x the training progress per second of the `seq 128` default. For a fixed
600 steps that is 613,200 tokens instead of 153,600.

A loss of 2.0 at 600 steps is not reachable at any setting. Scaling the measured
curve puts 6.2 nats at roughly 19M tokens, which is about 18,500 steps at
`batch 2 x seq 512` or 5.5 hours on this class of CPU; 2.0 needs billions. If
that is a hard requirement, change the token budget, not the model.

One cell, from a fresh Kaggle or Colab session:

```python
!git clone --depth 1 https://github.com/saurav3231/feather-v2.git
!pip -q install -e feather-v2 codecarbon datasets matplotlib psutil
!python feather-v2/kaggle/train_20M_simple.py --steps 600 --report-every 50 --out benchmark_20M_simple.json
```

That uses the default `batch-size 2` and `seq-len 128`, which is 256 tokens per
step and projects to roughly 48 minutes for 600 steps on a 2-core machine. The
full-size run is 4096 tokens per step and projects to roughly 12 hours:

```python
!python feather-v2/kaggle/train_20M_simple.py --steps 600 --batch-size 8 --seq-len 512 --time-budget-hours 12
```

Both projections come from a measured 4.8 s/step at batch 2 x seq 128 on a
2-core/4-thread laptop. They are not completed runs. The script measures the first
three timed steps, projects the finish time, and stops if the projection exceeds
`--time-budget-hours`, so it will not silently run for hours. Loss, throughput,
RAM, and energy are printed only after being measured; where codecarbon is
unavailable the energy fields are `null` rather than estimated.

The five figures in `docs/images/` are generated by a run, not committed by hand.
A short smoke run also writes all five, so `docs/images/*.png` is git-ignored to
stop a 4-step plot from being published next to a claim of 600 steps. A full run's
figures are added deliberately once its JSON is committed.

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
