# Feather v2 — Release notes v2.0.0

**Version:** 2.0.0
**Repository:** https://github.com/saurav3231/feather-v2
**Author:** Saurav Bhandari, Pokhara, Nepal
**License:** MIT + No Big Tech Clause (see `LICENSE`)

Measured numbers for this release are in [`docs/RESULTS_v2.0.md`](docs/RESULTS_v2.0.md),
which is generated from `configs/size_report.json` and `benchmark_report.json`.

---

## What v2.0.0 is

A trainable, CPU-only language model in PyTorch, built from seven components and a
library of differentiable mathematical operators, with a measured size ladder from 5M to
58M parameters.

The earlier releases of this project were inference-only scaffolding: the components
existed as NumPy reference implementations, and the published performance tables were not
produced by running anything. v2.0.0 replaces that with a real trainable model and a
benchmark that reports only what it measures.

---

## Added

- `nn_math.py`: 23 exported differentiable operators — Walsh-Hadamard transform and
  inverse, tropical and softmin matmul, fractional weights, p-adic weight profiles and
  distances, tensor-train factors/compression/matmul, Sinkhorn projection, sheaf
  consistency projection, Clifford-style gating, rough-path signatures, equilibrium and
  Jacobi iterations, Godel log coding, alpha dropout, and `KANLinear`.
- `nn_components.py`: seven trainable `nn.Module` components plus `TTExpert`.
- `model.py`: real `FeatherV2Model` with embeddings, `CognitiveBlock` stacks, cross-entropy
  plus MoE load-balancing loss, autoregressive generation, PTH checkpoints, and a measured
  `describe()` manifest.
- A five-step size ladder with each config's filename reflecting its **measured**
  parameter count.
- `scripts/measure_sizes.py` — parameter and byte-size measurement.
- `scripts/make_report.py` — generates `docs/RESULTS_v2.0.md` from measurement artifacts,
  so no documented number is hand-typed.
- A rewritten CPU benchmark that measures real forward, generation, RAM, throughput, loss,
  and energy, and records the value behind every pass/fail check.

## Fixed

- **Godel coder produced zero gradients.** The absolute-value formulation was replaced
  with a log-based one that stays finite and differentiable on large messages. Covered by
  a regression test.
- **Residual stream overflowed during training.** Added a per-component `LayerNorm`;
  a 60-step training run on the 5M config previously went non-finite.
- **p-adic profile shape error.** Contraction needed an unsqueezed mass axis.
- **Unused `SensoryEncoder.out_gate`.** Removed; the value path now feeds scale
  projections.
- **OpenWebText repository identifier** was the bare `openwebtext` name rather than the
  namespaced `Skylion007/openwebtext`.
- **Benchmark weights gate was broken.** `assert_real_weights` iterated
  `(name, parameter)` tuples as if they were parameters, raising a `TypeError` internally
  and reporting a failure unrelated to what it was checking.
- **Duplicate `_p_adic_group` definition** in `utils.py`; the second silently shadowed
  the first.

## Removed

These were published in earlier releases and were not real:

| Removed | Why |
| --- | --- |
| `feather-v2-40M-f16.gguf` | Not a GGUF file. A PTH renamed with a `.gguf` extension. |
| `feather-v2-40M-Q4_K_M.gguf` | No quantiser and no llama.cpp writer exist in this project. |
| `feather-v2-offline-v1.0.0.tar.gz` | Contained no verifiable offline bundle. |
| `feather_v2-1.0.0-py3-none-any.whl` | Built before the package was installable. |
| `test_v2_roundtrip.gguf` | Test fixture produced by the non-existent exporter. |
| 6 release charts and a release PDF | Blank 738-byte placeholders backing unmeasured claims. |
| `configs/feather_50M.json`, `configs/feather_100M.json` | Replaced by measured configs. |
| `configs/agent_env.json`, `configs/all_pcs.json`, `configs/i5_3337U.json` | Duplicated hardware guesses with no measurements behind them. |

`tests/test_gguf.py` now fails if a `.gguf` file appears in the tree without a real
exporter being registered, so mislabelled artifacts cannot silently return.

## Corrected claims

The following were published as measurements and were not. They are now stated as
unmeasured:

- Throughput on any machine other than the one in
  [`docs/BENCHMARK_v2.0.md`](docs/BENCHMARK_v2.0.md) — including i5-3337U, i7-12700,
  Kaggle, M3, and Raspberry Pi figures.
- Energy per token, and energy savings ratios.
- RAM figures. The real 5M model has ~19 MB of weights; process RSS is several hundred MB
  because of the PyTorch runtime.
- Comparisons against BitNet, Phi-4, GPU baselines, and "Transformer 7B".
- The `MOMR` metric and its ~120x claim. No implementation of that formula exists here.
- "Loss 18 -> 0.50" and similar end-to-end training results.
- "1M context" and context-recall claims beyond the measured sequence lengths.
- "0 multiplications" and the associated op-count savings.
- The CI status line "12/12 green 272+ tests". The suite has 144 tests.

---

## Verification for this release

```bash
python -m pytest tests/ -q          # 144 passed
python -m black --check src scripts tests kaggle
python scripts/measure_sizes.py
python kaggle/test_all_sizes_mega.py --loss-steps 60
python scripts/make_report.py
```

The training smoke test writes a real checkpoint:

```bash
python scripts/train.py --config configs/feather_5M.json --steps 40 --pth runs/feather.pth
```

---

## Known limitations

- **No quantised export.** PTH only, float32. No GGUF, no Q4_K_M, no int8 save path.
- **No accuracy evaluation.** No MMLU or any other benchmark score. The loss figures are
  training loss on raw corpus text over a short step budget.
- **No cross-project comparison.** No baseline was measured under identical conditions, so
  no speedup or efficiency ratio is stated.
- **No generation cache.** `generate` re-runs the full model per token.
- **The p-adic operators are not trainable.** Their descriptor is an exact integer mask
  and is therefore detached from the autograd graph.
- **Single reference machine.** An Intel i5-3337U with 2 physical cores, ~3 GB RAM, and AVX
  only. No AVX2, AVX-512, or AMX path has been measured.
- **MoE top-k is ambiguous in the specification.** Requirements referenced both `1` and
  `6`; the code and all shipped configs use `2`.

---

## License

MIT License plus the "No Big Tech Clause". See [`LICENSE`](LICENSE).
