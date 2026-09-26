# Feather v2 — Benchmark methodology and results

**Version:** 2.0.0
**Author:** Saurav Bhandari, Pokhara, Nepal

The current measured numbers are in [`RESULTS_v2.0.md`](RESULTS_v2.0.md), which is
generated from measurement artifacts. This document describes how those numbers are
produced and what they do and do not cover.

---

## 1. Why there is no comparison table

Earlier revisions of this file contained throughput, energy, and memory figures for other
projects — including BitNet and Phi-4 — alongside figures for Feather v2. None of those
numbers were measured. They were estimates presented in the same table and format as real
data, which made them indistinguishable from measurements to any reader.

They have been removed and are not replaced with new estimates. Producing a defensible
cross-project comparison would require running every baseline on the same machine under
the same conditions, which this repository does not do. Until that is done, this project
reports only its own measurements.

---

## 2. What the benchmark measures

`kaggle/test_all_sizes_mega.py` runs, for each config in the size ladder:

| Measurement | Method |
| --- | --- |
| Parameter count | Sum of `numel()` over `named_parameters()` after instantiation |
| Weight memory | `sum(p.numel() * p.element_size())` over parameters |
| Process RAM | Sampled RSS before model load and after, via `psutil` |
| Tokenisation rate | Wall-clock over the real tokenizer on real corpus text |
| Forward throughput | Median of repeated timed forward passes at seq len 32, 128, 512 |
| Bulk throughput | Timed forward passes at batch size 8 |
| Generation rate | Timed autoregressive `generate()` calls, batch size 1 |
| Training loss | Real backprop steps on real corpus text; first and last loss recorded |
| Energy | `codecarbon` over one forward pass plus a short generation |
| Per-component time | Timed forward of each component separately |
| Context similarity | Cosine similarity between initial and final hidden states |

Every check is recorded in `benchmark_report.json` alongside the value it asserted, so a
passing check can be audited against the measurement that justified it.

---

## 3. Verification checks

Each size must pass six checks:

| Check | Assertion |
| --- | --- |
| `weights_real` | Weight bytes > 0 and do not exceed total process RAM |
| `ram_real` | Measured RSS delta is consistent with the weight footprint |
| `timing_real` | Generation produced the requested token count in positive elapsed time |
| `loss_trend` | Training loss decreased over the step budget |
| `state_stability` | The same token prefix yields a similar final hidden state with and without a suffix |
| `init_ok` | The model constructed without error |

`assert_real_weights` compares measured weight bytes against measured RSS rather than
against an assumed constant. An earlier version of this gate iterated a list of
`(name, parameter)` tuples as if they were parameters, so the gate raised a `TypeError`
internally and was reported as a failure unrelated to the thing it was checking. That is
fixed and the gate now reports its own measurements.

---

## 4. Corpora

The benchmark attempts three public corpora and reports which ones were actually
available:

| Corpus | Identifier | Role |
| --- | --- | --- |
| OpenWebText | `Skylion007/openwebtext` | Tokenisation throughput |
| Wikipedia | `wikimedia/wikipedia`, config `20231101.en` | Training loss |
| no_robots | `HuggingFaceH4/no_robots` | Held-out generation text |

If a corpus cannot be loaded, the benchmark records that fact and marks the measurement
that depended on it as unavailable. It does not substitute a placeholder corpus or a
synthetic string.

Note that OpenWebText is published under the namespace `Skylion007`, not the bare
`openwebtext` name. The earlier identifier was wrong and has been corrected.

---

## 5. Reproducing

```bash
# Full ladder
python kaggle/test_all_sizes_mega.py --loss-steps 60

# Single size, fewer steps, for a quick check
python kaggle/test_all_sizes_mega.py --sizes 5M --loss-steps 20
```

Artifacts written:

- `benchmark_report.json` — every measurement and check, per size
- `docs/images/scaling.png` — parameter count against RAM
- `docs/images/memory_vs_params.png`
- `docs/images/loss_all_sizes.png`
- `docs/images/toks_vs_seqlen.png`
- `docs/images/energy_vs_size.png`

Then regenerate the results document:

```bash
python scripts/make_report.py
```

---

## 6. Reference machine

All committed measurements were taken on a single machine:

| Property | Value |
| --- | --- |
| CPU | Intel Core i5-3337U @ 1.80GHz |
| Cores | 2 physical / 4 logical |
| RAM | ~2.97 GB |
| SIMD | AVX. No AVX2, AVX-512, or AMX |
| Accelerator | None. CPU only |
| Platform | Windows |

This is a low-end 2013 mobile CPU. Throughput on other hardware has not been measured.
The exact feature flags and kernel binding chosen for this machine are recorded in the
`hardware` block of `benchmark_report.json`.

---

## 7. Limits of these results

- **One machine.** No cross-hardware comparison exists.
- **No accuracy evaluation.** No MMLU, no perplexity on a standard held-out benchmark, no
  task performance. The loss figures in `RESULTS_v2.0.md` are training loss on raw corpus
  text over a short step budget, which says nothing about model quality.
- **No baseline.** There is no reference model measured under identical conditions, so no
  speedup or efficiency ratio is reported.
- **Energy is not per-token.** The reported joules cover one forward pass plus a short
  generation as a single figure, not a normalised per-token cost, so it cannot be compared
  against any other energy figure.
- **RAM is dominated by the PyTorch runtime.** For the 5M config, weights are about
  19 MB while process RSS is several hundred MB. The gap is interpreter and library
  overhead, not the model.
- **State stability is not a context-length result.** It compares the final hidden state of
  a token prefix encoded alone against the same prefix encoded as the start of a longer
  sequence, at the same position. It measures how much a suffix perturbs an earlier
  representation. It is not evidence of long-context capability.

  An earlier version of this check compared the final *logits* of a short prefix against
  the final logits of a full sequence. That was wrong twice over: it compared a
  vocab-wide vector rather than a representation, and it compared two different positions.
  It now captures the hidden state via a forward hook and compares the same position.
