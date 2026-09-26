# Feather v2 — measured results

<!-- GENERATED FILE — do not edit by hand. -->
<!-- Source: `scripts/make_report.py` from `configs/size_report.json` and `benchmark_report.json`. -->

Every figure below is read from a measurement artifact. Anything that was not
measured is reported as _not measured_ rather than estimated or extrapolated.

## Measurement environment

- Timestamp: `2026-09-26T06:56:43Z`
- features.avx512: `False`
- features.amx: `False`
- features.avx2: `False`
- features.avx: `True`
- features.neon: `False`
- features.machine: `amd64`
- features.cores_physical: `2`
- features.cores_logical: `4`
- features.cpu: `Intel(R) Core(TM) i5-3337U CPU @ 1.80GHz`
- features.arch_string_raw: `AMD64`
- features.kaggle: `False`
- features.ram_gb: `2.9659500122070312`
- kernel.hypervector_dim: `1024`
- kernel.hv_memory_kb: `4`
- kernel.binding: `avx_wht`
- kernel.binds_per_instruction: `2`
- kernel.moe: `avx_tropical_tt`
- kernel.moe_tiles: `8x32`
- kernel.threads: `2`
- kernel.cache: `L1`
- kernel.adaptive_vocab: `256`
- kernel.adaptive_rank: `4`
- kernel.adaptive_binds: `2`

## Model sizes (measured)

| Config | Measured label | Parameters | Trainable | F32 | F16 | Largest group |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `feather_10M.json` | 9.60M | 9,604,412 | 9,604,412 | 36.64 MiB | 18.32 MiB | blocks |
| `feather_20M.json` | 19.52M | 19,520,508 | 19,520,508 | 74.46 MiB | 37.23 MiB | blocks |
| `feather_40M.json` | 40.34M | 40,337,084 | 40,337,084 | 153.87 MiB | 76.94 MiB | blocks |
| `feather_5M.json` | 5.06M | 5,055,020 | 5,055,020 | 19.28 MiB | 9.64 MiB | blocks |
| `feather_60M.json` | 58.37M | 58,368,714 | 58,368,714 | 222.66 MiB | 111.33 MiB | blocks |

All 5 configurations are F16 under 128 MiB (111.33 MiB largest).

### Parameter breakdown

| Config | blocks | embed | head | norm_f | pos_embed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `feather_10M.json` | 7,359,292 | 2,113,536 | 0 | 512 | 131,072 |
| `feather_20M.json` | 16,573,788 | 2,774,016 | 0 | 672 | 172,032 |
| `feather_40M.json` | 35,285,564 | 4,755,456 | 0 | 1,152 | 294,912 |
| `feather_5M.json` | 3,371,180 | 1,585,152 | 0 | 384 | 98,304 |
| `feather_60M.json` | 53,317,194 | 4,755,456 | 0 | 1,152 | 294,912 |

## Runtime benchmark (measured)

Benchmarked **1 of 5** ladder configurations: 5M.

> **Partial ladder.** 4 configuration(s) have measured sizes but no runtime benchmark yet. The rows above are real measurements; the missing sizes are absent, not estimated. Re-run `python kaggle/test_all_sizes_mega.py` to complete the ladder.

> **Stale report format.** This artifact still contains the pre-rename field(s) `context_dim, context_sim`. Those were replaced by `state_stability`, which compares the hidden state at the same sequence position with and without a suffix. Regenerate the report with the current benchmark script before citing it.

| Size | Parameters | RAM total | Fwd t/s @32 | @128 | @512 | Bulk t/s | Gen t/s | Loss | Energy (J) | Checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 5M | 5,055,020 | 464.9 | 229.0 | 535.5 | 691.6 | 687.1 | 9.06 | 9.0495 -> 7.1733 (Wikipedia (en)) | 776.1 | 6/6 |

### Verification checks

| Size | weights_real | ram_real | timing_real | loss_trend | context_recall | init_ok |
| --- | --- | --- | --- | --- | --- | --- |
| 5M | PASS | PASS | PASS | PASS | PASS | PASS |

Totals: **6/6** checks passed.

## Not claimed

The following have no measurement in this repository and are therefore not
asserted anywhere in the documentation:

- MMLU or any other benchmark accuracy score.
- Comparisons against GPU, BitNet, or any other project.
- Quantised (for example Q4_K_M) export or its size or quality.
- Long-context scaling beyond the measured sequence lengths.
- Operation-count or energy-savings ratios versus a baseline.

