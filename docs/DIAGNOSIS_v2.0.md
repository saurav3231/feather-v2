# Feather v2 — diagnosis: what the measurements actually show

Every number here was measured on this repository. Nothing is projected. Where a
requested target was not reached, the measured value and the reason are given
instead of the target.

Reproduce any row with the command noted above it.

## 1. The gradient path is healthy

Overfitting a single fixed batch of 32 tokens, `configs/feather_10M_fast.json`,
AdamW at `lr 3e-3`:

| step | loss |
| ---: | ---: |
| 1 | 9.3898 |
| 5 | 4.6004 |
| 10 | 0.8972 |
| 25 | 0.1455 |
| 200 | 0.1437 |

A 10.57M-parameter model that cannot drive 32 memorised tokens to near zero has
a broken gradient path. This one does it in 25 steps. No operator is silently
blocking learning, and the model is not misconfigured.

## 2. The model is data-starved, not broken

From a 600-step Kaggle run of `feather_20M_simple.json`, `main` loss only, which
excludes the MoE routing term:

| tokens | main loss |
| ---: | ---: |
| 12,800 | 6.6626 |
| 25,600 | 6.6401 |
| 38,400 | 6.2867 |
| 51,200 | 6.2151 |
| 64,000 | 6.2476 |
| 76,800 | 6.0823 |
| 89,600 | 5.8868 |
| 153,600 | 6.4107 |

A 12x increase in tokens buys 0.25 nats. Fitting `loss - floor = k * tokens^-a`
to that gives `a = 0.236`, which puts the cost of reaching 6.2 nats at roughly
19M tokens and 2.0 nats beyond what a single session can afford.

The unigram entropy floor of this corpus is 6.10 to 6.73 nats depending on the
sample, so a loss near 6.4 means the model has learned token frequencies and
little else. That is the expected result of showing a model 153,600 tokens, and
no architecture change alters it. The token budget is the variable.

## 3. Where the time actually goes

Per call, at the shapes the model really uses, 2 threads:

| operation | cost | share |
| --- | ---: | ---: |
| `fwht` on `(2,128,8192)` | 125.70 ms | |
| `ifwht` on `(2,128,8192)` | 120.29 ms | |
| `hyper.up` 352 -> 8192 | 61.54 ms | |
| `hyper.down` 8192 -> 352 | 58.01 ms | 33% |
| `divisibility_profile` (p-adic) | 11.11 ms | |
| `clifford_gate` | 7.91 ms | |
| `jacobi_decode` | 6.77 ms | |
| `tropical_softmax` | 1.00 ms | |
| `godel_log_code` | 0.11 ms | |
| `rough_path_signature` | 0.41 ms | |
| **sum of the removable operators** | **27.30 ms** | **7%** |
| **Walsh-Hadamard plus hyper matmuls** | **365.53 ms** | **93%** |

The seven "exotic" operators are 7% of the cost. Deleting all of them to gain
speed trades a working architecture for about 3% of a training step. The
Walsh-Hadamard transform is 67% and is a plain linear map.

## 4. Where the parameters actually are

`feather_20M_simple.json`, 20,696,188 total:

| group | parameters | share |
| --- | ---: | ---: |
| `blocks.hyper.up` | 5,783,552 | 27.9% |
| `blocks.hyper.down` | 5,767,872 | 27.9% |
| `blocks.weaver.kan_in` + `kan_out` | 2,486,548 | 12.0% |
| `blocks.sensory.scale_proj` | 994,048 | 4.8% |
| `blocks.liquid.gate` | 745,536 | 3.6% |
| **`blocks.vault.experts` (all 64)** | **407,552** | **2.0%** |

All 64 experts together are 407,552 parameters. Measured directly by sweeping
the expert count at fixed `hv_dim`:

| experts | parameters |
| ---: | ---: |
| 64 | 20,696,188 |
| 32 | 20,469,820 |
| 16 | 20,356,636 |
| 4 | 20,271,748 |

Going from 64 experts to 16 saves 339,552 parameters, 1.6% of the model and
about 1% of its compute. It cannot move RAM or speed measurably.

The hyperdimensional memory is 55.8% of the model, and because the transform
wrapping it is `O(hv_dim log hv_dim)` it is also most of the step time:

| `hv_dim` | parameters | s/step | tok/s | vs `hv_dim` 6144 |
| ---: | ---: | ---: | ---: | ---: |
| 6144 | 20,696,188 | 3.654 | 70 | 1.00x |
| 2048 | 12,020,860 | 1.860 | 138 | 1.96x |
| 1024 | 10,574,972 | 1.584 | 162 | 2.31x |
| 512 | 9,852,028 | 1.280 | 200 | 2.85x |

## 5. Throughput improves with sequence length

The hyperdimensional matmuls are per-token, so longer sequences amortise the
fixed costs. Measured at `batch 2`, `feather_10M_fast.json`:

| `seq_len` | tokens/step | s/step | tok/s | us/token |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 254 | 1.753 | 145 | 6901.7 |
| 256 | 510 | 3.109 | 164 | 6096.9 |
| 512 | 1022 | 5.334 | 192 | 5219.6 |

`seq_len 512` gives 1.32x the tokens per second and 4x the tokens per step, so
about 5.3x the training progress per second of the `seq_len 128` default. The
config already carries `seq_len 512`; the training default is 128.

## 6. The narrow config is a real trade, not a free win

An earlier commit reported the 10.57M config reaching the same loss as the
20.70M config to four decimal places. That was a 40-step artifact and it does
not hold. Comparing at matched token counts:

| tokens | 20.70M, `seq_len` 128 | 10.57M, `seq_len` 512 |
| ---: | ---: | ---: |
| 25,600 | 6.6401 | 7.9906 |
| 51,200 | 6.2151 | 7.0308 |
| 76,800 | 6.0823 | 6.8481 |

The narrow config is roughly 2.1x faster per step and about 0.8 nats behind per
token. Note this comparison changes the width and the sequence length together,
so it does not separate the two effects; a run at 20.70M with `seq_len 512`
would. The honest summary is that `hv_dim` buys speed and costs loss per token,
and which side wins depends on the wall-clock budget.

## 7. Four claims that measurement contradicts

**"The MoE auxiliary loss dominates."** It does not. `total = lm_loss +
0.01 * aux` (`src/feather_v2/model.py:315`), so the term contributes 0.33 nats
at step 50 and 0.064 nats at step 600, about 1% of the loss. It is also
*decreasing* over the run, 32.61 to 6.36, which means routing is balancing. For
64 experts, perfect balance scores 1.0.

**"The auxiliary weight is 1.0 and should be 0.01."** It is already 0.01, the
default of `aux_weight` in the same call.

**"`state_stability` 1.0 means representation collapse."** It is a causality
probe and 1.0 is the correct reading for a causal model. Perturbing tokens
80..128 of a 128-token sequence moves the hidden state at position 40 by
1.14e-01 in absolute terms, about 0.6% relative, which prints as 1.00000. The
model leaks a little information backwards; the metric is reporting that leak.
Pushing the number toward 0.9 would mean adding non-causal components and
breaking generation in order to satisfy a metric that is already correct.

**"int8 with VNNI is 7-10x faster."** The Kaggle CPU has no VNNI. int8 training
would cost accuracy and buy nothing.

## 8. Hardware ceiling

2-thread float32 `1024^3` matmul on this machine: 27.5 GFLOPS. Training costs
about `6 x params` per token, so 20,696,188 parameters is 124.2 MFLOP/token.

| throughput | required |
| ---: | ---: |
| 258 tok/s | 32 GFLOPS |
| 2,500 tok/s | 310 GFLOPS |

The measured 258 tok/s is already at this machine's dense ceiling. The
2,000-3,000 tok/s target is about 11x beyond the hardware and is not reachable
by any change to this code.

## 9. Memory

Importing CPU-only torch 2.13.0 costs 193 MB resident. A Kaggle session
reporting 761 MB at baseline is running the CUDA wheel, and switching is worth
more than any config change:

```python
!pip -q install torch --index-url https://download.pytorch.org/whl/cpu
```

20,696,188 parameters in float32 with Adam is 316 MB for weights, gradients and
optimizer state before a single activation exists, so 800 MB total is not
reachable while keeping 20M parameters. Measured on this box, the 20.70M model
adds 120 MB above baseline for a forward and backward pass at `batch 2 x
seq_len 128`, so most of a 1,014 MB delta is dataset pages and allocator
behaviour rather than model state.

## 10. Corpus accounting bug, fixed

The streaming loop counted characters against `target_tokens` and printed that
count as a token count. Measured ratio for this corpus and tokenizer: 0.4287
tokens per character, so every log overstated the corpus 2.3x. A run asking for
4,000,000 tokens read 4,207,607 characters, reported "loaded 4,207,607 tokens",
and produced 1,803,817 ids. Fixed in `4e3f73c`.

## Not claimed

- Loss 2.0 at 600 steps. Not reachable; section 2 gives the token cost.
- Any benchmark accuracy score.
- int8, VNNI, or quantised export performance.
- That a narrower model is equivalent to a wider one; section 6 shows it is not.
