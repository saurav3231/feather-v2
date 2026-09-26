# Feather v2 — Architecture

**Version:** 2.0.0
**Author:** Saurav Bhandari, Pokhara, Nepal
**License:** MIT + No Big Tech Clause (see `LICENSE`)

This document describes what the code actually does. Performance figures live in
[`RESULTS_v2.0.md`](RESULTS_v2.0.md), which is generated from measurement artifacts. No
speedup, energy-saving, or compression ratio is asserted here, because none of them has
been measured against a baseline.

---

## 1. Model structure

`FeatherV2Model` is a standard decoder-style language model with a non-standard block.

```
input_ids
  -> embed            nn.Embedding(vocab, dim)
  -> pos_embed        nn.Embedding(seq_len, dim)
  -> blocks           n_blocks x CognitiveBlock
  -> norm_f           nn.LayerNorm(dim)
  -> head             nn.Linear(dim, vocab, bias=False)
  -> logits
```

`CognitiveBlock` runs all seven components in sequence, each preceded by its own
`LayerNorm`:

| Order | Component | Class | Source docstring |
| --- | --- | --- | --- |
| 1 | Sensory encoder | `SensoryEncoder` | Multi-scale fractional encoder with p-adic scale selection. |
| 2 | Liquid memory | `LiquidMemory` | Hierarchical chunked gated recurrence. |
| 3 | Holographic memory | `HyperDimensionalMemory` | Holographic mixing through a Walsh-Hadamard basis. |
| 4 | Knowledge vault | `KnowledgeVault` | Sparse mixture of TT-compressed experts with Sinkhorn-balanced routing. |
| 5 | Cognitive weaver | `CognitiveWeaver` | Kolmogorov-Arnold feed-forward with a differentiable Godel loop. |
| 6 | Homeostasis governor | `HomeostasisGovernor` | Predictive entropy gate that rescales the residual stream. |
| 7 | Generative evolution | `GenerativeEvolution` | Jacobi-spectral refinement with a learned mutation step. |

The same seven components also appear once at the top level of the model, before the
`blocks` stack.

The per-component `LayerNorm` is load-bearing. Before it was added, the residual stream
grew without bound across the Godel loop and overflowed in float32 after a few dozen
training steps. There is a regression test for this.

### Tied embeddings

With `tie_embeddings: true` (the default), `head` shares its weight with `embed`, so
`head` contributes zero parameters. This is why the parameter breakdown reports
`head: 0`.

---

## 2. Operator library

`src/feather_v2/nn_math.py` exports 23 names. The operators that carry the architecture:

| Operator | Purpose | Exact or relaxed |
| --- | --- | --- |
| `fwht` / `ifwht` | Fast Walsh-Hadamard transform and inverse | Exact |
| `pad_to_pow2` / `next_pow2` | Shape helpers for the transform | Exact |
| `divisibility_profile` | p-adic divisibility descriptor | Exact integer, **detached** |
| `p_adic_weights` | p-adic weighted combination | Exact integer, **detached** |
| `p_adic_distance` | Ultrametric-style distance on p-adic digits | Exact integer, **detached** |
| `tropical_matmul` / `tropical_softmax` | Min-plus matmul | Softmin relaxation |
| `softmin` | Temperature-controlled min | Relaxation |
| `fractional_weights` | Hierarchical short/long-range weighting | Smooth relaxation |
| `tt_factors` / `tt_compress` / `tt_matmul` | Tensor-train construction and matmul | Low-rank approximation |
| `sinkhorn` | Entropic optimal transport projection | Iterative relaxation |
| `sheaf_project` | Sheaf consistency projection | Projection, differentiable |
| `clifford_gate` | Clifford-algebra style gate | Low-rank approximation |
| `equilibrium_update` | Fixed-point relaxation | Iterative |
| `jacobi_decode` | Jacobi-style iterative decode | Iterative |
| `rough_path_signature` | Rough-path style signature features | Relaxation |
| `godel_encode` / `godel_log_code` | Godel-style log coding | `log`-based, finite on large inputs |
| `alpha_dropout` | Dropout variant | Stochastic |
| `KANLinear` | Kolmogorov-Arnold spline layer | Learnable, differentiable |

### Honest limitations

Three of these deserve explicit statement, because the names suggest more than is
delivered:

1. **The p-adic operators are not trainable.** `divisibility_profile` produces an integer
   0/1 mask, so it is detached from the autograd graph. It influences routing decisions
   but receives no gradient. This is inherent: a discrete divisibility test has no useful
   derivative.

2. **"Tropical" is a softmin, not a true min-plus product.** `tropical_matmul` is
   differentiable by construction, which means it approximates rather than computes the
   exact min-plus result. It is a relaxation, and its output is not a tropical semiring
   element.

3. **"Godel coding" is log-based.** The first implementation used an absolute-value form
   that produced zero gradients and non-finite values for large messages. The current
   `godel_log_code` formulation is finite and differentiable, but it is a monotone
   encoding, not a true Godel numbering.

The TT, Clifford, rough-path, Sinkhorn, sheaf, equilibrium, and Jacobi operators are
smooth numerical relaxations. They are useful as inductive biases, but none of them
implements the exact discrete algorithm it is named after.

---

## 3. Mixture of experts

`KnowledgeVault` routes each token to `moe_top_k` of `moe_experts` `TTExpert` modules.
Routing is computed from the hidden state and combined with the top-k auxiliary
load-balancing penalty in `FeatherV2Model.loss`.

`moe_top_k` is `2`, and it is now set explicitly in all five ladder configs
(`configs/feather_*.json`) rather than being inherited from `DEFAULT_CONFIG`.

**Decision record.** Earlier revisions of this project referred to both `1` and `6` as
the intended value. The requirement was ambiguous, so `2` was kept and written down
explicitly rather than left to a default. The rationale is a judgement, not a measured
result: routing width trades capacity against per-token work, so `1` is the sparse end
and `6` the dense end. **No ablation over `moe_top_k` was run, so this document does
not claim `2` is optimal.** It is the value the code and the measured ladder use.

Routing width is genuinely config-driven: `FeatherV2Model` passes
`config["moe_top_k"]` into `KnowledgeVault`, which stores it as `self.top_k`. There is
no hardcoded `1` or `6` in the routing path. If a different width is wanted, change it
in the configs and re-run `scripts/measure_sizes.py` and the benchmark, because expert
count and routing width both change parameter count. Adaptive per-token routing width
is unimplemented and untested.

---

## 4. Tokenisation

`hybrid_adaptive_tokenizer` produces a fixed vocabulary of `8256` tokens: `256` adaptive
slots plus a byte-pair-merge vocabulary. Sequence length defaults to `512` and is bounded
by the learned `pos_embed` table.

---

## 5. Hardware adaptation

`feather_v2.hardware` detects CPU features and selects a Walsh-Hadamard binding, falling
back without raising:

```
AVX-512 -> AVX2 -> AVX -> NEON -> scalar
```

Run `python -m feather_v2.hardware` to see the detected features and the selected
binding on the current machine.

The binding selection is real and tested. The per-CPU throughput numbers that previously
appeared in this document were not measured and have been removed; see
[`RESULTS_v2.0.md`](RESULTS_v2.0.md) for the single machine that was actually measured.

---

## 6. Configuration

Defaults live in `DEFAULT_CONFIG`; the size ladder lives in `configs/`. Loading is strict
by default: an unknown key raises instead of being silently ignored, and renamed keys are
migrated with a warning.

| Key | Default | Meaning |
| --- | ---: | --- |
| `vocab` | 8256 | Token vocabulary size |
| `dim` | 512 | Model width |
| `n_blocks` | 2 | Number of `CognitiveBlock` stacks |
| `seq_len` | 512 | Maximum sequence length |
| `hv_dim` | 8192 | Hypervector dimensionality |
| `n_scales` | 4 | Sensory multi-scale count |
| `chunk` | 32 | Liquid memory chunk size |
| `p_adic_p` / `p_adic_levels` | 2 / 8 | p-adic descriptor base and levels |
| `tt_rank` | 6 | Tensor-train rank |
| `moe_experts` / `moe_top_k` | 96 / 2 | Expert count and routing width |
| `weaver_loops` / `weaver_gaussians` | 2 / 5 | Godel loop iterations and Gaussian basis |
| `evo_iters` / `evo_latent` | 6 / 16 | Refinement iterations and latent size |
| `tau` | 0.1 | Softmin temperature |
| `dropout` / `dropout_q` | 0.25 / 0.5 | Dropout rates |
| `alpha_fractional` | 0.7 | Fractional weighting exponent |
| `sig_dim` | 8 | Signature feature width |
| `tie_embeddings` | true | Share `head` weight with `embed` |

The ladder configs vary `dim`, `hv_dim`, `moe_experts`, `n_blocks`, and `tt_rank`. Each
config's filename reflects its **measured** parameter count, not a target — see
`configs/size_report.json`.

---

## 7. What is verified

- The test suite covers the operator library, component gradients, the detached p-adic
  descriptor, Godel-coder numerical stability, the per-component LayerNorm, real model
  training, checkpoint round trips, and config validation.
- `tests/test_gguf.py` fails if a `.gguf` file exists in the tree without a real exporter
  being registered, which prevents mislabelled artifacts from reappearing.
- Parameter counts, byte sizes, and CPU runtime figures in `RESULTS_v2.0.md` are read
  directly from JSON artifacts produced by `scripts/measure_sizes.py` and
  `kaggle/test_all_sizes_mega.py`.

Not verified, and therefore not claimed anywhere: benchmark accuracy (MMLU or
otherwise), comparison against any GPU or other project, quantised export, and
long-context behaviour beyond the measured sequence lengths.
