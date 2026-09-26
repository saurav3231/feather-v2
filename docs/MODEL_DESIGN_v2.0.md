# Feather v2 — Model design

**Version:** 2.0.0
**Author:** Saurav Bhandari, Pokhara, Nepal

Companion to [`ARCHITECTURE_FINAL_v2.0.md`](ARCHITECTURE_FINAL_v2.0.md), which
describes the component structure. This document describes the Python API as it actually
exists.

---

## 1. Public API

`FeatherV2Model` is a `torch.nn.Module`. Its interface:

```python
forward(input_ids: Tensor, return_aux: bool = False) -> Tensor | tuple[Tensor, Tensor]
loss(input_ids: Tensor, labels: Tensor | None = None,
     aux_weight: float = 0.01) -> tuple[Tensor, dict[str, float]]
generate(input_ids: Tensor, max_new_tokens: int = 16,
         temperature: float = 1.0, greedy: bool = True) -> Tensor
save_pth(path: str | Path) -> int
describe() -> dict[str, Any]
```

| Method | Returns | Notes |
| --- | --- | --- |
| `forward` | Logits `[batch, seq, vocab]`, or `(logits, aux)` | `aux` carries the MoE routing term |
| `loss` | `(total_loss, metrics)` | Next-token cross-entropy plus the load-balancing penalty |
| `generate` | Token ids `[batch, seq + max_new_tokens]` | Autoregressive; greedy by default |
| `save_pth` | Bytes written | `torch.save` of weights plus config |
| `describe` | Dict of measured facts | No estimates or targets |

If `labels` is omitted in `loss`, the model shifts `input_ids` internally to build
next-token targets. `aux_weight` scales the MoE balancing term; it defaults to `0.01`.

There is no `encode`, `route_and_apply`, `reasoning_loop`, `entropy_gate`, or
`speculative_generate` method. Earlier revisions of this document listed those names; they
do not exist in the codebase and have been removed.

---

## 2. Shapes

| Quantity | Shape |
| --- | --- |
| `input_ids` | `[batch, seq]` of int64 token ids, `seq <= seq_len` |
| Logits | `[batch, seq, vocab]` |
| Hidden state | `[batch, seq, dim]` |
| Generated ids | `[batch, seq + max_new_tokens]` |

With the default configuration `vocab=8256`, `dim=512`, `seq_len=512`, so logits for a
full-length sequence are `[batch, 512, 8256]`.

---

## 3. Construction

```python
from feather_v2 import FeatherV2Model, load_config

model = FeatherV2Model()                              # DEFAULT_CONFIG
model = FeatherV2Model(load_config("configs/feather_5M.json"))
```

`load_config` is strict by default: an unrecognised key raises `ValueError` rather than
being ignored. Renamed keys are migrated through `CONFIG_ALIASES` with a `UserWarning`.
Pass `strict=False` to accept unknown keys deliberately.

Attributes: `model.config` (the resolved dict) and `model.device`.

To read a config's real size instead of trusting its filename:

```python
!python -m feather_v2.model --config configs/feather_20M_simple.json --count-params
# 20,696,188 params (20.70M)
# dim=352 n_blocks=2
```

### Choosing a width that hits a size target

Parameter count is dominated by `blocks[i].hyper`, which is about 27% of the model
per block at `dim=384`, and by `embed`, which scales with `vocab * dim`. `dim` is
the effective lever. Measured counts for the quick-baseline architecture
(`hv_dim=6144`, `vocab=8256`, `moe_experts=64`, `moe_top_k=2`):

| `dim` | `n_blocks=1` | `n_blocks=2` | `n_blocks=8` |
| --- | --- | --- | --- |
| 352 | 11,891,614 (11.89M) | **20,696,188 (20.70M)** | 73,523,632 (73.52M) |
| 384 | 13,241,406 (13.24M) | 23,115,132 (23.12M) | 82,357,488 (82.36M) |

`configs/feather_20M_simple.json` therefore uses `dim=352`, which measures
20,696,188 parameters against a 20M target. The originally specified `dim=384`
measures 23,115,132 at `n_blocks=2`; that was an honest measurement, not a defect,
and `dim=352` was chosen so the shipped config lands near its stated size.
`n_blocks` alone cannot reach 20M from `dim=384`, because the values bracketing it
are 13.24M at `n_blocks=1` and 23.12M at `n_blocks=2`.

`hv_dim` does not affect the parameter count in this configuration: varying it
between 4864 and 6144 leaves the total at 23,115,132. It is kept explicit because
it still documents the intended hypernetwork width.

The module hierarchy is two levels, which matters when reaching for a component:

- `FeatherV2Model` children: `embed`, `pos_embed`, `blocks`, `norm_f`, `head`.
- Each `model.blocks[i]` is a `CognitiveBlock` whose children are the seven trainable
  components: `sensory`, `liquid`, `hyper`, `vault`, `weaver`, `governor`, `evolution`,
  each preceded by its own `norm_*` `LayerNorm`.

So the MoE is `model.blocks[i].vault`, **not** `model.vault`; a bare `model.vault`
raises `AttributeError`. Each component also exposes its own parameters, for example
`model.blocks[i].vault.experts`.

### MoE routing width

`moe_top_k` is the number of `moe_experts` `TTExpert` modules each token is routed to.
It is `2` in `DEFAULT_CONFIG` and is set explicitly in all five ladder configs, so the
measured ladder does not depend on a default that could change.

Earlier revisions referred to both `1` and `6`, so the requirement was ambiguous and the
value was fixed at `2` and written down. That is a judgement about the capacity-versus-
compute trade-off, **not a measured optimum**: no `moe_top_k` ablation was run, and these
docs do not claim `2` is best. Routing reads the config end to end
(`config["moe_top_k"]` -> `KnowledgeVault(top_k=...)` -> `self.top_k` -> `torch.topk`),
with no hardcoded width in the routing path. Changing it changes parameter count, so
re-run `scripts/measure_sizes.py` and the benchmark afterwards.

---

## 4. Training

```python
import torch
from feather_v2 import FeatherV2Model

model = FeatherV2Model()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

ids = torch.randint(0, model.config["vocab"], (2, 128))
opt.zero_grad()
loss, metrics = model.loss(ids)
loss.backward()
opt.step()
```

`model.parameters()` yields trainable tensors only. `describe()` reports
`parameters` and `parameters_trainable`; in the shipped configs these are equal, because
nothing is frozen.

`scripts/train.py` wraps this loop, writes a real PTH checkpoint, and saves a manifest
beside it:

```bash
python scripts/train.py --config configs/feather_5M.json --steps 40 --pth runs/feather.pth
```

---

## 5. Checkpoints

`save_pth` writes a PyTorch checkpoint with keys `format`, `config`, and `state_dict`.
`format` is `feather-v2-pth-1`. Reload with the classmethod:

```python
rebuilt = FeatherV2Model.from_pth("runs/feather.pth")
```

To read the raw payload instead:

```python
ckpt = torch.load("runs/feather.pth", map_location="cpu", weights_only=False)
rebuilt = FeatherV2Model(ckpt["config"])
rebuilt.load_state_dict(ckpt["state_dict"])
```

Checkpoints are float32. A 5M-parameter model produces a checkpoint of roughly 19.5 MiB.
There is no float16 or quantised save path, and no GGUF exporter.

A reloaded model reproduces logits bit-for-bit **only in evaluation mode**. `nn.Module`
defaults to `training=True`, and the model applies dropout (`dropout=0.25`,
`dropout_q=0.5`) during a training forward pass, so two forward passes on the same inputs
will differ. Call `.eval()` before comparing logits.

---

## 6. Generation

```python
out = model.generate(prompt_ids, max_new_tokens=32, greedy=True)
```

`greedy=True` takes the argmax at each step. With `greedy=False`, `temperature` scales
the logits before sampling. `generate` runs the full model per step; there is no KV cache,
which is a straightforward but unoptimised implementation.

---

## 7. Hardware adaptation

`feather_v2.hardware` provides `detect_cpu_features()` and `get_best_kernel()`, which
select a Walsh-Hadamard binding from AVX-512, AVX2, AVX, NEON, or scalar. Run
`python -m feather_v2.hardware` to print the detection for the current machine.

The binding is reported in `benchmark_report.json` and in the benchmark's `hardware`
block. The model itself computes in PyTorch; the binding selection does not change the
numerics of `forward`.

---

## 8. Verification

`python -m pytest tests/ -q` covers:

- Operator behaviour and gradient flow, including that the detached p-adic descriptor
  produces no gradient
- Per-component gradients in all seven components
- Numerical stability of the Godel coder on large messages
- Stability of the per-component `LayerNorm` across repeated training steps
- Real model training, loss decrease, and non-zero gradients
- Checkpoint round trips and byte-size reporting
- Config validation, alias migration, and size-label agreement
- A guard against mislabelled GGUF artifacts

Measured parameter counts, byte sizes, and CPU runtime figures are in
[`RESULTS_v2.0.md`](RESULTS_v2.0.md). No accuracy benchmark, cross-project comparison, or
energy-per-token figure is claimed.
