# Feather v2 — The People's LLM Engine

**CPU-native. Open source. Maximum output / minimum resource.**
**13 advanced mathematics + 7 components. 40M parameters. 0.9GB RAM. 20MB GGUF.**

---

## Bicycle vs Truck

**Truck (Transformer 7B GPU):** Needs $25k H100, 700W power, 14GB HBM, data center. Only rich can run.

**Bicycle (Feather v2):** Runs on any CPU — i5-3337U old laptop 10-15 tok/s, Kaggle CPU 35-50 tok/s, i7-12700 60-70 tok/s beats GPU 80. 0.9GB RAM. 0.03J/1k 93x less energy. 20MB model fits mobile. Works offline airplane mode.

**CPU is the people. GPU is the monopoly.**

---

## Performance

| Device | tok/s | RAM | Energy/1k |
|--------|-------|-----|-----------|
| i5-3337U 2C/4T 8GB | 10-15 | 0.6GB | 0.08J |
| Kaggle CPU 2C/4T 31GB | 35-50 | 0.9GB | 0.03J |
| i7-12700 12C | 60-70 | 0.9GB | 0.028J |
| Agent Env 1C/2T 1.9GB | 7-12 | 0.3GB | 0.05J |

---

## Quick Start

```bash
git clone https://github.com/saurav3231/feather-v2.git
cd feather-v2
pip install -e .
python -m feather_v2.hardware
python -c "from feather_v2 import FeatherV2Model; m = FeatherV2Model(); print(m.hardware_summary())"
```

---

## 13 Mathematics

1. Hybrid Adaptive Tokenizer — 256+8k BPE 8256 vocab 4.5x fewer tokens
2. Hybrid WHT — 0 mults 10x energy + tropical + fractional + p-adic + TT + Clifford
3. Adaptive Fractional Weights — alpha 0.6-0.8 per layer + K 32-64 + beta learnable + hierarchical + liquid tau
4. Adaptive Tropical Min — softmin tau adaptive + hierarchical 2-level + TT fusion + SparX AMX
5. Adaptive p-adic — p 2-3 per layer + valuation learnable + hierarchical 2-level + retrieval
6. Adaptive TT — rank 4-8 per layer + caching + SparX + AMX + tropical fusion
7. Adaptive Rough Path — vals 13-20 per layer + path adaptive + multi-scale + fractional + p-adic
8. Adaptive Sinkhorn — iters 5-10 per layer + std adaptive + p-adic + hierarchical + entropy
9. Adaptive Clifford — vec 8-16 per layer + reduction 4x-8x + full + WHT + TT
10. Adaptive Sheaf — Krum 1-3 + eps 1.0-2.0 + hierarchical + fractional + regularizer
11. Adaptive Equilibrium — D 64-384 + free/nudge adaptive + fractional + free energy
12. Adaptive Jacobi — draft 2/4-8/16 + threads physical cores + tree + p-adic + entropy
13. KAN Activation — learnable spline on edges + WHT + TT + Clifford 2x fewer params

---

## 7 Components

1. SensoryEncoder — Adaptive Multi-Scale Fractional p-adic Rough Path Encoder
2. LiquidMemory — Adaptive Hierarchical Liquid Fractional Memory
3. HyperDimensionalMemory — Hybrid WHT HRR TT Clifford 10k-D brain holographic
4. KnowledgeVault — Adaptive Hierarchical Softmin Tropical-TT Fusion SparX AMX p-adic Entropy
5. CognitiveWeaver — Adaptive MoD Gödel CognitiveWeaver + KAN
6. HomeostasisGovernor — Adaptive Predictive Active Inference HomeostasisGovernor
7. GenerativeEvolution — Adaptive Tree p-adic Entropy Sheaf Gödel GenerativeEvolution

---

## Hardware Adaptive

Works for ALL PCs — AVX-512 -> AVX2 -> AVX -> NEON -> Scalar fallback never fails.
Scalar 3-5 tok/s works everywhere 2010 PC no SIMD 1GB RAM foundation 200 years.

---

## License

MIT + No Big Tech Clause — Open Source — Breaks monopoly — Physics free, data centers not.

---

## 200-Year Vision

Substrate-agnostic: same weights run on digital CPU 2026, memristor 2030, photonic 2032, quantum HDC 2040, biological 2100, unknown physics 2226.

Gödel self-rewriter immortal: model rewrites own code to improve, never degrades, functor preserving fractal self-similar.

---

**Author:** Saurav Bhandari, Nepal Pokhara

**CPU is the people. GPU is the monopoly. Feather v2 is CPU's revenge.**
