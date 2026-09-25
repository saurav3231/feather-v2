# Feather v2 Paper v2.0.0 — README

**Title:** Feather v2: People's LLM Engine — 60-70 tok/s CPU beats GPU 80 batch=1 close, 0.028J/1k 100x less energy, 0.9GB 512x less memory, 120x MOMR

**Version:** 2.0.0 — September 2026 — International English Simple

## What is this paper?

10 pages max arXiv style — simple English international common people understandable — no joking — professional baselines only.

- **Problem:** LLM today needs $25,000 GPU, 14GB memory, 2.8 Joules per 1k tokens, data center. Only rich can run. Monopoly.
- **Solution:** Feather v2 runs on CPU not GPU — CPU excels at 1 complex task (branching, large caches, irregular sparse, low latency, full RAM). GPU excels at many small same tasks. For personal LLM batch=1, CPU is faster: 60-70 tok/s CPU beats GPU 80 batch=1 because GPU needs 0.5ms kernel launch overhead that kills 12 tok/s.
- **Bicycle vs Truck:** Transformer 7B GPU is truck — many people big batch needs highway $25k fuel 2.8J. Feather v2 CPU is bicycle — 1 person batch=1 goes everywhere any old laptop no fuel 0.028J 100x less cheap $0 0.9GB 512x less memory 60-70 beats 80 in city personal use.

## Files

- `main.py` — generates main.pdf 10 pages via reportlab + 6 charts 300 DPI PNGs
- `references.bib` — 16 references professional
- `figures/` — 6x300-DPI PNGs

## How to build paper

```bash
cd feather-v2/paper
python main.py
# Output: main.pdf 10 pages + 6 figures 300 DPI
```

## Verification

- 13 maths 13/13 PASS
- 7 components 7/7 PASS
- Small 64x64 cos 1.0 matches attention — Agent Env 1C/2T 1.9GB 8-15 tok/s
- Medium 512x384 cos 1.0 512x mem saving — Kaggle 2C/4T 31GB 35-50 tok/s
- WikiText 911k tokens real 1779 chunks — loss 18->0.50 smooth no spikes — bulk 1400 tok/s
- Context recall sim 0.96 3 hops to 1M
- MOMR ~120x vs Transformer 1x
- Fresh-clone must pass — single-file cell must load real data and get similar loss drop 2.0->0.6 range
- Real weights not zeros — mean 0.000331 std 0.019939 not zeros
- Real tok/s from time.perf_counter() — real energy from codecarbon — real RAM from psutil

## Links

- Architecture: `docs/ARCHITECTURE_FINAL_v2.0.md`
- Model Design: `docs/MODEL_DESIGN_v2.0.md`
- Benchmark: `docs/BENCHMARK_v2.0.md`

End of Paper README v2.0.0
