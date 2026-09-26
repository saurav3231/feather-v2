# TODO - which three documents are still missing?

This is an open question, not a work item that can be closed by guessing. It is
recorded here so the gap is visible instead of being filled with invented scope.

## Status

Three documents were requested that are not in this repository. **Their names and
required contents are not recoverable from the files currently present**, so they have
not been created. Writing plausible-looking documents under guessed names would produce
exactly the kind of unverifiable content this repository is being cleaned up to remove.

## What exists today

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview, install, quick start, verified status |
| `RELEASE_v2.0.0.md` | Release notes and verification commands |
| `docs/ARCHITECTURE_FINAL_v2.0.md` | Operator-by-operator design, exact vs relaxed |
| `docs/BENCHMARK_v2.0.md` | Benchmark methodology and check definitions |
| `docs/MODEL_DESIGN_v2.0.md` | Python API reference, verified against the code |
| `docs/RESULTS_v2.0.md` | Generated from measured JSON, no hand-typed numbers |
| `kaggle/README_MEGA_TEST.md` | How to run the full size ladder |
| `paper/README_PAPER.md` | How to build the paper |

## Where the spec was looked for

The following were searched across `A:\Project` and **none exist**:

- `FEATHER-V2_CODER_PROMPT_COMPACT_FOR_OPENCODE.md`
- any file matching `*PROMPT*`, `*MEGA*`, `*SPEC*`, `*PLAN*`
- `scripts/verify_v2_architecture.py`

The only match in the repository is `kaggle/README_MEGA_TEST.md`, which is a how-to for
the benchmark, not the requirements document.

## What is needed to close this

Please supply either:

1. The names and required contents of the three documents, or
2. The original requirements text, or
3. Confirmation that the existing eight documents above already cover the requirement.

Once known, the documents can be written against the same rule applied everywhere else
in this repository: every number measured, every API signature checked against the code,
and no claim that cannot be reproduced from a script in `scripts/` or `kaggle/`.

## Related open questions

- `moe_top_k` is now fixed at `2` and set explicitly in all five ladder configs. The
  original requirement mentioned both `1` and `6`. This was a judgement call, recorded in
  `docs/ARCHITECTURE_FINAL_v2.0.md`, and **no ablation was run**, so it is not claimed to
  be optimal. If a specific value was intended, say which and the ladder will be re-measured.
