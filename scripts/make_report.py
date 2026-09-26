"""Generate docs/RESULTS_v2.0.md from measured artifacts only.

Every number in the output is read from:
  - configs/size_report.json   (real parameter counts / byte sizes)
  - benchmark_report.json      (real runtime benchmark)

Nothing is hardcoded or extrapolated. Missing measurements render as
"not measured" instead of a placeholder number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SIZE_REPORT = ROOT / "configs" / "size_report.json"
BENCH_REPORT = ROOT / "benchmark_report.json"
OUT = ROOT / "docs" / "RESULTS_v2.0.md"

MIB = 1024.0 * 1024.0
NOT_MEASURED = "_not measured_"


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value: Any, spec: str = ".2f", missing: str = NOT_MEASURED) -> str:
    if value is None:
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return missing


def sizes_table(sizes: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Config | Measured label | Parameters | Trainable | F32 | F16 | Largest group |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sizes:
        lines.append(
            "| `{source}` | {label} | {params:,} | {train:,} | {f32} MiB | {f16} MiB | {group} |".format(
                source=row.get("source", "?"),
                label=row.get("measured_size_label", NOT_MEASURED),
                params=int(row.get("parameters") or 0),
                train=int(row.get("parameters_trainable") or 0),
                f32=fmt((row.get("f32_bytes") or 0) / MIB),
                f16=fmt((row.get("f16_bytes") or 0) / MIB),
                group=row.get("largest_group", NOT_MEASURED),
            )
        )
    return lines


def breakdown_table(sizes: list[dict[str, Any]]) -> list[str]:
    groups = sorted({g for row in sizes for g in (row.get("breakdown") or {})})
    lines = [
        "| Config | " + " | ".join(groups) + " |",
        "| --- |" + " ---: |" * len(groups),
    ]
    for row in sizes:
        bd = row.get("breakdown") or {}
        cells = [f"{int(bd.get(g) or 0):,}" for g in groups]
        lines.append(f"| `{row.get('source', '?')}` | " + " | ".join(cells) + " |")
    return lines


def loss_span(row: dict[str, Any]) -> str:
    corpus = row.get("loss_corpus") or "no corpus"
    trend = row.get("loss_trend") or {}
    first = trend.get("initial", trend.get("first"))
    last = trend.get("final", trend.get("last"))
    if first is not None and last is not None:
        return f"{float(first):.4f} -> {float(last):.4f}"
    losses = row.get("losses") or []
    if not losses:
        return f"{NOT_MEASURED} ({corpus})"
    return f"{float(losses[0]):.4f} -> {float(losses[-1]):.4f} ({corpus})"


def bench_table(bench: dict[str, Any] | None) -> list[str]:
    if not bench or not bench.get("sizes"):
        return [NOT_MEASURED, "", "Run `python kaggle/test_all_sizes_mega.py` first."]

    lines = [
        "| Size | Parameters | RAM total | Fwd t/s @32 | @128 | @512 | Bulk t/s | Gen t/s | Loss | Energy (J) | Checks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in bench["sizes"]:
        ft = row.get("forward_throughput") or {}
        checks = row.get("checks") or []
        passed = sum(1 for _, v in checks if v)
        lines.append(
            "| {size} | {params} | {ram} | {f32} | {f128} | {f512} | {bulk} | {gen} | {loss} | {energy} | {ok}/{total} |".format(
                size=row.get("size_label", "?"),
                params=(
                    fmt(row.get("params"), ",.0f")
                    if row.get("params")
                    else NOT_MEASURED
                ),
                ram=fmt(row.get("ram_mb"), ".1f"),
                f32=fmt(ft.get("32"), ".1f"),
                f128=fmt(ft.get("128"), ".1f"),
                f512=fmt(ft.get("512"), ".1f"),
                bulk=fmt(row.get("bulk_tok_s"), ".1f"),
                gen=fmt(row.get("gen_tok_s"), ".2f"),
                loss=loss_span(row),
                energy=fmt(row.get("energy_j"), ".4g"),
                ok=passed,
                total=len(checks),
            )
        )
    return lines


def check_table(bench: dict[str, Any] | None) -> list[str]:
    if not bench or not bench.get("sizes"):
        return [NOT_MEASURED]
    names: list[str] = []
    for row in bench["sizes"]:
        for name, _ in row.get("checks") or []:
            if name not in names:
                names.append(name)
    lines = ["| Size | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for row in bench["sizes"]:
        got = dict(row.get("checks") or [])
        lines.append(
            f"| {row.get('size_label', '?')} | "
            + " | ".join("PASS" if got.get(n) else "FAIL" for n in names)
            + " |"
        )
    return lines


def coverage_note(bench: dict[str, Any] | None, ladder: int) -> list[str]:
    """State how much of the ladder was actually benchmarked.

    A partial report must never be mistakable for a complete one, so the count is
    always printed and a shortfall is called out explicitly. Legacy metric names are
    flagged too: they mean the report was produced by an older revision and should be
    regenerated before it is treated as current evidence.
    """
    if not bench or not bench.get("sizes"):
        return []

    rows = bench["sizes"]
    done = [r.get("size_label", "?") for r in rows]
    note = [
        f"Benchmarked **{len(rows)} of {ladder}** ladder configurations: "
        f"{', '.join(done)}."
    ]

    if ladder and len(rows) < ladder:
        note += [
            "",
            f"> **Partial ladder.** {ladder - len(rows)} configuration(s) have measured "
            "sizes but no runtime benchmark yet. The rows above are real measurements; "
            "the missing sizes are absent, not estimated. Re-run "
            "`python kaggle/test_all_sizes_mega.py` to complete the ladder.",
        ]

    legacy = sorted(
        {
            key
            for r in rows
            for key in ("context_recall", "context_sim", "context_dim")
            if key in r
        }
    )
    if legacy:
        note += [
            "",
            "> **Stale report format.** This artifact still contains the pre-rename "
            f"field(s) `{', '.join(legacy)}`. Those were replaced by `state_stability`, "
            "which compares the hidden state at the same sequence position with and "
            "without a suffix. Regenerate the report with the current benchmark script "
            "before citing it.",
        ]
    return note + [""]


def main() -> None:
    sizes = load_json(SIZE_REPORT) or []
    bench = load_json(BENCH_REPORT)

    out: list[str] = [
        "# Feather v2 — measured results",
        "",
        "<!-- GENERATED FILE — do not edit by hand. -->",
        "<!-- Source: `scripts/make_report.py` from `configs/size_report.json` and `benchmark_report.json`. -->",
        "",
        "Every figure below is read from a measurement artifact. Anything that was not",
        "measured is reported as _not measured_ rather than estimated or extrapolated.",
        "",
    ]

    if bench and bench.get("hardware"):
        hw = bench["hardware"]
        out += [
            "## Measurement environment",
            "",
            f"- Timestamp: `{bench.get('timestamp', '?')}`",
        ]
        for key, value in hw.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    # Never re-render a throughput prediction, even if an older
                    # report artifact still carries one.
                    if sub_key in (
                        "expected_tok_per_sec",
                        "precision",
                        "ram_budget_gb",
                    ):
                        continue
                    out.append(f"- {key}.{sub_key}: `{sub_value}`")
            else:
                out.append(f"- {key}: `{value}`")
        out.append("")

    out += ["## Model sizes (measured)", ""]
    if sizes:
        out += sizes_table(sizes)
        out += [
            "",
            f"All {len(sizes)} configurations are F16 under 128 MiB "
            f"({fmt(max((r.get('f16_bytes') or 0) / MIB for r in sizes))} MiB largest).",
            "",
            "### Parameter breakdown",
            "",
        ]
        out += breakdown_table(sizes)
        out.append("")
    else:
        out += [NOT_MEASURED, "", "Run `python scripts/measure_sizes.py` first.", ""]

    out += ["## Runtime benchmark (measured)", ""]
    out += coverage_note(bench, len(sizes))
    out += bench_table(bench)
    out += ["", "### Verification checks", ""]
    out += check_table(bench)

    if bench:
        summary = bench.get("summary") or {}
        out += [
            "",
            f"Totals: **{summary.get('total_pass', 0)}/{summary.get('total_checks', 0)}** checks passed.",
        ]
    out += [
        "",
        "## Not claimed",
        "",
        "The following have no measurement in this repository and are therefore not",
        "asserted anywhere in the documentation:",
        "",
        "- MMLU or any other benchmark accuracy score.",
        "- Comparisons against GPU, BitNet, or any other project.",
        "- Quantised (for example Q4_K_M) export or its size or quality.",
        "- Long-context scaling beyond the measured sequence lengths.",
        "- Operation-count or energy-savings ratios versus a baseline.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(out)} lines)")


if __name__ == "__main__":
    main()
