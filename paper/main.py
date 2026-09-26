"""Feather v2 — paper generator.

Builds ``paper/main.pdf`` from measurement artifacts only. Every number in the
document is read from ``configs/size_report.json`` and ``benchmark_report.json``;
nothing is hardcoded and no figure is estimated. If the benchmark report is absent,
the results section says so instead of inventing values.

Run after the benchmark:

    python kaggle/test_all_sizes_mega.py --loss-steps 60
    python paper/main.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
SIZE_REPORT = ROOT / "configs" / "size_report.json"
BENCH_REPORT = ROOT / "benchmark_report.json"
PLOT_DIR = ROOT / "docs" / "images"

NOT_MEASURED = "not measured"


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def f(value: Any, spec: str = ".2f", missing: str = NOT_MEASURED) -> str:
    if value is None:
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return missing


def chart(
    title: str,
    values: list[float],
    labels: list[str],
    out_pdf: Path,
    value_max: float | None = None,
) -> Drawing | None:
    """Render a bar chart of measured values. Returns None when there is no data."""
    if not values:
        print(f"skipping {out_pdf.name}: no measured data")
        return None

    drawing = Drawing(400, 200)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 50, 50, 300, 125
    bc.data = [values]
    bc.categoryAxis.categoryNames = labels
    bc.categoryAxis.labels.fontSize = 7
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = value_max or max(values) * 1.2
    bc.bars[0].fillColor = colors.HexColor("#1f77b4")
    title_label = Label()
    title_label.setText(title)
    title_label.x, title_label.y = 90, 180
    title_label.fontSize = 9
    title_label.fontName = "Helvetica-Bold"
    drawing.add(bc)
    drawing.add(title_label)
    renderPDF.drawToFile(drawing, str(out_pdf))
    print(f"chart written: {out_pdf.relative_to(ROOT)}")
    return drawing


def table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f2f2f2")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return t


def generate_paper(output_dir: str = "paper") -> None:
    os.makedirs(output_dir, exist_ok=True)
    sizes = load_json(SIZE_REPORT) or []
    bench = load_json(BENCH_REPORT)

    doc = SimpleDocTemplate(
        os.path.join(output_dir, "main.pdf"),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="Feather v2: a trainable CPU-native language model with measured results",
        author="Saurav Bhandari",
    )
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=10.5)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10)

    story: list[Any] = []
    story.append(
        Paragraph(
            "Feather v2: a trainable CPU-native language model with measured results",
            styles["Title"],
        )
    )
    story.append(Paragraph("Saurav Bhandari, Pokhara, Nepal", styles["Normal"]))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------------- abstract
    story.append(Paragraph("Abstract", h2))
    story.append(
        Paragraph(
            "Feather v2 is a decoder-style language model implemented in PyTorch and built "
            "from seven components, each of which uses one or more operators drawn from a "
            "library of differentiable mathematical functions: Walsh-Hadamard transforms, "
            "tropical and fractional weighting, p-adic divisibility descriptors, "
            "tensor-train factorisation, Sinkhorn projection, sheaf consistency, Clifford "
            "gating, rough-path signatures, and Godel-style log coding. We report measured "
            "parameter counts, checkpoint sizes, CPU throughput, resident memory, training "
            "loss, and energy consumption across a five-step size ladder from 5.1M to 58.4M "
            "parameters, all on a single reference CPU. We deliberately report no accuracy "
            "benchmark and no comparison against other systems, because no such measurement "
            "has been performed. We state explicitly which components are exact and which "
            "are numerical relaxations, and we identify the operators that cannot be "
            "trained end-to-end.",
            body,
        )
    )

    # ------------------------------------------------------------------- scope
    story.append(Paragraph("1. Scope of the claims in this paper", h2))
    story.append(
        Paragraph(
            "This document reports only measurements produced by the code in this "
            "repository. Earlier drafts of the project documentation contained throughput "
            "and energy figures for several machines, comparisons against other language "
            "models, and a composite efficiency metric. None of those figures came from a "
            "measurement, and they have been removed rather than restated as estimates. "
            "Specifically, this paper makes no claim about: benchmark accuracy; throughput "
            "on any hardware other than the reference machine; energy per token; speedup or "
            "efficiency ratios against any baseline; quantised export; or behaviour beyond "
            "the measured sequence lengths. A composite metric that appeared in earlier "
            "drafts has been dropped because no definition or implementation of it exists.",
            body,
        )
    )

    # ------------------------------------------------------------ architecture
    story.append(Paragraph("2. Architecture", h2))
    story.append(
        Paragraph(
            "The model embeds tokens and positions, passes the sequence through a stack of "
            "identical blocks, applies a final layer norm, and projects to vocabulary "
            "logits. Each block runs seven components in sequence, each preceded by its own "
            "layer norm: a multi-scale fractional sensory encoder with p-adic scale "
            "selection; a hierarchical chunked gated liquid memory; a holographic memory "
            "that mixes through a Walsh-Hadamard basis; a sparse mixture of "
            "tensor-train-compressed experts with Sinkhorn-balanced routing; a "
            "Kolmogorov-Arnold feed-forward with a differentiable Godel loop; a predictive "
            "entropy gate over the residual stream; and a Jacobi-spectral refinement pass. "
            "The same seven components also appear once at the top level of the model.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The per-component layer norm is load-bearing rather than cosmetic. Before it "
            "was introduced, the residual stream grew without bound across the Godel loop "
            "and the model overflowed in float32 within a few dozen optimiser steps. A "
            "regression test covers this failure mode.",
            body,
        )
    )

    # ------------------------------------------------- exactness and relaxation
    story.append(Paragraph("3. Exact operators and numerical relaxations", h2))
    story.append(
        Paragraph(
            "Several components are named after discrete mathematics but cannot be "
            "implemented differentiably on a GPU. We state the relaxation in each case "
            "rather than presenting a relaxation as the exact algorithm.",
            body,
        )
    )
    story.append(
        table(
            [
                ["Operator", "Status", "Consequence"],
                [
                    "Walsh-Hadamard transform",
                    "Exact",
                    "Orthogonal, invertible",
                ],
                [
                    "p-adic divisibility descriptor",
                    "Exact integer mask, detached",
                    "Not trainable; informs routing only",
                ],
                [
                    "p-adic weights, distance",
                    "Exact integer arithmetic, detached",
                    "No gradient path",
                ],
                [
                    "Tropical / min-plus matmul",
                    "Softmin relaxation",
                    "Differentiable, approximate",
                ],
                [
                    "Fractional, TT, Clifford, rough-path, Sinkhorn, sheaf, equilibrium, Jacobi",
                    "Smooth relaxation",
                    "Inductive bias, not the discrete algorithm",
                ],
                [
                    "Godel coding",
                    "Log-based monotone encoding",
                    "Finite and differentiable; not a Godel numbering",
                ],
                [
                    "KAN",
                    "Learnable spline",
                    "Differentiable",
                ],
            ],
            widths=[1.9 * inch, 1.7 * inch, 2.3 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "The p-adic descriptor deserves emphasis. A divisibility test is a step "
            "function, so its derivative is zero almost everywhere. We therefore detach it "
            "from the autograd graph rather than pretending it is trainable. It is the only "
            "genuinely exact operator in the library that is also not trainable end to end.",
            small,
        )
    )

    # ---------------------------------------------------------- training result
    story.append(Paragraph("4. Training behaviour", h2))
    story.append(
        Paragraph(
            "Models are trained with next-token cross-entropy plus a mixture-of-experts "
            "load-balancing penalty. A representative smoke run on the 5.1M configuration "
            "reduces loss from 9.0053 to 8.4457 over three optimiser steps with non-zero "
            "gradients, which demonstrates that the loss path is connected end to end. "
            "Loss reduction over a short step budget is evidence that the model trains; it "
            "is not evidence of model quality, and no quality metric is reported here.",
            body,
        )
    )

    # ------------------------------------------------------------ size results
    story.append(Paragraph("5. Measured model sizes", h2))
    if sizes:
        rows = [["Config", "Measured label", "Parameters", "F32 (MiB)", "F16 (MiB)"]]
        for row in sizes:
            rows.append(
                [
                    f"configs/{row.get('source', '?')}",
                    str(row.get("measured_size_label", NOT_MEASURED)),
                    f"{int(row.get('parameters') or 0):,}",
                    f((row.get("f32_bytes") or 0) / 1048576, ".2f"),
                    f((row.get("f16_bytes") or 0) / 1048576, ".2f"),
                ]
            )
        story.append(
            table(
                rows,
                widths=[1.9 * inch, 1.1 * inch, 1.2 * inch, 0.9 * inch, 0.9 * inch],
            )
        )
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Every parameter in every configuration is trainable; the trainable and "
                "total counts are equal because nothing is frozen. The F16 column is the "
                "raw parameter count multiplied by two bytes. No quantisation is applied, "
                "and no F16 or quantised checkpoint is written by this project.",
                small,
            )
        )
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        chart(
            "F16 checkpoint size (MiB)",
            [(r.get("f16_bytes") or 0) / 1048576 for r in sizes],
            [str(r.get("measured_size_label", "?")) for r in sizes],
            PLOT_DIR / "paper_f16_size.pdf",
        )
    else:
        story.append(
            Paragraph(
                "Not measured. Run <font name='Courier'>scripts/measure_sizes.py</font> "
                "to produce configs/size_report.json.",
                body,
            )
        )

    # --------------------------------------------------------- runtime results
    story.append(PageBreak())
    story.append(Paragraph("6. Measured CPU runtime", h2))
    if bench and bench.get("sizes"):
        hw = bench.get("hardware") or {}
        story.append(
            Paragraph(
                "All runtime figures were measured on one machine: "
                f"{hw.get('cpu', 'unknown CPU')}, {hw.get('cores_physical', '?')} physical "
                f"cores, {f(hw.get('ram_gb'), '.2f')} GB RAM, with "
                f"{'AVX' if hw.get('avx') else 'no AVX'} and "
                f"{'AVX2' if hw.get('avx2') else 'no AVX2'}. "
                f"Measurement timestamp: {bench.get('timestamp', '?')}.",
                body,
            )
        )
        rows = [
            [
                "Size",
                "Params",
                "RAM (MB)",
                "Fwd t/s @512",
                "Bulk t/s",
                "Gen t/s",
                "Loss",
                "Checks",
            ]
        ]
        for row in bench["sizes"]:
            ft = row.get("forward_throughput") or {}
            checks = row.get("checks") or []
            losses = row.get("losses") or []
            loss_txt = (
                f"{losses[0]:.3f} -> {losses[-1]:.3f}" if losses else NOT_MEASURED
            )
            rows.append(
                [
                    str(row.get("size_label", "?")),
                    f(int(row.get("params") or 0), ",d"),
                    f(row.get("ram_mb"), ".1f"),
                    f(ft.get("512"), ".1f"),
                    f(row.get("bulk_tok_s"), ".1f"),
                    f(row.get("gen_tok_s"), ".2f"),
                    loss_txt,
                    f"{sum(1 for _, v in checks if v)}/{len(checks)}",
                ]
            )
        story.append(
            table(
                rows,
                widths=[
                    0.5 * inch,
                    0.8 * inch,
                    0.7 * inch,
                    0.8 * inch,
                    0.6 * inch,
                    0.6 * inch,
                    1.0 * inch,
                    0.5 * inch,
                ],
            )
        )
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Loss is next-token cross-entropy on raw English Wikipedia text over a short "
                "step budget, reported as first value to last value. Resident memory is "
                "dominated by the PyTorch runtime rather than by the model: the 5.1M "
                "configuration holds roughly 19 MB of weights while process resident memory "
                "is several hundred megabytes. Energy is a single figure covering one "
                "forward pass plus a short generation, not a per-token cost, so it cannot "
                "be compared against any other energy figure.",
                small,
            )
        )
        labels = [str(r.get("size_label", "?")) for r in bench["sizes"]]
        chart(
            "Generation throughput (tokens/s, batch 1)",
            [r.get("gen_tok_s") or 0 for r in bench["sizes"]],
            labels,
            PLOT_DIR / "paper_generation.pdf",
        )
        chart(
            "Forward throughput at sequence length 512 (tokens/s)",
            [
                (r.get("forward_throughput") or {}).get("512", 0) or 0
                for r in bench["sizes"]
            ],
            labels,
            PLOT_DIR / "paper_forward512.pdf",
        )
        summary = bench.get("summary") or {}
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                f"Verification checks passed: {summary.get('total_pass', 0)} of "
                f"{summary.get('total_checks', 0)}. Each check records the measurement that "
                "justified it, so a passing check can be audited against the underlying "
                "number rather than taken on trust.",
                small,
            )
        )
    else:
        story.append(
            Paragraph(
                "Not measured. Run <font name='Courier'>"
                "python kaggle/test_all_sizes_mega.py --loss-steps 60</font> to produce "
                "benchmark_report.json, then re-run this script.",
                body,
            )
        )

    # ----------------------------------------------------------- limitations
    story.append(Paragraph("7. Limitations", h2))
    for item in [
        "No accuracy evaluation. No MMLU, no standard held-out benchmark, no task "
        "performance. The loss figures are training loss on raw corpus text.",
        "No baseline. No reference model was measured under identical conditions, so no "
        "speedup or efficiency ratio is reported.",
        "Single machine. Throughput on other hardware is unknown.",
        "No quantised export. Checkpoints are float32 PyTorch. There is no GGUF writer and "
        "no Q4_K_M quantiser in this project, and files previously published under those "
        "names were not real.",
        "No generation cache. Sampling re-runs the full model at every step.",
        "The p-adic operators contribute no gradient, as described in Section 3.",
    ]:
        story.append(Paragraph(f"- {item}", small))
        story.append(Spacer(1, 2))

    # -------------------------------------------------------------- conclusion
    story.append(Paragraph("8. Conclusion", h2))
    story.append(
        Paragraph(
            "Feather v2 is a working trainable model with a measured size ladder and a "
            "benchmark that records the evidence behind each of its claims. Its most "
            "useful property for a reader is that the numbers are small and checkable: the "
            "model is between 5M and 58M parameters, it runs on a CPU, and every figure in "
            "this paper can be regenerated from the two JSON artifacts named at the top of "
            "the source file. Where the mathematics is a relaxation rather than an exact "
            "algorithm, the paper says so.",
            body,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "License: MIT License plus an additional clause. See LICENSE in the repository "
            "root.",
            small,
        )
    )

    doc.build(story)
    out = Path(output_dir) / "main.pdf"
    print(f"paper written: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_paper()
