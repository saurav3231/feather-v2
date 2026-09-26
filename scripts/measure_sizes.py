"""Measure real parameter counts and checkpoint sizes for the size ladder.

The previous release shipped configs named ``feather_5M`` ... ``feather_100M``
and artifacts labelled ``feather-v2-40M``, none of which were ever built. This
script instantiates each config for real and reports what came out, so the
labels can be corrected to match reality instead of aspiration.

Usage::

    python scripts/measure_sizes.py
    python scripts/measure_sizes.py --configs configs/feather_5M.json

The report is written to ``configs/size_report.json`` by default, which is where
the paper, the report generator and the docs read it from. Pass ``--out`` to
write somewhere else. Only ``feather_*.json`` files are treated as model
configs, so this script never tries to measure its own output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from feather_v2.model import FeatherV2Model, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def measure(config: dict) -> dict:
    """Instantiate and measure. No estimation, no naming assumptions."""
    model = FeatherV2Model(config)
    breakdown = model.parameter_breakdown()
    parameters = model.count_parameters()

    probe = torch.zeros(1, min(8, int(config["seq_len"])), dtype=torch.long)

    return {
        "config": {k: config[k] for k in sorted(config)},
        "parameters": parameters,
        "parameters_trainable": model.count_parameters(trainable_only=True),
        "measured_size_label": model.size_label(),
        "claimed_label": config.get("name", "unnamed"),
        "f32_bytes": parameters * 4,
        "f16_bytes": parameters * 2,
        "largest_group": (max(breakdown, key=breakdown.get) if breakdown else None),
        "breakdown": breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=sorted(str(p) for p in CONFIG_DIR.glob("feather_*.json")),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(CONFIG_DIR / "size_report.json"),
        help="Where to write the report (default: configs/size_report.json).",
    )
    args = parser.parse_args()

    rows = []
    for path in args.configs:
        name = Path(path).name
        try:
            config = load_config(path)
        except Exception as error:  # noqa: BLE001
            print(f"{name}: FAILED to load -- {error}")
            continue
        try:
            row = measure(config)
        except Exception as error:  # noqa: BLE001
            print(f"{name}: FAILED to build -- {error}")
            continue
        row["source"] = name
        rows.append(row)
        print(
            f"{name:28s} {row['parameters']:>12,} params  "
            f"-> {row['measured_size_label']:>6s}  "
            f"f16 {row['f16_bytes'] / 2**20:8.1f} MiB  "
            f"largest={row['largest_group']}"
        )

    if rows:
        total = sum(r["parameters"] for r in rows)
        print(f"{'TOTAL':28s} {total:>12,} params")
        print()
        print("Note: the measured label is derived from the real parameter count.")
        print("Any config whose name overstates its size is mislabelled.")

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
