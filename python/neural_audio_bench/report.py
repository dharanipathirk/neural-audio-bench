# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Generate a paper report: analysis tables + paper figures in one command.

``--preset dafx26`` reproduces the paper's tables and figures directly from the
archived expected CSVs, so no benchmark run is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from . import analyze, plot_paper
from .config import repo_root

PRESETS = {
    "dafx26": {
        "isolated": "experiments/dafx26-paper/expected/isolated.csv",
        "contention": "experiments/dafx26-paper/expected/contention.csv",
    }
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="Use archived expected CSVs for a known paper experiment.",
    )
    parser.add_argument("--isolated", default=None, help="Isolated CSV (overrides preset default).")
    parser.add_argument(
        "--contention", default=None, help="Contention CSV (overrides preset default)."
    )
    parser.add_argument("--output-dir", default=None, help="Report output directory.")
    parser.add_argument(
        "--pgf",
        action="store_true",
        help="Render paper figures through xelatex (requires a TeX install).",
    )


def run(args: argparse.Namespace) -> int:
    preset = PRESETS.get(args.preset, {}) if args.preset else {}

    def _resolve(value: str | None, preset_key: str, fallback: str) -> str:
        if value is not None:
            return value
        if preset_key in preset:
            return str(repo_root() / preset[preset_key])
        return fallback

    isolated = _resolve(args.isolated, "isolated", "results/isolated_merged.csv")
    contention = _resolve(args.contention, "contention", "results/contention.csv")
    output_dir = args.output_dir or "results/report"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REPORT")
    print(f"  Isolated:   {isolated}")
    print(f"  Contention: {contention}")
    print(f"  Output:     {out}")
    print("=" * 70)

    analyze.run(
        SimpleNamespace(
            isolated=isolated,
            contention=contention,
            output=str(out / "analysis.csv"),
        )
    )

    plot_paper.run(
        SimpleNamespace(
            isolated=isolated,
            contention=contention,
            output_dir=str(out),
            pgf=args.pgf,
        )
    )

    print(f"\nReport written to: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate analysis + paper figures")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
