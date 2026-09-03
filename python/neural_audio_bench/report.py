# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Generate a report from a run: analysis, summary tables, and publication figures.

Point it at the merged isolated CSV and the contention CSV of a run (the
defaults are the paths a run in ``results/`` produces) and it writes the
enriched analysis files, five summary tables, and the figure set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from . import analyze, plot_paper


def _write_summary_tables(isolated_path: str, contention_path: str, out: Path) -> None:
    """Write five machine-readable summary tables.

    Isolated throughput per backend/model/size, RTF at buffer 128 for the
    large tier, and one table each for contention Dimensions A (mix
    contention, levels 0 and 36), B (instance count, TCN large), and C
    (serial depth 1 and 7, TCN large, neural and full-callback p99).
    """
    isolated = analyze.load_isolated(isolated_path)
    neural, callbacks = analyze.load_contention(contention_path)

    throughput = (
        analyze.logical_throughput_rows(isolated)
        .groupby(["backend", "model", "model_size"], as_index=False)
        .agg(throughput_xrt=("throughput_xrt", "median"))
    )
    throughput.to_csv(out / "table_isolated_throughput.csv", index=False)

    rtf128 = (
        isolated[
            (isolated["mode"] == "callback")
            & (isolated["buffer_size"] == 128)
            & (isolated["model_size"] == "large")
        ]
        .groupby(["backend", "model"], as_index=False)
        .agg(rtf=("rtf", "median"))
    )
    rtf128.to_csv(out / "table_isolated_rtf128.csv", index=False)

    dim_a = (
        neural[
            (neural["dimension"] == "dim_a")
            & (neural["buffer_size"] == 128)
            & (neural["model_size"] == "large")
            & (neural["contention_level"].isin([0, 36]))
        ]
        .groupby(["backend", "model", "contention_level"], as_index=False)
        .agg(
            util_p50=("util_p50", "median"),
            util_p99=("util_p99", "median"),
            hw_xruns=("hw_xruns", "sum"),
            inf_underruns=("inf_underruns", "sum"),
        )
    )
    dim_a.to_csv(out / "table_contention_dim_a.csv", index=False)

    dim_b = (
        neural[
            (neural["dimension"] == "dim_b")
            & (neural["model"] == "TCN")
            & (neural["model_size"] == "large")
            & (neural["buffer_size"] == 128)
        ]
        .groupby(["backend", "instance_count"], as_index=False)
        .agg(
            util_p99=("util_p99", "median"),
            hw_xruns=("hw_xruns", "sum"),
            inf_underruns=("inf_underruns", "sum"),
        )
    )
    dim_b.to_csv(out / "table_contention_dim_b.csv", index=False)

    dim_c_neural = neural[
        (neural["dimension"] == "dim_c")
        & (neural["model"] == "TCN")
        & (neural["model_size"] == "large")
        & (neural["contention_level"].isin([1, 7]))
    ][["backend", "contention_level", "util_p99"]].rename(columns={"util_p99": "neural_util_p99"})
    dim_c_callbacks = callbacks[
        (callbacks["dimension"] == "dim_c_cb")
        & (callbacks["model"] == "TCN")
        & (callbacks["model_size"] == "large")
        & (callbacks["contention_level"].isin([1, 7]))
    ][["backend", "contention_level", "util_p99"]].rename(columns={"util_p99": "callback_util_p99"})
    dim_c = dim_c_neural.merge(dim_c_callbacks, on=["backend", "contention_level"])
    dim_c.to_csv(out / "table_contention_dim_c.csv", index=False)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--isolated",
        default="results/isolated_merged.csv",
        help="Merged isolated CSV of the run (default: results/isolated_merged.csv).",
    )
    parser.add_argument(
        "--contention",
        default="results/contention.csv",
        help="Contention CSV of the run (default: results/contention.csv).",
    )
    parser.add_argument("--output-dir", default="results/report", help="Report output directory.")
    parser.add_argument(
        "--pgf",
        action="store_true",
        help="Render figures through xelatex (requires a TeX install).",
    )


def run(args: argparse.Namespace) -> int:
    isolated = args.isolated
    contention = args.contention
    output_dir = args.output_dir

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

    _write_summary_tables(isolated, contention, out)

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
    parser = argparse.ArgumentParser(
        description="Generate analysis tables and figures from a run", allow_abbrev=False
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
