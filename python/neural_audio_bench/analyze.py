# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Analyze benchmark results: merge CSVs, compute derived metrics.

Usage:
  nab analyze [--isolated results/isolated.csv] [--contention results/contention.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _drop_non_ok(df: pd.DataFrame) -> pd.DataFrame:
    """Drop skipped/error rows (results schema v2). v1 CSVs have no status column."""
    if "status" in df.columns:
        return df[df["status"] == "ok"].copy()
    return df


def load_isolated(path: str) -> pd.DataFrame:
    """Load and enrich isolated benchmark CSV."""
    df = _drop_non_ok(pd.read_csv(path))

    # Backward-compatible model_size column
    if "model_size" not in df.columns:
        df["model_size"] = "unknown"

    # Add derived metrics
    # per_sample_ns is only meaningful for per-buffer rows (buffer_size > 0).
    # Throughput rows (mode=="throughput") have buffer_size==0; median_ns there
    # is already per-sample time, so the division is a no-op (0 → 1).
    df["per_sample_ns"] = df["median_ns"] / df["buffer_size"].replace(0, 1)
    df["throughput_xrt"] = np.where(df["rtf"] > 0, 1.0 / df["rtf"], np.nan)

    return df


def logical_throughput_rows(isolated: pd.DataFrame) -> pd.DataFrame:
    """Return one throughput row per backend/model/size configuration.

    ``nab run`` merges the two engine binaries' CSVs without duplicating the
    backends they share, so current runs already have one row per
    configuration. A CSV merged by simple concatenation carries a second
    physical row for each shared backend; this keeps the first one.
    """
    throughput = isolated[isolated["mode"] == "throughput"]
    return throughput.drop_duplicates(
        subset=["backend", "model", "model_size", "buffer_size", "rep"], keep="first"
    )


def load_contention(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and enrich contention benchmark CSV.

    Returns:
        (df_neural, df_cb) where df_neural contains rows whose dimension does
        NOT end in '_cb', and df_cb contains rows whose dimension ends in '_cb'.
        Both DataFrames have a 'row_type' column ('neural' or 'callback').
    """
    df = _drop_non_ok(pd.read_csv(path))

    # Ensure correct dtypes for integer columns
    for col in ["hw_xruns", "inf_underruns", "thread_count"]:
        df[col] = df[col].fillna(0).astype(int)

    # Split on dimension suffix
    is_cb = df["dimension"].str.endswith("_cb", na=False)
    df_neural = df[~is_cb].copy()
    df_cb = df[is_cb].copy()

    df_neural["row_type"] = "neural"
    df_cb["row_type"] = "callback"

    return df_neural, df_cb


def compute_degradation(isolated: pd.DataFrame, contention: pd.DataFrame) -> pd.DataFrame:
    """Compute RTF degradation: RTF_contention / RTF_isolated.

    Groups by (backend, model, model_size).
    Baseline is isolated RTF at buffer_size=128 (the standard comparison point).
    """
    # Get isolated RTF at buffer 128
    iso_128 = (
        isolated[(isolated["buffer_size"] == 128) & (isolated["mode"] == "callback")]
        .groupby(["backend", "model", "model_size"])["rtf"]
        .median()
        .reset_index()
    )
    iso_128.rename(columns={"rtf": "rtf_isolated"}, inplace=True)

    # Get contention RTF from neural rows only, dimension == "dim_a"
    cont = contention[contention["dimension"] == "dim_a"].copy()

    if cont.empty or iso_128.empty:
        return pd.DataFrame()

    merged = cont.merge(iso_128, on=["backend", "model", "model_size"], how="left")
    merged["degradation"] = merged["rtf"] / merged["rtf_isolated"]

    return merged


def print_summary(isolated: pd.DataFrame):
    """Print human-readable summary of isolated results."""
    print("\n" + "=" * 70)
    print("ISOLATED BENCHMARK SUMMARY")
    print("=" * 70)

    # Throughput summary
    throughput = logical_throughput_rows(isolated)
    if not throughput.empty:
        print("\n--- Throughput (x realtime) ---")
        for _, row in throughput.iterrows():
            xrt = 1.0 / row["rtf"] if row["rtf"] > 0 else float("inf")
            print(
                f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s}: "
                f"{xrt:>8.1f}x realtime"
            )

    # Per-buffer summary at buffer=128
    buf128 = isolated[(isolated["buffer_size"] == 128) & (isolated["mode"] == "callback")]
    if not buf128.empty:
        print("\n--- Per-buffer latency at buffer=128 (median, ns) ---")
        summary = (
            buf128.groupby(["backend", "model", "model_size"])
            .agg(
                median_ns=("median_ns", "median"),
                p99_ns=("p99_ns", "median"),
                rtf=("rtf", "median"),
                dropouts=("dropouts", "sum"),
            )
            .reset_index()
        )

        for _, row in summary.iterrows():
            print(
                f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s}: "
                f"median={row['median_ns']:>10,.0f} ns  "
                f"p99={row['p99_ns']:>10,.0f} ns  "
                f"RTF={row['rtf']:.4f}  "
                f"dropouts={int(row['dropouts'])}"
            )

    # Dropout analysis
    print("\n--- Dropout analysis (buffer=32, hardest deadline) ---")
    buf32 = isolated[(isolated["buffer_size"] == 32) & (isolated["mode"] == "callback")]
    if not buf32.empty:
        summary32 = (
            buf32.groupby(["backend", "model", "model_size"])["dropouts"].sum().reset_index()
        )
        for _, row in summary32.iterrows():
            status = "PASS" if row["dropouts"] == 0 else f"FAIL ({int(row['dropouts'])} xruns)"
            print(f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s}: {status}")


def print_contention_summary(df_neural: pd.DataFrame, df_cb: pd.DataFrame):
    """Print summary of contention results (neural and callback rows)."""
    print("\n" + "=" * 70)
    print("CONTENTION BENCHMARK SUMMARY")
    print("=" * 70)

    # --- Neural rows ---
    dim_a = df_neural[df_neural["dimension"] == "dim_a"]
    dim_b = df_neural[df_neural["dimension"] == "dim_b"]

    if not dim_a.empty:
        print("\n--- Dimension A: Utilization under contention (buffer=128) ---")
        summary = (
            dim_a[dim_a["buffer_size"] == 128]
            .groupby(["backend", "model", "model_size", "contention_level"])
            .agg(
                util_p50=("util_p50", "median"),
                util_p99=("util_p99", "median"),
                dropouts=("dropouts", "sum"),
                inf_underruns=("inf_underruns", "sum"),
            )
            .reset_index()
        )

        for _, row in summary.iterrows():
            inf_str = (
                f"  inf_underruns={int(row['inf_underruns'])}" if row["inf_underruns"] > 0 else ""
            )
            print(
                f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s} / "
                f"contention={row['contention_level']:>2d}: "
                f"p50={row['util_p50']:>6.1f}%  "
                f"p99={row['util_p99']:>6.1f}%  "
                f"xruns={int(row['dropouts'])}{inf_str}"
            )

    if not dim_b.empty:
        print("\n--- Dimension B: Max sustainable instances (p99 < 100%) ---")
        summary_b = (
            dim_b.groupby(["backend", "model", "model_size", "instance_count"])
            .agg(util_p99=("util_p99", "median"))
            .reset_index()
        )

        for (backend, model, model_size), group in summary_b.groupby(
            ["backend", "model", "model_size"]
        ):
            sustainable = group[group["util_p99"] < 100.0]
            max_inst = sustainable["instance_count"].max() if not sustainable.empty else 0
            print(f"  {backend:20s} / {model:8s} / {model_size:8s}: max {max_inst} instances")

    # --- hw_xruns summary ---
    all_neural = df_neural.copy()
    if not all_neural.empty and all_neural["hw_xruns"].sum() > 0:
        print("\n--- Hardware xruns per config (neural rows) ---")
        xrun_summary = (
            all_neural.groupby(["backend", "model", "model_size", "buffer_size"])
            .agg(total_hw_xruns=("hw_xruns", "sum"))
            .reset_index()
        )
        xrun_summary = xrun_summary[xrun_summary["total_hw_xruns"] > 0]
        for _, row in xrun_summary.iterrows():
            print(
                f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s} / "
                f"buf={int(row['buffer_size']):>4d}: "
                f"hw_xruns={int(row['total_hw_xruns'])}"
            )

    # --- Inference underruns summary (anira backends) ---
    if not all_neural.empty and all_neural["inf_underruns"].sum() > 0:
        print("\n--- Inference underruns per config (output not ready) ---")
        inf_summary = (
            all_neural.groupby(["backend", "model", "model_size", "buffer_size"])
            .agg(total_inf_underruns=("inf_underruns", "sum"))
            .reset_index()
        )
        inf_summary = inf_summary[inf_summary["total_inf_underruns"] > 0]
        for _, row in inf_summary.iterrows():
            print(
                f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s} / "
                f"buf={int(row['buffer_size']):>4d}: "
                f"inf_underruns={int(row['total_inf_underruns'])}"
            )

    # --- thread_count summary ---
    if not all_neural.empty and (all_neural["thread_count"] > 0).any():
        print("\n--- Observed audio thread count per config (neural rows) ---")
        thread_summary = (
            all_neural.groupby(["backend", "model", "model_size", "buffer_size"])
            .agg(thread_count=("thread_count", "max"))
            .reset_index()
        )
        for _, row in thread_summary.iterrows():
            print(
                f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s} / "
                f"buf={int(row['buffer_size']):>4d}: "
                f"threads={int(row['thread_count'])}"
            )

    # --- Callback rows summary ---
    if not df_cb.empty:
        print("\n--- Full-callback timing (dimension ends in _cb) ---")
        cb_summary = (
            df_cb.groupby(["backend", "model", "model_size", "dimension", "buffer_size"])
            .agg(
                util_p50=("util_p50", "median"),
                util_p99=("util_p99", "median"),
                hw_xruns=("hw_xruns", "sum"),
                inf_underruns=("inf_underruns", "sum"),
            )
            .reset_index()
        )
        for _, row in cb_summary.iterrows():
            inf_str = (
                f"  inf_underruns={int(row['inf_underruns'])}" if row["inf_underruns"] > 0 else ""
            )
            print(
                f"  {row['backend']:20s} / {row['model']:8s} / {row['model_size']:8s} / "
                f"{row['dimension']:12s} / buf={int(row['buffer_size']):>4d}: "
                f"p50={row['util_p50']:>6.1f}%  "
                f"p99={row['util_p99']:>6.1f}%  "
                f"hw_xruns={int(row['hw_xruns'])}{inf_str}"
            )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--isolated", default="results/isolated.csv")
    parser.add_argument("--contention", default="results/contention.csv")
    parser.add_argument("--output", default="results/analysis_summary.csv")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def derived_path(label: str) -> Path:
        suffix = output.suffix or ".csv"
        stem = output.stem if output.suffix else output.name
        return output.with_name(f"{stem}_{label}{suffix}")

    if Path(args.isolated).exists():
        isolated = load_isolated(args.isolated)
        print_summary(isolated)

        # Save enriched data
        out_path = derived_path("isolated")
        isolated.to_csv(out_path, index=False)
        print(f"\nEnriched isolated data saved to: {out_path}")
    else:
        print(f"Isolated results not found: {args.isolated}")
        isolated = None

    if Path(args.contention).exists():
        df_neural, df_cb = load_contention(args.contention)
        print_contention_summary(df_neural, df_cb)

        # Save enriched contention data
        cont_path = derived_path("contention")
        pd.concat([df_neural, df_cb], ignore_index=True).to_csv(cont_path, index=False)
        print(f"\nEnriched contention data (all rows) saved to: {cont_path}")

        cb_path = derived_path("contention_cb")
        df_cb.to_csv(cb_path, index=False)
        print(f"Callback-only contention data saved to: {cb_path}")

        if isolated is not None:
            degradation = compute_degradation(isolated, df_neural)
            if not degradation.empty:
                deg_path = derived_path("degradation")
                degradation.to_csv(deg_path, index=False)
                print(f"Degradation data saved to: {deg_path}")
    else:
        print(f"\nContention results not found: {args.contention}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze benchmark results", allow_abbrev=False)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
