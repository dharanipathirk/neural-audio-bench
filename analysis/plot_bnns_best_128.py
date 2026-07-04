#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Create a focused 128-sample real-time comparison figure for BNNSGraph."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BLUE = "#0077BB"
ORANGE = "#CC3311"
GREEN = "#009988"
GREY = "#BBBBBB"

MODELS = ["LSTM", "TCN", "WaveNet"]
MODEL_LABELS = {"LSTM": "LSTM-L", "TCN": "TCN-L", "WaveNet": "WaveNet-L"}

BACKENDS_MAIN = ["BNNSGraph", "RTNeural_XSIMD"]
BACKENDS_ASYNC = ["Anira_LibTorch", "Anira_ONNX"]
BACKEND_LABELS = {
    "BNNSGraph": "BNNSGraph",
    "RTNeural_XSIMD": "RTNeural-XSIMD",
    "Anira_LibTorch": "anira-LibTorch",
    "Anira_ONNX": "anira-ONNX RT",
}
BACKEND_COLORS = {
    "BNNSGraph": BLUE,
    "RTNeural_XSIMD": ORANGE,
    "Anira_LibTorch": GREEN,
    "Anira_ONNX": GREY,
}


def worst_case_rows(df: pd.DataFrame, backends: list[str]) -> pd.DataFrame:
    rows = []
    for backend in backends:
        for model in MODELS:
            sub = df[(df["backend"] == backend) & (df["model"] == model)]
            if sub.empty:
                continue
            row = sub.loc[sub["util_p99"].idxmax()].copy()
            row["model_label"] = MODEL_LABELS[model]
            rows.append(row)
    return pd.DataFrame(rows)


def annotate_bars(ax, bars, values, rows: pd.DataFrame, async_panel: bool = False):
    for bar, value, (_, row) in zip(bars, values, rows.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=".2",
        )

        if not async_panel and int(row["hw_xruns"]) > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 6, 166),
                f'{int(row["hw_xruns"])} xruns (c={int(row["contention_level"])})',
                ha="center",
                va="bottom",
                fontsize=7,
                color="#9E2A2B",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9E2A2B", lw=0.6, alpha=0.9),
            )


def main():
    root = Path(__file__).resolve().parents[1]
    contention_path = root / "results" / "contention.csv"
    out_dir = root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "grid.alpha": 0.25,
            "axes.edgecolor": ".3",
            "font.family": "serif",
        },
    )

    cont = pd.read_csv(contention_path)
    data = cont[
        (cont["dimension"] == "dim_a")
        & (cont["buffer_size"] == 128)
        & (cont["model_size"] == "large")
    ].copy()

    if data.empty:
        raise SystemExit("No dimension-A large-model rows found at buffer 128.")

    main_rows = worst_case_rows(data, BACKENDS_MAIN)
    async_rows = worst_case_rows(data, BACKENDS_ASYNC)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.75, 1.15]},
    )

    x = np.arange(len(MODELS))
    width = 0.34

    for i, backend in enumerate(BACKENDS_MAIN):
        rows = main_rows[main_rows["backend"] == backend].set_index("model").loc[MODELS].reset_index()
        vals = rows["util_p99"].to_numpy()
        bars = ax1.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=BACKEND_COLORS[backend],
            edgecolor=".25",
            linewidth=0.6,
            label=BACKEND_LABELS[backend],
        )
        annotate_bars(ax1, bars, vals, rows)

    async_width = 0.30
    for i, backend in enumerate(BACKENDS_ASYNC):
        rows = async_rows[async_rows["backend"] == backend].set_index("model").loc[MODELS].reset_index()
        vals = rows["util_p99"].to_numpy()
        bars = ax2.bar(
            x + (i - 0.5) * async_width,
            vals,
            width=async_width,
            color=BACKEND_COLORS[backend],
            edgecolor=".35",
            linewidth=0.6,
            hatch="//" if backend == "Anira_ONNX" else None,
            label=BACKEND_LABELS[backend],
        )
        annotate_bars(ax2, bars, vals, rows, async_panel=True)

    for ax in (ax1, ax2):
        ax.axhline(100, color="red", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_ylim(0, 170)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_title("Zero-Latency RT Backends", fontweight="bold")
    ax1.set_ylabel("Worst-case p99 deadline used (%)\nDimension A, buffer 128, large models")
    ax2.set_title("Async Wrappers (+latency)", fontweight="bold")
    ax2.set_facecolor("#F7F7F7")

    ax1.text(
        0.03,
        0.96,
        "BNNSGraph stays below 20% for all\nlarge models across contention levels.",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BLUE, lw=0.8, alpha=0.95),
    )

    ax2.text(
        0.03,
        0.96,
        "Shown for context only:\nbackground inference lowers\naudio-thread cost, but adds latency.",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=".6", lw=0.6, alpha=0.95),
    )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "BNNSGraph is the strongest real-time choice at 128 samples",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.95,
        "Direct LibTorch / ONNX Runtime are omitted here because they are not real-time safe and are excluded from contention runs.",
        ha="center",
        fontsize=8,
        color=".35",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.86])
    fig.savefig(out_dir / "bnns_best_realtime_128.png")
    fig.savefig(out_dir / "bnns_best_realtime_128.pdf")
    plt.close(fig)

    print("Saved:")
    print(out_dir / "bnns_best_realtime_128.png")
    print(out_dir / "bnns_best_realtime_128.pdf")


if __name__ == "__main__":
    main()
