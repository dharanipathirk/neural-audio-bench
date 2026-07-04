# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Generate the publication-quality figures for the DAFx-26 paper.

Four paper figures plus the focused 128-sample BNNSGraph comparison.

By default figures render with matplotlib's Agg backend so no TeX install is
required. Pass ``--pgf`` to render through xelatex for paper-exact fonts (needs
a working TeX installation on PATH).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

BLUE = "#0077BB"  # BNNSGraph
ORANGE = "#CC3311"  # RTNeural
GREEN = "#009988"  # Anira LibTorch
GREY = "#BBBBBB"  # Anira ONNX
AMBER = "#EE7733"  # Anira underruns


# ---------------------------------------------------------------------------
# Style configuration (moved out of module scope so path/backend are parameters)
# ---------------------------------------------------------------------------

_COMMON_RC = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.6",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "axes.edgecolor": ".3",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

# PGF backend: all text rendered by LaTeX => fonts match the paper exactly
_PGF_RC = {
    **_COMMON_RC,
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join(
        [
            r"\usepackage{fontspec}",
            r"\setmainfont{Times New Roman}[Ligatures=TeX]",
        ]
    ),
    "text.usetex": True,
}

_NONPGF_RC = {
    **_COMMON_RC,
    "text.usetex": False,
}


def _configure_backend(use_pgf: bool) -> None:
    if use_pgf:
        os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
        matplotlib.use("pgf", force=True)
        plt.rcParams.update(_PGF_RC)
    else:
        matplotlib.use("Agg", force=True)
        plt.rcParams.update(_NONPGF_RC)


# ================================================================
# FIGURE 1: Two-panel hero — Throughput (left) + Contention gap (right)
# ================================================================
def fig1_hero(iso: pd.DataFrame, cont: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))

    models = ["TCN", "WaveNet", "LSTM"]
    model_labels = {"TCN": "TCN-L", "WaveNet": "WaveNet-L", "LSTM": "LSTM-L"}

    iso_cb = iso[
        (iso["mode"] == "callback") & (iso["buffer_size"] == 128) & (iso["model_size"] == "large")
    ]
    cont_c0 = cont[
        (cont["dimension"] == "dim_a")
        & (cont["buffer_size"] == 128)
        & (cont["model_size"] == "large")
        & (cont["contention_level"] == 0)
    ]

    dumb_backends = ["BNNSGraph", "RTNeural_XSIMD"]
    dumb_colors = {"BNNSGraph": BLUE, "RTNeural_XSIMD": ORANGE}
    dumb_labels = {"BNNSGraph": "BNNSGraph", "RTNeural_XSIMD": "RTNeural"}

    y_pos = 0
    yticks, ylabels = [], []

    for model in models:
        for backend in dumb_backends:
            iso_rtf = iso_cb[(iso_cb["backend"] == backend) & (iso_cb["model"] == model)][
                "rtf"
            ].median()
            iso_pct = iso_rtf * 100
            c0_p99 = cont_c0[(cont_c0["backend"] == backend) & (cont_c0["model"] == model)][
                "util_p99"
            ].median()
            c0_xruns = int(
                cont_c0[(cont_c0["backend"] == backend) & (cont_c0["model"] == model)][
                    "hw_xruns"
                ].sum()
            )
            color = dumb_colors[backend]

            ax.plot(
                [iso_pct, c0_p99], [y_pos, y_pos], color=color, linewidth=1.2, alpha=0.5, zorder=2
            )
            ax.scatter(
                iso_pct, y_pos, s=30, facecolors="white", edgecolors=color, linewidths=1.2, zorder=3
            )
            ax.scatter(
                c0_p99, y_pos, s=30, facecolors=color, edgecolors=color, linewidths=0.8, zorder=3
            )

            if c0_xruns > 0:
                ax.annotate(
                    f"{c0_p99:.0f}\\%, {c0_xruns} xruns",
                    (c0_p99, y_pos),
                    fontsize=5.5,
                    fontweight="semibold",
                    color="#9E2A2B",
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="white", ec="#9E2A2B", alpha=0.8, lw=0.5
                    ),
                )

            yticks.append(y_pos)
            ylabels.append(f"{model_labels[model]}  {dumb_labels[backend]}")
            y_pos += 1
        y_pos += 0.3

    ax.axvline(x=100, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Deadline used (\\%)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlim(-5, 175)
    ax.invert_yaxis()

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="w",
            markeredgecolor="grey",
            markersize=4,
            label="Isolated",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markeredgecolor="grey",
            markersize=4,
            label="Core Audio p99",
        ),
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(outdir / "paper_fig1_hero.pdf")
    plt.close(fig)
    print("  Paper Fig 1: Isolated vs Core Audio p99 dumbbell")


# ================================================================
# FIGURE 2: P99 vs contention level (3 stacked panels)
# ================================================================
def fig2_p99_vs_contention(cont: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.5), sharex=True, sharey=True)

    models = ["LSTM", "TCN", "WaveNet"]
    da = cont[
        (cont["dimension"] == "dim_a")
        & (cont["buffer_size"] == 128)
        & (cont["model_size"] == "large")
    ]

    for ax, model in zip(axes, models, strict=False):
        for backend, color, marker, label in [
            ("BNNSGraph", BLUE, "o", "BNNSGraph"),
            ("RTNeural_XSIMD", ORANGE, "s", "RTNeural"),
        ]:
            sub = da[(da["backend"] == backend) & (da["model"] == model)]
            agg = sub.groupby("contention_level").agg(p99=("util_p99", "median")).reset_index()
            ax.plot(
                agg["contention_level"],
                agg["p99"],
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                label=label,
                zorder=3,
            )

        ax.axhline(y=100, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(model + "-Large", fontsize=9, fontweight="bold")
        ax.set_ylabel("p99 Util. (\\%)", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Active Conventional Tracks")
    axes[-1].set_xticks([0, 8, 24, 36])
    axes[0].set_ylim(-5, 170)
    axes[0].legend(fontsize=7, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(outdir / "paper_fig2_p99_contention.pdf")
    plt.close(fig)
    print("  Paper Fig 2: p99 vs contention level")


# ================================================================
# FIGURE 3: Instance scaling with two failure modes
# ================================================================
def fig3_instance_scaling(cont: pd.DataFrame, outdir: Path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.4, 4.0), gridspec_kw={"height_ratios": [3, 1.5]}
    )

    # Top panel: TCN-Large p99 vs instances
    db = cont[
        (cont["dimension"] == "dim_b") & (cont["model"] == "TCN") & (cont["model_size"] == "large")
    ]

    for backend, color, marker, label in [
        ("BNNSGraph", BLUE, "o", "BNNSGraph"),
        ("RTNeural_XSIMD", ORANGE, "s", "RTNeural"),
    ]:
        sub = db[db["backend"] == backend]
        agg = (
            sub.groupby("instance_count")
            .agg(p99=("util_p99", "median"), xruns=("hw_xruns", "sum"))
            .reset_index()
        )
        ax1.plot(
            agg["instance_count"],
            agg["p99"],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            label=label,
            zorder=3,
        )

        # Annotate xruns
        for _, row in agg.iterrows():
            if row["xruns"] > 0:
                xr_label = f"{int(row['xruns'])} xr"
                ax1.annotate(
                    xr_label,
                    (row["instance_count"], row["p99"]),
                    fontsize=6,
                    fontweight="semibold",
                    color="#9E2A2B",
                    xytext=(4, 6),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#9E2A2B", alpha=0.8),
                )

    ax1.axhline(y=100, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("p99 Callback Util. (\\%)", fontsize=8)
    ax1.set_title("TCN-Large: Audio Thread Failures", fontsize=9, fontweight="bold")
    ax1.set_xticks([1, 2, 4, 8, 16])
    ax1.legend(fontsize=7, loc="upper left", frameon=True)
    ax1.grid(alpha=0.3)

    # Bottom panel: Anira inference underruns at N=16, LSTM-Large
    db_lstm16 = cont[
        (cont["dimension"] == "dim_b")
        & (cont["model"] == "LSTM")
        & (cont["model_size"] == "large")
        & (cont["instance_count"] == 16)
    ]

    backends_anira = ["Anira_LibTorch", "Anira_ONNX"]
    labels_anira = ["anira\n(LibTorch)", "anira\n(ONNX~RT)"]
    colors_anira = [GREEN, GREY]
    underruns = []
    for b in backends_anira:
        sub = db_lstm16[db_lstm16["backend"] == b]
        underruns.append(sub["inf_underruns"].sum())

    bars = ax2.bar(
        labels_anira, underruns, color=colors_anira, edgecolor=".3", linewidth=0.5, width=0.5
    )
    for bar, ur in zip(bars, underruns, strict=False):
        ur_str = f"{ur:,}".replace(",", "{,}")  # LaTeX-safe thousands separator
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.5,
            ur_str,
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="semibold",
            color=AMBER,
        )

    ax2.set_ylabel("Inference\nUnderruns", fontsize=8)
    ax2.set_title("LSTM-Large N=16: Background Thread Failures", fontsize=8, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(10, 200000)
    ax2.text(
        0.98,
        0.85,
        "0 audio xruns",
        transform=ax2.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        style="italic",
        color=".4",
    )
    ax2.grid(axis="y", alpha=0.3)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(outdir / "paper_fig3_instance_failures.pdf")
    plt.close(fig)
    print("  Paper Fig 3: Instance scaling with dual failure modes")


# ================================================================
# FIGURE 4: BNNSGraph speedup ratio vs buffer size
# ================================================================
def fig4_speedup_vs_bufsize(iso: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))

    cb = iso[iso["mode"] == "callback"]
    bufsizes = [32, 64, 128, 256, 512, 1024]

    model_colors = {"TCN": BLUE, "WaveNet": ORANGE, "LSTM": GREEN}
    for model, marker in [("TCN", "o"), ("WaveNet", "s"), ("LSTM", "^")]:
        ratios = []
        for buf in bufsizes:
            bnns = cb[
                (cb["backend"] == "BNNSGraph")
                & (cb["model"] == model)
                & (cb["model_size"] == "large")
                & (cb["buffer_size"] == buf)
            ]
            rtn = cb[
                (cb["backend"] == "RTNeural_XSIMD")
                & (cb["model"] == model)
                & (cb["model_size"] == "large")
                & (cb["buffer_size"] == buf)
            ]
            if not bnns.empty and not rtn.empty:
                bnns_rtf = bnns["rtf"].median()
                rtn_rtf = rtn["rtf"].median()
                ratios.append(rtn_rtf / bnns_rtf if bnns_rtf > 0 else 0)
            else:
                ratios.append(0)
        ax.plot(
            bufsizes,
            ratios,
            color=model_colors[model],
            marker=marker,
            markersize=5,
            linewidth=1.5,
            label=f"{model}-L",
            zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(bufsizes)
    ax.set_xticklabels([str(b) for b in bufsizes])
    ax.set_xlabel("Buffer size (samples)")
    ax.set_ylabel("Speedup ($\\times$)")
    ax.set_title("\\texttt{BNNSGraph} speedup over \\texttt{RTNeural-XSIMD}", fontsize=8)
    ax.axhline(y=1, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", frameon=True)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "paper_fig4_speedup_bufsize.pdf")
    plt.close(fig)
    print("  Paper Fig 4: BNNSGraph speedup vs buffer size")


# ================================================================
# FIGURE 5 (folded from plot_bnns_best_128): focused 128-sample RT comparison
# ================================================================
BNNS_MODELS = ["LSTM", "TCN", "WaveNet"]
BNNS_MODEL_LABELS = {"LSTM": "LSTM-L", "TCN": "TCN-L", "WaveNet": "WaveNet-L"}

BNNS_BACKENDS_MAIN = ["BNNSGraph", "RTNeural_XSIMD"]
BNNS_BACKENDS_ASYNC = ["Anira_LibTorch", "Anira_ONNX"]
BNNS_BACKEND_LABELS = {
    "BNNSGraph": "BNNSGraph",
    "RTNeural_XSIMD": "RTNeural-XSIMD",
    "Anira_LibTorch": "anira-LibTorch",
    "Anira_ONNX": "anira-ONNX RT",
}
BNNS_BACKEND_COLORS = {
    "BNNSGraph": BLUE,
    "RTNeural_XSIMD": ORANGE,
    "Anira_LibTorch": GREEN,
    "Anira_ONNX": GREY,
}


def _bnns_worst_case_rows(df: pd.DataFrame, backends: list[str]) -> pd.DataFrame:
    rows = []
    for backend in backends:
        for model in BNNS_MODELS:
            sub = df[(df["backend"] == backend) & (df["model"] == model)]
            if sub.empty:
                continue
            row = sub.loc[sub["util_p99"].idxmax()].copy()
            row["model_label"] = BNNS_MODEL_LABELS[model]
            rows.append(row)
    return pd.DataFrame(rows)


def _bnns_annotate_bars(ax, bars, values, rows: pd.DataFrame, async_panel: bool = False):
    for bar, value, (_, row) in zip(bars, values, rows.iterrows(), strict=False):
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
                f"{int(row['hw_xruns'])} xruns (c={int(row['contention_level'])})",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#9E2A2B",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9E2A2B", lw=0.6, alpha=0.9),
            )


def fig_bnns_best_128(cont: pd.DataFrame, outdir: Path):
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

    data = cont[
        (cont["dimension"] == "dim_a")
        & (cont["buffer_size"] == 128)
        & (cont["model_size"] == "large")
    ].copy()

    if data.empty:
        print("  Paper Fig BNNS-128: SKIPPED (no dimension-A large-model rows at buffer 128)")
        return

    main_rows = _bnns_worst_case_rows(data, BNNS_BACKENDS_MAIN)
    async_rows = _bnns_worst_case_rows(data, BNNS_BACKENDS_ASYNC)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.75, 1.15]},
    )

    x = np.arange(len(BNNS_MODELS))
    width = 0.34

    for i, backend in enumerate(BNNS_BACKENDS_MAIN):
        rows = (
            main_rows[main_rows["backend"] == backend]
            .set_index("model")
            .loc[BNNS_MODELS]
            .reset_index()
        )
        vals = rows["util_p99"].to_numpy()
        bars = ax1.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=BNNS_BACKEND_COLORS[backend],
            edgecolor=".25",
            linewidth=0.6,
            label=BNNS_BACKEND_LABELS[backend],
        )
        _bnns_annotate_bars(ax1, bars, vals, rows)

    async_width = 0.30
    for i, backend in enumerate(BNNS_BACKENDS_ASYNC):
        rows = (
            async_rows[async_rows["backend"] == backend]
            .set_index("model")
            .loc[BNNS_MODELS]
            .reset_index()
        )
        vals = rows["util_p99"].to_numpy()
        bars = ax2.bar(
            x + (i - 0.5) * async_width,
            vals,
            width=async_width,
            color=BNNS_BACKEND_COLORS[backend],
            edgecolor=".35",
            linewidth=0.6,
            hatch="//" if backend == "Anira_ONNX" else None,
            label=BNNS_BACKEND_LABELS[backend],
        )
        _bnns_annotate_bars(ax2, bars, vals, rows, async_panel=True)

    for ax in (ax1, ax2):
        ax.axhline(100, color="red", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([BNNS_MODEL_LABELS[m] for m in BNNS_MODELS])
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
        "Direct LibTorch / ONNX Runtime are omitted here because they are not "
        "real-time safe and are excluded from contention runs.",
        ha="center",
        fontsize=8,
        color=".35",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.86])
    fig.savefig(outdir / "bnns_best_realtime_128.png")
    fig.savefig(outdir / "bnns_best_realtime_128.pdf")
    plt.close(fig)
    print("  Paper Fig BNNS-128: strongest real-time choice at 128 samples")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--isolated", default="results/isolated_xsimd.csv")
    parser.add_argument("--contention", default="results/contention.csv")
    parser.add_argument("--output-dir", default="results/figures")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pgf",
        dest="pgf",
        action="store_true",
        help="Render through xelatex for paper-exact fonts (requires a TeX install).",
    )
    group.add_argument(
        "--no-pgf",
        dest="pgf",
        action="store_false",
        help="Render with the Agg backend (default; no TeX required).",
    )
    parser.set_defaults(pgf=False)


def run(args: argparse.Namespace) -> int:
    _configure_backend(args.pgf)

    iso = pd.read_csv(args.isolated)
    cont = pd.read_csv(args.contention)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig1_hero(iso, cont, outdir)
    fig2_p99_vs_contention(cont, outdir)
    fig3_instance_scaling(cont, outdir)
    fig4_speedup_vs_bufsize(iso, outdir)
    fig_bnns_best_128(cont, outdir)

    print(f"\nDone. Figures saved to {outdir}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate DAFx-26 paper figures")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
