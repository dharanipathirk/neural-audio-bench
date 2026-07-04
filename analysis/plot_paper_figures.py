#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Generate 3 publication-quality figures for the DAFx26 paper."""

import os
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# PGF backend: all text rendered by LaTeX => fonts match the paper exactly
plt.rcParams.update({
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r"\usepackage{fontspec}",
        r"\setmainfont{Times New Roman}[Ligatures=TeX]",
    ]),
    "text.usetex": True,
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
})

BLUE = "#0077BB"    # BNNSGraph
ORANGE = "#CC3311"  # RTNeural
GREEN = "#009988"   # Anira LibTorch
GREY = "#BBBBBB"    # Anira ONNX
AMBER = "#EE7733"   # Anira underruns

iso = pd.read_csv("results/isolated_xsimd.csv")
cont = pd.read_csv("results/contention.csv")
outdir = Path("results/figures")
outdir.mkdir(exist_ok=True)

# ================================================================
# FIGURE 1: Two-panel hero — Throughput (left) + Contention gap (right)
# ================================================================
def fig1_hero():
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))

    models = ["TCN", "WaveNet", "LSTM"]
    model_labels = {"TCN": "TCN-L", "WaveNet": "WaveNet-L", "LSTM": "LSTM-L"}

    iso_cb = iso[(iso["mode"] == "callback") & (iso["buffer_size"] == 128) & (iso["model_size"] == "large")]
    cont_c0 = cont[(cont["dimension"] == "dim_a") & (cont["buffer_size"] == 128) &
                    (cont["model_size"] == "large") & (cont["contention_level"] == 0)]

    dumb_backends = ["BNNSGraph", "RTNeural_XSIMD"]
    dumb_colors = {"BNNSGraph": BLUE, "RTNeural_XSIMD": ORANGE}
    dumb_labels = {"BNNSGraph": "BNNSGraph", "RTNeural_XSIMD": "RTNeural"}

    y_pos = 0
    yticks, ylabels = [], []

    for model in models:
        for backend in dumb_backends:
            iso_rtf = iso_cb[(iso_cb["backend"] == backend) & (iso_cb["model"] == model)]["rtf"].median()
            iso_pct = iso_rtf * 100
            c0_p99 = cont_c0[(cont_c0["backend"] == backend) & (cont_c0["model"] == model)]["util_p99"].median()
            c0_xruns = int(cont_c0[(cont_c0["backend"] == backend) & (cont_c0["model"] == model)]["hw_xruns"].sum())
            color = dumb_colors[backend]

            ax.plot([iso_pct, c0_p99], [y_pos, y_pos], color=color, linewidth=1.2, alpha=0.5, zorder=2)
            ax.scatter(iso_pct, y_pos, s=30, facecolors="white", edgecolors=color, linewidths=1.2, zorder=3)
            ax.scatter(c0_p99, y_pos, s=30, facecolors=color, edgecolors=color, linewidths=0.8, zorder=3)

            if c0_xruns > 0:
                ax.annotate(f"{c0_p99:.0f}\\%, {c0_xruns} xruns",
                            (c0_p99, y_pos), fontsize=5.5, fontweight="semibold",
                            color="#9E2A2B", xytext=(6, 0), textcoords="offset points",
                            va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#9E2A2B", alpha=0.8, lw=0.5))

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
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w", markeredgecolor="grey",
               markersize=4, label="Isolated"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="grey",
               markersize=4, label="Core Audio p99"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(outdir / "paper_fig1_hero.pdf")
    plt.close(fig)
    print("  Paper Fig 1: Isolated vs Core Audio p99 dumbbell")

# ================================================================
# FIGURE 2: P99 vs contention level (3 stacked panels)
# ================================================================
def fig2_p99_vs_contention():
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.5), sharex=True, sharey=True)
    
    models = ["LSTM", "TCN", "WaveNet"]
    da = cont[(cont["dimension"]=="dim_a") & (cont["buffer_size"]==128) & (cont["model_size"]=="large")]
    
    for ax, model in zip(axes, models):
        for backend, color, marker, label in [
            ("BNNSGraph", BLUE, "o", "BNNSGraph"),
            ("RTNeural_XSIMD", ORANGE, "s", "RTNeural"),
        ]:
            sub = da[(da["backend"]==backend) & (da["model"]==model)]
            agg = sub.groupby("contention_level").agg(p99=("util_p99", "median")).reset_index()
            ax.plot(agg["contention_level"], agg["p99"], color=color, marker=marker, 
                    markersize=5, linewidth=1.5, label=label, zorder=3)
        
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
def fig3_instance_scaling():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.0), 
                                     gridspec_kw={"height_ratios": [3, 1.5]})
    
    # Top panel: TCN-Large p99 vs instances
    db = cont[(cont["dimension"]=="dim_b") & (cont["model"]=="TCN") & (cont["model_size"]=="large")]
    
    for backend, color, marker, label in [
        ("BNNSGraph", BLUE, "o", "BNNSGraph"),
        ("RTNeural_XSIMD", ORANGE, "s", "RTNeural"),
    ]:
        sub = db[db["backend"]==backend]
        agg = sub.groupby("instance_count").agg(
            p99=("util_p99", "median"), xruns=("hw_xruns", "sum")
        ).reset_index()
        ax1.plot(agg["instance_count"], agg["p99"], color=color, marker=marker,
                markersize=5, linewidth=1.5, label=label, zorder=3)
        
        # Annotate xruns
        for _, row in agg.iterrows():
            if row["xruns"] > 0:
                xr_label = f'{int(row["xruns"])} xr'
                ax1.annotate(xr_label,
                           (row["instance_count"], row["p99"]),
                           fontsize=6, fontweight="semibold", color="#9E2A2B",
                           xytext=(4, 6), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#9E2A2B", alpha=0.8))
    
    ax1.axhline(y=100, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("p99 Callback Util. (\\%)", fontsize=8)
    ax1.set_title("TCN-Large: Audio Thread Failures", fontsize=9, fontweight="bold")
    ax1.set_xticks([1, 2, 4, 8, 16])
    ax1.legend(fontsize=7, loc="upper left", frameon=True)
    ax1.grid(alpha=0.3)
    
    # Bottom panel: Anira inference underruns at N=16, LSTM-Large
    db_lstm16 = cont[(cont["dimension"]=="dim_b") & (cont["model"]=="LSTM") & 
                     (cont["model_size"]=="large") & (cont["instance_count"]==16)]
    
    backends_anira = ["Anira_LibTorch", "Anira_ONNX"]
    labels_anira = ["anira\n(LibTorch)", "anira\n(ONNX~RT)"]
    colors_anira = [GREEN, GREY]
    underruns = []
    for b in backends_anira:
        sub = db_lstm16[db_lstm16["backend"]==b]
        underruns.append(sub["inf_underruns"].sum())
    
    bars = ax2.bar(labels_anira, underruns, color=colors_anira, edgecolor=".3", linewidth=0.5, width=0.5)
    for bar, ur in zip(bars, underruns):
        ur_str = f"{ur:,}".replace(",", "{,}")  # LaTeX-safe thousands separator
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                ur_str, ha="center", va="bottom", fontsize=7, fontweight="semibold", color=AMBER)
    
    ax2.set_ylabel("Inference\nUnderruns", fontsize=8)
    ax2.set_title("LSTM-Large N=16: Background Thread Failures", fontsize=8, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(10, 200000)
    ax2.text(0.98, 0.85, "0 audio xruns", transform=ax2.transAxes, fontsize=7,
             ha="right", va="top", style="italic", color=".4")
    ax2.grid(axis="y", alpha=0.3)
    ax2.grid(axis="x", visible=False)
    
    fig.tight_layout()
    fig.savefig(outdir / "paper_fig3_instance_failures.pdf")
    plt.close(fig)
    print("  Paper Fig 3: Instance scaling with dual failure modes")

# ================================================================
# FIGURE 4: BNNSGraph speedup ratio vs buffer size
# ================================================================
def fig4_speedup_vs_bufsize():
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))

    cb = iso[iso["mode"] == "callback"]
    bufsizes = [32, 64, 128, 256, 512, 1024]

    model_colors = {"TCN": BLUE, "WaveNet": ORANGE, "LSTM": GREEN}
    for model, marker in [("TCN", "o"), ("WaveNet", "s"), ("LSTM", "^")]:
        ratios = []
        for buf in bufsizes:
            bnns = cb[(cb["backend"] == "BNNSGraph") & (cb["model"] == model) &
                      (cb["model_size"] == "large") & (cb["buffer_size"] == buf)]
            rtn = cb[(cb["backend"] == "RTNeural_XSIMD") & (cb["model"] == model) &
                     (cb["model_size"] == "large") & (cb["buffer_size"] == buf)]
            if not bnns.empty and not rtn.empty:
                bnns_rtf = bnns["rtf"].median()
                rtn_rtf = rtn["rtf"].median()
                ratios.append(rtn_rtf / bnns_rtf if bnns_rtf > 0 else 0)
            else:
                ratios.append(0)
        ax.plot(bufsizes, ratios, color=model_colors[model], marker=marker,
                markersize=5, linewidth=1.5, label=f"{model}-L", zorder=3)

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

fig1_hero()
fig2_p99_vs_contention()
fig3_instance_scaling()
fig4_speedup_vs_bufsize()
print("\nDone. Figures saved to results/figures/")
