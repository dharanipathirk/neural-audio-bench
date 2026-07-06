# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Generate publication-quality figures for the benchmark paper using seaborn.

Figures:
  1. Isolated RTF bar chart (backend × model, faceted by model_size)
  2. RTF vs buffer size (one panel per model)
  3. Contention degradation ratio (one panel per model)
  4. Utilization percentiles under full contention (one panel per model)
  5. RTF vs p99 divergence scatter (the key finding)
  6. Dropout rate vs contention level
  7. Jitter under load (stddev vs contention)
  8. Instance count cliff (p99 util vs instance count)
  9. Max sustainable instances (bar chart)
  10. AMX instruction mix (stacked bar)
  11. AMX compute intensity (compute/load ratio)

Usage:
  nab plot [--isolated results/isolated.csv] [--contention results/contention.csv]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------------------
# Theme and palette
# ---------------------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2,
    rc={
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.3,
        "axes.edgecolor": ".3",
        "font.family": "serif",
    },
)

# Backend display names (clean labels for figures)
BACKEND_LABELS = {
    "BNNSGraph": "BNNSGraph",
    "RTNeural_Eigen": "RTNeural\n(Eigen)",
    "RTNeural_XSIMD": "RTNeural\n(XSIMD)",
    "Direct_LibTorch": "Direct\nLibTorch",
    "Direct_ONNX": "Direct\nONNX",
    "Anira_LibTorch": "Anira\nLibTorch",
    "Anira_ONNX": "Anira\nONNX",
}

# Inline labels (no newlines, for legends)
BACKEND_LABELS_INLINE = {
    "BNNSGraph": "BNNSGraph",
    "RTNeural_Eigen": "RTNeural (Eigen)",
    "RTNeural_XSIMD": "RTNeural (XSIMD)",
    "Direct_LibTorch": "Direct LibTorch",
    "Direct_ONNX": "Direct ONNX",
    "Anira_LibTorch": "Anira LibTorch",
    "Anira_ONNX": "Anira ONNX",
}

BACKEND_ORDER = [
    "BNNSGraph",
    "RTNeural_Eigen",
    "RTNeural_XSIMD",
    "Direct_LibTorch",
    "Direct_ONNX",
    "Anira_LibTorch",
    "Anira_ONNX",
]

# Curated palette — colourblind-friendly, distinct in print
BACKEND_PALETTE = {
    "BNNSGraph": "#0077BB",
    "RTNeural_Eigen": "#EE7733",
    "RTNeural_XSIMD": "#CC3311",
    "Direct_LibTorch": "#33BBEE",
    "Direct_ONNX": "#EE3377",
    "Anira_LibTorch": "#009988",
    "Anira_ONNX": "#BBBBBB",
}

MODEL_ORDER = ["LSTM", "TCN", "WaveNet"]
SIZE_ORDER = ["small", "medium", "large"]
SIZE_LABELS = {"small": "Small", "medium": "Medium", "large": "Large"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_backend(name: str) -> str:
    return BACKEND_LABELS_INLINE.get(name, name)


def _active(df: pd.DataFrame, col: str, order: list) -> list:
    return [v for v in order if v in df[col].values]


def _cb_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "mode" in df.columns:
        return df[df["mode"] == "callback"]
    if "dimension" in df.columns:
        return df[df["dimension"].str.endswith("_cb", na=False)]
    return df


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add display-friendly label columns for plotting."""
    df = df.copy()
    df["Backend"] = df["backend"].map(BACKEND_LABELS_INLINE).fillna(df["backend"])
    if "model_size" in df.columns:
        df["Size"] = df["model_size"].map(SIZE_LABELS).fillna(df["model_size"])
    else:
        df["Size"] = "All"
    return df


def _backend_hue_order(df: pd.DataFrame) -> list[str]:
    """Return inline-label backend names in canonical order, filtered to those present."""
    present = set(df["Backend"].unique())
    return [
        BACKEND_LABELS_INLINE[b] for b in BACKEND_ORDER if BACKEND_LABELS_INLINE.get(b) in present
    ]


def _backend_palette(df: pd.DataFrame) -> dict:
    """Return palette keyed by inline label, filtered to those present."""
    present = set(df["Backend"].unique())
    return {
        BACKEND_LABELS_INLINE[b]: BACKEND_PALETTE[b]
        for b in BACKEND_ORDER
        if BACKEND_LABELS_INLINE.get(b) in present
    }


def _save(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 – Isolated RTF bar chart
# ---------------------------------------------------------------------------


def fig1_isolated_rtf_bar(df: pd.DataFrame, out_dir: Path):
    data = _cb_filter(df)
    data = data[data["buffer_size"] == 128]
    if data.empty:
        print("  Figure 1: SKIPPED (no callback data at buffer 128)")
        return

    data = _add_labels(data)
    summary = (
        data.groupby(["Backend", "model", "Size"], observed=True)["rtf"].median().reset_index()
    )

    sizes = [SIZE_LABELS[s] for s in SIZE_ORDER if SIZE_LABELS[s] in summary["Size"].values]
    if not sizes:
        sizes = summary["Size"].unique().tolist()

    hue_order = _backend_hue_order(summary)
    palette = _backend_palette(summary)

    n_rows = len(sizes)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes, strict=False):
        sub = summary[summary["Size"] == size]
        sns.barplot(
            data=sub,
            x="model",
            y="rtf",
            hue="Backend",
            hue_order=hue_order,
            palette=palette,
            order=MODEL_ORDER,
            ax=ax,
            edgecolor=".2",
            linewidth=0.5,
        )
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_ylabel("RTF")
        ax.set_xlabel("")
        ax.set_title(size, fontweight="bold")
        ax.legend_.remove()

    axes[-1].set_xlabel("Model Architecture")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(hue_order), 4),
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        fontsize=9,
    )
    fig.suptitle("Isolated Inference RTF at Buffer Size 128", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    _save(fig, out_dir, "fig1_isolated_rtf_bar")
    print("  Figure 1: Isolated RTF bar chart")


# ---------------------------------------------------------------------------
# Figure 2 – RTF vs buffer size
# ---------------------------------------------------------------------------


def fig2_rtf_vs_bufsize(df: pd.DataFrame, out_dir: Path):
    data = _cb_filter(df)
    if data.empty:
        print("  Figure 2: SKIPPED")
        return

    data = _add_labels(data)
    summary = (
        data.groupby(["Backend", "model", "buffer_size"], observed=True)["rtf"]
        .median()
        .reset_index()
    )

    models = _active(summary, "model", MODEL_ORDER)
    hue_order = _backend_hue_order(summary)
    palette = _backend_palette(summary)
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        mdata = summary[summary["model"] == model]
        sns.lineplot(
            data=mdata,
            x="buffer_size",
            y="rtf",
            hue="Backend",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            ax=ax,
            legend=(ax is axes[0]),
        )
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("Buffer Size (samples)")
        ax.set_title(model, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Real-Time Factor")
            sns.move_legend(ax, "upper left", fontsize=8, frameon=True)
        else:
            ax.set_ylabel("")

    fig.suptitle("RTF vs Buffer Size", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig2_rtf_vs_bufsize")
    print("  Figure 2: RTF vs buffer size")


# ---------------------------------------------------------------------------
# Figure 3 – Contention degradation ratio
# ---------------------------------------------------------------------------


def fig3_contention_degradation(contention: pd.DataFrame, isolated: pd.DataFrame, out_dir: Path):
    dim_a = contention[(contention["dimension"] == "dim_a") & (contention["buffer_size"] == 128)]
    if dim_a.empty:
        print("  Figure 3: SKIPPED")
        return

    iso = _cb_filter(isolated)
    iso = iso[iso["buffer_size"] == 128]
    iso_group = ["backend", "model"]
    cont_group = ["backend", "contention_level"]
    if "model_size" in iso.columns and "model_size" in dim_a.columns:
        iso_group = ["backend", "model", "model_size"]
        cont_group = ["backend", "contention_level", "model_size"]
    iso_rtf = iso.groupby(iso_group)["rtf"].median().reset_index()
    iso_rtf.rename(columns={"rtf": "rtf_isolated"}, inplace=True)

    models = _active(dim_a, "model", MODEL_ORDER)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        mdata = dim_a[dim_a["model"] == model]
        summary = mdata.groupby(cont_group)["rtf"].median().reset_index()
        summary["model"] = model
        merge_on = [c for c in iso_group if c in summary.columns]
        summary = summary.merge(iso_rtf, on=merge_on, how="left")
        summary["degradation"] = summary["rtf"] / summary["rtf_isolated"]
        summary = (
            summary.groupby(["backend", "contention_level"])["degradation"].median().reset_index()
        )
        summary = _add_labels(summary)

        hue_order = _backend_hue_order(summary)
        palette = _backend_palette(summary)
        sns.lineplot(
            data=summary,
            x="contention_level",
            y="degradation",
            hue="Backend",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            ax=ax,
            legend=(ax is axes[0]),
        )
        ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Active Conventional Tracks")
        ax.set_title(model, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Degradation Ratio\n(contention / isolated)")
            sns.move_legend(ax, "upper left", fontsize=8, frameon=True)
        else:
            ax.set_ylabel("")

    fig.suptitle(
        "RTF Degradation Under DAW Contention (buffer=128)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig3_contention_degradation")
    print("  Figure 3: Contention degradation")


# ---------------------------------------------------------------------------
# Figure 4 – Utilization percentiles
# ---------------------------------------------------------------------------


def fig4_utilization_distribution(contention: pd.DataFrame, out_dir: Path):
    max_cont = contention["contention_level"].max() if not contention.empty else 36
    dim_a = contention[
        (contention["dimension"] == "dim_a")
        & (contention["buffer_size"] == 128)
        & (contention["contention_level"] == max_cont)
    ]
    if dim_a.empty:
        print("  Figure 4: SKIPPED")
        return

    dim_a = _add_labels(dim_a)
    models = _active(dim_a, "model", MODEL_ORDER)
    pct_cols = ["util_p50", "util_p95", "util_p99", "util_p999", "util_max"]
    pct_labels = ["p50", "p95", "p99", "p99.9", "max"]
    pct_markers = ["o", "s", "D", "^", "v"]

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    hue_order = _backend_hue_order(dim_a)

    for ax, model in zip(axes, models, strict=False):
        mdata = dim_a[dim_a["model"] == model]
        backends = [b for b in hue_order if b in mdata["Backend"].values]
        x = np.arange(len(backends))

        for j, (col, plabel, marker) in enumerate(
            zip(pct_cols, pct_labels, pct_markers, strict=False)
        ):
            if col not in mdata.columns:
                continue
            vals = [
                mdata[mdata["Backend"] == b][col].median()
                if not mdata[mdata["Backend"] == b].empty
                else np.nan
                for b in backends
            ]
            offset = (j - 2) * 0.1
            ax.scatter(
                x + offset,
                vals,
                marker=marker,
                s=60,
                zorder=3,
                edgecolors="black",
                linewidths=0.5,
                label=plabel if (model == models[0]) else "",
            )

        ax.axhline(
            y=100.0,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="100% = xrun" if model == models[0] else "",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(backends, rotation=20, ha="right", fontsize=8)
        ax.set_title(model, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Callback Utilization (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(pct_labels) + 1,
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        fontsize=8,
    )
    fig.suptitle(
        f"Utilization Percentiles Under Full Contention (buffer=128, level={max_cont})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    _save(fig, out_dir, "fig4_utilization_distribution")
    print("  Figure 4: Utilization percentiles")


# ---------------------------------------------------------------------------
# Figure 5 – RTF vs p99 divergence scatter
# ---------------------------------------------------------------------------


def fig5_divergence_scatter(contention: pd.DataFrame, isolated: pd.DataFrame, out_dir: Path):
    if contention.empty or isolated.empty:
        print("  Figure 5: SKIPPED")
        return

    iso = _cb_filter(isolated)
    iso = iso[iso["buffer_size"] == 128]
    group_cols = ["backend", "model"]
    if "model_size" in iso.columns:
        group_cols = ["backend", "model", "model_size"]
    iso_rtf = iso.groupby(group_cols)["rtf"].median().reset_index()
    iso_rtf.rename(columns={"rtf": "isolated_rtf"}, inplace=True)

    max_cont = contention["contention_level"].max() if not contention.empty else 36
    cont = contention[
        (contention["dimension"] == "dim_a")
        & (contention["buffer_size"] == 128)
        & (contention["contention_level"] == max_cont)
    ]
    if cont.empty:
        print("  Figure 5: SKIPPED (no contention data)")
        return

    cont_agg = cont.groupby(group_cols)["util_p99"].median().reset_index()
    merged = cont_agg.merge(iso_rtf, on=group_cols)
    merged = _add_labels(merged)

    hue_order = _backend_hue_order(merged)
    palette = _backend_palette(merged)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sns.scatterplot(
        data=merged,
        x="isolated_rtf",
        y="util_p99",
        hue="Backend",
        style="model" if "model" in merged.columns else None,
        hue_order=hue_order,
        palette=palette,
        s=50,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        zorder=3,
    )

    # Only annotate points above 40% p99 or with large model size (reduce clutter)
    for _, row in merged.iterrows():
        model_size = row.get("model_size", "")
        if row["util_p99"] > 40 or model_size == "large":
            size_tag = f" ({SIZE_LABELS.get(model_size, model_size)})" if model_size else ""
            ax.annotate(
                f"{row['model']}{size_tag}",
                (row["isolated_rtf"], row["util_p99"]),
                fontsize=5.5,
                textcoords="offset points",
                xytext=(4, 4),
            )

    ax.axhline(y=100.0, color="red", linestyle="--", alpha=0.7, linewidth=0.8, label="p99 = 100%")

    # Shade the "false safe" quadrant
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([xlim[0], 1.0], ylim[1], 100.0, alpha=0.06, color="red", zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Isolated RTF", fontsize=8)
    ax.set_ylabel("Contention p99 Util. (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=True, loc="upper left")

    fig.tight_layout()
    _save(fig, out_dir, "fig5_divergence_scatter")
    print("  Figure 5: Divergence scatter")


# ---------------------------------------------------------------------------
# Figure 6 – Dropout rate vs contention
# ---------------------------------------------------------------------------


def fig6_dropout_vs_contention(contention: pd.DataFrame, out_dir: Path):
    dim_a = contention[(contention["dimension"] == "dim_a") & (contention["buffer_size"] == 128)]
    if dim_a.empty:
        print("  Figure 6: SKIPPED")
        return

    if "hw_xruns" not in dim_a.columns:
        dim_a = dim_a.copy()
        dim_a["hw_xruns"] = 0

    agg = (
        dim_a.groupby(["backend", "model", "contention_level"])
        .agg(dropouts=("dropouts", "sum"), hw_xruns=("hw_xruns", "sum"))
        .reset_index()
    )
    agg["total_xruns"] = agg["dropouts"] + agg["hw_xruns"]
    agg = _add_labels(agg)
    agg["contention_level"] = agg["contention_level"].astype(str)

    models = _active(agg, "model", MODEL_ORDER)
    hue_order = _backend_hue_order(agg)
    palette = _backend_palette(agg)
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        mdata = agg[agg["model"] == model]
        sns.barplot(
            data=mdata,
            x="contention_level",
            y="total_xruns",
            hue="Backend",
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            edgecolor=".2",
            linewidth=0.5,
        )
        ax.set_xlabel("Contention Level")
        ax.set_title(model, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Total Xruns (hw + sw)")
            sns.move_legend(ax, "upper left", fontsize=8, frameon=True)
        else:
            ax.set_ylabel("")
            if ax.get_legend():
                ax.get_legend().remove()

    fig.suptitle("Dropout Rate vs Contention Level", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig6_dropout_vs_contention")
    print("  Figure 6: Dropout rate vs contention")


# ---------------------------------------------------------------------------
# Figure 7 – Jitter under load
# ---------------------------------------------------------------------------


def fig7_jitter_under_load(contention: pd.DataFrame, out_dir: Path):
    dim_a = contention[(contention["dimension"] == "dim_a") & (contention["buffer_size"] == 128)]
    if dim_a.empty or "stddev_ns" not in dim_a.columns:
        print("  Figure 7: SKIPPED")
        return

    dim_a = _add_labels(dim_a.copy())
    dim_a["jitter_us"] = dim_a["stddev_ns"] / 1e3
    summary = (
        dim_a.groupby(["Backend", "model", "contention_level"], observed=True)["jitter_us"]
        .median()
        .reset_index()
    )

    models = _active(summary, "model", MODEL_ORDER)
    hue_order = _backend_hue_order(summary)
    palette = _backend_palette(summary)
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        mdata = summary[summary["model"] == model]
        sns.lineplot(
            data=mdata,
            x="contention_level",
            y="jitter_us",
            hue="Backend",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            ax=ax,
            legend=(ax is axes[0]),
        )
        ax.set_xlabel("Active Conventional Tracks")
        ax.set_title(model, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Timing Jitter — stddev (µs)")
            sns.move_legend(ax, "upper left", fontsize=8, frameon=True)
        else:
            ax.set_ylabel("")

    fig.suptitle("Timing Jitter Under Load (buffer=128)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig7_jitter_under_load")
    print("  Figure 7: Jitter under load")


# ---------------------------------------------------------------------------
# Figure 8 – Instance count cliff
# ---------------------------------------------------------------------------


def fig8_instance_cliff(contention: pd.DataFrame, out_dir: Path):
    dim_b = contention[contention["dimension"] == "dim_b"]
    if dim_b.empty:
        print("  Figure 8: SKIPPED")
        return

    dim_b = _add_labels(dim_b)
    summary = (
        dim_b.groupby(["Backend", "model", "instance_count"], observed=True)["util_p99"]
        .median()
        .reset_index()
    )

    models = _active(summary, "model", MODEL_ORDER)
    hue_order = _backend_hue_order(summary)
    palette = _backend_palette(summary)
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(7, 2.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        mdata = summary[summary["model"] == model]
        sns.lineplot(
            data=mdata,
            x="instance_count",
            y="util_p99",
            hue="Backend",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            ax=ax,
            legend=(ax is axes[0]),
        )
        ax.axhline(y=100.0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Neural Plugin Instances")
        ax.set_title(model, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("p99 Callback Utilization (%)")
            sns.move_legend(ax, "upper left", fontsize=8, frameon=True)
        else:
            ax.set_ylabel("")

    fig.suptitle("Instance Count Cliff (buffer=128)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig8_instance_cliff")
    print("  Figure 8: Instance count cliff")


# ---------------------------------------------------------------------------
# Figure 9 – Max sustainable instances
# ---------------------------------------------------------------------------


def fig9_max_sustainable(contention: pd.DataFrame, out_dir: Path):
    dim_b = contention[contention["dimension"] == "dim_b"]
    if dim_b.empty:
        print("  Figure 9: SKIPPED")
        return

    group_cols = ["backend", "model"]
    if "model_size" in dim_b.columns:
        group_cols = ["backend", "model", "model_size"]

    results = []
    for keys, group in dim_b.groupby(group_cols):
        if len(group_cols) == 3:
            backend, model, model_size = keys
        else:
            backend, model = keys
            model_size = "unknown"
        summary = group.groupby("instance_count")["util_p99"].median()
        sustainable = summary[summary < 100.0]
        max_inst = int(sustainable.index.max()) if not sustainable.empty else 0
        results.append(
            {
                "backend": backend,
                "model": model,
                "model_size": model_size,
                "max_instances": max_inst,
            }
        )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return

    result_df = _add_labels(result_df)
    sizes = [
        SIZE_LABELS.get(s, s)
        for s in SIZE_ORDER
        if SIZE_LABELS.get(s, s) in result_df.get("Size", pd.Series()).values
    ]
    if not sizes:
        sizes = result_df["Size"].unique().tolist() if "Size" in result_df.columns else ["Unknown"]

    hue_order = _backend_hue_order(result_df)
    palette = _backend_palette(result_df)

    n_rows = len(sizes)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes, strict=False):
        sub = (
            result_df[result_df.get("Size", pd.Series(dtype=str)) == size]
            if "Size" in result_df.columns
            else result_df
        )
        sns.barplot(
            data=sub,
            x="model",
            y="max_instances",
            hue="Backend",
            hue_order=hue_order,
            palette=palette,
            order=MODEL_ORDER,
            ax=ax,
            edgecolor=".2",
            linewidth=0.5,
        )
        ax.set_ylabel("Max Instances")
        ax.set_xlabel("")
        ax.set_title(size, fontweight="bold")
        if ax is not axes[0] and ax.get_legend():
            ax.get_legend().remove()
        elif ax.get_legend():
            sns.move_legend(ax, "upper right", fontsize=8, frameon=True)

    axes[-1].set_xlabel("Model Architecture")
    fig.suptitle(
        "Max Neural Plugin Instances Before Xrun (p99 < 100%)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig9_max_sustainable")
    print("  Figure 9: Max sustainable instances")


# ---------------------------------------------------------------------------
# AMX Helpers
# ---------------------------------------------------------------------------

AMX_CATEGORIES = ["loads", "compute", "stores", "extract", "control"]
AMX_PALETTE = {
    "Loads": "#4878d0",
    "Compute": "#ee854a",
    "Stores": "#6acc65",
    "Extract": "#d65f5f",
    "Control": "#b47cc7",
}


def _load_amx_results(script_file: Path) -> list[dict]:
    # AMX scan output lands in microarch/amx_results (see
    # microarch/run_amx_analysis.sh); these figures are optional and skipped
    # when no scan has been run.
    from neural_audio_bench.config import repo_root

    amx_dir = repo_root() / "microarch" / "amx_results"
    if not amx_dir.exists():
        print(
            f"  AMX: directory not found ({amx_dir}) — skipping (optional; run microarch/run_amx_analysis.sh to generate)"
        )
        return []
    json_files = sorted(amx_dir.glob("*.json"))
    if not json_files:
        print(f"  AMX: no JSON files in {amx_dir} — skipping")
        return []

    records = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text())
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                flat = {"backend": item.get("backend", jf.stem)}
                libs = item.get("libraries", {})
                if libs and isinstance(libs, dict):
                    for cat in AMX_CATEGORIES:
                        flat[cat] = sum(v.get(cat, 0) for v in libs.values())
                else:
                    for cat in AMX_CATEGORIES:
                        flat[cat] = item.get(cat, 0)
                records.append(flat)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  AMX: could not read {jf.name}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Figure 10 – AMX instruction mix
# ---------------------------------------------------------------------------


def fig10_amx_instruction_mix(out_dir: Path, script_file: Path):
    records = _load_amx_results(script_file)
    if not records:
        return

    amx_df = pd.DataFrame(records)
    if "backend" not in amx_df.columns:
        print("  Figure 10: SKIPPED")
        return

    backends = _active(amx_df, "backend", BACKEND_ORDER)
    if not backends:
        backends = amx_df["backend"].unique().tolist()

    for col in AMX_CATEGORIES:
        if col not in amx_df.columns:
            amx_df[col] = 0

    agg = amx_df.groupby("backend")[AMX_CATEGORIES].sum().reset_index()
    agg["Backend"] = agg["backend"].map(BACKEND_LABELS_INLINE).fillna(agg["backend"])

    fig, ax = plt.subplots(figsize=(8, 5))
    backend_labels = [BACKEND_LABELS_INLINE.get(b, b) for b in backends]
    x = np.arange(len(backends))
    bottom = np.zeros(len(backends))

    for cat in AMX_CATEGORIES:
        vals = np.array(
            [
                agg.loc[agg["backend"] == b, cat].values[0]
                if not agg.loc[agg["backend"] == b].empty
                else 0
                for b in backends
            ],
            dtype=float,
        )
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=cat.capitalize(),
            color=AMX_PALETTE.get(cat.capitalize(), "#888"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(backend_labels, rotation=15, ha="right")
    ax.set_ylabel("AMX Instruction Count")
    ax.set_title("AMX Instruction Mix per Backend", fontsize=14, fontweight="bold")
    ax.legend(title="Category", frameon=True)

    fig.tight_layout()
    _save(fig, out_dir, "fig10_amx_instruction_mix")
    print("  Figure 10: AMX instruction mix")


# ---------------------------------------------------------------------------
# Figure 11 – AMX compute intensity
# ---------------------------------------------------------------------------


def fig11_amx_compute_intensity(out_dir: Path, script_file: Path):
    records = _load_amx_results(script_file)
    if not records:
        return

    amx_df = pd.DataFrame(records)
    if "backend" not in amx_df.columns:
        print("  Figure 11: SKIPPED")
        return

    for col in ["loads", "compute"]:
        if col not in amx_df.columns:
            amx_df[col] = 0

    agg = amx_df.groupby("backend")[["loads", "compute"]].sum().reset_index()
    agg["intensity"] = agg.apply(
        lambda r: r["compute"] / r["loads"] if r["loads"] > 0 else 0.0, axis=1
    )
    agg["Backend"] = agg["backend"].map(BACKEND_LABELS_INLINE).fillna(agg["backend"])

    backends = _active(agg, "backend", BACKEND_ORDER)
    if not backends:
        backends = agg["backend"].unique().tolist()

    plot_df = agg[agg["backend"].isin(backends)].copy()
    plot_df["backend"] = pd.Categorical(plot_df["backend"], categories=backends, ordered=True)
    plot_df = plot_df.sort_values("backend")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [BACKEND_PALETTE.get(b, "#555") for b in plot_df["backend"]]
    bars = ax.bar(
        np.arange(len(plot_df)),
        plot_df["intensity"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, val in zip(bars, plot_df["intensity"].values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(
        [BACKEND_LABELS_INLINE.get(b, b) for b in plot_df["backend"]], rotation=15, ha="right"
    )
    ax.set_ylabel("Compute ops per load\n(higher = better weight reuse)")
    ax.set_title("AMX Compute Intensity per Backend", fontsize=14, fontweight="bold")

    fig.tight_layout()
    _save(fig, out_dir, "fig11_amx_compute_intensity")
    print("  Figure 11: AMX compute intensity")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--isolated", default="results/isolated.csv")
    parser.add_argument("--contention", default="results/contention.csv")
    parser.add_argument("--output-dir", default="results/figures")


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script_file = Path(__file__).resolve()

    print("Generating figures...")

    isolated = pd.DataFrame()
    contention = pd.DataFrame()

    if Path(args.isolated).exists():
        isolated = pd.read_csv(args.isolated)
        if "dimension" in isolated.columns and "mode" not in isolated.columns:
            isolated["mode"] = isolated["dimension"].apply(
                lambda d: "callback" if str(d).endswith("_cb") else "infer"
            )
        fig1_isolated_rtf_bar(isolated, out_dir)
        fig2_rtf_vs_bufsize(isolated, out_dir)
    else:
        print(f"  Isolated CSV not found: {args.isolated}")

    if Path(args.contention).exists():
        contention = pd.read_csv(args.contention)
        fig4_utilization_distribution(contention, out_dir)
        fig6_dropout_vs_contention(contention, out_dir)
        fig7_jitter_under_load(contention, out_dir)
        fig8_instance_cliff(contention, out_dir)
        fig9_max_sustainable(contention, out_dir)
    else:
        print(f"  Contention CSV not found: {args.contention}")

    if not isolated.empty and not contention.empty:
        fig3_contention_degradation(contention, isolated, out_dir)
        fig5_divergence_scatter(contention, isolated, out_dir)

    fig10_amx_instruction_mix(out_dir, script_file)
    fig11_amx_compute_intensity(out_dir, script_file)

    print(f"\nAll figures saved to: {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
