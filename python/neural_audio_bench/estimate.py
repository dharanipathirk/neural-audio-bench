# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Estimate benchmark runtime from a resolved configuration.

What this can do exactly:
- Compute the exact benchmark-controlled wall time for contention mode from the
  current C++ control flow and config values.

What it cannot do from config alone:
- Compute the exact isolated-mode wall time. Isolated mode adapts its iteration
  count from measured backend speed at runtime, so the true duration depends on
  the machine, binary, backend, model, and buffer size.

Optional:
- If you pass an existing isolated CSV from the same machine/binary/config, the
  script can estimate isolated runtime from those measured rows.

Compile-time backend availability is read from ``build/nab_build_info.json`` when
present, otherwise from the CMake-generated ``flags.make``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import default_base_config, repo_root

DEFAULT_CONFIG = default_base_config()
DEFAULT_BUILD_DIR = repo_root() / "build"
DEFAULT_TARGET = "nab_engine"


FIXED_CONTENTION_OVERHEAD_SECONDS = 1.2
# From runSingleConfig():
#   0.2 settle after device config
#   0.2 after ensureContextAllocated()
#   0.2 after first play()
#   0.2 after warmup stop()
#   0.2 after restart play()
#   0.2 after measurement stop()


@dataclass(frozen=True)
class BuildFeatures:
    rtneural_backend: str
    has_libtorch: bool
    has_onnx: bool
    has_anira: bool


@dataclass
class RuntimeSummary:
    configs: int
    seconds: float


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_flags_make(path: Path) -> BuildFeatures:
    if not path.exists():
        raise FileNotFoundError(f"flags.make not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    defines: set[str] = set()
    for match in re.finditer(r"-D([A-Za-z0-9_]+)(?:=(?:\"[^\"]*\"|[^ \n]+))?", text):
        defines.add(match.group(1))

    if "RTNEURAL_USE_XSIMD" in defines:
        rt_backend = "RTNeural_XSIMD"
    elif "RTNEURAL_USE_EIGEN" in defines:
        rt_backend = "RTNeural_Eigen"
    else:
        raise RuntimeError(f"Could not infer RTNeural variant from {path}")

    return BuildFeatures(
        rtneural_backend=rt_backend,
        has_libtorch=("HAS_LIBTORCH" in defines and "USE_LIBTORCH" in defines),
        has_onnx=("HAS_ONNXRUNTIME" in defines and "USE_ONNXRUNTIME" in defines),
        has_anira=("HAS_ANIRA" in defines),
    )


def build_features_from_info(info: dict, target: str) -> BuildFeatures:
    """Derive build features from a ``nab_build_info.json`` document.

    Expected shape: ``{"targets": {"<target>": {"HAS_LIBTORCH": true, ...}}}``.
    """
    targets = info.get("targets", {})
    if target not in targets:
        raise KeyError(f"target '{target}' not present in nab_build_info.json")
    t = targets[target]

    rt_backend = t.get("rtneural_backend")
    if not rt_backend:
        if t.get("RTNEURAL_USE_XSIMD"):
            rt_backend = "RTNeural_XSIMD"
        elif t.get("RTNEURAL_USE_EIGEN"):
            rt_backend = "RTNeural_Eigen"
    if rt_backend not in ("RTNeural_XSIMD", "RTNeural_Eigen"):
        raise RuntimeError(f"Could not infer RTNeural variant for target '{target}'")

    return BuildFeatures(
        rtneural_backend=rt_backend,
        has_libtorch=bool(t.get("HAS_LIBTORCH")) and bool(t.get("USE_LIBTORCH", True)),
        has_onnx=bool(t.get("HAS_ONNXRUNTIME")) and bool(t.get("USE_ONNXRUNTIME", True)),
        has_anira=bool(t.get("HAS_ANIRA")),
    )


def infer_build_features(build_dir: Path, target: str) -> BuildFeatures:
    info_path = build_dir / "nab_build_info.json"
    if info_path.exists():
        try:
            return build_features_from_info(load_json(info_path), target)
        except (KeyError, RuntimeError, json.JSONDecodeError, OSError):
            pass  # fall back to flags.make parsing
    flags_make = build_dir / "CMakeFiles" / f"{target}.dir" / "flags.make"
    return parse_flags_make(flags_make)


def enabled_sizes(cfg: dict) -> list[str]:
    sizes_cfg = cfg.get("model_sizes", {})
    ordered = ["small", "medium", "large"]
    return [s for s in ordered if sizes_cfg.get(s, True)]


def enabled_isolated_backends(cfg: dict, features: BuildFeatures) -> list[str]:
    be = cfg.get("backends", {})
    backends = []
    if be.get("BNNSGraph", True):
        backends.append("BNNSGraph")
    if be.get(features.rtneural_backend, True):
        backends.append(features.rtneural_backend)
    if features.has_libtorch and be.get("Direct_LibTorch", True):
        backends.append("Direct_LibTorch")
    if features.has_onnx and be.get("Direct_ONNX", True):
        backends.append("Direct_ONNX")
    return backends


def enabled_contention_backends(cfg: dict, features: BuildFeatures) -> list[str]:
    be = cfg.get("backends", {})
    backends = []
    if be.get("BNNSGraph", True):
        backends.append("BNNSGraph")
    if be.get(features.rtneural_backend, True):
        backends.append(features.rtneural_backend)
    if features.has_anira and features.has_libtorch and be.get("Anira_LibTorch", True):
        backends.append("Anira_LibTorch")
    if features.has_anira and features.has_onnx and be.get("Anira_ONNX", True):
        backends.append("Anira_ONNX")
    return backends


def fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if hrs:
        parts.append(f"{hrs}h")
    if mins or hrs:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def contention_per_config_seconds(sample_rate: float, buffer_size: int, cfg: dict) -> float:
    ct = cfg["contention"]
    warmup = max(
        ct["warmup_callbacks"] * buffer_size / sample_rate,
        float(ct["warmup_min_seconds"]),
    )
    return FIXED_CONTENTION_OVERHEAD_SECONDS + warmup + float(ct["measure_seconds"])


def compute_contention_runtime(cfg: dict, features: BuildFeatures) -> dict[str, RuntimeSummary]:
    sample_rate = float(cfg["sample_rate"])
    models = ["LSTM", "TCN", "WaveNet"]
    sizes = enabled_sizes(cfg)
    backends = enabled_contention_backends(cfg, features)
    ct = cfg["contention"]
    reps = int(ct["num_reps"])

    dim_a_seconds = 0.0
    dim_a_configs = 0
    for _size in sizes:
        for _model in models:
            for _backend in backends:
                for buffer_size in ct["buffer_sizes"]:
                    for _level in ct["contention_levels"]:
                        for _rep in range(reps):
                            dim_a_configs += 1
                            dim_a_seconds += contention_per_config_seconds(
                                sample_rate, int(buffer_size), cfg
                            )

    dim_b_buffer = 128
    dim_b_per = contention_per_config_seconds(sample_rate, dim_b_buffer, cfg)
    dim_b_configs = len(sizes) * len(models) * len(backends) * len(ct["instance_counts"]) * reps
    dim_b_seconds = dim_b_configs * dim_b_per

    dim_c_buffer = 128
    dim_c_per = contention_per_config_seconds(sample_rate, dim_c_buffer, cfg)
    dim_c_configs = (
        len(sizes) * len(models) * len(backends) * len(ct.get("neural_track_depths", [])) * reps
    )
    dim_c_seconds = dim_c_configs * dim_c_per

    total_configs = dim_a_configs + dim_b_configs + dim_c_configs
    total_seconds = dim_a_seconds + dim_b_seconds + dim_c_seconds

    return {
        "dim_a": RuntimeSummary(dim_a_configs, dim_a_seconds),
        "dim_b": RuntimeSummary(dim_b_configs, dim_b_seconds),
        "dim_c": RuntimeSummary(dim_c_configs, dim_c_seconds),
        "total": RuntimeSummary(total_configs, total_seconds),
    }


def compute_isolated_counts(cfg: dict, features: BuildFeatures) -> dict[str, int]:
    models = 3
    sizes = len(enabled_sizes(cfg))
    backends = len(enabled_isolated_backends(cfg, features))
    iso = cfg["isolated"]
    num_bufs = len(iso["buffer_sizes"])
    reps = int(iso["num_reps"])

    throughput_runs = models * sizes * backends
    callback_reps = throughput_runs * num_bufs * reps
    warmup_probe_groups = throughput_runs * num_bufs

    return {
        "throughput_runs": throughput_runs,
        "callback_reps": callback_reps,
        "warmup_probe_groups": warmup_probe_groups,
        "active_backends": backends,
        "active_sizes": sizes,
    }


def load_isolated_runtime_from_csv(
    csv_path: Path, sample_rate: float, cfg: dict
) -> tuple[float, float]:
    """
    Returns:
    - estimated total isolated seconds including warmup+probe approximation
    - measured-only seconds from CSV rows
    """
    throughput_seconds = 0.0
    callback_seconds = 0.0
    rep1_mean_by_key: dict[tuple[str, str, str, int], float] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        mode = row["mode"]
        if mode == "throughput":
            audio_seconds = float(row["total_samples"]) / sample_rate
            throughput_seconds += audio_seconds * float(row["rtf"])
        elif mode == "callback":
            mean_ns = float(row["mean_ns"])
            total_samples = int(row["total_samples"])
            callback_seconds += (mean_ns * total_samples) / 1e9

            key = (
                row["backend"],
                row["model"],
                row["model_size"],
                int(row["buffer_size"]),
            )
            if int(row["rep"]) == 1 and key not in rep1_mean_by_key:
                rep1_mean_by_key[key] = mean_ns

    warmup_probe_seconds = 0.0
    extra_iters = int(cfg["isolated"]["warmup_iterations"]) + 10
    for mean_ns in rep1_mean_by_key.values():
        warmup_probe_seconds += (mean_ns * extra_iters) / 1e9

    return (
        throughput_seconds + callback_seconds + warmup_probe_seconds,
        throughput_seconds + callback_seconds,
    )


def print_contention_report(cfg: dict, features: BuildFeatures) -> None:
    backends = enabled_contention_backends(cfg, features)
    report = compute_contention_runtime(cfg, features)
    ct = cfg["contention"]

    print("Contention")
    print(f"  Active backends: {', '.join(backends) if backends else '(none)'}")
    print(f"  Sample rate: {cfg['sample_rate']:.0f} Hz")
    print(
        "  Per-config fixed overhead: "
        f"{FIXED_CONTENTION_OVERHEAD_SECONDS:.1f}s "
        "(6 x 200ms settle/start-stop waits)"
    )
    print(
        "  Warmup formula: "
        f"max({ct['warmup_callbacks']} * buffer / sample_rate, {ct['warmup_min_seconds']}s)"
    )
    print(f"  Measurement time per config: {ct['measure_seconds']}s")
    print(
        f"  Dimension A: {report['dim_a'].configs} configs, {fmt_duration(report['dim_a'].seconds)}"
    )
    print(
        f"  Dimension B: {report['dim_b'].configs} configs, {fmt_duration(report['dim_b'].seconds)}"
    )
    print(
        f"  Dimension C: {report['dim_c'].configs} configs, {fmt_duration(report['dim_c'].seconds)}"
    )
    print(f"  Total: {report['total'].configs} configs, {fmt_duration(report['total'].seconds)}")
    if ct.get("use_system_au", False):
        print(
            "  Note: excludes one-time AU scan overhead in Dimension A; "
            "that cost is not deterministic."
        )


def print_isolated_report(cfg: dict, features: BuildFeatures, isolated_csv: Path | None) -> None:
    backends = enabled_isolated_backends(cfg, features)
    counts = compute_isolated_counts(cfg, features)
    iso = cfg["isolated"]

    print("Isolated")
    print(f"  Active backends: {', '.join(backends) if backends else '(none)'}")
    print(f"  Throughput runs (Mode A): {counts['throughput_runs']}")
    print(f"  Callback reps (Mode B): {counts['callback_reps']}")
    print(f"  Warmup+probe groups: {counts['warmup_probe_groups']}")
    print(
        "  Exact runtime cannot be derived from config alone because Mode B "
        "adapts iteration count from measured backend speed."
    )
    nominal_measure_budget = counts["callback_reps"] * float(iso["target_measure_seconds"])
    print(
        "  Nominal measurement budget only "
        f"(callback reps x target_measure_seconds): {fmt_duration(nominal_measure_budget)}"
    )
    print(
        "  Throughput rows, warmup iterations, probe iterations, and min_iterations "
        "can make the real isolated runtime materially larger."
    )

    if isolated_csv:
        total_est, measured_only = load_isolated_runtime_from_csv(
            isolated_csv, float(cfg["sample_rate"]), cfg
        )
        print(f"  Estimated isolated runtime from CSV: {fmt_duration(total_est)}")
        print(f"    Measured rows only: {fmt_duration(measured_only)}")
        print("    Warmup+probe added using rep-1 mean_ns per (backend, model, size, buffer).")


def print_full_run_estimate(
    cfg: dict,
    build_dir: Path,
    cooldown: int,
    isolated_csv: Path | None,
) -> None:
    """Estimate wall time for the interleaved runner:
    Eigen isolated → cool → XSIMD isolated → cool → XSIMD contention.
    """
    eigen_features = infer_build_features(build_dir, "nab_engine")
    xsimd_features = infer_build_features(build_dir, "nab_engine_xsimd")

    print("Full interleaved run estimate")
    print(f"  Cooldown between phases: {cooldown}s")
    print()

    # Phase 1: Eigen isolated
    eigen_iso_counts = compute_isolated_counts(cfg, eigen_features)
    eigen_iso_budget = eigen_iso_counts["callback_reps"] * float(
        cfg["isolated"]["target_measure_seconds"]
    )

    # Phase 2: XSIMD isolated
    xsimd_iso_counts = compute_isolated_counts(cfg, xsimd_features)
    xsimd_iso_budget = xsimd_iso_counts["callback_reps"] * float(
        cfg["isolated"]["target_measure_seconds"]
    )

    # Phase 3: XSIMD contention only
    xsimd_contention = compute_contention_runtime(cfg, xsimd_features)

    # If we have an isolated CSV, use it for a tighter estimate
    eigen_iso_est: float | None = None
    xsimd_iso_est: float | None = None
    if isolated_csv and isolated_csv.exists():
        total_est, _ = load_isolated_runtime_from_csv(isolated_csv, float(cfg["sample_rate"]), cfg)
        # Rough split: attribute proportionally by backend count
        eigen_n = len(enabled_isolated_backends(cfg, eigen_features))
        xsimd_n = len(enabled_isolated_backends(cfg, xsimd_features))
        total_n = eigen_n + xsimd_n
        if total_n > 0:
            eigen_iso_est = total_est * eigen_n / total_n
            xsimd_iso_est = total_est * xsimd_n / total_n

    print("  Phase 1: Eigen isolated (nab_engine)")
    eigen_backends = enabled_isolated_backends(cfg, eigen_features)
    print(f"    Backends: {', '.join(eigen_backends)}")
    if eigen_iso_est is not None:
        print(f"    Estimated from CSV: {fmt_duration(eigen_iso_est)}")
    else:
        print(f"    Nominal budget: {fmt_duration(eigen_iso_budget)}  (actual varies)")
    print()

    print("  Phase 2: XSIMD isolated (nab_engine_xsimd)")
    xsimd_backends = enabled_isolated_backends(cfg, xsimd_features)
    print(f"    Backends: {', '.join(xsimd_backends)}")
    if xsimd_iso_est is not None:
        print(f"    Estimated from CSV: {fmt_duration(xsimd_iso_est)}")
    else:
        print(f"    Nominal budget: {fmt_duration(xsimd_iso_budget)}  (actual varies)")
    print()

    print("  Phase 3: XSIMD contention (nab_engine_xsimd)")
    xsimd_ct_backends = enabled_contention_backends(cfg, xsimd_features)
    print(f"    Backends: {', '.join(xsimd_ct_backends)}")
    print(
        f"    Dim A: {xsimd_contention['dim_a'].configs} configs, "
        f"{fmt_duration(xsimd_contention['dim_a'].seconds)}"
    )
    print(
        f"    Dim B: {xsimd_contention['dim_b'].configs} configs, "
        f"{fmt_duration(xsimd_contention['dim_b'].seconds)}"
    )
    print(
        f"    Dim C: {xsimd_contention['dim_c'].configs} configs, "
        f"{fmt_duration(xsimd_contention['dim_c'].seconds)}"
    )
    print(f"    Subtotal: {fmt_duration(xsimd_contention['total'].seconds)}")
    print()

    # Total: 3 phases, 2 cooldowns between them
    total_cooldown = cooldown * 2
    contention_seconds = xsimd_contention["total"].seconds

    if eigen_iso_est is not None and xsimd_iso_est is not None:
        total = eigen_iso_est + xsimd_iso_est + contention_seconds + total_cooldown
        print(f"  Total (from CSV): {fmt_duration(total)}")
        print(
            f"    = isolated Eigen {fmt_duration(eigen_iso_est)}"
            f" + cool {cooldown}s"
            f" + isolated XSIMD {fmt_duration(xsimd_iso_est)}"
            f" + cool {cooldown}s"
            f" + contention {fmt_duration(contention_seconds)}"
        )
    else:
        budget_total = eigen_iso_budget + xsimd_iso_budget + contention_seconds + total_cooldown
        print(f"  Total (nominal): {fmt_duration(budget_total)}")
        print("    Isolated budgets are nominal — actual depends on hardware speed.")
        print(f"    Contention is exact: {fmt_duration(contention_seconds)}")
        print(f"    Cooldowns: 2 x {cooldown}s = {fmt_duration(total_cooldown)}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the config JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=(
            "Build directory used to infer compile-time backend availability "
            f"(default: {DEFAULT_BUILD_DIR})"
        ),
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=["nab_engine", "nab_engine_xsimd"],
        help="Benchmark target to model. Determines the RTNeural variant and compiled backends.",
    )
    parser.add_argument(
        "--isolated-csv",
        type=Path,
        default=None,
        help="Optional isolated CSV from a prior run to estimate actual isolated runtime.",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=120,
        help="Cooldown seconds between phases in the interleaved runner (default: 120).",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help=(
            "Estimate total wall time for the full interleaved run "
            "(Eigen isolated + XSIMD isolated + XSIMD contention + cooldowns)."
        ),
    )


def run(args: argparse.Namespace) -> int:
    cfg = load_json(args.config)
    features = infer_build_features(args.build_dir, args.target)

    print(f"Config:  {args.config}")
    print(f"Build:   {args.build_dir}")
    print(f"Target:  {args.target}")
    print(f"RTNeural backend in target: {features.rtneural_backend}")
    print(
        "Compiled features: "
        f"anira={'yes' if features.has_anira else 'no'}, "
        f"libtorch={'yes' if features.has_libtorch else 'no'}, "
        f"onnx={'yes' if features.has_onnx else 'no'}"
    )
    print()

    print_contention_report(cfg, features)
    print()
    print_isolated_report(cfg, features, args.isolated_csv)

    if args.full_run:
        print()
        print("=" * 60)
        print_full_run_estimate(cfg, args.build_dir, args.cooldown, args.isolated_csv)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
