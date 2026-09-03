# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Estimate benchmark runtime from a resolved configuration.

What this can compute from config:
- The contention protocol time for successful first attempts: configured
  warmup, measurement windows, and fixed waits.

What it cannot do from config alone:
- Compute the exact isolated-mode wall time. Isolated mode adapts its iteration
  count from measured backend speed at runtime, so the true duration depends on
  the machine, binary, backend, model, and buffer size.
- Predict session construction, Audio Unit scan/device delays, or transient
  contention retries.

Optional:
- If you pass an existing isolated CSV from the same machine/binary/config, the
  script can estimate isolated runtime from those measured rows.

Compile-time backend availability is read from CMake's generated ``flags.make``.
Before CMake has been configured, the estimator clearly labels and uses the
project's default full-build feature set so it remains useful as a planning
command.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import default_base_config, repo_root, resolve_config

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
    source: str = "CMake flags"


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
        source=str(path),
    )


def infer_build_features(build_dir: Path, target: str) -> BuildFeatures:
    flags_make = build_dir / "CMakeFiles" / f"{target}.dir" / "flags.make"
    if flags_make.exists():
        return parse_flags_make(flags_make)

    rt_backend = "RTNeural_XSIMD" if target == "nab_engine_xsimd" else "RTNeural_Eigen"
    print(
        f"WARNING: no configured build metadata found for {target} at {flags_make}; "
        "assuming the default full build (LibTorch, ONNX Runtime, and anira enabled).",
        file=sys.stderr,
    )
    return BuildFeatures(
        rtneural_backend=rt_backend,
        has_libtorch=True,
        has_onnx=True,
        has_anira=True,
        source="unconfigured default full build (assumed)",
    )


def enabled_model_specs(cfg: dict) -> list[tuple[str, str]]:
    """Return manifest/config model entries selected by architecture and size."""
    model_entries: list[tuple[str, str]] = []
    manifest_value = cfg.get("models_manifest")
    if manifest_value:
        manifest_path = Path(manifest_value)
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            model_entries = [
                (str(item["arch"]), str(item.get("size", "")))
                for item in manifest.get("models", [])
            ]

    if not model_entries:
        for arch, sizes in cfg.get("models", {}).items():
            if arch.startswith("_") or not isinstance(sizes, dict):
                continue
            for size in sizes:
                if not size.startswith("_"):
                    model_entries.append((arch, size))

    model_selector = cfg.get("model_types", {})
    size_selector = cfg.get("model_sizes", {})
    return [
        (arch, size)
        for arch, size in model_entries
        if model_selector.get(arch, True) and size_selector.get(size, True)
    ]


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
    model_specs = enabled_model_specs(cfg)
    backends = enabled_contention_backends(cfg, features)
    ct = cfg["contention"]
    reps = int(ct["num_reps"])

    dim_a_seconds = 0.0
    dim_a_configs = 0
    for _model in model_specs:
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
    dim_b_configs = len(model_specs) * len(backends) * len(ct["instance_counts"]) * reps
    dim_b_seconds = dim_b_configs * dim_b_per

    dim_c_buffer = 128
    dim_c_per = contention_per_config_seconds(sample_rate, dim_c_buffer, cfg)
    dim_c_configs = len(model_specs) * len(backends) * len(ct.get("neural_track_depths", [])) * reps
    dim_c_seconds = dim_c_configs * dim_c_per

    report = {
        "dim_a": RuntimeSummary(dim_a_configs, dim_a_seconds),
        "dim_b": RuntimeSummary(dim_b_configs, dim_b_seconds),
        "dim_c": RuntimeSummary(dim_c_configs, dim_c_seconds),
    }

    for scenario in cfg.get("custom_scenarios", []):
        scenario_configs = 0
        scenario_seconds = 0.0
        for buffer_size in scenario["buffer_sizes"]:
            count = len(model_specs) * len(backends) * len(scenario["sweep"]["values"]) * reps
            scenario_configs += count
            scenario_seconds += count * contention_per_config_seconds(
                sample_rate, int(buffer_size), cfg
            )
        report[str(scenario["id"])] = RuntimeSummary(scenario_configs, scenario_seconds)

    report["total"] = RuntimeSummary(
        sum(summary.configs for summary in report.values()),
        sum(summary.seconds for summary in report.values()),
    )
    return report


def compute_isolated_counts(cfg: dict, features: BuildFeatures) -> dict[str, int]:
    models = len(enabled_model_specs(cfg))
    backends = len(enabled_isolated_backends(cfg, features))
    iso = cfg["isolated"]
    num_bufs = len(iso["buffer_sizes"])
    reps = int(iso["num_reps"])

    throughput_runs = models * backends
    callback_reps = throughput_runs * num_bufs * reps
    warmup_probe_groups = throughput_runs * num_bufs

    return {
        "throughput_runs": throughput_runs,
        "callback_reps": callback_reps,
        "warmup_probe_groups": warmup_probe_groups,
        "active_backends": backends,
        "active_models": models,
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
    for scenario_id, summary in report.items():
        if scenario_id == "total":
            continue
        label = {"dim_a": "Dimension A", "dim_b": "Dimension B", "dim_c": "Dimension C"}.get(
            scenario_id, f"Custom scenario {scenario_id}"
        )
        print(f"  {label}: {summary.configs} configs, {fmt_duration(summary.seconds)}")
    print(f"  Total: {report['total'].configs} configs, {fmt_duration(report['total'].seconds)}")
    print(
        "  Note: total is the successful first-attempt protocol time; setup delays and retries vary."
    )
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
    print(f"  Active manifest/config model entries: {counts['active_models']}")
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
    warmup: int,
    cooldown: int,
    isolated_csv: Path | None,
) -> None:
    """Estimate wall time for the interleaved runner:
    Eigen isolated → cool → XSIMD isolated → cool → XSIMD contention.
    """
    eigen_features = infer_build_features(build_dir, "nab_engine")
    xsimd_features = infer_build_features(build_dir, "nab_engine_xsimd")

    print("Full interleaved run estimate")
    print(f"  CPU warmup before each phase: {warmup}s")
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
    print(f"    Feature source: {eigen_features.source}")
    eigen_backends = enabled_isolated_backends(cfg, eigen_features)
    print(f"    Backends: {', '.join(eigen_backends)}")
    if eigen_iso_est is not None:
        print(f"    Estimated from CSV: {fmt_duration(eigen_iso_est)}")
    else:
        print(f"    Nominal budget: {fmt_duration(eigen_iso_budget)}  (actual varies)")
    print()

    print("  Phase 2: XSIMD isolated (nab_engine_xsimd)")
    print(f"    Feature source: {xsimd_features.source}")
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
    for scenario_id, summary in xsimd_contention.items():
        if scenario_id == "total":
            continue
        print(f"    {scenario_id}: {summary.configs} configs, {fmt_duration(summary.seconds)}")
    print(f"    Subtotal: {fmt_duration(xsimd_contention['total'].seconds)}")
    print()

    # Total: 3 phase warmups and 2 cooldowns between the phases.
    total_warmup = max(0, warmup) * 3
    total_cooldown = max(0, cooldown) * 2
    contention_seconds = xsimd_contention["total"].seconds

    if eigen_iso_est is not None and xsimd_iso_est is not None:
        total = eigen_iso_est + xsimd_iso_est + contention_seconds + total_warmup + total_cooldown
        print(f"  Total (from CSV): {fmt_duration(total)}")
        print(
            f"    = warmups {fmt_duration(total_warmup)}"
            f" + isolated Eigen {fmt_duration(eigen_iso_est)}"
            f" + cool {cooldown}s"
            f" + isolated XSIMD {fmt_duration(xsimd_iso_est)}"
            f" + cool {cooldown}s"
            f" + contention {fmt_duration(contention_seconds)}"
        )
    else:
        budget_total = (
            eigen_iso_budget + xsimd_iso_budget + contention_seconds + total_warmup + total_cooldown
        )
        print(f"  Total (nominal): {fmt_duration(budget_total)}")
        print("    Isolated budgets are nominal — actual depends on hardware speed.")
        print(
            "    Contention first-attempt protocol: "
            f"{fmt_duration(contention_seconds)} (excludes setup delays and retries)"
        )
        print(f"    CPU warmups: 3 x {warmup}s = {fmt_duration(total_warmup)}")
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
        "--warmup",
        type=int,
        default=45,
        help="CPU warmup seconds before each phase in the interleaved runner (default: 45).",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=120,
        help="Cooldown seconds between phases in the interleaved runner (default: 120).",
    )
    parser.add_argument(
        "--full-run",
        dest="full_run",
        action="store_true",
        default=True,
        help=(
            "Estimate the full interleaved run (default): "
            "(Eigen isolated + XSIMD isolated + XSIMD contention + cooldowns)."
        ),
    )
    parser.add_argument(
        "--single-target",
        dest="full_run",
        action="store_false",
        help="Estimate only --target, for direct engine invocations.",
    )


def run(args: argparse.Namespace) -> int:
    if args.warmup < 0 or args.cooldown < 0:
        print("ERROR: --warmup and --cooldown must be non-negative", file=sys.stderr)
        return 2
    cfg = resolve_config(preset=args.config)

    print(f"Config:  {args.config}")
    print(f"Build:   {args.build_dir}")

    if args.full_run:
        print_full_run_estimate(cfg, args.build_dir, args.warmup, args.cooldown, args.isolated_csv)
    else:
        features = infer_build_features(args.build_dir, args.target)
        print(f"Target:  {args.target}")
        print(f"Feature source: {features.source}")
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

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
