# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Run the full benchmark suite with thermal fairness.

Orchestrates the benchmark protocol:
  - CPU warmup burn before each phase (reaches thermal steady state)
  - Cooldown between phases (resets to a consistent baseline)

Phases:
  1. Eigen isolated       (nab-engine)
  2. XSIMD isolated       (nab-engine-xsimd)
  3. XSIMD contention     (nab-engine-xsimd; Eigen contention is skipped)

Merges the isolated CSVs and runs analysis + figures automatically.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .config import (
    config_sha256,
    default_base_config,
    resolve_config,
    validate_config,
    write_resolved,
)
from .machine import machine_info


class PhaseError(RuntimeError):
    """Raised when a benchmark phase binary exits non-zero."""


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parent),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or ``None`` when the path is absent/not a file."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Logger:
    """Timestamped logger that tees to the console and a log file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} | {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def stream(self, command: list[str]) -> int:
        """Run a command, streaming combined stdout/stderr to console and log."""
        with self.log_path.open("a", encoding="utf-8") as f:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
            return proc.wait()


def _ncpu() -> int:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return os.cpu_count() or 1


def cpu_warmup(duration: int, logger: Logger) -> None:
    """Saturate all cores for ``duration`` seconds to reach thermal steady state."""
    if duration <= 0:
        return

    ncpu = _ncpu()
    logger.log(f"CPU warmup: burning {ncpu} cores for {duration}s to reach thermal steady state...")

    procs: list[subprocess.Popen] = []
    devnull = open(os.devnull, "w")  # noqa: SIM115 - kept open for the burn duration
    try:
        for _ in range(ncpu):
            procs.append(subprocess.Popen(["yes"], stdout=devnull, stderr=devnull))
        time.sleep(duration)
    finally:
        for proc in procs:
            proc.kill()
        for proc in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        devnull.close()

    logger.log("CPU warmup complete — starting benchmark immediately.")


def merge_isolated_csvs(eigen_csv: Path, xsimd_csv: Path, out_csv: Path, logger: Logger) -> None:
    """Merge engine CSVs without duplicating backends present in both.

    The two executables differ only in their RTNeural implementation. Shared
    backends run in both binaries, so appending the second file wholesale would
    double-weight them during aggregation.
    """
    with eigen_csv.open("r", encoding="utf-8") as f:
        eigen_lines = f.read().splitlines()
    merged = list(eigen_lines)
    if xsimd_csv.exists():
        with xsimd_csv.open("r", encoding="utf-8") as f:
            xsimd_lines = f.read().splitlines()
        if eigen_lines and xsimd_lines:
            eigen_rows = list(csv.reader(eigen_lines))
            xsimd_rows = list(csv.reader(xsimd_lines))
            if eigen_rows[0] != xsimd_rows[0]:
                raise PhaseError("isolated CSV headers differ between engine variants")
            try:
                backend_idx = eigen_rows[0].index("backend")
            except ValueError as exc:
                raise PhaseError("isolated CSV has no 'backend' column") from exc
            eigen_backends = {row[backend_idx] for row in eigen_rows[1:] if len(row) > backend_idx}
            xsimd_only = {
                row[backend_idx]
                for row in xsimd_rows[1:]
                if len(row) > backend_idx and row[backend_idx] not in eigen_backends
            }
            merged.extend(
                raw
                for raw, row in zip(xsimd_lines[1:], xsimd_rows[1:], strict=True)
                if len(row) > backend_idx and row[backend_idx] in xsimd_only
            )
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("\n".join(merged) + "\n")
    logger.log(f"  {out_csv.name}: {max(0, len(merged) - 1)} data rows")


@contextlib.contextmanager
def _power_assertion(logger: Logger):
    """Keep macOS awake for the complete benchmark run, if available."""
    caffeinate = shutil.which("caffeinate") if sys.platform == "darwin" else None
    if caffeinate is None:
        yield False
        return

    proc = subprocess.Popen(
        [caffeinate, "-dims", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.log("macOS power assertion active (caffeinate -dims).")
    try:
        yield True
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def _run_phase(logger: Logger, label: str, command: list[str]) -> None:
    logger.log(label)
    code = logger.stream(command)
    if code != 0:
        raise PhaseError(f"{label} exited with code {code}: {' '.join(command)}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--experiment", help="Experiment name (experiments/<NAME>/config.json).")
    src.add_argument("--config", help="Path to a config JSON (layered on base.json).")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override a config value (JSON-parsed, repeatable).",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: runs/<ts>).")
    parser.add_argument(
        "--mode",
        choices=["isolated", "contention", "all"],
        default="all",
        help="Which phase groups to run (default: all).",
    )
    parser.add_argument("--warmup", type=int, default=45, help="CPU warmup seconds (default: 45).")
    parser.add_argument(
        "--cooldown", type=int, default=120, help="Cooldown seconds between phases (default: 120)."
    )
    parser.add_argument(
        "--allow-any-device",
        action="store_true",
        help="Pass --allow-any-device to the engine binaries.",
    )
    parser.add_argument("--engine-dir", default=None, help="Directory with engine binaries.")


def run(args: argparse.Namespace) -> int:
    from . import analyze, plot_figures
    from .config import repo_root

    if args.warmup < 0 or args.cooldown < 0:
        print("ERROR: --warmup and --cooldown must be non-negative", file=sys.stderr)
        return 2

    # --- Resolve + validate config ---
    preset = args.experiment or args.config
    config = resolve_config(base=default_base_config(), preset=preset, overrides=args.overrides)
    validate_config(config)

    # Ensure the manifest path is absolute (load_config already resolves it).
    mm = config.get("models_manifest")
    if isinstance(mm, str) and mm and not os.path.isabs(mm):
        config["models_manifest"] = str(Path(mm).resolve())
    mm = config.get("models_manifest")

    output_dir = Path(args.output_dir) if args.output_dir else repo_root() / "runs" / _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_dir = Path(args.engine_dir) if args.engine_dir else repo_root() / "build"
    eigen_bin = engine_dir / "nab-engine"
    xsimd_bin = engine_dir / "nab-engine-xsimd"

    # --- Write resolved config + run manifest ---
    resolved_path = write_resolved(config, output_dir / "resolved.json")

    run_manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "nab_version": __version__,
        "git_rev": _git_head(),
        "config_sha256": config_sha256(config),
        "resolved_config": str(resolved_path),
        "models_manifest": str(mm) if mm else None,
        "models_manifest_sha256": _sha256_file(mm) if mm else None,
        "dependency_pins_sha256": _sha256_file(repo_root() / "cmake" / "Versions.cmake"),
        "engine_sha256": {
            "nab-engine": _sha256_file(eigen_bin),
            "nab-engine-xsimd": _sha256_file(xsimd_bin),
        },
        "mode": args.mode,
        "warmup_seconds": args.warmup,
        "cooldown_seconds": args.cooldown,
        "machine": machine_info(),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    logger = Logger(output_dir / "benchmark_log.txt")

    extra_args = ["--allow-any-device"] if args.allow_any_device else []
    config_args = ["--config", str(resolved_path)]

    isolated_csv = output_dir / "isolated.csv"
    isolated_xsimd_csv = output_dir / "isolated_xsimd.csv"
    contention_csv = output_dir / "contention.csv"
    merged_csv = output_dir / "isolated_merged.csv"

    run_isolated = args.mode in ("isolated", "all")
    run_contention = args.mode in ("contention", "all")

    logger.log("==========================================")
    logger.log("Benchmark Suite")
    logger.log(f"  Eigen:    {eigen_bin}")
    logger.log(f"  XSIMD:    {xsimd_bin}")
    logger.log(f"  Output:   {output_dir}")
    logger.log(f"  Mode:     {args.mode}")
    logger.log(f"  Cooldown: {args.cooldown}s")
    logger.log(f"  Warmup:   {args.warmup}s")
    logger.log("==========================================")

    xsimd_available = xsimd_bin.exists() and os.access(xsimd_bin, os.X_OK)

    try:
        with _power_assertion(logger):
            if run_isolated:
                # Phase 1: Eigen isolated (required binary for this phase group)
                if not (eigen_bin.exists() and os.access(eigen_bin, os.X_OK)):
                    raise PhaseError(f"Eigen binary not found or not executable: {eigen_bin}")
                cpu_warmup(args.warmup, logger)
                _run_phase(
                    logger,
                    "PHASE 1/3: Eigen isolated",
                    [str(eigen_bin), "--mode", "isolated", "--output", str(isolated_csv)]
                    + config_args
                    + extra_args,
                )

                logger.log(f"Cooling down {args.cooldown}s...")
                time.sleep(args.cooldown)

                # Phase 2: XSIMD isolated
                if xsimd_available:
                    cpu_warmup(args.warmup, logger)
                    _run_phase(
                        logger,
                        "PHASE 2/3: XSIMD isolated",
                        [str(xsimd_bin), "--mode", "isolated", "--output", str(isolated_xsimd_csv)]
                        + config_args
                        + extra_args,
                    )
                else:
                    logger.log(f"Skipping XSIMD isolated: binary not found ({xsimd_bin}).")

                # Merge isolated CSVs
                if isolated_csv.exists():
                    logger.log("Merging isolated CSVs...")
                    merge_isolated_csvs(isolated_csv, isolated_xsimd_csv, merged_csv, logger)

            if run_contention:
                if xsimd_available:
                    if run_isolated:
                        logger.log(f"Cooling down {args.cooldown}s...")
                        time.sleep(args.cooldown)
                    cpu_warmup(args.warmup, logger)
                    _run_phase(
                        logger,
                        "PHASE 3/3: XSIMD contention",
                        [str(xsimd_bin), "--mode", "contention", "--output-dir", str(output_dir)]
                        + config_args
                        + extra_args,
                    )
                else:
                    logger.log(f"Skipping XSIMD contention: binary not found ({xsimd_bin}).")
    except PhaseError as exc:
        logger.log(f"ERROR: {exc}")
        return 1

    # --- Analysis (best-effort, non-fatal) ---
    analysis_isolated = merged_csv if merged_csv.exists() else isolated_csv
    logger.log("Running analysis...")
    try:
        analyze.run(
            argparse.Namespace(
                isolated=str(analysis_isolated),
                contention=str(contention_csv),
                output=str(output_dir / "analysis_xsimd.csv"),
            )
        )
    except Exception as exc:  # noqa: BLE001 - analysis is best-effort
        logger.log(f"Analysis failed (non-fatal): {exc}")

    try:
        plot_figures.run(
            argparse.Namespace(
                isolated=str(analysis_isolated),
                contention=str(contention_csv),
                output_dir=str(output_dir / "figures"),
            )
        )
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        logger.log(f"Figures failed (non-fatal): {exc}")

    logger.log("==========================================")
    logger.log("ALL DONE")
    logger.log(f"  Output:     {output_dir}")
    logger.log(f"  Merged iso: {merged_csv}")
    logger.log(f"  Contention: {contention_csv}")
    logger.log(f"  Figures:    {output_dir / 'figures'}")
    logger.log("==========================================")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark suite", allow_abbrev=False)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
