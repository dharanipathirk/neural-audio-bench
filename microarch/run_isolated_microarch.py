#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Run isolated-only microarchitecture side analysis without mixing outputs into
the main benchmark results directory.

Each run isolates exactly one backend + one model + one size by generating a
temporary benchmark config and invoking nab-engine with --mode isolated.
While the benchmark is running, decode_amx.py attaches via lldb and writes a
separate AMX JSON file for that combination.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "benchmark_config.json"
DEFAULT_BENCH = REPO_ROOT / "build" / "nab-engine"
DEFAULT_MODELS = REPO_ROOT / "models"
DECODE_AMX = REPO_ROOT / "analysis" / "decode_amx.py"

ALL_BACKENDS = [
    "BNNSGraph",
    "RTNeural_Eigen",
    "RTNeural_XSIMD",
    "Direct_LibTorch",
    "Direct_ONNX",
    "Anira_LibTorch",
    "Anira_ONNX",
]
ALL_MODELS = ["LSTM", "TCN", "WaveNet"]
ALL_SIZES = ["small", "medium", "large"]


@dataclass(frozen=True)
class RunSpec:
    backend: str
    model: str
    size: str

    @property
    def stem(self) -> str:
        return f"{self.backend}__{self.model}__{self.size}"


def infer_rtneural_backend(bench_path: Path) -> str:
    return "RTNeural_XSIMD" if "xsimd" in bench_path.name.lower() else "RTNeural_Eigen"


def parse_csv_list(raw: str | None, allowed: list[str]) -> list[str]:
    if not raw:
        return []
    out = []
    allowed_set = set(allowed)
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token not in allowed_set:
            raise ValueError(f"Unsupported value '{token}'. Allowed: {', '.join(allowed)}")
        out.append(token)
    return out


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def make_temp_config(
    base_cfg: dict,
    spec: RunSpec,
    sample_rate: float,
    buffer_size: int,
    target_seconds: float,
    min_iterations: int,
    throughput_seconds: float,
) -> dict:
    cfg = deepcopy(base_cfg)

    cfg["sample_rate"] = sample_rate

    iso = cfg.setdefault("isolated", {})
    iso["buffer_sizes"] = [buffer_size]
    iso["warmup_iterations"] = max(int(iso.get("warmup_iterations", 100)), 100)
    iso["target_measure_seconds"] = target_seconds
    iso["min_iterations"] = min_iterations
    iso["num_reps"] = 1
    iso["throughput_seconds"] = throughput_seconds

    # Keep contention inert for these runs.
    ct = cfg.setdefault("contention", {})
    ct["num_reps"] = 1

    cfg["model_sizes"] = {
        "small": spec.size == "small",
        "medium": spec.size == "medium",
        "large": spec.size == "large",
    }

    cfg["model_types"] = {
        "lstm": spec.model == "LSTM",
        "tcn": spec.model == "TCN",
        "wavenet": spec.model == "WaveNet",
    }

    cfg["backends"] = {name: name == spec.backend for name in ALL_BACKENDS}
    return cfg


def run_decode_amx(pid: int, backend: str, json_out: Path, verbose: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(DECODE_AMX),
        "--pid",
        str(pid),
        "--backend",
        backend,
        "--json-output",
        str(json_out),
    ]
    if verbose:
        cmd.append("--verbose")
    return subprocess.run(cmd, text=True, capture_output=True)


def wait_for_process(proc: subprocess.Popen[str], timeout_s: float) -> int:
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated-only AMX side analysis in a separate results tree.")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help="Path to nab-engine binary")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base benchmark_config.json")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="Model directory override")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: microarch/results/<timestamp>)")
    parser.add_argument("--backends", type=str, default=None, help="Comma-separated backend list")
    parser.add_argument("--model-types", type=str, default=None, help="Comma-separated model list")
    parser.add_argument("--sizes", type=str, default="large", help="Comma-separated model sizes")
    parser.add_argument("--sample-rate", type=float, default=48000.0)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--target-seconds", type=float, default=6.0)
    parser.add_argument("--min-iterations", type=int, default=20000)
    parser.add_argument("--throughput-seconds", type=float, default=5.0)
    parser.add_argument("--attach-delay", type=float, default=3.0, help="Seconds to wait before attaching decode_amx")
    parser.add_argument("--post-timeout", type=float, default=120.0, help="Seconds to wait for benchmark completion after AMX scan")
    parser.add_argument("--verbose-amx", action="store_true", help="Pass --verbose to decode_amx.py")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and manifest without launching runs")
    args = parser.parse_args()

    bench = args.bench.resolve()
    config = args.config.resolve()
    models_dir = args.models.resolve()

    if not bench.exists():
        print(f"ERROR: benchmark binary not found: {bench}", file=sys.stderr)
        return 1
    if not os.access(bench, os.X_OK):
        print(f"ERROR: benchmark binary is not executable: {bench}", file=sys.stderr)
        return 1
    if not config.exists():
        print(f"ERROR: config not found: {config}", file=sys.stderr)
        return 1
    if not DECODE_AMX.exists():
        print(f"ERROR: decode_amx.py not found: {DECODE_AMX}", file=sys.stderr)
        return 1

    rt_backend = infer_rtneural_backend(bench)
    default_backends = ["BNNSGraph", rt_backend, "Direct_LibTorch", "Direct_ONNX"]

    try:
        backends = parse_csv_list(args.backends, ALL_BACKENDS) or default_backends
        models = parse_csv_list(args.model_types, ALL_MODELS) or ALL_MODELS
        sizes = parse_csv_list(args.sizes, ALL_SIZES) or ["large"]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir.resolve() if args.out_dir else (REPO_ROOT / "microarch" / "results" / ts))
    cfg_dir = out_dir / "configs"
    csv_dir = out_dir / "isolated_csv"
    log_dir = out_dir / "logs"
    amx_dir = out_dir / "amx_json"
    for path in (cfg_dir, csv_dir, log_dir, amx_dir):
        path.mkdir(parents=True, exist_ok=True)

    base_cfg = load_json(config)
    specs = [RunSpec(backend, model, size) for backend in backends for model in models for size in sizes]

    manifest: list[dict] = []
    print(f"Output directory: {out_dir}")
    print(f"Benchmark binary: {bench}")
    print(f"Models directory: {models_dir}")
    print(f"Run count: {len(specs)}")

    for idx, spec in enumerate(specs, start=1):
        print(f"[{idx}/{len(specs)}] {spec.backend} / {spec.model} / {spec.size}")

        run_cfg = make_temp_config(
            base_cfg,
            spec,
            sample_rate=args.sample_rate,
            buffer_size=args.buffer_size,
            target_seconds=args.target_seconds,
            min_iterations=args.min_iterations,
            throughput_seconds=args.throughput_seconds,
        )
        cfg_path = cfg_dir / f"{spec.stem}.json"
        csv_path = csv_dir / f"{spec.stem}.csv"
        log_path = log_dir / f"{spec.stem}.log"
        amx_path = amx_dir / f"{spec.stem}.json"
        write_json(cfg_path, run_cfg)

        entry = {
            "backend": spec.backend,
            "model": spec.model,
            "size": spec.size,
            "config": str(cfg_path),
            "csv": str(csv_path),
            "log": str(log_path),
            "amx_json": str(amx_path),
            "attach_delay_s": args.attach_delay,
        }

        if args.dry_run:
            entry["status"] = "dry_run"
            manifest.append(entry)
            continue

        with log_path.open("w", encoding="utf-8") as log_fh:
            cmd = [
                str(bench),
                "--mode",
                "isolated",
                "--models",
                str(models_dir),
                "--config",
                str(cfg_path),
                "--output",
                str(csv_path),
            ]
            log_fh.write("COMMAND: " + " ".join(cmd) + "\n")
            log_fh.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )

            time.sleep(args.attach_delay)

            decode_result = run_decode_amx(proc.pid, spec.backend, amx_path, args.verbose_amx)
            if decode_result.stdout:
                log_fh.write("\n=== decode_amx stdout ===\n")
                log_fh.write(decode_result.stdout)
            if decode_result.stderr:
                log_fh.write("\n=== decode_amx stderr ===\n")
                log_fh.write(decode_result.stderr)
            log_fh.flush()

            exit_code = wait_for_process(proc, args.post_timeout)

        entry["benchmark_exit_code"] = exit_code
        entry["decode_amx_exit_code"] = decode_result.returncode
        entry["status"] = "ok" if exit_code == 0 and decode_result.returncode == 0 else "error"
        manifest.append(entry)

    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, {"runs": manifest})
    print(f"Manifest: {manifest_path}")

    failures = [m for m in manifest if m.get("status") == "error"]
    if failures:
        print(f"Completed with {len(failures)} failing runs.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
