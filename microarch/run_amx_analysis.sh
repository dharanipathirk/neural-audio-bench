#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
set -euo pipefail

# ---------------------------------------------------------------------------
# run_amx_analysis.sh
#
# Runs nab-engine in isolated mode for each backend, attaches decode_amx.py
# to capture AMX instruction mix, and writes per-backend JSON files to
# microarch/amx_results/.
#
# Environment overrides:
#   BENCH_BIN  – path to nab-engine binary  (default: ./build/nab-engine)
#   PYTHON     – python runner                  (default: uv run)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_BIN="${BENCH_BIN:-./build/nab-engine}"
PYTHON="${PYTHON:-uv run}"
RESULTS_DIR="${SCRIPT_DIR}/amx_results"
DECODE_SCRIPT="${SCRIPT_DIR}/decode_amx.py"

mkdir -p "${RESULTS_DIR}"

# Backends to scan.  Each entry must match a BackendType name in BenchmarkConfig.h.
# The benchmark binary runs all available backends; the AMX decoder observes
# whichever AMX instructions fire while it is attached.  To isolate a single
# backend, set "backends" in the benchmark config (only the target → true,
# all others → false) before running this script.
BACKENDS=(
    "BNNSGraph"
    "RTNeural_Eigen"
    # RTNeural_XSIMD uses the separate nab-engine-xsimd binary;
    # set BENCH_BIN=./build/nab-engine-xsimd and run this script again for XSIMD.
    "Direct_LibTorch"
    "Direct_ONNX"
    "Anira_LibTorch"
    "Anira_ONNX"
)

if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "ERROR: bench binary not found or not executable: ${BENCH_BIN}" >&2
    echo "  Set BENCH_BIN to the correct path, or build first." >&2
    exit 1
fi

scan_backend() {
    local backend="$1"
    local json_out="${RESULTS_DIR}/${backend}.json"

    echo "──────────────────────────────────────────────"
    echo "Scanning backend: ${backend}"

    # Launch bench in background; suppress its output.
    # Note: --backend is NOT a valid CLI flag; backend selection is done via
    # the benchmark config.  The binary runs whatever backends are enabled.
    "${BENCH_BIN}" --mode isolated --output /dev/null \
        >/dev/null 2>&1 &
    local bench_pid=$!
    echo "  Launched PID ${bench_pid}"

    # Give the process time to warm up and start inference
    sleep 3

    # Verify the process is still alive before attaching
    if ! kill -0 "${bench_pid}" 2>/dev/null; then
        echo "  WARNING: bench process ${bench_pid} exited before scan – skipping ${backend}" >&2
        return
    fi

    # Attach the decoder
    ${PYTHON} "${DECODE_SCRIPT}" \
        --pid "${bench_pid}" \
        --backend "${backend}" \
        --json-output "${json_out}" \
        || true   # don't abort the whole run on decoder failure

    # Clean up the bench process
    if kill -0 "${bench_pid}" 2>/dev/null; then
        kill "${bench_pid}" && wait "${bench_pid}" 2>/dev/null || true
        echo "  Killed PID ${bench_pid}"
    fi

    echo "  JSON → ${json_out}"
}

for backend in "${BACKENDS[@]}"; do
    scan_backend "${backend}"
done

echo "══════════════════════════════════════════════"
echo "AMX analysis complete.  Results in: ${RESULTS_DIR}"
