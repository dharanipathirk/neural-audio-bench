#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
set -euo pipefail

# ---------------------------------------------------------------------------
# run_benchmark.sh
#
# Runs the full benchmark suite with thermal fairness:
#   - CPU warmup burn before each phase (reaches thermal steady state)
#   - Cooldown between phases (resets to consistent baseline)
#
# Phases:
#   1. Eigen isolated
#   2. XSIMD isolated
#   3. XSIMD contention (Eigen contention skipped)
#
# Merges CSVs and runs analysis automatically.
#
# Usage:
#   caffeinate -dims ./scripts/run_benchmark.sh
#   caffeinate -dims ./scripts/run_benchmark.sh --cooldown 60 --warmup 30
#   caffeinate -dims ./scripts/run_benchmark.sh --config benchmark_config_test.json
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EIGEN_BIN="${REPO_ROOT}/build/nab-engine"
XSIMD_BIN="${REPO_ROOT}/build/nab-engine-xsimd"
OUTPUT_DIR="${REPO_ROOT}/results"
CONFIG_FLAG=""
COOLDOWN=120
WARMUP=45
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cooldown)         COOLDOWN="$2"; shift 2 ;;
        --warmup)           WARMUP="$2"; shift 2 ;;
        --config)           CONFIG_FLAG="--config $2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --allow-any-device) EXTRA_ARGS="$EXTRA_ARGS --allow-any-device"; shift ;;
        *)                  echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "${OUTPUT_DIR}/benchmark_log.txt"; }

# ---------------------------------------------------------------------------
# CPU thermal warmup — saturate all cores to reach steady-state frequency.
# This ensures the first backend in each phase doesn't benefit from a cold CPU.
# ---------------------------------------------------------------------------
cpu_warmup() {
    local duration=$1
    if [[ "${duration}" -le 0 ]]; then return; fi

    local ncpu
    ncpu=$(sysctl -n hw.ncpu)
    log "CPU warmup: burning ${ncpu} cores for ${duration}s to reach thermal steady state..."

    local pids=()
    for ((i = 0; i < ncpu; i++)); do
        yes > /dev/null 2>&1 &
        pids+=($!)
    done

    sleep "${duration}"

    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true

    log "CPU warmup complete — starting benchmark immediately."
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
for bin in "${EIGEN_BIN}" "${XSIMD_BIN}"; do
    if [[ ! -x "${bin}" ]]; then
        log "ERROR: binary not found: ${bin}"
        exit 1
    fi
done

log "=========================================="
log "Benchmark Suite"
log "  Eigen:    ${EIGEN_BIN}"
log "  XSIMD:    ${XSIMD_BIN}"
log "  Output:   ${OUTPUT_DIR}"
log "  Cooldown: ${COOLDOWN}s"
log "  Warmup:   ${WARMUP}s"
log "=========================================="

# ---------------------------------------------------------------------------
# Phase 1: Eigen isolated
# ---------------------------------------------------------------------------
cpu_warmup "${WARMUP}"
log "PHASE 1/3: Eigen isolated"
"${EIGEN_BIN}" --mode isolated \
    --output "${OUTPUT_DIR}/isolated.csv" \
    ${CONFIG_FLAG} ${EXTRA_ARGS} \
    2>&1 | tee -a "${OUTPUT_DIR}/benchmark_log.txt"

log "Cooling down ${COOLDOWN}s..."
sleep "${COOLDOWN}"

# ---------------------------------------------------------------------------
# Phase 2: XSIMD isolated
# ---------------------------------------------------------------------------
cpu_warmup "${WARMUP}"
log "PHASE 2/3: XSIMD isolated"
"${XSIMD_BIN}" --mode isolated \
    --output "${OUTPUT_DIR}/isolated_xsimd.csv" \
    ${CONFIG_FLAG} ${EXTRA_ARGS} \
    2>&1 | tee -a "${OUTPUT_DIR}/benchmark_log.txt"

log "Cooling down ${COOLDOWN}s..."
sleep "${COOLDOWN}"

# ---------------------------------------------------------------------------
# Phase 3: XSIMD contention
# ---------------------------------------------------------------------------
cpu_warmup "${WARMUP}"
log "PHASE 3/3: XSIMD contention"
"${XSIMD_BIN}" --mode contention \
    --output-dir "${OUTPUT_DIR}" \
    ${CONFIG_FLAG} ${EXTRA_ARGS} \
    2>&1 | tee -a "${OUTPUT_DIR}/benchmark_log.txt"

# ---------------------------------------------------------------------------
# Merge CSVs
# ---------------------------------------------------------------------------
log "Merging isolated CSVs..."
cp "${OUTPUT_DIR}/isolated.csv" "${OUTPUT_DIR}/isolated_merged.csv"
tail -n +2 "${OUTPUT_DIR}/isolated_xsimd.csv" >> "${OUTPUT_DIR}/isolated_merged.csv"
log "  isolated_merged.csv: $(wc -l < "${OUTPUT_DIR}/isolated_merged.csv") rows"

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
log "Running analysis..."
PYTHON="${PYTHON:-uv run}"

${PYTHON} "${REPO_ROOT}/analysis/analyze_results.py" \
    --isolated "${OUTPUT_DIR}/isolated_merged.csv" \
    --contention "${OUTPUT_DIR}/contention.csv" \
    --output "${OUTPUT_DIR}/analysis_xsimd.csv" \
    2>&1 | tee -a "${OUTPUT_DIR}/benchmark_log.txt" || true

${PYTHON} "${REPO_ROOT}/analysis/plot_figures.py" \
    --isolated "${OUTPUT_DIR}/isolated_merged.csv" \
    --contention "${OUTPUT_DIR}/contention.csv" \
    --output-dir "${OUTPUT_DIR}/figures" \
    2>&1 | tee -a "${OUTPUT_DIR}/benchmark_log.txt" || true

log "=========================================="
log "ALL DONE"
log "  Merged isolated: ${OUTPUT_DIR}/isolated_merged.csv"
log "  Contention:      ${OUTPUT_DIR}/contention.csv"
log "  Figures:         ${OUTPUT_DIR}/figures/"
log "=========================================="
