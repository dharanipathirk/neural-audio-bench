#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
set -euo pipefail

# ---------------------------------------------------------------------------
# run_amx_analysis.sh
#
# Compatibility entry point for the isolated microarchitecture harness.
# Outputs are written below microarch/results/<timestamp>/.
#
# Environment overrides:
#   BENCH_BIN  – path to nab-engine binary (default: build/nab-engine)
#   CONFIG     – base config (default: configs/base.json)
#   MODELS     – model directory (default: models)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_BIN="${BENCH_BIN:-${REPO_ROOT}/build/nab-engine}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/base.json}"
MODELS="${MODELS:-${REPO_ROOT}/models}"

exec uv run python "${SCRIPT_DIR}/run_isolated_microarch.py" \
    --bench "${BENCH_BIN}" \
    --config "${CONFIG}" \
    --models "${MODELS}" \
    "$@"
