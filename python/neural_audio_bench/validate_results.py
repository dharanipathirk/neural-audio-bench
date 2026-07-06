# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Validate a benchmark result CSV: header shape, row sanity, non-emptiness.

Used by CI after the smoke run, and useful for checking any result file:

    uv run python -m neural_audio_bench.validate_results results/isolated.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ISOLATED_COLUMNS = [
    "mode", "backend", "model", "model_size", "buffer_size", "rep",
    "median_ns", "mean_ns", "p95_ns", "p99_ns", "p999_ns", "min_ns",
    "max_ns", "stddev_ns", "rtf", "dropouts", "total_samples",
]

CONTENTION_COLUMNS = [
    "dimension", "backend", "model", "model_size", "buffer_size",
    "contention_level", "instance_count", "rep",
    "median_ns", "mean_ns", "p95_ns", "p99_ns", "p999_ns", "min_ns",
    "max_ns", "stddev_ns", "rtf", "dropouts", "total_samples",
    "util_p50", "util_p95", "util_p99", "util_p999", "util_max",
    "hw_xruns", "inf_underruns", "thread_count",
]

# Columns added by results-schema v2 (status rows + versioning); accepted
# in addition to the v1 layout so both eras of CSV validate.
V2_PREFIX = ["schema_version", "status", "error_msg"]


def validate_csv(path: str | Path) -> list[str]:
    """Return a list of problems (empty = valid)."""
    path = Path(path)
    problems: list[str] = []
    if not path.exists():
        return [f"{path}: file does not exist"]

    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [f"{path}: empty file"]
        rows = list(reader)

    base = header[len(V2_PREFIX):] if header[: len(V2_PREFIX)] == V2_PREFIX else header

    if base == ISOLATED_COLUMNS:
        kind = "isolated"
    elif base == CONTENTION_COLUMNS:
        kind = "contention"
    else:
        return [f"{path}: header matches neither isolated nor contention schema: {header}"]

    if not rows:
        problems.append(f"{path}: no data rows")

    ncols = len(header)
    idx = {name: header.index(name) for name in header}
    ok_rows = 0
    for i, row in enumerate(rows, start=2):
        if len(row) != ncols:
            problems.append(f"{path}:{i}: expected {ncols} columns, got {len(row)}")
            continue
        if "status" in idx and row[idx["status"]] != "ok":
            continue  # skipped/error rows carry no timing numbers
        try:
            median = float(row[idx["median_ns"]])
            total = int(row[idx["total_samples"]])
        except ValueError:
            problems.append(f"{path}:{i}: non-numeric timing fields")
            continue
        if median < 0:
            problems.append(f"{path}:{i}: negative median_ns")
        if total <= 0:
            problems.append(f"{path}:{i}: total_samples must be positive")
        ok_rows += 1

    if not problems and ok_rows == 0:
        problems.append(f"{path}: contains no successful (status=ok) rows")

    if not problems:
        print(f"{path}: OK ({kind}, {len(rows)} rows, {ok_rows} measured)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+", help="Result CSV files to validate")
    args = parser.parse_args(argv)

    all_problems: list[str] = []
    for f in args.csv_files:
        all_problems.extend(validate_csv(f))

    for p in all_problems:
        print(f"ERROR: {p}", file=sys.stderr)
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
