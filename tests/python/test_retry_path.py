# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Contention transient-retry path, driven by the NAB_TEST_FORCE_TRANSIENT hook.

Requires the built nab-engine and BlackHole as the default output device, so it
is skipped when either is unavailable (the isolated CI runner has neither). The
retry logic itself (fault injection -> retry -> succeed / exhaust) is exercised
here; on a machine without BlackHole the test no-ops rather than failing.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest
from neural_audio_bench import config as cfg

ENGINE = cfg.repo_root() / "build" / "nab-engine"


def _has_blackhole() -> bool:
    # The engine refuses contention mode unless a virtual output device is
    # default; probe cheaply by checking for the driver's presence.
    out = subprocess.run(["system_profiler", "SPAudioDataType"], capture_output=True, text=True)
    return "BlackHole" in out.stdout


pytestmark = pytest.mark.skipif(
    not ENGINE.exists() or not _has_blackhole(),
    reason="requires built nab-engine and BlackHole as default output device",
)


def _make_config(tmp_path: Path) -> Path:
    c = json.loads((cfg.repo_root() / "configs" / "smoke.json").read_text())
    c["models_manifest"] = str(cfg.repo_root() / "models" / "manifest.json")
    c["contention"].update(
        contention_levels=[],
        neural_track_depths=[],
        instance_counts=[1],
        measure_seconds=3,
        warmup_min_seconds=2,
        warmup_callbacks=30,
    )
    c["model_types"] = {"lstm": True, "tcn": False, "wavenet": False}
    c["model_sizes"] = {"small": True, "medium": False, "large": False}
    c["backends"] = {k: (k == "BNNSGraph") for k in c["backends"] if not k.startswith("_")}
    p = tmp_path / "retry.json"
    p.write_text(json.dumps(c))
    return p


def _run(config: Path, out: Path, forced: int) -> str:
    env = {"NAB_TEST_FORCE_TRANSIENT": str(forced)}
    subprocess.run(
        [
            "caffeinate",
            "-dims",
            str(ENGINE),
            "--mode",
            "contention",
            "--config",
            str(config),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
        timeout=300,
    )
    with open(out) as f:
        for r in csv.DictReader(f):
            if r["dimension"] == "dim_b":
                return r["status"]
    return "missing"


def test_retry_recovers_before_exhaustion(tmp_path):
    # 2 forced transients < 3 attempts -> succeeds on the 3rd attempt.
    assert _run(_make_config(tmp_path), tmp_path / "r2.csv", forced=2) == "ok"


def test_retry_exhaustion_writes_error(tmp_path):
    # 3 forced transients == max attempts -> error row.
    assert _run(_make_config(tmp_path), tmp_path / "r3.csv", forced=3) == "error"
