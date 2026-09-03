# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Microarchitecture result-layout tests."""

import json
from pathlib import Path

from neural_audio_bench import config, plot_figures


def _amx_payload(backend: str, loads: int) -> dict:
    return {
        "backend": backend,
        "libraries": {
            "libBNNS.dylib": {
                "loads": loads,
                "compute": 2,
                "stores": 1,
                "extract": 0,
                "control": 2,
            }
        },
    }


def test_plotter_reads_only_latest_timestamped_amx_run(tmp_path, monkeypatch):
    for run_name, backend, loads in [
        ("20260101_000000", "old", 1),
        ("20260102_000000", "new", 3),
    ]:
        amx_dir = tmp_path / "microarch" / "results" / run_name / "amx_json"
        amx_dir.mkdir(parents=True)
        (amx_dir / "result.json").write_text(json.dumps(_amx_payload(backend, loads)))

    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    records = plot_figures._load_amx_results(Path(__file__))
    assert records == [
        {
            "backend": "new",
            "loads": 3,
            "compute": 2,
            "stores": 1,
            "extract": 0,
            "control": 2,
        }
    ]
