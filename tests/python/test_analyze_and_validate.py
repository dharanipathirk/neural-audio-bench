# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Analysis pipeline and result validation on the archived paper CSVs."""

import csv

from neural_audio_bench import analyze, validate_results
from neural_audio_bench import config as cfg

EXPECTED = cfg.repo_root() / "experiments" / "dafx26-paper" / "expected"


def test_paper_csvs_validate():
    assert validate_results.validate_csv(EXPECTED / "isolated.csv") == []
    assert validate_results.validate_csv(EXPECTED / "contention.csv") == []


def test_validate_results_rejects_garbage(tmp_path):
    p = tmp_path / "junk.csv"
    p.write_text("foo,bar\n1,2\n")
    assert validate_results.validate_csv(p) != []


def test_analyze_reproduces_paper_headline_numbers(tmp_path):
    out = tmp_path / "analysis.csv"
    analyze.main(
        [
            "--isolated",
            str(EXPECTED / "isolated.csv"),
            "--contention",
            str(EXPECTED / "contention.csv"),
            "--output",
            str(out),
        ]
    )
    iso = tmp_path / "analysis_isolated.csv"
    assert iso.exists()
    # Paper Table 4: BNNSGraph LSTM-Large RTF 0.053 at buffer 128.
    with open(iso) as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["backend"] == "BNNSGraph"
            and r["model"] == "LSTM"
            and r["model_size"] == "large"
            and r["buffer_size"] == "128"
            and r["mode"] == "callback"
        ]
    assert rows, "expected BNNSGraph LSTM large @128 row in enriched output"
    assert abs(float(rows[0]["rtf"]) - 0.0533) < 0.001
