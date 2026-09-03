# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Analysis pipeline, summary tables, and result validation on fixture CSVs.

The fixtures in tests/data/ are small synthetic result files in the current
results schema (v2): two backends, one model, two buffer sizes, all three
contention dimensions, one skipped row, and one duplicated throughput row.
"""

import pandas as pd
import pytest
from neural_audio_bench import analyze, report, validate_results
from neural_audio_bench import config as cfg

DATA = cfg.repo_root() / "tests" / "data"
ISOLATED = DATA / "isolated.csv"
CONTENTION = DATA / "contention.csv"

DEADLINE_128_NS = 128 / 48000 * 1e9


def test_fixture_csvs_validate():
    assert validate_results.validate_csv(ISOLATED) == []
    assert validate_results.validate_csv(CONTENTION) == []


def test_validate_results_rejects_garbage(tmp_path):
    p = tmp_path / "junk.csv"
    p.write_text("foo,bar\n1,2\n")
    assert validate_results.validate_csv(p) != []


def test_load_isolated_drops_non_ok_rows_and_enriches():
    iso = analyze.load_isolated(str(ISOLATED))
    assert "skipped" not in set(iso.get("status", pd.Series(dtype=str)))
    assert (iso["backend"] != "Direct_ONNX").all()

    bnns_cb64 = iso[
        (iso["backend"] == "BNNSGraph") & (iso["mode"] == "callback") & (iso["buffer_size"] == 64)
    ].iloc[0]
    assert bnns_cb64["per_sample_ns"] == pytest.approx(49500 / 64)
    assert bnns_cb64["throughput_xrt"] == pytest.approx(1.0 / bnns_cb64["rtf"])


def test_logical_throughput_rows_keeps_first_physical_row():
    iso = analyze.load_isolated(str(ISOLATED))
    physical = iso[iso["mode"] == "throughput"]
    logical = analyze.logical_throughput_rows(iso)
    assert len(physical) == 3  # BNNSGraph twice (two engine runs), RTNeural once
    assert len(logical) == 2
    bnns = logical[logical["backend"] == "BNNSGraph"].iloc[0]
    assert bnns["median_ns"] == 177  # the first run's value, not the duplicate's 180


def test_summary_tables_cover_all_dimensions(tmp_path):
    report._write_summary_tables(str(ISOLATED), str(CONTENTION), tmp_path)

    throughput = pd.read_csv(tmp_path / "table_isolated_throughput.csv")
    assert set(throughput["backend"]) == {"BNNSGraph", "RTNeural_Eigen"}

    rtf128 = pd.read_csv(tmp_path / "table_isolated_rtf128.csv")
    bnns = rtf128[rtf128["backend"] == "BNNSGraph"].iloc[0]
    assert bnns["rtf"] == pytest.approx(58000 / DEADLINE_128_NS, rel=1e-4)

    dim_a = pd.read_csv(tmp_path / "table_contention_dim_a.csv")
    assert len(dim_a) == 4  # 2 backends x levels 0 and 36
    rt0 = dim_a[(dim_a["backend"] == "RTNeural_XSIMD") & (dim_a["contention_level"] == 0)].iloc[0]
    assert rt0["util_p99"] == pytest.approx(150.0)
    assert rt0["hw_xruns"] == 300

    dim_b = pd.read_csv(tmp_path / "table_contention_dim_b.csv")
    assert len(dim_b) == 6  # 3 backends x instance counts 1 and 2
    anira2 = dim_b[(dim_b["backend"] == "Anira_LibTorch") & (dim_b["instance_count"] == 2)].iloc[0]
    assert anira2["inf_underruns"] == 1234

    dim_c = pd.read_csv(tmp_path / "table_contention_dim_c.csv")
    assert len(dim_c) == 4  # 2 backends x depths 1 and 7
    assert {"neural_util_p99", "callback_util_p99"} <= set(dim_c.columns)
    bnns7 = dim_c[(dim_c["backend"] == "BNNSGraph") & (dim_c["contention_level"] == 7)].iloc[0]
    assert bnns7["callback_util_p99"] == pytest.approx(4.4)


def test_compute_degradation_uses_isolated_rtf_at_128():
    iso = analyze.load_isolated(str(ISOLATED))
    neural, _cb = analyze.load_contention(str(CONTENTION))
    deg = analyze.compute_degradation(iso, neural)
    assert not deg.empty
    row = deg[
        (deg["backend"] == "BNNSGraph")
        & (deg["contention_level"] == 0)
        & (deg["dimension"] == "dim_a")
    ].iloc[0]
    # contention rtf (p50 3% of deadline) over isolated rtf (58000 ns / deadline)
    assert row["degradation"] == pytest.approx(0.03 / (58000 / DEADLINE_128_NS), rel=1e-3)


def test_analyze_cli_writes_enriched_files(tmp_path):
    out = tmp_path / "analysis.csv"
    assert (
        analyze.main(
            ["--isolated", str(ISOLATED), "--contention", str(CONTENTION), "--output", str(out)]
        )
        == 0
    )
    for suffix in ("_isolated", "_contention", "_contention_cb", "_degradation"):
        assert (tmp_path / f"analysis{suffix}.csv").exists(), suffix
