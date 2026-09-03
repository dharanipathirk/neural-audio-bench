# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Run orchestration and variant-merge tests."""

import csv

from neural_audio_bench.run import Logger, merge_isolated_csvs


def test_merge_keeps_only_backend_unique_to_second_engine(tmp_path):
    header = "mode,backend,model,model_size,buffer_size,status\n"
    eigen = tmp_path / "eigen.csv"
    xsimd = tmp_path / "xsimd.csv"
    output = tmp_path / "merged.csv"
    eigen.write_text(
        header
        + "callback,BNNSGraph,LSTM,small,128,ok\n"
        + "callback,RTNeural_Eigen,LSTM,small,128,ok\n",
        encoding="utf-8",
    )
    xsimd.write_text(
        header
        + "callback,BNNSGraph,LSTM,small,128,ok\n"
        + "callback,RTNeural_XSIMD,LSTM,small,128,ok\n",
        encoding="utf-8",
    )

    merge_isolated_csvs(eigen, xsimd, output, Logger(tmp_path / "merge.log"))

    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["backend"] for row in rows] == [
        "BNNSGraph",
        "RTNeural_Eigen",
        "RTNeural_XSIMD",
    ]
