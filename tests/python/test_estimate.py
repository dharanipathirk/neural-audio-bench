# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Runtime-estimator regression tests."""

from neural_audio_bench import estimate
from neural_audio_bench.config import default_base_config, resolve_config


def test_estimate_works_before_cmake_configuration(tmp_path, capsys):
    assert (
        estimate.main(
            [
                "--config",
                str(default_base_config()),
                "--build-dir",
                str(tmp_path / "not-configured"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Full interleaved run estimate" in captured.out
    assert "CPU warmups: 3 x 45s" in captured.out
    assert "unconfigured default full build (assumed)" in captured.out
    assert "no configured build metadata found" in captured.err


def test_estimate_counts_manifest_models_and_custom_scenarios():
    cfg = resolve_config()
    cfg["model_types"] = {"lstm": True, "tcn": False, "wavenet": False}
    cfg["custom_scenarios"] = [
        {
            "id": "custom_load",
            "sweep": {"parameter": "tracks", "values": [1, 4]},
            "buffer_sizes": [64, 128],
            "tracks": [{"count": "sweep", "neural": True, "clip": "noise"}],
        }
    ]
    features = estimate.BuildFeatures(
        rtneural_backend="RTNeural_XSIMD",
        has_libtorch=True,
        has_onnx=True,
        has_anira=True,
    )

    # Three enabled LSTM size entries and four contention-capable backends.
    assert len(estimate.enabled_model_specs(cfg)) == 3
    report = estimate.compute_contention_runtime(cfg, features)
    assert report["custom_load"].configs == 3 * 4 * 2 * 2
    assert report["total"].configs == sum(
        summary.configs for name, summary in report.items() if name != "total"
    )


def test_estimate_rejects_negative_orchestration_delays(capsys):
    assert estimate.main(["--warmup", "-1"]) == 2
    assert "must be non-negative" in capsys.readouterr().err
