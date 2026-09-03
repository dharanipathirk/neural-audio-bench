# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Model manifest loading and validation."""

import json

import pytest
from neural_audio_bench import config as cfg
from neural_audio_bench import manifest as mf

FIXTURE_MANIFEST = cfg.repo_root() / "tests" / "data" / "models.manifest.json"

# Parameter counts of the nine built-in models as `nab export` computes them.
BUILTIN_PARAM_COUNTS = {
    "lstm_small": 1861,
    "lstm_medium": 6921,
    "lstm_large": 38113,
    "tcn_small": 4337,
    "tcn_medium": 33633,
    "tcn_large": 93745,
    "wavenet_small": 841,
    "wavenet_medium": 10609,
    "wavenet_large": 41697,
}


def _entry(model_id: str) -> dict:
    return {
        "id": model_id,
        "arch": "custom",
        "state": "stateless",
        "channels": 1,
        "formats": {"onnx": "model.onnx"},
    }


def test_fixture_manifest_validates_and_matches_builtin_catalog():
    manifest = mf.load_manifest(FIXTURE_MANIFEST)
    mf.validate_manifest(manifest)
    counts = {entry["id"]: entry["param_count"] for entry in manifest["models"]}
    assert counts == BUILTIN_PARAM_COUNTS


def test_manifest_missing_required_field_rejected(tmp_path):
    import jsonschema

    bad = {
        "schema_version": 1,
        "models": [{"id": "x", "arch": "lstm"}],
    }  # no state/channels/formats
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(jsonschema.ValidationError):
        mf.validate_manifest(mf.load_manifest(path))


def test_rtneural_format_requires_topology_metadata():
    import jsonschema

    entry = _entry("rt_model")
    entry["formats"]["rtneural"] = "weights.json"
    manifest = {"schema_version": 1, "models": [entry]}
    with pytest.raises(jsonschema.ValidationError):
        mf.validate_manifest(manifest)

    entry["param_count"] = 1
    entry["hyperparams"] = {"hidden": 1}
    mf.validate_manifest(manifest)


def test_duplicate_model_ids_rejected():
    manifest = {"schema_version": 1, "models": [_entry("same"), _entry("same")]}
    with pytest.raises(ValueError, match="duplicate model id"):
        mf.validate_manifest(manifest)


def test_resolved_paths_are_absolute():
    resolved = mf.resolve_manifest(FIXTURE_MANIFEST)
    for entry in resolved["models"]:
        for fmt, path in entry["formats"].items():
            assert str(path).startswith("/"), f"{entry['id']}/{fmt} not absolute: {path}"
