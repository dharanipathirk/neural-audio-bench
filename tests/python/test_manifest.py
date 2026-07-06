# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Model manifest loading and validation."""

import json

import pytest
from neural_audio_bench import config as cfg
from neural_audio_bench import manifest as mf

PAPER_MANIFEST = cfg.repo_root() / "experiments" / "dafx26-paper" / "models.manifest.json"

# Parameter counts from paper Table 1 — the manifest is a frozen record.
PAPER_PARAM_COUNTS = {
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


def test_paper_manifest_validates_and_matches_table1():
    m = mf.load_manifest(PAPER_MANIFEST)
    mf.validate_manifest(m)
    counts = {entry["id"]: entry["param_count"] for entry in m["models"]}
    assert counts == PAPER_PARAM_COUNTS


def test_manifest_missing_required_field_rejected(tmp_path):
    import jsonschema

    bad = {
        "schema_version": 1,
        "models": [{"id": "x", "arch": "lstm"}],
    }  # no state/channels/formats
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(jsonschema.ValidationError):
        mf.validate_manifest(mf.load_manifest(p))


def test_resolved_paths_are_absolute():
    resolved = mf.resolve_manifest(PAPER_MANIFEST)
    for entry in resolved["models"]:
        for fmt, path in entry["formats"].items():
            assert str(path).startswith("/"), f"{entry['id']}/{fmt} not absolute: {path}"
