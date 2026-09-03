# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Config loading, layering, and validation."""

import pytest
from neural_audio_bench import config as cfg


def test_base_config_validates():
    c = cfg.load_config(cfg.default_base_config())
    cfg.validate_config(c)  # must not raise


def test_base_config_default_protocol_values():
    c = cfg.load_config(cfg.default_base_config())
    cfg.validate_config(c)
    # The shipped default protocol — guards against accidental edits.
    assert c["isolated"]["min_iterations"] == 10000
    assert c["isolated"]["num_reps"] == 1
    assert c["isolated"]["target_measure_seconds"] == 5.0
    assert c["contention"]["instance_counts"] == [1, 2, 4, 8, 16]
    assert c["contention"]["contention_levels"] == [0, 8, 24, 36]
    assert c["contention"]["neural_track_depths"] == [1, 3, 5, 7]
    assert c["contention"]["measure_seconds"] == 15
    assert c["contention"]["num_tracks"] == 36
    assert c["contention"]["use_system_au"] is True


def test_merge_deep_and_lists_replaced():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2, 3]}
    override = {"a": {"y": 5}, "l": [9]}
    merged = cfg.merge(base, override)
    assert merged["a"] == {"x": 1, "y": 5}
    assert merged["l"] == [9]  # lists replace, never concatenate


def test_apply_override_parses_json_values():
    c = {"isolated": {"num_reps": 1, "buffer_sizes": [128]}}
    cfg.apply_override(c, "isolated.num_reps=3")
    cfg.apply_override(c, "isolated.buffer_sizes=[64, 256]")
    assert c["isolated"]["num_reps"] == 3
    assert c["isolated"]["buffer_sizes"] == [64, 256]


def test_invalid_config_rejected():
    import jsonschema

    c = cfg.load_config(cfg.default_base_config())
    del c["isolated"]
    with pytest.raises(jsonschema.ValidationError):
        cfg.validate_config(c)


def test_config_sha256_stable_under_key_order():
    a = {"x": 1, "y": {"a": 2, "b": 3}}
    b = {"y": {"b": 3, "a": 2}, "x": 1}
    assert cfg.config_sha256(a) == cfg.config_sha256(b)


def test_custom_model_and_backend_selectors_validate():
    c = cfg.load_config(cfg.default_base_config())
    c["model_types"] = {"custom_arch": True}
    c["model_sizes"]["research"] = True
    c["backends"]["ThirdPartyBackend"] = True
    cfg.validate_config(c)


def test_buffer_larger_than_engine_capacity_rejected():
    import jsonschema

    c = cfg.load_config(cfg.default_base_config())
    c["isolated"]["buffer_sizes"] = [4096]
    with pytest.raises(jsonschema.ValidationError):
        cfg.validate_config(c)
