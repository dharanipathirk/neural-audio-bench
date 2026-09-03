# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Command-line parser behaviour."""

import pytest
from neural_audio_bench import cli


def test_abbreviated_options_are_rejected():
    # With argparse's default prefix matching, `--manifest` used to expand to
    # `--manifest-out` and overwrite the frozen paper manifest.
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["export", "--manifest", "somewhere/models.manifest.json"])


def test_full_option_names_still_parse():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["export", "--manifest-out", "out/manifest.json", "--only", "lstm/small"]
    )
    assert args.manifest_out == "out/manifest.json"
    assert args.only == ["lstm/small"]


def test_run_rejects_negative_orchestration_delays(capsys):
    assert cli.main(["run", "--cooldown", "-1"]) == 2
    assert "must be non-negative" in capsys.readouterr().err
