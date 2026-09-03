# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""``nab`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, analyze, estimate, plot_figures, report
from . import export as export_cmd
from . import manifest as manifest_mod
from . import run as run_cmd
from .config import default_base_config, resolve_config, validate_config


def _cmd_validate_config(args: argparse.Namespace) -> int:
    import jsonschema

    preset = args.experiment or args.config
    try:
        config = resolve_config(base=default_base_config(), preset=preset, overrides=args.overrides)
    except Exception as exc:  # noqa: BLE001 - surface any resolution error cleanly
        print(f"ERROR resolving config: {exc}", file=sys.stderr)
        return 1

    try:
        validate_config(config)
    except jsonschema.ValidationError as exc:
        print(f"CONFIG INVALID: {exc.message}", file=sys.stderr)
        return 1

    manifest_ok = True
    mm = config.get("models_manifest")
    if mm:
        mm_path = Path(mm)
        if mm_path.exists():
            try:
                manifest_mod.validate_manifest(manifest_mod.load_manifest(mm_path))
                print(f"Manifest OK: {mm_path}", file=sys.stderr)
            except (jsonschema.ValidationError, ValueError) as exc:
                message = getattr(exc, "message", str(exc))
                print(f"MANIFEST INVALID ({mm_path}): {message}", file=sys.stderr)
                manifest_ok = False
        else:
            print(f"WARNING: models_manifest not found: {mm_path}", file=sys.stderr)

    if not manifest_ok:
        return 1

    print(json.dumps(config, indent=2))
    print("Config valid.", file=sys.stderr)
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        preset = args.experiment or args.config
        config = resolve_config(base=default_base_config(), preset=preset)
        mm = config.get("models_manifest")
        if not mm:
            print("No models_manifest in the resolved config.", file=sys.stderr)
            return 1
        manifest_path = Path(mm)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = manifest_mod.load_manifest(manifest_path)
    try:
        manifest_mod.validate_manifest(manifest)
    except Exception as exc:  # noqa: BLE001 - still list even if schema check trips
        print(f"WARNING: manifest failed schema validation: {exc}", file=sys.stderr)

    models = manifest.get("models", [])
    print(f"{'id':18} {'arch':9} {'size':8} {'params':>10}  formats")
    print("-" * 70)
    for model in models:
        formats = ",".join(sorted(model.get("formats", {}).keys()))
        print(
            f"{model.get('id', ''):18} "
            f"{model.get('arch', ''):9} "
            f"{model.get('size', ''):8} "
            f"{model.get('param_count', 0):>10,}  "
            f"{formats}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False everywhere: with argparse's default prefix matching,
    # `nab export --manifest X` silently became `--manifest-out X` and
    # overwrote the file it was meant to read.
    parser = argparse.ArgumentParser(
        prog="nab",
        description="neural-audio-bench: benchmark real-time neural audio inference.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"nab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser(
        "export", allow_abbrev=False, help="Export benchmark models to all formats."
    )
    export_cmd.add_arguments(p_export)
    p_export.set_defaults(func=export_cmd.run)

    p_run = sub.add_parser("run", allow_abbrev=False, help="Run the benchmark suite.")
    run_cmd.add_arguments(p_run)
    p_run.set_defaults(func=run_cmd.run)

    p_estimate = sub.add_parser(
        "estimate", allow_abbrev=False, help="Estimate benchmark runtime from a config."
    )
    estimate.add_arguments(p_estimate)
    p_estimate.set_defaults(func=estimate.run)

    p_analyze = sub.add_parser("analyze", allow_abbrev=False, help="Analyze benchmark result CSVs.")
    analyze.add_arguments(p_analyze)
    p_analyze.set_defaults(func=analyze.run)

    p_plot = sub.add_parser(
        "plot", allow_abbrev=False, help="Generate exploratory benchmark figures."
    )
    plot_figures.add_arguments(p_plot)
    p_plot.set_defaults(func=plot_figures.run)

    p_report = sub.add_parser(
        "report", allow_abbrev=False, help="Generate analysis tables + paper figures."
    )
    report.add_arguments(p_report)
    p_report.set_defaults(func=report.run)

    p_list = sub.add_parser("list-models", allow_abbrev=False, help="List model manifest entries.")
    p_list.add_argument("--manifest", default=None, help="Manifest path (default: from config).")
    p_list.add_argument(
        "--config",
        default=str(default_base_config()),
        help="Config to derive the manifest from (default: configs/base.json).",
    )
    p_list.add_argument("--experiment", default=None, help="Experiment name to derive config from.")
    p_list.set_defaults(func=_cmd_list_models)

    p_validate = sub.add_parser(
        "validate-config", allow_abbrev=False, help="Resolve and schema-validate a config."
    )
    p_validate.add_argument(
        "--config",
        default=str(default_base_config()),
        help="Config path to validate (default: configs/base.json).",
    )
    p_validate.add_argument("--experiment", default=None, help="Experiment name to validate.")
    p_validate.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override a config value before validation (repeatable).",
    )
    p_validate.set_defaults(func=_cmd_validate_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
