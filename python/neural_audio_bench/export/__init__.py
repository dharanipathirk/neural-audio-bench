# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Export all benchmark models to CoreML, RTNeural JSON, ONNX, and TorchScript.

Reads model configurations from a resolved benchmark config.
Exports small/medium/large tiers for each architecture, then emits a model
manifest with real parameter counts.

Seeds (torch.manual_seed(42) at each tier and before each architecture) are
preserved from the original exporter so weights are byte-reproducible.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..config import default_base_config, repo_root, resolve_config, validate_config
from ..manifest import ARCH_DISPLAY, SIZE_ABBR, emit_manifest

# torch / coremltools are imported lazily inside the export functions so that
# lightweight commands (help, estimate, analyze, validate-config) stay fast.

SEED = 42

CONV_ONNX_NOTE = (
    "ONNX export is stateless (ONNX Runtime has no InOut mechanism for conv state buffers)."
)


def _model_info(arch: str, size: str, model, hyperparams: dict) -> dict:
    """Build a manifest entry for an exported model (paths relative to models_root)."""
    from .models import count_params

    name = f"stateful_{arch}_{size}"
    base = f"{arch}/{size}/{name}"
    info = {
        "id": f"{arch}_{size}",
        "arch": arch,
        "size": size,
        "display_name": f"{ARCH_DISPLAY.get(arch, arch)}-{SIZE_ABBR.get(size, size)}",
        "state": "stateful",
        "channels": 1,
        "param_count": count_params(model),
        "hyperparams": {k: v for k, v in hyperparams.items() if not str(k).startswith("_")},
        "formats": {
            "coreml": f"{base}.mlmodelc",
            "onnx": f"{base}.onnx",
            "torchscript": f"{base}.pt",
            "rtneural": f"{base}_weights.json",
        },
    }
    if arch in ("tcn", "wavenet"):
        info["notes"] = CONV_ONNX_NOTE
    return info


def export_model(arch_name, size_name, model, states_fn, models_out: Path, is_lstm=False):
    """Export a single model to all formats and return its manifest entry."""
    from .exporters import (
        export_coreml,
        export_onnx_conv,
        export_onnx_lstm,
        export_rtneural_json,
        export_torchscript,
    )
    from .models import count_params
    from .validate import generate_reference_output

    name = f"stateful_{arch_name}_{size_name}"
    out_dir = models_out / arch_name / size_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  --- {arch_name}/{size_name}: {count_params(model):,} params ---")

    states = states_fn(model)
    export_coreml(model, name, states, out_dir)

    # Reset state for other exports
    for _buf_name, buf in model.named_buffers():
        buf.zero_()

    export_rtneural_json(model, name, out_dir)

    # ONNX: LSTM uses explicit state I/O; TCN/WaveNet use internal state
    if is_lstm:
        export_onnx_lstm(model, name, out_dir)
    else:
        export_onnx_conv(model, name, out_dir)

    export_torchscript(model, name, out_dir)
    generate_reference_output(model, name, out_dir)


def _parse_only(only: list[str] | None) -> set[tuple[str, str]] | None:
    if not only:
        return None
    selected: set[tuple[str, str]] = set()
    for item in only:
        arch, _, size = item.partition("/")
        if not arch or not size:
            raise ValueError(f"Invalid --only '{item}', expected ARCH/SIZE (e.g. lstm/small)")
        selected.add((arch.strip(), size.strip()))
    return selected


def export_all(
    config: dict,
    manifest_out: str | Path,
    models_out: str | Path,
    only: list[str] | None = None,
) -> Path:
    """Export the enabled models and emit a manifest. Returns the manifest path."""
    import numpy as np
    import torch

    from .models import (
        StatefulLSTM,
        StatefulTCN,
        StatefulWaveNet,
        lstm_states,
        tcn_states,
        wavenet_states,
    )

    models_out = Path(models_out)
    manifest_out = Path(manifest_out)
    only_set = _parse_only(only)

    print("=" * 60)
    print("Neural Audio Benchmark -- Model Export Pipeline")
    print("Processing mode: BUFFER-AT-A-TIME")
    print("=" * 60)

    model_cfg = config["models"]
    size_cfg = config.get("model_sizes", {"small": True, "medium": True, "large": True})

    def selected(arch: str, size: str) -> bool:
        return only_set is None or (arch, size) in only_set

    models_info: list[dict] = []

    for size_name in ["small", "medium", "large"]:
        if not size_cfg.get(size_name, True):
            print(f"\nSkipping size: {size_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Size tier: {size_name.upper()}")
        print(f"{'=' * 60}")

        # Reset seed for each tier so weights are reproducible
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # LSTM
        cfg = model_cfg["lstm"][size_name]
        if selected("lstm", size_name):
            model = StatefulLSTM(hidden=cfg["hidden"])
            export_model("lstm", size_name, model, lstm_states, models_out, is_lstm=True)
            models_info.append(_model_info("lstm", size_name, model, cfg))

        # TCN
        torch.manual_seed(SEED)
        cfg = model_cfg["tcn"][size_name]
        if selected("tcn", size_name):
            model = StatefulTCN(
                channels=cfg["channels"],
                kernel_size=cfg["kernel_size"],
                num_layers=cfg["num_layers"],
            )
            export_model("tcn", size_name, model, tcn_states, models_out)
            models_info.append(_model_info("tcn", size_name, model, cfg))

        # WaveNet
        torch.manual_seed(SEED)
        cfg = model_cfg["wavenet"][size_name]
        if selected("wavenet", size_name):
            model = StatefulWaveNet(
                channels=cfg["channels"],
                kernel_size=cfg["kernel_size"],
                num_layers=cfg["num_layers"],
            )
            export_model("wavenet", size_name, model, wavenet_states, models_out)
            models_info.append(_model_info("wavenet", size_name, model, cfg))

    # models_root makes the manifest's relative format paths resolve to models_out
    models_root = os.path.relpath(models_out.resolve(), manifest_out.resolve().parent)
    emit_manifest(models_info, manifest_out, models_root=models_root)

    print(f"\n{'=' * 60}")
    print("All models exported successfully!")
    print(f"  Manifest: {manifest_out}")
    print(f"{'=' * 60}")
    return manifest_out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(default_base_config()),
        help="Config JSON providing model hyperparameters (default: configs/base.json).",
    )
    parser.add_argument(
        "--out", default=None, help="Output models directory (default: <repo>/models)."
    )
    parser.add_argument(
        "--manifest-out", default=None, help="Manifest path (default: <out>/manifest.json)."
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ARCH/SIZE",
        help="Export only this ARCH/SIZE (repeatable, e.g. --only lstm/small).",
    )


def run(args: argparse.Namespace) -> int:
    config = resolve_config(base=args.config)
    validate_config(config)

    models_out = Path(args.out) if args.out else repo_root() / "models"
    manifest_out = Path(args.manifest_out) if args.manifest_out else models_out / "manifest.json"

    export_all(config, manifest_out, models_out, only=args.only or None)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export benchmark models to all formats")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
