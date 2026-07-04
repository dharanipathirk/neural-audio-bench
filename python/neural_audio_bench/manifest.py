# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Model manifest loading, validation, resolution, and emission.

A manifest catalogs the models available to the benchmark. Format paths are
relative to ``models_root``, which is itself relative to the manifest file's
directory. Resolution turns every path into an absolute path.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import load_schema


def load_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest: dict) -> None:
    """Validate a manifest against the JSON schema. Raises on error."""
    import jsonschema

    jsonschema.validate(manifest, load_schema("model-manifest.schema.json"))


def models_root_dir(path: str | Path, manifest: dict) -> Path:
    """Absolute directory that format paths are relative to."""
    manifest_dir = Path(path).resolve().parent
    root = manifest.get("models_root")
    if not root:
        return manifest_dir
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path
    return (manifest_dir / root_path).resolve()


def resolve_manifest(path: str | Path, validate: bool = True) -> dict:
    """Load a manifest and resolve ``models_root`` and every format path to
    absolute paths. Returns a copy of the manifest with resolved paths."""
    path = Path(path)
    manifest = load_manifest(path)
    if validate:
        validate_manifest(manifest)
    root = models_root_dir(path, manifest)
    resolved = json.loads(json.dumps(manifest))
    resolved["models_root"] = str(root)
    for model in resolved.get("models", []):
        formats = model.get("formats", {})
        for fmt, rel in list(formats.items()):
            formats[fmt] = str((root / rel).resolve())
    return resolved


ARCH_DISPLAY = {"lstm": "LSTM", "tcn": "TCN", "wavenet": "WaveNet"}
SIZE_ABBR = {"small": "S", "medium": "M", "large": "L"}


def emit_manifest(
    models_info: list[dict],
    out_path: str | Path,
    models_root: str = ".",
    validate: bool = True,
) -> Path:
    """Build a manifest document from per-model info dicts and write it.

    ``models_info`` entries are used verbatim as manifest ``models`` items.
    """
    manifest = {
        "schema_version": 1,
        "models_root": models_root,
        "models": models_info,
    }
    if validate:
        validate_manifest(manifest)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path
