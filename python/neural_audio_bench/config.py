# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Configuration loading, layering, validation, and resolution.

Layering model (deep merge, lists replaced): base.json <- preset <- CLI overrides.
Every config file's ``models_manifest`` path is relative to that file's own
directory; it is resolved to an absolute path at load time so merged configs
carry a self-consistent, provenance-correct manifest path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Locate the repository root (the directory containing ``schemas/``)."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "schemas" / "config.schema.json").exists():
            return parent
    # Fallback: python/neural_audio_bench/config.py -> repo root
    return here.parents[2]


def default_base_config() -> Path:
    """Path to the authoritative base configuration."""
    return repo_root() / "configs" / "base.json"


def schema_path(name: str) -> Path:
    return repo_root() / "schemas" / name


def load_schema(name: str) -> dict:
    with schema_path(name).open(encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str | Path) -> dict:
    """Load a config JSON, resolving ``models_manifest`` to an absolute path.

    The manifest path is resolved relative to this config file's directory so
    that later merging preserves the correct base directory for whichever file
    supplied the value.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    mm = data.get("models_manifest")
    if isinstance(mm, str) and mm and not os.path.isabs(mm):
        data["models_manifest"] = str((path.parent / mm).resolve())
    return data


def merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base``. Dicts merge recursively; every
    other value (including lists) is replaced wholesale."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_preset_path(preset: str | Path) -> Path:
    """Resolve a preset given as an experiment name or a filesystem path."""
    candidate = Path(preset)
    if candidate.exists():
        return candidate
    experiment = repo_root() / "experiments" / str(preset) / "config.json"
    if experiment.exists():
        return experiment
    raise FileNotFoundError(
        f"Could not resolve preset '{preset}' as a config path or experiment name."
    )


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_override(config: dict, override: str) -> None:
    """Apply a single ``key.path=value`` override in place.

    The value is parsed as JSON when possible, otherwise kept as a string.
    """
    if "=" not in override:
        raise ValueError(f"Invalid --set override '{override}', expected key.path=value")
    key, _, raw = override.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid --set override '{override}', empty key")
    value = _parse_override_value(raw)
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def resolve_config(
    base: str | Path | dict | None = None,
    preset: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Resolve a fully-layered config: base <- preset <- overrides."""
    if isinstance(base, dict):
        result = copy.deepcopy(base)
    else:
        base_path = Path(base) if base is not None else default_base_config()
        result = load_config(base_path)
    if preset is not None:
        result = merge(result, load_config(resolve_preset_path(preset)))
    for override in overrides or []:
        apply_override(result, override)
    return result


def validate_config(config: dict) -> None:
    """Validate a resolved config against the JSON schema. Raises on error."""
    import jsonschema

    jsonschema.validate(config, load_schema("config.schema.json"))


def config_sha256(config: dict) -> str:
    """Stable SHA-256 of a resolved config document."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_resolved(config: dict, out_path: str | Path) -> Path:
    """Write a resolved config to ``out_path`` as pretty JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return out_path
