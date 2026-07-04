# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Best-effort machine metadata capture for run provenance.

Every probe degrades gracefully to ``None`` if the underlying tool is missing
or fails, so this module is safe to call on any platform.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def _sysctl(key: str) -> str | None:
    return _run(["sysctl", "-n", key])


def _sysctl_int(key: str) -> int | None:
    value = _sysctl(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _torch_version() -> str | None:
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch import may fail for many reasons
        return None
    return getattr(torch, "__version__", None)


def _thermal_state() -> str | None:
    """Parse ``pmset -g therm`` output best-effort."""
    text = _run(["pmset", "-g", "therm"])
    if text is None:
        return None
    return text


def machine_info() -> dict:
    """Collect machine metadata as a JSON-serializable dict."""
    return {
        "hw_model": _sysctl("hw.model"),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "performance_cores": _sysctl_int("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl_int("hw.perflevel1.physicalcpu"),
        "physical_cpu": _sysctl_int("hw.physicalcpu"),
        "logical_cpu": _sysctl_int("hw.ncpu"),
        "mem_bytes": _sysctl_int("hw.memsize"),
        "macos_version": _run(["sw_vers", "-productVersion"]),
        "macos_build": _run(["sw_vers", "-buildVersion"]),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": _torch_version(),
        "thermal_state": _thermal_state(),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = "Print best-effort machine metadata as JSON."


def run(_args: argparse.Namespace) -> int:
    print(json.dumps(machine_info(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print machine metadata as JSON")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
