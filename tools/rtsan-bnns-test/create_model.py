# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Create and compile the small CoreML model used by the BNNS RTSan probe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


class SimpleAudioMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(42)
    model = SimpleAudioMLP().eval()
    traced = torch.jit.trace(model, torch.randn(1, 64))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=(1, 64), name="input")],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.macOS15,
    )

    package = Path("SimpleAudioMLP.mlpackage")
    compiled = Path("SimpleAudioMLP.mlmodelc")
    mlmodel.save(package)
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), "."],
        check=True,
    )
    if not compiled.is_dir():
        raise RuntimeError(f"coremlcompiler did not create {compiled}")

    prediction = mlmodel.predict({"input": np.zeros((1, 64), dtype=np.float32)})
    if prediction["output"].shape != (1, 64):
        raise RuntimeError("unexpected CoreML output shape")
    print(f"Compiled RTSan fixture: {compiled}")


if __name__ == "__main__":
    main()
