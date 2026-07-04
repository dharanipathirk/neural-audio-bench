# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""Reference-output generation with buffer-vs-sample cross-validation."""

import json

import numpy as np
import torch


def generate_reference_output(model, name, out_dir, num_samples=100):
    """Generate reference outputs for C++ validation.

    Processes all num_samples as ONE buffer (buffer-at-a-time), not
    sample-by-sample.  Also generates a sample-by-sample reference
    for cross-validation.
    """
    model.eval()
    for _buf_name, buf in model.named_buffers():
        buf.zero_()

    rng = np.random.RandomState(42)
    inputs = rng.randn(num_samples).astype(np.float32)

    # --- Buffer-at-a-time reference (primary) ---
    # Reset state
    for _buf_name, buf in model.named_buffers():
        buf.zero_()
    with torch.no_grad():
        x = torch.tensor(inputs, dtype=torch.float32).reshape(1, 1, -1)
        y = model(x)
        buffer_outputs = y.squeeze().tolist()
    # Ensure it's always a list even for single-sample edge case
    if isinstance(buffer_outputs, float):
        buffer_outputs = [buffer_outputs]

    # --- Sample-by-sample reference (cross-validation) ---
    for _buf_name, buf in model.named_buffers():
        buf.zero_()
    sample_outputs = []
    with torch.no_grad():
        for i in range(num_samples):
            x = torch.tensor([[[inputs[i]]]], dtype=torch.float32)
            y = model(x)
            sample_outputs.append(y.item())

    # Verify buffer == sample-by-sample (they must be identical)
    max_diff = max(abs(a - b) for a, b in zip(buffer_outputs, sample_outputs, strict=False))
    if max_diff > 1e-5:
        print(f"    WARNING: buffer vs sample-by-sample diff = {max_diff:.8f}")
    else:
        print(f"    Buffer == sample-by-sample: max_diff={max_diff:.2e}")

    ref_path = out_dir / f"{name}_reference.json"
    with open(ref_path, "w") as f:
        json.dump(
            {
                "inputs": inputs.tolist(),
                "outputs": buffer_outputs,
                "outputs_sample_by_sample": sample_outputs,
                "num_samples": num_samples,
                "processing_mode": "buffer_at_a_time",
            },
            f,
        )
    print(f"  Reference: {ref_path} ({num_samples} samples, buffer-at-a-time)")
