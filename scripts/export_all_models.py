#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Export all benchmark models to CoreML, RTNeural JSON, ONNX, and TorchScript.

Reads model configurations from benchmark_config.json.
Exports small/medium/large tiers for each architecture.

BUFFER-AT-A-TIME processing: all models accept variable-length input
x: (1, 1, N) where N can be 1..2048.  State is carried across calls.

LSTM is sequential (LSTM -> Dense).
TCN has residual connections (h = h + Conv1x1(PReLU(Conv(state)))).
WaveNet has residual + skip connections, following RTNeural-NAM's pattern.
All backends process architecturally identical models.

Usage:
  uv run scripts/export_all_models.py
"""

import json
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
CONFIG_PATH = PROJECT_DIR / "benchmark_config.json"

# Load config
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)


# ---------------------------------------------------------------------------
# Model 1: Stateful LSTM
# LSTM(1, hidden) -> Dense(hidden, 1) + hidden/cell state
#
# nn.LSTM handles variable seq_len natively -- no forward() change needed.
# ---------------------------------------------------------------------------
class StatefulLSTM(nn.Module):
    def __init__(self, hidden=20):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.dense = nn.Linear(hidden, 1)
        self.register_buffer("hidden_state", torch.zeros(1, 1, hidden))
        self.register_buffer("cell_state", torch.zeros(1, 1, hidden))

    def forward(self, x):
        # x: (1, 1, N) -> transpose to (1, N, 1) for batch_first LSTM
        out, (h_n, c_n) = self.lstm(x.transpose(1, 2), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h_n
        self.cell_state[:] = c_n
        # out: (1, N, hidden) -> dense -> (1, N, 1) -> transpose to (1, 1, N)
        return self.dense(out).transpose(1, 2)


# ---------------------------------------------------------------------------
# Model 2: Stateful TCN
# Conv1D(1->ch,k=1) -> [Conv1D(ch,ch,k,dil) + PReLU + 1x1 residual]xN -> Conv1D(ch->1,k=1)
#
# Buffer-at-a-time: prepend state, valid convolution, slice state back.
# Each layer: full = cat([state, h])           <- prepend old samples
#             activated = PReLU(Conv(full))      <- valid conv -> N outputs
#             state[:] = full[:, :, -rf:]        <- save last rf for next call
#             h = h + Conv1x1(activated)         <- residual connection
# ---------------------------------------------------------------------------
class StatefulTCN(nn.Module):
    def __init__(self, channels=16, kernel_size=3, num_layers=4):
        super().__init__()
        self.channels = channels
        self.input_conv = nn.Conv1d(1, channels, 1, bias=True)
        self.output_conv = nn.Conv1d(channels, 1, 1, bias=True)

        dilations = [2**i for i in range(num_layers)]
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.state_sizes = []
        # Store receptive fields as Python ints -- NOT tensors.
        # CoreML converter fails if negative slice uses a tensor expression.
        self.rfs: list[int] = []

        for d in dilations:
            rf = (kernel_size - 1) * d + 1  # Python int
            self.state_sizes.append(rf)
            self.rfs.append(rf)
            self.convs.append(nn.Conv1d(channels, channels, kernel_size, dilation=d, bias=True))
            self.activations.append(nn.PReLU(num_parameters=channels))
            self.residual_convs.append(nn.Conv1d(channels, channels, 1, bias=True))

        for i, sz in enumerate(self.state_sizes):
            self.register_buffer(f"layer_state_{i}", torch.zeros(1, channels, sz))

    def forward(self, x):
        h = self.input_conv(x)  # (1, channels, N)
        for i, (conv, act, res_conv) in enumerate(
            zip(self.convs, self.activations, self.residual_convs)
        ):
            state = getattr(self, f"layer_state_{i}")
            rf = self.rfs[i]  # Python int
            full = torch.cat([state[:, :, 1:], h], dim=2)  # (1, ch, rf-1+N)
            activated = act(conv(full))                      # valid conv -> (1, ch, N)
            state[:] = full[:, :, -rf:]                      # save last rf samples
            h = h + res_conv(activated)                      # residual connection
        return self.output_conv(h)  # (1, 1, N)


# ---------------------------------------------------------------------------
# Model 3: Stateful WaveNet
# Matches RTNeural-NAM architecture (wavenet_layer.hpp):
#   Conv1D(1->ch,k=1) input projection
#   Per layer:
#     activated = Tanh(Conv(full))           <- dilated conv + activation
#     skip_sum += skip_conv(activated)       <- skip connection (accumulator)
#     h = h + res_conv(activated)            <- residual connection
#   Output: Conv1x1(skip_sum)
#
# Buffer-at-a-time: same prepend-convolve-slice pattern as TCN.
# Note: RTNeural-NAM uses Tanh only (no gated tanh*sigmoid). We match this.
# ---------------------------------------------------------------------------
class StatefulWaveNet(nn.Module):
    def __init__(self, channels=8, kernel_size=2, num_layers=3):
        super().__init__()
        self.channels = channels
        self.input_conv = nn.Conv1d(1, channels, 1, bias=True)

        dilations = [2**i for i in range(num_layers)]
        self.convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.state_sizes = []
        self.rfs: list[int] = []

        for d in dilations:
            rf = (kernel_size - 1) * d + 1  # Python int
            self.state_sizes.append(rf)
            self.rfs.append(rf)
            self.convs.append(nn.Conv1d(channels, channels, kernel_size, dilation=d, bias=True))
            self.residual_convs.append(nn.Conv1d(channels, channels, 1, bias=True))
            self.skip_convs.append(nn.Conv1d(channels, channels, 1, bias=False))

        # Output projection from skip accumulator
        self.output_conv = nn.Conv1d(channels, 1, 1, bias=True)

        # One state buffer per layer
        for i, sz in enumerate(self.state_sizes):
            self.register_buffer(f"layer_state_{i}", torch.zeros(1, channels, sz))

    def forward(self, x):
        h = self.input_conv(x)  # (1, channels, N)
        skip_sum = torch.zeros_like(h)

        for i, (conv, res_conv, skip_conv) in enumerate(
            zip(self.convs, self.residual_convs, self.skip_convs)
        ):
            state = getattr(self, f"layer_state_{i}")
            rf = self.rfs[i]  # Python int
            full = torch.cat([state[:, :, 1:], h], dim=2)  # (1, ch, rf-1+N)
            activated = torch.tanh(conv(full))               # valid conv -> (1, ch, N)
            state[:] = full[:, :, -rf:]                      # save last rf samples
            skip_sum = skip_sum + skip_conv(activated)       # skip connection
            h = h + res_conv(activated)                       # residual connection

        return self.output_conv(skip_sum)  # (1, 1, N)


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def export_coreml(model, name, states, out_dir):
    """Export to CoreML .mlpackage with stateful buffers and variable seq_len."""
    model.eval()
    # Trace at size 1 -- tracing at larger sizes bakes in concrete shapes
    # that break RangeDim.  TracerWarning about mismatched outputs between
    # trace runs is expected (state changes between runs) and harmless.
    traced = torch.jit.trace(model, torch.randn(1, 1, 1))

    seq_dim = ct.RangeDim(lower_bound=1, upper_bound=2048, default=128)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 1, seq_dim))],
        outputs=[ct.TensorType(name="y")],
        states=states,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.macOS15,
    )

    pkg_path = out_dir / f"{name}.mlpackage"
    mlmodel.save(str(pkg_path))
    print(f"  CoreML: {pkg_path}")

    # Validate write_state in MIL
    prog = mlmodel._mil_program
    has_write_state = False
    for func in prog.functions.values():
        for op in func.operations:
            if op.op_type == "coreml_update_state":
                has_write_state = True
                break
    if not has_write_state:
        spec_str = str(mlmodel.get_spec())
        has_write_state = "updateState" in spec_str or "write_state" in spec_str

    if has_write_state:
        print(f"    write_state ops found in MIL")
    else:
        print(f"    WARNING: write_state NOT found -- state may not persist!")

    # Validate state carries across calls with different buffer sizes
    state = mlmodel.make_state()
    x1 = np.random.randn(1, 1, 32).astype(np.float32)
    x2 = np.random.randn(1, 1, 32).astype(np.float32)
    mlmodel.predict({"x": x1}, state=state)
    y2_carried = mlmodel.predict({"x": x2}, state=state)
    fresh_state = mlmodel.make_state()
    y2_fresh = mlmodel.predict({"x": x2}, state=fresh_state)
    diff = np.max(np.abs(y2_carried["y"] - y2_fresh["y"]))
    if diff > 1e-5:
        print(f"    State verified: diff={diff:.6f}")
    else:
        print(f"    WARNING: State NOT carrying! diff={diff:.8f}")

    # Validate different buffer sizes work
    for buf_sz in [1, 32, 128]:
        try:
            test_state = mlmodel.make_state()
            test_x = np.random.randn(1, 1, buf_sz).astype(np.float32)
            test_y = mlmodel.predict({"x": test_x}, state=test_state)
            assert test_y["y"].shape[-1] == buf_sz, (
                f"Output shape mismatch: expected {buf_sz}, got {test_y['y'].shape[-1]}"
            )
            print(f"    Buffer size {buf_sz}: OK (shape {test_y['y'].shape})")
        except Exception as e:
            print(f"    Buffer size {buf_sz}: FAILED -- {e}")

    # Compile to .mlmodelc
    compiled_dir = out_dir / f"{name}.mlmodelc"
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(out_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    COMPILE FAILED: {result.stderr}")
    elif compiled_dir.exists():
        print(f"    Compiled: {compiled_dir}")
    else:
        print(f"    Compiled (check {out_dir} for .mlmodelc)")

    return mlmodel


def export_rtneural_json(model, name, out_dir):
    """Export weights in RTNeural-compatible JSON format."""
    model.eval()
    weights_dict = {}
    for param_name, param in model.named_parameters():
        weights_dict[param_name] = param.detach().cpu().numpy().tolist()
    meta = {"model_type": name, "params": count_params(model)}
    weights_dict["__meta__"] = meta
    json_path = out_dir / f"{name}_weights.json"
    with open(json_path, "w") as f:
        json.dump(weights_dict, f)
    print(f"  RTNeural JSON: {json_path}")


def export_onnx_lstm(model, name, out_dir):
    """Export LSTM to ONNX with explicit h_in/c_in inputs and h_out/c_out outputs.

    Follows the iPlug2OnnxRuntime pattern: the C++ side manages state tensors
    and feeds them back as inputs each call, with output state pointing to the
    same backing memory.
    """
    model.eval()
    for buf_name, buf in model.named_buffers():
        buf.zero_()

    h = model.hidden

    class LSTMExplicitState(nn.Module):
        """Wrapper that takes (x, h_in, c_in) -> (y, h_out, c_out)."""
        def __init__(self, lstm_model):
            super().__init__()
            self.lstm = lstm_model.lstm
            self.dense = lstm_model.dense

        def forward(self, x, h_in, c_in):
            # x: (1, 1, N), h_in: (1, 1, H), c_in: (1, 1, H)
            out, (h_out, c_out) = self.lstm(x.transpose(1, 2), (h_in, c_in))
            y = self.dense(out).transpose(1, 2)  # (1, 1, N)
            return y, h_out, c_out

    wrapper = LSTMExplicitState(model)
    wrapper.eval()

    dummy_x = torch.randn(1, 1, 1)
    dummy_h = torch.zeros(1, 1, h)
    dummy_c = torch.zeros(1, 1, h)

    onnx_path = out_dir / f"{name}.onnx"
    try:
        torch.onnx.export(
            wrapper, (dummy_x, dummy_h, dummy_c), str(onnx_path),
            input_names=["x", "h_in", "c_in"],
            output_names=["y", "h_out", "c_out"],
            opset_version=18,
            dynamo=False,  # Force legacy export — dynamo ignores dynamic_axes
            dynamic_axes={
                "x": {2: "seq_len"},
                "y": {2: "seq_len"},
            },
        )
        print(f"  ONNX: {onnx_path}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")


def export_onnx_conv(model, name, out_dir):
    """Export TCN/WaveNet to ONNX with dynamic sequence dimension.

    State is managed internally by the model (registered buffers).
    The ONNX graph includes the state update operations.
    """
    model.eval()
    for buf_name, buf in model.named_buffers():
        buf.zero_()

    dummy = torch.randn(1, 1, 1)
    onnx_path = out_dir / f"{name}.onnx"
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["x"], output_names=["y"],
            opset_version=18,
            dynamo=False,  # Force legacy export — dynamo ignores dynamic_axes
            dynamic_axes={
                "x": {2: "seq_len"},
                "y": {2: "seq_len"},
            },
        )
        print(f"  ONNX: {onnx_path}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")


def export_torchscript(model, name, out_dir):
    """Export to TorchScript .pt format with variable sequence length support.

    Trace at a representative size.  The traced model will handle different
    seq_len values because all ops (conv, cat, LSTM) support dynamic sizes.
    """
    model.eval()
    for buf_name, buf in model.named_buffers():
        buf.zero_()
    # Trace at size 128 -- a representative buffer size.
    # Tracing runs a forward pass which mutates state buffers.
    traced = torch.jit.trace(model, torch.randn(1, 1, 128))
    # Zero all state buffers on the traced model BEFORE saving so the .pt
    # file starts from clean zero state (not post-trace state).
    with torch.no_grad():
        for buf_name, buf in traced.named_buffers():
            buf.zero_()
    pt_path = out_dir / f"{name}.pt"
    traced.save(str(pt_path))
    print(f"  TorchScript: {pt_path}")


def generate_reference_output(model, name, out_dir, num_samples=100):
    """Generate reference outputs for C++ validation.

    Processes all num_samples as ONE buffer (buffer-at-a-time), not
    sample-by-sample.  Also generates a sample-by-sample reference
    for cross-validation.
    """
    model.eval()
    for buf_name, buf in model.named_buffers():
        buf.zero_()

    rng = np.random.RandomState(42)
    inputs = rng.randn(num_samples).astype(np.float32)

    # --- Buffer-at-a-time reference (primary) ---
    # Reset state
    for buf_name, buf in model.named_buffers():
        buf.zero_()
    with torch.no_grad():
        x = torch.tensor(inputs, dtype=torch.float32).reshape(1, 1, -1)
        y = model(x)
        buffer_outputs = y.squeeze().tolist()
    # Ensure it's always a list even for single-sample edge case
    if isinstance(buffer_outputs, float):
        buffer_outputs = [buffer_outputs]

    # --- Sample-by-sample reference (cross-validation) ---
    for buf_name, buf in model.named_buffers():
        buf.zero_()
    sample_outputs = []
    with torch.no_grad():
        for i in range(num_samples):
            x = torch.tensor([[[inputs[i]]]], dtype=torch.float32)
            y = model(x)
            sample_outputs.append(y.item())

    # Verify buffer == sample-by-sample (they must be identical)
    max_diff = max(abs(a - b) for a, b in zip(buffer_outputs, sample_outputs))
    if max_diff > 1e-5:
        print(f"    WARNING: buffer vs sample-by-sample diff = {max_diff:.8f}")
    else:
        print(f"    Buffer == sample-by-sample: max_diff={max_diff:.2e}")

    ref_path = out_dir / f"{name}_reference.json"
    with open(ref_path, "w") as f:
        json.dump({
            "inputs": inputs.tolist(),
            "outputs": buffer_outputs,
            "outputs_sample_by_sample": sample_outputs,
            "num_samples": num_samples,
            "processing_mode": "buffer_at_a_time",
        }, f)
    print(f"  Reference: {ref_path} ({num_samples} samples, buffer-at-a-time)")


# ---------------------------------------------------------------------------
# Export pipeline for each model type + size
# ---------------------------------------------------------------------------
def export_model(arch_name, size_name, model, states_fn, is_lstm=False):
    """Export a single model to all formats."""
    name = f"stateful_{arch_name}_{size_name}"
    out_dir = MODELS_DIR / arch_name / size_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  --- {arch_name}/{size_name}: {count_params(model):,} params ---")

    states = states_fn(model)
    export_coreml(model, name, states, out_dir)

    # Reset state for other exports
    for buf_name, buf in model.named_buffers():
        buf.zero_()

    export_rtneural_json(model, name, out_dir)

    # ONNX: LSTM uses explicit state I/O; TCN/WaveNet use internal state
    if is_lstm:
        export_onnx_lstm(model, name, out_dir)
    else:
        export_onnx_conv(model, name, out_dir)

    export_torchscript(model, name, out_dir)
    generate_reference_output(model, name, out_dir)


def lstm_states(model):
    h = model.hidden
    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, 1, h)), name="hidden_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, 1, h)), name="cell_state"),
    ]


def tcn_states(model):
    ch = model.channels
    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, ch, sz)), name=f"layer_state_{i}")
        for i, sz in enumerate(model.state_sizes)
    ]


def wavenet_states(model):
    ch = model.channels
    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, ch, sz)), name=f"layer_state_{i}")
        for i, sz in enumerate(model.state_sizes)
    ]


def main():
    print("=" * 60)
    print("Neural Audio Benchmark -- Model Export Pipeline")
    print("Processing mode: BUFFER-AT-A-TIME")
    print(f"Config: {CONFIG_PATH}")
    print("=" * 60)

    model_cfg = CONFIG["models"]
    size_cfg = CONFIG.get("model_sizes", {"small": True, "medium": True, "large": True})

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
        model = StatefulLSTM(hidden=cfg["hidden"])
        export_model("lstm", size_name, model, lstm_states, is_lstm=True)

        # TCN
        torch.manual_seed(SEED)
        cfg = model_cfg["tcn"][size_name]
        model = StatefulTCN(channels=cfg["channels"], kernel_size=cfg["kernel_size"],
                            num_layers=cfg["num_layers"])
        export_model("tcn", size_name, model, tcn_states)

        # WaveNet
        torch.manual_seed(SEED)
        cfg = model_cfg["wavenet"][size_name]
        model = StatefulWaveNet(channels=cfg["channels"], kernel_size=cfg["kernel_size"],
                                num_layers=cfg["num_layers"])
        export_model("wavenet", size_name, model, wavenet_states)

    print(f"\n{'=' * 60}")
    print("All models exported successfully!")
    print(f"{'=' * 60}")

    # Summary
    print("\nExported files:")
    for arch in ["lstm", "tcn", "wavenet"]:
        for size in ["small", "medium", "large"]:
            p = MODELS_DIR / arch / size
            if p.exists():
                total = sum(
                    ff.stat().st_size for ff in p.rglob("*") if ff.is_file()
                )
                print(f"  {arch}/{size}: {total:,} bytes")


if __name__ == "__main__":
    main()
