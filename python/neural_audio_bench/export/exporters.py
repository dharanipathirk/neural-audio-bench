# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Format exporters: CoreML (.mlpackage + compiled .mlmodelc), RTNeural JSON,
ONNX (explicit-state LSTM / stateless conv), and TorchScript.
"""

import json
import subprocess

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from .models import count_params


def export_coreml(model, name, states, out_dir):
    """Export to CoreML .mlpackage with stateful buffers and variable seq_len.

    Compute precision is fp16 and cannot be raised: CoreML state tensors must
    be fp16 (coremltools rejects fp32 state with "State only support fp16
    dtype"), and fp16 is also coremltools' mlprogram default. BNNSGraph
    therefore executes every weight and activation in half precision, unlike
    the fp32 ONNX/TorchScript/RTNeural paths; the manifest notes record this.
    """
    compute_precision = ct.precision.FLOAT16

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
        compute_precision=compute_precision,
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
        print("    write_state ops found in MIL")
    else:
        print("    WARNING: write_state NOT found -- state may not persist!")

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
        except Exception as e:  # noqa: BLE001 - report any per-size failure and continue
            print(f"    Buffer size {buf_sz}: FAILED -- {e}")

    # Compile to .mlmodelc
    compiled_dir = out_dir / f"{name}.mlmodelc"
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(out_dir)],
        capture_output=True,
        text=True,
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
    for _buf_name, buf in model.named_buffers():
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
            wrapper,
            (dummy_x, dummy_h, dummy_c),
            str(onnx_path),
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
    except Exception as e:  # noqa: BLE001 - ONNX export is best-effort
        print(f"  ONNX export failed: {e}")


def export_onnx_conv(model, name, out_dir):
    """Export TCN/WaveNet to ONNX with dynamic sequence dimension.

    Registered PyTorch convolution buffers are not mutable ONNX state, so
    these exports are intentionally stateless. The manifest records this
    asymmetry; stateful convolution would require explicit state I/O.
    """
    model.eval()
    for _buf_name, buf in model.named_buffers():
        buf.zero_()

    dummy = torch.randn(1, 1, 1)
    onnx_path = out_dir / f"{name}.onnx"
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["x"],
            output_names=["y"],
            opset_version=18,
            dynamo=False,  # Force legacy export — dynamo ignores dynamic_axes
            dynamic_axes={
                "x": {2: "seq_len"},
                "y": {2: "seq_len"},
            },
        )
        print(f"  ONNX: {onnx_path}")
    except Exception as e:  # noqa: BLE001 - ONNX export is best-effort
        print(f"  ONNX export failed: {e}")


def export_torchscript(model, name, out_dir):
    """Export to TorchScript .pt format with variable sequence length support.

    Trace at a representative size.  The traced model will handle different
    seq_len values because all ops (conv, cat, LSTM) support dynamic sizes.
    """
    model.eval()
    for _buf_name, buf in model.named_buffers():
        buf.zero_()
    # Trace at size 128 -- a representative buffer size.
    # Tracing runs a forward pass which mutates state buffers.
    traced = torch.jit.trace(model, torch.randn(1, 1, 128))
    # Zero all state buffers on the traced model BEFORE saving so the .pt
    # file starts from clean zero state (not post-trace state).
    with torch.no_grad():
        for _buf_name, buf in traced.named_buffers():
            buf.zero_()
    pt_path = out_dir / f"{name}.pt"
    traced.save(str(pt_path))
    print(f"  TorchScript: {pt_path}")
