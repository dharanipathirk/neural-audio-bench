# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
Benchmark model definitions and CoreML state descriptors.

BUFFER-AT-A-TIME processing: all models accept variable-length input
x: (1, 1, N) where N can be 1..2048.  State is carried across calls.

LSTM is sequential (LSTM -> Dense).
TCN has residual connections (h = h + Conv1x1(PReLU(Conv(state)))).
WaveNet has residual + skip connections, following RTNeural-NAM's pattern.
All backends process architecturally identical models.
"""

import torch
import torch.nn as nn


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
            zip(self.convs, self.activations, self.residual_convs, strict=False)
        ):
            state = getattr(self, f"layer_state_{i}")
            rf = self.rfs[i]  # Python int
            full = torch.cat([state[:, :, 1:], h], dim=2)  # (1, ch, rf-1+N)
            activated = act(conv(full))  # valid conv -> (1, ch, N)
            state[:] = full[:, :, -rf:]  # save last rf samples
            h = h + res_conv(activated)  # residual connection
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
            zip(self.convs, self.residual_convs, self.skip_convs, strict=False)
        ):
            state = getattr(self, f"layer_state_{i}")
            rf = self.rfs[i]  # Python int
            full = torch.cat([state[:, :, 1:], h], dim=2)  # (1, ch, rf-1+N)
            activated = torch.tanh(conv(full))  # valid conv -> (1, ch, N)
            state[:] = full[:, :, -rf:]  # save last rf samples
            skip_sum = skip_sum + skip_conv(activated)  # skip connection
            h = h + res_conv(activated)  # residual connection

        return self.output_conv(skip_sum)  # (1, 1, N)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def lstm_states(model):
    import coremltools as ct

    h = model.hidden
    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, 1, h)), name="hidden_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, 1, h)), name="cell_state"),
    ]


def tcn_states(model):
    import coremltools as ct

    ch = model.channels
    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, ch, sz)), name=f"layer_state_{i}")
        for i, sz in enumerate(model.state_sizes)
    ]


def wavenet_states(model):
    import coremltools as ct

    ch = model.channels
    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, ch, sz)), name=f"layer_state_{i}")
        for i, sz in enumerate(model.state_sizes)
    ]
