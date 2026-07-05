// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "InferenceBackend.h"
#include <memory>
#include <string>
#include <vector>

#if HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// ---------------------------------------------------------------------------
// Direct ONNX Runtime inference engine — buffer-at-a-time processing.
//
// For LSTM: Uses explicit state I/O (x, h_in, c_in) -> (y, h_out, c_out).
// State buffers are shared between input and output (iPlug2OnnxRuntime
// pattern) so state persists across Run() calls with zero copy.
//
// For TCN/WaveNet: State is managed internally by the ONNX model.
// Single Run() per buffer with dynamic sequence length.
// ---------------------------------------------------------------------------
class OnnxRuntimeEngine
{
public:
    bool initialize(const std::string& onnxPath);
    void processBlock(const float* input, float* output, int numSamples);
    void resetState();
    bool isValid() const { return valid; }

    void processChunk(const float* input, float* output, int numSamples);

private:
    bool valid = false;
    bool hasExplicitState = false;  // true for LSTM (has h_in/c_in/h_out/c_out)
    int maxBufferSize = 2048;

#if HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memInfo;

    // Pre-allocated buffers (avoid per-call allocation)
    std::vector<float> xBuf;
    std::vector<float> yBuf;
    std::vector<float> hBuf;  // LSTM hidden state (shared in/out)
    std::vector<float> cBuf;  // LSTM cell state (shared in/out)
    int hiddenSize = 0;

    // Input/output names discovered from the model
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
#endif
};

// ---------------------------------------------------------------------------
// Direct_ONNX backend adapter. Not real-time safe (allocates per call),
// so it is excluded from contention automatically.
// ---------------------------------------------------------------------------
class OnnxRuntimeBackend : public InferenceBackend
{
public:
    bool prepare(const PrepareContext& ctx) override;
    void process(const float* in, float* out, int n) noexcept override { engine.processBlock(in, out, n); }
    void reset() noexcept override { engine.resetState(); }

    const char* name() const override { return "Direct_ONNX"; }
    bool isRealtimeSafe() const override { return false; }
    const char* requiredFormat() const override { return "onnx"; }

private:
    OnnxRuntimeEngine engine;
};
