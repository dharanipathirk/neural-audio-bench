// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "InferenceBackend.h"
#include <string>

#if HAS_LIBTORCH
#include <torch/script.h>
#include <ATen/Parallel.h>
#endif

// ---------------------------------------------------------------------------
// Direct LibTorch inference engine — buffer-at-a-time processing.
// Processes the full audio buffer in a single forward() call.
// State persists in the TorchScript model's registered buffers.
// ---------------------------------------------------------------------------
class LibTorchEngine
{
public:
    bool initialize(const std::string& torchscriptPath);
    void processBlock(const float* input, float* output, int numSamples);
    void resetState();
    bool isValid() const { return valid; }

private:
    bool valid = false;
    std::string modelPath;
#if HAS_LIBTORCH
    torch::jit::script::Module model;
#endif
};

// ---------------------------------------------------------------------------
// Direct_LibTorch backend adapter. Not real-time safe (allocates per call),
// so it is excluded from contention automatically.
// ---------------------------------------------------------------------------
class LibTorchBackend : public InferenceBackend
{
public:
    bool prepare(const PrepareContext& ctx) override;
    void process(const float* in, float* out, int n) noexcept override { engine.processBlock(in, out, n); }
    void reset() noexcept override { engine.resetState(); }

    const char* name() const override { return "Direct_LibTorch"; }
    bool isRealtimeSafe() const override { return false; }
    const char* requiredFormat() const override { return "torchscript"; }

private:
    LibTorchEngine engine;
};
