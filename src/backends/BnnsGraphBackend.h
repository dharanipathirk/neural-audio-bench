// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "InferenceBackend.h"
#include <Accelerate/Accelerate.h>
#include <array>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Generic stateful BNNSGraph inference engine.
// Works with any .mlmodelc that uses the stateful pattern (InOut arguments).
// Dynamic argument discovery -- no model-specific code needed.
//
// Buffer-at-a-time: processes the full audio buffer in a single
// BNNSGraphContextExecute() call using dynamic shapes.  The CoreML model
// must be exported with ct.RangeDim on the sequence dimension.
// ---------------------------------------------------------------------------
class BNNSGraphEngine
{
public:
    BNNSGraphEngine() = default;
    ~BNNSGraphEngine() { deinitialize(); }

    bool initialize(const std::string& mlmodelcPath, int preparedBlockSize);
    void deinitialize();

    // Process a full buffer in one Execute call (buffer-at-a-time)
    void processBlock(const float* input, float* output, int numSamples);

    // Reset all state buffers to zero
    void resetState();

    bool isValid() const { return graph.data != nullptr; }

private:
    bnns_graph_t graph{};
    bnns_graph_context_t ctx{};

    size_t argCount = 0;
    std::vector<void*> buffers;
    std::vector<size_t> bufferSizes;       // current allocation sizes (bytes)
    std::vector<size_t> baseBytesPerElem;  // bytes per element (without seq_len multiplier)
    std::vector<BNNSGraphArgumentIntent> intents;

    // Tensor shape info for dynamic shape updates
    struct ArgShape {
        size_t rank = 0;
        int64_t shape[8] = {};  // static shape from graph
        int seqDim = -1;        // which dimension is the dynamic seq_len (-1 = none)
    };
    std::vector<ArgShape> argShapes;
    std::vector<bnns_graph_shape_t> dynamicShapes;
    std::vector<std::array<uint64_t, 8>> dynamicShapeData;

    size_t xIdx = 0;
    size_t yIdx = 0;
    int currentSeqLen = 0;    // last seq_len we configured

    static constexpr int kMaxBufferSize = 2048;

    char* workspace = nullptr;
    size_t wsSize = 0;

    std::vector<bnns_graph_argument_t> args;
    std::vector<const char*> argNames;

    bool setDynamicShape(int seqLen);
    bool ensureWorkspace(size_t requiredBytes);
};

// ---------------------------------------------------------------------------
// Backend adapter wrapping BNNSGraphEngine
// ---------------------------------------------------------------------------
class BnnsGraphBackend : public InferenceBackend
{
public:
    bool prepare(const PrepareContext& ctx) override;
    void process(const float* in, float* out, int n) noexcept override { engine.processBlock(in, out, n); }
    void reset() noexcept override { engine.resetState(); }
    void teardown() override { engine.deinitialize(); }

    const char* name() const override { return "BNNSGraph"; }
    bool isRealtimeSafe() const override { return true; }
    const char* requiredFormat() const override { return "coreml"; }

private:
    BNNSGraphEngine engine;
};
