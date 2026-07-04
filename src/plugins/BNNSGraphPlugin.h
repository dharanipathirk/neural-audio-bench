// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <tracktion_engine/tracktion_engine.h>
#include <Accelerate/Accelerate.h>
#include "../TimingLogger.h"
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

    bool initialize(const std::string& mlmodelcPath);
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

    size_t xIdx = 0;
    size_t yIdx = 0;
    int currentSeqLen = 0;    // last seq_len we configured

    static constexpr int kMaxBufferSize = 2048;

    char* workspace = nullptr;
    size_t wsSize = 0;

    std::vector<bnns_graph_argument_t> args;
    std::vector<const char*> argNames;

    void setDynamicShape(int seqLen);
    void reallocBuffersForSeqLen(int seqLen);
};

// ---------------------------------------------------------------------------
// Tracktion Engine plugin wrapping BNNSGraphEngine
// ---------------------------------------------------------------------------
namespace tracktion { namespace engine {

class BNNSGraphPlugin : public Plugin
{
public:
    BNNSGraphPlugin(PluginCreationInfo info);
    ~BNNSGraphPlugin() override;

    static const char* xmlTypeName;

    juce::String getName() const override { return pluginName; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }

    void initialise(const PluginInitialisationInfo& info) override;
    void deinitialise() override;
    void applyToBuffer(const PluginRenderContext& ctx) override;
    void reset() override;

    int getNumOutputChannelsGivenInputs(int numInputs) override { return juce::jmin(numInputs, 1); }
    bool takesAudioInput() override { return true; }
    bool isSynth() override { return false; }

    // Configuration
    void setModelPath(const std::string& path) { modelPath = path; }
    void setPluginName(const juce::String& name) { pluginName = name; }

    TimingLogger& getTimingLogger() { return timingLogger; }

private:
    BNNSGraphEngine engine;
    std::string modelPath;
    juce::String pluginName{"BNNSGraph"};
    TimingLogger timingLogger;
};

}} // namespace tracktion::engine
