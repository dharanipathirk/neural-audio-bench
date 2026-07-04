// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <tracktion_engine/tracktion_engine.h>
#include "../TimingLogger.h"
#include "../BenchmarkConfig.h"
#include <string>
#include <memory>

#if HAS_LIBTORCH
#include <torch/script.h>
#include <ATen/Parallel.h>
#endif

#if HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
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
// Tracktion Engine plugins for LibTorch and ONNX backends
// ---------------------------------------------------------------------------
namespace tracktion { namespace engine {

class DirectLibTorchPlugin : public Plugin
{
public:
    DirectLibTorchPlugin(PluginCreationInfo info);
    ~DirectLibTorchPlugin() override;

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

    void setModelPath(const std::string& path) { modelPath = path; }
    void setPluginName(const juce::String& name) { pluginName = name; }
    TimingLogger& getTimingLogger() { return timingLogger; }

private:
    LibTorchEngine engine;
    std::string modelPath;
    juce::String pluginName{"Direct_LibTorch"};
    TimingLogger timingLogger;
};

class DirectOnnxPlugin : public Plugin
{
public:
    DirectOnnxPlugin(PluginCreationInfo info);
    ~DirectOnnxPlugin() override;

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

    void setModelPath(const std::string& path) { modelPath = path; }
    void setPluginName(const juce::String& name) { pluginName = name; }
    TimingLogger& getTimingLogger() { return timingLogger; }

private:
    OnnxRuntimeEngine engine;
    std::string modelPath;
    juce::String pluginName{"Direct_ONNX"};
    TimingLogger timingLogger;
};

}} // namespace tracktion::engine
