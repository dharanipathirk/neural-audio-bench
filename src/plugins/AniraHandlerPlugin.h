// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <tracktion_engine/tracktion_engine.h>
#include "../TimingLogger.h"
#include "../BenchmarkConfig.h"
#include <string>
#include <memory>

#if HAS_ANIRA
#include <anira/anira.h>
#endif

// ---------------------------------------------------------------------------
// Tracktion Engine plugins that use anira's InferenceHandler for LibTorch and
// ONNX inference. Unlike Direct_LibTorch/Direct_ONNX (which call the APIs
// directly on the audio thread), these route inference through anira's
// background-thread scheduler with ring buffers — the way a production plugin
// built on anira would work.
//
// This introduces additional buffering latency determined by anira's scheduler,
// but keeps the audio thread free of any inference work.
// ---------------------------------------------------------------------------

namespace tracktion { namespace engine {

class AniraLibTorchHandlerPlugin : public Plugin
{
public:
    AniraLibTorchHandlerPlugin(PluginCreationInfo info);
    ~AniraLibTorchHandlerPlugin() override;

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
    bool introduceLatency() const { return true; }

    void setModelPath(const std::string& path) { modelPath = path; }
    void setPluginName(const juce::String& name) { pluginName = name; }
    TimingLogger& getTimingLogger() { return timingLogger; }

    // Number of callbacks where inference output was not ready (background
    // thread didn't finish in time). This is anira's failure mode — the
    // audio thread doesn't miss a deadline, but the neural output is stale.
    // Callbacks where background inference output was not ready.
    int getInferenceUnderruns() const { return inferenceUnderruns; }
    void resetInferenceUnderruns() { inferenceUnderruns = 0; }

private:
    std::string modelPath;
    juce::String pluginName{"Anira_LibTorch"};
    TimingLogger timingLogger;
    int inferenceUnderruns = 0;

#if HAS_ANIRA
    std::unique_ptr<anira::InferenceConfig> config;
    std::unique_ptr<anira::PrePostProcessor> processor;
    std::unique_ptr<anira::InferenceHandler> handler;
#endif
};

class AniraOnnxHandlerPlugin : public Plugin
{
public:
    AniraOnnxHandlerPlugin(PluginCreationInfo info);
    ~AniraOnnxHandlerPlugin() override;

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
    bool introduceLatency() const { return true; }

    void setModelPath(const std::string& path) { modelPath = path; }
    void setPluginName(const juce::String& name) { pluginName = name; }
    TimingLogger& getTimingLogger() { return timingLogger; }
    int getInferenceUnderruns() const { return inferenceUnderruns; }
    void resetInferenceUnderruns() { inferenceUnderruns = 0; }

private:
    std::string modelPath;
    juce::String pluginName{"Anira_ONNX"};
    TimingLogger timingLogger;
    int inferenceUnderruns = 0;

#if HAS_ANIRA
    std::unique_ptr<anira::InferenceConfig> config;
    std::unique_ptr<anira::PrePostProcessor> processor;
    std::unique_ptr<anira::InferenceHandler> handler;

    bool hasExplicitStateLstm = false;
    int hiddenSize = 0;
    std::vector<float> audioInputScratch;
    std::vector<float> hStateIn;
    std::vector<float> cStateIn;
    std::vector<float> hStateOut;
    std::vector<float> cStateOut;
#endif
};

}} // namespace tracktion::engine
