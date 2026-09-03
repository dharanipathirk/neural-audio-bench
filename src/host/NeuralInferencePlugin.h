// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <tracktion_engine/tracktion_engine.h>
#include "../TimingLogger.h"
#include "../backends/InferenceBackend.h"

#include <memory>

// ---------------------------------------------------------------------------
// Unified Tracktion Engine plugin. Replaces the six per-engine plugins
// (BNNSGraphPlugin, RTNeuralPlugin, DirectLibTorchPlugin, DirectOnnxPlugin,
// AniraLibTorchHandlerPlugin, AniraOnnxHandlerPlugin). The concrete inference
// engine is supplied as an InferenceBackend via setBackend().
// ---------------------------------------------------------------------------
namespace tracktion { inline namespace engine {

class NeuralInferencePlugin : public Plugin
{
public:
    NeuralInferencePlugin(PluginCreationInfo info);
    ~NeuralInferencePlugin() override;

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
    void setBackend(std::unique_ptr<InferenceBackend> b) { backend = std::move(b); }
    void setModelSpec(const ModelSpec& s) { spec = s; }
    void setPluginName(const juce::String& name) { pluginName = name; }

    TimingLogger& getTimingLogger() { return timingLogger; }

    // Inference underrun accessors delegate to the backend (0 for non-anira).
    int getInferenceUnderruns() const { return backend ? backend->underrunCount() : 0; }
    void resetInferenceUnderruns() { if (backend) backend->resetUnderruns(); }

private:
    std::unique_ptr<InferenceBackend> backend;
    ModelSpec spec;
    juce::String pluginName{"Neural"};
    TimingLogger timingLogger;
    bool prepared = false;
};

}} // namespace tracktion::engine
