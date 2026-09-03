// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../BenchmarkConfig.h"
#include "../TimingLogger.h"

#include "../plugins/ContentionPlugins.h"
#include "../core/ModelManifest.h"   // ModelSpec, findModelSpec

#include <tracktion_engine/tracktion_engine.h>

#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace te = tracktion::engine;

// Result from building an Edit session — contains all timing loggers needed
struct SessionTimingInfo
{
    TimingLogger* neuralLogger = nullptr;             // per-plugin neural inference timing
    std::vector<TimingLogger*> neuralLoggers;          // (dim_b: one per instance)
    TimingLogger* callbackLogger = nullptr;            // full audio callback timing
    std::unique_ptr<CallbackTimer> callbackTimer;      // shared timer (owns lifetime)
    std::unique_ptr<ThreadIDLogger> threadIdLogger;    // audio thread IDs (owns lifetime)

    // For anira backends: functions to reset and query inference underruns.
    // Each anira plugin instance adds a getter/resetter pair.
    std::vector<std::function<int()>> inferenceUnderrunGetters;
    std::vector<std::function<void()>> inferenceUnderrunResetters;

    int getTotalInferenceUnderruns() const
    {
        int total = 0;
        for (auto& g : inferenceUnderrunGetters)
            total += g();
        return total;
    }

    void resetInferenceUnderruns()
    {
        for (auto& r : inferenceUnderrunResetters)
            r();
    }
};

// ---------------------------------------------------------------------------
// Shared helper used by EditBuilder and the scenario session builders: create
// the unified NeuralInferencePlugin on a track, wire it to a registry backend
// and the given model spec, and register its underrun getter/resetter. Returns
// the plugin's neural TimingLogger (nullptr if the spec/backend is unavailable).
// The plugin is inserted at position 0 (front of the track).
// ---------------------------------------------------------------------------
TimingLogger* attachNeuralPlugin(te::AudioTrack& track, const std::string& backend,
                                 const ModelSpec& spec, const juce::String& pluginName,
                                 SessionTimingInfo& sessionInfo);

// ---------------------------------------------------------------------------
// Shared session-building toolbox for the contention scenarios. Each scenario
// (see src/scenarios/) builds its Tracktion Engine layout by calling these
// helpers; the helpers themselves are backend/model agnostic. Session-layout
// logic lives in the scenarios, not here.
// ---------------------------------------------------------------------------
class EditBuilder
{
public:
    EditBuilder(te::Engine& engine, const std::vector<ModelSpec>& specs);
    ~EditBuilder();

    // Lazily create (once) the shared noise WAV clip source used by every
    // scenario. Idempotent after the first call.
    void ensureNoiseWavFile(double durationSeconds, double sampleRate);

    void addAudioClip(te::AudioTrack& track, double durationSeconds);
    void addConventionalDSP(te::AudioTrack& track, double clipDuration);
    TimingLogger* addNeuralPlugin(te::AudioTrack& track, const std::string& backend,
                                   const ModelSpec& spec, double clipDuration,
                                   SessionTimingInfo& sessionInfo);

    // Add CallbackStartPlugin to a track, connected to shared timer + thread ID logger
    void addCallbackStart(te::AudioTrack& track, CallbackTimer* timer, ThreadIDLogger* threadIdLogger);

    // Add CallbackEndPlugin to master bus, returns its TimingLogger
    TimingLogger* addCallbackEnd(te::Edit& edit, CallbackTimer* timer);

    // Register the built-in contention/neural/callback plugin types. Idempotent.
    void registerPluginTypes();

private:
    te::Engine& engine;
    const std::vector<ModelSpec>& specs;

    juce::File noiseWavFile;
    bool pluginsRegistered = false;
};
