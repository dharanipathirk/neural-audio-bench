// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../BenchmarkConfig.h"
#include "../TimingLogger.h"

#include "../plugins/ContentionPlugins.h"
#include "../core/ModelManifest.h"   // ModelSpec, findModelSpec
#include "AUSessionBuilder.h"

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
// Shared helper used by both EditBuilder and AUSessionBuilder: create the
// unified NeuralInferencePlugin on a track, wire it to a registry backend and
// the given model spec, and register its underrun getter/resetter. Returns the
// plugin's neural TimingLogger (nullptr if the spec/backend is unavailable).
// The plugin is inserted at position 0 (front of the track).
// ---------------------------------------------------------------------------
TimingLogger* attachNeuralPlugin(te::AudioTrack& track, BackendType backend,
                                 const ModelSpec* spec, const juce::String& pluginName,
                                 SessionTimingInfo& sessionInfo);

// ---------------------------------------------------------------------------
// Builds Tracktion Engine Edits for contention benchmarking.
// ---------------------------------------------------------------------------
class EditBuilder
{
public:
    EditBuilder(te::Engine& engine, const std::string& modelDir,
                const std::vector<ModelSpec>& specs);
    ~EditBuilder();

    SessionTimingInfo buildDimensionA(
        te::Edit& edit,
        BackendType backend,
        ModelType model,
        ModelSize size,
        int activeTracks,
        double sampleRate,
        int numTracks = 36);

    SessionTimingInfo buildDimensionB(
        te::Edit& edit,
        BackendType backend,
        ModelType model,
        ModelSize size,
        int instanceCount,
        double sampleRate);

    // Dimension C: neural track serial depth.
    // Single track with the neural plugin surrounded by a configurable chain.
    // depth presets: "bare" (1), "channel_strip" (3), "mix_fx" (5), "heavy_chain" (7)
    SessionTimingInfo buildDimensionC(
        te::Edit& edit,
        BackendType backend,
        ModelType model,
        ModelSize size,
        int depth,
        double sampleRate);

private:
    te::Engine& engine;
    std::string modelDir;
    const std::vector<ModelSpec>& specs;

    juce::File noiseWavFile;
    void ensureNoiseWavFile(double durationSeconds, double sampleRate);

    void addAudioClip(te::AudioTrack& track, double durationSeconds);
    void addConventionalDSP(te::AudioTrack& track, double clipDuration);
    TimingLogger* addNeuralPlugin(te::AudioTrack& track, BackendType backend,
                                   ModelType model, ModelSize size, double clipDuration,
                                   SessionTimingInfo& sessionInfo);

    // Add CallbackStartPlugin to a track, connected to shared timer + thread ID logger
    void addCallbackStart(te::AudioTrack& track, CallbackTimer* timer, ThreadIDLogger* threadIdLogger);

    // Add CallbackEndPlugin to master bus, returns its TimingLogger
    TimingLogger* addCallbackEnd(te::Edit& edit, CallbackTimer* timer);

    void registerPluginTypes();
    bool pluginsRegistered = false;
};

// ---------------------------------------------------------------------------
// Runs the full contention benchmark suite using real-time CoreAudio playback.
// ---------------------------------------------------------------------------
class ContentionBenchmark
{
public:
    ContentionBenchmark(te::Engine& engine, const std::string& modelDir, const std::string& configPath,
                        const std::vector<ModelSpec>& specs)
        : builder(engine, modelDir, specs), auBuilder(engine, modelDir, specs),
          engine(engine), modelDir(modelDir), configPath(configPath) {}

    void runDimensionA(FILE* csvFile);
    void runDimensionB(FILE* csvFile);
    void runDimensionC(FILE* csvFile);
    void runAll(FILE* csvFile);

private:
    EditBuilder builder;
    AUSessionBuilder auBuilder;
    bool auScanned = false;
    bool auAvailable = false;
    te::Engine& engine;
    std::string modelDir;
    std::string configPath;

    void runSingleConfig(
        FILE* csvFile,
        const char* dimension,
        BackendType backend,
        ModelType model,
        ModelSize size,
        int bufferSize,
        int contentionLevel,
        int instanceCount,
        int rep);
};
