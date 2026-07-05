// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../BenchmarkConfig.h"
#include "../TimingLogger.h"
#include "../plugins/ContentionPlugins.h"
#include "../core/ModelManifest.h"   // ModelSpec, findModelSpec

#include <tracktion_engine/tracktion_engine.h>

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace te = tracktion::engine;

// ---------------------------------------------------------------------------
// Builds a realistic 36-track mixing session using Apple system Audio Units
// as described in benchmark_session_layout.md.
//
// 24 source tracks + group buses + FX returns + mix bus.
// Track 15 (Electric Lead Guitar) hosts the neural model under test.
// All other tracks use macOS system AUs (AUParametricEQ, AUDynamicsProcessor,
// AUMatrixReverb, etc.) for realistic CPU contention.
// ---------------------------------------------------------------------------

// Forward from EditBuilder.h
struct SessionTimingInfo;

class AUSessionBuilder
{
public:
    AUSessionBuilder(te::Engine& engine, const std::string& modelDir,
                     const std::vector<ModelSpec>& specs);
    ~AUSessionBuilder();

    // Scan for required Apple system AUs. Returns true if all found.
    // Must be called before buildSession().
    bool scanForRequiredAUs();

    // Build the full 32-track session.
    // activeTracks: how many conventional tracks are active (0-24).
    // Track 15 (neural) is always active.
    SessionTimingInfo buildSession(
        te::Edit& edit,
        BackendType backend,
        ModelType model,
        ModelSize size,
        int activeTracks,
        double sampleRate);

    // Report which AUs were found/missing.
    void printAUStatus() const;

private:
    te::Engine& engine;
    std::string modelDir;
    const std::vector<ModelSpec>& specs;

    // Plugin descriptions for each AU type, keyed by AU name
    std::map<std::string, juce::PluginDescription> auMap;
    bool scanned = false;

    // Noise WAV file for source audio
    juce::File noiseWavFile;
    void ensureNoiseWavFile(double durationSeconds, double sampleRate);

    // Insert an Apple system AU by name. Returns the plugin, or nullptr if AU not found.
    te::Plugin::Ptr insertAU(te::Edit& edit, te::AudioTrack& track, const std::string& auName, int index);

    // Insert an aux send plugin (returns pointer for parameter access)
    te::Plugin::Ptr insertAuxSend(te::Edit& edit, te::AudioTrack& track, int busNumber, float gainDb);

    // Add audio clip to a track
    void addAudioClip(te::AudioTrack& track, double durationSeconds);

    // Build the chain for a specific track number (per layout doc)
    void buildTrackChain(te::Edit& edit, te::AudioTrack& track, int trackNumber);

    // Build bus chains
    void buildDrumBus(te::Edit& edit, te::AudioTrack& bus);
    void buildBassBus(te::Edit& edit, te::AudioTrack& bus);
    void buildGuitarBus(te::Edit& edit, te::AudioTrack& bus);
    void buildKeysBus(te::Edit& edit, te::AudioTrack& bus);
    void buildVocalBus(te::Edit& edit, te::AudioTrack& bus);
    void buildMusicBus(te::Edit& edit, te::AudioTrack& bus);
    void buildMixBus(te::Edit& edit, te::AudioTrack& bus);

    // Build FX return chains
    void buildFXReturn(te::Edit& edit, te::AudioTrack& track, int fxNumber);

    // Neural model insertion (reused from EditBuilder logic)
    TimingLogger* addNeuralPlugin(te::AudioTrack& track, BackendType backend,
                                   ModelType model, ModelSize size, double clipDuration,
                                   SessionTimingInfo& sessionInfo);

    // Add CallbackStartPlugin to a track, connected to shared timer + thread ID logger
    void addCallbackStart(te::AudioTrack& track, CallbackTimer* timer, ThreadIDLogger* threadIdLogger);

    void registerPluginTypes();
    bool pluginsRegistered = false;

    // Required Apple system AU names
    static const std::vector<std::string>& requiredAUNames();
};
