// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "AUSessionBuilder.h"
#include "EditBuilder.h"  // for SessionTimingInfo
#include "../plugins/BNNSGraphPlugin.h"
#include "../plugins/RTNeuralPlugin.h"
#include "../plugins/AniraPlugin.h"
#include "../plugins/AniraHandlerPlugin.h"

#include <random>

// ---------------------------------------------------------------------------
// AUSessionBuilder
// ---------------------------------------------------------------------------

AUSessionBuilder::AUSessionBuilder(te::Engine& engine, const std::string& modelDir)
    : engine(engine), modelDir(modelDir) {}

AUSessionBuilder::~AUSessionBuilder()
{
    if (noiseWavFile.existsAsFile())
        noiseWavFile.deleteFile();
}

const std::vector<std::string>& AUSessionBuilder::requiredAUNames()
{
    static const std::vector<std::string> names = {
        "AUHipass",
        "AULowpass",
        "AUBandpass",
        "AUHighShelfFilter",
        "AULowShelfFilter",
        "AUParametricEQ",
        "AUNBandEQ",
        "AUDynamicsProcessor",
        "AUMultibandCompressor",
        "AUPeakLimiter",
        "AUDelay",
        "AUSampleDelay",
        "AUMatrixReverb",
        "AUDistortion",
    };
    return names;
}

bool AUSessionBuilder::scanForRequiredAUs()
{
    if (scanned) return !auMap.empty();

    auto& pm = engine.getPluginManager();

    // Trigger a plugin scan if the known list is empty
    auto knownTypes = pm.knownPluginList.getTypes();
    if (knownTypes.isEmpty())
    {
        fprintf(stderr, "  Scanning for Audio Unit plugins...\n");

        // Apple system AUs are registered AudioComponents, not .component bundles.
        // PluginDirectoryScanner scans filesystem paths and finds nothing.
        // Use searchPathsForPlugins() which calls AudioComponentFindNext internally.
        auto& formatManager = pm.pluginFormatManager;
        for (int i = 0; i < formatManager.getNumFormats(); i++)
        {
            auto* format = formatManager.getFormat(i);
            if (format->getName() == "AudioUnit")
            {
                auto auIdentifiers = format->searchPathsForPlugins(
                    juce::FileSearchPath(), true, false);

                for (auto& id : auIdentifiers)
                {
                    juce::OwnedArray<juce::PluginDescription> results;
                    format->findAllTypesForFile(results, id);
                    for (auto* desc : results)
                        pm.knownPluginList.addType(*desc);
                }
                break;
            }
        }

        knownTypes = pm.knownPluginList.getTypes();
        fprintf(stderr, "  Found %d Audio Unit plugins\n", knownTypes.size());
    }

    // Find each required AU
    for (auto& name : requiredAUNames())
    {
        bool found = false;
        for (auto& desc : knownTypes)
        {
            if (desc.name == juce::String(name) &&
                desc.manufacturerName == "Apple")
            {
                auMap[name] = desc;
                found = true;
                break;
            }
        }
        if (!found)
            fprintf(stderr, "  WARNING: System AU '%s' not found\n", name.c_str());
    }

    scanned = true;
    fprintf(stderr, "  AU scan complete: %zu of %zu required AUs found\n",
            auMap.size(), requiredAUNames().size());

    return auMap.size() == requiredAUNames().size();
}

void AUSessionBuilder::printAUStatus() const
{
    fprintf(stderr, "\nSystem AU availability:\n");
    for (auto& name : requiredAUNames())
    {
        bool found = auMap.count(name) > 0;
        fprintf(stderr, "  %s: %s\n", name.c_str(), found ? "OK" : "MISSING");
    }
}

void AUSessionBuilder::ensureNoiseWavFile(double durationSeconds, double sampleRate)
{
    if (noiseWavFile.existsAsFile())
        return;

    const int numSamples = static_cast<int>(sampleRate * durationSeconds);
    juce::AudioBuffer<float> buffer(1, numSamples);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto* data = buffer.getWritePointer(0);
    for (int i = 0; i < numSamples; i++)
        data[i] = dist(rng);

    auto tempDir = juce::File::getSpecialLocation(juce::File::tempDirectory)
                       .getChildFile("nab-engine");
    tempDir.createDirectory();
    noiseWavFile = tempDir.getChildFile("au_session_noise.wav");
    noiseWavFile.deleteFile();

    juce::WavAudioFormat wavFormat;
    std::unique_ptr<juce::AudioFormatWriter> writer(
        wavFormat.createWriterFor(new juce::FileOutputStream(noiseWavFile),
                                   sampleRate, 1, 16, {}, 0));
    if (writer)
        writer->writeFromAudioSampleBuffer(buffer, 0, numSamples);
}

te::Plugin::Ptr AUSessionBuilder::insertAU(te::Edit& edit, te::AudioTrack& track,
                                            const std::string& auName, int index)
{
    auto it = auMap.find(auName);
    if (it == auMap.end())
        return nullptr;

    auto plugin = edit.getPluginCache().createNewPlugin(
        te::ExternalPlugin::xmlTypeName, it->second);
    if (plugin)
        track.pluginList.insertPlugin(plugin, index, nullptr);
    return plugin;
}

te::Plugin::Ptr AUSessionBuilder::insertAuxSend(te::Edit& edit, te::AudioTrack& track,
                                                  int busNumber, float gainDb)
{
    // Tracktion Engine doesn't have a direct "aux send" plugin for custom buses.
    // We skip sends for now -- the CPU contention from insert plugins is the
    // primary measurement. Sends would add routing complexity but minimal CPU.
    (void)edit; (void)track; (void)busNumber; (void)gainDb;
    return nullptr;
}

void AUSessionBuilder::addAudioClip(te::AudioTrack& track, double durationSeconds)
{
    auto clipDuration = tracktion::TimeDuration::fromSeconds(durationSeconds);
    track.insertWaveClip("Noise", noiseWavFile,
        { { {}, clipDuration } }, false);
    track.setMute(false);
}

void AUSessionBuilder::registerPluginTypes()
{
    if (pluginsRegistered) return;

    auto& pm = engine.getPluginManager();
    pm.createBuiltInType<te::BNNSGraphPlugin>();
    pm.createBuiltInType<te::RTNeuralPlugin>();
    pm.createBuiltInType<te::DirectLibTorchPlugin>();
    pm.createBuiltInType<te::DirectOnnxPlugin>();
    pm.createBuiltInType<te::CallbackStartPlugin>();
    pm.createBuiltInType<te::CallbackEndPlugin>();
    pm.createBuiltInType<te::AniraLibTorchHandlerPlugin>();
    pm.createBuiltInType<te::AniraOnnxHandlerPlugin>();

    pluginsRegistered = true;
}

// ---------------------------------------------------------------------------
// Per-track plugin chains (from benchmark_session_layout.md)
// ---------------------------------------------------------------------------

void AUSessionBuilder::buildTrackChain(te::Edit& edit, te::AudioTrack& track, int trackNumber)
{
    // Track numbering: 1-indexed in the layout doc, 0-indexed here.
    // Each track gets insert plugins per the session layout.
    int t = trackNumber + 1; // Convert to 1-indexed for readability

    switch (t)
    {
        case 1: // Kick In
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1); // gate
            insertAU(edit, track, "AUDynamicsProcessor", -1); // compressor
            break;

        case 2: // Kick Out
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowShelfFilter", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 3: // Snare Top
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUHighShelfFilter", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 4: // Snare Bottom
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 5: // Hi-Hat
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 6: // Tom 1
        case 7: // Tom 2
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 8: // Floor Tom
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowShelfFilter", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 9: // Overheads
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUHighShelfFilter", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 10: // Room Mics
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDistortion", -1);
            break;

        case 11: // Bass DI
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 12: // Bass Amp
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDistortion", -1);
            break;

        case 13: // Electric Rhythm Guitar L
        case 14: // Electric Rhythm Guitar R
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 15: // Electric Lead Guitar — NEURAL MODEL — handled separately
            break;

        case 16: // Acoustic Guitar
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 17: // Piano
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUHighShelfFilter", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 18: // Synth Pad
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUSampleDelay", -1);
            break;

        case 19: // Lead Vocal
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUHighShelfFilter", -1);
            insertAU(edit, track, "AUBandpass", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        case 20: // Lead Vocal Double
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUSampleDelay", -1);
            break;

        case 21: // BG Vocal 1
        case 22: // BG Vocal 2
        case 23: // BG Vocal 3
        case 24: // BG Vocal 4
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            break;

        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Bus chains
// ---------------------------------------------------------------------------

void AUSessionBuilder::buildDrumBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUDistortion", -1);
    insertAU(edit, bus, "AUPeakLimiter", -1);
}

void AUSessionBuilder::buildBassBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUMultibandCompressor", -1);
}

void AUSessionBuilder::buildGuitarBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
}

void AUSessionBuilder::buildKeysBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
}

void AUSessionBuilder::buildVocalBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUMultibandCompressor", -1);
    insertAU(edit, bus, "AUPeakLimiter", -1);
}

void AUSessionBuilder::buildMusicBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
}

void AUSessionBuilder::buildMixBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUMultibandCompressor", -1);
    insertAU(edit, bus, "AUPeakLimiter", -1);
}

// ---------------------------------------------------------------------------
// FX returns
// ---------------------------------------------------------------------------

void AUSessionBuilder::buildFXReturn(te::Edit& edit, te::AudioTrack& track, int fxNumber)
{
    switch (fxNumber)
    {
        case 1: // Short Plate Reverb
            insertAU(edit, track, "AUMatrixReverb", -1);
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            break;

        case 2: // Long Hall Reverb
            insertAU(edit, track, "AUMatrixReverb", -1);
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            break;

        case 3: // Slapback Delay
            insertAU(edit, track, "AUDelay", -1);
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowpass", -1);
            break;

        case 4: // Stereo Ping-Pong Delay
            insertAU(edit, track, "AUDelay", -1);
            insertAU(edit, track, "AUHipass", -1);
            insertAU(edit, track, "AULowpass", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            break;

        case 5: // Parallel Compression (NY Compression)
            insertAU(edit, track, "AUDynamicsProcessor", -1);
            insertAU(edit, track, "AUParametricEQ", -1);
            insertAU(edit, track, "AUDistortion", -1);
            insertAU(edit, track, "AUHighShelfFilter", -1);
            break;
    }
}

// ---------------------------------------------------------------------------
// Neural model insertion (delegates to same logic as EditBuilder)
// ---------------------------------------------------------------------------

void AUSessionBuilder::addCallbackStart(te::AudioTrack& track, CallbackTimer* timer,
                                         ThreadIDLogger* threadIdLogger)
{
    auto plugin = track.edit.getPluginCache().createNewPlugin(
        te::CallbackStartPlugin::xmlTypeName, {});
    if (auto* p = dynamic_cast<te::CallbackStartPlugin*>(plugin.get()))
    {
        p->setCallbackTimer(timer);
        p->setThreadIDLogger(threadIdLogger);
        track.pluginList.insertPlugin(plugin, 0, nullptr);
    }
}

TimingLogger* AUSessionBuilder::addNeuralPlugin(te::AudioTrack& track, BackendType backend,
                                                 ModelType model, ModelSize size, double clipDuration,
                                                 SessionTimingInfo& sessionInfo)
{
    addAudioClip(track, clipDuration);

    te::Plugin::Ptr plugin;

    switch (backend)
    {
        case BackendType::BNNSGraph:
        {
            plugin = track.edit.getPluginCache().createNewPlugin(
                te::BNNSGraphPlugin::xmlTypeName, {});
            if (auto* p = dynamic_cast<te::BNNSGraphPlugin*>(plugin.get()))
            {
                p->setModelPath(modelCoreMLPath(model, size, modelDir));
                p->setPluginName("BNNS_" + juce::String(modelTypeName(model)));
                track.pluginList.insertPlugin(plugin, 0, nullptr);
                return &p->getTimingLogger();
            }
            break;
        }
        case BackendType::RTNeural_Eigen:
        case BackendType::RTNeural_XSIMD:
        {
            plugin = track.edit.getPluginCache().createNewPlugin(
                te::RTNeuralPlugin::xmlTypeName, {});
            if (auto* p = dynamic_cast<te::RTNeuralPlugin*>(plugin.get()))
            {
                p->setModelConfig(model, size, modelWeightsPath(model, size, modelDir));
                p->setPluginName("RTNeural_" + juce::String(modelTypeName(model)));
                track.pluginList.insertPlugin(plugin, 0, nullptr);
                return &p->getTimingLogger();
            }
            break;
        }
        case BackendType::Direct_LibTorch:
        {
            plugin = track.edit.getPluginCache().createNewPlugin(
                te::DirectLibTorchPlugin::xmlTypeName, {});
            if (auto* p = dynamic_cast<te::DirectLibTorchPlugin*>(plugin.get()))
            {
                p->setModelPath(modelTorchScriptPath(model, size, modelDir));
                p->setPluginName("DirectLT_" + juce::String(modelTypeName(model)));
                track.pluginList.insertPlugin(plugin, 0, nullptr);
                return &p->getTimingLogger();
            }
            break;
        }
        case BackendType::Direct_ONNX:
        {
            plugin = track.edit.getPluginCache().createNewPlugin(
                te::DirectOnnxPlugin::xmlTypeName, {});
            if (auto* p = dynamic_cast<te::DirectOnnxPlugin*>(plugin.get()))
            {
                p->setModelPath(modelOnnxPath(model, size, modelDir));
                p->setPluginName("DirectONNX_" + juce::String(modelTypeName(model)));
                track.pluginList.insertPlugin(plugin, 0, nullptr);
                return &p->getTimingLogger();
            }
            break;
        }
        case BackendType::Anira_LibTorch:
        {
            plugin = track.edit.getPluginCache().createNewPlugin(
                te::AniraLibTorchHandlerPlugin::xmlTypeName, {});
            if (auto* p = dynamic_cast<te::AniraLibTorchHandlerPlugin*>(plugin.get()))
            {
                p->setModelPath(modelTorchScriptPath(model, size, modelDir));
                p->setPluginName("AniraLT_" + juce::String(modelTypeName(model)));
                track.pluginList.insertPlugin(plugin, 0, nullptr);
                sessionInfo.inferenceUnderrunGetters.push_back([p]() { return p->getInferenceUnderruns(); });
                sessionInfo.inferenceUnderrunResetters.push_back([p]() { p->resetInferenceUnderruns(); });
                return &p->getTimingLogger();
            }
            break;
        }
        case BackendType::Anira_ONNX:
        {
            plugin = track.edit.getPluginCache().createNewPlugin(
                te::AniraOnnxHandlerPlugin::xmlTypeName, {});
            if (auto* p = dynamic_cast<te::AniraOnnxHandlerPlugin*>(plugin.get()))
            {
                p->setModelPath(modelOnnxPath(model, size, modelDir));
                p->setPluginName("AniraONNX_" + juce::String(modelTypeName(model)));
                track.pluginList.insertPlugin(plugin, 0, nullptr);
                sessionInfo.inferenceUnderrunGetters.push_back([p]() { return p->getInferenceUnderruns(); });
                sessionInfo.inferenceUnderrunResetters.push_back([p]() { p->resetInferenceUnderruns(); });
                return &p->getTimingLogger();
            }
            break;
        }
        default:
            break;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Build the full session
// ---------------------------------------------------------------------------

SessionTimingInfo AUSessionBuilder::buildSession(te::Edit& edit, BackendType backend,
                                                  ModelType model, ModelSize size, int activeTracks,
                                                  double sampleRate)
{
    registerPluginTypes();

    double clipDuration = 30.0; // Long enough for warmup + measurement
    ensureNoiseWavFile(clipDuration, sampleRate);

    SessionTimingInfo info;
    info.callbackTimer = std::make_unique<CallbackTimer>();
    info.threadIdLogger = std::make_unique<ThreadIDLogger>();
    info.threadIdLogger->allocate(32);

    // ---------------------------------------------------------------------------
    // Track layout (36 tracks total):
    //   0-9:   Drums (Kick In, Kick Out, Snare Top/Bottom, HiHat, Tom1/2, FloorTom, OHs, Room)
    //   10-11: Bass (DI, Amp)
    //   12-15: Guitars (Rhythm L, Rhythm R, *Lead Guitar [neural]*, Acoustic)
    //   16-17: Keys (Piano, Synth Pad)
    //   18-23: Vocals (Lead, Double, BG1-4)
    //   24:    Drum Bus
    //   25:    Bass Bus
    //   26:    Guitar Bus
    //   27:    Keys Bus
    //   28:    Vocal Bus
    //   29:    Music Bus
    //   30:    Mix Bus (master processing before actual master)
    //   31-35: FX Returns (Plate Reverb, Hall Reverb, Slapback, Ping-Pong, Parallel Comp)
    // ---------------------------------------------------------------------------

    edit.ensureNumberOfAudioTracks(36);
    auto tracks = te::getAudioTracks(edit);

    if (static_cast<int>(tracks.size()) < 36)
    {
        fprintf(stderr, "  ERROR: Could not create 36 tracks\n");
        return info;
    }

    // Count of plugin inserts for verification
    int totalInserts = 0;

    // --- Source tracks (0-23) ---
    // activeTracks = number of CONVENTIONAL tracks alongside the neural track.
    // Track 14 (neural) is always active and does NOT count toward activeTracks.
    int conventionalCount = 0;
    for (int i = 0; i < 24; i++)
    {
        auto* track = tracks[static_cast<size_t>(i)];

        if (i == 14) // Track 15 = Electric Lead Guitar = NEURAL MODEL
        {
            info.neuralLogger = addNeuralPlugin(*track, backend, model, size, clipDuration, info);

            // Post-neural processing (from session layout)
            insertAU(edit, *track, "AUHipass", -1);
            insertAU(edit, *track, "AUParametricEQ", -1);
            insertAU(edit, *track, "AUParametricEQ", -1);
            insertAU(edit, *track, "AUHighShelfFilter", -1);
            insertAU(edit, *track, "AUDynamicsProcessor", -1);
            totalInserts += 6; // 1 neural + 5 conventional

            addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
            track->setMute(false);
        }
        else if (conventionalCount < activeTracks)
        {
            // Active conventional track with real AU chain
            addAudioClip(*track, clipDuration);
            buildTrackChain(edit, *track, i);
            conventionalCount++;

            addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
            track->setMute(false);
            totalInserts += track->pluginList.size() - 2; // subtract default Volume+Level
        }
        else
        {
            // Inactive (muted) track
            track->setMute(true);
        }

        // Route to default output
        track->getOutput().setOutputToDefaultDevice(false);
    }

    // --- Group buses (24-30) ---
    struct BusConfig {
        int index;
        const char* name;
        void (AUSessionBuilder::*buildFn)(te::Edit&, te::AudioTrack&);
    };

    BusConfig buses[] = {
        {24, "Drum Bus",   &AUSessionBuilder::buildDrumBus},
        {25, "Bass Bus",   &AUSessionBuilder::buildBassBus},
        {26, "Guitar Bus", &AUSessionBuilder::buildGuitarBus},
        {27, "Keys Bus",   &AUSessionBuilder::buildKeysBus},
        {28, "Vocal Bus",  &AUSessionBuilder::buildVocalBus},
        {29, "Music Bus",  &AUSessionBuilder::buildMusicBus},
        {30, "Mix Bus",    &AUSessionBuilder::buildMixBus},
    };

    for (auto& bus : buses)
    {
        auto* track = tracks[static_cast<size_t>(bus.index)];
        track->setName(juce::String(bus.name));
        // Buses need audio input to process — add a noise clip
        addAudioClip(*track, clipDuration);
        (this->*(bus.buildFn))(edit, *track);
        track->setMute(false);
        track->getOutput().setOutputToDefaultDevice(false);
        totalInserts += track->pluginList.size() - 2;
    }

    // --- FX Return buses (31-35) ---
    for (int fx = 1; fx <= 5; fx++)
    {
        int idx = 30 + fx;
        auto* track = tracks[static_cast<size_t>(idx)];

        const char* fxNames[] = {"", "FX1 Plate Reverb", "FX2 Hall Reverb",
                                  "FX3 Slapback Delay", "FX4 Ping-Pong Delay",
                                  "FX5 Parallel Comp"};
        track->setName(juce::String(fxNames[fx]));
        addAudioClip(*track, clipDuration);
        buildFXReturn(edit, *track, fx);
        track->setMute(false);
        track->getOutput().setOutputToDefaultDevice(false);
        totalInserts += track->pluginList.size() - 2;
    }

    // --- Callback timing: CallbackEnd on master bus ---
    auto endPlugin = edit.getPluginCache().createNewPlugin(
        te::CallbackEndPlugin::xmlTypeName, {});
    if (auto* p = dynamic_cast<te::CallbackEndPlugin*>(endPlugin.get()))
    {
        p->setCallbackTimer(info.callbackTimer.get());
        edit.getMasterPluginList().insertPlugin(endPlugin, -1, nullptr);
        info.callbackLogger = &p->getTimingLogger();
    }

    fprintf(stderr, "  AU session built: %d conventional + neural track 15, ~%d insert plugins\n",
            conventionalCount, totalInserts);

    return info;
}
