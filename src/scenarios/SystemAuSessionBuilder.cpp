// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "SystemAuSessionBuilder.h"
#include "../contention/EditBuilder.h"  // for SessionTimingInfo + attachNeuralPlugin
#include "../host/NeuralInferencePlugin.h"

#include <algorithm>
#include <random>

// ---------------------------------------------------------------------------
// SystemAuSessionBuilder
// ---------------------------------------------------------------------------

SystemAuSessionBuilder::SystemAuSessionBuilder(te::Engine& engine, const std::string& modelDir,
                                   const std::vector<ModelSpec>& specs)
    : engine(engine), modelDir(modelDir), specs(specs) {}

SystemAuSessionBuilder::~SystemAuSessionBuilder()
{
    if (noiseWavFile.existsAsFile())
        noiseWavFile.deleteFile();
}

const std::vector<std::string>& SystemAuSessionBuilder::requiredAUNames()
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

bool SystemAuSessionBuilder::scanForRequiredAUs()
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

void SystemAuSessionBuilder::printAUStatus() const
{
    fprintf(stderr, "\nSystem AU availability:\n");
    for (auto& name : requiredAUNames())
    {
        bool found = auMap.count(name) > 0;
        fprintf(stderr, "  %s: %s\n", name.c_str(), found ? "OK" : "MISSING");
    }
}

void SystemAuSessionBuilder::ensureNoiseWavFile(double durationSeconds, double sampleRate)
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
    std::unique_ptr<juce::OutputStream> stream =
        std::make_unique<juce::FileOutputStream>(noiseWavFile);
    auto writer = wavFormat.createWriterFor(
        stream,
        juce::AudioFormatWriterOptions{}
            .withSampleRate(sampleRate)
            .withNumChannels(1)
            .withBitsPerSample(16));
    if (writer)
        writer->writeFromAudioSampleBuffer(buffer, 0, numSamples);
}

te::Plugin::Ptr SystemAuSessionBuilder::insertAU(te::Edit& edit, te::AudioTrack& track,
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

te::Plugin::Ptr SystemAuSessionBuilder::insertAuxSend(te::Edit& edit, te::AudioTrack& track,
                                                  int busNumber, float gainDb)
{
    // Tracktion Engine doesn't have a direct "aux send" plugin for custom buses.
    // We skip sends for now -- the CPU contention from insert plugins is the
    // primary measurement. Sends would add routing complexity but minimal CPU.
    (void)edit; (void)track; (void)busNumber; (void)gainDb;
    return nullptr;
}

void SystemAuSessionBuilder::addAudioClip(te::AudioTrack& track, double durationSeconds)
{
    auto clipDuration = tracktion::TimeDuration::fromSeconds(durationSeconds);
    track.insertWaveClip("Noise", noiseWavFile,
        { { {}, clipDuration } }, false);
    track.setMute(false);
}

void SystemAuSessionBuilder::registerPluginTypes()
{
    if (pluginsRegistered) return;

    auto& pm = engine.getPluginManager();
    pm.createBuiltInType<te::NeuralInferencePlugin>();
    pm.createBuiltInType<te::CallbackStartPlugin>();
    pm.createBuiltInType<te::CallbackEndPlugin>();

    pluginsRegistered = true;
}

// ---------------------------------------------------------------------------
// Per-track plugin chains for the fixed system-AU session
// ---------------------------------------------------------------------------

void SystemAuSessionBuilder::buildTrackChain(te::Edit& edit, te::AudioTrack& track, int trackNumber)
{
    // Track numbering: 1-indexed in the documented layout, 0-indexed here.
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

void SystemAuSessionBuilder::buildDrumBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUDistortion", -1);
    insertAU(edit, bus, "AUPeakLimiter", -1);
}

void SystemAuSessionBuilder::buildBassBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUMultibandCompressor", -1);
}

void SystemAuSessionBuilder::buildGuitarBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
}

void SystemAuSessionBuilder::buildKeysBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
}

void SystemAuSessionBuilder::buildVocalBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUMultibandCompressor", -1);
    insertAU(edit, bus, "AUPeakLimiter", -1);
}

void SystemAuSessionBuilder::buildMusicBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUParametricEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
}

void SystemAuSessionBuilder::buildMixBus(te::Edit& edit, te::AudioTrack& bus)
{
    insertAU(edit, bus, "AUNBandEQ", -1);
    insertAU(edit, bus, "AUDynamicsProcessor", -1);
    insertAU(edit, bus, "AUMultibandCompressor", -1);
    insertAU(edit, bus, "AUPeakLimiter", -1);
}

// ---------------------------------------------------------------------------
// FX returns
// ---------------------------------------------------------------------------

void SystemAuSessionBuilder::buildFXReturn(te::Edit& edit, te::AudioTrack& track, int fxNumber)
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

void SystemAuSessionBuilder::addCallbackStart(te::AudioTrack& track, CallbackTimer* timer,
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

TimingLogger* SystemAuSessionBuilder::addNeuralPlugin(te::AudioTrack& track,
                                                 const std::string& backend,
                                                 const ModelSpec& model, double clipDuration,
                                                 SessionTimingInfo& sessionInfo)
{
    addAudioClip(track, clipDuration);

    // Plugin display name — kept byte-identical to the pre-refactor
    // SystemAuSessionBuilder (note the Direct/Anira prefixes differ from EditBuilder).
    const auto modelName = modelArchDisplayName(model);
    juce::String prefix = juce::String(backend);
    if (backend == "BNNSGraph") prefix = "BNNS";
    else if (backend.rfind("RTNeural_", 0) == 0) prefix = "RTNeural";
    else if (backend == "Direct_LibTorch") prefix = "DirectLT";
    else if (backend == "Direct_ONNX") prefix = "DirectONNX";
    else if (backend == "Anira_LibTorch") prefix = "AniraLT";
    else if (backend == "Anira_ONNX") prefix = "AniraONNX";

    const auto name = prefix + "_" + juce::String(modelName);
    return attachNeuralPlugin(track, backend, model, name, sessionInfo);
}

// ---------------------------------------------------------------------------
// Build the full session
// ---------------------------------------------------------------------------

SessionTimingInfo SystemAuSessionBuilder::buildSession(te::Edit& edit,
                                                  const std::string& backend,
                                                  const ModelSpec& model, int activeTracks,
                                                  double sampleRate)
{
    registerPluginTypes();

    const int effectiveActiveTracks =
        std::clamp(activeTracks, 0, kMaxConventionalTracks);
    if (effectiveActiveTracks != activeTracks)
    {
        fprintf(stderr,
                "  WARNING: requested Dimension A contention level %d maps to %d "
                "conventional source tracks (valid range 0-%d); the CSV retains "
                "the requested level for paper compatibility\n",
                activeTracks, effectiveActiveTracks, kMaxConventionalTracks);
    }

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

    edit.ensureNumberOfAudioTracks(kSessionTrackCount);
    auto tracks = te::getAudioTracks(edit);

    if (static_cast<int>(tracks.size()) < kSessionTrackCount)
    {
        fprintf(stderr, "  ERROR: Could not create %d tracks\n", kSessionTrackCount);
        return info;
    }

    // Count of plugin inserts for verification
    int totalInserts = 0;

    // --- Source tracks (0-23) ---
    // activeTracks = number of CONVENTIONAL tracks alongside the neural track.
    // Track 14 (neural) is always active and does NOT count toward activeTracks.
    int conventionalCount = 0;
    for (int i = 0; i < kSourceTrackCount; i++)
    {
        auto* track = tracks[static_cast<size_t>(i)];

        if (i == kNeuralTrackIndex) // Track 15 = Electric Lead Guitar = NEURAL MODEL
        {
            info.neuralLogger = addNeuralPlugin(*track, backend, model, clipDuration, info);

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
        else if (conventionalCount < effectiveActiveTracks)
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
        void (SystemAuSessionBuilder::*buildFn)(te::Edit&, te::AudioTrack&);
    };

    BusConfig buses[] = {
        {24, "Drum Bus",   &SystemAuSessionBuilder::buildDrumBus},
        {25, "Bass Bus",   &SystemAuSessionBuilder::buildBassBus},
        {26, "Guitar Bus", &SystemAuSessionBuilder::buildGuitarBus},
        {27, "Keys Bus",   &SystemAuSessionBuilder::buildKeysBus},
        {28, "Vocal Bus",  &SystemAuSessionBuilder::buildVocalBus},
        {29, "Music Bus",  &SystemAuSessionBuilder::buildMusicBus},
        {30, "Mix Bus",    &SystemAuSessionBuilder::buildMixBus},
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
