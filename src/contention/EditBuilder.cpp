// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "EditBuilder.h"
#include "../plugins/BNNSGraphPlugin.h"
#include "../plugins/RTNeuralPlugin.h"
#include "../plugins/AniraPlugin.h"
#include "../plugins/AniraHandlerPlugin.h"
#include "../plugins/ContentionPlugins.h"

#include <random>

// ---------------------------------------------------------------------------
// EditBuilder
// ---------------------------------------------------------------------------

EditBuilder::EditBuilder(te::Engine& engine, const std::string& modelDir)
    : engine(engine), modelDir(modelDir) {}

EditBuilder::~EditBuilder()
{
    // Clean up the temp WAV file
    if (noiseWavFile.existsAsFile())
        noiseWavFile.deleteFile();
}

void EditBuilder::ensureNoiseWavFile(double durationSeconds, double sampleRate)
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
    noiseWavFile = tempDir.getChildFile("contention_noise.wav");
    noiseWavFile.deleteFile();

    juce::WavAudioFormat wavFormat;
    std::unique_ptr<juce::AudioFormatWriter> writer(
        wavFormat.createWriterFor(new juce::FileOutputStream(noiseWavFile),
                                   sampleRate, 1, 16, {}, 0));
    if (writer)
        writer->writeFromAudioSampleBuffer(buffer, 0, numSamples);

    fprintf(stderr, "  Created noise WAV: %s\n", noiseWavFile.getFullPathName().toRawUTF8());
}

void EditBuilder::registerPluginTypes()
{
    if (pluginsRegistered) return;

    auto& pm = engine.getPluginManager();
    pm.createBuiltInType<te::BNNSGraphPlugin>();
    pm.createBuiltInType<te::RTNeuralPlugin>();
    pm.createBuiltInType<te::DirectLibTorchPlugin>();
    pm.createBuiltInType<te::DirectOnnxPlugin>();
    pm.createBuiltInType<te::ContentionEQPlugin>();
    pm.createBuiltInType<te::ContentionCompPlugin>();
    pm.createBuiltInType<te::ContentionReverbPlugin>();
    pm.createBuiltInType<te::ContentionDelayPlugin>();
    pm.createBuiltInType<te::NoiseGeneratorPlugin>();
    pm.createBuiltInType<te::CallbackStartPlugin>();
    pm.createBuiltInType<te::CallbackEndPlugin>();
    pm.createBuiltInType<te::AniraLibTorchHandlerPlugin>();
    pm.createBuiltInType<te::AniraOnnxHandlerPlugin>();

    pluginsRegistered = true;
}

void EditBuilder::addAudioClip(te::AudioTrack& track, double durationSeconds)
{
    auto clipDuration = tracktion::TimeDuration::fromSeconds(durationSeconds);
    track.insertWaveClip("Noise", noiseWavFile,
        { { {}, clipDuration } }, false);
    track.setMute(false);
}

void EditBuilder::addConventionalDSP(te::AudioTrack& track, double clipDuration)
{
    addAudioClip(track, clipDuration);

    auto eqPlugin = track.edit.getPluginCache().createNewPlugin(
        te::ContentionEQPlugin::xmlTypeName, {});
    track.pluginList.insertPlugin(eqPlugin, 0, nullptr);

    auto compPlugin = track.edit.getPluginCache().createNewPlugin(
        te::ContentionCompPlugin::xmlTypeName, {});
    track.pluginList.insertPlugin(compPlugin, -1, nullptr);
}

TimingLogger* EditBuilder::addNeuralPlugin(te::AudioTrack& track, BackendType backend,
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
                p->setPluginName("DirectLibTorch_" + juce::String(modelTypeName(model)));
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
                p->setPluginName("AniraLibTorch_" + juce::String(modelTypeName(model)));
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

void EditBuilder::addCallbackStart(te::AudioTrack& track, CallbackTimer* timer,
                                   ThreadIDLogger* threadIdLogger)
{
    auto plugin = track.edit.getPluginCache().createNewPlugin(
        te::CallbackStartPlugin::xmlTypeName, {});
    if (auto* p = dynamic_cast<te::CallbackStartPlugin*>(plugin.get()))
    {
        p->setCallbackTimer(timer);
        p->setThreadIDLogger(threadIdLogger);
        // Insert at position 0 so it runs BEFORE all other plugins on this track
        track.pluginList.insertPlugin(plugin, 0, nullptr);
    }
}

TimingLogger* EditBuilder::addCallbackEnd(te::Edit& edit, CallbackTimer* timer)
{
    auto plugin = edit.getPluginCache().createNewPlugin(
        te::CallbackEndPlugin::xmlTypeName, {});
    if (auto* p = dynamic_cast<te::CallbackEndPlugin*>(plugin.get()))
    {
        p->setCallbackTimer(timer);
        // Insert on master bus — runs AFTER all tracks are mixed
        edit.getMasterPluginList().insertPlugin(plugin, -1, nullptr);
        return &p->getTimingLogger();
    }
    return nullptr;
}

SessionTimingInfo EditBuilder::buildDimensionA(te::Edit& edit, BackendType backend,
                                                ModelType model, ModelSize size, int activeTracks,
                                                double sampleRate, int numTracks)
{
    registerPluginTypes();

    // Audio clips are mono.  The session layout spec mentions stereo tracks
    // (overheads, room mics, synth pad) but this benchmark uses mono throughout
    // for simplicity.  This underestimates the CPU cost of stereo processing
    // chains but does not affect the relative comparison between backends.
    double clipDuration = 30.0;  // Long enough for warmup + measurement
    ensureNoiseWavFile(clipDuration, sampleRate);

    edit.ensureNumberOfAudioTracks(numTracks);
    auto tracks = te::getAudioTracks(edit);

    SessionTimingInfo info;
    info.callbackTimer = std::make_unique<CallbackTimer>();
    info.threadIdLogger = std::make_unique<ThreadIDLogger>();
    info.threadIdLogger->allocate(32);

    // activeTracks = number of CONVENTIONAL DSP tracks to activate alongside
    // the neural track.  The neural track (index 14) is always active and does
    // NOT count toward activeTracks.  A separate counter ensures exactly
    // activeTracks conventional tracks are created regardless of the neural
    // track's position.  The CSV column 'contention_level' equals activeTracks.
    int conventionalCount = 0;
    for (int i = 0; i < static_cast<int>(tracks.size()) && i < numTracks; i++)
    {
        auto* track = tracks[static_cast<size_t>(i)];

        if (i == 14) // Track 15 = neural model track
        {
            info.neuralLogger = addNeuralPlugin(*track, backend, model, size, clipDuration, info);

            auto eqPlugin = edit.getPluginCache().createNewPlugin(
                te::ContentionEQPlugin::xmlTypeName, {});
            track->pluginList.insertPlugin(eqPlugin, -1, nullptr);

            auto compPlugin = edit.getPluginCache().createNewPlugin(
                te::ContentionCompPlugin::xmlTypeName, {});
            track->pluginList.insertPlugin(compPlugin, -1, nullptr);

            // Add callback start FIRST on this track (before neural plugin)
            addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
        }
        else if (conventionalCount < activeTracks)
        {
            addConventionalDSP(*track, clipDuration);
            conventionalCount++;

            if (conventionalCount % 4 == 1)
            {
                auto reverbPlugin = edit.getPluginCache().createNewPlugin(
                    te::ContentionReverbPlugin::xmlTypeName, {});
                track->pluginList.insertPlugin(reverbPlugin, -1, nullptr);
            }
            else if (conventionalCount % 4 == 3)
            {
                auto delayPlugin = edit.getPluginCache().createNewPlugin(
                    te::ContentionDelayPlugin::xmlTypeName, {});
                track->pluginList.insertPlugin(delayPlugin, -1, nullptr);
            }

            // Add callback start as FIRST plugin on every active track
            addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
        }
        else
        {
            track->setMute(true);
        }
    }

    // Add callback end on master bus (runs after all tracks mixed)
    info.callbackLogger = addCallbackEnd(edit, info.callbackTimer.get());

    return info;
}

SessionTimingInfo EditBuilder::buildDimensionB(te::Edit& edit, BackendType backend,
                                                ModelType model, ModelSize size, int instanceCount,
                                                double sampleRate)
{
    registerPluginTypes();

    double clipDuration = 30.0;  // Long enough for warmup + measurement
    ensureNoiseWavFile(clipDuration, sampleRate);

    SessionTimingInfo info;
    info.callbackTimer = std::make_unique<CallbackTimer>();
    info.threadIdLogger = std::make_unique<ThreadIDLogger>();
    info.threadIdLogger->allocate(32);

    edit.ensureNumberOfAudioTracks(instanceCount);
    auto tracks = te::getAudioTracks(edit);

    for (int i = 0; i < instanceCount && i < static_cast<int>(tracks.size()); i++)
    {
        auto* logger = addNeuralPlugin(*tracks[static_cast<size_t>(i)], backend, model, size, clipDuration, info);
        if (logger)
            info.neuralLoggers.push_back(logger);

        addCallbackStart(*tracks[static_cast<size_t>(i)], info.callbackTimer.get(),
                         info.threadIdLogger.get());
    }

    if (!info.neuralLoggers.empty())
        info.neuralLogger = info.neuralLoggers[0];

    info.callbackLogger = addCallbackEnd(edit, info.callbackTimer.get());

    return info;
}

// ---------------------------------------------------------------------------
// Dimension C: neural track serial depth
// ---------------------------------------------------------------------------

SessionTimingInfo EditBuilder::buildDimensionC(te::Edit& edit, BackendType backend,
                                                ModelType model, ModelSize size, int depth,
                                                double sampleRate)
{
    registerPluginTypes();

    double clipDuration = 30.0;
    ensureNoiseWavFile(clipDuration, sampleRate);

    SessionTimingInfo info;
    info.callbackTimer = std::make_unique<CallbackTimer>();
    info.threadIdLogger = std::make_unique<ThreadIDLogger>();
    info.threadIdLogger->allocate(32);

    edit.ensureNumberOfAudioTracks(1);
    auto tracks = te::getAudioTracks(edit);
    auto* track = tracks[0];

    // Chain layout: pre-neural plugins → neural → post-neural plugins.
    // Realistic signal chain: EQ → Compressor → Neural → Delay → Reverb → EQ → Compressor
    // depth=1: neural only (bare)
    // depth=3: EQ → Comp → Neural  (channel_strip)
    // depth=5: EQ → Comp → Neural → Delay → Reverb  (mix_fx)
    // depth=7: EQ → Comp → Neural → Delay → Reverb → EQ → Comp  (heavy_chain)
    //
    // NOTE: addNeuralPlugin inserts at position 0 (front) and adds its own audio clip.
    // So we call addNeuralPlugin FIRST, then insert pre-neural plugins at position 0
    // (pushing the neural plugin down), then append post-neural plugins at -1 (end).
    // This gives the correct chain order.

    // Step 1: Neural plugin (inserts at pos 0, adds audio clip)
    info.neuralLogger = addNeuralPlugin(*track, backend, model, size, clipDuration, info);

    // Step 2: Pre-neural plugins inserted at position 0 — BEFORE the neural plugin.
    // Insert in reverse order so they end up in the right sequence.
    if (depth >= 3)
    {
        auto comp = edit.getPluginCache().createNewPlugin(te::ContentionCompPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(comp, 0, nullptr);  // now at front

        auto eq = edit.getPluginCache().createNewPlugin(te::ContentionEQPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(eq, 0, nullptr);     // pushes comp after it
        // Chain so far: EQ → Comp → Neural
    }

    // Step 3: Post-neural plugins appended at -1 (end)
    if (depth >= 5)
    {
        auto delay = edit.getPluginCache().createNewPlugin(te::ContentionDelayPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(delay, -1, nullptr);

        auto reverb = edit.getPluginCache().createNewPlugin(te::ContentionReverbPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(reverb, -1, nullptr);
    }

    if (depth >= 7)
    {
        auto eq2 = edit.getPluginCache().createNewPlugin(te::ContentionEQPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(eq2, -1, nullptr);

        auto comp2 = edit.getPluginCache().createNewPlugin(te::ContentionCompPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(comp2, -1, nullptr);
    }

    // Step 4: CallbackStart at position 0 (before everything)
    addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
    info.callbackLogger = addCallbackEnd(edit, info.callbackTimer.get());

    return info;
}

// ---------------------------------------------------------------------------
// ContentionBenchmark — real-time CoreAudio playback via TransportControl
// ---------------------------------------------------------------------------

void ContentionBenchmark::runSingleConfig(
    FILE* csvFile, const char* dimension, BackendType backend,
    ModelType model, ModelSize size, int bufferSize, int contentionLevel, int instanceCount, int rep)
{
    using namespace tracktion::engine;

    auto cfg = BenchmarkRuntimeConfig::load(configPath);

    fprintf(stderr, "  %s: %s/%s/%s buf=%d contention=%d instances=%d rep=%d\n",
            dimension, backendTypeName(backend), modelTypeName(model), modelSizeName(size),
            bufferSize, contentionLevel, instanceCount, rep);

    // --- Configure audio device for this buffer size ---
    auto& dm = engine.getDeviceManager();
    auto setup = dm.deviceManager.getAudioDeviceSetup();
    if (setup.bufferSize != bufferSize || setup.sampleRate != cfg.sampleRate)
    {
        setup.sampleRate = cfg.sampleRate;
        setup.bufferSize = bufferSize;
        dm.deviceManager.setAudioDeviceSetup(setup, true);
    }

    auto* device = dm.deviceManager.getCurrentAudioDevice();
    double actualSR = device ? device->getCurrentSampleRate() : cfg.sampleRate;
    int actualBS = device ? device->getCurrentBufferSizeSamples() : bufferSize;

    // Guard against hidden resampling: if the device runs at a different
    // sample rate than the config, Tracktion will resample the noise WAV
    // (created at cfg.sampleRate), inflating processing cost.  Fail early.
    if (std::abs(actualSR - cfg.sampleRate) > 1.0)
    {
        fprintf(stderr, "    SKIP: device sample rate (%.0f Hz) differs from config (%.0f Hz).\n"
                "    Resampling would contaminate timing results.\n",
                actualSR, cfg.sampleRate);
        return;
    }

    // --- Create Edit and build session ---
    auto edit = Edit::createSingleTrackEdit(engine);
    edit->tempoSequence.getTempo(0)->setBpm(60.0);
    edit->getMasterVolumePlugin()->setVolumeDb(0.0f);

    SessionTimingInfo timing;

    std::string dimStr(dimension);
    if (dimStr == "dim_a")
    {
        // use_system_au: real macOS AUs for Dimension A contention.
        // Lazy scan: only probe for AUs on first dim_a call.
        if (cfg.useSystemAU)
        {
            if (!auScanned)
            {
                auAvailable = auBuilder.scanForRequiredAUs();
                auScanned = true;
                if (auAvailable)
                    fprintf(stderr, "AU session builder: all system AUs available\n");
                else
                    fprintf(stderr, "AU session builder: some AUs missing, falling back to custom DSP\n");
            }

            if (auAvailable)
                timing = auBuilder.buildSession(*edit, backend, model, size, contentionLevel, cfg.sampleRate);
            else
                timing = builder.buildDimensionA(*edit, backend, model, size, contentionLevel, cfg.sampleRate, cfg.contentionNumTracks);
        }
        else
        {
            timing = builder.buildDimensionA(*edit, backend, model, size, contentionLevel, cfg.sampleRate, cfg.contentionNumTracks);
        }
    }
    else if (dimStr == "dim_b")
        timing = builder.buildDimensionB(*edit, backend, model, size, instanceCount, cfg.sampleRate);
    else if (dimStr == "dim_c")
        timing = builder.buildDimensionC(*edit, backend, model, size, contentionLevel, cfg.sampleRate);
    else
        timing = builder.buildDimensionA(*edit, backend, model, size, contentionLevel, cfg.sampleRate, cfg.contentionNumTracks);

    if (!timing.neuralLogger)
    {
        fprintf(stderr, "    SKIP: failed to create neural plugin\n");
        return;
    }

    // Route all tracks to default output
    for (auto* track : getAudioTracks(*edit))
        track->getOutput().setOutputToDefaultDevice(false);

    // --- Start real-time playback ---
    auto& transport = edit->getTransport();
    transport.setPosition(tracktion::TimePosition());

    // Let device manager settle after any config change
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    transport.ensureContextAllocated(true);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    transport.play(false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    if (!transport.isPlaying())
    {
        fprintf(stderr, "    SKIP: transport failed to start\n");
        transport.freePlaybackContext();
        return;
    }

    // --- Warmup phase: play for a few seconds to stabilise ---
    // JIT compilation, cache warming, and thread scheduler need 2-3s minimum
    double warmupSeconds = static_cast<double>(cfg.contentionWarmupCallbacks) * bufferSize / actualSR;
    warmupSeconds = std::max(warmupSeconds, cfg.contentionWarmupMinSeconds);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(
        static_cast<int>(warmupSeconds * 1000.0));

    // Stop briefly to safely reset timing loggers (no concurrent audio thread access)
    transport.stop(false, false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    timing.neuralLogger->reset();
    for (auto* l : timing.neuralLoggers)
        l->reset();
    if (timing.callbackLogger)
        timing.callbackLogger->reset();
    if (timing.threadIdLogger)
        timing.threadIdLogger->reset();
    timing.resetInferenceUnderruns();  // zero anira underrun counters at measurement boundary

    // Restart playback for measurement phase
    transport.play(false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    if (!transport.isPlaying())
    {
        fprintf(stderr, "    SKIP: transport failed to restart for measurement\n");
        transport.freePlaybackContext();
        return;
    }

    // --- Snapshot xrun count AFTER the 200ms restart settle ---
    // Taking the baseline here (not before restart) excludes xruns caused
    // by the transport restart transient, which is not part of steady-state
    // measurement.
    int xrunsBefore = device ? device->getXRunCount() : 0;

    // --- Measurement phase ---
    TimingUtils::init();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(cfg.contentionMeasureSeconds * 1000);

    // --- Stop playback ---
    transport.stop(false, false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    // --- Snapshot xrun count AFTER measurement (1B) ---
    int xrunsAfter = device ? device->getXRunCount() : 0;
    int hwXruns = xrunsAfter - xrunsBefore;

    // --- Collect thread count (1A) ---
    int threadCount = timing.threadIdLogger ? timing.threadIdLogger->getUniqueCount() : 0;

    // --- Collect results ---
    double deadline_ns = TimingUtils::bufferDurationNs(actualBS, actualSR);

    // Neural inference timing (per-plugin)
    auto neuralDurations = timing.neuralLogger->getDurationsNs();
    if (neuralDurations.empty())
    {
        fprintf(stderr, "    SKIP: no timing data collected (applyToBuffer not called)\n");
        transport.freePlaybackContext();
        return;
    }

    // Full callback timing (graph start → master bus end)
    auto callbackDurations = timing.callbackLogger
        ? timing.callbackLogger->getDurationsNs()
        : std::vector<double>{};

    // Discard entries from the restart→measurement settle period (200ms wait
    // at line 427 above).  Trim 300ms worth of callbacks (200ms + margin) to
    // exclude transport restart transients, cache warming, and JIT.
    auto trimFront = [actualBS, actualSR](std::vector<double>& v) {
        int settleCallbacks = static_cast<int>(0.3 * actualSR / actualBS);
        int trim = std::max(settleCallbacks, static_cast<int>(v.size()) / 10);
        trim = std::min(trim, static_cast<int>(v.size()) / 2); // never trim more than half
        if (trim > 0 && static_cast<int>(v.size()) > trim)
            v.erase(v.begin(), v.begin() + trim);
    };
    trimFront(neuralDurations);
    trimFront(callbackDurations);

    auto neuralStats = TimingStats::compute(neuralDurations, deadline_ns);

    // Query anira inference underruns (0 for non-anira backends).
    // This counts callbacks where background inference output was not ready.
    int inferenceUnderruns = timing.getTotalInferenceUnderruns();

    // Print neural row (includes hw_xruns and thread_count)
    CSVOutput::printContentionRow(csvFile, dimension, backendTypeName(backend),
                                   modelTypeName(model), modelSizeName(size), actualBS,
                                   contentionLevel, instanceCount, rep, neuralStats,
                                   hwXruns, inferenceUnderruns, threadCount);

    // Print callback row (same config, dimension suffixed with "_cb")
    if (!callbackDurations.empty())
    {
        auto cbStats = TimingStats::compute(callbackDurations, deadline_ns);
        std::string cbDim = std::string(dimension) + "_cb";
        CSVOutput::printContentionRow(csvFile, cbDim.c_str(), backendTypeName(backend),
                                       modelTypeName(model), modelSizeName(size), actualBS,
                                       contentionLevel, instanceCount, rep, cbStats,
                                       hwXruns, inferenceUnderruns, threadCount);

        fprintf(stderr, "    -> neural: %d cb, p50=%.1f%% p99=%.1f%% | callback: %d cb, p50=%.1f%% p99=%.1f%% | hw_xruns=%d inf_underruns=%d threads=%d\n",
                neuralStats.total_samples, neuralStats.util_p50, neuralStats.util_p99,
                cbStats.total_samples, cbStats.util_p50, cbStats.util_p99,
                hwXruns, inferenceUnderruns, threadCount);
    }
    else
    {
        fprintf(stderr, "    -> %d callbacks, p50=%.1f%% p99=%.1f%% | hw_xruns=%d threads=%d\n",
                neuralStats.total_samples, neuralStats.util_p50, neuralStats.util_p99,
                hwXruns, threadCount);
    }

    fflush(csvFile);
    transport.freePlaybackContext();
}

void ContentionBenchmark::runDimensionA(FILE* csvFile)
{
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Dimension A: Mix Contention Sweep\n");
    fprintf(stderr, "========================================\n");

    // Load runtime config
    auto cfg = BenchmarkRuntimeConfig::load(configPath);

    // Contention benchmarks use only RT-safe backends on the audio thread.
    // Direct_LibTorch and Direct_ONNX allocate per-call (not RT-safe) and
    // must be evaluated via anira's background-thread scheduler instead.
    // Direct backends are still available in isolated mode for raw speed comparison.
    BackendType backends[] = {
        BackendType::BNNSGraph,
#if defined(RTNEURAL_USE_XSIMD)
        BackendType::RTNeural_XSIMD,
#else
        BackendType::RTNeural_Eigen,
#endif
#if HAS_ANIRA && HAS_LIBTORCH
        BackendType::Anira_LibTorch,
#endif
#if HAS_ANIRA && HAS_ONNXRUNTIME
        BackendType::Anira_ONNX,
#endif
    };
    int numBackends = sizeof(backends) / sizeof(backends[0]);

    for (int si = 0; si < static_cast<int>(ModelSize::COUNT); si++)
    {
        auto size = static_cast<ModelSize>(si);
        if (!cfg.isSizeEnabled(size)) continue;

        for (int mi = 0; mi < static_cast<int>(ModelType::COUNT); mi++)
        {
            auto model = static_cast<ModelType>(mi);
            if (!cfg.isModelEnabled(model)) continue;

            for (int bi = 0; bi < numBackends; bi++)
            {
                if (!cfg.isBackendEnabled(backends[bi])) continue;

                for (auto bufSize : cfg.contentionBufferSizes)
                {
                    for (auto contLevel : cfg.contentionLevels)
                    {
                        for (int rep = 1; rep <= cfg.contentionReps; rep++)
                        {
                            runSingleConfig(csvFile, "dim_a", backends[bi], model, size,
                                            bufSize, contLevel, 1, rep);
                        }
                    }
                }
            }
        }
    }
}

void ContentionBenchmark::runDimensionB(FILE* csvFile)
{
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Dimension B: Instance Count Sweep\n");
    fprintf(stderr, "========================================\n");

    auto cfg = BenchmarkRuntimeConfig::load(configPath);

    // RT-safe backends only (same as Dimension A)
    BackendType backends[] = {
        BackendType::BNNSGraph,
#if defined(RTNEURAL_USE_XSIMD)
        BackendType::RTNeural_XSIMD,
#else
        BackendType::RTNeural_Eigen,
#endif
#if HAS_ANIRA && HAS_LIBTORCH
        BackendType::Anira_LibTorch,
#endif
#if HAS_ANIRA && HAS_ONNXRUNTIME
        BackendType::Anira_ONNX,
#endif
    };
    int numBackends = sizeof(backends) / sizeof(backends[0]);

    for (int si = 0; si < static_cast<int>(ModelSize::COUNT); si++)
    {
        auto size = static_cast<ModelSize>(si);
        if (!cfg.isSizeEnabled(size)) continue;

        for (int mi = 0; mi < static_cast<int>(ModelType::COUNT); mi++)
        {
            auto model = static_cast<ModelType>(mi);
            if (!cfg.isModelEnabled(model)) continue;

            for (int bi = 0; bi < numBackends; bi++)
            {
                if (!cfg.isBackendEnabled(backends[bi])) continue;

                // Dimension B uses a fixed buffer size of 128 samples.
                // Instance scaling at a single buffer size isolates the
                // effect of adding more neural plugins without confounding
                // buffer-size variability.
                const int dimBBufferSize = 128;
                for (auto instCount : cfg.instanceCounts)
                {
                    for (int rep = 1; rep <= cfg.contentionReps; rep++)
                    {
                        runSingleConfig(csvFile, "dim_b", backends[bi], model, size,
                                        dimBBufferSize, 0, instCount, rep);
                    }
                }
            }
        }
    }
}

void ContentionBenchmark::runDimensionC(FILE* csvFile)
{
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Dimension C: Neural Track Serial Depth\n");
    fprintf(stderr, "========================================\n");

    auto cfg = BenchmarkRuntimeConfig::load(configPath);

    // RT-safe backends only (same as Dim A/B)
    BackendType backends[] = {
        BackendType::BNNSGraph,
#if defined(RTNEURAL_USE_XSIMD)
        BackendType::RTNeural_XSIMD,
#else
        BackendType::RTNeural_Eigen,
#endif
#if HAS_ANIRA && HAS_LIBTORCH
        BackendType::Anira_LibTorch,
#endif
#if HAS_ANIRA && HAS_ONNXRUNTIME
        BackendType::Anira_ONNX,
#endif
    };
    int numBackends = sizeof(backends) / sizeof(backends[0]);

    // Fixed buffer size 128, no conventional contention tracks.
    // The "contention_level" CSV column stores the depth value.
    const int dimCBufferSize = 128;

    for (int si = 0; si < static_cast<int>(ModelSize::COUNT); si++)
    {
        auto size = static_cast<ModelSize>(si);
        if (!cfg.isSizeEnabled(size)) continue;

        for (int mi = 0; mi < static_cast<int>(ModelType::COUNT); mi++)
        {
            auto model = static_cast<ModelType>(mi);
            if (!cfg.isModelEnabled(model)) continue;

            for (int bi = 0; bi < numBackends; bi++)
            {
                if (!cfg.isBackendEnabled(backends[bi])) continue;

                for (auto depth : cfg.neuralTrackDepths)
                {
                    for (int rep = 1; rep <= cfg.contentionReps; rep++)
                    {
                        runSingleConfig(csvFile, "dim_c", backends[bi], model, size,
                                        dimCBufferSize, depth, 1, rep);
                    }
                }
            }
        }
    }
}

void ContentionBenchmark::runAll(FILE* csvFile)
{
    CSVOutput::printContentionHeader(csvFile);
    fflush(csvFile);
    runDimensionA(csvFile);
    runDimensionB(csvFile);
    runDimensionC(csvFile);
    fprintf(stderr, "\nContention benchmark complete.\n");
}
