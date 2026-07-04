// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "TracktionPlaybackTest.h"

// ---------------------------------------------------------------------------
// Helper: create a WAV file filled with noise
// ---------------------------------------------------------------------------
juce::File TracktionPlaybackTest::createNoiseWavFile(double durationSeconds, double sampleRate)
{
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
    auto wavFile = tempDir.getChildFile("noise_source.wav");
    wavFile.deleteFile();

    juce::WavAudioFormat wavFormat;
    std::unique_ptr<juce::AudioFormatWriter> writer(
        wavFormat.createWriterFor(new juce::FileOutputStream(wavFile),
                                   sampleRate, 1, 16, {}, 0));
    if (writer)
        writer->writeFromAudioSampleBuffer(buffer, 0, numSamples);

    return wavFile;
}

// ---------------------------------------------------------------------------
// Register plugin types
// ---------------------------------------------------------------------------
void TracktionPlaybackTest::registerPluginTypes()
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

    pluginsRegistered = true;
}

// ---------------------------------------------------------------------------
// Run the test — real-time playback via CoreAudio
// ---------------------------------------------------------------------------
bool TracktionPlaybackTest::run()
{
    using namespace tracktion::engine;

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Tracktion Real-Time Playback Test\n");
    fprintf(stderr, "========================================\n");

    registerPluginTypes();

    const double testSampleRate = SAMPLE_RATE;
    const int testBlockSize = 128;
    const double testDuration = 3.0; // 3 seconds of playback

    // --- Step 1: Configure audio device ---
    fprintf(stderr, "  Configuring audio device...\n");
    auto& dm = engine.getDeviceManager();
    dm.initialise(0, 2); // 0 inputs, 2 outputs (stereo)

    auto* device = dm.deviceManager.getCurrentAudioDevice();
    if (!device)
    {
        fprintf(stderr, "  ERROR: No audio device available!\n");
        fprintf(stderr, "  Ensure an audio output device is available (e.g., BlackHole, speakers).\n");
        return false;
    }
    fprintf(stderr, "  Audio device: %s\n", device->getName().toRawUTF8());

    // Set sample rate and buffer size
    auto setup = dm.deviceManager.getAudioDeviceSetup();
    setup.sampleRate = testSampleRate;
    setup.bufferSize = testBlockSize;
    auto err = dm.deviceManager.setAudioDeviceSetup(setup, true);
    if (err.isNotEmpty())
    {
        fprintf(stderr, "  WARNING: Could not set exact audio config: %s\n", err.toRawUTF8());
        fprintf(stderr, "  Using device defaults instead.\n");
    }

    // Re-read actual device settings
    device = dm.deviceManager.getCurrentAudioDevice();
    double actualSR = device->getCurrentSampleRate();
    int actualBS = device->getCurrentBufferSizeSamples();
    fprintf(stderr, "  Actual: sr=%.0f bs=%d\n", actualSR, actualBS);

    // --- Step 2: Create noise WAV file ---
    fprintf(stderr, "  Creating noise WAV file...\n");
    auto noiseFile = createNoiseWavFile(testDuration + 2.0, actualSR);
    if (!noiseFile.existsAsFile())
    {
        fprintf(stderr, "  ERROR: Failed to create noise WAV file\n");
        return false;
    }

    // --- Step 3: Create Edit with tracks + plugins ---
    fprintf(stderr, "  Creating Edit with 4 tracks...\n");
    auto edit = Edit::createSingleTrackEdit(engine);
    edit->ensureNumberOfAudioTracks(4);
    edit->tempoSequence.getTempo(0)->setBpm(60.0);
    edit->getMasterVolumePlugin()->setVolumeDb(0.0f);

    auto tracks = getAudioTracks(*edit);
    auto clipDuration = tracktion::TimeDuration::fromSeconds(testDuration + 2.0);

    // Track 0: WAV clip + BNNSGraph LSTM
    TimingLogger* neuralLogger = nullptr;
    {
        auto* track = tracks[0];
        track->insertWaveClip("Noise", noiseFile, { { {}, clipDuration } }, false);
        auto plugin = edit->getPluginCache().createNewPlugin(BNNSGraphPlugin::xmlTypeName, {});
        if (auto* p = dynamic_cast<BNNSGraphPlugin*>(plugin.get()))
        {
            p->setModelPath(modelCoreMLPath(ModelType::LSTM, ModelSize::Small, modelDir));
            p->setPluginName("Test_BNNS_LSTM");
            track->pluginList.insertPlugin(plugin, 0, nullptr);
            neuralLogger = &p->getTimingLogger();
        }
        track->setMute(false);
        fprintf(stderr, "  Track 0: WAV clip + BNNSGraph LSTM\n");
    }

    // Track 1: WAV clip + EQ + Comp
    {
        auto* track = tracks[1];
        track->insertWaveClip("Noise", noiseFile, { { {}, clipDuration } }, false);
        auto eq = edit->getPluginCache().createNewPlugin(ContentionEQPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(eq, 0, nullptr);
        auto comp = edit->getPluginCache().createNewPlugin(ContentionCompPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(comp, -1, nullptr);
        track->setMute(false);
        fprintf(stderr, "  Track 1: WAV clip + EQ + Comp\n");
    }

    // Track 2: WAV clip + Reverb
    {
        auto* track = tracks[2];
        track->insertWaveClip("Noise", noiseFile, { { {}, clipDuration } }, false);
        auto reverb = edit->getPluginCache().createNewPlugin(ContentionReverbPlugin::xmlTypeName, {});
        track->pluginList.insertPlugin(reverb, 0, nullptr);
        track->setMute(false);
        fprintf(stderr, "  Track 2: WAV clip + Reverb\n");
    }

    // Track 3: WAV clip only (baseline)
    {
        auto* track = tracks[3];
        track->insertWaveClip("Noise", noiseFile, { { {}, clipDuration } }, false);
        track->setMute(false);
        fprintf(stderr, "  Track 3: WAV clip only\n");
    }

    // --- Step 4: Ensure output routing for all tracks ---
    for (auto* track : tracks)
        track->getOutput().setOutputToDefaultDevice(false);

    // --- Step 5: Start real-time playback ---
    fprintf(stderr, "  Starting real-time playback...\n");

    auto& transport = edit->getTransport();
    transport.setPosition(tracktion::TimePosition());

    // Pump message loop to let device manager fully initialise
    juce::MessageManager::getInstance()->runDispatchLoopUntil(500);

    // Allocate playback context (builds the node graph)
    transport.ensureContextAllocated(true);

    // Pump to let context build complete
    juce::MessageManager::getInstance()->runDispatchLoopUntil(500);

    // Start playback
    transport.play(false);

    // Pump to let the play command propagate through ValueTree listeners
    juce::MessageManager::getInstance()->runDispatchLoopUntil(500);

    fprintf(stderr, "  Transport playing: %s\n", transport.isPlaying() ? "YES" : "NO");
    fprintf(stderr, "  Playback context active: %s\n",
            transport.isPlayContextActive() ? "YES" : "NO");

    // --- Step 6: Wait for playback duration ---
    fprintf(stderr, "  Waiting %.1f seconds for real-time playback...\n", testDuration);

    int waitMs = static_cast<int>(testDuration * 1000.0);
    int elapsed = 0;
    while (elapsed < waitMs)
    {
        juce::MessageManager::getInstance()->runDispatchLoopUntil(100);
        elapsed += 100;
    }

    // --- Step 7: Stop playback ---
    fprintf(stderr, "  Stopping playback...\n");
    transport.stop(false, false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    // --- Step 8: Check results ---
    fprintf(stderr, "\n  --- Results ---\n");

    auto pos = transport.getPosition();
    fprintf(stderr, "  Transport position: %.2fs\n", pos.inSeconds());

    int neuralCallCount = neuralLogger ? neuralLogger->getCount() : 0;
    fprintf(stderr, "  BNNSGraph applyToBuffer calls: %d\n", neuralCallCount);

    bool success = (neuralCallCount > 0);
    if (success)
    {
        fprintf(stderr, "  SUCCESS: Plugins called via real-time CoreAudio callback!\n");

        auto durations = neuralLogger->getDurationsNs();
        if (!durations.empty())
        {
            double deadline_ns = TimingUtils::bufferDurationNs(actualBS, actualSR);
            auto stats = TimingStats::compute(durations, deadline_ns);
            fprintf(stderr, "  Neural timing: median=%.0fns p99=%.0fns RTF=%.4f dropouts=%d/%d\n",
                    stats.median_ns, stats.p99_ns, stats.rtf,
                    stats.dropout_count, stats.total_samples);
            fprintf(stderr, "  Utilization: p50=%.1f%% p95=%.1f%% p99=%.1f%% max=%.1f%%\n",
                    stats.util_p50, stats.util_p95, stats.util_p99, stats.util_max);
        }
    }
    else
    {
        fprintf(stderr, "  FAILURE: applyToBuffer was never called during real-time playback!\n");
    }

    // Cleanup
    transport.freePlaybackContext();
    noiseFile.deleteFile();

    fprintf(stderr, "========================================\n\n");
    return success;
}
