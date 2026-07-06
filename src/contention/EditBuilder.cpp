// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "EditBuilder.h"
#include "../host/NeuralInferencePlugin.h"
#include "../backends/BackendRegistry.h"
#include "../plugins/ContentionPlugins.h"

#include <random>

// ---------------------------------------------------------------------------
// Shared neural-plugin attach (used by EditBuilder and the scenario session
// builders, e.g. SystemAuSessionBuilder).
// ---------------------------------------------------------------------------
TimingLogger* attachNeuralPlugin(te::AudioTrack& track, BackendType backend,
                                 const ModelSpec* spec, const juce::String& pluginName,
                                 SessionTimingInfo& sessionInfo)
{
    if (spec == nullptr)
        return nullptr;

    auto plugin = track.edit.getPluginCache().createNewPlugin(
        te::NeuralInferencePlugin::xmlTypeName, {});
    auto* p = dynamic_cast<te::NeuralInferencePlugin*>(plugin.get());
    if (p == nullptr)
        return nullptr;

    auto backendImpl = BackendRegistry::instance().create(backendTypeName(backend));
    if (!backendImpl)
        return nullptr;

    p->setBackend(std::move(backendImpl));
    p->setModelSpec(*spec);
    p->setPluginName(pluginName);
    track.pluginList.insertPlugin(plugin, 0, nullptr);

    // Wire underrun getter/resetter uniformly (non-anira backends return 0).
    sessionInfo.inferenceUnderrunGetters.push_back([p]() { return p->getInferenceUnderruns(); });
    sessionInfo.inferenceUnderrunResetters.push_back([p]() { p->resetInferenceUnderruns(); });

    return &p->getTimingLogger();
}

// ---------------------------------------------------------------------------
// EditBuilder — shared session-building toolbox
// ---------------------------------------------------------------------------

EditBuilder::EditBuilder(te::Engine& engine, const std::vector<ModelSpec>& specs)
    : engine(engine), specs(specs) {}

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
    pm.createBuiltInType<te::NeuralInferencePlugin>();
    pm.createBuiltInType<te::ContentionEQPlugin>();
    pm.createBuiltInType<te::ContentionCompPlugin>();
    pm.createBuiltInType<te::ContentionReverbPlugin>();
    pm.createBuiltInType<te::ContentionDelayPlugin>();
    pm.createBuiltInType<te::NoiseGeneratorPlugin>();
    pm.createBuiltInType<te::CallbackStartPlugin>();
    pm.createBuiltInType<te::CallbackEndPlugin>();

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

    // Plugin display name — kept byte-identical to the pre-refactor EditBuilder.
    juce::String name;
    switch (backend)
    {
        case BackendType::BNNSGraph:
            name = "BNNS_" + juce::String(modelTypeName(model)); break;
        case BackendType::RTNeural_Eigen:
        case BackendType::RTNeural_XSIMD:
            name = "RTNeural_" + juce::String(modelTypeName(model)); break;
        case BackendType::Direct_LibTorch:
            name = "DirectLibTorch_" + juce::String(modelTypeName(model)); break;
        case BackendType::Direct_ONNX:
            name = "DirectONNX_" + juce::String(modelTypeName(model)); break;
        case BackendType::Anira_LibTorch:
            name = "AniraLibTorch_" + juce::String(modelTypeName(model)); break;
        case BackendType::Anira_ONNX:
            name = "AniraONNX_" + juce::String(modelTypeName(model)); break;
        default:
            break;
    }

    const ModelSpec* spec = nab::findModelSpec(specs, model, size);
    return attachNeuralPlugin(track, backend, spec, name, sessionInfo);
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
