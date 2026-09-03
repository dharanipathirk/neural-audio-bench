// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "ConfigurableScenario.h"

#include "../plugins/ContentionPlugins.h"

namespace nab {

namespace {

// Map a chain element name to a contention DSP plugin and insert it on the
// track at `index` (-1 = append). Returns false (with a warning) for unknown
// names, so a typo in the config is visible but non-fatal.
bool insertContentionPlugin(te::Edit& edit, te::AudioTrack& track,
                            const std::string& name, int index)
{
    const char* xmlType = nullptr;
    if (name == "eq")              xmlType = te::ContentionEQPlugin::xmlTypeName;
    else if (name == "compressor") xmlType = te::ContentionCompPlugin::xmlTypeName;
    else if (name == "reverb")     xmlType = te::ContentionReverbPlugin::xmlTypeName;
    else if (name == "delay")      xmlType = te::ContentionDelayPlugin::xmlTypeName;

    if (xmlType == nullptr)
    {
        fprintf(stderr, "  WARNING: custom scenario: unknown chain plugin '%s' (skipped)\n",
                name.c_str());
        return false;
    }

    auto plugin = edit.getPluginCache().createNewPlugin(xmlType, {});
    track.pluginList.insertPlugin(plugin, index, nullptr);
    return true;
}

} // namespace

SessionTimingInfo ConfigurableScenario::build(te::Edit& edit, EditBuilder& builder,
                                              const std::string& backend, const ModelSpec& model,
                                              int sweepValue, const BenchmarkRuntimeConfig& /*cfg*/,
                                              double sampleRate)
{
    builder.registerPluginTypes();

    double clipDuration = 30.0;  // Long enough for warmup + measurement
    builder.ensureNoiseWavFile(clipDuration, sampleRate);

    SessionTimingInfo info;
    info.callbackTimer = std::make_unique<CallbackTimer>();
    info.threadIdLogger = std::make_unique<ThreadIDLogger>();
    info.threadIdLogger->allocate(32);

    // Resolve the total number of tracks across all groups (a "sweep" group
    // contributes `sweepValue` tracks).
    int totalTracks = 0;
    for (const auto& ts : spec.tracks)
        totalTracks += ts.countIsSweep ? sweepValue : ts.count;
    if (totalTracks < 1)
        totalTracks = 1;

    edit.ensureNumberOfAudioTracks(totalTracks);
    auto tracks = te::getAudioTracks(edit);

    int trackIdx = 0;
    for (const auto& ts : spec.tracks)
    {
        int groupCount = ts.countIsSweep ? sweepValue : ts.count;
        for (int k = 0; k < groupCount; k++)
        {
            if (trackIdx >= static_cast<int>(tracks.size()))
                break;
            auto* track = tracks[static_cast<size_t>(trackIdx++)];

            if (ts.neural)
            {
                // addNeuralPlugin inserts the neural plugin at position 0 and
                // adds the audio clip. Pre-neural "chain" is then inserted at 0
                // in reverse so it ends up before the neural plugin; "chain_after"
                // is appended at the end.
                auto* logger = builder.addNeuralPlugin(*track, backend, model, clipDuration, info);
                if (logger)
                    info.neuralLoggers.push_back(logger);

                for (auto it = ts.chain.rbegin(); it != ts.chain.rend(); ++it)
                    insertContentionPlugin(edit, *track, *it, 0);

                for (const auto& name : ts.chainAfter)
                    insertContentionPlugin(edit, *track, name, -1);
            }
            else
            {
                if (ts.clip == "noise")
                    builder.addAudioClip(*track, clipDuration);

                for (const auto& name : ts.chain)
                    insertContentionPlugin(edit, *track, name, -1);
                for (const auto& name : ts.chainAfter)
                    insertContentionPlugin(edit, *track, name, -1);
            }

            // Every track gets a CallbackStart (inserted at 0 → runs first).
            builder.addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
        }
    }

    if (!info.neuralLoggers.empty())
        info.neuralLogger = info.neuralLoggers[0];

    info.callbackLogger = builder.addCallbackEnd(edit, info.callbackTimer.get());

    return info;
}

} // namespace nab
