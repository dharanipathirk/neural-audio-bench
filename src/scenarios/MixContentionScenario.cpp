// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "MixContentionScenario.h"

#include "../plugins/ContentionPlugins.h"

#include <algorithm>

namespace nab {

SessionTimingInfo MixContentionScenario::build(te::Edit& edit, EditBuilder& builder,
                                               const std::string& backend, const ModelSpec& model,
                                               int sweepValue, const BenchmarkRuntimeConfig& cfg,
                                               double sampleRate)
{
    // use_system_au: real macOS AUs for Dimension A contention.
    // Lazy scan: only probe for AUs on the first dim_a build.
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
            return auBuilder.buildSession(edit, backend, model, sweepValue, sampleRate);

        return buildCustomDsp(edit, builder, backend, model, sweepValue, sampleRate,
                              cfg.contentionNumTracks);
    }

    return buildCustomDsp(edit, builder, backend, model, sweepValue, sampleRate,
                          cfg.contentionNumTracks);
}

// ---------------------------------------------------------------------------
// Custom-DSP fallback layout — moved verbatim from EditBuilder::buildDimensionA.
// ---------------------------------------------------------------------------
SessionTimingInfo MixContentionScenario::buildCustomDsp(te::Edit& edit, EditBuilder& builder,
                                                        const std::string& backend,
                                                        const ModelSpec& model, int activeTracks,
                                                        double sampleRate, int numTracks)
{
    builder.registerPluginTypes();

    const int maxConventionalTracks = std::max(0, numTracks - 1);
    const int effectiveActiveTracks = std::clamp(activeTracks, 0, maxConventionalTracks);
    if (effectiveActiveTracks != activeTracks)
    {
        fprintf(stderr,
                "  WARNING: requested Dimension A contention level %d maps to %d "
                "conventional tracks in the %d-track custom-DSP layout; the CSV "
                "retains the requested level\n",
                activeTracks, effectiveActiveTracks, numTracks);
    }

    // Audio clips are mono.  The session layout spec mentions stereo tracks
    // (overheads, room mics, synth pad) but this benchmark uses mono throughout
    // for simplicity.  This underestimates the CPU cost of stereo processing
    // chains but does not affect the relative comparison between backends.
    double clipDuration = 30.0;  // Long enough for warmup + measurement
    builder.ensureNoiseWavFile(clipDuration, sampleRate);

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
            info.neuralLogger = builder.addNeuralPlugin(*track, backend, model, clipDuration, info);

            auto eqPlugin = edit.getPluginCache().createNewPlugin(
                te::ContentionEQPlugin::xmlTypeName, {});
            track->pluginList.insertPlugin(eqPlugin, -1, nullptr);

            auto compPlugin = edit.getPluginCache().createNewPlugin(
                te::ContentionCompPlugin::xmlTypeName, {});
            track->pluginList.insertPlugin(compPlugin, -1, nullptr);

            // Add callback start FIRST on this track (before neural plugin)
            builder.addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
        }
        else if (conventionalCount < effectiveActiveTracks)
        {
            builder.addConventionalDSP(*track, clipDuration);
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
            builder.addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
        }
        else
        {
            track->setMute(true);
        }
    }

    // Add callback end on master bus (runs after all tracks mixed)
    info.callbackLogger = builder.addCallbackEnd(edit, info.callbackTimer.get());

    return info;
}

} // namespace nab
