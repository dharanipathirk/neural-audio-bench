// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "SerialDepthScenario.h"

#include "../plugins/ContentionPlugins.h"

namespace nab {

// Moved verbatim from EditBuilder::buildDimensionC. `depth` is the current
// sweep value.
SessionTimingInfo SerialDepthScenario::build(te::Edit& edit, EditBuilder& builder,
                                             const std::string& backend, const ModelSpec& model,
                                             int sweepValue, const BenchmarkRuntimeConfig& /*cfg*/,
                                             double sampleRate)
{
    const int depth = sweepValue;

    builder.registerPluginTypes();

    double clipDuration = 30.0;
    builder.ensureNoiseWavFile(clipDuration, sampleRate);

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
    info.neuralLogger = builder.addNeuralPlugin(*track, backend, model, clipDuration, info);

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
    builder.addCallbackStart(*track, info.callbackTimer.get(), info.threadIdLogger.get());
    info.callbackLogger = builder.addCallbackEnd(edit, info.callbackTimer.get());

    return info;
}

} // namespace nab
