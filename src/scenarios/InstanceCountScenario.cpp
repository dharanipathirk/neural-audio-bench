// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "InstanceCountScenario.h"

namespace nab {

// Moved verbatim from EditBuilder::buildDimensionB. `instanceCount` is the
// current sweep value.
SessionTimingInfo InstanceCountScenario::build(te::Edit& edit, EditBuilder& builder,
                                               const std::string& backend, const ModelSpec& model,
                                               int sweepValue, const BenchmarkRuntimeConfig& /*cfg*/,
                                               double sampleRate)
{
    const int instanceCount = sweepValue;

    builder.registerPluginTypes();

    double clipDuration = 30.0;  // Long enough for warmup + measurement
    builder.ensureNoiseWavFile(clipDuration, sampleRate);

    SessionTimingInfo info;
    info.callbackTimer = std::make_unique<CallbackTimer>();
    info.threadIdLogger = std::make_unique<ThreadIDLogger>();
    info.threadIdLogger->allocate(32);

    edit.ensureNumberOfAudioTracks(instanceCount);
    auto tracks = te::getAudioTracks(edit);

    for (int i = 0; i < instanceCount && i < static_cast<int>(tracks.size()); i++)
    {
        auto* logger = builder.addNeuralPlugin(*tracks[static_cast<size_t>(i)], backend, model, clipDuration, info);
        if (logger)
            info.neuralLoggers.push_back(logger);

        builder.addCallbackStart(*tracks[static_cast<size_t>(i)], info.callbackTimer.get(),
                                 info.threadIdLogger.get());
    }

    if (!info.neuralLoggers.empty())
        info.neuralLogger = info.neuralLoggers[0];

    info.callbackLogger = builder.addCallbackEnd(edit, info.callbackTimer.get());

    return info;
}

} // namespace nab
