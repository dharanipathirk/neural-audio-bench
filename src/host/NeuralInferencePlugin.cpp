// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "NeuralInferencePlugin.h"

namespace tracktion { namespace engine {

const char* NeuralInferencePlugin::xmlTypeName = "neuralInferencePlugin";

NeuralInferencePlugin::NeuralInferencePlugin(PluginCreationInfo info)
    : Plugin(info)
{
}

NeuralInferencePlugin::~NeuralInferencePlugin()
{
    deinitialise();
}

void NeuralInferencePlugin::initialise(const PluginInitialisationInfo& info)
{
    if (backend)
    {
        PrepareContext pc{ info.sampleRate, info.blockSizeSamples, &spec };
        prepared = backend->prepare(pc);

        // Per-backend timing-logger capacity (see InferenceBackend): the
        // on-thread backends use SAMPLE_RATE*30/32, the anira backends use
        // sampleRate*30/blockSize — matching the legacy per-plugin formula.
        timingLogger.allocate(backend->timingLoggerCapacity(info.sampleRate, info.blockSizeSamples));
    }
}

void NeuralInferencePlugin::deinitialise()
{
    if (backend)
        backend->teardown();
}

void NeuralInferencePlugin::applyToBuffer(const PluginRenderContext& ctx)
{
    if (ctx.destBuffer == nullptr || !prepared || !backend)
        return;

    const int numSamples = ctx.bufferNumSamples;
    const int startSample = ctx.bufferStartSample;

    // Non-timed pre-process hook (Anira_LibTorch's out-of-band underrun check,
    // which the old plugin performed before recordStart()).
    backend->preProcess(numSamples);

    // Everything between recordStart/recordEnd matches the old plugins exactly,
    // including the getWritePointer(0) call, which they made inside the window.
    timingLogger.recordStart();
    auto* data = ctx.destBuffer->getWritePointer(0) + startSample;
    backend->process(data, data, numSamples);
    timingLogger.recordEnd();
}

void NeuralInferencePlugin::reset()
{
    if (backend)
        backend->reset();
}

}} // namespace tracktion::engine
