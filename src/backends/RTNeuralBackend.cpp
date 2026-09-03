// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "RTNeuralBackend.h"
#include "RTNeuralTopology.h"

#include <algorithm>

// ---------------------------------------------------------------------------
// RTNeuralEngine — dispatches to compile-time templates based on model+size
// ---------------------------------------------------------------------------

bool RTNeuralEngine::initialize(ModelType modelType, ModelSize modelSize, const std::string& weightsPath)
{
    (void) weightsPath;
    currentModel = modelType;
    currentSize = modelSize;

    // Reset all models for the selected size
    switch (currentSize)
    {
        case ModelSize::Small:
            lstmSmall.reset(); tcnSmall.reset(); wavenetSmall.reset(); break;
        case ModelSize::Medium:
            lstmMedium.reset(); tcnMedium.reset(); wavenetMedium.reset(); break;
        case ModelSize::Large:
            lstmLarge.reset(); tcnLarge.reset(); wavenetLarge.reset(); break;
        default: break;
    }

    // Weights: RTNeural's compile-time templates use their default zero/template
    // initialization, which differs from the Python-seeded weights exported
    // for BNNSGraph/LibTorch/ONNX backends. This is acceptable because:
    // 1. This is a speed benchmark — timing depends on architecture and
    //    parameter count, not weight values.
    // 2. For LSTM/TCN/WaveNet at these sizes, weight values do not affect
    //    compute time (no conditional sparsity, no early-exit paths).
    // 3. The JSON export exists for reference/validation but is not loaded.

    // Prepare arena for buffer processing (TCN and WaveNet only)
    static constexpr int kMaxBlock = 2048;
    switch (currentSize)
    {
        case ModelSize::Small:  tcnSmall.prepare(kMaxBlock);  wavenetSmall.prepare(kMaxBlock);  break;
        case ModelSize::Medium: tcnMedium.prepare(kMaxBlock); wavenetMedium.prepare(kMaxBlock); break;
        case ModelSize::Large:  tcnLarge.prepare(kMaxBlock);  wavenetLarge.prepare(kMaxBlock);  break;
        default: break;
    }

    initialized = true;
    fprintf(stderr, "RTNeuralEngine: Initialized %s/%s (template-initialized weights, buffer-at-a-time for TCN/WaveNet)\n",
            modelTypeName(modelType), modelSizeName(modelSize));
    return true;
}

float RTNeuralEngine::processSample(float input)
{
    float in[1] = {input};

    switch (currentSize)
    {
        case ModelSize::Small:
            switch (currentModel)
            {
                case ModelType::LSTM:    return lstmSmall.forward(in);
                case ModelType::TCN:     return tcnSmall.forward(in);
                case ModelType::WaveNet: return wavenetSmall.forward(in);
                default: return input;
            }
        case ModelSize::Medium:
            switch (currentModel)
            {
                case ModelType::LSTM:    return lstmMedium.forward(in);
                case ModelType::TCN:     return tcnMedium.forward(in);
                case ModelType::WaveNet: return wavenetMedium.forward(in);
                default: return input;
            }
        case ModelSize::Large:
            switch (currentModel)
            {
                case ModelType::LSTM:    return lstmLarge.forward(in);
                case ModelType::TCN:     return tcnLarge.forward(in);
                case ModelType::WaveNet: return wavenetLarge.forward(in);
                default: return input;
            }
        default:
            return input;
    }
}

void RTNeuralEngine::processBlock(const float* input, float* output, int numSamples)
{
    // LSTM: per-sample (ModelT has no buffer overload)
    // TCN/WaveNet: buffer forward with arena (RTNeural-NAM pattern)
    if (currentModel == ModelType::LSTM)
    {
        for (int i = 0; i < numSamples; i++)
            output[i] = processSample(input[i]);
        return;
    }

    // Process in chunks of kMaxBlock for TCN/WaveNet
    static constexpr int kMaxBlock = 2048;
    int offset = 0;
    while (offset < numSamples)
    {
        int chunk = std::min(numSamples - offset, kMaxBlock);

        switch (currentSize)
        {
            case ModelSize::Small:
                if (currentModel == ModelType::TCN)
                    tcnSmall.forward(input + offset, output + offset, chunk);
                else
                    wavenetSmall.forward(input + offset, output + offset, chunk);
                break;
            case ModelSize::Medium:
                if (currentModel == ModelType::TCN)
                    tcnMedium.forward(input + offset, output + offset, chunk);
                else
                    wavenetMedium.forward(input + offset, output + offset, chunk);
                break;
            case ModelSize::Large:
                if (currentModel == ModelType::TCN)
                    tcnLarge.forward(input + offset, output + offset, chunk);
                else
                    wavenetLarge.forward(input + offset, output + offset, chunk);
                break;
            default: break;
        }
        offset += chunk;
    }
}

void RTNeuralEngine::resetState()
{
    switch (currentSize)
    {
        case ModelSize::Small:
            lstmSmall.reset(); tcnSmall.reset(); wavenetSmall.reset(); break;
        case ModelSize::Medium:
            lstmMedium.reset(); tcnMedium.reset(); wavenetMedium.reset(); break;
        case ModelSize::Large:
            lstmLarge.reset(); tcnLarge.reset(); wavenetLarge.reset(); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// RTNeuralBackend adapter
// ---------------------------------------------------------------------------

bool RTNeuralBackend::mapArchSize(const ModelSpec& spec, ModelType& m, ModelSize& s)
{
    if (spec.arch == "lstm")         m = ModelType::LSTM;
    else if (spec.arch == "tcn")     m = ModelType::TCN;
    else if (spec.arch == "wavenet") m = ModelType::WaveNet;
    else return false;

    if (spec.size == "small")        s = ModelSize::Small;
    else if (spec.size == "medium")  s = ModelSize::Medium;
    else if (spec.size == "large")   s = ModelSize::Large;
    else return false;

    return true;
}

bool RTNeuralBackend::supports(const ModelSpec& spec, std::string& whyNot) const
{
    ModelType m; ModelSize s;
    if (!mapArchSize(spec, m, s))
    {
        whyNot = std::string("RTNeural has no compiled variant for arch=") + spec.arch +
                 " size=" + spec.size;
        return false;
    }

    if (!InferenceBackend::supports(spec, whyNot))
        return false;

    return nab::validateRTNeuralCompiledTopology(spec, whyNot);
}

bool RTNeuralBackend::prepare(const PrepareContext& ctx)
{
    if (ctx.model == nullptr)
        return false;

    ModelType m; ModelSize s;
    if (!mapArchSize(*ctx.model, m, s))
        return false;

    // RTNeural uses template-initialized weights (see initialize()); the
    // weights path is passed through for parity but is not loaded.
    std::string weights;
    auto it = ctx.model->formatPaths.find("rtneural");
    if (it != ctx.model->formatPaths.end())
        weights = it->second;

    return engine.initialize(m, s, weights);
}
