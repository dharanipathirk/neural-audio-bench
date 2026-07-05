// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "AniraBackend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#if HAS_ANIRA && defined(USE_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

namespace {

#if HAS_ANIRA && defined(USE_ONNXRUNTIME)
struct OnnxModelInfo
{
    bool hasExplicitStateLstm = false;
    bool hasStateInputs = false;
    bool hasStateOutputs = false;
    int hiddenSize = 0;
};

OnnxModelInfo inspectOnnxModel(const std::string& modelPath)
{
    OnnxModelInfo info;

    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "anira_onnx_inspect");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        Ort::Session session(env, modelPath.c_str(), opts);
        Ort::AllocatorWithDefaultOptions alloc;

        size_t numInputs = session.GetInputCount();
        bool hasHIn = false;
        bool hasCIn = false;
        for (size_t i = 0; i < numInputs; ++i)
        {
            auto name = session.GetInputNameAllocated(i, alloc);
            if (std::strcmp(name.get(), "h_in") == 0)
            {
                hasHIn = true;
                auto typeInfo = session.GetInputTypeInfo(i);
                auto shape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
                if (!shape.empty() && shape.back() > 0)
                    info.hiddenSize = static_cast<int>(shape.back());
            }
            else if (std::strcmp(name.get(), "c_in") == 0)
            {
                hasCIn = true;
            }
        }
        info.hasStateInputs = hasHIn && hasCIn;

        bool hasHOut = false;
        bool hasCOut = false;
        const size_t numOutputs = session.GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i)
        {
            auto name = session.GetOutputNameAllocated(i, alloc);
            if (std::strcmp(name.get(), "h_out") == 0)
                hasHOut = true;
            else if (std::strcmp(name.get(), "c_out") == 0)
                hasCOut = true;
        }
        info.hasStateOutputs = hasHOut && hasCOut;
        info.hasExplicitStateLstm = info.hasStateInputs && info.hasStateOutputs;
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "    AniraOnnxHandler: ONNX inspect failed for %s: %s\n",
                modelPath.c_str(), e.what());
    }

    return info;
}
#endif

} // namespace

// ===========================================================================
// AniraLibTorchBackend
// ===========================================================================

bool AniraLibTorchBackend::prepare(const PrepareContext& ctx)
{
#if HAS_ANIRA && defined(USE_LIBTORCH)
    if (ctx.model == nullptr)
        return false;
    auto it = ctx.model->formatPaths.find("torchscript");
    if (it == ctx.model->formatPaths.end())
        return false;
    const std::string modelPath = it->second;

    int blockSize = ctx.maxBlockSize;
    double sampleRate = ctx.sampleRate;

    // Model processes [1, blockSize] -> [1, blockSize] (mono, block-at-a-time)
    config = std::make_unique<anira::InferenceConfig>(
        std::vector<anira::ModelData>{
            anira::ModelData(modelPath, anira::InferenceBackend::LIBTORCH)
        },
        std::vector<anira::TensorShape>{
            anira::TensorShape(
                anira::TensorShapeList{{1, 1, static_cast<int64_t>(blockSize)}},
                anira::TensorShapeList{{1, 1, static_cast<int64_t>(blockSize)}}
            )
        },
        anira::ProcessingSpec(
            {1},                                    // 1 input channel
            {1},                                    // 1 output channel
            {static_cast<size_t>(blockSize)},       // preprocess input size
            {static_cast<size_t>(blockSize)}        // postprocess output size
        ),
        static_cast<float>(blockSize) / static_cast<float>(sampleRate) * 1000.0f  // max inference time: 1x buffer duration
    );

    processor = std::make_unique<anira::PrePostProcessor>(*config);
    handler = std::make_unique<anira::InferenceHandler>(*processor, *config);
    handler->set_inference_backend(anira::InferenceBackend::LIBTORCH);
    handler->prepare(anira::HostConfig(
        static_cast<float>(blockSize),
        static_cast<float>(sampleRate),
        true
    ));

    latency = static_cast<int>(handler->get_latency());
    fprintf(stderr, "    AniraLibTorchHandler: model=%s block=%d latency=%u samples\n",
            modelPath.c_str(), blockSize, handler->get_latency());
    return true;
#else
    (void)ctx;
    fprintf(stderr, "    AniraLibTorchHandler: anira or LibTorch not available\n");
    return false;
#endif
}

void AniraLibTorchBackend::preProcess(int n) noexcept
{
#if HAS_ANIRA
    if (!handler) return;

    // Check if background inference finished in time — if available samples
    // are less than the buffer size, the output will be stale/repeated.
    // (In the original AniraLibTorchHandlerPlugin this ran BEFORE recordStart,
    // i.e. outside the timing window.)
    size_t available = handler->get_available_samples(0, 0);
    if (available < static_cast<size_t>(n))
        inferenceUnderruns++;
#else
    (void)n;
#endif
}

void AniraLibTorchBackend::process(const float* in, float* out, int n) noexcept
{
#if HAS_ANIRA
    (void)in;
    if (!handler) return;

    float* channelData = out;
    float* channels[] = { channelData };
    handler->process(channels, static_cast<size_t>(n));
#else
    (void)in; (void)out; (void)n;
#endif
}

void AniraLibTorchBackend::reset() noexcept
{
#if HAS_ANIRA
    if (handler)
        handler->reset();
#endif
}

void AniraLibTorchBackend::teardown()
{
#if HAS_ANIRA
    handler.reset();
    processor.reset();
    config.reset();
#endif
}

// ===========================================================================
// AniraOnnxBackend
// ===========================================================================

bool AniraOnnxBackend::prepare(const PrepareContext& ctx)
{
#if HAS_ANIRA && defined(USE_ONNXRUNTIME)
    if (ctx.model == nullptr)
        return false;
    auto it = ctx.model->formatPaths.find("onnx");
    if (it == ctx.model->formatPaths.end())
        return false;
    const std::string modelPath = it->second;

    int blockSize = ctx.maxBlockSize;
    double sampleRate = ctx.sampleRate;
    const float maxInferenceTimeMs =
        static_cast<float>(blockSize) / static_cast<float>(sampleRate) * 1000.0f;

    const auto modelInfo = inspectOnnxModel(modelPath);
    hasExplicitStateLstm = modelInfo.hasExplicitStateLstm && modelInfo.hiddenSize > 0;
    hiddenSize = hasExplicitStateLstm ? modelInfo.hiddenSize : 0;

    if (modelInfo.hasStateInputs != modelInfo.hasStateOutputs)
    {
        fprintf(stderr,
                "    AniraOnnxHandler: ONNX model has partial state I/O for %s; using single-tensor mode\n",
                modelPath.c_str());
    }

    if (modelInfo.hasExplicitStateLstm && hiddenSize <= 0)
    {
        fprintf(stderr,
                "    AniraOnnxHandler: explicit state detected but hidden size is invalid for %s; falling back to single-tensor mode\n",
                modelPath.c_str());
    }

    if (hasExplicitStateLstm)
    {
        audioInputScratch.assign(static_cast<size_t>(blockSize), 0.0f);
        hStateIn.assign(static_cast<size_t>(hiddenSize), 0.0f);
        cStateIn.assign(static_cast<size_t>(hiddenSize), 0.0f);
        hStateOut.assign(static_cast<size_t>(hiddenSize), 0.0f);
        cStateOut.assign(static_cast<size_t>(hiddenSize), 0.0f);

        config = std::make_unique<anira::InferenceConfig>(
            std::vector<anira::ModelData>{
                anira::ModelData(modelPath, anira::InferenceBackend::ONNX)
            },
            std::vector<anira::TensorShape>{
                anira::TensorShape(
                    anira::TensorShapeList{
                        {1, 1, static_cast<int64_t>(blockSize)},
                        {1, 1, static_cast<int64_t>(hiddenSize)},
                        {1, 1, static_cast<int64_t>(hiddenSize)}
                    },
                    anira::TensorShapeList{
                        {1, 1, static_cast<int64_t>(blockSize)},
                        {1, 1, static_cast<int64_t>(hiddenSize)},
                        {1, 1, static_cast<int64_t>(hiddenSize)}
                    }
                )
            },
            anira::ProcessingSpec(
                {1, 1, 1},
                {1, 1, 1},
                {static_cast<size_t>(blockSize), 0, 0},
                {static_cast<size_t>(blockSize), 0, 0}
            ),
            maxInferenceTimeMs,
            0,
            true,
            0.0f,
            1
        );
    }
    else
    {
        config = std::make_unique<anira::InferenceConfig>(
            std::vector<anira::ModelData>{
                anira::ModelData(modelPath, anira::InferenceBackend::ONNX)
            },
            std::vector<anira::TensorShape>{
                anira::TensorShape(
                    anira::TensorShapeList{{1, 1, static_cast<int64_t>(blockSize)}},
                    anira::TensorShapeList{{1, 1, static_cast<int64_t>(blockSize)}}
                )
            },
            anira::ProcessingSpec(
                {1},
                {1},
                {static_cast<size_t>(blockSize)},
                {static_cast<size_t>(blockSize)}
            ),
            maxInferenceTimeMs
        );
    }

    processor = std::make_unique<anira::PrePostProcessor>(*config);
    handler = std::make_unique<anira::InferenceHandler>(*processor, *config);
    handler->set_inference_backend(anira::InferenceBackend::ONNX);
    handler->prepare(anira::HostConfig(
        static_cast<float>(blockSize),
        static_cast<float>(sampleRate),
        true
    ));

    latency = static_cast<int>(handler->get_latency());

    if (hasExplicitStateLstm)
    {
        fprintf(stderr,
                "    AniraOnnxHandler: LSTM explicit state model=%s block=%d hidden=%d latency=%u samples\n",
                modelPath.c_str(), blockSize, hiddenSize, handler->get_latency());
    }
    else
    {
        fprintf(stderr, "    AniraOnnxHandler: model=%s block=%d latency=%u samples\n",
                modelPath.c_str(), blockSize, handler->get_latency());
    }
    return true;
#else
    (void)ctx;
    fprintf(stderr, "    AniraOnnxHandler: anira or ONNX Runtime not available\n");
    return false;
#endif
}

void AniraOnnxBackend::process(const float* in, float* out, int n) noexcept
{
#if HAS_ANIRA
    (void)in;
    if (!handler) return;

    float* channelData = out;
    const size_t numSamples = static_cast<size_t>(n);

    if (hasExplicitStateLstm)
    {
        std::memcpy(audioInputScratch.data(), channelData, numSamples * sizeof(float));

        float* audioOutChannels[] = { channelData };
        float* hOutChannels[] = { hStateOut.data() };
        float* cOutChannels[] = { cStateOut.data() };
        float* const* outputTensors[] = { audioOutChannels, hOutChannels, cOutChannels };
        size_t numOutputSamples[] = { numSamples, hStateOut.size(), cStateOut.size() };
        size_t* receivedOutput = handler->pop_data(outputTensors, numOutputSamples);

        if (receivedOutput[0] < numSamples)
        {
            inferenceUnderruns++;
        }
        else
        {
            std::copy(hStateOut.begin(), hStateOut.end(), hStateIn.begin());
            std::copy(cStateOut.begin(), cStateOut.end(), cStateIn.begin());
        }

        const float* audioInChannels[] = { audioInputScratch.data() };
        const float* hInChannels[] = { hStateIn.data() };
        const float* cInChannels[] = { cStateIn.data() };
        const float* const* inputTensors[] = { audioInChannels, hInChannels, cInChannels };
        size_t numInputSamples[] = { numSamples, hStateIn.size(), cStateIn.size() };
        handler->push_data(inputTensors, numInputSamples);
    }
    else
    {
        size_t available = handler->get_available_samples(0, 0);
        if (available < numSamples)
            inferenceUnderruns++;

        float* channels[] = { channelData };
        handler->process(channels, numSamples);
    }
#else
    (void)in; (void)out; (void)n;
#endif
}

void AniraOnnxBackend::reset() noexcept
{
#if HAS_ANIRA
    if (handler)
        handler->reset();

    if (hasExplicitStateLstm)
    {
        std::fill(audioInputScratch.begin(), audioInputScratch.end(), 0.0f);
        std::fill(hStateIn.begin(), hStateIn.end(), 0.0f);
        std::fill(cStateIn.begin(), cStateIn.end(), 0.0f);
        std::fill(hStateOut.begin(), hStateOut.end(), 0.0f);
        std::fill(cStateOut.begin(), cStateOut.end(), 0.0f);

        if (processor)
        {
            for (size_t i = 0; i < hStateIn.size(); ++i)
            {
                processor->set_input(0.0f, 1, i);
                processor->set_input(0.0f, 2, i);
                processor->set_output(0.0f, 1, i);
                processor->set_output(0.0f, 2, i);
            }
        }
    }
#endif
}

void AniraOnnxBackend::teardown()
{
#if HAS_ANIRA
    handler.reset();
    processor.reset();
    config.reset();
    hasExplicitStateLstm = false;
    hiddenSize = 0;
    audioInputScratch.clear();
    hStateIn.clear();
    cStateIn.clear();
    hStateOut.clear();
    cStateOut.clear();
#endif
}
