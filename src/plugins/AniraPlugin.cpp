// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "AniraPlugin.h"

#if HAS_LIBTORCH
namespace {

void zeroTorchScriptState(torch::jit::script::Module& model)
{
    torch::NoGradGuard no_grad;
    for (const auto& buffer : model.named_buffers(/*recurse=*/true))
        buffer.value.zero_();
}

} // namespace
#endif

// ---------------------------------------------------------------------------
// LibTorchEngine — buffer-at-a-time processing
// ---------------------------------------------------------------------------

bool LibTorchEngine::initialize(const std::string& torchscriptPath)
{
#if HAS_LIBTORCH
    try
    {
        // Single-threaded inference for fair comparison with other backends
        at::set_num_threads(1);
        static bool interopSet = false;
        if (!interopSet) { at::set_num_interop_threads(1); interopSet = true; }

        modelPath = torchscriptPath;
        model = torch::jit::load(torchscriptPath);
        model.eval();
        torch::NoGradGuard no_grad;

        // Warmup with a buffer (primes JIT, allocators)
        auto dummy = torch::randn({1, 1, 128});
        model.forward({dummy});

        // Reset traced/exported state buffers without discarding the warmed
        // module. This preserves JIT/allocator warmup for isolated Mode B.
        zeroTorchScriptState(model);

        valid = true;
        fprintf(stderr, "LibTorchEngine: Loaded %s (buffer-at-a-time)\n",
                torchscriptPath.c_str());
        return true;
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "LibTorchEngine: Failed to load %s: %s\n",
                torchscriptPath.c_str(), e.what());
        return false;
    }
#else
    fprintf(stderr, "LibTorchEngine: LibTorch not available\n");
    return false;
#endif
}

void LibTorchEngine::processBlock(const float* input, float* output, int numSamples)
{
#if HAS_LIBTORCH
    if (!valid) {
        for (int i = 0; i < numSamples; i++) output[i] = input[i];
        return;
    }
    torch::NoGradGuard no_grad;

    // Process in chunks to handle Mode A throughput (48k+ samples).
    // Audio callbacks are always <= 1024.
    static constexpr int kMaxChunk = 2048;
    int offset = 0;
    while (offset < numSamples)
    {
        int chunk = std::min(numSamples - offset, kMaxChunk);

        auto x = torch::from_blob(const_cast<float*>(input + offset),
                                   {1, 1, chunk},
                                   torch::kFloat32).clone();

        auto y = model.forward({x}).toTensor();

        auto y_accessor = y.accessor<float, 3>();
        for (int i = 0; i < chunk; i++)
            output[offset + i] = y_accessor[0][0][i];

        offset += chunk;
    }
#else
    for (int i = 0; i < numSamples; i++) output[i] = input[i];
#endif
}

void LibTorchEngine::resetState()
{
#if HAS_LIBTORCH
    if (valid && !modelPath.empty())
    {
        try
        {
            zeroTorchScriptState(model);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "LibTorchEngine::resetState failed: %s\n", e.what());
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// OnnxRuntimeEngine — buffer-at-a-time processing
// ---------------------------------------------------------------------------

bool OnnxRuntimeEngine::initialize(const std::string& onnxPath)
{
#if HAS_ONNXRUNTIME
    try
    {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "bench");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);

        session = std::make_unique<Ort::Session>(*env, onnxPath.c_str(), opts);
        memInfo = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        // Discover input/output names
        Ort::AllocatorWithDefaultOptions alloc;
        size_t numInputs = session->GetInputCount();
        size_t numOutputs = session->GetOutputCount();

        inputNames.clear();
        outputNames.clear();
        for (size_t i = 0; i < numInputs; i++)
        {
            auto name = session->GetInputNameAllocated(i, alloc);
            inputNames.push_back(name.get());
        }
        for (size_t i = 0; i < numOutputs; i++)
        {
            auto name = session->GetOutputNameAllocated(i, alloc);
            outputNames.push_back(name.get());
        }

        // Detect LSTM explicit state pattern: inputs include h_in and c_in
        hasExplicitState = false;
        for (auto& name : inputNames)
        {
            if (name == "h_in")
            {
                hasExplicitState = true;
                break;
            }
        }

        // Pre-allocate buffers
        xBuf.resize(maxBufferSize, 0.0f);
        yBuf.resize(maxBufferSize, 0.0f);

        if (hasExplicitState)
        {
            // Query h_in shape to determine hidden size
            for (size_t i = 0; i < numInputs; i++)
            {
                if (inputNames[i] == "h_in")
                {
                    auto typeInfo = session->GetInputTypeInfo(i);
                    auto shape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
                    // h_in shape: (1, 1, hidden_size)
                    hiddenSize = static_cast<int>(shape.back());
                    break;
                }
            }
            hBuf.resize(hiddenSize, 0.0f);
            cBuf.resize(hiddenSize, 0.0f);

            // Prime the session to avoid first-call allocations
            std::vector<int64_t> xShape = {1, 1, 1};
            std::vector<int64_t> hShape = {1, 1, hiddenSize};
            float dummy = 0.0f;

            std::vector<Ort::Value> inTensors;
            inTensors.push_back(Ort::Value::CreateTensor<float>(*memInfo, &dummy, 1, xShape.data(), 3));
            inTensors.push_back(Ort::Value::CreateTensor<float>(*memInfo, hBuf.data(), hBuf.size(), hShape.data(), 3));
            inTensors.push_back(Ort::Value::CreateTensor<float>(*memInfo, cBuf.data(), cBuf.size(), hShape.data(), 3));

            std::vector<Ort::Value> outTensors;
            outTensors.push_back(Ort::Value::CreateTensor<float>(*memInfo, &dummy, 1, xShape.data(), 3));
            outTensors.push_back(Ort::Value::CreateTensor<float>(*memInfo, hBuf.data(), hBuf.size(), hShape.data(), 3));
            outTensors.push_back(Ort::Value::CreateTensor<float>(*memInfo, cBuf.data(), cBuf.size(), hShape.data(), 3));

            std::vector<const char*> inNames, outNames;
            for (auto& n : inputNames) inNames.push_back(n.c_str());
            for (auto& n : outputNames) outNames.push_back(n.c_str());

            session->Run(Ort::RunOptions{nullptr},
                         inNames.data(), inTensors.data(), inTensors.size(),
                         outNames.data(), outTensors.data(), outTensors.size());

            // Reset state after priming
            std::fill(hBuf.begin(), hBuf.end(), 0.0f);
            std::fill(cBuf.begin(), cBuf.end(), 0.0f);

            fprintf(stderr, "OnnxRuntimeEngine: LSTM explicit state (hidden=%d) %s\n",
                    hiddenSize, onnxPath.c_str());
        }
        else
        {
            fprintf(stderr, "OnnxRuntimeEngine: Conv model (internal state) %s\n",
                    onnxPath.c_str());
        }

        valid = true;
        return true;
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "OnnxRuntimeEngine: Failed to load %s: %s\n",
                onnxPath.c_str(), e.what());
        return false;
    }
#else
    fprintf(stderr, "OnnxRuntimeEngine: ONNX Runtime not available\n");
    return false;
#endif
}

void OnnxRuntimeEngine::processBlock(const float* input, float* output, int numSamples)
{
#if HAS_ONNXRUNTIME
    if (!valid) {
        for (int i = 0; i < numSamples; i++) output[i] = input[i];
        return;
    }

    // Process in chunks to handle Mode A throughput (48k+ samples).
    int offset = 0;
    while (offset < numSamples)
    {
        int chunk = std::min(numSamples - offset, maxBufferSize);
        processChunk(input + offset, output + offset, chunk);
        offset += chunk;
    }
}

void OnnxRuntimeEngine::processChunk(const float* input, float* output, int numSamples)
{
    // Copy input to pre-allocated buffer
    memcpy(xBuf.data(), input, numSamples * sizeof(float));

    // Following iPlug2OnnxRuntime pattern: always use max-buffer-size tensors.
    // Actual audio is memcpy'd in/out. Pre-allocated tensors avoid per-call alloc.
    // h/c output tensors share memory with input tensors (zero-copy state carry).

    std::vector<const char*> inNamesC, outNamesC;
    for (auto& n : inputNames) inNamesC.push_back(n.c_str());
    for (auto& n : outputNames) outNamesC.push_back(n.c_str());

    if (hasExplicitState)
    {
        // LSTM: (x, h_in, c_in) -> (y, h_out, c_out)
        std::vector<int64_t> xShape = {1, 1, static_cast<int64_t>(numSamples)};
        std::vector<int64_t> hShape = {1, 1, static_cast<int64_t>(hiddenSize)};

        // Build input tensors — wrapping pre-allocated buffers
        std::vector<Ort::Value> inTensors;
        inTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, xBuf.data(), numSamples, xShape.data(), 3));
        inTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, hBuf.data(), hBuf.size(), hShape.data(), 3));
        inTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, cBuf.data(), cBuf.size(), hShape.data(), 3));

        // Output tensors: y uses yBuf, h/c output SHARE memory with input
        std::vector<Ort::Value> outTensors;
        outTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, yBuf.data(), numSamples, xShape.data(), 3));
        outTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, hBuf.data(), hBuf.size(), hShape.data(), 3));
        outTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, cBuf.data(), cBuf.size(), hShape.data(), 3));

        session->Run(Ort::RunOptions{nullptr},
                     inNamesC.data(), inTensors.data(), inTensors.size(),
                     outNamesC.data(), outTensors.data(), outTensors.size());
    }
    else
    {
        // TCN/WaveNet: (x) -> (y), state managed internally
        std::vector<int64_t> xShape = {1, 1, static_cast<int64_t>(numSamples)};

        std::vector<Ort::Value> inTensors;
        inTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, xBuf.data(), numSamples, xShape.data(), 3));

        std::vector<Ort::Value> outTensors;
        outTensors.push_back(Ort::Value::CreateTensor<float>(
            *memInfo, yBuf.data(), numSamples, xShape.data(), 3));

        session->Run(Ort::RunOptions{nullptr},
                     inNamesC.data(), inTensors.data(), inTensors.size(),
                     outNamesC.data(), outTensors.data(), outTensors.size());
    }

    // Copy output
    memcpy(output, yBuf.data(), numSamples * sizeof(float));
#else
    for (int i = 0; i < numSamples; i++) output[i] = input[i];
#endif
}

void OnnxRuntimeEngine::resetState()
{
#if HAS_ONNXRUNTIME
    if (hasExplicitState)
    {
        std::fill(hBuf.begin(), hBuf.end(), 0.0f);
        std::fill(cBuf.begin(), cBuf.end(), 0.0f);
    }
    // For TCN/WaveNet: ONNX state is internal to the model graph.
    // There's no way to reset it without reloading the session.
    // This is a known ONNX limitation for stateful conv models.
#endif
}

// ---------------------------------------------------------------------------
// DirectLibTorchPlugin
// ---------------------------------------------------------------------------

namespace tracktion { namespace engine {

const char* DirectLibTorchPlugin::xmlTypeName = "directLibTorchPlugin";

DirectLibTorchPlugin::DirectLibTorchPlugin(PluginCreationInfo info) : Plugin(info) {}
DirectLibTorchPlugin::~DirectLibTorchPlugin() { deinitialise(); }

void DirectLibTorchPlugin::initialise(const PluginInitialisationInfo& info)
{
    if (!modelPath.empty())
        engine.initialize(modelPath);
    timingLogger.allocate(static_cast<int>(SAMPLE_RATE * 30.0 / 32.0));
}

void DirectLibTorchPlugin::deinitialise() {}

void DirectLibTorchPlugin::applyToBuffer(const PluginRenderContext& ctx)
{
    if (ctx.destBuffer == nullptr || !engine.isValid())
        return;

    timingLogger.recordStart();
    auto* data = ctx.destBuffer->getWritePointer(0);
    engine.processBlock(data + ctx.bufferStartSample,
                        data + ctx.bufferStartSample,
                        ctx.bufferNumSamples);
    timingLogger.recordEnd();
}

void DirectLibTorchPlugin::reset() { engine.resetState(); }

// ---------------------------------------------------------------------------
// DirectOnnxPlugin
// ---------------------------------------------------------------------------

const char* DirectOnnxPlugin::xmlTypeName = "directOnnxPlugin";

DirectOnnxPlugin::DirectOnnxPlugin(PluginCreationInfo info) : Plugin(info) {}
DirectOnnxPlugin::~DirectOnnxPlugin() { deinitialise(); }

void DirectOnnxPlugin::initialise(const PluginInitialisationInfo& info)
{
    if (!modelPath.empty())
        engine.initialize(modelPath);
    timingLogger.allocate(static_cast<int>(SAMPLE_RATE * 30.0 / 32.0));
}

void DirectOnnxPlugin::deinitialise() {}

void DirectOnnxPlugin::applyToBuffer(const PluginRenderContext& ctx)
{
    if (ctx.destBuffer == nullptr || !engine.isValid())
        return;

    timingLogger.recordStart();
    auto* data = ctx.destBuffer->getWritePointer(0);
    engine.processBlock(data + ctx.bufferStartSample,
                        data + ctx.bufferStartSample,
                        ctx.bufferNumSamples);
    timingLogger.recordEnd();
}

void DirectOnnxPlugin::reset() { engine.resetState(); }

}} // namespace tracktion::engine
