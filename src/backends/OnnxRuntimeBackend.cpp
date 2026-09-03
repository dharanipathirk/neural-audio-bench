// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "OnnxRuntimeBackend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

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
            fprintf(stderr, "OnnxRuntimeEngine: Stateless convolution model %s\n",
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
        // TCN/WaveNet: stateless (x) -> (y)
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
    // TCN/WaveNet ONNX exports are stateless, so there is nothing to reset.
#endif
}

// ---------------------------------------------------------------------------
// OnnxRuntimeBackend adapter
// ---------------------------------------------------------------------------

bool OnnxRuntimeBackend::prepare(const PrepareContext& ctx)
{
    if (ctx.model == nullptr)
        return false;
    auto it = ctx.model->formatPaths.find("onnx");
    if (it == ctx.model->formatPaths.end())
        return false;
    return engine.initialize(it->second);
}
