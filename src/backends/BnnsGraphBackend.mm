// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "BnnsGraphBackend.h"
#import <Foundation/Foundation.h>
#include <algorithm>

// ---------------------------------------------------------------------------
// BNNSGraphEngine implementation — buffer-at-a-time via dynamic shapes
// ---------------------------------------------------------------------------

static size_t roundUpToPage(size_t bytes)
{
    return ((bytes + 4095) / 4096) * 4096;
}

bool BNNSGraphEngine::initialize(const std::string& mlmodelcPath, int preparedBlockSize)
{
    deinitialize();

    // Compile graph (single-threaded for audio)
    auto options = BNNSGraphCompileOptionsMakeDefault();
    BNNSGraphCompileOptionsSetTargetSingleThread(options, true);
    graph = BNNSGraphCompileFromFile(mlmodelcPath.c_str(), nullptr, options);
    BNNSGraphCompileOptionsDestroy(options);

    if (graph.data == nullptr)
    {
        fprintf(stderr, "BNNSGraphEngine: Failed to compile %s\n", mlmodelcPath.c_str());
        return false;
    }

    // Create context
    ctx = BNNSGraphContextMake(graph);
    if (ctx.data == nullptr
        || BNNSGraphContextSetArgumentType(ctx, BNNSGraphArgumentTypePointer) != 0)
    {
        fprintf(stderr, "BNNSGraphEngine: failed to create graph context\n");
        deinitialize();
        return false;
    }

    // Discover arguments
    argCount = BNNSGraphGetArgumentCount(graph, nullptr);
    if (argCount == 0 || argCount == SIZE_MAX)
    {
        fprintf(stderr, "BNNSGraphEngine: failed to discover graph arguments\n");
        deinitialize();
        return false;
    }
    buffers.resize(argCount, nullptr);
    bufferSizes.resize(argCount, 0);
    baseBytesPerElem.resize(argCount, 0);
    intents.resize(argCount);
    args.resize(argCount);
    argShapes.resize(argCount);
    argNames.resize(argCount);
    dynamicShapes.resize(argCount);
    dynamicShapeData.resize(argCount);

    if (BNNSGraphGetArgumentNames(graph, nullptr, argCount, argNames.data()) != 0
        || BNNSGraphGetArgumentIntents(graph, nullptr, argCount, intents.data()) != 0)
    {
        fprintf(stderr, "BNNSGraphEngine: failed to query graph arguments\n");
        deinitialize();
        return false;
    }

    // Find input/output positions by name
    xIdx = BNNSGraphGetArgumentPosition(graph, nullptr, "x");
    yIdx = BNNSGraphGetArgumentPosition(graph, nullptr, "y");

    if (xIdx >= argCount || yIdx >= argCount)
    {
        fprintf(stderr, "BNNSGraphEngine: Could not find 'x' (idx=%zu) or 'y' (idx=%zu)\n",
                xIdx, yIdx);
        deinitialize();
        return false;
    }

    // Query tensor shapes and identify the dynamic seq_len dimension
    for (size_t i = 0; i < argCount; i++)
    {
        BNNSTensor tensor{};
        if (BNNSGraphContextGetTensor(ctx, nullptr, argNames[i], true, &tensor) != 0
            || tensor.rank == 0 || tensor.rank > 8)
        {
            fprintf(stderr, "BNNSGraphEngine: invalid tensor metadata for argument '%s'\n",
                    argNames[i]);
            deinitialize();
            return false;
        }

        auto& as = argShapes[i];
        as.rank = tensor.rank;
        as.seqDim = -1;

        size_t elemBytes = 0;
        switch (tensor.data_type)
        {
            case BNNSDataTypeFloat16: elemBytes = sizeof(_Float16); break;
            case BNNSDataTypeFloat32: elemBytes = sizeof(float); break;
            default:
                fprintf(stderr, "BNNSGraphEngine: unsupported data type for argument '%s'\n",
                        argNames[i]);
                deinitialize();
                return false;
        }

        size_t staticBytes = elemBytes;
        for (size_t d = 0; d < tensor.rank; d++)
        {
            ssize_t dim = tensor.shape[d];
            as.shape[d] = (dim > 0) ? dim : 1;

            // The dynamic dimension (from ct.RangeDim) shows as the default value.
            // For x and y, the last dimension is the sequence length.
            // For state buffers (InOut), all dimensions are static.
            staticBytes *= as.shape[d];
        }
        baseBytesPerElem[i] = elemBytes;  // per-element size for reallocation

        // For x and y, the last dimension is the seq_len (dynamic)
        if (i == xIdx || i == yIdx)
            as.seqDim = static_cast<int>(tensor.rank) - 1;

        // Allocate at max buffer size for x/y, at static size for state
        size_t allocBytes;
        if (as.seqDim >= 0)
        {
            // Dynamic: allocate for max buffer
            allocBytes = elemBytes;
            for (size_t d = 0; d < tensor.rank; d++)
                allocBytes *= (d == static_cast<size_t>(as.seqDim))
                    ? kMaxBufferSize : as.shape[d];
        }
        else
        {
            allocBytes = staticBytes;
        }

        size_t pageAligned = roundUpToPage(allocBytes);
        buffers[i] = aligned_alloc(4096, pageAligned);
        if (buffers[i] == nullptr)
        {
            fprintf(stderr, "BNNSGraphEngine: failed to allocate argument '%s'\n", argNames[i]);
            deinitialize();
            return false;
        }
        memset(buffers[i], 0, pageAligned);
        bufferSizes[i] = pageAligned;

    }

    // Configure the host's normal block size and allocate the exact workspace
    // required for that dynamic shape before any audio callback runs. Further
    // configured shapes are established during the benchmark's untimed warmup.
    currentSeqLen = 0;
    const int initialSeqLen = std::clamp(preparedBlockSize, 1, kMaxBufferSize);
    if (!setDynamicShape(initialSeqLen))
    {
        deinitialize();
        return false;
    }

    fprintf(stderr, "BNNSGraphEngine: Loaded %s (%zu args, x=%zu, y=%zu, buffer-at-a-time)\n",
            mlmodelcPath.c_str(), argCount, xIdx, yIdx);

    return true;
}

void BNNSGraphEngine::deinitialize()
{
    for (auto& buf : buffers)
    {
        if (buf) free(buf);
        buf = nullptr;
    }
    buffers.clear();
    bufferSizes.clear();
    baseBytesPerElem.clear();
    intents.clear();
    args.clear();
    argShapes.clear();
    argNames.clear();
    dynamicShapes.clear();
    dynamicShapeData.clear();

    if (workspace) { free(workspace); workspace = nullptr; }
    wsSize = 0;
    if (ctx.data)  { BNNSGraphContextDestroy(ctx); ctx = {}; }
    if (graph.data) { free(graph.data); graph = {}; }

    currentSeqLen = 0;
}

bool BNNSGraphEngine::ensureWorkspace(size_t requiredBytes)
{
    if (requiredBytes <= wsSize && workspace != nullptr)
        return true;

    const size_t allocationSize = roundUpToPage(std::max<size_t>(requiredBytes, 1));
    auto* replacement = static_cast<char*>(aligned_alloc(4096, allocationSize));
    if (replacement == nullptr)
    {
        fprintf(stderr, "BNNSGraphEngine: failed to allocate %zu-byte workspace\n",
                allocationSize);
        return false;
    }

    free(workspace);
    workspace = replacement;
    wsSize = allocationSize;
    return true;
}

bool BNNSGraphEngine::setDynamicShape(int seqLen)
{
    if (seqLen == currentSeqLen)
        return true;
    if (seqLen <= 0 || seqLen > kMaxBufferSize)
        return false;

    // Reuse storage allocated by initialize(): changing shape during an audio
    // callback must not allocate merely to construct the BNNS descriptors.

    for (size_t i = 0; i < argCount; i++)
    {
        auto& as = argShapes[i];
        if (as.seqDim >= 0)
        {
            for (size_t d = 0; d < as.rank; d++)
                dynamicShapeData[i][d] = (d == static_cast<size_t>(as.seqDim))
                    ? static_cast<uint64_t>(seqLen)
                    : static_cast<uint64_t>(as.shape[d]);
            dynamicShapes[i].rank = as.rank;
            dynamicShapes[i].shape = dynamicShapeData[i].data();
        }
        else
        {
            dynamicShapes[i].rank = 0;
            dynamicShapes[i].shape = nullptr;
        }
    }

    const int shapeResult = BNNSGraphContextSetDynamicShapes(
        ctx, nullptr, argCount, dynamicShapes.data());
    if (shapeResult < 0)
    {
        fprintf(stderr, "BNNSGraphEngine: failed to set dynamic shape %d (status %d)\n",
                seqLen, shapeResult);
        return false;
    }

    const size_t requiredWorkspace = BNNSGraphContextGetWorkspaceSize(ctx, nullptr);
    if (!ensureWorkspace(requiredWorkspace))
        return false;

    currentSeqLen = seqLen;
    return true;
}

void BNNSGraphEngine::processBlock(const float* input, float* output, int numSamples)
{
    if (!graph.data || numSamples <= 0)
        return;

    // Process in chunks of kMaxBufferSize if needed (Mode A throughput may
    // pass 48,000+ samples; audio callbacks are always <= 1024).
    int offset = 0;
    while (offset < numSamples)
    {
        int chunk = std::min(numSamples - offset, kMaxBufferSize);

        // Set dynamic shapes if chunk size changed
        if (!setDynamicShape(chunk))
        {
            std::fill_n(output + offset, chunk, 0.0f);
            offset += chunk;
            continue;
        }

        if (baseBytesPerElem[xIdx] == sizeof(float))
        {
            memcpy(buffers[xIdx], input + offset, static_cast<size_t>(chunk) * sizeof(float));
        }
        else
        {
            auto* xBuf = static_cast<_Float16*>(buffers[xIdx]);
            for (int i = 0; i < chunk; i++)
                xBuf[i] = static_cast<_Float16>(input[offset + i]);
        }

        // Build argument list — compute actual byte sizes for this chunk
        for (size_t i = 0; i < argCount; i++)
        {
            auto& as = argShapes[i];
            size_t bytes = baseBytesPerElem[i];
            for (size_t d = 0; d < as.rank; d++)
                bytes *= (d == static_cast<size_t>(as.seqDim) && as.seqDim >= 0)
                    ? static_cast<size_t>(chunk) : as.shape[d];
            args[i] = {.data_ptr = buffers[i], .data_ptr_size = bytes};
        }

        // Single Execute for this chunk
        const int executeResult = BNNSGraphContextExecute(
            ctx, nullptr, argCount, args.data(), wsSize, workspace);
        if (executeResult != 0)
        {
            fprintf(stderr, "BNNSGraphEngine: execution failed (status %d)\n", executeResult);
            std::fill_n(output + offset, chunk, 0.0f);
            offset += chunk;
            continue;
        }

        if (baseBytesPerElem[yIdx] == sizeof(float))
        {
            memcpy(output + offset, buffers[yIdx], static_cast<size_t>(chunk) * sizeof(float));
        }
        else
        {
            auto* yBuf = static_cast<_Float16*>(buffers[yIdx]);
            for (int i = 0; i < chunk; i++)
                output[offset + i] = static_cast<float>(yBuf[i]);
        }

        offset += chunk;
    }
}

void BNNSGraphEngine::resetState()
{
    for (size_t i = 0; i < argCount; i++)
        if (intents[i] == BNNSGraphArgumentIntentInOut)
            memset(buffers[i], 0, bufferSizes[i]);
}

// ---------------------------------------------------------------------------
// BnnsGraphBackend adapter
// ---------------------------------------------------------------------------

bool BnnsGraphBackend::prepare(const PrepareContext& ctx)
{
    if (ctx.model == nullptr)
        return false;
    auto it = ctx.model->formatPaths.find("coreml");
    if (it == ctx.model->formatPaths.end())
        return false;
    return engine.initialize(it->second, ctx.maxBlockSize);
}
