// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "BNNSGraphPlugin.h"
#include "../BenchmarkConfig.h"
#import <Foundation/Foundation.h>

// ---------------------------------------------------------------------------
// BNNSGraphEngine implementation — buffer-at-a-time via dynamic shapes
// ---------------------------------------------------------------------------

static size_t roundUpToPage(size_t bytes)
{
    return ((bytes + 4095) / 4096) * 4096;
}

bool BNNSGraphEngine::initialize(const std::string& mlmodelcPath)
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
    BNNSGraphContextSetArgumentType(ctx, BNNSGraphArgumentTypePointer);

    // Discover arguments
    argCount = BNNSGraphGetArgumentCount(graph, nullptr);
    buffers.resize(argCount, nullptr);
    bufferSizes.resize(argCount, 0);
    baseBytesPerElem.resize(argCount, 0);
    intents.resize(argCount);
    args.resize(argCount);
    argShapes.resize(argCount);
    argNames.resize(argCount);

    BNNSGraphGetArgumentNames(graph, nullptr, argCount, argNames.data());
    BNNSGraphGetArgumentIntents(graph, nullptr, argCount, intents.data());

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
        BNNSGraphContextGetTensor(ctx, nullptr, argNames[i], true, &tensor);

        auto& as = argShapes[i];
        as.rank = tensor.rank;
        as.seqDim = -1;

        size_t elemBytes;
        switch (tensor.data_type)
        {
            case BNNSDataTypeFloat16: elemBytes = sizeof(_Float16); break;
            case BNNSDataTypeFloat32: elemBytes = sizeof(float); break;
            default: elemBytes = sizeof(_Float16); break;
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
        memset(buffers[i], 0, pageAligned);
        bufferSizes[i] = pageAligned;
    }

    // Allocate workspace
    wsSize = BNNSGraphContextGetWorkspaceSize(ctx, nullptr) + 4096;
    workspace = (char*)aligned_alloc(4096, wsSize);

    currentSeqLen = 0;  // will be set on first processBlock call

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

    if (workspace) { free(workspace); workspace = nullptr; }
    if (ctx.data)  { BNNSGraphContextDestroy(ctx); ctx = {}; }
    if (graph.data) { free(graph.data); graph = {}; }

    currentSeqLen = 0;
}

void BNNSGraphEngine::setDynamicShape(int seqLen)
{
    if (seqLen == currentSeqLen)
        return;

    // Build bnns_graph_shape_t array — one entry per argument.
    // For static arguments, shape is 0-filled (rank=0) which tells BNNS to
    // use the compiled default.  For dynamic arguments (x, y), we set the
    // actual seq_len.
    std::vector<bnns_graph_shape_t> shapes(argCount);
    std::vector<std::vector<uint64_t>> shapeData(argCount);

    for (size_t i = 0; i < argCount; i++)
    {
        auto& as = argShapes[i];
        if (as.seqDim >= 0)
        {
            shapeData[i].resize(as.rank);
            for (size_t d = 0; d < as.rank; d++)
                shapeData[i][d] = (d == static_cast<size_t>(as.seqDim))
                    ? static_cast<uint64_t>(seqLen)
                    : static_cast<uint64_t>(as.shape[d]);
            shapes[i].rank = as.rank;
            shapes[i].shape = shapeData[i].data();
        }
        else
        {
            // Static shape — rank=0 tells BNNS to keep compiled shape
            shapes[i].rank = 0;
            shapes[i].shape = nullptr;
        }
    }

    BNNSGraphContextSetDynamicShapes(ctx, nullptr, argCount, shapes.data());
    currentSeqLen = seqLen;
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
        setDynamicShape(chunk);

        // Convert float input -> fp16 input buffer
        _Float16* xBuf = (_Float16*)buffers[xIdx];
        for (int i = 0; i < chunk; i++)
            xBuf[i] = (_Float16)input[offset + i];

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
        BNNSGraphContextExecute(ctx, nullptr, argCount, args.data(), wsSize, workspace);

        // Convert fp16 output -> float
        _Float16* yBuf = (_Float16*)buffers[yIdx];
        for (int i = 0; i < chunk; i++)
            output[offset + i] = (float)yBuf[i];

        offset += chunk;
    }
}

void BNNSGraphEngine::resetState()
{
    for (size_t i = 0; i < argCount; i++)
        if (intents[i] == BNNSGraphArgumentIntentInOut)
            memset(buffers[i], 0, bufferSizes[i]);
    currentSeqLen = 0;
}

// ---------------------------------------------------------------------------
// BNNSGraphPlugin (Tracktion Engine plugin)
// ---------------------------------------------------------------------------

namespace tracktion { namespace engine {

const char* BNNSGraphPlugin::xmlTypeName = "bnnsGraphPlugin";

BNNSGraphPlugin::BNNSGraphPlugin(PluginCreationInfo info)
    : Plugin(info)
{
}

BNNSGraphPlugin::~BNNSGraphPlugin()
{
    deinitialise();
}

void BNNSGraphPlugin::initialise(const PluginInitialisationInfo& info)
{
    fprintf(stderr, "    BNNSGraphPlugin::initialise called! sr=%.0f bs=%d model=%s\n",
            info.sampleRate, info.blockSizeSamples, modelPath.c_str());

    if (!modelPath.empty() && !engine.isValid())
        engine.initialize(modelPath);

    timingLogger.allocate(static_cast<int>(SAMPLE_RATE * 30.0 / 32.0));
}

void BNNSGraphPlugin::deinitialise()
{
    engine.deinitialize();
}

void BNNSGraphPlugin::applyToBuffer(const PluginRenderContext& ctx)
{
    if (ctx.destBuffer == nullptr || !engine.isValid())
        return;

    auto* buffer = ctx.destBuffer;
    const int numSamples = ctx.bufferNumSamples;
    const int startSample = ctx.bufferStartSample;

    timingLogger.recordStart();

    // In-place: read from buffer, process, write back
    // Need a temp output since processBlock writes to a separate output buffer
    auto* data = buffer->getWritePointer(0);
    engine.processBlock(data + startSample, data + startSample, numSamples);

    timingLogger.recordEnd();
}

void BNNSGraphPlugin::reset()
{
    engine.resetState();
}

}} // namespace tracktion::engine
