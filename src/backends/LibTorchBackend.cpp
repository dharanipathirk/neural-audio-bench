// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "LibTorchBackend.h"

#include <algorithm>
#include <cstdio>

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
// LibTorchBackend adapter
// ---------------------------------------------------------------------------

bool LibTorchBackend::prepare(const PrepareContext& ctx)
{
    if (ctx.model == nullptr)
        return false;
    auto it = ctx.model->formatPaths.find("torchscript");
    if (it == ctx.model->formatPaths.end())
        return false;
    return engine.initialize(it->second);
}
