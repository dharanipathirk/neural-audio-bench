// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../BenchmarkConfig.h"
#include "../TimingLogger.h"
#include "../plugins/BNNSGraphPlugin.h"
#include "../plugins/RTNeuralPlugin.h"
#include "../plugins/AniraPlugin.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Isolated benchmark: measures raw inference speed without DAW overhead.
// Mode A: raw throughput (process N seconds of audio, measure wall-clock)
// Mode B: simulated DAW callbacks (per-buffer timing, stats)
// ---------------------------------------------------------------------------
class IsolatedBenchmark
{
public:
    IsolatedBenchmark(const std::string& modelDir, const std::string& configPath)
        : modelDir(modelDir), configPath(configPath) {}

    void runAll(FILE* csvFile);

private:
    std::string modelDir;
    std::string configPath;

    // Generate deterministic random signal
    std::vector<float> generateSignal(size_t numSamples, uint32_t seed = 42);

    // Mode A: measure throughput (x realtime)
    void runModeA(FILE* csvFile, const char* backend, ModelType model, ModelSize size,
                  std::function<void(const float*, float*, int)> processBlock,
                  const BenchmarkRuntimeConfig& cfg);

    // Mode B: per-buffer latency at each buffer size
    void runModeB(FILE* csvFile, const char* backend, ModelType model, ModelSize size,
                  std::function<void(const float*, float*, int)> processBlock,
                  std::function<void()> resetFn,
                  const BenchmarkRuntimeConfig& cfg);

    // Run all backends for a given model+size combination
    void benchmarkModel(FILE* csvFile, ModelType model, ModelSize size,
                        const BenchmarkRuntimeConfig& cfg);
};
