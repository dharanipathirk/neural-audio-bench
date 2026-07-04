// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "IsolatedBenchmark.h"

std::vector<float> IsolatedBenchmark::generateSignal(size_t numSamples, uint32_t seed)
{
    std::vector<float> signal(numSamples);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& s : signal)
        s = dist(gen);
    return signal;
}

void IsolatedBenchmark::runModeA(FILE* csvFile, const char* backend, ModelType model, ModelSize size,
                                  std::function<void(const float*, float*, int)> processBlock,
                                  const BenchmarkRuntimeConfig& cfg)
{
    double throughputSeconds = cfg.isolatedThroughputSeconds;
    const size_t totalSamples = static_cast<size_t>(cfg.sampleRate * throughputSeconds);
    auto signal = generateSignal(totalSamples);
    std::vector<float> output(totalSamples);

    // Warmup
    processBlock(signal.data(), output.data(), static_cast<int>(totalSamples));

    // Measure
    TimingUtils::init();
    auto t0 = TimingUtils::now();
    processBlock(signal.data(), output.data(), static_cast<int>(totalSamples));
    auto t1 = TimingUtils::now();

    double elapsed_ns = TimingUtils::toNanoseconds(t1 - t0);
    double elapsed_s = elapsed_ns / 1e9;
    double xRT = throughputSeconds / elapsed_s;

    fprintf(stderr, "  Mode A [%s/%s/%s]: %.1fx real-time (%.4fs for %.1fs audio)\n",
            backend, modelTypeName(model), modelSizeName(size), xRT, elapsed_s, throughputSeconds);

    // Build a TimingStats for the throughput row (single measurement)
    double perSample = elapsed_ns / static_cast<double>(totalSamples);
    TimingStats ts{};
    ts.median_ns = perSample;
    ts.mean_ns = perSample;
    ts.rtf = 1.0 / xRT;
    ts.total_samples = static_cast<int>(totalSamples);
    CSVOutput::printIsolatedRow(csvFile, "throughput", backend,
                                 modelTypeName(model), modelSizeName(size),
                                 0, 1, ts);
}

void IsolatedBenchmark::runModeB(FILE* csvFile, const char* backend, ModelType model, ModelSize size,
                                  std::function<void(const float*, float*, int)> processBlock,
                                  std::function<void()> resetFn,
                                  const BenchmarkRuntimeConfig& cfg)
{
    TimingUtils::init();

    for (size_t si = 0; si < cfg.isolatedBufferSizes.size(); si++)
    {
        int bufSize = cfg.isolatedBufferSizes[si];
        double deadline_ns = TimingUtils::bufferDurationNs(bufSize, cfg.sampleRate);

        // Pre-generate a long deterministic signal. Each iteration consumes the
        // next segment, so the model sees varied input (not the same block 500K
        // times). This avoids artificial cache-hot behaviour on the input path
        // while keeping runs fully reproducible.
        // Size: enough for warmup + probe + max iterations, or at least 10s of audio.
        size_t streamSamples = std::max(
            static_cast<size_t>(cfg.sampleRate * 10),  // 10s minimum
            static_cast<size_t>(bufSize) * 1100000      // headroom for 1M iterations
        );
        auto stream = generateSignal(streamSamples);
        std::vector<float> output(static_cast<size_t>(bufSize));

        // Adaptive warmup — consumes first portion of stream
        size_t offset = 0;
        for (int w = 0; w < cfg.isolatedWarmupIterations; w++)
        {
            processBlock(stream.data() + offset, output.data(), bufSize);
            offset += static_cast<size_t>(bufSize);
            if (offset + static_cast<size_t>(bufSize) > streamSamples) offset = 0;
        }

        // Time 10 iterations to decide iteration count
        auto probe_t0 = TimingUtils::now();
        for (int w = 0; w < 10; w++)
        {
            processBlock(stream.data() + offset, output.data(), bufSize);
            offset += static_cast<size_t>(bufSize);
            if (offset + static_cast<size_t>(bufSize) > streamSamples) offset = 0;
        }
        auto probe_t1 = TimingUtils::now();
        double per_iter_ns = TimingUtils::toNanoseconds(probe_t1 - probe_t0) / 10.0;

        // Target cfg.isolatedTargetSeconds max per rep, at least cfg.isolatedMinIterations,
        // capped at 1M to prevent excessive memory use for very fast configs.
        int iterations = static_cast<int>(
            std::min(1000000.0,
                std::max(static_cast<double>(cfg.isolatedMinIterations),
                         cfg.isolatedTargetSeconds * 1e9 / per_iter_ns))
        );

        for (int rep = 0; rep < cfg.isolatedReps; rep++)
        {
            resetFn();
            offset = 0;  // reset stream position at each rep for reproducibility
            std::vector<double> times;
            times.reserve(iterations);

            for (int m = 0; m < iterations; m++)
            {
                const float* in = stream.data() + offset;
                auto t0 = TimingUtils::now();
                processBlock(in, output.data(), bufSize);
                auto t1 = TimingUtils::now();
                times.push_back(TimingUtils::toNanoseconds(t1 - t0));

                offset += static_cast<size_t>(bufSize);
                if (offset + static_cast<size_t>(bufSize) > streamSamples) offset = 0;
            }

            auto stats = TimingStats::compute(times, deadline_ns);
            CSVOutput::printIsolatedRow(csvFile, "callback", backend,
                                         modelTypeName(model), modelSizeName(size),
                                         bufSize, rep + 1, stats);
        }
    }
}

void IsolatedBenchmark::benchmarkModel(FILE* csvFile, ModelType model, ModelSize size,
                                        const BenchmarkRuntimeConfig& cfg)
{
    fprintf(stderr, "\n--- %s / %s ---\n", modelTypeName(model), modelSizeName(size));

    // 1. BNNSGraph
    if (cfg.isBackendEnabled(BackendType::BNNSGraph))
    {
        BNNSGraphEngine engine;
        std::string path = modelCoreMLPath(model, size, modelDir);
        if (engine.initialize(path))
        {
            auto proc = [&](const float* in, float* out, int n) { engine.processBlock(in, out, n); };
            auto reset = [&]() { engine.resetState(); };

            runModeA(csvFile, "BNNSGraph", model, size, proc, cfg);
            runModeB(csvFile, "BNNSGraph", model, size, proc, reset, cfg);
        }
    }

    // 2. RTNeural (Eigen or XSIMD, selected at compile time)
    {
#if defined(RTNEURAL_USE_XSIMD)
        const auto rtneuralBackendType = BackendType::RTNeural_XSIMD;
        const char* rtneuralBackend = "RTNeural_XSIMD";
#else
        const auto rtneuralBackendType = BackendType::RTNeural_Eigen;
        const char* rtneuralBackend = "RTNeural_Eigen";
#endif
        if (cfg.isBackendEnabled(rtneuralBackendType))
        {
            RTNeuralEngine engine;
            std::string weights = modelWeightsPath(model, size, modelDir);
            if (engine.initialize(model, size, weights))
            {
                auto proc = [&](const float* in, float* out, int n) { engine.processBlock(in, out, n); };
                auto reset = [&]() { engine.resetState(); };

                runModeA(csvFile, rtneuralBackend, model, size, proc, cfg);
                runModeB(csvFile, rtneuralBackend, model, size, proc, reset, cfg);
            }
        }
    }

    // 3. LibTorch (direct, no anira scheduling)
    if (cfg.isBackendEnabled(BackendType::Direct_LibTorch))
    {
        LibTorchEngine engine;
        std::string path = modelTorchScriptPath(model, size, modelDir);
        if (engine.initialize(path))
        {
            auto proc = [&](const float* in, float* out, int n) { engine.processBlock(in, out, n); };
            auto reset = [&]() { engine.resetState(); };

            runModeA(csvFile, "Direct_LibTorch", model, size, proc, cfg);
            runModeB(csvFile, "Direct_LibTorch", model, size, proc, reset, cfg);
        }
    }

    // 4. ONNX Runtime (direct, no anira scheduling)
    if (cfg.isBackendEnabled(BackendType::Direct_ONNX))
    {
        OnnxRuntimeEngine engine;
        std::string path = modelOnnxPath(model, size, modelDir);
        if (engine.initialize(path))
        {
            auto proc = [&](const float* in, float* out, int n) { engine.processBlock(in, out, n); };
            auto reset = [&]() { engine.resetState(); };

            runModeA(csvFile, "Direct_ONNX", model, size, proc, cfg);
            runModeB(csvFile, "Direct_ONNX", model, size, proc, reset, cfg);
        }
    }
}

void IsolatedBenchmark::runAll(FILE* csvFile)
{
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Isolated Benchmark\n");
    fprintf(stderr, "========================================\n");

    CSVOutput::printIsolatedHeader(csvFile);

    auto cfg = BenchmarkRuntimeConfig::load(configPath);

    for (int si = 0; si < static_cast<int>(ModelSize::COUNT); si++)
    {
        auto size = static_cast<ModelSize>(si);
        if (!cfg.isSizeEnabled(size)) continue;

        for (int mi = 0; mi < static_cast<int>(ModelType::COUNT); mi++)
        {
            auto model = static_cast<ModelType>(mi);
            if (!cfg.isModelEnabled(model)) continue;
            benchmarkModel(csvFile, model, size, cfg);
        }
    }

    fprintf(stderr, "\nIsolated benchmark complete.\n");
}
