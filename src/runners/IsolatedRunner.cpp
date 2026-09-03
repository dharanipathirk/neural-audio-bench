// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "IsolatedRunner.h"
#include "../backends/BackendRegistry.h"

#include <algorithm>

std::vector<float> IsolatedRunner::generateSignal(size_t numSamples, uint32_t seed)
{
    std::vector<float> signal(numSamples);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& s : signal)
        s = dist(gen);
    return signal;
}

void IsolatedRunner::runModeA(FILE* csvFile, const char* backend, const ModelSpec& spec,
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

    const auto modelName = modelArchDisplayName(spec);
    const auto sizeName = modelSizeDisplayName(spec);
    fprintf(stderr, "  Mode A [%s/%s/%s]: %.1fx real-time (%.4fs for %.1fs audio)\n",
            backend, modelName.c_str(), sizeName.c_str(), xRT, elapsed_s, throughputSeconds);

    // Build a TimingStats for the throughput row (single measurement)
    double perSample = elapsed_ns / static_cast<double>(totalSamples);
    TimingStats ts{};
    ts.median_ns = perSample;
    ts.mean_ns = perSample;
    ts.rtf = 1.0 / xRT;
    ts.total_samples = static_cast<int>(totalSamples);
    CSVOutput::printIsolatedRow(csvFile, "throughput", backend,
                                 modelName.c_str(), sizeName.c_str(),
                                 0, 1, ts);
}

void IsolatedRunner::runModeB(FILE* csvFile, const char* backend, const ModelSpec& spec,
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
            const auto modelName = modelArchDisplayName(spec);
            const auto sizeName = modelSizeDisplayName(spec);
            CSVOutput::printIsolatedRow(csvFile, "callback", backend,
                                         modelName.c_str(), sizeName.c_str(),
                                         bufSize, rep + 1, stats);
        }
    }
}

void IsolatedRunner::benchmarkModel(FILE* csvFile, const ModelSpec& spec,
                                    const BenchmarkRuntimeConfig& cfg)
{
    const auto modelName = modelArchDisplayName(spec);
    const auto sizeName = modelSizeDisplayName(spec);
    fprintf(stderr, "\n--- %s / %s ---\n", modelName.c_str(), sizeName.c_str());

    auto& reg = BackendRegistry::instance();

    // Iterate backends in registration order (BNNSGraph, RTNeural_(variant),
    // Direct_LibTorch, Direct_ONNX, Anira_LibTorch, Anira_ONNX) so the CSV row
    // order matches the pre-refactor benchmark. Anira backends are skipped in
    // isolated mode (supportsIsolated() == false).
    for (const auto& backendName : reg.names())
    {
        if (!cfg.isBackendEnabled(backendName))
            continue;

        auto backend = reg.create(backendName);
        if (!backend)
            continue;
        if (!backend->supportsIsolated())
            continue;

        std::string whyNot;
        if (!backend->supports(spec, whyNot))
        {
            fprintf(stderr, "  SKIP %s: %s\n", backendName.c_str(), whyNot.c_str());
            CSVOutput::printIsolatedStatusRow(csvFile, "skipped", whyNot,
                                              backendName.c_str(), modelName.c_str(),
                                              sizeName.c_str());
            continue;
        }

        const int maxBlockSize = cfg.isolatedBufferSizes.empty()
            ? 0 : *std::max_element(cfg.isolatedBufferSizes.begin(), cfg.isolatedBufferSizes.end());
        PrepareContext pc{ cfg.sampleRate, maxBlockSize, &spec };
        if (!backend->prepare(pc))
        {
            fprintf(stderr, "  SKIP %s: prepare failed for %s/%s\n",
                    backendName.c_str(), modelName.c_str(), sizeName.c_str());
            CSVOutput::printIsolatedStatusRow(csvFile, "error", "prepare failed",
                                              backendName.c_str(), modelName.c_str(),
                                              sizeName.c_str());
            continue;
        }

        auto proc = [&](const float* in, float* out, int n) { backend->process(in, out, n); };
        auto reset = [&]() { backend->reset(); };

        runModeA(csvFile, backend->name(), spec, proc, cfg);
        runModeB(csvFile, backend->name(), spec, proc, reset, cfg);
    }
}

void IsolatedRunner::runAll(FILE* csvFile)
{
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Isolated Benchmark\n");
    fprintf(stderr, "========================================\n");

    CSVOutput::printIsolatedHeader(csvFile);

    auto cfg = BenchmarkRuntimeConfig::load(configPath);

    for (const auto* spec : nab::benchmarkModelOrder(specs))
    {
        if (!cfg.isSizeEnabled(spec->size) || !cfg.isModelEnabled(spec->arch))
            continue;
        benchmarkModel(csvFile, *spec, cfg);
    }

    fprintf(stderr, "\nIsolated benchmark complete.\n");
}
