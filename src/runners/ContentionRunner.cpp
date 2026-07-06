// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "ContentionRunner.h"

#include "../TimingLogger.h"
#include "../backends/BackendRegistry.h"

#include <cmath>

// ---------------------------------------------------------------------------
// RT-safe backend set for contention.
//
// Contention benchmarks use only RT-safe backends on the audio thread.
// Direct_LibTorch and Direct_ONNX allocate per-call (not RT-safe) and
// must be evaluated via anira's background-thread scheduler instead.
// Direct backends are still available in isolated mode for raw speed
// comparison.
//
// The set comes from the BackendRegistry (compile-gated registration) filtered
// by isRealtimeSafe(), preserving registry order. This reproduces the old
// hardcoded per-dimension arrays exactly:
//   BNNSGraph, RTNeural_(Eigen|XSIMD per compile flag),
//   Anira_LibTorch (HAS_ANIRA && HAS_LIBTORCH),
//   Anira_ONNX     (HAS_ANIRA && HAS_ONNXRUNTIME).
// Enabled-in-config filtering happens in the sweep loop, matching the old code.
// ---------------------------------------------------------------------------
std::vector<std::string> ContentionRunner::rtSafeBackendNames() const
{
    std::vector<std::string> names;
    auto& reg = BackendRegistry::instance();
    for (const auto& name : reg.names())
    {
        auto backend = reg.create(name);
        if (backend && backend->isRealtimeSafe())
            names.push_back(name);
    }
    return names;
}

// ---------------------------------------------------------------------------
// Measurement protocol core — real-time CoreAudio playback via TransportControl.
// Preserved verbatim from ContentionBenchmark::runSingleConfig, with only the
// session-build dispatch changed: the dim_a/dim_b/dim_c if-chain is replaced by
// scenario.build(), and the CSV sweep columns come from scenario.csvColumns().
// ---------------------------------------------------------------------------
void ContentionRunner::runSingleConfig(
    FILE* csvFile, nab::Scenario& scenario, BackendType backend,
    ModelType model, ModelSize size, int bufferSize, int sweepValue, int rep)
{
    using namespace tracktion::engine;

    auto cfg = BenchmarkRuntimeConfig::load(configPath);

    const char* dimension = scenario.id();
    int contentionLevel = 0, instanceCount = 1;
    scenario.csvColumns(sweepValue, contentionLevel, instanceCount);

    fprintf(stderr, "  %s: %s/%s/%s buf=%d contention=%d instances=%d rep=%d\n",
            dimension, backendTypeName(backend), modelTypeName(model), modelSizeName(size),
            bufferSize, contentionLevel, instanceCount, rep);

    // --- Configure audio device for this buffer size ---
    auto& dm = engine.getDeviceManager();
    auto setup = dm.deviceManager.getAudioDeviceSetup();
    if (setup.bufferSize != bufferSize || setup.sampleRate != cfg.sampleRate)
    {
        setup.sampleRate = cfg.sampleRate;
        setup.bufferSize = bufferSize;
        dm.deviceManager.setAudioDeviceSetup(setup, true);
    }

    auto* device = dm.deviceManager.getCurrentAudioDevice();
    double actualSR = device ? device->getCurrentSampleRate() : cfg.sampleRate;
    int actualBS = device ? device->getCurrentBufferSizeSamples() : bufferSize;

    // Guard against hidden resampling: if the device runs at a different
    // sample rate than the config, Tracktion will resample the noise WAV
    // (created at cfg.sampleRate), inflating processing cost.  Fail early.
    if (std::abs(actualSR - cfg.sampleRate) > 1.0)
    {
        fprintf(stderr, "    SKIP: device sample rate (%.0f Hz) differs from config (%.0f Hz).\n"
                "    Resampling would contaminate timing results.\n",
                actualSR, cfg.sampleRate);
        CSVOutput::printContentionStatusRow(csvFile, "error", "device sample rate differs from config",
                                          scenario.id(), backendTypeName(backend),
                                          modelTypeName(model), modelSizeName(size),
                                          bufferSize, contentionLevel, instanceCount, rep);
        return;
    }

    // --- Create Edit and build session ---
    auto edit = Edit::createSingleTrackEdit(engine);
    edit->tempoSequence.getTempo(0)->setBpm(60.0);
    edit->getMasterVolumePlugin()->setVolumeDb(0.0f);

    SessionTimingInfo timing =
        scenario.build(*edit, builder, backend, model, size, sweepValue, cfg, cfg.sampleRate);

    if (!timing.neuralLogger)
    {
        fprintf(stderr, "    SKIP: failed to create neural plugin\n");
        CSVOutput::printContentionStatusRow(csvFile, "skipped", "failed to create neural plugin",
                                          scenario.id(), backendTypeName(backend),
                                          modelTypeName(model), modelSizeName(size),
                                          bufferSize, contentionLevel, instanceCount, rep);
        return;
    }

    // Route all tracks to default output
    for (auto* track : getAudioTracks(*edit))
        track->getOutput().setOutputToDefaultDevice(false);

    // --- Start real-time playback ---
    auto& transport = edit->getTransport();
    transport.setPosition(tracktion::TimePosition());

    // Let device manager settle after any config change
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    transport.ensureContextAllocated(true);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    transport.play(false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    if (!transport.isPlaying())
    {
        fprintf(stderr, "    SKIP: transport failed to start\n");
        CSVOutput::printContentionStatusRow(csvFile, "error", "transport failed to start",
                                          scenario.id(), backendTypeName(backend),
                                          modelTypeName(model), modelSizeName(size),
                                          bufferSize, contentionLevel, instanceCount, rep);
        transport.freePlaybackContext();
        return;
    }

    // --- Warmup phase: play for a few seconds to stabilise ---
    // JIT compilation, cache warming, and thread scheduler need 2-3s minimum
    double warmupSeconds = static_cast<double>(cfg.contentionWarmupCallbacks) * bufferSize / actualSR;
    warmupSeconds = std::max(warmupSeconds, cfg.contentionWarmupMinSeconds);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(
        static_cast<int>(warmupSeconds * 1000.0));

    // Stop briefly to safely reset timing loggers (no concurrent audio thread access)
    transport.stop(false, false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    timing.neuralLogger->reset();
    for (auto* l : timing.neuralLoggers)
        l->reset();
    if (timing.callbackLogger)
        timing.callbackLogger->reset();
    if (timing.threadIdLogger)
        timing.threadIdLogger->reset();
    timing.resetInferenceUnderruns();  // zero anira underrun counters at measurement boundary

    // Restart playback for measurement phase
    transport.play(false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    if (!transport.isPlaying())
    {
        fprintf(stderr, "    SKIP: transport failed to restart for measurement\n");
        CSVOutput::printContentionStatusRow(csvFile, "error", "transport failed to restart for measurement",
                                          scenario.id(), backendTypeName(backend),
                                          modelTypeName(model), modelSizeName(size),
                                          bufferSize, contentionLevel, instanceCount, rep);
        transport.freePlaybackContext();
        return;
    }

    // --- Snapshot xrun count AFTER the 200ms restart settle ---
    // Taking the baseline here (not before restart) excludes xruns caused
    // by the transport restart transient, which is not part of steady-state
    // measurement.
    int xrunsBefore = device ? device->getXRunCount() : 0;

    // --- Measurement phase ---
    TimingUtils::init();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(cfg.contentionMeasureSeconds * 1000);

    // --- Stop playback ---
    transport.stop(false, false);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);

    // --- Snapshot xrun count AFTER measurement (1B) ---
    int xrunsAfter = device ? device->getXRunCount() : 0;
    int hwXruns = xrunsAfter - xrunsBefore;

    // --- Collect thread count (1A) ---
    int threadCount = timing.threadIdLogger ? timing.threadIdLogger->getUniqueCount() : 0;

    // --- Collect results ---
    double deadline_ns = TimingUtils::bufferDurationNs(actualBS, actualSR);

    // Neural inference timing (per-plugin)
    auto neuralDurations = timing.neuralLogger->getDurationsNs();
    if (neuralDurations.empty())
    {
        fprintf(stderr, "    SKIP: no timing data collected (applyToBuffer not called)\n");
        CSVOutput::printContentionStatusRow(csvFile, "error", "no timing data collected",
                                          scenario.id(), backendTypeName(backend),
                                          modelTypeName(model), modelSizeName(size),
                                          bufferSize, contentionLevel, instanceCount, rep);
        transport.freePlaybackContext();
        return;
    }

    // Full callback timing (graph start → master bus end)
    auto callbackDurations = timing.callbackLogger
        ? timing.callbackLogger->getDurationsNs()
        : std::vector<double>{};

    // Discard entries from the restart→measurement settle period (200ms wait
    // at line 427 above).  Trim 300ms worth of callbacks (200ms + margin) to
    // exclude transport restart transients, cache warming, and JIT.
    auto trimFront = [actualBS, actualSR](std::vector<double>& v) {
        int settleCallbacks = static_cast<int>(0.3 * actualSR / actualBS);
        int trim = std::max(settleCallbacks, static_cast<int>(v.size()) / 10);
        trim = std::min(trim, static_cast<int>(v.size()) / 2); // never trim more than half
        if (trim > 0 && static_cast<int>(v.size()) > trim)
            v.erase(v.begin(), v.begin() + trim);
    };
    trimFront(neuralDurations);
    trimFront(callbackDurations);

    auto neuralStats = TimingStats::compute(neuralDurations, deadline_ns);

    // Query anira inference underruns (0 for non-anira backends).
    // This counts callbacks where background inference output was not ready.
    int inferenceUnderruns = timing.getTotalInferenceUnderruns();

    // Print neural row (includes hw_xruns and thread_count)
    CSVOutput::printContentionRow(csvFile, dimension, backendTypeName(backend),
                                   modelTypeName(model), modelSizeName(size), actualBS,
                                   contentionLevel, instanceCount, rep, neuralStats,
                                   hwXruns, inferenceUnderruns, threadCount);

    // Print callback row (same config, dimension suffixed with "_cb")
    if (!callbackDurations.empty())
    {
        auto cbStats = TimingStats::compute(callbackDurations, deadline_ns);
        std::string cbDim = std::string(dimension) + "_cb";
        CSVOutput::printContentionRow(csvFile, cbDim.c_str(), backendTypeName(backend),
                                       modelTypeName(model), modelSizeName(size), actualBS,
                                       contentionLevel, instanceCount, rep, cbStats,
                                       hwXruns, inferenceUnderruns, threadCount);

        fprintf(stderr, "    -> neural: %d cb, p50=%.1f%% p99=%.1f%% | callback: %d cb, p50=%.1f%% p99=%.1f%% | hw_xruns=%d inf_underruns=%d threads=%d\n",
                neuralStats.total_samples, neuralStats.util_p50, neuralStats.util_p99,
                cbStats.total_samples, cbStats.util_p50, cbStats.util_p99,
                hwXruns, inferenceUnderruns, threadCount);
    }
    else
    {
        fprintf(stderr, "    -> %d callbacks, p50=%.1f%% p99=%.1f%% | hw_xruns=%d threads=%d\n",
                neuralStats.total_samples, neuralStats.util_p50, neuralStats.util_p99,
                hwXruns, threadCount);
    }

    fflush(csvFile);
    transport.freePlaybackContext();
}

// ---------------------------------------------------------------------------
// Generic sweep driver — one nested loop per registered scenario, replacing the
// three near-identical runDimensionA/B/C loops. Iteration order per scenario is
// preserved exactly: sizes → models → RT-safe backends → scenario buffer sizes
// → scenario sweep values → reps. Scenarios run in registration order (A, B, C,
// then any custom scenarios).
// ---------------------------------------------------------------------------
void ContentionRunner::runAll(FILE* csvFile)
{
    CSVOutput::printContentionHeader(csvFile);
    fflush(csvFile);

    const auto backendNames = rtSafeBackendNames();

    for (const auto& scenarioPtr : scenarios.all())
    {
        auto& scenario = *scenarioPtr;

        fprintf(stderr, "\n========================================\n");
        fprintf(stderr, "%s\n", scenario.title());
        fprintf(stderr, "========================================\n");

        // Load runtime config
        auto cfg = BenchmarkRuntimeConfig::load(configPath);

        const auto bufferSizes = scenario.bufferSizes(cfg);
        const auto sweepValues = scenario.sweepValues(cfg);

        for (int si = 0; si < static_cast<int>(ModelSize::COUNT); si++)
        {
            auto size = static_cast<ModelSize>(si);
            if (!cfg.isSizeEnabled(size)) continue;

            for (int mi = 0; mi < static_cast<int>(ModelType::COUNT); mi++)
            {
                auto model = static_cast<ModelType>(mi);
                if (!cfg.isModelEnabled(model)) continue;

                // Same catalog check as IsolatedRunner: skip size/model pairs
                // with no ModelSpec (the old code discovered this later, when
                // the neural plugin failed to attach — CSV output is identical).
                if (nab::findModelSpec(specs, model, size) == nullptr)
                {
                    fprintf(stderr, "  SKIP %s/%s: no model spec available\n",
                            modelTypeName(model), modelSizeName(size));
                    continue;
                }

                for (const auto& backendName : backendNames)
                {
                    BackendType backend;
                    if (!backendTypeFromName(backendName, backend)) continue;
                    if (!cfg.isBackendEnabled(backend)) continue;

                    for (auto bufSize : bufferSizes)
                    {
                        for (auto sweepValue : sweepValues)
                        {
                            for (int rep = 1; rep <= cfg.contentionReps; rep++)
                            {
                                runSingleConfig(csvFile, scenario, backend, model, size,
                                                bufSize, sweepValue, rep);
                            }
                        }
                    }
                }
            }
        }
    }

    fprintf(stderr, "\nContention benchmark complete.\n");
}
