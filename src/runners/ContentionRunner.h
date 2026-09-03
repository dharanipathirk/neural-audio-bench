// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../BenchmarkConfig.h"
#include "../core/ModelManifest.h"                 // ModelSpec, findModelSpec
#include "../contention/EditBuilder.h"             // EditBuilder, SessionTimingInfo
#include "../scenarios/Scenario.h"
#include "../scenarios/ScenarioRegistry.h"

#include <tracktion_engine/tracktion_engine.h>

#include <cstdio>
#include <string>
#include <vector>

namespace te = tracktion::engine;

// Outcome of one contention configuration. Transient failures (transport
// start/restart hiccups, missing timing data — typically the audio graph not
// yet settled) are retried before an error row is written; permanent failures
// (device sample-rate mismatch, plugin creation failure) are recorded at once.
enum class ConfigResult { Ok, TransientFailure, PermanentFailure };

// ---------------------------------------------------------------------------
// Runs the full contention benchmark suite using real-time CoreAudio playback.
//
// Replaces the former ContentionBenchmark. Instead of three near-identical
// per-dimension loops, it drives ONE generic nested sweep per registered
// Scenario (sizes → models → RT-safe backends → scenario buffer sizes →
// scenario sweep values → reps), calling the scenario-agnostic runSingleConfig.
// The measurement protocol in runSingleConfig is preserved verbatim from the
// old ContentionBenchmark::runSingleConfig.
// ---------------------------------------------------------------------------
class ContentionRunner
{
public:
    ContentionRunner(te::Engine& engine, const std::string& configPath,
                     const std::vector<ModelSpec>& specs,
                     const nab::ScenarioRegistry& scenarios)
        : builder(engine, specs), engine(engine),
          configPath(configPath), specs(specs), scenarios(scenarios) {}

    void runAll(FILE* csvFile);

private:
    EditBuilder builder;
    te::Engine& engine;
    std::string configPath;
    const std::vector<ModelSpec>& specs;
    const nab::ScenarioRegistry& scenarios;

    // Ordered list of compiled-in, real-time-safe backend names taken from the
    // BackendRegistry (registry order). Enabled-in-config is checked per config
    // in the sweep loop, matching the old per-dimension loops.
    std::vector<std::string> rtSafeBackendNames() const;

    // Measurement protocol core — preserved verbatim from the old
    // ContentionBenchmark::runSingleConfig, except the session is built via the
    // scenario and the CSV sweep columns come from scenario.csvColumns().
    //
    // Writes the ok result rows on success and returns Ok. On failure it writes
    // NOTHING and returns the failure kind, setting failStatus ("error"/
    // "skipped") and failReason for the caller to record after retries are
    // exhausted. This keeps the successful-measurement path byte-identical while
    // letting the caller retry transient failures.
    ConfigResult runSingleConfig(
        FILE* csvFile,
        nab::Scenario& scenario,
        const std::string& backend,
        const ModelSpec& model,
        int bufferSize,
        int sweepValue,
        int rep,
        std::string& failStatus,
        std::string& failReason);
};
