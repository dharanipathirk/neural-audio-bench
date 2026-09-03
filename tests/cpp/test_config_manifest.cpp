// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
//
// Config-loader and model-manifest tests. The loader is fail-fast (calls
// exit(1) on invalid input), so failure cases run as separate ctest
// invocations marked WILL_FAIL; this binary exercises one path per run.
//
// Usage: nab_test_config <mode> <path>
//   load-config    — load a config; exit 0 iff valid & complete
//   load-manifest  — load a manifest; verifies 9 specs with absolute paths
//   check-defaults — load a config and assert the shipped default protocol values

#include "BenchmarkConfig.h"
#include "backends/RTNeuralTopology.h"
#include "core/ModelManifest.h"

// The benchmark must be built in Release (NDEBUG), which would compile every
// assert() below out and make this test pass unconditionally. Re-enable them.
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <load-config|load-manifest|check-defaults> <path>\n", argv[0]);
        return 2;
    }
    const char* mode = argv[1];
    const char* path = argv[2];

    if (strcmp(mode, "load-config") == 0)
    {
        auto cfg = BenchmarkRuntimeConfig::load(path);  // exits 1 on invalid
        assert(cfg.sampleRate > 0);
        assert(!cfg.isolatedBufferSizes.empty());
        printf("config OK: %s\n", path);
        return 0;
    }

    if (strcmp(mode, "load-manifest") == 0)
    {
        auto specs = nab::ModelManifest::load(path);  // exits on invalid
        assert(specs.size() == 9);
        long total = 0;
        for (const auto& s : specs)
        {
            assert(!s.id.empty() && !s.arch.empty());
            assert(s.formatPaths.count("coreml") == 1);
            assert(s.formatPaths.at("coreml").front() == '/');  // absolute
            std::string topologyError;
            assert(nab::validateRTNeuralCompiledTopology(s, topologyError));
            total += s.paramCount;
        }
        // Sum of the nine built-in models' parameter counts.
        assert(total == 1861 + 6921 + 38113 + 4337 + 33633 + 93745 + 841 + 10609 + 41697);

        // The runner keeps the built-in size-major order (small, medium, large;
        // lstm, tcn, wavenet inside each), then schedules third-party manifest
        // entries in declaration order.
        ModelSpec custom;
        custom.id = "custom_model";
        custom.arch = "custom_arch";
        custom.size = "research";
        specs.push_back(custom);
        const auto ordered = nab::benchmarkModelOrder(specs);
        const std::vector<std::string> expectedIds = {
            "lstm_small", "tcn_small", "wavenet_small",
            "lstm_medium", "tcn_medium", "wavenet_medium",
            "lstm_large", "tcn_large", "wavenet_large", "custom_model"
        };
        assert(ordered.size() == expectedIds.size());
        for (size_t i = 0; i < expectedIds.size(); ++i)
            assert(ordered[i]->id == expectedIds[i]);

        // A manifest label must never silently select a different compile-time
        // RTNeural network. Missing and mismatched metadata are both rejected.
        std::string topologyError;
        ModelSpec missingHyperparam = specs.front();
        missingHyperparam.hyperparams.erase("hidden");
        assert(!nab::validateRTNeuralCompiledTopology(missingHyperparam, topologyError));
        assert(topologyError.find("missing required hyperparameter") != std::string::npos);

        ModelSpec mismatchedHyperparam = specs.front();
        mismatchedHyperparam.hyperparams["hidden"] = 999;
        assert(!nab::validateRTNeuralCompiledTopology(mismatchedHyperparam, topologyError));
        assert(topologyError.find("manifest declares") != std::string::npos);

        ModelSpec missingParamCount = specs.front();
        missingParamCount.paramCount = 0;
        assert(!nab::validateRTNeuralCompiledTopology(missingParamCount, topologyError));
        assert(topologyError.find("missing a positive param_count") != std::string::npos);

        ModelSpec mismatchedParamCount = specs.front();
        mismatchedParamCount.paramCount += 1;
        assert(!nab::validateRTNeuralCompiledTopology(mismatchedParamCount, topologyError));
        assert(topologyError.find("manifest declares") != std::string::npos);

        BenchmarkRuntimeConfig legacyConfig;
        const auto legacySpecs = nab::ModelManifest::synthesize("models", legacyConfig);
        assert(legacySpecs.size() == 9);
        for (const auto& s : legacySpecs)
            assert(nab::validateRTNeuralCompiledTopology(s, topologyError));

        printf("manifest OK: %zu specs\n", specs.size());
        return 0;
    }

    if (strcmp(mode, "check-defaults") == 0)
    {
        auto cfg = BenchmarkRuntimeConfig::load(path);
        // The shipped default protocol (configs/base.json) — guards against
        // accidental edits.
        assert(cfg.isolatedMinIterations == 10000);
        assert(cfg.isolatedReps == 1);
        assert(cfg.isolatedTargetSeconds == 5.0);
        assert(cfg.contentionMeasureSeconds == 15);
        assert(cfg.contentionNumTracks == 36);
        assert(cfg.useSystemAU == true);
        assert((cfg.instanceCounts == std::vector<int>{1, 2, 4, 8, 16}));
        assert((cfg.contentionLevels == std::vector<int>{0, 8, 24, 36}));
        assert((cfg.neuralTrackDepths == std::vector<int>{1, 3, 5, 7}));
        assert(cfg.isModelEnabled("third_party_arch"));
        assert(cfg.isSizeEnabled("custom_tier"));
        assert(!cfg.isBackendEnabled("not_registered"));
        printf("default protocol values OK\n");
        return 0;
    }

    fprintf(stderr, "unknown mode: %s\n", mode);
    return 2;
}
