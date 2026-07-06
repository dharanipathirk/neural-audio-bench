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
//   check-paper    — load a config and assert the frozen dafx26 paper values

#include "BenchmarkConfig.h"
#include "core/ModelManifest.h"

#include <cassert>
#include <cstdio>
#include <cstring>

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <load-config|load-manifest|check-paper> <path>\n", argv[0]);
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
            total += s.paramCount;
        }
        // Sum of paper Table 1 parameter counts.
        assert(total == 1861 + 6921 + 38113 + 4337 + 33633 + 93745 + 841 + 10609 + 41697);
        printf("manifest OK: %zu specs\n", specs.size());
        return 0;
    }

    if (strcmp(mode, "check-paper") == 0)
    {
        auto cfg = BenchmarkRuntimeConfig::load(path);
        // Frozen values of the final paper run — guards the preset.
        assert(cfg.isolatedMinIterations == 10000);
        assert(cfg.isolatedReps == 1);
        assert(cfg.isolatedTargetSeconds == 5.0);
        assert(cfg.contentionMeasureSeconds == 15);
        assert(cfg.contentionNumTracks == 36);
        assert(cfg.useSystemAU == true);
        assert((cfg.instanceCounts == std::vector<int>{1, 2, 4, 8, 16}));
        assert((cfg.contentionLevels == std::vector<int>{0, 8, 24, 36}));
        assert((cfg.neuralTrackDepths == std::vector<int>{1, 3, 5, 7}));
        printf("paper preset values OK\n");
        return 0;
    }

    fprintf(stderr, "unknown mode: %s\n", mode);
    return 2;
}
