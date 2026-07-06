// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "ScenarioRegistry.h"

#include "MixContentionScenario.h"
#include "InstanceCountScenario.h"
#include "SerialDepthScenario.h"
#include "ConfigurableScenario.h"

#include <cstdio>

namespace nab {

void registerBuiltinScenarios(ScenarioRegistry& registry,
                              te::Engine& engine,
                              const std::string& modelDir,
                              const std::vector<ModelSpec>& specs,
                              const BenchmarkRuntimeConfig& cfg)
{
    // Built-ins first, in execution/CSV order: A, B, C.
    registry.registerScenario(std::make_unique<MixContentionScenario>(engine, modelDir, specs));
    registry.registerScenario(std::make_unique<InstanceCountScenario>());
    registry.registerScenario(std::make_unique<SerialDepthScenario>());

    // Then one ConfigurableScenario per custom_scenarios entry (if any).
    for (const auto& sc : cfg.customScenarios)
    {
        fprintf(stderr, "Registered custom scenario '%s' (%zu sweep values, %zu track groups)\n",
                sc.id.c_str(), sc.sweepValues.size(), sc.tracks.size());
        registry.registerScenario(std::make_unique<ConfigurableScenario>(sc));
    }
}

} // namespace nab
