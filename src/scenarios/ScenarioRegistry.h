// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "Scenario.h"

#include <tracktion_engine/tracktion_engine.h>

#include <memory>
#include <string>
#include <vector>

namespace nab {

// ---------------------------------------------------------------------------
// Ordered registry of contention scenarios. Registration order == execution
// order == CSV grouping order (the built-ins register A, B, C; custom
// scenarios register after). Mirrors BackendRegistry, but is a plain owned
// object rather than a singleton: scenarios hold references to the locally
// constructed Engine, so their lifetime is tied to the contention run.
// ---------------------------------------------------------------------------
class ScenarioRegistry
{
public:
    void registerScenario(std::unique_ptr<Scenario> scenario)
    {
        scenarios.push_back(std::move(scenario));
    }

    const std::vector<std::unique_ptr<Scenario>>& all() const { return scenarios; }

private:
    std::vector<std::unique_ptr<Scenario>> scenarios;
};

// Populate the registry with the three built-in scenarios (dim_a Mix
// Contention, dim_b Instance Count, dim_c Serial Depth) followed by one
// ConfigurableScenario per cfg.customScenarios entry. Registration order is the
// execution/CSV order.
void registerBuiltinScenarios(ScenarioRegistry& registry,
                              te::Engine& engine,
                              const std::string& modelDir,
                              const std::vector<ModelSpec>& specs,
                              const BenchmarkRuntimeConfig& cfg);

} // namespace nab
