// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "Scenario.h"

#include <vector>

namespace nab {

// ---------------------------------------------------------------------------
// Dimension B — Instance Count Sweep ("dim_b").
//
// One neural track per instance, no conventional contention tracks. Sweeps the
// number of neural plugin instances (instance_counts). Isolates the effect of
// adding more neural plugins at a single, fixed buffer size.
// ---------------------------------------------------------------------------
class InstanceCountScenario : public Scenario
{
public:
    const char* id() const override { return "dim_b"; }
    const char* title() const override { return "Dimension B: Instance Count Sweep"; }

    std::vector<int> sweepValues(const BenchmarkRuntimeConfig& cfg) const override
    {
        return cfg.instanceCounts;
    }

    std::vector<int> bufferSizes(const BenchmarkRuntimeConfig&) const override
    {
        // Dimension B uses a fixed buffer size of 128 samples.
        // Instance scaling at a single buffer size isolates the
        // effect of adding more neural plugins without confounding
        // buffer-size variability.
        return {128};
    }

    SessionTimingInfo build(te::Edit& edit, EditBuilder& builder,
                            BackendType backend, ModelType model, ModelSize size,
                            int sweepValue, const BenchmarkRuntimeConfig& cfg,
                            double sampleRate) override;

    void csvColumns(int sweepValue, int& contentionLevel, int& instanceCount) const override
    {
        contentionLevel = 0;
        instanceCount = sweepValue;  // number of neural instances
    }
};

} // namespace nab
