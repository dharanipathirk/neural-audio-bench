// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "Scenario.h"
#include "SystemAuSessionBuilder.h"

#include <string>
#include <vector>

namespace nab {

// ---------------------------------------------------------------------------
// Dimension A — Mix Contention Sweep ("dim_a").
//
// Sweeps the number of active conventional DSP tracks alongside a single neural
// track (contention_levels). When use_system_au is set and all required Apple
// system AUs are present, the session is built with real system AUs (via
// SystemAuSessionBuilder); otherwise it falls back to the lightweight custom
// DSP layout (the original buildDimensionA). The AU scan happens lazily on the
// first build and its result is cached here.
// ---------------------------------------------------------------------------
class MixContentionScenario : public Scenario
{
public:
    MixContentionScenario(te::Engine& engine, const std::string& modelDir,
                          const std::vector<ModelSpec>& specs)
        : auBuilder(engine, modelDir, specs) {}

    const char* id() const override { return "dim_a"; }
    const char* title() const override { return "Dimension A: Mix Contention Sweep"; }

    std::vector<int> sweepValues(const BenchmarkRuntimeConfig& cfg) const override
    {
        return cfg.contentionLevels;
    }
    std::vector<int> bufferSizes(const BenchmarkRuntimeConfig& cfg) const override
    {
        return cfg.contentionBufferSizes;
    }

    SessionTimingInfo build(te::Edit& edit, EditBuilder& builder,
                            BackendType backend, ModelType model, ModelSize size,
                            int sweepValue, const BenchmarkRuntimeConfig& cfg,
                            double sampleRate) override;

    void csvColumns(int sweepValue, int& contentionLevel, int& instanceCount) const override
    {
        contentionLevel = sweepValue;  // active conventional tracks
        instanceCount = 1;
    }

private:
    // Custom-DSP fallback layout (the original EditBuilder::buildDimensionA).
    SessionTimingInfo buildCustomDsp(te::Edit& edit, EditBuilder& builder,
                                     BackendType backend, ModelType model, ModelSize size,
                                     int activeTracks, double sampleRate, int numTracks);

    SystemAuSessionBuilder auBuilder;
    bool auScanned = false;
    bool auAvailable = false;
};

} // namespace nab
