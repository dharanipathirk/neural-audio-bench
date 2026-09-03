// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "Scenario.h"

#include <vector>

namespace nab {

// ---------------------------------------------------------------------------
// Dimension C — Neural Track Serial Depth ("dim_c").
//
// A single track with the neural plugin surrounded by a configurable serial
// chain. Sweeps the chain depth (neural_track_depths):
//   depth=1: neural only (bare)
//   depth=3: EQ -> Comp -> Neural (channel_strip)
//   depth=5: EQ -> Comp -> Neural -> Delay -> Reverb (mix_fx)
//   depth=7: EQ -> Comp -> Neural -> Delay -> Reverb -> EQ -> Comp (heavy_chain)
// Runs at a fixed buffer size; the depth value is stored in the CSV
// contention_level column.
// ---------------------------------------------------------------------------
class SerialDepthScenario : public Scenario
{
public:
    const char* id() const override { return "dim_c"; }
    const char* title() const override { return "Dimension C: Neural Track Serial Depth"; }

    std::vector<int> sweepValues(const BenchmarkRuntimeConfig& cfg) const override
    {
        return cfg.neuralTrackDepths;
    }

    std::vector<int> bufferSizes(const BenchmarkRuntimeConfig&) const override
    {
        // Fixed buffer size 128, no conventional contention tracks.
        return {128};
    }

    SessionTimingInfo build(te::Edit& edit, EditBuilder& builder,
                            const std::string& backend, const ModelSpec& model,
                            int sweepValue, const BenchmarkRuntimeConfig& cfg,
                            double sampleRate) override;

    void csvColumns(int sweepValue, int& contentionLevel, int& instanceCount) const override
    {
        contentionLevel = sweepValue;  // serial chain depth
        instanceCount = 1;
    }
};

} // namespace nab
