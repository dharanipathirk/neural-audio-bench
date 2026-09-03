// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "Scenario.h"

#include <string>
#include <vector>

namespace nab {

// ---------------------------------------------------------------------------
// ConfigurableScenario — a data-driven contention scenario built entirely from
// a JSON spec (config "custom_scenarios"), so users can define custom track
// layouts without writing C++.
//
// This is a CONVENIENCE, not a replacement for the C++ scenarios. It is kept
// deliberately simple; anything the built-in scenarios do that this cannot
// express (staggered per-track chains, mute logic, bus processing, ...) should
// be written as a first-class C++ Scenario.
//
// Spec shape (one entry of the "custom_scenarios" array):
//
//   {
//     "id": "my_scenario",
//     "sweep": { "parameter": "track_count", "values": [1, 4, 8] },
//     "buffer_sizes": [128],
//     "tracks": [
//       { "count": "sweep", "chain": ["eq", "compressor"], "clip": "noise" },
//       { "count": 1, "neural": true, "chain_after": ["delay"] }
//     ]
//   }
//
// Semantics:
//   - Each element of "tracks" describes a GROUP of identical tracks.
//   - "count": an integer, or the string "sweep" (the current sweep value sets
//     this group's track count).
//   - "clip": "noise" adds the shared noise clip as the track's audio source
//     (the default; currently the only source).
//   - "chain": plugin chain by name. Names map to the contention DSP plugins:
//       "eq" -> EQ, "compressor" -> Compressor, "reverb" -> Reverb, "delay" -> Delay.
//     For a non-neural track the chain is appended in order. For a neural track
//     the chain is the PRE-neural chain (runs before the neural plugin).
//   - "neural": true marks the group's tracks as hosting the neural plugin under
//     test. "chain_after" is the POST-neural chain (neural tracks only).
//   - Every created track gets a CallbackStart; the master bus gets a
//     CallbackEnd (via the shared EditBuilder helpers).
//   - At least one neural track is required for the run to be measurable.
//
// csvColumns: the sweep value is written to the CSV contention_level column;
// instance_count is fixed at 1.
// ---------------------------------------------------------------------------
class ConfigurableScenario : public Scenario
{
public:
    explicit ConfigurableScenario(const CustomScenarioSpec& spec)
        : spec(spec), titleStr("Custom Scenario: " + spec.id) {}

    const char* id() const override { return spec.id.c_str(); }
    const char* title() const override { return titleStr.c_str(); }

    std::vector<int> sweepValues(const BenchmarkRuntimeConfig&) const override
    {
        return spec.sweepValues;
    }
    std::vector<int> bufferSizes(const BenchmarkRuntimeConfig&) const override
    {
        return spec.bufferSizes;
    }

    SessionTimingInfo build(te::Edit& edit, EditBuilder& builder,
                            const std::string& backend, const ModelSpec& model,
                            int sweepValue, const BenchmarkRuntimeConfig& cfg,
                            double sampleRate) override;

    void csvColumns(int sweepValue, int& contentionLevel, int& instanceCount) const override
    {
        contentionLevel = sweepValue;
        instanceCount = 1;
    }

private:
    CustomScenarioSpec spec;
    std::string titleStr;
};

} // namespace nab
