// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../BenchmarkConfig.h"
#include "../contention/EditBuilder.h"   // SessionTimingInfo, EditBuilder

#include <tracktion_engine/tracktion_engine.h>

#include <vector>

namespace nab {

// ---------------------------------------------------------------------------
// A contention Scenario builds one Tracktion Engine session for a single sweep
// value (a contention level, an instance count, a serial depth, or a custom
// parameter). Adding a new contention "dimension" is: subclass Scenario +
// register it — ContentionRunner and runSingleConfig are scenario-agnostic.
//
// The measurement protocol (device setup, warmup/measure/trim, CSV emission)
// lives in ContentionRunner::runSingleConfig and never changes per scenario.
// A scenario only decides:
//   - its CSV dimension string      (id)
//   - its stderr banner             (title)
//   - what values it sweeps over    (sweepValues)   and at what buffer sizes
//     (bufferSizes)
//   - how to build the session      (build)         for a given sweep value
//   - how the sweep value maps onto the two CSV sweep columns
//     contention_level / instance_count (csvColumns)
// ---------------------------------------------------------------------------
class Scenario
{
public:
    virtual ~Scenario() = default;

    // CSV "dimension" column value, e.g. "dim_a"/"dim_b"/"dim_c" or a custom id.
    virtual const char* id() const = 0;

    // stderr banner printed once before the scenario's sweep runs.
    virtual const char* title() const = 0;

    // Sweep values for this scenario (dim_a: contention_levels, dim_b:
    // instance_counts, dim_c: neural_track_depths, custom: its own list).
    virtual std::vector<int> sweepValues(const BenchmarkRuntimeConfig& cfg) const = 0;

    // Buffer sizes to run (dim_a: cfg.contentionBufferSizes; dim_b/dim_c and
    // typical custom scenarios: a fixed single size, e.g. {128}).
    virtual std::vector<int> bufferSizes(const BenchmarkRuntimeConfig& cfg) const = 0;

    // Build the session for one sweep value. `builder` is the shared toolbox of
    // session-building helpers; scenarios call into it. `sweepValue` is the
    // current contention_level / instance_count / depth / custom value.
    virtual SessionTimingInfo build(te::Edit& edit, EditBuilder& builder,
                                    BackendType backend, ModelType model, ModelSize size,
                                    int sweepValue, const BenchmarkRuntimeConfig& cfg,
                                    double sampleRate) = 0;

    // Map the sweep value onto the CSV columns contention_level and
    // instance_count (also drives the stderr progress line).
    virtual void csvColumns(int sweepValue, int& contentionLevel, int& instanceCount) const = 0;
};

} // namespace nab
