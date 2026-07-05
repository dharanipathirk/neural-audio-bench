// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "InferenceBackend.h"
#include "../BenchmarkConfig.h"

bool InferenceBackend::supports(const ModelSpec& spec, std::string& whyNot) const
{
    const std::string fmt = requiredFormat();
    if (spec.formatPaths.find(fmt) == spec.formatPaths.end())
    {
        whyNot = std::string(name()) + " requires format '" + fmt +
                 "' which model '" + spec.id + "' does not provide";
        return false;
    }
    return true;
}

int InferenceBackend::timingLoggerCapacity(double /*sampleRate*/, int /*blockSize*/) const
{
    // Matches the legacy on-thread plugins: SAMPLE_RATE * 30 / 32.
    return static_cast<int>(SAMPLE_RATE * 30.0 / 32.0);
}
