// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "InferenceBackend.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Singleton registry of inference backends. Backends are registered once (see
// registerBuiltinBackends) in a fixed order that preserves the benchmark
// execution order — and therefore the CSV row order:
//   BNNSGraph, RTNeural_(Eigen|XSIMD), Direct_LibTorch, Direct_ONNX,
//   Anira_LibTorch, Anira_ONNX.
// Compile-gated availability (#if HAS_LIBTORCH etc.) simply leaves unavailable
// backends absent from the registry.
// ---------------------------------------------------------------------------
class BackendRegistry
{
public:
    using Factory = std::function<std::unique_ptr<InferenceBackend>()>;

    static BackendRegistry& instance();

    void registerBackend(const std::string& name, Factory factory);
    std::unique_ptr<InferenceBackend> create(const std::string& name) const;
    bool has(const std::string& name) const;
    const std::vector<std::string>& names() const { return order; }

private:
    std::vector<std::string> order;                 // registration order
    std::map<std::string, Factory> factories;
};

namespace nab {
// Populates the singleton registry. Idempotent — safe to call more than once.
void registerBuiltinBackends();
}
