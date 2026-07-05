// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../backends/InferenceBackend.h"   // ModelSpec
#include "../BenchmarkConfig.h"             // ModelType/ModelSize, BenchmarkRuntimeConfig

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Model manifest loader.
//   load()      — parse a manifest JSON into ModelSpecs with ABSOLUTE paths.
//   synthesize()— build the 9 classic specs from the legacy directory layout.
//   resolve()   — pick load() (manifest present) or synthesize() (fallback),
//                 resolving the manifest path relative to the CONFIG FILE dir.
// ---------------------------------------------------------------------------
namespace nab {

class ModelManifest
{
public:
    static std::vector<ModelSpec> load(const std::string& manifestPath);
    static std::vector<ModelSpec> synthesize(const std::string& modelDir,
                                             const BenchmarkRuntimeConfig& cfg);
    static std::vector<ModelSpec> resolve(const std::string& configPath,
                                          const std::string& modelDir,
                                          const BenchmarkRuntimeConfig& cfg);
};

// Look up the spec matching a known architecture/size enum pair, or nullptr.
const ModelSpec* findModelSpec(const std::vector<ModelSpec>& specs,
                               ModelType arch, ModelSize size);

} // namespace nab
