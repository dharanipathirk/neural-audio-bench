// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "InferenceBackend.h"

#include <string>

namespace nab {

// RTNeural's benchmark models are compile-time template instantiations. These
// helpers keep manifest metadata and the compiled variants in lockstep without
// requiring the lightweight config tests to include RTNeural itself.
bool populateRTNeuralCompiledMetadata(ModelSpec& spec, std::string& whyNot);
bool validateRTNeuralCompiledTopology(const ModelSpec& spec, std::string& whyNot);

} // namespace nab
