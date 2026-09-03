// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "RTNeuralTopology.h"

#include <map>
#include <string>
#include <vector>

namespace nab {

namespace {

// Mirrors the RTNeuralLSTM_*, RTNeuralTCN_*, and RTNeuralWaveNet_* aliases in
// RTNeuralBackend.h. Keep this table in sync when adding a compiled variant.
struct CompiledTopology
{
    const char* arch;
    const char* size;
    long paramCount;
    std::map<std::string, double> hyperparams;
};

const std::vector<CompiledTopology>& compiledTopologies()
{
    static const std::vector<CompiledTopology> table = {
        {"lstm", "small",  1861,  {{"hidden", 20}}},
        {"lstm", "medium", 6921,  {{"hidden", 40}}},
        {"lstm", "large",  38113, {{"hidden", 96}}},
        {"tcn", "small",  4337,  {{"channels", 16}, {"kernel_size", 3}, {"num_layers", 4}}},
        {"tcn", "medium", 33633, {{"channels", 32}, {"kernel_size", 3}, {"num_layers", 8}}},
        {"tcn", "large",  93745, {{"channels", 48}, {"kernel_size", 3}, {"num_layers", 10}}},
        {"wavenet", "small",  841,   {{"channels", 8},  {"kernel_size", 2}, {"num_layers", 3}}},
        {"wavenet", "medium", 10609, {{"channels", 16}, {"kernel_size", 2}, {"num_layers", 10}}},
        {"wavenet", "large",  41697, {{"channels", 32}, {"kernel_size", 2}, {"num_layers", 10}}},
    };
    return table;
}

const CompiledTopology* findTopology(const ModelSpec& spec)
{
    for (const auto& topology : compiledTopologies())
        if (spec.arch == topology.arch && spec.size == topology.size)
            return &topology;
    return nullptr;
}

std::string formatHyperparam(double value)
{
    const auto asInteger = static_cast<long long>(value);
    if (static_cast<double>(asInteger) == value)
        return std::to_string(asInteger);
    return std::to_string(value);
}

std::string variantName(const ModelSpec& spec)
{
    return spec.arch + "/" + spec.size;
}

std::string rebuildAdvice()
{
    return "; add a matching template variant to RTNeuralBackend.h and "
           "RTNeuralTopology.cpp, then rebuild";
}

} // namespace

bool populateRTNeuralCompiledMetadata(ModelSpec& spec, std::string& whyNot)
{
    const auto* topology = findTopology(spec);
    if (topology == nullptr)
    {
        whyNot = "RTNeural has no compiled variant for arch=" + spec.arch +
                 " size=" + spec.size;
        return false;
    }

    spec.paramCount = topology->paramCount;
    spec.hyperparams = topology->hyperparams;
    whyNot.clear();
    return true;
}

bool validateRTNeuralCompiledTopology(const ModelSpec& spec, std::string& whyNot)
{
    const auto* topology = findTopology(spec);
    if (topology == nullptr)
    {
        whyNot = "RTNeural has no compiled variant for arch=" + spec.arch +
                 " size=" + spec.size;
        return false;
    }

    const std::string variant = variantName(spec);
    for (const auto& [key, expected] : topology->hyperparams)
    {
        const auto it = spec.hyperparams.find(key);
        if (it == spec.hyperparams.end())
        {
            whyNot = "manifest entry for RTNeural's compiled " + variant +
                     " variant is missing required hyperparameter '" + key + "'" +
                     rebuildAdvice();
            return false;
        }
        if (it->second != expected)
        {
            whyNot = "RTNeural's compiled " + variant + " variant has " + key + "=" +
                     formatHyperparam(expected) + " but the manifest declares " + key +
                     "=" + formatHyperparam(it->second) + rebuildAdvice();
            return false;
        }
    }

    if (spec.paramCount <= 0)
    {
        whyNot = "manifest entry for RTNeural's compiled " + variant +
                 " variant is missing a positive param_count" + rebuildAdvice();
        return false;
    }
    if (spec.paramCount != topology->paramCount)
    {
        whyNot = "RTNeural's compiled " + variant + " variant has " +
                 std::to_string(topology->paramCount) +
                 " parameters but the manifest declares " +
                 std::to_string(spec.paramCount) + rebuildAdvice();
        return false;
    }

    whyNot.clear();
    return true;
}

} // namespace nab
