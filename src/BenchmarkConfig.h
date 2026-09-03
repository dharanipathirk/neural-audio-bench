// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <json.hpp>
#include <fstream>
#include <array>
#include <map>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Runtime benchmark configuration, loaded from a complete JSON document.
// ---------------------------------------------------------------------------

// Model architectures
enum class ModelType { LSTM, TCN, WaveNet, COUNT };

inline const char* modelTypeName(ModelType m)
{
    switch (m)
    {
        case ModelType::LSTM:    return "LSTM";
        case ModelType::TCN:     return "TCN";
        case ModelType::WaveNet: return "WaveNet";
        default:                 return "Unknown";
    }
}

// Model size tiers
enum class ModelSize { Small, Medium, Large, COUNT };

inline const char* modelSizeName(ModelSize s)
{
    switch (s)
    {
        case ModelSize::Small:  return "small";
        case ModelSize::Medium: return "medium";
        case ModelSize::Large:  return "large";
        default:                return "unknown";
    }
}

// Backend types
enum class BackendType {
    BNNSGraph,
    RTNeural_Eigen,
    RTNeural_XSIMD,
    Direct_LibTorch,    // Direct LibTorch API calls on audio thread (no anira scheduling)
    Direct_ONNX,        // Direct ONNX Runtime API calls on audio thread (no anira scheduling)
    Anira_LibTorch,     // LibTorch via anira InferenceHandler (background thread + ring buffers)
    Anira_ONNX,         // ONNX via anira InferenceHandler (background thread + ring buffers)
    COUNT
};

// Returns true for backends that are safe to run directly on the real-time
// audio thread (no heap allocation, no locks in the processing path).
// Direct_LibTorch and Direct_ONNX allocate per-call and must use anira's
// background-thread scheduler for contention benchmarks.
inline bool isBackendRealtimeSafe(BackendType b)
{
    switch (b)
    {
        case BackendType::BNNSGraph:
        case BackendType::RTNeural_Eigen:
        case BackendType::RTNeural_XSIMD:
        case BackendType::Anira_LibTorch:
        case BackendType::Anira_ONNX:
            return true;
        default:
            return false;
    }
}

inline const char* backendTypeName(BackendType b)
{
    switch (b)
    {
        case BackendType::BNNSGraph:        return "BNNSGraph";
        case BackendType::RTNeural_Eigen:   return "RTNeural_Eigen";
        case BackendType::RTNeural_XSIMD:   return "RTNeural_XSIMD";
        case BackendType::Direct_LibTorch:  return "Direct_LibTorch";
        case BackendType::Direct_ONNX:      return "Direct_ONNX";
        case BackendType::Anira_LibTorch:   return "Anira_LibTorch";
        case BackendType::Anira_ONNX:       return "Anira_ONNX";
        default:                            return "Unknown";
    }
}

// Reverse of backendTypeName: map a backend CSV name back to its enum.
// Returns false if the name is not a known backend.
inline bool backendTypeFromName(const std::string& name, BackendType& out)
{
    for (int bi = 0; bi < static_cast<int>(BackendType::COUNT); bi++)
    {
        auto b = static_cast<BackendType>(bi);
        if (name == backendTypeName(b))
        {
            out = b;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Model file paths — include size tier in directory structure
// Structure: models/{arch}/{size}/stateful_{arch}_{size}.{ext}
// ---------------------------------------------------------------------------

// Lowercase version for path construction
inline std::string modelTypeNameLower(ModelType m)
{
    switch (m)
    {
        case ModelType::LSTM:    return "lstm";
        case ModelType::TCN:     return "tcn";
        case ModelType::WaveNet: return "wavenet";
        default:                 return "unknown";
    }
}

inline std::string modelCoreMLPath(ModelType m, ModelSize s, const std::string& modelDir)
{
    return modelDir + "/" + modelTypeNameLower(m) + "/" + modelSizeName(s) +
           "/stateful_" + modelTypeNameLower(m) + "_" + modelSizeName(s) + ".mlmodelc";
}

inline std::string modelWeightsPath(ModelType m, ModelSize s, const std::string& modelDir)
{
    return modelDir + "/" + modelTypeNameLower(m) + "/" + modelSizeName(s) +
           "/stateful_" + modelTypeNameLower(m) + "_" + modelSizeName(s) + "_weights.json";
}

inline std::string modelOnnxPath(ModelType m, ModelSize s, const std::string& modelDir)
{
    return modelDir + "/" + modelTypeNameLower(m) + "/" + modelSizeName(s) +
           "/stateful_" + modelTypeNameLower(m) + "_" + modelSizeName(s) + ".onnx";
}

inline std::string modelTorchScriptPath(ModelType m, ModelSize s, const std::string& modelDir)
{
    return modelDir + "/" + modelTypeNameLower(m) + "/" + modelSizeName(s) +
           "/stateful_" + modelTypeNameLower(m) + "_" + modelSizeName(s) + ".pt";
}

// ---------------------------------------------------------------------------
// Optional data-driven contention scenario (consumed by ConfigurableScenario).
// Lets a user define a custom track layout in JSON without writing C++. These
// are absent from the checked-in configs, so paper behaviour is unchanged.
// ---------------------------------------------------------------------------
struct CustomScenarioTrackSpec
{
    bool countIsSweep = false;              // "count": "sweep" — the sweep value sets this group's track count
    int count = 1;                          // fixed track count (when not sweep-driven)
    bool neural = false;                    // this group hosts the neural plugin under test
    std::string clip = "noise";             // audio source ("noise" is currently the only option)
    std::vector<std::string> chain;         // plugin chain (pre-neural for a neural track)
    std::vector<std::string> chainAfter;    // post-neural plugin chain (neural tracks only)
};

struct CustomScenarioSpec
{
    std::string id;                         // CSV dimension string + scenario id
    std::string sweepParameter;             // human-readable name of the swept parameter
    std::vector<int> sweepValues;
    std::vector<int> bufferSizes;
    std::vector<CustomScenarioTrackSpec> tracks;
};

// ---------------------------------------------------------------------------
// Runtime config loaded from JSON.
//
// The config file is REQUIRED and must be complete: there are no compiled-in
// parameter defaults. A missing file, a parse error, or a missing key is a
// fatal error. This guarantees that what ran is exactly what the config says
// — the resolved config written next to the results is the full record.
// Layering (base config <- experiment preset <- CLI overrides) happens in the
// Python `nab run` frontend, which passes a fully-resolved file down.
// ---------------------------------------------------------------------------
struct BenchmarkRuntimeConfig
{
    double sampleRate = 0.0;

    // Isolated benchmark
    std::vector<int> isolatedBufferSizes;
    int isolatedWarmupIterations = 0;
    double isolatedTargetSeconds = 0.0;
    int isolatedMinIterations = 0;
    int isolatedReps = 0;
    double isolatedThroughputSeconds = 0.0;

    // Contention benchmark
    std::vector<int> contentionBufferSizes;
    std::vector<int> contentionLevels;
    std::vector<int> instanceCounts;
    std::vector<int> neuralTrackDepths;  // Dimension C: serial depth on neural track
    bool useSystemAU = true;  // Dimension A: true = real macOS system AUs, false = custom DSP
    int contentionWarmupCallbacks = 0;
    double contentionWarmupMinSeconds = 0.0;
    int contentionMeasureSeconds = 0;
    int contentionReps = 0;
    int contentionNumTracks = 0;

    // Optional: path to the model manifest (relative to the config file's
    // directory) and accepted virtual output device names for contention mode.
    std::string modelsManifest;
    std::vector<std::string> virtualOutputDevices = {"BlackHole"};

    // Optional data-driven custom scenarios (absent in the checked-in configs).
    std::vector<CustomScenarioSpec> customScenarios;

    // Dynamic selectors. Unknown model architectures/sizes default to enabled
    // so a manifest entry is benchmarked without requiring a second catalog in
    // the config. Backends are opt-in and therefore must have an explicit true
    // entry whose key matches InferenceBackend::name().
    std::map<std::string, bool> modelEnabled;
    std::map<std::string, bool> sizeEnabled;
    std::map<std::string, bool> backendEnabled;

    [[noreturn]] static void fail(const std::string& configPath, const std::string& msg)
    {
        fprintf(stderr, "ERROR: config %s: %s\n", configPath.c_str(), msg.c_str());
        fprintf(stderr, "  The config must be a complete document (see configs/base.json and\n"
                        "  schemas/config.schema.json). There are no compiled-in defaults.\n");
        exit(1);
    }

    static BenchmarkRuntimeConfig load(const std::string& configPath)
    {
        BenchmarkRuntimeConfig cfg;
        std::ifstream f(configPath);
        if (!f.is_open())
            fail(configPath, "file not found");

        nlohmann::json j;
        try
        {
            f >> j;
        }
        catch (const std::exception& e)
        {
            fail(configPath, std::string("parse error: ") + e.what());
        }

        try
        {
            // Every parameter key is required — a missing key is a config bug,
            // not a request for a default.
            auto require = [&](const nlohmann::json& obj, const char* section, const char* key) -> const nlohmann::json& {
                if (!obj.contains(key))
                    fail(configPath, std::string("missing required key '") + section + "." + key + "'");
                return obj.at(key);
            };

            if (!j.contains("sample_rate")) fail(configPath, "missing required key 'sample_rate'");
            cfg.sampleRate = j["sample_rate"];

            if (!j.contains("isolated")) fail(configPath, "missing required section 'isolated'");
            {
                auto& iso = j["isolated"];
                cfg.isolatedBufferSizes = require(iso, "isolated", "buffer_sizes").get<std::vector<int>>();
                cfg.isolatedWarmupIterations = require(iso, "isolated", "warmup_iterations");
                cfg.isolatedTargetSeconds = require(iso, "isolated", "target_measure_seconds");
                cfg.isolatedMinIterations = require(iso, "isolated", "min_iterations");
                cfg.isolatedReps = require(iso, "isolated", "num_reps");
                cfg.isolatedThroughputSeconds = require(iso, "isolated", "throughput_seconds");
            }

            if (!j.contains("contention")) fail(configPath, "missing required section 'contention'");
            {
                auto& ct = j["contention"];
                cfg.contentionBufferSizes = require(ct, "contention", "buffer_sizes").get<std::vector<int>>();
                cfg.contentionLevels = require(ct, "contention", "contention_levels").get<std::vector<int>>();
                cfg.instanceCounts = require(ct, "contention", "instance_counts").get<std::vector<int>>();
                cfg.neuralTrackDepths = require(ct, "contention", "neural_track_depths").get<std::vector<int>>();
                cfg.useSystemAU = require(ct, "contention", "use_system_au").get<bool>();
                cfg.contentionWarmupCallbacks = require(ct, "contention", "warmup_callbacks");
                cfg.contentionWarmupMinSeconds = require(ct, "contention", "warmup_min_seconds");
                cfg.contentionMeasureSeconds = require(ct, "contention", "measure_seconds");
                cfg.contentionReps = require(ct, "contention", "num_reps");
                cfg.contentionNumTracks = require(ct, "contention", "num_tracks");
                if (cfg.contentionNumTracks < 15)
                    fail(configPath, "'contention.num_tracks' must be at least 15 (track 15 hosts the neural model)");
            }

            if (j.contains("models_manifest"))
                cfg.modelsManifest = j["models_manifest"].get<std::string>();

            if (j.contains("virtual_output_devices"))
                cfg.virtualOutputDevices = j["virtual_output_devices"].get<std::vector<std::string>>();

            if (!j.contains("model_sizes")) fail(configPath, "missing required section 'model_sizes'");
            {
                auto& ms = j["model_sizes"];
                for (auto it = ms.begin(); it != ms.end(); ++it)
                    if (it.key().empty() || it.key()[0] != '_')
                        cfg.sizeEnabled[it.key()] = it.value().get<bool>();
            }

            if (j.contains("model_types"))
            {
                auto& mt = j["model_types"];
                for (auto it = mt.begin(); it != mt.end(); ++it)
                    if (it.key().empty() || it.key()[0] != '_')
                        cfg.modelEnabled[it.key()] = it.value().get<bool>();
            }

            if (!j.contains("backends")) fail(configPath, "missing required section 'backends'");
            {
                auto& be = j["backends"];
                for (auto it = be.begin(); it != be.end(); ++it)
                    if (it.key().empty() || it.key()[0] != '_')
                        cfg.backendEnabled[it.key()] = it.value().get<bool>();
            }

            // Optional: data-driven custom contention scenarios. Absent in the
            // checked-in configs; parsing here keeps the whole config in one place.
            if (j.contains("custom_scenarios"))
            {
                auto& arr = j["custom_scenarios"];
                if (!arr.is_array())
                    fail(configPath, "'custom_scenarios' must be an array");
                for (const auto& sc : arr)
                {
                    CustomScenarioSpec cs;
                    cs.id = require(sc, "custom_scenarios", "id").get<std::string>();
                    auto& sweep = require(sc, "custom_scenarios", "sweep");
                    cs.sweepParameter = require(sweep, "custom_scenarios.sweep", "parameter").get<std::string>();
                    cs.sweepValues = require(sweep, "custom_scenarios.sweep", "values").get<std::vector<int>>();
                    cs.bufferSizes = require(sc, "custom_scenarios", "buffer_sizes").get<std::vector<int>>();
                    for (const auto& t : require(sc, "custom_scenarios", "tracks"))
                    {
                        CustomScenarioTrackSpec ts;
                        if (t.contains("count"))
                        {
                            if (t["count"].is_string())
                            {
                                if (t["count"].get<std::string>() == "sweep")
                                    ts.countIsSweep = true;
                                else
                                    fail(configPath, "custom_scenarios track 'count' string must be \"sweep\"");
                            }
                            else
                                ts.count = t["count"].get<int>();
                        }
                        if (t.contains("neural")) ts.neural = t["neural"].get<bool>();
                        if (t.contains("clip")) ts.clip = t["clip"].get<std::string>();
                        if (t.contains("chain")) ts.chain = t["chain"].get<std::vector<std::string>>();
                        if (t.contains("chain_after")) ts.chainAfter = t["chain_after"].get<std::vector<std::string>>();
                        cs.tracks.push_back(std::move(ts));
                    }
                    cfg.customScenarios.push_back(std::move(cs));
                }
            }

            fprintf(stderr, "Config: loaded from %s\n", configPath.c_str());
        }
        catch (const std::exception& e)
        {
            fail(configPath, std::string("invalid value: ") + e.what());
        }

        return cfg;
    }

    bool isSizeEnabled(ModelSize s) const
    {
        return isSizeEnabled(modelSizeName(s));
    }

    bool isModelEnabled(ModelType m) const
    {
        return isModelEnabled(modelTypeNameLower(m));
    }

    bool isBackendEnabled(BackendType b) const
    {
        return isBackendEnabled(backendTypeName(b));
    }

    bool isSizeEnabled(const std::string& size) const
    {
        auto it = sizeEnabled.find(size);
        return it == sizeEnabled.end() ? true : it->second;
    }

    bool isModelEnabled(const std::string& arch) const
    {
        auto it = modelEnabled.find(arch);
        return it == modelEnabled.end() ? true : it->second;
    }

    bool isBackendEnabled(const std::string& backend) const
    {
        auto it = backendEnabled.find(backend);
        return it != backendEnabled.end() && it->second;
    }
};

// ---------------------------------------------------------------------------
// Compile-time defaults — only SAMPLE_RATE remains as a fallback for places
// that don't have access to BenchmarkRuntimeConfig, mostly timing logger
// allocation and WAV/test setup. All benchmark timing parameters come from
// BenchmarkRuntimeConfig (JSON).
// ---------------------------------------------------------------------------
constexpr double SAMPLE_RATE = 48000.0;
