// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <json.hpp>
#include <fstream>
#include <array>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Runtime benchmark configuration, loaded from benchmark_config.json.
// Falls back to compiled defaults if the config file is missing.
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

    // Which model architectures to test
    bool modelEnabled[3] = {true, true, true}; // lstm, tcn, wavenet

    // Which sizes to test
    bool sizeEnabled[3] = {true, true, true}; // small, medium, large

    // Which backends to test (indexed by BackendType enum)
    bool backendEnabled[static_cast<int>(BackendType::COUNT)] = {true, true, true, true, true, true, true};

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
            }

            if (j.contains("models_manifest"))
                cfg.modelsManifest = j["models_manifest"].get<std::string>();

            if (j.contains("virtual_output_devices"))
                cfg.virtualOutputDevices = j["virtual_output_devices"].get<std::vector<std::string>>();

            if (!j.contains("model_sizes")) fail(configPath, "missing required section 'model_sizes'");
            {
                auto& ms = j["model_sizes"];
                cfg.sizeEnabled[0] = require(ms, "model_sizes", "small").get<bool>();
                cfg.sizeEnabled[1] = require(ms, "model_sizes", "medium").get<bool>();
                cfg.sizeEnabled[2] = require(ms, "model_sizes", "large").get<bool>();
            }

            if (j.contains("model_types"))
            {
                auto& mt = j["model_types"];
                if (mt.contains("lstm")) cfg.modelEnabled[0] = mt["lstm"];
                if (mt.contains("tcn")) cfg.modelEnabled[1] = mt["tcn"];
                if (mt.contains("wavenet")) cfg.modelEnabled[2] = mt["wavenet"];
            }

            if (!j.contains("backends")) fail(configPath, "missing required section 'backends'");
            {
                auto& be = j["backends"];
                // Reject unknown backend names (typo protection). Keys starting
                // with '_' are comments.
                for (auto it = be.begin(); it != be.end(); ++it)
                {
                    const std::string& key = it.key();
                    if (!key.empty() && key[0] == '_')
                        continue;
                    bool known = false;
                    for (int bi = 0; bi < static_cast<int>(BackendType::COUNT); bi++)
                        if (key == backendTypeName(static_cast<BackendType>(bi)))
                            known = true;
                    if (!known)
                        fail(configPath, "unknown backend '" + key + "' in 'backends'");
                }
                for (int bi = 0; bi < static_cast<int>(BackendType::COUNT); bi++)
                {
                    auto name = std::string(backendTypeName(static_cast<BackendType>(bi)));
                    if (!be.contains(name))
                        fail(configPath, "missing required key 'backends." + name + "'");
                    cfg.backendEnabled[bi] = be[name].get<bool>();
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
        return sizeEnabled[static_cast<int>(s)];
    }

    bool isModelEnabled(ModelType m) const
    {
        return modelEnabled[static_cast<int>(m)];
    }

    bool isBackendEnabled(BackendType b) const
    {
        int idx = static_cast<int>(b);
        return idx >= 0 && idx < static_cast<int>(BackendType::COUNT) && backendEnabled[idx];
    }
};

// ---------------------------------------------------------------------------
// Compile-time defaults — only SAMPLE_RATE remains as a fallback for places
// that don't have access to BenchmarkRuntimeConfig, mostly timing logger
// allocation and WAV/test setup. All benchmark timing parameters come from
// BenchmarkRuntimeConfig (JSON).
// ---------------------------------------------------------------------------
constexpr double SAMPLE_RATE = 48000.0;
