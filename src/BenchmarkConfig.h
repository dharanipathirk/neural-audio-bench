// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <json.hpp>
#include <fstream>
#include <array>
#include <string>
#include <vector>

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
// Runtime config loaded from JSON
// ---------------------------------------------------------------------------
struct BenchmarkRuntimeConfig
{
    double sampleRate = 48000.0;

    // Isolated benchmark
    std::vector<int> isolatedBufferSizes = {32, 64, 128, 256, 512, 1024};
    int isolatedWarmupIterations = 100;
    double isolatedTargetSeconds = 5.0;
    int isolatedMinIterations = 500;
    int isolatedReps = 3;
    double isolatedThroughputSeconds = 10.0;

    // Contention benchmark
    std::vector<int> contentionBufferSizes = {64, 128, 512};
    std::vector<int> contentionLevels = {0, 8, 24, 36};
    std::vector<int> instanceCounts = {1, 2, 4, 8, 12, 16};
    std::vector<int> neuralTrackDepths = {1, 3, 5, 7};  // Dimension C: serial depth on neural track
    bool useSystemAU = true;  // Dimension A: true = real macOS system AUs, false = custom DSP
    int contentionWarmupCallbacks = 200;
    double contentionWarmupMinSeconds = 3.0;
    int contentionMeasureSeconds = 15;
    int contentionReps = 2;
    int contentionNumTracks = 36;

    // Which model architectures to test
    bool modelEnabled[3] = {true, true, true}; // lstm, tcn, wavenet

    // Which sizes to test
    bool sizeEnabled[3] = {true, true, true}; // small, medium, large

    // Which backends to test (indexed by BackendType enum, default: all enabled)
    bool backendEnabled[static_cast<int>(BackendType::COUNT)] = {true, true, true, true, true, true, true};

    static BenchmarkRuntimeConfig load(const std::string& configPath)
    {
        BenchmarkRuntimeConfig cfg;
        std::ifstream f(configPath);
        if (!f.is_open())
        {
            fprintf(stderr, "Config: %s not found, using defaults\n", configPath.c_str());
            return cfg;
        }

        try
        {
            nlohmann::json j;
            f >> j;

            if (j.contains("sample_rate")) cfg.sampleRate = j["sample_rate"];

            if (j.contains("isolated"))
            {
                auto& iso = j["isolated"];
                if (iso.contains("buffer_sizes")) cfg.isolatedBufferSizes = iso["buffer_sizes"].get<std::vector<int>>();
                if (iso.contains("warmup_iterations")) cfg.isolatedWarmupIterations = iso["warmup_iterations"];
                if (iso.contains("target_measure_seconds")) cfg.isolatedTargetSeconds = iso["target_measure_seconds"];
                if (iso.contains("min_iterations")) cfg.isolatedMinIterations = iso["min_iterations"];
                if (iso.contains("num_reps")) cfg.isolatedReps = iso["num_reps"];
                if (iso.contains("throughput_seconds")) cfg.isolatedThroughputSeconds = iso["throughput_seconds"];
            }

            if (j.contains("contention"))
            {
                auto& ct = j["contention"];
                if (ct.contains("buffer_sizes")) cfg.contentionBufferSizes = ct["buffer_sizes"].get<std::vector<int>>();
                if (ct.contains("contention_levels")) cfg.contentionLevels = ct["contention_levels"].get<std::vector<int>>();
                if (ct.contains("instance_counts")) cfg.instanceCounts = ct["instance_counts"].get<std::vector<int>>();
                if (ct.contains("neural_track_depths")) cfg.neuralTrackDepths = ct["neural_track_depths"].get<std::vector<int>>();
                if (ct.contains("use_system_au")) cfg.useSystemAU = ct["use_system_au"].get<bool>();
                if (ct.contains("warmup_callbacks")) cfg.contentionWarmupCallbacks = ct["warmup_callbacks"];
                if (ct.contains("warmup_min_seconds")) cfg.contentionWarmupMinSeconds = ct["warmup_min_seconds"];
                if (ct.contains("measure_seconds")) cfg.contentionMeasureSeconds = ct["measure_seconds"];
                if (ct.contains("num_reps")) cfg.contentionReps = ct["num_reps"];
                if (ct.contains("num_tracks")) cfg.contentionNumTracks = ct["num_tracks"];
            }

            if (j.contains("model_sizes"))
            {
                auto& ms = j["model_sizes"];
                if (ms.contains("small")) cfg.sizeEnabled[0] = ms["small"];
                if (ms.contains("medium")) cfg.sizeEnabled[1] = ms["medium"];
                if (ms.contains("large")) cfg.sizeEnabled[2] = ms["large"];
            }

            if (j.contains("model_types"))
            {
                auto& mt = j["model_types"];
                if (mt.contains("lstm")) cfg.modelEnabled[0] = mt["lstm"];
                if (mt.contains("tcn")) cfg.modelEnabled[1] = mt["tcn"];
                if (mt.contains("wavenet")) cfg.modelEnabled[2] = mt["wavenet"];
            }

            if (j.contains("backends"))
            {
                auto& be = j["backends"];
                for (int bi = 0; bi < static_cast<int>(BackendType::COUNT); bi++)
                {
                    auto name = std::string(backendTypeName(static_cast<BackendType>(bi)));
                    if (be.contains(name) && be[name].is_boolean())
                        cfg.backendEnabled[bi] = be[name].get<bool>();
                }
            }

            fprintf(stderr, "Config: loaded from %s\n", configPath.c_str());
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "Config: parse error in %s: %s, using defaults\n",
                    configPath.c_str(), e.what());
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
