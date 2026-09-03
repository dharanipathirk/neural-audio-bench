// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "ModelManifest.h"
#include "backends/RTNeuralTopology.h"

#include <json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>

namespace fs = std::filesystem;

namespace nab {

namespace {

[[noreturn]] void manifestFail(const std::string& path, const std::string& msg)
{
    fprintf(stderr, "ERROR: model manifest %s: %s\n", path.c_str(), msg.c_str());
    fprintf(stderr, "  See schemas/model-manifest.schema.json for the required document shape.\n");
    exit(1);
}

} // namespace

std::vector<ModelSpec> ModelManifest::load(const std::string& manifestPath)
{
    std::ifstream f(manifestPath);
    if (!f.is_open())
        manifestFail(manifestPath, "file not found");

    nlohmann::json j;
    try
    {
        f >> j;
    }
    catch (const std::exception& e)
    {
        manifestFail(manifestPath, std::string("parse error: ") + e.what());
    }

    if (!j.contains("schema_version"))
        manifestFail(manifestPath, "missing required key 'schema_version'");
    if (!j.contains("models") || !j["models"].is_array())
        manifestFail(manifestPath, "missing required 'models' array");

    const fs::path manifestAbs = fs::absolute(manifestPath);
    const fs::path manifestDir = manifestAbs.parent_path();
    fs::path modelsRoot = manifestDir;
    if (j.contains("models_root"))
        modelsRoot = manifestDir / j["models_root"].get<std::string>();

    std::vector<ModelSpec> specs;
    std::set<std::string> ids;
    try
    {
        for (const auto& m : j["models"])
        {
            ModelSpec sp;

            if (!m.contains("id"))
                manifestFail(manifestPath, "a model entry is missing required key 'id'");
            sp.id = m["id"].get<std::string>();
            if (!ids.insert(sp.id).second)
                manifestFail(manifestPath, "duplicate model id '" + sp.id + "'");

            if (!m.contains("arch"))
                manifestFail(manifestPath, "model '" + sp.id + "' is missing required key 'arch'");
            sp.arch = m["arch"].get<std::string>();

            if (m.contains("size"))
                sp.size = m["size"].get<std::string>();

            sp.displayName = m.contains("display_name")
                ? m["display_name"].get<std::string>() : sp.id;

            if (m.contains("state"))
                sp.stateful = (m["state"].get<std::string>() == "stateful");

            if (m.contains("channels"))
                sp.channels = m["channels"].get<int>();

            if (m.contains("param_count"))
                sp.paramCount = m["param_count"].get<long>();

            if (m.contains("hyperparams"))
                for (auto it = m["hyperparams"].begin(); it != m["hyperparams"].end(); ++it)
                    sp.hyperparams[it.key()] = it.value().get<double>();

            if (!m.contains("formats") || !m["formats"].is_object() || m["formats"].empty())
                manifestFail(manifestPath, "model '" + sp.id + "' is missing required 'formats'");

            for (auto it = m["formats"].begin(); it != m["formats"].end(); ++it)
            {
                fs::path abs = (modelsRoot / it.value().get<std::string>()).lexically_normal();
                sp.formatPaths[it.key()] = abs.string();
            }

            specs.push_back(std::move(sp));
        }
    }
    catch (const std::exception& e)
    {
        manifestFail(manifestPath, std::string("invalid value: ") + e.what());
    }

    fprintf(stderr, "Model manifest: loaded %zu models from %s\n",
            specs.size(), manifestPath.c_str());
    return specs;
}

std::vector<ModelSpec> ModelManifest::synthesize(const std::string& modelDir,
                                                 const BenchmarkRuntimeConfig& /*cfg*/)
{
    std::vector<ModelSpec> specs;
    for (int mi = 0; mi < static_cast<int>(ModelType::COUNT); mi++)
    {
        auto m = static_cast<ModelType>(mi);
        for (int si = 0; si < static_cast<int>(ModelSize::COUNT); si++)
        {
            auto s = static_cast<ModelSize>(si);

            ModelSpec sp;
            sp.arch = modelTypeNameLower(m);
            sp.size = modelSizeName(s);
            sp.id = sp.arch + "_" + sp.size;
            sp.displayName = std::string(modelTypeName(m)) + "-" + sp.size;
            sp.stateful = true;
            sp.channels = 1;
            sp.formatPaths["coreml"]      = modelCoreMLPath(m, s, modelDir);
            sp.formatPaths["onnx"]        = modelOnnxPath(m, s, modelDir);
            sp.formatPaths["torchscript"] = modelTorchScriptPath(m, s, modelDir);
            sp.formatPaths["rtneural"]    = modelWeightsPath(m, s, modelDir);

            std::string metadataError;
            if (!populateRTNeuralCompiledMetadata(sp, metadataError))
                manifestFail("legacy directory layout", metadataError);

            specs.push_back(std::move(sp));
        }
    }
    return specs;
}

std::vector<ModelSpec> ModelManifest::resolve(const std::string& configPath,
                                              const std::string& modelDir,
                                              const BenchmarkRuntimeConfig& cfg)
{
    if (!cfg.modelsManifest.empty())
    {
        const fs::path manifestPath =
            (fs::path(configPath).parent_path() / cfg.modelsManifest).lexically_normal();

        if (fs::exists(manifestPath))
            return load(manifestPath.string());

        fprintf(stderr,
                "model manifest not found at %s; using legacy model directory layout %s\n",
                manifestPath.string().c_str(), modelDir.c_str());
    }

    return synthesize(modelDir, cfg);
}

const ModelSpec* findModelSpec(const std::vector<ModelSpec>& specs,
                               ModelType arch, ModelSize size)
{
    const std::string a = modelTypeNameLower(arch);
    const std::string s = modelSizeName(size);
    for (const auto& sp : specs)
        if (sp.arch == a && sp.size == s)
            return &sp;
    return nullptr;
}

std::vector<const ModelSpec*> benchmarkModelOrder(const std::vector<ModelSpec>& specs)
{
    std::vector<const ModelSpec*> ordered;
    std::set<const ModelSpec*> added;

    static const char* legacySizes[] = {"small", "medium", "large"};
    static const char* legacyArchs[] = {"lstm", "tcn", "wavenet"};
    for (const auto* size : legacySizes)
        for (const auto* arch : legacyArchs)
            for (const auto& spec : specs)
                if (spec.arch == arch && spec.size == size)
                {
                    ordered.push_back(&spec);
                    added.insert(&spec);
                }

    for (const auto& spec : specs)
        if (added.find(&spec) == added.end())
            ordered.push_back(&spec);

    return ordered;
}

} // namespace nab
