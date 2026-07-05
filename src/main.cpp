// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
/*
    Neural Audio Benchmark: BNNSGraph vs RTNeural vs anira

    Measures inference framework performance under:
    1. Isolated conditions (raw speed)
    2. Realistic DAW contention (real-time playback via BlackHole)

    Usage:
      ./nab-engine --mode isolated   [--output results/isolated.csv]
      ./nab-engine --mode contention [--output results/contention.csv]
      ./nab-engine --mode all        [--output-dir results/]
*/

#include <juce_core/juce_core.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_events/juce_events.h>
#include <tracktion_engine/tracktion_engine.h>

#include "BenchmarkConfig.h"
#include "TimingLogger.h"
#include "backends/BackendRegistry.h"
#include "core/ModelManifest.h"
#include "runners/IsolatedRunner.h"
#include "contention/EditBuilder.h"
#include "contention/TracktionPlaybackTest.h"

#include <cstdio>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// Headless engine behaviour (no device auto-init for isolated mode)
// ---------------------------------------------------------------------------
class BenchmarkEngineBehaviour : public tracktion::engine::EngineBehaviour
{
public:
    bool autoInitialiseDeviceManager() override { return initDevice; }
    bool initDevice = false;

    // Report the number of audio processing threads at startup.
    // Tracktion Engine uses this to set its thread pool size.
    // Default is typically (numCPUs - 1). We log whatever the engine decides.
    int getNumberOfCPUsToUseForAudio() override
    {
        int n = tracktion::engine::EngineBehaviour::getNumberOfCPUsToUseForAudio();
        fprintf(stderr, "Audio thread pool size: %d\n", n);
        return n;
    }
};

class BenchmarkUIBehaviour : public tracktion::engine::UIBehaviour
{
public:
    void runTaskWithProgressBar(tracktion::engine::ThreadPoolJobWithProgress&) override {}
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Parse arguments
    std::string mode = "isolated";
    std::string outputPath;
    std::string outputDir = "results";
    std::string modelDirOverride;
    std::string configPathOverride;
    bool allowAnyDevice = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            mode = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            outputPath = argv[++i];
        else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc)
            outputDir = argv[++i];
        else if (strcmp(argv[i], "--models") == 0 && i + 1 < argc)
            modelDirOverride = argv[++i];
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            configPathOverride = argv[++i];
        else if (strcmp(argv[i], "--allow-any-device") == 0)
            allowAnyDevice = true;
        else if (strcmp(argv[i], "--help") == 0)
        {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --mode <isolated|contention|test-playback|all>  Benchmark mode\n");
            printf("  --output <path>                   Output CSV file\n");
            printf("  --output-dir <dir>                Output directory (for --mode all)\n");
            printf("  --models <dir>                    Override model directory\n");
            printf("  --config <path>                   Override benchmark_config.json path\n");
            printf("  --allow-any-device                Allow contention mode on non-BlackHole devices\n");
            return 0;
        }
    }

    std::string modelDir = modelDirOverride.empty() ? MODEL_DIR : modelDirOverride;

    // A config file is required — there are no compiled-in defaults.
    if (configPathOverride.empty())
    {
        fprintf(stderr, "ERROR: --config is required (e.g. --config configs/base.json).\n"
                        "  Use `nab run` for config layering (base <- experiment <- overrides),\n"
                        "  or pass a complete config file directly.\n");
        return 1;
    }
    std::string configPath = configPathOverride;

    // Load config early so we can print the actual sample rate
    auto globalCfg = BenchmarkRuntimeConfig::load(configPath);

    fprintf(stderr, "========================================\n");
    fprintf(stderr, "neural-audio-bench engine\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "Mode: %s\n", mode.c_str());
    fprintf(stderr, "Models: %s\n", modelDir.c_str());
    fprintf(stderr, "Config: %s\n", configPath.c_str());
    fprintf(stderr, "Sample rate: %.0f Hz\n", globalCfg.sampleRate);
    fprintf(stderr, "Throughput: %.1fs, Reps: %d\n",
            globalCfg.isolatedThroughputSeconds, globalCfg.isolatedReps);

    // Initialize JUCE message manager (required even for CLI)
    juce::ScopedJuceInitialiser_GUI juceInit;

    TimingUtils::init();

    // Register the compile-available inference backends (fixed order preserves
    // benchmark execution order / CSV row order).
    nab::registerBuiltinBackends();

    // Resolve the model catalog once. Uses the manifest referenced by the
    // config (path relative to the config file) if present, else falls back to
    // the legacy models/{arch}/{size} directory layout.
    const std::vector<ModelSpec> modelSpecs =
        nab::ModelManifest::resolve(configPath, modelDir, globalCfg);

    if (mode == "isolated" || mode == "all")
    {
        std::string isoPath = outputPath.empty()
            ? outputDir + "/isolated.csv" : outputPath;

        // Ensure output directory exists
        juce::File(juce::String(outputDir)).createDirectory();

        FILE* csvFile = fopen(isoPath.c_str(), "w");
        if (!csvFile)
        {
            fprintf(stderr, "ERROR: Cannot open %s for writing\n", isoPath.c_str());
            return 1;
        }

        IsolatedRunner iso(configPath, modelSpecs);
        iso.runAll(csvFile);

        fclose(csvFile);
        fprintf(stderr, "\nIsolated results written to: %s\n", isoPath.c_str());
    }

    if (mode == "test-playback")
    {
        // Real-time playback test via CoreAudio
        auto behaviour = std::make_unique<BenchmarkEngineBehaviour>();
        behaviour->initDevice = true; // Need real audio device for CoreAudio callbacks

        tracktion::engine::Engine engine(
            "PlaybackTest",
            std::make_unique<BenchmarkUIBehaviour>(),
            std::move(behaviour)
        );

        TracktionPlaybackTest test(engine, modelDir);
        bool success = test.run();
        return success ? 0 : 1;
    }

    if (mode == "contention" || mode == "all")
    {
        std::string contPath = (mode == "all")
            ? outputDir + "/contention.csv" : outputPath;

        if (contPath.empty())
            contPath = outputDir + "/contention.csv";

        juce::File(juce::String(outputDir)).createDirectory();

        // Real-time CoreAudio playback — need audio device for deadline pressure
        auto behaviour = std::make_unique<BenchmarkEngineBehaviour>();
        behaviour->initDevice = true;

        tracktion::engine::Engine engine(
            "BenchmarkApp",
            std::make_unique<BenchmarkUIBehaviour>(),
            std::move(behaviour)
        );

        // Initialize device manager
        engine.getDeviceManager().initialise(0, 2);

        auto* device = engine.getDeviceManager().deviceManager.getCurrentAudioDevice();
        fprintf(stderr, "\nAudio device: %s\n",
                device ? device->getName().toRawUTF8() : "none");
        if (!device)
        {
            fprintf(stderr, "ERROR: No audio device available. Cannot run contention benchmark.\n");
            return 1;
        }

        // Require a virtual loopback device for contention benchmarks —
        // real-time deadline pressure on a physical device produces
        // non-reproducible results. Accepted device names come from the
        // config's virtual_output_devices list (default: BlackHole).
        {
            auto deviceName = device->getName();
            bool accepted = false;
            for (const auto& v : globalCfg.virtualOutputDevices)
                if (deviceName.containsIgnoreCase(juce::String(v)))
                    accepted = true;

            if (!accepted)
            {
                fprintf(stderr, "ERROR: Audio device '%s' is not an accepted virtual output device.\n",
                        deviceName.toRawUTF8());
                fprintf(stderr, "  Contention benchmarks require a virtual loopback device for\n"
                        "  reproducible real-time deadline pressure without acoustic output.\n"
                        "  Install: brew install blackhole-2ch\n"
                        "  Then set BlackHole as the default output device.\n"
                        "  To override: --allow-any-device\n");
                if (!allowAnyDevice)
                    return 1;
                fprintf(stderr, "  --allow-any-device set, proceeding anyway.\n");
            }
        }

        FILE* csvFile = fopen(contPath.c_str(), "w");
        if (!csvFile)
        {
            fprintf(stderr, "ERROR: Cannot open %s for writing\n", contPath.c_str());
            return 1;
        }

        ContentionBenchmark contention(engine, modelDir, configPath, modelSpecs);
        contention.runAll(csvFile);

        fclose(csvFile);
        fprintf(stderr, "\nContention results written to: %s\n", contPath.c_str());
    }

    fprintf(stderr, "\nBenchmark complete.\n");
    return 0;
}
