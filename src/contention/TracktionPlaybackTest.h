// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "../BenchmarkConfig.h"
#include "../TimingLogger.h"
#include "../host/NeuralInferencePlugin.h"
#include "../backends/BackendRegistry.h"
#include "../plugins/ContentionPlugins.h"

#include <tracktion_engine/tracktion_engine.h>
// Graph module: full definitions for PlayHead, PlayHeadState, Node, ThreadPoolStrategy
#include <tracktion_graph/tracktion_graph.h>
// Internal engine headers for graph-level processing
#include <tracktion_engine/playback/graph/tracktion_TracktionEngineNode.h>
#include <tracktion_engine/playback/graph/tracktion_EditNodeBuilder.h>
#include <tracktion_engine/playback/graph/tracktion_TracktionNodePlayer.h>

#include <string>

namespace te = tracktion::engine;

// ---------------------------------------------------------------------------
// Standalone test that verifies Tracktion Engine graph-level processing works:
//   - Creates an Edit with tracks + WAV clips + custom plugins
//   - Builds the node graph via createNodeForEdit()
//   - Processes blocks via TracktionNodePlayer
//   - Verifies applyToBuffer is called (via TimingLogger count)
// ---------------------------------------------------------------------------
class TracktionPlaybackTest
{
public:
    TracktionPlaybackTest(te::Engine& engine, const std::string& modelDir)
        : engine(engine), modelDir(modelDir) {}

    bool run();

private:
    te::Engine& engine;
    std::string modelDir;

    void registerPluginTypes();
    bool pluginsRegistered = false;

    juce::File createNoiseWavFile(double durationSeconds, double sampleRate);
};
