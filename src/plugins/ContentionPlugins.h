// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <tracktion_engine/tracktion_engine.h>
#include "../TimingLogger.h"
#include "../BenchmarkConfig.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Lightweight DSP plugins for generating realistic CPU contention.
// These simulate what conventional audio plugins do in a mix session.
// All are designed to be used as Tracktion Engine Plugin subclasses.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 4-band Biquad Parametric EQ
// Each band is a second-order IIR (biquad) filter.
// ---------------------------------------------------------------------------
class BiquadEQ
{
public:
    void prepare(double sampleRate, int numBands = 4);
    void process(float* data, int numSamples);
    void reset();

private:
    struct Biquad
    {
        float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
        float z1 = 0, z2 = 0;

        float processSample(float x)
        {
            float y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }

        void setParams(double freq, double q, double gainDb, double sr);
        void resetState() { z1 = z2 = 0; }
    };

    std::vector<Biquad> bands;
};

// ---------------------------------------------------------------------------
// Simple envelope-follower compressor
// ---------------------------------------------------------------------------
class SimpleCompressor
{
public:
    void prepare(double sampleRate, float ratio = 4.0f, float threshDb = -20.0f,
                 float attackMs = 10.0f, float releaseMs = 100.0f);
    void process(float* data, int numSamples);
    void reset();

private:
    float ratio = 4.0f;
    float threshLin = 0.1f;
    float attackCoeff = 0.0f;
    float releaseCoeff = 0.0f;
    float envelope = 0.0f;
};

// ---------------------------------------------------------------------------
// Freeverb-style algorithmic reverb (Schroeder/Moorer)
// 8 parallel comb filters + 4 series allpass filters
// ---------------------------------------------------------------------------
class FreeverbReverb
{
public:
    void prepare(double sampleRate, float roomSize = 0.85f, float damping = 0.5f);
    void process(float* data, int numSamples);
    void reset();

private:
    struct CombFilter
    {
        std::vector<float> buffer;
        int writeIdx = 0;
        float feedback = 0;
        float damp1 = 0, damp2 = 0;
        float filterStore = 0;

        void init(int size, float fb, float damp);
        float processSample(float input);
        void resetState();
    };

    struct AllpassFilter
    {
        std::vector<float> buffer;
        int writeIdx = 0;
        float feedback = 0.5f;

        void init(int size);
        float processSample(float input);
        void resetState();
    };

    static constexpr int NUM_COMBS = 8;
    static constexpr int NUM_ALLPASSES = 4;

    CombFilter combs[NUM_COMBS];
    AllpassFilter allpasses[NUM_ALLPASSES];
    float wet = 0.3f;
};

// ---------------------------------------------------------------------------
// Stereo delay with feedback (used as mono for benchmark)
// ---------------------------------------------------------------------------
class StereoDelay
{
public:
    void prepare(double sampleRate, float delayMs = 375.0f, float feedback = 0.35f);
    void process(float* data, int numSamples);
    void reset();

private:
    std::vector<float> delayBuffer;
    int writeIdx = 0;
    int delaySamples = 0;
    float feedback = 0.35f;
};

// ---------------------------------------------------------------------------
// Shared callback timer — used by CallbackStartPlugin and CallbackEndPlugin
// to measure the full audio graph processing time per audio callback.
// Defined outside namespaces so EditBuilder can own its lifetime.
// ---------------------------------------------------------------------------
struct CallbackTimer
{
    std::atomic<uint64_t> blockStart{0};
};

// ---------------------------------------------------------------------------
// Tracktion Engine plugin wrappers for contention DSP
// ---------------------------------------------------------------------------
namespace tracktion { inline namespace engine {

class ContentionEQPlugin : public Plugin
{
public:
    ContentionEQPlugin(PluginCreationInfo info);
    static const char* xmlTypeName;
    juce::String getName() const override { return "ContentionEQ"; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    void initialise(const PluginInitialisationInfo&) override;
    void deinitialise() override {}
    void applyToBuffer(const PluginRenderContext& ctx) override;
    void reset() override { eq.reset(); }
    int getNumOutputChannelsGivenInputs(int n) override { return juce::jmin(n, 1); }
    bool takesAudioInput() override { return true; }
    bool isSynth() override { return false; }
private:
    BiquadEQ eq;
};

class ContentionCompPlugin : public Plugin
{
public:
    ContentionCompPlugin(PluginCreationInfo info);
    static const char* xmlTypeName;
    juce::String getName() const override { return "ContentionComp"; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    void initialise(const PluginInitialisationInfo&) override;
    void deinitialise() override {}
    void applyToBuffer(const PluginRenderContext& ctx) override;
    void reset() override { comp.reset(); }
    int getNumOutputChannelsGivenInputs(int n) override { return juce::jmin(n, 1); }
    bool takesAudioInput() override { return true; }
    bool isSynth() override { return false; }
private:
    SimpleCompressor comp;
};

class ContentionReverbPlugin : public Plugin
{
public:
    ContentionReverbPlugin(PluginCreationInfo info);
    static const char* xmlTypeName;
    juce::String getName() const override { return "ContentionReverb"; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    void initialise(const PluginInitialisationInfo&) override;
    void deinitialise() override {}
    void applyToBuffer(const PluginRenderContext& ctx) override;
    void reset() override { reverb.reset(); }
    int getNumOutputChannelsGivenInputs(int n) override { return juce::jmin(n, 1); }
    bool takesAudioInput() override { return true; }
    bool isSynth() override { return false; }
private:
    FreeverbReverb reverb;
};

class ContentionDelayPlugin : public Plugin
{
public:
    ContentionDelayPlugin(PluginCreationInfo info);
    static const char* xmlTypeName;
    juce::String getName() const override { return "ContentionDelay"; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    void initialise(const PluginInitialisationInfo&) override;
    void deinitialise() override {}
    void applyToBuffer(const PluginRenderContext& ctx) override;
    void reset() override { delay.reset(); }
    int getNumOutputChannelsGivenInputs(int n) override { return juce::jmin(n, 1); }
    bool takesAudioInput() override { return true; }
    bool isSynth() override { return false; }
private:
    StereoDelay delay;
};

// ---------------------------------------------------------------------------
// Noise generator plugin -- produces random audio so tracks process even
// without clips. Must be first plugin on every track.
// ---------------------------------------------------------------------------
class NoiseGeneratorPlugin : public Plugin
{
public:
    NoiseGeneratorPlugin(PluginCreationInfo info);
    static const char* xmlTypeName;
    juce::String getName() const override { return "NoiseGen"; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    void initialise(const PluginInitialisationInfo&) override;
    void deinitialise() override {}
    void applyToBuffer(const PluginRenderContext& ctx) override;
    void reset() override { rng.seed(42); }
    int getNumOutputChannelsGivenInputs(int) override { return 1; }
    bool takesAudioInput() override { return false; }
    bool producesAudioWhenNoAudioInput() override { return true; }
    bool isSynth() override { return false; }
private:
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-0.5f, 0.5f};
};

// ---------------------------------------------------------------------------
// Callback-level timing: measures the FULL audio graph processing cost per
// audio callback, not just one plugin's inference.
//
// CallbackStartPlugin: placed as the first plugin on every active track.
//   Each instance atomically writes mach_absolute_time to a shared counter
//   using compare-and-swap (only the earliest write wins per block).
//
// CallbackEndPlugin: placed on the master bus (runs after all tracks mixed).
//   Reads the earliest start time, records (start, end) into its TimingLogger.
// ---------------------------------------------------------------------------

class CallbackStartPlugin : public Plugin
{
public:
    CallbackStartPlugin(PluginCreationInfo info);
    static const char* xmlTypeName;
    juce::String getName() const override { return "CallbackStart"; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    void initialise(const PluginInitialisationInfo&) override {}
    void deinitialise() override {}
    void applyToBuffer(const PluginRenderContext& ctx) override;
    int getNumOutputChannelsGivenInputs(int n) override { return n; }
    bool takesAudioInput() override { return true; }
    bool isSynth() override { return false; }

    void setCallbackTimer(CallbackTimer* t) { timer = t; }
    void setThreadIDLogger(ThreadIDLogger* l) { threadIdLogger = l; }
private:
    CallbackTimer* timer = nullptr;
    ThreadIDLogger* threadIdLogger = nullptr;
};

class CallbackEndPlugin : public Plugin
{
public:
    CallbackEndPlugin(PluginCreationInfo info);
    static const char* xmlTypeName;
    juce::String getName() const override { return "CallbackEnd"; }
    juce::String getSelectableDescription() override { return getName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    void initialise(const PluginInitialisationInfo&) override;
    void deinitialise() override {}
    void applyToBuffer(const PluginRenderContext& ctx) override;
    int getNumOutputChannelsGivenInputs(int n) override { return n; }
    bool takesAudioInput() override { return true; }
    bool isSynth() override { return false; }

    void setCallbackTimer(CallbackTimer* t) { timer = t; }
    TimingLogger& getTimingLogger() { return timingLogger; }
private:
    CallbackTimer* timer = nullptr;
    TimingLogger timingLogger;
};

}} // namespace tracktion::engine
