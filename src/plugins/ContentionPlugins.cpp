// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "ContentionPlugins.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// BiquadEQ
// ---------------------------------------------------------------------------

void BiquadEQ::Biquad::setParams(double freq, double q, double gainDb, double sr)
{
    double A = std::pow(10.0, gainDb / 40.0);
    double w0 = 2.0 * M_PI * freq / sr;
    double alpha = std::sin(w0) / (2.0 * q);

    double b0d = 1.0 + alpha * A;
    double b1d = -2.0 * std::cos(w0);
    double b2d = 1.0 - alpha * A;
    double a0d = 1.0 + alpha / A;
    double a1d = -2.0 * std::cos(w0);
    double a2d = 1.0 - alpha / A;

    this->b0 = static_cast<float>(b0d / a0d);
    this->b1 = static_cast<float>(b1d / a0d);
    this->b2 = static_cast<float>(b2d / a0d);
    this->a1 = static_cast<float>(a1d / a0d);
    this->a2 = static_cast<float>(a2d / a0d);
}

void BiquadEQ::prepare(double sampleRate, int numBands)
{
    bands.resize(static_cast<size_t>(numBands));
    // Typical mix EQ: cut 300Hz, boost 1.2kHz, boost 3kHz, peaking cut 8kHz
    double freqs[] = {300.0, 1200.0, 3000.0, 8000.0};
    double gains[] = {-3.0, 2.0, 3.0, -2.0};
    double qs[] = {1.4, 1.0, 1.5, 0.7};

    for (int i = 0; i < numBands && i < 4; i++)
        bands[static_cast<size_t>(i)].setParams(freqs[i], qs[i], gains[i], sampleRate);
}

void BiquadEQ::process(float* data, int numSamples)
{
    for (auto& band : bands)
        for (int i = 0; i < numSamples; i++)
            data[i] = band.processSample(data[i]);
}

void BiquadEQ::reset()
{
    for (auto& band : bands)
        band.resetState();
}

// ---------------------------------------------------------------------------
// SimpleCompressor
// ---------------------------------------------------------------------------

void SimpleCompressor::prepare(double sampleRate, float r, float threshDb,
                                float attackMs, float releaseMs)
{
    ratio = r;
    threshLin = std::pow(10.0f, threshDb / 20.0f);
    attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * attackMs / 1000.0f));
    releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * releaseMs / 1000.0f));
    envelope = 0.0f;
}

void SimpleCompressor::process(float* data, int numSamples)
{
    for (int i = 0; i < numSamples; i++)
    {
        float absVal = std::fabs(data[i]);
        float coeff = (absVal > envelope) ? attackCoeff : releaseCoeff;
        envelope = coeff * envelope + (1.0f - coeff) * absVal;

        float gain = 1.0f;
        if (envelope > threshLin)
        {
            float overDb = 20.0f * std::log10(envelope / threshLin);
            float reduceDb = overDb * (1.0f - 1.0f / ratio);
            gain = std::pow(10.0f, -reduceDb / 20.0f);
        }
        data[i] *= gain;
    }
}

void SimpleCompressor::reset() { envelope = 0.0f; }

// ---------------------------------------------------------------------------
// FreeverbReverb
// ---------------------------------------------------------------------------

void FreeverbReverb::CombFilter::init(int size, float fb, float damp)
{
    buffer.assign(static_cast<size_t>(size), 0.0f);
    writeIdx = 0;
    feedback = fb;
    damp1 = damp;
    damp2 = 1.0f - damp;
    filterStore = 0.0f;
}

float FreeverbReverb::CombFilter::processSample(float input)
{
    float output = buffer[static_cast<size_t>(writeIdx)];
    filterStore = output * damp2 + filterStore * damp1;
    buffer[static_cast<size_t>(writeIdx)] = input + filterStore * feedback;
    writeIdx = (writeIdx + 1) % static_cast<int>(buffer.size());
    return output;
}

void FreeverbReverb::CombFilter::resetState()
{
    std::fill(buffer.begin(), buffer.end(), 0.0f);
    filterStore = 0.0f;
    writeIdx = 0;
}

void FreeverbReverb::AllpassFilter::init(int size)
{
    buffer.assign(static_cast<size_t>(size), 0.0f);
    writeIdx = 0;
}

float FreeverbReverb::AllpassFilter::processSample(float input)
{
    float bufout = buffer[static_cast<size_t>(writeIdx)];
    float output = -input + bufout;
    buffer[static_cast<size_t>(writeIdx)] = input + bufout * feedback;
    writeIdx = (writeIdx + 1) % static_cast<int>(buffer.size());
    return output;
}

void FreeverbReverb::AllpassFilter::resetState()
{
    std::fill(buffer.begin(), buffer.end(), 0.0f);
    writeIdx = 0;
}

void FreeverbReverb::prepare(double sampleRate, float roomSize, float damping)
{
    // Freeverb comb filter sizes (scaled from 44.1kHz reference)
    double scale = sampleRate / 44100.0;
    int combSizes[] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
    int apSizes[] = {556, 441, 341, 225};

    for (int i = 0; i < NUM_COMBS; i++)
        combs[i].init(static_cast<int>(combSizes[i] * scale), roomSize, damping);

    for (int i = 0; i < NUM_ALLPASSES; i++)
        allpasses[i].init(static_cast<int>(apSizes[i] * scale));

    wet = 0.3f;
}

void FreeverbReverb::process(float* data, int numSamples)
{
    for (int i = 0; i < numSamples; i++)
    {
        float input = data[i];
        float combOut = 0.0f;

        for (int c = 0; c < NUM_COMBS; c++)
            combOut += combs[c].processSample(input);

        combOut /= static_cast<float>(NUM_COMBS);

        for (int a = 0; a < NUM_ALLPASSES; a++)
            combOut = allpasses[a].processSample(combOut);

        data[i] = input * (1.0f - wet) + combOut * wet;
    }
}

void FreeverbReverb::reset()
{
    for (auto& c : combs) c.resetState();
    for (auto& a : allpasses) a.resetState();
}

// ---------------------------------------------------------------------------
// StereoDelay
// ---------------------------------------------------------------------------

void StereoDelay::prepare(double sampleRate, float delayMs, float fb)
{
    delaySamples = static_cast<int>(sampleRate * delayMs / 1000.0);
    delayBuffer.assign(static_cast<size_t>(delaySamples), 0.0f);
    writeIdx = 0;
    feedback = fb;
}

void StereoDelay::process(float* data, int numSamples)
{
    if (delayBuffer.empty()) return;
    int bufSize = static_cast<int>(delayBuffer.size());

    for (int i = 0; i < numSamples; i++)
    {
        int readIdx = (writeIdx - delaySamples + bufSize) % bufSize;
        float delayed = delayBuffer[static_cast<size_t>(readIdx)];
        delayBuffer[static_cast<size_t>(writeIdx)] = data[i] + delayed * feedback;
        data[i] = data[i] * 0.7f + delayed * 0.3f;
        writeIdx = (writeIdx + 1) % bufSize;
    }
}

void StereoDelay::reset()
{
    std::fill(delayBuffer.begin(), delayBuffer.end(), 0.0f);
    writeIdx = 0;
}

// ---------------------------------------------------------------------------
// Tracktion Engine plugin wrappers
// ---------------------------------------------------------------------------

namespace tracktion { inline namespace engine {

// EQ
const char* ContentionEQPlugin::xmlTypeName = "contentionEQ";
ContentionEQPlugin::ContentionEQPlugin(PluginCreationInfo info) : Plugin(info) {}
void ContentionEQPlugin::initialise(const PluginInitialisationInfo& info) { eq.prepare(info.sampleRate); }
void ContentionEQPlugin::applyToBuffer(const PluginRenderContext& ctx) {
    if (ctx.destBuffer == nullptr) return;
    eq.process(ctx.destBuffer->getWritePointer(0) + ctx.bufferStartSample, ctx.bufferNumSamples);
}

// Compressor
const char* ContentionCompPlugin::xmlTypeName = "contentionComp";
ContentionCompPlugin::ContentionCompPlugin(PluginCreationInfo info) : Plugin(info) {}
void ContentionCompPlugin::initialise(const PluginInitialisationInfo& info) { comp.prepare(info.sampleRate); }
void ContentionCompPlugin::applyToBuffer(const PluginRenderContext& ctx) {
    if (ctx.destBuffer == nullptr) return;
    comp.process(ctx.destBuffer->getWritePointer(0) + ctx.bufferStartSample, ctx.bufferNumSamples);
}

// Reverb
const char* ContentionReverbPlugin::xmlTypeName = "contentionReverb";
ContentionReverbPlugin::ContentionReverbPlugin(PluginCreationInfo info) : Plugin(info) {}
void ContentionReverbPlugin::initialise(const PluginInitialisationInfo& info) { reverb.prepare(info.sampleRate); }
void ContentionReverbPlugin::applyToBuffer(const PluginRenderContext& ctx) {
    if (ctx.destBuffer == nullptr) return;
    reverb.process(ctx.destBuffer->getWritePointer(0) + ctx.bufferStartSample, ctx.bufferNumSamples);
}

// Delay
const char* ContentionDelayPlugin::xmlTypeName = "contentionDelay";
ContentionDelayPlugin::ContentionDelayPlugin(PluginCreationInfo info) : Plugin(info) {}
void ContentionDelayPlugin::initialise(const PluginInitialisationInfo& info) { delay.prepare(info.sampleRate); }
void ContentionDelayPlugin::applyToBuffer(const PluginRenderContext& ctx) {
    if (ctx.destBuffer == nullptr) return;
    delay.process(ctx.destBuffer->getWritePointer(0) + ctx.bufferStartSample, ctx.bufferNumSamples);
}

// NoiseGenerator
const char* NoiseGeneratorPlugin::xmlTypeName = "noiseGenerator";
NoiseGeneratorPlugin::NoiseGeneratorPlugin(PluginCreationInfo info) : Plugin(info) {}
void NoiseGeneratorPlugin::initialise(const PluginInitialisationInfo&) { rng.seed(42); }
void NoiseGeneratorPlugin::applyToBuffer(const PluginRenderContext& ctx) {
    if (ctx.destBuffer == nullptr) return;
    auto* data = ctx.destBuffer->getWritePointer(0);
    for (int i = ctx.bufferStartSample; i < ctx.bufferStartSample + ctx.bufferNumSamples; i++)
        data[i] = dist(rng);
}

// ---------------------------------------------------------------------------
// Callback timing plugins
// ---------------------------------------------------------------------------

// CallbackStart: atomically record the earliest timestamp per block
const char* CallbackStartPlugin::xmlTypeName = "callbackStart";
CallbackStartPlugin::CallbackStartPlugin(PluginCreationInfo info) : Plugin(info) {}
void CallbackStartPlugin::applyToBuffer(const PluginRenderContext& ctx) {
    if (!timer || ctx.destBuffer == nullptr) return;
    // Record the calling thread's ID (lock-free, once per generation).
    if (threadIdLogger)
        threadIdLogger->record();
    // CAS: only the first plugin to reach this per block writes the start time.
    // Subsequent tracks in the same block see a non-zero value and skip.
    uint64_t expected = 0;
    uint64_t now = TimingUtils::now();
    timer->blockStart.compare_exchange_strong(expected, now, std::memory_order_relaxed);
}

// CallbackEnd: runs on master bus after all tracks. Records (start, now) pair.
const char* CallbackEndPlugin::xmlTypeName = "callbackEnd";
CallbackEndPlugin::CallbackEndPlugin(PluginCreationInfo info) : Plugin(info) {}
void CallbackEndPlugin::initialise(const PluginInitialisationInfo&) {
    timingLogger.allocate(static_cast<int>(SAMPLE_RATE * 30.0 / 32.0));
}
void CallbackEndPlugin::applyToBuffer(const PluginRenderContext& ctx) {
    if (!timer || ctx.destBuffer == nullptr) return;
    uint64_t start = timer->blockStart.exchange(0, std::memory_order_relaxed);
    if (start == 0) return; // no start recorded (shouldn't happen)
    // Record the full callback span
    timingLogger.recordExternalPair(start, TimingUtils::now());
}

}} // namespace tracktion::engine
