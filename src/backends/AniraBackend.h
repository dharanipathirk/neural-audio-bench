// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "InferenceBackend.h"
#include <memory>
#include <string>
#include <vector>

#if HAS_ANIRA
#include <anira/anira.h>
#endif

// ---------------------------------------------------------------------------
// Backends that route inference through anira's InferenceHandler (background
// thread + ring buffers), the way a production plugin built on anira works.
//
// The per-callback work between the host plugin's timing record points is
// ported VERBATIM from the old AniraHandlerPlugin. One subtlety is preserved:
//   - Anira_LibTorch performed its underrun check (get_available_samples)
//     BEFORE recordStart() — i.e. OUTSIDE the timing window. That check now
//     lives in preProcess(), which the host plugin calls before recordStart.
//   - Anira_ONNX performed its underrun check / state pop-push INSIDE the
//     timing window, so it lives entirely in process().
//
// supportsIsolated() == false: anira's async scheduler makes per-call isolated
// timing meaningless, so these backends only run under contention.
// ---------------------------------------------------------------------------

class AniraLibTorchBackend : public InferenceBackend
{
public:
    bool prepare(const PrepareContext& ctx) override;
    void preProcess(int n) noexcept override;
    void process(const float* in, float* out, int n) noexcept override;
    void reset() noexcept override;
    void teardown() override;

    const char* name() const override { return "Anira_LibTorch"; }
    bool isRealtimeSafe() const override { return true; }
    bool supportsIsolated() const override { return false; }
    int latencySamples() const override { return latency; }
    const char* requiredFormat() const override { return "torchscript"; }
    int underrunCount() const override { return inferenceUnderruns; }
    void resetUnderruns() override { inferenceUnderruns = 0; }
    int timingLoggerCapacity(double sampleRate, int blockSize) const override
    {
        return static_cast<int>(sampleRate * 30.0 / blockSize);
    }

private:
    int inferenceUnderruns = 0;
    int latency = 0;

#if HAS_ANIRA
    std::unique_ptr<anira::InferenceConfig> config;
    std::unique_ptr<anira::PrePostProcessor> processor;
    std::unique_ptr<anira::InferenceHandler> handler;
#endif
};

class AniraOnnxBackend : public InferenceBackend
{
public:
    bool prepare(const PrepareContext& ctx) override;
    void process(const float* in, float* out, int n) noexcept override;
    void reset() noexcept override;
    void teardown() override;

    const char* name() const override { return "Anira_ONNX"; }
    bool isRealtimeSafe() const override { return true; }
    bool supportsIsolated() const override { return false; }
    int latencySamples() const override { return latency; }
    const char* requiredFormat() const override { return "onnx"; }
    int underrunCount() const override { return inferenceUnderruns; }
    void resetUnderruns() override { inferenceUnderruns = 0; }
    int timingLoggerCapacity(double sampleRate, int blockSize) const override
    {
        return static_cast<int>(sampleRate * 30.0 / blockSize);
    }

private:
    int inferenceUnderruns = 0;
    int latency = 0;

#if HAS_ANIRA
    std::unique_ptr<anira::InferenceConfig> config;
    std::unique_ptr<anira::PrePostProcessor> processor;
    std::unique_ptr<anira::InferenceHandler> handler;

    bool hasExplicitStateLstm = false;
    int hiddenSize = 0;
    std::vector<float> audioInputScratch;
    std::vector<float> hStateIn;
    std::vector<float> cStateIn;
    std::vector<float> hStateOut;
    std::vector<float> cStateOut;
#endif
};
