// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ModelSpec describes one model from the manifest (or synthesized legacy paths)
// ---------------------------------------------------------------------------
struct ModelSpec
{
    std::string id;            // "lstm_small"
    std::string arch;          // "lstm" | "tcn" | "wavenet" | custom
    std::string size;          // "small" | "medium" | "large" | custom ("" allowed)
    std::string displayName;
    bool stateful = true;
    int channels = 1;
    long paramCount = 0;
    std::map<std::string, double> hyperparams;
    std::map<std::string, std::string> formatPaths; // format name -> ABSOLUTE path ("coreml","onnx","torchscript","rtneural")
};

struct PrepareContext
{
    double sampleRate = 48000.0;
    int maxBlockSize = 0;      // 0 = unknown (isolated mode prepares without a device)
    const ModelSpec* model = nullptr;
};

// ---------------------------------------------------------------------------
// Abstract inference backend. Each concrete backend wraps one of the original
// verbatim inference engines and exposes a uniform process() surface used by
// both the isolated runner and the unified host plugin.
// ---------------------------------------------------------------------------
class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    virtual bool prepare(const PrepareContext&) = 0;

    // Optional non-timed pre-process hook, invoked immediately BEFORE the host
    // plugin's timing window opens (before TimingLogger::recordStart). Used to
    // replicate Anira_LibTorch's out-of-band underrun check, which the old
    // plugin performed before recordStart(). Default: no-op.
    virtual void preProcess(int /*n*/) noexcept {}

    virtual void process(const float* in, float* out, int n) noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual void teardown() {}

    // Declarations
    virtual const char* name() const = 0;               // EXACT CSV string
    virtual bool isRealtimeSafe() const = 0;            // may run on the audio thread under contention
    virtual bool supportsIsolated() const { return true; } // anira returns false (async scheduler; isolated per-call timing is not meaningful)
    virtual int latencySamples() const { return 0; }
    virtual const char* requiredFormat() const = 0;     // key into ModelSpec::formatPaths
    virtual bool supports(const ModelSpec& spec, std::string& whyNot) const; // default: requiredFormat present in spec.formatPaths
    virtual int underrunCount() const { return 0; }
    virtual void resetUnderruns() {}

    // Capacity (number of entries) for the host plugin's per-callback
    // TimingLogger. The legacy plugins differed here: the on-thread backends
    // (BNNSGraph/RTNeural/Direct_*) sized for SAMPLE_RATE*30/32, while the
    // anira handler backends sized for sampleRate*30/blockSize. Preserved
    // per-backend so the recorded timing set is byte-identical to before.
    virtual int timingLoggerCapacity(double sampleRate, int blockSize) const;
};
