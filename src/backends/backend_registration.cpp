// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "BackendRegistry.h"
#include "BnnsGraphBackend.h"
#include "RTNeuralBackend.h"

#if HAS_LIBTORCH
#include "LibTorchBackend.h"
#endif
#if HAS_ONNXRUNTIME
#include "OnnxRuntimeBackend.h"
#endif
#if HAS_ANIRA
#include "AniraBackend.h"
#endif

#include <memory>

namespace nab {

// Registration order preserves the benchmark execution order — and thus the
// CSV row order:
//   BNNSGraph, RTNeural_(Eigen|XSIMD), Direct_LibTorch, Direct_ONNX,
//   Anira_LibTorch, Anira_ONNX.
// Compile-gated backends are simply absent from the registry.
void registerBuiltinBackends()
{
    static bool done = false;
    if (done)
        return;
    done = true;

    auto& reg = BackendRegistry::instance();

    reg.registerBackend("BNNSGraph",
        [] { return std::unique_ptr<InferenceBackend>(std::make_unique<BnnsGraphBackend>()); });

    reg.registerBackend(RTNeuralBackend::kName,
        [] { return std::unique_ptr<InferenceBackend>(std::make_unique<RTNeuralBackend>()); });

#if HAS_LIBTORCH
    reg.registerBackend("Direct_LibTorch",
        [] { return std::unique_ptr<InferenceBackend>(std::make_unique<LibTorchBackend>()); });
#endif
#if HAS_ONNXRUNTIME
    reg.registerBackend("Direct_ONNX",
        [] { return std::unique_ptr<InferenceBackend>(std::make_unique<OnnxRuntimeBackend>()); });
#endif
#if HAS_ANIRA && defined(USE_LIBTORCH)
    reg.registerBackend("Anira_LibTorch",
        [] { return std::unique_ptr<InferenceBackend>(std::make_unique<AniraLibTorchBackend>()); });
#endif
#if HAS_ANIRA && defined(USE_ONNXRUNTIME)
    reg.registerBackend("Anira_ONNX",
        [] { return std::unique_ptr<InferenceBackend>(std::make_unique<AniraOnnxBackend>()); });
#endif
}

} // namespace nab
