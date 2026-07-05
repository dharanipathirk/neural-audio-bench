// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include "InferenceBackend.h"
#include "../BenchmarkConfig.h"

// RTNeural backend is selected via CMake compile definitions:
//   RTNEURAL_USE_EIGEN=1  → Eigen backend (default, nab-engine)
//   RTNEURAL_USE_XSIMD=1  → XSIMD backend (nab-engine-xsimd)
// If neither is defined, fall back to Eigen so the project builds standalone.
#if !defined(RTNEURAL_USE_EIGEN) && !defined(RTNEURAL_USE_XSIMD)
#define RTNEURAL_USE_EIGEN 1
#endif
#include <RTNeural/RTNeural.h>

#include <json.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "arena.hpp"  // RTNeural-NAM memory arena for buffer processing

// ---------------------------------------------------------------------------
// RTNeural engine wrapping all 3 model architectures at all 3 size tiers.
// Uses compile-time templates for maximum performance (Eigen backend).
//
// LSTM uses RTNeural::ModelT (purely sequential, sufficient for LSTM→Dense).
// TCN and WaveNet use CUSTOM structs that call RTNeural layer primitives
// directly, following the RTNeural-NAM pattern (wavenet_layer.hpp).
// This enables residual connections (TCN, WaveNet) and skip connections
// (WaveNet) that ModelT cannot express.
//
// NOTE: Changing model sizes requires recompiling. The sizes below must match
// the values in benchmark_config.json.
// ---------------------------------------------------------------------------

// ===================== LSTM =====================
// LSTM(1, H) -> Dense(H, 1)

// Small: hidden=20
using RTNeuralLSTM_Small = RTNeural::ModelT<float, 1, 1,
    RTNeural::LSTMLayerT<float, 1, 20>,
    RTNeural::DenseT<float, 20, 1>>;

// Medium: hidden=40
using RTNeuralLSTM_Medium = RTNeural::ModelT<float, 1, 1,
    RTNeural::LSTMLayerT<float, 1, 40>,
    RTNeural::DenseT<float, 40, 1>>;

// Large: hidden=96
using RTNeuralLSTM_Large = RTNeural::ModelT<float, 1, 1,
    RTNeural::LSTMLayerT<float, 1, 96>,
    RTNeural::DenseT<float, 96, 1>>;

// ===================== TCN =====================
// Uses RTNeural primitives directly (NOT ModelT) to implement residual connections.
// Per layer: activated = PReLU(Conv(state))
//            h = h + Dense1x1(activated)      <- residual
// Following RTNeural-NAM pattern (wavenet_layer.hpp).

template <typename T, int channels, int kernel_size, int dilation>
struct TCNLayer
{
    static constexpr int ch = channels;
    RTNeural::Conv1DT<T, channels, channels, kernel_size, dilation> conv;
    RTNeural::PReLUActivationT<T, channels> activation;
    RTNeural::DenseT<T, channels, channels> residual_1x1;

#if RTNEURAL_USE_EIGEN
    Eigen::Matrix<T, channels, 1> outs;
#elif RTNEURAL_USE_XSIMD
    static constexpr int vec_size = RTNeural::ceil_div(channels, (int)xsimd::batch<T>::size);
    xsimd::batch<T> outs[vec_size];
#endif

    void reset() { conv.reset(); }

    RTNEURAL_REALTIME void forward(const auto& ins) noexcept
    {
        conv.forward(ins);
        activation.forward(conv.outs);
        residual_1x1.forward(activation.outs);
#if RTNEURAL_USE_EIGEN
        outs = ins + residual_1x1.outs;
#elif RTNEURAL_USE_XSIMD
        for (int i = 0; i < vec_size; ++i)
            outs[i] = ins[i] + residual_1x1.outs[i];
#endif
    }

    // Buffer forward: process N samples with better cache locality
#if RTNEURAL_USE_EIGEN
    using Vec = Eigen::Matrix<T, channels, 1>;
    void forward(const Vec* ins, Vec* layer_outs, int N, wavenet::Memory_Arena<>& arena) noexcept
    {
        const auto _ = arena.create_frame();
        auto* temp = arena.template allocate<Vec>(N, RTNEURAL_DEFAULT_ALIGNMENT);

        for (int n = 0; n < N; ++n) { conv.forward(ins[n]); temp[n] = conv.outs; }
        for (int n = 0; n < N; ++n) { activation.forward(temp[n]); temp[n] = activation.outs; }
        for (int n = 0; n < N; ++n)
        {
            residual_1x1.forward(temp[n]);
            layer_outs[n] = ins[n] + residual_1x1.outs;
        }
    }
#elif RTNEURAL_USE_XSIMD
    using Vec = xsimd::batch<T>[vec_size];
    void forward(const Vec* ins, Vec* layer_outs, int N, wavenet::Memory_Arena<>& arena) noexcept
    {
        const auto _ = arena.create_frame();
        auto* temp = arena.template allocate<Vec>(N, RTNEURAL_DEFAULT_ALIGNMENT);

        for (int n = 0; n < N; ++n) { conv.forward(ins[n]); for (int i = 0; i < vec_size; ++i) temp[n][i] = conv.outs[i]; }
        for (int n = 0; n < N; ++n) { activation.forward(temp[n]); for (int i = 0; i < vec_size; ++i) temp[n][i] = activation.outs[i]; }
        for (int n = 0; n < N; ++n)
        {
            residual_1x1.forward(temp[n]);
            for (int i = 0; i < vec_size; ++i) layer_outs[n][i] = ins[n][i] + residual_1x1.outs[i];
        }
    }
#endif
};

// TCN model: input_conv -> N x TCNLayer -> output_conv
template <typename T, int channels, int kernel_size, typename... Layers>
struct TCNModel
{
    RTNeural::Conv1DT<T, 1, channels, 1, 1> input_conv;
    std::tuple<Layers...> layers;
    RTNeural::Conv1DT<T, channels, 1, 1, 1> output_conv;

    wavenet::Memory_Arena<> arena;

    void reset()
    {
        input_conv.reset();
        output_conv.reset();
        std::apply([](auto&... l) { (l.reset(), ...); }, layers);
    }

    void prepare(int block_size)
    {
#if RTNEURAL_USE_EIGEN
        using Vec = Eigen::Matrix<T, channels, 1>;
#elif RTNEURAL_USE_XSIMD
        using Vec = xsimd::batch<T>[RTNeural::ceil_div(channels, (int)xsimd::batch<T>::size)];
#endif
        // Each layer needs temp storage for N samples, plus input_conv outputs
        size_t bytes = sizeof(Vec) * block_size * (sizeof...(Layers) + 2);
        arena.resize(bytes + 256);
    }

    // Per-sample forward (unchanged)
    RTNEURAL_REALTIME T forward(const T* input) noexcept
    {
#if RTNEURAL_USE_EIGEN
        input_conv.forward(Eigen::Matrix<T, 1, 1>::Map(input));
#elif RTNEURAL_USE_XSIMD
        using v_type = xsimd::batch<T>;
        static constexpr int v_in = RTNeural::ceil_div(1, (int)v_type::size);
        v_type in_arr[v_in] { v_type(*input) };
        input_conv.forward(in_arr);
#endif
        forwardLayers(std::make_index_sequence<sizeof...(Layers)>{});
        output_conv.forward(lastOuts());
#if RTNEURAL_USE_EIGEN
        return output_conv.outs(0);
#elif RTNEURAL_USE_XSIMD
        return output_conv.outs[0].get(0);
#endif
    }

    // Buffer forward: process N samples with improved cache locality
    void forward(const T* input, T* output, int N) noexcept
    {
#if RTNEURAL_USE_EIGEN
        using Vec = Eigen::Matrix<T, channels, 1>;
        const auto* v_ins = reinterpret_cast<const Eigen::Matrix<T, 1, 1>*>(input);
#elif RTNEURAL_USE_XSIMD
        using v_type = xsimd::batch<T>;
        static constexpr int v_ch = RTNeural::ceil_div(channels, (int)v_type::size);
        using Vec = xsimd::batch<T>[v_ch];
        auto* v_ins = arena.template allocate<xsimd::batch<T>[1]>(N, RTNEURAL_DEFAULT_ALIGNMENT);
        for (int n = 0; n < N; ++n)
            v_ins[n][0] = xsimd::batch<T>(input[n]);
#endif

        // Process all N through input_conv
        auto* layer_io = arena.template allocate<Vec>(N, RTNEURAL_DEFAULT_ALIGNMENT);
        for (int n = 0; n < N; ++n)
        {
            input_conv.forward(v_ins[n]);
#if RTNEURAL_USE_EIGEN
            layer_io[n] = input_conv.outs;
#elif RTNEURAL_USE_XSIMD
            for (int i = 0; i < v_ch; ++i) layer_io[n][i] = input_conv.outs[i];
#endif
        }

        // Process all N through each layer
        forwardLayersBuffer(layer_io, N, std::make_index_sequence<sizeof...(Layers)>{});

        // Process all N through output_conv
        for (int n = 0; n < N; ++n)
        {
            output_conv.forward(layer_io[n]);
#if RTNEURAL_USE_EIGEN
            output[n] = output_conv.outs(0);
#elif RTNEURAL_USE_XSIMD
            output[n] = output_conv.outs[0].get(0);
#endif
        }

        arena.clear();
    }

private:
    template <size_t... Is>
    RTNEURAL_REALTIME void forwardLayers(std::index_sequence<Is...>) noexcept
    {
        (forwardLayer<Is>(), ...);
    }

    template <size_t I>
    RTNEURAL_REALTIME void forwardLayer() noexcept
    {
        if constexpr (I == 0)
            std::get<0>(layers).forward(input_conv.outs);
        else
            std::get<I>(layers).forward(std::get<I - 1>(layers).outs);
    }

    RTNEURAL_REALTIME auto& lastOuts() noexcept
    {
        return std::get<sizeof...(Layers) - 1>(layers).outs;
    }

#if RTNEURAL_USE_EIGEN
    using LayerVec = Eigen::Matrix<T, channels, 1>;
#elif RTNEURAL_USE_XSIMD
    using LayerVec = xsimd::batch<T>[RTNeural::ceil_div(channels, (int)xsimd::batch<T>::size)];
#endif

    template <size_t... Is>
    void forwardLayersBuffer(LayerVec* io, int N, std::index_sequence<Is...>) noexcept
    {
        (forwardLayerBuffer<Is>(io, N), ...);
    }

    template <size_t I>
    void forwardLayerBuffer(LayerVec* io, int N) noexcept
    {
        std::get<I>(layers).forward(io, io, N, arena);
    }
};

// Concrete TCN types for each size tier
using RTNeuralTCN_Small = TCNModel<float, 16, 3,
    TCNLayer<float, 16, 3, 1>,
    TCNLayer<float, 16, 3, 2>,
    TCNLayer<float, 16, 3, 4>,
    TCNLayer<float, 16, 3, 8>>;

using RTNeuralTCN_Medium = TCNModel<float, 32, 3,
    TCNLayer<float, 32, 3, 1>,
    TCNLayer<float, 32, 3, 2>,
    TCNLayer<float, 32, 3, 4>,
    TCNLayer<float, 32, 3, 8>,
    TCNLayer<float, 32, 3, 16>,
    TCNLayer<float, 32, 3, 32>,
    TCNLayer<float, 32, 3, 64>,
    TCNLayer<float, 32, 3, 128>>;

using RTNeuralTCN_Large = TCNModel<float, 48, 3,
    TCNLayer<float, 48, 3, 1>,
    TCNLayer<float, 48, 3, 2>,
    TCNLayer<float, 48, 3, 4>,
    TCNLayer<float, 48, 3, 8>,
    TCNLayer<float, 48, 3, 16>,
    TCNLayer<float, 48, 3, 32>,
    TCNLayer<float, 48, 3, 64>,
    TCNLayer<float, 48, 3, 128>,
    TCNLayer<float, 48, 3, 256>,
    TCNLayer<float, 48, 3, 512>>;

// ===================== WaveNet =====================
// Uses RTNeural primitives directly (NOT ModelT) to implement residual + skip.
// Per layer: activated = Tanh(Conv(state))
//            skip_sum += Dense_skip(activated)      <- skip connection
//            h = h + Dense_1x1(activated)           <- residual connection
// Output: Conv1x1(skip_sum)
// Following RTNeural-NAM pattern (wavenet_layer.hpp).

template <typename T, int channels, int kernel_size, int dilation>
struct WaveNetLayer
{
    static constexpr int ch = channels;
    RTNeural::Conv1DT<T, channels, channels, kernel_size, dilation> conv;
    RTNeural::TanhActivationT<T, channels> activation;
    RTNeural::DenseT<T, channels, channels> residual_1x1;
    RTNeural::DenseT<T, channels, channels, false> skip_1x1;  // no bias (matches PyTorch)

#if RTNEURAL_USE_EIGEN
    Eigen::Matrix<T, channels, 1> outs;
#elif RTNEURAL_USE_XSIMD
    static constexpr int vec_size = RTNeural::ceil_div(channels, (int)xsimd::batch<T>::size);
    xsimd::batch<T> outs[vec_size];
#endif

    void reset() { conv.reset(); }

    RTNEURAL_REALTIME void forward(const auto& ins, auto& skip_sum) noexcept
    {
        conv.forward(ins);
        activation.forward(conv.outs);

        skip_1x1.forward(activation.outs);
#if RTNEURAL_USE_EIGEN
        skip_sum += skip_1x1.outs;
#elif RTNEURAL_USE_XSIMD
        for (int i = 0; i < vec_size; ++i)
            skip_sum[i] += skip_1x1.outs[i];
#endif

        residual_1x1.forward(activation.outs);
#if RTNEURAL_USE_EIGEN
        outs = ins + residual_1x1.outs;
#elif RTNEURAL_USE_XSIMD
        for (int i = 0; i < vec_size; ++i)
            outs[i] = ins[i] + residual_1x1.outs[i];
#endif
    }

    // Buffer forward: process N samples
#if RTNEURAL_USE_EIGEN
    using Vec = Eigen::Matrix<T, channels, 1>;
    void forward(const Vec* ins, Vec* skip_sums, Vec* layer_outs, int N, wavenet::Memory_Arena<>& arena) noexcept
    {
        const auto _ = arena.create_frame();
        auto* temp = arena.template allocate<Vec>(N, RTNEURAL_DEFAULT_ALIGNMENT);

        for (int n = 0; n < N; ++n) { conv.forward(ins[n]); temp[n] = conv.outs; }
        for (int n = 0; n < N; ++n) { activation.forward(temp[n]); temp[n] = activation.outs; }
        for (int n = 0; n < N; ++n) { skip_1x1.forward(temp[n]); skip_sums[n] += skip_1x1.outs; }
        for (int n = 0; n < N; ++n)
        {
            residual_1x1.forward(temp[n]);
            layer_outs[n] = ins[n] + residual_1x1.outs;
        }
    }
#elif RTNEURAL_USE_XSIMD
    using Vec = xsimd::batch<T>[vec_size];
    void forward(const Vec* ins, Vec* skip_sums, Vec* layer_outs, int N, wavenet::Memory_Arena<>& arena) noexcept
    {
        const auto _ = arena.create_frame();
        auto* temp = arena.template allocate<Vec>(N, RTNEURAL_DEFAULT_ALIGNMENT);

        for (int n = 0; n < N; ++n) { conv.forward(ins[n]); for (int i = 0; i < vec_size; ++i) temp[n][i] = conv.outs[i]; }
        for (int n = 0; n < N; ++n) { activation.forward(temp[n]); for (int i = 0; i < vec_size; ++i) temp[n][i] = activation.outs[i]; }
        for (int n = 0; n < N; ++n) { skip_1x1.forward(temp[n]); for (int i = 0; i < vec_size; ++i) skip_sums[n][i] += skip_1x1.outs[i]; }
        for (int n = 0; n < N; ++n)
        {
            residual_1x1.forward(temp[n]);
            for (int i = 0; i < vec_size; ++i) layer_outs[n][i] = ins[n][i] + residual_1x1.outs[i];
        }
    }
#endif
};

// WaveNet model: input_conv -> N x WaveNetLayer -> output_conv(skip_sum)
template <typename T, int channels, int kernel_size, typename... Layers>
struct WaveNetModel
{
    static constexpr int ch = channels;
    RTNeural::Conv1DT<T, 1, channels, 1, 1> input_conv;
    std::tuple<Layers...> layers;
    RTNeural::Conv1DT<T, channels, 1, 1, 1> output_conv;

#if RTNEURAL_USE_EIGEN
    Eigen::Matrix<T, channels, 1> skip_sum;
#elif RTNEURAL_USE_XSIMD
    static constexpr int vec_size = RTNeural::ceil_div(channels, (int)xsimd::batch<T>::size);
    xsimd::batch<T> skip_sum[vec_size];
#endif

    wavenet::Memory_Arena<> arena;

    void reset()
    {
        input_conv.reset();
        output_conv.reset();
        std::apply([](auto&... l) { (l.reset(), ...); }, layers);
    }

    void prepare(int block_size)
    {
#if RTNEURAL_USE_EIGEN
        using Vec = Eigen::Matrix<T, channels, 1>;
#elif RTNEURAL_USE_XSIMD
        using Vec = xsimd::batch<T>[vec_size];
#endif
        // skip_sums[N] + layer_io[N] + temp per layer + input conversion
        size_t bytes = sizeof(Vec) * block_size * (sizeof...(Layers) + 3);
#if RTNEURAL_USE_XSIMD
        bytes += sizeof(xsimd::batch<T>) * block_size;  // v_ins
#endif
        arena.resize(bytes + 256);
    }

    // Per-sample forward (unchanged)
    RTNEURAL_REALTIME T forward(const T* input) noexcept
    {
#if RTNEURAL_USE_EIGEN
        input_conv.forward(Eigen::Matrix<T, 1, 1>::Map(input));
        skip_sum.setZero();
#elif RTNEURAL_USE_XSIMD
        using v_type = xsimd::batch<T>;
        static constexpr int v_in = RTNeural::ceil_div(1, (int)v_type::size);
        v_type in_arr[v_in] { v_type(*input) };
        input_conv.forward(in_arr);
        for (int i = 0; i < vec_size; ++i)
            skip_sum[i] = v_type(T(0));
#endif
        forwardLayers(std::make_index_sequence<sizeof...(Layers)>{});
        output_conv.forward(skip_sum);
#if RTNEURAL_USE_EIGEN
        return output_conv.outs(0);
#elif RTNEURAL_USE_XSIMD
        return output_conv.outs[0].get(0);
#endif
    }

    // Buffer forward: process N samples with improved cache locality
    void forward(const T* input, T* output, int N) noexcept
    {
#if RTNEURAL_USE_EIGEN
        using Vec = Eigen::Matrix<T, channels, 1>;
        const auto* v_ins = reinterpret_cast<const Eigen::Matrix<T, 1, 1>*>(input);
#elif RTNEURAL_USE_XSIMD
        using v_type = xsimd::batch<T>;
        using Vec = xsimd::batch<T>[vec_size];
        auto* v_ins = arena.template allocate<xsimd::batch<T>[1]>(N, RTNEURAL_DEFAULT_ALIGNMENT);
        for (int n = 0; n < N; ++n)
            v_ins[n][0] = xsimd::batch<T>(input[n]);
#endif

        // Allocate per-sample arrays
        auto* layer_io = arena.template allocate<Vec>(N, RTNEURAL_DEFAULT_ALIGNMENT);
        auto* skip_sums = arena.template allocate<Vec>(N, RTNEURAL_DEFAULT_ALIGNMENT);

        // input_conv all N
        for (int n = 0; n < N; ++n)
        {
            input_conv.forward(v_ins[n]);
#if RTNEURAL_USE_EIGEN
            layer_io[n] = input_conv.outs;
            skip_sums[n].setZero();
#elif RTNEURAL_USE_XSIMD
            for (int i = 0; i < vec_size; ++i) { layer_io[n][i] = input_conv.outs[i]; skip_sums[n][i] = v_type(T(0)); }
#endif
        }

        // Each layer processes all N
        forwardLayersBuffer(layer_io, skip_sums, N, std::make_index_sequence<sizeof...(Layers)>{});

        // output_conv all N
        for (int n = 0; n < N; ++n)
        {
            output_conv.forward(skip_sums[n]);
#if RTNEURAL_USE_EIGEN
            output[n] = output_conv.outs(0);
#elif RTNEURAL_USE_XSIMD
            output[n] = output_conv.outs[0].get(0);
#endif
        }

        arena.clear();
    }

private:
    template <size_t... Is>
    RTNEURAL_REALTIME void forwardLayers(std::index_sequence<Is...>) noexcept
    {
        (forwardLayer<Is>(), ...);
    }

    template <size_t I>
    RTNEURAL_REALTIME void forwardLayer() noexcept
    {
        if constexpr (I == 0)
            std::get<0>(layers).forward(input_conv.outs, skip_sum);
        else
            std::get<I>(layers).forward(std::get<I - 1>(layers).outs, skip_sum);
    }

#if RTNEURAL_USE_EIGEN
    using LayerVec = Eigen::Matrix<T, channels, 1>;
#elif RTNEURAL_USE_XSIMD
    using LayerVec = xsimd::batch<T>[vec_size];
#endif

    template <size_t... Is>
    void forwardLayersBuffer(LayerVec* io, LayerVec* skips, int N, std::index_sequence<Is...>) noexcept
    {
        (forwardLayerBuffer<Is>(io, skips, N), ...);
    }

    template <size_t I>
    void forwardLayerBuffer(LayerVec* io, LayerVec* skips, int N) noexcept
    {
        std::get<I>(layers).forward(io, skips, io, N, arena);
    }
};

// Concrete WaveNet types for each size tier
using RTNeuralWaveNet_Small = WaveNetModel<float, 8, 2,
    WaveNetLayer<float, 8, 2, 1>,
    WaveNetLayer<float, 8, 2, 2>,
    WaveNetLayer<float, 8, 2, 4>>;

using RTNeuralWaveNet_Medium = WaveNetModel<float, 16, 2,
    WaveNetLayer<float, 16, 2, 1>,
    WaveNetLayer<float, 16, 2, 2>,
    WaveNetLayer<float, 16, 2, 4>,
    WaveNetLayer<float, 16, 2, 8>,
    WaveNetLayer<float, 16, 2, 16>,
    WaveNetLayer<float, 16, 2, 32>,
    WaveNetLayer<float, 16, 2, 64>,
    WaveNetLayer<float, 16, 2, 128>,
    WaveNetLayer<float, 16, 2, 256>,
    WaveNetLayer<float, 16, 2, 512>>;

using RTNeuralWaveNet_Large = WaveNetModel<float, 32, 2,
    WaveNetLayer<float, 32, 2, 1>,
    WaveNetLayer<float, 32, 2, 2>,
    WaveNetLayer<float, 32, 2, 4>,
    WaveNetLayer<float, 32, 2, 8>,
    WaveNetLayer<float, 32, 2, 16>,
    WaveNetLayer<float, 32, 2, 32>,
    WaveNetLayer<float, 32, 2, 64>,
    WaveNetLayer<float, 32, 2, 128>,
    WaveNetLayer<float, 32, 2, 256>,
    WaveNetLayer<float, 32, 2, 512>>;

// ---------------------------------------------------------------------------
// RTNeural engine — dispatches to the correct size/architecture at runtime
// ---------------------------------------------------------------------------
class RTNeuralEngine
{
public:
    RTNeuralEngine() = default;

    bool initialize(ModelType modelType, ModelSize modelSize, const std::string& weightsPath);
    float processSample(float input);
    void processBlock(const float* input, float* output, int numSamples);
    void resetState();
    bool isValid() const { return initialized; }

private:
    ModelType currentModel = ModelType::LSTM;
    ModelSize currentSize = ModelSize::Small;
    bool initialized = false;

    // Small
    RTNeuralLSTM_Small     lstmSmall;
    RTNeuralTCN_Small      tcnSmall;
    RTNeuralWaveNet_Small  wavenetSmall;

    // Medium
    RTNeuralLSTM_Medium    lstmMedium;
    RTNeuralTCN_Medium     tcnMedium;
    RTNeuralWaveNet_Medium wavenetMedium;

    // Large
    RTNeuralLSTM_Large     lstmLarge;
    RTNeuralTCN_Large      tcnLarge;
    RTNeuralWaveNet_Large  wavenetLarge;
};

// ---------------------------------------------------------------------------
// RTNeural backend adapter. name() reflects the compile-time backend
// (RTNeural_XSIMD under RTNEURAL_USE_XSIMD, else RTNeural_Eigen).
// ---------------------------------------------------------------------------
class RTNeuralBackend : public InferenceBackend
{
public:
#if defined(RTNEURAL_USE_XSIMD)
    static constexpr const char* kName = "RTNeural_XSIMD";
#else
    static constexpr const char* kName = "RTNeural_Eigen";
#endif

    bool prepare(const PrepareContext& ctx) override;
    void process(const float* in, float* out, int n) noexcept override { engine.processBlock(in, out, n); }
    void reset() noexcept override { engine.resetState(); }

    const char* name() const override { return kName; }
    bool isRealtimeSafe() const override { return true; }
    const char* requiredFormat() const override { return "rtneural"; }
    bool supports(const ModelSpec& spec, std::string& whyNot) const override;

private:
    RTNeuralEngine engine;
    static bool mapArchSize(const ModelSpec& spec, ModelType& m, ModelSize& s);
};
