# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
cmake_minimum_required(VERSION 3.28)
project(neural_audio_bench C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")

# ---------------------------------------------------------------------------
# Benchmarks MUST be built in Release mode to produce valid timing results.
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set — defaulting to Release")
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
        "Benchmark must be built in Release mode for valid timing results. "
        "Got CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "Use: cmake -DCMAKE_BUILD_TYPE=Release ..")
endif()

# ---------------------------------------------------------------------------
# Dependencies (pinned; see cmake/Versions.cmake)
# ---------------------------------------------------------------------------
include(cmake/Versions.cmake)
include(cmake/Dependencies.cmake)

set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(COMMON_HEADERS
    src/BenchmarkConfig.h
    src/TimingLogger.h
)

# Backend abstraction layer: verbatim inference engines + adapters + registry
set(BACKEND_SOURCES
    src/backends/InferenceBackend.h
    src/backends/InferenceBackend.cpp
    src/backends/BackendRegistry.h
    src/backends/BackendRegistry.cpp
    src/backends/backend_registration.cpp
    src/backends/arena.hpp
    src/backends/BnnsGraphBackend.h
    src/backends/BnnsGraphBackend.mm
    src/backends/RTNeuralBackend.h
    src/backends/RTNeuralBackend.cpp
    src/backends/RTNeuralTopology.h
    src/backends/RTNeuralTopology.cpp
    src/backends/LibTorchBackend.h
    src/backends/LibTorchBackend.cpp
    src/backends/OnnxRuntimeBackend.h
    src/backends/OnnxRuntimeBackend.cpp
    src/backends/AniraBackend.h
    src/backends/AniraBackend.cpp
)

# Unified host plugin
set(HOST_SOURCES
    src/host/NeuralInferencePlugin.h
    src/host/NeuralInferencePlugin.cpp
)

# Model manifest / catalog
set(CORE_SOURCES
    src/core/ModelManifest.h
    src/core/ModelManifest.cpp
)

set(PLUGIN_SOURCES
    src/plugins/ContentionPlugins.h
    src/plugins/ContentionPlugins.cpp
)

set(RUNNER_SOURCES
    src/runners/IsolatedRunner.h
    src/runners/IsolatedRunner.cpp
    src/runners/ContentionRunner.h
    src/runners/ContentionRunner.cpp
)

set(SCENARIO_SOURCES
    src/scenarios/Scenario.h
    src/scenarios/ScenarioRegistry.h
    src/scenarios/ScenarioRegistry.cpp
    src/scenarios/MixContentionScenario.h
    src/scenarios/MixContentionScenario.cpp
    src/scenarios/InstanceCountScenario.h
    src/scenarios/InstanceCountScenario.cpp
    src/scenarios/SerialDepthScenario.h
    src/scenarios/SerialDepthScenario.cpp
    src/scenarios/ConfigurableScenario.h
    src/scenarios/ConfigurableScenario.cpp
    src/scenarios/SystemAuSessionBuilder.h
    src/scenarios/SystemAuSessionBuilder.cpp
)

set(CONTENTION_SOURCES
    src/contention/EditBuilder.h
    src/contention/EditBuilder.cpp
    src/contention/TracktionPlaybackTest.h
    src/contention/TracktionPlaybackTest.cpp
)

# ---------------------------------------------------------------------------
# Benchmark target factory
# The Eigen and XSIMD RTNeural backends require separate executables because
# the backend is selected by compile definition. Everything else is identical.
# Extra per-variant compile definitions are passed after the output name.
# ---------------------------------------------------------------------------
function(nab_add_benchmark_target TARGET_NAME BINARY_NAME)
    add_executable(${TARGET_NAME}
        src/main.cpp
        ${COMMON_HEADERS}
        ${BACKEND_SOURCES}
        ${HOST_SOURCES}
        ${CORE_SOURCES}
        ${PLUGIN_SOURCES}
        ${RUNNER_SOURCES}
        ${SCENARIO_SOURCES}
        ${CONTENTION_SOURCES}
    )
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${BINARY_NAME}")

    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${NAB_RTNEURAL_ROOT}
        ${NAB_RTNEURAL_ROOT}/modules/Eigen
        ${NAB_RTNEURAL_ROOT}/modules/json
        ${NAB_RTNEURAL_ROOT}/modules/xsimd/include
    )

    target_compile_definitions(${TARGET_NAME} PRIVATE
        JUCE_STANDALONE_APPLICATION=1
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0
        JUCE_MODAL_LOOPS_PERMITTED=1
        JUCE_PLUGINHOST_AU=1
        RTNEURAL_DEFAULT_ALIGNMENT=16
        MODEL_DIR="${MODEL_DIR}"
        ${ARGN}
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        tracktion::tracktion_engine
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_events
        "-framework Accelerate"
        "-framework Foundation"
        "-framework CoreAudio"
        "-framework AudioToolbox"
    )

    # LibTorch (direct backend)
    if(Torch_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(${TARGET_NAME} PRIVATE HAS_LIBTORCH=1)
    endif()

    # ONNX Runtime (direct backend)
    if(EXISTS "${NAB_ORT_ROOT}/lib/libonnxruntime.dylib")
        target_include_directories(${TARGET_NAME} PRIVATE "${NAB_ORT_ROOT}/include")
        target_link_directories(${TARGET_NAME} PRIVATE "${NAB_ORT_ROOT}/lib")
        target_link_libraries(${TARGET_NAME} PRIVATE onnxruntime)
        target_compile_definitions(${TARGET_NAME} PRIVATE HAS_ONNXRUNTIME=1)
    endif()

    # anira InferenceHandler backends
    if(TARGET anira::anira)
        target_link_libraries(${TARGET_NAME} PRIVATE anira::anira)
        target_compile_definitions(${TARGET_NAME} PRIVATE HAS_ANIRA=1)
    endif()

    set_source_files_properties(
        src/backends/BnnsGraphBackend.mm
        PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    )
endfunction()

# ---------------------------------------------------------------------------
# LibTorch discovery (once, shared by all targets)
# ---------------------------------------------------------------------------
if(EXISTS "${NAB_LIBTORCH_ROOT}/lib/libtorch_cpu.dylib")
    list(APPEND CMAKE_PREFIX_PATH "${NAB_LIBTORCH_ROOT}/share/cmake/Torch")
    find_package(Torch QUIET)
    if(Torch_FOUND)
        message(STATUS "LibTorch enabled: ${NAB_LIBTORCH_ROOT}")
    else()
        message(STATUS "LibTorch CMake config not found, direct LibTorch backend disabled")
    endif()
endif()
if(EXISTS "${NAB_ORT_ROOT}/lib/libonnxruntime.dylib")
    message(STATUS "ONNX Runtime enabled: ${NAB_ORT_ROOT}")
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
nab_add_benchmark_target(nab_engine nab-engine
    RTNEURAL_USE_EIGEN=1
    EIGEN_STACK_ALLOCATION_LIMIT=524288
)

option(BUILD_XSIMD_VARIANT "Build nab_engine_xsimd with the RTNeural XSIMD backend" ON)
if(BUILD_XSIMD_VARIANT)
    nab_add_benchmark_target(nab_engine_xsimd nab-engine-xsimd
        RTNEURAL_USE_XSIMD=1
    )
    message(STATUS "Building XSIMD variant: nab-engine-xsimd")
endif()

# ---------------------------------------------------------------------------
# Unit tests (pure logic — no audio device, models, or heavy deps required)
# ---------------------------------------------------------------------------
enable_testing()

add_executable(nab_test_stats tests/cpp/test_stats.cpp)
target_include_directories(nab_test_stats PRIVATE src)
add_test(NAME stats COMMAND nab_test_stats)

add_executable(nab_test_config
    tests/cpp/test_config_manifest.cpp
    src/core/ModelManifest.cpp
    src/backends/InferenceBackend.cpp
    src/backends/RTNeuralTopology.cpp
)
target_include_directories(nab_test_config PRIVATE
    src
    ${NAB_RTNEURAL_ROOT}/modules/json
)

add_test(NAME config-base
    COMMAND nab_test_config load-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/base.json)
add_test(NAME config-smoke
    COMMAND nab_test_config load-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/smoke.json)
add_test(NAME config-default-values
    COMMAND nab_test_config check-defaults ${CMAKE_CURRENT_SOURCE_DIR}/configs/base.json)
add_test(NAME manifest-fixture
    COMMAND nab_test_config load-manifest ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/models.manifest.json)
add_test(NAME config-incomplete-rejected
    COMMAND nab_test_config load-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/data/incomplete-config.json)
set_tests_properties(config-incomplete-rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME config-missing-rejected
    COMMAND nab_test_config load-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/data/does-not-exist.json)
set_tests_properties(config-missing-rejected PROPERTIES WILL_FAIL TRUE)

# ---------------------------------------------------------------------------
# CoreML model compilation (custom command)
# Directory structure: models/{arch}/{size}/stateful_{arch}_{size}.mlpackage
# ---------------------------------------------------------------------------
set(MODEL_ARCHS lstm tcn wavenet)
set(MODEL_SIZES small medium large)

foreach(ARCH ${MODEL_ARCHS})
    foreach(SIZE ${MODEL_SIZES})
        set(MODEL_NAME "stateful_${ARCH}_${SIZE}")
        set(PKG "${MODEL_DIR}/${ARCH}/${SIZE}/${MODEL_NAME}.mlpackage")
        set(OUT "${MODEL_DIR}/${ARCH}/${SIZE}/${MODEL_NAME}.mlmodelc")
        if(EXISTS ${PKG} AND NOT EXISTS ${OUT})
            add_custom_command(
                OUTPUT "${OUT}"
                COMMAND xcrun coremlcompiler compile "${PKG}" "${MODEL_DIR}/${ARCH}/${SIZE}"
                DEPENDS "${PKG}"
                COMMENT "Compiling ${MODEL_NAME}.mlpackage -> .mlmodelc"
            )
            add_custom_target(compile_${MODEL_NAME} DEPENDS "${OUT}")
            add_dependencies(nab_engine compile_${MODEL_NAME})
        endif()
    endforeach()
endforeach()
