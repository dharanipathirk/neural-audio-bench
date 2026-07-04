# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
cmake_minimum_required(VERSION 3.20)
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
# Paths to local dependencies (sibling directories)
# ---------------------------------------------------------------------------
set(DAFX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(JUCE_ROOT "${DAFX_ROOT}/JUCE")
set(TRACKTION_ROOT "${DAFX_ROOT}/tracktion_engine")
set(RTNEURAL_ROOT "${DAFX_ROOT}/RTNeural")
set(ANIRA_ROOT "${DAFX_ROOT}/anira")
set(ANIRA_MODULES "${ANIRA_ROOT}/modules")
set(LIBTORCH_ROOT "${ANIRA_MODULES}/libtorch-2.4.1-macOS-arm64")
set(ORT_ROOT "${ANIRA_MODULES}/onnxruntime-1.19.2-macOS-arm64")
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")

# ---------------------------------------------------------------------------
# JUCE
# ---------------------------------------------------------------------------
add_subdirectory(${JUCE_ROOT} ${CMAKE_BINARY_DIR}/juce EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Tracktion Engine
# ---------------------------------------------------------------------------
add_subdirectory(${TRACKTION_ROOT} ${CMAKE_BINARY_DIR}/tracktion EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Common sources
# ---------------------------------------------------------------------------
set(COMMON_HEADERS
    src/BenchmarkConfig.h
    src/TimingLogger.h
)

set(PLUGIN_SOURCES
    src/plugins/BNNSGraphPlugin.h
    src/plugins/BNNSGraphPlugin.mm
    src/plugins/RTNeuralPlugin.h
    src/plugins/RTNeuralPlugin.cpp
    src/plugins/AniraPlugin.h
    src/plugins/AniraPlugin.cpp
    src/plugins/AniraHandlerPlugin.h
    src/plugins/AniraHandlerPlugin.cpp
    src/plugins/ContentionPlugins.h
    src/plugins/ContentionPlugins.cpp
)

set(ISOLATED_SOURCES
    src/isolated/IsolatedBenchmark.h
    src/isolated/IsolatedBenchmark.cpp
)

set(CONTENTION_SOURCES
    src/contention/EditBuilder.h
    src/contention/EditBuilder.cpp
    src/contention/TracktionPlaybackTest.h
    src/contention/TracktionPlaybackTest.cpp
    src/contention/AUSessionBuilder.h
    src/contention/AUSessionBuilder.cpp
)

# ---------------------------------------------------------------------------
# Main benchmark executable
# ---------------------------------------------------------------------------
add_executable(nab_engine
    src/main.cpp
    ${COMMON_HEADERS}
    ${PLUGIN_SOURCES}
    ${ISOLATED_SOURCES}
    ${CONTENTION_SOURCES}
)
set_target_properties(nab_engine PROPERTIES OUTPUT_NAME "nab-engine")

target_include_directories(nab_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RTNEURAL_ROOT}
    ${RTNEURAL_ROOT}/modules/Eigen
    ${RTNEURAL_ROOT}/modules/json
    ${RTNEURAL_ROOT}/modules/xsimd/include
)

target_compile_definitions(nab_engine PRIVATE
    JUCE_STANDALONE_APPLICATION=1
    JUCE_USE_CURL=0
    JUCE_WEB_BROWSER=0
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_PLUGINHOST_AU=1
    RTNEURAL_USE_EIGEN=1
    RTNEURAL_DEFAULT_ALIGNMENT=16
    EIGEN_STACK_ALLOCATION_LIMIT=524288
    MODEL_DIR="${MODEL_DIR}"
)

target_link_libraries(nab_engine PRIVATE
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

# Compile .mm files as Objective-C++
set_source_files_properties(
    src/plugins/BNNSGraphPlugin.mm
    PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
)

# ---------------------------------------------------------------------------
# LibTorch backend (optional)
# ---------------------------------------------------------------------------
if(EXISTS "${LIBTORCH_ROOT}/lib/libtorch_cpu.dylib")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}/share/cmake/Torch")
    find_package(Torch QUIET)
    if(Torch_FOUND)
        target_link_libraries(nab_engine PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(nab_engine PRIVATE ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(nab_engine PRIVATE HAS_LIBTORCH=1)
        message(STATUS "LibTorch enabled: ${LIBTORCH_ROOT}")
    else()
        message(STATUS "LibTorch CMake config not found, skipping")
    endif()
else()
    message(STATUS "LibTorch not found at ${LIBTORCH_ROOT}, skipping")
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime backend (optional)
# ---------------------------------------------------------------------------
if(EXISTS "${ORT_ROOT}/lib/libonnxruntime.dylib")
    target_include_directories(nab_engine PRIVATE "${ORT_ROOT}/include")
    target_link_directories(nab_engine PRIVATE "${ORT_ROOT}/lib")
    target_link_libraries(nab_engine PRIVATE onnxruntime)
    target_compile_definitions(nab_engine PRIVATE HAS_ONNXRUNTIME=1)
    message(STATUS "ONNX Runtime enabled: ${ORT_ROOT}")
else()
    message(STATUS "ONNX Runtime not found at ${ORT_ROOT}, skipping")
endif()

# ---------------------------------------------------------------------------
# anira InferenceHandler (optional — real anira scheduling layer)
# ---------------------------------------------------------------------------
option(BUILD_WITH_ANIRA "Build with anira InferenceHandler for Anira_LibTorch/Anira_ONNX backends" OFF)

if(BUILD_WITH_ANIRA AND EXISTS "${ANIRA_ROOT}/CMakeLists.txt")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_INSTALL OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_TESTS OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_DOCS OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_TFLITE OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_LOGGING ON CACHE BOOL "" FORCE)

    add_subdirectory(${ANIRA_ROOT} ${CMAKE_BINARY_DIR}/anira EXCLUDE_FROM_ALL)

    target_link_libraries(nab_engine PRIVATE anira::anira)
    target_compile_definitions(nab_engine PRIVATE HAS_ANIRA=1)
    message(STATUS "anira InferenceHandler enabled: ${ANIRA_ROOT}")
elseif(BUILD_WITH_ANIRA)
    message(WARNING "BUILD_WITH_ANIRA=ON but anira not found at ${ANIRA_ROOT}")
endif()

# ---------------------------------------------------------------------------
# RTNeural XSIMD build variant (optional second executable)
# ---------------------------------------------------------------------------
option(BUILD_XSIMD_VARIANT "Build nab_engine_xsimd with RTNeural XSIMD backend" OFF)

if(BUILD_XSIMD_VARIANT)
    add_executable(nab_engine_xsimd
        src/main.cpp
        ${COMMON_HEADERS}
        ${PLUGIN_SOURCES}
        ${ISOLATED_SOURCES}
        ${CONTENTION_SOURCES}
    )
    set_target_properties(nab_engine_xsimd PROPERTIES OUTPUT_NAME "nab-engine-xsimd")

    target_include_directories(nab_engine_xsimd PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${RTNEURAL_ROOT}
        ${RTNEURAL_ROOT}/modules/Eigen
        ${RTNEURAL_ROOT}/modules/json
        ${RTNEURAL_ROOT}/modules/xsimd/include
    )

    target_compile_definitions(nab_engine_xsimd PRIVATE
        JUCE_STANDALONE_APPLICATION=1
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0
        JUCE_MODAL_LOOPS_PERMITTED=1
        JUCE_PLUGINHOST_AU=1
        RTNEURAL_USE_XSIMD=1
        RTNEURAL_DEFAULT_ALIGNMENT=16
        MODEL_DIR="${MODEL_DIR}"
    )

    target_link_libraries(nab_engine_xsimd PRIVATE
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

    set_source_files_properties(
        src/plugins/BNNSGraphPlugin.mm
        PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    )

    # Optionally link LibTorch and ONNX Runtime if available
    if(Torch_FOUND)
        target_link_libraries(nab_engine_xsimd PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(nab_engine_xsimd PRIVATE ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(nab_engine_xsimd PRIVATE HAS_LIBTORCH=1)
    endif()

    if(EXISTS "${ORT_ROOT}/lib/libonnxruntime.dylib")
        target_include_directories(nab_engine_xsimd PRIVATE "${ORT_ROOT}/include")
        target_link_directories(nab_engine_xsimd PRIVATE "${ORT_ROOT}/lib")
        target_link_libraries(nab_engine_xsimd PRIVATE onnxruntime)
        target_compile_definitions(nab_engine_xsimd PRIVATE HAS_ONNXRUNTIME=1)
    endif()

    if(BUILD_WITH_ANIRA AND TARGET anira::anira)
        target_link_libraries(nab_engine_xsimd PRIVATE anira::anira)
        target_compile_definitions(nab_engine_xsimd PRIVATE HAS_ANIRA=1)
    endif()

    message(STATUS "Building XSIMD variant: nab_engine_xsimd")
endif()

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
