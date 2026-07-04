# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#
# Fetches every dependency at the pinned versions from Versions.cmake.
# No sibling directories or preinstalled libraries are required.
#
# Local development override (uses an existing checkout instead of fetching):
#   cmake -B build -DFETCHCONTENT_SOURCE_DIR_JUCE=/path/to/JUCE ...
# (same pattern for TRACKTION_ENGINE, RTNEURAL, ANIRA — note that overriding
#  bypasses the version pins, so results may not match the published baseline.)

include(FetchContent)

# --- JUCE ------------------------------------------------------------------
FetchContent_Declare(juce
    GIT_REPOSITORY ${NAB_JUCE_GIT_REPO}
    GIT_TAG        ${NAB_JUCE_GIT_TAG}
    EXCLUDE_FROM_ALL
)

# --- Tracktion Engine ------------------------------------------------------
# GIT_SUBMODULES is empty: Tracktion only uses its bundled JUCE submodule
# when no juce::juce_core target exists, and ours is added first.
# The patch fixes a JUCE 8.0.12 API rename (userBounds -> userArea) in
# window-positioning code that never runs in the headless benchmark.
FetchContent_Declare(tracktion_engine
    GIT_REPOSITORY ${NAB_TRACKTION_GIT_REPO}
    GIT_TAG        ${NAB_TRACKTION_GIT_TAG}
    GIT_SUBMODULES ""
    PATCH_COMMAND  git apply --ignore-whitespace
                   "${CMAKE_CURRENT_SOURCE_DIR}/patches/tracktion-juce8-compile-fix.patch"
    UPDATE_DISCONNECTED ON
    EXCLUDE_FROM_ALL
)

# --- RTNeural (header-only; populated but NOT added as a subproject) --------
# The benchmark consumes RTNeural and its bundled Eigen/json/xsimd purely as
# include directories with our own compile definitions, exactly as the paper
# build did. SOURCE_SUBDIR points at a directory with no CMakeLists.txt so
# FetchContent_MakeAvailable populates the source without configuring it.
FetchContent_Declare(rtneural
    GIT_REPOSITORY ${NAB_RTNEURAL_GIT_REPO}
    GIT_TAG        ${NAB_RTNEURAL_GIT_TAG}
    SOURCE_SUBDIR  .nab-populate-only
)

# --- anira -----------------------------------------------------------------
# Populated without configuring (SOURCE_SUBDIR has no CMakeLists.txt) so the
# LibTorch/ORT symlinks below can be placed before anira's CMake runs.
FetchContent_Declare(anira
    GIT_REPOSITORY ${NAB_ANIRA_GIT_REPO}
    GIT_TAG        ${NAB_ANIRA_GIT_TAG}
    SOURCE_SUBDIR  .nab-populate-only
)

FetchContent_MakeAvailable(juce tracktion_engine rtneural anira)

set(NAB_RTNEURAL_ROOT "${rtneural_SOURCE_DIR}")

# --- Prebuilt LibTorch / ONNX Runtime ---------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/FetchLibTorch.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/FetchOnnxRuntime.cmake)

# --- anira (configured last so it reuses the fetched LibTorch/ORT) ----------
option(BUILD_WITH_ANIRA "Build the anira InferenceHandler backends (Anira_LibTorch/Anira_ONNX)" ON)

if(BUILD_WITH_ANIRA)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_INSTALL OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_TESTS OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_DOCS OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_TFLITE OFF CACHE BOOL "" FORCE)
    set(ANIRA_WITH_LOGGING ON CACHE BOOL "" FORCE)

    # Symlink our checksummed LibTorch/ORT into the modules/ layout anira's
    # Setup*.cmake scripts expect, so anira skips its own (unchecksummed)
    # downloads and every backend links the same bytes.
    file(MAKE_DIRECTORY "${anira_SOURCE_DIR}/modules")
    set(_anira_libtorch "${anira_SOURCE_DIR}/modules/libtorch-${NAB_LIBTORCH_VERSION}-macOS-${CMAKE_SYSTEM_PROCESSOR}")
    set(_anira_ort      "${anira_SOURCE_DIR}/modules/onnxruntime-${NAB_ORT_VERSION}-macOS-${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT EXISTS "${_anira_libtorch}")
        file(CREATE_LINK "${NAB_LIBTORCH_ROOT}" "${_anira_libtorch}" SYMBOLIC)
    endif()
    if(NOT EXISTS "${_anira_ort}")
        file(CREATE_LINK "${NAB_ORT_ROOT}" "${_anira_ort}" SYMBOLIC)
    endif()

    add_subdirectory("${anira_SOURCE_DIR}" "${anira_BINARY_DIR}" EXCLUDE_FROM_ALL)
    message(STATUS "anira enabled: ${anira_SOURCE_DIR}")
endif()
