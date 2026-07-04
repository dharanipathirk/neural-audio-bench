# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#
# Single source of truth for every third-party dependency pin.
# These exact versions produced the DAFx-26 paper results (see
# experiments/dafx26-paper/). Bumping any pin requires re-baselining
# the golden benchmark numbers.

# --- Source dependencies (pinned to full commit SHAs) ----------------------
set(NAB_JUCE_GIT_REPO      https://github.com/juce-framework/JUCE.git)
set(NAB_JUCE_GIT_TAG       501c07674e1ad693085a7e7c398f205c2677f5da) # 8.0.12 + 4

set(NAB_TRACKTION_GIT_REPO https://github.com/Tracktion/tracktion_engine.git)
set(NAB_TRACKTION_GIT_TAG  2877b621f2fbee564d0696a616b86bf8ba8c8ab0) # v3.2.0 + 72
# Requires patches/tracktion-juce8-compile-fix.patch to compile against JUCE 8.0.12.

set(NAB_RTNEURAL_GIT_REPO  https://github.com/jatinchowdhury18/RTNeural.git)
set(NAB_RTNEURAL_GIT_TAG   1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d)
# RTNeural's xsimd submodule is pinned by this commit (a00c81f7).

set(NAB_ANIRA_GIT_REPO     https://github.com/anira-project/anira.git)
set(NAB_ANIRA_GIT_TAG      48e6b2aad3c54eaf8de735ba4b0caae451f1dbc1) # v2.0.3, Apache-2.0

# --- Prebuilt binary dependencies (pinned by SHA256) -----------------------
# Same archives anira v2.0.3 fetches on macOS arm64; fetched here explicitly
# so the direct LibTorch/ONNX backends work without anira, and so the
# checksums are enforced.
set(NAB_LIBTORCH_VERSION   2.4.1)
set(NAB_LIBTORCH_URL
    https://github.com/faressc/libtorch-cpp-lib/releases/download/v2.4.1/libtorch-2.4.1-macOS-arm64.zip)
set(NAB_LIBTORCH_SHA256    d92e872221dc69a78b4dc7798715f7c0ab4462b082d143f6f6355a7d1e44146d)

set(NAB_ORT_VERSION        1.19.2)
set(NAB_ORT_URL
    https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-universal2-1.19.2.tgz)
set(NAB_ORT_SHA256         b0289ddbc32f76e5d385abc7b74cc7c2c51cdf2285b7d118bf9d71206e5aee3a)
