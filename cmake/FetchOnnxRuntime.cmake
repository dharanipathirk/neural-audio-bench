# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#
# Fetches the official prebuilt ONNX Runtime (macOS universal2) archive and
# exposes it as NAB_ORT_ROOT (containing lib/ and include/).

include(FetchContent)

FetchContent_Declare(nab_onnxruntime
    URL      ${NAB_ORT_URL}
    URL_HASH SHA256=${NAB_ORT_SHA256}
)
FetchContent_MakeAvailable(nab_onnxruntime)

set(NAB_ORT_ROOT "${nab_onnxruntime_SOURCE_DIR}")
