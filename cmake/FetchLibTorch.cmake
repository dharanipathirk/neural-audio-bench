# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#
# Fetches the prebuilt LibTorch (CPU, macOS arm64) archive and exposes it as
# NAB_LIBTORCH_ROOT. The archive's single top-level directory is stripped by
# FetchContent, so NAB_LIBTORCH_ROOT contains lib/, include/ and share/.

include(FetchContent)

FetchContent_Declare(nab_libtorch
    URL      ${NAB_LIBTORCH_URL}
    URL_HASH SHA256=${NAB_LIBTORCH_SHA256}
)
FetchContent_MakeAvailable(nab_libtorch)

set(NAB_LIBTORCH_ROOT "${nab_libtorch_SOURCE_DIR}")
