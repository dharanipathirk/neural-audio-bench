# Third-Party Licenses

neural-audio-bench is licensed under GPL-3.0-or-later (see [LICENSE](LICENSE)).

One third-party source file is included in this repository:
`src/backends/arena.hpp`, the memory arena from
[RTNeural-NAM](https://github.com/jatinchowdhury18/RTNeural-NAM)
(`wavenet/arena.hpp`, BSD-3-Clause, Copyright (c) 2024 jatinchowdhury18),
reproduced unmodified with its licence text in the file header. The TCN and
WaveNet buffer-processing structs in `src/backends/RTNeuralBackend.h` follow
the pattern of RTNeural-NAM's `wavenet_layer.hpp` but are original code.

Every other dependency is fetched at build time at the pinned versions in
`cmake/Versions.cmake`. When you build and/or redistribute binaries, the
licenses below apply to the linked components.

| Component | License | Notes |
|---|---|---|
| [JUCE](https://github.com/juce-framework/JUCE) | AGPLv3 (or commercial) | Framework for audio infrastructure. JUCE 8 modules are dual-licensed under the AGPLv3 and the commercial JUCE licence. The GPLv3 permits combining with AGPLv3 code (GPLv3 section 13); the AGPL's network-use terms apply to the JUCE portion of any derived work, and a commercial JUCE licence is required to redistribute non-(A)GPL derivatives. |
| [Tracktion Engine](https://github.com/Tracktion/tracktion_engine) | GPLv3 (or commercial) | DAW session hosting for the contention benchmark. GPLv3 tier applies to this open-source build; a commercial licence is required for non-GPL derivatives. |
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | BSD-3-Clause | Real-time neural inference library (Eigen/XSIMD backends). |
| [RTNeural-NAM](https://github.com/jatinchowdhury18/RTNeural-NAM) | BSD-3-Clause | Source of `src/backends/arena.hpp` (see above). |
| [anira](https://github.com/anira-project/anira) | Apache-2.0 | Background-thread inference scheduling. |
| [moodycamel concurrentqueue](https://github.com/cameron314/concurrentqueue) | Simplified BSD (BSD-2-Clause) | Lock-free queue fetched and linked by anira. |
| [LibTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | PyTorch C++ runtime (prebuilt arm64 archive from [faressc/libtorch-cpp-lib](https://github.com/faressc/libtorch-cpp-lib), the same archive anira uses). |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Official prebuilt macOS archive. |
| [Eigen](https://eigen.tuxfamily.org) | MPL-2.0 | Bundled with RTNeural. |
| [xsimd](https://github.com/xtensor-stack/xsimd) | BSD-3-Clause | Bundled with RTNeural. |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | Bundled with RTNeural; also fetched by anira. |
| BNNS / Accelerate, Core Audio, CoreML | Apple SLA | macOS system frameworks; not distributed with this project. |
| [BlackHole](https://github.com/ExistentialAudio/BlackHole) | GPLv3 | Virtual audio driver, installed separately by the user (runtime requirement for contention mode only; not linked). |

Python dependencies (torch, coremltools, onnx, onnxscript, jsonschema, pandas,
matplotlib, seaborn, numpy) are installed from PyPI under their respective
licenses (BSD/MIT/Apache-2.0/PSF family) and are listed with pinned versions
in `uv.lock`.
