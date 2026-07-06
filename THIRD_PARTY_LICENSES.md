# Third-Party Licenses

neural-audio-bench is licensed under GPL-3.0-or-later (see [LICENSE](LICENSE)).
It does not vendor third-party code; all dependencies are fetched at build
time at the pinned versions in `cmake/Versions.cmake`. When you build and/or
redistribute binaries, the licenses below apply to the linked components.

| Component | License | Notes |
|---|---|---|
| [JUCE](https://github.com/juce-framework/JUCE) | GPLv3 (or commercial) | Framework for audio infrastructure. GPLv3 tier applies to this open-source build; a commercial JUCE license is required to redistribute non-GPL derivatives. |
| [Tracktion Engine](https://github.com/Tracktion/tracktion_engine) | GPLv3 (or commercial) | DAW session hosting for the contention benchmark. Same dual-licensing model as JUCE. |
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | BSD-3-Clause | Real-time neural inference library (Eigen/XSIMD backends). |
| [anira](https://github.com/anira-project/anira) | Apache-2.0 | Background-thread inference scheduling. |
| [LibTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | PyTorch C++ runtime (prebuilt arm64 archive from [faressc/libtorch-cpp-lib](https://github.com/faressc/libtorch-cpp-lib), the same archive anira uses). |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Official prebuilt macOS archive. |
| [Eigen](https://eigen.tuxfamily.org) | MPL-2.0 | Bundled with RTNeural. |
| [xsimd](https://github.com/xtensor-stack/xsimd) | BSD-3-Clause | Bundled with RTNeural. |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | Bundled with RTNeural; also fetched by anira. |
| BNNS / Accelerate, Core Audio, CoreML | Apple SLA | macOS system frameworks; not distributed with this project. |
| [BlackHole](https://github.com/ExistentialAudio/BlackHole) | GPLv3 | Virtual audio driver, installed separately by the user (runtime requirement for contention mode only; not linked). |

Python dependencies (torch, coremltools, onnx, pandas, matplotlib, seaborn,
numpy) are installed from PyPI under their respective licenses (BSD/MIT/
Apache-2.0 family) and are listed with pinned versions in `uv.lock`.
