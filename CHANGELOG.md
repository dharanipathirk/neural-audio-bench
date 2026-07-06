# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-05

First public release, accompanying the DAFx-26 paper *"Real-Time Neural
Audio on Apple Silicon: Benchmarking Inference Frameworks Under Realistic
DAW Contention"*.

### Added
- Isolated benchmark: throughput and per-callback inference timing for
  BNNSGraph, RTNeural (Eigen/XSIMD), LibTorch, and ONNX Runtime.
- Contention benchmark on Tracktion Engine + BlackHole: realistic mix
  sessions (Dimension A), neural instance scaling (Dimension B), and serial
  insert-chain depth (Dimension C), with hardware xrun and anira
  inference-underrun accounting.
- Pluggable backend registry (`InferenceBackend`), model manifest
  (bring-your-own-model), and scenario abstraction.
- `nab` CLI: model export (CoreML/ONNX/TorchScript/RTNeural), thermally
  controlled run orchestration, runtime estimation, analysis, and
  figure/report generation.
- Pinned, checksummed dependency fetching (no manual setup); CMake presets.
- `experiments/dafx26-paper/`: frozen configuration, model manifest, and
  reference results reproducing the paper.
