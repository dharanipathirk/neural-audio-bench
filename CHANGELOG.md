# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Planned 1.0.0 public release, accompanying the DAFx-26 paper *"Real-Time Neural
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

### Changed
- The CoreML exporter sets fp16 compute precision explicitly (CoreML state
  tensors must be fp16, so stateful exports cannot be fp32), and every
  manifest entry's `notes` now records that BNNSGraph runs in half precision
  while the other backends run fp32.
- RTNeural refuses (with a `skipped` results row) manifest entries whose
  hyperparameters or parameter count differ from its compiled topologies,
  instead of running the compiled network under the manifest's label.

### Fixed
- `nab` no longer accepts abbreviated options (`allow_abbrev=False`), so
  `nab export --manifest …` is an error instead of silently expanding to
  `--manifest-out` and overwriting the file.
- Paper figures rendered without `--pgf` printed literal `\%` and
  `\texttt{}` in axis labels and titles.
- The C++ unit tests were compiled with `NDEBUG`, which disabled every
  assertion.
- `src/backends/arena.hpp` (RTNeural-NAM, BSD-3-Clause) carries its original
  copyright notice; the licence table lists JUCE 8 as AGPLv3 and adds the
  remaining linked and Python dependencies.
