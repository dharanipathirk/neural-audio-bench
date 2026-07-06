# Architecture

neural-audio-bench separates *what is measured* (backends × models ×
scenarios) from *how it is measured* (runners + timing core). Each of the
three axes is pluggable through a registry, so an extension is a localized
change that automatically appears in config, CLI, and results.

```
┌────────────────────────── Python (nab CLI) ──────────────────────────┐
│ export        run             estimate   analyze   plot   report     │
│ models +      config layering,                                       │
│ manifest      thermal protocol, run manifests                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ resolved config JSON
┌──────────────────────────────▼──────────────────── C++ (nab-engine) ─┐
│  core/        Config (fail-fast), ModelManifest, TimingLogger,       │
│               Stats, CSV writers                                     │
│  backends/    InferenceBackend interface + BackendRegistry           │
│               BNNSGraph · RTNeural (Eigen/XSIMD) · LibTorch ·        │
│               ONNX Runtime · anira (LibTorch/ONNX)                   │
│  host/        NeuralInferencePlugin (one Tracktion plugin hosting    │
│               any backend) + contention DSP + callback probes        │
│  scenarios/   Scenario interface + registry: mix contention (A),     │
│               instance count (B), serial depth (C), configurable     │
│  runners/     IsolatedRunner · ContentionRunner (measurement         │
│               protocol: warmup → reset → measure → trim)             │
└───────────────────────────────────────────────────────────────────────┘
```

## The three extension axes

### Backends (`src/backends/`)

`InferenceBackend` is the contract:

- lifecycle: `prepare(PrepareContext)` → `process(in, out, n)` (called on
  the audio thread; must be allocation- and lock-free if the backend
  declares itself RT-safe) → `reset()` → `teardown()`
- declarations: `name()` (the exact string that appears in the results),
  `isRealtimeSafe()`, `supportsIsolated()`, `latencySamples()`,
  `requiredFormat()`, `supports(model, whyNot)`, underrun accessors for
  asynchronous backends

The declarations drive placement automatically: isolated mode runs every
backend that `supportsIsolated()`; contention mode runs every backend that
`isRealtimeSafe()`. That is how the paper's backend sets fall out: direct
LibTorch/ONNX allocate per call (not RT-safe → isolated only), anira
schedules inference on a background thread (RT-safe wrapper, but isolated
per-call timing would be meaningless → contention only).

### Models (`models/manifest.json`)

Models are data, not code. A manifest entry declares id, architecture,
state semantics, parameter count, hyperparameters, and one file path per
export format; each backend picks the path for its `requiredFormat()`.
`nab export` generates the built-in catalog (seeded random weights) and
emits the manifest; bring-your-own-model means adding an entry that points
at your files. A model missing a backend's format yields an explicit
skipped row, not an error.

### Scenarios (`src/scenarios/`)

A `Scenario` builds a Tracktion session for each sweep value (track
layouts, plugin chains, where the neural plugin sits) and declares its
sweep parameter and buffer sizes. The measurement protocol around it —
device configuration, warmup, logger reset at the measurement boundary,
xrun baselining, timed window, trim, statistics — lives in
`ContentionRunner` and is scenario-agnostic. JSON-defined
`custom_scenarios` cover simple layouts without C++.

## Measurement core (deliberately boring)

`TimingLogger` records `mach_absolute_time` pairs lock-free from the audio
thread into preallocated buffers. `CallbackStartPlugin` (first on every
active track, earliest-start via atomic CAS) and `CallbackEndPlugin` (on
the master bus) bracket the full callback. Statistics use interpolated
medians and nearest-rank tail percentiles. This code is the paper's
measurement instrument: changes here invalidate baselines and require
re-validation against `experiments/dafx26-paper/expected/`
(see CONTRIBUTING.md).

## Two binaries

RTNeural selects its SIMD backend (Eigen vs XSIMD) at compile time, so the
suite builds `nab-engine` (Eigen) and `nab-engine-xsimd`. Everything else
is identical; `nab run` orchestrates the phase sequence across both.

## Cross-platform contract

The suite currently targets macOS on Apple Silicon (BNNSGraph, Core Audio,
`mach_absolute_time`). The seams for porting exist by design:

- backend availability is compile-gated and registry-driven — a Linux
  build simply registers fewer backends
- the virtual-device check is config-driven (`virtual_output_devices`)
- timing is isolated behind one utility that a `std::chrono::steady_clock`
  implementation can back on other platforms

Porting is a contribution opportunity, not a v1 promise: absolute numbers
are platform-specific either way; the methodology transfers.
