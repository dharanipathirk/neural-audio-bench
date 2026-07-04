# neural-audio-bench

Benchmarking tools for real-time neural audio inference on Apple Silicon,
with support for custom models and inference backends.

Neural audio plugins run inference inside a hard real-time callback. A
backend that is fast in isolation can still miss deadlines when it shares
CPU cores, cache, and memory bandwidth with the rest of a mix. This project
measures both cases:

- **Isolated benchmarks:** throughput and per-buffer latency distributions
  from the median through p99.9 for each backend, model, and buffer size.
- **Contention benchmarks:** the same inference running in Tracktion Engine
  sessions under Core Audio deadline pressure through the BlackHole virtual
  driver. The scenarios cover 36-track mixes, neural instance scaling, and
  serial insert chains, with hardware xrun and background-inference underrun
  counts.

## What's in the box

| Axis | Built in | Extensible via |
|---|---|---|
| **Backends** | BNNSGraph (Apple AMX), RTNeural (Eigen & XSIMD), LibTorch, ONNX Runtime, anira (background-thread LibTorch/ONNX) | one C++ class and one registration line ([guide](docs/adding-a-backend.md)) |
| **Models** | LSTM, TCN, WaveNet × small/medium/large (0.8k–94k params, seeded exports; RTNeural uses template initialization) | a JSON manifest entry pointing at your exported files ([guide](docs/adding-a-model.md)) |
| **Scenarios** | mix contention (real macOS Audio Units), instance count, serial chain depth | JSON track layouts or a C++ `Scenario` ([guide](docs/adding-a-scenario.md)) |

Example results from the paper, measured on an Apple M3 with large models
and a 128-sample buffer at 48 kHz:

| Backend | Isolated RTF | p99 utilization under Core Audio | xruns |
|---|---|---|---|
| BNNSGraph (TCN-L) | 0.022 | < 20% at requested level 36 | 0 |
| RTNeural-XSIMD (TCN-L) | 0.211 (~5× headroom) | **158.5% at requested level 0** | 355 |

Dimension A always runs a fixed 12-track bus/FX-return bed plus the neural
source track. Its historical `contention_level` is a requested conventional
source-track count: levels 24 and 36 both fill the 23 available conventional
source slots. The labels are retained to reproduce the accepted paper.

## Quick start

Requirements: Apple Silicon Mac, macOS 15+, Xcode CLI tools, CMake ≥ 3.28,
[uv](https://docs.astral.sh/uv/). Details: [hardware requirements](docs/hardware-requirements.md).

```bash
git clone https://github.com/dharanipathirk/neural-audio-bench
cd neural-audio-bench

uv sync                          # Python env
uv run nab export                # generate the model catalog (seeded, ~1 min)

cmake --preset default           # fetch all pinned dependencies
cmake --build --preset default   # builds nab-engine + nab-engine-xsimd

# Isolated benchmarks (no audio device needed)
uv run nab run --config configs/base.json --mode isolated

# Contention benchmarks (install BlackHole first: brew install blackhole-2ch,
# set it as the default output device)
uv run nab run --config configs/base.json --mode all

# Analyze + plot
uv run nab analyze --isolated runs/<ts>/isolated_merged.csv --contention runs/<ts>/contention.csv
uv run nab plot    --isolated runs/<ts>/isolated_merged.csv --contention runs/<ts>/contention.csv
```

`nab run` handles the thermal protocol (all-core warmup burns, cooldowns,
`caffeinate`), config layering, and writes a `run_manifest.json` recording
machine metadata and configuration hashes alongside the results. Use
`nab estimate` to predict runtime before committing to a full sweep, and
`configs/smoke.json` for a minutes-long sanity pass. To freeze a study,
keep its config at `experiments/<name>/config.json` and run
`nab run --experiment <name>`; the run's `resolved.json` and
`run_manifest.json` record exactly what ran. `nab report` turns a run's
CSVs into summary tables and figures.

Dependencies are pinned (LibTorch 2.4.1, ONNX Runtime 1.19.2, exact
JUCE/Tracktion/RTNeural/anira commits in `cmake/Versions.cmake`) and the
model catalog is regenerated from a fixed seed, so two machines running the
same config measure the same code and models.

## Documentation

- [Architecture](docs/architecture.md): registries, interfaces, and the measurement core
- [Methodology](docs/methodology.md): protocol, statistics, and tail latency
- [Results schema](docs/results-schema.md): CSV columns and provenance manifests
- [Adding a backend](docs/adding-a-backend.md) · [Adding a model](docs/adding-a-model.md) · [Adding a scenario](docs/adding-a-scenario.md)
- [Contributing](CONTRIBUTING.md): re-baselining policy and release checks

## Citing

If you use this suite in research, please cite the DAFx-26 paper and the
software described in [CITATION.cff](CITATION.cff):

```bibtex
@inproceedings{balasubramaniam2026nab,
  title     = {Real-Time Neural Audio on Apple Silicon: Benchmarking
               Inference Frameworks Under Realistic DAW Contention},
  author    = {Rathna Kumar Balasubramaniam, Dharanipathi and
               Ramachandran, Saravanabalagi and Timoney, Joseph},
  booktitle = {Proceedings of the 29th International Conference on
               Digital Audio Effects (DAFx-26)},
  year      = {2026}
}
```

## License

GPL-3.0-or-later. The contention benchmark links JUCE 8 (AGPLv3 or
commercial) and Tracktion Engine (GPLv3 or commercial); the GPLv3 permits
combining with AGPLv3 code, and the AGPL's network-use terms apply to the
JUCE portion of any derived work. Third-party components are
listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
