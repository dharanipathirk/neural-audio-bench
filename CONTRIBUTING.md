# Contributing to neural-audio-bench

Thanks for your interest! Contributions are welcome — new inference
backends, model architectures, contention scenarios, platform ports, and
documentation improvements all make the suite more useful.

## Development setup

```bash
git clone https://github.com/dharanipathirk/neural-audio-bench
cd neural-audio-bench
uv sync                                   # Python environment (uv required)
uv run nab export                         # generate the model catalog
cmake --preset default                    # fetches all pinned C++ deps
cmake --build --preset default
uv run pre-commit install                 # formatting/lint hooks
```

Requirements: macOS 15+ on Apple Silicon, Xcode command-line tools,
CMake ≥ 3.28, [uv](https://docs.astral.sh/uv/). Contention mode
additionally needs [BlackHole](https://github.com/ExistentialAudio/BlackHole)
(`brew install blackhole-2ch`). See `docs/hardware-requirements.md`.

## What to know before changing benchmark code

The measurement core produced published results. Changes to timing loops,
warmup/trim protocol, statistics, or compile flags shift the numbers and
must be re-baselined against a run made with the previous code on the
same machine:

- **Behavior-neutral changes** (new backends, scenarios, docs, CLI): normal
  PR flow.
- **Measurement-affecting changes**: state this explicitly in the PR, run
  the isolated suite before/after on the same machine, and include both
  CSVs. Real hardware and thermal control are required for this.

## Extending the suite

- **New backend**: implement `InferenceBackend`, register it, and it
  appears in config/CLI/results automatically — see `docs/adding-a-backend.md`.
- **New model**: add a manifest entry pointing at your exported files —
  see `docs/adding-a-model.md`. All backends except RTNeural load models at
  runtime; RTNeural only runs the nine topologies compiled into
  `src/backends/RTNeuralBackend.h` and reports any other entry as
  unsupported, so a new size for it means adding a template variant and
  rebuilding.
- **New contention scenario**: a JSON track layout covers most cases;
  C++ `Scenario` subclasses handle the rest — see `docs/adding-a-scenario.md`.

## Style

- C++: C++20, Allman braces, 4-space indent — match the surrounding code
  (`.clang-format` approximates the style for new files; the existing
  measurement core is deliberately kept byte-identical to the code that
  produced the paper results, so don't mass-reformat). No allocation,
  locking, or logging on the audio thread.
- Python: ruff (lint + format), settings in `pyproject.toml`.
- Every source file carries an SPDX `GPL-3.0-or-later` header.
- Dependency changes go through `cmake/Versions.cmake` (pinned SHA or
  SHA256-checksummed archive) — never a floating branch.

## Tests

`ctest --test-dir build` for C++ units; `uv run pytest` for Python. There is
no hosted CI: run the release checks below locally before opening a PR.
Contention mode needs BlackHole and controlled thermals and is validated on
reference hardware per release.

## Release checks

From the repository root, after `uv run nab export` and a build:

```bash
uv lock --check
uv run ruff check python tests
uv run ruff format --check python tests
uv run pre-commit run --all-files
uv run pytest
cmake --build --preset default
ctest --test-dir build --output-on-failure
uv run nab validate-config --config configs/base.json
./build/nab-engine --mode isolated --config configs/smoke.json --output /tmp/nab-smoke-eigen.csv
./build/nab-engine-xsimd --mode isolated --config configs/smoke.json --output /tmp/nab-smoke-xsimd.csv
uv run python -m neural_audio_bench.validate_results /tmp/nab-smoke-eigen.csv /tmp/nab-smoke-xsimd.csv
```

## License

By contributing you agree that your contributions are licensed under
GPL-3.0-or-later.
