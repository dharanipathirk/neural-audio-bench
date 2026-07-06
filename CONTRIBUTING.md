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
require re-baselining against `experiments/dafx26-paper/expected/`:

- **Behavior-neutral changes** (new backends, scenarios, docs, CLI): normal
  PR flow.
- **Measurement-affecting changes**: state this explicitly in the PR, run
  the isolated suite before/after on the same machine, and include both
  CSVs. CI cannot do this — real hardware and thermal control are required.

## Extending the suite

- **New backend**: implement `InferenceBackend`, register it, and it
  appears in config/CLI/results automatically — see `docs/adding-a-backend.md`.
- **New model**: add a manifest entry pointing at your exported files —
  see `docs/adding-a-model.md`. RTNeural requires a rebuild (compile-time
  templates); all other backends load models at runtime.
- **New contention scenario**: a JSON track layout covers most cases;
  C++ `Scenario` subclasses handle the rest — see `docs/adding-a-scenario.md`.

## Style

- C++: C++20, formatted with the repo `.clang-format` (Allman braces,
  4-space indent). No allocation, locking, or logging on the audio thread.
- Python: ruff (lint + format), settings in `pyproject.toml`.
- Every source file carries an SPDX `GPL-3.0-or-later` header.
- Dependency changes go through `cmake/Versions.cmake` (pinned SHA or
  SHA256-checksummed archive) — never a floating branch.

## Tests

`ctest --test-dir build` for C++ units; `uv run pytest` for Python.
CI builds both binaries, runs the unit suites, and executes an isolated
smoke benchmark. Contention mode cannot run in CI (needs BlackHole and
controlled thermals) — it is validated on reference hardware per release.

## License

By contributing you agree that your contributions are licensed under
GPL-3.0-or-later.
