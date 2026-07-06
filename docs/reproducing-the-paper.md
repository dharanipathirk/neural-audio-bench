# Reproducing the DAFx-26 Paper

Everything behind the paper's tables and figures is preserved in this
repository. There are three levels of reproduction, from minutes to hours.

## Level 1 — Regenerate tables and figures from the archived data (minutes)

The raw CSVs from the paper's final benchmark run are checked in at
`experiments/dafx26-paper/expected/`. Regenerate the analysis and figures
without any benchmark hardware:

```bash
uv sync
uv run nab report --preset dafx26
```

Add `--pgf` (requires a TeX installation) for the publication-exact
Times/PGF typography.

## Level 2 — Re-run the isolated benchmarks (about an hour)

```bash
uv run nab export                                  # seeded models (byte-reproducible)
cmake --preset default && cmake --build --preset default
uv run nab run --experiment dafx26-paper --mode isolated
```

## Level 3 — Full reproduction including contention (several hours)

```bash
brew install blackhole-2ch    # then set BlackHole 2ch as default output
caffeinate -dims uv run nab run --experiment dafx26-paper
```

`nab estimate --config experiments/dafx26-paper/config.json` predicts the
wall-clock for your machine before you commit.

## What "reproduce" means here

**Pinned, verifiable inputs:**

- Dependencies: exact commit SHAs for JUCE, Tracktion Engine, RTNeural,
  anira and SHA256-checksummed LibTorch 2.4.1 / ONNX Runtime 1.19.2
  archives (`cmake/Versions.cmake`) — the same builds the paper used.
- Models: `nab export` regenerates the nine models byte-identically from
  fixed seed 42 (weights JSON, ONNX, and reference outputs are
  bit-reproducible; CoreML/TorchScript containers may differ in packaging
  metadata but encode identical weights). The paper's original artifacts
  are attached to the `v1.0.0` GitHub Release with a `SHA256SUMS` file.
- Configuration: `experiments/dafx26-paper/config.json` is the exact
  document from the final paper run — the run that produced
  `expected/*.csv`.
- Code: the `dafx26` git tag marks the release whose measurement paths
  were validated as timing-equivalent to the paper's original codebase.

**Reference environment** (paper): MacBook Air, Apple M3 (4P+4E, 24 GB),
macOS 26.4, Apple Clang 21.0.0, BlackHole 2ch.

**What to expect on your machine:**

- *Same chip (M3)*: isolated medians and tail percentiles should agree
  with `expected/isolated.csv` within run-to-run noise (~2–3% on medians;
  tails are noisier). Contention xrun counts are inherently more variable
  but the qualitative failure points (which backend/model/level produces
  xruns) reproduce.
- *Different Apple Silicon*: absolute numbers scale with the chip;
  rankings, contention effects, and the isolated-vs-contention gap — the
  paper's actual claims — reproduce across generations.
- *macOS/toolchain drift*: newer OS or Clang versions can shift absolute
  numbers a few percent; the run manifest records both so comparisons are
  attributable.

## Checking a reproduction

Compare your run against the archived one:

```bash
uv run nab analyze --isolated runs/<ts>/isolated_merged.csv \
                   --contention runs/<ts>/contention.csv
# then diff the summary tables against a `nab analyze` pass over
# experiments/dafx26-paper/expected/*.csv
```

If you publish results from a reproduction, include the
`run_manifest.json` — it captures machine, OS, dependency versions, and
config hashes, which is exactly the context a reader needs.
