# Isolated Microarchitecture Analysis

This folder keeps optional microarchitecture-side experiments separate from the
main benchmark CSVs and figures.

The main benchmark answers real-time audio questions:

- isolated throughput
- callback deadline utilization
- xruns / inference underruns under contention

This folder is for isolated-only explanatory analysis on Apple Silicon:

- AMX instruction mix via `microarch/decode_amx.py`
- one backend + one model + one size per run
- separate configs, logs, CSVs, and JSON outputs under `microarch/results/`

## Why separate?

These runs are not part of the core publication benchmark pipeline. They are
meant to explain backend behavior, not replace the real-time results.

## What the harness does

`run_isolated_microarch.py`:

1. creates a timestamped output directory
2. generates a temporary benchmark config for each combination
3. runs `nab-engine --mode isolated` with exactly one enabled:
   - backend
   - model type
   - model size
4. attaches `decode_amx.py` to the running process
5. saves:
   - per-run config JSON
   - isolated CSV
   - benchmark log
   - AMX JSON
   - a manifest summarizing all runs

## Default experiment matrix

Defaults are intentionally small:

- backends:
  - `BNNSGraph`
  - `RTNeural_Eigen` or `RTNeural_XSIMD` depending on the binary
  - `Direct_LibTorch`
  - `Direct_ONNX`
- models:
  - `LSTM`
  - `TCN`
  - `WaveNet`
- sizes:
  - `large`

This gives a focused isolated-only explanatory pass without mixing results into
`results/`.

## Example

```bash
uv run python microarch/run_isolated_microarch.py \
  --bench build/nab-engine-xsimd
```

Outputs go to:

```text
microarch/results/YYYYMMDD_HHMMSS/
```

## Notes

- This harness currently automates AMX-side analysis only.
- It does not collect PMU counters such as cycles, instructions, IPC, or
  L1/L2 misses.
- Those hardware-counter experiments should remain isolated-only if added later.
