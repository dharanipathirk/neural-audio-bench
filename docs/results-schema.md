# Results Schema

Benchmark results are CSV files with a stable, versioned schema
(`schemas/results.schema.json`). The first column of every row is
`schema_version`; it is bumped whenever a column is added, removed, or
changes meaning. Analysis code should check it before interpreting rows.
Current version: **2** (v1 was the pre-release paper format without the
`schema_version`/`status`/`error_msg` columns).

Alongside the CSVs, every `nab run` writes:

- `resolved.json` — the complete configuration the run actually used
  (after base ← experiment ← CLI-override layering).
- `run_manifest.json` — provenance: timestamp, machine metadata (chip,
  core counts, RAM, macOS version), thermal state, tool versions, git
  revision, config/manifest SHA-256 hashes.
- `benchmark_log.txt` — full engine output.

## Common columns

| Column | Meaning |
|---|---|
| `schema_version` | Integer schema version of this row |
| `status` | `ok`, `skipped`, or `error` — a backend that fails to load or doesn't support a model produces an explicit row, never a silent hole |
| `error_msg` | Reason when `status != ok`, else empty |
| `backend` | `BNNSGraph`, `RTNeural_Eigen`, `RTNeural_XSIMD`, `Direct_LibTorch`, `Direct_ONNX`, `Anira_LibTorch`, `Anira_ONNX`, or a registered custom backend |
| `model` | Architecture display name (`LSTM`, `TCN`, `WaveNet`, or custom) |
| `model_size` | Size tier (`small`/`medium`/`large`, or custom) |
| `buffer_size` | Audio buffer size in samples (0 for throughput rows) |
| `rep` | 1-based repetition index |
| `median_ns`, `mean_ns`, `p95_ns`, `p99_ns`, `p999_ns`, `min_ns`, `max_ns`, `stddev_ns` | Per-iteration (isolated) or per-callback (contention) wall-clock duration statistics, nanoseconds. Median is interpolated; p95/p99/p99.9 are nearest-rank |
| `rtf` | Real-time factor = median duration / buffer deadline |
| `dropouts` | Iterations/callbacks exceeding the deadline |
| `total_samples` | Callback rows: number of timed iterations/callbacks. Throughput rows: number of audio samples processed by the single timed call |

## Isolated results (`isolated.csv`)

Additional column: `mode` — `throughput` (one big process call, measures
×real-time; `buffer_size` is 0 and `median_ns`/`mean_ns` hold per-sample
cost) or `callback` (per-buffer timing at each configured buffer size).

## Contention results (`contention.csv`)

Additional columns:

| Column | Meaning |
|---|---|
| `dimension` | `dim_a` (mix contention), `dim_b` (instance count), `dim_c` (serial depth); a `_cb` suffix (`dim_a_cb`, …) marks the full-callback row measured from track start to master-bus end for the same configuration |
| `contention_level` | Dim A: requested conventional track count. In the paper/system-AU layout this is clamped to 0–23 and the fixed 12-track bus/return bed remains active, so historical levels 24 and 36 both mean 23 active conventional sources. Dim C: serial chain depth. Custom scenarios: configured sweep value |
| `instance_count` | Neural plugin instances (dim_b; 1 elsewhere) |
| `util_p50`, `util_p95`, `util_p99`, `util_p999`, `util_max` | Per-callback utilization percentiles: (duration / deadline) × 100% |
| `hw_xruns` | Core Audio hardware overload count during the measurement window (`getXRunCount` delta) |
| `inf_underruns` | anira backends: callbacks where background inference output was not ready (summed across instances); 0 for on-thread backends |
| `thread_count` | Unique audio threads observed during measurement |

## Interpreting results

- Compare **tail percentiles** (`util_p99`, `util_p999`), not medians —
  audio fails on the worst callback.
- `hw_xruns > 0` means audible glitches occurred.
- For anira backends, low audio-thread utilization with high
  `inf_underruns` means the output was silently stale/zeroed — a failure
  mode that utilization alone does not show.
- Cross-machine comparisons: normalize within a machine first; absolute
  numbers are chip- and thermal-dependent (see `run_manifest.json`).
