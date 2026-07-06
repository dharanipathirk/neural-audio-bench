# Benchmark Methodology

This page documents the measurement methodology, as introduced in the
DAFx-26 paper (*Real-Time Neural Audio on Apple Silicon: Benchmarking
Inference Frameworks Under Realistic DAW Contention*). The central thesis:
**isolated real-time factor is a misleading metric** — a backend can look
comfortably real-time-safe alone yet cause audible glitches when sharing
CPU, cache, memory bandwidth, and the callback budget with a real mix.

## Processing model

All measurements use **buffer-at-a-time** processing — each backend
processes the full audio buffer in its natural mode:

- BNNSGraph: one `BNNSGraphContextExecute` per buffer with dynamic shapes
- LibTorch / ONNX Runtime: one `forward()` / `Run()` per buffer
- RTNeural: per-sample for LSTM (`ModelT` limitation), arena-based buffer
  forward for TCN/WaveNet (RTNeural-NAM pattern)

Models are **stateful** across calls, mono, 48 kHz, with **seeded random
weights** — inference cost depends on topology and parameter count, not
weight values, and no data-dependent branching exists in these architectures.

## Isolated benchmark

Measures raw inference speed without audio-device overhead.

- **Throughput mode**: process N seconds of audio in one call → ×real-time
  factor.
- **Per-callback mode**: after `warmup_iterations` warmup calls, a 10-probe
  measurement picks an iteration count targeting `target_measure_seconds`
  of wall-clock (at least `min_iterations`, capped at 1M). Each iteration
  processes the next segment of a long pre-generated seeded input stream
  (varied input, no artificial cache-hot input path). Per-iteration
  wall-clock is recorded with `mach_absolute_time` (ns resolution).
- **Reported**: median (interpolated), mean, p95/p99/p99.9 (nearest-rank),
  min, max, sample std-dev, RTF (median / deadline), dropout count.

## Contention benchmark

Real DAW sessions built with Tracktion Engine, played in real time through
the BlackHole virtual audio driver — real Core Audio callback deadline
pressure, no acoustic output, reproducible without physical hardware
variance.

**Instrumentation**: a `CallbackStartPlugin` runs first on every active
track (earliest `mach_absolute_time` per callback via atomic CAS); a
`CallbackEndPlugin` on the master bus records the end. The neural plugin
records its own inference duration per callback. Hardware xruns come from
Core Audio's `getXRunCount`; anira backends additionally report
**inference underruns** (callbacks where the background thread missed its
deadline and output was stale/zeroed) — the audio thread doesn't miss its
deadline, but the output is wrong, so both failure modes are counted.

**Scenarios (dimensions)**:

- **A — Mix contention**: a realistic 36-track session (drums, bass,
  guitars, keys, vocals, buses, FX returns) with per-track channel strips
  of real macOS system Audio Units (`use_system_au: true`;
  AUParametricEQ, AUDynamicsProcessor, AUMatrixReverb, …) or lightweight
  custom DSP. The neural plugin sits on one guitar track; the number of
  active conventional tracks sweeps the contention level.
- **B — Instance count**: N neural instances on separate tracks, no
  conventional DSP, buffer 128. Per-instance timing from the first
  instance; session health via xruns and summed inference underruns.
- **C — Serial depth**: one track, one neural instance inside an insert
  chain of depth 1/3/5/7 (bare → channel strip → mix FX → heavy chain),
  buffer 128.

Only **real-time-safe** backends run under contention (declared by each
backend; direct LibTorch/ONNX allocate per call and are excluded — they
participate via anira's background-thread scheduler instead, at the price
of +1 buffer latency).

**Protocol per configuration** (four stages):

1. **Warmup** — set device buffer size and sample rate, build the session
   with 30 s noise clips, play ≥ `warmup_min_seconds` to reach steady
   state (JIT, caches, thread scheduler).
2. **Reset boundary** — stop, reset all timing loggers and underrun
   counters, restart, settle 200 ms, and only then snapshot the xrun
   baseline, so restart transients corrupt neither timing nor xrun counts.
3. **Measure** — record for `measure_seconds` (15 s in the paper: long
   enough to populate tail percentiles, short enough to limit thermal
   drift within a configuration).
4. **Trim & reduce** — discard initial callbacks (≥300 ms worth, at least
   10% of the run, never more than half) to exclude residual settle
   outliers; compute statistics; write CSV rows (a `dim_*` row for the
   neural plugin and a `dim_*_cb` row for the full callback).

**Thermal fairness** (orchestrated by `nab run`): a 45 s all-core CPU burn
precedes each benchmark phase so the first configuration doesn't benefit
from a cold CPU, and a 120 s cooldown between phases restores a consistent
baseline. Run under `caffeinate` to prevent sleep; `nab run` handles this.

## Statistics notes

- Median is interpolated; p95/p99/p99.9 use nearest-rank — tail
  percentiles report an actually-observed worst-case-class value rather
  than an interpolation between two observations.
- Utilization = (duration / deadline) × 100%, reported at
  p50/p95/p99/p99.9/max for contention runs.
- Why tails, not means: audio fails on the worst callback, not the average
  one. A backend with a great median and a fat tail glitches.

## Known caveats

- ONNX Runtime has no InOut mechanism for convolutional state buffers, so
  the ONNX exports of TCN/WaveNet are stateless (LSTM uses explicit state
  I/O and is stateful). Isolated conv comparisons for ONNX carry this
  asymmetry; the model manifest records it per model.
- Contention results depend on machine, macOS version, and thermal
  conditions. Compare backends within a run, not absolute numbers across
  machines. The run manifest records machine metadata and thermal state
  for provenance.
- RTSan verification: the BNNSGraph execution path was verified
  realtime-safe with RealtimeSanitizer (`-fsanitize=realtime`,
  `[[clang::nonblocking]]`) over repeated invocations.
