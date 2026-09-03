# Hardware & System Requirements

## Required

| Requirement | Why |
|---|---|
| Apple Silicon Mac (M1 or later) | BNNSGraph/AMX backends are Apple-specific; timing uses `mach_absolute_time` |
| macOS 15+ | BNNSGraph stateful-model API, CoreML `mlprogram` state support |
| Xcode command-line tools | Compiler + `coremlcompiler` for `.mlpackage → .mlmodelc` |
| CMake ≥ 3.28 | `brew install cmake` |
| [uv](https://docs.astral.sh/uv/) | Python environment + `nab` CLI |
| ~3 GB disk | Fetched dependencies (JUCE, Tracktion, LibTorch, ONNX Runtime) + build |

An 8 GB machine can install and build the project, but is likely to swap during
the full isolated sweep and should not be used for trustworthy timings.
**16 GB RAM is the practical minimum; 24 GB or more is recommended.** The
isolated runner deliberately pre-generates a long varied input stream, which
can approach 4.5 GB for a 1024-sample callback configuration before model
runtimes and the build are counted.

## Contention mode only: BlackHole

The contention benchmark plays real DAW sessions through
[BlackHole](https://github.com/ExistentialAudio/BlackHole), a virtual
loopback audio driver. It delivers genuine Core Audio callback deadline
pressure with no acoustic output and no physical-device variance.

```bash
brew install blackhole-2ch
```

Then set **BlackHole 2ch as the default output device** (System Settings →
Sound → Output) before running `nab run` with contention enabled. The
engine refuses to run contention benchmarks on a non-virtual device
(override with `--allow-any-device`, at the cost of reproducibility and
audible noise). Isolated mode needs no audio device at all.

## Getting trustworthy numbers

- **Close everything else.** Browsers, Spotlight indexing, backup jobs all
  steal cache and bandwidth.
- **Use `nab run`** rather than invoking the engine directly: it runs the
  thermal protocol (all-core warmup burn before each phase, cooldowns
  between phases) and `caffeinate` so the machine neither sleeps nor
  starts a phase cold.
- **Plug in the power adapter** on laptops; battery power management
  throttles differently.
- **Expect scale, not identity, across chips.** An M3 Air (paper reference
  machine) and an M2 Ultra Studio give different absolute numbers; the
  backend rankings and contention effects are what reproduce.
- Runtime for the full paper configuration is several hours; use
  `nab estimate` to predict it from your config before committing.

## Smoke run

`configs/smoke.json` runs the isolated benchmark on the small models in a
couple of minutes and verifies the toolchain, models, and both engine
binaries without an audio device. It is a functional check, not a
measurement; the contention suite is validated on reference hardware for
each release.
