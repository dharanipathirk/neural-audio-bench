# Adding an Inference Backend

Benchmarking your own inference engine takes one class and one
registration line. After that it appears in `--list-backends` (via
`nab-engine`), the config `backends` section, and the results CSVs
automatically.

## 1. Implement `InferenceBackend`

Create `src/backends/MyEngineBackend.{h,cpp}`:

```cpp
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "InferenceBackend.h"
#include <my_engine/api.h>

class MyEngineBackend : public InferenceBackend
{
public:
    bool prepare(const PrepareContext& ctx) override
    {
        // Load the model file for your format, allocate every buffer you
        // will ever need, warm any lazy initialization.
        // ctx.model->formatPaths.at(requiredFormat()) is an absolute path.
        // ctx.sampleRate and ctx.maxBlockSize describe the session
        // (maxBlockSize is 0 in isolated mode — size for the largest
        // configured buffer instead).
        return engine.load(ctx.model->formatPaths.at("myformat"));
    }

    void process(const float* in, float* out, int n) noexcept override
    {
        // Audio-thread hot path. If isRealtimeSafe() is true this must not
        // allocate, lock, or block — it runs inside real Core Audio
        // callbacks under contention.
        engine.infer(in, out, n);
    }

    void reset() noexcept override { engine.clearState(); }
    void teardown() override { engine.unload(); }

    const char* name() const override { return "MyEngine"; }   // CSV string
    bool isRealtimeSafe() const override { return true; }      // audio-thread eligible?
    const char* requiredFormat() const override { return "myformat"; }
};
```

Key declarations and what they control:

| Method | Effect |
|---|---|
| `name()` | The `backend` column value in results and the config toggle key |
| `isRealtimeSafe()` | `true` → included in contention benchmarks (audio thread) |
| `supportsIsolated()` | default `true`; return `false` for asynchronous/background-thread designs where per-call timing is meaningless |
| `requiredFormat()` | which manifest format path this backend loads |
| `supports(spec, why)` | veto specific models with a human-readable reason (emitted in results) |
| `latencySamples()` | additional latency your design introduces (reported, not compensated) |
| `underrunCount()/resetUnderruns()` | for asynchronous backends: callbacks where output wasn't ready |

## 2. Register it

In `src/backends/backend_registration.cpp`, add alongside the others
(registration order = execution order = CSV row order):

```cpp
NAB_REGISTER_BACKEND(MyEngineBackend);
```

Gate it with `#if HAS_MYENGINE` if the dependency is optional, and add the
dependency itself to `cmake/Versions.cmake` + `cmake/Dependencies.cmake`
(pinned SHA or checksummed archive — never a floating branch).

## 3. Provide model files

Add your format to each manifest entry you want to benchmark (see
[adding-a-model.md](adding-a-model.md)), or extend
`python/neural_audio_bench/export/exporters.py` so `nab export` emits your
format for the built-in catalog.

## 4. Enable and run

```jsonc
// in your config's "backends" section
"MyEngine": true
```

```bash
cmake --build --preset default
uv run nab run --config configs/base.json --mode isolated
```

## RT-safety honesty check

`isRealtimeSafe()` is a claim the contention benchmark will test — a
backend that allocates in `process()` will show up with xruns and fat
tails. For a stronger guarantee, build your backend's inference loop with
RealtimeSanitizer (`-fsanitize=realtime`, `[[clang::nonblocking]]`) as was
done for BNNSGraph in the paper.
