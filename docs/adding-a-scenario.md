# Adding a Contention Scenario

A scenario defines a session layout (tracks, plugin chains, where the
neural plugin sits) and what it sweeps. The measurement protocol around it
— device setup, warmup, logger reset at the measurement boundary, xrun
baselining, timed window, trim, statistics — is scenario-agnostic and
never changes.

## Route 1: JSON (no C++)

For layouts of the form "N tracks of chain X, a neural track, sweep some
count", add a `custom_scenarios` array to your config:

```jsonc
"custom_scenarios": [
  {
    "id": "synth_stack",
    "sweep": { "parameter": "track_count", "values": [2, 8, 16] },
    "buffer_sizes": [128],
    "tracks": [
      { "count": "sweep", "chain": ["eq", "reverb"], "clip": "noise" },
      { "count": 1, "neural": true, "chain": ["eq"], "chain_after": ["delay"] }
    ]
  }
]
```

Semantics:

- Each `tracks` element describes a group of identical tracks. `count` is
  an integer or `"sweep"` (the sweep value sets the group's track count).
- `chain` names map to the built-in contention DSP: `eq`, `compressor`,
  `reverb`, `delay`. On a neural track, `chain` runs before the neural
  plugin and `chain_after` runs after it.
- Every track gets a callback-start probe; the master bus gets the
  callback-end probe — full-callback rows (`<id>_cb`) come for free.
- Results rows carry your `id` in the `dimension` column, with the sweep
  value in `contention_level`.

Custom scenarios run after the built-ins, for every enabled RT-safe
backend × model, with the standard protocol and reps.

## Route 2: C++ (full control)

For anything the JSON spec can't express (staggered per-track chains,
mute logic, bus routing, multiple neural placements), subclass `Scenario`
(`src/scenarios/Scenario.h`):

```cpp
class MyScenario : public nab::Scenario
{
public:
    const char* id() const override { return "my_dim"; }        // CSV dimension
    const char* title() const override { return "My Scenario"; } // stderr banner

    std::vector<int> sweepValues(const BenchmarkRuntimeConfig& cfg) const override;
    std::vector<int> bufferSizes(const BenchmarkRuntimeConfig& cfg) const override;

    SessionTimingInfo build(te::Edit& edit, EditBuilder& builder,
                            const std::string& backend, const ModelSpec& model,
                            int sweepValue, const BenchmarkRuntimeConfig& cfg,
                            double sampleRate) override;

    void csvColumns(int sweepValue, int& contentionLevel, int& instanceCount) const override;
};
```

Inside `build()`, use the `EditBuilder` toolbox: `ensureNoiseWavFile`,
`addAudioClip`, `addConventionalDSP`, `addNeuralPlugin` (creates the
unified neural plugin for whatever backend is under test and wires its
timing logger + underrun accounting), `addCallbackStart` on every active
track, and `addCallbackEnd` on the master bus. The three built-ins —
`MixContentionScenario` (36-track mix with real system AUs),
`InstanceCountScenario`, `SerialDepthScenario` — are the reference
implementations.

Register it in `nab::registerBuiltinScenarios`
(`src/scenarios/ScenarioRegistry.cpp`); registration order is execution
and CSV-grouping order.

## Design guidance

- **Sweep one variable.** The built-in dimensions each isolate a single
  contention mechanism (parallel mix load, instance count, serial depth).
  A scenario that varies two things at once produces data nobody can
  attribute.
- **Keep sessions steady-state.** Layouts should produce constant load for
  the whole measurement window — automation ramps or clip gaps would
  smear the distribution the tail percentiles are trying to capture.
- **Mind the measurement budget.** Each (backend × model × buffer × sweep
  value × rep) costs warmup + measure seconds; `nab estimate` includes
  custom scenarios when predicting runtime.
