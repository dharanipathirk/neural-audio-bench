# Adding a Model

The benchmark treats models as data: a manifest entry pointing at exported
files. There are two routes.

## Route 1: your own exported files (no C++ for most backends)

Export your model to the formats you care about, then add an entry to a
manifest (run `nab export` to generate `models/manifest.json` and extend it,
start from the fixture `tests/data/models.manifest.json`, or keep a separate
manifest and point your config's `models_manifest` at it):

```jsonc
{
  "id": "my_amp_model",
  "arch": "lstm",                 // lstm | tcn | wavenet | custom name
  "size": "custom",
  "display_name": "MyAmp-LSTM64",
  "state": "stateful",
  "channels": 1,
  "param_count": 17217,
  "hyperparams": { "hidden": 64 },
  "formats": {
    "coreml":      "my_amp/my_amp.mlmodelc",
    "onnx":        "my_amp/my_amp.onnx",
    "torchscript": "my_amp/my_amp.pt",
    "rtneural":    "my_amp/my_amp_weights.json"
  }
}
```

Paths are relative to the manifest's `models_root`. A model may omit
formats — backends whose format is missing produce an explicit `skipped`
row rather than failing the run.

Export requirements (what the paper's models satisfy; match them for
comparable numbers):

- **Buffer-at-a-time**: the model accepts `(1, 1, N)` input with variable
  N (1–2048) and carries state across calls.
- **CoreML**: `mlprogram` with `ct.StateType` state buffers and a
  `RangeDim` sequence dimension, compiled to `.mlmodelc`
  (`xcrun coremlcompiler compile`). Mind the compute precision: CoreML state
  tensors must be fp16, so a stateful `mlprogram` export runs in half
  precision (this is what the paper's BNNSGraph numbers use) while every
  other backend runs fp32. Record the precision of your own export in
  `notes`.
- **ONNX**: dynamic `seq_len` axis. Recurrent state as explicit
  inputs/outputs (`h_in/c_in → h_out/c_out`); note ONNX Runtime has no
  InOut mechanism for internal conv state (the paper's TCN/WaveNet ONNX
  exports are stateless — document such caveats in `notes`).
- **TorchScript**: traced with state in registered buffers, zeroed before
  saving.
- **RTNeural**: weights JSON (see `python/neural_audio_bench/export/exporters.py`).

Use `python/neural_audio_bench/export/` as the reference implementation —
subclass or copy its exporters for your architecture, and reuse its
validation (state carrying, buffer-size invariance, buffer-vs-sample
equivalence) to catch export bugs before they become benchmark artifacts.

## Route 2: extend the built-in generated catalog

If your model is a size variant of the built-in architectures, edit the
`models` section of your config (hyperparameters per size tier) and run
`nab export` — it regenerates the catalog and manifest with seeded random
weights (timing depends on topology, not weight values). This covers the
runtime-loading backends (BNNSGraph, LibTorch, ONNX Runtime, anira); for
RTNeural see the caveat below.

## The RTNeural caveat

RTNeural fixes layer sizes at **compile time** (that is its performance
model). The compiled catalog covers exactly the nine paper topologies:

| arch | small | medium | large |
|---|---|---|---|
| lstm | hidden 20 | hidden 40 | hidden 96 |
| tcn | 16 ch, k=3, 4 layers | 32 ch, k=3, 8 layers | 48 ch, k=3, 10 layers |
| wavenet | 8 ch, k=2, 3 layers | 16 ch, k=2, 10 layers | 32 ch, k=2, 10 layers |

The backend selects a compiled network by the manifest's `arch` and `size`
names and then checks the entry's `hyperparams` and `param_count` against
the compiled constants in `src/backends/RTNeuralTopology.cpp`. Missing
metadata, any mismatch, or an unknown arch/size
name, produces a `skipped` results row with the reason; RTNeural never runs
a compiled network under a manifest label that describes a different one. To
benchmark a new size with RTNeural, add a template variant in
`src/backends/RTNeuralBackend.h` (an `RTNeuralLSTM_*` alias, or a
`TCNModel`/`WaveNetModel` instantiation), dispatch to it in
`RTNeuralEngine::initialize`, add its row to `RTNeuralTopology.cpp`, and
rebuild. The RTNeural path benchmarks compiled topology and does not load the
manifest's JSON weights, so do not use its output to assert cross-backend
numerical equivalence.

## Verify before benchmarking

```bash
uv run nab validate-config --config yourconfig.json   # manifest + schema check
uv run nab list-models --config yourconfig.json       # catalog as the suite sees it
uv run nab run --config yourconfig.json --mode isolated --warmup 0 --cooldown 0
```

`nab run` layers your file over `configs/base.json` and writes the complete
resolved config beside the results. If you invoke `nab-engine` directly, its
`--config` argument must instead be a complete, already-resolved document; the
C++ engine intentionally does no layering or default filling.

Reference outputs (`*_reference.json` from `nab export`) let you confirm
numerical equivalence of a new export against its PyTorch source.
