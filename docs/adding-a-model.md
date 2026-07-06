# Adding a Model

The benchmark treats models as data: a manifest entry pointing at exported
files. There are two routes.

## Route 1: your own exported files (no C++ for most backends)

Export your model to the formats you care about, then add an entry to a
manifest (start from `models/manifest.json`, or keep a separate manifest
and point your config's `models_manifest` at it):

```jsonc
{
  "id": "my_amp_model",
  "arch": "lstm",                 // lstm | tcn | wavenet | custom name
  "size": "custom",
  "display_name": "MyAmp-LSTM64",
  "state": "stateful",
  "channels": 1,
  "param_count": 17921,
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
  (`xcrun coremlcompiler compile`).
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
weights (timing depends on topology, not weight values).

## The RTNeural caveat

RTNeural fixes layer sizes at **compile time** (that is its performance
model). The compiled catalog currently covers the nine paper models; a
manifest entry with new hyperparameters will be reported as unsupported by
the RTNeural backend until a matching template variant is compiled in. All
runtime-loading backends (BNNSGraph, LibTorch, ONNX Runtime, anira) accept
new models without recompiling.

## Verify before benchmarking

```bash
uv run nab validate-config --config yourconfig.json   # manifest + schema check
uv run nab list-models                                # catalog as the suite sees it
./build/nab-engine --mode isolated --config yourconfig.json --output-dir /tmp/check
```

Reference outputs (`*_reference.json` from `nab export`) let you confirm
numerical equivalence of a new export against its PyTorch source.
