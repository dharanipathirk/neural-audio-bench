# BNNSGraph RealtimeSanitizer probe

This standalone probe is the source used to check that repeated
`BNNSGraphContextExecute` calls do not allocate, lock, or invoke another
operation rejected by Clang's RealtimeSanitizer. Setup, model compilation,
workspace allocation, and teardown happen outside the annotated callback.

Requirements: Apple Silicon, the Xcode command-line tools, `uv`, and a Homebrew
LLVM release with RealtimeSanitizer support (LLVM 20 or newer).

```bash
brew install llvm
cd tools/rtsan-bnns-test
make run
```

`make run` creates and compiles a small CoreML model, builds the C++ probe with
`-fsanitize=realtime`, and executes 100 calls inside a
`[[clang::nonblocking]]` function. A sanitizer diagnostic or nonzero exit is a
failure.

This is an API-level check of the execution call, not a numerical model test or
a substitute for measuring real Core Audio xruns. The benchmark separately
prepares its dynamic shape and workspace during setup/warmup before recording
the callback window.
