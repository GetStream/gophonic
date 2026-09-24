# Numerical and allocation validation

The tests keep the model checkpoint, input fixture, and expected execution
semantics explicit. Most tests run without weights. Tests needing a model skip
unless a bundle is supplied through the corresponding environment variable;
they never download one.

## Run the gates

Basic scalar and SIMD checks:

```sh
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 GOEXPERIMENT=simd go test ./...
```

Run with converted models to include model-level parity and allocation
checks:

```sh
export GOPHONIC_TEST_MODEL="$PWD/smart-turn-v3.2.gophonic"
export GOPHONIC_TEST_TINYMEL_MODEL="$PWD/tinymel.gophonic"
export GOPHONIC_WHISPER_MODEL="$PWD/tiny.en.gophonic"

go test ./...
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -race ./...
go vet ./...
```

The [CI workflow](../.github/workflows/ci.yml) runs tests, vet, and CLI builds
on AMD64 Linux and ARM64 macOS with scalar and SIMD configurations. It does not
download model weights, so the model-level gates above remain explicit local
or release checks.

The race detector uses Go's supported race build environment, which may require
a C toolchain. It is a development check; the inference binary can still be
built with `CGO_ENABLED=0`.

## What is checked

| Boundary | Evidence |
| --- | --- |
| 400-point FFT | Independent direct-DFT comparison |
| Whisper frontend | Saved 16 kHz tone features: max absolute error `2e-5`, RMSE `2e-6`; silence output |
| Parallel frontend | Bitwise comparison with serial output across helper counts |
| Standalone frontend | Comparison with the existing frontend, warmed allocation and closed-state checks |
| Smart Turn prediction | Silence and tone ONNX probabilities within `2e-5` |
| TinyMelNet prediction | Zero-feature, tone, and deterministic-pattern ONNX probabilities within `2e-5` |
| TinyMelNet convolution stages | Selected ONNX tensor samples within `3e-5`, scale and zero-point checks |
| TinyMelNet GRU | ONNX equation reference, gate/direction semantics, full saved GRU outputs within `2e-5` |
| SIMD kernels | Scalar comparisons, tails/ranges, quantization rounding and saturation |
| Reused workspaces | Warm public prediction allocation assertions and benchmark allocation reports |
| Built-in session adapters | Direct-call prediction parity, nil/closed lifecycle checks, warm allocations through `AudioSession` |
| Whisper bundles | Official checkpoint SHA-256, exact tensor manifest and shapes for the model's dimensions, bundle checksum, finite FP32 validation |
| Whisper whole-file frontend | Pinned PyTorch JFK and long-file mel samples; separate single-window full-array oracle |
| Whisper encoder and decoder | Pinned PyTorch stem, all four encoder blocks, final encoder, prefix/next-token logits, cache reset, and JFK token sequence; scalar/SIMD worker parity |
| Whisper full transcription | Pinned OpenAI JFK, silence, and JFK plus 35 seconds of silence transcripts; warm allocation checks |

An external-package conformance test also implements `AudioSession` using only
the standalone Whisper frontend. Its model head is a test stand-in; this proves
that another package can implement the interface without access to internal
model types, not that a third model architecture has been ported.

“Zero features” means an all-zero input tensor. It is distinct from the
frontend's representation of silent PCM, which is `-1.5` in every feature bin.

The TinyMelNet manifest at `testdata/tinymel_oracle.json` records the source
checkpoint SHA-256, tensor shapes, stage hashes, quantization metadata, and
oracle runtime. The checked-in oracle was generated with ONNX Runtime 1.30.0,
`CPUExecutionProvider`, sequential graph execution, eight intra-op threads, and
all graph optimizations enabled.

## Regenerate an independent oracle

Use the original, SHA-pinned ONNX checkpoint and an ONNX Runtime environment:

```sh
.venv/bin/python -m pip install numpy onnx onnxruntime
.venv/bin/python tools/tinymel_to_gophonic.py \
  model_tinymel_int8.onnx --oracle-dir /tmp/gophonic-oracle
```

The converter reads `testdata/tone.mel.f32le` by default and can take another
path with `--tone-mel`. Inspect the generated manifest and compare it with the
existing fixtures before replacing them. Expected outputs must come from the
independent reference, not from the Go implementation under test. Do not widen
tolerances or refresh an oracle merely to make an optimization pass.

## What the tests do not establish

The current model oracles cover a small deterministic fixture set. They do not
constitute a broad speech corpus evaluation or prove identical decisions near
a threshold on all inputs. Scalar and SIMD reductions can differ within the
accepted floating-point tolerances.

Resampling tests cover basic channel mixing and window behavior; the saved
Whisper oracle covers 16 kHz input. Resampling at other rates is not asserted
bitwise equivalent to a particular external audio library. ARM64 development
results do not establish performance or runtime validation on every supported
Go target.

The zero-allocation claim applies to the
[documented public-call boundary](api.md#ownership-lifetime-and-allocations).
Kernel microbenchmarks alone do not prove it. Performance changes must also
pass the relevant numerical checks and improve an exercised public prediction
benchmark under the same CPU and worker settings.
