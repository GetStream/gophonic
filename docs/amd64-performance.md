# AMD64 CPU performance

Go 1.27's `GOEXPERIMENT=simd` and `GOAMD64=v3` select native AVX2/FMA
kernels for Whisper's packed FP32 matrix products, unpacked decoder matrix-vector
products, GELU, softmax, LayerNorm, attention preparation, and portable
quantized projections. The v3 selection is made at build time; there is no
per-call CPU feature dispatch. Build a separate v1/v2 binary for older CPUs.
Those builds retain the scalar Go path and the same model and API behavior.
Apple ARM64 keeps its NEON/SME paths.

```sh
GOAMD64=v3 GOEXPERIMENT=simd CGO_ENABLED=0 go build ./cmd/gophonic
```

## Native measurements

On an AMD EPYC 9V74 GitHub runner with Go 1.27.1, the same process measured
both each dispatched kernel and its scalar baseline. Results are medians of
three runs from [the validation workflow](https://github.com/GetStream/gophonic/actions/runs/36059212641).
Each kernel test uses prepacked or preloaded weights and observable output.

| Operation | AMD64 v3 SIMD | Scalar Go | Speedup |
| --- | ---: | ---: | ---: |
| FP32 encoder projection, 1500×384×384 | 11.26 ms | 136.38 ms | 12.1× |
| FP32 decoder projection, 1×384×384, unpacked weights | 18.28 µs | 87.87 µs | 4.8× |
| Q8 vector projection, K=4096, N=4096 | 2.68 ms | 35.00 ms | 13.1× |
| Audio attention softmax, 32×1500 | 50.50 µs | 184.42 µs | 3.7× |
| GELU, 1536 values | 2.41 µs | 6.90 µs | 2.9× |
| LayerNorm, 1024 values | 0.61 µs | 6.93 µs | 11.3× |

The 4×16 FP32 tile measured 11.23 ms against 15.56 ms for the 2×16 tile
on the same input and runner. Both use the packed panel layout; the wider
tile reuses each weight load across four rows without vector spills in the
compiled amd64 loop.

The official tiny.en JFK **full-file PCM-to-text** benchmark, including mel,
encoder, decoder, and tokenizer, measured 0.825 s with SIMD and 3.498 s with
scalar Go: **4.24× faster**. Both used the same converted, checksum-verified
OpenAI FP32 checkpoint and passed the pinned transcript test. Warm calls
reported **0 B/op and 0 allocs/op**. These timings use one Go process per mode
on the same runner; the model, input, and test harness are identical.

The full-file benchmark is distinct from the padded 30-second-window benchmark
used for a matched whisper.cpp comparison. On an AMD EPYC 9V74 runner, a
[separate matched run](https://github.com/GetStream/gophonic/actions/runs/36060210384)
measured the warm, single-worker 30-second window at **1.952 s for Go** and
**1.459 s for whisper.cpp** (Go/C = **1.338**). Both ran the official tiny.en
FP32 tensors, checked the same JFK transcript on every call, used five warm
calls per block, and alternated process order. The Go implementation is still
about 34% slower on this CPU and workload. The result is an end-to-end latency
measurement, not a claim that each individual kernel is slower.

## Dispatch and correctness

The v3 routines execute through the production API, including executor shards.
Native amd64 CI runs the scalar build and v3 SIMD build, while local cross-builds
exercise v1, arm64, and the v3 code under Rosetta for correctness. FP32 results
are checked against independent FP64 references under the package's stated
rounding bound; signed int8 extremes, tails, output padding, and nonfinite
cases are covered. Direct and public-path allocation tests assert zero warm
heap allocations. Weight packing and workspace construction occur outside the
measured inference loop.
