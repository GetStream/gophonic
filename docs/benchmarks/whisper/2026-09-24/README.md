# Whisper CPU comparison, September 24, 2026

Official OpenAI `tiny.en`, the same bit-exact FP32 weights, and JFK PCM. Both
helpers transcribe the 30-second padded window to the same 24-token text.
The C reference is pinned whisper.cpp `a664346ea5c6dddff3e61a2b7b32dd4514613f50`
with Metal and CoreML disabled, Apple Accelerate and BLAS enabled, and the
library thread cap set to the reported worker budget. Go uses Go 1.27.0 with
`GOEXPERIMENT=simd`, no cgo, and an explicit matching worker count.

Each [alternating comparison](../../../../tools/benchmark_whisper_warm_compare.py)
uses six Go/C block pairs, five timed calls per block, and five untimed warm
calls before each block. These are in-process PCM-to-text times, excluding
model loading. The raw JSON files contain every sample and model/audio hashes.

| Workers | Go median | Optimized C median | Go / C | Raw samples |
| ---: | ---: | ---: | ---: | --- |
| 1, before 2×32 GEMM | 826.660 ms | 435.651 ms | 1.898 | [JSON](matched-go-c-1.json) |
| 1, with 2×32 GEMM | 782.065 ms | 438.468 ms | 1.784 | [JSON](matched-go-c-1-wide.json) |
| 8 | 210.634 ms | 125.730 ms | 1.675 | [JSON](matched-go-c-8.json) |
| 12 | 247.736 ms | 190.228 ms | 1.302 | [JSON](matched-go-c-12.json) |

The accepted four-query SIMD softmax was included in all three Go rows; its
separate ablation is recorded in [REPORT.md](REPORT.md), [paired samples](paired.json),
and [independent ABBA samples](independent-abba.json). The separate
[single-core profile](single-core-profile/README.md) identifies encoder GEMM
as the dominant Go cost. The accepted 2×32 GEMM ablation is in
[gemm2x32-REPORT.md](gemm2x32-REPORT.md). Rejected long-K and compact-vocabulary experiments
are recorded in [longk-REPORT.md](longk-REPORT.md) and
[vocab16-REPORT.md](vocab16-REPORT.md).
The later [full-context one-core kernel and toolchain study](onecore-ceiling/README.md)
records rejected exact kernel variants, the measured NEON FMA ceiling,
experimental Strassen results, and the Go 1.27.1 comparison.

These ratios describe the pinned reference and this machine. The optimized C
runtime uses default FP16 key/value caches and flash attention despite FP32
weights; token output agrees on JFK, but its internal arithmetic is not all
FP32. `VECLIB_MAXIMUM_THREADS` requests a native library limit; the actual
hardware scheduling was not independently counted. A win against the separate
Accelerate/BLAS-disabled C build would not establish a win against this
optimized CPU reference.
