# CLM and Qwen3-8B CPU performance

Measured on 2026-09-24 on an Apple M4 Max (12 performance + 4 efficiency
cores, 64 GB), CPU only, model loading excluded unless stated. The local path
loads the official `Qwen/Qwen3-8B` BF16 safetensors snapshot directly; the
llama.cpp comparison uses the official Qwen3-8B `Q8_0` GGUF with `-ngl 0 -dev
none` and its best thread count (8).

## Results

| Workload | gophonic before | llama.cpp Q8_0, CPU | gophonic now |
| --- | ---: | ---: | ---: |
| 1 token | 338 ms | 31.8 ms | 65 ms |
| 12 tokens (one text) | 413 ms | 90 ms | **69 ms** |
| 64–70 tokens (one text) | ≈2 s | 487 ms (64) | **342 ms (70)** |
| 16 texts × ~12 tokens | ≈6.6 s | — | 1.08 s |
| Rank 16 candidates, cached | 6.6 s | — | **1.65 ms** |
| Cosine vs official BF16 (`hello`) | 0.99738 | 0.99929 | **0.99991** |
| Load (safetensors → packed) | goinfer load + 5–8 s repack | — | 3.1 s |

All warmed paths report 0 allocs/op. “Before” is the previous single-thread
per-row int8 path. Probabilities for the pinned CLM ranking now differ from the
official BF16 PyTorch pipeline by at most 7.1e-5 (previously 1.0e-3).

## What changed

**Exact weights.** BF16 converts to FP16 exactly after multiplying a row by a
power of two, so each projection row is stored as FP16 with a power-of-two row
scale. Every official weight is represented without rounding (a loader test
checks this bit for bit). This is the main quality gain: the previous per-row
int8 weights were the dominant error. Per-row int8 remains available with
`Options{Weights: "int8"}` for half the memory.

**FP16-activation SME kernel.** Activations are scaled per row by a power of
two into `[2^14, 2^15)`, rounded to FP16 (11-bit significand, finer than the
BF16 activations of the reference runtime), and multiplied with the widening
FP16 `FMOPA` into FP32 accumulators, 16 rows × 64 columns per tile. Column and
row scales are applied as ZA rows are stored. Measured over 2–4 GB of distinct
weights, FP16 and int8 weights run at the same ~1.9 TMAC/s: prefill is bound
by the matrix units, not DRAM, so exact weights cost no speed.

**Parallel layer pipeline.** A persistent worker pool (8 participants by
default; more only adds contention on the two P-cluster SME units) runs every
stage. Projections are claimed one 64-column panel at a time so the per-stage
barrier never waits for more than one panel; Q/K/V and gate/up share one
activation pack and one dispatch. Norms, QK-norm/RoPE, streaming-softmax
attention, and SwiGLU use NEON through Go's `simd/archsimd`. Idle workers spin
for 1 ms before parking. On this machine the projection stage runs at
~62 ms per 16-token tile, within 5% of a static-split lower bound.

**Batched texts.** All texts of one `Embed` call are packed into shared
16-row tiles (up to 512 tokens per pass); each sequence attends only to its
own tokens. The CLM head scores all candidates as one batched GEMM instead of
one matrix-vector product per candidate.

**Exact embedding cache.** Finished embeddings are cached by a 128-bit hash of
their token IDs (4096 entries, 64 MiB, CLOCK eviction, no allocation). Inference
is deterministic, so a hit returns exactly what recomputation would.

**No third-party inference runtime.** A native safetensors loader (JSON via
`vibejson`) replaces goinfer; the tokenizer was already native.

## Limits

Prefill costs ≈3.7 ms per token (one 16-row tile ≈ 60 ms) at the SME FP16
ceiling. A single token pays for a full tile; a bandwidth-bound matrix-vector
path would help single-token requests with int8 weights. Long texts (up to the
2048-token CLM limit) scale linearly in projection time; attention is still a
simple streaming loop. CPUs without SME use a portable panel-decode kernel
(NEON on arm64) that is correct but far slower.

## Reproducing

```sh
cd examples/clm-qwen
export GOPHONIC_QWEN3_MODEL=/path/to/Qwen3-8B
export GOPHONIC_QWEN3_TOKENIZER=$GOPHONIC_QWEN3_MODEL
export GOPHONIC_QWEN3_HELLO_REFERENCE=/path/to/qwen3-8b-hello-reference.f32
export GOPHONIC_CLM_HEAD_BUNDLE=/path/to/CLM_v0.1-8B.gclm
CGO_ENABLED=0 GOEXPERIMENT=simd go test -run TestOfficial -v
CGO_ENABLED=0 GOEXPERIMENT=simd go test -run '^$' -bench 'Official' -benchtime=20x
llama-bench -m Qwen3-8B-Q8_0.gguf -p 1,12,64 -n 0 -embd 1 -t 8 -ngl 0 -dev none
```

`examples/clm-qwen/tools/reference_hidden.py` writes the BF16 reference vector. Set
`GOPHONIC_QWEN_WEIGHTS=int8` or `GOPHONIC_QWEN_THREADS=N` to vary benchmarks.
