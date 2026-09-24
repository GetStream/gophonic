# CLM and Qwen3-8B CPU performance

Measured on 2026-09-24 on an Apple M4 Max (12 performance + 4 efficiency
cores, 64 GB), CPU only, model loading excluded unless stated. The local path
loads the official `Qwen/Qwen3-8B` BF16 safetensors snapshot directly; the
llama.cpp comparison uses the official Qwen3-8B `Q8_0` GGUF with `-ngl 0 -dev
none` and its best thread count (8).

## Results

| Workload | gophonic before | llama.cpp Q8_0, CPU | exact (default) | int8 fast mode |
| --- | ---: | ---: | ---: | ---: |
| 1 token | 338 ms | 31.8 ms | 51 ms | **28 ms** |
| 12 tokens (one text) | 413 ms | 90 ms | 66 ms | **43 ms** |
| 64–70 tokens (one text) | ≈2 s | 487 ms (64) | 304 ms (70) | **150 ms (70)** |
| 2048 tokens (one text) | minutes | — | 8.19 s | **4.89 s** |
| New 30-token turn on an 1800-token state | minutes | — | **155 ms** | — |
| 16 texts × ~12 tokens | ≈6.6 s | — | 841 ms | **415 ms** |
| `Question.Choose`, one new input | — | — | 128 ms | — |
| `Question.ChooseBatch`, per input (16) | — | — | 94 ms | **47 ms** |
| Rank 16 candidates, cached | 6.6 s | — | **1.7 ms** | — |
| Cosine vs official BF16 (`hello`) | 0.99738 | 0.99929 | **0.99991** | 0.99866 |
| CLM probability error vs official | 1.0e-3 | — | **7.1e-5** | 2.2e-4 |
| Load (safetensors → packed) | goinfer load + 5–8 s repack | — | 3.0 s | 4.3 s |

The int8 mode (`Options{Weights: "int8"}`) rotates each projection's input
with a randomized Hadamard transform and runs int8×int8 `SMOPA` with exact
int32 accumulation (≈35 ms per 16-row tile, against ≈62 ms for FP16). With
four or more tiles, four SME workers and the remaining performance cores
split each projection from opposite ends: SME claims 64-column panels, NEON
claims 16-column strips with `SDOT`, and both produce bit-identical results.
Its
answers on the 31 `Question` probes match the exact mode. GPTQ-rounded
weights raise its fidelity further in an offline study (cosine 0.99961);
that conversion is not yet part of the loader.

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

**Blocked SME attention.** Sequences of 64 tokens or more compute attention
per KV-head group and 128-query block: keys and values up to the block's
causal limit are packed once, and Q·Kᵀ and P·V run as FP32 SME matrix
products with a vectorized softmax between them. Per-token cost stays near
3.9 ms from 280 to 2048 tokens; the previous streaming loop made 2048 tokens
take 23 s.

**Prefix key/value store.** The encoder keeps the post-RoPE keys and values
of the last long input for all 36 layers in FP32 (288 KiB per token, one
2048-token store by default). A later input that shares a token prefix, such
as a conversation state with a new turn, evaluates only its new tokens.
Branching from the middle of the stored sequence reuses the shared part.

**Final-layer pruning.** Only each sequence's last row reaches the output, so
the last layer's output projection and MLP run on one row per sequence.

**Exact embedding cache.** Finished embeddings are cached by a 128-bit hash of
their token IDs (4096 entries, 64 MiB, CLOCK eviction, no allocation). Inference
is deterministic, so a hit returns exactly what recomputation would.

**No third-party inference runtime.** A native safetensors loader (JSON via
`vibejson`) replaces goinfer; the tokenizer was already native.

## Limits

Every floating-point SME format (FP32, FP16, BF16, and int16) peaks at
2.07 TMAC/s across the chip in a register-only loop, with or without
efficiency cores; the projection kernel sustains about 1.8–1.9 TMAC/s while
streaming weights from DRAM. Fresh prefill therefore costs ≈3.7–3.9 ms per
token (one 16-row tile ≈ 60 ms), and a single token pays for a full tile.
Only int8×int8 `SMOPA` runs faster (4.1 TMAC/s), and it would require
quantized activations. Further speedups on this CPU come from avoiding work:
the prefix store and the embedding cache. CPUs without SME use a portable
panel-decode kernel (NEON on arm64) that is correct but far slower.

## Reproducing

```sh
export GOPHONIC_QWEN3_MODEL=/path/to/Qwen3-8B
export GOPHONIC_QWEN3_TOKENIZER=$GOPHONIC_QWEN3_MODEL
export GOPHONIC_QWEN3_HELLO_REFERENCE=/path/to/qwen3-8b-hello-reference.f32
export GOPHONIC_CLM_HEAD_BUNDLE=/path/to/CLM_v0.1-8B.gclm
CGO_ENABLED=0 GOEXPERIMENT=simd go test ./qwen3 -run TestOfficial -v
CGO_ENABLED=0 GOEXPERIMENT=simd go test ./qwen3 -run '^$' -bench 'Official' -benchtime=5x
llama-bench -m Qwen3-8B-Q8_0.gguf -p 1,12,64 -n 0 -embd 1 -t 8 -ngl 0 -dev none
```

`qwen3/tools/reference_hidden.py` writes the BF16 reference vector. Set
`GOPHONIC_QWEN_WEIGHTS=int8` or `GOPHONIC_QWEN_THREADS=N` to vary benchmarks.
