# Whisper performance

gophonic's Whisper runs on Apple's SME matrix unit from pure Go assembly. On an
M4 Max it transcribes a 30-second window about 3× faster than whisper.cpp on
one core, with the same FP32 weights and identical transcripts. On eight
threads it is 1.1–1.3× faster.

## Results

Warm latency for one reused 30-second window (the 11-second JFK clip plus
padding), PCM to text. Apple M4 Max, macOS High Power mode, Go 1.27.1 with
`GOEXPERIMENT=simd` and `CGO_ENABLED=0`.

| Model | Threads | gophonic | whisper.cpp | Go / C | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| tiny.en | 1 | 100.3 ms | 298.0 ms | 0.337 (2.97× faster) | 0.333–0.341 |
| base.en | 1 | 198.2 ms | 565.0 ms | 0.351 (2.85× faster) | 0.348–0.353 |
| small.en | 1 | 646.1 ms | 1,734 ms | 0.373 (2.68× faster) | 0.364–0.378 |
| tiny.en | 8 | 57.3 ms | 73.0 ms | 0.785 (1.27× faster) | 0.773–0.811 |
| base.en | 8 | 117.5 ms | 129.0 ms | 0.910 (1.10× faster) | 0.897–0.921 |

Raw samples and provenance hashes are in [`benchmarks/whisper/`](benchmarks/whisper/).

### What is compared

| | gophonic | whisper.cpp |
| --- | --- | --- |
| Build | pure Go, `CGO_ENABLED=0` | commit `a664346e`, Release, Accelerate + Apple BLAS, Metal off |
| Weights | official checkpoint converted to FP32 | the same FP32 tensors in ggml format ([converter](../tools/whisper_bundle_to_ggml_f32.py)) |
| Threads | `N` workers, `GOMAXPROCS=N` | `n_threads=N`, `VECLIB_MAXIMUM_THREADS=N` |
| Decoding | greedy, temperature 0 | greedy, temperature 0, default flash attention |
| Check | exact transcript on every call | exact transcript on every call |

[`benchmark_whisper_warm_compare.py`](../tools/benchmark_whisper_warm_compare.py)
alternates the two runtimes in blocks of five timed calls, after five warm
calls per block: ten blocks for the one-core tiny.en and base.en runs, six for
the others. It reports medians and a block-bootstrap 95% interval for
the latency ratio. whisper.cpp stores its KV caches in FP16 by default; gophonic
keeps every activation in FP32.

## Where the time goes

One core, tiny.en, per 30-second window:

| Stage | Time |
| --- | ---: |
| Log-mel frontend | 2.6 ms |
| Encoder | 60.3 ms |
|   matrix products (SME) | ≈27 ms |
|   softmax exponential | ≈18 ms |
|   bias + GELU | ≈6 ms |
|   LayerNorm, residuals, other | ≈9 ms |
| Cross-attention key/value projection | 4.5 ms |
| 26 decoder steps | 31.8 ms |
| **Window total** | **99.2 ms** |

The stage times come from an in-process run; the encoder split is taken from
its CPU profile. The decoder steps are dominated by the vocabulary projection.

- **Encoder matrix products** run at about 1.19 TFLOP/s. The measured peak of
  the FP32 `FMOPA` instruction on one M4 Max core is 1.32 TFLOP/s.
- **The softmax exponential** runs at the NEON throughput limit.
- **The vocabulary projection** reads 40 MB of FP16 weights per token. It is
  limited by FP16-to-FP32 widening in the matrix unit rather than by memory
  bandwidth.

## How it works

### SME from Go assembly

Go's assembler has no SME mnemonics. The kernel sources in
[`internal/whispergemm/smesrc`](../internal/whispergemm/smesrc) are assembled
with clang (`python3 build.py`), and the encodings are embedded as `WORD`
directives. Support is detected at run time through `hw.optional.arm.FEAT_SME`,
and the kernels require 512-bit streaming vectors.

- **GEMM.** Up to 32 rows of A are transposed through ZA into a scratch panel.
  Four 16×16 FP32 ZA tiles then accumulate each 32×32 output block with
  `FMOPA`. Operand loads are double-buffered so the matrix unit never waits.
  Row strides may be shorter than a row, so the STFT frames and both
  convolutions read overlapping windows of their input with no copies.
- **GEMV.** Decoder weights are packed so that 512 outputs accumulate in eight
  ZA vector groups with `FMLA ZA.S[..., VGx4]`. Weights are stored as FP16
  only when every value converts back to identical FP32 bits, and each one is
  widened with `FCVT`/`FCVTLT` before an FP32 fused multiply-add.
- **Signal safety.** When a signal handler returns to a thread in streaming
  mode, Darwin restores only the low 128 bits of each Z register. ZA,
  predicates, and streaming mode survive. Each kernel keeps a sentinel in
  `z31` and recomputes a tile, or repeats a store phase, when the sentinel was
  cleared. Stress tests run the kernels under profiling signals, GC, and
  oversubscribed goroutines.

### The rest of the pipeline

| Stage | Implementation |
| --- | --- |
| Log-mel frontend | STFT as one GEMM with a Hann-windowed DFT matrix, then mel as a second GEMM |
| Stem convolutions | GEMMs over overlapping input rows, with weights packed tap-major |
| Attention | per-tile QKᵀ on SME, NEON softmax, P·V on SME, and 1/sum applied to the 64-wide output |
| Norms and activations | NEON LayerNorm with float64 statistics, fused with the residual add; NEON bias + GELU |
| Decoder | SME GEMV for projections, vocabulary, and cross-attention; NEON greedy argmax |
| Threads | the frontend, row operations, attention tiles, and vocabulary chunks run in parallel on persistent workers |

The M4 Max has one SME unit per performance cluster. Matrix throughput stops
scaling after about two concurrent streaming threads, which is why the
8-thread speedup is smaller than the single-core one.

## Numerical contract

- **Arithmetic:** everything is FP32, and FP16 is used only as lossless
  storage.
- **SME products:** each output starts at +0 and adds products in increasing K
  order with fused FP32 rounding. Results are therefore independent of row
  blocking and worker count, and bit-exact against a sequential FMA oracle.
  `FMOPA` returns the default NaN rather than propagating NaN payloads.
- **Model-level checks:** the encoder is compared with PyTorch activations
  stage by stage (2e-4 absolute plus 2e-4 relative). The frontend is compared
  with OpenAI's reference features (max 5e-5, RMSE 2e-6; measured 2.3e-5 and
  2.9e-7). Every model's JFK transcript must match exactly.
- **NEON kernels** are tested against scalar references with explicit error
  bounds: softmax, GELU over its full accuracy grid, LayerNorm, and argmax
  with ties, NaN, and infinities.

## Limits and next steps

- **Scope:** the SME path needs M4-class hardware. Other ARM64 and AMD64 CPUs
  run the NEON or scalar Go kernels, which have not been compared with
  whisper.cpp in this study.
- **Remaining one-core time** is mostly at hardware limits: `FMOPA` throughput,
  NEON throughput for the exponential, and FP16 widening. The largest
  quality-preserving idea left is to screen the vocabulary with a
  reduced-precision pass and recompute only the candidates exactly, which
  keeps the greedy token identical.
- **Multicore** is limited by the two SME units and by the serial token loop.

## Reproduce

```sh
python3 tools/whisper_pt_to_gophonic.py base.en.pt base.en.gophonic
python3 tools/whisper_bundle_to_ggml_f32.py base.en.gophonic ggml-tiny.en.bin ggml-base.en-f32.bin

GOEXPERIMENT=simd CGO_ENABLED=0 go build -o go-warm ./tools/benchmark_whisper_go_warm
clang++ -O3 -std=c++17 -I$WHISPER_CPP/include -I$WHISPER_CPP/ggml/include \
  tools/benchmark_whisper_cpp_warm.cpp -L$WHISPER_CPP/build/bin -lwhisper \
  -Wl,-rpath,$WHISPER_CPP/build/bin -o cpp-warm

python3 tools/benchmark_whisper_warm_compare.py go-warm base.en.gophonic \
  cpp-warm ggml-base.en-f32.bin testdata/whisper_jfk.pcm.f32le --workers 1
```

The ggml template (`ggml-tiny.en.bin`) supplies only the shared mel filters
and vocabulary. Pass `--expected` for models whose transcript differs from
tiny.en's; small.en adds punctuation.
