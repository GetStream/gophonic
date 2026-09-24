# One-core full-context Whisper kernel research, 2026-09-24

No production kernel changes are recommended from this round. The experiments
ran against `codex/whisper-cpu-performance` at `6635909`. All prototypes are
isolated Go overlays in [`raw.tar.gz`](raw.tar.gz). The primary
performance contract is the original **1500-position context**, official
Whisper tiny.en FP32 weights, one Go worker and `GOMAXPROCS=1`, pure Go,
`GOEXPERIMENT=simd`, and zero warm allocations. Hardware: Apple M4 Max;
Go 1.27.0; darwin/arm64; CGO disabled.

## Results

| Experiment | Full-work result | Decision |
| --- | --- | --- |
| Wider 3x32 / 6x16 / 5x16 tiles with direct scalar broadcasts | About 70–80% slower for full encoder matrices | Reject: register spills and expensive scalar broadcast lowering |
| Ordered-K blocking around the current 2x32 kernel | PV about 1–2% better; contraction up to 2%; conv2 up to 5% | Too small to retain extra kernel and dispatch complexity |
| Cache-blocked replicated activations + 5x16 tile | Safe version effectively flat; pointer version expansion 5.2%, contraction 3.7%, but attention PV 17.8% slower | Reject; packing/call/load overhead remains material |
| Original 2x32 with static broadcast-index tables and/or bounds-validated pointers | 0–0.7% differences across full shapes | Reject: the removed instructions were not the throughput bottleneck |
| Strassen-Winograd, shape-selected one/two levels, vectorized combination passes | Full PCM-to-text **781.10 -> 744.29 ms**, **4.71%** lower latency | Keep as research only: changed arithmetic and extra prepacked weights, modest gain, no corpus quality claim |

Strassen used 24 alternating baseline/candidate pairs after five warm calls
per mode. It won all 24 pairs. A paired-bootstrap ratio-of-medians 95% interval
is **0.9513–0.9546** (seed 20260924, 10,000 replicates). Both modes produced
identical 24 text tokens, the same two prompt tokens, and EOT, for the official
JFK PCM fixture. `testing.AllocsPerRun(2)` returned zero for both public-call
modes after warmup. This single fixture establishes integration only; it does
not establish equal transcription quality. No arithmetic mode is installed
in the default `TranscribeWindowInto` path.

The exact candidate tests compare every float bit against the prior four-row
reduction path, preserve padding sentinels, exercise tails/misalignment and
NaN/Inf/signed-zero inputs, and assert zero allocations. The Strassen matrix
screen instead measures numerical error because its addition order changes.
The first Strassen version lost theoretical savings to scalar combination
passes; vectorizing these and using Winograd's lower-addition schedule helped.
Three levels still lose to overhead for the important shapes.

## Empirical compute limit

Five register-only FMA loops were generated using distinct live accumulators,
preloaded vector operands, no vector loads or stores inside the reduction
loop, and observable output. Thirty timings followed five warmups per shape.
Disassembly is saved and confirms the expected FMAs; source and raw durations
are in the archive.

| Register geometry | Median GFLOP/s | Best GFLOP/s |
| --- | ---: | ---: |
| 2x32 | 82.99 | 83.71 |
| 4x16 | 83.06 | 83.66 |
| 5x16 | 82.21 | 82.93 |
| 3x24 | 83.02 | 83.70 |
| 7x12 | 81.23 | 82.22 |

The shipped full projection/expansion/contraction GEMMs are approximately
69–72 GFLOP/s, about 83–87% of this observed register-only rate. This is an
empirical limit of these compiler/instruction patterns on this run, not a
formal physical ceiling. A preliminary loop with the same two source
registers for every FMA measured only 54.6 GFLOP/s; it is retained in the raw
artifacts to show why operand/register layout matters and is not used as the
ceiling.

The full encoder plus decoder cross-KV preparation performs **40.477 GFLOP**
of dense matrix arithmetic: 1.880 GFLOP stem convolutions, 7.078 GFLOP QKV/output
projections, 14.156 GFLOP MLP, 13.824 GFLOP QK/PV attention, and 3.539 GFLOP
cross-KV. At the best observed 83.71 GFLOP/s, that arithmetic alone takes
approximately **484 ms**, before frontend, normalization, activations, softmax,
and token decoding. The pinned optimized CPU C full call is about **438 ms**.
This does not support expecting ordinary exact FP32 NEON loop tuning to close
the complete gap, much less halve C's latency. It does not rule out a different
algorithm or wider/matrix arithmetic capability.

## Accelerate control

The existing independent Accelerate SGEMM reference uses the same shapes,
FP32 input/output buffers, row-major input A, transposed row-major weights,
alpha=1, beta=0, the same deterministic xorshift input distribution, five
warm calls, and 100 timed calls per shape. It checks 17 output elements against
an independent FP64 reduction. Two separate runs with
`VECLIB_MAXIMUM_THREADS=1` reported:

| Shape M,K,N | Run A ms | Run B ms | Approx. GFLOP/s |
| --- | ---: | ---: | ---: |
| 1500,384,384 | 0.4200 | 0.3991 | 1053–1108 |
| 1500,384,1536 | 1.5446 | 1.5485 | 1143–1146 |
| 1500,1536,384 | 1.6719 | 1.6715 | 1058–1059 |

Per-call process CPU time divided by wall time was **0.999–1.000**, so the
observed throughput is not explained by multiple simultaneously busy CPU
threads. The worst sampled FP64 absolute error was 4.14e-6, 3.48e-6, and 1.47e-5,
respectively. Output checksums and raw timings are retained. This control
measures Accelerate capability; it does not establish which individual
whisper.cpp graph nodes use it or identify its private internal instructions.
**AMX/SME use was not directly observed and is not asserted.** The operative
comparison remains the matched CPU-only whisper.cpp full-call timing.

## Recommendation

Preserve the exact implementation and these negative results. The highest
leverage next investigation is a CPU backend/node audit and instruction
capability expansion, or a separately accuracy-tested algorithmic change at
full context. The Go 1.27.1 arm64 archsimd API exposes FP32/FP64 and integer
vectors, but no Float16/BFloat16 type, DotProduct, or widening multiply-add.
Available Int8 `MulWidenLo` plus widening/reduction adds instructions; it does
not currently provide a credible >2x arithmetic-throughput path over FP32 FMA.
Avoid another full quantized model prototype until a kernel-only throughput
probe or new supported intrinsic establishes that leverage.

## Reproduction and files

Use `/Users/thesyncim/.codex/bin/project-env` and an agent-specific
`CODEX_AGENT_ID`. This run used `CODEX_PROJECT_CACHE_ROOT=/private/tmp/goinfer-build-cache`,
`GOMODCACHE=/private/tmp/gophonic-modcache`, `GOTOOLCHAIN=local`, and the binary
`/Users/thesyncim/go/pkg/mod/golang.org/toolchain@v0.0.1-go1.27.0.darwin-arm64/bin/go`.
The model is `/private/tmp/gophonic-tiny.en.gophonic`; the PCM fixture is
`testdata/whisper_jfk.pcm.f32le`.

Each `*-overlay.json` in the archive maps temporary source files over repository files.
Build tests with `go test -c -overlay=...` through project-env, then run the
binaries with GOMAXPROCS=1. Run Whisper binaries with working directory
`whisper` so fixture paths resolve. `strassen-overlay.json` and
`TestStrassenFastWindowPaired` reproduce the full-context result despite the
historical test name. No reduced-context API is called there. The completed
historical L608 profile in `profile.json` is diagnostic background only; its
overlay needs the unmerged research branch `d4130b8` and is not a target or
performance claim. All full-shape and full-call results are usable on 6635909.

[`summary.json`](summary.json) collates medians, bootstrap interval, tokens,
and FLOP accounting. Raw JSON, source generators, emitted assembly, and
reference C source are in [`raw.tar.gz`](raw.tar.gz), along with
`SHA256.json` pinning the experiment inputs and artifacts.

## Go 1.27.1 follow-up

The official Go 1.27.1 patch toolchain passed the full scalar and SIMD suites
with the official model. Its arm64 `archsimd` API file is byte-identical to
Go 1.27.0. Four alternating one-core blocks of the original full-context JFK
call measured **779.44 ms** median with Go 1.27.0 and **778.64 ms** with
Go 1.27.1 (ratio **0.9990**). Both binaries used the same source and model;
each block contained five warm calls and three timed calls. The difference is
too small to attribute to a compiler speedup. The project now recommends
Go 1.27.1 through the `toolchain` directive while retaining Go 1.27.0 as
the minimum version. Per-call durations and hashes are in
[`go1271-ab.json`](go1271-ab.json).
The three attached data artifacts are pinned in [`SHA256SUMS`](SHA256SUMS).
