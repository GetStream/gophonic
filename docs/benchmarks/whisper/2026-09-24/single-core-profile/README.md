# One-worker Whisper stage profile

September 24, 2026. Apple M4 Max; Go 1.27.0, `GOEXPERIMENT=simd`,
`GOMAXPROCS=1`, explicit one-worker transcriber. Official tiny.en FP32 model;
JFK padded-window workload. Every call matched all 24 text token IDs, plus
prefix `[50257,50362]` and EOS `50256` in Go. C returned the same 24 text IDs
and EOS. The Go executor starts zero helper workers at this budget.

The largest target is the live `whispergemm.kernel4x16` GEMM kernel: **72.02%
of sampled CPU time**. The dominant QKV, QK/PV, MLP, and cross-KV matrix groups run near
**63–66 GFLOP/s** on this worker. Encoder attention and MLP dominate wall time.

## Stage wall time

These are medians of 16 Go calls after five warm calls. Go was instrumented
with `time.Now/time.Since` and CPU-profiled during the measured calls. They are
diagnostic timings, not a replacement for an uninstrumented, alternating
Go/C production comparison. Stage medians do not necessarily add to the
median total; nested rows must not be added again to their parent.

| Stage | Median ms |
| --- | ---: |
| Complete Go window | 827.683 |
| Frontend | 10.605 |
| Encoder | 668.473 |
| Cross-KV preparation | 57.337 |
| Token loop, greedy policy, prompt | 90.546 |

| Encoder work, summed across four blocks | Median ms |
| --- | ---: |
| Stem convolutions, activation and position addition | 32.730 |
| Attention normalization | 7.332 |
| Q/K/V projections and biases | 83.312 |
| Attention, including packing | 261.213 |
| Output projection, bias, residual | 29.186 |
| MLP normalization | 7.348 |
| MLP expansion | 107.958 |
| MLP bias and GELU | 26.166 |
| MLP contraction | 108.727 |
| MLP bias and residual | 1.918 |
| Final normalization | 1.833 |

| Nested work | Median ms |
| --- | ---: |
| Attention scaling and K/V packing | 3.409 |
| Attention QK products | 110.157 |
| Attention softmax | 41.429 |
| Attention probability/value products | 105.710 |
| Cross-KV matrix products | 53.951 |
| Cross-KV bias, transposition and scaling | 3.396 |
| Decoder self attention, projections and residuals | 5.662 |
| Decoder cross attention, projections and residuals | 30.738 |
| Decoder MLP, norms, activation and residuals | 12.852 |
| Decoder vocabulary projections, 26 positions | 38.032 |

For the same 24-token workload, the optimized CPU C helper measured **434.031
ms** median over eight calls after five warm calls, with both `n_threads=1`
and `VECLIB_MAXIMUM_THREADS=1`. Its build has Accelerate and Apple BLAS on,
Metal/CoreML off, native CPU optimization on; runtime selected BLAS and flash
attention. The model's `ftype=0`. The environment records a requested library
thread limit; this probe did not inspect actual internal library scheduling.

| C internal timing counter | Median ms | Boundary |
| --- | ---: | --- |
| Mel | 13.540 | PCM frontend |
| Encode | 330.220 | Includes cross-KV preparation |
| Decode | 75.505 | Sum of 24 decoder calls |
| Batched decode | 6.280 | Two prompt tokens |
| Sample | 7.950 | Token sampling and policy |

C counters are printed to 0.01 ms. These labels differ from Go boundaries:
compare Go encoder plus cross-KV with C encode, and Go full token loop with C
decode plus batched prompt plus sampling. Both run the official greedy comma
transcript, but matching FP32 weight storage does not imply identical runtime
activation precision or reduction order.

## Kernel evidence and priorities

The CPU profile captured 11.51 seconds of samples over 13.36 seconds elapsed.
Top flat symbols were `kernel4x16` 72.02%, `vector4` 5.30%, `vector8` 3.91%, and
`softmaxFourRows` 3.82% (5.56% cumulative). Profile percentages describe sampled
CPU work, not percentages of elapsed time or removable scheduler latency.

1. Tune `kernel4x16` and actual shape dispatch first. Live disassembly contains
   vector FMA, 16 `VTBL` broadcasts per four reduction values, and no vector
   spills inside the main reduction loop. It also synthesizes four table
   constants at every microkernel entry and retains bounds branches in the
   reduction loop. Short-K attention QK makes setup amortization relevant;
   long-K MLP contraction and attention PV make cache layout and blocking
   relevant. Benchmark both short and long K before accepting a variant.
2. Keep changes to fast math attributable. Softmax costs 41.43 ms and MLP
   bias/GELU costs 26.17 ms. Together they are substantially smaller than dense
   multiplication. `ReduceMin` contributes 1.48% cumulative CPU samples;
   GELU's existing scalar fallback is live. Any changed approximation needs
   current numerical and exact token gates; this profile alone establishes
   no permissible relaxation.
3. Address decoder cross-attention and vocabulary after dense single-core
   work. They cost 30.74 and 38.03 ms respectively over the full token loop.
   Vocabulary streams about 2.07 GB of logical FP32 weights over 26 positions,
   so storage/conversion experiments need complete-loop evidence.

No inference worker is parked in this single-worker executor. Do not derive
synchronization savings from operating-system thread wait samples or from
the difference between profile sample duration and wall duration.

## Reproduction and files

`metadata.json` records commits, configuration and model hashes from the
existing matched model audit. `source-sha256.json` pins the four production
files instrumented by the temporary overlay. `prepare.py` creates instrumented
copies under `/private/tmp/goinfer-singlecore-profile` without modifying
production source. It contains the local repository/model paths used here.
Reproduction requires that source snapshot and the same official model files.

From the repository root:

```sh
python3 docs/benchmarks/whisper/2026-09-24/single-core-profile/prepare.py
CODEX_AGENT_ID=astra_singlecore_profile CODEX_PROJECT_CACHE_ROOT=/private/tmp/goinfer-build-cache GOEXPERIMENT=simd /Users/thesyncim/.codex/bin/project-env go test -overlay=/private/tmp/goinfer-singlecore-profile/overlay.json -c -o /private/tmp/goinfer-singlecore-profile/profile.test ./whisper
```

Run the test binary from the repository's `whisper` directory:

```sh
CODEX_AGENT_ID=astra_singlecore_profile CODEX_PROJECT_CACHE_ROOT=/private/tmp/goinfer-build-cache GOEXPERIMENT=simd GOMAXPROCS=1 /Users/thesyncim/.codex/bin/project-env /private/tmp/goinfer-singlecore-profile/profile.test -test.run '^TestSingleCoreStageProfile$' -test.v
```

Build and run the C helper with the already configured optimized C reference:

```sh
c++ -O3 -std=c++17 -I/private/tmp/gophonic-whisper.cpp/include -I/private/tmp/gophonic-whisper.cpp/ggml/include docs/benchmarks/whisper/2026-09-24/single-core-profile/cpp-stage.cpp -L/private/tmp/gophonic-whisper.cpp/build-cpu-accelerate/bin -Wl,-rpath,/private/tmp/gophonic-whisper.cpp/build-cpu-accelerate/bin -lwhisper -o /private/tmp/goinfer-singlecore-profile/cpp-stage
VECLIB_MAXIMUM_THREADS=1 /private/tmp/goinfer-singlecore-profile/cpp-stage /private/tmp/gophonic-ggml-model-f32.bin testdata/whisper_jfk.pcm.f32le 1 8
```

`go-stage-samples.json` and `cpp-summary.json` retain every measured sample.
`cpp-stage.log` retains the internal counters and token IDs. `cpu.pprof`,
`pprof-top.txt`, `pprof-cum.txt` and `kernel4x16.asm` retain profile and dispatch
proof. The instrumented binary remains at
`/private/tmp/goinfer-singlecore-profile/profile.test` for local profile use.
`stage-accounting.json` confirms top-level Go stages leave only ~2 microseconds
outside their sum; attention tile clock/loop overhead is ~0.21 ms per window.
