# One-worker NEON 2x32 GEMM, 2026-09-24

Status: accepted and installed in the shared tree after the matched one-worker gate. No custom compiler, unsafe code, assembly, precision change, or allocation is introduced.

## Hypothesis and emitted code

The 4x16 kernel uses sixteen FP32 vector accumulators and sixteen byte-table broadcasts per four reduction values. A 2x32 tile keeps the same sixteen accumulators while halving the broadcasts to eight, reusing each source broadcast across eight output vectors instead of four. It reads two existing packed panels, trading additional weight loads for fewer shuffles. The emitted inner loop uses V0–V31 without vector stack spills. Both versions visit each output in increasing K FMA order. The single-row path retains its existing four-stream order.

Full groups of four output rows use pairs of 2x32 tiles; the final one to three rows still call the original single-row path. Preserving these boundaries keeps outputs independent of executor worker count. One remaining complete 16-column panel still uses 4x16; column tails retain the existing single-row path.

## Environment

Apple M4 Max, darwin/arm64, stock Go 1.27.0, `GOEXPERIMENT=simd`, `CGO_ENABLED=0`; runtime GOMAXPROCS=1 and one explicit inference worker for all performance evidence. All Go builds/tests used `/Users/thesyncim/.codex/bin/project-env`, project cache `/private/tmp/goinfer-build-cache/f3d97b1e269cddc2/go-build`, and `CODEX_AGENT_ID=astra_longk_kernel`. The official tiny.en model is `/private/tmp/gophonic-tiny.en.gophonic`; matched-window audio/oracle are the repository JFK fixtures.

## Shape gate

The test calls production `PackedB.Mul` dispatch through a temporary runtime switch. Inputs/packing are identical, output is consumed, and 16 alternating baseline/candidate measurements are collected per shape after warming each call. Times exclude packing and setup.

| Shape | Baseline ms | 2x32 ms | Change |
|---|---:|---:|---:|
| attention_scores_tile | 0.09544 | 0.08963 | -6.09% |
| attention_values_tile | 0.09289 | 0.08794 | -5.33% |
| encoder_projection | 6.74187 | 6.18423 | -8.27% |
| encoder_expand | 27.04729 | 24.86654 | -8.06% |
| encoder_contract | 27.54410 | 25.70988 | -6.66% |
| conv2 | 20.44779 | 19.66727 | -3.82% |

## Matched one-worker evidence

Each benchmark uses one process and one reused workspace, with alternating AB/BA order. The window benchmark verifies the exact 24 text tokens and comma transcript after every call; the encoder benchmark compares every FP32 output bit to the original kernel. Twenty-four pairs follow eight window warm calls or six encoder warm calls.

| Work | Baseline median ms | Candidate median ms | Median paired change | Bootstrap 95% paired interval | Wins | Warm bytes/allocations |
|---|---:|---:|---:|---:|---:|---:|
| window | 823.205 | 782.410 | -5.00% | [-5.09%, -4.84%] | 23/24 | 0/0 |
| encoder | 672.565 | 634.336 | -5.63% | [-5.93%, -5.57%] | 23/24 | 0/0 |

The interval uses 10,000 bootstrap resamples of paired ratios with fixed seed 926. This measures an improvement in Go relative to its previous kernel, not a claim of beating whisper.cpp or reaching the 50% goal. No multicore speedup is claimed from these measurements.

## Production files

- `internal/whispergemm/kernel_arm64.go`: paired-panel dispatch, preserving four-row tail boundaries.
- `internal/whispergemm/kernel_wide_arm64.go`: new ordered-FMA 2x32 kernel.
- `internal/whispergemm/kernel_wide_arm64_test.go`: exact pre-existing reduction-order, special-value, padding/stride, and allocation tests.
- `internal/whispergemm/gemm_test.go`: independent oracle coverage across the 32-column boundary and dyadic tails through N=35.
- `internal/whispergemm/README.md`: dispatch and arithmetic contract.

The runtime experiment switch exists only in `/private/tmp` overlays, not in production source.

## Validation

Production source passed both complete package suites with the official model and the encoder allocation gate enabled:

```text
GOMAXPROCS=8 CGO_ENABLED=0 GOEXPERIMENT=simd GOPHONIC_WHISPER_MODEL=/private/tmp/gophonic-tiny.en.gophonic GOPHONIC_TEST_ENCODER_ALLOCS=1
project-env go test ./internal/whispergemm ./whisper -count=1
internal/whispergemm: PASS (0.452 s)
whisper: PASS (8.341 s)

GOMAXPROCS=8 CGO_ENABLED=0 GOEXPERIMENT= GOPHONIC_WHISPER_MODEL=/private/tmp/gophonic-tiny.en.gophonic GOPHONIC_TEST_ENCODER_ALLOCS=1
project-env go test ./internal/whispergemm ./whisper -count=1
internal/whispergemm: PASS (0.550 s)
whisper: PASS (23.735 s)
```

These include the official encoder/PyTorch stage checks, SIMD token-preservation check, full-file/window/silence/long-input transcription cases, worker-count identity, exact dyadic tails, special values, independent FP64 oracle, and zero-allocation assertions. `git diff --check` passed.

## Raw evidence

`shapes.json`, `window.json`, `encoder.json`, and `summary.json` contain timing samples/statistics. `kernel2x32.asm` contains emitted ARM64 code. `baseline-kernel_arm64.go`, `tile-kernel_arm64.go`, `overlay.json`, `window-overlay.json`, and the temporary test sources reproduce the same-process experiment. The production code was installed only after the full-window result passed.
