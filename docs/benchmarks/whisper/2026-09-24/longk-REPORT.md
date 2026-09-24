# Long-K packed GEMM experiments, 2026-09-24

Both candidates were rejected. No source changes from this lane remain in the repository. CPU was released to `/root/astra_vocab16`.

Environment: Apple M4 Max, darwin/arm64, stock Go 1.27.0, `GOEXPERIMENT=simd`, `CGO_ENABLED=0`, `GOMAXPROCS=8`. All builds used `/Users/thesyncim/.codex/bin/project-env`, `CODEX_AGENT_ID=astra_longk_kernel`, and `CODEX_PROJECT_CACHE_ROOT=/private/tmp/goinfer-build-cache`. Verified GOCACHE `/private/tmp/goinfer-build-cache/f3d97b1e269cddc2/go-build`; project identity `.git`.

## K blocking

Hypothesis: a K=1500/1536 sixteen-column panel occupies about 96 KB and may evict source rows; reuse a 128/256/384/512/768-value panel segment across all owned rows. Every output retains increasing-K FP32 FMA order; intermediate FP32 stores do not reassociate. The 1-row stream-reduction tails are unchanged. The initial candidate dispatch covered K>=1024 and M>=16, including conv2 as a regression guard.

Tests called the real `PackedB.Mul` path with a temporary runtime switch, alternating six block choices in a rotating order. Each choice has 12 measurements. Setup, packing, and warm calls are excluded; output is consumed. Shapes match the 32-row attention value tile and 188-row encoder shards.

| Shape | Block | Median ms | Relative latency |
|---|---:|---:|---:|
| attention_values_tile | 0 | 0.09385 | +0.00% |
| attention_values_tile | 128 | 0.09371 | -0.15% |
| attention_values_tile | 256 | 0.09328 | -0.61% |
| attention_values_tile | 384 | 0.09411 | +0.28% |
| attention_values_tile | 512 | 0.09360 | -0.26% |
| attention_values_tile | 768 | 0.09446 | +0.66% |
| encoder_contract_shard | 0 | 3.40695 | +0.00% |
| encoder_contract_shard | 128 | 3.45415 | +1.39% |
| encoder_contract_shard | 256 | 3.43830 | +0.92% |
| encoder_contract_shard | 384 | 3.41503 | +0.24% |
| encoder_contract_shard | 512 | 3.43182 | +0.73% |
| encoder_contract_shard | 768 | 3.40411 | -0.08% |
| conv2_shard | 0 | 2.53269 | +0.00% |
| conv2_shard | 128 | 2.57475 | +1.66% |
| conv2_shard | 256 | 2.56044 | +1.10% |
| conv2_shard | 384 | 2.55253 | +0.78% |
| conv2_shard | 512 | 2.56412 | +1.24% |
| conv2_shard | 768 | 2.54665 | +0.55% |

Block 0 is unblocked. No block size produced a useful improvement across the target shapes. The apparent -0.61% at attention with block256 accompanies +0.92% contraction and +1.10% conv2; it does not justify a full-window experiment.

## 8x8 tile

A separate overlay used eight source rows and eight output columns with the same packed layout and increasing-K FMA order. Two calls cover each 16-column panel, leaving existing 4-row and 1-row tails intact. It reduces explicit weight loads per output by reusing each vector over eight rows. It doubles source broadcasts per output and rereads each source row for the second half-panel. Each mode has 24 alternating measurements.

| Shape | Baseline 4x16 ms | Candidate 8x8 ms | Relative latency |
|---|---:|---:|---:|
| attention_values_tile | 0.09365 | 0.11303 | +20.69% |
| encoder_contract_shard | 3.39522 | 4.09415 | +20.59% |
| conv2_shard | 2.54342 | 3.06594 | +20.54% |

Both variants were exercised live through production-shaped `PackedB.Mul` dispatch. Inspection of the baseline, modified 4x16, and 8x8 disassembly found no vector stack spills inside their main reduction loops. The 8x8 body uses V0–V31 and 32 table lookups per four K values, compared with 16 table lookups for the 4x16 body, for the same 64 accumulators. The reduced weight-vector traffic did not compensate for extra broadcasts/source loads and address work.

## Correctness and allocation gates

For each candidate, these tests passed:

- `TestMulAgainstIndependentOracle` (independent float64 oracle, existing tolerance unchanged).
- `TestMulExactDyadicTails`, `TestMulSpecialValues`, `TestValidationAndZeroAllocations`.
- `TestLongKBlockedExact`: byte-identical results against the unblocked 4x16 path for K=1024/1025/1152/1499/1500/1501/1536/1537, row/column tails, padded strides, and an unaligned source offset.
- `TestLongKBlockedSpecialValues`: exact FP32 bits for cancellation, signed zeros, subnormal values, infinities, and a payload NaN on the long-K path.
- Warm `testing.AllocsPerRun` returned 0 for every tested shape and candidate block.

No scalar code changed. Official model stage/full-window/race gates were deliberately not run: both candidates failed the initial performance gate, and the parent required >=5% kernel gain before the full-window stage. Thus encoder/window speedup is **not measured**, and no end-to-end benefit is claimed. No candidate is retained.

## Evidence

- `shapes.json`: raw K-block rotating measurements.
- `tile-shapes.json`: raw 4x16/8x8 measurements.
- `candidate-kernel_arm64.go`: ungated K-block candidate.
- `experiment-kernel_arm64.go`, `overlay.json`: K-block runtime switch overlay.
- `tile-kernel_arm64.go`, `tile-overlay.json`: 8x8 runtime switch overlay.
- `longk_experiment_test.go`, `tile_experiment_test.go`: exact-output/zero-allocation tests and timed shape probes.
- `kernel4x16.asm`, `kernel8x8.asm`: generated ARM64 disassembly.
- `longk_window_experiment_test.go`: prepared but unrun encoder/window A/B harness.
