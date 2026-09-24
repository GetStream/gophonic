# Lossless vocabulary storage experiment, 2026-09-24

Status: rejected for integration. All implementation and tests are archived in
this directory. The shared repository has no vector16 files or retained edits
from this experiment. CPU slot has been released.

## Contract and fairness

- Machine: Apple M4 Max, darwin/arm64; Go 1.27.0; GOEXPERIMENT=simd;
  CGO_ENABLED=0; GOMAXPROCS=1 and Executor(1) for final trials.
- Official converted tiny.en embedding: 51,864 rows × 384 input values =
  19,915,776 scalar products per call.
- Every variant uses the same original weights and the same 384-value input.
  Weight packing, model reads, output allocation, executor construction and
  per-group zero detection are outside timed loops for all contenders.
- Both baseline and candidate calls pass through Executor.Rows with one worker.
  Output is retained in benchmarkSink. testing.B.Loop protects live work.
- Baseline goes through MulVector to vector4: K=384 is below vector8's K>=512
  threshold and weight stride equals K.
- Original packing tests exhaust all 65,536 encodings, preserve signed zero,
  test unsupported FP32 fallback, unaligned inputs, row slices and tails, and
  compare the complete official embedding projection on four random inputs.
- Final specialized candidates compare the complete 51,864-output projection
  bit-for-bit on four random inputs per test run. Three repeats all passed.
- Both final candidates and FP32 measure 0 B/op and 0 allocs/op.
- The final compact16 candidate processes 16 input coefficients per iteration,
  matching baseline vector4's unroll, four output rows, eight FMA accumulators,
  FMA sequence and final pairwise reduction. It skips vector zero checks for
  12,928 groups and uses original FP32 vector4 for the 38 groups containing a
  zero (38 of 12,966 groups, 0.293%). No implicit full-matrix fallback occurs.

## Measured one-core results

Final benchmark: 150ms target time per subbenchmark, three repeats. The FP32
baseline runs before and after each candidate. Raw results are in
`bench-final-formats.txt`.

| Variant | FP32 median | Candidate median | Result |
|---|---:|---:|---|
| 16-bit, unroll16, no-zero groups | 1.464224 ms | 3.451719 ms | 2.357× latency |
| 24-bit, one byte-table unpack | 1.476082 ms | 1.442274 ms | 2.29% lower latency |

The earlier generic 16-bit path (unroll8, per-vector signed-zero reconstruction)
measured 7.65ms, then 4.90ms after hoisting shift vectors. Those timings exposed
avoidable implementation cost and do not describe the corrected final kernel.
The corrected result still fails the >=10% vocabulary-gain integration gate.
The 24-bit result would save only approximately 0.9ms across 26 vocabulary
projections, before model integration; it has no end-to-end performance claim.

## Generated-code audit

`asm-final.txt` contains the exact binary's three kernels. Count each steady
iteration over four rows × sixteen input coefficients (64 scalar products):

| Loop | Instructions | NEON FMAs | Vector loads | Decode instructions | Bounds comparisons |
|---|---:|---:|---:|---:|---:|
| FP32 vector4 | 53 | 16 | 20 | 0 | 4 |
| Corrected compact16 | 142 | 16 | 12 | 96 | 4 |
| Compact24 | 72 | 16 | 20 | 16 table lookups | 5 |

The compact16 decode consists of 16 widen operations, 32 shifts, 32 ORs and 16
ANDs. Broadcast vectors are outside the loop. All three loops have zero helper
calls and zero vector spills. Scalar row-pointer stack loads remain in both
baseline and compact16; ordinary bounds checks exist in both at equal count.
Thus the remaining loss comes from reconstruction instruction throughput and
latency, not hidden allocation, a slower FMA reduction, missing SIMD dispatch,
or a hidden FP32 fallback.

This is lossless weight storage followed by FP32 FMA, not native FP16 arithmetic.
The conclusion applies to these formats and this Go-generated implementation;
it does not establish a general limit on compact weights or reduced precision.
The 24-bit trial demonstrates that cutting unpack work to one instruction does
recover performance, but its measured gain is too small for this task.

## Reproduce

The Go overlay adds experimental files without modifying the shared package:

```sh
CODEX_PROJECT_CACHE_ROOT=/private/tmp/goinfer-build-cache \
CODEX_AGENT_ID=astra_vocab16 GOTOOLCHAIN=local GOEXPERIMENT=simd \
CGO_ENABLED=0 GOPHONIC_WHISPER_MODEL=/private/tmp/gophonic-tiny.en.gophonic \
GOMAXPROCS=1 /Users/thesyncim/.codex/bin/project-env \
/Users/thesyncim/go/pkg/mod/golang.org/toolchain@v0.0.1-go1.27.0.darwin-arm64/bin/go test \
-overlay=/private/tmp/gophonic-vector16/overlay.json ./internal/whispergemm \
-run '^(TestVector16NoZeroOfficialBitExact|TestVector24OfficialBitExact)$' \
-bench '^(BenchmarkVector16NonzeroOfficial|BenchmarkVector24Official)$' \
-benchtime=150ms -count=3 -v
```

The earlier generic encoding/tail suite is `-run '^TestVector16'`. Test binaries,
overlay, original/hoisted/final assembly and benchmark output remain beside this
report for further review. No production integration was attempted.
