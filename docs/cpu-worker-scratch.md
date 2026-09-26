# Private Whisper attention memory

Whisper encoder attention now borrows each executing worker's existing matrix
scratch through `Executor.RowsWithScratch`. The worker reuses its own storage
between matrix and attention phases. No attention tile borrows a numeric buffer
from the process-wide scratch pool. Inputs become read-only at the phase
boundary; workers write disjoint query tiles and private score regions.

Scores and packed per-head K/V now occupy one lane-owned arena. It uses the
existing 64-byte-aligned bump allocator, with bounded padding and no per-region
free list. On Unix the numeric payload is anonymous mapped memory outside the
Go heap and GC pacing; other platforms retain the pointer-free heap fallback.
Small Go metadata remains on the heap. Closing the encoder stops its executor
before releasing the attention arena; explicit owner liveness protects borrowed
views during execution. Arithmetic, kernels, and reduction order are unchanged.

## Measured memory

Apple M4 Max, Go 1.27.1, SIMD enabled, eight-worker Whisper tiny.en:

| Payload | Before | After |
| --- | ---: | ---: |
| Duplicate global GEMM scratch retained after transcription | 1,540,096 B | 0 B |
| Worker matrix scratch arena | 1,572,864 B | 1,572,864 B |
| Attention scores and packed K/V | 6,150,144 B on Go heap | 6,150,144 B in one private arena |
| Attention/worker arenas after lane close | — | 0 B |

The scratch change eliminates **1.47 MiB** of duplicate retained storage in
this fresh-process fixture. The arena change moves **5.87 MiB** of existing
numeric payload outside the Go heap; that migration is not a reduction in total
payload bytes. This layout needs no alignment padding. These are payload counts,
not RSS or GC pause measurements, and pooled storage was process-wide rather
than a fixed overhead for every lane.

Raw [pooled](benchmarks/cpu-stt/whisper-worker-scratch-shared-memory.json),
[private scratch](benchmarks/cpu-stt/whisper-worker-scratch-private-memory.json),
and [final arena](benchmarks/cpu-stt/whisper-final-arena-state.json) reports are
retained. The final report covers 2/4/8/16 workers and multiple audio windows;
every lane releases both arenas and leaves the global scratch pool empty.

## Latency and exactness

The scratch-only experiment compared adjacent, alternating complete public
`TranscribeInto` calls on the same warmed lane against `5c03c5d` behavior, before
the attention arena migration. `CGO_ENABLED=0`, `GOMAXPROCS=16`. Every timed call
was followed by untimed bitwise checks of encoder output, final logits, hidden
state, self-KV and tokens, plus transcript equality.

| Workload | Pairs | Baseline median | Private scratch median | Paired ratio, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| JFK, 8 workers | 64 | 67.020 ms | 65.368 ms | 0.9885 [0.9737, 1.0033] |
| Multiple windows, 8 workers | 32 | 210.350 ms | 211.014 ms | 1.0220 [0.9776, 1.0751] |
| JFK, 2 workers | 24 | 94.324 ms | 95.396 ms | 0.9648 [0.9161, 1.0105] |
| JFK, 4 workers | 24 | 72.999 ms | 74.418 ms | 0.9939 [0.9558, 1.0351] |
| JFK, 16 workers | 24 | 74.106 ms | 74.135 ms | 0.9911 [0.9692, 1.0103] |

Ratios are geometric means of candidate/baseline pairs; intervals bootstrap
10,000 complete pairs with seed 0. **No additional latency gain is established.**
Keep this change for its measured memory and ownership benefit. The
[discovery](benchmarks/cpu-stt/whisper-worker-scratch-discovery.json),
[confirmation](benchmarks/cpu-stt/whisper-worker-scratch-confirmation.json), and
[analysis](benchmarks/cpu-stt/whisper-worker-scratch-analysis.json) retain all
samples. All 336 timed calls reported zero Go allocations.

After the arena migration, a separate final public-path check with `GOGC=10`
matched the previous baseline's complete numeric-state fingerprints at every
worker count. Four calls reported zero allocations; one reported four small
allocations totaling 240 bytes after forced GC. This does not promise that the
Go runtime never allocates. The permanent focused warm allocation tests pass.

Other experiments (Qwen paired-row GEMV, narrower Whisper SME tiles, and private
completion counters) did not establish a reliable end-to-end improvement and
are absent from production. No extra Qwen speedup is claimed by this follow-up.

## Validation and reproduction

The final source passes the full SIMD repository suite, race and non-SIMD tests
for `internal/whispergemm`, `whisper`, and `qwen3asr`, repository vet and CLI
build. Real Whisper model/reference, allocation, and lifecycle tests pass with
`GOGC=10`. Changed packages and their Qwen consumer cross-compile for Linux arm64
with SIMD and Windows amd64 without SIMD. These cross-builds are not runtime
performance measurements.

The [scratch experiment patch](benchmarks/cpu-stt/whisper-worker-scratch-experiment.patch)
applies to **`5c03c5d`**, in a separate checkout. It adds only experiment switches
and harnesses alongside the scratch change. Build using the project wrapper:

```sh
git apply /path/to/whisper-worker-scratch-experiment.patch
CODEX_AGENT_ID=whisper-scratch GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/stt-scratch.test ./whisper
cd whisper
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  STT_WHISPER_SCRATCH=/tmp/scratch.json \
  /tmp/stt-scratch.test -test.run '^TestCPUScratchDispatch$' -test.v
# Run each footprint variant in its own fresh process:
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  STT_SCRATCH_SHARED=1 STT_SCRATCH_MEMORY=/tmp/pooled.json \
  /tmp/stt-scratch.test -test.run '^TestCPUScratchFootprint$' -test.v
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  STT_SCRATCH_SHARED=0 STT_SCRATCH_MEMORY=/tmp/private.json \
  /tmp/stt-scratch.test -test.run '^TestCPUScratchFootprint$' -test.v
```

The [final arena check patch](benchmarks/cpu-stt/whisper-final-arena-state.patch)
instead applies to the final source. Build the same way and run from `whisper`
with `GOMAXPROCS=16 GOGC=10`, the model path, and
`STT_FINAL_ARENA=/tmp/final-arena.json`, selecting `^TestCPUFinalArenaState$`.
Its stored numeric fingerprints are specific to the tested M4 Max SIMD path.
Neither instrumentation patch is compiled into production.
