# CPU STT performance and memory

This change removes unnecessary Whisper vocabulary work and moves the largest
CPU inference payloads outside the Go heap. It preserves existing numerical
precision. It does not establish a new competitive ranking, and it does not
speed up Qwen's matrix arithmetic.

## Initial step measurements

Baseline: `origin/main` at `0b9bb91`. Apple M4 Max, macOS arm64,
Go 1.27.1, `GOEXPERIMENT=simd`, `CGO_ENABLED=0`, GOMAXPROCS 16. Whisper
transcribers use eight workers; Qwen CPU stages use at most eight active
workers. Prompt-only Whisper measurements use one worker.

Both revisions used identical benchmark sources and local official weights.
Six paired blocks alternate execution order; each sample includes 10 warm
iterations for Whisper or eight for Qwen. Setup and weight packing are outside
the timed loops. No profiling or other tests ran alongside these measurements.
The machine was running ordinary desktop applications, so paired comparisons
are more useful than comparing these absolute latencies to earlier reports.

| Workload | Main median | Candidate median | Candidate/main, bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| Whisper tiny.en, 33 s repeated JFK, full multi-window transcription | 215.07 ms | 191.31 ms | 0.872–0.901 |
| Whisper tiny.en, 64 prompt tokens + next greedy token | 83.98 ms | 36.46 ms | 0.416–0.441 |
| Whisper tiny.en, 128 prompt tokens + next greedy token | 165.99 ms | 68.93 ms | 0.408–0.423 |
| Whisper tiny.en, JFK full-file transcription | 63.76 ms | 64.78 ms | 0.987–1.062 |
| Whisper tiny.en, JFK padded 30 s window | 57.01 ms | 57.31 ms | 0.982–1.017 |
| Qwen3-ASR-1.7B F16, JFK | 675.60 ms | 688.13 ms | 0.954–1.035 |
| Qwen3-ASR-1.7B F16, Chinese | 255.45 ms | 256.69 ms | 0.944–1.028 |
| Qwen3-ASR-1.7B F16, decoder token including logits | 15.80 ms | 15.61 ms | 0.955–1.048 |

The multi-window Whisper reduction is 11.0%; its long prompt work is
2.3–2.4 times faster. Short Whisper clips and Qwen latency show no statistically
resolved change in this run. The longer Whisper input is a synthetic workload
formed by repeating the 11-second fixture three times, not an accuracy corpus.

Raw samples, binary hashes, and intervals:
[Whisper](benchmarks/cpu-stt/whisper.json),
[Qwen](benchmarks/cpu-stt/qwen.json).

## Initial step memory

These first-step numbers describe commit `af8387c`, before the single-copy KV
follow-up below. They are separate measurements, with explicit GC before each live-heap
snapshot and two warm lanes kept alive on one shared model. Heap figures are
increases over the empty test process. Peak RSS comes from macOS `time -l` for
the complete load-and-two-lanes test; it is not steady-state model size.

| Go live heap | Main | Candidate |
| --- | ---: | ---: |
| Whisper tiny.en, loaded model | 144.07 MiB | 0.02 MiB |
| Whisper, model + one warm lane | 333.77 MiB | 153.64 MiB |
| Whisper, model + two warm lanes | 464.48 MiB | 219.21 MiB |
| Qwen3-ASR-1.7B F16, loaded model | 3,902.88 MiB | 12.31 MiB |
| Qwen, model + one warm lane | 4,272.96 MiB | 100.31 MiB |
| Qwen, model + two warm lanes | 4,642.25 MiB | 187.54 MiB |

Whisper peak RSS fell from 441.06 to 412.55 MiB (6.5%). Qwen peak RSS was
approximately unchanged: 4,386.92 versus 4,369.08 MiB. Mapping a payload removes
it from Go heap accounting; it does not remove its physical-memory cost.
[Raw memory and resource reports](benchmarks/cpu-stt/memory.json).

The mapped payloads are:

- Whisper's validated FP32 model tensors, encoder activation slab, and each
  GEMM executor's private worker scratch regions.
- Qwen F16 decoder/head weights and CPU encoder weights, packed directly into
  one arena per component with no intermediate packed-weight copy.
- Qwen decoder and encoder activation slabs and, initially, the raw per-layer prefix KV
  store (replaced by the canonical packed store below). Anonymous pages are committed as used, so reserved KV capacity does
  not need to be eagerly touched.

This is not a heap-free runtime. Tokenizer data, metadata, quantization tiles,
temporary attention packing, Whisper packed weight copies and decoder buffers,
and other small workspaces still use Go allocations. Numeric heap slices are
already non-scannable; moving their backing storage primarily changes GC
pacing and heap accounting. The existing CPU int8 decoder format is unchanged.

## Single-copy Qwen KV follow-up

The CPU prefix cache now owns its final attention layout directly in one
bounded arena per lane. The old path retained both row-major KV and a packed
copy, then repacked the unfinished value chunk on every decoded token. The
new path writes each projected row once into the canonical cache after the
attention stage completes. Worker tasks write disjoint KV-head groups; no
global cache pool or implicit cross-lane prefix sharing is introduced.

Keys append columns in 16-column panels. Complete new panels use the existing
SME transpose kernel; partial panels preserve their preceding columns. Values
use 256-token pages with a fixed physical pitch and a separate logical row
count. Appending one value row never relocates older rows or repacks the tail.
The SME kernel takes the physical pitch as an additional argument, while its
reduction loop still visits only initialized rows in the original order.
NEON and scalar kernels consume those same pages through one-panel views.
There is no quantization or changed softmax/chunk reduction order.

A cache reserves virtual storage for its requested capacity once, with pages
committed on access. In-use keys round to 16 columns; on a 16 KiB-page system,
touching a value panel can commit its entire 256-token page. That tail waste
is bounded, but it matters for short prefixes. Truncation reuses capacity;
Close releases the entire mapping. Headers and token IDs remain on the heap.
This design avoids numeric-cache heap growth and interior allocation holes,
not all process allocations or all unused capacity. It preserves the existing
explicit read-only prefix API for direct evaluator users; STT lanes each own
an independent cache. Prefix copies into already-used destinations replace
both payload and dimensions; consumers no longer lazily mutate the cache.

Measurements below compare the preceding commit `af8387c` with this follow-up
on the same machine and flags. Three paired blocks each load one model and
keep two JFK transcription lanes alive, with GC before each heap snapshot.

| Qwen F16 memory metric | Previous step | Single-copy KV |
| --- | ---: | ---: |
| Go live heap, model + one warm lane | 100.31 MiB | 37.36 MiB |
| Go live heap, model + two warm lanes | 187.54 MiB | 61.64 MiB |
| Two-lane live RSS, median | 4,360.55 MiB | 4,383.50 MiB |
| macOS peak physical footprint, median | 4,269.32 MiB | 4,208.33 MiB |

The two-lane Go heap falls 67.1%, and macOS peak physical footprint falls
about 61 MiB (1.4% of the model-dominated process). RSS instead rises about
23 MiB (0.5%); these are different OS accounting metrics, and this result must
not be described as an RSS reduction. A separate matched diagnostic with
`debug.FreeOSMemory` after GC retains the same direction: peak physical
footprint 4,268.91 to 4,194.71 MiB, RSS 4,353.52 to 4,394.31 MiB. Scavenging is
opt-in measurement code, not production behavior. `vmmap -summary` also
reported lower dirty memory/physical footprint, without identifying every
cause of the RSS difference.

Six paired blocks of 16 real cached decode steps, including the vocabulary
head, give the following synthetic context-length results. Prompts use the
JFK audio embeddings followed by repeated valid text tokens. Warm steps have
zero allocations; this is a scaling benchmark, not an accuracy corpus.

| Prefix tokens | Previous median | New median | New/previous, bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| 256 | 14.79 ms | 14.88 ms | 0.974–1.033 |
| 512 | 15.57 ms | 15.92 ms | 0.918–1.101 |
| 2,048 | 20.72 ms | 20.42 ms | 0.764–1.013 |

These timings do not resolve a latency improvement. A subsequent full-clip
comparison encountered a roughly tenfold slowdown already in the unchanged
baseline and was stopped before any paired result; it is not evidence for a
candidate speed claim. The accepted improvement here is cache ownership,
duplicate removal and heap/physical-footprint reduction.

Raw reports: [memory](benchmarks/cpu-stt/kv-memory.json),
[scavenging diagnostic](benchmarks/cpu-stt/kv-memory-scavenged.json),
[context-length latency](benchmarks/cpu-stt/kv-latency.json).

The design draws on established blocked storage and virtual-memory ideas.
[PagedAttention](https://arxiv.org/abs/2309.06180) manages KV in blocks;
[vAttention](https://arxiv.org/abs/2405.04437) separates virtual layout from
physical allocation. Those papers target GPU serving. This implementation
uses ordinary CPU anonymous mappings and private lane ownership; it does not
claim their algorithms, cache-sharing policies or a new competitive record.

## Ownership and lifetime

`internal/arena` is a bounded bump allocator over anonymous memory on Unix.
Each region is rounded to 64 bytes and cannot append into the next region.
There are no per-object frees or interior holes: a capacity change replaces a
complete slab only between operations. Padding is at most 63 bytes per region,
plus the OS page rounding of each mapping. Growth temporarily holds the old
and new slab; it does not copy activation contents. Non-Unix platforms retain
the existing pointer-free heap fallback.

Workers own disjoint writable outputs and scratch regions. Immutable weights
remain shared. The GEMM executor no longer borrows scratch from the global
channel pool. Its completion barrier runs before a slab can be replaced or
released. Qwen allocates attention scratch only for participants that can
actually execute attention.

Owners retain their arenas across assembly and worker calls using explicit
`runtime.KeepAlive` guards. Close joins workers before unmapping. Model loads
release failed allocations; cleanups are a backstop for abandoned owners.
Never put Go pointers in an arena or keep a borrowed numeric view past its
owner's lifetime. Close lanes before their model. Direct Whisper users now
have `Model.Close`; `gophonic.Model.Close` invokes it automatically.

```go
model, err := whisper.Load(path)
if err != nil { return err }
defer model.Close()
lane, err := whisper.NewTranscriber(model)
if err != nil { return err }
defer lane.Close() // runs before model.Close
```

Go's `GOMEMLIMIT` does not account for Unix mappings. Services must budget
resident model memory, maximum lane capacity and temporary growth separately;
this work does not add an admission controller or a process RSS limit.

## Exact work removal

Whisper still computes every prompt token's layer state and KV entries. It
omits only the final normalization and vocabulary projection when no caller
uses that position's logits. Full transcription retains SOT logits for the
no-speech rule and final-prompt logits for generation. Word alignment retains
every probability it consumes. An explicit differential test verifies
bit-identical subsequent logits after skipped projections, including short
and full audio cache shapes.

Immutable encoder packing is cached once per Whisper model. On accelerated
cross-attention, raw preparation buffers hold one layer instead of duplicating
the complete packed cache for every layer. No precision, vocabulary filtering,
activation rounding or greedy policy changed.

Fixed equal Qwen worker ranges regressed one-token latency from about 16 ms
to 21–22 ms. Coarser dynamic claims and a longer Whisper spin interval also
regressed the measured workload. They were discarded; fine-grained balancing
remains where it was faster.

## Validation and reproduction

Validation includes the complete repository suite with SIMD, the relevant
packages without the SIMD experiment, race checks for arena lifetimes,
borrowed buffers and worker dispatch, and the installed official model tests
under `GOGC=10` (Whisper tiny/base and Qwen3-ASR CPU/GPU).
The portable Qwen evaluator is exercised by its scalar-reference tests.
Linux amd64 SIMD and Windows amd64 scalar test binaries cross-compile;
these are compilation checks, not measurements on those CPUs.

The model gates check independent reference encoder activations, exact greedy
transcripts and warm zero-allocation calls. Off-heap storage has explicit
zero/overflow, alignment, non-overlap, guard-region, repeated Close, forced-GC,
workspace growth and prefix-copy coverage. The single-copy cache adds reused
copy/branch tests at 0/1/15/16/17/255/256/257-token boundaries, forced GC,
concurrent explicit read-only consumers, and FP16/int8 evaluator reference
comparisons. Fixed-pitch matrices fill unused rows with NaNs and compare
live products against an independent FP64 oracle; canaries check output and
allocation boundaries on SME and portable dispatch.

Build test binaries on the baseline and candidate through the required
`project-env` wrapper, with the same flags. Copy `whisper/cpu_benchmark_test.go`,
`qwen3asr/cpu_memory_test.go`, and `qwen3asr/benchmark_test.go`
into the baseline checkout so both execute the same instrumentation.

```sh
CODEX_AGENT_ID=stt-cpu GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/whisper.test ./whisper
CODEX_AGENT_ID=stt-cpu GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/qwen.test ./qwen3asr

python3 tools/benchmark_stt_cpu_compare.py /tmp/whisper-before.test /tmp/whisper.test \
  --cwd whisper --models /path/to/models --samples 6 --iterations 10 \
  --bench 'Benchmark(OfficialTinyEN(Transcribe|Window)$|CPUContextPrompt|CPURepeatedAudio)' \
  --output /tmp/whisper-comparison.json
python3 tools/benchmark_stt_cpu_compare.py /tmp/qwen-before.test /tmp/qwen.test \
  --cwd qwen3asr --models /path/to/models --samples 6 --iterations 8 \
  --bench 'Benchmark(Transcribe|Decoder)/f16' --output /tmp/qwen-comparison.json
```

Run memory reports from the corresponding package directory with
`GOPHONIC_MEMORY_REPORT=1 GOPHONIC_MODELS=/path/to/models`,
`-test.run '^TestCPUMemoryFootprint$' -test.v`, and `/usr/bin/time -l` on macOS.
Do not run benchmarks alongside tests, profiling, or another benchmark process.

For the cache follow-up, copy `qwen3asr/kv_benchmark_test.go` and the updated
memory report into the reference checkout, build both binaries identically,
and run:

```sh
GOMAXPROCS=16 python3 tools/benchmark_stt_cpu_compare.py /tmp/qwen-before.test /tmp/qwen.test \
  --cwd qwen3asr --models /path/to/models --samples 6 --iterations 16 \
  --bench '^BenchmarkDecoderPrefixLength$' --output /tmp/kv-latency.json
```

`GOPHONIC_HEAP_PROFILE=/tmp/live.heap` optionally writes a warm live-heap
profile. Qwen's report supports `GOPHONIC_MEMORY_SCAVENGE=1` for the separate
forced-scavenge diagnostic and `GOPHONIC_VMMAP_REPORT=1` for a macOS VM summary.
