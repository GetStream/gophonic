# Exact CPU streaming improvements

The follow-up [decoder continuation work](cpu-continuation-exact.md) reuses
exact audio KV blocks while preserving the original attention reductions, and
reduces small Whisper decoder dispatches.

This follow-up to the one-row packing change targets work repeated across
Qwen streaming partials. It preserves the existing F16 weight representation,
FP32 accumulation order, token verification, and public transcript semantics.
The baseline is PR #37 at `2086220ac9534e13450e8ead26afb84c3aadc91b`, including
main's merged prefix-cache work and the row-kernel signal correction.

## Reuse vocabulary weights across draft states

The CPU verifier previously called the vocabulary projection independently
for every draft state. Thirty-two states traversed the approximately 600 MB
head thirty-two times. Ordinary multi-row matrix multiplication changes the
FP32 reduction tree, so it cannot satisfy the bit-exact single-row contract.

The new SME kernel assigns half of ZA's vector groups to each of two rows.
Each weight load feeds both rows while each row keeps its original eight
partial sums, FDOT pair rounding, ordered additions, and scaling. An outer
panel loop keeps a weight panel cache-local across up to sixteen rows. An odd
last row uses the original kernel. This combines register-level weight reuse
with cache reuse without changing a single row's arithmetic.

The workspace packs contiguous rows into its existing activation storage.
No new numeric buffer is allocated. Workers own disjoint output panels;
packed inputs and weights are immutable until the completion barrier. The
projection is enabled only for eligible CPU F16 SME heads. Other formats,
widths, and platforms keep their existing projection paths. Signal handling
rebuilds live streaming state per panel and retries both rows after detecting
cleared upper vector lanes.

## Reuse completed encoder windows

Qwen's audio encoder uses independent attention windows. A continuing CPU
transcription can keep the embeddings for complete earlier windows when their
normalized mel features are identical. The cache compares the full feature
prefix bit-for-bit; there is no hash collision or approximate acceptance.
Changes in earlier audio or global mel normalization force a full encode.

The cached boundary must align both to whole attention windows and to the
sixteen-row activation tile. This preserves the row-versus-tile kernel choice
and FP32 reduction order in the recomputed suffix. In the measured 1.7B model,
the alignment is sixteen seconds. The final two STFT frames remain uncached
because right-edge reflection can change them when audio grows. The suffix
keeps the original global feature stride and chunk-padding geometry.

Reuse requires an explicit `Options.Partial` and strictly growing frame
counts. Shrinking, repeated-length, and fresh offline inputs do a full encode.
Decoder audio KV is still recomputed, preserving its original attention
reduction grouping. This is encoder work reuse within a live continuation,
not a whole-transcription result cache.

Each lane owns a pointer-free native feature snapshot whose retained capacity
is capped at 8 MiB.
The snapshot grows geometrically and is released with the lane. Growth briefly
holds the old and replacement mappings until allocation succeeds. Existing
output embeddings retain the valid prefix; there is no second embedding
cache. Allocation failure disables this optional reuse for that call.
Workers add no shared mutable cache or global allocator. On Unix the snapshot
payload is outside the Go heap; metadata remains ordinary small Go objects.

## End-to-end confirmation

The complete public `Transcribe` workload uses seven strictly growing PCM
inputs: 4, 8, 16.02, 17, 24, 32.02, and 33 seconds. The JFK recording is
repeated on a hop-aligned boundary to obtain the longer fixture. This is a
controlled continuation workload, not an independent long-form accuracy corpus.
Each call supplies the preceding transcript as `Options.Partial`.

Measurements use the Qwen3-ASR-1.7B F16 checkpoint on an Apple M4 Max with
64 GiB RAM, Go 1.27.1, `GOEXPERIMENT=simd`, `CGO_ENABLED=0`, and
`GOMAXPROCS=16`. A fresh 32-block comparison rotates all six execution orders
of three variants: original serial head/full encoder, batched head/full
encoder, and batched head/reused encoder. Every sample is retained. No
coordinating-agent builds, tests, or profiles overlap timing; other desktop
activity still adds noise.

| Seven-call trace | Median time | Paired ratio versus serial/full | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| Serial head, full encoder | 6,208.68 ms | 1.00000 | — |
| Batched head, full encoder | 5,491.26 ms | 0.96261 | 0.89750–1.03969 |
| Batched head, reused encoder | 5,039.24 ms | 0.83516 | 0.79465–0.88317 |

Both improvements reduce paired total latency by **16.5%**, with an interval
of 11.7–20.5%. The combined path also improves over head batching alone:
ratio 0.86760, interval 0.79807–0.93585. This longer experiment does not resolve
a separate head-only gain. The independent shorter English and Chinese
traces do; see [head batching measurements](cpu-head-batching.md).

| Audio at each call | Frames reused | Baseline median | Combined median | Paired combined/baseline | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 s | 0 | 335.60 ms | 329.19 ms | 1.08402 | 0.98067–1.22394 |
| 8 s | 0 | 407.61 ms | 379.59 ms | 0.98462 | 0.91016–1.07091 |
| 16.02 s | 0 | 789.99 ms | 749.84 ms | 1.02466 | 0.94531–1.11968 |
| 17 s | 1,600 | 919.49 ms | 753.82 ms | 0.87998 | 0.81099–0.96025 |
| 24 s | 1,600 | 1,116.72 ms | 926.87 ms | 0.85055 | 0.79567–0.91550 |
| 32.02 s | 1,600 | 1,395.83 ms | 1,163.36 ms | 0.78383 | 0.71711–0.84927 |
| 33 s | 3,200 | 1,092.19 ms | 702.68 ms | 0.61921 | 0.56350–0.67642 |

The last call's paired reduction is 38.1%, with an interval of 32.4–43.7%.
Earlier calls that cannot reuse a window show no resolved change in this
experiment. Individual distribution medians differ from the paired estimate
because host load varies. Ratios are geometric means of neighboring paired
ratios; intervals resample paired log-ratios 10,000 times with seed 0.

The [raw confirmation](benchmarks/cpu-stt/stream-prefix-confirmation.json)
contains all 96 traces and 672 timed public calls, per-call timings, exact
fingerprints, allocation results, binary/patch hashes, and analysis. The
[two-block discovery](benchmarks/cpu-stt/stream-prefix-discovery.json) is kept
separately and does not contribute to the reported intervals. These are CPU
streaming results for this host and fixture, not offline, Whisper, other-CPU,
or competitor performance claims.

## Exactness and memory

The existing non-audio prompt cache and lane capacities are warmed before
comparing variants. The warmed baseline must reproduce its own fingerprints
first. This matters because the existing cold and cached prompt paths group
attention differently. Every variant then compares embeddings, hidden states,
final and verification logits, verification tail states, generated tokens,
text, and language at all seven steps. Any bit mismatch fails the run. Every
timed call also checks text and language. The standard `testing.AllocsPerRun` check returns zero for all three traces;
that Go helper temporarily sets `GOMAXPROCS=1`. The final production benchmark
at 16 Ps also exposed small allocations outside those one-P gates; see the
allocation qualification below.

Independent full encodes match incremental encodes bit-for-bit across growing
real PCM, changing tail tiles, chunk/window boundaries, and shrinking input.
Tests reject invalid alignment, changed earlier features, and signed-zero
changes; they enforce the snapshot cap and check allocation behavior while
audio grows. In the real-PCM test through 33 seconds, the primary encoder
activation arena plus snapshot retains 38,541,312 bytes versus 52,254,720 bytes
for the full-encode activation arena: 26.2% less. This excludes weights,
decoder state, other encoder scratch, and process RSS. Capacity depends on
previously seen audio; an already-large arena is retained for reuse.

The full SIMD repository suite, q8gemm/qwen3lm/qwen3asr race checks, real-model
encoder/token/continuation checks under `GOGC=10`, non-SIMD package tests, and
vet pass. Linux arm64 SIMD and Windows amd64 scalar binaries cross-compile.
Generated SME/NEON assembly reproduces exactly. The new pair kernel passed
40 signal-stress repetitions; a warm-only public-call profile confirms the
kernel executes and preserves exact state despite thousands of signal retries.

### Allocation qualification

The final [production benchmarks](benchmarks/cpu-stt/stream-production-benchmark.txt)
at 16 Ps recorded six allocations / 672 bytes for the four-call JFK trace,
one / 112 bytes for Chinese, and one / 112 bytes for the seven-call trace.
These are real allocations, so the implementation does not promise a
GC-invisible Go scheduler or strict zero allocation for every multi-P call.

A post-GC [allocation-stack comparison](benchmarks/cpu-stt/head-batch-allocations.json)
confirms `runtime.acquireSudog → runtime.chanrecv → workerPool.loop` in both
serial and batched paths. Go allocates a 112-byte channel wait record when
its per-P and central caches are empty; runtime timer storage can also grow.
This pre-existing scheduler behavior is separate from the numeric payloads.
The four serial/batch/batch/serial trials measured 5/1/2/0 allocations and
464/112/224/0 bytes. They are diagnostic trials, not latency measurements.

`testing.AllocsPerRun` uses one P and integer-divides the count, so its recorded
zero does not prove multi-P allocation freedom. The new permanent allocation
tests now measure a single call or trace to avoid fractional masking; they
pass, still subject to the helper's one-P scope. The feature snapshot remains
outside Go's heap on Unix, and batching introduces no new numeric buffer.
No scheduler experiment or allocation-hiding warmup change ships.

## Reproduce

Use an isolated checkout of the final PR implementation. Apply both temporary
patches; the encoder experiment uses the head experiment's toggle and hash
helper. Production contains neither switch nor instrumentation counter.

```sh
git apply docs/benchmarks/cpu-stt/head-batch-experiment.patch
git apply docs/benchmarks/cpu-stt/stream-prefix-experiment.patch
CODEX_AGENT_ID=stt-stream GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/stt-stream.test ./qwen3asr
cd qwen3asr
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  STT_COMBINED_PAIRS=32 STT_COMBINED_OUTPUT=/tmp/stt-stream.json \
  /tmp/stt-stream.test -test.run '^TestCPUStreamCombinedExperiment$' -test.v -test.timeout=20m
```

The permanent `BenchmarkCPUStreaming` measures the production seven-call
trace without switches. To recompute the combined paired interval:

```python
import json, math, random, statistics
report = json.load(open("docs/benchmarks/cpu-stt/stream-prefix-confirmation.json"))
pairs = {}
for sample in report["samples"]:
    pairs.setdefault(sample["pair"], {})[sample["variant"]] = sample["ns"]
logs = [math.log(p["batch-prefix"] / p["serial-full"]) for p in pairs.values()]
rng = random.Random(0)
boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
              for _ in range(10000))
print(math.exp(statistics.mean(logs)), boot[249], boot[9749])
```
