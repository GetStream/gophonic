# Exact CPU decoder row packing

A one-token Qwen projection used to fill a sixteen-row activation tile,
zero its fifteen unused rows, then copy the live row into the contiguous
buffer consumed by the SME row kernel. The new path packs directly into that
existing contiguous buffer with NEON. It uses the same power-of-two scaling,
FP32 multiplication, FP16 conversion, packed weights, and SME accumulation.

For eligible F16 projections, the lane owner packs the row before publishing
the projection job. Workers read it without mutation until the completion
barrier. Mutable state stays within its existing lane; immutable model
weights remain shared. The change adds no buffers, quantization, speculative
acceptance, or cache compression. It preserves the general tile path for
prefill, int8 weights, unsupported widths, and CPUs without SME.

Packing writes fall from `34*K` to `2*K` bytes: the sixteen-row FP16 tile and
its separate live-row copy become one FP16 row. At K=2048 that is 68 KiB to
4 KiB per projection. This is a 17-fold reduction in those writes, not a
17-fold reduction in transcription time or resident memory. Existing tile
storage remains available for prefill. Warm calls still allocate nothing.

## Measurement method

Baseline is main at `c19bb3f`, on an Apple M4 Max with Go 1.27.1,
`GOEXPERIMENT=simd`, `CGO_ENABLED=0`, and `GOMAXPROCS=16`. The official
Qwen3-ASR-1.7B checkpoint uses the existing exact-weight F16 format.

Separate-process measurements varied substantially with desktop activity.
We therefore also compare both paths within the same warmed lane, alternating
baseline/candidate and candidate/baseline. Every measured operation is the
complete public `Transcribe` call, from PCM through audio encoding, decoding,
and text production. The ordinary non-audio prompt-prefix reuse applies
identically to both paths; no audio or transcript result cache is added.

The temporary [instrumentation patch](benchmarks/cpu-stt/compact-crossover.patch)
only adds a switch between the old packing path and the candidate, plus a test
harness. It is not compiled into production. Each clip warms three times,
then runs 64 paired blocks. An untimed SHA-256 comparison after every call
checks all encoder embeddings, the final hidden state and logits, generated
token IDs, and transcript bytes. Any mismatch fails the run. Expected hashes
also match the independently built main binary. All samples are retained;
there is no fastest-run filtering. No profiling, builds, or other project
tests overlap the timed experiment.

[Raw samples and binary hash](benchmarks/cpu-stt/compact-crossover.json)
include all 256 measured transcriptions. Intervals resample paired blocks
10,000 times with seed 0. They describe these fixtures and this host, not an
accuracy corpus or a competitive ranking.

| Complete transcription | Baseline median | Candidate median | Median paired candidate/baseline | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| English JFK | 815.55 ms | 818.06 ms | 0.9854 | 0.9743–0.9987 |
| Chinese | 317.90 ms | 314.60 ms | 0.9836 | 0.9766–0.9909 |

The median paired reductions are 1.46% and 1.64%. The individual distribution
medians for JFK are essentially flat; pairing neighboring calls accounts for
changing host load. This is a small latency improvement with a narrow English
margin, not a large end-to-end breakthrough. An earlier 36-block comparison
with additional experimental variants found 2.54% and 2.18%; the final isolated
64-block run above is the reported result.

The [packing microbenchmark](benchmarks/cpu-stt/compact-packing.txt) has five
samples per shape, zero B/op and zero allocs/op. Median K=2048 conversion falls
from 4,010 ns to 84.3 ns; K=6144 falls from 13,660 ns to 221.5 ns. These 48–62×
conversion gains are confined to this small stage; they are not transcription
speedups.

## Rejected experiments

The same investigation rejected fused gate/up activation work, private
completion counters, channel and pipe parking, and fused Q/K head completion.
A tighter, counterbalanced comparison showed that merely serializing the old
packing stage regressed the Chinese fixture by about 1%; adding head completion
regressed it by about 2%. Their apparently larger wins in earlier noisy runs
were not reliable. Neither change ships independently.

A version with per-worker compact row buffers added no resolved benefit over
the single immutable row and hit one intermittent exact-state mismatch. A
subsequent full run passed, but the cause remains unresolved. That prototype,
its extra native buffers, and all runtime experiment switches are excluded.

## Validation

The compact packer is compared with the original packer for K=16 through 6144,
large and tiny magnitudes, negative zero, infinities, NaNs, and arbitrary legal
subranges including scalar tails. Finite, infinite, and zero half encodings
match exactly; NaN payload identity is not promised between existing scalar
and hardware conversion paths. Normal inference remains finite. Final F16
matrix outputs are compared bit-for-bit, including partial output panels.
Tests also check that the unused tile stays untouched, compact mode rejects
incompatible consumers, and the workspace can return to multi-row prefill.

The full repository suite, relevant race checks, real-model encoder and token
oracles, streaming continuation, and warm zero-allocation checks pass. Real
model checks also run with `GOGC=10`. Generated assembly is reproduced from its
source and checked for an exact match. Non-SIMD package checks and vet also
pass. Linux amd64 SIMD and Windows amd64 scalar binaries cross-compile; no
runtime speed claim is made for those platforms.

## Reproduce

Use an isolated worktree and the project environment wrapper for Go commands.
Apply the instrumentation only to the experimental worktree:

```sh
git apply docs/benchmarks/cpu-stt/compact-crossover.patch
CODEX_AGENT_ID=stt-crossover GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/stt-crossover.test ./qwen3asr
cd qwen3asr
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  GOPHONIC_CROSSOVER_REPORT=/tmp/stt-crossover.json \
  /tmp/stt-crossover.test -test.run '^TestCPUCounterbalancedLatency$' -test.v -test.timeout=15m
```

For independent binaries, `BenchmarkCPUTranscribe` prints the same state
fingerprints. `tools/benchmark_stt_cpu_compare.py --require-state` rejects
missing or mismatched fingerprints while alternating pinned binaries. The
`BenchmarkF16RowPacking` microbenchmark isolates conversion work; it does not
substitute for complete transcription measurements.

To recompute the paired interval from the raw report:

```python
import json, random, statistics
report = json.load(open("docs/benchmarks/cpu-stt/compact-crossover.json"))
rng = random.Random(0)
for clip in ("jfk", "zh"):
    values = {v: [s["NS"] for s in report["samples"]
                  if s["Clip"] == clip and s["Variant"] == v] for v in (0, 3)}
    pairs = list(zip(values[0], values[3]))
    boot = sorted(statistics.median(after / before for before, after in
                  rng.choices(pairs, k=len(pairs))) for _ in range(10000))
    print(clip, statistics.median(b / a for a, b in pairs),
          (boot[249], boot[9749]))
```

## Follow-up after the prefix-cache merge

A further investigation used PR #37 at `7185106b80384b729100a45d55f8aeb25503fc4a`,
after integrating #36 from main, plus the
[row signal fix](benchmarks/cpu-stt/row-signal-fix.patch) as its baseline.
A warm-only profile, started after model loading and three transcriptions,
put 29.52% of sampled CPU time in the F16 row kernel, 15.19% in the F16 tile
kernel, and 28.47% in the worker loop. These are CPU sample shares, not wall-time
fractions.

This investigation reproduced a correctness bug in the existing F16 row
kernel under profiling signals, garbage collection, and worker oversubscription.
A signal between panels could clear the upper row-scale lanes before the next
panel reset its sentinel. The F16 and int8 row kernels now rebuild their live
predicates and row scale inside each panel's sentinel-protected region. Their
checks first move an upper vector block into the signal-preserved low block,
then read it into a general-purpose register; a cleared predicate cannot hide
a retry. The F16 stress test failed three of 30 repetitions before the fix.
Both row stress tests passed 40 repetitions each after it. This correction
adds no storage and preserves the existing arithmetic and precision.

The follow-up passes the full SIMD repository suite, q8gemm/qwen3lm race
checks, real-model encoder/token/streaming and warm allocation checks with
`GOGC=10`, non-SIMD package tests, and vet. Generated assembly reproduces
exactly. Linux arm64 SIMD and Windows amd64 scalar binaries cross-compile.

The performance experiments tested reordered F16 weight loads, a tighter row
loop, software prefetch, removal of an empty prefix-attention dispatch,
deferring one-token KV appends to the following output-projection barrier,
one-chunk SwiGLU claims, and two- or four-panel claims restricted to the large
vocabulary projection. Combinations were also tested. No additional
performance change is retained.

The selected scheduling combination received a fresh confirmation separate
from those discovery runs: 64 counterbalanced baseline/candidate pairs per
clip, with all 256 complete `Transcribe` calls retained. Every untimed state
fingerprint matched the original English and Chinese hashes above. The
[raw samples and metadata](benchmarks/cpu-stt/scheduling-confirmation.json)
record the baseline and patch identities, instrumented binary SHA-256,
Go 1.27.1, `GOEXPERIMENT=simd`, `CGO_ENABLED=0`, `GOMAXPROCS=16`, order, and
bootstrap procedure.

| Complete transcription | Baseline median | Experimental median | Median paired experimental/baseline | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| English JFK | 1,493.48 ms | 1,480.28 ms | 0.99891 | 0.98660–1.02165 |
| Chinese | 813.86 ms | 783.87 ms | 0.99090 | 0.97875–1.00492 |

Both intervals include no change. This run establishes no incremental latency
improvement. Other desktop workloads were present, and absolute latency
varied substantially between discovery and confirmation; these absolute
medians should not be compared with the earlier compact-packing table.
No coordinating-agent builds, profiles, or validation jobs overlapped the
confirmation. The speculative scheduling changes and all runtime switches
were removed from production.

The [experimental instrumentation patch](benchmarks/cpu-stt/scheduling-experiment.patch)
is retained only for reproducibility. In an isolated checkout of `7185106`,
apply saved copies of `row-signal-fix.patch` and then `scheduling-experiment.patch`.
Build and run with the same project-scoped Go environment:

```sh
CODEX_AGENT_ID=stt-scheduling GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/stt-scheduling.test ./qwen3asr
cd qwen3asr
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  GOPHONIC_ASTRA_REPORT=/tmp/stt-scheduling.json \
  GOPHONIC_ASTRA_VARIANTS=0,7 GOPHONIC_ASTRA_BLOCKS=64 \
  /tmp/stt-scheduling.test -test.run '^TestAstraCounterbalancedLatency$' -test.v -test.timeout=10m
```
