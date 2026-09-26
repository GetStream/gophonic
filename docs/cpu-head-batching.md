# Exact batched CPU vocabulary projection

`LogitsRowsInto` now reuses each F16 vocabulary weight panel across the draft
states being verified. The former CPU loop traversed the roughly 600 MB
Qwen3-ASR-1.7B head separately for every row. A 32-row verification therefore
traversed roughly 19 GB of weights. The new order assigns one 64-column panel
to a worker, which processes all rows before claiming another panel.

A new two-row SME kernel shares each weight load between two independent
sets of eight FP32 partial sums. Each row retains the original FDOT pair
rounding, ordered final additions, column scaling, and inverse row scaling.
An odd remaining row uses the existing single-row kernel. This preserves
single-row results bit-for-bit, unlike substituting the general multi-row
matrix kernel, whose reduction order differs.

The existing activation tiles hold the contiguous rows. No additional
numeric buffer or weight representation is introduced. Mutable workspaces
and output panels belong to their lane or worker; packed activations and
weights remain immutable during projection. The path is limited to F16
weights, SME with the existing 512-bit dispatch, and widths divisible by 16.
Single-row calls, other formats, and other platforms retain their existing
computation. The new kernel reconstructs streaming predicates and scales
inside each panel's signal sentinel, and retries both rows after a detected
signal interruption.

## Fresh paired confirmation

Baseline is PR #37 at `2086220ac9534e13450e8ead26afb84c3aadc91b`, after the
prefix-KV merge and row-kernel signal correction. Measurements use the
Qwen3-ASR-1.7B checkpoint on an Apple M4 Max with 64 GiB RAM, Go 1.27.1,
`GOEXPERIMENT=simd`, `CGO_ENABLED=0`, and `GOMAXPROCS=16`.

Both paths run in the same warmed process and lane. Each of 64 fresh pairs
alternates serial/batch or batch/serial order. A preceding full transcription
warms the existing non-audio prompt prefix before state comparisons. The
[raw report](benchmarks/cpu-stt/head-batch-confirmation.json) preserves all
samples, fingerprints, allocation counts, binary SHA-256, and environment.
The eight-pair [discovery run](benchmarks/cpu-stt/head-batch-discovery.json)
is separate from this confirmation.

| Real prompt-tail states | Serial median | Batched median | Paired speedup | Paired batch/serial 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 3 rows | 6.776 ms | 2.655 ms | 2.56× | 0.38163–0.39688 |
| 8 rows | 18.039 ms | 3.944 ms | 4.53× | 0.21733–0.22596 |
| 16 rows | 36.135 ms | 6.423 ms | 5.62× | 0.17670–0.17958 |
| 32 rows | 72.313 ms | 11.701 ms | 6.20× | 0.16086–0.16195 |

The complete public-call workload transcribes one-quarter, one-half,
three-quarters, then all of a clip's PCM. Each call after the first supplies
the preceding transcript as `Options.Partial`. The audio strictly grows;
there is no repeated-transcript cache. These clips are shorter than the
16-second encoder-reuse boundary, so this experiment isolates the head change.
The measurements include feature extraction, audio encoding, decoder prefill,
draft verification, subsequent decoding, and text production.

| Four-call growing-audio trace | Serial median | Batched median | Paired batch/serial | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| English JFK | 1,299.09 ms | 1,180.53 ms | 0.92226 | 0.88326–0.96104 |
| Chinese | 695.38 ms | 650.39 ms | 0.93802 | 0.88593–0.99264 |

The paired reductions are 7.8% and 6.2% for these whole traces. Live public
verification batches contain 11, 14, and 21 rows for JFK and 7, 8, and 11 rows
for Chinese. The original `AllocsPerRun` gates reported zero after warmup;
that helper forces one P and averages using integer division. Strengthened
single-measured-run checks also pass at one P. At 16 Ps the existing worker
scheduler can allocate 112-byte channel wait records in both the serial and
batched paths, so these gates do not establish universal zero allocations.
See the [allocation qualification](cpu-streaming-exact.md#allocation-qualification)
and [serial/batch allocation stacks](benchmarks/cpu-stt/head-batch-allocations.json).
Every vocabulary logit matches the serial path exactly for the head measurements.
Before timing, each public trace step checks encoder embeddings, final hidden
state and logits, verification hidden states and logits, generated token IDs,
text, and language. Every timed call also checks its text and language.

Ratios are geometric means of neighboring paired latency ratios. Intervals
resample those paired log-ratios 10,000 times with seed 0. All samples are
retained. No coordinating-agent builds, profiles, or validation jobs overlap
timing; unrelated desktop activity can still affect the measurements. These
results cover these fixtures and this host. They are not a 6.2× speedup of
whole transcription, an offline transcription speedup, or a competitive
ranking. Combined longer-stream results are documented in
[exact CPU streaming improvements](cpu-streaming-exact.md).

## Correctness checks

The new kernel is compared directly with repeated calls to the original
single-row kernel, including odd row counts, partial output panels, non-aligned
input strides, legal split packing ranges, very small and large values,
signed zeros, infinities, and NaNs. Evaluator tests exercise F16 and int8
fallbacks with one, three, and eight workers and row counts through the
256-row API limit. They also return to a single-row call after batching and
check warm allocations with the one-P helper described above. A separate
stress test combines CPU profiling
signals, repeated GC, and oversubscribed private lanes; the initial candidate
passed 20 repetitions before performance confirmation, and another 20 passed
afterward with 1,946 actual panel retries.

A separate [warm-only public Partial profile](benchmarks/cpu-stt/head-batch-warm-profile.txt)
starts after three complete traces and covers 48 further public calls. It
samples `smeRows2F16` directly (2.05 seconds, 1.40% of CPU samples), confirming
that the new kernel runs in the real verifier. The final full-state fingerprint
still matches the warm reference after 10,547 signal retries across the SME
kernels. Model loading is outside the profile. Concurrent correctness jobs
were allowed during this run, so it provides dispatch and signal evidence,
not another latency measurement. The [raw profile](benchmarks/cpu-stt/head-batch-warm.pprof)
and [temporary profiling harness](benchmarks/cpu-stt/head-batch-profile.patch)
are retained for reproduction.

## Reproduce

The permanent `BenchmarkCPULogitsRows` compares separate public row calls with
the public batch API. `BenchmarkCPUPartialTranscribe` measures the four-call
trace. For counterbalanced baseline/candidate samples in one process, apply
[the temporary experiment patch](benchmarks/cpu-stt/head-batch-experiment.patch)
to an isolated checkout of the final implementation. It restores only the
benchmark switch, per-row-count call counters, and experiment harness; these
are absent from production.

```sh
git apply docs/benchmarks/cpu-stt/head-batch-experiment.patch
CODEX_AGENT_ID=stt-head GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/stt-head.test ./qwen3asr
cd qwen3asr
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  STT_HEAD_PAIRS=64 STT_HEAD_OUTPUT=/tmp/stt-head.json \
  /tmp/stt-head.test -test.run '^TestHeadBatchExperiment$' -test.v -test.timeout=20m
```

To recompute each paired interval:

```python
import json, math, random, statistics
report = json.load(open("docs/benchmarks/cpu-stt/head-batch-confirmation.json"))
for case in report["cases"]:
    pairs = {}
    for sample in case["samples"]:
        pairs.setdefault(sample["pair"], {})[sample["variant"]] = sample["ns"]
    logs = [math.log(p["batch"] / p["serial"]) for p in pairs.values()]
    rng = random.Random(0)
    boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                  for _ in range(10000))
    print(case["name"], math.exp(statistics.mean(logs)), boot[249], boot[9749])
```
