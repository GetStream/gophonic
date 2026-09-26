# Concurrent CPU speech inference

This change bounds busy waits in private worker pools and gives the HTTP
server a CPU budget per concurrent request. It does not change weights,
precision, matrix reductions, KV layout, or token selection.

## Why concurrency stalled

Both executors previously yielded at completion barriers only when their
own participant count exceeded `GOMAXPROCS`. Several independent pools can
each fit that limit while collectively oversubscribing it. Qwen's held
workers could also spin throughout a forward pass while the caller that
would publish their next job was descheduled. The overloaded reference's
stacks show callers in `workerPool.runN` and helpers in `workerPool.loop`.

After 50 microseconds of a completion wait, callers now yield to other
runnable goroutines. Held Qwen helpers also yield after that short spin
budget. Yielding adds no batching timer or sleep. Ordinary short barriers
still poll, and Qwen's existing idle parking and forward-pass holds remain:
parking inside a forward pass broke the warm allocation checks during
validation. There is no new global scheduler or shared mutable call state.

With `-threads 0` and multiple HTTP request slots, the server now chooses
`max(1, min(64, GOMAXPROCS / workers))` CPU participants per lane. One request
slot keeps the model default, and explicit `-threads` keeps its meaning.
The caller counts as a participant. This is a startup budget, not OS CPU
pinning or a dynamically enforced process-wide quota.

## Measurements

Reference: main at `1712e54`. Candidate: this change. Apple M4 Max,
Go 1.27.1, darwin/arm64, `GOEXPERIMENT=simd`, `CGO_ENABLED=0`.
These are local inference measurements, not x86/Linux server or HTTP
capacity claims. No timed processes overlapped, and no build or test ran
alongside a timed process. Two runs used opposite variant order.

Each request calls the public `Transcribe` API, including the frontend,
encoder, decoder, and text conversion. Model loading, lane creation, two
sequential warmups, and one concurrent warmup are excluded. Even-numbered
lanes use the full fixture; odd-numbered lanes use its first three quarters.
Qwen uses the existing Mandarin fixture and exact F16 1.7B checkpoint;
Whisper uses the JFK fixture and tiny.en checkpoint. No partial transcript
is supplied. All lanes retain private mutable state.

| Workload | Before cohort median | After cohort median | Throughput change | CPU time per call |
| --- | ---: | ---: | ---: | ---: |
| Qwen, 4 calls × 4 workers, `GOMAXPROCS=8`, run 1 | 3,058.6 ms | 900.8 ms | 3.40× | 5,863.7 → 1,666.0 ms |
| Same Qwen configuration, reverse-order run | 1,483.2 ms | 729.0 ms | 2.03× | 2,857.1 → 1,380.6 ms |
| Whisper, 8 calls × 2 workers, `GOMAXPROCS=16`, run 1 | 175.4 ms | 161.2 ms | 1.09× | 255.8 → 240.9 ms |
| Same Whisper configuration, second run | 176.0 ms | 157.9 ms | 1.11× | 257.5 → 241.4 ms |
| Whisper, 4 calls × 4 workers, `GOMAXPROCS=8` | 135.2 ms | 119.0 ms | 1.14× | 236.0 → 209.4 ms |

Qwen contention runs have three and four measured cohorts respectively.
Whisper runs have five and eight. CPU time is process user plus system time
from `getrusage`, divided by completed requests. Cohort throughput is the
number of requests divided by the time until the last request finishes.
These short traces do not establish production p99 latency or concurrent
live-call capacity.

At sensible budgets on all 16 CPU slots, Qwen was effectively unchanged:
238.7 → 237.1 ms for one call with eight workers, 701.3 → 682.7 ms for four
calls with four workers, and 1,318.7 → 1,328.3 ms for eight calls with two
workers. The change fixes contention; it is not a universal kernel speedup.

The Qwen eight-call/eight-worker reference warmed every lane but did not
finish its first concurrent cohort before the whole test's 45-second timeout.
The candidate completed both measured cohorts in 1.94–1.97 seconds. The
timeout includes setup, so no speedup ratio is assigned to that case.

Whisper's eight-call/eight-worker case was noisy: 310.5 → 338.2 ms in one
run and 209.1 → 185.3 ms in the next. Two workers per lane were faster than
eight in both runs. The new server default selects two for eight request
slots on `GOMAXPROCS=16`; previously each Whisper lane selected eight and
each Qwen lane selected sixteen. Explicit overprovisioning remains possible.

## Exactness and memory

Every timed result is checked against its own sequential warm reference.
The saved fingerprints also match between the main and candidate binaries
for every comparable request. Qwen fingerprints include embeddings, hidden
state, logits, generated tokens, text, and language; Whisper fingerprints
include encoder output, logits, tokens, text, and language.

The worker pools and their private numeric arenas are retained. There is no
new allocation, scratch buffer, cross-call KV sharing, or batching queue in
the inference path. The server's smaller default worker count also reduces
per-worker scratch and helper goroutines. Existing warm zero-allocation
assertions remain intact; this does not claim that every Go runtime or HTTP
allocation is eliminated.

The [raw measurements](benchmarks/cpu-stt/server-throughput.json) retain all
request timings, fingerprints, CPU times, and bounded-run statuses.

## Validation

The full repository suite passes with SIMD enabled and disabled. Race checks
pass for both executors, the HTTP server, and its CLI. Official Qwen frontend,
encoder, transcript and warm-allocation fixtures pass, as do Whisper's JFK
and warm word-timestamp fixtures. `go vet ./...` passes and the server builds
for Linux amd64 with SIMD enabled; that build is not a Linux performance run.

## Reproduction

Build the same test harness against the reference and candidate sources,
using the same compiler, CPU, settings, model cache, and fixtures. Execute
one binary at a time from its package directory:

```sh
GOEXPERIMENT=simd CGO_ENABLED=0 go test -c ./qwen3asr -o /tmp/qwen-throughput.test
cd qwen3asr
GOMAXPROCS=8 GOPHONIC_MODELS=/path/to/models \
  STT_SERVER_CPU='your CPU' STT_SERVER_OUTPUT=/tmp/qwen-throughput.json \
  STT_SERVER_CONFIGS=4:4 STT_SERVER_ROUNDS=8 \
  /tmp/qwen-throughput.test -test.run '^TestCPUConcurrentThroughput$' -test.v -test.timeout=2m
```

For Whisper, build `./whisper` and run from `whisper/`. For ordinary load,
use `GOMAXPROCS=16` and `STT_SERVER_CONFIGS=1:8,4:4,8:2`. Each completed
cohort checkpoints its JSON. The tests are opt-in and skip without
`STT_SERVER_OUTPUT`. Server-core tuning still needs measurements on the
intended EPYC, Xeon, or ARM server and representative call durations.
