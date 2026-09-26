# Exact CPU decoder continuation

This follows the [encoder-window and vocabulary improvements](cpu-streaming-exact.md)
merged in `14f4aa9`. It removes repeated decoder work in growing Qwen audio
and avoids small worker handoffs in Whisper. Neither change introduces a new
weight format, approximation, or speculative acceptance rule.

A subsequent [worker scratch and attention arena change](cpu-worker-scratch.md)
removes duplicate pooled Whisper scratch and moves attention payloads outside
the Go heap on Unix. Its memory benefit is measured separately; it does not
add a confirmed latency improvement to the numbers below.

## Qwen: preserve the arithmetic anchor

A growing audio prefix can produce identical encoder embeddings, yet ordinary
KV-prefix extension changes floating-point reduction groups. The new path keeps
the original non-audio prompt anchor and restores cached K/V into the existing
one-layer scratch. Complete 128-row blocks skip their projections, normalization,
RoPE, and attention queries; requested tail outputs always run.

The cache is private to its transcription lane. Reuse requires exact encoder
validation, a matching prompt/attention anchor, and the immediately preceding
encode revision. Cancellation after encoding and changed prompts are covered by
real-model regression tests. GPU and ineligible calls recompute normally.

There is no second persistent KV cache, another numeric arena, or additional
retained audio copy. Skipping rows also preserves the original scheduling policy,
so reuse does not increase attention-descriptor capacity for the same shape.
See [decoder continuation](cpu-decoder-continuation.md) for the exactness proof,
measurements, raw samples, profiling evidence, and reproduction commands.

On the M4 Max fixture, a fresh 24-pair comparison with equivalent updates
adjacent on independent warmed lanes measured **11.7% lower total latency**
(paired ratio 0.8827, 95% interval 0.8400–0.9454). The final 33-second update
measured **49.4% lower latency**. The earlier whole-trace comparison had an
unresolved aggregate result; both complete datasets are retained in the
detailed report.

## Whisper: keep small packed attention on the caller

A warm profile of 100 complete public `TranscribeInto` calls showed substantial
worker scheduling and wakeup activity. CPU sample percentages are not wall-time
percentages. The tiny model's six packed cross-attention heads finish sooner on
the caller than after another worker handoff. The final change runs packed
attention with at most six heads of at most 64 dimensions on the caller. Larger
heads and the unpacked fallback retain their previous scheduling.

No kernel or arithmetic changes. There are no new buffers, longer spin windows,
or permanently active workers. The investigation rejected longer polling: a
500 microsecond polling window did not resolve a standalone gain and made the
combined candidate slower.

A fresh confirmation used Apple M4 Max, Go 1.27.1, `GOEXPERIMENT=simd`,
`CGO_ENABLED=0`, and `GOMAXPROCS=16`. Variants alternate within neighboring
pairs on the same warmed lane. Model loading is excluded; each sample includes
the complete public transcription. Every timed call is followed by an untimed
bitwise check of encoder output, final logits, hidden state, self-KV, token IDs,
and transcript. All samples are retained.

The ratio is the geometric mean of paired candidate/baseline latencies. The
95% intervals resample complete pairs 10,000 times with seed 0. These are local
fixture results, not a competitive ranking or an accuracy-corpus evaluation.

| Public transcription | Pairs | Baseline median | Candidate median | Paired ratio, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| JFK, 8 workers (default) | 64 | 70.03 ms | 64.35 ms | 0.9242 [0.9093, 0.9384] |
| Repeated JFK, multiple windows, 8 workers | 32 | 207.51 ms | 181.52 ms | 0.8794 [0.8657, 0.8938] |
| JFK, 2 workers | 24 | 71.50 ms | 69.85 ms | 0.9629 [0.9420, 0.9802] |
| JFK, 4 workers | 24 | 72.77 ms | 70.47 ms | 0.9665 [0.9430, 0.9906] |
| JFK, 16 workers | 24 | 71.96 ms | 66.83 ms | 0.8933 [0.8695, 0.9133] |

The complete [raw report](benchmarks/cpu-stt/whisper-attention-confirmation.json)
records allocations as well as timings and state fingerprints. All 272 measured
short transcriptions reported zero Go allocations. One multiwindow call reported
four allocations; the remaining 63 reported zero. This does not establish that
the runtime can never allocate. The separate allocation tests use a single
measured `AllocsPerRun` invocation, whose GOMAXPROCS=1 limitation still applies.

The [warm profile summary](benchmarks/cpu-stt/whisper-attention-profile-top.txt),
[initial dispatch trial](benchmarks/cpu-stt/whisper-attention-discovery.txt), and
[longer-polling trial](benchmarks/cpu-stt/whisper-spin-discovery.txt) are retained.
The first dispatch trial's overall interval crossed no change; the later fresh
confirmation above is the reported result. No parent or Astra build, profile,
or validation job overlapped the confirmation.

## Reproduce the Whisper comparison

The [instrumentation patch](benchmarks/cpu-stt/whisper-attention-confirmation.patch)
applies to this change in an isolated checkout. It restores the old dispatch
through a test-only switch and adds the paired public-call harness. Neither is
present in production.

```sh
git apply docs/benchmarks/cpu-stt/whisper-attention-confirmation.patch
CODEX_AGENT_ID=whisper-continuation GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/stt-whisper.test ./whisper
cd whisper
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  STT_WHISPER_DISPATCH=/tmp/whisper-confirmation.json \
  /tmp/stt-whisper.test -test.run '^TestCPUNextAttentionDispatch$' -test.v -test.timeout=10m
```

## Correctness coverage

The decoder differential suite compares hidden states, logits, token IDs and
all layers' complete K/V contents against recomputation. It covers existing
F16 and int8 weights, 1/3/8 workers, anchors 0/1/17/255/256/257, audio lengths
around 128-row block and 256-row value-page boundaries, and tail outputs of
1/3/16/32 rows. Ordinary one-token decoding after each continuation also
matches. Invalid reuse and failed forward passes clear temporary skip state.

A real-model growing-PCM test verifies positive reuse and two lifecycle
hazards: cancellation after a changed audio encode, before its decoder runs,
and a changed static prompt. The former must reject old decoder K/V even when
the encoder can reuse its newer audio prefix.

Packed extraction tests preserve float bits, including signed zero and NaN
payloads, check output padding, reject invalid ranges, and exercise fixed-pitch
value pages. Whisper's packed attention comparison covers both sides of its
small-head scheduling cutoff and checks exact equality with the previous
parallel evaluation. No assembly changes are required.

## Final validation

The final production source passes the complete SIMD-enabled repository suite,
race checks for `internal/qwen3lm`, `internal/whispergemm`, `qwen3asr` and
`whisper`, and the same packages' non-SIMD tests. Targeted real Whisper and
Qwen model/reference, continuation, lifecycle and allocation checks also pass
with `GOGC=10`. Repository vet and the CLI build pass. The modified packages
cross-compile for Linux arm64 with SIMD and Windows amd64 without it; no
runtime performance claim is made for those cross-compiled configurations.

Temporary benchmark/profile test files and production experiment switches are
absent. Their archived patches are the reproducibility artifacts.
