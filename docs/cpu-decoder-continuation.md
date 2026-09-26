# Exact CPU decoder continuation

On the measured Qwen3-ASR-1.7B growing-audio trace, fresh adjacent pairing
shows **11.7% lower total latency** (95% CI 5.5–16.0%) and **49.4% lower
latency for the 33-second update** (CI 38.8–56.0%). All exact-state gates pass.
These measurements use F16 weights on an Apple M4 Max with 64 GiB, Go 1.27.1,
`GOEXPERIMENT=simd`, `CGO_ENABLED=0` and `GOMAXPROCS=16`.

Growing partial Qwen3-ASR calls can now avoid recomputing decoder projections
for a proven unchanged audio prefix. The encoder already validates its reused
normalized features bit for bit. The decoder uses that validation to reuse
complete 128-row attention blocks from its existing packed KV cache.

No new numerical cache is allocated. Each lane keeps only small provenance
fields: the prompt anchor, audio row count, encode revision and validity. Its
existing bounded KV arena remains the sole persistent copy. Each layer restores
the reused rows into the existing one-layer K/V workspace before processing
the new suffix.

## Why the result is bit-exact

Ordinary prefix extension cannot be used here: moving audio from the current
sequence into the kept prefix changes how FP32 attention products are grouped.
The previous implementation computes the current sequence's value product
first, then adds one product for each 256-row prefix page. Changing that
partition can change a float bit even when every cached key and value is exact.

The continuation path retains the original non-audio `keep` as its arithmetic
anchor. For each layer, workers copy the reused audio KV into its original
row positions, then truncate that layer's packed prefix back to `keep`.
Attention still sees the original current-sequence matrix and prefix pages;
its operation order, causal limits, softmax and value-product grouping are
unchanged. Afterwards the usual store operation publishes the reconstructed
prefix together with the newly computed suffix.

Reuse is rounded down to complete 128-row attention blocks. This preserves
the previous query block's full key bound, and also aligns the 16-row
projection tiles and 32-row attention tiles. The original logical batch size
and row positions remain unchanged, so the remaining rows use the same kernel
and tail shapes. Final output compaction resets row skipping before the last
output projection and MLP. An ordinary following decode step sees the same KV
bits as a complete recomputation.

## Lifecycle and fallback

Reuse requires all of the following:

- The CPU encoder validated an unchanged prefix during a strictly growing
  partial call.
- The decoder cache belongs to the immediately preceding encode revision.
- The static prompt and arithmetic anchor match, with audio beginning at that
  anchor.
- The previous prefill completed successfully and the same KV arena is live.

Cold prefills, changed features, shrinking or repeated inputs, fresh calls,
changed prompt anchors, replaced KV arenas and abandoned encodes recompute the
decoder normally. In particular, an encode that succeeds before cancellation
or prompt failure advances the revision: a later encoder cache hit cannot
mistakenly authorize decoder KV belonging to an older audio input. The GPU
path retains complete recomputation.

## Correctness evidence

The synthetic differential test compares every hidden state, logit, cached
token, and every layer/group's key and value bit. It covers F16 and existing
int8 formats; worker counts 1, 3 and 8; anchors 0, 1, 17, 255, 256 and 257;
audio lengths around 128- and 256-row boundaries; 1/3/16/32 output rows; and
ordinary token decode immediately after each continuation. Invalid reuse and
failed forward passes also check cleanup. A threshold test verifies that
skipping blocks does not switch a warmed group dispatch to a larger per-head
descriptor list: scheduling is selected from the original unskipped shape.

Packed extraction tests independently cover exact FP32 bits, signed zero,
NaN payloads, column tails, fixed-pitch value pages, destination padding and
bounds. Single-measured-run `AllocsPerRun` checks pass for the warmed internal
continuation path. Those checks run at one P; they do not establish universal
zero allocation at production concurrency. See the existing
[allocation qualification](cpu-streaming-exact.md#allocation-qualification).

The permanent real-model regression uses actual growing JFK-derived PCM and
compares embeddings, hidden states, logits, verification tails, generated
tokens, text and language against the same warmed lane with decoder reuse
disabled. It exercises positive reuse, a changed prompt, and cancellation
after an encode that changed the audio prefix; the next call has positive
encoder reuse but must reject stale decoder KV.

## Measurement protocol

The reference is main `14f4aa9a8fd5fccfc26865b9b7c38c5d67d95609`, including
the previously merged exact encoder prefix and vocabulary-head improvements.
Both measurement protocols disable only decoder continuation by clearing its
provenance before each baseline public call. The initial protocol runs both
variants in the same warmed lane, with identical capacities, prompt history
and public partial transcription calls. The fresh adjacent protocol uses two
private warmed lanes to place matched inputs next to each other. Neither
protocol caches a completed transcription.

One trace contains seven strictly growing inputs: 4, 8, 16.02, 17, 24, 32.02
and 33 seconds. The audio repeats the repository JFK fixture to provide a
deterministic long input; it is a performance workload, not a quality corpus.
The decoder reuses 0, 0, 0, 128, 128, 128 and 384 rows respectively. Complete
float-state hashes are checked outside timed runs, including warm baseline
reproducibility. Timed calls check text and language. Whole-trace pairs reverse
order on alternate blocks, and every sample is retained.

The four-pair same-lane discovery run passed exactness but had extreme unrelated
desktop-load variance (whole traces ranged from 4.0 to 41.5 seconds). Its
paired ratio is exploratory evidence only. The raw
[discovery report](benchmarks/cpu-stt/decoder-continuation-discovery.json),
[log](benchmarks/cpu-stt/decoder-continuation-discovery.log), and
[reproducible harness](benchmarks/cpu-stt/decoder-continuation-experiment.patch)
are preserved.

The fresh 24-pair whole-trace confirmation also experienced substantial
desktop-load drift. Its total paired ratio is **0.994**, with 95% CI
**[0.864, 1.206]**; aggregate latency improvement is unresolved in that run.
The 33-second update, which reuses 384 rows, has a paired ratio of **0.482**
with CI **[0.428, 0.556]**: a resolved **51.8% lower latency** for that update.
The first three calls cannot reuse decoder rows, yet their paired ratios
range from 1.11 to 1.25, illustrating the unrelated drift. All samples remain
in the [confirmation report](benchmarks/cpu-stt/decoder-continuation-confirmation.json)
and [log](benchmarks/cpu-stt/decoder-continuation-confirmation.log).

Same-lane whole-trace process allocation counters at 16 Ps range from 0 to 11
allocations / 0–1,232 bytes for recomputation, and 0 to 10 / 0–1,120 bytes for
reuse. These include the pre-existing worker scheduling metadata; the result
does not support a blanket zero-allocation claim.

## Adjacent confirmation

The primary result is a fresh set of **24 paired traces** on two separate
Transcribers sharing only immutable model weights. Each lane owns its
activations, KV cache, encoder snapshot and workers; idle workers retain their
normal bounded wait policy. Both lanes follow the same warm input history.
The baseline must reproduce its complete float-state hashes before the other
lane is compared against it. All seven cross-lane hash gates pass.

During timing, the baseline and candidate process each matching audio length
adjacently. Their execution order is `(pair + step) % 2`, balancing both
orders for every step. Each lane receives its own previous partial transcript.
Text and language are checked after every timed call; complete numeric hashes
are checked in the untimed gates. Every sample is retained, including the
remaining unrelated desktop stalls.

| Public update | Median recompute / reuse | Paired ratio | 95% CI |
| --- | ---: | ---: | ---: |
| 4 s, no reusable decoder rows | 336 / 344 ms | 1.032 | 0.958–1.149 |
| 8 s, no reusable decoder rows | 399 / 411 ms | 1.061 | 0.987–1.188 |
| 16.02 s, no reusable decoder rows | 798 / 796 ms | 1.121 | 0.989–1.338 |
| 17 s, 128 reused rows | 786 / 638 ms | 0.780 | 0.702–0.832 |
| 24 s, 128 reused rows | 957 / 807 ms | 0.883 | 0.827–0.972 |
| 32.02 s, 128 reused rows | 1,158 / 1,024 ms | 0.895 | 0.845–0.956 |
| 33 s, 384 reused rows | 737 / 322 ms | 0.506 | 0.440–0.612 |
| Complete seven-call trace | 5,217 / 4,322 ms | 0.883 | 0.840–0.945 |

Ratios are paired geometric means, which differ from ratios of the displayed
medians. The no-reuse controls do not resolve a change. All four calls with
reused decoder rows and the aggregate trace resolve lower latency under
adjacent pairing. The earlier same-lane aggregate remains unresolved and is
reported separately above, rather than being discarded.

Combined process counters across both lanes range from 1 to 12 allocations
and 112–1,344 bytes per fourteen-call paired trace. They include worker
scheduling metadata and are not a per-lane zero-allocation result.

The [raw adjacent report](benchmarks/cpu-stt/decoder-continuation-adjacent-confirmation.json),
[log](benchmarks/cpu-stt/decoder-continuation-adjacent-confirmation.log),
[harness](benchmarks/cpu-stt/decoder-continuation-adjacent.patch), and
[all per-step statistics](benchmarks/cpu-stt/decoder-continuation-statistics.json)
include provenance, binary SHA-256, raw samples and the uncertainty estimates.

## Warm dispatch profile

Warm profiles exclude loading and initial capacity growth. Each variant
executes 21 public calls and checks exact states after every call. The reuse
profile exercises 12 reused prefixes totaling 2,304 rows. Profiled projection
CPU samples total 60.94 seconds for recomputation and 43.88 seconds for reuse;
these samples demonstrate the avoided projection work, not an independently
controlled latency comparison. The live `restorePrefix`, `UnpackColumns` and
`UnpackRows` stack accounts for 0.04 sampled CPU seconds in the reuse profile.

The [profile log](benchmarks/cpu-stt/decoder-continuation-profile.log),
[recomputation profile](benchmarks/cpu-stt/decoder-continuation-recompute-profile.txt),
[reuse profile](benchmarks/cpu-stt/decoder-continuation-reuse-profile.txt), and
[profile harness](benchmarks/cpu-stt/decoder-continuation-profile.patch) are
preserved, together with both raw `.pprof` files. The profiling harness also
hashes state; that test-only work is visible in the profile.

Ratios and per-step confidence intervals are computed from paired log ratios
with 30,000 bootstrap draws, seed 250925, and no sample filtering. The
[analysis script](benchmarks/cpu-stt/decoder-continuation-analysis.py) uses only
the Python standard library.

## Reproduce the adjacent comparison

Apply the archived harness to an isolated checkout of this change. The harness
contains no production switch; its baseline clears only decoder provenance.

```sh
git apply docs/benchmarks/cpu-stt/decoder-continuation-adjacent.patch
CODEX_AGENT_ID=decoder-continuation GOEXPERIMENT=simd CGO_ENABLED=0 \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/stt-decoder.test ./qwen3asr
cd qwen3asr
GOMAXPROCS=16 GOPHONIC_MODELS=/absolute/path/to/models \
  STT_ADJACENT_OUTPUT=/tmp/decoder-adjacent.json STT_ADJACENT_PAIRS=24 \
  /tmp/stt-decoder.test -test.run '^TestDecoderAdjacentExperiment$' -test.v -test.timeout=20m
cd ..
python3 docs/benchmarks/cpu-stt/decoder-continuation-analysis.py /tmp/decoder-adjacent.json
```

See the [combined validation report](cpu-continuation-exact.md#final-validation)
for final repository, race, fallback, real-model and cross-compile checks.
