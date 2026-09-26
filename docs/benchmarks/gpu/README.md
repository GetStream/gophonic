# GPU qualification measurements

These measurements are development evidence, not a state-of-the-art claim.
The local device is an Apple M4 Max with 64 GiB unified memory, macOS
26.3.1(a), and Go 1.27.1. NVIDIA/AMD measurements are still required.

## External ASR reference

`mlx-qwen3asr-decoder-q8.jsonl` records five warm repetitions per clip using
MLX Audio revision `4ab7e6f7dedd69a136cfaa318c5dc8aed5119446`, MLX 0.32.2,
and the local official Qwen3-ASR-1.7B checkpoint. The benchmark uses the same
PCM fixtures and peak normalization as gophonic. Greedy generation is capped
at 256 tokens. Transcript text, generated-token counts, all latency samples,
and MLX memory counters are included in the raw output.

The script quantizes decoder linear layers and a separate output head to
MLX affine 8-bit groups of 32. Embedding lookup and audio encoder weights
retain their original precision. The 197 quantized layers include the
separate head; this avoids leaving the tied head in BF16 while labeling the
comparison decoder-Q8. Median PCM-to-text times were 249.1 ms for JFK and
123.5 ms for the Chinese fixture.

This is a size-comparable reference, **not identical arithmetic**: MLX affine
quantization differs from gophonic's rotated Q8B, and activation computation
also differs. Matching these two transcripts is not a corpus-level accuracy
measurement or proof of numerical equivalence. Device allocator counters and
host peak RSS describe different quantities; do not add or directly equate
them with another implementation's logical buffer capacities.

Run in an isolated Python environment with the pinned MLX Audio revision:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/benchmark_mlx_asr.py \
  --model /path/to/Qwen3-ASR-1.7B \
  --precision decoder-q8 \
  --implementation-revision 4ab7e6f7dedd69a136cfaa318c5dc8aed5119446
```

Use `--precision bf16` for the unquantized checkpoint reference. The script
is optional development tooling and adds no gophonic runtime dependency.

## Independent-worker measurements

The public ASR concurrency test compares exact fingerprints of encoder
outputs, decoder states, logits, generated tokens, transcript, and language
between serial and concurrent execution. Each lane uses its own transcriber
and K/V cache. It passed for the English and Chinese fixtures on the local
Metal device with CGO disabled.

`BenchmarkGPUConcurrentTranscribe` measures complete JFK transcriptions
through 1, 2, 4, or 8 persistent workers. Setup, model loading, serial warmup,
and concurrent warmup are outside the timed loop. One `ns/op` is a wave of
one complete call per worker; `calls/s` is aggregate throughput. It does not
measure an HTTP server or batched decoder implementation.

```sh
CODEX_AGENT_ID=gpu-measure CGO_ENABLED=0 GOEXPERIMENT=simd \
GOPHONIC_MODELS=/path/to/models \
/Users/thesyncim/.codex/bin/project-env go test ./qwen3asr \
  -run '^TestGPUConcurrentTranscribers$' \
  -bench '^BenchmarkGPUConcurrentTranscribe$' -benchtime=5x -count=3
```

Do not run another GPU workload during measurement. Compare full-call results
alongside projections; changing threadgroup geometry can improve an isolated
shape while slowing a complete decode.

## Full-call native batches

The opt-in `qwen3asr.BatchTranscriber` reuses immutable Q8B weights across
FP32 token projections. Each call keeps private K/V and encoder state. Lane
workers hand off decode ownership at token boundaries and wait for its return.
Independent attention dispatches run in a concurrent encoder, separated from
dependent projection stages by explicit buffer barriers. The scalar API and
Apple provider default remain unchanged.

Public parity tests compare **exact fingerprints** of encoder output, final
hidden state, recomputed final logits, draft verification, generated tokens,
text, and language against independent lanes with the same call history.
The greedy path intentionally does not copy full logits on each step; the
internal decoder tests separately compare selected tokens and full-logit
results from identical private prefixes. They pass for English
and Chinese fixtures, varying active counts (including empty lanes), forced
language/context, segments, partial transcripts, cancellation/recovery, and
forced collections between calls. Invalid/aliased outputs are rejected before
worker dispatch. This is observed equivalence on these fixtures, not a
corpus-wide accuracy claim.

A first long sequential benchmark sweep was rejected: the unchanged scalar
control drifted from about 233 ms to 443–459 ms, and later waves varied by
several times. `BenchmarkGPUPairedTranscribe` instead alternates the execution
order of baseline and batch waves. Both paths use persistent private lanes,
serial and concurrent warmup, identical input PCM, and transcript validation
inside each timed wave. A wave finishes one call per lane; `ns/op` covers both
methods, while the named metrics report them separately. Allocation counts
also cover both methods together.

[Four-lane raw output](asr-paired-four-lanes.txt) records three counts of four
pairs each on the local M4 Max. The English-only throughput ratios were
1.393×, 1.394×, and 1.326×. The mixed workload alternates the 11-second English
and 4.2-second Chinese fixtures; its ratios were 1.260×, 1.257×, and 1.091×.
English wave times were 596–692 ms batched versus 830–918 ms independent.
Single-call controls were near 1.00× (0.997–1.047× across both workload labels).
Machine load still varied, so retain the range rather than quoting the best
sample. These results establish a local opt-in throughput improvement, not
state-of-the-art performance or a reduction in every caller's latency.

[Eight-lane and two-lane output](asr-paired-two-eight-lanes.txt) adds
process CPU time (user plus system, via `getrusage`). Eight English calls
measured 1.260–1.356× throughput and 31–50% less process CPU time per call.
The mixed eight-call workload measured 1.032–1.161× throughput and 6–23%
less process CPU per call. These CPU counters do not include all system
or driver activity and do not pin execution to particular cores.

The initial mixed two-lane batch was not reliably faster (0.877–1.115×).
A remaining single lane now uses its ordinary decoder: it has no other
call's weight loads to reuse, so batch barriers add no value. The
[tail-path rerun](asr-paired-single-lane-tail.txt) passed exact public parity
three times and measured 1.015, 1.104, and 1.031× for mixed two-lane calls.
Runs had different desktop load; these are paired comparisons against the
contemporaneous baseline, not proof that the tail change alone accounts
for the difference. Small heterogeneous batches remain workload-sensitive.

```sh
CODEX_AGENT_ID=gpu-measure CGO_ENABLED=0 GOEXPERIMENT=simd \
GOPHONIC_MODELS=/path/to/models \
/Users/thesyncim/.codex/bin/project-env go test ./qwen3asr \
  -run '^TestGPUBatchTranscribe' \
  -bench '^BenchmarkGPUPairedTranscribe$/(jfk|mixed)/lanes=(1|4)$' \
  -benchtime=4x -count=3 -v
```

## Token selection and resource lifetime

The final native greedy batch selects token IDs on the GPU immediately
after the unchanged Q8B head, within its existing command buffer. For this
151,936-token vocabulary it avoids copying 607,744 bytes of logits per lane
per step and scanning them on the CPU; vocabulary-result readback becomes
one 32-bit token ID per lane. Hidden-state output remains unchanged. The
selection kernel compares integer keys derived from the existing FP32 bits,
with explicit NaN and signed-zero handling, to preserve scalar first-maximum
semantics even with Metal fast-math enabled. Actual-device tests cover ties,
infinities, NaNs, subnormals, padded rows, and a tail-element winner.

A [12-pair, three-repeat comparison](native-greedy-selection.txt) alternates
full-copy/CPU-scan, direct-mapped/CPU-scan, and GPU selection over the same
eight-lane decoder step and fixed 32-token private prefixes. GPU selection
measured 30.88, 27.58, and 29.33 ms versus 33.20, 30.36, and 32.08 ms for
mapped scanning. Process CPU time was 1.06–1.17 ms versus 3.84–3.96 ms per
step, about 70–73% lower. Full-copy/CPU-scan controls measured 29.95–32.32 ms
and 4.02–4.11 ms of process CPU. The earlier two-pair pilot was too variable
to choose a path. These are decoder-step measurements, not complete-call
latencies; do not compare absolute timings with earlier sessions as if
machine load were fixed.

Native encoders also skip unchanged pipeline and buffer bindings, with
cached state private to each encoder and reset at Begin. Tests exercise
inline-byte replacement, buffer/offset/pipeline changes, and encoder reuse
on the actual GPU. The branch explicitly releases compiled libraries and
pipelines; closed model, workspace, KV, and transcriber handles drop retained
weights and high-water scratch. Failed optional pipeline initialization
releases already-created pipelines, and encoder-worker initialization errors
stop their workers. Portable readback failures now cancel pending mappings
before returning, allowing the staging buffer to be reused safely.

## Matched projection stream

The native and portable benchmarks use the same deterministic, nonzero Q8B
weight generator, 28 distinct 2048×2048 matrices, and one nonzero activation
vector. The packed weights total 124,780,544 bytes, larger than a single
projection. Outputs are distinct per matrix; one submission executes all
projections and reads back the final output. Input/weight uploads are outside
the timed loop. The native benchmark independently checks every final output
row with FP64 accumulation before timing.

Recorded local samples (see the adjacent raw files):

| Provider | Elapsed samples | Median | Go allocations/op |
| --- | --- | --- | --- |
| Native Metal | 463, 706, 429 µs | 463 µs | 0 |
| Pinned GoGPU, logical 32-lane tiles, WG128/RPS2 | 698, 701, 686 µs | 698 µs | 2,282–2,286 |
| Isolated GoGPU ABI-signature cache experiment | 748, 784, 945 µs | 784 µs | 2,016–2,020 |

The experiment cached immutable prepared call interfaces and copied them per
invocation. It reduced allocations but did **not** improve latency, so it is
not part of the runtime dependencies. Native samples used 1 second per count;
portable samples used 300 ms. Variability is visible, and these are projection
measurements, not complete ASR speedups. The portable path has not cleared the
replacement gate. Older zero-filled native buffer measurements are excluded
from this comparison.

The final portable geometry is WG128 with two rows per logical 32-lane
tile. These are software workgroup tiles, not an assumption about a vendor's
physical subgroup width. Skipping redundant pipeline bindings reduces the
28-dispatch stream to about 2,120 Go allocations; a later three-count run
recorded 792, 661, and 772 µs. The allocation reduction is repeatable, while
the timing samples are noisy.

A same-input projection bank concatenates the same 28 matrices into one
row-major matrix and issues one dispatch. It measured 536 and 529 µs with
394 Go allocations, versus 943 and 750 µs with 2,120–2,122 allocations for
independent portable dispatches in that session. The bank preserves the
independent dependency graph and reads back the same final output rows. It
models fusion opportunities such as QKV or gate/up; it cannot combine
sequential transformer layers whose activations depend on previous results.
See [bank qualification samples](portable-q8b-bank.txt) for the full setup.
This primitive remains qualification code, and neither portable form has
cleared the native replacement or allocation gate.

```sh
CODEX_AGENT_ID=gpu-measure CGO_ENABLED=0 GOEXPERIMENT=simd \
/Users/thesyncim/.codex/bin/project-env go test ./internal/qwen3lm \
  -run '^$' -bench '^BenchmarkGPUQ8BProjectionStream$' -benchtime=1s -count=3

CODEX_AGENT_ID=gpu-measure CGO_ENABLED=0 GOEXPERIMENT=simd \
GOPHONIC_GPU_BENCH=1 \
/Users/thesyncim/.codex/bin/project-env go test ./internal/gpuportable \
  -run '^$' -bench '^BenchmarkLinearProjectionStream$/Q8B/wg128-r2$' \
  -benchtime=300ms -count=3
```
