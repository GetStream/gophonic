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
