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
