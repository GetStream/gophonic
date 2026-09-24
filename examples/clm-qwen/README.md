# Local Qwen3-8B embeddings for CLM

This optional nested module connects `gophonic/clm` to a pure-Go CPU Qwen3
runtime. It keeps the large text-decoder dependency out of gophonic's core
module. `Open` loads an official Qwen3-8B safetensors snapshot directory or a
GGUF with matching geometry; `Embed` tokenizes text, keeps the final 2048
tokens, runs Qwen3's final hidden state at the last token, then writes
the raw 4096-value hidden vector. The CLM head normalizes it once. Both state
and action roles use the same encoder. The CLM heads and scoring remain inside
gophonic.

## Run a local ranking

Convert the official 75 MB CLM head with the repository's
`tools/clm_pt_to_gophonic.py` first. Then run from this directory:

```sh
CGO_ENABLED=0 GOEXPERIMENT=simd go run ./cmd/rank \
  -qwen /path/to/Qwen3-8B \
  -head /path/to/CLM_v0.1-8B.gclm \
  -state 'What causes tides on Earth?' \
  'The Moon’s gravitational pull.' \
  'Photosynthesis in plants.'
```

The command prints candidates in descending probability order. `-quant int8`
is an opt-in weight-only quantized mode. On the current 64 GB development
machine, the full FP32 Qwen load is refused by the decoder's memory-fit guard.
The int8 model loads and runs locally, but its ranking accuracy still needs a
corpus gate.

This adapter is built against a pinned commit of
[`townsendmerino/goinfer`](https://github.com/townsendmerino/goinfer), whose
[`HiddenLast`](https://github.com/townsendmerino/goinfer/blob/2f2b429898b2819434c08c51f06b01b0be391ba2/decoder/embed.go)
returns the post-final-norm last-token state before the language-model head.
The adapter checks Qwen3-8B's hidden size and layer/head geometry at load.
Supply the exact [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)
checkpoint for the published CLM head. `Open` uses full FP32 weights.
`OpenWithOptions(path, Options{Quant: "int8"})` reduces the safetensors weight
footprint at the cost of altered embeddings. Int8 uses gophonic's reusable
Qwen3 workspace. On Apple M4 CPUs with 512-bit SME, all input lengths use
a packed Q8 matrix kernel, processing prompts longer than 16 tokens in tiles.
The owning encoder releases its original Q8 projection arrays after packing
all layers. Other platforms retain the portable batched path. FP32 continues to use
GoInfer's reference forward path. Quantized weights and GGUF checkpoints need
a separate ranking-accuracy gate against the reference.

CLM's upstream reference uses vLLM pooling. Token IDs, 4096-value embeddings,
and final rankings must be compared with that reference before claiming parity.
The int8 fast-path gate compares one-token and 12-token outputs to GoInfer on the
official checkpoint. The FP32 CLM-ranking gate covers one short input; a
corpus-level ranking evaluation is still needed.

`Encoder.EmbedTokensInto` accepts pretokenized IDs and caller-owned output.
For a Hugging Face Qwen3 tokenizer, both it and public text `Encoder.Embed`
perform zero heap allocations after their workspaces have warmed to the
longest input. The GGUF tokenizer uses its existing allocation behavior.
Calls on one encoder are serialized; use a separate encoder per concurrent lane.

On an Apple M4 Max with `GOMAXPROCS=1`, an isolated 30-call run of the
owned packed encoder measured 338 ms for one token and 382 ms for 12 tokens,
with 0 B/op and 0 allocs/op. Earlier 15-call public text runs averaged 392 ms
and 398 ms; a prior run averaged 502 ms, so latency varies between processes.
The pinned CLM ranking matches the existing int8 path. A llama.cpp Q8_0 CPU
prefill benchmark measured 778 ms for 12 random tokens, using a different
quantized weight format and token contents. The SME path repacks weights once
(about 5–8 s). Releasing the owning decoder's original Q8 projections lowered
Go heap from 14,457 to 7,828 MiB after collection. Peak loading memory
remains high, and sampled process RSS did not decrease after collection. See [the performance report](../../docs/clm-performance.md) for the measurement details. The
100 ms single-core target remains open.

## Reference gates

Run the int8 fast evaluator's official-checkpoint parity and allocation gates with:

```sh
GOPHONIC_QWEN3_FAST_MODEL=/path/to/Qwen3-8B \
CGO_ENABLED=0 GOEXPERIMENT=simd go test -run '^TestFastQwenOfficialParity$' -v
```


The SME gates use the same official snapshot and the converted CLM head:

```sh
GOPHONIC_QWEN3_FAST_MODEL=/path/to/Qwen3-8B \
GOPHONIC_CLM_HEAD_BUNDLE=/path/to/CLM_v0.1-8B.gclm \
CGO_ENABLED=0 GOEXPERIMENT=simd go test \
  -run '^TestOfficialSME' \
  -bench '^BenchmarkOfficialQwenSMEPublicTwelveTokens$' \
  -benchtime=15x -count=1 -v
```

The tests assert real SME dispatch, agreement with the existing int8 decoder
on the official Qwen checkpoint, CLM ranking order and probabilities, and zero
warmed allocations. On a CPU without 512-bit SME, they skip and the existing
batched CPU path remains available.

The opt-in `TestOfficialCLMRanking` runs the full Go FP32 Qwen decoder and
converted head against the official BF16 Qwen + PyTorch head output. Regenerate
the three golden probabilities with `tools/reference_rank.py`. On the development
Mac, the full Go FP32 path needed `GOINFER_NO_FIT_GUARD=1` because the runtime
prices a maximum-length KV cache at load even though this test uses short text.
That override was used only for this measured test. The reference test passes
with probabilities within 0.001 and identical candidate order. One short
example cannot establish corpus-level ranking accuracy.

```sh
GOINFER_NO_FIT_GUARD=1 \
GOPHONIC_QWEN3_TOKENIZER=/path/to/Qwen3-8B \
GOPHONIC_QWEN3_MODEL=/path/to/Qwen3-8B \
GOPHONIC_CLM_HEAD_BUNDLE=/path/to/CLM_v0.1-8B.gclm \
CGO_ENABLED=0 GOEXPERIMENT=simd go test -run TestOfficial -v ./...
```
