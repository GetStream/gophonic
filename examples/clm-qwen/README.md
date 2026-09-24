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
is an opt-in lower-memory mode. On the current 64 GB development machine, the
full FP32 Qwen load is refused by the decoder's memory-fit guard; int8 loads
and embeds correctly, but its ranking accuracy still needs a corpus gate.

This adapter is built against a pinned commit of
[`townsendmerino/goinfer`](https://github.com/townsendmerino/goinfer), whose
[`HiddenLast`](https://github.com/townsendmerino/goinfer/blob/2f2b429898b2819434c08c51f06b01b0be391ba2/decoder/embed.go)
returns the post-final-norm last-token state before the language-model head.
The adapter checks Qwen3-8B's hidden size and layer/head geometry at load.
Supply the exact [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)
checkpoint for the published CLM head. `Open` uses full FP32 weights.
`OpenWithOptions(path, Options{Quant: "int8"})` reduces the safetensors weight
footprint at the cost of altered embeddings. Quantized weights and GGUF
checkpoints need a separate ranking-accuracy gate against the reference.

CLM's upstream reference uses vLLM pooling. Token IDs, 4096-value embeddings,
and final rankings must be compared with that reference before claiming parity.
The included reference gate covers one short input; a corpus-level accuracy
evaluation is still needed. The Qwen decoder currently
allocates a fresh hidden vector and KV state per call, so the full text path is
not zero allocation even when the CLM head's workspace is reused. A single
`Encoder` serializes calls; use separate instances for concurrent lanes.

## Reference gate

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
