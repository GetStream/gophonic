# Local Qwen3-8B embeddings for CLM

This optional nested module connects `gophonic/clm` to a pure-Go CPU Qwen3-8B
encoder. `Open` loads an official Qwen3-8B safetensors snapshot directory;
`Embed` tokenizes text, keeps the final 2048 tokens, runs Qwen3 to the
post-final-norm hidden state of the last token, and writes the raw
4096-value vector. The CLM head normalizes it once. Both state and action
roles use the same encoder.

There is no cgo and no third-party inference runtime: the safetensors loader,
tokenizer, transformer, and matrix kernels are local. The only dependencies are
gophonic, `golang.org/x/text` (NFC normalization), and
[`vibejson`](https://github.com/thesyncim/vibejson) for JSON.

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

The command prints candidates in descending probability order.
`-weights int8` halves weight memory at a measurable accuracy cost, and
`-threads N` overrides the worker count.

## Precision

By default every BF16 checkpoint weight is stored exactly (FP16 with a
power-of-two row scale, 12.9 GiB). Activations entering each projection are
rounded to FP16 after an exact per-row power-of-two scale, and all products
accumulate in FP32. Against the official BF16 PyTorch hidden state for
`hello`, the default path reaches cosine 0.99991 (llama.cpp Q8_0: 0.99929),
and the pinned CLM ranking matches the official BF16 probabilities within
7.1e-5. `Options{Weights: "int8"}` (6.5 GiB) reaches cosine 0.99737 and
probabilities within 1.0e-3.

## Performance

On an Apple M4 Max CPU, one 12-token text embeds in about 70 ms, a 70-token
text in 317 ms, 16 short texts in 850 ms, and a full 2048-token text in
8.2 s, all with zero warmed allocations; the model loads in about 3 s.
llama.cpp's best CPU run takes 90 ms for 12 tokens and 487 ms for 64. A new
30-token turn on an 1800-token conversation takes 203 ms because the stored
prefix is reused, and re-ranking 16 cached candidates takes 1.7 ms. See
[the performance report](../../docs/clm-performance.md).

- **Kernels.** On CPUs with 512-bit SME (Apple M4), projections run a
  16-row × 64-column FP16 `FMOPA` tile. Other CPUs use a portable panel kernel
  (NEON on arm64 with `GOEXPERIMENT=simd`); it is correct but much slower.
- **Threads.** `Options.Threads` defaults to min(performance cores,
  `GOMAXPROCS`, 8). Workers persist for the encoder's lifetime and spin for
  1 ms between operations before parking; call `Close` to stop them.
- **Batching.** Texts in one `Embed` call share 16-row tiles, up to 512
  tokens per forward pass, so many short candidates cost far less than
  separate calls.
- **Cache.** `Options.CacheEntries` sizes the exact embedding cache (default
  4096 entries, 16 KiB each; negative disables). `CacheStats` reports hits.
- **Prefix store.** Inputs of 64 tokens or more run through a store of the
  last long input's keys and values (`Options.PrefixCacheTokens`, default
  2048 tokens ≈ 576 MiB; negative disables). A later input sharing its token
  prefix evaluates only the new tokens; `PrefixStats` reports reuse.
  `Evaluator.NewPrefixKV` and `HiddenLastExtendInto` expose it directly.
- **Long inputs.** From 64 tokens, attention runs as blocked SME matrix
  products, so cost per token stays flat up to the 2048-token limit.

Calls on one `Encoder` are serialized; use one encoder per concurrent lane.
For lower-level use, `LoadModel`, `NewEvaluator`, and
`Evaluator.HiddenLastBatchInto` evaluate caller-tokenized batches directly.
`Encoder.EmbedTokensInto` accepts pretokenized IDs.

## Reference gates

The unit tests build a small random Qwen3 checkpoint, load it through the
real safetensors loader, and compare every path (both weight formats, SME and
portable kernels, 1–8 workers, packed batches) with a float64 reference
implementation. The official-checkpoint gates are opt-in:

```sh
GOPHONIC_QWEN3_MODEL=/path/to/Qwen3-8B \
GOPHONIC_QWEN3_TOKENIZER=/path/to/Qwen3-8B \
GOPHONIC_QWEN3_HELLO_REFERENCE=/path/to/hello.f32 \
GOPHONIC_CLM_HEAD_BUNDLE=/path/to/CLM_v0.1-8B.gclm \
CGO_ENABLED=0 GOEXPERIMENT=simd go test -run 'TestOfficial|TestQwenTokenizer' -v
```

`tools/reference_hidden.py MODEL hello.f32` writes the BF16 reference vector,
and `tools/reference_rank.py` regenerates the golden CLM probabilities. The
tokenizer goldens in `testdata/` were produced by the official Hugging Face
tokenizer. One short ranking cannot establish corpus-level accuracy.
