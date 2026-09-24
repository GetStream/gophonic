# qwen3: Qwen3-8B in pure Go, on the CPU or the Apple GPU

Package `qwen3` runs the official `Qwen/Qwen3-8B` checkpoint with no cgo and
no inference runtime: the safetensors loader, tokenizer, transformer, and
matrix kernels are Go and Go assembly. On Apple M4 the CPU path uses the SME
matrix units, and the GPU path drives Metal through a pure-Go binding; other
CPUs use portable kernels (NEON on arm64).

| M4 Max, one text | 1 token | 12 tokens | ~70 tokens | 16 × 12 tokens | cosine vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpu` | **15.8 ms** | **25 ms** | **179 ms** | **356 ms** | 0.99933 |
| `gpu-q4` | **11.0 ms** | 25 ms | 159 ms | 331 ms | 0.953 |
| CPU `int8` | 28 ms | 43 ms | 150 ms | 415 ms | 0.99866 |
| CPU exact (default) | 51 ms | 66 ms | 304 ms | 841 ms | **0.99991** |
| llama.cpp Metal Q8_0 | 19.5 ms | 57 ms | 92 ms (64) | — | 0.99933 |
| llama.cpp CPU Q8_0 | 31.8 ms | 90 ms | 487 ms (64) | — | 0.99929 |

```go
m, err := qwen3.Open("/path/to/Qwen3-8B", qwen3.Options{})
if err != nil {
	return err
}
defer m.Close()

// Zero-shot multiple choice: prepare the question once, then one prefill
// per input and no text generation.
q, err := m.Question("What emotion does the writer express?",
	[]string{"joy", "anger", "neutral"})
probs := make([]float32, 3)
err = q.Choose(ctx, "I waited two hours for nothing.", probs)

// Last-token hidden states, e.g. for the CLM ranking heads in ../clm.
dst := [][]float32{make([]float32, 4096)}
err = m.Embed(ctx, []string{"hello"}, dst)
```

The API is allocation-free by design: setup (`Open`, `Question`) allocates
everything up front, outputs go to caller-owned buffers, and hot-path calls
build no strings and do no map lookups (the tokenizer's merge table is a flat
open-addressing array). `Question.ChooseTokens` and `Model.EmbedTokensInto`
take token IDs and involve no strings at all. Calls on one `Model` are
serialized; open one per concurrent lane.

## Question

`Model.Question` builds the Qwen3 chat prompt (non-thinking mode) with the
options listed as `A)`, `B)`, …, tokenizes it once, and evaluates its prefix
once, keeping that prefix's keys and values (about 288 KiB per prompt token).
`Choose` then tokenizes only the input, evaluates the input and a short
suffix against the stored prefix, and compares the next-token logits of the
answer letters; it needs only the language-model head rows of the letters
(read at `Open`, not the 1.2 GB head). The fixed parts end on pre-tokenizer
boundaries, so this equals tokenizing the whole prompt, which a test checks.
Questions do not share state, so alternating between them costs nothing.

`Question.NewStream` follows a growing input, such as a speech recognizer's
partial transcripts: each `Update` evaluates only the tokens added (or
revised) since the previous one plus the 9-token prompt suffix, and matches a
fresh `Choose` on the same text. `examples/qwen3/turn` uses it for
word-by-word turn detection at about 72 ms per word in exact mode.

`Model.NewContext` holds a long shared text, such as a conversation, that
several questions are asked about: `Set` evaluates it once and afterwards only
the tokens that changed, and `Ask` answers any number of `ContextQuestion`s as
short tails against it in shared forward passes. Three questions about a
five-turn support conversation take about 520 ms after each new turn, against
1.5 s for three `Choose` calls. The context comes first in this prompt, which
agrees with `Question` on classification probes but not on turn detection;
`examples/qwen3/conversation` tracks mood, topic, and resolution turn by turn.

`ChooseBatch` answers many inputs in shared forward passes. On an M4 Max, a
new input costs about 140 ms alone, 94 ms per input in a batch of 16, and
47 ms per input in a batch with `Weights: "int8"`.

The programs in [`../examples/qwen3`](../examples/qwen3) are each one short
`main.go`: `turn` (has a voice-agent user finished speaking?), `sentiment`,
`intent`, `tools`, and `subtitles`. On the official checkpoint they answer
30 of 31 inputs as expected, with probability ≥ 0.99 each (the miss:
"footsteps upstairs, I live alone" read as anger rather than fear). A few
hand-written inputs are not an accuracy study; validate on your own data.

```sh
export GOPHONIC_QWEN3_MODEL=/path/to/Qwen3-8B
CGO_ENABLED=0 GOEXPERIMENT=simd go run ./examples/qwen3/turn
```

## CLM ranking

`Embed` returns the post-final-norm last-token state that the
[CLM v0.1 heads](../docs/clm.md) were trained on. Adapt it with one line:

```go
engine, err := clm.NewEngine(head, clm.EmbedFunc(
	func(ctx context.Context, _ clm.Role, texts []string, dst [][]float32) error {
		return m.Embed(ctx, texts, dst)
	}))
```

`examples/qwen3/reply` ranks candidate agent replies (the helpful one wins
with ≥ 0.99 in each conversation) and `examples/qwen3/rank` is a CLI. CLM
v0.1 is not a zero-shot classifier: framing classification as candidate
sentences scored near chance in our probes, which is why the classifiers
above use `Choose`.

## Precision

By default every BF16 weight is stored exactly (FP16 with a power-of-two row
scale, 12.9 GiB); activations entering each projection are rounded to FP16
after an exact per-row power-of-two scale, and products accumulate in FP32.
Against the official BF16 PyTorch hidden state for `hello` this reaches
cosine 0.99991 (llama.cpp Q8_0: 0.99929), and the pinned CLM ranking matches
the official probabilities within 7.1e-5.

`Options{Weights: "int8"}` is a fast mode on the int8 matrix units. Each
projection's input space is rotated with a fixed randomized Hadamard
transform (applied to weight rows at load and to activation rows at run
time, so W·x is unchanged before rounding); weights are stored as per-row
int8 (6.5 GiB) and activations are quantized to int8 per row. Integer
products are exact, and SME and portable kernels give identical results.
Against official BF16 it reaches cosine 0.99866 on `hello` and the pinned
CLM probabilities within 2.2e-4, and the 31 `Choose` probes give the same
answers as the exact mode. It runs a 12-token text in 43 ms, one token in
28 ms, and 16 short texts in 415 ms.

## GPU

`Options{Weights: "gpu"}` runs the model on the Apple GPU (darwin/arm64)
through a pure-Go Metal binding (`internal/metal`, no cgo); the kernels are
Metal shading language source embedded in the package. The residual stream
is kept in a Hadamard-rotated basis, RMSNorm weights and a per-head value
rotation are folded into the weights at load, and each projection is stored
as per-row int8 while activations stay in FP32. A layer is six dispatches,
and weight streaming runs at the measured memory bandwidth (≈440 GB/s).

The table at the top compares it with the CPU paths and llama.cpp; the
llama.cpp Metal 4-bit files reach cosine 0.863 (Q4_0, 12.0 ms for one token)
and 0.942 (Q4_K_M, 12.5 ms), below `gpu-q4`.

A single token streams every weight once through GEMV kernels. Longer
inputs, and several texts packed into one pass, run batched simdgroup-matrix
kernels that read each weight once per 16 or 32 tokens, splitting K across
threadgroups when a projection alone would leave GPU cores idle.

`gpu-q4` stores blocks of 32 weights as 4-bit codes with one FP16 scale,
chosen per block to minimize rounding error. Prefix stores (`Question`,
`Context`, `Stream`) are not yet supported on the GPU.

## Performance

On the M4 Max CPU, 2048 tokens take 8.2 s exact and 4.9 s in `int8`; a
30-token turn appended to an 1800-token state takes 155 ms because the stored
prefix is reused; `Question.Choose` takes 128 ms per new input and
`ChooseBatch` 94 ms (47 ms in `int8`); cached re-ranks take under 2 ms. See
[the performance report](../docs/clm-performance.md).

- **Threads.** `Options.Threads` defaults to min(performance cores,
  `GOMAXPROCS`, 16). Workers live as long as the `Model`; call `Close`.
- **Batching.** Texts in one `Embed` call share 16-row matrix tiles.
- **Caches.** `Options.CacheEntries` sizes an exact embedding cache (default
  4096 × 16 KiB) and `Options.PrefixCacheTokens` a store of the last long
  input's keys and values (default 2048 tokens ≈ 576 MiB); negative values
  disable them. `CacheStats` and `PrefixStats` report reuse.
- **Low-level.** `LoadWeights`, `NewEvaluator`, `Evaluator.HiddenLastBatchInto`,
  and `PrefixKV` evaluate pretokenized batches directly; `EmbedTokensInto`
  accepts token IDs.

## Tests

Unit tests build a small random Qwen3 checkpoint, load it through the real
safetensors loader, and compare every path (both weight formats, SME and
portable kernels, 1–8 workers, packed batches, prefix extension) with a
float64 reference implementation. Official-checkpoint gates are opt-in:

```sh
GOPHONIC_QWEN3_MODEL=/path/to/Qwen3-8B \
GOPHONIC_QWEN3_TOKENIZER=/path/to/Qwen3-8B \
GOPHONIC_QWEN3_HELLO_REFERENCE=/path/to/hello.f32 \
GOPHONIC_CLM_HEAD_BUNDLE=/path/to/CLM_v0.1-8B.gclm \
CGO_ENABLED=0 GOEXPERIMENT=simd go test ./qwen3 ./examples/qwen3 -run 'Official|Tokenizer' -v
```

`tools/reference_hidden.py MODEL hello.f32` writes the BF16 reference vector
and `tools/reference_rank.py` regenerates the golden CLM probabilities. The
tokenizer goldens in `testdata/` come from the official Hugging Face
tokenizer.
