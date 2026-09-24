# Whisper tiny.en implementation and validation ledger

This package provides **speech-to-text with the official Whisper tiny.en model**. The
original Gophonic models predict whether an audio turn has ended. Smart Turn
contains a Whisper-like audio encoder, but its 8-second classifier bundle has
no text decoder, vocabulary, or transcription output. `AudioSession` remains a
PCM-to-turn-prediction contract; it is not a Whisper transcription interface.

## Pinned sources and first checkpoint

| Item | Pin and source | Use |
| --- | --- | --- |
| OpenAI Whisper source | commit [`86098128c0b4f24f0e2aa2994de830614b474227`](https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227) | [Model](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/model.py), [decoding](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/decoding.py), [audio](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/audio.py), [tokenizer](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/tokenizer.py), and [transcription](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/transcribe.py) oracle |
| First model | official [`tiny.en.pt`](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt) | English; target FP32 execution. The [upstream loader](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/__init__.py) specifies expected SHA-256 `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03` and checks it after download. The converter must independently verify the downloaded bytes. |
| Frontend assets | [`mel_filters.npz`](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/assets/mel_filters.npz) and [`gpt2.tiktoken`](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/assets/gpt2.tiktoken) at the same commit | Convert offline; retain provenance and license notice. The [upstream README](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/README.md#license) says code and model weights are MIT licensed. |
| CPU comparison | [whisper.cpp](https://github.com/ggml-org/whisper.cpp/tree/a664346ea5c6dddff3e61a2b7b32dd4514613f50), built commit `a664346ea5c6dddff3e61a2b7b32dd4514613f50` | Local CPU-only build disables Metal, Accelerate, and BLAS; record the complete build flags and matching run options with each result. Its [bench tool](https://github.com/ggml-org/whisper.cpp/blob/a664346ea5c6dddff3e61a2b7b32dd4514613f50/examples/bench/bench.cpp) covers encoder and decoder stages. |

The published [`tiny.en` configuration](https://huggingface.co/openai/whisper-tiny.en/blob/main/config.json)
maps to the following expected checkpoint dimensions: `n_mels=80`, audio context `1500`,
audio width `384`, `6` heads, `4` blocks; vocabulary `51864`, text context
`448`, text width `384`, `6` heads, `4` blocks. Each attention head has width
`64`; each MLP has width `1536`. The offline converter must reject a checkpoint
whose embedded `dims` disagree with these values. The source and checkpoint
pins above are distinct: the source commit fixes semantics; the checkpoint
SHA fixes weights. The checkpoint was downloaded, verified against that SHA,
converted to FP32, and loaded by the Go parser with all 167 tensor shapes checked.

## Required tensor and operator inventory

`model.py` defines these state-dictionary shapes for `tiny.en`. The first
runtime converts the checkpoint tensors to FP32 during offline export:

| Scope | Named tensors / shapes | Runtime result |
| --- | --- | --- |
| Audio input | mono 16 kHz, at most `480000` PCM samples; `80 × 3000` log-mel | one 30-second window, right padded when short |
| Audio stem | `encoder.conv1.weight [384,80,3]`, `encoder.conv2.weight [384,384,3]`, biases `[384]`, `encoder.positional_embedding [1500,384]` | GELU after each convolution; stride 2 on conv2, producing `[1500,384]` |
| Audio blocks `i=0..3` | `encoder.blocks.i.attn.{query,key,value,out}.weight [384,384]`; query/value/out bias `[384]`, **no key bias**; `attn_ln` and `mlp_ln` weight/bias `[384]`; `mlp.0.weight [1536,384]`, bias `[1536]`; `mlp.2.weight [384,1536]`, bias `[384]` | unmasked self-attention, residuals, LayerNorm, GELU MLP |
| Audio finish | `encoder.ln_post` weight/bias `[384]` | encoded audio `[1500,384]` |
| Text input | `decoder.token_embedding.weight [51864,384]`, `decoder.positional_embedding [448,384]` | embed prefix and each generated token at its absolute text position |
| Text blocks `i=0..3` | self `attn`, `cross_attn`: each query/key/value/out weight `[384,384]`, query/value/out bias `[384]`, no key bias; `attn_ln`, `cross_attn_ln`, `mlp_ln` weight/bias `[384]`; MLP weights/biases as audio blocks | causal self-attention, full-audio cross-attention, residuals, GELU MLP |
| Text finish | `decoder.ln` weight/bias `[384]`; logits use the **same** `token_embedding.weight` transposed | `[51864]` logits for the newest token; no separate output-head tensor |

The reusable operator layer needs FP32 convolution, tiled GEMM and GEMV,
LayerNorm, GELU, stable softmax, and attention. Keep fixed-shape, specialized
dispatch for known Whisper dimensions and a scalar Go reference for parity.
Go 1.27 `simd/archsimd` kernels and packed weight layouts may then be selected
at build time without a runtime ONNX interpreter. Bundle conversion is an
offline operation: validate all names, shapes, dtypes and finite values, pin
the source SHA, and emit a versioned bundle with an integrity checksum.

## Audio and decoding semantics

OpenAI's `audio.py` uses a 400-point Hann-windowed power STFT, hop `160`,
centered reflect padding, drops the final STFT frame, applies the pinned
80-band mel filters, clamps at `1e-10`, then applies the max-minus-8 floor and
`(log10(mel)+4)/4` scaling. Its single-window example **right-pads** PCM to
30 seconds. Gophonic's current turn-detector frontend **left-pads** an 8-second
window and normalizes waveform mean and variance. Reuse its FFT and mel
kernels only through a new official-Whisper preprocessing path; keep the
existing turn-detector feature contract unchanged.

The English `tiny.en` path uses FP32 and greedy decoding at temperature
zero and `without_timestamps=true`. For this model, the initial token sequence
is `<|startoftranscript|>` (`50257`), `<|notimestamps|>` (`50362`); the source
tokenizer supplies suppression rules. The first decoder call processes the full prefix
with a causal mask. Every later call processes one token, using its absolute
position and per-layer self-attention cache. Precompute each decoder layer's
cross-attention keys and values from `[1500,384]` encoded audio once per window.
The four self K/V pairs have capacity `[448,384]` each; the four cross K/V
pairs have shape `[1500,384]` each. Reset token count and both cache lifetimes
between windows. Stop at end-of-text (`50256`) or the text-context/sample limit
(the default sample limit is `448/2 = 224` generated tokens). Apply
the official blank and non-speech token filters before argmax; decode BPE
token byte payloads to UTF-8. Preallocate token IDs, logits, attention scores,
KV storage, features, and output scratch in one session; an `Into` API writes
to caller-provided text storage and reports capacity errors. A string-returning
convenience API may allocate.

`TranscribeWindowInto` produces speech-to-text for one `<=30s` English PCM
window. `TranscribeInto` applies full-file mel normalization, repeated mel
windows, timestamp-token seeking, previous-text prompting, and no-speech
skipping. It fixes temperature at zero and disables timestamp prompting;
temperature fallback, multilingual checkpoints, beam search, and word
timestamps remain future extensions. The API and tests distinguish raw
single-window decoding from full-file transcription because short audio
produces different mel inputs and can produce different text.

## Validation gates

1. **Converter:** verify checkpoint SHA-256, embedded dimensions, exhaustive
   tensor manifest, bundle round-trip, license/provenance record; compare a
   sample of converted weights bitwise with PyTorch.
2. **Frontend:** compare official `pad_or_trim` + `log_mel_spectrogram` against
   Go on silence, edge impulses, tone, and real 16 kHz speech. Validate the
   complete `[80,3000]` output, including beginning/end frames.
3. **Graph:** compare each stem output, every encoder block, final encoder
   output, first decoder prefix logits, subsequent one-token logits, cache
   lengths, and selected token IDs with official PyTorch CPU FP32. Assert exact
   token sequences for a speech corpus where top-logit margins are sufficient;
   use documented numerical tolerances for floating tensors.
4. **User result:** compare final English transcripts for short speech,
   silence, and a full 30-second sample. Add long-audio oracle tests when the
   multi-window stage is implemented.
5. **Runtime:** zero steady-state allocations for `TranscribeInto` after model
   and session creation; repeat-call correctness, concurrent independent
   sessions sharing immutable weights, scalar/SIMD parity, and race tests.

## Performance gate and repository scope

Benchmark the same official `tiny.en` weights and the same 16 kHz audio with
Go, official PyTorch CPU, and CPU-only whisper.cpp. Pin commits and model
checksums. Match FP32/quantization, greedy/no-timestamp options, thread count,
30-second padding and actual output tokens; record the quality result with
latency. Exclude model load from warm inference and report it separately.
Report frontend, encoder, prefix, per-token decoder, and full transcription
medians with repetitions, allocations, CPU model, Go/compiler flags, and
whisper.cpp backend/build flags. Disable GPU/Metal/CoreML in the C++ run;
identify any CPU BLAS/Accelerate use. Treat a speedup as established only when
matched CPU configurations and transcript quality are measured. Existing
2.09 ms Gophonic numbers are for TinyMelNet turn detection, not Whisper STT.

Gophonic now includes a distinct `whisper` package for full transcription.
Keep its model, session, text result, and benchmark track distinct from the
turn-detector `AudioSession` interface. That interface alone does not provide
the encoder, autoregressive decoder, tokenizer, or text result required here.
