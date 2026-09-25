# Whisper design

The `whisper` package runs OpenAI's English Whisper checkpoints end to end in
Go: log-mel frontend, audio encoder, incremental text decoder, and BPE
tokenizer. `Transcriber` implements `speech.Transcriber`.

## Sources

| Item | Pin |
| --- | --- |
| Model semantics | OpenAI Whisper [`86098128`](https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227): `model.py`, `audio.py`, `decoding.py`, `tokenizer.py`, `transcribe.py` |
| Frontend assets | `mel_filters.npz` and `gpt2.tiktoken` at the same commit (MIT) |
| Checkpoints | official `tiny.en`, `base.en`, `small.en`, and `medium.en`, accepted only with their published SHA-256 |

## Models and bundles

[`tools/whisper_pt_to_gophonic.py`](../tools/whisper_pt_to_gophonic.py) reads
the PyTorch archive with a restricted unpickler. It checks the checkpoint's
embedded dimensions and exports every tensor as FP32 with its shape and a
SHA-256 over the payload. Version-2 bundles also record the model dimensions
inside that checksummed payload. At load time, names, shapes, finite values,
and the checksum are all validated.

| Model | Width | Heads | Layers (encoder / decoder) |
| --- | ---: | ---: | ---: |
| tiny.en | 384 | 6 | 4 / 4 |
| base.en | 512 | 8 | 6 / 6 |
| small.en | 768 | 12 | 12 / 12 |
| medium.en | 1024 | 16 | 24 / 24 |

All English models share 80 mel bins, 1500 audio frames, a 448-token text
context, and a 51,864-token vocabulary. `Model.Dims()` reports the rest, and
transcribers size their scratch from it.

## Pipeline

1. **Frontend:** 400-point periodic-Hann power STFT with hop 160 and centered
   reflect padding; the final frame is dropped. Then the 80-band mel filters,
   a `1e-10` clamp, the max-minus-8 floor, and `(log10(x)+4)/4`.
2. **Encoder:** two convolutions with GELU (the second has stride 2), sinusoidal
   positions, then N pre-norm blocks of unmasked self-attention and a GELU MLP,
   and a final LayerNorm.
3. **Decoder:** token and position embeddings, then N blocks of causal
   self-attention, cross-attention over the encoded audio, and an MLP. Logits
   reuse the token embedding. Cross-attention keys and values are projected
   once per window. Each token step appends to the self-attention cache and
   runs one position.
4. **Decoding:** greedy at temperature zero, with the reference blank and
   non-speech suppression. Decoding stops at end-of-text or the sample limit.

`TranscribeWindowInto` decodes one right-padded window of at most 30 seconds.
`TranscribeInto` handles audio of any length. It normalizes mel over the
whole file and applies timestamp-token seeking, previous-text prompting, and
the no-speech rule. Temperature fallback, beam search, multilingual models,
and word timestamps are not implemented.

## Execution

Scratch is preallocated per `Transcriber`, and weights are immutable and
shared, so warm calls allocate nothing. Kernels and numerical guarantees are
described in [Whisper performance](whisper-performance.md).

## Validation

- **Converter:** exact checkpoint SHA, embedded dimensions, the complete
  tensor manifest, and strided-tensor reads compared with PyTorch values.
- **Frontend:** OpenAI reference features on speech, silence, and padding
  (max 5e-5, RMSE 2e-6).
- **Graph:** every encoder stage compared with PyTorch activations;
  decoder logits and exact token sequences checked.
- **Transcripts:** the test suite requires the exact JFK text for tiny.en,
  plus full-file, silence, and long-input cases. The benchmark harness checks
  the exact text for tiny.en, base.en, and small.en on every call.
- **Runtime:** zero warm allocations, results independent of worker count,
  race tests, and SIMD and scalar builds.
