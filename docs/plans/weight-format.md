# A weight format for a 4 GB voice agent

Status: plan, 2026-09-26, branch `perf/footprint`. Goal: Gopher (ears, brain,
voice) in about 4 GB instead of 44 GB, a tenth, with losses we measure and
accept, not guess.

## 1. Where the memory is, measured

| Part | Now | Bytes |
| --- | --- | ---: |
| Language model | Qwen3.6-35B-A3B, `gpu-q8` | 36.3 GB |
| Recognizer | Qwen3-ASR-1.7B, `gpu-q8` + encoder | 2.5 GB |
| Voice | Qwen3-TTS-12Hz-1.7B, `gpu-q8` + codec | 1.6 GB |
| GPU scratch | 68 buffers of 8 MB or more | 4.2 GB |
| **Total** | | **44.6 GB** |

The weights are file-backed and never wired (wired memory stayed at 6.0 GB
with Gopher running), but they are Gopher's working set: evicted, they are
read again from disk. Loading is already solved: the cache file is the GPU's
layout, mapped without copying, and three models open in 1.5 s. What is
left to win is bytes per weight, the models' sizes, and scratch.

## 2. What smaller models cost, measured

| Swap | Quality | Size |
| --- | --- | ---: |
| ASR 1.7B → 0.6B | LibriSpeech test-clean WER 1.64% → 2.38% (200 utterances) | 2.5 → ~1.0 GB |
| TTS 1.7B → 0.6B | heard back by Qwen3-ASR-1.7B: no misheard word (only number spelling) | 1.6 → ~1.0 GB |
| LLM → Qwen3-4B-Instruct-2507 | 9 of 11 scenarios; its zero-shot wake judgment says yes to anything | 36.3 → 4.4 GB (8-bit) |
| LLM → Qwen3-8B | wake judgment best of the three (0.01 on an unrelated question) | 8.0 GB (8-bit) |

Bits per weight, Qwen3-4B against its BF16 weights, next-token KL divergence
over 9,960 positions of LibriSpeech's book text:

| Format | bits/weight | KL (nats) | top-1 agreement | perplexity |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 16 | 0 | 100% | 46.61 |
| `gpu-q8` | 8.5 | 0.0006 | 98.4% | 46.72 |
| `gpu-q4`, rounded | 4.5 | 0.115 | 81.8% | 51.82 |
| `gpu-q4`, GPTQ | 4.5 | 0.038 | 89.0% | 47.86 |

For scale: llama.cpp's Q8_0 sits near KL 0.001, Q4_K_M near 0.03–0.05.
GPTQ at 4 bits is already usable; the format below aims at KL ≤ 0.01 at
about 4 bits, and ≤ 0.02 at about 3.

## 3. The format: rotated, codebook-quantized, mixed-rate, mapped

Each piece is established in the literature; together they fit this engine
unusually well, because its GPU formats already rotate every projection.

**3.1 Incoherence, already paid for.** Every projection input is rotated by a
randomized Hadamard transform (`hadamard.go`), folded into the weights. After
it, each block of a weight row is close to Gaussian with no outliers: the
precondition of QuIP#, QTIP, and QuaRot. Nothing new at run time.

**3.2 Levels that fit a Gaussian.** `gpu-q4` spreads 16 evenly spaced levels
over ±max. For Gaussian values, the mean-squared-error-optimal 16 levels
(Lloyd–Max) are denser near zero; at 4 bits they cut the rounding error's
power by about a third, the equivalent of a quarter to half a bit. The level
table is fixed per rate (computed once, in the file), so decoding is a
16-entry lookup the GEMV already has registers for. At 3 bits, 8 levels.

**3.3 Scales that cost less.** Today every 32 values carry a 16-bit scale:
0.5 bits per weight. Two levels of scale, as llama.cpp's K-quants: an FP16
scale per 256 values and a 6-bit scale per 32 within it, 0.25 bits per
weight, with each 32's scale searched for least squared error, as
`q4GroupScale` already does.

**3.4 Error-compensated rounding.** GPTQ rounds each column knowing the
error of the ones before, against the Hessian of calibration activations in
the rotated basis. It accepts any grid of levels, so it runs unchanged on
3.2's. It now runs at any width (the 4096-multiple limit is gone). The
calibration text becomes in-domain: conversations like Gopher's, with tool
calls and moments, not only prose.

**3.5 Bits where they matter.** Some tensors are far more sensitive than
others (attention value and output, `down_proj`, the tied head). For each
tensor and each candidate rate (3, 4, 5, 6, 8 bits), measure the KL on
held-out text with only that tensor quantized, then choose rates that
minimize the summed KL under a byte budget: a knapsack solved greedily by
KL saved per byte, as EXL2 and EXL3 allocate rates. The budget, not a
format name, is what one asks for: "4.0 bits per weight on average".

**3.6 Mapped as it runs.** The file stays what the GPU reads: 16 KB-aligned
tensors, codes and scales in the layout the kernels stream, the level tables
beside them, and a header listing each tensor's rate, offset, and geometry.
No decoding at load; file-backed, so the system can reclaim it.

**3.7 One file to ship.** Today the cache is made on first load from the
BF16 checkpoint: 16 GB downloaded for an 8B model, and a minute of
preparing. A packed model (`gophonic pack Qwen3-8B -bits 4`) writes the same
bytes as a single versioned, checksummed file that `gophonic.Open` maps
directly: 4 GB downloaded, no preparing. The checkpoint's tokenizer and
config ride along.

**Beyond (only if 3.1–3.5 miss the budget):** trellis-coded quantization
(QTIP, the basis of EXL3) reaches near-lossless 4-bit and good 2–3 bit
quality by coding 256 weights at a time along a trellis whose values are
computed, not looked up. It needs new GEMV and matmul kernels with a serial
decode per block; QTIP reports memory-bandwidth-bound matvec on GPUs. It is
the next step up, not the first.

## 4. Scratch, sized to need

The 4.2 GB of scratch is all sized by worst-case contexts, and some is
duplicated:

- **ASR and TTS lanes:** two lanes each, and every lane allocates a 224 MB
  workspace and a 112–224 MB KV prefix, 1.6 GB together. Grow them to the
  longest turn seen, starting from a turn's size, and give the transcriber's
  partials and judgments one lane.
- **Qwen3.6's states:** 60 MB per zero-shot question (three), per session
  prefill, and per rewind snapshot (five). The hybrid model's snapshots can
  be kept as deltas or fewer. A dense model has none.
- **KV caches:** 8-bit keys and values per head halve them, measured by the
  same KL gate.

## 5. Gates: what "acceptable" means

Every step is measured before it is kept:

- **KL and top-1 agreement** against BF16, on held-out prose and on
  Gopher-style conversations (a tool in `tools/`, not a scratch test).
- **Gopher's scenarios:** 11 of 11 on the real models, the harness judging
  with the reference models.
- **ASR:** LibriSpeech WER within 0.2 points of `gpu-q8`; turn accuracy on
  the 538 held-out clips within a point.
- **TTS:** heard-back WER, and `TestVoicedFollowsWords`.
- **Speed:** GEMV bandwidth at least `gpu-q4`'s; first audio no later.
- Zero allocations on warm paths; deterministic preparation.

## 6. Steps

1. **Measure.** The KL tool and corpora in `tools/`; baselines for every
   model and format above.
2. **Gaussian levels** (`gpu-n4`, `gpu-n3`): 3.2 and 3.3 in the GEMV and
   matmul kernels, with GPTQ. Expect KL ≤ 0.02 at 4.25 bits.
3. **Mixed rates** (3.5): sensitivity measurement and the knapsack; a model
   prepared for a byte budget.
4. **Packed files** (3.7): `gophonic pack`, and `Open` of a packed file.
5. **ASR and TTS decoders** in the same format, calibrated with real audio
   inputs (their decoders read embeddings, not only tokens), and the 0.6B
   heads: the recognizer's turn head (training data being written) and the
   voice's alignment head (ranked against Whisper's word timings).
6. **Scratch** (section 4).
7. **Choose Gopher's default** from the measurements: probably Qwen3-8B at
   about 3.5–4 bits (3.6–4.1 GB), both 0.6B speech models, and scratch under
   1 GB: about 5–6 GB, or 4B-based about 3.5–4 GB.
8. **Only if needed:** trellis coding for 3 bits and below.

## 7. Risks

- GPTQ on small models overfits calibration text: held-out gates catch it.
- The tied head and embeddings are a tenth of a 4B model and read every
  token: they get a rate of their own in 3.5.
- 3-bit codes do not align to bytes: 256-value super-blocks (96 bytes) keep
  loads aligned.
- A 4B model's zero-shot judgments are weaker: the gates measure the
  judgments themselves, and the 8B stays the reference choice.
