# SFU showcase plan: "Gopher", a local voice co-pilot on Stream Video

Status: plan, 2026-09-25. Scope: audio only (speech in, speech and text
out). Nothing here is built yet; every number that is not a citation is an
estimate derived from measurements already in this repository and is marked
as such.

## 0. Summary

**Flagship.** `examples/gopher`: a Go binary that joins a Pronto call as a
real participant and behaves like a colleague on the call:

1. **Talks.** Address it ("Gopher, ...") and it answers by voice with
   **≤ 500 ms** from the end of your sentence to the first audio the other
   participants hear (target p50; ≤ 800 ms p95), with barge-in: talk over
   it and it stops within one 20 ms frame.
2. **Captions.** Live partial captions for every speaker, in the language
   spoken (30 languages), pushed to the call as native closed captions
   (server API) or as chat messages (user token only).
3. **Interprets.** A second participant, "Gopher (Deutsch)" (any of 10
   target languages), speaks each person's turn in the target language **in
   that person's own voice** (3-second voice cloning), about 1.5–2.5 s
   behind the speaker.
4. **Remembers.** "Gopher, what did we decide about the rollout?" is
   answered from the whole meeting; at the end it posts a summary and
   action items to the chat.

Every model runs in-process through gophonic in pure Go (SME/NEON assembly
and the pure-Go Metal binding); Stream is transport and chat only.

**Models (all Apache-2.0, all Qwen-family so one transformer core serves
them):**

| Role | Model | Why |
| --- | --- | --- |
| STT, streaming | Qwen3-ASR-1.7B (already ported), streaming mode | Best open multilingual ASR at this size (LibriSpeech 1.63 / Fleurs-en 3.35 WER offline; streaming +0.64 avg); trained for streaming; 226 ms for 11 s here today |
| Turn taking | Smart Turn v3.2 (ported) + text end-of-turn on Qwen3.6 + SFU audio levels | 3.6–40 ms per decision; no newer open model found (v3.x is current) |
| LLM | **Qwen3.6-35B-A3B** (Apr 2026, hybrid Gated DeltaNet + MoE, 3 B active); stage on Qwen3-8B (ported) and Qwen3.5-9B | Intelligence Index 43, second only to Qwen3.6-27B (46) under 150 B and ahead of Gemma 4 31B (39); ~2.9 GB streamed per token in int8 → estimated ~8–9 ms/token on this M4 Max (MLX 4-bit: 130 tok/s; llama.cpp Q4: ~71 tok/s) |
| TTS, streaming | **Qwen3-TTS-12Hz-1.7B** (CustomVoice for the agent; Base for cloning) | Qwen3-1.7B geometry talker → `internal/qwen3lm` reuse; 97–101 ms first packet in the paper; Seed-TTS-eval WER 1.24 en / 0.77 zh, beats CosyVoice 3 and MiniMax; causal codec decoder |
| Translation | The same Qwen3.6-35B-A3B, non-thinking mode | No separate model; clause-level streaming translation |

**Memory.** ≈ 41 GB of weights at int8 (LLM 35, ASR 2.3, TTS 2.4, turn
0.03) plus ≈ 1 GB of caches and the WebRTC process: fits the 64 GB M4 Max
with the Metal wired limit raised; the light preset (Qwen3.5-9B) needs
≈ 15 GB.

**Milestones (one engineer; two can overlap M2 and M4):** M1 text agent on
the call (1.5 wk) → M2 voice agent with Qwen3-TTS on Qwen3-8B (3 wk) →
M3 streaming ASR, captions, ≤ 500 ms voice-to-voice (2 wk) → M4 hybrid
core, Qwen3.5-9B then Qwen3.6-35B-A3B (4–5 wk) → M5 interpreters with
voice cloning, summary, benchmark report (3 wk). ≈ 14–15 weeks serial,
≈ 10 with two people.

**Top risks.** (1) The hybrid Gated DeltaNet + MoE core is new numerical
work with no existing Go reference; mitigated by validating each block
alone (MoE on Qwen3-30B-A3B, GDN on Qwen3.5-0.8B/9B) against float64 and
official BF16 states. (2) Qwen3-TTS naturalness: objective metrics are
state of the art but the hosted "Qwen3 TTS Flash" sits low on the
Artificial Analysis arena; the best open-weight voices (Breeze TTS 2,
Voxtral TTS) are non-commercial. (3) One GPU serving three models at once;
mitigated by a priority arbiter with bounded work units. (4) Thermal
throttling (≈ 35 % loss under sustained load) and the 48 GB default Metal
working set.

## 1. What exists today (verified in the tree)

- **Typed lanes.** `gophonic.Open` → `Model`; `Lane[T]` opens any interface
  a format provides (`gophonic.go`, `formats.go`). Lanes own scratch; warm
  calls allocate nothing; `Pool` shares models. Built-in lanes:
  `speech.Transcriber` (Qwen3-ASR, Whisper), `speech.TurnDetector` /
  `AudioClassifier` (Smart Turn, TinyMelNet), `speech.ZeroShot` (Qwen3).
- **`internal/qwen3lm`.** Qwen3 dense transformer with formats `f16`,
  `int8` (SME), `gpu` (int8 per row), `gpu-q8` (blocks of 32, Hadamard
  rotated), `gpu-q4`; `PrefixKV` with `CommonPrefix` /
  `HiddenLastExtendInto` (prefix reuse), `Embeds` (placeholder rows spliced
  from an encoder), `LogitsInto` (head on the GPU), an allocation-free
  tokenizer, and the Metal kernels specialized by geometry at load
  (`gpu_darwin.go`: `gemv`, batched `pass`, tiled attention, one command
  buffer per pass). GPTQ rounding for the GPU formats. There is **no
  sampling loop, no chat template, no linear attention, no MoE**.
- **Qwen3-ASR** (`qwen3asr/`): whole-clip mel → AuT encoder (windowed
  attention, 104-token = 8 s windows, one embedding per 80 ms) → Qwen3
  decoder with spliced audio embeddings, greedy decode, prompt-prefix KV
  reuse. GPU: 226 ms for 11 s (19 ms encoder, 49 ms prefill, ~5 ms per
  token). **No partial results**: `Transcribe` takes a finished clip.
- **Turn detection**: Smart Turn v3.2 at 3.6–46 ms depending on threads.
- **Weight cache**: Qwen3-8B opens in 0.09 s, Qwen3-ASR in 0.085 s.
- **`examples/streamcall`** (own module): joins Pronto with a token from
  `/api/auth/create-token`, subscribes to every microphone, decodes Opus
  to 16 kHz with gopus (also its SILK VAD), Smart Turn every 200 ms of
  silence, transcribes finished turns, classifies mood with Qwen3-8B,
  posts to the call chat over the Chat REST API. It publishes nothing.
- Measured on this machine (docs): Qwen3-8B 15.2 ms/token GPU
  (llama.cpp Metal Q8_0 18.6 ms), Qwen3-1.7B 4.67 ms/token, prefill
  ≈ 670 tok/s at 512 tokens, decode within 12 % of the 440 GB/s bandwidth
  floor.

## 2. What the Stream SDK and SFU allow (verified in the module cache)

`github.com/GetStream/getstream-go-webrtc@v0.0.0-20260923215301-80daf64c2fc1`:

- **Audio in.** `audiortc.NewTrackReader(track, ReaderConfig{Opus:
  opus.Config{SampleRate: 16000}})` → `Frames()` yields 16 kHz mono PCM
  per RTP packet, with packet-loss concealment (`MaxConceal`). One reader
  per remote track; an unread track stalls.
- **Audio out.** `audiortc.NewTrackWriter(WriterConfig{Opus:
  opus.Config{SampleRate: 24000}})` accepts PCM at any rate (Opus encodes
  24 kHz natively, so Qwen3-TTS output needs no resampling), queues 20 ms
  Opus frames, emits encoded silence when the queue runs dry (keeps the RTP
  timeline continuous), reports the audio level for the header extension,
  and has **`Clear()`** (drop queued audio: barge-in), **`Buffered()`**
  (backpressure), **`Flush()`** (end of utterance). `NewAudioTrack` +
  `call.AddTrack` publish it; the SFU's `PublishOptions` are advisory.
  Encoding is pure Go (gopus). RED redundancy is available
  (`WithRedTranscodingEnabledForAudio`).
- **Events.** `HandleCallEvent` for `TrackPublished/Unpublished`,
  `ParticipantJoined/Left`, **`DominantSpeakerChanged`**,
  **`AudioLevelChanged`** (SFU-side levels for every participant: free
  activity hints), `ChangePublishOptions`, reconnect/migration handled by
  the SDK (`restorePublishedTracks`).
- **Data channels are not used** by the SDK (`OnDataChannel(nil)`), so text
  goes through Stream's REST APIs: the Chat channel (works with the Pronto
  user token, as today) and, with an API secret, the server SDK's
  `Call.SendClosedCaption{SpeakerID, Text, StartTime, EndTime, Language,
  Translated}` (`POST /api/v2/video/call/{type}/{id}/closed_captions`),
  which clients receive as `call.closed_caption` events, the same event
  Stream's own captioning emits, and `Call.SendCallEvent{Custom}` for
  custom events.
- **Multiple voices.** A participant publishes one `TRACK_TYPE_AUDIO`;
  each interpreter language is therefore its own `rtc.Client`/participant
  ("Gopher (Deutsch)"), which also lets every listener mute the languages
  they do not want with Pronto's per-participant volume.
- **Echo.** The SFU never sends our own track back (we do not subscribe
  to ourselves), and browsers run AEC on their microphones, so the bot
  needs no acoustic echo canceller. Residual echo from a participant whose
  AEC fails is handled in software (section 8.3).
- Video and screen-share tracks exist in the protocol but are out of scope
  for this audio-only plan.

## 3. The model landscape as of September 2026, per role

Research date 2026-09-25. Sources are linked inline; numbers are the
publishers' unless marked "measured here".

### 3.1 Speech recognition

| Model | Open? | Streaming | Quality | Notes |
| --- | --- | --- | --- | --- |
| **Qwen3-ASR-1.7B / 0.6B** (Jan 2026) | Apache-2.0 | Yes: unified offline/streaming model, dynamic 1–8 s attention window; evaluated with 2 s chunks, 5-token fallback, last four chunks unfixed; streaming WER 3.33 avg vs 2.69 offline (LibriSpeech/Fleurs), TTFT 92 ms (0.6B) | LibriSpeech clean 1.63 vs Whisper-large-v3 1.51; WenetSpeech 4.97 vs 9.86; 30 languages + 22 dialects; Qwen3-ForcedAligner-0.6B gives word timestamps (AAS 28–43 ms) | Already ported; streaming recipe reproduced by [qwen3-asr.cpp](https://github.com/JohnsonChang123/qwen3-asr.cpp) (cached 104-token windows; streaming differs from offline "by one punctuation mark in 327 characters"). [Report](https://arxiv.org/html/2601.21337v1), [repo](https://github.com/QwenLM/Qwen3-ASR) |
| Qwen-Audio-3.0-ASR (+ -Streaming) (Sep 2026 report) | Not found as open weights | Yes | MoE LLM-ASR, 30 languages | [Report](https://arxiv.org/abs/2609.07549); no HF weights located, so not plannable |
| VibeVoice-ASR-Streaming 1.5B/7B (Microsoft, Sep 2026) | MIT | Yes, speaker-attributed, 2.9 s chunks + 0.5 s lookahead, ≈ 2 s latency | Best of five sets in the report | Different backbone (not Qwen), 2 s inherent latency; interesting for diarized captions later. [HF](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B) |
| Voxtral Mini 4B Realtime (Mistral) | Apache-2.0 | Yes, 80–2400 ms configurable, 480 ms recommended | Competitive | Mistral architecture: a second transformer core to port. [Northflank survey](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks) |
| Parakeet TDT 0.6B/1.1B v3 (NVIDIA) | CC-BY-4.0 | RNN-T style | RTFx > 2000; English-centric/25 langs | FastConformer + TDT: a new architecture; fastest, not the most accurate |
| Whisper-large-v3-turbo | MIT | No (chunked) | Baseline | Slower and worse than Qwen3-ASR on non-English |

**Choice: Qwen3-ASR-1.7B in streaming mode** (0.6B as the low-latency
preset when four or more people talk at once). It is already validated
here, the streaming recipe is published, and the decoder is the shared
Qwen3 core. The June 2026 "native Transformers" `-hf` checkpoints are the
same weights.

### 3.2 Turn taking

Smart Turn v3.2 (ported, 23 languages) is the current open model; the
[Pipecat repo](https://github.com/pipecat-ai/smart-turn) shows v3.x only,
no v4. Complement it with (a) text end-of-turn on the running partial
transcript with the LLM (this repo's `Question.NewStream` already does
word-by-word turn detection at 31 ms per update on Qwen3-8B GPU) and (b)
the SFU's `AudioLevelChanged` events. No new model to port.

### 3.3 Dialogue LLM

Fits in 64 GB alongside ASR + TTS, decodes fast enough for speech (a
40-token spoken answer must take well under a second), and quality:

| Model | Arch | Total / active | AA Intelligence Index | Est. int8 bytes streamed per token | Est. ms/token here (440 GB/s, +30 % overhead) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B (ported) | dense, classic Qwen3 | 8.2 B | (2025 model; below every 2026 entry) | 7.0 GB | **15.2 measured** |
| Qwen3-30B-A3B-Instruct-2507 | classic Qwen3 attention + MoE (128 experts, top-8) | 30.5 / 3.3 B | (2025) | ≈ 3.5 GB | ≈ 10 |
| Qwen3.5-9B (Feb 2026) | hybrid GDN 3:1 dense | 9 B | 2026-class, "9B beats 120B" claims | ≈ 8 GB | ≈ 22 |
| **Qwen3.6-35B-A3B** (Apr 2026) | hybrid GDN 3:1 + MoE (256 experts, top-8 + 1 shared), 40 layers, hidden 2048, vocab 248 320 | 35 / 3 B | **43** (Qwen3.6-27B: 46, Qwen3.5-27B: 42, Gemma 4 31B: 39) | ≈ 2.9 GB | **≈ 8–9** |
| Qwen3.6-27B / Qwen3.8-27B (Aug 2026) | hybrid GDN dense | 27 B | 46 / newer | ≈ 27 GB | ≈ 70 (too slow to speak) |
| Gemma 4 26B-A4B / 31B (Apr 2026) | Gemma (new core) | 26 / 3.8 B | 31 (26B, older index) / 39 | ≈ 4.5 GB | ≈ 13 |
| gpt-oss-20b | MoE (new core: attention sinks, sliding windows, MXFP4) | 21 / 3.6 B | ≈ 24 | ≈ 4 GB | ≈ 12 |

Sources: [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
(config fetched: 40 layers as 10 × (3 GDN + 1 gated attention), GDN 16 K
heads / 32 V heads × 128, conv kernel 4, attention 16 Q / 2 KV heads × 256
with partial RoPE 0.25, 256 experts of width 512, 8 routed + 1 shared,
vocab 248 320, 262 K context; SWE-bench Verified 73.4, AIME 2026 92.7,
MMLU-Pro 85.2; non-thinking mode supported with temperature 0.7, top-p
0.8, top-k 20, presence penalty 1.5);
[Artificial Analysis](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)
and their [sub-32B article](https://artificialanalysis.ai/articles/sub-32b-open-weights)
(Qwen3.6 35B A3B 43, Qwen3.5 27B 42, Gemma 4 31B 39; Qwen3.6-27B 46 is the
open leader under 150 B);
[Qwen 3.5→3.8 lineup](https://codersera.com/blog/qwen-3-5-complete-guide-2026/)
(Qwen3.8 released 27B dense and 2.4T only, no A3B);
[Qwen3.5 small models](https://rits.shanghai.nyu.edu/ai/qwen-3-5-small-models-9b-parameters-that-beat-120b/);
[Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/);
Apple-silicon references for Qwen3.5-35B-A3B on an M4 Max:
[MLX 4-bit 126–131 tok/s, llama.cpp Q4_K_XL ≈ 71 tok/s](https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/).

**Choice: Qwen3.6-35B-A3B.** It is the strongest open model that can
answer at conversational speed on this machine: 3 B active parameters keep
decode near 8–9 ms/token in int8 while the Intelligence Index (43) is
within three points of the best sub-150 B open model, which is 9× slower.
The 27B dense models are better but decode at ≈ 70 ms/token in int8, which
is a 2.8 s wait for a 40-token sentence; 4-bit would halve that but falls
below the Q8_0 fidelity bar. Gemma 4 and gpt-oss need a second transformer
core for no quality gain. Non-thinking mode for speech; thinking mode can
be switched on for "think about it" questions with a spoken "let me think".

**Staging.** Qwen3-8B runs today and carries M1–M3. Qwen3.5-9B is the first
hybrid target (no MoE: isolates the Gated DeltaNet work) and stays as the
light preset (≈ 15 GB total). Qwen3-30B-A3B-Instruct-2507 is an optional
intermediate that isolates the MoE work on the classic core.

**Speed estimate detail for Qwen3.6-35B-A3B (estimate, per token, int8):**
30 GDN layers × (in_proj 2048→12 288: 25.2 M, out_proj 4096→2048: 8.4 M)
= 1.0 GB; 10 attention layers × (q 2048→8192 with gate, k/v 2048→512,
o 4096→2048 ≈ 27 M) = 0.27 GB; 40 MoE layers × 9 experts × 3 × 2048 × 512
= 1.13 GB; head 248 320 × 2048 = 0.51 GB; routers, norms, embeddings row
reads: negligible. ≈ 2.9 GB → 6.6 ms at 440 GB/s; with 40 layers of
dispatch overhead expect 8–9 ms (110–125 tok/s), i.e. above llama.cpp's
4-bit and near MLX's 4-bit at Q8 fidelity.

### 3.4 Text to speech

| Model | License | Backbone | Codec | First packet | Quality evidence |
| --- | --- | --- | --- | --- | --- |
| **Qwen3-TTS-12Hz-1.7B / 0.6B** (Jan 2026) | Apache-2.0 | Qwen3 LM: talker 28 layers × 2048 hidden, 16 Q / 8 KV heads × 128, MLP 6144 (= Qwen3-1.7B geometry); code predictor 5 layers × 1024 | Qwen3-TTS-Tokenizer-12Hz: 12.5 Hz, 16 RVQ codebooks × 2048, causal ConvNet decoder (8-layer 512-wide transformer + upsampling 8·5·4·3·2·2 to 24 kHz) | 101 ms (LM 97 + decoder 4), RTF 0.31 at 1× on their server; "first packet after a single character" | Seed-TTS-eval WER 1.24 en / 0.77 zh (CosyVoice 3: 1.45 / 0.71, MiniMax 1.65 / 0.83); best WER in 6/10 and best speaker similarity in 10/10 languages vs MiniMax and ElevenLabs; 10 languages; 3 s cloning (Base), 9 preset voices (CustomVoice), instruction voice design |
| Breeze TTS 2 (Aug 2026) | code Apache-2.0, **weights non-commercial** | T5Gemma2 text encoder + backbone + depth decoder | Qwen3-TTS tokenizer | < 40 ms TTFA on H100 | **#1 open weights on Artificial Analysis (Elo 1205–1215)**, beats ElevenLabs v3; weights en/zh |
| Voxtral TTS (Mar 2026) | **CC BY-NC 4.0** | 3.4 B decoder + 390 M flow-matching + 300 M codec | 12.5 Hz | ≈ 90 ms | Elo 1078 open weights |
| Fish Audio S2 Pro | restrictive | | | | Elo 1122 |
| CosyVoice 3 | Apache-2.0 | Qwen2-style LM + flow matching + vocoder | 25 Hz | ≈ 150 ms | Strong WER; needs a DiT/flow pipeline |
| Kokoro-82M | Apache-2.0 | StyleTTS2 | | < 100 ms | Fast, fixed voices, no cloning, dated quality |

Sources: [Qwen3-TTS report](https://arxiv.org/html/2601.15621v1),
[repo](https://github.com/QwenLM/Qwen3-TTS),
[talker config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/raw/main/config.json),
[tokenizer config](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/raw/main/config.json),
[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) and
[its license split](https://wavespeed.ai/blog/commercial-compliance/breeze-tts-2-review/),
[Voxtral TTS](https://www.marktechpost.com/2026/03/28/mistral-ai-releases-voxtral-tts-a-4b-open-weight-streaming-speech-model-for-low-latency-multilingual-voice-generation/),
[Artificial Analysis TTS arena](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice)
via [this summary](https://offlinetts.com/blog/tts-arena-leaderboard-2026/)
(open-weight top 5: Breeze TTS 2 1205, Fish S2 Pro 1122, Step Audio EditX
1095, Voxtral TTS 1078, Magpie 357M 1063; hosted "Qwen3 TTS Flash" ≈ 929).
Apple-silicon reference for the 2.1 B Qwen3-TTS-12Hz pipeline (talker
1.41 B + code predictor 175 M + codec 171 M): RTF 0.876 → 0.253 after
tuning on an M4 MacBook Pro
([drmhse](https://www.drmhse.com/posts/tuning-qwen3-tts-apple-silicon-m4/)).

**Choice: Qwen3-TTS-12Hz-1.7B** (CustomVoice "Ryan"/"Aiden" for the agent,
Base for interpreter voice cloning), 0.6B as the fallback preset. It is the
only top-tier model that is commercially usable, streams from the first
frame with a causal decoder, and whose talker is literally the Qwen3-1.7B
geometry that `internal/qwen3lm` already runs at 4.67 ms/token. Its
weakness is the arena Elo of the hosted variant; the objective metrics
(WER, speaker similarity) lead the field. Breeze TTS 2 shares the codec
and would be the quality upgrade if its weights were relicensed; it is not
plannable for a Stream showcase.

**Estimate, Qwen3-TTS on this machine** (from measured Qwen3-1.7B decode):
one 80 ms frame = 1 talker step (≈ 4.7 ms, gpu-q8) + 15 code-predictor
steps of a 5-layer 1024-wide model (≈ 3–5 ms as one command buffer with
on-GPU sampling) + codec decode of one frame (≈ 1–2 ms) ≈ 10 ms per frame:
**RTF ≈ 0.13**, first audio (one frame, since the decoder is causal)
≈ 40–60 ms after the first text tokens including the text prefill. The
paper packets 4 frames (320 ms); we can ship 1–2 frames per packet.

### 3.5 Speech-to-speech models, considered and rejected

Qwen3-Omni's talker is not self-hostable (vLLM serves only the thinker;
[source](https://www.callmissed.com/blog/qwen-voice-agent-benchmarks-builder-guide)),
Qwen3.5-Omni-Plus is proprietary, Moshi/LFM2-Audio-1.5B are end-to-end but
below cascade quality on VoiceBench (LFM2-Audio 56.8) and cannot be
instructed about the meeting the way an LLM with a transcript can. A
cascade with streaming at every seam reaches the same ≈ 200–500 ms
region (Qwen3-Omni's own claim is 211 ms audio-only on a GPU server).

## 4. Concepts considered

Scored on "no way this runs locally in Go" (wow), feasibility in pure Go,
reuse of existing work, and demo robustness.

| Concept | Wow | Feasibility | Reuse | Notes |
| --- | --- | --- | --- | --- |
| A. Voice agent on the call, ≤ 500 ms, barge-in, 30 languages in, 10 out | Very high | High (all Qwen-core) | High | The core; every other concept adds to it |
| B. Live captions with partials, per speaker, native closed captions | High | High | High (Qwen3-ASR) | Cheap once streaming ASR exists; visible to everyone |
| C. Simultaneous interpreters that dub each speaker into other languages in their own voice | Very high | Medium (cloning path: tokenizer encoder + speaker encoder) | High | 1.5–2.5 s lag is inherent; one participant per language |
| D. Meeting memory: questions about the meeting, summary and action items to chat | High | High (long context; prefix KV already exists) | High | Free with the LLM; end-of-demo payoff |
| E. Screen-share copilot | High | Medium-low in pure Go (no VP8/H264 decoder; ViT port) | Medium | Excluded: audio only |
| F. Audio-only "spatial" tricks (per-listener language mixes) | Medium | Low (SFU has no per-subscriber mixing) | | Dropped |

**Flagship = A + B + D, then C.** One binary, staged so that each milestone
is a complete demo.

## 5. The experience: demo script (≈ 9 minutes)

Setup: MacBook Pro M4 Max on a table, Activity Monitor's GPU history and
`nettop` filtered to the process visible on a second display; two or three
people on Pronto from laptops/phones. Wi-Fi stays on (the SFU needs it);
the only remote hosts the process talks to are `*.stream-io-api.com` and
the SFU edge, which `nettop` shows.

- **0:00 Cold start.** `go run ./examples/gopher`. Three models open from
  the weight cache in under a second (measured today: 0.09 s + 0.085 s;
  TTS estimated similar). The bot appears in the participant list as
  "Gopher" and says "Hi, I'm Gopher. I'm running on this laptop." First
  wow: the voice is natural and it started in one second.
- **0:30 Latency.** Alice: "Gopher, what's the capital of Portugal?" The
  answer starts within half a second. The terminal prints the timeline of
  that turn (pause 200 ms · Smart Turn 4 ms · ASR final 38 ms · LLM first
  token 61 ms · TTS first frame 47 ms · Opus 1 ms · 0 allocations). Chat
  gets the same line. Engineers stare at the numbers.
- **1:00 Barge-in.** Bob: "Gopher, explain how WebRTC ICE works." It starts
  a long answer; Bob interrupts after two seconds: "shorter". It stops
  mid-word (queue cleared within one frame) and gives a two-sentence
  version. It also remembers what it was saying if asked "go on".
- **1:45 Languages.** Carla asks in Spanish; it answers in Spanish, same
  voice. Alice asks in German. Captions in the chat/closed captions show
  each speaker's words in the language spoken, appearing while they are
  still talking (partials), then finalized.
- **2:30 Interruption by a human conversation.** Alice and Bob talk to
  each other about the rollout plan for two minutes. Gopher stays silent
  (not addressed; text end-of-turn and address detection), captions keep
  flowing for both, per speaker.
- **4:30 Memory.** Bob: "Gopher, what did Alice say the blocker was?" It
  answers from the transcript in one sentence. "Who's on point for the
  rollback plan?" Answered.
- **5:15 Interpreter (M5).** Alice: "Gopher, bring in a German
  interpreter." A second participant "Gopher (Deutsch)" joins. When Bob
  speaks English, two seconds later Bob's own voice says the German
  translation. A German listener mutes the English participants and keeps
  the interpreter. Show that the interpreter is a separate participant
  everyone can mute.
- **7:00 Load.** Three people talk at once for 20 seconds (captions for
  all), while asking Gopher a question. The GPU graph pins at 100 %; the
  answer still comes within a second; the terminal shows the arbiter's
  queue depths.
- **8:00 Wrap.** "Gopher, summarize and post the action items." A summary
  with owners lands in the chat. Show the model sizes and the memory
  footprint (≈ 43 GB resident), then `Ctrl-C` and restart in one second to
  prove there is no warm-up cost.

Wow moments, in order of impact: the half-second reply; stopping mid-word;
answering in the language it was asked in; a participant's own voice
speaking German; the zero-allocation timeline; the one-second cold start.

## 6. Chosen models and budgets

| Role | Model, format | Weights resident | Per-call compute (estimate unless noted) | Fallback |
| --- | --- | --- | --- | --- |
| STT | Qwen3-ASR-1.7B, `gpu-q8` decoder + exact FP16 encoder | 2.3 GB | Partial every 500 ms: encode the open ≤ 8 s window ≈ 10 ms, ≈ 6 new audio tokens + 5-token fallback re-decode ≈ 60 ms; final on turn end ≈ 40 ms (measured: 19 ms encoder for 11 s, 5 ms/token) | Qwen3-ASR-0.6B (1.8 GB, ≈ 2× faster) |
| Turn | Smart Turn v3.2 FP32 | 0.03 GB | 3.6–40 ms per pause, CPU | TinyMelNet |
| LLM | Qwen3.6-35B-A3B, int8 (GPU, GPTQ-rounded) | ≈ 35 GB | ≈ 8–9 ms/token decode; prefill ≈ 600 tok/s; conversation prefix cached | Qwen3.5-9B (≈ 9 GB, ≈ 22 ms/token); Qwen3-8B (measured 15.2 ms) |
| TTS | Qwen3-TTS-12Hz-1.7B, `gpu-q8` talker, exact code predictor and codec | ≈ 2.4 GB | ≈ 10 ms per 80 ms frame (RTF ≈ 0.13), first frame ≈ 50 ms | 0.6B (≈ 1 GB) |
| Caches | LLM KV: 10 attention layers × 2 KV heads × 256 × K,V × 4 B = 41 KB/token → 8 K tokens = 0.33 GB; GDN state 30 × 32 × 128 × 128 × 4 B = 63 MB per session; ASR/TTS prefixes < 0.2 GB | ≈ 0.7 GB | | |
| Total | | **≈ 41 GB + process** | | Light preset ≈ 15 GB |

Metal's default working-set limit on a 64 GB machine is about 48 GB; the
demo host sets `sudo sysctl iogpu.wired_limit_mb=57344` (documented in the
example README) so the 41 GB of mapped weights plus scratch never page.

Quality gate for every quantized path: hidden-state cosine against
official BF16 at or above llama.cpp Q8_0 of the same checkpoint (the
existing bar: `gpu` GPTQ 0.99990 vs Q8_0 0.99933 on Qwen3-8B), plus
task-level checks (ASR transcripts identical to `f16`, TTS codec tokens
identical under greedy decoding, LLM answers identical on the probe set).

## 7. Latency budget, end to end

Timeline for "user stops talking → other participants hear the first
syllable", M3 target, Qwen3-8B numbers where measured and Qwen3.6
estimates in brackets:

| Stage | Where | Time | How it is hidden or shortened |
| --- | --- | --- | --- |
| Speaker's browser capture + Opus encode + uplink to SFU + SFU → us | network | 40–80 ms | Not ours; Stream edge selection |
| Opus decode (gopus, 16 kHz) + VAD per 20 ms frame | CPU | < 0.5 ms | Per frame, per participant |
| Silence needed to suspect an end of turn | policy | **200 ms** (dominant term) | Smart Turn asked at 200 ms of silence; text end-of-turn asked on every partial *before* the pause; a confident text decision at a 120 ms pause skips the wait |
| Smart Turn v3.2 | CPU | 4–40 ms | Runs on the 8 s tail, overlaps the pause |
| ASR final | GPU | ≈ 40 ms (encoder of the open window 10 ms + ≈ 6 tokens × 5 ms) | Partials every 500 ms mean only the last half second is new; committed tokens' KV are reused |
| LLM first token | GPU | 25 ms measured for a 12-token prompt on Qwen3-8B (≈ 30–60 ms for the tail on 3.6) | Speculative prefill: the user's partial transcript is pushed into the conversation KV as it arrives (extend/rollback with `PrefixKV.CommonPrefix`), so at turn end only the last words and the template suffix are prefilled |
| LLM first clause (≈ 8 tokens) | GPU | 8 × 15.2 = 122 ms measured (8 × 9 ≈ 70 ms est.) | TTS starts on the first clause boundary or after 6 tokens, whichever first |
| TTS first frame | GPU | ≈ 50 ms (text prefill + 1 talker step + 15 code-predictor steps + codec frame) | Frames are 80 ms; emit one frame per packet at the start, four later |
| Opus encode (24 kHz mono, 20 ms frames) | CPU | ≈ 1 ms | |
| Us → SFU → listeners' jitter buffer + decode + playout | network | 60–120 ms | Not ours; RED can be enabled for lossy links |
| **Total, end of speech → first audio at listeners** | | **≈ 480–620 ms** with Qwen3-8B, **≈ 420–560 ms** with Qwen3.6 (estimates); ≈ 300 ms of it is the pause and the network | Target p50 ≤ 500 ms measured at a listener's browser (`getStats` on the listener), p95 ≤ 800 ms |

For reference, production voice agents measured from real phone calls sit
at 600–1800 ms with a fleet median of 680 ms p50 / 1180 ms p95, and the
human turn gap is ≈ 200 ms
([Openbenchmarks](https://openbenchmarks.com/voice-agent-latency),
[DestiLabs](https://www.destilabs.com/blog/ai-voice-agent-benchmark-2026),
[Trillet](https://trillet.ai/blogs/voice-ai-latency-benchmarks)).

Overlaps that make the budget work:

1. **Streaming ASR partials** (every 500 ms; 250 ms for the dominant
   speaker when the GPU is idle) so the final decode is small.
2. **Speculative LLM prefill** of the partial transcript; on a revision
   (the 5-token fallback changed committed words) roll the prefix back to
   the common prefix, which the KV store already supports.
3. **Early end-of-turn** from text (the running partial + "is the speaker
   done?" on the LLM, 31 ms measured) so the 200 ms silence rule shrinks
   when the sentence is obviously complete, and grows when it is obviously
   not ("and then I", "so, umm").
4. **Streaming TTS** on the LLM's token stream with clause boundaries
   (punctuation) or a 6-token minimum; the talker's prompt is extended per
   clause (its own prefix KV), never rebuilt.
5. **Barge-in**: on speech from a human while we speak, `TrackWriter.Clear()`
   (drops queued frames within 20 ms), cancel the LLM and TTS contexts
   (both loops check `ctx` per token/frame), and hand the interrupted text
   to the next turn's context so "go on" works. Echo of our own voice is
   rejected before it can trigger this (8.3).
6. **Interpreter lag** is a policy, not compute: translate at clause
   boundaries of the partial transcript (≈ 1–2 s of speech), synthesize as
   the translation streams; total lag 1.5–2.5 s, matching human
   simultaneous interpretation.

## 8. gophonic work required

Everything below follows the existing rules: a leaf interface package,
one package per model family, shared numerics in `internal/`, lanes that
own scratch, outputs into caller buffers, `AllocsPerRun` tests on warm
paths, and validation against the official PyTorch reference of each
model.

### 8.1 Text generation: package `chat` (new leaf) + `qwen3` lanes

Interfaces (leaf package, no model imports, mirrors `speech`):

```go
package chat

type Role uint8 // System, User, Assistant, Tool
type Message struct{ Role Role; Content string }

// Options for one generation. Zero value: model defaults (Qwen3.6
// non-thinking: temperature 0.7, top-p 0.8, top-k 20, presence 1.5).
type Options struct {
	Temperature, TopP     float32
	TopK                  int
	PresencePenalty       float32
	MaxTokens             int
	Think                 bool     // thinking mode; off for speech
	Stop                  []string
}

// Sink receives the generation as it happens. Piece is valid only during
// the call; implementations copy nothing on the caller's behalf.
type Sink interface {
	Token(id int, piece []byte) error // returning an error stops generation
}

// Session is one conversation whose prefix keys/values stay on the
// model. A session belongs to one goroutine at a time.
type Session interface {
	// Append adds messages to the conversation and evaluates them
	// (prefix extension); it is what speculative prefill calls with
	// partial user text, then again with the final text (rollback to the
	// common prefix is automatic).
	Append(ctx context.Context, msgs ...Message) error
	// Generate produces the assistant's reply for the current
	// conversation, streaming into sink, and appends it. ctx cancellation
	// is barge-in: the partial reply is kept as an interrupted turn.
	Generate(ctx context.Context, opts Options, sink Sink) error
	Tokens() int         // context length used
	Truncate(keepFirst, keepLast int) error // drop middle turns, keep the system prompt
	Close() error
}

type Generator interface { // the lane type: Lane[chat.Generator](model)
	NewSession(system string, maxTokens int) (Session, error)
	Close() error
}
```

Implementation in `qwen3` (and later the hybrid core): chat template
builders for Qwen3 and Qwen3.5/3.6 (`<|im_start|>` format, `<think>\n\n</think>`
for non-thinking on 3.x, tool-call syntax later), tokenization into a lane
buffer, `HiddenLastExtendInto` + `LogitsInto` per token, a sampler
(temperature, top-k, top-p, presence penalty over a fixed-size seen-token
bitmap, all on caller-owned scratch; top-k on 248 K logits is a partial
selection on the GPU for the hybrid core), incremental detokenization with
the existing allocation-free decoder, stop-token handling. Zero-alloc:
messages are tokenized into a lane-owned ring; `Sink.Token` receives a
slice into it. Later: speculative decoding with Qwen3.5-0.8B as the draft
(optional; decode is already 8–9 ms).

Validation: token-for-token equality with `transformers` greedy generation
on 20 prompts per checkpoint (fixture generated by a `tools/reference_generate.py`
in the style of the existing reference scripts), sampler unit tests
against a NumPy implementation with fixed RNG, template tests against
`apply_chat_template`, `AllocsPerRun(Generate) == 0` after warm-up.

Effort: 1.5 weeks (M1).

### 8.2 Streaming ASR: `speech.StreamTranscriber` on Qwen3-ASR

```go
package speech

// StreamTranscriber transcribes audio as it arrives. Feed appends PCM;
// Partial writes the current best transcript (stable prefix plus a
// revisable tail) into dst; Finish decodes the remainder and resets for
// the next utterance. All results index a lane-owned text buffer that is
// valid until the next call.
type StreamTranscriber interface {
	Feed(pcm []float32) error
	Partial(ctx context.Context, dst *Transcript) error // Transcript.Stable marks the committed byte length
	Finish(ctx context.Context, dst *Transcript) error
	Reset()
	Close() error
}
```

Qwen3-ASR implementation, following the published recipe
([report](https://arxiv.org/html/2601.21337v1) §streaming; independently
reproduced by qwen3-asr.cpp): the encoder's attention is block-diagonal in
104-token (8 s) windows, so a completed window's output never changes and
is encoded once (cache per lane, ring of windows); only the open tail
window is re-encoded per partial (≈ 10 ms on the GPU). The decoder keeps
the prompt-prefix KV as today plus the committed text tokens; on each
partial it splices the new audio embeddings (the prompt's `<|audio_pad|>`
count grows: the prompt layout must be arranged so audio tokens are
appended, not inserted; verify against the processor's prompt), drops the
last 5 generated tokens (fallback), and greedily decodes until end or a
per-partial token cap. Chunks older than the last four are frozen
(`Transcript.Stable`). Cadence is the caller's (500 ms default).

Mel frontend: `mel.Spectrogram` gains an incremental mode (append frames;
the 400-point FFT and 160 hop are unchanged; centered reflection padding
means the last two frames are provisional).

Validation: streaming output on the test clips equals the offline
transcript (the C++ port reports a one-punctuation difference on a 105 s
file; assert equality on the JFK and Chinese fixtures and ≤ 0.5 CER on a
longer clip); streaming WER on LibriSpeech test-clean (subset) within the
paper's 1.95 vs 1.63 offline; latency benchmark: partial ≤ 80 ms p95 with
one speaker, ≤ 150 ms with four; `AllocsPerRun(Feed/Partial/Finish) == 0`.

Effort: 2 weeks (M3), including the incremental mel and the window cache.

### 8.3 Turn taking and barge-in policy (example code, small library helper)

- `speech.TurnDetector` (Smart Turn) as today, on 200 ms pauses, then every
  200 ms up to 2 s.
- Text end-of-turn on the partial transcript via `chat`/`qwen3.Question.NewStream`
  ("Has the speaker finished their sentence?"), ≈ 31 ms measured, updated
  per partial; combined rule: speak when Smart Turn > 0.5 **or** (text
  complete > 0.9 and pause ≥ 120 ms); wait when text says incomplete even
  if Smart Turn fires (up to 1 s).
- Address detection: is Gopher addressed? A zero-shot question on the
  final transcript plus the conversation ("Is this directed at the
  assistant?"), 45 ms measured on Qwen3-8B; a wake word is not required.
- Barge-in: human speech (VAD ≥ 96/255 and energy above −45 dBFS for
  ≥ 120 ms) while the writer has audio buffered → `Clear()`, cancel
  contexts. Self-echo guard: while we speak, any incoming speech whose
  partial transcript is a fuzzy match (normalized edit distance < 0.3) of
  the last 3 s of our own spoken text is ignored, and the SFU's
  `AudioLevelChanged` for our own track vs the speaker's helps break ties.

Effort: inside M1/M3.

### 8.4 Text to speech: package `qwen3tts`, lane `speech.Synthesizer`

```go
package speech

// Synthesizer turns text into speech as the text arrives.
type Synthesizer interface {
	// Speak synthesizes text appended so far; call it per clause. PCM is
	// delivered to out in frames of 80 ms at 24 kHz (SampleRateOut) as they
	// are decoded; out must not retain the slice. Returns when the clause's
	// audio has been delivered or ctx is cancelled (barge-in).
	Speak(ctx context.Context, text []byte, out func(pcm []float32) error) error
	// End flushes the utterance (eos) and resets the utterance state.
	End(ctx context.Context, out func(pcm []float32) error) error
	// SetVoice selects a preset (CustomVoice) or a cloned voice.
	SetVoice(v Voice) error
	Close() error
}

type Voice struct {
	Preset    string     // "Ryan", "Aiden", ... (CustomVoice)
	Reference []float32  // 3–10 s at 16 kHz for cloning (Base model); nil for presets
	Language  string     // ISO 639-1; "" = auto from text
	Instruct  string     // optional style instruction
}
```

Components (`qwen3tts/`):

1. **Talker** = `internal/qwen3lm` with the 1.7B geometry, loaded from the
   `talker.` tensors with the 3072-row codec head. New core feature: input
   rows that are the **sum** of a projected text embedding and a codec
   embedding ("dual track"): generalize `Embeds` to `Rows` input (the
   evaluator accepts precomputed hidden-wide rows for a range of
   positions, which also removes the placeholder-count bookkeeping). The
   text projection (ResizeMLP) and the codec embedding table (3072 × 2048)
   run as small kernels. Prefix KV per utterance: prefix = language/speaker
   tokens (+ speaker embedding row for cloning) + text so far; each new
   frame appends one row. Sampling: temperature 0.9, top-k 50, top-p 1.0,
   repetition penalty 1.05, suppression of the last 1024 vocab rows except
   eos (from the reference code), on the GPU to avoid a 3072-float readback.
2. **Code predictor**: a 5-layer Qwen3-style transformer (hidden 1024,
   16 Q / 8 KV heads, MLP 3072) with 15 embedding tables and 15 heads
   (2048 each), run 15 steps per frame on the talker's hidden state through
   `small_to_mtp_projection`. Same core, second geometry (the kernels are
   specialized per geometry at load, already supported). One command
   buffer per frame with sampling on the GPU; CPU-SME alternative if
   dispatch latency dominates (≈ 50 MB of weights per step is 0.5 ms on the
   CPU).
3. **Codec decoder** (`Qwen3-TTS-Tokenizer-12Hz` decoder half): 16 codebook
   embeddings (× 512) summed → 1024 latent → 8-layer 512-wide causal
   transformer → conv upsampling stack (decoder_dim 1536, rates 8·5·4·3
   then 2·2: 1920 samples per frame at 24 kHz). Implemented on the CPU with
   the existing `whispergemm`/`q8gemm` tiles (im2col for the convolutions;
   per frame the work is a few tens of MFLOP) or as Metal kernels; exact
   FP16 weights. The encoder half (32 quantizers, causal) is needed only
   for cloning (M5).
4. **Speaker encoder** for cloning (Base model): extracts the speaker
   embedding from reference audio; port in M5 with the tokenizer encoder.
5. `speech.Synthesizer` lane and a `gophonic` format for the Qwen3-TTS
   snapshot directory (`model_type` "qwen3_tts"); weight-cache entries for
   the talker like Qwen3.

Validation: `tools/reference_tts.py` runs the official `qwen-tts` package
in FP32 with greedy decoding (temperature 0) on fixed texts and records
the talker prompt ids, the first-step logits, the codec token matrix
(16 × T), the decoder's latent, and the waveform. Tests require prompt ids
exact, talker tokens exact under greedy in the exact CPU format and the
same first frame in `gpu-q8`, codec tokens exact, waveform max abs error
≤ 1e-3 in FP32, and loopback WER through Qwen3-ASR on 50 sentences within
0.5 points of the paper's 1.24; `AllocsPerRun(Speak) == 0` warm; a
benchmark reporting first-frame latency and RTF against the paper's 101 ms
/ 0.31 (their server) and the M4 community figure (RTF 0.253).

Effort: 3 weeks (M2): 1 week talker + rows API + sampling, 1 week code
predictor + codec decoder, 1 week streaming, validation, benchmarks.

### 8.5 The Qwen3.5/3.6 hybrid core (`internal/qwen3lm`, formats and kernels)

Per the fetched configs and the reference implementation
([Qwen3-Next modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py),
[Gated DeltaNet paper](https://arxiv.org/abs/2412.06464),
[vLLM's kernel breakdown](https://vllm.ai/blog/2025-09-11-qwen3-next)):

- **Layer types**: 3 × (GDN → MoE) then 1 × (gated attention → MoE), 40
  layers (35B-A3B) or dense MLP instead of MoE (0.8B–27B).
- **Gated DeltaNet layer**: `in_proj_qkvz` (2048 → q 2048, k 2048, v 4096,
  z 4096), `in_proj_ba` (2048 → 64); depthwise causal conv1d (kernel 4) over
  [q,k,v] with SiLU; L2-normalize q and k per head, q scaled by
  128^-0.5; `g = -exp(A_log) · softplus(a + dt_bias)`, `β = sigmoid(b)`;
  per value head (32, keys repeated 2×) with state S ∈ R^{128×128}:
  `S ← S·exp(g); kv = Sᵀk; Δ = (v − kv)·β; S ← S + k Δᵀ; o = Sᵀq`; then
  gated RMSNorm `norm(o) · silu(z)` and `out_proj` (4096 → 2048).
  Decode = a 128×128 rank-1 update per head per layer: 32 × 16 K MACs, trivial;
  the conv keeps a 3-sample ring per channel. Prefill = the chunked
  algorithm (chunk 64: gate cumsum, KKᵀ with decay, `(I + A)^-1` by
  triangular solve, WY representation, inter-chunk state recurrence,
  chunk output), six Metal kernels; the CPU path uses SME for the chunk
  matrices. State lives in the `PrefixKV` alongside K/V of the attention
  layers, with `CopyPrefix` snapshotting states at chunk boundaries so
  rollback (speculative prefill) works: keep a snapshot every 64 tokens
  and recompute forward from the nearest one.
- **Gated attention**: q_proj outputs 2 × 16 × 256 (query and a sigmoid
  gate per head), RMSNorm on q and k per head, partial RoPE on the first
  64 of 256 dims, GQA with 2 KV heads, output × sigmoid(gate), o_proj.
  Existing attention kernels gain the gate and partial-RoPE parameters.
- **MoE**: router (2048 → 256), softmax, top-8, renormalized; experts
  512-wide SwiGLU; shared expert 512-wide with a sigmoid gate (2048 → 1).
  Decode: a **grouped GEMV** kernel taking 8 expert indices and weights
  per token; expert weights stored int8 (blocks of 32) contiguously per
  expert in the weight cache so each token touches 9 × 3 × 1 MB. Prefill:
  sort tokens by expert, batched simdgroup GEMMs per expert. RMSNorm
  weights fold into every expert's gate/up rows as they do for dense
  projections; the Hadamard rotation of the residual applies unchanged.
- **248 320-row head**: 0.5 GB int8; logits into a GPU buffer with a
  top-k partial selection kernel so sampling never reads 1 MB back.
- **Loader/cache**: new layout version, tensor names for `qwen3_5` /
  `qwen3_5_moe`; `config.json` `model_type` detection in `qwen3.IsModelDir`.
- **GPTQ** for the GPU formats extends to experts (calibration routes
  tokens; experts that see few tokens fall back to round-to-nearest).

Validation, in this order: (1) `lmtest` random hybrid checkpoints (with
and without MoE) against a float64 reference of the same equations, every
format and worker count, prefix extension and rollback; (2) official
Qwen3.5-0.8B and 9B BF16 hidden states for fixed texts
(`tools/reference_hidden.py` extended), cosine ≥ llama.cpp Q8_0 on the
same GGUF (llama.cpp supports the family; GGUFs exist); (3) official
Qwen3.6-35B-A3B; greedy generation equality on 20 prompts; (4)
benchmarks: decode ms/token and prefill tok/s vs llama.cpp Metal Q8_0 and
MLX 8-bit on the same machine, alternating engines from a cool state as
`docs/clm-performance.md` prescribes.

Effort: 4–5 weeks (M4): 1.5 wk GDN (decode + chunked prefill, CPU and
GPU), 1 wk gated attention + loader + 0.8B/9B validation, 1.5 wk MoE +
35B-A3B, 0.5–1 wk GPTQ, benchmarks, docs. The optional Qwen3-30B-A3B-2507
step (classic attention + MoE) can front-load the MoE kernels by a week if
two people work in parallel.

### 8.6 Interpreter path (M5)

- Tokenizer **encoder** (causal conv + 8-layer transformer, 32 quantizers)
  and the speaker encoder of the Base model, to turn 3–10 s of a
  participant into the prompt rows the talker needs; per participant,
  computed once from their first clean turn (cached by user ID).
- `chat.Session` in "translator" mode: system prompt fixes the target
  language; each committed clause of the source partial is appended and
  translated; output streams to that participant's `Synthesizer` with
  their cloned voice; one interpreter participant per language, each with
  its own `TrackWriter`.
- Ordering: one synthesizer per interpreter, a FIFO of (speaker, clause);
  overlapping speakers are serialized, and the queue is trimmed if lag
  exceeds 4 s (skip to the newest clause of the newest speaker, announce
  nothing).

Effort: 3 weeks including summary/action items and the benchmark report.

### 8.7 Small shared items

- `speech.Transcript.Stable` (committed byte length) and a `Partial` flag.
- `gophonic.Pool` gains a per-model lane cap and a GPU arbiter hook
  (8.8 in the app; the library only exposes bounded work units: a token, a
  frame, one partial).
- `cmd/gophonic-server`: `/v1/chat/completions` (streaming SSE) and
  `/v1/audio/speech` land for free on the new lanes; not required for the
  demo but they make the showcase reusable.

## 9. Example app architecture (`examples/gopher`)

One process, one `rtc.Client` per participant identity (Gopher, plus one
per interpreter language), all sharing the models through `gophonic.Pool`.

```
SFU ──► per-speaker pipeline (goroutine per remote audio track)
        TrackReader(16 kHz) → 20 ms frames → VAD/energy → ring buffer (8 s)
          ├─► StreamTranscriber.Feed  (every frame)
          ├─► every 500 ms: Partial → captions (chat / closed captions), text end-of-turn, speculative Append
          ├─► on pause ≥ 200 ms: Smart Turn (CPU lane) ─┐
          └─► turn end ────────────────────────────────┴─► Finish → final caption → dialogue manager

dialogue manager (one goroutine, owns the chat.Session)
        addressed? → Generate(stream) → clause splitter → Synthesizer.Speak → TrackWriter.Write (24 kHz)
        barge-in: cancel ctx, TrackWriter.Clear(), keep interrupted text
        memory: Session holds the full transcript ("[Alice] ...") as user turns; Truncate keeps the system prompt + last N tokens

interpreters (one goroutine per language)
        clause queue ← committed clauses of every speaker → translate Session → Speak(voice=speaker) → its own TrackWriter

GPU arbiter (one goroutine): priority queue of work units
        1. TTS frame (≈ 10 ms)   2. LLM token (≈ 9–15 ms)   3. ASR partial (≈ 60 ms)   4. LLM prefill chunk (≤ 64 tokens, ≈ 100 ms)
        rule: a unit waits at most one unit of a higher class; prefill is chunked so nothing waits > 100 ms
```

- **Backpressure**: `TrackWriter.Buffered()` keeps at most 600 ms queued
  (barge-in responsiveness), the synthesizer pauses on the arbiter;
  partial cadence degrades from 250 → 500 → 1000 ms as arbiter queue depth
  grows; captions batch per speaker.
- **Concurrency on one GPU**: separate `metal` command buffers per lane
  already exist; the arbiter serializes admission, not execution, so short
  units interleave. Measure whether concurrent command buffers improve
  utilization; if the GPU serializes them anyway, the arbiter's ordering is
  what bounds latency.
- **CPU budget**: 12 P-cores: Opus decode/encode and VAD (< 5 %), Smart
  Turn lanes (GOMAXPROCS-1 helpers each, but one prediction at a time per
  speaker), codec decoder if on CPU, SME ASR encoder fallback. Set
  `Threads` per lane so the sum stays under the core count.
- **Failure modes and handling**: SFU reconnect/migration (SDK restores
  tracks; pipelines survive because they own their readers by track ID);
  a model error in TTS falls back to posting text; ASR hallucination on
  silence/noise (energy gate, min 400 ms speech, drop transcripts that are
  only filler); overlapping speakers (independent pipelines; the dialogue
  manager answers the addressed one); thermal throttling (the arbiter
  reports unit times; when TTS RTF > 0.5 switch to the 0.6B TTS and stop
  partials for non-dominant speakers); memory pressure (start with the
  light preset if `hw.memsize` < 48 GB or the wired limit is low); Chat
  REST failures (log, keep talking).
- **Observability**: a per-turn timeline struct (all stages, ms, allocs
  via `runtime.MemStats` deltas) printed and posted; Prometheus-free.

## 10. Milestones

| Milestone | Deliverable (shippable demo) | Library work | Effort | Exit criteria |
| --- | --- | --- | --- | --- |
| **M1 Text agent on the call** | Gopher joins, listens (as today), answers addressed questions in the chat, meeting memory | `chat` package + Qwen3 generation/sampling/template (8.1); address detection; timeline | 1.5 wk | Greedy equality with transformers on 20 prompts; 0 allocs per token; answer in chat < 1 s after turn end |
| **M2 Voice agent** | Gopher speaks (Qwen3-8B + Qwen3-TTS-1.7B), barge-in | `qwen3tts` (8.4), `speech.Synthesizer`, rows input in the core, Opus publish, barge-in | 3 wk | TTS parity tests; first frame ≤ 80 ms after text; voice-to-voice ≤ 900 ms p50 measured at a listener |
| **M3 Streaming and captions** | Partials, closed captions, ≤ 500 ms voice-to-voice, four-speaker load | `speech.StreamTranscriber` (8.2), incremental mel, speculative prefill, text end-of-turn, GPU arbiter | 2 wk | Streaming = offline on fixtures; partial ≤ 80 ms p95; **≤ 500 ms p50 / ≤ 800 ms p95** with Qwen3-8B |
| **M4 Frontier LLM** | Same demo on Qwen3.6-35B-A3B (Qwen3.5-9B light preset) | Hybrid core (8.5), GPTQ for experts, loader, benchmarks | 4–5 wk | Cosine ≥ llama.cpp Q8_0 on 0.8B/9B/35B-A3B; ≥ 100 tok/s decode; total resident ≤ 45 GB; voice-to-voice unchanged or better |
| **M5 Interpreters and wrap-up** | "Gopher (Deutsch)" dubbing in each speaker's voice; summary/action items; benchmark report; README + docs | Tokenizer encoder + speaker encoder (8.6), translator sessions, report | 3 wk | Cloning similarity within the paper's range on a held-out set (via a speaker-verification model or the paper's protocol); lag ≤ 2.5 s p50; report published |

Total ≈ 14–15 weeks for one engineer; M2 (TTS) and M4 (core) are
independent, so two engineers finish in ≈ 10 weeks. Each milestone is
demoable on its own and lands on `main` behind the example's flags.

### Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Hybrid core correctness (GDN chunked prefill, MoE routing, gated attention) | M4 slips or ships below the fidelity bar | Validate blocks in isolation (0.8B dense first; optional classic-attention MoE on Qwen3-30B-A3B), float64 `lmtest` oracles before any official weights, GGUF Q8_0 as the fidelity yardstick |
| Qwen3-TTS naturalness judged weaker than hosted voices | "Wow" reduced | Pick the best preset by ear (Ryan/Aiden), use the 1.7B, tune sampling; state the objective wins; keep Breeze TTS 2 as a documented non-commercial comparison only |
| GPU contention (three models, four speakers) | Latency spikes | Arbiter with bounded units, partial cadence degradation, 0.6B ASR for non-dominant speakers, measure concurrent command buffers early (M3) |
| Thermal throttling (≈ 35 %) | Demo drifts slower over 10 minutes | Cool machine before the demo; adaptive presets; keep the demo ≤ 10 minutes; report both cool and sustained numbers |
| Metal working-set limit / 41 GB resident | Paging or allocation failure | `iogpu.wired_limit_mb`; light preset auto-selected; measure with `vm_stat` in CI-like runs |
| Closed captions need an API secret | Captions only in chat on Pronto staging | Chat fallback is already working; ask for the pronto-staging secret; custom events as a middle ground |
| Streaming ASR prompt layout (audio placeholders must be appendable) | Partial path recomputes the whole prompt | Verify against the processor's prompt ids in week 1 of M3; worst case re-prefill ≈ 50 ms per partial, still within budget |
| Self-echo from a participant without AEC | False barge-in | Fuzzy match of incoming partials against our spoken text; SFU audio levels; a "push-to-interrupt" flag for the demo |
| Interpreter voice cloning quality on noisy 3 s references | Odd voices | Choose the reference from the speaker's cleanest, longest turn (energy, Smart Turn confidence); fall back to a preset voice per speaker |

### Benchmarks that prove "state of the art"

All on this M4 Max, cool start, alternating engines, published in
`docs/sfu-showcase-results.md` with the raw runs:

1. **LLM**: Qwen3.6-35B-A3B decode ms/token and prefill tok/s at 128/512/2048
   tokens: gophonic int8 vs llama.cpp Metal Q8_0 and Q4_K_M vs MLX 8-bit
   and 4-bit, with the hidden-state cosine of each against BF16 on the same
   texts (speed *and* fidelity in one table, as `docs/clm-performance.md`
   does).
2. **ASR**: streaming partial latency (p50/p95) and WER on LibriSpeech
   test-clean and Fleurs (en, de, es, zh) subsets vs the paper's streaming
   numbers and vs whisper.cpp large-v3-turbo offline; offline 11 s clip
   time vs qwen3-asr.cpp/MLX ports where available on Apple silicon.
3. **TTS**: first-frame latency and RTF vs the paper (101 ms / 0.31) and the
   community M4 figure (RTF 0.253); loopback WER through Qwen3-ASR; codec
   waveform parity.
4. **Voice-to-voice**: end of speech → first audio at a listener's browser
   (WebRTC `getStats` timestamps plus a loopback tone method), p50/p95 over
   50 turns, single speaker and four speakers, vs the published
   commercial-platform range (600–1800 ms) and fleet medians (680/1180 ms).
5. **Allocations**: `AllocsPerRun == 0` for token, frame, partial, and the
   per-frame audio path; a 10-minute soak with heap growth = 0.
6. **Memory and load time**: resident bytes per model, cold open times
   from the weight cache.

## 11. Spikes to run first (each ≤ 1 day)

1. Streaming ASR prompt layout: confirm the audio placeholders sit at the
   end of the prompt so partials extend the prefix (else measure the
   re-prefill cost).
2. Qwen3-TTS dual-track input and the code predictor's exact inputs from
   the reference code; dump a greedy fixture with the official package.
3. Metal concurrency: two lanes issuing command buffers at once, does
   wall time overlap? Decides the arbiter's granularity.
4. Closed captions: does Pronto render `call.closed_caption` events sent
   with an API secret? Else chat.
5. Gated DeltaNet decode on Qwen3.5-0.8B on the CPU in float32 against a
   PyTorch dump: the smallest end-to-end proof that the hybrid math is
   understood before any kernel work.

## Appendix: sources

- Qwen3-ASR: [technical report](https://arxiv.org/html/2601.21337v1),
  [GitHub](https://github.com/QwenLM/Qwen3-ASR),
  [qwen3-asr.cpp streaming port](https://github.com/JohnsonChang123/qwen3-asr.cpp),
  [Baseten streaming notes](https://www.baseten.co/library/qwen3-asr-1-7b-streaming/).
- Qwen3-TTS: [technical report](https://arxiv.org/html/2601.15621v1),
  [GitHub](https://github.com/QwenLM/Qwen3-TTS),
  [1.7B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice),
  [tokenizer](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz),
  [M4 tuning post](https://www.drmhse.com/posts/tuning-qwen3-tts-apple-silicon-m4/).
- Qwen3.5/3.6/3.8: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
  [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
  [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5),
  [lineup guide](https://codersera.com/blog/qwen-3-5-complete-guide-2026/),
  [Qwen3.8 release note](https://www.latent.space/p/ainews-qwen-38-max24t-and-27b-new),
  [Gated DeltaNet](https://arxiv.org/abs/2412.06464),
  [vLLM Qwen3-Next post](https://vllm.ai/blog/2025-09-11-qwen3-next),
  [transformers modeling_qwen3_next](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py).
- Rankings: [Artificial Analysis sub-32B](https://artificialanalysis.ai/articles/sub-32b-open-weights),
  [Qwen3.6 35B A3B page](https://artificialanalysis.ai/models/qwen3-6-35b-a3b),
  [TTS arena](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice),
  [arena summary](https://offlinetts.com/blog/tts-arena-leaderboard-2026/),
  [Open ASR leaderboard](https://huggingface.co/blog/open-asr-leaderboard),
  [2026 ASR survey](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks).
- Alternatives: [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2),
  [Breeze license](https://wavespeed.ai/blog/commercial-compliance/breeze-tts-2-review/),
  [Voxtral TTS](https://www.marktechpost.com/2026/03/28/mistral-ai-releases-voxtral-tts-a-4b-open-weight-streaming-speech-model-for-low-latency-multilingual-voice-generation/),
  [VibeVoice-ASR-Streaming](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B),
  [Qwen-Audio-3.0-ASR report](https://arxiv.org/abs/2609.07549),
  [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/),
  [LFM2-Audio](https://www.liquid.ai/blog/lfm2-audio-an-end-to-end-audio-foundation-model),
  [Qwen3-Omni](https://qwen.ai/blog?from=research.latest-advancements-list&id=65f766fc2dcba7905c1cb69cc4cab90e94126bf4),
  [Smart Turn](https://github.com/pipecat-ai/smart-turn).
- Apple silicon speed references: [MLX vs llama.cpp, Qwen3.5-35B-A3B on M4 Max](https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/),
  [MLX vs llama.cpp overview](https://yage.ai/share/mlx-apple-silicon-en-20260331.html).
- Voice-agent latency: [Openbenchmarks](https://openbenchmarks.com/voice-agent-latency),
  [DestiLabs](https://www.destilabs.com/blog/ai-voice-agent-benchmark-2026),
  [Trillet](https://trillet.ai/blogs/voice-ai-latency-benchmarks).
- Stream: `getstream-go-webrtc` (module cache, `audio/rtc/{reader,writer}.go`,
  `track/local.go`, `call.go`), `getstream-go/v5` (`Call.SendClosedCaption`,
  `Call.SendCallEvent`), `examples/streamcall/main.go`.
