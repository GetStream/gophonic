# gophonic

**Local speech and language models in pure Go.** Hear 30 languages with
Qwen3-ASR, think with Qwen3, speak with Qwen3-TTS, and put them together in
a voice agent that talks without turns, all inside your Go process: no cgo,
no Python, no ONNX Runtime, no network.

```sh
tools/fetch-models.sh                                   # Qwen3-ASR + turn detectors into models/
go install ./cmd/gophonic ./cmd/gophonic-server

gophonic -model models/Qwen3-ASR-1.7B meeting.wav       # {"text":"..."}
gophonic-server models/                                 # OpenAI-compatible API on :8080
```

| Task | Model | Speed on an M4 Max |
| --- | --- | --- |
| **Speech to text** | **Qwen3-ASR** 1.7B / 0.6B: 30 languages, 22 Chinese dialects | 11 s of audio in **226 ms** |
| Speech to text, English | Whisper `tiny.en` / `base.en` / `small.en` | ~3× whisper.cpp on one core |
| End of turn | Qwen3-ASR-1.7B as it transcribes; Smart Turn v3.2, TinyMelNet | free with the transcript; **3.6 ms** per prediction |
| **Text to speech** | **Qwen3-TTS-12Hz-1.7B**, 10 languages, 9 voices, streaming | first audio **18 ms** after the text; 5× real time |
| Generate text | Qwen3, any size: chat with streaming replies | Qwen3-8B: 19 ms per token, reply starts 52 ms after a message |
| Classify text | Qwen3, any size: moderation, routing, intent, sentiment | one Qwen3-8B token in **15 ms** (llama.cpp: 19 ms) |
| **Voice agent** | any of the above as a `speech.Duplex`: listens and speaks at once | stops within 300 ms when talked over |

Qwen3-ASR is the state-of-the-art open speech recognizer, and the one to
use unless you need Whisper's word timestamps or an English-only CPU model.

## Why it is fast

- **Loads in milliseconds.** On Apple silicon, the first load converts a
  checkpoint once and caches the result. Every later load maps that file and runs from it in
  place, with no reading, no conversion, and no copy onto the GPU:
  Qwen3-8B opens in 0.09 s instead of 4.5 s, and Qwen3-ASR in 0.085 s.
- **Uses the whole chip.** The Apple GPU runs through a pure-Go Metal
  binding, Apple's SME matrix unit runs Go assembly, and portable kernels
  cover everything else.
- **Allocates nothing.** Warm calls, including a full HTTP transcription
  request, allocate nothing, so latency stays flat under load.
- **Exact where it counts.** Every output is checked against the official
  PyTorch or ONNX reference. Where weights are quantized, fidelity stays at
  or above llama.cpp's Q8_0 ([validation](docs/validation.md)).

## Get the models

```sh
tools/fetch-models.sh                  # Qwen3-ASR-1.7B, Smart Turn, TinyMelNet
tools/fetch-models.sh whisper qwen3    # also Whisper (English) and Qwen3-8B
```

The script downloads official checkpoints at pinned revisions into
[`models/`](models) and converts the ones that need it. Hugging Face
snapshot directories, such as `Qwen/Qwen3-ASR-1.7B` or any `Qwen/Qwen3-*`,
load as they are, so you can point gophonic at a snapshot you already have.

## Command line

```sh
gophonic -model models/Qwen3-ASR-1.7B a.wav b.ogg c.opus       # one JSON line each
gophonic -model models/Qwen3-ASR-1.7B -language de -response-format srt talk.wav
gophonic -model models/smart-turn-v3.2.gophonic turn.wav       # {"probability":0.95,"complete":true}
gophonic -model models/Qwen3-8B -question "Is this acceptable at work?" \
  -labels acceptable,rude,spam "you absolute clown"            # [...,{"label":"rude","probability":1},...]
```

The model is loaded once per run, whatever the number of inputs. The CLI
reads WAV and Ogg Opus. See [CLI](docs/api.md#cli) for every flag.

## Server

```sh
gophonic-server models/
```

The server serves every model in `models/`. It loads a model on the first
request that needs it and closes it after 15 minutes without requests.
Thanks to the cache, reopening it takes a fraction of a second.
Transcription and `/v1/models` follow the OpenAI API, so existing clients
only need a new base URL:

```python
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
client.audio.transcriptions.create(model="Qwen3-ASR-1.7B", file=open("talk.wav", "rb"))
```

```sh
curl localhost:8080/v1/audio/transcriptions -F model=whisper-1 -F file=@talk.wav   # default transcriber
curl localhost:8080/v1/audio/classifications -F file=@turn.wav                     # turn detection
curl localhost:8080/v1/classifications -d '{"input":"WIN A FREE CRUISE",
  "question":"Is this message spam?","labels":["spam","not spam"]}'                 # zero-shot text
```

See [Server](docs/server.md) for every endpoint and flag.

## Go

```go
model, err := gophonic.Open("models/Qwen3-ASR-1.7B", gophonic.Options{})
if err != nil {
	return err
}
defer model.Close()

lane, err := gophonic.Lane[speech.Transcriber](model) // one lane per concurrent caller
if err != nil {
	return err
}
defer lane.Close()

var t speech.Transcript // reuse across calls: warm calls allocate nothing
err = lane.Transcribe(ctx, pcm16k, speech.Options{}, &t)
fmt.Println(t.Text, t.Language)
```

A model provides lanes of the interfaces it supports:

| Interface | Provided by |
| --- | --- |
| `speech.Transcriber` | Qwen3-ASR, Whisper |
| `speech.TurnDetector`, `speech.AudioClassifier` | Smart Turn, TinyMelNet |
| `speech.Synthesizer` | Qwen3-TTS |
| `chat.Generator`, `speech.ZeroShot` (text classifiers from a question and labels) | Qwen3 |

`gophonic.Lane[T](model)` opens any of them, including interfaces your own
package defines. `gophonic.Register` adds model formats. `gophonic.Pool`
loads models on demand and closes idle ones, the way the server does. The
[Go API](docs/api.md) covers all of it, and each model package (`qwen3asr`,
`whisper`, `smartturn`, `tinymel`, `qwen3`) exposes lower-level entry points.

## Performance

Warm latency on an Apple M4 Max. Every run is checked against the reference
output.

| Workload | gophonic | Reference engine |
| --- | ---: | ---: |
| Qwen3-ASR-1.7B, 11 s English, PCM to text | **226 ms** (GPU) · 751 ms (CPU, exact weights) | n/a |
| Qwen3-1.7B decoder, one token | **4.67 ms** | llama.cpp Metal Q8_0: 5.54 ms |
| Qwen3-8B, one token | **15.2 ms** (GPU) · 27.5 ms (CPU) | llama.cpp Q8_0: 18.6 ms (Metal) · 30.3 ms (CPU) |
| Qwen3-8B, 12-token prompt | **25 ms** | llama.cpp Metal Q8_0: 58 ms |
| Whisper base.en, 30 s, one thread | **198 ms** | whisper.cpp: 565 ms |
| TinyMelNet, PCM to prediction | **3.6 ms** | n/a |

Long prompts are the exception: on a 2048-token Qwen3-8B prompt, llama.cpp's
Metal backend is faster (3.1 s against our 3.5 s). The details:
[Qwen3-ASR](docs/qwen3asr.md), [Qwen3](docs/clm-performance.md),
[Whisper](docs/whisper-performance.md), [turn detection](docs/benchmarks.md).

## A voice agent without turns

`speech.Duplex` is an agent that listens and speaks at the same time: audio
goes in and comes out 20 ms at a time, and the agent decides when to talk.
`duplex.New` builds one from whatever models you give it:

```go
agent, err := duplex.New(duplex.Config{
	Prompt: "You are Gopher.",
	Listen: speech.Options{Languages: speech.Languages(speech.English, speech.Portuguese)},
	Tools:  []chat.Tool{chat.Func("now", "The current time.", now)},
	Idle:   45 * time.Second,
}, asr, llm, tts)
for { // every 20 ms
	state, err := agent.Step(micFrame, speakerFrame)
}
agent.Note("Ana joined the call.") // anything that happens, in words
```

It works while you talk, so that almost nothing is left when you stop:

- **It transcribes as you speak.** Each pass checks the last transcript
  against the audio in one step and decodes only what changed; the result is
  exactly the offline transcript. The conversation is evaluated up to what
  you have said so far.
- **It drafts the answer at your first pause.** The answer is transcribed,
  judged, written, and voiced while the turn is still open, and held.
- **The recognizer knows when you are done.** Qwen3-ASR's state as it
  ends your transcript has heard how you spoke and read what you said; a
  head trained on labeled speech judges the turn from it in the same pass,
  for free. On held-out human speech 40 ms into a pause it is right
  95.7% of the time, where Smart Turn, a separate model, is right
  89.8%. An unsure pause is judged again as it grows.
- **It lets you go on.** Speech before or just as the answer starts means
  you were not done: the draft is dropped and forgotten, and your whole
  utterance is heard again. Talk over it later and it judges, in context,
  whether you are interrupting or just saying "mm-hmm".

- **It acts at moments, and may say nothing.** The agent is asked what to
  say after each utterance, after each note (a chat message, someone
  joining), when a pause it asked for is over, and after a quiet spell.
  Speech and silence are its only primitives: it answers with words, with
  `<silent>` (to speech meant for someone else, or while asked to be quiet
  until something happens: it judges that in context, every time), or with
  words that end in `<silent 30s>`, which asks to be asked again then, so
  "remind me in thirty seconds" needs no tool.
- **It acts with tools.** A tool is a Go function whose arguments struct is
  its schema (`chat.Func`); the model calls it in Qwen3's own format, the
  call's fixed parts are drafted and checked in one pass, and its result
  becomes part of the answer, on the same voice.
- **It speaks your languages.** `Languages` limits the recognizer to the
  languages of the call, their names and their scripts: a noise in an
  English and Portuguese call cannot come out as Chinese.

It remembers only what you heard, to the word: the voice reports how far it
has spoken from the TTS model's own attention, which also times the
captions. It keeps a meeting's typed chat and who comes and goes as context
(`Note`), knows who is speaking (`Speaker`), and
[`examples/gopher`](examples/gopher) puts it in a video call. Package
`scenario` tests any `speech.Duplex` from a text script, with real audio:
the user's lines are synthesized, the agent's answers transcribed, and a
local model judges what it said.

## Example: an AI listener on a video call

[`examples/streamcall`](examples/streamcall) joins a Stream video call as a
silent participant. With Smart Turn it finds when each speaker finishes, then
transcribes the turn with Qwen3-ASR and reads the speaker's mood with Qwen3.
It posts all of that to the call's chat in real time, with every model
running on the local machine.

## Documentation

- [Go API](docs/api.md): models, lanes, the pool, the weight cache, adding backends, the CLI
- [Server](docs/server.md): endpoints, OpenAI compatibility, operation
- [Architecture](docs/architecture.md): packages, dependency rules, shared kernels
- [Models](docs/models.md): converters, provenance, licenses
- [Validation](docs/validation.md): how outputs are checked against references
- Deep dives: [Qwen3-ASR](docs/qwen3asr.md) · [Qwen3 and CLM](docs/clm-performance.md) ·
  [Whisper design](docs/whisper-design.md) · [Whisper performance](docs/whisper-performance.md) ·
  [Benchmarks](docs/benchmarks.md)

## Build and test

Go 1.27 or newer. Build with `GOEXPERIMENT=simd` for the SIMD kernels.
`CGO_ENABLED=0` works everywhere.

```sh
GOEXPERIMENT=simd go test ./...
```

Model tests look for their weights in [`models/`](models), or in
`$GOPHONIC_MODELS`, and skip when a file is missing. They never download
anything.

## License

[BSD-2-Clause](LICENSE). Model weights keep their own licenses; see
[provenance](docs/models.md#provenance-and-licenses).
