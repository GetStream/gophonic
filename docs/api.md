# Go API

Load a model once, then open one lane per concurrent caller. A model's
weights are immutable and shared by every lane; a lane owns its scratch and
workers, and warm calls on it allocate nothing.

There are two layers:

- **Model-independent.** `gophonic.Open` loads any registered model by
  path. A model provides lanes of the interface types it supports: the
  interfaces of package `speech` (`Transcriber`, `TurnDetector`,
  `AudioClassifier`, `TextClassifier`, `ZeroShot`) or any interface a
  third-party package defines. Applications, the CLI, and the HTTP server use
  only this layer.
- **Per model.** Packages `qwen3asr`, `whisper`, `smartturn`, `tinymel`, and
  `qwen3` expose each model's own entry points: decoder formats, explicit
  workspaces, feature-level prediction, caller-sized result buffers, and
  Qwen3's text tasks.

## Open a model

```go
model, err := gophonic.Open(path, gophonic.Options{Threads: 4})
if err != nil {
	return err
}
defer model.Close()
if gophonic.Supports[speech.Transcriber](model) {
	lane, err := gophonic.Lane[speech.Transcriber](model)
	// ...
}
```

A model's capabilities are the lane types it provides, not a fixed list:
`Lane[T]` opens a lane of interface type `T` and fails with
`speech.ErrUnsupported` when the model does not provide it, `Supports[T]`
asks first, and `Model.Provides` lists them.

| Model | Provides |
| --- | --- |
| Qwen3-ASR, Whisper | `speech.Transcriber` |
| Smart Turn, TinyMelNet | `speech.TurnDetector`, `speech.AudioClassifier` (incomplete, complete) |
| Qwen3 (any size) | `speech.ZeroShot`: text classifiers from a question and its answers |

```go
llm, err := gophonic.Open("models/Qwen3-8B", gophonic.Options{})
zeroShot, err := gophonic.Lane[speech.ZeroShot](llm)
moderate, err := zeroShot.Classifier("Is this message acceptable in a workplace chat?",
	[]string{"acceptable", "rude", "harassment", "spam"})
probs := make([]float32, 4)
err = moderate.ClassifyInto(ctx, message, probs) // warm calls allocate nothing
```

`Open` tries each registered format's `Match` and loads the path with the
first that claims it. The built-in formats recognize official Qwen3-ASR and
Qwen3 snapshot directories (by `config.json`'s `model_type`) and converted
Whisper, Smart Turn, and TinyMelNet `.gophonic` bundles; `Register` adds more
([Add a backend](#add-a-backend)). It returns `ErrUnknownFormat` for anything
no format claims. `Model.Close` releases the model's resources, such as GPU
memory, once its lanes are closed. `Model.Name` reports the architecture
(`"qwen3-asr"`, `"qwen3-tts"`, `"qwen3"`, `"whisper"`, `"smart-turn"`,
`"tinymel"`), and `Model.Path` the path it was loaded from.

`Detect` reports the format `Open` would use without loading anything, and
a format's `Provides` lists the lane types its models provide when that is
known up front, which is how the server picks a model for a request. `Open`
checks that the model it loads provides them.

`Options.Format` picks the weight format of the Qwen models, one vocabulary
for all of them:

| `Options.Format` | Weights |
| --- | --- |
| `gophonic.FormatF16` (`"f16"`) | every BF16 weight exactly, on the CPU's matrix units |
| `gophonic.FormatInt8` (`"int8"`) | int8 rows in a rotated basis, on the CPU |
| `gophonic.FormatGPU` (`"gpu"`) | int8 rows in a rotated basis, on the Apple GPU |
| `gophonic.FormatGPUQ8` (`"gpu-q8"`) | int8 blocks of 32, on the Apple GPU |
| `gophonic.FormatGPUQ4` (`"gpu-q4"`) | 4-bit blocks of 32, on the Apple GPU: lower fidelity |

Empty picks the fastest format of llama.cpp Q8_0 fidelity on the machine:
the Apple GPU where Metal is present, `f16` elsewhere. Whisper and the turn
detectors have one format and ignore it; an unknown name fails with
`speech.ErrUnsupported`. The model packages take the same names in their
own `Options.Format`.

`Options.Threads` bounds each lane's CPU workers, including the caller: a
Qwen3-ASR transcriber uses that many (default `min(GOMAXPROCS, 16)`, at most
8 for the encoder), a Whisper transcriber that many execution slots (default
`min(GOMAXPROCS, 8)`), a Qwen3-TTS model's codec decoder that many, and a
TinyMelNet detector `Threads-1` helper goroutines (default none). Smart Turn
uses `GOMAXPROCS-1` helpers regardless. In the model packages the same
bound is each lane's `LaneOptions.Threads`.

## Loading and the weight cache

The first load of a checkpoint converts its weights for the backend that
runs them (rotating, quantizing, and packing Qwen3's projections for the GPU,
for example) and writes the result to a cache entry. Every later load maps
that entry and uses it in place: nothing is read or converted, pages load
from the page cache as the model first touches them, and on Apple silicon the
mapped pages become GPU buffers without a copy, as llama.cpp maps GGUF files.

| Load (warm page cache, M4 Max) | Without the cache | First load, writing the entry | From the cache |
| --- | ---: | ---: | ---: |
| Qwen3-8B, GPU (`qwen3.Open`) | 4.5 s | 5.0 s | **0.09 s** |
| Qwen3-ASR-1.7B, GPU (`qwen3asr.Load`) | 1.3 s | 1.8 s | **0.085 s** |

The first load writes each finished layer to disk while it converts the
next, so building the entry costs little more than converting.

The GPU formats of Qwen3 and Qwen3-ASR, the defaults on Apple silicon, load
this way; the CPU formats still convert at every load, and Whisper and the
turn detectors load their small converted bundles directly.

Entries live in `gophonic.CacheDir()`: the user cache directory
(`~/Library/Caches/gophonic` on macOS, `~/.cache/gophonic` on Linux), or
`$GOPHONIC_CACHE`; setting it to an empty string disables the cache. An entry is keyed by the checkpoint's
path, the names, sizes, and modification times of its files, the weight
format, and a layout version, so a changed checkpoint or a new gophonic
release rebuilds it; building an entry removes the one it replaces. One
process builds an entry while others wait for it, and a crash leaves no
partial entry. A Qwen3-8B GPU entry takes about 7 GB of disk; delete the
directory at any time to reclaim it. Without a writable cache directory,
every load converts the weights in memory.

Embedding tables are used as the checkpoint stores them, so they are mapped
from the safetensors file itself rather than copied.

## Serve many models: Pool

A `Pool` opens models on demand, shares them, reuses their lanes, and closes
each model after it has gone unused for a while (15 minutes by default).
Services that load models by request use it instead of `Open`:

```go
pool := gophonic.NewPool(gophonic.Options{}, gophonic.DefaultKeepAlive)
defer pool.Close()

lease, err := gophonic.Acquire[speech.Transcriber](pool, "models/Qwen3-ASR-1.7B")
if err != nil {
	return err
}
defer lease.Release()
err = lease.Lane.Transcribe(ctx, pcm, speech.Options{}, &transcript)
```

The first `Acquire` of a model opens it (concurrent callers wait for the one
load); later ones reuse released lanes and allocate nothing. A model stays
open while any lease of it is out, and closes, with its lanes, once it has
been idle for the keep-alive time; the next `Acquire` opens it again from
the weight cache. `Pool.Close` closes idle models at once and the others as
their last leases are released.

## Transcription

```go
lane, err := gophonic.Lane[speech.Transcriber](model)
if err != nil {
	return err
}
defer lane.Close()

var t speech.Transcript
opts := speech.Options{Language: speech.English, Segments: true, Words: true}
if err := lane.Transcribe(ctx, mono16kPCM, opts, &t); err != nil {
	return err
}
for _, w := range t.Words {
	fmt.Printf("%6.2f %s\n", w.Start, t.Text[w.TextStart:w.TextEnd])
}
```

`Transcribe` takes mono float32 PCM at 16 kHz (`speech.SampleRate`) of any
length and writes the result into `dst`:

| Field | Contents |
| --- | --- |
| `Text` | The transcript bytes; offsets in segments and words index it |
| `Language` | The detected or requested language, or `speech.Unknown` |
| `Segments` | Timed spans, when `Options.Segments` or `Options.Words` is set |
| `Words` | Aligned words with a confidence, when `Options.Words` is set |
| `Turn` | Whether the speaker's turn ends where the audio does, when `Options.Turn` is set |

Languages are values: `speech.Language` is a small integer
(`speech.English`, `speech.Portuguese`, …) with `Code()` and `Name()`, and
`speech.LanguageSet` a set of them, one bit each. `speech.ParseLanguage`
reads a code or an English name in any case, and `speech.ParseLanguages` a
comma-separated list; neither allocates per language.
`Options.Language` forces a language; `Options.Languages` instead limits
the languages that may be spoken: Qwen3-ASR detects the likeliest of them
and writes only in their scripts, so noise in an English and Portuguese
call cannot come out as Chinese. A lane prepares each set once, and tells
it is the same set with one comparison, so alternating sets costs
nothing. `Options.Partial` continues an earlier
transcript of the start of the same audio, checking it in one pass and
decoding only where it differs; the result is the same.

`Options.Turn` judges the end of the speaker's turn from the transcriber's
own state, at no extra cost: Qwen3-ASR-1.7B carries a head trained on
labeled human and synthetic speech cut at pauses
([`qwen3asr/tools/turn.py`](../qwen3asr/tools/turn.py) reproduces it).
Transcribers without one fail with `speech.ErrUnsupported`. An option the model cannot honor, such
as a language it does not know or a `Context` prompt it cannot use, fails with
an error that wraps `speech.ErrUnsupported`; the Whisper English models accept
only English and no context. A closed lane returns an error wrapping
`speech.ErrClosed`.

`dst`'s slices are reused: the first call on a new audio length may grow them,
and later calls whose results fit allocate nothing. A lane must not be used
by two goroutines at once; open one per concurrent caller.

### Audio input

`speech.Resampler` converts interleaved mono or stereo PCM at 8–96 kHz to mono
16 kHz, with a 32-tap Hann-windowed sinc filter:

```go
n, err := speech.Samples16k(len(pcm), sampleRate, channels)
if err != nil {
	return err
}
mono := make([]float32, n)
resampler := speech.NewResampler()
n, err = resampler.Resample16kInto(pcm, sampleRate, channels, mono)
```

No padding, truncation, or normalization is applied. Mono 16 kHz input needs
no conversion. The CLI decodes Ogg Opus directly at 16 kHz with `gopus`.

## Turn detection

```go
detector, err := gophonic.Lane[speech.TurnDetector](model)
if err != nil {
	return err
}
defer detector.Close()

prediction, err := detector.Predict(pcm, 48000, 2) // on each VAD pause
if err != nil {
	return err
}
if prediction.Complete {
	// Let the dialogue policy decide whether to respond.
}
```

`Predict` takes interleaved mono or stereo PCM at 8–96 kHz. Audio must be
nonempty, contain complete frames, and be finite where it is read.
`Prediction.Probability` is the model's probability that the turn is complete;
`Complete` applies that model's threshold (Smart Turn `> 0.5`, TinyMelNet
`> 0.57`). The result does not borrow lane storage.

## Speech synthesis

A `speech.Synthesizer` lane speaks one utterance at a time, pushed as text
and pulled as audio: `Begin` starts an utterance, `Write` adds its text as a
language model writes it, `End` completes it, and `Read` returns samples as
they are decoded. Text and audio may be on two goroutines, and neither
waits for the other.

```go
tts, err := gophonic.Lane[speech.Synthesizer](model)
if err != nil {
	return err
}
defer tts.Close()

err = tts.Begin(ctx, speech.SpeakOptions{Voice: "ryan", Language: speech.English})
go func() { // a Synthesizer is an io.Writer: the model writes at its own pace
	session.Reply(ctx, chat.Options{}, tts)
	tts.End()
}()
frame := make([]float32, tts.SampleRate()/50)
for played := 0; ; {
	n, err := tts.Read(frame)
	if err == io.EOF {
		break
	}
	play(frame[:n])
	played += n
	spoken(tts.Voiced(played)) // bytes of the reply spoken so far
}
```

`Voiced(n)` reports how many bytes of the text the utterance's first `n`
samples speak: it never decreases as `n` grows and is all of the text once
the utterance ends. Captions follow the voice with it, an interrupted agent
keeps only what was heard, and subtitles time each word by it. `Begin` drops an utterance in progress, and cancelling its context
cuts it: `Read` then returns the context's error. `speech.Synthesize` speaks
a whole text in one call. A warm utterance allocates nothing, on any of the
lane's goroutines.

Package `speechtest` checks any implementation against this contract
(`speechtest.TestSynthesizer`) and provides `Tone`, a synthesizer whose audio
and `Voiced` are exact, for testing code that drives one.

## Model packages

### qwen3tts

Qwen3-TTS-12Hz-1.7B-CustomVoice's lane decodes each 80 ms frame on the CPU
while the GPU generates the next, and caches the voice prompt (speaker,
language, and `SpeakOptions.Style`) across utterances. A frame is the
talker's step and fifteen of the code predictor's, one per codebook; the
predictor's run whole on the GPU, heads, sampling, and all, in one
submission, so a frame takes 10.3 ms on an M4 Max, where a round trip per
codebook took 14.4. Its `Voiced` is the
talker's own position in the text: one attention head of the talker (layer
3, head 0) weighs most the text token being spoken, as Whisper's alignment
heads follow the audio, and each frame is read with it for the cost of one
head's softmax. Against Whisper's word timings of the speech, streamed a
word at a time, it is 0.27 words from the word being spoken on average and
two at worst (`TestVoicedFollowsWords`).

### qwen3asr

```go
model, err := qwen3asr.Load("models/Qwen3-ASR-1.7B", qwen3asr.Options{})
if err != nil { return err }
lane, err := qwen3asr.NewTranscriber(model, qwen3asr.LaneOptions{})
if err != nil { return err }
defer lane.Close()

err = lane.Transcribe(ctx, mono16kPCM, speech.Options{Language: speech.German, Context: "Bundestag"}, &t)
```

`Options.Format` picks the decoder's weights: `"gpu-q8"` (int8 blocks on
the Apple GPU, with the encoder on the GPU too; the default where Metal is
present), `"f16"` (every BF16 weight exactly, on the CPU), or any other
format of the table above. `speech.Options.Language` forces one of
`Model.Languages()`, a `speech.LanguageSet`, and skips detection; `Context` primes recognition with
names and terms. Qwen3-ASR produces no timestamps: `Segments` yields one
segment per decoded piece (the whole clip, or each piece of audio longer
than 20 minutes), and `Words` fails with `speech.ErrUnsupported`. Input whose
peak exceeds 1 is scaled down by its peak, as the reference normalizes
decoded audio. See [Qwen3-ASR](qwen3asr.md).

### whisper

```go
model, err := whisper.Load("tiny.en.gophonic")
if err != nil { return err }
worker, err := whisper.NewTranscriber(model, whisper.LaneOptions{})
if err != nil { return err }
defer worker.Close()

text, err := worker.TranscribeInto(mono16kPCM, make([]byte, 0, 4096))
```

`TranscribeInto` computes Whisper's full-file log-mel transform, runs
successive 30-second encoder and decoder windows, carries prior text tokens,
applies the no-speech rule, and returns text in caller storage. Decoding is
greedy at temperature zero with `without_timestamps=true`; timestamp tokens
still control seeking. `TranscribeSegmentsInto` adds segment times, and
`TranscribeWordsInto` aligns words from selected cross-attention heads with a
second decoder pass. These methods need caller buffers with room for the
result and return an error otherwise; `Transcribe` (the `speech` interface)
sizes its buffers itself. Word timing is supported for the official
`tiny.en`, `base.en`, `small.en`, and `medium.en` checkpoints; the
`medium.en` alignment mask has not been exercised with local weights.

`TranscribeWindowInto` handles one right-padded 30-second window, so its
result can differ from the full-file path for short audio;
`TranscribeFixedWindowsInto` decodes independent windows. Text keeps the
tokenizer's leading spaces, and token bytes are joined across segments before
Whisper's UTF-8 replacement rule, so the output is always valid UTF-8.

### smartturn and tinymel

| Operation | `smartturn` | `tinymel` |
| --- | --- | --- |
| Read a file | `Load(path)` | `Load(path)` |
| Read an `io.Reader` | `ReadWeights(r)` | `ReadWeights(r)` |
| Scratch | `NewWorkspace()` | `NewWorkspace()`, `NewWorkspaceWithWorkers(n)` |
| `speech.TurnDetector` lane | `NewDetector(model)` | `NewDetector(model, LaneOptions{Threads: n})` |

Both models offer three prediction entry points on a workspace:

| Method | Input |
| --- | --- |
| `PredictInto(pcm, sampleRate, channels, ws)` | Interleaved `[]float32`, mono or stereo, 8–96 kHz |
| `PredictMono16kInto(pcm, ws)` | Mono 16 kHz `[]float32` |
| `PredictFeaturesInto(features, ws)` | Exactly 64,000 float32 values in row-major `[80,800]` order |

The audio path keeps the most recent eight seconds, left-pads shorter input
with zeros, normalizes the waveform, and computes Whisper's 80-band log-mel
features. It reads the caller's PCM and writes into workspace scratch. The
feature path expects the finished tensor, indexed `features[band*800+frame]`,
and performs no resampling, padding, or normalization.

A Smart Turn workspace starts `GOMAXPROCS-1` helper goroutines at
construction. A TinyMelNet workspace is serial by default;
`NewWorkspaceWithWorkers(n)` requests `n` helpers alongside the caller,
capped at `tinymel.MaxWorkers` (seven) and at `GOMAXPROCS-1`. Set the process
CPU budget before constructing workspaces. `smartturn.FeaturesInto` exposes
Smart Turn's frontend on its own workspace.

### qwen3

See the [package README](../qwen3/README.md): `Open`, `Embed`, `Question`,
and `Context`. One `qwen3.Model` serves all of them and `chat.Generator`
from one copy of the weights, each prepared at its first use: the first
session loads the language-model head (620 MB for Qwen3-8B), the first
question or embedding its workspace and caches.

Qwen3 models provide `chat.Generator`, conversations that keep their
context evaluated between replies. `Session.Reply` writes the reply to an
`io.Writer` as it is decoded. A session may offer tools: a `chat.Tool` is
a spec (what the model sees) and a `Call`; `chat.Func` makes one of a Go
function whose arguments struct is its schema. The model calls tools in
Qwen3's own format, calls reach `Session.Calls` rather than the reply's
text, and results join the conversation as `chat.ToolResult` messages.
`chat.Answer` is the loop: reply, run the calls, reply again knowing them.

```go
tools := []chat.Tool{chat.Func("now", "The current time.",
	func(ctx context.Context, args struct{}) (string, error) { return time.Now().String(), nil })}
s, err := gen.NewSession("You are a helpful assistant.", chat.Specs(tools)...)
s.Add(chat.User, "What time is it?")
err = chat.Answer(ctx, s, tools, chat.Options{}, os.Stdout, 4)
```

A call's fixed parts, `{"name": "now", "arguments":`, are drafted and checked
in one pass, sampling each position as decoding one token at a time would.

## The turn detectors' frontend

A detector trained on the same features can reuse the frontend, which
lives in package `speech` and holds no model scratch:

```go
frontend := speech.NewTurnFeatures()
defer frontend.Close()
features := make([]float32, speech.TurnFeatureBands*speech.TurnFeatureFrames)

err := frontend.Into(pcm, 48000, 2, features)
```

It accepts mono or stereo PCM at 8–96 kHz and writes the normalized,
row-major `[80,800]` tensor with the built-in eight-second window. Once the
sample rate is warm, successful extraction does not allocate.

## Add a backend

A new model implements whatever lane interfaces fit it, from package
`speech` or its own, and is passed to the application like a built-in lane.
To make `gophonic.Open` (and so the CLI and server) load it by path, register
a format whose `Open` builds a `Model` and provides its lanes:

```go
gophonic.Register(gophonic.Format{
	Name:  "my-turn",
	Match: func(path string) bool { return strings.HasSuffix(path, ".myturn") },
	Open: func(path string, opts gophonic.Options) (*gophonic.Model, error) {
		model, err := LoadModel(path)
		if err != nil {
			return nil, err
		}
		m := gophonic.NewModel("my-turn", nil)
		gophonic.Provide(m, func() (speech.TurnDetector, error) { return OpenSession(model, 0.5) })
		// A capability gophonic does not define is provided the same way.
		return gophonic.Provide(m, func() (mypkg.Diarizer, error) { return NewDiarizer(model) }), nil
	},
})
```

Formats registered later are tried first, so a registered format can take
over paths a built-in one would claim; `gophonic.Formats` lists them in
order. This sketch assumes your package defines `LoadModel`, `NewScratch`,
and `Model.PredictPCMInto`; those names are your backend's, not
gophonic's:

```go
type Session struct {
	model     *Model
	scratch   *Scratch
	threshold float32
	closed    bool
}

var _ speech.TurnDetector = (*Session)(nil)

func OpenSession(model *Model, threshold float32) (*Session, error) {
	return &Session{model: model, scratch: NewScratch(), threshold: threshold}, nil
}

func (s *Session) Predict(pcm []float32, rate, channels int) (speech.Prediction, error) {
	if s == nil || s.closed {
		return speech.Prediction{}, speech.ErrClosed
	}
	p, err := s.model.PredictPCMInto(pcm, rate, channels, s.scratch)
	if err != nil {
		return speech.Prediction{}, err
	}
	return speech.Prediction{Probability: p, Complete: p > s.threshold}, nil
}

func (s *Session) Close() error {
	if s != nil && !s.closed {
		s.closed = true
		s.scratch.Close()
	}
	return nil
}
```

A backend trained on the turn detectors' features can hold a
`speech.TurnFeatures` in its scratch and call its `Into` before its own
model, importing nothing but `speech`. The interfaces perform no loading, feature conversion,
or allocation management on a backend's behalf; its implementation
establishes its own numerical and allocation guarantees. The
[conformance test](../speech/turnfeatures_test.go) implements
`speech.TurnDetector` from outside package `speech` with only the
standalone frontend.

## Ownership, lifetime, and allocations

A lane or workspace belongs to one caller at a time. Do not overlap calls on
it or race `Close` with a call. Helper goroutines divide one call internally;
they do not make a lane safe for several callers. `Close` is idempotent when
called serially, waits for helpers to exit, and leaves the lane unusable.

The zero-allocation contract covers successful repeated calls with loaded
weights, reusable scratch, and warmed sizes: a sample rate already seen, an
audio length no longer than before, and result buffers with enough capacity.
Dispatch through the `speech` interfaces adds no allocation. Loading, lane
construction, error formatting, file decoding, and result serialization are
outside the contract.

A service handling many conversations should budget total CPU: helpers for
every simultaneous lane can oversubscribe the process. Serial lanes or fewer
helpers may give better throughput even when more helpers lower isolated
latency.

## CLI

```sh
./gophonic -model models/Qwen3-ASR-1.7B one.wav two.ogg three.opus
./gophonic -model models/base.en.gophonic -response-format verbose_json -word-timestamps speech.wav
./gophonic -model models/tinymel.gophonic -threads 4 turn.wav
./gophonic -model models/Qwen3-1.7B -question "Is this message spam?" -labels "spam,not spam" "WIN A FREE CRUISE" "lunch at 1?"
```

`-model` takes any path `gophonic.Open` recognizes, and the command runs the
model once per input, loading it once. Transcription models print `json`
(`{"text":"..."}`, the default), `text`, `verbose_json`, `srt`, or `vtt`,
chosen with `-response-format`; `-word-timestamps` adds words to
`verbose_json`; `-language` and `-context` set the corresponding
`speech.Options`. Turn detectors print each prediction as a JSON line, and
classifiers each input's label probabilities. A language model answers
`-question` about each text argument with one of `-labels`. Input formats:

- WAV: RIFF/WAVE PCM at 8, 16, 24, or 32 bits, or IEEE float at 32 or 64 bits;
  one or two channels at 8–96 kHz.
- Ogg Opus: `.ogg` or `.opus`, one or two channels, decoded by `gopus` to
  16 kHz with pre-skip and the final granule position applied.
