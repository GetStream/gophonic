# Go API

Load a model once, then open one lane per concurrent caller. A model's
weights are immutable and shared by every lane; a lane owns its scratch and
workers, and warm calls on it allocate nothing.

There are two layers:

- **Model-independent.** `gophonic.Open` loads any supported model by path.
  Its lanes implement the interfaces of package `speech`:
  `speech.Transcriber` for speech-to-text and `speech.TurnDetector` for
  end-of-turn detection. Applications, the CLI, and the HTTP server use only
  this layer.
- **Per model.** Packages `whisper`, `smartturn`, `tinymel`, and `qwen3`
  expose each model's own entry points: explicit workspaces, feature-level
  prediction, caller-sized result buffers, and Qwen3's text tasks.

## Open a model

```go
model, err := gophonic.Open(path, gophonic.Options{Threads: 4})
if err != nil {
	return err
}
switch model.Kind() {
case gophonic.Transcription:
	lane, err := model.NewTranscriber()
	// ...
case gophonic.TurnDetection:
	detector, err := model.NewTurnDetector()
	// ...
}
```

`Open` recognizes the format from the file: a converted Whisper, Smart Turn,
or TinyMelNet `.gophonic` bundle. It returns `ErrUnknownFormat` for anything
else. `Model.Name` reports the architecture (`"whisper"`, `"smart-turn"`,
`"tinymel"`). Asking a model for the other kind of lane fails with
`speech.ErrUnsupported`.

`Options.Threads` bounds each lane's CPU workers, including the caller: a
Whisper transcriber uses that many execution slots (default
`min(GOMAXPROCS, 8)`), and a TinyMelNet detector uses `Threads-1` helper
goroutines (default none). Smart Turn uses `GOMAXPROCS-1` helpers regardless.

## Transcription

```go
lane, err := model.NewTranscriber()
if err != nil {
	return err
}
defer lane.Close()

var t speech.Transcript
opts := speech.Options{Language: "en", Segments: true, Words: true}
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
| `Language` | English name of the detected or requested language |
| `Segments` | Timed spans, when `Options.Segments` or `Options.Words` is set |
| `Words` | Aligned words with a confidence, when `Options.Words` is set |

`Options.Language` takes an ISO 639-1 code or an English name
(`speech.LanguageName` normalizes both). An option the model cannot honor, such
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
detector, err := model.NewTurnDetector()
if err != nil {
	return err
}
defer detector.Close()

prediction, err := detector.PredictInto(pcm, 48000, 2) // on each VAD pause
if err != nil {
	return err
}
if prediction.Complete {
	// Let the dialogue policy decide whether to respond.
}
```

`PredictInto` takes interleaved mono or stereo PCM at 8–96 kHz. Audio must be
nonempty, contain complete frames, and be finite where it is read.
`Prediction.Probability` is the model's probability that the turn is complete;
`Complete` applies that model's threshold (Smart Turn `> 0.5`, TinyMelNet
`> 0.57`). The result does not borrow lane storage.

## Model packages

### whisper

```go
model, err := whisper.Load("tiny.en.gophonic")
if err != nil { return err }
worker, err := whisper.NewTranscriber(model) // or NewTranscriberWithWorkers
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
| `speech.TurnDetector` lane | `NewSession(model)` | `NewSession(model, helpers)` |

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
`Context`, and the low-level `Evaluator` for pretokenized batches.

## The turn detectors' frontend

A detector trained on the same features can reuse the frontend without any
model scratch:

```go
frontend := gophonic.NewWhisperFeatureWorkspace()
defer frontend.Close()
features := make([]float32, 80*800)

err := gophonic.ExtractWhisperFeaturesInto(pcm, 48000, 2, features, frontend)
```

It accepts mono or stereo PCM at 8–96 kHz and writes the normalized,
row-major `[80,800]` tensor with the built-in eight-second window. Once the
sample rate is warm, successful extraction does not allocate.

## Add a backend

A new model implements `speech.TurnDetector` or `speech.Transcriber` in its
own package and is passed to the application like a built-in lane. No
registration is needed. This sketch assumes your package defines `LoadModel`,
`NewScratch`, and `Model.PredictPCMInto`; those names are your backend's, not
gophonic's:

```go
type Session struct {
	model     *Model
	scratch   *Scratch
	threshold float32
	closed    bool
}

var _ speech.TurnDetector = (*Session)(nil)

func OpenSession(path string, threshold float32) (*Session, error) {
	model, err := LoadModel(path)
	if err != nil {
		return nil, err
	}
	return &Session{model: model, scratch: NewScratch(), threshold: threshold}, nil
}

func (s *Session) PredictInto(pcm []float32, rate, channels int) (speech.Prediction, error) {
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

A backend trained on Whisper's turn-detector features can hold a
`WhisperFeatureWorkspace` in its scratch and call `ExtractWhisperFeaturesInto`
before its own model. The interfaces perform no loading, feature conversion,
or allocation management on a backend's behalf; its implementation
establishes its own numerical and allocation guarantees. The
[conformance test](../external_test.go) implements `speech.TurnDetector` from
outside the module's packages with only the standalone frontend.

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
./gophonic -model base.en.gophonic -response-format verbose_json -word-timestamps speech.wav
./gophonic -model tinymel.gophonic -threads 4 speech.ogg
```

`-model` takes any bundle `gophonic.Open` recognizes. Transcription models
print `json` (`{"text":"..."}`, the default), `text`, `verbose_json`, `srt`, or
`vtt`, chosen with `-response-format`; `-word-timestamps` adds words to
`verbose_json`; `-language` and `-context` set the corresponding
`speech.Options`. Turn detectors print the prediction as JSON. Input formats:

- WAV: RIFF/WAVE PCM at 8, 16, 24, or 32 bits, or IEEE float at 32 or 64 bits;
  one or two channels at 8–96 kHz.
- Ogg Opus: `.ogg` or `.opus`, one or two channels, decoded by `gopus` to
  16 kHz with pre-skip and the final granule position applied.

The CLI reads and decodes the file, loads the model, runs once, and prints the
result. Warm library benchmarks exclude these per-process setup and I/O
costs.
