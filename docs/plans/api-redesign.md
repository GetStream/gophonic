# API redesign: every public surface

Status: design and first five steps, 2026-09-26, branch `feat/asr-turn`.
Nothing is released, so every change below is breaking and meant to be.
The migration plan at the end orders the work as PR-sized steps that each
keep the tests green.

Implemented in the tree (steps 1–5 of section 12, with what the MoE branch
needed carried over): `chat.Session.Reply` into an `io.Writer`,
`chat.Tool`/`Func`/`Specs`/`Answer`, `chat.Options.Presence`, the role
`chat.ToolResult`; `speech.Duplex.Step(in, out)` and `Note`,
`speech.TurnDetector.Predict`, no `Model.NewTranscriber`/`NewTurnDetector`;
the cascade's moments (`Note`, `Idle`, `<silent 30s>`, `<silent until
"...">` with a reminder note at each later moment), `Speaker`, the
`Observer` with typed stages, `Config.Quiet` (a silence is overruled when
nothing asked for quiet: any silence alone with one person, one until
something happens in a meeting; and a request for quiet is pointed out to
the model before it answers), `Config.Wake` (while the model keeps a
silence it chose until something happens, each utterance is judged against
what it named: the model answers, told so, when it happened, and the words
join the conversation unanswered otherwise; after `MaxSilence` the model
judges for itself, reminded), and the scenario harness (package
`scenario`) with Gopher's ten scripts. Measured on Qwen3-8B, the model
alone neither held a chosen silence ("What is two plus two?" got "Four.")
nor ended it on "Hi!", answered `<silent>` after a tool's result, and read
a chat note aloud; the harness found each of these in minutes, and the
two judges and one note fixed all but the last. The `qwen3`
package reads the checkpoint's chat template (`thinks`) so the
Instruct-2507 models, whose template has no thinking block, open replies
and zero-shot answers after the assistant header, and `IsModelDir` accepts
`qwen3_moe`.

Step 5 is in the tree too: `speech.Synthesizer` is `Begin`, `Write`, `End`,
`Read`, and `Voiced`, with `speech.Synthesize` and package `speechtest`
(`Tone` and the `TestSynthesizer` conformance suite). `Voiced` departs from
section 7's design, which took the offset of the last text token fed: the
talker is fed 12.5 tokens a second and speaks three or four, so that offset
runs seconds ahead of the voice. It comes instead from the talker's
alignment head (layer 3, head 0 of 448, found by ranking every head against
Whisper's word timings; 0.27 words from the word being spoken, where the
median head is 6 away), read each frame through `qwen3lm.Probe`, a
one-token extension's view of one head, on the CPU and in a `probe1` Metal
kernel. `Voiced(samples)` answers for any sample of the utterance, so the
cascade asks it for the sample playing: captions show whole words up to
it, and an interruption keeps the reply's bytes that were heard. The code
predictor's fifteen steps per frame run as one GPU submission through
`qwen3lm.Decoder` (FP16 heads with the final norm folded in, a radix-select
top-k sampler, input rows gathered on the GPU): 10.3 ms per frame, from
14.4. Steps 6–12 remain as planned.

## 0. The decisions in one page

1. **One way to open, one way to get a lane.** `gophonic.Open(path,
   Options{Threads, Format})` then `gophonic.Lane[T](model)`. The
   per-model conveniences (`Model.NewTranscriber`, `NewTurnDetector`), the
   root-package turn frontend, and the eager Qwen3 "questions" model go.
   `Format` is one vocabulary for every model (`f16`, `int8`, `gpu`,
   `gpu-q8`, `gpu-q4`), and every model type is closed with `Close`.
2. **Languages are values.** `speech.Language` (a byte) and
   `speech.LanguageSet` (a bitmask) replace strings and string slices in
   options and transcripts: comparable, no allocation, no per-call
   normalization, and a script mask keyed by a set instead of by
   `slices.Equal` on strings.
3. **Replies stream into an `io.Writer`, not a closure.**
   `chat.Session.Reply(ctx, opts, w io.Writer)`. Tools become
   `chat.Tool` (spec plus `Call`), `chat.Func` derives one from a typed Go
   function, and `chat.Answer` is the reply-run-tools-reply loop every
   agent, text or voice, uses.
4. **Speech is pushed text and pulled audio.** `speech.Synthesizer` is a
   stateful lane: `Begin(ctx, opts)`, `Write(text)`, `End()`, `Read(pcm)`,
   `Voiced()`. No callbacks, no goroutines in the caller, and `Voiced`
   reports how many bytes of the text the audio read so far has spoken, so
   captions follow the voice from the talker's own text position instead of
   a characters-per-second guess.
5. **The duplex acts at moments, and its only primitives are speech and
   silence.** A moment is any time the agent is asked what to say: after an
   utterance, after a `Note` (chat, presence, a tool result), when a pause
   it asked for is over (`<silent 30s>`), and every `Idle` of quiet. The
   model answers with words, `<silent>`, or words ending in `<silent 30s>`.
   That one grammar subsumes today's `SilencePrompt` with
   `<silent until "...">`, the zero-shot `Wake` check, `maxSilence`, the
   `Silent()` tool flag, and the `Heard` addressing hook: the model reads
   what happened and decides, in context, every time. Reminders are a
   planned moment, not a tool.
6. **Multi-party attribution is a note, not a prefix.** The application
   tells the cascade who is speaking (`Speaker(name)`); the cascade adds
   "Ana is speaking." as a system note when the speaker changes and passes
   the words through untouched, so small models are not handed "Ana said:
   ..." to read back. The echo guard stays as a last line of defence on
   the reply, no longer as the thing that makes attribution work.
7. **Observers are an interface with typed stages.** `duplex.Observer`
   (`Heard`, `Said`, `Stage`, `Error`) replaces four callback fields and
   the `fmt.Sprintf`-built stage strings.
8. **A scenario harness for any `speech.Duplex`.** Package `scenario` runs
   plain text scripts (`user: Stop talking until I say hi.` /
   `gopher: silent`) end to end: lines are spoken with a synthesizer, the
   agent's audio is transcribed, and a local model judges claims. The
   Gopher example ships the behaviours the user tested by voice as
   scenario files.
9. **The engine stays internal.** `qwen3` stops re-exporting
   `internal/qwen3lm`'s `Weights`, `Evaluator`, `Workspace`, `PrefixKV`,
   and `Embeds`, so the hybrid Gated DeltaNet and MoE cores can change the
   engine's shape (states that are not keys and values, experts in the
   geometry) without a public API change. One `qwen3.Model` provides chat,
   questions, and embeddings from one loaded copy of the weights.
10. **Optional interfaces for the fast paths.** Batching
    (`speech.BatchTranscriber`), draft models (`chat.Drafted`), and native
    duplex models are discovered by type assertion, so the common
    interfaces stay small and a backend adds what it can.

## 1. Rules every API follows

These restate the standing requirements as design rules, so each proposal
below can be checked against them.

- **Setup allocates, steady state does not.** Opening a model, opening a
  lane, preparing a question, starting a session, and the first call at a
  new size may allocate. Every later call at that size or smaller writes
  into caller-owned or lane-owned storage and allocates nothing. Every
  lane method that is called per frame, per token, or per utterance has an
  `AllocsPerRun == 0` test.
- **No strings, maps, closures, or boxing on hot paths.** Text crosses hot
  paths as `[]byte` into reusable buffers; options are value types
  (`Language`, `LanguageSet`, enums); callbacks are interfaces implemented
  by a struct the caller already owns, or `io.Writer`; nothing on a hot
  path is keyed by a string.
- **A lane owns its scratch and its goroutines; a model is immutable.**
  Weights are shared read-only. Lanes may run concurrently with each
  other; a lane is used by one goroutine at a time. `Close` is idempotent.
- **The caller's clock never blocks on a model.** `Duplex.Step` runs on an
  audio thread and does no inference; lanes that stream (`Synthesizer`)
  keep their own worker and bounded rings.
- **Batching, speculation, and overlap are inside lanes, exposed by
  capability.** Prefix-KV reuse, checkpoint and restore, draft-and-verify,
  and CPU/GPU overlap are how lanes are implemented; the interfaces only
  make them possible (sessions keep state, transcripts continue partials,
  synthesizers decode while they generate) and optional interfaces expose
  batching where a backend has it.
- **Extension is by small interfaces, in leaf packages.** `speech` and
  `chat` import no model code; any package implements them.
  `gophonic.Register` adds formats; `Provide[T]` adds capabilities the
  root has never heard of; `chat.Tool` adds tools; `scenario` tests any
  `speech.Duplex`.
- **Local inference, `vibejson`, Q8_0 fidelity** are unchanged and
  restated where an API touches them (tools that fetch data are fine;
  every `Format` that is not exact is validated against BF16).

## 2. Package `gophonic` (root)

### Problems

- `Model.NewTranscriber` and `Model.NewTurnDetector` (`gophonic.go:111-115`)
  are two of five lane types with a method; the others (`Synthesizer`,
  `Generator`, `ZeroShot`) have none. Two ways to do one thing.
- `Options` has only `Threads` (`gophonic.go:25-29`); the format choice
  that every Qwen model offers is reachable only through the model
  packages, with three names for it: `qwen3.Options.Weights`
  (`qwen3/model.go:89`), `qwen3asr.Options.Format` (`qwen3asr/model.go:54`),
  `qwen3tts.Options.Format` (`qwen3tts/model.go`), and constants
  `qwen3.WeightsGPU` versus `qwen3asr.FormatGPU`. `openQwen3ASR` ignores
  the options entirely (`formats.go:38`).
- Opening a Qwen3 model builds a `Chat` and then a `Questions` model
  (`formats.go:63-67`): a second workspace, a 2048-token prefix store
  (288 KiB per token on Qwen3-8B, so about 590 MB), a 4096-entry embedding
  cache, and the letter head, whether or not anyone asks a question. The
  Gopher agent never does. Its "lanes" (`formats.go:79-95`) are wrappers
  whose `Close` is a no-op, and every session serializes on one mutex
  (`qwen3/chat.go:312`), so the lane contract ("lanes run concurrently")
  is not what the model does.
- Models release resources with `Release()` (`qwen3asr/model.go:259`,
  `qwen3tts/model.go:324`, `qwen3lm.Weights.Release`) but `Close()`
  everywhere else (`whisper.Model.Close`, every lane, `Model.Close`).
- `Model.Name()` is the format name, not the checkpoint; there is no
  `Path()`, so the server keeps its own name-to-path table
  (`internal/httpserver/server.go:36-43`) and `Lease.Path()` exists only for
  it.
- `Format.Provides` (`gophonic.go:139-142`) is declared separately from
  what `Open` provides and can drift; the server relies on it to choose a
  default model without loading (`server.go:174-179`).
- The turn detectors' mel frontend lives in the root package
  (`features.go`), against the architecture rule that the root only
  dispatches; third-party detectors must import the root to reuse it.
- The weight cache is configured by `$GOPHONIC_CACHE` only, undocumented in
  the API, with no way to ask where entries live.
- `Pool` has no per-model lane cap and no per-model options; every model
  opens with the pool's one `Options`.

### Proposed API

```go
package gophonic

// Options configures Open. The zero value picks each model's defaults.
type Options struct {
	// Threads bounds the CPU workers of each lane, including the caller.
	// Zero picks the model's default.
	Threads int
	// Format names the weight format of models that offer a choice: one of
	// the Format constants. Empty picks the fastest format that keeps
	// llama.cpp Q8_0 fidelity on this machine (the Apple GPU where Metal is
	// present, exact FP16 weights on the CPU elsewhere). Models with one
	// format ignore it.
	Format string
}

// Weight formats of the Qwen models. Every format that is not exact is
// validated against BF16 hidden states at or above llama.cpp Q8_0.
const (
	FormatF16   = "f16"    // every BF16 weight exactly, on the CPU
	FormatInt8  = "int8"   // rotated int8 rows on the CPU's matrix units
	FormatGPU   = "gpu"    // int8 rows on the Apple GPU
	FormatGPUQ8 = "gpu-q8" // int8 blocks of 32 on the Apple GPU
	FormatGPUQ4 = "gpu-q4" // 4-bit blocks of 32 on the Apple GPU: lower fidelity
)

// Model is a loaded model: immutable weights shared by the lanes opened
// from it. What it can do is the set of lane types it provides.
type Model struct{ /* unexported */ }

func Open(path string, opts Options) (*Model, error)
func Detect(path string) (Format, bool)

func (m *Model) Name() string             // the architecture: "qwen3-asr"
func (m *Model) Path() string             // what Open was given
func (m *Model) Provides() []reflect.Type // lane types, in the format's order
func (m *Model) Close() error             // once its lanes are closed; idempotent

// Lane opens a lane of type T, or fails with speech.ErrUnsupported.
func Lane[T any](m *Model) (T, error)
func Supports[T any](m *Model) bool

// Backends. A Format's Open builds a Model and provides its lanes.
type Format struct {
	Name     string
	Match    func(path string) bool
	Open     func(path string, opts Options) (*Model, error)
	Provides []reflect.Type // known without loading; nil if not
}
func Register(f Format)
func Formats() []Format
func NewModel(name, path string, close func() error) *Model
func Provide[T any](m *Model, open func() (T, error)) *Model

// CacheDir reports where prepared weights are cached: $GOPHONIC_CACHE, or
// the user cache directory. An empty $GOPHONIC_CACHE disables the cache.
func CacheDir() string

// Pool, Lease, Acquire, DefaultKeepAlive, ErrPoolClosed: unchanged.
```

Removed: `Model.NewTranscriber`, `Model.NewTurnDetector`,
`WhisperFeatureWorkspace`, `ExtractWhisperFeaturesInto` (they move to
`speech.TurnFeatures`, section 3).

`Format.Provides` stays, but the root checks it: after a format's `Open`
returns, `Open` verifies that the model provides at least what the format
declared and fails otherwise, so the declaration cannot drift silently.

### Examples

```go
model, err := gophonic.Open("models/Qwen3-ASR-1.7B", gophonic.Options{})
defer model.Close()
asr, err := gophonic.Lane[speech.Transcriber](model)
defer asr.Close()

var t speech.Transcript
err = asr.Transcribe(ctx, pcm16k, speech.Options{Languages: speech.Languages(speech.English, speech.Portuguese)}, &t)
fmt.Println(string(t.Text), t.Language)
```

```go
// A backend: a format whose model provides a lane type gophonic never
// heard of.
gophonic.Register(gophonic.Format{
	Name:  "my-diarizer",
	Match: func(p string) bool { return strings.HasSuffix(p, ".diar") },
	Open: func(p string, opts gophonic.Options) (*gophonic.Model, error) {
		w, err := LoadWeights(p)
		if err != nil {
			return nil, err
		}
		m := gophonic.NewModel("my-diarizer", p, w.Close)
		return gophonic.Provide(m, func() (mypkg.Diarizer, error) { return NewDiarizer(w, opts.Threads) }), nil
	},
	Provides: []reflect.Type{reflect.TypeFor[mypkg.Diarizer]()},
})
```

### Requirements

- Zero-alloc: `Open`, `Lane`, `Register` are setup. `Pool.Acquire` of a
  released lane allocates nothing (tested today, `pool_test.go:97`).
- Perf: `Format` is passed down so the GPU formats are chosen in one place;
  the Qwen3 model no longer pays for an unused questions workspace.
- Extensibility: unchanged mechanism (`Register`, `Provide[T]`), now with a
  checked `Provides` and a `Path` so servers need no side table.

## 3. Package `speech`

### Problems

- `Options.Language string` and `Languages []string`
  (`speech/speech.go:52-57`) are normalized on every call through a map
  (`speech.LanguageName`, `language.go:40`), and `qwen3asr` decides whether
  its allowed-token list and script mask are stale with
  `slices.Equal(t.allowedFor, names)` (`qwen3asr/transcriber.go:502,527`):
  a caller that alternates two language sets rebuilds a 151 936-entry mask
  on every call. `Transcript.Language` is a string too.
- `TurnDetector.PredictInto(pcm, rate, channels) (Prediction, error)`
  (`speech.go:167`) is named `Into` but writes into nothing.
- `Synthesizer.Speak(ctx, opts, next func() ([]byte, error), out func([]float32) error)`
  (`synthesis.go:23`) takes two closures per utterance. Every caller that
  streams text from a language model has to build a goroutine, a channel,
  and the closures itself; the cascade does (`duplex/cascade.go:1162-1196`)
  with a `chan string` that boxes a string per piece and a `[]byte(p)`
  copy per piece (`cascade.go:1184`). There is no way to learn how much of
  the text has been voiced, so captions are timed by
  `charsPerSecond = 14` (`cascade.go:89, 1153`).
- `Duplex.Step(ctx, in, out)` (`duplex.go:32`) takes a context on an
  audio thread and only returns `ctx.Err()` (`cascade.go:431`). The
  `Cascade` has `Add(role, text)` and `Buffered()` that the interface lacks,
  so an application written against `speech.Duplex` cannot give the agent
  context.
- The turn frontend for third-party detectors lives in the root package.

### Proposed API

```go
package speech

const SampleRate = 16000

var (
	ErrClosed      = errors.New("speech: closed")
	ErrUnsupported = errors.New("speech: unsupported option")
)

// Language is a spoken language known to at least one gophonic model.
// The zero value is unknown. It is a byte: comparable, allocation-free,
// and the index of a bit in LanguageSet.
type Language uint8

const (
	Unknown Language = iota
	Arabic
	Cantonese
	Chinese
	// ... every language of today's table, in its order ...
	Vietnamese
)

// ParseLanguage reads an ISO 639 code ("pt") or an English name
// ("Portuguese", "portuguese"). It does not allocate.
func ParseLanguage(s string) (Language, bool)
func (l Language) Code() string
func (l Language) Name() string
func (l Language) String() string // Name

// LanguageSet is a set of languages. It is comparable, so a lane can tell
// in one comparison whether a call's set is the one it prepared for.
type LanguageSet uint64

func Languages(l ...Language) LanguageSet
func (s LanguageSet) Has(l Language) bool
func (s LanguageSet) With(l Language) LanguageSet
func (s LanguageSet) Len() int
func (s LanguageSet) All() iter.Seq[Language]
func (s LanguageSet) String() string // "en,pt"; allocates

// Transcriber turns speech into text. One lane per concurrent caller.
type Transcriber interface {
	// Transcribe transcribes mono 16 kHz PCM into dst, reusing the capacity
	// of dst's slices. Warm calls whose results fit allocate nothing.
	Transcribe(ctx context.Context, pcm []float32, opts Options, dst *Transcript) error
	Close() error
}

// Options adjusts one transcription. The zero value detects the language
// and returns text only.
type Options struct {
	Language  Language    // forces the language; Unknown detects it
	Languages LanguageSet // when Language is Unknown: the languages that may be spoken
	Context   string      // primes recognition: names, terms
	Segments  bool
	Words     bool
	Partial   *Transcript // an earlier transcript of the start of the same audio
	Turn      bool        // judge Transcript.Turn from the recognizer's state
}

type Transcript struct {
	Text     []byte
	Language Language
	Segments []Segment
	Words    []Word
	Turn     Prediction
}
func (t *Transcript) Reset()

// Segment, Word, Prediction: unchanged.

// BatchTranscriber is a Transcriber that can transcribe several clips in
// shared forward passes. Servers and tools ask for it with a type
// assertion; a Transcriber need not provide it.
type BatchTranscriber interface {
	Transcriber
	// TranscribeBatch transcribes clips[i] into dst[i]. Warm calls on a
	// batch no larger than an earlier one allocate nothing.
	TranscribeBatch(ctx context.Context, clips [][]float32, opts Options, dst []Transcript) error
}

// TurnDetector predicts from the latest audio whether a speaker is done.
// PCM is interleaved mono or stereo at 8–96 kHz.
type TurnDetector interface {
	Predict(pcm []float32, sampleRate, channels int) (Prediction, error)
	Close() error
}

// AudioClassifier, TextClassifier, ZeroShot: unchanged.

// Synthesizer turns text into speech as the text arrives. It is a lane
// with one utterance at a time: Begin starts one, Write adds its text,
// End says the text is complete, and Read pulls its audio, on any two
// goroutines. Begin while an utterance is in progress drops it, as an
// interruption does. Warm utterances allocate nothing.
type Synthesizer interface {
	SampleRate() int
	Voices() []string
	// Begin starts an utterance in opts's voice. Speech ends when ctx is
	// done, when the text ends, or when the next Begin starts.
	Begin(ctx context.Context, opts SpeakOptions) error
	// Write adds text to the utterance. It returns once the text is
	// buffered; the buffer holds a reply's worth, so a language model
	// writing at its own pace is not throttled by playback.
	Write(text []byte) (int, error)
	// End marks the text complete: Read returns io.EOF after the last
	// sample.
	End() error
	// Read fills pcm with the next samples and reports how many. It blocks
	// until at least one sample is decoded or the utterance ends.
	Read(pcm []float32) (int, error)
	// Voiced reports how many bytes of the text written so far the samples
	// read so far have spoken, from the model's own position in the text.
	Voiced() int
	Close() error
}

type SpeakOptions struct {
	Voice    string
	Language Language
	// Style says how to speak, in words ("Calm and even; never laugh."),
	// for models that take an instruction. It is part of the voice prompt,
	// cached with the voice, so a warm styled utterance allocates nothing.
	Style string
}

// Synthesize speaks text in one call, appending the audio to dst: the
// simple path for tools and tests.
func Synthesize(ctx context.Context, s Synthesizer, opts SpeakOptions, text string, dst []float32) ([]float32, error)

// Duplex is a conversational agent that listens and speaks at once.
// Step runs on the caller's clock and does no inference.
type Duplex interface {
	Rates() (in, out int)
	Frame() (in, out int)
	// Step consumes one frame the agent hears (nil: a gap) and writes one
	// frame it speaks. Warm steps allocate nothing.
	Step(in, out []float32) (DuplexState, error)
	// Say speaks text as soon as the agent can.
	Say(text string) error
	// Note tells the agent, in words, that something happened outside the
	// conversation: a chat message, someone joining, a timer. It joins the
	// agent's context, and the agent may answer it.
	Note(text string) error
	Close() error
}

type DuplexState uint8 // Listening, Thinking, Speaking: unchanged

// TurnFeatures is the turn detectors' waveform-to-log-mel frontend, for
// detectors trained on the same features. Moved from the root package.
type TurnFeatures struct{ /* unexported */ }
const TurnFeatureBands, TurnFeatureFrames = 80, 800
func NewTurnFeatures() *TurnFeatures
func (f *TurnFeatures) Into(pcm []float32, sampleRate, channels int, dst []float32) error
func (f *TurnFeatures) Close()

// Resampler, Samples16k, Resample16kInto: unchanged.
```

### Examples

```go
// Transcribe a file (the CLI's path).
pcm, rate, channels, _ := audiofile.ReadPath("talk.wav")
mono := make([]float32, must(speech.Samples16k(len(pcm), rate, channels)))
n, _ := speech.NewResampler().Resample16kInto(pcm, rate, channels, mono)
var t speech.Transcript
lang, _ := speech.ParseLanguage("de")
err := asr.Transcribe(ctx, mono[:n], speech.Options{Language: lang, Context: "Bundestag"}, &t)
```

```go
// Stream a call: transcribe while the speaker talks, each pass continuing
// the last; the result equals the offline transcript.
var prev, cur speech.Transcript
for chunk := range chunks { // every 500 ms
	err := asr.Transcribe(ctx, audioSoFar, speech.Options{Partial: &prev, Turn: true}, &cur)
	prev, cur = cur, prev
	if prev.Turn.Complete { break }
}
```

```go
// Speak what a language model writes, and caption it as it is voiced.
tts.Begin(ctx, speech.SpeakOptions{Voice: "ryan"})
go func() { session.Reply(ctx, opts, ttsWriter{tts}); tts.End() }()
frame := make([]float32, tts.SampleRate()/50)
for {
	n, err := tts.Read(frame)
	if err == io.EOF { break }
	play(frame[:n])
	captions.Show(replyText[:tts.Voiced()])
}
```

### Requirements

- Zero-alloc: `Language` and `LanguageSet` are values; `Options` holds no
  slice but `Partial`'s pointer; `Transcript` buffers are caller-owned and
  reused. The synthesizer's text and audio rings are lane-owned and sized
  at `Begin` for one reply; `Read` copies into the caller's frame.
- Perf: `Partial` is the draft-and-verify path (`qwen3asr/transcriber.go:460`)
  and stays the one API for streaming transcription; `BatchTranscriber`
  lets the server pack clips into shared passes; the synthesizer decodes
  frame k on the CPU while the GPU generates frame k+1 as today, now
  behind `Read`.
- Extensibility: every interface is implementable outside the module
  (`external_test.go` keeps proving it for `TurnDetector`; a new
  conformance test does the same for `Synthesizer` with a tone
  generator).

## 4. Package `chat`

### Problems

- `Session.Reply(ctx, opts, sink func(piece []byte) error)`
  (`chat/chat.go:81`): a closure per reply; the cascade's closure captures
  eight variables (`cascade.go:1200-1268`) and is built per reply, and the
  same shape is needed by the HTTP server (SSE), the CLI, and tests.
- Tools are split between `chat.ToolSpec` (the schema) and
  `duplex.Tool{Run}` (`duplex/tool.go:19-27`), so a text agent (the
  `examples/qwen3/tools` example, a chat completions endpoint) has no
  runnable tool type, and the reply-run-reply loop exists only inside the
  cascade (`cascade.go:1272-1277, 1468-1495`). `Run` returns a `reply bool`
  whose only use is the `Silent()` hack for "be quiet" tools
  (`tool.go:63-72`).
- `Options` lacks the presence penalty Qwen3.6 is tuned with (1.5) and any
  way to ask for thinking mode.

### Proposed API

```go
package chat

type Role uint8 // System, User, Assistant, Tool: unchanged

type Options struct {
	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	// Presence lowers the logit of every token already in the reply by
	// this much; Qwen3.6's non-thinking mode is tuned for 1.5.
	Presence float32
	Seed     uint64
}

type Generator interface {
	NewSession(system string, tools ...ToolSpec) (Session, error)
	Close() error
}

type Session interface {
	Add(role Role, text string) error // does not retain text
	// Reply generates the assistant's next message, writing its text to w
	// in pieces that end on UTF-8 boundaries; tool calls go to Calls. It
	// ends when the model ends it, after MaxTokens, when w returns an
	// error, or when ctx is done. Warm replies allocate nothing.
	Reply(ctx context.Context, opts Options, w io.Writer) error
	Calls() []Call
	Finished(ctx context.Context, role Role, text string) (float32, error)
	Prefill(ctx context.Context) error
	Truncate(n int) error
	Checkpoint() int
	Restore(mark int) error
	Close() error
}

// Drafted is a Generator that decodes with a draft model, checking its
// tokens in one pass of the main model. Optional.
type Drafted interface {
	Generator
	SetDraft(draft Generator) error
}

// Tools.
type ToolSpec struct{ Name, Description, Parameters string } // unchanged
type Call struct { Name string; Arguments []byte }           // unchanged

// Tool is a function the model may call. Its spec tells the model what
// it does; Call does it. Tools are slow paths (a lookup, a clock): they
// may allocate.
type Tool interface {
	Spec() ToolSpec
	Call(ctx context.Context, arguments []byte) (result string, err error)
}

// Func makes a Tool of a Go function whose arguments struct is its
// schema (json, desc, enum, omitempty tags), as duplex.Func does today.
func Func[Args any](name, description string, run func(ctx context.Context, args Args) (string, error)) Tool

// Specs returns the tools' specs, for NewSession.
func Specs(tools []Tool) []ToolSpec

// Answer replies, runs the calls the reply made, adds their results, and
// replies again knowing them, up to rounds times: the loop every agent
// runs. A tool the session was not offered, or a call that fails, is
// answered with an error result the model can read.
func Answer(ctx context.Context, s Session, tools []Tool, opts Options, w io.Writer, rounds int) error
```

### Examples

```go
llm, _ := gophonic.Open("models/Qwen3-8B", gophonic.Options{})
gen, _ := gophonic.Lane[chat.Generator](llm)
tools := []chat.Tool{chat.Func("now", "The current time.", now)}
s, _ := gen.NewSession("You are helpful.", chat.Specs(tools)...)
s.Add(chat.User, "What time is it?")
err := chat.Answer(ctx, s, tools, chat.Options{}, os.Stdout, 4)
```

```go
// A tool: the arguments struct is the schema.
var now = chat.Func("now", "The current date and time.",
	func(ctx context.Context, args struct {
		Timezone string `json:"timezone,omitempty" desc:"an IANA time zone"`
	}) (string, error) { ... })
```

### Requirements

- Zero-alloc: `Reply` writes lane-owned bytes to `w`; the caller's writer
  is a struct it already owns. Tool calls are recorded in lane-owned
  `[][]byte` (`qwen3/chat.go:427-441` today). `Add` does not retain
  `text`, so a caller with bytes may pass `unsafe.String` and copy nothing.
- Perf: prefix-KV reuse, `Checkpoint`/`Restore` for speculative prefill,
  `Finished` sharing its pass with the reply's header, and drafted tool
  scaffolds are unchanged; `Drafted` adds draft-model decoding without
  touching `Session`. A `Generator` may batch the decode steps of its
  sessions on one GPU pass internally (continuous batching); the interface
  does not preclude it because sessions never share a caller.
- Extensibility: `Tool` is an interface, so an MCP client can expose a
  server's tools as `[]chat.Tool` (package `mcp`, later) with no change
  here; `Answer` serves text agents, the HTTP chat endpoint, and the
  cascade alike.

## 5. Package `duplex`: moments

### Problems

- `Config` (`cascade.go:98-153`) mixes lanes, options, three policies, and
  four callbacks. `Heard func(text string) (message string, answer bool)`
  conflates attribution with addressing; Gopher uses it to prefix
  "Ana said: " (`examples/gopher/main.go:111-116`), which makes Qwen3-8B
  read the words back (hence `echoOf`, `cascade.go:1372-1418`).
- The agent acts only after an utterance. Silence over time is three
  special cases: `SilencePrompt` with `<silent until "...">`
  (`cascade.go:1361`), the zero-shot `Wake` check with its own labels
  (`cascade.go:1438-1458`), and `maxSilence` (`cascade.go:1366`). A
  reminder cannot be expressed at all, and the tool flag `reply bool`
  (`tool.go:26`) exists to end a reply after "be quiet".
- Stage names are `fmt.Sprintf`-built strings on worker paths
  (`cascade.go:659, 673, 929, 1486`); captions are estimated from
  characters per second (`cascade.go:1153`); `showUser` re-derives
  `speakable(reply.String())` for every piece (`cascade.go:1152`), which is
  quadratic in the reply.
- The responder's reply path allocates per piece: `strings.Builder`s, a
  `chan string`, `[]byte(p)`, `speakable(string(p))` (`cascade.go:1136,
  1184, 1205, 1234`).
- `Add(role, text)` is a method on `*Cascade` only; multi-party presence
  and chat cannot be given to a `speech.Duplex`.
- Interruptions need a `speech.ZeroShot` lane, which is why every Qwen3
  model opens its questions workspace (section 2).

### The moments design

A **moment** is a time at which the agent is asked what to say. There are
four kinds, and they share one path (a job for the responder) and one
answer grammar:

| Moment | Trigger | What the model is asked |
| --- | --- | --- |
| Utterance | a speaker's turn ends (as today: prepared at the pause, released when the turn is over) | the words, as a user message |
| Note | `Note(text)`: a chat message, someone joining or leaving, a tool result | the note, as a system message |
| Planned | the model ended a reply with `<silent 30s>` | a system note: "The 30 seconds you asked for have passed." |
| Idle | `Config.Idle` of quiet with nothing heard or noted (zero: never) | a system note: "Nothing has happened for a minute." |

The model's reply is speech, or `<silent>`, and either may end with
`<silent Ns>` to plan the next moment:

```
Sure, I'll remind you. <silent 30s>      → says "Sure, I'll remind you.", asks again in 30 s
<silent>                                  → nothing, until the next moment
<silent 2m>                               → nothing, and asks again in two minutes
Lisbon.                                   → says "Lisbon."
```

Silence "until I say hi" needs no wake classifier: the instruction and
the model's own `<silent>` replies are in its context, so at the next
utterance it decides, in context, whether "hi" happened. The cost of a
silent moment is a prefill of the note plus one or two tokens, since the
`<silent` scaffold is drafted and verified in one pass like a tool call's
(`qwen3/chat.go:444-470`): about 50 ms on Qwen3-8B. Moments that arrive
while a reply is being spoken queue and are folded into one job after it.

Addressing in a meeting is the same decision: the prompt says to answer
only what is meant for it, and the model replies `<silent>` to the rest.
The words still join the conversation, so "what did Ana say?" works.

Attribution: the application reports the current speaker
(`Cascade.Speaker(name)`, from the SFU's dominant-speaker or audio-level
events); when an utterance's speaker differs from the last one, the
cascade adds "Ana is speaking." as a system note before the message, and
the message is the words alone. The echo guard remains as a reply filter
(a reply that begins by repeating the message it answers is cut), but the
framing no longer invites it.

### Proposed API

```go
package duplex

// Config shapes a Cascade. Every field is optional.
type Config struct {
	Prompt string

	// Lanes. New opens those left nil from its models and owns them.
	Transcriber speech.Transcriber
	Turns       speech.TurnDetector // when the transcriber does not judge turns
	Session     chat.Session
	Voice       speech.Synthesizer

	// Options.
	Listen speech.Options
	Speak  speech.SpeakOptions
	Reply  chat.Options
	Tools  []chat.Tool

	// Idle is how long the agent may hear nothing before it is asked
	// whether it has something to say; zero never asks.
	Idle time.Duration

	// Judges, each a zero-shot question to the models' ZeroShot lane when
	// one provides it: Interruptions (speech over the agent's voice: stop,
	// or go on), Quiet (do the words ask for quiet? a silence chosen
	// otherwise is overruled), and Wake (do the words end the silence the
	// model named?). They are small, in-grammar guards around the model's
	// own choice of speech or silence; see the status note at the top for
	// what Qwen3-8B needed.
	Interruptions, Quiet, Wake speech.TextClassifier

	// Observer sees what is heard and said, each reply's stages, and
	// errors, from worker goroutines. Nil observes nothing.
	Observer Observer
}

// Observer receives the cascade's events. Embed Base to implement some.
type Observer interface {
	// Heard reports the speaker's words: partial while spoken, then final.
	Heard(speaker string, text []byte, final bool)
	// Said reports the agent's reply as it grows: text so far, how many of
	// its bytes the voice has spoken, and whether it is finished or cut.
	Said(text []byte, voiced int, final bool)
	// Stage reports a reply's progress since the speaker's pause.
	Stage(s Stage, elapsed time.Duration)
	Error(err error)
}
type Base struct{} // no-op Observer

type Stage uint8
const (
	Transcribed Stage = iota + 1
	Judged     // the words judged finished or not
	FirstText
	FirstAudio
	TurnOver   // the turn found over, by ear or by the recognizer
	Silent     // the model chose silence
	Planned    // ... and asked to be asked again
	Called     // a tool ran
	Dropped    // a reply superseded before it was heard
)
func (s Stage) String() string

type Cascade struct{ /* unexported */ }
func New(cfg Config, models ...*gophonic.Model) (*Cascade, error)

func (c *Cascade) Rates() (in, out int)
func (c *Cascade) Frame() (in, out int)
func (c *Cascade) Step(in, out []float32) (speech.DuplexState, error)
func (c *Cascade) Say(text string) error
func (c *Cascade) Note(text string) error
// Speaker names who is speaking now, for the conversation's attribution.
// It allocates nothing and may be called every frame.
func (c *Cascade) Speaker(name string)
func (c *Cascade) Buffered() time.Duration
func (c *Cascade) Close() error

// MomentsPrompt is appended to the system prompt: the answer grammar.
const MomentsPrompt = `You are asked what to say at moments: after someone speaks, when something is noted, and when you asked to be. Reply with your words, or with <silent> to say nothing. End a reply with <silent 30s> (any number of seconds or minutes) to be asked again after that time, as when you must wait for something or remind someone. Never repeat or read back what someone said.`
```

### Example: build an agent, add a tool

```go
agent, err := duplex.New(duplex.Config{
	Prompt: "You are Gopher, in a video call.",
	Listen: speech.Options{Languages: speech.Languages(speech.English, speech.Portuguese)},
	Tools:  []chat.Tool{chat.Func("now", "The current time.", now), search},
	Idle:   45 * time.Second,
	Observer: captions,
}, asr, llm, tts)
defer agent.Close()

for range tick.C { // every 20 ms
	state, _ := agent.Step(micFrame, speakerFrame)
	if newSpeaker { agent.Speaker(name) }
}
agent.Note("Ana joined the call.")
agent.Note("Ana wrote in the call's chat: let's meet at 3")
```

### Requirements

- Zero-alloc: `Step` is unchanged (tested). The responder keeps one
  `[]byte` reply buffer per cascade and writes it through `io.Writer`
  into the synthesizer; captions come from `Voiced()`; stages are an enum;
  `Speaker` stores a string. Utterance-level work (transcribing, adding to
  the session) uses lane buffers and `unsafe.String` views; the tests add
  `AllocsPerRun` on a scripted reply with fake lanes.
- Perf: the pause-prepare-release pipeline, scribe passes with `Partial`,
  speculative `Finished` prefill, and drafted `<silent` scaffolds are kept;
  planned and idle moments are one timer, so nothing polls.
- Extensibility: `Tools` are `chat.Tool`; a native speech-to-speech model
  implements `speech.Duplex` directly and everything above `Step`
  (mixer, captions, scenarios) works unchanged.

## 6. Package `scenario`: the testing harness

Any `speech.Duplex` is driven end to end by a text script with real audio:
the user's lines are synthesized with a `speech.Synthesizer`, the agent's
audio is transcribed with a `speech.Transcriber`, and a `chat.Generator`
judges semantic claims. It runs in real time (the cascade's turn taking is
wall-clock), so a scenario takes as long as the conversation.

### Script format

One step per line, `who: what`. Blank lines and `#` comments are skipped.

```
# examples/gopher/scenarios/stop-talking.txt
user: Stop talking until I say hi.
gopher: silent
user: What is the capital of Portugal?
gopher: silent
user: Hi!
gopher: speaks
user: What is the capital of Portugal?
gopher: says that the capital is Lisbon
```

| Line | Meaning |
| --- | --- |
| `user: text` | speaks text to the agent (the `Speak` voice), then waits for the agent's reply or silence |
| `user(pt): text` | speaks in Portuguese |
| `note: text` | `agent.Note(text)` |
| `chat: Ana: text` | `Note("Ana wrote in the call's chat: text")` |
| `join: Ana` / `leave: Ana` | `Note("Ana joined the call.")`, and `Speaker` for the next lines |
| `wait: 20s` | lets time pass |
| `gopher: silent` | nothing spoken for `Quiet` (default 6 s) after the last step |
| `gopher: silent for 20s` | nothing spoken for 20 s |
| `gopher: speaks` | speech starts within `Answer` (default 6 s) |
| `gopher: speaks within 40s` | speech starts within 40 s |
| `gopher: says claim` | speaks, and the judge agrees the transcript says the claim |
| `gopher(pt): says claim` | transcribed as Portuguese |
| `gopher: does not repeat the user` | the transcript does not contain the user's last line |
| `gopher: captions follow the voice` | `Said` was observed with `voiced` growing before `final` |

Any name that is not a directive is the agent; scripts read as dialogue.

### Proposed API

```go
package scenario

type Config struct {
	Agent   speech.Duplex
	Voice   speech.Synthesizer // speaks the user's lines
	Speak   speech.SpeakOptions
	Ears    speech.Transcriber // hears the agent
	Judge   chat.Generator     // judges "says"; nil fails such steps
	Answer  time.Duration      // how long an answer may take; default 6 s
	Quiet   time.Duration      // how long silence is watched; default 6 s
	Observe func(o duplex.Observer) // lets the harness observe captions; optional
	Log     func(format string, args ...any)
}

// Run drives cfg.Agent through script and returns each step's outcome.
func Run(ctx context.Context, cfg Config, script string) (Report, error)

// Test runs every scenario file matching glob as a subtest.
func Test(t *testing.T, cfg Config, glob string)

type Report struct{ Steps []Step }
type Step struct {
	Line   string
	Heard  string        // what the agent said, transcribed
	After  time.Duration // from the end of the user's line to the agent's first audio
	OK     bool
	Reason string
}
```

The harness's clock: a goroutine steps the agent every 20 ms, feeds the
synthesized user audio frame by frame (silence otherwise), collects the
output frames, and detects speech by state and energy. The agent's audio
is resampled to 16 kHz for `Ears`.

### Example: a scenario test for Gopher

```go
func TestScenarios(t *testing.T) {
	asr, llm, tts := openModels(t) // skips without models
	agent, _ := duplex.New(config("ryan", "", nil), asr, llm, tts)
	ears, _ := gophonic.Lane[speech.Transcriber](asr)
	voice, _ := gophonic.Lane[speech.Synthesizer](tts)
	judge, _ := gophonic.Lane[chat.Generator](llm)
	scenario.Test(t, scenario.Config{Agent: agent, Voice: voice, Ears: ears, Judge: judge,
		Speak: speech.SpeakOptions{Voice: "serena"}}, "scenarios/*.txt")
}
```

The files shipped with Gopher, one per behaviour the user tested by voice:
`hear-me.txt`, `stop-talking.txt`, `no-echo.txt`, `jokes-captions.txt`,
`chat-context.txt`, `search.txt`, `time.txt`, `portuguese.txt`,
`reminder.txt` (expected to fail until moments land, marked `# expect:
fail` so the harness reports it without failing the suite).

## 7. Model packages

### qwen3

Problems: two objects over one set of weights (`Model` for embeddings and
questions, `Chat` for generation, `Chat.Questions` to share,
`qwen3/chat.go:86-109`); `Options.Weights` versus everyone else's `Format`;
the whole engine re-exported (`qwen3/lowlevel.go`), which ties
`internal/qwen3lm`'s shape to a public API just as it gains experts and
recurrent state; `IsModelDir` accepts only `model_type == "qwen3"`
(`qwen3/model.go:108-117`) so the MoE (`qwen3_moe`) and 3.5/3.6 hybrids
(`qwen3_next`-style types) need a public change to be recognized.

```go
package qwen3

type Options struct {
	Format       string // gophonic.Format*; empty picks the fastest faithful one
	Threads      int
	CacheEntries int // exact embedding cache; 0 default, <0 off
	PrefixTokens int // prefix store for long inputs; 0 default, <0 off
}

// Model is one loaded Qwen3-family checkpoint (dense, MoE, or hybrid),
// serving conversations, questions, and embeddings from one copy of its
// weights. Sessions and questions of one Model share its workspace and
// are serialized; open a second Model for a second concurrent lane.
type Model struct{ /* unexported */ }

func IsModelDir(dir string) bool // qwen3, qwen3_moe, and the 3.5/3.6 hybrid types
func Open(path string, opts Options) (*Model, error)
func (m *Model) Close() error
func (m *Model) Family() string // "qwen3", "qwen3-moe", "qwen3.6"; for logs

// chat.Generator
func (m *Model) NewSession(system string, tools ...chat.ToolSpec) (chat.Session, error)
// speech.ZeroShot; the letter head loads on first use
func (m *Model) Classifier(question string, labels []string) (speech.TextClassifier, error)
func (m *Model) Question(question string, options []string) (*Question, error)
func (m *Model) NewContext(maxTokens int) (*Context, error)
func (m *Model) ContextQuestion(question string, options []string) (*ContextQuestion, error)
// Embeddings
func (m *Model) Embed(ctx context.Context, texts []string, dst [][]float32) error
func (m *Model) EmbedTokensInto(ctx context.Context, ids [][]int, dst [][]float32) error
func (m *Model) Width() int

// Question, Stream, Context, ContextQuestion: unchanged.
// Tokenizer stays public (LoadTokenizer, EncodeInto, DecodeAppend, Piece):
// applications tokenize. Weights, Evaluator, Workspace, PrefixKV, Embeds,
// LoadWeights, NewEvaluator, and QuantizeGPTQ leave the public API;
// cmd/qwen3-gptq calls the internal package directly.
```

Zero-alloc and perf as today (`Choose`, `Update`, `Reply`, `Embed` have
`AllocsPerRun` tests). The single workspace removes 600 MB of prefix
store and a second worker pool from every Qwen3 open; questions allocate
their own prefix store on first use.

### qwen3asr

Problems: `Release` (`model.go:259`); `Languages()` allocates
(`model.go:247`) and returns names; `NewTranscriber(m, workers int)` takes
a bare int while every other lane takes options; language options compared
as strings (section 3).

```go
package qwen3asr

type Options struct{ Format string } // FormatF16, FormatGPU (= gpu-q8), or empty
func IsModelDir(dir string) bool
func Load(dir string, opts Options) (*Model, error)
func (m *Model) Languages() speech.LanguageSet
func (m *Model) JudgesTurns() bool // a turn head exists for this checkpoint
func (m *Model) Close() error
func NewTranscriber(m *Model, opts LaneOptions) (*Transcriber, error)
type LaneOptions struct{ Threads int }
// Transcriber: speech.Transcriber and speech.BatchTranscriber (later).
```

The script mask and allowed-language list are keyed by `LanguageSet`
(one integer comparison). Streaming stays `Options.Partial`; the encoder's
windowed attention makes an incremental encoder (encode only the open
8-second window) a later internal optimization behind the same call.

### qwen3tts

Problems: `Synthesizer.Greedy bool` is an exported test hook
(`synth.go:52`); `Speakers()`/`Languages()` allocate (`model.go:302,312`);
`Release`.

```go
package qwen3tts

type Options struct{ Format string; Threads int }
func Load(dir string, opts Options) (*Model, error)
func (m *Model) Voices() []string           // built once at Load
func (m *Model) Languages() speech.LanguageSet
func (m *Model) Close() error
func NewSynthesizer(m *Model, opts LaneOptions) (*Synthesizer, error)
type LaneOptions struct{ Greedy bool }      // deterministic decoding, for tests and fixtures
// Synthesizer: speech.Synthesizer (Begin/Write/End/Read/Voiced).
const SampleRate, FrameSamples // unchanged
```

`Voiced` comes from the talker's text feed: each generated frame records
the byte offset of the last text token fed before it; `Read` reports the
offset of the last frame it handed out. `SpeakOptions.Style` is the
official instruct, projected as text rows before the voice prompt and
cached with it (commit 6de5707), unchanged by this redesign.

### whisper

Problems: `NewTranscriber` and `NewTranscriberWithWorkers`
(`whisper/transcriber.go:60,67`); a large graph-level surface (`Dims`,
`DecoderScratch`, `GreedyPolicy`, `Tokenizer`, four `Transcribe*Into`
variants, twenty error variables) that no application uses.

```go
package whisper

func Load(path string) (*Model, error)
func (m *Model) Close() error
func NewTranscriber(m *Model, opts LaneOptions) (*Transcriber, error)
type LaneOptions struct{ Threads int }
// Transcriber: speech.Transcriber, plus the caller-buffer variants
// TranscribeInto, TranscribeSegmentsInto, TranscribeWordsInto.
```

The encoder, decoder, tokenizer, and policy types move to
`whisper/internal/graph` (tests and tools follow); `FeaturesInto` and
`FullFeaturesInto` stay for feature-level callers.

### smartturn and tinymel

Problems: `NewSession(model)` versus `NewSession(model, helpers)`;
`Session` versus `Transcriber`/`Synthesizer` naming; `PredictInto`.

```go
package smartturn // and tinymel, identically

func Load(path string) (*Model, error)
func ReadWeights(r io.Reader) (*Model, error)
func NewDetector(m *Model, opts LaneOptions) (*Detector, error)
type LaneOptions struct{ Threads int }
// Detector: speech.TurnDetector (Predict) and speech.AudioClassifier.
// Workspace and Model.PredictInto/PredictFeaturesInto stay for callers
// that own preprocessing.
```

## 8. `internal/qwen3lm`: where its shape leaks

- Through `qwen3`'s aliases (section 7): removed from the public API.
- Through `qwen3asr` and `qwen3tts`: they consume `Evaluator`,
  `PrefixKV`, `Embeds`, and `HiddenLastExtendEmbedInto`; that is
  module-internal and unchanged.
- Through `Config` (`Hidden, Layers, Heads, KVHeads, HeadDim,
  Intermediate, Vocab, MaxPositions`): the turn head keys on
  `Hidden/Layers/Vocab` (`qwen3asr/turn.go:45`); fine.

Recommendations for the MoE and hybrid work (not done here, owned by the
engine branch): rename `PrefixKV` to `State` (keys and values for
attention layers, a conv ring and a 128×128 matrix per head for Gated
DeltaNet layers, snapshotted every 64 tokens so `CommonPrefix`/`CopyPrefix`
still give rollback); keep `HiddenLastExtendInto` and `LogitsInto` as the
two entry points every lane uses, so `Session`, `Transcriber`, and
`Synthesizer` do not change when the core does; add a grouped-GEMV
expert kernel behind the same `Evaluator`. The public API above does not
name any of these.

## 9. `httpserver`, `cmd/gophonic`, `cmd/gophonic-server`

Problems: `httpserver.Model{Name, Path, Format}` duplicates what a
`gophonic.Model` should know (`server.go:36-43`); no chat or speech
endpoints, so the server cannot serve the two lanes the agent is built
on; the CLI's `-question/-labels` mode is the only text task.

```go
package httpserver

type Config struct {
	Pool       *gophonic.Pool
	Models     []Model // Name and Path; Format from gophonic.Detect
	Workers    int
	MaxSeconds int
}
```

Endpoints added, each on the new interfaces without adapters:

- `POST /v1/chat/completions` (`chat.Generator`): messages become
  `Session.Add`, tools become `ToolSpec`s, `stream: true` writes SSE
  through an `io.Writer` that frames each piece; `tool_calls` come from
  `Calls()`. A request slot holds one session; sessions are not kept
  between requests (stateless API), but a `conversation` extension keyed
  by the client's id reuses a session's prefix.
- `POST /v1/audio/speech` (`speech.Synthesizer`): `Begin`, `Write`,
  `End`, then `Read` into a WAV or raw PCM stream; `response_format:
  pcm` streams frames as they are decoded.
- Transcription uses `BatchTranscriber` when a model provides it and the
  queue holds several requests for the same model.

CLI: `gophonic -model TTS -voice ryan -o out.wav "text"` speaks;
`gophonic -model LLM -chat` reads a conversation from stdin. Both are
thin, since the lanes stream.

## 10. `examples/gopher`

Problems: `Heard` builds "name said: " (`main.go:111-116`); `OnText`
distinguishes roles by a `chat.Role` and infers the speaker from the
mixer's loudest track at caption time (`main.go:120-139`), which can name
the wrong person when two talk; presence and chat go through `agent.Add`
with `chat.System` (`main.go:189, 247`), which the interface lacks.

With the redesign, `main.go` implements `duplex.Observer` on a `captions`
struct (`Heard(speaker, text, final)` posts closed captions with the
speaker the cascade attributed; `Said(text, voiced, final)` posts the
reply as the voice reaches it), calls `agent.Speaker(name)` from the
mixer's per-frame loudest track, and `agent.Note` for chat and presence.
`config()` sets `Idle: 45 * time.Second` and keeps the tools. `scenarios/`
holds the nine scripts and `scenario_test.go` runs them.

## 11. How what is coming fits

| Coming | Where the API makes it natural |
| --- | --- |
| Qwen3-30B-A3B (MoE), Qwen3.6-35B-A3B (GDN + MoE, 248k vocab) | `qwen3.IsModelDir` accepts the family; `qwen3.Open` picks the core by `model_type`; `Format` is one vocabulary; `chat.Options.Presence` and top-k on a 248k head are engine details behind `Reply`; nothing public names KV, experts, or heads |
| Time-aware duplex ("moments") | section 5: `Note`, `Idle`, `<silent 30s>`, one job path, one grammar |
| Scenario harness | section 6: `scenario.Run/Test` on any `speech.Duplex` |
| Tools: typed Go functions, MCP, drafted calls | `chat.Tool`, `chat.Func`, `chat.Answer`; an `mcp` package returns `[]chat.Tool`; the drafted scaffolds stay inside `Session.Reply` |
| Multi-party: attribution, chat and presence as context, captions that follow the voice | `Cascade.Speaker`, `Duplex.Note`, `Observer.Heard(speaker, ...)`, `Observer.Said(text, voiced, ...)` fed by `Synthesizer.Voiced` |
| Turn taking by the recognizer, streaming with `Partial`, language limits with a script mask | `Options.Turn`, `Options.Partial`, `Options.Languages` as a `LanguageSet` keyed mask |
| Serving: OpenAI-compatible server, multi-model pool | section 9; `Pool` unchanged; `BatchTranscriber` for throughput |

## 12. Migration plan

Ordered by leverage over risk. Each step is one PR-sized commit that
builds, vets, and passes the tests of the packages it touches
(`GOEXPERIMENT=simd go test ./pkg...`, plus `examples/gopher` in its own
module), with every existing `AllocsPerRun` test kept.

1. **Chat replies into `io.Writer`; tools move to `chat`.**
   `chat/chat.go` (Reply signature, `Tool`, `Func`, `Specs`, `Answer`,
   `Options.Presence`), `chat/tool.go` (from `duplex/tool.go`),
   `chat/tool_test.go`, `qwen3/chat.go` (emit through `w`, presence
   penalty in the sampler), `qwen3/chat_test.go`, `qwen3/tools_test.go`,
   `duplex/cascade.go` (sink struct, `chat.Answer`), `duplex/cascade_test.go`,
   `examples/gopher/tools.go`, `tools_test.go`, `docs/api.md`.
   Risk: low; mechanical.
2. **Duplex interface: `Step(in, out)`, `Note`, `Observer`, typed
   stages; drop `Heard`, `Wake`, `SilencePrompt`, `maxSilence`, `Silent()`.**
   `speech/duplex.go`, `duplex/cascade.go`, `duplex/observer.go`,
   `duplex/cascade_test.go`, `examples/gopher/main.go`, `agent.go`,
   `README.md`. The moment grammar lands here with the utterance and note
   moments; planned and idle moments in step 3. Risk: medium; the cascade
   tests with fake lanes cover the paths.
3. **Moments: planned (`<silent 30s>`) and idle; `Speaker` attribution.**
   `duplex/cascade.go` (one timer, `Speaker`, speaker-change notes, reply
   parsing), `duplex/cascade_test.go` (a session that answers
   `<silent 1s>` and is asked again; a note that gets answered; idle),
   `examples/gopher/main.go` (`Speaker` from the mixer). Risk: medium;
   behaviour is verified by scenarios in step 4.
4. **Scenario harness and Gopher's scenario files.** `scenario/scenario.go`,
   `scenario/script.go`, `scenario/scenario_test.go` (a scripted fake
   duplex), `examples/gopher/scenarios/*.txt`,
   `examples/gopher/scenario_test.go`. Risk: low for the harness; the
   scenario results are the measure of steps 2 and 3.
5. **Synthesizer as push-text, pull-audio with `Voiced`.**
   `speech/synthesis.go`, `qwen3tts/synth.go` (a lane goroutine, text and
   audio rings, byte offsets per frame), `qwen3tts/qwen3tts_test.go`
   (allocs, `Voiced` monotone, `Begin` drops the previous utterance),
   `duplex/cascade.go` (replace the channel and closures; captions from
   `Voiced`), `internal/zz*` untouched (scratch). Risk: medium; the
   synthesizer's reference tests pin the output.
6. **Root options and naming: `Options.Format`, `Model.Path`, checked
   `Provides`, `Close` everywhere, lane option structs, `Predict`.**
   `gophonic.go`, `formats.go`, `qwen3asr/model.go`, `transcriber.go`,
   `qwen3tts/model.go`, `synth.go`, `whisper/transcriber.go`,
   `smartturn/session.go` → `detector.go`, `tinymel/session.go` →
   `detector.go`, `speech/speech.go`, `internal/httpserver/server.go`,
   `cmd/gophonic/main.go`, `duplex/cascade.go`, all tests, `docs/api.md`.
   Risk: low; compiler-driven.
7. **`speech.Language` and `LanguageSet`.** `speech/language.go`,
   `speech/speech.go`, `speech/synthesis.go`, `qwen3asr/transcriber.go`
   (mask keyed by set), `qwen3asr/scripts.go`, `qwen3asr/model.go`,
   `qwen3asr/languages_test.go`, `qwen3tts/synth.go`, `whisper/speech.go`,
   `internal/httpserver/server.go`, `upload.go`, `cmd/gophonic/main.go`,
   `duplex/cascade.go`, `examples/gopher/agent.go`, `main.go`. Risk: low;
   compiler-driven, and the language tests pin behaviour.
8. **One `qwen3.Model`; the engine leaves the public API; lazy questions.**
   `qwen3/model.go`, `chat.go`, `choose.go`, `context.go`, `lowlevel.go`
   (deleted), `cache_test.go` and `official_test.go` (use the internal
   package through an internal test helper), `formats.go`,
   `cmd/qwen3-gptq/main.go`. Coordinate with the engine branch: it must
   not depend on `qwen3.NewEvaluator`. Risk: medium; the official tests
   pin every path.
9. **Root turn frontend to `speech.TurnFeatures`.** `features.go` →
   `speech/turnfeatures.go`, `features_test.go`, `external_test.go`,
   `docs/api.md`, `docs/architecture.md`. Risk: low.
10. **Server endpoints: chat completions (SSE) and speech; batch
    transcription.** `internal/httpserver/chat.go`, `speech.go`,
    `server.go`, `server_test.go`, `docs/server.md`; `qwen3asr`
    `TranscribeBatch`. Risk: medium; new code, warm-alloc tests as today.
11. **Whisper's graph surface moves internal.** `whisper/*.go` →
    `whisper/internal/graph`, tests follow. Risk: low, large diff; last
    because nothing depends on it.
12. **`mcp` package: MCP servers as `[]chat.Tool`.** New package; stdio
    JSON-RPC with `vibejson`. Risk: low; additive.

## 13. Open questions and risks

- **Does the model reliably answer `<silent>` when not addressed?**
  Qwen3-8B answered a `stay_quiet` tool call when asked to be quiet
  (`qwen3/tools_test.go`), and chose `<silent until ...>` in today's
  cascade; the moments grammar is closer to plain text than a tool call.
  Recommendation: keep the drafted `<silent` scaffold so a silent moment
  costs one pass, and measure with `stop-talking.txt` and a meeting
  scenario (`join: Ana`, `join: Bob`, Ana talks to Bob) on Qwen3-8B before
  Qwen3.6; if 8B over-answers, add a one-line addressing rule to the
  prompt rather than a classifier.
- **Reminders need the model to compute durations.** "Remind me in thirty
  seconds" must become `<silent 30s>`; small models may write `<silent
  30>` or "thirty". Recommendation: accept `Ns`, `Nm`, `N` (seconds), and
  number words for one to sixty; `reminder.txt` tells us the rest.
- **Idle moments cost tokens and can make the agent chatty.**
  Recommendation: default `Idle` to zero; Gopher sets 45 s and its prompt
  says idle notes are usually answered with `<silent>`.
- **Real-time scenarios are slow** (a reminder scenario is 40 s).
  Recommendation: accept it for the Gopher suite (skipped without
  models); a `Clock` in the cascade for faster-than-real-time runs is a
  later, isolated change, since the cascade's timers are already in one
  place after step 3.
- **`Synthesizer.Write` never blocks on playback,** so a language model
  can run a whole reply ahead of the voice and compete for the GPU with
  it. Recommendation: the cascade paces `Write` against `Buffered()`
  (two pieces ahead until the first audio, as today), a policy in the
  cascade, not the lane.
- **`LanguageSet` as `uint64` holds 63 languages.** Qwen3-ASR has 30 (its
  22 Chinese dialects are not in the table); multilingual Whisper would
  add 99. Recommendation: `uint64` now with a static check; widen to
  `[2]uint64` if a multilingual Whisper lands. The API does not change.
- **Removing `qwen3`'s engine aliases** could break the engine branch's
  tests if they use `qwen3.NewEvaluator`. Recommendation: step 8 after the
  MoE work merges, with a grep on both branches first.
- **The echo guard is kept**, though the moments design and note-based
  attribution should make it idle. Recommendation: `no-echo.txt` reports
  how often it fires (`Stage` `Dropped` with a reason); remove it when the
  count is zero on Qwen3.6.
- **Continuous batching for chat** is not in this plan; the API allows it.
  Recommendation: build it for the server when the MoE core lands, since
  MoE decode is where batching pays most.
