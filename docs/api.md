# Embedding GoFloor

Import `github.com/GetStream/gofloor`. Load weights once, then create one session
or workspace for every concurrent prediction lane. Built-in models can be
shared across goroutines because inference reads their weights without
modifying them.

## Sessions across architectures

Applications can depend on one small interface:

```go
type AudioSession interface {
	PredictInto(pcm []float32, sampleRate, channels int) (Prediction, error)
	Close() error
}
```

`NewSmartTurnSession(model)` returns `(*SmartTurnSession, error)`;
`NewTinyMelSession(model, helpers)` returns `(*TinyMelSession, error)`. Each owns
one workspace and delegates to the model's existing prediction path. Load the
model separately, then create as many independent sessions as needed:

```go
concrete, err := gofloor.NewTinyMelSession(model, 3)
if err != nil {
	return err
}
var detector gofloor.AudioSession = concrete
defer detector.Close()

prediction, err := detector.PredictInto(pcm, 16000, 1)
```

The common contract is PCM-only. An architecture owns its weights, frontend,
sample-rate support, input window, and completion threshold; the interface does
not impose a feature tensor shape. `Prediction.Probability` means probability
of a completed turn, and `Complete` applies that backend's decision rule.

Built-in sessions accept the PCM formats described below. Their constructors
reject a nil model with `ErrNilModel`; a nil or closed built-in session returns
`ErrSessionClosed` from prediction. Serial `Close` calls are idempotent. Keep
prediction and close operations exclusive to the session's owner.

The following direct APIs expose each built-in model's workspace and feature
entry point when those are more useful to the application.

## Choose a model and workspace

| Operation | Smart Turn v3.2 | TinyMelNet |
| --- | --- | --- |
| Read a file | `Load(path)` | `LoadTinyMel(path)` |
| Read an `io.Reader` | `ReadWeights(reader)` | `ReadTinyMelWeights(reader)` |
| Construct scratch | `NewWorkspace()` | `NewTinyMelWorkspace()` |
| Request helpers | Uses `GOMAXPROCS-1` at construction | `NewTinyMelWorkspaceWithWorkers(n)` |
| Release helpers | `workspace.Close()` | `workspace.Close()` |

TinyMelNet defaults to serial execution. Its worker constructor requests `n`
helper goroutines in addition to the calling goroutine, capped at seven and at
`GOMAXPROCS-1`. Negative requests become zero. Set the process CPU budget before
constructing workspaces; a constructor does not reserve OS threads or pin work
to physical cores.

For example, load and initialize at startup:

```go
model, err := gofloor.LoadTinyMel("tinymel.gofloor")
if err != nil {
	return err
}
workspace := gofloor.NewTinyMelWorkspaceWithWorkers(3)
defer workspace.Close()
```

Each pause decision then reuses both objects:

```go
prediction, err := model.PredictInto(pcm, 48000, 2, workspace)
if err != nil {
	return err
}
if prediction.Complete {
	// Let the application's dialogue policy decide whether to respond.
}
```

`pcm` in this example is interleaved stereo audio: left, right, left, right.
`Prediction` is a small value with `Probability float32` and `Complete bool`;
the result does not borrow workspace storage.

## Audio and feature contracts

Both model types provide these entry points:

| Method | Input |
| --- | --- |
| `PredictInto(pcm, sampleRate, channels, workspace)` | Interleaved `[]float32`, mono or stereo, 8–96 kHz |
| `PredictMono16kInto(pcm, workspace)` | Mono 16 kHz `[]float32` |
| `PredictFeaturesInto(features, workspace)` | Exactly 64,000 float32 values in row-major `[80,800]` order |

Use PCM amplitude conventions with full scale around `[-1,1]`. Audio must be
nonempty and contain complete frames. Consumed samples must be finite. Stereo
channels are averaged; other sample rates are resampled to 16 kHz.

The audio path retains the most recent eight seconds and left-pads shorter
input with zeros. It then applies waveform normalization and Whisper feature
extraction. It reads the caller's PCM and writes normalized audio into workspace
scratch, so the caller's samples remain unchanged. Keep input slices immutable
for the duration of the call.

The feature path expects the completed, normalized Whisper log-mel tensor. Its
index is `features[melBand*800+frame]`; this entry point performs no resampling,
padding, or feature normalization. It reads the feature slice without retaining
it after the call.

The standalone frontend allocates only feature-extraction scratch:

```go
frontend := gofloor.NewWhisperFeatureWorkspace()
defer frontend.Close()
features := make([]float32, 80*800)

// Reuse frontend and features for each prediction.
err := gofloor.ExtractWhisperFeaturesInto(pcm, 48000, 2, features, frontend)
```

`ExtractWhisperFeaturesInto` accepts mono/stereo PCM at 8–96 kHz and writes the
normalized, row-major `[80,800]` tensor. It preserves the built-in eight-second
windowing rules, runs the serial frontend, and needs neither built-in model's
inference buffers nor a worker pool. Its scratch and destination belong to one
caller at a time. Once the sample rate is warm, successful extraction does not
allocate.

`ExtractWhisperFeatures16k(pcm, dst, workspace)` remains available for callers
already holding a Smart Turn `Workspace`. TinyMelNet's audio prediction path
uses its existing helpers to parallelize preprocessing.

## Add an audio backend

Implement `AudioSession` in your backend package and pass it directly to the
application. A new architecture provides its own loader, model math, reusable
scratch, and threshold. No registration or changes to GoFloor's model dispatch
are needed.

This adapter sketch assumes your package already defines `LoadModel`,
`NewScratch`, and a `Model.PredictPCMInto` method returning a probability. Those
names represent your backend's implementation, not GoFloor APIs:

```go
type Session struct {
	model     *Model
	scratch   *Scratch
	threshold float32
	closed    bool
}

var _ gofloor.AudioSession = (*Session)(nil)

func OpenSession(path string, threshold float32) (*Session, error) {
	model, err := LoadModel(path)
	if err != nil {
		return nil, err
	}
	return &Session{model: model, scratch: NewScratch(), threshold: threshold}, nil
}

func (s *Session) PredictInto(pcm []float32, rate, channels int) (gofloor.Prediction, error) {
	if s == nil || s.closed {
		return gofloor.Prediction{}, gofloor.ErrSessionClosed
	}
	p, err := s.model.PredictPCMInto(pcm, rate, channels, s.scratch)
	if err != nil {
		return gofloor.Prediction{}, err
	}
	return gofloor.Prediction{Probability: p, Complete: p > s.threshold}, nil
}

func (s *Session) Close() error {
	if s != nil && !s.closed {
		s.closed = true
		s.scratch.Close()
	}
	return nil
}
```

An architecture trained on the same Whisper feature representation can hold a
`WhisperFeatureWorkspace` and an `80*800` feature buffer inside its own scratch.
Call `ExtractWhisperFeaturesInto`, then pass those features to its model. An
architecture with different preprocessing implements that frontend itself.

The interface performs no model loading, graph interpretation, feature
conversion, or allocation management on behalf of an external backend. Its
implementation must establish its own numerical and allocation guarantees.
The [external-package conformance test](../session_external_test.go) demonstrates
implementing the contract and reusing only the Whisper frontend; its test head
is a stand-in, not an additional turn-detector model.

## Ownership, lifetime, and allocations

A workspace is owned by one caller at a time. Do not overlap predictions on the
same workspace, or call `Close` concurrently with a prediction. Helpers divide
the current prediction internally; that does not make the workspace safe for
multiple external callers.

`Close` is idempotent when called serially and waits for persistent helpers to
exit. A closed workspace cannot be reused. Calling a prediction with a nil
model or a nil/closed workspace returns an error.

For built-in sessions and direct model calls, the **zero-allocation contract**
covers successful repeated predictions with loaded weights, reusable scratch,
and a warmed sample-rate configuration. Session construction allocates its
workspace once; adapting a prediction through `AudioSession` adds no per-call
allocation.
The 16 kHz path needs no resampler table. A non-16 kHz rate initializes its
coefficient table on first use; changing to a rate requiring greater table
capacity can allocate again. Load/setup, error formatting, file decoding, and
result serialization are outside the contract.

An application serving many independent conversations should benchmark its
total CPU budget. Seven helpers for every simultaneous conversation can
oversubscribe the process. Serial workspaces or smaller helper counts may
provide better throughput even when a larger pool lowers isolated latency.

## CLI input and output

```sh
./gofloor -model smart-turn-v3.2.gofloor speech.wav
./gofloor -tiny-model tinymel.gofloor -tiny-workers 3 speech.ogg
```

Exactly one model flag and one audio path are required. `-tiny-workers` accepts
0–7 and applies only to `-tiny-model`. CLI input formats are:

- WAV: RIFF/WAVE PCM at 8, 16, 24, or 32 bits; IEEE float at 32 or 64 bits;
  one or two channels at 8–96 kHz.
- Ogg Opus: `.ogg` or `.opus`, one or two channels, decoded by `gopus` to
  16 kHz with pre-skip and the final granule position applied.

The CLI selects the two built-in model loaders. External sessions are integrated
in Go applications. The CLI reads and decodes a file, loads weights, creates a
workspace, makes one prediction, and prints JSON. It is a demonstration and integration utility;
warm library benchmarks exclude these per-process setup and I/O costs.
