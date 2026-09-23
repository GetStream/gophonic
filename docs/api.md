# Embedding GoFloor

Import `github.com/GetStream/gofloor`. Load weights once, then create one
workspace for every concurrent prediction lane. A model can be shared across
goroutines because inference reads its weights without modifying them.

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

`ExtractWhisperFeatures16k(pcm, dst, workspace)` exposes the shared frontend for
mono 16 kHz audio. Provide a `dst` slice with exactly `80*800` elements and a
`Workspace`. The standalone extractor uses its serial frontend. TinyMelNet's
audio prediction path can parallelize preprocessing using its existing helpers.

## Ownership, lifetime, and allocations

A workspace is owned by one caller at a time. Do not overlap predictions on the
same workspace, or call `Close` concurrently with a prediction. Helpers divide
the current prediction internally; that does not make the workspace safe for
multiple external callers.

`Close` is idempotent when called serially and waits for persistent helpers to
exit. A closed workspace cannot be reused. Calling a prediction with a nil
model or a nil/closed workspace returns an error.

The **zero-allocation contract** covers successful repeated predictions with
loaded weights, a reusable workspace, and a warmed sample-rate configuration.
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

The CLI reads and decodes a file, loads weights, creates a workspace, makes one
prediction, and prints JSON. It is a demonstration and integration utility;
warm library benchmarks exclude these per-process setup and I/O costs.
