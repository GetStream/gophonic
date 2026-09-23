# gofloor

**Audio turn detection, written in Go.**

`gofloor` estimates whether a speaker has finished their turn from the last
eight seconds of audio. Embed it in a voice agent, reuse a workspace, and get a
probability without leaving your Go process.

- **CPU inference in Go.** Build with `CGO_ENABLED=0`; optional Go 1.27
  `simd/archsimd` accelerates FP32 work on ARM64 and AMD64, with additional
  TinyMelNet integer kernels on ARM64 NEON.
- **Zero allocations during warm prediction.** Models share immutable weights;
  each concurrent prediction owns its scratch and persistent workers.
- **Two explicit model choices.** Pipecat Smart Turn v3.2 FP32 and the smaller
  TinyMelNet INT8 graph have separate loaders and completion thresholds.
- **PCM in your application; WAV or Opus at the command line.** Ogg Opus decoding
  uses [`gopus`](https://github.com/thesyncim/gopus).

Inference, feature extraction, and resampling run in Go. Python is used once to
convert a published ONNX checkpoint into a weight bundle. The runtime implements
these two model graphs; arbitrary ONNX models are outside its current scope.

[Models](docs/models.md) · [Go API](docs/api.md) ·
[Architecture](docs/architecture.md) · [Benchmarks](docs/benchmarks.md) ·
[Validation](docs/validation.md)

## Quick start

You need Go 1.27 and Python 3 with NumPy and ONNX for the one-time conversion.
Run the following from a checkout of this repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy onnx

curl -fL \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx \
  -o smart-turn-v3.2-gpu.onnx
.venv/bin/python tools/onnx_to_gofloor.py \
  smart-turn-v3.2-gpu.onnx smart-turn-v3.2.gofloor

CGO_ENABLED=0 GOEXPERIMENT=simd go build -o gofloor ./cmd/gofloor
./gofloor -model smart-turn-v3.2.gofloor speech.wav
```

The `-gpu` suffix is the upstream FP32 checkpoint's filename. GoFloor executes
it on the CPU. The converter verifies the checkpoint's SHA-256 before reading
its weights. No model download occurs during prediction or tests.

The command writes one JSON object:

```json
{"probability":0.91,"complete":true}
```

This is an example result. Smart Turn uses `probability > 0.5` for `complete`.
Replace `speech.wav` with `speech.ogg` or `speech.opus` to decode Ogg Opus. Omit
`GOEXPERIMENT=simd` to build the scalar Go kernels.

## Choose a model

| Model | Runtime graph | Load / CLI | Completion threshold |
| --- | --- | --- | ---: |
| [Pipecat Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) | Whisper encoder, FP32 | `Load` / `-model` | `> 0.5` |
| [TinyMelNet](https://huggingface.co/deveshu/hinglish-turn-detector) | Quantized convolutions, bidirectional GRU | `LoadTinyMel` / `-tiny-model` | `> 0.57` |

TinyMelNet trades model quality for lower compute cost. Its published evaluation
focuses on English, Hindi, and Hinglish; model quality and weight licensing are
covered in [Models](docs/models.md). Choosing it is explicit:

```sh
curl -fL \
  https://huggingface.co/deveshu/hinglish-turn-detector/resolve/main/model_tinymel_int8.onnx \
  -o model_tinymel_int8.onnx
.venv/bin/python tools/tinymel_to_gofloor.py \
  model_tinymel_int8.onnx --bundle tinymel.gofloor

GOMAXPROCS=8 ./gofloor \
  -tiny-model tinymel.gofloor -tiny-workers 7 speech.wav
```

`-tiny-workers` counts helper goroutines; the caller also does work. The default
is serial. A request for seven helpers is capped at `GOMAXPROCS-1`.

## Use it from Go

Load a model at startup. Reuse one workspace for each concurrent prediction
lane; share the model between lanes. In the following excerpt, `pcm` is the
application's mono 16 kHz `[]float32` audio:

```go
import "github.com/GetStream/gofloor"

model, err := gofloor.LoadTinyMel("tinymel.gofloor")
if err != nil {
	return err
}

workspace := gofloor.NewTinyMelWorkspaceWithWorkers(7)
defer workspace.Close()

// Repeat this call whenever your VAD detects a pause.
prediction, err := model.PredictMono16kInto(pcm, workspace)
if err != nil {
	return err
}
// prediction.Probability is P(turn complete).
// prediction.Complete applies this model's threshold.
```

Use `NewTinyMelWorkspace()` for serial execution. For Smart Turn, use `Load`
with `NewWorkspace()`. Both models also accept interleaved mono/stereo PCM at
8–96 kHz through `PredictInto`, or normalized `[80,800]` log-mel features through
`PredictFeaturesInto`.

Short input is left-padded; long input keeps the most recent eight seconds.
Call from a VAD-gated pause decision and apply your application's turn policy
to the result. The library does not contain a streaming VAD or dialogue policy.

## Performance and correctness

On the development Apple M4 Max, Go 1.27 SIMD TinyMelNet measured **2.090 ms**
for the model and **3.553 ms** from mono 16 kHz PCM to a prediction, using seven
helpers at `GOMAXPROCS=8`. Both measured **0 B/op and 0 allocs/op**. These are
medians of three warm, 200-iteration run means; file decoding and setup are
excluded. See [benchmarks and the ONNX Runtime CPU comparison](docs/benchmarks.md)
for conditions and reproduction commands.

The allocation contract covers successful calls with a reused workspace and a
warmed sample-rate configuration. Model loading, workspace construction,
resampler growth, file decoding, and JSON output are outside that boundary.

Tests compare the frontend with saved Whisper features, model probabilities
with ONNX Runtime outputs, and SIMD kernels with scalar references. GoFloor is
under active development; these parity checks establish numerical behavior on
the covered fixtures, not application-level accuracy. See
[validation coverage and limits](docs/validation.md).

```sh
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 GOEXPERIMENT=simd go test ./...

GOFLOOR_TEST_MODEL=smart-turn-v3.2.gofloor \
GOFLOOR_TEST_TINYMEL_MODEL=tinymel.gofloor \
GOEXPERIMENT=simd go test ./...
```

## License

GoFloor code is [BSD-2-Clause](LICENSE). Model weights and `gopus` retain their
own licenses. Weights are downloaded separately; see
[model provenance and terms](docs/models.md#provenance-and-licenses).
