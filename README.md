# gophonic

**Speech and language models in pure Go: transcription, turn detection, and
Qwen3.**

gophonic runs speech-to-text, end-of-turn detection, and Qwen3-8B inside your
Go process. There is no cgo, no ONNX Runtime, and no Python at inference time.
Matrix work runs on Apple's SME matrix unit from Go assembly, or on the Apple
GPU through a pure-Go Metal binding, with portable kernels everywhere else.

| Task | Model | Package | On an M4 Max |
| --- | --- | --- | --- |
| Transcription | OpenAI Whisper `tiny.en`, `base.en`, `small.en` | [`whisper`](whisper) | about 3× whisper.cpp on one core, identical transcripts |
| Turn detection | Pipecat Smart Turn v3.2 | [`smartturn`](smartturn) | FP32 Whisper encoder, ONNX parity within 2e-5 |
| Turn detection | TinyMelNet | [`tinymel`](tinymel) | 3.6 ms from PCM to prediction, zero allocations |
| Language | Qwen3-8B, CLM action ranking | [`qwen3`](qwen3), [`clm`](clm) | one token in 15 ms on the GPU (llama.cpp Metal: 19 ms) |

- **One interface per task.** Every transcription model implements
  `speech.Transcriber` and every turn detector `speech.TurnDetector`;
  `gophonic.Open` loads any supported model by path. The CLI and the HTTP
  server see only those interfaces.
- **Built for servers.** A loaded model is immutable and shared. Each
  concurrent lane owns its scratch, and warm calls allocate nothing.
- **Pure Go toolchain.** `CGO_ENABLED=0` builds everything. Hand-written
  kernels are Go assembly or Go 1.27 SIMD intrinsics, and every accelerated
  path has a portable fallback.

[Go API](docs/api.md) · [Architecture](docs/architecture.md) ·
[Models](docs/models.md) · [Local server](docs/server.md) ·
[Whisper design](docs/whisper-design.md) ·
[Whisper performance](docs/whisper-performance.md) ·
[Qwen3 performance](docs/clm-performance.md) ·
[Benchmarks](docs/benchmarks.md) · [Validation](docs/validation.md)

## Quick start

Fetch the official models once; the script downloads each checkpoint,
verifies its pinned digest, and converts it into [`models/`](models). Then
build the CLI and point it at any model; the format is detected from the
file:

```sh
tools/fetch-models.sh   # Whisper tiny.en and base.en, Smart Turn, TinyMelNet

CGO_ENABLED=0 GOEXPERIMENT=simd go build -o gophonic ./cmd/gophonic
./gophonic -model models/base.en.gophonic speech.wav          # {"text":"..."}
./gophonic -model models/base.en.gophonic -response-format srt speech.wav
./gophonic -model models/base.en.gophonic -response-format verbose_json -word-timestamps speech.wav
./gophonic -model models/smart-turn-v3.2.gophonic speech.wav  # {"probability":0.91,"complete":true}
```

The CLI reads WAV and Ogg Opus. [`gophonic-server`](docs/server.md) serves the
same transcription models over an OpenAI-style HTTP endpoint.

From Go, open the model once and give each concurrent caller its own lane:

```go
model, err := gophonic.Open("models/base.en.gophonic", gophonic.Options{})
if err != nil { return err }
lane, err := model.NewTranscriber()
if err != nil { return err }
defer lane.Close()

var t speech.Transcript // reused across calls
err = lane.Transcribe(ctx, mono16kPCM, speech.Options{Words: true}, &t)
fmt.Printf("%s (%s, %d words)\n", t.Text, t.Language, len(t.Words))
```

Turn detectors work the same way, on PCM at any rate from 8 to 96 kHz:

```go
model, err := gophonic.Open("models/smart-turn-v3.2.gophonic", gophonic.Options{})
if err != nil { return err }
detector, err := model.NewTurnDetector()
if err != nil { return err }
defer detector.Close()

prediction, err := detector.PredictInto(pcm, 48000, 2) // call on each VAD pause
```

`speech.NewResampler` converts other formats to the 16 kHz mono PCM a
transcriber expects. Each model package also offers lower-level entry points,
such as feature-level prediction and caller-sized result buffers; see the
[Go API](docs/api.md).

## Whisper transcription

| Model | Parameters | Checkpoint |
| --- | ---: | --- |
| `tiny.en` | 39 M | [tiny.en.pt](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt) |
| `base.en` | 74 M | [base.en.pt](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt) |
| `small.en` | 244 M | [small.en.pt](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt) |

The converter also accepts `medium.en`, which has not been benchmarked.
Transcription is greedy at temperature zero, with whole-file mel
normalization, previous-window context, timestamp seeking, the reference
no-speech rule, and segment and word timestamps. Temperature fallback, beam
search, and multilingual Whisper checkpoints are not supported.

Warm latency for one 30-second window, PCM to text, against whisper.cpp with
its fastest CPU backend (Accelerate/BLAS), the same FP32 weights, the same
audio, and the same thread budget. Every call checks the exact transcript.

| Model | Threads | gophonic | whisper.cpp | Speedup |
| --- | ---: | ---: | ---: | ---: |
| tiny.en | 1 | 100.3 ms | 298.0 ms | **2.97×** |
| base.en | 1 | 198.2 ms | 565.0 ms | **2.85×** |
| small.en | 1 | 646.1 ms | 1,734 ms | **2.68×** |
| tiny.en | 8 | 57.3 ms | 73.0 ms | **1.27×** |
| base.en | 8 | 117.5 ms | 129.0 ms | **1.10×** |

Encoder projections, attention, convolutions, and the STFT run as 32×32 SME
outer-product tiles at about 90% of the unit's measured FP32 peak. The decoder
streams FP16 copies of weights that convert back to FP32 exactly, halving
memory traffic without changing a weight. Softmax, GELU, LayerNorm, and argmax
are hand-written NEON. All arithmetic is FP32; tests check the encoder against
PyTorch activations stage by stage and require the exact reference token
sequences. [Whisper performance](docs/whisper-performance.md) has the method,
raw samples, and per-stage costs.

## Turn detection

Both detectors read the last eight seconds of audio and return the
probability that the speaker has finished.

| Model | Runtime graph | Complete when |
| --- | --- | ---: |
| [Pipecat Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) | Four-layer Whisper encoder, FP32 | `> 0.5` |
| [TinyMelNet](https://huggingface.co/deveshu/hinglish-turn-detector) | INT8 convolutions, bidirectional GRU | `> 0.57` |

`tools/fetch-models.sh turn` converts both from their pinned ONNX files.
`Options.Threads`
(CLI `-threads`) sets its worker count: with eight it takes 3.6 ms from PCM
to prediction ([details](docs/benchmarks.md)). A custom
detector implements `speech.TurnDetector` and plugs into the same code
([example](docs/api.md#add-a-backend)).

## Qwen3-8B and CLM ranking

[`qwen3`](qwen3) runs the official Qwen3-8B checkpoint on the Apple GPU
(chosen automatically) or the CPU. It embeds texts for the
[CLM ranking heads](docs/clm.md), answers zero-shot multiple-choice questions
without generating text, and reuses the keys and values of growing
conversations. On an M4 Max:

| | gophonic GPU | llama.cpp Metal Q8_0 | gophonic CPU int8 | llama.cpp CPU Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| 1 token | **15.2 ms** | 18.6 ms | 27.5 ms | 30.3 ms |
| 12 tokens | **25 ms** | 58 ms | 43 ms | 87 ms |
| 2048 tokens | 3.5 s | **3.1 s** | 4.4 s | 28 s |

The GPU's int8 weights match llama.cpp Q8_0 fidelity (cosine 0.99933 against
official BF16) and exceed it once rounded with GPTQ (0.99990); the exact CPU
path stores every BF16 weight without rounding (0.99991). See the [package README](qwen3/README.md) and the
[performance report](docs/clm-performance.md).

## Layout

| Package | Contents |
| --- | --- |
| `gophonic` | `Open`, which recognizes and loads any supported model, and the turn detectors' standalone log-mel frontend |
| `speech` | `Transcriber`, `TurnDetector`, transcripts, predictions, languages, and the 16 kHz resampler |
| `whisper`, `smartturn`, `tinymel`, `qwen3`, `clm` | One model family each; model packages never import each other |
| `internal/mel`, `internal/resample` | The log-mel frontends and the resampling filter every model shares |
| `internal/qwen3lm` | The Qwen3 transformer: loader, tokenizer, CPU and GPU forward pass |
| `internal/whispergemm`, `internal/q8gemm`, `internal/q8gemv`, `internal/vec`, `internal/metal` | SME, NEON, and scalar kernels and the Metal binding |
| `cmd/gophonic`, `cmd/gophonic-server`, `cmd/qwen3-gptq` | CLI, HTTP server, offline GPTQ rounding |

[Architecture](docs/architecture.md) explains the dependency rules and the
shared building blocks.

## Testing

```sh
tools/fetch-models.sh                 # optional: official models into models/
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 GOEXPERIMENT=simd go test ./...
```

Model-level tests find their weights in [`models/`](models) and skip when a
file is absent; they never download anything. Parity tests establish
numerical behavior on the covered fixtures, not application-level accuracy;
see [Validation](docs/validation.md).

## License

gophonic is [BSD-2-Clause](LICENSE). Model weights and `gopus` keep their own
licenses; see [model provenance](docs/models.md#provenance-and-licenses).
