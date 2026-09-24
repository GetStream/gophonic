# gophonic

**Speech inference in pure Go: Whisper transcription and turn detection.**

gophonic runs OpenAI Whisper and audio turn-detection models inside your Go
process. There is no cgo, no ONNX Runtime, and no Python at inference time.
On Apple Silicon with SME (M4 and later), Whisper runs **about 3× faster than
whisper.cpp on one core**, with the same FP32 weights and identical
transcripts.

- **Whisper speech-to-text:** official English checkpoints `tiny.en`,
  `base.en` and `small.en`, with an encoder, incremental decoder, tokenizer,
  and long-audio transcription.
- **Turn detection:** Pipecat Smart Turn v3.2 and TinyMelNet estimate whether
  a speaker has finished talking.
- **Built for servers:** a loaded model is shared and immutable. Each
  concurrent lane reuses its own scratch, and warm calls allocate nothing.
- **Pure Go toolchain:** builds with `CGO_ENABLED=0`. Hand-written ARM64
  kernels are plain Go assembly, and every accelerated path has a portable
  fallback.

[Whisper performance](docs/whisper-performance.md) ·
[Whisper design](docs/whisper-design.md) · [Models](docs/models.md) ·
[Go API](docs/api.md) · [Architecture](docs/architecture.md) ·
[Benchmarks](docs/benchmarks.md) · [Validation](docs/validation.md)

## Whisper transcription

Download an official English checkpoint and convert it once. The converter
accepts only the pinned OpenAI SHA-256 for each model and needs Python with
NumPy:

```sh
curl -fLO https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt
python3 tools/whisper_pt_to_gophonic.py base.en.pt base.en.gophonic

CGO_ENABLED=0 GOEXPERIMENT=simd go build -o gophonic ./cmd/gophonic
./gophonic -whisper-model base.en.gophonic speech.wav   # {"text":"..."}
```

| Model | Parameters | Checkpoint |
| --- | ---: | --- |
| `tiny.en` | 39 M | [tiny.en.pt](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt) |
| `base.en` | 74 M | [base.en.pt](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt) |
| `small.en` | 244 M | [small.en.pt](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt) |

The converter also accepts `medium.en`, which has not been benchmarked. The CLI
reads WAV and Ogg Opus. From Go, share one `*whisper.Model` and create one
`*whisper.Transcriber` per concurrent caller:

```go
model, err := whisper.Load("base.en.gophonic")
if err != nil { return err }
worker, err := whisper.NewTranscriber(model)
if err != nil { return err }
defer worker.Close()

text, err := worker.TranscribeInto(mono16kPCM, make([]byte, 0, 4096))
```

Transcription is greedy at temperature zero. It uses whole-file mel
normalization, previous-window context, timestamp seeking, and the reference
no-speech rule. It does not include temperature fallback, beam search,
multilingual models, or word timestamps.

### Performance

Warm latency for one 30-second window, PCM to text, on an Apple M4 Max. The
comparison uses whisper.cpp with its fastest CPU backend (Accelerate/BLAS),
the same FP32 weights, the same audio, and the same thread budget. Every call
checks the exact transcript.

| Model | Threads | gophonic | whisper.cpp | Speedup |
| --- | ---: | ---: | ---: | ---: |
| tiny.en | 1 | 100.3 ms | 298.0 ms | **2.97×** |
| base.en | 1 | 198.2 ms | 565.0 ms | **2.85×** |
| small.en | 1 | 646.1 ms | 1,734 ms | **2.68×** |
| tiny.en | 8 | 57.3 ms | 73.0 ms | **1.27×** |
| base.en | 8 | 117.5 ms | 129.0 ms | **1.10×** |

[Whisper performance](docs/whisper-performance.md) covers the method, raw
samples, per-stage costs, and remaining limits.

The speed comes from Apple's **SME** matrix unit, driven from Go assembly:

- **Matrix products:** encoder projections, attention, convolutions, and the
  STFT run as 32×32 outer-product tiles at about 90% of the unit's measured
  FP32 peak.
- **Matrix-vector products:** the decoder's projections accumulate in the matrix
  unit, reading FP16 copies of weights that convert back to FP32 exactly. This
  halves memory traffic without changing a single weight.
- **Elementwise work:** softmax, GELU, LayerNorm, and argmax are hand-written
  NEON, with fused passes wherever data would otherwise be read twice.

SME is detected at run time. Other CPUs use portable NEON or scalar Go kernels.

### Accuracy

All arithmetic is FP32. FP16 appears only as lossless weight storage, and only
when every value round-trips bit-exactly. Tests check the encoder against
PyTorch activations stage by stage and require exact reference token
sequences. The SME kernels are bit-exact against a sequential fused
multiply-add oracle.

## Turn detection

Both detectors look at the last eight seconds of speech and return the
probability that the turn is complete.

| Model | Runtime graph | Load / CLI | Complete when |
| --- | --- | --- | ---: |
| [Pipecat Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) | Whisper encoder, FP32 | `Load` / `-model` | `> 0.5` |
| [TinyMelNet](https://huggingface.co/deveshu/hinglish-turn-detector) | INT8 convolutions, bidirectional GRU | `LoadTinyMel` / `-tiny-model` | `> 0.57` |

Convert a checkpoint once (Python with NumPy and ONNX), then run it:

```sh
curl -fLO https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx
python3 tools/onnx_to_gophonic.py smart-turn-v3.2-gpu.onnx smart-turn-v3.2.gophonic
./gophonic -model smart-turn-v3.2.gophonic speech.wav   # {"probability":0.91,"complete":true}
```

For TinyMelNet, convert `model_tinymel_int8.onnx` with
`tools/tinymel_to_gophonic.py` and pass `-tiny-model`; `-tiny-workers` adds
helper goroutines. From Go:

```go
model, err := gophonic.LoadTinyMel("tinymel.gophonic")
if err != nil { return err }
session, err := gophonic.NewTinyMelSession(model, 7) // 7 helper goroutines
if err != nil { return err }
defer session.Close()

prediction, err := session.PredictInto(pcm, 16000, 1) // call on each VAD pause
```

Sessions accept mono or stereo PCM at 8–96 kHz. Both detectors implement
`AudioSession`, which is also the interface for plugging in another backend
([example](docs/api.md#add-an-audio-backend)). With seven helpers, TinyMelNet
takes **3.6 ms** from PCM to prediction on an M4 Max, with no allocations
([details](docs/benchmarks.md)).

## Testing

```sh
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 GOEXPERIMENT=simd go test ./...

# With converted models, add the reference-parity suites:
GOPHONIC_WHISPER_MODEL=tiny.en.gophonic \
GOPHONIC_TEST_MODEL=smart-turn-v3.2.gophonic \
GOPHONIC_TEST_TINYMEL_MODEL=tinymel.gophonic \
GOEXPERIMENT=simd go test ./...
```

Parity tests establish numerical behavior on the covered fixtures, not
application-level accuracy; see [validation](docs/validation.md).

## License

gophonic is [BSD-2-Clause](LICENSE). Model weights and `gopus` keep their own
licenses; see [model provenance](docs/models.md#provenance-and-licenses).
