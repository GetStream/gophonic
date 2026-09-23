# Models and weight bundles

Gophonic ships two specialized inference graphs. Each built-in loader accepts
the bundle for its corresponding graph. Additional audio turn-detector
architectures implement [`AudioSession`](api.md#add-an-audio-backend) with their
own loader and execution code; they can optionally reuse the standalone Whisper
frontend. The built-in converters do not import arbitrary ONNX graphs.

| | Smart Turn v3.2 | TinyMelNet |
| --- | --- | --- |
| Source | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) | [deveshu/hinglish-turn-detector](https://huggingface.co/deveshu/hinglish-turn-detector) |
| ONNX file | `smart-turn-v3.2-gpu.onnx` | `model_tinymel_int8.onnx` |
| Go type | `Model` | `TinyMelModel` |
| Loader | `Load` / `ReadWeights` | `LoadTinyMel` / `ReadTinyMelWeights` |
| Feature input | 64,000 float32 values, row-major `[80,800]` | Same |
| Output | `Prediction{Probability, Complete}` | Same |
| Decision | `Probability > 0.5` | `Probability > 0.57` |

## Convert Smart Turn

Install the offline converter dependencies in an environment of your choice:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy onnx
```

Download and convert the audited FP32 artifact:

```sh
curl -fL \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx \
  -o smart-turn-v3.2-gpu.onnx
.venv/bin/python tools/onnx_to_gophonic.py \
  smart-turn-v3.2-gpu.onnx smart-turn-v3.2.gophonic
```

Expected source SHA-256:

```text
ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be
```

The upstream `-gpu` file contains the FP32 graph. The Go runtime uses the CPU;
the filename does not select an execution device. The separate upstream INT8
Smart Turn artifact is not supported by this loader.

The converter verifies the source digest, input/output shape, initializer
types, and tensor shapes. It maps the checkpoint's generated initializer names
to the runtime's stable names, transposes projection matrices where needed, and
writes a roughly 32 MB bundle. A changed upstream file fails the hash check and
requires a newly audited converter mapping.

## Convert TinyMelNet

The same Python dependencies are enough for bundle conversion:

```sh
curl -fL \
  https://huggingface.co/deveshu/hinglish-turn-detector/resolve/main/model_tinymel_int8.onnx \
  -o model_tinymel_int8.onnx
.venv/bin/python tools/tinymel_to_gophonic.py \
  model_tinymel_int8.onnx --bundle tinymel.gophonic
```

Expected source SHA-256:

```text
6b986a0440b30f533f0f7e473695939c347a2c95c5280eb2e090d0076b3dbbd1
```

The bundle preserves all 51 initializers and 25 tensor-valued constants,
including the ONNX names, shapes, dtypes, scales, and zero points. Quantized
weights remain `uint8`; model loading prepares additional immutable layouts for
the convolution kernels. The runtime also executes FP32 operators, including
the bidirectional GRU. “INT8” names the upstream quantized model artifact; it
does not mean that every operator uses integer arithmetic.

Install ONNX Runtime only if you want to regenerate independent oracle fixtures:

```sh
.venv/bin/python -m pip install onnxruntime
.venv/bin/python tools/tinymel_to_gophonic.py \
  model_tinymel_int8.onnx --oracle-dir /tmp/gophonic-tinymel-oracle
```

This writes CPU reference probabilities, selected intermediate tensors,
quantization metadata, and the complete GRU stage fixtures. It does not modify
the repository's checked-in fixtures. See [Validation](validation.md) before
replacing an existing oracle.

## Model quality and deployment

TinyMelNet's model card reports **0.896 overall** and **0.871 Hinglish** accuracy,
versus **0.938** and **0.951** for its Whisper teacher. These are **FP32 evaluation
figures**, not measurements of this Go runtime or a direct comparison with
Pipecat Smart Turn. The INT8 TinyMelNet file uses the separately published
threshold of **0.57**. The Hinglish test slice is synthetic and template-disjoint;
voice and domain coverage are limited. Its intended use is end-of-turn scoring
after a VAD detects a pause. [Source model card](https://huggingface.co/deveshu/hinglish-turn-detector).

Evaluate the chosen model on your own speakers, languages, microphones, and
pause policy. Numerical agreement with an ONNX reference establishes that the
runtime executes the intended model; it does not establish how often that model
will interrupt or wait in a particular application. Application code can use
`Probability` to apply a separately calibrated threshold.

## Provenance and licenses

| Component | Published license / terms |
| --- | --- |
| Gophonic implementation | [BSD-2-Clause](../LICENSE) |
| Pipecat Smart Turn model | [BSD-2-Clause model metadata](https://huggingface.co/pipecat-ai/smart-turn-v3), [upstream code license](https://github.com/pipecat-ai/smart-turn/blob/main/LICENSE) |
| TinyMelNet model | [MIT model metadata](https://huggingface.co/deveshu/hinglish-turn-detector); its card states that upstream dataset terms govern redistribution and commercial use of trained weights |
| `gopus` dependency | [BSD-3-Clause](https://github.com/thesyncim/gopus/blob/main/LICENSE) |

LiveKit's turn-detector weights are not used. Its
[model license](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE)
restricts use to the LiveKit Agents framework, which does not fit this
standalone runtime.

Keep the source artifact digest with any deployed bundle. The converters pin
the exact upstream bytes they accept. Model downloads and converted bundles
are separate from the source repository.
