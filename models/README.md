# models

Model files live here. Tests, benchmarks, and examples find them by name, so
`go test ./...` runs every check whose model is present and skips the rest.
Set `GOPHONIC_MODELS` to keep them elsewhere. Git ignores everything in this
directory except this file; symbolic links work.

```sh
tools/fetch-models.sh                 # whisper and turn (the default)
tools/fetch-models.sh qwen3 clm       # large Hugging Face checkpoints
```

The script downloads each official checkpoint, verifies the SHA-256 digest
its converter pins, and converts it. Files already present are kept.

| File | Model | Target | Source |
| --- | --- | --- | --- |
| `tiny.en.gophonic` | Whisper tiny.en | `whisper` | [OpenAI](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt), converted |
| `base.en.gophonic` | Whisper base.en | `whisper` | [OpenAI](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt), converted |
| `small.en.gophonic` | Whisper small.en | by hand | [OpenAI](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt), converted |
| `smart-turn-v3.2.gophonic` | Smart Turn v3.2 | `turn` | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3), converted |
| `tinymel.gophonic` | TinyMelNet | `turn` | [deveshu/hinglish-turn-detector](https://huggingface.co/deveshu/hinglish-turn-detector), converted |
| `Qwen3-8B/` | Qwen3-8B snapshot | `qwen3` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| `qwen3-8b-hello-reference.f32` | BF16 PyTorch hidden state of "hello" | `qwen3` | `qwen3/tools/reference_hidden.py` |
| `CLM_v0.1-8B.gclm` | CLM v0.1 head | `clm` | [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), converted |

The Qwen3 targets download about 16 GB and compute the reference vector with
PyTorch in BF16. [Models](../docs/models.md) documents each converter and its
provenance.
