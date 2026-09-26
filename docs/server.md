# HTTP server

`gophonic-server` serves every model you point it at. Transcription and the
model list follow the OpenAI API, so OpenAI clients work unchanged.

```sh
CGO_ENABLED=0 GOEXPERIMENT=simd go build -o gophonic-server ./cmd/gophonic-server
./gophonic-server models/          # every model in models/, on 127.0.0.1:8080
```

Pass model paths, directories of models, or both. A model's name is its file
or directory name without `.gophonic`: `Qwen3-ASR-1.7B`, `smart-turn-v3.2`,
`Qwen3-1.7B`.

Models load on the first request that needs them and close after 15 minutes
without requests (`-keep-alive`). Prepared weights are cached on disk
([Loading](api.md#loading-and-the-weight-cache)), so reloading a closed model
takes a fraction of a second.

| Endpoint | Does |
| --- | --- |
| `POST /v1/chat/completions` | Converses with a language model, streamed or whole, with tools. OpenAI-compatible. |
| `POST /v1/audio/speech` | Speaks text as WAV, or streams PCM. OpenAI-compatible. |
| `POST /v1/audio/transcriptions` | Transcribes a WAV upload. OpenAI-compatible. |
| `GET /v1/models` | Lists the models. OpenAI-compatible. |
| `POST /v1/audio/classifications` | Scores a WAV upload with an audio classifier, such as a turn detector. |
| `POST /v1/classifications` | Classifies text: zero-shot with a language model, or with a text classifier. |
| `GET /healthz`, `GET /readyz` | Return 200 while the server runs. |

Errors have OpenAI's shape: `{"error":{"message":"..."}}`.

## Choosing a model

Every endpoint takes a `model` field. It names a served model; when it is
empty or an OpenAI model name (`whisper-1`, `gpt-4o-transcribe`,
`gpt-4o-mini-transcribe`), the server uses the first model that can do the
job, preferring those whose format declares it without loading anything.
Any other unknown name returns 404.

## Transcription

```sh
curl -sS localhost:8080/v1/audio/transcriptions -F model=whisper-1 -F file=@speech.wav
# {"text":"..."}
```

With the official OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.audio.transcriptions.create(model="Qwen3-ASR-1.7B", file=open("speech.wav", "rb")).text)
```

Form fields:

| Field | Meaning |
| --- | --- |
| `file` | WAV audio, at most 25 MiB and `-max-audio-seconds` (default 120). |
| `model` | See [Choosing a model](#choosing-a-model). |
| `language` | ISO 639-1 code or English name; default: detect. |
| `prompt` | Context text, such as names and terms. |
| `response_format` | `json` (default, `{"text":...}`), `text`, `verbose_json`, `srt`, or `vtt`. |
| `timestamp_granularities[]` | `segment` or `word`, with `verbose_json`. Word times need a Whisper model. |

A value the model cannot honor, such as word timestamps from Qwen3-ASR or
German for Whisper's English checkpoints, returns 400.

## Audio classification

Upload a WAV file as for transcription. Turn detectors classify a speaker's
last eight seconds as `incomplete` or `complete`:

```sh
curl -sS localhost:8080/v1/audio/classifications -F model=smart-turn-v3.2 -F file=@turn.wav
# {"model":"smart-turn-v3.2","classes":[{"label":"incomplete","probability":0.0513348},{"label":"complete","probability":0.948665}]}
```

## Text classification

A language model answers a question about each input with one of the
labels you give, without generating text: moderation, routing, sentiment,
intent, whatever the question asks.

```sh
curl -sS localhost:8080/v1/classifications -d '{
  "model": "Qwen3-1.7B",
  "input": ["you absolute clown", "see you at 3 for the design review"],
  "question": "Is this message acceptable in a workplace chat?",
  "labels": ["acceptable", "rude", "spam"]
}'
# {"model":"Qwen3-1.7B","results":[
#   {"classes":[{"label":"acceptable","probability":7.85132e-12},{"label":"rude","probability":1},{"label":"spam","probability":2.24497e-09}]},
#   {"classes":[{"label":"acceptable","probability":0.999999},{"label":"rude","probability":9.86703e-07},{"label":"spam","probability":1.70266e-09}]}]}
```

`input` is a string or a list of strings. The server evaluates each question
once and keeps up to 16 prepared, so requests that repeat a question pay only
for their inputs: the request above takes 20 ms once its question is
prepared, on an M4 Max. A model with a text classifier of its own takes neither
`question` nor `labels`.

## Chat

```sh
curl -sS localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3-8B",
  "messages": [{"role": "user", "content": "What is the capital of Portugal?"}],
  "stream": true}'
```

The OpenAI clients work unchanged, streaming included. A request's
conversation is replayed into a new session of the model, so the server
keeps no state between requests. With `"stream": true` the reply comes as
server-sent events, a chunk per piece as the model writes it, then
`data: [DONE]`.

Tools follow OpenAI's shape. The server runs none: a reply that calls
tools comes back with `tool_calls` and `finish_reason: "tool_calls"`, and
the client runs them and sends their results as `tool` messages. The
assistant message that made the calls, replayed in the next request, is
written in the model's own format, so Qwen3 and Qwen3.6 each see calls as
they were trained to.

| Field | Meaning |
| --- | --- |
| `messages` | `system` (or `developer`), `user`, `assistant` (with `tool_calls`), and `tool` messages; content is text, or text parts |
| `tools` | Functions the model may call |
| `stream` | Server-sent events |
| `max_tokens`, `max_completion_tokens` | Bound the reply |
| `temperature`, `top_p`, `presence_penalty`, `seed` | Sampling, OpenAI's defaults (temperature 1); `top_k` too |

`n` other than 1, `stop`, and content other than text fail with 400.

## Speech

```sh
curl -sS localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "The capital of Portugal is Lisbon.", "voice": "ryan", "instructions": "Calm and even."}' -o speech.wav
```

`input` is spoken in `voice` (a voice of the model, or an OpenAI voice name
for the model's default), styled by `instructions` as OpenAI's are; a
`language` field forces a language. `response_format` is `wav` (the
default) or `pcm`: 16-bit little-endian mono at the voice's rate (24 kHz for
Qwen3-TTS), streamed as it is decoded.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-listen` | `127.0.0.1:8080` | Listen address. |
| `-workers` | 1 | Requests running inference at once; as many more may queue, and the rest get 503. |
| `-keep-alive` | `15m` | Close a model after this long without requests; negative keeps models open. |
| `-max-audio-seconds` | 120 | Longest audio a request may send. |
| `-threads` | 0 | CPU workers per lane; 0 divides `GOMAXPROCS` across request slots (minimum 1, maximum 64). A single slot keeps the model default. |

With `GOMAXPROCS=16`, `-workers 8` defaults to two CPU workers per lane.
Each lane keeps its own mutable scratch and KV cache. The budget counts the
caller as one worker and is selected at startup; explicit `-threads` overrides
it. When there are more request slots than CPU slots, Go schedules the
single-worker lanes. Tune concurrency against your latency target and memory
limit; more concurrent calls do not always increase throughput.

## Operation

The server binds to loopback by default and has no authentication or TLS: put
it behind an authenticating reverse proxy before exposing it. Ctrl-C drains
active requests, then closes the models.

Each request slot owns reusable upload, audio, and response buffers, and
models hand out lanes from a pool, so a warm transcription request allocates
nothing between the handler's entry and its response
(`TestWarmedWAVHandlerAllocations`); Go's `net/http` still allocates for each
connection and request. The HTTP endpoints accept WAV; the CLI also reads Ogg
Opus.

```sh
GOEXPERIMENT=simd go test ./internal/httpserver -count=1
```
