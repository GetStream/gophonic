# Local Whisper HTTP server

`gophonic-server` serves completed audio recordings with one immutable English
Whisper model and a fixed pool of independent transcribers. Its inference path
uses the same Go runtime as the CLI. Each worker owns reusable request, decoded
audio, inference, and response buffers. On the measured warmed WAV handler
path, multipart parsing through response writing takes zero heap allocations.
Go's `net/http` transport still allocates for each connection/request; this
measurement starts at the handler with a reusable request object.

```sh
CGO_ENABLED=0 GOEXPERIMENT=simd go build -o gophonic-server ./cmd/gophonic-server
./gophonic-server -whisper-model tiny.en.gophonic -listen 127.0.0.1:8080 -workers 1
```

In another terminal:

```sh
curl -sS http://127.0.0.1:8080/healthz
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=gophonic-whisper -F file=@recording.wav
# {"text":"..."}
```

`POST /v1/audio/transcriptions` accepts WAV files, returning JSON
`{"text":"..."}` or plain text when `response_format=text`. It accepts
`model=gophonic-whisper` or `model=whisper-1` for clients that require a model
field, and `language=en`. Other transcription options are rejected explicitly.
This is a small file-transcription subset of the OpenAI API, not a claim of full
API compatibility. `GET /healthz` and `GET /readyz` return 200 while the server
is running.

The CLI also decodes Ogg Opus. The HTTP endpoint currently accepts WAV so its
hot path can reuse all parser and decoder storage. Uploads are limited to
25 MiB, and decoded audio defaults to at most 120
seconds (`-max-audio-seconds` changes that limit). The default address is
loopback. The server has no authentication or TLS; put it behind an
authenticating reverse proxy before binding to a non-loopback address. A bounded
admission queue holds at most twice the configured number of workers; additional requests receive HTTP 503. One worker uses its own upload, decoded PCM, mono PCM, text, response,
transcriber, and resampling workspace, so concurrent requests do not share
mutable state. The first upload at a new maximum size or sample rate may
grow scratch; repeated requests within prepared capacities allocate zero
heap objects at the handler boundary. Shut down with Ctrl-C; the server drains active requests
before closing workspaces.

The model currently provides English greedy transcription without word
timestamps, beam search, or true incremental decoding. A live WebRTC service
needs an audio/Opus ingestion loop, buffering and VAD, and a separate contract
for partial versus committed transcripts. The file endpoint does not pretend
that repeated complete-file requests are incremental decoding.

To validate a converted official `tiny.en` bundle without HTTP, run the pinned
JFK test documented in [README](../README.md#testing). With a local bundle,
this server's official JFK test also exercises the multipart endpoint:

```sh
GOPHONIC_WHISPER_MODEL=/path/to/tiny.en.gophonic \
  GOEXPERIMENT=simd go test ./internal/httpserver -run TestWhisperServerOfficialJFK -count=1
```
