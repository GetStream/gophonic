# Stream WebRTC + local gophonic inference

This example joins a Stream call as a receive-only client. For each remote audio
track it decodes Opus/RED to mono 16 kHz float32 PCM using
[`audio/rtc.TrackReader`](https://github.com/GetStream/getstream-go-webrtc/blob/80daf64c2fc1a0702c0e6df175fc4239deaacd03/audio/rtc/reader.go).
A per-track `gopus.VAD` checks 20 ms frames and a Smart Turn session checks 500 ms pauses. Completed utterances go to
one separate Whisper tiny.en worker. A bounded queue prevents Whisper latency
from stalling the RTP reader; if all eight slots are occupied, the example logs
and drops a transcript. The 30-second limit also bounds retained PCM memory.

This is a nested Go module so the core `gophonic` module does not acquire the
WebRTC SDK's dependencies. It uses the current local checkout through a
relative `replace` directive and pins the WebRTC SDK to commit `80daf64`.

## Run locally

Convert or obtain compatible `.gophonic` model bundles as described in the
[root README](../../README.md). From this directory:

```sh
export STREAM_API_KEY=...
export STREAM_CALL_ID=...
export STREAM_USER_TOKEN=... # or STREAM_API_SECRET for local server-side experiments
export STREAM_USER_ID=gophonic-listener
# Optional: STREAM_CALL_TYPE=default

CGO_ENABLED=0 GOEXPERIMENT=simd go run . \
  -turn-model /path/to/smart-turn-v3.2.gophonic \
  -whisper-model /path/to/tiny.en.gophonic
```

Run another participant in the call to supply audio. The listener subscribes
to audio already published when it joins and to newly published audio tracks.
It logs turn probabilities and transcripts by participant. Stop with Ctrl-C.

The `gopus` SILK VAD returns activity on a 0–255 scale. This example uses
a threshold of 128; tune it and the 500 ms hangover on real call audio. Smart
Turn predicts whether a paused utterance is complete; it does not detect speech
activity. This example runs one Whisper worker on one CPU core. Add workers
only after measuring single-core latency, queue depth, and accuracy with your
real audio; each worker needs its own `whisper.Transcriber` while the model can
be shared.

## Ownership and allocation boundary

`TrackReader.Read()` returns float32 PCM. The example converts each 20 ms frame
into a reusable int16 buffer for `gopus.VAD`, then copies accepted audio into an
utterance buffer. On completion, `submit` transfers ownership of that buffer to
the Whisper worker. The gophonic session and transcriber keep reusable inference
scratch, but this whole ingress path is **not zero allocation**: the SDK's Opus
wrapper currently allocates and copies each decoded frame in
[`Decoder.output`](https://github.com/GetStream/getstream-go-webrtc/blob/80daf64c2fc1a0702c0e6df175fc4239deaacd03/audio/opus/opus.go).
For an end-to-end zero-allocation path, the SDK would need a borrowed-frame or
`ReadInto` API backed by caller-owned buffers, including explicit lifetime rules
for RED recovery and packet-loss concealment. Measure it before adding another
copy or claiming a throughput improvement.
