# streamcall

A local AI listener for a [Pronto](https://pronto-staging.getstream.io) call,
Stream's video app. It joins as a silent participant, subscribes to every
microphone, and for each speaker:

1. marks speech with the Opus SILK voice detector from
   [gopus](https://github.com/thesyncim/gopus), which also decodes the audio
   straight to 16 kHz;
2. asks [Smart Turn v3.2](../../smartturn) at every pause whether the turn is
   over;
3. transcribes the finished turn with [Qwen3-ASR](../../qwen3asr), in
   whatever language was spoken;
4. reads the speaker's mood with [Qwen3-8B](../../qwen3) as a zero-shot
   multiple-choice question;
5. posts who spoke, what they said, the language, and the mood to the call's
   chat.

Every model runs in-process through gophonic. Audio never leaves the machine,
and nothing is published into the call.

```sh
tools/fetch-models.sh asr turn qwen3   # from the repository root
cd examples/streamcall
GOEXPERIMENT=simd go run . -stt ../../models/Qwen3-ASR-1.7B \
  -turn ../../models/smart-turn-v3.2.gophonic -llm ../../models/Qwen3-8B
```

It prints a link to the call; open it, allow the microphone, and talk. Pass
`-call default:ID` to join an existing call. Pronto issues the user token
for its own app, so no Stream credentials are needed.

The example is its own module, so its WebRTC dependencies stay out of
gophonic's.
