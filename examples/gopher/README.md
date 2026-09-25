# gopher

A voice agent you can just talk to, in a [Pronto](https://pronto-staging.getstream.io)
call (Stream's video app). Gopher listens to everyone, answers out loud, lets
you interrupt it, and knows when you have not finished. In a meeting it
answers only when addressed by name. Every model runs in this process:

| Role | Model | On an M4 Max |
| --- | --- | --- |
| Hear | Qwen3-ASR-1.7B | 11 s of speech in 226 ms |
| Know when you are done | Smart Turn v3.2 | a few ms per pause |
| Think | Qwen3-8B | reply starts ~50 ms after your words reach it; 19 ms per token |
| Speak | Qwen3-TTS-12Hz-1.7B | first audio 43 ms after the text; 0.21 of real time |

The four models take about 15 GB and load in about 2.5 s from gophonic's
weight cache. Audio never leaves the machine; Stream carries the call.

```sh
tools/fetch-models.sh asr turn qwen3 tts   # from the repository root
cd examples/gopher
GOEXPERIMENT=simd go run .
```

Open the printed link, allow the microphone, and talk. `-voice` picks one of
Qwen3-TTS's voices (ryan, aiden, serena, vivian, eric, dylan, uncle_fu,
ono_anna, sohee); `-call default:ID` joins an existing call.

## How it works

The program has no turns. The agent is a `speech.Duplex`: every 20 ms the
call's mixed audio goes in and the agent's speech comes out, on one clock.
`duplex.New` builds it from the four models by what they provide:

```go
agent, err := duplex.New(duplex.Config{Prompt: prompt}, asr, turn, llm, tts)
...
state, err := agent.Step(ctx, in, out) // every 20 ms
```

Inside, the cascade listens continuously while it thinks and speaks:

- A pause of 200 ms asks Smart Turn whether you are done; Qwen3-ASR then
  transcribes the utterance and Qwen3-8B answers, its text streaming into
  Qwen3-TTS as it is written, so speech starts after the first words.
- Talk before Gopher's answer is heard and it drops the answer and keeps
  listening to your whole sentence. Talk over its speech for 300 ms and it
  stops within a frame; a short "mm-hmm" lets it go on.
- Its memory holds only what you heard: an interrupted answer is cut where
  it stopped.

The example is its own module, so its WebRTC dependencies stay out of
gophonic's.
