# gopher

A voice agent you can just talk to, in a [Pronto](https://pronto-staging.getstream.io)
call (Stream's video app). Gopher listens to everyone, answers out loud, lets
you interrupt it, and knows when you have not finished. It can look things
up, tell the time anywhere, keep quiet until you want it back, and in a
meeting it answers what is meant for it. Every model runs in this process:

| Role | Model | On an M4 Max |
| --- | --- | --- |
| Hear, and know when you are done | Qwen3-ASR-1.7B | 11 s of speech in 226 ms; the end of your turn in the same pass |
| Think, and decide whether to speak | Qwen3-8B | reply starts ~50 ms after your words reach it; 19 ms per token |
| Speak | Qwen3-TTS-12Hz-1.7B | first audio 18 ms after the text; 0.21 of real time |

The three models take about 15 GB and load in under 2 s from gophonic's
weight cache. Audio never leaves the machine; Stream carries the call, and
the search tool reads Wikipedia.

```sh
tools/fetch-models.sh asr qwen3 tts   # from the repository root
cd examples/gopher
GOEXPERIMENT=simd go run . -languages en,pt
```

Open the printed link, allow the microphone, and talk. `-languages` lists
the languages spoken in the call: what Gopher hears is transcribed in one of
them, never another, and it answers in the one it is spoken to. `-voice`
picks one of Qwen3-TTS's voices (ryan, aiden, serena, vivian, eric, dylan,
uncle_fu, ono_anna, sohee); `-call default:ID` joins an existing call.

## How it works

The program has no turns. The agent is a `speech.Duplex`: every 20 ms the
call's mixed audio goes in and the agent's speech comes out, on one clock.
`duplex.New` builds it from the models by what they provide:

```go
agent, err := duplex.New(duplex.Config{Prompt: prompt, Tools: tools(), Listen: listen}, asr, llm, tts)
...
state, err := agent.Step(ctx, in, out) // every 20 ms
```

Inside, the cascade works while you talk:

- Qwen3-ASR transcribes you as you speak, and Qwen3-8B reads along. At your
  first pause the answer is prepared and held; Qwen3-ASR's own judgment,
  made as it finishes your transcript, releases it the moment you are done,
  and waits, judging again, when you only paused.
- Talk before Gopher's answer is heard and it drops the answer and keeps
  listening to your whole sentence. Talk over its speech and it stops within
  a frame; a short "mm-hmm" lets it go on.
- Gopher may choose to say nothing: to speech meant for someone else, or
  when asked to be quiet ("stop talking until I say hi"), which lasts until
  that happens.
- Its memory holds only what you heard: an interrupted answer is cut where
  it stopped.

Its tools are plain Go functions ([`tools.go`](tools.go)); the arguments
struct tells the model how to call each:

```go
duplex.Func("now", "The current date and time: here, or in another time zone.",
	func(ctx context.Context, args struct {
		Timezone string `json:"timezone,omitempty" desc:"an IANA time zone such as Asia/Tokyo"`
	}) (string, error) { ... })
```

Gopher also watches the call's text chat over Stream Chat's realtime API,
and who comes and goes: both join its conversation as notes, never spoken on
their own, so it can answer "what did Alice write?" or summarize the meeting
later.

`e2e` joins a call as a second participant, speaks a recording three times
(after joining, after muting and unmuting, after rejoining), and reports how
soon Gopher answers and whether it talked over the speech.

The example is its own module, so its WebRTC dependencies stay out of
gophonic's.
