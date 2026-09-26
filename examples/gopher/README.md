# gopher

A voice agent you can just talk to, in a [Pronto](https://pronto-staging.getstream.io)
call (Stream's video app). Gopher listens to everyone, answers out loud, lets
you interrupt it, and knows when you have not finished. It can look things
up, tell the time anywhere, keep quiet until you want it back, and in a
meeting it answers what is meant for it. Every model runs in this process:

| Role | Model | On an M4 Max |
| --- | --- | --- |
| Hear, and know when you are done | Qwen3-ASR-1.7B | 11 s of speech in 226 ms; the end of your turn in the same pass |
| Think, and decide whether to speak | Qwen3.6-35B-A3B | a mixture of experts: 8 ms per token |
| Speak | Qwen3-TTS-12Hz-1.7B | first audio 18 ms after the text; 0.21 of real time |

The three models take about 40 GB and load in under 3 s from gophonic's
weight cache. Audio never leaves the machine; Stream carries the call, and
the search tool reads Wikipedia.

```sh
tools/fetch-models.sh asr qwen36 tts   # from the repository root
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
agent, err := duplex.New(duplex.Config{Prompt: prompt, Tools: tools(), Listen: listen, Idle: idle}, asr, llm, tts)
...
state, err := agent.Step(in, out) // every 20 ms
agent.Speaker(name)               // who is talking, in a meeting
agent.Note("Ana joined the call.")
```

Inside, the cascade works while you talk:

- Qwen3-ASR transcribes you as you speak, and Qwen3.6 reads along. At your
  first pause the answer is prepared and held; Qwen3-ASR's own judgment,
  made as it finishes your transcript, releases it the moment you are done,
  and waits, judging again, when you only paused.
- Talk before Gopher's answer is heard and it drops the answer and keeps
  listening to your whole sentence. Talk over its speech and it stops within
  a frame; a short "mm-hmm" lets it go on.
- Gopher acts at moments: after you speak, when something is noted (a chat
  message, someone joining), when a pause it asked for is over, and after a
  quiet spell. Each time it may say nothing: to speech meant for someone
  else, or when asked to be quiet ("stop talking until I say hi"), which it
  judges again, in context, every time. "Remind me in thirty seconds" is a
  reply that ends in `<silent 30s>`: it is asked again then.
- Its memory holds only what you heard: an interrupted answer is cut where
  it stopped.

Its tools are plain Go functions ([`tools.go`](tools.go)); the arguments
struct tells the model how to call each:

```go
chat.Func("now", "The current date and time: here, or in another time zone.",
	func(ctx context.Context, args struct {
		Timezone string `json:"timezone,omitempty" desc:"an IANA time zone such as Asia/Tokyo"`
	}) (string, error) { ... })
```

Gopher also watches the call's text chat over Stream Chat's realtime API,
and who comes and goes: both reach it as notes (`Note`), never spoken on
their own, so it can answer "what did Alice write?" or summarize the meeting
later. In a meeting the loudest participant is named as the speaker
(`Speaker`), so the conversation knows whose words it hears without the
words themselves being reframed.

## Scenarios

[`scenarios/`](scenarios) holds what Gopher must do, as dialogues:

```
user: Gopher, stop talking until I say hi.
gopher: silent
user: Hi!
gopher: speaks
```

`go test -run TestScenarios .` speaks each `user:` line with Qwen3-TTS in
another voice, transcribes Gopher's answers with Qwen3-ASR, and has Qwen3.6
judge each `says` claim, in real time with the three models (see package
`scenario`). A script whose first line is `# expect: fail` documents a
behavior that does not work yet.

`e2e` joins a call as a second participant, speaks a recording three times
(after joining, after muting and unmuting, after rejoining), and reports how
soon Gopher answers and whether it talked over the speech.

The example is its own module, so its WebRTC dependencies stay out of
gophonic's.
