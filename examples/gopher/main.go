// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command gopher joins a Pronto call (Stream's video app) as a voice
// agent you can just talk to. It listens to everyone, answers out loud,
// lets you interrupt it, and knows when you have not finished, with every
// model running in this process: Smart Turn and Qwen3-ASR to listen,
// Qwen3-8B to think, and Qwen3-TTS to speak.
//
//	go run .   # then open the printed link and talk
//
// The agent is a speech.Duplex: the call's audio goes in and the agent's
// speech comes out, 20 ms at a time, on one clock. There are no turns in
// this program; the duplex decides when to speak.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	webaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	"github.com/GetStream/getstream-go-webrtc/logger"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/GetStream/getstream-go-webrtc/track"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/sirupsen/logrus"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/duplex"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

const self = "gopher" // our user ID; we never subscribe to ourselves

var verbose *bool

const prompt = `You are Gopher, a friendly voice assistant taking part in a live video call.
Everything you write is spoken aloud, so answer in one to three short, natural sentences,
without lists, markdown, or emoji. You run entirely on the user's own laptop, in Go.`

func main() {
	callFlag := flag.String("call", "", "call to join as type:id (default: a new call)")
	asrPath := flag.String("asr", "../../models/Qwen3-ASR-1.7B", "speech recognition model")
	turnPath := flag.String("turn", "../../models/smart-turn-v3.2.gophonic", "turn detection model")
	llmPath := flag.String("llm", "../../models/Qwen3-8B", "language model")
	ttsPath := flag.String("tts", "../../models/Qwen3-TTS-12Hz-1.7B-CustomVoice", "speech synthesis model")
	language := flag.String("language", "en", "language spoken in the call (ISO 639-1); empty detects it")
	voice := flag.String("voice", "ryan", "voice: ryan, aiden, serena, vivian, eric, dylan, uncle_fu, ono_anna, sohee")
	pronto := flag.String("pronto", "https://pronto-staging.getstream.io", "Pronto app whose call to join")
	verbose = flag.Bool("v", false, "log the input level every second")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Load the four models at once; each maps its cached weights.
	start := time.Now()
	models := make([]*gophonic.Model, 4)
	var wg sync.WaitGroup
	for i, path := range []string{*asrPath, *turnPath, *llmPath, *ttsPath} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m, err := gophonic.Open(path, gophonic.Options{})
			check(err)
			models[i] = m
		}()
	}
	wg.Wait()
	log.Printf("models loaded in %v", time.Since(start).Round(time.Millisecond))

	apiKey, token, err := prontoToken(*pronto, self)
	check(err)
	callType, callID, ok := strings.Cut(*callFlag, ":")
	if !ok {
		callType, callID = "default", "gopher-"+randomHex(3)
	}
	// The call's chat opens once Gopher has joined, which creates its user.
	var room *chatChannel
	humans := &roster{}
	agent, err := duplex.New(duplex.Config{
		Prompt: prompt,
		Voice:  speech.SpeakOptions{Voice: *voice, Language: *language},
		Listen: speech.Options{Language: *language},
		Reply:  chat.Options{Temperature: 0.7, TopP: 0.9, MaxTokens: 160},
		// Alone with one person Gopher answers everything; in a meeting,
		// only what is addressed to it.
		Addressed: func(text string) bool {
			return humans.count() <= 1 || strings.Contains(strings.ToLower(text), "gopher")
		},
		OnText: func(role chat.Role, text string) {
			who := "🧑"
			if role == chat.Assistant {
				who = "🐹 Gopher:"
			}
			fmt.Printf("%s %s\n", who, text)
			if room != nil {
				go room.send(who + " " + text)
			}
		},
		OnError: func(err error) { log.Printf("agent: %v", err) },
		OnStage: func(stage string, elapsed time.Duration) {
			if *verbose {
				log.Printf("reply %s after %v", stage, elapsed.Round(time.Millisecond))
			}
		},
	}, models...)
	check(err)
	defer agent.Close()

	sdkLog := logrus.New()
	sdkLog.SetLevel(logrus.ErrorLevel)
	if *verbose {
		sdkLog.SetLevel(logrus.WarnLevel)
	}
	client, err := rtc.NewClient(apiKey, rtc.User{ID: self, Name: "Gopher 🐹"}, rtc.StaticToken(token),
		rtc.WithLogger(logger.FromLogrus(sdkLog)))
	check(err)
	defer client.Close()
	call := client.Call(callType, callID)
	mix := newMixer()
	join, err := call.Join(ctx, rtc.WithOnTrack(rtc.SubscriberFunc(func(t rtc.OnTrackReceived) {
		if t.TrackType == sfu_models.TrackType_TRACK_TYPE_AUDIO {
			go hear(ctx, t, mix, humans)
		}
	})))
	check(err)
	defer call.Leave("done")
	room = &chatChannel{apiKey: apiKey, token: token, path: "/channels/videocall/" + url.PathEscape(callID)}
	if err := room.open(); err != nil {
		log.Printf("chat is off: %v", err)
		room = nil
	}

	// Gopher's voice: the writer encodes and paces what the agent says.
	writer, err := audiortc.NewTrackWriter(audiortc.WriterConfig{})
	check(err)
	info := &sfu_models.TrackInfo{TrackId: "gopher-voice-" + randomHex(4), TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO}
	// Opus is always negotiated as two channels (RFC 7587); the stream
	// itself is mono. Advertising one channel makes the SFU reject it.
	codec := writer.Codec()
	codec.Channels = 2
	voiceTrack, err := track.NewAudioTrack(info, writer, codec)
	check(err)
	_, err = call.AddTrack(info, voiceTrack)
	check(err)
	// Tell the SFU the microphone is live, or clients treat Gopher as muted.
	if _, err := call.Client().UpdateMuteStates(ctx, &signal_rpc.UpdateMuteStatesRequest{SessionId: call.SessionID.Load(),
		MuteStates: []*signal_rpc.TrackMuteState{{TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO, Muted: false}}}); err != nil {
		log.Printf("unmute: %v", err)
	}


	// Hear every microphone: those already live, then each that starts.
	var mics []*signal_rpc.TrackSubscriptionDetails
	subscribe := func(user, session string) {
		for _, m := range mics {
			if m.SessionId == session {
				return
			}
		}
		mics = append(mics, &signal_rpc.TrackSubscriptionDetails{UserId: user, SessionId: session,
			TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO})
		if err := call.SubscribeToTracks(ctx, mics...); err != nil {
			log.Printf("subscribe: %v", err)
		}
	}
	for _, p := range join.GetCallState().GetParticipants() {
		for _, t := range p.GetPublishedTracks() {
			if t == sfu_models.TrackType_TRACK_TYPE_AUDIO && p.GetUserId() != self {
				subscribe(p.GetUserId(), p.GetSessionId())
			}
		}
	}
	defer rtc.HandleCallEvent(call, func(e *sfu_events.SfuEvent_TrackPublished) {
		if p := e.TrackPublished; p.GetType() == sfu_models.TrackType_TRACK_TYPE_AUDIO && p.GetUserId() != self {
			subscribe(p.GetUserId(), p.GetSessionId())
		}
	})()

	fmt.Printf("🐹 Gopher is in the call. Join and say hi: %s/join/%s?type=%s\n", *pronto, url.PathEscape(callID), callType)
	converse(ctx, agent, mix, writer)
}

// converse runs the agent on the call's clock: every 20 ms the room's audio
// goes in and the agent's speech comes out.
func converse(ctx context.Context, agent speech.Duplex, mix *mixer, writer *audiortc.TrackWriter) {
	inSize, outSize := agent.Frame()
	_, outRate := agent.Rates()
	in, out := make([]float32, inSize), make([]float32, outSize)
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	was := speech.Listening
	var peak float32 // loudest input of the last second, for the log
	for frames := 1; ; frames++ {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		mix.read(in)
		for _, v := range in {
			peak = max(peak, v, -v)
		}
		state, err := agent.Step(ctx, in, out)
		if err != nil {
			return
		}
		if *verbose && frames%50 == 0 {
			if peak > 0.001 {
				log.Printf("input peak %.1f dBFS, agent %v", 20*math.Log10(float64(peak)), state)
			}
			peak = 0
		}
		if state != was {
			log.Printf("agent %v", state)
		}
		switch {
		case state == speech.Speaking:
			writer.Write(webaudio.FromFloat32(out, outRate, 1))
		case was == speech.Speaking:
			// Cut off or finished: drop what the encoder still holds.
			writer.Clear()
		}
		was = state
	}
}

// hear decodes one participant's microphone into the mixer.
func hear(ctx context.Context, t rtc.OnTrackReceived, mix *mixer, humans *roster) {
	id := string(t.ParticipantID.UserID)
	humans.add(id)
	defer humans.remove(id)
	// A person can publish more than one microphone track (a second tab,
	// a re-publish): each gets its own queue so their audio never
	// interleaves.
	key := id + "/" + t.Track.ID()
	log.Printf("hearing %s (track %s)", id, t.Track.ID())
	defer mix.drop(key)
	reader, err := audiortc.NewTrackReader(t.Track, audiortc.ReaderConfig{Opus: opus.Config{SampleRate: speech.SampleRate}})
	if err != nil {
		log.Printf("hear %s: %v", id, err)
		return
	}
	defer reader.Close()
	for frame, err := range reader.Frames() {
		if err != nil || ctx.Err() != nil {
			return
		}
		mix.write(key, frame.Float32())
	}
}

// mixer sums the participants' audio on the agent's clock. Each track has
// a queue that doubles as a jitter buffer: a read takes a whole frame from
// each queue that has one and adds them. A queue short of a frame, because
// its packet is late, sits this frame out rather than splicing silence into
// speech, and one that falls behind the clock is trimmed so latency stays
// low.
type mixer struct {
	mu     sync.Mutex
	queues map[string][]float32
}

func newMixer() *mixer { return &mixer{queues: map[string][]float32{}} }

const maxQueue = speech.SampleRate / 5 // 200 ms

func (m *mixer) write(id string, pcm []float32) {
	m.mu.Lock()
	q := append(m.queues[id], pcm...)
	if over := len(q) - maxQueue; over > 0 {
		q = q[over:]
	}
	m.queues[id] = q
	m.mu.Unlock()
}

func (m *mixer) drop(key string) {
	m.mu.Lock()
	delete(m.queues, key)
	m.mu.Unlock()
}

func (m *mixer) read(dst []float32) {
	clear(dst)
	m.mu.Lock()
	for key, q := range m.queues {
		if len(q) < len(dst) {
			continue
		}
		for i, v := range q[:len(dst)] {
			dst[i] += v
		}
		m.queues[key] = q[:copy(q, q[len(dst):])]
	}
	m.mu.Unlock()
}

// roster counts the people in the call.
type roster struct {
	mu  sync.Mutex
	ids map[string]int
}

func (r *roster) add(id string) {
	r.mu.Lock()
	if r.ids == nil {
		r.ids = map[string]int{}
	}
	r.ids[id]++
	r.mu.Unlock()
}

func (r *roster) remove(id string) {
	r.mu.Lock()
	if r.ids[id]--; r.ids[id] <= 0 {
		delete(r.ids, id)
	}
	r.mu.Unlock()
}

func (r *roster) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.ids)
}

// prontoToken asks a Pronto deployment for a user token for its app.
func prontoToken(pronto, user string) (apiKey, token string, err error) {
	resp, err := http.Get(pronto + "/api/auth/create-token?environment=pronto&user_id=" + url.QueryEscape(user))
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	type credentials struct{ APIKey, Token string }
	var v credentials
	dec, err := vibejson.CompileDecoder[credentials](vibejson.DecoderOptions{})
	if err != nil {
		return "", "", err
	}
	r := vibejson.NewReader(resp.Body)
	if !vibejson.DecodeNext(r, dec, &v) || v.Token == "" {
		return "", "", fmt.Errorf("pronto token: %s: %v", resp.Status, r.Err())
	}
	return v.APIKey, v.Token, nil
}

// chatChannel posts to one Stream Chat channel as the token's user, through
// the chat REST API.
type chatChannel struct{ apiKey, token, path string }

// open creates the channel if nobody has opened the call's chat yet.
func (c *chatChannel) open() error { return c.post("/query", map[string]any{"state": false}) }

func (c *chatChannel) send(text string) error {
	return c.post("/message", map[string]any{"message": map[string]any{"text": text}})
}

func (c *chatChannel) post(endpoint string, body any) error {
	payload, err := vibejson.Marshal(&body)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost,
		"https://chat.stream-io-api.com"+c.path+endpoint+"?api_key="+url.QueryEscape(c.apiKey), bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", c.token)
	req.Header.Set("Stream-Auth-Type", "jwt")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s: %s", resp.Status, msg)
	}
	return nil
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
