// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command gopher joins a Pronto call (Stream's video app) as a voice
// agent you can just talk to. It listens to everyone, answers out loud,
// lets you interrupt it, and knows when you have not finished, with every
// model running in this process: Smart Turn and Qwen3-ASR to listen,
// Qwen3.6-35B-A3B to think, and Qwen3-TTS to speak.
//
//	go run .   # then open the printed link and talk
//
// The agent is a speech.Duplex: the call's audio goes in and the agent's
// speech comes out, 20 ms at a time, on one clock. There are no turns in
// this program; the duplex decides when to speak.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	webaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/GetStream/getstream-go-webrtc/logger"
	"github.com/GetStream/getstream-go-webrtc/track"
	getstream "github.com/GetStream/getstream-go/v5"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/sirupsen/logrus"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/duplex"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

const self = "gopher" // our user ID; we never subscribe to ourselves

var verbose *bool

func main() {
	callFlag := flag.String("call", "", "call to join as type:id (default: a new call)")
	asrPath := flag.String("asr", "../../models/Qwen3-ASR-1.7B", "speech recognition model")
	turnPath := flag.String("turn", "", "turn detection model, for a speech recognizer that does not judge turns itself as Qwen3-ASR-1.7B does")
	llmPath := flag.String("llm", "../../models/Qwen3.6-35B-A3B", "language model")
	ttsPath := flag.String("tts", "../../models/Qwen3-TTS-12Hz-1.7B-CustomVoice", "speech synthesis model")
	language := flag.String("language", "", "language spoken in the call (ISO 639-1); empty detects it, and Gopher answers in kind")
	languages := flag.String("languages", "", "languages spoken in the call, comma-separated ISO 639-1 codes such as en,pt: what is heard is transcribed in one of them, and Gopher answers in kind")
	voice := flag.String("voice", "aiden", "voice: aiden, ryan, serena, vivian, eric, dylan, uncle_fu, ono_anna, sohee")
	pronto := flag.String("pronto", "https://pronto-staging.getstream.io", "Pronto app whose call to join")
	verbose = flag.Bool("v", false, "log the input level every second")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Load the models at once; each maps its cached weights.
	start := time.Now()
	paths := []string{*asrPath, *llmPath, *ttsPath}
	if *turnPath != "" {
		paths = append(paths, *turnPath)
	}
	models := make([]*gophonic.Model, len(paths))
	var wg sync.WaitGroup
	for i, path := range paths {
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
	var live atomic.Pointer[liveCaptions] // the chat's captions, when closed captions are off
	humans := &roster{}
	present := &people{}
	mix := newMixer()
	// Closed captions are a server-side API: with the app's secret, what is
	// said appears as the call's captions; without it, in the chat.
	captions := newCaptions(apiKey, callType, callID)
	var spoken speech.Language
	if *language != "" {
		var ok bool
		if spoken, ok = speech.ParseLanguage(*language); !ok {
			log.Fatalf("unknown language %q", *language)
		}
	}
	allowed, err := speech.ParseLanguages(*languages)
	if err != nil {
		log.Fatal(err)
	}
	cfg := config(*voice, spoken, allowed)
	// Captions follow the voice: closed captions sentence by sentence with
	// the app's secret; otherwise the call's chat, where each answer is one
	// message that grows as it is spoken.
	cfg.Observer = &observer{captions: captions, live: &live}
	agent, err := duplex.New(cfg, models...)
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
	if room != nil && captions.call == nil {
		live.Store(newLiveCaptions(room))
	}
	if room != nil {
		// The call's chat is silent context: what people type joins the
		// conversation without being spoken or answered, so a later
		// question or summary can draw on it. Gopher's own captions and
		// replies land in the same channel, so its own messages are
		// skipped; they are already in the conversation.
		if err := room.watch(ctx, func(userID, name, text string) {
			if userID == self {
				return
			}
			if name == "" {
				name = userID
			}
			present.learn(userID, name)
			if err := agent.Note(present.name(userID) + " wrote in the call's chat: " + text); err != nil {
				log.Printf("chat: %v", err)
			}
		}); err != nil {
			log.Printf("chat history and live updates are off: %v", err)
		}
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

	// Hear every microphone: those already live, then each that starts,
	// stops, or changes. A browser re-publishes its microphone on mute,
	// device changes, and rejoins; each time the subscription is sent
	// anew (without the session, then with it) so the SFU renegotiates
	// the new stream instead of forwarding packets Gopher cannot route.
	mics := map[string]*signal_rpc.TrackSubscriptionDetails{}
	var micMu sync.Mutex
	resubscribe := func(changed string) {
		micMu.Lock()
		defer micMu.Unlock()
		list := func(skip string) []*signal_rpc.TrackSubscriptionDetails {
			var out []*signal_rpc.TrackSubscriptionDetails
			for session, d := range mics {
				if session != skip {
					out = append(out, d)
				}
			}
			return out
		}
		if changed != "" {
			if err := call.SubscribeToTracks(ctx, list(changed)...); err != nil {
				log.Printf("subscribe: %v", err)
			}
		}
		if err := call.SubscribeToTracks(ctx, list("")...); err != nil {
			log.Printf("subscribe: %v", err)
		}
	}
	// Who is in the call, and who comes and goes, reaches the agent as
	// notes, by name, as the chat does.
	arrive := func(p *sfu_models.Participant, note string) {
		if id := p.GetUserId(); id != self && present.arrive(id, p.GetName()) {
			if err := agent.Note(present.name(id) + note); err != nil {
				log.Printf("agent: %v", err)
			}
		}
	}
	defer rtc.HandleCallEvent(call, func(e *sfu_events.SfuEvent_ParticipantJoined) {
		arrive(e.ParticipantJoined.GetParticipant(), " joined the call.")
	})()
	defer rtc.HandleCallEvent(call, func(e *sfu_events.SfuEvent_ParticipantLeft) {
		if id := e.ParticipantLeft.GetParticipant().GetUserId(); id != self && present.leave(id) {
			if err := agent.Note(present.name(id) + " left the call."); err != nil {
				log.Printf("agent: %v", err)
			}
		}
	})()
	for _, p := range join.GetCallState().GetParticipants() {
		arrive(p, " is in the call.")
		for _, t := range p.GetPublishedTracks() {
			if t == sfu_models.TrackType_TRACK_TYPE_AUDIO && p.GetUserId() != self {
				mics[p.GetSessionId()] = &signal_rpc.TrackSubscriptionDetails{UserId: p.GetUserId(), SessionId: p.GetSessionId(),
					TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO}
			}
		}
	}
	resubscribe("")
	defer rtc.HandleCallEvent(call, func(e *sfu_events.SfuEvent_TrackPublished) {
		if p := e.TrackPublished; p.GetType() == sfu_models.TrackType_TRACK_TYPE_AUDIO && p.GetUserId() != self {
			micMu.Lock()
			mics[p.GetSessionId()] = &signal_rpc.TrackSubscriptionDetails{UserId: p.GetUserId(), SessionId: p.GetSessionId(),
				TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO}
			micMu.Unlock()
			log.Printf("%s published a microphone", p.GetUserId())
			resubscribe(p.GetSessionId())
		}
	})()
	defer rtc.HandleCallEvent(call, func(e *sfu_events.SfuEvent_TrackUnpublished) {
		if p := e.TrackUnpublished; p.GetType() == sfu_models.TrackType_TRACK_TYPE_AUDIO && p.GetUserId() != self {
			micMu.Lock()
			delete(mics, p.GetSessionId())
			micMu.Unlock()
			log.Printf("%s unpublished a microphone", p.GetUserId())
			resubscribe("")
		}
	})()

	fmt.Printf("🐹 Gopher is in the call. Join and say hi: %s/join/%s?type=%s\n", *pronto, url.PathEscape(callID), callType)
	converse(ctx, agent, mix, humans, present, writer)
}

// converse runs the agent on the call's clock: every 20 ms the room's audio
// goes in and the agent's speech comes out. In a meeting, the loudest
// participant is named as the speaker, so the conversation knows whose
// words it hears.
func converse(ctx context.Context, agent *duplex.Cascade, mix *mixer, humans *roster, present *people, writer *audiortc.TrackWriter) {
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
		heard := in
		if !mix.read(in) {
			heard = nil // a packet is late: a gap, not silence
		}
		for _, v := range heard {
			peak = max(peak, v, -v)
		}
		if frames%10 == 0 {
			if humans.count() > 1 {
				agent.Speaker(present.name(mix.loudest()))
			} else {
				agent.Speaker("")
			}
		}
		state, err := agent.Step(heard, out)
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
	energy map[string]float32 // recent loudness per track, decaying
	missed map[string]int     // frames each track has been late in a row
}

func newMixer() *mixer {
	return &mixer{queues: map[string][]float32{}, energy: map[string]float32{}, missed: map[string]int{}}
}

// maxGap is how long a track that was sending may be late before its
// silence is taken for real: longer, and the sender stopped (a muted or
// quiet microphone sends nothing).
const maxGap = 3 // frames: 60 ms

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
	delete(m.energy, key)
	delete(m.missed, key)
	m.mu.Unlock()
}

// read mixes the next frame of every track into dst. It reports false,
// consuming nothing, when a track that was sending is late: the frame is
// not known yet, and its absence is not silence. The audio still arrives,
// and is heard in order, at most maxGap frames later.
func (m *mixer) read(dst []float32) bool {
	clear(dst)
	m.mu.Lock()
	defer m.mu.Unlock()
	late := false
	for key, q := range m.queues {
		if len(q) >= len(dst) {
			continue
		}
		if m.missed[key] < maxGap {
			late = true
		}
		m.missed[key]++
	}
	if late {
		return false
	}
	for key, q := range m.queues {
		if len(q) < len(dst) {
			continue
		}
		m.missed[key] = 0
		var e float32
		for i, v := range q[:len(dst)] {
			dst[i] += v
			e += v * v
		}
		m.energy[key] = 0.95*m.energy[key] + e
		m.queues[key] = q[:copy(q, q[len(dst):])]
	}
	return true
}

// loudest returns the user whose audio was loudest recently, the likely
// speaker of what was just transcribed.
func (m *mixer) loudest() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	best, who := float32(0), self
	for key, e := range m.energy {
		if e > best {
			best, who = e, key
		}
	}
	id, _, _ := strings.Cut(who, "/")
	return id
}

// observer shows what is said: as the call's closed captions with the
// app's secret, otherwise in its chat, and on the terminal.
type observer struct {
	duplex.Base
	captions *captions
	live     *atomic.Pointer[liveCaptions]
}

func (o *observer) Heard(speaker string, text []byte, final bool) {
	if !final {
		return
	}
	if speaker == "" {
		speaker = "you"
	}
	o.captions.show(speaker, string(text))
	if l := o.live.Load(); l != nil {
		l.say(speaker + ": " + string(text))
	}
	fmt.Printf("%s: %s\n", speaker, text)
}

func (o *observer) Said(text []byte, voiced int, final bool) {
	shown := string(text[:voiced])
	if final {
		shown = string(text)
	}
	o.captions.assistant(shown, final)
	if l := o.live.Load(); l != nil {
		l.answer("Gopher: "+shown, final)
	}
	if final {
		fmt.Printf("Gopher: %s\n", text)
	}
}

func (o *observer) Stage(s duplex.Stage, elapsed time.Duration) {
	if *verbose {
		log.Printf("reply %s after %v", s, elapsed.Round(time.Millisecond))
	}
}

func (o *observer) Error(err error) { log.Printf("agent: %v", err) }

// captions sends what is said as the call's closed captions.
type captions struct {
	call *getstream.Call // nil without the app's secret
	send chan getstream.SendClosedCaptionRequest
	sent int // bytes of the current reply already captioned
}

func newCaptions(apiKey, callType, callID string) *captions {
	c := &captions{}
	secret := os.Getenv("STREAM_API_SECRET")
	if secret == "" {
		log.Printf("closed captions are off: set STREAM_API_SECRET to show them")
		return c
	}
	client, err := getstream.NewClient(apiKey, secret)
	if err != nil {
		log.Printf("closed captions are off: %v", err)
		return c
	}
	c.call = client.Video().Call(callType, callID)
	c.send = make(chan getstream.SendClosedCaptionRequest, 64)
	go func() { // one request at a time, in order
		for req := range c.send {
			if _, err := c.call.SendClosedCaption(context.Background(), &req); err != nil {
				log.Printf("caption: %v", err)
			}
		}
	}()
	log.Printf("closed captions are on")
	return c
}

func (c *captions) show(speaker, text string) {
	if c.call == nil || strings.TrimSpace(text) == "" {
		return
	}
	select {
	case c.send <- getstream.SendClosedCaptionRequest{SpeakerID: speaker, Text: strings.TrimSpace(text)}:
	default: // captions are falling behind; drop rather than delay the call
	}
}

// assistant captions each finished sentence of the reply growing in text.
func (c *captions) assistant(text string, final bool) {
	if c.sent > len(text) {
		c.sent = 0 // a new reply
	}
	rest := text[c.sent:]
	end := strings.LastIndexAny(rest, ".!?…")
	if final {
		end = len(rest) - 1
	}
	if end >= 0 {
		c.show(self, rest[:end+1])
		c.sent += end + 1
	}
	if final {
		c.sent = 0
	}
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

// people names the call's participants, from the call and its chat, and
// counts their sessions, so that a second tab is not a second arrival.
type people struct {
	mu       sync.Mutex
	names    map[string]string // user ID → display name
	sessions map[string]int
}

// name returns id's display name, or id until one is known.
func (p *people) name(id string) string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if n := p.names[id]; n != "" {
		return n
	}
	return id
}

// learn records id's display name.
func (p *people) learn(id, name string) {
	if name = strings.TrimSpace(name); name == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.names == nil {
		p.names = map[string]string{}
	}
	p.names[id] = name
}

// arrive counts a session of id, reporting whether id just came.
func (p *people) arrive(id, name string) bool {
	p.learn(id, name)
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.sessions == nil {
		p.sessions = map[string]int{}
	}
	p.sessions[id]++
	return p.sessions[id] == 1
}

// leave ends a session of id, reporting whether id is gone.
func (p *people) leave(id string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.sessions[id] == 0 {
		return false
	}
	p.sessions[id]--
	return p.sessions[id] == 0
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
