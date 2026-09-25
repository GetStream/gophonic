// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command streamcall joins a Pronto call (Stream's video app) as a silent
// listener. For each speaker it finds the end of every turn with Smart Turn
// v3.2, transcribes the turn with Qwen3-ASR in whatever language they speak,
// and reads how they feel with Qwen3-8B, all in-process with gophonic, then
// posts the result to the call's chat. Audio never leaves the machine, and
// nothing is published into the call.
//
//	go run .   # then open the printed link and talk
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
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"slices"
	"strings"
	"sync"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/thesyncim/gopus"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

const (
	rate     = speech.SampleRate
	frame    = rate / 50              // 20 ms, the voice detector's unit
	voiced   = 96                     // Opus SILK speech activity (of 255) that counts as speech
	quiet    = 0.0056                 // RMS below which a frame is silence whatever the VAD says (-45 dBFS)
	pause    = 200 * time.Millisecond // silence after which Smart Turn is asked, and again as it grows
	giveUp   = 2 * time.Second        // silence that ends a turn whatever Smart Turn says
	shortest = 400 * time.Millisecond // speech a turn needs before it is transcribed
	self     = "gophonic"             // our user ID; we never subscribe to ourselves
	step     = 20 * time.Millisecond
)

func main() {
	callFlag := flag.String("call", "", "call to join as type:id (default: a new call)")
	turnPath := flag.String("turn", "models/smart-turn-v3.2.gophonic", "turn-detection model")
	sttPath := flag.String("stt", "models/Qwen3-ASR-1.7B", "transcription model")
	llmPath := flag.String("llm", "models/Qwen3-8B", "Qwen3-8B snapshot for sentiment")
	pronto := flag.String("pronto", "https://pronto-staging.getstream.io", "Pronto app whose call to join")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	turn, err := gophonic.Open(*turnPath, gophonic.Options{})
	check(err)
	stt, err := gophonic.Open(*sttPath, gophonic.Options{})
	check(err)

	// Pronto hands out user tokens for its app, so no Stream credentials are needed.
	apiKey, token, err := prontoToken(*pronto, self)
	check(err)
	client, err := rtc.NewClient(apiKey, rtc.User{ID: self, Name: "gophonic"}, rtc.StaticToken(token))
	check(err)
	defer client.Close()
	callType, callID, ok := strings.Cut(*callFlag, ":")
	if !ok {
		callType, callID = "default", "gophonic-"+randomHex(3)
	}
	call := client.Call(callType, callID)
	// Results go to the call's chat, the channel Pronto shows beside the video.
	chat := &chatChannel{apiKey: apiKey, token: token, path: "/channels/videocall/" + url.PathEscape(callID)}
	if err := chat.open(); err != nil {
		log.Printf("chat is off: %v", err)
		chat = nil
	}
	// Any model that answers questions about text will do; Qwen3 does.
	llm, err := gophonic.Open(*llmPath, gophonic.Options{})
	check(err)
	defer llm.Close()
	zeroShot, err := gophonic.Lane[speech.ZeroShot](llm)
	check(err)
	mood, err := zeroShot.Classifier("What is the speaker's mood?", moods)
	check(err)
	judge := &analyst{mood: mood, probs: make([]float32, len(moods)), chat: chat}
	join, err := call.Join(ctx, rtc.WithOnTrack(rtc.SubscriberFunc(func(t rtc.OnTrackReceived) {
		if t.TrackType == sfu_models.TrackType_TRACK_TYPE_AUDIO {
			go listen(ctx, t, turn, stt, judge)
		}
	})))
	check(err)
	defer call.Leave("done")

	// The SFU forwards only what we ask for: every microphone already live,
	// then each one that starts later.
	var mics []*signal_rpc.TrackSubscriptionDetails
	subscribe := func(user, session string) {
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

	fmt.Printf("gophonic is listening. Join and talk: %s/join/%s?type=%s\n", *pronto, url.PathEscape(callID), callType)
	<-ctx.Done()
}

// listen follows one speaker: voice activity marks speech, a pause asks
// Smart Turn whether the turn is over every 200 ms while it lasts, and a
// finished turn is transcribed and judged.
func listen(ctx context.Context, t rtc.OnTrackReceived, turn, stt *gophonic.Model, judge *analyst) {
	name := string(t.ParticipantID.UserID)
	if t.Participant != nil && t.Participant.Name != "" {
		name = t.Participant.Name
	}
	// gopus decodes straight to 16 kHz, resampling inside the Opus decoder.
	reader, err := audiortc.NewTrackReader(t.Track, audiortc.ReaderConfig{Opus: opus.Config{SampleRate: rate}})
	check(err)
	defer reader.Close()
	detector, err := turn.NewTurnDetector()
	check(err)
	defer detector.Close()
	transcriber, err := stt.NewTranscriber()
	check(err)
	defer transcriber.Close()
	vad, err := gopus.NewVAD(rate)
	check(err)

	var (
		pending         []float32 // decoded audio not yet seen by the voice detector
		pcm16           [frame]int16
		preroll         []float32 // the last 300 ms before anyone speaks: VAD fires late
		spoken          []float32 // the current turn
		talked, silence time.Duration
		transcript      speech.Transcript
	)
	for decoded, err := range reader.Frames() {
		if err != nil || ctx.Err() != nil {
			return
		}
		pending = append(pending, decoded.Float32()...)
		for ; len(pending) >= frame; pending = pending[frame:] {
			samples := pending[:frame]
			var energy float32
			for i, s := range samples {
				pcm16[i] = int16(max(-1, min(1, s)) * 32767)
				energy += s * s
			}
			activity, _ := vad.AnalyzeInt16(pcm16[:])
			if energy < quiet*quiet*frame {
				activity = 0 // room noise the detector still hears as speech
			}
			if len(spoken) == 0 {
				if activity < voiced { // nobody is talking yet
					preroll = append(preroll, samples...)
					preroll = preroll[max(0, len(preroll)-15*frame):]
					continue
				}
				spoken, preroll = append(spoken, preroll...), preroll[:0]
			}
			spoken = append(spoken, samples...)
			if activity >= voiced {
				talked, silence = talked+step, 0
				continue
			}
			if silence += step; silence%pause != 0 && silence < giveUp {
				continue
			}
			start := time.Now()
			p, err := detector.PredictInto(spoken, rate, 1)
			if err != nil || (!p.Complete && silence < giveUp) {
				if err == nil && talked >= shortest && silence == pause {
					fmt.Printf("  · %s paused, not done yet (%.2f)\n", name, p.Probability)
				}
				continue // wait for more speech or a longer pause
			}
			judged := time.Since(start)
			if talked >= shortest {
				start = time.Now()
				if transcriber.Transcribe(ctx, spoken, speech.Options{}, &transcript) == nil {
					if text := strings.TrimSpace(string(transcript.Text)); text != "" {
						fmt.Printf("\n● %s (%s): %q\n  turn over %.2f (Smart Turn %v) · transcribed in %v\n", name,
							transcript.Language, text, p.Probability, judged.Round(time.Millisecond), time.Since(start).Round(time.Millisecond))
						go judge.analyze(ctx, name, transcript.Language, text, p.Probability)
					}
				}
			}
			spoken, talked, silence = spoken[:0], 0, 0
		}
	}
}

// analyst judges each finished turn with a zero-shot text classifier: for
// Qwen3-8B, one prefill of the turn against a prepared multiple-choice
// question, then a softmax over the answer letters. One classifier serves
// every speaker, one turn at a time.
type analyst struct {
	mu    sync.Mutex
	mood  speech.TextClassifier
	probs []float32
	chat  *chatChannel // nil when the call has no chat
}

// moods are the answers Qwen3-8B chooses between. Asking for the speaker's
// mood, with a calm option first, keeps ordinary speech neutral; asking which
// emotion they express reads anger into it.
var (
	moods = []string{"calm or neutral", "happy or excited", "frustrated or angry", "sad", "worried or anxious"}
	faces = []string{"😐", "😄", "😠", "😢", "😟"}
)

func (a *analyst) analyze(ctx context.Context, speaker, language, text string, turn float32) {
	a.mu.Lock()
	start := time.Now()
	err := a.mood.ClassifyInto(ctx, text, a.probs)
	best := slices.Index(a.probs, slices.Max(a.probs))
	feeling := fmt.Sprintf("%s %s %.2f (Qwen3-8B, %v)", faces[best], moods[best], a.probs[best],
		time.Since(start).Round(time.Millisecond))
	a.mu.Unlock()
	if err != nil {
		log.Printf("sentiment: %v", err)
		return
	}
	fmt.Printf("  %s\n", feeling)
	if a.chat == nil {
		return
	}
	// One message per turn: who spoke, what they said and in which language,
	// that the turn is over, and how they feel.
	msg := fmt.Sprintf("**%s** finished their turn (Smart Turn %.2f)\n> %s\n%s · %s", speaker, turn, text, language, feeling)
	if err := a.chat.send(msg); err != nil {
		log.Printf("chat: %v", err)
	}
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
