// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command e2e checks Gopher end to end, through the SFU: it joins the call
// as a test participant, speaks recorded speech, and listens for Gopher's
// voice, reporting whether it answered and how long after the voice ended.
// Each round waits for Gopher to be quiet first. It loads no models; talk
// over Gopher is tested in the scenarios, which see its decisions.
//
//	go run ./e2e -call default:ID
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	webaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/GetStream/getstream-go-webrtc/track"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/thesyncim/vibejson"
)

func main() {
	callFlag := flag.String("call", "", "call to join as type:id")
	pronto := flag.String("pronto", "https://pronto-staging.getstream.io", "Pronto app")
	audioPath := flag.String("audio", "../../testdata/whisper_jfk.pcm.f32le", "speech to say: mono 16 kHz float32 PCM")
	seconds := flag.Float64("seconds", 11, "how much of the speech to say")
	flag.Parse()
	callType, callID, _ := strings.Cut(*callFlag, ":")
	raw, err := os.ReadFile(*audioPath)
	check(err)
	speechPCM := make([]float32, min(len(raw)/4, int(*seconds*16000)))
	for i := range speechPCM {
		speechPCM[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	// Where the voice starts and ends within the speech, as the clips were
	// cut: 20 ms frames louder than 5% of the loudest. Latency counts from
	// the end of the voice, as a listener hears it, not of the silence
	// after it.
	voiceStart, voiceEnd := voiced(speechPCM)
	ctx := context.Background()
	apiKey, token := prontoToken(*pronto, "gopher-tester")
	client, err := rtc.NewClient(apiKey, rtc.User{ID: "gopher-tester", Name: "Tester"}, rtc.StaticToken(token))
	check(err)
	defer client.Close()
	call := client.Call(callType, callID)

	// Gopher's voice: energy per 20 ms, with arrival times.
	var mu sync.Mutex
	var loud []time.Time
	onTrack := rtc.SubscriberFunc(func(t rtc.OnTrackReceived) {
		if t.TrackType != sfu_models.TrackType_TRACK_TYPE_AUDIO || t.ParticipantID.UserID != "gopher" {
			return
		}
		log.Printf("hearing Gopher")
		reader, err := audiortc.NewTrackReader(t.Track, audiortc.ReaderConfig{Opus: opus.Config{SampleRate: 16000}})
		if err != nil {
			log.Print(err)
			return
		}
		for frame, err := range reader.Frames() {
			if err != nil {
				return
			}
			var e float64
			for _, v := range frame.Float32() {
				e += float64(v * v)
			}
			if rms := math.Sqrt(e / float64(max(1, len(frame.Float32())))); rms > 0.01 {
				mu.Lock()
				loud = append(loud, time.Now())
				mu.Unlock()
			}
		}
	})
	join, err := call.Join(ctx, rtc.WithOnTrack(onTrack))
	check(err)
	defer func() { call.Leave("done") }()
	subscribeGopher(call, join)

	// A microphone, published once; rounds then mute and unmute it or
	// rejoin, as a browser does.
	writer, err := audiortc.NewTrackWriter(audiortc.WriterConfig{})
	check(err)
	publish := func(call *rtc.Call) {
		info := &sfu_models.TrackInfo{TrackId: fmt.Sprintf("tester-mic-%d", time.Now().UnixNano()), TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO}
		codec := writer.Codec()
		codec.Channels = 2
		local, err := track.NewAudioTrack(info, writer, codec)
		check(err)
		_, err = call.AddTrack(info, local)
		check(err)
		mute(call, false)
	}
	publish(call)
	// quiet waits until Gopher has been silent for a while, so that a round
	// measures a turn rather than an answer to the round before.
	quiet := func() {
		time.Sleep(3 * time.Second) // a mute or a rejoin settles, and Gopher subscribes
		for deadline := time.Now().Add(30 * time.Second); time.Now().Before(deadline); time.Sleep(100 * time.Millisecond) {
			mu.Lock()
			last := time.Time{}
			if len(loud) > 0 {
				last = loud[len(loud)-1]
			}
			mu.Unlock()
			if time.Since(last) > 2*time.Second {
				return
			}
		}
	}
	// speak says the speech in real time and returns when it began and ended.
	speak := func() (started, ended time.Time) {
		tick := time.NewTicker(20 * time.Millisecond)
		defer tick.Stop()
		started = time.Now()
		for pos := 0; pos+320 <= len(speechPCM); pos += 320 {
			<-tick.C
			writer.Write(webaudio.FromFloat32(speechPCM[pos:pos+320], 16000, 1))
		}
		writer.Flush()
		return started, time.Now()
	}
	say := func(round int, what string) {
		quiet()
		started, _ := speak()
		began, ended := started.Add(voiceStart), started.Add(voiceEnd)
		time.Sleep(6 * time.Second)
		mu.Lock()
		var first time.Time
		var over time.Duration // Gopher's voice while the speech went on, at its pauses
		for _, t := range loud {
			if t.After(began) && t.Before(ended) {
				over += 20 * time.Millisecond
			}
			if t.After(ended) {
				first = t
				break
			}
		}
		mu.Unlock()
		if over > 0 {
			fmt.Printf("round %d (%s): Gopher spoke for %v before the voice ended\n", round, what, over)
		}
		if first.IsZero() {
			fmt.Printf("round %d (%s): Gopher did not answer\n", round, what)
		} else {
			fmt.Printf("round %d (%s): Gopher answered %v after the voice ended\n", round, what, first.Sub(ended).Round(10*time.Millisecond))
		}
	}
	say(0, "first utterance")
	mute(call, true)
	time.Sleep(2 * time.Second)
	mute(call, false)
	say(1, "after mute and unmute")
	call.Leave("rejoin")
	call = client.Call(callType, callID)
	join, err = call.Join(ctx, rtc.WithOnTrack(onTrack))
	check(err)
	subscribeGopher(call, join)
	publish(call)
	say(2, "after rejoining")

}

// voiced returns where the voice in pcm, mono 16 kHz, starts and ends: the
// first and last 20 ms frames louder than 5% of the loudest.
func voiced(pcm []float32) (start, end time.Duration) {
	const frame = 320
	rms := make([]float64, len(pcm)/frame)
	loudest := 0.0
	for i := range rms {
		var e float64
		for _, v := range pcm[i*frame : (i+1)*frame] {
			e += float64(v) * float64(v)
		}
		rms[i] = math.Sqrt(e / frame)
		loudest = max(loudest, rms[i])
	}
	first, last := -1, -1
	for i, r := range rms {
		if r > max(0.05*loudest, 0.003) {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if first < 0 {
		return 0, 0
	}
	return time.Duration(first) * 20 * time.Millisecond, time.Duration(last+1) * 20 * time.Millisecond
}

func mute(call *rtc.Call, muted bool) {
	_, err := call.Client().UpdateMuteStates(context.Background(), &signal_rpc.UpdateMuteStatesRequest{SessionId: call.SessionID.Load(),
		MuteStates: []*signal_rpc.TrackMuteState{{TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO, Muted: muted}}})
	check(err)
}

func subscribeGopher(call *rtc.Call, join *sfu_events.JoinResponse) {
	var subs []*signal_rpc.TrackSubscriptionDetails
	for _, p := range join.GetCallState().GetParticipants() {
		if p.GetUserId() == "gopher" {
			subs = append(subs, &signal_rpc.TrackSubscriptionDetails{UserId: p.GetUserId(), SessionId: p.GetSessionId(), TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO})
		}
	}
	check(call.SubscribeToTracks(context.Background(), subs...))
}

func prontoToken(pronto, user string) (string, string) {
	resp, err := http.Get(pronto + "/api/auth/create-token?environment=pronto&user_id=" + url.QueryEscape(user))
	check(err)
	defer resp.Body.Close()
	type credentials struct{ APIKey, Token string }
	var v credentials
	dec, err := vibejson.CompileDecoder[credentials](vibejson.DecoderOptions{})
	check(err)
	r := vibejson.NewReader(resp.Body)
	if !vibejson.DecodeNext(r, dec, &v) {
		log.Fatal("token")
	}
	return v.APIKey, v.Token
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
