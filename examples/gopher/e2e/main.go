// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command e2e checks Gopher end to end, through the SFU: it joins the call
// as a test participant, speaks recorded speech, and listens for Gopher's
// voice, reporting whether it answered and how long after the speech ended.
// It loads no models.
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
	say := func(round int, what string) {
		time.Sleep(3 * time.Second)
		mu.Lock()
		loud = loud[:0]
		mu.Unlock()
		tick := time.NewTicker(20 * time.Millisecond)
		for pos := 0; pos+320 <= len(speechPCM); pos += 320 {
			<-tick.C
			writer.Write(webaudio.FromFloat32(speechPCM[pos:pos+320], 16000, 1))
		}
		tick.Stop()
		writer.Flush()
		ended := time.Now()
		time.Sleep(6 * time.Second)
		mu.Lock()
		var first time.Time
		for _, t := range loud {
			if t.After(ended) {
				first = t
				break
			}
		}
		mu.Unlock()
		if first.IsZero() {
			fmt.Printf("round %d (%s): Gopher did not answer\n", round, what)
		} else {
			fmt.Printf("round %d (%s): Gopher answered %v after the speech ended\n", round, what, first.Sub(ended).Round(10*time.Millisecond))
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
