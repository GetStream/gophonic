// Command webrtc listens to a Stream call and runs local turn detection and Whisper.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"syscall"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/whisper"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	signal_rpc "github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/thesyncim/gopus"
)

const (
	sampleRate          = 16000
	pauseSamples        = sampleRate / 2
	maxUtteranceSamples = 30 * sampleRate
	vadFrameSamples     = sampleRate / 50 // gopus VAD accepts 20 ms frames.
	vadThreshold        = 128             // SILK activity Q8; tune on call audio.
)

type utterance struct {
	participant string
	pcm         []float32
}

func main() {
	turnPath := flag.String("turn-model", "", "converted Smart Turn v3.2 model")
	whisperPath := flag.String("whisper-model", "", "converted Whisper tiny.en model")
	flag.Parse()
	if err := run(*turnPath, *whisperPath); err != nil {
		log.Fatal(err)
	}
}

func run(turnPath, whisperPath string) error {
	if turnPath == "" || whisperPath == "" {
		return errors.New("set -turn-model and -whisper-model")
	}
	apiKey, callID := os.Getenv("STREAM_API_KEY"), os.Getenv("STREAM_CALL_ID")
	if apiKey == "" || callID == "" {
		return errors.New("set STREAM_API_KEY and STREAM_CALL_ID")
	}
	userID := envOr("STREAM_USER_ID", "gophonic-listener")
	turnModel, err := gophonic.Load(turnPath)
	if err != nil {
		return fmt.Errorf("load turn model: %w", err)
	}
	whisperModel, err := whisper.Load(whisperPath)
	if err != nil {
		return fmt.Errorf("load Whisper model: %w", err)
	}

	user := rtc.User{ID: userID, Name: userID}
	var client *rtc.Client
	if token := os.Getenv("STREAM_USER_TOKEN"); token != "" {
		client, err = rtc.NewClient(apiKey, user, rtc.StaticToken(token))
	} else if secret := os.Getenv("STREAM_API_SECRET"); secret != "" {
		client, err = rtc.NewRTCClient(apiKey, secret, rtc.WithUser(user))
	} else {
		return errors.New("set STREAM_USER_TOKEN or STREAM_API_SECRET")
	}
	if err != nil {
		return fmt.Errorf("create Stream client: %w", err)
	}
	defer client.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	worker, err := whisper.NewTranscriberWithWorkers(whisperModel, 1)
	if err != nil {
		return fmt.Errorf("create Whisper worker: %w", err)
	}
	jobs := make(chan utterance, 8)
	go transcribe(ctx, worker, jobs)

	call := client.Call(envOr("STREAM_CALL_TYPE", "default"), callID)
	joined, err := call.Join(ctx, rtc.WithOnTrack(rtc.SubscriberFunc(func(t rtc.OnTrackReceived) {
		if t.TrackType == sfu_models.TrackType_TRACK_TYPE_AUDIO {
			go readTrack(ctx, t, turnModel, jobs)
		}
	})))
	if err != nil {
		return fmt.Errorf("join call: %w", err)
	}
	defer call.Leave("gophonic listener stopped")
	log.Printf("joined %s", call.CID())

	remove := rtc.HandleCallEvent(call, func(e *sfu_events.SfuEvent_TrackPublished) {
		p := e.TrackPublished
		if p.GetUserId() == userID || p.GetType() != sfu_models.TrackType_TRACK_TYPE_AUDIO {
			return
		}
		if err := call.SubscribeToTracks(ctx, &signal_rpc.TrackSubscriptionDetails{
			UserId: p.GetUserId(), SessionId: p.GetSessionId(), TrackType: p.GetType(),
		}); err != nil {
			log.Printf("subscribe new track: %v", err)
		}
	})
	defer remove()
	if err := subscribeExisting(ctx, call, joined, userID); err != nil {
		return err
	}
	<-ctx.Done()
	return nil
}

func subscribeExisting(ctx context.Context, call *rtc.Call, joined *sfu_events.JoinResponse, self string) error {
	var subs []*signal_rpc.TrackSubscriptionDetails
	for _, p := range joined.GetCallState().GetParticipants() {
		if p.GetUserId() == self {
			continue
		}
		for _, tt := range p.GetPublishedTracks() {
			if tt == sfu_models.TrackType_TRACK_TYPE_AUDIO {
				subs = append(subs, &signal_rpc.TrackSubscriptionDetails{
					UserId: p.GetUserId(), SessionId: p.GetSessionId(), TrackType: tt,
				})
			}
		}
	}
	if len(subs) == 0 {
		return nil
	}
	return call.SubscribeToTracks(ctx, subs...)
}

func readTrack(ctx context.Context, t rtc.OnTrackReceived, model *gophonic.Model, jobs chan<- utterance) {
	reader, err := audiortc.NewTrackReader(t.Track, audiortc.ReaderConfig{
		Opus: opus.Config{SampleRate: sampleRate, Channels: 1},
	})
	if err != nil {
		log.Printf("track %s: %v", t.ParticipantID, err)
		return
	}
	defer reader.Close()
	vad, err := gopus.NewVAD(sampleRate)
	if err != nil {
		log.Printf("VAD: %v", err)
		return
	}
	session, err := gophonic.NewSmartTurnSession(model)
	if err != nil {
		log.Printf("session: %v", err)
		return
	}
	defer session.Close()

	var pcm []float32
	var frame [vadFrameSamples]float32
	var vadPCM [vadFrameSamples]int16
	var frameN, silence int
	var hadVoice, voicedSinceCheck bool
	participant := fmt.Sprint(t.ParticipantID)
	for {
		decoded, err := reader.Read()
		if err != nil {
			if !errors.Is(err, io.EOF) && ctx.Err() == nil {
				log.Printf("track %s: %v", participant, err)
			}
			if hadVoice && len(pcm) > 0 && ctx.Err() == nil {
				pcm = append(pcm, frame[:frameN]...)
				submit(ctx, jobs, participant, pcm)
			}
			return
		}
		samples := decoded.Float32() // TrackReader is configured for float32 mono 16 kHz.
		for len(samples) > 0 {
			n := copy(frame[frameN:], samples)
			frameN += n
			samples = samples[n:]
			if frameN != vadFrameSamples {
				continue
			}
			frameN = 0
			for i, x := range frame {
				if x > 1 {
					x = 1
				} else if x < -1 {
					x = -1
				}
				vadPCM[i] = int16(x * 32767)
			}
			activity, err := vad.AnalyzeInt16(vadPCM[:])
			if err != nil {
				log.Printf("VAD %s: %v", participant, err)
				return
			}
			if activity >= vadThreshold {
				hadVoice, voicedSinceCheck, silence = true, true, 0
			} else {
				silence += vadFrameSamples
			}
			if !hadVoice {
				continue
			}
			pcm = append(pcm, frame[:]...)
			if len(pcm) >= maxUtteranceSamples {
				pcm = submit(ctx, jobs, participant, pcm[:maxUtteranceSamples])
				hadVoice, voicedSinceCheck, silence = false, false, 0
				continue
			}
			if silence < pauseSamples || !voicedSinceCheck {
				continue
			}
			voicedSinceCheck = false
			pred, err := session.PredictInto(pcm, sampleRate, 1)
			if err != nil {
				log.Printf("turn %s: %v", participant, err)
				continue
			}
			log.Printf("turn %s: p=%.3f complete=%v", participant, pred.Probability, pred.Complete)
			if pred.Complete {
				pcm = submit(ctx, jobs, participant, pcm)
				hadVoice, silence = false, 0
			}
		}
	}
}

// submit transfers ownership of the utterance buffer. A full queue drops the
// transcript rather than stalling the RTP reader or accumulating unbounded audio.
func submit(ctx context.Context, jobs chan<- utterance, participant string, pcm []float32) []float32 {
	select {
	case jobs <- utterance{participant: participant, pcm: pcm}:
	case <-ctx.Done():
	default:
		log.Printf("Whisper queue full; dropping %s utterance", participant)
	}
	return nil
}

func transcribe(ctx context.Context, worker *whisper.Transcriber, jobs <-chan utterance) {
	defer worker.Close()
	text := make([]byte, 0, 4096)
	for {
		select {
		case <-ctx.Done():
			return
		case job := <-jobs:
			result, err := worker.TranscribeWindowInto(job.pcm, text[:0])
			if err != nil {
				log.Printf("transcribe %s: %v", job.participant, err)
				continue
			}
			log.Printf("transcript %s: %s", job.participant, result)
		}
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
