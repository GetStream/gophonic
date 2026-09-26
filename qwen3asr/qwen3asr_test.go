// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

// reference is testdata/qwen3asr/reference.json, written by
// tools/reference.py from the official model in FP32.
type reference struct {
	Clips map[string]struct {
		Samples, Frames, Tokens int
		FeatureFrames           []int       `json:"feature_frames"`
		Features                []float32   `json:"features"`
		EncoderRows             []int       `json:"encoder_rows"`
		InputIDs                []int       `json:"input_ids"`
		LogitsTop               [][]float64 `json:"logits_top"`
		Generated               []int       `json:"generated"`
		Language, Text          string
	} `json:"clips"`
}

func loadReference(t testing.TB) *reference {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "qwen3asr", "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r reference
	if err := vibejson.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return &r
}

// clipPCM returns a test clip as the transcriber receives it: jfk is the
// raw Whisper fixture, whose peak above 1 the transcriber normalizes.
func clipPCM(t testing.TB, name string) []float32 {
	path := map[string]string{"jfk": "whisper_jfk.pcm.f32le", "zh": "qwen3asr/zh.pcm.s16le"}[name]
	raw, err := os.ReadFile(filepath.Join("..", "testdata", path))
	if err != nil {
		t.Fatal(err)
	}
	if name == "zh" {
		pcm := make([]float32, len(raw)/2)
		for i := range pcm {
			pcm[i] = float32(int16(binary.LittleEndian.Uint16(raw[2*i:]))) / 32768
		}
		return pcm
	}
	return readFloats(t, raw)
}

func readFloats(t testing.TB, raw []byte) []float32 {
	v := make([]float32, len(raw)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return v
}

var models sync.Map // checkpoint and format → *Model

// loadModel loads the checkpoint once per format for the package's tests.
func loadModel(t testing.TB, format string) *Model {
	t.Helper()
	return loadNamed(t, testmodels.Qwen3ASR, format)
}

// loadNamed loads the checkpoint name, in the models directory, once per
// format.
func loadNamed(t testing.TB, name, format string) *Model {
	t.Helper()
	dir := testmodels.Path(t, name)
	if (format == qwen3lm.WeightsGPUQ8 || format == qwen3lm.WeightsGPU) && !qwen3lm.GPUAvailable() {
		t.Skip("no Metal GPU")
	}
	key := name + "\x00" + format
	if m, ok := models.Load(key); ok {
		return m.(*Model)
	}
	m, err := Load(dir, Options{Format: format})
	if err != nil {
		t.Fatal(err)
	}
	models.Store(key, m)
	return m
}

func cosine(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
		maxAbs = max(maxAbs, math.Abs(float64(a[i]-b[i])))
	}
	return dot / math.Sqrt(na*nb), maxAbs
}

// The frontend, encoder, and prompt match the FP32 reference: features to
// FP32 rounding, encoder rows at every window edge to within the FP16
// rounding of activations that both encoders use, and the prompt ids
// exactly.
func TestFrontendEncoderAndPromptMatchReference(t *testing.T) {
	ref := loadReference(t)
	for _, format := range []string{qwen3lm.WeightsF16, qwen3lm.WeightsGPUQ8} {
		t.Run(format, func(t *testing.T) {
			m := loadModel(t, format)
			tr, err := NewTranscriber(m, LaneOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer tr.Close()
			const minCos, maxDiff = 0.999999, 2e-4
			for name, want := range ref.Clips {
				var dst speech.Transcript
				if err := tr.Transcribe(context.Background(), clipPCM(t, name), speech.Options{}, &dst); err != nil {
					t.Fatal(err)
				}
				frames := len(tr.features) / 128
				if frames != want.Frames || len(tr.embeds) != want.Tokens*m.enc.out {
					t.Fatalf("%s: %d frames and %d tokens, want %d and %d", name, frames, len(tr.embeds)/m.enc.out, want.Frames, want.Tokens)
				}
				for i, f := range want.FeatureFrames {
					for bin := range 128 {
						got, w := tr.features[bin*frames+f], want.Features[i*128+bin]
						if math.Abs(float64(got-w)) > 1e-4 {
							t.Fatalf("%s: feature (bin %d, frame %d) = %g, want %g", name, bin, f, got, w)
						}
					}
				}
				raw, err := os.ReadFile(filepath.Join("..", "testdata", "qwen3asr", name+".encoder.f32le"))
				if err != nil {
					t.Fatal(err)
				}
				rows := readFloats(t, raw)
				d := m.enc.out
				worstCos, worstDiff := 1.0, 0.0
				for i, r := range want.EncoderRows {
					cos, diff := cosine(tr.embeds[r*d:(r+1)*d], rows[i*d:(i+1)*d])
					worstCos, worstDiff = min(worstCos, cos), max(worstDiff, diff)
				}
				t.Logf("%s: encoder rows cosine ≥ %.9f, max abs %g", name, worstCos, worstDiff)
				if worstCos < minCos || worstDiff > maxDiff {
					t.Fatalf("%s: encoder rows cosine %.9f, max abs %g", name, worstCos, worstDiff)
				}
				if !slices.Equal(tr.ids, want.InputIDs) {
					t.Fatalf("%s: prompt ids differ from the processor's", name)
				}
			}
		})
	}
}

// Transcripts match the reference in the exact format and on the GPU, whose
// first logits stay within Q8_0-level error.
func TestTranscriptsMatchReference(t *testing.T) {
	ref := loadReference(t)
	for _, format := range []string{qwen3lm.WeightsF16, qwen3lm.WeightsGPUQ8} {
		t.Run(format, func(t *testing.T) {
			m := loadModel(t, format)
			tr, err := NewTranscriber(m, LaneOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer tr.Close()
			for name, want := range ref.Clips {
				var dst speech.Transcript
				if err := tr.Transcribe(context.Background(), clipPCM(t, name), speech.Options{}, &dst); err != nil {
					t.Fatal(err)
				}
				if string(dst.Text) != want.Text || dst.Language.Name() != want.Language {
					t.Errorf("%s: %q (%s), want %q (%s)", name, dst.Text, dst.Language, want.Text, want.Language)
				}
				if format == qwen3lm.WeightsF16 && !slices.Equal(tr.gen, want.Generated[:len(want.Generated)-1]) {
					t.Errorf("%s: generated %v, want %v", name, tr.gen, want.Generated)
				}
				// First-step logits of a fresh prefill against the reference's
				// 32 largest.
				kv, err := m.eval.NewPrefixKV(len(tr.ids))
				if err != nil {
					t.Fatal(err)
				}
				embeds := qwen3lm.Embeds{Token: m.ids.audioPad, Rows: tr.embeds}
				if err := m.eval.HiddenLastExtendEmbedInto(kv, 0, tr.ids, embeds, tr.hidden, tr.lm); err != nil {
					t.Fatal(err)
				}
				if err := m.eval.LogitsInto(tr.hidden, tr.logits, tr.lm); err != nil {
					t.Fatal(err)
				}
				tolerance := 0.05 // exact weights, FP16 activations
				if format == qwen3lm.WeightsGPUQ8 {
					tolerance = 1.5
				}
				if top := int(want.LogitsTop[0][0]); argmax(tr.logits) != top {
					t.Errorf("%s: first token %d, want %d", name, argmax(tr.logits), top)
				}
				for _, p := range want.LogitsTop {
					if d := math.Abs(float64(tr.logits[int(p[0])]) - p[1]); d > tolerance {
						t.Errorf("%s: logit %d is %g, want %g", name, int(p[0]), tr.logits[int(p[0])], p[1])
					}
				}
			}
		})
	}
}

func TestWarmTranscribeDoesNotAllocate(t *testing.T) {
	for _, format := range []string{qwen3lm.WeightsF16, qwen3lm.WeightsGPUQ8} {
		t.Run(format, func(t *testing.T) {
			tr, err := NewTranscriber(loadModel(t, format), LaneOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer tr.Close()
			pcm := clipPCM(t, "zh")
			var dst speech.Transcript
			opts := speech.Options{Language: speech.Chinese, Context: "交易", Segments: true}
			if err := tr.Transcribe(context.Background(), pcm, opts, &dst); err != nil {
				t.Fatal(err)
			}
			if n := testing.AllocsPerRun(3, func() {
				if err := tr.Transcribe(context.Background(), pcm[:len(pcm)*3/4], opts, &dst); err != nil {
					t.Fatal(err)
				}
			}); n != 0 {
				t.Fatalf("warm Transcribe allocated %.1f times", n)
			}
		})
	}
}

func TestOptions(t *testing.T) {
	tr, err := NewTranscriber(loadModel(t, qwen3lm.WeightsF16), LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	ref := loadReference(t)
	pcm := clipPCM(t, "zh")
	var dst speech.Transcript
	// A forced language is appended to the prompt; the model then writes
	// the text alone.
	if err := tr.Transcribe(context.Background(), pcm, speech.Options{Language: speech.Chinese, Segments: true}, &dst); err != nil {
		t.Fatal(err)
	}
	if string(dst.Text) != ref.Clips["zh"].Text || dst.Language != speech.Chinese {
		t.Errorf("forced Chinese: %q (%s)", dst.Text, dst.Language)
	}
	suffix, _ := tr.m.tok.EncodeInto("assistant\nlanguage Chinese<asr_text>", make([]int, 0, 64), &tr.tokWS)
	if !slices.Equal(tr.ids[len(tr.ids)-len(suffix):], suffix) {
		t.Errorf("prompt does not end with the forced language")
	}
	if len(dst.Segments) != 1 || dst.Segments[0].End != float64(len(pcm))/sampleRate || dst.Segments[0].TextEnd != len(dst.Text) {
		t.Errorf("segments %+v", dst.Segments)
	}
	for _, opts := range []speech.Options{{Language: speech.Language(200)}, {Words: true}} {
		if err := tr.Transcribe(context.Background(), pcm, opts, &dst); !errors.Is(err, speech.ErrUnsupported) {
			t.Errorf("%+v: error %v, want ErrUnsupported", opts, err)
		}
	}
	if err := tr.Transcribe(context.Background(), pcm, speech.Options{Context: "<|audio_pad|>"}, &dst); err == nil {
		t.Error("accepted a context holding the audio placeholder")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := tr.Transcribe(ctx, pcm, speech.Options{}, &dst); !errors.Is(err, context.Canceled) {
		t.Errorf("canceled context: %v", err)
	}
	// Silence yields no text.
	if err := tr.Transcribe(context.Background(), make([]float32, sampleRate), speech.Options{}, &dst); err != nil || len(dst.Text) != 0 {
		t.Errorf("silence: %q, %v", dst.Text, err)
	}
	if err := tr.Transcribe(context.Background(), make([]float32, 100), speech.Options{}, &dst); err != nil || len(dst.Text) != 0 {
		t.Errorf("6 ms of audio: %q, %v", dst.Text, err)
	}
}
