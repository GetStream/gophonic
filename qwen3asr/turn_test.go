// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/GetStream/gophonic/speech"
)

// The embedded head matches the checkpoint it was trained on.
func TestTurnHeadFile(t *testing.T) {
	f, err := parseTurnHead(turn17)
	if err != nil {
		t.Fatal(err)
	}
	if f.hidden != 2048 || f.layers != 28 || f.vocab != 151936 || f.threshold <= 0 || f.threshold >= 1 {
		t.Fatalf("head for %d×%d, vocabulary %d, threshold %v", f.hidden, f.layers, f.vocab, f.threshold)
	}
}

// Qwen3-ASR hears whether a speaker is done: the whole of JFK's sentence
// surely ends the turn, "ask not" surely does not, and "my fellow
// Americans," with its falling voice, is not sure, so an agent waits and
// judges again as the pause grows. Each is judged 40 ms into the pause.
func TestTurn(t *testing.T) {
	m := loadModel(t, FormatGPU)
	tr, err := NewTranscriber(m, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	pcm := clipPCM(t, "jfk")
	pause := make([]float32, 40*sampleRate/1000)
	var dst speech.Transcript
	for _, c := range []struct {
		end    float64 // seconds of speech
		lo, hi float32 // bounds of the judgment
	}{{2.12, 0, 0.9}, {4.30, 0, 0.5}, {11.0, 0.9, 1}} {
		audio := append(append([]float32(nil), pcm[:int(c.end*sampleRate)]...), pause...)
		if err := tr.Transcribe(context.Background(), audio, speech.Options{Turn: true}, &dst); err != nil {
			t.Fatal(err)
		}
		t.Logf("%q: %.3f", dst.Text, dst.Turn.Probability)
		if p := dst.Turn.Probability; p < c.lo || p > c.hi {
			t.Errorf("%q: turn over with probability %.3f, want %v to %v", dst.Text, p, c.lo, c.hi)
		}
		// Continuing a transcript judges the same.
		partial := dst
		partial.Text = append([]byte(nil), dst.Text...)
		want := dst.Turn.Probability
		if err := tr.Transcribe(context.Background(), audio, speech.Options{Turn: true, Partial: &partial}, &dst); err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(float64(dst.Turn.Probability - want)); d > 0.02 {
			t.Errorf("%q continued: %.3f, %.3f alone", dst.Text, dst.Turn.Probability, want)
		}
	}
	audio := append(append([]float32(nil), pcm...), pause...)
	if n := testing.AllocsPerRun(3, func() {
		if err := tr.Transcribe(context.Background(), audio, speech.Options{Turn: true}, &dst); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("%v allocations per warm call", n)
	}
	// Models without a head say so.
	saved := m.turn
	m.turn = nil
	defer func() { m.turn = saved }()
	if err := tr.Transcribe(context.Background(), audio, speech.Options{Turn: true}, &dst); !errors.Is(err, speech.ErrUnsupported) {
		t.Errorf("without a head: %v", err)
	}
}

// TestTurnFeatures writes the training data of the turn head (see
// tools/turn.py): for each clip of TURN_LIST (WAV path, label, pause in ms),
// the label, the pause, the audio rows and the state that ends its
// transcript, as float32 to TURN_OUT.
func TestTurnFeatures(t *testing.T) {
	list, out := os.Getenv("TURN_LIST"), os.Getenv("TURN_OUT")
	if list == "" || out == "" {
		t.Skip("TURN_LIST and TURN_OUT name the clips and the output")
	}
	m := loadModel(t, FormatGPU)
	tr, err := NewTranscriber(m, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	in, err := os.Open(list)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	f, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	var dst speech.Transcript
	sc := bufio.NewScanner(in)
	for sc.Scan() {
		fields := strings.Split(sc.Text(), "\t")
		label, _ := strconv.ParseFloat(fields[1], 32)
		pause, _ := strconv.Atoi(fields[2])
		pcm := readWAV16(t, fields[0])
		end := speechEnd(pcm) + pause*sampleRate/1000
		if end <= len(pcm) {
			pcm = pcm[:end]
		} else {
			pcm = append(pcm, make([]float32, end-len(pcm))...)
		}
		if err := tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
			t.Fatal(err)
		}
		binary.Write(w, binary.LittleEndian, []float32{float32(label), float32(pause), float32(len(tr.embeds) / m.enc.out)})
		binary.Write(w, binary.LittleEndian, tr.hidden)
	}
}

// readWAV16 reads a 16-bit PCM WAV file.
func readWAV16(t *testing.T, path string) []float32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 12; i+8 <= len(raw); {
		id, size := string(raw[i:i+4]), int(binary.LittleEndian.Uint32(raw[i+4:]))
		i += 8
		if id == "data" {
			data := raw[i:min(i+size, len(raw))]
			pcm := make([]float32, len(data)/2)
			for j := range pcm {
				pcm[j] = float32(int16(binary.LittleEndian.Uint16(data[2*j:]))) / 32768
			}
			return pcm
		}
		i += size + size&1
	}
	t.Fatalf("%s: no PCM data", path)
	return nil
}

// speechEnd returns the sample after the last 20 ms frame louder than 5% of
// the loudest.
func speechEnd(pcm []float32) int {
	const frame = sampleRate / 50
	rms := make([]float64, len(pcm)/frame)
	var peak float64
	for i := range rms {
		var e float64
		for _, v := range pcm[i*frame : (i+1)*frame] {
			e += float64(v) * float64(v)
		}
		rms[i] = math.Sqrt(e / frame)
		peak = max(peak, rms[i])
	}
	for i := len(rms) - 1; i >= 0; i-- {
		if rms[i] > max(0.05*peak, 0.003) {
			return (i + 1) * frame
		}
	}
	return len(pcm)
}
