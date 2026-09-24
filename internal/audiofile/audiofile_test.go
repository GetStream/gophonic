// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package audiofile

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func TestDecodeWAVBytesAndDurationLimit(t *testing.T) {
	var wav bytes.Buffer
	wav.WriteString("RIFF")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(36+8))
	wav.WriteString("WAVEfmt ")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(1))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(1))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16000))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(32000))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(2))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(16))
	wav.WriteString("data")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(8))
	for _, sample := range []int16{0, 32767, -32768, 16384} {
		_ = binary.Write(&wav, binary.LittleEndian, sample)
	}
	pcm, rate, channels, err := DecodeBytes("sample.wav", wav.Bytes(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 || channels != 1 || len(pcm) != 4 || pcm[0] != 0 || pcm[1] != 32767.0/32768 || pcm[2] != -1 || pcm[3] != 0.5 {
		t.Fatalf("decoded WAV: rate=%d channels=%d pcm=%v", rate, channels, pcm)
	}
	if _, _, _, err := Decode("sample.wav", bytes.NewReader(wav.Bytes()), 1); err != nil {
		t.Fatalf("reader decoder: %v", err)
	}
	if _, _, _, err := DecodeBytes("sample.mp3", wav.Bytes(), 1); err == nil {
		t.Fatal("unsupported format accepted")
	}
	// Four frames at 16 kHz are accepted; 16001 frames exceed a one-second limit.
	oversized := bytes.Repeat([]byte{0, 0}, 16001)
	limited := append([]byte(nil), wav.Bytes()[:44]...)
	binary.LittleEndian.PutUint32(limited[4:8], uint32(36+len(oversized)))
	binary.LittleEndian.PutUint32(limited[40:44], uint32(len(oversized)))
	limited = append(limited, oversized...)
	if _, _, _, err := DecodeBytes("long.wav", limited, 1); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("duration limit error = %v", err)
	}
}

func TestDecodeWAVIntoUsesCallerStorage(t *testing.T) {
	wav := make([]byte, 48)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], 40)
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 1)
	binary.LittleEndian.PutUint32(wav[24:], 16000)
	binary.LittleEndian.PutUint32(wav[28:], 32000)
	binary.LittleEndian.PutUint16(wav[32:], 2)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 4)
	binary.LittleEndian.PutUint16(wav[44:], 32767)
	binary.LittleEndian.PutUint16(wav[46:], 0)
	scratch := make([]float32, 2)
	allocations := testing.AllocsPerRun(100, func() {
		pcm, rate, channels, err := DecodeWAVInto(wav, scratch, 1)
		if err != nil || rate != 16000 || channels != 1 || len(pcm) != 2 || &pcm[0] != &scratch[0] {
			panic("decode failed")
		}
	})
	if allocations != 0 {
		t.Fatalf("WAV decode: %g allocs/op", allocations)
	}
}
