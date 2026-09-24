// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package audiofile

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

func decodeWAV(data []byte, maxSeconds int) ([]float32, int, int, error) {
	return DecodeWAVInto(data, nil, maxSeconds)
}

// DecodeWAVInto decodes into caller-owned storage. When dst has enough
// capacity, decoding itself performs no heap allocation.
func DecodeWAVInto(data []byte, dst []float32, maxSeconds int) ([]float32, int, int, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, 0, errors.New("invalid RIFF/WAVE header")
	}
	var format, channels, bits, blockAlign int
	var sampleRate int
	var audioData []byte
	for pos := 12; pos+8 <= len(data); {
		chunkID := string(data[pos : pos+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		pos += 8
		if chunkSize < 0 || pos+chunkSize > len(data) {
			return nil, 0, 0, errors.New("truncated WAV chunk")
		}
		chunk := data[pos : pos+chunkSize]
		switch chunkID {
		case "fmt ":
			if len(chunk) < 16 {
				return nil, 0, 0, errors.New("short WAV fmt chunk")
			}
			format = int(binary.LittleEndian.Uint16(chunk[0:2]))
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			blockAlign = int(binary.LittleEndian.Uint16(chunk[12:14]))
			bits = int(binary.LittleEndian.Uint16(chunk[14:16]))
		case "data":
			if audioData == nil {
				audioData = chunk
			}
		}
		pos += chunkSize
		if pos&1 != 0 {
			pos++
		}
	}
	if format != 1 && format != 3 {
		return nil, 0, 0, fmt.Errorf("unsupported WAV encoding %d; expected PCM or IEEE float", format)
	}
	if channels < 1 || channels > 2 || sampleRate < 8000 || sampleRate > 96000 || len(audioData) == 0 {
		return nil, 0, 0, errors.New("WAV must contain mono or stereo audio at 8–96 kHz")
	}
	if format == 1 && bits != 8 && bits != 16 && bits != 24 && bits != 32 {
		return nil, 0, 0, fmt.Errorf("unsupported PCM bit depth %d", bits)
	}
	if format == 3 && bits != 32 && bits != 64 {
		return nil, 0, 0, fmt.Errorf("unsupported float bit depth %d", bits)
	}
	bytesPerSample := bits / 8
	if blockAlign != channels*bytesPerSample || len(audioData)%blockAlign != 0 {
		return nil, 0, 0, errors.New("invalid WAV block alignment")
	}
	frames := len(audioData) / blockAlign
	if maxSeconds > 0 && frames > sampleRate*maxSeconds {
		return nil, 0, 0, fmt.Errorf("audio exceeds %d seconds", maxSeconds)
	}
	if cap(dst) < frames*channels {
		dst = make([]float32, frames*channels)
	} else {
		dst = dst[:frames*channels]
	}
	samples := dst
	for i := range samples {
		at := i * bytesPerSample
		switch {
		case format == 1 && bits == 8:
			samples[i] = (float32(audioData[at]) - 128) / 128
		case format == 1 && bits == 16:
			samples[i] = float32(int16(binary.LittleEndian.Uint16(audioData[at:at+2]))) / 32768
		case format == 1 && bits == 24:
			v := int32(audioData[at]) | int32(audioData[at+1])<<8 | int32(audioData[at+2])<<16
			if v&0x800000 != 0 {
				v |= ^int32(0xffffff)
			}
			samples[i] = float32(v) / 8388608
		case format == 1 && bits == 32:
			samples[i] = float32(int32(binary.LittleEndian.Uint32(audioData[at:at+4]))) / 2147483648
		case format == 3 && bits == 32:
			samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(audioData[at : at+4]))
		case format == 3 && bits == 64:
			samples[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(audioData[at : at+8])))
		}
	}
	return samples, sampleRate, channels, nil
}
