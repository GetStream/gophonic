// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"errors"
	"io"
	"os"

	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"
)

func readOggOpus(path string) ([]float32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	reader, err := ogg.NewReader(f)
	if err != nil {
		return nil, 0, 0, err
	}
	channels := int(reader.Channels())
	if channels < 1 || channels > 2 {
		return nil, 0, 0, errors.New("Ogg Opus must have one or two channels")
	}
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(16000, channels))
	if err != nil {
		return nil, 0, 0, err
	}
	pcm := make([]float32, 0, 16000*8*channels)
	// An Ogg page can carry packets larger than a typical 20 ms Opus frame.
	// The maximum Ogg packet size is bounded by the page lacing table.
	packet := make([]byte, 65536)
	frame := make([]float32, 5760*channels)
	var finalGranule uint64
	for {
		n, granule, readErr := reader.ReadPacketInto(packet)
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, 0, 0, readErr
		}
		if granule > finalGranule {
			finalGranule = granule
		}
		samples, decodeErr := decoder.Decode(packet[:n], frame)
		if decodeErr != nil {
			return nil, 0, 0, decodeErr
		}
		pcm = append(pcm, frame[:samples*channels]...)
	}
	if finalGranule == 0 {
		return nil, 0, 0, errors.New("Ogg Opus contains no audio packets")
	}
	decodedFrames := len(pcm) / channels
	validEnd := int(finalGranule / 3) // Ogg granule positions are measured at 48 kHz.
	if validEnd > decodedFrames {
		validEnd = decodedFrames
	}
	preSkip := int(reader.PreSkip()) / 3
	if validEnd <= preSkip {
		return nil, 0, 0, errors.New("Ogg Opus stream is shorter than its pre-skip")
	}
	pcm = pcm[preSkip*channels : validEnd*channels]
	return pcm, 16000, channels, nil
}
