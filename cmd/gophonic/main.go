// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/audiofile"
	"github.com/GetStream/gophonic/internal/transcriptformat"
	"github.com/GetStream/gophonic/whisper"
)

func main() {
	modelPath := flag.String("model", "", "converted Pipecat Smart Turn v3.2 .gophonic bundle")
	tinyModelPath := flag.String("tiny-model", "", "converted TinyMelNet .gophonic bundle (uses its own threshold)")
	whisperModelPath := flag.String("whisper-model", "", "converted OpenAI Whisper English .gophonic bundle (tiny.en, base.en, small.en) for transcription")
	responseFormat := flag.String("response-format", "json", "Whisper output: json, text, verbose_json, srt, or vtt")
	wordTimestamps := flag.Bool("word-timestamps", false, "include aligned word timestamps in Whisper verbose_json output")
	tinyWorkers := flag.Int("tiny-workers", 0, "TinyMelNet persistent CPU helpers (0=serial, max 7; only with -tiny-model)")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: gophonic (-model smart-turn.gophonic | -tiny-model tinymel.gophonic [-tiny-workers 0..7] | -whisper-model whisper.gophonic) audio.wav|audio.ogg|audio.opus\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	modes := 0
	for _, path := range []string{*modelPath, *tinyModelPath, *whisperModelPath} {
		if path != "" {
			modes++
		}
	}
	if modes != 1 || flag.NArg() != 1 || *tinyWorkers < 0 || *tinyWorkers > gophonic.MaxTinyMelWorkers || (*tinyModelPath == "" && *tinyWorkers != 0) ||
		(*whisperModelPath == "" && (*responseFormat != "json" || *wordTimestamps)) ||
		(*wordTimestamps && *responseFormat != "verbose_json") ||
		(*whisperModelPath != "" && *responseFormat != "json" && *responseFormat != "text" && *responseFormat != "verbose_json" && *responseFormat != "srt" && *responseFormat != "vtt") {
		flag.Usage()
		os.Exit(2)
	}
	pcm, rate, channels, err := audiofile.ReadPath(flag.Arg(0))
	if err != nil {
		fatal(err)
	}
	if *whisperModelPath != "" {
		model, err := whisper.Load(*whisperModelPath)
		if err != nil {
			fatal(err)
		}
		worker, err := whisper.NewTranscriber(model)
		if err != nil {
			fatal(err)
		}
		defer worker.Close()
		count, err := whisper.PCM16kSamples(len(pcm), rate, channels)
		if err != nil {
			fatal(err)
		}
		mono := make([]float32, count)
		pcmWorkspace := whisper.NewPCMWorkspace()
		defer pcmWorkspace.Close()
		n, err := pcmWorkspace.Resample16kInto(pcm, rate, channels, mono)
		if err != nil {
			fatal(err)
		}
		textBuffer := make([]byte, 0, max(4096, n/16))
		var result []byte
		var segments []whisper.Segment
		var words []whisper.Word
		if *wordTimestamps {
			result, segments, words, err = worker.TranscribeWordsInto(mono[:n], textBuffer, make([]whisper.Segment, 0, max(64, n/2000)), make([]whisper.Word, 0, max(128, n/200)))
		} else if *responseFormat == "verbose_json" || *responseFormat == "srt" || *responseFormat == "vtt" {
			result, segments, err = worker.TranscribeSegmentsInto(mono[:n], textBuffer, make([]whisper.Segment, 0, max(64, n/2000)))
		} else {
			result, err = worker.TranscribeInto(mono[:n], textBuffer)
		}
		if err != nil {
			fatal(err)
		}
		output := make([]byte, 0, max(4096, len(result)*4))
		trimmed := bytes.TrimSpace(result)
		switch *responseFormat {
		case "json":
			output = append(output, `{"text":`...)
			output = transcriptformat.AppendJSONString(output, trimmed)
			output = append(output, '}', '\n')
		case "text":
			output = append(output, trimmed...)
			output = append(output, '\n')
		case "verbose_json":
			output = transcriptformat.AppendVerboseJSON(output, trimmed, result, segments, words, *wordTimestamps, n)
		case "srt", "vtt":
			output = transcriptformat.AppendSubtitles(output, result, segments, *responseFormat == "vtt")
		}
		if _, err := os.Stdout.Write(output); err != nil {
			fatal(err)
		}
		return
	}
	var prediction gophonic.Prediction
	if *tinyModelPath != "" {
		model, err := gophonic.LoadTinyMel(*tinyModelPath)
		if err != nil {
			fatal(err)
		}
		workspace := gophonic.NewTinyMelWorkspaceWithWorkers(*tinyWorkers)
		defer workspace.Close()
		prediction, err = model.PredictInto(pcm, rate, channels, workspace)
	} else {
		model, err := gophonic.Load(*modelPath)
		if err != nil {
			fatal(err)
		}
		workspace := gophonic.NewWorkspace()
		defer workspace.Close()
		prediction, err = model.PredictInto(pcm, rate, channels, workspace)
	}
	if err != nil {
		fatal(err)
	}
	if err := json.NewEncoder(os.Stdout).Encode(prediction); err != nil {
		fatal(err)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "gophonic:", err)
	os.Exit(1)
}
