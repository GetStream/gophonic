// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command gophonic runs any model gophonic.Open recognizes on its inputs,
// loading the model once: it transcribes audio files, predicts whether
// their speakers have finished their turns, or classifies them, according
// to what the model provides. A language model classifies text arguments
// instead, answering -question with one of -labels. Each input's result is
// written in input order, one JSON or text line per input (subtitles are
// blocks).
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/audiofile"
	"github.com/GetStream/gophonic/internal/transcriptformat"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

var (
	modelPath      = flag.String("model", "", "model: a Qwen3-ASR or Qwen3 checkpoint directory, or a converted .gophonic bundle (Whisper, Smart Turn, TinyMelNet)")
	question       = flag.String("question", "", "for a language model: the question to answer about each text argument")
	labels         = flag.String("labels", "", "for a language model: the comma-separated answers to choose from")
	responseFormat = flag.String("response-format", "json", "transcript output: json, text, verbose_json, srt, or vtt")
	wordTimestamps = flag.Bool("word-timestamps", false, "include word timestamps in verbose_json output")
	language       = flag.String("language", "", "spoken language as a code or English name (default: detect)")
	contextText    = flag.String("context", "", "text that primes transcription, such as names or terms")
	threads        = flag.Int("threads", 0, "CPU workers per lane (0: model default)")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: gophonic -model PATH [flags] audio.wav|audio.ogg|audio.opus...\n"+
			"       gophonic -model LANGUAGE-MODEL -question Q -labels A,B,... text...\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	switch {
	case *modelPath == "" || flag.NArg() == 0 || *threads < 0,
		*wordTimestamps && *responseFormat != "verbose_json":
		flag.Usage()
		os.Exit(2)
	}
	switch *responseFormat {
	case "json", "text", "verbose_json", "srt", "vtt":
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err := run(flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "gophonic:", err)
		os.Exit(1)
	}
}

// run loads the model once, runs it on every input, and closes it.
func run(inputs []string) error {
	if *language != "" {
		if _, ok := speech.LanguageName(*language); !ok {
			return fmt.Errorf("unknown language %q", *language)
		}
	}
	model, err := gophonic.Open(*modelPath, gophonic.Options{Threads: *threads})
	if err != nil {
		return err
	}
	defer model.Close()
	out := vibejson.NewWriter(os.Stdout)
	switch {
	case gophonic.Supports[speech.ZeroShot](model):
		return classifyTexts(model, out, inputs)
	case gophonic.Supports[speech.Transcriber](model):
		return transcribe(model, inputs)
	case gophonic.Supports[speech.TurnDetector](model):
		return detectTurns(model, out, inputs)
	case gophonic.Supports[speech.AudioClassifier](model):
		return classifyAudio(model, out, inputs)
	}
	return fmt.Errorf("%s provides nothing this command runs", model.Name())
}

func transcribe(model *gophonic.Model, paths []string) error {
	transcriber, err := model.NewTranscriber()
	if err != nil {
		return err
	}
	defer transcriber.Close()
	resampler := speech.NewResampler()
	defer resampler.Close()
	opts := speech.Options{
		Language: *language,
		Context:  *contextText,
		Segments: *responseFormat == "verbose_json" || *responseFormat == "srt" || *responseFormat == "vtt",
		Words:    *wordTimestamps,
	}
	var (
		mono   []float32
		t      speech.Transcript
		output []byte
	)
	for _, path := range paths {
		pcm, rate, channels, err := audiofile.ReadPath(path)
		if err != nil {
			return err
		}
		count, err := speech.Samples16k(len(pcm), rate, channels)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if cap(mono) < count {
			mono = make([]float32, count)
		}
		n, err := resampler.Resample16kInto(pcm, rate, channels, mono[:count])
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := transcriber.Transcribe(context.Background(), mono[:n], opts, &t); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		trimmed := bytes.TrimSpace(t.Text)
		output = output[:0]
		switch *responseFormat {
		case "json":
			output = append(output, `{"text":`...)
			output = transcriptformat.AppendJSONString(output, trimmed)
			output = append(output, '}', '\n')
		case "text":
			output = append(output, trimmed...)
			output = append(output, '\n')
		case "verbose_json":
			output = transcriptformat.AppendVerboseJSON(output, &t, *wordTimestamps, n)
		case "srt", "vtt":
			output = transcriptformat.AppendSubtitles(output, &t, *responseFormat == "vtt")
		}
		if _, err := os.Stdout.Write(output); err != nil {
			return err
		}
	}
	return nil
}

func detectTurns(model *gophonic.Model, out *vibejson.Writer, paths []string) error {
	detector, err := model.NewTurnDetector()
	if err != nil {
		return err
	}
	defer detector.Close()
	enc, err := vibejson.CompileEncoder[speech.Prediction](vibejson.EncoderOptions{})
	if err != nil {
		return err
	}
	for _, path := range paths {
		pcm, rate, channels, err := audiofile.ReadPath(path)
		if err != nil {
			return err
		}
		prediction, err := detector.PredictInto(pcm, rate, channels)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := writeLine(out, enc, &prediction); err != nil {
			return err
		}
	}
	return nil
}

func classifyAudio(model *gophonic.Model, out *vibejson.Writer, paths []string) error {
	classifier, err := gophonic.Lane[speech.AudioClassifier](model)
	if err != nil {
		return err
	}
	defer classifier.Close()
	probs := make([]float32, len(classifier.Labels()))
	for _, path := range paths {
		pcm, rate, channels, err := audiofile.ReadPath(path)
		if err != nil {
			return err
		}
		if err := classifier.ClassifyInto(pcm, rate, channels, probs); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := writeClasses(out, classifier.Labels(), probs); err != nil {
			return err
		}
	}
	return nil
}

// classifyTexts answers -question about each text with one of -labels.
func classifyTexts(model *gophonic.Model, out *vibejson.Writer, texts []string) error {
	choices := strings.Split(*labels, ",")
	if *question == "" || len(choices) < 2 {
		return fmt.Errorf("%s answers questions about text: give -question and at least two -labels", model.Name())
	}
	zeroShot, err := gophonic.Lane[speech.ZeroShot](model)
	if err != nil {
		return err
	}
	defer zeroShot.Close()
	classifier, err := zeroShot.Classifier(*question, choices)
	if err != nil {
		return err
	}
	defer classifier.Close()
	probs := make([]float32, len(choices))
	for _, text := range texts {
		if err := classifier.ClassifyInto(context.Background(), text, probs); err != nil {
			return err
		}
		if err := writeClasses(out, choices, probs); err != nil {
			return err
		}
	}
	return nil
}

// class is one label's probability in the output.
type class struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

var classesEncoder, _ = vibejson.CompileEncoder[[]class](vibejson.EncoderOptions{})

// writeClasses writes each label's probability, in order, as a JSON line.
func writeClasses(out *vibejson.Writer, labels []string, probs []float32) error {
	classes := make([]class, len(labels))
	for i := range labels {
		classes[i] = class{labels[i], probs[i]}
	}
	return writeLine(out, classesEncoder, &classes)
}

// writeLine streams v as one line of JSON.
func writeLine[T any](out *vibejson.Writer, enc vibejson.Encoder[T], v *T) error {
	if err := vibejson.EncodeTo(out, enc, v); err != nil {
		return err
	}
	if err := out.Newline(); err != nil {
		return err
	}
	return out.Flush()
}
