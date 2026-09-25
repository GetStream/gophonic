// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command gophonic runs any model gophonic.Open recognizes on one input: it
// transcribes an audio file, predicts whether its speaker has finished their
// turn, or classifies it, according to what the model provides. A language
// model classifies the text argument instead, answering -question with one
// of -labels.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/audiofile"
	"github.com/GetStream/gophonic/internal/transcriptformat"
	"github.com/GetStream/gophonic/speech"
)

func main() {
	modelPath := flag.String("model", "", "model: a Qwen3-ASR or Qwen3 checkpoint directory, or a converted .gophonic bundle (Whisper, Smart Turn, TinyMelNet)")
	question := flag.String("question", "", "for a language model: the question to answer about the text argument")
	labels := flag.String("labels", "", "for a language model: the comma-separated answers to choose from")
	responseFormat := flag.String("response-format", "json", "transcript output: json, text, verbose_json, srt, or vtt")
	wordTimestamps := flag.Bool("word-timestamps", false, "include word timestamps in verbose_json output")
	language := flag.String("language", "", "spoken language as a code or English name (default: detect)")
	contextText := flag.String("context", "", "text that primes transcription, such as names or terms")
	threads := flag.Int("threads", 0, "CPU workers per lane (0: model default)")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: gophonic -model PATH [flags] audio.wav|audio.ogg|audio.opus\n"+
			"       gophonic -model LANGUAGE-MODEL -question Q -labels A,B,... text\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	switch {
	case *modelPath == "" || flag.NArg() != 1 || *threads < 0,
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
	if *language != "" {
		if _, ok := speech.LanguageName(*language); !ok {
			fatal(fmt.Errorf("unknown language %q", *language))
		}
	}
	model, err := gophonic.Open(*modelPath, gophonic.Options{Threads: *threads})
	if err != nil {
		fatal(err)
	}
	if gophonic.Supports[speech.ZeroShot](model) {
		classifyText(model, *question, strings.Split(*labels, ","), flag.Arg(0))
		return
	}
	pcm, rate, channels, err := audiofile.ReadPath(flag.Arg(0))
	if err != nil {
		fatal(err)
	}
	if !gophonic.Supports[speech.Transcriber](model) && !gophonic.Supports[speech.TurnDetector](model) &&
		gophonic.Supports[speech.AudioClassifier](model) {
		classifier, err := gophonic.Lane[speech.AudioClassifier](model)
		if err != nil {
			fatal(err)
		}
		defer classifier.Close()
		probs := make([]float32, len(classifier.Labels()))
		if err := classifier.ClassifyInto(pcm, rate, channels, probs); err != nil {
			fatal(err)
		}
		printClasses(classifier.Labels(), probs)
		return
	}
	if gophonic.Supports[speech.TurnDetector](model) {
		detector, err := model.NewTurnDetector()
		if err != nil {
			fatal(err)
		}
		defer detector.Close()
		prediction, err := detector.PredictInto(pcm, rate, channels)
		if err != nil {
			fatal(err)
		}
		if err := json.NewEncoder(os.Stdout).Encode(prediction); err != nil {
			fatal(err)
		}
		return
	}

	transcriber, err := model.NewTranscriber()
	if err != nil {
		fatal(err)
	}
	defer transcriber.Close()
	count, err := speech.Samples16k(len(pcm), rate, channels)
	if err != nil {
		fatal(err)
	}
	mono := make([]float32, count)
	resampler := speech.NewResampler()
	defer resampler.Close()
	n, err := resampler.Resample16kInto(pcm, rate, channels, mono)
	if err != nil {
		fatal(err)
	}
	opts := speech.Options{
		Language: *language,
		Context:  *contextText,
		Segments: *responseFormat == "verbose_json" || *responseFormat == "srt" || *responseFormat == "vtt",
		Words:    *wordTimestamps,
	}
	var t speech.Transcript
	if err := transcriber.Transcribe(context.Background(), mono[:n], opts, &t); err != nil {
		fatal(err)
	}
	output := make([]byte, 0, max(4096, len(t.Text)*4))
	trimmed := bytes.TrimSpace(t.Text)
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
		fatal(err)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "gophonic:", err)
	os.Exit(1)
}

// classifyText answers question about text with one of labels.
func classifyText(model *gophonic.Model, question string, labels []string, text string) {
	if question == "" || len(labels) < 2 {
		fatal(fmt.Errorf("%s answers questions about text: give -question and at least two -labels", model.Name()))
	}
	zeroShot, err := gophonic.Lane[speech.ZeroShot](model)
	if err != nil {
		fatal(err)
	}
	defer zeroShot.Close()
	classifier, err := zeroShot.Classifier(question, labels)
	if err != nil {
		fatal(err)
	}
	defer classifier.Close()
	probs := make([]float32, len(labels))
	if err := classifier.ClassifyInto(context.Background(), text, probs); err != nil {
		fatal(err)
	}
	printClasses(labels, probs)
}

// printClasses writes each label's probability, in order, as JSON.
func printClasses(labels []string, probs []float32) {
	type class struct {
		Label       string  `json:"label"`
		Probability float32 `json:"probability"`
	}
	classes := make([]class, len(labels))
	for i := range labels {
		classes[i] = class{labels[i], probs[i]}
	}
	if err := json.NewEncoder(os.Stdout).Encode(classes); err != nil {
		fatal(err)
	}
}
