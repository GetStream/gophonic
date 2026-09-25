// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command gophonic-server serves bounded local file transcription with any
// transcription model gophonic.Open recognizes, over an OpenAI-compatible
// POST /v1/audio/transcriptions endpoint.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/httpserver"
)

func main() {
	modelPath := flag.String("model", "", "transcription model: a converted Whisper .gophonic bundle or a Qwen3-ASR checkpoint directory")
	listen := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	workers := flag.Int("workers", 1, "independent transcription lanes")
	maxSeconds := flag.Int("max-audio-seconds", 120, "maximum decoded audio duration")
	threads := flag.Int("threads", 0, "CPU workers per lane (0: model default)")
	flag.Parse()
	if *modelPath == "" || flag.NArg() != 0 || *threads < 0 {
		fmt.Fprintln(os.Stderr, "usage: gophonic-server -model MODEL [-listen 127.0.0.1:8080] [-workers 1] [-threads 0]")
		os.Exit(2)
	}
	model, err := gophonic.Open(*modelPath, gophonic.Options{Threads: *threads})
	if err != nil {
		fatal(err)
	}
	if model.Kind() != gophonic.Transcription {
		fatal(fmt.Errorf("%s is a %s model; the server needs a transcription model", model.Name(), model.Kind()))
	}
	handler, err := httpserver.NewServer(model.NewTranscriber, *workers, *maxSeconds)
	if err != nil {
		fatal(err)
	}
	defer handler.Close()
	httpd := &http.Server{
		Addr:              *listen,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	shutdownDone := make(chan struct{})
	go func() {
		<-ctx.Done()
		_ = httpd.Shutdown(context.Background())
		close(shutdownDone)
	}()
	fmt.Fprintf(os.Stderr, "gophonic-server listening on %s\n", *listen)
	if err := httpd.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fatal(err)
	}
	if ctx.Err() != nil {
		<-shutdownDone
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "gophonic-server:", err)
	os.Exit(1)
}
