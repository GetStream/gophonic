// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command gophonic-server serves bounded local Whisper file transcription.
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

	"github.com/GetStream/gophonic/internal/httpserver"
	"github.com/GetStream/gophonic/whisper"
)

func main() {
	modelPath := flag.String("whisper-model", "", "converted English Whisper .gophonic bundle")
	listen := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	workers := flag.Int("workers", 1, "independent transcription lanes")
	maxSeconds := flag.Int("max-audio-seconds", 120, "maximum decoded audio duration")
	flag.Parse()
	if *modelPath == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: gophonic-server -whisper-model MODEL [-listen 127.0.0.1:8080] [-workers 1]")
		os.Exit(2)
	}
	model, err := whisper.Load(*modelPath)
	if err != nil {
		fatal(err)
	}
	handler, err := httpserver.NewWhisperServer(model, *workers, *maxSeconds)
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
