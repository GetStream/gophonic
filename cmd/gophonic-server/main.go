// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command gophonic-server serves models over HTTP:
//
//	gophonic-server ~/models
//
// It serves every model it finds in the directories and paths it is given.
// Transcription (POST /v1/audio/transcriptions) and the model list (GET
// /v1/models) follow the OpenAI API; audio classification, such as turn
// detection (POST /v1/audio/classifications), and text classification (POST
// /v1/classifications) follow its conventions. Models load on first use and
// close after -keep-alive without requests.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/httpserver"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	workers := flag.Int("workers", 1, "requests running inference at once")
	maxSeconds := flag.Int("max-audio-seconds", 120, "longest audio a request may send")
	threads := flag.Int("threads", 0, "CPU workers per lane (0: model default)")
	keepAlive := flag.Duration("keep-alive", gophonic.DefaultKeepAlive, "close a model after this long without requests (negative: never)")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: gophonic-server [flags] MODEL|DIRECTORY...\n\n"+
			"Serves every model in the given paths; a directory is searched one level deep.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() == 0 || *threads < 0 {
		flag.Usage()
		os.Exit(2)
	}
	models, err := find(flag.Args())
	if err != nil {
		fatal(err)
	}
	pool := gophonic.NewPool(gophonic.Options{Threads: *threads}, *keepAlive)
	defer pool.Close()
	handler, err := httpserver.NewServer(httpserver.Config{Pool: pool, Models: models, Workers: *workers, MaxSeconds: *maxSeconds})
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
	for _, m := range models {
		fmt.Fprintf(os.Stderr, "serving %s (%s)\n", m.Name, m.Format.Name)
	}
	fmt.Fprintf(os.Stderr, "gophonic-server listening on http://%s\n", *listen)
	if err := httpd.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fatal(err)
	}
	if ctx.Err() != nil {
		<-shutdownDone
	}
}

// find lists the models at paths: each path that is a model, and the
// models directly inside each directory that is not.
func find(paths []string) ([]httpserver.Model, error) {
	var models []httpserver.Model
	add := func(path string, f gophonic.Format) {
		name := strings.TrimSuffix(filepath.Base(path), ".gophonic")
		models = append(models, httpserver.Model{Name: name, Path: path, Format: f})
	}
	for _, path := range paths {
		if f, ok := gophonic.Detect(path); ok {
			add(path, f)
			continue
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, fmt.Errorf("%s is neither a model nor a directory of models", path)
		}
		found := len(models)
		for _, e := range entries {
			child := filepath.Join(path, e.Name())
			if f, ok := gophonic.Detect(child); ok {
				add(child, f)
			}
		}
		if len(models) == found {
			return nil, fmt.Errorf("found no model in %s", path)
		}
	}
	return models, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "gophonic-server:", err)
	os.Exit(1)
}
