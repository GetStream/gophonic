// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestNow(t *testing.T) {
	clock := tools()[0]
	got, reply, err := clock.Run(context.Background(), []byte(`{"timezone": "Asia/Tokyo"}`))
	if err != nil || !reply || !strings.Contains(got, "Asia/Tokyo") || !strings.Contains(got, time.Now().In(mustZone(t, "Asia/Tokyo")).Format("January 2, 2006")) {
		t.Fatalf("%q %v %v", got, reply, err)
	}
	if _, _, err := clock.Run(context.Background(), []byte(`{"timezone": "Mars/Olympus"}`)); err == nil {
		t.Fatal("an unknown zone")
	}
}

func mustZone(t *testing.T, name string) *time.Location {
	loc, err := time.LoadLocation(name)
	if err != nil {
		t.Fatal(err)
	}
	return loc
}

func TestSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("needs Wikipedia")
	}
	got, reply, err := tools()[1].Run(context.Background(), []byte(`{"query": "Eiffel Tower height"}`))
	if err != nil {
		t.Skip("Wikipedia unreachable:", err)
	}
	t.Log(got)
	if !reply || !strings.Contains(got, "Eiffel") {
		t.Fatalf("%q", got)
	}
}
