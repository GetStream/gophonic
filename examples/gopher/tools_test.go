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
	got, err := clock.Call(context.Background(), []byte(`{"timezone": "Asia/Tokyo"}`))
	if err != nil || !strings.Contains(got, "Asia/Tokyo") || !strings.Contains(got, time.Now().In(mustZone(t, "Asia/Tokyo")).Format("January 2, 2006")) {
		t.Fatalf("%q %v", got, err)
	}
	if _, err := clock.Call(context.Background(), []byte(`{"timezone": "Mars/Olympus"}`)); err == nil {
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
	got, err := tools()[1].Call(context.Background(), []byte(`{"query": "Eiffel Tower height"}`))
	if err != nil {
		t.Skip("Wikipedia unreachable:", err)
	}
	t.Log(got)
	if !strings.Contains(got, "Eiffel") {
		t.Fatalf("%q", got)
	}
}
