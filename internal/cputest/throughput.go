// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin || linux

// Package cputest provides an opt-in concurrent inference measurement harness.
// It is imported only by model tests, never by inference code.
package cputest

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// Lane owns independent inference state. Run measures a complete public call;
// Fingerprint checks its numeric state and output outside the timed region.
type Lane struct {
	Run         func() error
	Fingerprint func() [32]byte
	Close       func()
}

type Sample struct {
	Lanes, Threads, Round, Lane int
	NS                          int64
	Fingerprint                 string
}

type Wave struct {
	Lanes, Threads, Round int
	CPUNS, NS             int64
}

// Run records warmed concurrent cohorts. Each lane's full and shorter input
// must be deterministic; the factory selects them by lane index. Results are
// checkpointed after every cohort, so a timed-out overload run preserves all
// completed measurements. These are inference timings, not HTTP timings.
func Run(t *testing.T, newLane func(index, workers int) Lane) {
	t.Helper()
	path := os.Getenv("STT_SERVER_OUTPUT")
	if path == "" {
		t.Skip("set STT_SERVER_OUTPUT for concurrent CPU measurements")
	}
	rounds := 4
	if n, err := strconv.Atoi(os.Getenv("STT_SERVER_ROUNDS")); err == nil && n > 0 {
		rounds = n
	}
	report := struct {
		CPU, Go, OS, Arch string
		Procs             int
		Samples           []Sample
		Waves             []Wave
	}{CPU: os.Getenv("STT_SERVER_CPU"), Go: runtime.Version(), OS: runtime.GOOS, Arch: runtime.GOARCH, Procs: runtime.GOMAXPROCS(0)}
	configs := os.Getenv("STT_SERVER_CONFIGS")
	if configs == "" {
		configs = "1:8,4:4,8:2,8:8"
	}
	for _, conf := range strings.Split(configs, ",") {
		var count, threads int
		if n, err := fmt.Sscanf(conf, "%d:%d", &count, &threads); err != nil || n != 2 || count < 1 || count > 64 || threads < 1 || threads > 64 {
			t.Fatalf("invalid lanes:threads configuration %q", conf)
		}
		lanes := make([]Lane, count)
		want := make([][32]byte, count)
		for i := range lanes {
			lanes[i] = newLane(i, threads)
			// Also clean up after an exactness failure or a canceled call.
			t.Cleanup(lanes[i].Close)
			for range 2 {
				if err := lanes[i].Run(); err != nil {
					t.Fatal(err)
				}
			}
			want[i] = lanes[i].Fingerprint()
		}
		t.Logf("%s warmed", conf)
		for round := -1; round < rounds; round++ {
			var ready, done sync.WaitGroup
			ready.Add(count)
			done.Add(count)
			start := make(chan struct{})
			durations := make([]int64, count)
			errs := make([]error, count)
			for i := range count {
				go func(i int) {
					defer done.Done()
					ready.Done()
					<-start
					began := time.Now()
					errs[i] = lanes[i].Run()
					durations[i] = time.Since(began).Nanoseconds()
				}(i)
			}
			ready.Wait()
			cpu := cpuTime()
			began := time.Now()
			close(start)
			done.Wait()
			elapsed := time.Since(began).Nanoseconds()
			cpu = cpuTime() - cpu
			if round >= 0 {
				report.Waves = append(report.Waves, Wave{count, threads, round, cpu, elapsed})
			}
			for i := range count {
				if errs[i] != nil {
					t.Fatal(errs[i])
				}
				got := lanes[i].Fingerprint()
				if got != want[i] {
					t.Fatalf("%s lane%d changed numeric state or output", conf, i)
				}
				if round >= 0 {
					report.Samples = append(report.Samples, Sample{count, threads, round, i, durations[i], fmt.Sprintf("%x", got)})
				}
			}
			data, err := json.MarshalIndent(report, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(path, append(data, '\n'), 0644); err != nil {
				t.Fatal(err)
			}
			t.Logf("%s round%d elapsed=%s CPU=%s", conf, round, time.Duration(elapsed), time.Duration(cpu))
		}
		for _, lane := range lanes {
			lane.Close()
		}
	}
}

func cpuTime() int64 {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		panic(err)
	}
	return usage.Utime.Nano() + usage.Stime.Nano()
}
