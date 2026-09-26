// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/speech"
)

// counted is a model format that counts what it opens and closes.
type counted struct {
	opens, closes, lanes, laneCloses atomic.Int32
}

type countedLane struct{ c *counted }

func (l countedLane) Flag(string) bool { return true }

func (l countedLane) Close() error { l.c.laneCloses.Add(1); return nil }

// poolCounts counts for the running test.
var poolCounts atomic.Pointer[counted]

var registerPoolFormat = sync.OnceFunc(func() {
	gophonic.Register(gophonic.Format{
		Name:  "pooled",
		Match: func(p string) bool { return strings.HasSuffix(p, ".pooled") },
		Open: func(path string, _ gophonic.Options) (*gophonic.Model, error) {
			c := poolCounts.Load()
			c.opens.Add(1)
			time.Sleep(5 * time.Millisecond) // concurrent Acquires overlap the load
			m := gophonic.NewModel("pooled", path, func() error { c.closes.Add(1); return nil })
			return gophonic.Provide(m, func() (moderator, error) {
				c.lanes.Add(1)
				return countedLane{c}, nil
			}), nil
		},
	})
})

func pooledModel(t *testing.T) (string, *counted) {
	registerPoolFormat()
	c := &counted{}
	poolCounts.Store(c)
	path := filepath.Join(t.TempDir(), "m.pooled")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	return path, c
}

func TestPoolSharesAndReusesLanes(t *testing.T) {
	path, c := pooledModel(t)
	pool := gophonic.NewPool(gophonic.Options{}, -1)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			l, err := gophonic.Acquire[moderator](pool, path)
			if err != nil {
				t.Error(err)
				return
			}
			l.Lane.Flag("x")
			l.Release()
		}()
	}
	wg.Wait()
	if c.opens.Load() != 1 {
		t.Fatalf("model opened %d times; want once", c.opens.Load())
	}
	lanes := c.lanes.Load()
	for range 100 {
		l, err := gophonic.Acquire[moderator](pool, path)
		if err != nil {
			t.Fatal(err)
		}
		l.Release()
	}
	if c.lanes.Load() != lanes {
		t.Fatalf("sequential Acquires opened %d new lanes; want reuse", c.lanes.Load()-lanes)
	}
	if _, err := gophonic.Acquire[speech.Transcriber](pool, path); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("Acquire of an unprovided type = %v", err)
	}
	if allocs := testing.AllocsPerRun(100, func() {
		l, _ := gophonic.Acquire[moderator](pool, path)
		l.Release()
	}); allocs != 0 {
		t.Fatalf("warm Acquire and Release allocate %v times", allocs)
	}
	pool.Close()
	if c.closes.Load() != 1 || c.laneCloses.Load() != c.lanes.Load() {
		t.Fatalf("Close closed the model %d times and %d of %d lanes", c.closes.Load(), c.laneCloses.Load(), c.lanes.Load())
	}
	if _, err := gophonic.Acquire[moderator](pool, path); !errors.Is(err, gophonic.ErrPoolClosed) {
		t.Fatalf("Acquire after Close = %v", err)
	}
}

func TestPoolClosesIdleModels(t *testing.T) {
	path, c := pooledModel(t)
	pool := gophonic.NewPool(gophonic.Options{}, 30*time.Millisecond)
	defer pool.Close()
	l, err := gophonic.Acquire[moderator](pool, path)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(60 * time.Millisecond)
	if c.closes.Load() != 0 {
		t.Fatal("a model in use was closed")
	}
	l.Release()
	deadline := time.Now().Add(2 * time.Second)
	for c.closes.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if c.closes.Load() != 1 || c.laneCloses.Load() != 1 || len(pool.Loaded()) != 0 {
		t.Fatalf("idle model: %d closes, %d lane closes, loaded %v", c.closes.Load(), c.laneCloses.Load(), pool.Loaded())
	}
	// The next Acquire opens it again.
	l, err = gophonic.Acquire[moderator](pool, path)
	if err != nil || c.opens.Load() != 2 {
		t.Fatalf("reacquire: %v after %d opens", err, c.opens.Load())
	}
	l.Release()
}

func TestPoolCloseWaitsForLeases(t *testing.T) {
	path, c := pooledModel(t)
	pool := gophonic.NewPool(gophonic.Options{}, 0)
	l, err := gophonic.Acquire[moderator](pool, path)
	if err != nil {
		t.Fatal(err)
	}
	pool.Close()
	if c.closes.Load() != 0 {
		t.Fatal("Close closed a model with a lease out")
	}
	l.Release()
	if c.closes.Load() != 1 {
		t.Fatal("the last Release after Close left the model open")
	}
}
