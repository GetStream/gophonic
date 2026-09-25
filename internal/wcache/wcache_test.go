// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package wcache

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFreshThenMapped(t *testing.T) {
	Dir = t.TempDir()
	ckpt := t.TempDir()
	if err := os.WriteFile(filepath.Join(ckpt, "model.safetensors"), []byte("v1"), 0o644); err != nil {
		t.Fatal(err)
	}
	key, err := Key(ckpt, "gpu", "v1")
	if err != nil {
		t.Fatal(err)
	}
	fill := func(f *File) (a, b []byte) {
		l := NewLayout(f.Payload())
		return l.Take(10), l.Take(Align + 1)
	}
	f, err := Open(key, 3*Align)
	if err != nil || !f.Fresh() {
		t.Fatalf("Open = %v, fresh %v; want a fresh entry", err, f.Fresh())
	}
	a, b := fill(f)
	if len(a) != Align || len(b) != 2*Align {
		t.Fatalf("regions %d, %d bytes; want %d, %d", len(a), len(b), Align, 2*Align)
	}
	a[0], b[len(b)-1] = 7, 9
	f.Done(a)
	f.Commit()
	f.Close()

	f, err = Open(key, 3*Align)
	if err != nil || f.Fresh() {
		t.Fatalf("reopen = %v, fresh %v; want the committed entry", err, f.Fresh())
	}
	if a, b := fill(f); a[0] != 7 || b[len(b)-1] != 9 {
		t.Fatalf("reopened entry lost its bytes")
	}
	f.Close()

	// A changed checkpoint gets a new entry, which replaces the old one.
	if err := os.WriteFile(filepath.Join(ckpt, "model.safetensors"), []byte("v2!"), 0o644); err != nil {
		t.Fatal(err)
	}
	key2, _ := Key(ckpt, "gpu", "v1")
	other, _ := Key(ckpt, "gpu-q4", "v1")
	if key2 == key {
		t.Fatal("key ignores checkpoint changes")
	}
	for _, k := range []string{other, key2} {
		f, err := Open(k, Align)
		if err != nil || !f.Fresh() {
			t.Fatalf("Open(%s) = %v, fresh %v", k, err, f.Fresh())
		}
		f.Commit()
		f.Close()
	}
	names, _ := filepath.Glob(filepath.Join(Dir, "*.bin*"))
	if len(names) != 2 {
		t.Fatalf("cache holds %v; want the new gpu and gpu-q4 entries", names)
	}
}

func TestUncommittedIsDiscarded(t *testing.T) {
	Dir = t.TempDir()
	key, _ := Key(t.TempDir(), "gpu")
	f, err := Open(key, Align)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	if names, _ := filepath.Glob(filepath.Join(Dir, "*.bin*")); len(names) != 0 {
		t.Fatalf("cache holds %v after an uncommitted entry closed", names)
	}
	if f, _ = Open(key, Align); !f.Fresh() {
		t.Fatal("an uncommitted entry was reused")
	}
	f.Close()
}

func TestNoDirectory(t *testing.T) {
	Dir = ""
	f, err := Open("x.gpu.00", 5)
	if err != nil || !f.Fresh() || len(f.Payload()) != Align {
		t.Fatalf("Open without a cache directory = %v", err)
	}
	f.Payload()[0] = 1
	f.Commit()
	f.Close()
}

func TestOnePreparer(t *testing.T) {
	Dir = t.TempDir()
	key, _ := Key(t.TempDir(), "gpu")
	first, err := Open(key, Align)
	if err != nil || !first.Fresh() {
		t.Fatalf("Open = %v", err)
	}
	second := make(chan *File)
	go func() {
		f, _ := Open(key, Align) // waits for first to be committed
		second <- f
	}()
	first.Payload()[0] = 42
	time.Sleep(20 * time.Millisecond)
	first.Commit()
	f := <-second
	if f.Fresh() || f.Payload()[0] != 42 {
		t.Fatalf("a concurrent Open prepared the entry again")
	}
	f.Close()
	first.Close()
}
