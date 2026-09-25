// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package wcache keeps prepared model weights in memory-mapped files, so a
// checkpoint is converted once. The first load writes the weights it
// prepares (quantized, rotated, packed) straight into a new cache file;
// later loads map that file and use it in place, and with a warm page cache
// they read nothing at all. On Apple silicon the mapped pages become GPU
// buffers without a copy.
//
// Entries live in the user cache directory (~/Library/Caches/gophonic on
// macOS, ~/.cache/gophonic on Linux), or in $GOPHONIC_CACHE. An entry is
// named after its checkpoint directory and kind of weights, plus a hash of
// everything the prepared bytes depend on; committing an entry removes the
// entries it replaces. When no directory can be written, loads still work,
// preparing weights in anonymous memory each time.
package wcache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/gophonic/internal/mmap"
)

// Align is the alignment of every region: the largest page size of the
// supported platforms (16 KiB on Apple silicon), which GPU buffers made from
// mapped memory require.
const Align = 16 << 10

const (
	magic      = "GPHWCACH"
	headerSize = Align
	complete   = 1
)

// Dir is where entries are kept; empty disables the cache.
var Dir = defaultDir()

func defaultDir() string {
	if dir, ok := os.LookupEnv("GOPHONIC_CACHE"); ok {
		return dir
	}
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "gophonic")
}

// Key names the entry of one kind of prepared weights (such as "gpu-q8")
// for the checkpoint in dir. parts are everything else the prepared bytes
// depend on, such as a layout version and options; the checkpoint's files
// are fingerprinted by name, size, and modification time, so an entry is
// rebuilt after they change.
func Key(dir, kind string, parts ...string) (string, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	for _, p := range parts {
		fmt.Fprintf(h, "%s\x00", p)
	}
	for _, e := range entries {
		if info, err := e.Info(); err == nil && info.Mode().IsRegular() {
			fmt.Fprintf(h, "%s\x00%d\x00%d\x00", e.Name(), info.Size(), info.ModTime().UnixNano())
		}
	}
	where := sha256.Sum256([]byte(abs))
	return fmt.Sprintf("%s-%s.%s.%s", sanitize(filepath.Base(abs)), hex.EncodeToString(where[:4]),
		sanitize(kind), hex.EncodeToString(h.Sum(nil)[:12])), nil
}

// sanitize keeps names portable, and free of the dots that separate a key's
// fields.
func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, s)
}

// File is one mapped cache entry.
type File struct {
	data      []byte // the whole mapping: header, then the payload
	fd        *os.File
	path, tmp string
	fresh     bool
	lock      *os.File // held while the entry is prepared
}

// Open maps the entry key with a payload of size bytes. When a complete
// entry exists it is mapped read-only and Fresh reports false. Otherwise
// Open returns a zeroed, writable mapping that the caller fills and then
// commits.
func Open(key string, size int) (*File, error) {
	if size < 0 {
		return nil, errors.New("wcache: negative size")
	}
	total := headerSize + roundUp(max(size, 1))
	if Dir != "" {
		path := filepath.Join(Dir, key+".bin")
		if f, err := openComplete(path, total); err == nil {
			return f, nil
		}
		// One loader prepares an entry; others wait for it and map it.
		os.MkdirAll(Dir, 0o755)
		lk := lock(filepath.Join(Dir, key[:strings.LastIndexByte(key, '.')+1]+"lock"))
		if f, err := openComplete(path, total); err == nil {
			lk.Close()
			return f, nil
		}
		if f, err := create(path, total); err == nil {
			f.lock = lk
			return f, nil
		}
		lk.Close()
	}
	data, err := mmap.Anonymous(total)
	if err != nil {
		return nil, err
	}
	return &File{data: data, fresh: true}, nil
}

func openComplete(path string, total int) (*File, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	if info, err := fd.Stat(); err != nil || info.Size() != int64(total) {
		return nil, errors.New("wcache: size mismatch")
	}
	data, err := mmap.File(fd, 0, total, false)
	if err != nil {
		return nil, err
	}
	if string(data[:8]) != magic || binary.LittleEndian.Uint64(data[8:]) != uint64(total) ||
		binary.LittleEndian.Uint64(data[16:]) != complete {
		mmap.Unmap(data)
		return nil, errors.New("wcache: incomplete entry")
	}
	return &File{data: data, path: path}, nil
}

func create(path string, total int) (*File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	fd, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*.tmp")
	if err != nil {
		return nil, err
	}
	data, err := func() ([]byte, error) {
		if err := fd.Truncate(int64(total)); err != nil {
			return nil, err
		}
		return mmap.File(fd, 0, total, true)
	}()
	if err != nil {
		fd.Close()
		os.Remove(fd.Name())
		return nil, err
	}
	copy(data, magic)
	binary.LittleEndian.PutUint64(data[8:], uint64(total))
	return &File{data: data, fd: fd, path: path, tmp: fd.Name(), fresh: true}, nil
}

// Fresh reports whether the entry is new: the caller must fill the payload
// and then call Commit.
func (f *File) Fresh() bool { return f.fresh }

// Payload is the entry's bytes after its header, Align-aligned: writable
// while the entry is fresh, read-only once it is complete. They stay valid
// until Commit, which moves them, or Close.
func (f *File) Payload() []byte { return f.data[headerSize:] }

// Done tells a fresh entry that region, part of its payload, is filled, so
// its pages can be written to disk while the caller fills the rest and
// Commit has less to wait for. It returns at once.
func (f *File) Done(region []byte) {
	if f.fresh && f.fd != nil {
		mmap.Flush(region)
	}
}

// Commit publishes a fresh entry once its payload is filled: it writes the
// payload to disk, marks it complete, renames it into place, removing the
// entries it replaces (those of the same checkpoint and kind), and maps it
// again read-only. The payload moves: take regions of Payload after Commit,
// and only then hand them to the GPU, which so only ever reads complete,
// read-only mappings, as llama.cpp's do. When the entry cannot be
// published, its bytes stay usable from anonymous memory, and the next load
// prepares them again. Commit does nothing for an entry that was already
// complete.
func (f *File) Commit() {
	if !f.fresh {
		return
	}
	f.fresh = false
	if f.fd == nil {
		return // anonymous memory: nothing to publish
	}
	defer f.lock.Close()
	defer f.fd.Close()
	if err := f.publish(); err != nil {
		os.Remove(f.tmp)
		if mem, err := mmap.Anonymous(len(f.data)); err == nil {
			copy(mem, f.data)
			mmap.Unmap(f.data)
			f.data = mem
		}
		return
	}
	done, err := openComplete(f.path, len(f.data))
	if err != nil {
		return // keep the writable mapping of the published file
	}
	mmap.Unmap(f.data)
	f.data = done.data
}

func (f *File) publish() error {
	// The payload reaches the disk before the mark that says it is complete.
	if err := f.fd.Sync(); err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(f.data[16:], complete)
	if err := f.fd.Sync(); err != nil {
		return err
	}
	if err := os.Rename(f.tmp, f.path); err != nil {
		return err
	}
	name := filepath.Base(f.path)
	prefix := name[:strings.LastIndexByte(strings.TrimSuffix(name, ".bin"), '.')+1]
	old, _ := filepath.Glob(filepath.Join(filepath.Dir(f.path), prefix+"*.bin"))
	for _, p := range slices.DeleteFunc(old, func(p string) bool {
		rest := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(p), prefix), ".bin")
		return p == f.path || strings.Contains(rest, ".")
	}) {
		os.Remove(p)
	}
	// Temporary files an hour old were left by processes that died.
	tmps, _ := filepath.Glob(filepath.Join(filepath.Dir(f.path), prefix+"*.tmp"))
	for _, p := range tmps {
		if info, err := os.Stat(p); err == nil && time.Since(info.ModTime()) > time.Hour {
			os.Remove(p)
		}
	}
	return nil
}

// Close unmaps the entry, discarding it if it is still fresh. Nothing may
// use its memory afterwards, GPU buffers made from it included.
func (f *File) Close() error {
	if f == nil || f.data == nil {
		return nil
	}
	if f.fresh && f.fd != nil {
		f.fd.Close()
		os.Remove(f.tmp)
		f.lock.Close()
	}
	err := mmap.Unmap(f.data)
	f.data = nil
	return err
}

// Layout cuts a payload into Align-aligned regions. Run the same sequence
// of Take calls twice: first on a nil payload to measure it, then on the
// mapped payload to cut it.
type Layout struct {
	payload []byte
	off     int
}

// NewLayout starts cutting payload, or measuring when it is nil.
func NewLayout(payload []byte) *Layout { return &Layout{payload: payload} }

// Take returns the next n bytes of the payload, n rounded up to Align, or
// nil while measuring.
func (l *Layout) Take(n int) []byte {
	start := l.off
	l.off += roundUp(max(n, 1))
	if l.payload == nil {
		return nil
	}
	return l.payload[start:l.off:l.off]
}

// Size reports the bytes taken so far.
func (l *Layout) Size() int { return l.off }

func roundUp(n int) int { return (n + Align - 1) &^ (Align - 1) }
