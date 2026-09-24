// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestPinnedOracleFixtureIntegrity(t *testing.T) {
	dir := filepath.Join("..", "testdata", "whisper")
	data, err := os.ReadFile(filepath.Join(dir, "jfk.oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		SourceCommit     string            `json:"source_commit"`
		CheckpointSHA256 string            `json:"checkpoint_sha256"`
		PCMSHA256        string            `json:"pcm_sha256"`
		FilesSHA256      map[string]string `json:"files_sha256"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.SourceCommit != "86098128c0b4f24f0e2aa2994de830614b474227" ||
		manifest.CheckpointSHA256 != "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03" ||
		manifest.PCMSHA256 != "80a6b1e2dcc00e55e5341e4de9c44f65bc576202c93106ff6b4bfe251a87e02f" {
		t.Fatal("oracle provenance differs from pinned source, checkpoint, or PCM")
	}
	if err := verifySHA256(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"), manifest.PCMSHA256); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) != ".f32le" {
			continue
		}
		want, ok := manifest.FilesSHA256[entry.Name()]
		if !ok {
			t.Fatalf("oracle file %s lacks a pinned SHA-256", entry.Name())
		}
		if err := verifySHA256(filepath.Join(dir, entry.Name()), want); err != nil {
			t.Fatal(err)
		}
		seen++
	}
	if seen != len(manifest.FilesSHA256) {
		t.Fatalf("manifest lists %d oracle files; found %d", len(manifest.FilesSHA256), seen)
	}
}

func verifySHA256(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		return fmt.Errorf("%s SHA-256 %s, want %s", path, got, want)
	}
	return nil
}
