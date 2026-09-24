// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"testing"
)

func identityFixture(t *testing.T) *HeadPair {
	t.Helper()
	cfg := Config{EncoderDim: 2, Width: 2, Depth: 2, ProjectionDim: 2, Activation: "gelu"}
	tensors := make(map[string][]float32)
	for _, spec := range tensorSpecs(cfg) {
		count := 1
		for _, d := range spec.shape {
			count *= d
		}
		tensors[spec.name] = make([]float32, count)
	}
	for _, head := range []string{"state_head", "action_head"} {
		for i := 0; i < 2; i++ {
			tensors[head+".inp.weight"][i*2+i] = 1
			tensors[head+".out.weight"][i*2+i] = 1
		}
	}
	manifest := bundleManifest{Format: "gophonic-clm", Version: 1, Config: cfg, LogitScale: float32(math.Log(2)), Scale: 2,
		Provenance: Provenance{SourceSHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}
	for _, spec := range tensorSpecs(cfg) {
		var raw bytes.Buffer
		for _, x := range tensors[spec.name] {
			_ = binary.Write(&raw, binary.LittleEndian, x)
		}
		sum := sha256.Sum256(raw.Bytes())
		manifest.Tensors = append(manifest.Tensors, tensorRecord{Name: spec.name, Shape: spec.shape, SHA256: hex.EncodeToString(sum[:])})
	}
	var bundle bytes.Buffer
	if err := writeBundle(&bundle, manifest, tensors); err != nil {
		t.Fatal(err)
	}
	head, err := ReadWeights(bytes.NewReader(bundle.Bytes()))
	if err != nil {
		t.Fatalf("ReadWeights: %v", err)
	}
	return head
}

func TestHeadProjectionAndScoreReference(t *testing.T) {
	head := identityFixture(t)
	ws := head.NewWorkspace()
	state := []float32{1, 1}
	projected := make([]float32, 2)
	if err := head.ProjectStateInto(state, projected, ws); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(projected[0])-1/math.Sqrt(2)) > 1e-6 || math.Abs(float64(projected[1])-1/math.Sqrt(2)) > 1e-6 {
		t.Fatalf("normalized state projection = %v, want [1/sqrt(2) 1/sqrt(2)]", projected)
	}
	actions := [][]float32{{1, 0}, {0, 1}, {-1, 0}}
	scores := make([]float32, len(actions))
	if err := head.ScoreInto(state, actions, 1, scores, ws); err != nil {
		t.Fatal(err)
	}
	want := []float32{float32(math.Sqrt(2)), float32(math.Sqrt(2)), -float32(math.Sqrt(2))}
	for i := range want {
		if math.Abs(float64(scores[i]-want[i])) > 2e-6 {
			t.Errorf("scaled cosine score %d = %.8f, want %.8f", i, scores[i], want[i])
		}
	}
}

func TestLayerNormActivationResidualOrder(t *testing.T) {
	cfg := Config{EncoderDim: 2, Width: 2, Depth: 3, ProjectionDim: 2, Activation: "relu", LayerNorm: true, Residual: true}
	tensors := make(map[string][]float32)
	for _, spec := range tensorSpecs(cfg) {
		count := 1
		for _, d := range spec.shape {
			count *= d
		}
		tensors[spec.name] = make([]float32, count)
	}
	for _, head := range []string{"state_head", "action_head"} {
		for i := 0; i < 2; i++ {
			tensors[head+".inp.weight"][i*2+i] = 1
			tensors[head+".hidden.0.weight"][i*2+i] = 1
			tensors[head+".out.weight"][i*2+i] = 1
			tensors[head+".hidden.0.norm.weight"][i] = 1
		}
		tensors[head+".hidden.0.norm.bias"][1] = 2
	}
	manifest := manifestFor(cfg, tensors)
	var bundle bytes.Buffer
	if err := writeBundle(&bundle, manifest, tensors); err != nil {
		t.Fatal(err)
	}
	head, err := ReadWeights(bytes.NewReader(bundle.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, 2)
	if err := head.ProjectStateInto([]float32{1, 0}, out, head.NewWorkspace()); err != nil {
		t.Fatal(err)
	}
	// PyTorch LayerNorm over [1,0] is approximately [1,-1]; norm bias, ReLU,
	// residual addition, and final normalization yield [2,1]/sqrt(5).
	if math.Abs(float64(out[0])-2/math.Sqrt(5)) > 2e-5 || math.Abs(float64(out[1])-1/math.Sqrt(5)) > 2e-5 {
		t.Fatalf("layernorm/activation/residual projection = %v, want [2,1]/sqrt(5)", out)
	}
}

func manifestFor(cfg Config, tensors map[string][]float32) bundleManifest {
	m := bundleManifest{Format: "gophonic-clm", Version: 1, Config: cfg, LogitScale: 0, Scale: 1}
	for _, spec := range tensorSpecs(cfg) {
		var raw bytes.Buffer
		for _, x := range tensors[spec.name] {
			_ = binary.Write(&raw, binary.LittleEndian, x)
		}
		sum := sha256.Sum256(raw.Bytes())
		m.Tensors = append(m.Tensors, tensorRecord{Name: spec.name, Shape: spec.shape, SHA256: hex.EncodeToString(sum[:])})
	}
	return m
}

type fixtureEmbedder struct{ values map[string][]float32 }

func (e *fixtureEmbedder) Embed(_ context.Context, _ Role, texts []string, dst [][]float32) error {
	for i, text := range texts {
		copy(dst[i], e.values[text])
	}
	return nil
}

func TestRankReferenceSoftmaxAndStableOrdering(t *testing.T) {
	head := identityFixture(t)
	embedder := &fixtureEmbedder{values: map[string][]float32{
		"state": {1, 1}, "left": {1, 0}, "up": {0, 1}, "down": {-1, 0}, "up-again": {0, 1},
	}}
	engine, err := NewEngine(head, embedder)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := engine.NewWorkspace(4)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	candidates := []string{"left", "up", "down"}
	got := make([]RankedCandidate, 3)
	if err := engine.RankInto(ctx, "state", candidates, 1, got, ws); err != nil {
		t.Fatal(err)
	}
	want := []struct {
		candidate   string
		probability float64
	}{
		{"left", 0.4856477146303332}, {"up", 0.4856477146303332}, {"down", 0.028704570739333718},
	}
	for i := range want {
		if got[i].Candidate != want[i].candidate || got[i].Rank != i+1 || math.Abs(float64(got[i].Probability)-want[i].probability) > 2e-6 {
			t.Errorf("rank %d = %+v, want candidate %q probability %.9f", i, got[i], want[i].candidate, want[i].probability)
		}
	}
	candidates = []string{"up", "up-again", "left", "down"}
	got = make([]RankedCandidate, 4)
	if err := engine.RankInto(ctx, "state", candidates, 1, got, ws); err != nil {
		t.Fatal(err)
	}
	if got[0].Candidate != "up" || got[1].Candidate != "up-again" {
		t.Fatalf("equal scores must preserve input order, got %q then %q", got[0].Candidate, got[1].Candidate)
	}
	allocs := testing.AllocsPerRun(100, func() {
		if err := engine.RankInto(ctx, "state", candidates, 1, got, ws); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Errorf("warmed RankInto allocated %.2f objects per call, want 0", allocs)
	}
}

func TestNewWorkspaceRejectsOverflowAndExcessiveCapacity(t *testing.T) {
	head := identityFixture(t)
	engine, err := NewEngine(head, &fixtureEmbedder{values: map[string][]float32{}})
	if err != nil {
		t.Fatal(err)
	}
	bytesPerCandidate := 4*head.config.EncoderDim + 64
	tooManyForMemoryCap := maxWorkspaceBytes/bytesPerCandidate + 1
	if _, err := engine.NewWorkspace(tooManyForMemoryCap); err == nil {
		t.Fatal("NewWorkspace accepted capacity above its memory limit")
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := engine.NewWorkspace(maxInt); err == nil {
		t.Fatal("NewWorkspace accepted an overflowing capacity")
	}
}

func TestReadWeightsDetectsPayloadCorruption(t *testing.T) {
	// Build a valid small bundle, then flip a byte in its final tensor payload.
	cfg := Config{EncoderDim: 2, Width: 2, Depth: 2, ProjectionDim: 2, Activation: "relu"}
	tensors := make(map[string][]float32)
	for _, spec := range tensorSpecs(cfg) {
		count := 1
		for _, d := range spec.shape {
			count *= d
		}
		tensors[spec.name] = make([]float32, count)
	}
	manifest := manifestFor(cfg, tensors)
	var bundle bytes.Buffer
	if err := writeBundle(&bundle, manifest, tensors); err != nil {
		t.Fatal(err)
	}
	raw := append([]byte(nil), bundle.Bytes()...)
	raw[len(raw)-1] ^= 0x01
	if _, err := ReadWeights(bytes.NewReader(raw)); err == nil {
		t.Fatal("ReadWeights accepted a corrupted tensor payload")
	}
}

func TestGELUExactMatchesPyTorchErfDefinition(t *testing.T) {
	values := []float32{-1, 0, 1}
	activateInPlace("gelu", values)
	want := []float32{-0.158655256, 0, 0.841344744}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 2e-7 {
			t.Errorf("gelu(%d) = %.9f, want %.9f", i-1, values[i], want[i])
		}
	}
}

// TestReferenceHeadOfficialCheckpointOracle is an optional release gate. Its
// expected values were produced by the upstream CLM PyTorch HeadPair for the
// checkpoint whose SHA-256 is pinned below. CI can run the same test by setting
// GOPHONIC_CLM_BUNDLE to a bundle converted from that checkpoint.
func TestReferenceHeadOfficialCheckpointOracle(t *testing.T) {
	path := os.Getenv("GOPHONIC_CLM_BUNDLE")
	if path == "" {
		t.Skip("set GOPHONIC_CLM_BUNDLE to a converted CLM-v0.1-8B checkpoint")
	}
	head, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := head.Provenance().SourceSHA256; got != "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5" {
		t.Fatalf("checkpoint provenance SHA-256 = %q, want official reference hash", got)
	}
	cfg := head.Config()
	if cfg.EncoderDim != 4096 || cfg.Width != 1536 || cfg.Depth != 3 || cfg.ProjectionDim != 512 || cfg.Activation != "gelu" || !cfg.LayerNorm || cfg.Residual {
		t.Fatalf("official reference config = %+v", cfg)
	}
	state := oracleInput(7, 0, 101, 50)
	actionA := oracleInput(13, 17, 97, 48)
	actionB := oracleInput(19, 23, 89, 44)
	stateProjection := make([]float32, 512)
	actionProjectionA := make([]float32, 512)
	actionProjectionB := make([]float32, 512)
	ws := head.NewWorkspace()
	if err := head.ProjectStateInto(state, stateProjection, ws); err != nil {
		t.Fatal(err)
	}
	if err := head.ProjectActionInto(actionA, actionProjectionA, ws); err != nil {
		t.Fatal(err)
	}
	if err := head.ProjectActionInto(actionB, actionProjectionB, ws); err != nil {
		t.Fatal(err)
	}
	check := func(label string, got []float32, first, last []float32) {
		t.Helper()
		for i, want := range first {
			if delta := math.Abs(float64(got[i] - want)); delta > 3e-4 {
				t.Errorf("%s projection[%d] = %.8f, PyTorch oracle %.8f (delta %.3g)", label, i, got[i], want, delta)
			}
		}
		for i, want := range last {
			at := len(got) - len(last) + i
			if delta := math.Abs(float64(got[at] - want)); delta > 3e-4 {
				t.Errorf("%s projection[%d] = %.8f, PyTorch oracle %.8f (delta %.3g)", label, at, got[at], want, delta)
			}
		}
	}
	check("state", stateProjection,
		[]float32{-0.049917173, 0.007642230, 0.000976227, 0.031786099, -0.034906231, -0.009084376, 0.017271379, 0.070618190},
		[]float32{0.026716953, 0.028201947, -0.006836416, -0.024881443, 0.003527065, 0.008419422, -0.000754512, 0.147522330})
	check("action A", actionProjectionA,
		[]float32{0.009574241, 0.007566117, 0.056999248, -0.004690434, 0.054093204, -0.002233587, -0.010026574, -0.003137844},
		[]float32{0.056849502, 0.017425874, 0.024173254, 0.083122559, 0.046671305, 0.002280591, 0.063061960, 0.043439336})
	check("action B", actionProjectionB,
		[]float32{-0.064450182, -0.035190210, -0.055921104, 0.014680642, 0.032341473, -0.026056137, -0.050183628, 0.038220882},
		[]float32{0.009265338, -0.014286243, -0.031100389, -0.031574849, -0.003927135, -0.038878612, 0.015758855, 0.026741277})
	scores := make([]float32, 2)
	if err := head.ScoreInto(state, [][]float32{actionA, actionB}, 1, scores, ws); err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{11.94599247, 17.04429626} {
		if delta := math.Abs(float64(scores[i] - want)); delta > 2e-3 {
			t.Errorf("official score %d = %.7f, PyTorch oracle %.7f (delta %.3g)", i, scores[i], want, delta)
		}
	}
}

func oracleInput(multiplier, offset, modulus, denominator int) []float32 {
	out := make([]float32, 4096)
	for i := range out {
		out[i] = float32((i*multiplier+offset)%modulus-modulus/2) / float32(denominator)
	}
	return out
}
