// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package clm loads and runs the trained CLM state/action projection heads.
// The Qwen3 text encoder is a separate dependency and is supplied through an
// Embedder when ranking text.
package clm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"github.com/GetStream/gophonic/internal/whispergemm"
	"github.com/thesyncim/vibejson"
	"io"
	"math"
	"os"
	"strings"
)

const (
	bundleMagic   = "GCLMCPU1"
	bundleVersion = uint32(1)
	maxDepth      = 64
	maxWidth      = 16384
	maxProjection = 16384
	maxParameters = 250_000_000
)

var (
	ErrInvalidModel = errors.New("clm: invalid model")
	ErrWorkspace    = errors.New("clm: invalid workspace")
)

// Config describes the two projection heads. Depth counts linear layers: the
// first maps encoder_dim to width, the last maps width to projection_dim, and
// any intervening layers map width to width.
type Config struct {
	EncoderDim    int    `json:"encoder_dim"`
	Width         int    `json:"width"`
	Depth         int    `json:"depth"`
	ProjectionDim int    `json:"projection_dim"`
	Activation    string `json:"activation"`
	LayerNorm     bool   `json:"layernorm"`
	Residual      bool   `json:"residual"`
}

// Provenance records the original checkpoint from which the CPU bundle was
// converted. It is informational unless the caller separately verifies the
// file hash before conversion.
type Provenance struct {
	SourceRepo     string `json:"source_repo,omitempty"`
	SourceRevision string `json:"source_revision,omitempty"`
	SourceFile     string `json:"source_file,omitempty"`
	SourceSHA256   string `json:"source_sha256"`
}

type tensorRecord struct {
	Name   string `json:"name"`
	Shape  []int  `json:"shape"`
	SHA256 string `json:"sha256"`
}

type bundleManifest struct {
	Format     string         `json:"format"`
	Version    int            `json:"version"`
	Config     Config         `json:"config"`
	LogitScale float32        `json:"logit_scale"`
	Scale      float32        `json:"scale"`
	Provenance Provenance     `json:"provenance"`
	Tensors    []tensorRecord `json:"tensors"`
}

// linear holds torch.nn.Linear weights packed once for batched GEMM. The
// row-major [out,in] checkpoint tensor is packed as the transposed right
// matrix, so each call multiplies all rows of a batch against one weight read.
type linear struct {
	packed *whispergemm.PackedB // K=in, N=out
	bias   []float32
	in     int
	out    int
}

type block struct {
	linear linear
	normW  []float32
	normB  []float32
}

type projectionHead struct {
	in         linear
	hidden     []block
	out        linear
	activation string
	layerNorm  bool
	residual   bool
}

// HeadPair is an immutable state/action head pair from a converted CLM
// checkpoint. It is safe for concurrent use when each call has its own
// Workspace.
type HeadPair struct {
	config     Config
	scale      float32
	state      projectionHead
	action     projectionHead
	provenance Provenance
}

// Load opens a converted .gclm CPU bundle.
func Load(path string) (*HeadPair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadWeights(f)
}

// ReadWeights reads the versioned little-endian bundle emitted by
// tools/clm_pt_to_gophonic.py.
func ReadWeights(r io.Reader) (*HeadPair, error) {
	if r == nil {
		return nil, fmt.Errorf("clm: nil bundle reader")
	}
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("clm: read bundle header: %w", err)
	}
	if string(magic[:]) != bundleMagic {
		return nil, fmt.Errorf("clm: unsupported bundle magic %q", string(magic[:]))
	}
	var version, manifestSize uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("clm: read bundle version: %w", err)
	}
	if version != bundleVersion {
		return nil, fmt.Errorf("clm: unsupported bundle version %d", version)
	}
	if err := binary.Read(r, binary.LittleEndian, &manifestSize); err != nil {
		return nil, fmt.Errorf("clm: read manifest length: %w", err)
	}
	if manifestSize == 0 || manifestSize > 1<<20 {
		return nil, fmt.Errorf("clm: invalid manifest length %d", manifestSize)
	}
	manifestBytes := make([]byte, manifestSize)
	if _, err := io.ReadFull(r, manifestBytes); err != nil {
		return nil, fmt.Errorf("clm: read manifest: %w", err)
	}
	var manifest bundleManifest
	if err := vibejson.Unmarshal(manifestBytes, &manifest); err != nil {
		return nil, fmt.Errorf("clm: decode manifest: %w", err)
	}
	if err := validateManifest(manifest); err != nil {
		return nil, err
	}
	state, action := projectionHead{}, projectionHead{}
	seenParams := int64(0)
	recordIndex := 0
	readHead := func(name string, dst *projectionHead) error {
		*dst = projectionHead{activation: manifest.Config.Activation, layerNorm: manifest.Config.LayerNorm, residual: manifest.Config.Residual}
		var err error
		if dst.in, err = readLinear(r, &manifest, &recordIndex, name+".inp", manifest.Config.EncoderDim, manifest.Config.Width); err != nil {
			return err
		}
		seenParams += int64(dst.in.in*dst.in.out + len(dst.in.bias))
		dst.hidden = make([]block, manifest.Config.Depth-2)
		for i := range dst.hidden {
			p := fmt.Sprintf("%s.hidden.%d", name, i)
			if dst.hidden[i].linear, err = readLinear(r, &manifest, &recordIndex, p, manifest.Config.Width, manifest.Config.Width); err != nil {
				return err
			}
			seenParams += int64(dst.hidden[i].linear.in*dst.hidden[i].linear.out + len(dst.hidden[i].linear.bias))
			if manifest.Config.LayerNorm {
				if dst.hidden[i].normW, err = readTensor(r, &manifest, &recordIndex, p+".norm.weight", []int{manifest.Config.Width}); err != nil {
					return err
				}
				if dst.hidden[i].normB, err = readTensor(r, &manifest, &recordIndex, p+".norm.bias", []int{manifest.Config.Width}); err != nil {
					return err
				}
				seenParams += int64(2 * manifest.Config.Width)
			}
		}
		if dst.out, err = readLinear(r, &manifest, &recordIndex, name+".out", manifest.Config.Width, manifest.Config.ProjectionDim); err != nil {
			return err
		}
		seenParams += int64(dst.out.in*dst.out.out + len(dst.out.bias))
		return nil
	}
	if err := readHead("state_head", &state); err != nil {
		return nil, err
	}
	if err := readHead("action_head", &action); err != nil {
		return nil, err
	}
	if recordIndex != len(manifest.Tensors) {
		return nil, fmt.Errorf("clm: manifest has %d unexpected trailing tensors", len(manifest.Tensors)-recordIndex)
	}
	if seenParams > maxParameters {
		return nil, fmt.Errorf("clm: model parameter count %d exceeds limit %d", seenParams, maxParameters)
	}
	var trailing [1]byte
	if n, err := io.ReadFull(r, trailing[:]); n != 0 {
		return nil, fmt.Errorf("clm: trailing bytes after weight bundle")
	} else if !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("clm: check bundle end: %w", err)
	}
	return &HeadPair{config: manifest.Config, scale: manifest.Scale, state: state, action: action, provenance: manifest.Provenance}, nil
}

func validateManifest(m bundleManifest) error {
	if m.Format != "gophonic-clm" || m.Version != int(bundleVersion) {
		return fmt.Errorf("clm: unsupported bundle format %q version %d", m.Format, m.Version)
	}
	c := m.Config
	if c.EncoderDim <= 0 || c.EncoderDim > maxWidth || c.Width <= 0 || c.Width > maxWidth || c.Depth < 2 || c.Depth > maxDepth || c.ProjectionDim <= 0 || c.ProjectionDim > maxProjection {
		return fmt.Errorf("clm: %w: invalid dimensions: encoder=%d width=%d depth=%d projection=%d", ErrInvalidModel, c.EncoderDim, c.Width, c.Depth, c.ProjectionDim)
	}
	if c.Activation != "gelu" && c.Activation != "relu" && c.Activation != "silu" {
		return fmt.Errorf("clm: %w: unsupported activation %q", ErrInvalidModel, c.Activation)
	}
	if !finite32(m.LogitScale) || !finite32(m.Scale) || m.Scale < 0 || m.Scale > 100 {
		return fmt.Errorf("clm: %w: invalid scale %g", ErrInvalidModel, m.Scale)
	}
	expectedScale := float32(math.Exp(float64(m.LogitScale)))
	if expectedScale > 100 {
		expectedScale = 100
	}
	if math.Abs(float64(expectedScale-m.Scale)) > math.Max(1e-6, float64(expectedScale)*1e-6) {
		return fmt.Errorf("clm: %w: logit_scale %g yields scale %g, manifest says %g", ErrInvalidModel, m.LogitScale, expectedScale, m.Scale)
	}
	if m.Provenance.SourceSHA256 != "" {
		if len(m.Provenance.SourceSHA256) != sha256.Size*2 {
			return fmt.Errorf("clm: %w: invalid source SHA-256", ErrInvalidModel)
		}
		if _, err := hex.DecodeString(m.Provenance.SourceSHA256); err != nil {
			return fmt.Errorf("clm: %w: invalid source SHA-256: %v", ErrInvalidModel, err)
		}
	}
	expected := tensorSpecs(c)
	if len(m.Tensors) != len(expected) {
		return fmt.Errorf("clm: %w: tensor count %d, want %d", ErrInvalidModel, len(m.Tensors), len(expected))
	}
	var total int64
	for i, spec := range expected {
		got := m.Tensors[i]
		if got.Name != spec.name || !sameShape(got.Shape, spec.shape) {
			return fmt.Errorf("clm: %w: tensor %d is %q %v, want %q %v", ErrInvalidModel, i, got.Name, got.Shape, spec.name, spec.shape)
		}
		if len(got.SHA256) != sha256.Size*2 {
			return fmt.Errorf("clm: %w: tensor %q has invalid SHA-256", ErrInvalidModel, got.Name)
		}
		if _, err := hex.DecodeString(got.SHA256); err != nil {
			return fmt.Errorf("clm: %w: tensor %q has invalid SHA-256: %v", ErrInvalidModel, got.Name, err)
		}
		var count int64 = 1
		for _, dim := range spec.shape {
			count *= int64(dim)
		}
		total += count
		if total > maxParameters {
			return fmt.Errorf("clm: model parameter count exceeds limit %d", maxParameters)
		}
	}
	return nil
}

type tensorSpec struct {
	name  string
	shape []int
}

func tensorSpecs(c Config) []tensorSpec {
	out := make([]tensorSpec, 0, 2*(2*c.Depth+2))
	appendHead := func(name string) {
		out = append(out,
			tensorSpec{name + ".inp.weight", []int{c.Width, c.EncoderDim}},
			tensorSpec{name + ".inp.bias", []int{c.Width}},
		)
		for i := 0; i < c.Depth-2; i++ {
			p := fmt.Sprintf("%s.hidden.%d", name, i)
			out = append(out, tensorSpec{p + ".weight", []int{c.Width, c.Width}}, tensorSpec{p + ".bias", []int{c.Width}})
			if c.LayerNorm {
				out = append(out, tensorSpec{p + ".norm.weight", []int{c.Width}}, tensorSpec{p + ".norm.bias", []int{c.Width}})
			}
		}
		out = append(out, tensorSpec{name + ".out.weight", []int{c.ProjectionDim, c.Width}}, tensorSpec{name + ".out.bias", []int{c.ProjectionDim}})
	}
	appendHead("state_head")
	appendHead("action_head")
	return out
}

func readLinear(r io.Reader, m *bundleManifest, index *int, prefix string, in, out int) (linear, error) {
	w, err := readTensor(r, m, index, prefix+".weight", []int{out, in})
	if err != nil {
		return linear{}, err
	}
	b, err := readTensor(r, m, index, prefix+".bias", []int{out})
	if err != nil {
		return linear{}, err
	}
	packed, err := whispergemm.NewPackedB(in, out)
	if err != nil {
		return linear{}, fmt.Errorf("clm: %s: %w", prefix, err)
	}
	if err := packed.Pack(w, in, true); err != nil {
		return linear{}, fmt.Errorf("clm: %s: %w", prefix, err)
	}
	return linear{packed: packed, bias: b, in: in, out: out}, nil
}

func readTensor(r io.Reader, m *bundleManifest, index *int, name string, shape []int) ([]float32, error) {
	if *index >= len(m.Tensors) {
		return nil, fmt.Errorf("clm: missing tensor %q", name)
	}
	record := m.Tensors[*index]
	if record.Name != name || !sameShape(record.Shape, shape) {
		return nil, fmt.Errorf("clm: tensor %d is %q %v, want %q %v", *index, record.Name, record.Shape, name, shape)
	}
	(*index)++
	count := 1
	for _, dim := range shape {
		count *= dim
	}
	values := make([]float32, count)
	const chunkFloats = 1 << 16
	buf := make([]byte, chunkFloats*4)
	hasher := sha256.New()
	for start := 0; start < count; {
		n := min(chunkFloats, count-start)
		bytes := buf[:n*4]
		if _, err := io.ReadFull(r, bytes); err != nil {
			return nil, fmt.Errorf("clm: read tensor %q: %w", name, err)
		}
		_, _ = hasher.Write(bytes)
		for i := 0; i < n; i++ {
			values[start+i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
		}
		start += n
	}
	if !strings.EqualFold(hex.EncodeToString(hasher.Sum(nil)), record.SHA256) {
		return nil, fmt.Errorf("clm: tensor %q SHA-256 mismatch", name)
	}
	for i, v := range values {
		if !finite32(v) {
			return nil, fmt.Errorf("clm: tensor %q has non-finite value at %d", name, i)
		}
	}
	return values, nil
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func finite32(x float32) bool   { return !float32NaN(x) && !math.IsInf(float64(x), 0) }
func float32NaN(x float32) bool { return x != x }

// Config returns an immutable copy of the checkpoint architecture.
func (h *HeadPair) Config() Config {
	if h == nil {
		return Config{}
	}
	return h.config
}

// Scale returns exp(logit_scale), capped at 100 as in the CLM reference.
func (h *HeadPair) Scale() float32 {
	if h == nil {
		return 0
	}
	return h.scale
}

// Provenance returns the source checkpoint metadata copied by the converter.
func (h *HeadPair) Provenance() Provenance {
	if h == nil {
		return Provenance{}
	}
	return h.provenance
}

// NewWorkspace allocates reusable scratch for one caller. A workspace must not
// be used by concurrent calls; the immutable HeadPair can be shared freely.
// Batch buffers grow to the largest candidate count seen; warmed calls up to
// that count allocate nothing.
func (h *HeadPair) NewWorkspace() *Workspace {
	if h == nil {
		return nil
	}
	ws := &Workspace{head: h, stateProjection: make([]float32, h.config.ProjectionDim),
		gemm: make([]float32, whispergemm.ScratchLen(max(h.config.EncoderDim, h.config.Width)))}
	ws.grow(1)
	return ws
}

// Workspace is caller-owned scratch for allocation-free head projection and
// scoring after construction. It is tied to the HeadPair that created it.
type Workspace struct {
	head                    *HeadPair
	rows                    int
	input, hiddenA, hiddenB []float32 // [rows][dim]
	projected               []float32 // [rows][ProjectionDim]
	gemm                    []float32 // GEMM scratch, owned so no call allocates
	stateProjection         []float32
	one                     [1][]float32
}

func (ws *Workspace) grow(rows int) {
	if rows <= ws.rows {
		return
	}
	c := ws.head.config
	ws.rows = rows
	ws.input = make([]float32, rows*c.EncoderDim)
	ws.hiddenA = make([]float32, rows*c.Width)
	ws.hiddenB = make([]float32, rows*c.Width)
	ws.projected = make([]float32, rows*c.ProjectionDim)
}

// ProjectStateInto applies the state MLP to one raw encoder embedding, L2
// normalizes the embedding before the MLP (matching CLM's embedder), then L2
// normalizes the projected result. dst must have ProjectionDim elements.
func (h *HeadPair) ProjectStateInto(embedding, dst []float32, ws *Workspace) error {
	if err := h.checkWorkspace(embedding, dst, ws); err != nil {
		return err
	}
	return h.projectOne(&h.state, embedding, dst, ws)
}

// ProjectActionInto is the action-head counterpart to ProjectStateInto.
func (h *HeadPair) ProjectActionInto(embedding, dst []float32, ws *Workspace) error {
	if err := h.checkWorkspace(embedding, dst, ws); err != nil {
		return err
	}
	return h.projectOne(&h.action, embedding, dst, ws)
}

// ScoreInto computes scale*cosine(state, action)/temperature for every raw
// candidate embedding. The caller provides scores and a reusable Workspace.
func (h *HeadPair) ScoreInto(stateEmbedding []float32, actionEmbeddings [][]float32, temperature float32, scores []float32, ws *Workspace) error {
	if h == nil {
		return fmt.Errorf("clm: nil head pair")
	}
	if ws == nil || ws.head != h {
		return ErrWorkspace
	}
	if len(stateEmbedding) != h.config.EncoderDim {
		return fmt.Errorf("clm: state embedding has %d values, want %d", len(stateEmbedding), h.config.EncoderDim)
	}
	if len(scores) < len(actionEmbeddings) {
		return fmt.Errorf("clm: scores buffer has %d values, want %d", len(scores), len(actionEmbeddings))
	}
	if !finite32(temperature) || temperature <= 0 || temperature > 100 {
		return fmt.Errorf("clm: temperature must be in (0,100], got %g", temperature)
	}
	if err := h.ProjectStateInto(stateEmbedding, ws.stateProjection, ws); err != nil {
		return err
	}
	for i, a := range actionEmbeddings {
		if len(a) != h.config.EncoderDim {
			return fmt.Errorf("clm: action embedding %d has %d values, want %d", i, len(a), h.config.EncoderDim)
		}
	}
	// All candidates pass through each head layer together, so every packed
	// weight matrix is read once per call rather than once per candidate.
	const chunk = 64
	p := h.config.ProjectionDim
	for start := 0; start < len(actionEmbeddings); start += chunk {
		batch := actionEmbeddings[start:min(start+chunk, len(actionEmbeddings))]
		if err := h.projectRows(&h.action, batch, ws); err != nil {
			return fmt.Errorf("clm: action embeddings %d-%d: %w", start, start+len(batch)-1, err)
		}
		for i := range batch {
			cos := dot32(ws.stateProjection, ws.projected[i*p:(i+1)*p])
			scores[start+i] = (h.scale * cos) / temperature
		}
	}
	return nil
}

func (h *HeadPair) checkWorkspace(input, dst []float32, ws *Workspace) error {
	if h == nil {
		return fmt.Errorf("clm: nil head pair")
	}
	if ws == nil || ws.head != h {
		return ErrWorkspace
	}
	if len(input) != h.config.EncoderDim {
		return fmt.Errorf("clm: embedding has %d values, want %d", len(input), h.config.EncoderDim)
	}
	if len(dst) != h.config.ProjectionDim {
		return fmt.Errorf("clm: projection buffer has %d values, want %d", len(dst), h.config.ProjectionDim)
	}
	return nil
}

func (h *HeadPair) projectOne(head *projectionHead, embedding, dst []float32, ws *Workspace) error {
	ws.one[0] = embedding
	err := h.projectRows(head, ws.one[:], ws)
	ws.one[0] = nil
	if err == nil {
		copy(dst, ws.projected[:h.config.ProjectionDim])
	}
	return err
}

// projectRows applies head to every embedding and leaves the L2-normalized
// projections in ws.projected, one ProjectionDim row per embedding.
func (h *HeadPair) projectRows(head *projectionHead, embeddings [][]float32, ws *Workspace) error {
	n := len(embeddings)
	ws.grow(n)
	c := h.config
	for i, e := range embeddings {
		if err := normalizeEncoderInto(e, ws.input[i*c.EncoderDim:(i+1)*c.EncoderDim]); err != nil {
			return err
		}
	}
	linearRows(&head.in, ws.input, ws.hiddenA, n, ws.gemm)
	for i := range n {
		activateInPlace(head.activation, ws.hiddenA[i*c.Width:(i+1)*c.Width])
	}
	cur, next := ws.hiddenA, ws.hiddenB
	for l := range head.hidden {
		layer := &head.hidden[l]
		linearRows(&layer.linear, cur, next, n, ws.gemm)
		for i := range n {
			row := next[i*c.Width : (i+1)*c.Width]
			if head.layerNorm {
				layerNormInPlace(row, layer.normW, layer.normB)
			}
			activateInPlace(head.activation, row)
			if head.residual {
				for j, v := range cur[i*c.Width : (i+1)*c.Width] {
					row[j] += v
				}
			}
		}
		cur, next = next, cur
	}
	linearRows(&head.out, cur, ws.projected, n, ws.gemm)
	for i := range n {
		if err := normalizeProjectedInPlace(ws.projected[i*c.ProjectionDim : (i+1)*c.ProjectionDim]); err != nil {
			return err
		}
	}
	return nil
}

// linearRows computes output[r] = input[r] · Wᵀ + bias for rows packed at
// the layer's input and output widths.
func linearRows(l *linear, input, output []float32, rows int, scratch []float32) {
	if err := l.packed.MulScratch(output, l.out, input, l.in, rows, scratch); err != nil {
		panic("clm: head GEMM: " + err.Error())
	}
	for r := range rows {
		out := output[r*l.out : (r+1)*l.out]
		for i, b := range l.bias {
			out[i] += b
		}
	}
}

func activateInPlace(kind string, values []float32) {
	switch kind {
	case "relu":
		for i, x := range values {
			if x < 0 {
				values[i] = 0
			}
		}
	case "silu":
		for i, x := range values {
			values[i] = x / (1 + float32(math.Exp(-float64(x))))
		}
	default: // PyTorch nn.GELU's exact erf form (approximate="none").
		const invSqrt2 = 0.7071067811865475244
		for i, x := range values {
			values[i] = 0.5 * x * (1 + float32(math.Erf(float64(x)*invSqrt2)))
		}
	}
}

func layerNormInPlace(values, weight, bias []float32) {
	var mean float64
	for _, x := range values {
		mean += float64(x)
	}
	mean /= float64(len(values))
	var variance float64
	for _, x := range values {
		d := float64(x) - mean
		variance += d * d
	}
	variance /= float64(len(values))
	inv := float32(1 / math.Sqrt(variance+1e-5))
	mu := float32(mean)
	for i, x := range values {
		values[i] = (x-mu)*inv*weight[i] + bias[i]
	}
}

func normalizeEncoderInto(input, output []float32) error {
	var sum float64
	for i, x := range input {
		if !finite32(x) {
			return fmt.Errorf("clm: non-finite encoder embedding at %d", i)
		}
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum) + 1e-12 // clm.embedder.l2 uses norm + epsilon.
	inv := float32(1 / norm)
	for i, x := range input {
		output[i] = x * inv
	}
	return nil
}

func normalizeProjectedInPlace(values []float32) error {
	var sum float64
	for _, x := range values {
		if !finite32(x) {
			return fmt.Errorf("clm: projection produced non-finite value")
		}
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if norm < 1e-12 {
		norm = 1e-12
	}
	inv := float32(1 / norm)
	for i := range values {
		values[i] *= inv
	}
	return nil
}

func dot32(a, b []float32) float32 {
	var sum float32
	for i, x := range a {
		sum += x * b[i]
	}
	return sum
}

// writeBundle is shared by tests and conversion-facing tools. Tensor payloads
// are serialized in the order returned by tensorSpecs and each tensor's
// manifest SHA-256 is checked while loading.
func writeBundle(w io.Writer, m bundleManifest, tensors map[string][]float32) error {
	if err := validateManifest(m); err != nil {
		return err
	}
	manifestBytes, err := vibejson.Marshal(&m)
	if err != nil {
		return err
	}
	if len(manifestBytes) > 1<<20 {
		return fmt.Errorf("clm: manifest too large")
	}
	if _, err = w.Write([]byte(bundleMagic)); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, bundleVersion); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, uint32(len(manifestBytes))); err != nil {
		return err
	}
	if _, err = w.Write(manifestBytes); err != nil {
		return err
	}
	for _, spec := range m.Tensors {
		values, ok := tensors[spec.Name]
		if !ok {
			return fmt.Errorf("clm: missing tensor data %q", spec.Name)
		}
		buf := make([]byte, len(values)*4)
		for i, x := range values {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
		}
		_, err = w.Write(buf)
		if err != nil {
			return err
		}
	}
	return nil
}
