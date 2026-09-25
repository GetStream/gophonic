// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

// dense is an FP32 linear layer, y = W·x + b, packed for the SME kernel in
// chunks of output columns, so that a product over a few rows (whose time
// is the weights' memory traffic) can spread over workers a chunk each.
type dense struct {
	parts   []*whispergemm.PackedB // cols columns each; the last may be narrower
	b       []float32              // nil without a bias
	in, out int
	cols    int
}

// Column chunks: up to denseChunks of at least minChunk columns, in whole
// kernel tiles.
const (
	denseChunks = 8
	minChunk    = 128
)

// loadDense reads name.weight, [out][in] (a one-wide convolution kernel
// also fits), and name.bias when present.
func loadDense(st *safetensors.Checkpoint, name string, out, in int, shape ...int) (dense, error) {
	if shape == nil {
		shape = []int{out, in}
	}
	w, err := st.Float32(name+".weight", shape...)
	if err != nil {
		return dense{}, err
	}
	d, err := packDense(w, in, out)
	if err != nil {
		return dense{}, err
	}
	if st.Has(name + ".bias") {
		if d.b, err = st.Float32(name+".bias", out); err != nil {
			return dense{}, err
		}
	}
	return d, nil
}

// packDense packs the [out][in] weights w.
func packDense(w []float32, in, out int) (dense, error) {
	d := dense{in: in, out: out, cols: chunkWidth(out)}
	for c0 := 0; c0 < out; c0 += d.cols {
		width := min(d.cols, out-c0)
		p, err := whispergemm.NewPackedB(in, width)
		if err != nil {
			return dense{}, err
		}
		if err := p.Pack(w[c0*in:(c0+width)*in], in, true); err != nil {
			return dense{}, err
		}
		d.parts = append(d.parts, p)
	}
	return d, nil
}

func chunkWidth(out int) int {
	w := (out + denseChunks - 1) / denseChunks
	w = max(minChunk, (w+31)&^31)
	return min(w, out)
}

// apply writes rows × out outputs of rows × in inputs.
func (d *dense) apply(exec *whispergemm.Executor, dst, src []float32, rows int) error {
	if len(dst) < rows*d.out || len(src) < rows*d.in {
		return fmt.Errorf("qwen3tts: dense %dx%d on %d rows", d.out, d.in, rows)
	}
	for c, p := range d.parts {
		if err := exec.Mul(p, dst[c*d.cols:], d.out, src, d.in, rows); err != nil {
			return err
		}
	}
	d.bias(dst, rows)
	return nil
}

// mul writes rows × out products (without the bias) of rows of src,
// aStride apart (overlapping when narrower than in), on the calling
// goroutine with the given scratch.
func (d *dense) mul(dst, src []float32, aStride, rows int, scratch []float32) {
	for c, p := range d.parts {
		p.MulScratch(dst[c*d.cols:], d.out, src, aStride, rows, scratch)
	}
}

// bias adds the bias to rows of dst.
func (d *dense) bias(dst []float32, rows int) {
	if d.b == nil {
		return
	}
	for r := range rows {
		addTo(dst[r*d.out:(r+1)*d.out], d.b)
	}
}

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }
