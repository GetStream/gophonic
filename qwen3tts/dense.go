// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

// dense is an FP32 linear layer, y = W·x + b, packed for the SME kernel.
type dense struct {
	w       *whispergemm.PackedB
	b       []float32 // nil without a bias
	in, out int
}

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
	d := dense{in: in, out: out}
	if d.w, err = whispergemm.NewPackedB(in, out); err != nil {
		return dense{}, err
	}
	if err := d.w.Pack(w, in, true); err != nil {
		return dense{}, err
	}
	if st.Has(name + ".bias") {
		if d.b, err = st.Float32(name+".bias", out); err != nil {
			return dense{}, err
		}
	}
	return d, nil
}

// apply writes rows × out outputs of rows × in inputs.
func (d *dense) apply(exec *whispergemm.Executor, dst, src []float32, rows int) error {
	if len(dst) < rows*d.out || len(src) < rows*d.in {
		return fmt.Errorf("qwen3tts: dense %dx%d on %d rows", d.out, d.in, rows)
	}
	if err := exec.Mul(d.w, dst, d.out, src, d.in, rows); err != nil {
		return err
	}
	if d.b != nil {
		for r := range rows {
			addTo(dst[r*d.out:(r+1)*d.out], d.b)
		}
	}
	return nil
}

func addTo(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += src[i]
	}
}

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }
