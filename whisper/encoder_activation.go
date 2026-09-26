// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import "github.com/GetStream/gophonic/internal/nn"

// encoderActivation is stored in the workspace so submitting an activation
// does not allocate a closure or a boxed slice. Each worker owns whole rows.
type encoderActivation struct {
	values, bias []float32
	width        int
}

func (a *encoderActivation) ApplyRows(start, end int) {
	nn.BiasGELU(a.values[start*a.width:end*a.width], a.bias, end-start, a.width)
}

func (w *encoderWorkspace) activate(values, bias []float32, rows, width int) error {
	w.activation = encoderActivation{values: values, bias: bias, width: width}
	err := w.gemm.Rows(&w.activation, rows, 32)
	w.activation = encoderActivation{}
	return err
}
