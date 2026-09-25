// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

// PackedLen reports the float32 storage required for a K-by-N packed matrix.
func PackedLen(k, n int) (int, error) {
	const maxInt = int(^uint(0) >> 1)
	if k < 0 || n < 0 || n > maxInt-(panelColumns-1) {
		return 0, ErrShape
	}
	columns := (n + panelColumns - 1) / panelColumns * panelColumns
	if columns != 0 && k > (maxInt/4)/columns {
		return 0, ErrShape
	}
	return k * columns, nil
}

// NewPackedBBuffer borrows zeroed caller-owned storage. Its owner must remain
// alive until every Pack/Mul call finishes. Reshape cannot exceed the original
// capacity; it returns ErrShape instead of allocating a replacement.
func NewPackedBBuffer(k, n int, storage []float32) (*PackedB, error) {
	size, err := PackedLen(k, n)
	if err != nil || len(storage) < size {
		return nil, ErrShape
	}
	return &PackedB{k: k, n: n, data: storage[:size:size], borrowed: true}, nil
}

// NewPackedBRows reserves a fixed-pitch value page with initially zero rows.
// Appending or truncating rows changes only logical K, never the location of
// existing values. Multiplication reads just the initialized row prefix.
func NewPackedBRows(capacity, n int, storage []float32) (*PackedB, error) {
	b, err := NewPackedBBuffer(capacity, n, storage)
	if err != nil {
		return nil, err
	}
	b.pitch, b.fixedRows, b.k = capacity, true, 0
	return b, nil
}

// CopyColumnsFrom keeps the first columns of src, preserving K and the exact
// packed values. The destination must have sufficient capacity. src may be b;
// other overlapping storage is not supported. Padding is reset after a cut.
func (b *PackedB) CopyColumnsFrom(src *PackedB, columns int) error {
	if b == nil || src == nil {
		return ErrNilMatrix
	}
	size, err := PackedLen(src.k, columns)
	if err != nil || b.fixedRows || src.fixedRows || columns > src.n || src.k != b.k || size > cap(b.data) {
		return ErrShape
	}
	b.data = b.data[:size]
	copy(b.data, src.data[:size])
	b.n = columns
	return b.PackColumns(nil, b.k, columns)
}

// CopyRowsFrom keeps the first rows of src, preserving N and every stored
// value. It repacks the panel strides without arithmetic or allocation.
// src may be b; other overlapping storage is not supported.
func (b *PackedB) CopyRowsFrom(src *PackedB, rows int) error {
	if b == nil || src == nil {
		return ErrNilMatrix
	}
	if rows < 0 || rows > src.k || b.n != src.n {
		return ErrShape
	}
	if b == src {
		return b.resizeRows(rows)
	}
	size, err := PackedLen(rows, b.n)
	if err != nil || size > cap(b.data) {
		return ErrShape
	}
	if b.fixedRows && rows > b.pitch {
		return ErrShape
	}
	if !b.fixedRows {
		b.data = b.data[:size]
	}
	b.k = rows
	for panel := 0; panel < (b.n+panelColumns-1)/panelColumns; panel++ {
		to, from := panel*b.panelRows()*panelColumns, panel*src.panelRows()*panelColumns
		copy(b.data[to:to+rows*panelColumns], src.data[from:from+rows*panelColumns])
	}
	return nil
}

// PackRows writes rows of a row-major matrix at row start and discards the
// previous suffix. The prefix [0,start) is preserved, including when changing
// K requires moving overlapping panels. Storage capacity is fixed here.
func (b *PackedB) PackRows(src []float32, stride, start, rows int) error {
	if b == nil {
		return ErrNilMatrix
	}
	if start < 0 || start > b.k || rows < 0 || start > int(^uint(0)>>1)-rows || !validMatrix(src, rows, b.n, stride) {
		return ErrShape
	}
	if err := b.resizeRows(start + rows); err != nil {
		return err
	}
	for col := 0; col < b.n; col += panelColumns {
		width := min(panelColumns, b.n-col)
		for r := 0; r < rows; r++ {
			off := col*b.panelRows() + (start+r)*panelColumns
			dst := b.data[off : off+panelColumns]
			copy(dst, src[r*stride+col:r*stride+col+width])
			clear(dst[width:])
		}
	}
	return nil
}

// resizeRows changes the stride between panels while preserving their common
// row prefix. Growing moves panels backwards; shrinking moves them forwards.
func (b *PackedB) resizeRows(rows int) error {
	if b.fixedRows {
		if rows < 0 || rows > b.pitch {
			return ErrShape
		}
		b.k = rows
		return nil
	}
	size, err := PackedLen(rows, b.n)
	if err != nil || size > cap(b.data) {
		return ErrShape
	}
	if rows == b.k {
		return nil
	}
	old := b.k
	common := min(old, rows) * panelColumns
	data := b.data[:max(len(b.data), size)]
	panels := (b.n + panelColumns - 1) / panelColumns
	for i := 0; i < panels; i++ {
		panel := i
		if rows > old {
			panel = panels - 1 - i
		}
		copy(data[panel*rows*panelColumns:panel*rows*panelColumns+common],
			data[panel*old*panelColumns:panel*old*panelColumns+common])
	}
	b.k, b.data = rows, data[:size]
	return nil
}
