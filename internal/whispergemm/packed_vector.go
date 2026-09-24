// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import "math"

const (
	vectorGroup      = 64 // outputs per ZA vector group
	vectorChunkGroup = 8  // vector groups accumulated together
)

// PackedVector stores an N-by-K matrix for repeated matrix-vector products
// on hardware with a streaming matrix unit. Weights are kept as FP16 when
// every value converts to FP16 and back to the identical FP32 bits, which
// halves the memory traffic of bandwidth-bound decoder projections without
// changing any arithmetic: each weight widens exactly before an FP32 fused
// multiply-add. Otherwise the FP32 values are stored.
//
// Each output accumulates in increasing K order from +0 with fused FP32
// multiply-add, matching PackedB.Mul on the SME path for one row.
type PackedVector struct {
	k, n, kp int
	w16      []uint16
	w32      []float32
}

// PackedVectorAccelerated reports whether PackedVector.Mul runs on the
// streaming matrix unit. Callers can keep their existing MulVector path on
// other machines, where the packed fallback is not tuned.
func PackedVectorAccelerated() bool { return smeEnabled }

// NewPackedVector packs rows N-by-K weights whose rows start stride elements
// apart. It allocates the packed copy; Mul then allocates nothing.
func NewPackedVector(weights []float32, stride, rows, k int) (*PackedVector, error) {
	p := &PackedVector{}
	if err := p.pack(weights, stride, rows, k, true); err != nil {
		return nil, err
	}
	return p, nil
}

// NewPackedVectorFP32 allocates FP32 storage for dynamic data such as
// attention caches; Repack refills it without allocating.
func NewPackedVectorFP32(rows, k int) (*PackedVector, error) {
	if rows < 0 || k < 0 {
		return nil, ErrShape
	}
	p := &PackedVector{k: k, n: rows, kp: (k + 3) &^ 3}
	p.w32 = make([]float32, p.size())
	return p, nil
}

// Dims returns K and N.
func (p *PackedVector) Dims() (k, n int) { return p.k, p.n }

// Half reports whether the weights are stored as exact FP16.
func (p *PackedVector) Half() bool { return p.w16 != nil }

func (p *PackedVector) size() int {
	groups := (p.n + vectorGroup - 1) / vectorGroup
	return groups * vectorGroup * p.kp
}

// Repack replaces FP32 contents from rows-by-K data with the given stride.
// The dimensions must match NewPackedVectorFP32.
func (p *PackedVector) Repack(weights []float32, stride int) error {
	return p.RepackShape(weights, stride, p.n, p.k)
}

// RepackShape replaces FP32 contents with a rows-by-K matrix that may be
// smaller than the allocated shape, such as attention caches for a shorter
// audio sequence. It allocates nothing.
func (p *PackedVector) RepackShape(weights []float32, stride, rows, k int) error {
	if p.w32 == nil || rows < 0 || k < 0 || !validMatrix(weights, rows, k, stride) {
		return ErrShape
	}
	kp := (k + 3) &^ 3
	groups := (rows + vectorGroup - 1) / vectorGroup
	if groups*vectorGroup*kp > cap(p.w32) {
		return ErrShape
	}
	p.n, p.k, p.kp = rows, k, kp
	p.w32 = p.w32[:cap(p.w32)]
	p.fill32(weights, stride)
	return nil
}

// RepackColumns fills FP32 storage with a rows-by-K matrix whose element
// (row, col) is src[col*stride+row]*scale + bias[row]. Consecutive rows are
// adjacent in src, so each packed lane group is a contiguous copy. A scale of
// one and a nil bias leave values unchanged. It allocates nothing.
func (p *PackedVector) RepackColumns(src []float32, stride, rows, k int, scale float32, bias []float32) error {
	if p.w32 == nil || rows < 0 || k < 0 || !validMatrix(src, k, rows, stride) || (bias != nil && len(bias) < rows) {
		return ErrShape
	}
	kp := (k + 3) &^ 3
	groups := (rows + vectorGroup - 1) / vectorGroup
	if groups*vectorGroup*kp > cap(p.w32) {
		return ErrShape
	}
	p.n, p.k, p.kp = rows, k, kp
	p.w32 = p.w32[:cap(p.w32)]
	for first := 0; first < groups; first += vectorChunkGroup {
		g := min(vectorChunkGroup, groups-first)
		base := first * vectorGroup * kp
		for c := 0; c < kp; c++ {
			for gi := 0; gi < g; gi++ {
				dst := p.w32[base+(c*g+gi)*vectorGroup : base+(c*g+gi+1)*vectorGroup]
				row0 := (first + gi) * vectorGroup
				width := min(vectorGroup, rows-row0)
				if c >= k {
					clear(dst)
					continue
				}
				in := src[c*stride+row0 : c*stride+row0+width]
				switch {
				case bias != nil:
					b := bias[row0 : row0+width]
					for i, v := range in {
						if scale != 1 {
							v *= scale
						}
						dst[i] = v + b[i]
					}
				case scale != 1:
					for i, v := range in {
						dst[i] = v * scale
					}
				default:
					copy(dst, in)
				}
				clear(dst[width:])
			}
		}
	}
	return nil
}

func (p *PackedVector) pack(weights []float32, stride, rows, k int, allowHalf bool) error {
	if rows < 0 || k < 0 || !validMatrix(weights, rows, k, stride) {
		return ErrShape
	}
	p.k, p.n, p.kp = k, rows, (k+3)&^3
	p.w16, p.w32 = nil, nil
	if allowHalf && exactHalfMatrix(weights, stride, rows, k) {
		p.w16 = make([]uint16, p.size())
		p.forEach(func(index, row, col int) {
			if row < p.n && col < p.k {
				h, _ := halfFromFloat32(weights[row*stride+col])
				p.w16[halfLane(index)] = h
			}
		})
		return nil
	}
	p.w32 = make([]float32, p.size())
	p.fill32(weights, stride)
	return nil
}

func (p *PackedVector) fill32(weights []float32, stride int) {
	groups := (p.n + vectorGroup - 1) / vectorGroup
	for first := 0; first < groups; first += vectorChunkGroup {
		g := min(vectorChunkGroup, groups-first)
		base := first * vectorGroup * p.kp
		for gi := 0; gi < g; gi++ {
			row0 := (first + gi) * vectorGroup
			width := min(vectorGroup, p.n-row0)
			for c := 0; c < p.kp; c++ {
				dst := p.w32[base+(c*g+gi)*vectorGroup : base+(c*g+gi+1)*vectorGroup]
				if c >= p.k {
					clear(dst)
					continue
				}
				src := weights[row0*stride+c:]
				for lane := 0; lane < width; lane++ {
					dst[lane] = src[lane*stride]
				}
				clear(dst[width:])
			}
		}
	}
}

// forEach visits every packed slot in chunk-major order with its source
// coordinates; padding slots report rows or columns outside the matrix.
func (p *PackedVector) forEach(visit func(index, row, col int)) {
	groups := (p.n + vectorGroup - 1) / vectorGroup
	for first := 0; first < groups; first += vectorChunkGroup {
		g := min(vectorChunkGroup, groups-first)
		base := first * vectorGroup * p.kp
		for c := 0; c < p.kp; c++ {
			for gi := 0; gi < g; gi++ {
				for lane := 0; lane < vectorGroup; lane++ {
					visit(base+(c*g+gi)*vectorGroup+lane, (first+gi)*vectorGroup+lane, c)
				}
			}
		}
	}
}

// halfLane maps an output lane to its FP16 slot. FCVT widens even halves and
// FCVTLT odd halves, so outputs 0-15 and 16-31 interleave in the first
// vector and 32-47 and 48-63 in the second.
func halfLane(index int) int {
	lane := index % vectorGroup
	half := lane / 32
	within := lane % 32
	return index - lane + half*32 + 2*(within%16) + within/16
}

// Chunks returns the number of independently computable output chunks.
func (p *PackedVector) Chunks() int {
	groups := (p.n + vectorGroup - 1) / vectorGroup
	return (groups + vectorChunkGroup - 1) / vectorChunkGroup
}

// MulChunks computes outputs of chunks [first,last) only, writing the same
// values Mul would. Disjoint ranges can run concurrently on different
// goroutines, which spreads bandwidth-bound products over matrix units.
func (p *PackedVector) MulChunks(dst, x []float32, first, last int) error {
	if p == nil {
		return ErrNilMatrix
	}
	if len(x) < p.k || len(dst) < p.n || first < 0 || last > p.Chunks() || first > last {
		return ErrShape
	}
	start := first * vectorChunkGroup * vectorGroup
	end := min(p.n, last*vectorChunkGroup*vectorGroup)
	if start >= end {
		return nil
	}
	if p.k == 0 {
		clear(dst[start:end])
		return nil
	}
	// Every chunk before the last is full, so chunk c starts at a fixed offset.
	offset := start * p.kp
	sub := PackedVector{k: p.k, n: end - start, kp: p.kp}
	if p.w16 != nil {
		sub.w16 = p.w16[offset:]
	} else {
		sub.w32 = p.w32[offset:]
	}
	if mulPackedVectorSME(&sub, dst[start:end], x) {
		return nil
	}
	sub.mulGeneric(dst[start:end], x)
	return nil
}

// Mul writes dst[:N] = W * x[:K]. It allocates no memory and may run
// concurrently with other Mul calls on the same matrix.
func (p *PackedVector) Mul(dst, x []float32) error {
	if p == nil {
		return ErrNilMatrix
	}
	if len(x) < p.k || len(dst) < p.n {
		return ErrShape
	}
	if p.n == 0 {
		return nil
	}
	if p.k == 0 {
		clear(dst[:p.n])
		return nil
	}
	if mulPackedVectorSME(p, dst, x) {
		return nil
	}
	p.mulGeneric(dst, x)
	return nil
}

func (p *PackedVector) mulGeneric(dst, x []float32) {
	groups := (p.n + vectorGroup - 1) / vectorGroup
	for first := 0; first < groups; first += vectorChunkGroup {
		g := min(vectorChunkGroup, groups-first)
		base := first * vectorGroup * p.kp
		for gi := 0; gi < g; gi++ {
			for lane := 0; lane < vectorGroup; lane++ {
				row := (first+gi)*vectorGroup + lane
				if row >= p.n {
					break
				}
				var sum float32
				for c := 0; c < p.k; c++ {
					index := base + (c*g+gi)*vectorGroup + lane
					var w float32
					if p.w16 != nil {
						w = float32FromHalf(p.w16[halfLane(index)])
					} else {
						w = p.w32[index]
					}
					sum = float32(math.FMA(float64(x[c]), float64(w), float64(sum)))
				}
				dst[row] = sum
			}
		}
	}
}

func exactHalfMatrix(weights []float32, stride, rows, k int) bool {
	for r := 0; r < rows; r++ {
		for _, v := range weights[r*stride : r*stride+k] {
			if _, ok := halfFromFloat32(v); !ok {
				return false
			}
		}
	}
	return true
}

// halfFromFloat32 returns v as IEEE binary16 when the conversion is exact.
// NaNs report false so their payloads are never altered.
func halfFromFloat32(v float32) (uint16, bool) {
	bits := math.Float32bits(v)
	sign := uint16(bits>>16) & 0x8000
	exp := int(bits>>23) & 0xff
	mant := bits & 0x7fffff
	switch {
	case exp == 0xff:
		if mant != 0 {
			return 0, false
		}
		return sign | 0x7c00, true
	case exp == 0 && mant == 0:
		return sign, true
	case exp == 0:
		return 0, false // FP32 subnormals are far below FP16 range
	}
	e := exp - 127
	switch {
	case e >= -14 && e <= 15:
		if mant&0x1fff != 0 {
			return 0, false
		}
		return sign | uint16(e+15)<<10 | uint16(mant>>13), true
	case e >= -24 && e < -14:
		full := mant | 0x800000
		shift := uint(-e - 14 + 13)
		if full&(1<<shift-1) != 0 {
			return 0, false
		}
		return sign | uint16(full>>shift), true
	}
	return 0, false
}

func float32FromHalf(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		v := float32(mant) * (1.0 / (1 << 24))
		if sign != 0 {
			v = -v
		}
		return v
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}
