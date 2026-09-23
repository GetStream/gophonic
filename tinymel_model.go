// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	tinyBundleMagic       = "GOTMEL1\x00"
	tinyBundleVersion     = uint32(1)
	tinyBundleTensorCount = uint32(76)
	tinySourceSHA256      = "6b986a0440b30f533f0f7e473695939c347a2c95c5280eb2e090d0076b3dbbd1"
	tinyFrameCount        = 800
	tinyMelCount          = 80
	tinyStemChannels      = 192
	tinySequenceLength    = 100
	tinyGRUHidden         = 128
	tinyGRUDirections     = 2
)

const (
	tinyDTypeFloat32 uint8 = 1
	tinyDTypeUint8   uint8 = 2
	tinyDTypeInt64   uint8 = 3
	tinyDTypeInt32   uint8 = 4
)

var (
	errInvalidTinyBundle = errors.New("invalid TinyMelNet model bundle")
	errTinyModelClosed   = errors.New("TinyMelNet workspace is closed")
)

type tinyTensor struct {
	dtype uint8
	shape []uint32
	f32   []float32
	u8    []uint8
	i64   []int64
	i32   []int32
}

type tinyConv1D struct {
	weight          []uint8 // ONNX OIK order, with O=output, I=input/group, K=kernel
	packed          []uint8 // output, kernel, input/group; used by contiguous SIMD dots
	depthwisePacked []uint8 // kernel, channel; used by depthwise SIMD convolution
	weightScale     float32
	weightZero      uint8
	bias            []float32
	inChannels      int
	outChannels     int
	kernel          int
	stride          int
	groups          int
}

type tinySeparableBlock struct {
	depthwise tinyConv1D
	pointwise tinyConv1D
}

type tinyQuantizedLinear struct {
	weight      []uint8 // ONNX MatMul order [input,output]
	weightScale float32
	weightZero  uint8
	bias        []float32
	in          int
	out         int
}

// TinyMelModel is the optional TinyMelNet model from the MIT-licensed
// Hinglish turn-detector repository. It is kept separate from Model because
// it has a different graph, checkpoint, and completion threshold.
type TinyMelModel struct {
	stem   tinyConv1D
	blocks [3]tinySeparableBlock

	gruW, gruR, gruB []float32

	poolQuery      []uint8
	poolQueryScale float32
	poolQueryZero  uint8

	headNormW, headNormB []float32
	head1, head2         tinyQuantizedLinear
}

// LoadTinyMel loads the pinned TinyMelNet INT8 model bundle produced by
// tools/tinymel_to_gofloor.py.
func LoadTinyMel(path string) (*TinyMelModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadTinyMelWeights(f)
}

// ReadTinyMelWeights reads the pinned TinyMelNet bundle produced by the
// offline converter. The bundle stores the original ONNX initializer names,
// dtypes, and layouts so runtime operator math remains auditable.
func ReadTinyMelWeights(r io.Reader) (*TinyMelModel, error) {
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read TinyMelNet header: %w", err)
	}
	if string(magic[:]) != tinyBundleMagic {
		return nil, fmt.Errorf("%w: bad magic %q", errInvalidTinyBundle, string(magic[:]))
	}
	var version, count uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read TinyMelNet version: %w", err)
	}
	if version != tinyBundleVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", errInvalidTinyBundle, version)
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read TinyMelNet tensor count: %w", err)
	}
	if count != tinyBundleTensorCount {
		return nil, fmt.Errorf("%w: expected %d tensors, bundle contains %d", errInvalidTinyBundle, tinyBundleTensorCount, count)
	}
	var sourceHash [32]byte
	if _, err := io.ReadFull(r, sourceHash[:]); err != nil {
		return nil, fmt.Errorf("read TinyMelNet source hash: %w", err)
	}
	if fmt.Sprintf("%x", sourceHash) != tinySourceSHA256 {
		return nil, fmt.Errorf("%w: source ONNX SHA-256 does not match audited checkpoint", errInvalidTinyBundle)
	}

	tensors := make(map[string]tinyTensor, count)
	for ti := uint32(0); ti < count; ti++ {
		var nameLen uint16
		if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
			return nil, fmt.Errorf("read TinyMelNet tensor %d name length: %w", ti, err)
		}
		if nameLen == 0 || nameLen > 256 {
			return nil, fmt.Errorf("%w: tensor %d has invalid name length %d", errInvalidTinyBundle, ti, nameLen)
		}
		nameBytes := make([]byte, int(nameLen))
		if _, err := io.ReadFull(r, nameBytes); err != nil {
			return nil, fmt.Errorf("read TinyMelNet tensor %d name: %w", ti, err)
		}
		name := string(nameBytes)
		if _, exists := tensors[name]; exists {
			return nil, fmt.Errorf("%w: duplicate tensor %q", errInvalidTinyBundle, name)
		}
		var dtype, rank uint8
		if err := binary.Read(r, binary.LittleEndian, &dtype); err != nil {
			return nil, fmt.Errorf("read TinyMelNet tensor %q dtype: %w", name, err)
		}
		if err := binary.Read(r, binary.LittleEndian, &rank); err != nil {
			return nil, fmt.Errorf("read TinyMelNet tensor %q rank: %w", name, err)
		}
		if rank > 8 {
			return nil, fmt.Errorf("%w: tensor %q rank %d is too large", errInvalidTinyBundle, name, rank)
		}
		t := tinyTensor{dtype: dtype, shape: make([]uint32, int(rank))}
		length := uint64(1)
		for i := range t.shape {
			if err := binary.Read(r, binary.LittleEndian, &t.shape[i]); err != nil {
				return nil, fmt.Errorf("read TinyMelNet tensor %q dimension %d: %w", name, i, err)
			}
			dim := uint64(t.shape[i])
			if dim == 0 || length > 1<<28/dim {
				return nil, fmt.Errorf("%w: tensor %q has invalid shape %v", errInvalidTinyBundle, name, t.shape)
			}
			length *= dim
		}
		var valueCount uint32
		if err := binary.Read(r, binary.LittleEndian, &valueCount); err != nil {
			return nil, fmt.Errorf("read TinyMelNet tensor %q value count: %w", name, err)
		}
		if uint64(valueCount) != length {
			return nil, fmt.Errorf("%w: tensor %q shape has %d values, bundle says %d", errInvalidTinyBundle, name, length, valueCount)
		}
		t, err := readTinyTensorData(r, name, t, int(valueCount))
		if err != nil {
			return nil, err
		}
		tensors[name] = t
	}
	var extra [1]byte
	if n, err := r.Read(extra[:]); err != io.EOF || n != 0 {
		return nil, fmt.Errorf("%w: trailing bundle data", errInvalidTinyBundle)
	}
	return tinyModelFromTensors(tensors)
}

func readTinyTensorData(r io.Reader, name string, t tinyTensor, count int) (tinyTensor, error) {
	switch t.dtype {
	case tinyDTypeFloat32:
		raw := make([]byte, count*4)
		if _, err := io.ReadFull(r, raw); err != nil {
			return tinyTensor{}, fmt.Errorf("read TinyMelNet tensor %q data: %w", name, err)
		}
		t.f32 = make([]float32, count)
		for i := range t.f32 {
			t.f32[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case tinyDTypeUint8:
		t.u8 = make([]uint8, count)
		if _, err := io.ReadFull(r, t.u8); err != nil {
			return tinyTensor{}, fmt.Errorf("read TinyMelNet tensor %q data: %w", name, err)
		}
	case tinyDTypeInt64:
		raw := make([]byte, count*8)
		if _, err := io.ReadFull(r, raw); err != nil {
			return tinyTensor{}, fmt.Errorf("read TinyMelNet tensor %q data: %w", name, err)
		}
		t.i64 = make([]int64, count)
		for i := range t.i64 {
			t.i64[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
		}
	case tinyDTypeInt32:
		raw := make([]byte, count*4)
		if _, err := io.ReadFull(r, raw); err != nil {
			return tinyTensor{}, fmt.Errorf("read TinyMelNet tensor %q data: %w", name, err)
		}
		t.i32 = make([]int32, count)
		for i := range t.i32 {
			t.i32[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	default:
		return tinyTensor{}, fmt.Errorf("%w: tensor %q has unsupported dtype %d", errInvalidTinyBundle, name, t.dtype)
	}
	return t, nil
}

func tinyModelFromTensors(tensors map[string]tinyTensor) (*TinyMelModel, error) {
	get := func(name string, dtype uint8, shape ...uint32) (tinyTensor, error) {
		t, ok := tensors[name]
		if !ok {
			return tinyTensor{}, fmt.Errorf("%w: missing tensor %q", errInvalidTinyBundle, name)
		}
		if t.dtype != dtype || !bytes.Equal(uint32Bytes(t.shape), uint32Bytes(shape)) {
			return tinyTensor{}, fmt.Errorf("%w: tensor %q has dtype %d shape %v, expected dtype %d shape %v", errInvalidTinyBundle, name, t.dtype, t.shape, dtype, shape)
		}
		return t, nil
	}
	float := func(name string, shape ...uint32) ([]float32, error) {
		t, err := get(name, tinyDTypeFloat32, shape...)
		return t.f32, err
	}
	u8 := func(name string, shape ...uint32) ([]uint8, error) {
		t, err := get(name, tinyDTypeUint8, shape...)
		return t.u8, err
	}
	floatScalar := func(name string) (float32, error) {
		v, err := float(name)
		if err != nil {
			return 0, err
		}
		return v[0], nil
	}
	u8Scalar := func(name string) (uint8, error) {
		v, err := u8(name)
		if err != nil {
			return 0, err
		}
		return v[0], nil
	}
	conv := func(weightName, scaleName, zeroName, biasName string, inC, outC, kernel, stride, groups int, weightShape ...uint32) (tinyConv1D, error) {
		w, err := u8(weightName, weightShape...)
		if err != nil {
			return tinyConv1D{}, err
		}
		scale, err := floatScalar(scaleName)
		if err != nil {
			return tinyConv1D{}, err
		}
		zero, err := u8Scalar(zeroName)
		if err != nil {
			return tinyConv1D{}, err
		}
		bias, err := float(biasName, uint32(outC))
		if err != nil {
			return tinyConv1D{}, err
		}
		inPerGroup := inC / groups
		packed := make([]uint8, len(w))
		for oc := 0; oc < outC; oc++ {
			for ic := 0; ic < inPerGroup; ic++ {
				for k := 0; k < kernel; k++ {
					source := (oc*inPerGroup+ic)*kernel + k
					destination := (oc*kernel+k)*inPerGroup + ic
					packed[destination] = w[source]
				}
			}
		}
		conv := tinyConv1D{weight: w, packed: packed, weightScale: scale, weightZero: zero, bias: bias, inChannels: inC, outChannels: outC, kernel: kernel, stride: stride, groups: groups}
		conv.depthwisePacked = packTinyDepthwiseWeights(conv)
		return conv, nil
	}

	m := &TinyMelModel{}
	var err error
	if m.stem, err = conv("onnx::Conv_257_quantized", "onnx::Conv_257_scale", "onnx::Conv_257_zero_point", "onnx::Conv_258", 80, 192, 5, 2, 1, 192, 80, 5); err != nil {
		return nil, err
	}
	blockSpecs := [...]struct {
		depthW, depthScale, depthZero, depthBias string
		pointW, pointScale, pointZero, pointBias string
		depthStride                              int
	}{
		{"stem.3.depthwise.weight_quantized", "stem.3.depthwise.weight_scale", "stem.3.depthwise.weight_zero_point", "stem.3.depthwise.bias", "onnx::Conv_260_quantized", "onnx::Conv_260_scale", "onnx::Conv_260_zero_point", "onnx::Conv_261", 2},
		{"stem.4.depthwise.weight_quantized", "stem.4.depthwise.weight_scale", "stem.4.depthwise.weight_zero_point", "stem.4.depthwise.bias", "onnx::Conv_263_quantized", "onnx::Conv_263_scale", "onnx::Conv_263_zero_point", "onnx::Conv_264", 2},
		{"stem.5.depthwise.weight_quantized", "stem.5.depthwise.weight_scale", "stem.5.depthwise.weight_zero_point", "stem.5.depthwise.bias", "onnx::Conv_266_quantized", "onnx::Conv_266_scale", "onnx::Conv_266_zero_point", "onnx::Conv_267", 1},
	}
	for i, spec := range blockSpecs {
		stride := spec.depthStride
		m.blocks[i].depthwise, err = conv(spec.depthW, spec.depthScale, spec.depthZero, spec.depthBias, 192, 192, 5, stride, 192, 192, 1, 5)
		if err != nil {
			return nil, err
		}
		m.blocks[i].pointwise, err = conv(spec.pointW, spec.pointScale, spec.pointZero, spec.pointBias, 192, 192, 1, 1, 1, 192, 192, 1)
		if err != nil {
			return nil, err
		}
	}
	if m.gruW, err = float("onnx::GRU_309", 2, 384, 192); err != nil {
		return nil, err
	}
	if m.gruR, err = float("onnx::GRU_310", 2, 384, 128); err != nil {
		return nil, err
	}
	if m.gruB, err = float("onnx::GRU_308", 2, 768); err != nil {
		return nil, err
	}
	if m.poolQuery, err = u8("pool.query_quantized", 256); err != nil {
		return nil, err
	}
	if m.poolQueryScale, err = floatScalar("pool.query_scale"); err != nil {
		return nil, err
	}
	if m.poolQueryZero, err = u8Scalar("pool.query_zero_point"); err != nil {
		return nil, err
	}
	if m.headNormW, err = float("head.0.weight", 256); err != nil {
		return nil, err
	}
	if m.headNormB, err = float("head.0.bias", 256); err != nil {
		return nil, err
	}
	if m.head1, err = tinyLinearFromTensors(u8, floatScalar, u8Scalar, float, "head.1", 256, 256, 256, 256); err != nil {
		return nil, err
	}
	if m.head2, err = tinyLinearFromTensors(u8, floatScalar, u8Scalar, float, "head.4", 256, 1, 256, 1); err != nil {
		return nil, err
	}
	return m, nil
}

func tinyLinearFromTensors(
	u8 func(string, ...uint32) ([]uint8, error),
	floatScalar func(string) (float32, error),
	u8Scalar func(string) (uint8, error),
	float func(string, ...uint32) ([]float32, error),
	prefix string, in, out, weightDim0, weightDim1 int,
) (tinyQuantizedLinear, error) {
	weight, err := u8(prefix+".weight_quantized", uint32(weightDim0), uint32(weightDim1))
	if err != nil {
		return tinyQuantizedLinear{}, err
	}
	scale, err := floatScalar(prefix + ".weight_scale")
	if err != nil {
		return tinyQuantizedLinear{}, err
	}
	zero, err := u8Scalar(prefix + ".weight_zero_point")
	if err != nil {
		return tinyQuantizedLinear{}, err
	}
	bias, err := float(prefix+".bias", uint32(out))
	if err != nil {
		return tinyQuantizedLinear{}, err
	}
	return tinyQuantizedLinear{weight: weight, weightScale: scale, weightZero: zero, bias: bias, in: in, out: out}, nil
}

func uint32Bytes(values []uint32) []byte {
	var out []byte
	for _, value := range values {
		out = binary.LittleEndian.AppendUint32(out, value)
	}
	return out
}
