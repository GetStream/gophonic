// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

// dimsBundle writes a version-2 bundle whose tensors hold small finite values.
func dimsBundle(d Dims, raw [6]uint32) []byte {
	var payload bytes.Buffer
	_ = binary.Write(&payload, binary.LittleEndian, raw)
	for i, spec := range expectedTensors(d) {
		payload.Write(binary.LittleEndian.AppendUint16(nil, uint16(len(spec.name))))
		payload.WriteString(spec.name)
		payload.WriteByte(byte(len(spec.shape)))
		count := 1
		for _, dim := range spec.shape {
			_ = binary.Write(&payload, binary.LittleEndian, uint32(dim))
			count *= dim
		}
		_ = binary.Write(&payload, binary.LittleEndian, uint32(count))
		value := math.Float32bits(float32(i%7) / 8)
		for range count {
			_ = binary.Write(&payload, binary.LittleEndian, value)
		}
	}
	var out bytes.Buffer
	out.WriteString("WHISPER1")
	_ = binary.Write(&out, binary.LittleEndian, bundleVersionDims)
	_ = binary.Write(&out, binary.LittleEndian, uint32(len(expectedTensors(d))))
	out.Write(payload.Bytes())
	sum := sha256.Sum256(payload.Bytes())
	out.Write(sum[:])
	return out.Bytes()
}

func TestReadWeightsVersion2Dims(t *testing.T) {
	d := Dims{AudioState: 64, AudioHeads: 2, AudioLayers: 2, TextState: 64, TextHeads: 2, TextLayers: 3}
	raw := [6]uint32{64, 2, 2, 64, 2, 3}
	data := dimsBundle(d, raw)
	m, err := ReadWeights(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if m.Dims() != d {
		t.Fatalf("dims %+v, want %+v", m.Dims(), d)
	}
	if got := len(m.tensor("decoder.blocks.2.mlp.0.weight")); got != 4*64*64 {
		t.Fatalf("decoder MLP has %d values", got)
	}
	// The dimensions are inside the checksummed payload.
	tampered := bytes.Clone(data)
	tampered[16+8] ^= 1
	if _, err := ReadWeights(bytes.NewReader(tampered)); err == nil {
		t.Fatal("accepted tampered dimensions")
	}
	for _, bad := range [][6]uint32{{0, 2, 2, 64, 2, 3}, {64, 3, 2, 64, 2, 3}, {64, 2, 2, 128, 2, 3}, {64, 2, 0, 64, 2, 3}} {
		if _, err := ReadWeights(bytes.NewReader(dimsBundle(d, bad))); err == nil {
			t.Fatalf("accepted invalid dimensions %v", bad)
		}
	}
}

func TestReadWeightsRejectsInvalidHeaders(t *testing.T) {
	for _, data := range [][]byte{nil, []byte("GOINFER1"), []byte("WHISPER1")} {
		if _, err := ReadWeights(bytes.NewReader(data)); err == nil {
			t.Fatalf("accepted invalid header %q", data)
		}
	}
	var wrongVersion bytes.Buffer
	wrongVersion.WriteString("WHISPER1")
	_ = binary.Write(&wrongVersion, binary.LittleEndian, uint32(3))
	_ = binary.Write(&wrongVersion, binary.LittleEndian, uint32(len(expectedTensors(TinyENDims))))
	if _, err := ReadWeights(&wrongVersion); err == nil {
		t.Fatal("accepted unsupported bundle version")
	}
}

func TestDecodeFiniteFloat32RejectsNaN(t *testing.T) {
	data := make([]byte, 8)
	binary.LittleEndian.PutUint32(data, 0x3f800000)
	binary.LittleEndian.PutUint32(data[4:], 0x7fc00000)
	values := make([]float32, 2)
	if err := decodeFiniteFloat32(data, values); err == nil {
		t.Fatal("accepted NaN in model weights")
	}
}

func TestOfficialTinyENBundle(t *testing.T) {
	path := testmodels.Path(t, testmodels.WhisperTinyEN)
	m, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range expectedTensors(TinyENDims) {
		count := 1
		for _, dim := range spec.shape {
			count *= dim
		}
		if len(m.tensor(spec.name)) != count {
			t.Fatalf("%s: got %d values, want %d", spec.name, len(m.tensor(spec.name)), count)
		}
	}
	if len(m.tensors) != 167 {
		t.Fatalf("got %d tensors, want 167", len(m.tensors))
	}
	// The official checkpoint stores this projection with a transposed stride.
	// These values are from PyTorch's contiguous FP32 view, and catch a converter
	// that copies raw storage without applying tensor strides.
	want := []float32{
		0.0018625259399414062, -0.004398345947265625,
		0.0037136077880859375, 0.0014696121215820312,
		-0.0032749176025390625, 0.006084442138671875,
		-0.0009918212890625, -0.1939697265625,
	}
	got := m.tensor("encoder.blocks.0.attn.query.weight")
	for i, value := range want {
		if got[i] != value {
			t.Fatalf("transposed weight[%d]: got %g, want %g", i, got[i], value)
		}
	}
}
