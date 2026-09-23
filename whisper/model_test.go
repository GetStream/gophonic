// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

func TestReadWeightsRejectsInvalidHeaders(t *testing.T) {
	for _, data := range [][]byte{nil, []byte("GOINFER1"), []byte("WHISPER1")} {
		if _, err := ReadWeights(bytes.NewReader(data)); err == nil {
			t.Fatalf("accepted invalid header %q", data)
		}
	}
	var wrongVersion bytes.Buffer
	wrongVersion.WriteString("WHISPER1")
	_ = binary.Write(&wrongVersion, binary.LittleEndian, uint32(2))
	_ = binary.Write(&wrongVersion, binary.LittleEndian, uint32(len(expectedTensors())))
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
	path := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if path == "" {
		t.Skip("set GOPHONIC_WHISPER_MODEL to the converted official tiny.en bundle")
	}
	m, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range expectedTensors() {
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
