// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import "math"

// The official English-model alignment masks decoded from OpenAI Whisper's
// _ALIGNMENT_HEADS. Keep them out of the model file: they are model-family
// metadata, not learned weight tensors. Source: OpenAI Whisper commit
// 86098128c0b4f24f0e2aa2994de830614b474227, whisper/__init__.py.
type alignmentHead struct{ layer, head int }

var tinyENAlignmentHeads = [...]alignmentHead{{1, 0}, {2, 0}, {2, 5}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}}
var baseENAlignmentHeads = [...]alignmentHead{{3, 3}, {4, 7}, {5, 1}, {5, 5}, {5, 7}}
var smallENAlignmentHeads = [...]alignmentHead{{6, 6}, {7, 0}, {7, 3}, {7, 8}, {8, 2}, {8, 5}, {8, 7}, {9, 0}, {9, 4}, {9, 8}, {9, 10}, {10, 0}, {10, 1}, {10, 2}, {10, 3}, {10, 6}, {10, 11}, {11, 2}, {11, 4}}

func alignmentHeadsForDims(d Dims) []alignmentHead {
	switch {
	case d.TextState == 384 && d.TextLayers == 4 && d.TextHeads == 6:
		return tinyENAlignmentHeads[:]
	case d.TextState == 512 && d.TextLayers == 6 && d.TextHeads == 8:
		return baseENAlignmentHeads[:]
	case d.TextState == 768 && d.TextLayers == 12 && d.TextHeads == 12:
		return smallENAlignmentHeads[:]
	default:
		return nil
	}
}

// alignmentCapture stores selected cross-attention probabilities. It is
// enabled only for word timing, and is owned by one Transcriber.
type alignmentCapture struct {
	probabilities []float32 // [selected head, actual positions, actual audio frames]
	heads         []alignmentHead
	frames        int
	positions     int
}

func (a *alignmentCapture) capture(layer, position int, scores []float32) {
	if a == nil || position >= a.positions {
		return
	}
	for selected, head := range a.heads {
		if head.layer != layer {
			continue
		}
		src := scores[head.head*AudioFrames : head.head*AudioFrames+a.frames]
		dst := a.probabilities[(selected*a.positions+position)*a.frames:]
		var sum float64
		for _, value := range src {
			sum += float64(value)
		}
		if sum == 0 {
			continue
		}
		inverse := float32(1 / sum)
		for i, value := range src {
			dst[i] = value * inverse
		}
	}
}

// normalizeAlignmentInto computes the reference per-head, per-frame token
// standardization, width-seven reflected median filter, and head mean.
func normalizeAlignmentInto(dst []float32, capture *alignmentCapture, positions int) {
	frames := capture.frames
	for i := range dst[:(positions-2)*frames] {
		dst[i] = 0
	}
	for head := range capture.heads {
		base := head * positions * frames
		for frame := 0; frame < frames; frame++ {
			var sum, squares float64
			for pos := 0; pos < positions; pos++ {
				value := float64(capture.probabilities[base+pos*frames+frame])
				sum += value
				squares += value * value
			}
			mean := sum / float64(positions)
			variance := squares/float64(positions) - mean*mean
			if variance < 1e-16 {
				variance = 1e-16
			}
			inv := 1 / math.Sqrt(variance)
			// Store normalized scores back in the optional capture workspace.
			for pos := 0; pos < positions; pos++ {
				at := base + pos*frames + frame
				capture.probabilities[at] = float32((float64(capture.probabilities[at]) - mean) * inv)
			}
		}
		// Omit SOT at position zero and EOT at the final position.
		for row := 0; row < positions-2; row++ {
			pos := row + 1
			for frame := 0; frame < frames; frame++ {
				var seven [7]float32
				for tap := -3; tap <= 3; tap++ {
					index := frame + tap
					if index < 0 {
						index = -index
					}
					if index >= frames {
						index = 2*frames - index - 2
					}
					if index < 0 {
						index = 0
					}
					if index >= frames {
						index = frames - 1
					}
					seven[tap+3] = capture.probabilities[base+pos*frames+index]
				}
				for i := 1; i < 7; i++ {
					value := seven[i]
					j := i
					for j > 0 && seven[j-1] > value {
						seven[j] = seven[j-1]
						j--
					}
					seven[j] = value
				}
				dst[row*frames+frame] += seven[3] / float32(len(capture.heads))
			}
		}
	}
}
