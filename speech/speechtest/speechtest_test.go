// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speechtest

import (
	"testing"
	"time"

	"github.com/GetStream/gophonic/speech"
)

func TestTone(t *testing.T) {
	TestSynthesizer(t, NewTone(10*time.Millisecond), speech.SpeakOptions{}, "Sure, the answer is simple.")
}
