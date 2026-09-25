// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

// Model is a loaded Qwen3-ASR checkpoint: the audio encoder, the Qwen3
// decoder with its language-model head, and the tokenizer. It is immutable
// and shared by every Transcriber opened from it.
type Model struct {
	enc       *encoder    // geometry, with CPU weights unless genc is set
	genc      *gpuEncoder // the encoder in GPU memory, for GPU formats
	lm        *qwen3lm.Weights
	eval      *qwen3lm.Evaluator
	tok       *qwen3lm.Tokenizer
	ids       tokenIDs
	languages map[string]bool // English names the model supports
}

// tokenIDs are the special tokens of the chat prompt and its output.
type tokenIDs struct {
	audioPad, asrText int
	eos               [2]int
}

// Decoder weight formats for Options.Format.
const (
	// FormatF16 runs the encoder and decoder on the CPU's SME matrix units,
	// keeping every BF16 weight exactly.
	FormatF16 = qwen3lm.WeightsF16
	// FormatGPU runs the encoder and decoder on the Apple GPU (darwin/arm64):
	// the encoder with every BF16 weight exact, the decoder with FP32
	// activations and int8 weights in blocks of 32 sharing an FP16 scale, in
	// a Hadamard-rotated basis, more faithful than GGML's Q8_0.
	FormatGPU = qwen3lm.WeightsGPUQ8
)

// Options configures Load.
type Options struct {
	// Format is the decoder's weight format: FormatF16, FormatGPU, or empty
	// for FormatGPU where a Metal GPU is present and FormatF16 elsewhere.
	Format string
}

type config struct {
	ModelType        string   `json:"model_type"`
	SupportLanguages []string `json:"support_languages"`
	Thinker          struct {
		ModelType  string             `json:"model_type"`
		Audio      audioConfig        `json:"audio_config"`
		Text       qwen3lm.TextConfig `json:"text_config"`
		AudioToken int                `json:"audio_token_id"`
	} `json:"thinker_config"`
}

type audioConfig struct {
	DModel       int    `json:"d_model"`
	Layers       int    `json:"encoder_layers"`
	Heads        int    `json:"encoder_attention_heads"`
	FFN          int    `json:"encoder_ffn_dim"`
	MelBins      int    `json:"num_mel_bins"`
	Downsample   int    `json:"downsample_hidden_size"`
	OutputDim    int    `json:"output_dim"`
	Window       int    `json:"n_window"`
	WindowInfer  int    `json:"n_window_infer"`
	MaxPositions int    `json:"max_source_positions"`
	Activation   string `json:"activation_function"`
	ScaleEmbed   bool   `json:"scale_embedding"`
}

// IsModelDir reports whether dir holds a Qwen3-ASR checkpoint: a
// config.json whose model_type is qwen3_asr.
func IsModelDir(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	return vibejson.Unmarshal(raw, &c) == nil && c.ModelType == "qwen3_asr"
}

// Load reads an official Qwen3-ASR snapshot directory (Qwen/Qwen3-ASR-1.7B
// or Qwen/Qwen3-ASR-0.6B from Hugging Face), for the GPU where Metal is
// present and for the CPU otherwise.
func Load(dir string, opts Options) (_ *Model, err error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("qwen3asr: %w", err)
	}
	var c config
	if err := vibejson.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("qwen3asr: parse config.json: %w", err)
	}
	if c.ModelType != "qwen3_asr" {
		return nil, fmt.Errorf("qwen3asr: model_type %q is not qwen3_asr", c.ModelType)
	}
	text := c.Thinker.Text
	if err := checkRoPE(&text); err != nil {
		return nil, err
	}
	switch opts.Format {
	case "":
		opts.Format = FormatF16
		if qwen3lm.GPUAvailable() {
			opts.Format = FormatGPU
		}
	case FormatF16, FormatGPU, qwen3lm.WeightsInt8, qwen3lm.WeightsGPU, qwen3lm.WeightsGPUQ4:
	default:
		return nil, fmt.Errorf("qwen3asr: unsupported decoder format %q", opts.Format)
	}
	if c.Thinker.Audio.OutputDim != text.HiddenSize {
		return nil, fmt.Errorf("qwen3asr: audio output width %d does not match decoder width %d", c.Thinker.Audio.OutputDim, text.HiddenSize)
	}
	st, err := safetensors.Open(dir)
	if err != nil {
		return nil, fmt.Errorf("qwen3asr: %w", err)
	}
	// GPU formats run the encoder on the GPU too; its kernels need 64-wide
	// heads, which every Qwen3-ASR size has.
	var (
		enc  *encoder
		genc *gpuEncoder
	)
	const audioPrefix = "thinker.audio_tower."
	if opts.Format == FormatGPU || opts.Format == qwen3lm.WeightsGPU || opts.Format == qwen3lm.WeightsGPUQ4 {
		if enc, err = newEncoder(c.Thinker.Audio); err == nil {
			genc, err = loadGPUEncoder(st, enc, audioPrefix)
		}
	} else {
		enc, err = loadEncoder(st, c.Thinker.Audio, audioPrefix)
	}
	head := "thinker.lm_head.weight"
	if !st.Has(head) {
		head = "thinker.model.embed_tokens.weight" // tied embeddings
	}
	st.Close()
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil && genc != nil {
			genc.release()
		}
	}()
	lm, err := qwen3lm.Load(dir, qwen3lm.LoadOptions{Format: opts.Format, Prefix: "thinker.model.", Config: &text, Head: head})
	if err != nil {
		return nil, fmt.Errorf("qwen3asr: decoder: %w", err)
	}
	defer func() {
		if err != nil {
			lm.Release()
		}
	}()
	eval, err := qwen3lm.NewEvaluator(lm)
	if err != nil {
		return nil, err
	}
	tok, err := qwen3lm.LoadTokenizer(dir)
	if err != nil {
		return nil, fmt.Errorf("qwen3asr: %w", err)
	}
	m := &Model{enc: enc, genc: genc, lm: lm, eval: eval, tok: tok, languages: map[string]bool{}}
	for _, name := range c.SupportLanguages {
		if canonical, ok := speech.LanguageName(name); ok {
			m.languages[canonical] = true
		}
	}
	lookup := func(content string) int {
		id, ok := tok.AddedID(content)
		if !ok {
			err = fmt.Errorf("qwen3asr: tokenizer lacks %s", content)
		}
		return id
	}
	m.ids = tokenIDs{audioPad: lookup("<|audio_pad|>"), asrText: lookup("<asr_text>"),
		eos: [2]int{lookup("<|im_end|>"), lookup("<|endoftext|>")}}
	if err != nil {
		return nil, err
	}
	if m.ids.audioPad != c.Thinker.AudioToken {
		return nil, fmt.Errorf("qwen3asr: audio token %d in config, %d in tokenizer", c.Thinker.AudioToken, m.ids.audioPad)
	}
	return m, nil
}

// checkRoPE accepts the multimodal RoPE of Qwen3-ASR's decoder and clears
// it. Every token of a transcription prompt has the same position on all
// three MRoPE axes, which makes MRoPE equal to the decoder's plain RoPE.
func checkRoPE(c *qwen3lm.TextConfig) error {
	if c.RopeScaling == nil {
		return nil
	}
	scaling, ok := c.RopeScaling.(map[string]any)
	if !ok {
		return errors.New("qwen3asr: unsupported rope_scaling")
	}
	if t, _ := scaling["rope_type"].(string); t != "default" {
		return fmt.Errorf("qwen3asr: unsupported rope_type %v", scaling["rope_type"])
	}
	sections, _ := scaling["mrope_section"].([]any)
	sum := 0
	for _, s := range sections {
		v, ok := s.(float64)
		if !ok {
			return errors.New("qwen3asr: malformed mrope_section")
		}
		sum += int(v)
	}
	if head := c.HeadDim; head == 0 || sum != head/2 {
		return fmt.Errorf("qwen3asr: mrope_section covers %d of %d frequencies", sum, head/2)
	}
	c.RopeScaling = nil
	return nil
}

// Languages reports the English names of the languages the model accepts in
// speech.Options.Language, in speech.Languages order.
func (m *Model) Languages() []string {
	var names []string
	for _, l := range speech.Languages {
		if m.languages[l.Name] {
			names = append(names, l.Name)
		}
	}
	return names
}

// Release frees resources the decoder holds outside the Go heap. The model
// is unusable afterwards.
func (m *Model) Release() {
	m.lm.Release()
	if m.genc != nil {
		m.genc.release()
	}
}
