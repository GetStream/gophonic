// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

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
	languages speech.LanguageSet // English names the model supports
	// Limits of the language sets transcriptions have given.
	limitsMu sync.RWMutex
	limits   map[speech.LanguageSet]*limits
	turn     *turnHead // judges the end of a turn, for the checkpoints it was trained on
}

// tokenIDs are the special tokens of the chat prompt and its output.
type tokenIDs struct {
	audioPad, asrText int
	eos               [2]int
	// language starts the output, "language English<asr_text>...", and
	// none is the name it gives audio without speech.
	language, none int
}

// Options configures Load.
type Options struct {
	// Format is the decoder's weight format, as gophonic.Options names it:
	// "f16" (every BF16 weight exactly, on the CPU's matrix units), "int8",
	// "gpu", "gpu-q8" (int8 blocks of 32, on the Apple GPU), or "gpu-q4".
	// Empty picks "gpu-q8" where a Metal GPU is present and "f16" elsewhere.
	// The GPU formats run the encoder on the GPU too, every BF16 weight
	// exact.
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
		opts.Format = qwen3lm.WeightsF16
		if qwen3lm.GPUAvailable() {
			opts.Format = qwen3lm.WeightsGPUQ8
		}
	case qwen3lm.WeightsF16, qwen3lm.WeightsInt8, qwen3lm.WeightsGPU, qwen3lm.WeightsGPUQ8, qwen3lm.WeightsGPUQ4:
	default:
		return nil, fmt.Errorf("qwen3asr: unknown weight format %q: %w", opts.Format, speech.ErrUnsupported)
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
	if opts.Format == qwen3lm.WeightsGPUQ8 || opts.Format == qwen3lm.WeightsGPU || opts.Format == qwen3lm.WeightsGPUQ4 {
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
		if err != nil {
			if genc != nil {
				genc.release()
			}
			if enc != nil {
				_ = enc.memory.Close()
			}
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
	m := &Model{enc: enc, genc: genc, lm: lm, eval: eval, tok: tok}
	for _, name := range c.SupportLanguages {
		if l, ok := speech.ParseLanguage(name); ok {
			m.languages = m.languages.With(l)
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
	var tws qwen3lm.TokenizerWorkspace
	named, err := tok.EncodeInto("language None", make([]int, 0, 16), &tws)
	if err != nil || len(named) != 2 {
		return nil, fmt.Errorf("qwen3asr: tokenizer splits \"language None\" as %v: %v", named, err)
	}
	m.ids.language, m.ids.none = named[0], named[1]
	if m.ids.audioPad != c.Thinker.AudioToken {
		return nil, fmt.Errorf("qwen3asr: audio token %d in config, %d in tokenizer", c.Thinker.AudioToken, m.ids.audioPad)
	}
	m.turn = turnHeadFor(lm.Config())
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

// Languages is the set of languages the model accepts in
// speech.Options.Language and Languages.
func (m *Model) Languages() speech.LanguageSet { return m.languages }

// Close releases native resources and drops references to loaded weights,
// tokenizer data, and language limits. All lanes must be closed first; close
// must not overlap inference. Repeated calls are harmless.
func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	var err error
	if m.enc != nil {
		err = m.enc.memory.Close()
	}
	if m.lm != nil {
		m.lm.Release()
	}
	if m.genc != nil {
		m.genc.release()
	}
	// Do not copy or reset limitsMu after use. No lanes remain to access it.
	m.enc, m.genc, m.lm, m.eval, m.tok = nil, nil, nil, nil, nil
	m.limits, m.turn = nil, nil
	m.languages = 0
	return err
}
