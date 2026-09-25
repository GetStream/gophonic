// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package qwen3tts runs Qwen3-TTS-12Hz text-to-speech in pure Go: the
// official Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice snapshot, as it is
// published. Speech streams as it is generated: every 80 ms frame of 24 kHz
// audio is decoded as soon as its codec tokens exist, and text can keep
// arriving while earlier text is spoken, as a language model writes it.
//
// Three models run per frame. The talker, a Qwen3 decoder with Qwen3-1.7B's
// geometry, reads one input row per step (a text token's projected
// embedding plus the previous frame's codec embeddings) and predicts the
// frame's first codebook. The code predictor, a five-layer Qwen3 decoder,
// predicts the other fifteen codebooks from the talker's state. Both run on
// the Apple GPU through gophonic's Qwen3 core when Metal is present. The
// codec decoder turns the sixteen codebooks into 1920 samples on the CPU's
// matrix unit, concurrently with the GPU.
package qwen3tts

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/whispergemm"
	"github.com/thesyncim/vibejson"
)

// SampleRate is the rate of the synthesized speech.
const SampleRate = 24000

// FrameSamples is the audio of one codec frame: 80 ms.
const FrameSamples = 1920

const (
	groups = 16   // codebooks per frame
	codes  = 2048 // entries per codebook
)

type config struct {
	ModelType     string `json:"model_type"`
	TTSModelType  string `json:"tts_model_type"`
	TokenizerType string `json:"tokenizer_type"`
	TTSBOS        int    `json:"tts_bos_token_id"`
	TTSEOS        int    `json:"tts_eos_token_id"`
	TTSPad        int    `json:"tts_pad_token_id"`
	Talker        struct {
		qwen3lm.TextConfig
		CodePredictor qwen3lm.TextConfig `json:"code_predictor_config"`
		NumCodeGroups int                `json:"num_code_groups"`
		TextHidden    int                `json:"text_hidden_size"`
		TextVocab     int                `json:"text_vocab_size"`
		CodecBOS      int                `json:"codec_bos_id"`
		CodecEOS      int                `json:"codec_eos_token_id"`
		CodecPad      int                `json:"codec_pad_id"`
		Think         int                `json:"codec_think_id"`
		NoThink       int                `json:"codec_nothink_id"`
		ThinkBOS      int                `json:"codec_think_bos_id"`
		ThinkEOS      int                `json:"codec_think_eos_id"`
		Languages     map[string]int     `json:"codec_language_id"`
		Speakers      map[string]int     `json:"spk_id"`
		Dialects      map[string]any     `json:"spk_is_dialect"`
	} `json:"talker_config"`
}

// IsModelDir reports whether dir holds a Qwen3-TTS snapshot.
func IsModelDir(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	return vibejson.Unmarshal(raw, &c) == nil && c.ModelType == "qwen3_tts"
}

// Options selects how Load prepares the model.
type Options struct {
	// Format is the talker's and code predictor's weight format, as in
	// package qwen3: empty picks the GPU with int8 blocks (gpu-q8) where
	// Metal is present, and exact FP16 weights on the CPU elsewhere.
	Format string
	// Threads bounds the CPU workers of the codec decoder and the CPU
	// projections; zero picks the performance cores.
	Threads int
}

// Model is a loaded Qwen3-TTS model, immutable and shared by every
// Synthesizer made from it.
type Model struct {
	cfg     config
	tokens  *qwen3lm.Tokenizer
	talker  *qwen3lm.Weights
	tEval   *qwen3lm.Evaluator
	cp      *qwen3lm.Weights
	cpEval  *qwen3lm.Evaluator
	unmap   []func() error
	threads int

	hidden, cpHidden int

	// textEmbed is the talker's text embedding table (BF16, mapped), and
	// textFC1 and textFC2 its projection into the talker's width.
	textEmbed        []uint16
	textFC1, textFC2 dense
	// codecEmbed is the talker's codec embedding table (BF16, mapped),
	// [3072][hidden]; cpEmbed the code predictor's fifteen input tables,
	// [2048][hidden] each.
	codecEmbed []uint16
	cpEmbed    [groups - 1][]uint16
	// cpRows holds every code predictor input row, projected to its width:
	// cpRows[0] for the talker's first-codebook codes, cpRows[g] for
	// codebook g's codes. proj projects the talker state.
	cpRows [groups - 1][]float32
	proj   dense
	// heads are the code predictor's fifteen output heads.
	heads [groups - 1]*whispergemm.PackedVector
	// Text rows of the TTS control tokens.
	bosRow, eosRow, padRow []float32

	codec *codec
}

// Load reads a Qwen3-TTS snapshot directory, including its speech_tokenizer
// subdirectory (the codec).
func Load(dir string, opts Options) (_ *Model, err error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("qwen3tts: %w", err)
	}
	var c config
	if err := vibejson.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("qwen3tts: parse config.json: %w", err)
	}
	if c.ModelType != "qwen3_tts" || c.TokenizerType != "qwen3_tts_tokenizer_12hz" {
		return nil, fmt.Errorf("qwen3tts: model_type %q with tokenizer %q is not Qwen3-TTS-12Hz", c.ModelType, c.TokenizerType)
	}
	if c.Talker.NumCodeGroups != groups || c.Talker.TextHidden != c.Talker.HiddenSize {
		return nil, errors.New("qwen3tts: unsupported codebook count or text width")
	}
	m := &Model{cfg: c, threads: opts.Threads, hidden: c.Talker.HiddenSize, cpHidden: c.Talker.CodePredictor.HiddenSize}
	if m.threads <= 0 {
		m.threads = max(1, qwen3lm.PerformanceCores())
	}
	defer func() {
		if err != nil {
			m.Release()
		}
	}()
	if m.tokens, err = qwen3lm.LoadTokenizer(dir); err != nil {
		return nil, fmt.Errorf("qwen3tts: %w", err)
	}
	format := opts.Format
	if format == "" {
		format = qwen3lm.WeightsF16
		if qwen3lm.GPUAvailable() {
			format = qwen3lm.WeightsGPUQ8
		}
	}
	// The talker: a Qwen3 decoder whose inputs are always precomputed rows,
	// and whose head predicts the first codebook.
	tc := c.Talker.TextConfig
	tc.ModelType, tc.RopeScaling = "qwen3", nil // identical positions on all MRoPE axes make it plain RoPE
	if m.talker, err = qwen3lm.Load(dir, qwen3lm.LoadOptions{Format: format, Prefix: "talker.model.", Config: &tc,
		Head: "talker.codec_head.weight", NoEmbed: true}); err != nil {
		return nil, fmt.Errorf("qwen3tts: talker: %w", err)
	}
	if m.tEval, err = qwen3lm.NewEvaluator(m.talker); err != nil {
		return nil, err
	}
	pc := c.Talker.CodePredictor
	pc.ModelType = "qwen3"
	if m.cp, err = qwen3lm.Load(dir, qwen3lm.LoadOptions{Format: format, Prefix: "talker.code_predictor.model.", Config: &pc, NoEmbed: true}); err != nil {
		return nil, fmt.Errorf("qwen3tts: code predictor: %w", err)
	}
	if m.cpEval, err = qwen3lm.NewEvaluator(m.cp); err != nil {
		return nil, err
	}
	st, err := safetensors.Open(dir)
	if err != nil {
		return nil, fmt.Errorf("qwen3tts: %w", err)
	}
	defer st.Close()
	if err := m.loadTables(st); err != nil {
		return nil, fmt.Errorf("qwen3tts: %w", err)
	}
	if m.codec, err = loadCodec(filepath.Join(dir, "speech_tokenizer"), m.threads); err != nil {
		return nil, fmt.Errorf("qwen3tts: codec: %w", err)
	}
	return m, nil
}

// loadTables reads the embedding tables and small projections that run on
// the CPU, and projects every code predictor input once.
func (m *Model) loadTables(st *safetensors.Checkpoint) error {
	c, h, ch := &m.cfg.Talker, m.hidden, m.cpHidden
	mapped := func(name string, rows, cols int) ([]uint16, error) {
		b, unmap, err := st.MapBF16(name, rows, cols)
		if err == nil {
			m.unmap = append(m.unmap, unmap)
		}
		return b, err
	}
	var err error
	if m.textEmbed, err = mapped("talker.model.text_embedding.weight", c.TextVocab, h); err != nil {
		return err
	}
	if m.codecEmbed, err = mapped("talker.model.codec_embedding.weight", c.Vocab, h); err != nil {
		return err
	}
	for g := range m.cpEmbed {
		if m.cpEmbed[g], err = mapped(fmt.Sprintf("talker.code_predictor.model.codec_embedding.%d.weight", g), codes, h); err != nil {
			return err
		}
	}
	if m.textFC1, err = loadDense(st, "talker.text_projection.linear_fc1", h, h); err != nil {
		return err
	}
	if m.textFC2, err = loadDense(st, "talker.text_projection.linear_fc2", h, h); err != nil {
		return err
	}
	if m.proj, err = loadDense(st, "talker.code_predictor.small_to_mtp_projection", ch, h); err != nil {
		return err
	}
	for g := range m.heads {
		w, err := st.Float32(fmt.Sprintf("talker.code_predictor.lm_head.%d.weight", g), codes, ch)
		if err != nil {
			return err
		}
		if m.heads[g], err = whispergemm.NewPackedVector(w, ch, codes, ch); err != nil {
			return err
		}
	}
	// Every code predictor input is a projected embedding row: project the
	// tables once. Row g holds codebook g's inputs (g = 0: the talker's
	// first-codebook table).
	exec, err := whispergemm.NewExecutor(m.threads)
	if err != nil {
		return err
	}
	defer exec.Close()
	table := make([]float32, codes*h)
	for g := range m.cpRows {
		src := m.codecEmbed
		if g > 0 {
			src = m.cpEmbed[g-1]
		}
		for i := range table {
			table[i] = q8gemm.BF16ToF32(src[i])
		}
		m.cpRows[g] = make([]float32, codes*ch)
		if err := m.proj.apply(exec, m.cpRows[g], table, codes); err != nil {
			return err
		}
	}
	// The TTS control tokens' text rows.
	rows := make([]float32, 3*h)
	for i, id := range []int{m.cfg.TTSBOS, m.cfg.TTSEOS, m.cfg.TTSPad} {
		m.textRows(exec, rows[i*h:(i+1)*h], []int{id}, make([]float32, h))
	}
	m.bosRow, m.eosRow, m.padRow = rows[:h], rows[h:2*h], rows[2*h:]
	return nil
}

// textRows writes the projected text embeddings of ids to dst, one hidden-
// wide row each; tmp holds len(ids) rows of scratch.
func (m *Model) textRows(exec *whispergemm.Executor, dst []float32, ids []int, tmp []float32) error {
	h := m.hidden
	for i, id := range ids {
		row := m.textEmbed[id*h : (id+1)*h]
		for j, b := range row {
			dst[i*h+j] = q8gemm.BF16ToF32(b)
		}
	}
	n := len(ids)
	if err := m.textFC1.apply(exec, tmp[:n*h], dst[:n*h], n); err != nil {
		return err
	}
	for i, v := range tmp[:n*h] {
		tmp[i] = silu(v)
	}
	return m.textFC2.apply(exec, dst[:n*h], tmp[:n*h], n)
}

// Speakers lists the preset voices, in lower case.
func (m *Model) Speakers() []string {
	var names []string
	for name := range m.cfg.Talker.Speakers {
		names = append(names, name)
	}
	return names
}

// Languages lists the languages a voice can be asked to speak, in lower
// case; an empty language lets the model follow the text.
func (m *Model) Languages() []string {
	var names []string
	for name := range m.cfg.Talker.Languages {
		if !strings.Contains(name, "dialect") {
			names = append(names, name)
		}
	}
	return names
}

// Release frees the model's GPU memory and mappings; the model is unusable
// afterwards.
func (m *Model) Release() {
	if m.talker != nil {
		m.talker.Release()
	}
	if m.cp != nil {
		m.cp.Release()
	}
	for _, unmap := range m.unmap {
		unmap()
	}
	m.unmap = nil
}
