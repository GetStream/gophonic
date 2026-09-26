// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package qwen3asr runs Qwen3-ASR, Alibaba's multilingual speech recognizer
// (30 languages and 22 Chinese dialects), from the official Hugging Face
// checkpoints Qwen/Qwen3-ASR-1.7B and Qwen/Qwen3-ASR-0.6B.
//
// The model is an audio encoder (AuT) feeding a Qwen3 decoder. The encoder
// turns 128-band log-mel features into one embedding per 80 ms: three
// stride-two convolutions per one-second chunk, then transformer layers
// that attend within eight-second windows. The embeddings replace the
// <|audio_pad|> tokens of a chat prompt, and the decoder answers greedily
// with "language X<asr_text>text". A Transcriber implements
// speech.Transcriber: it detects the language or takes a forced one, primes
// recognition with context text, and cuts audio longer than 20 minutes at
// quiet points.
//
// In the "gpu-q8" format, the default where a Metal GPU is present, both
// run on the Apple GPU: the encoder with every BF16 weight exact, the
// decoder with int8 weights in blocks of 32. In "f16" both run on the CPU's
// SME matrix units with every BF16 weight exact. The encoders round activations to FP16
// for their matrix products, which accumulate in FP32. Warm transcriptions
// allocate nothing.
//
// Outputs match the official qwen-asr package: the tests compare features,
// encoder rows, prompt ids, first-step logits, and transcripts with its FP32
// run. Like vLLM, and unlike its Transformers CPU path, the encoder attends
// within windows of n_window_infer frames.
package qwen3asr
