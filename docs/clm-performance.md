# CLM and Qwen3 CPU performance campaign

Measured on 2026-09-24 on an Apple M4 Max, with one CPU thread/core reported
and GPU layers disabled. Model loading is excluded from steady-state timings.
The local Qwen3-8B model is the official safetensors checkpoint; the competitor
uses the official Qwen3-8B Q8_0 GGUF. These weight formats differ: the local
int8 representation has one FP32 scale per row, while GGUF Q8_0 has one FP16
scale per 32 weights. Hidden
vectors and CLM rankings must be checked separately from timing.

| Path | One-token last-hidden latency | Allocation boundary | Notes |
| --- | ---: | ---: | --- |
| GoInfer weight-only int8, existing `HiddenLast` | 959 ms | ~1.8 MB, 2,418 allocs | Same pretokenized input, `GOMAXPROCS=1` |
| gophonic fused Q8 evaluator | 436 ms | 0 B, 0 allocs | Same loaded int8 weights and token ID, pretokenized workspace path |
| llama.cpp Q8_0 CPU, `llama-bench` | 111.5 ms | not measured | `-p 1 -n 0 -embd 1 -t 1 -ngl 0 -dev none -r 5` |

For 12 tokens, isolated sequential Go runs measured 5.251 s/op in the
original token-major fast evaluator and 3.880 s/op in the current public text
API with layer-batched prefill and reusable tokenizer. Both have zero warmed
allocations. The older GoInfer batched path measured 4.0–4.2 s/op on the same
input. llama.cpp CPU prefill measured 0.778 s/op (`-p 12 -n 0 -embd 1 -t 1
-ngl 0 -dev none -r 5`). The original token-major loop streams about 83.35 GB
of int8 projection weights for 12 tokens; the current layer-batched pass reuses
weight rows across tokens. These numbers compare different quantized weight
formats and input token contents. The 500 ms target remains unmet.

The public text API includes tokenization. The new Qwen3 tokenizer matches
official token IDs for the tested Unicode and special-token cases, and uses
zero warmed allocations, including decomposed Unicode that needs NFC
normalization. The previous GoInfer tokenizer used about 792 B and 39
allocations to tokenize `hello`.

For `hello`, cosine against the official BF16 Qwen3-8B last hidden vector was
0.99738 for local per-row weight-only int8 and 0.99929 for llama.cpp Q8_0.
These are one-example checks, not corpus quality results. The local W8A8 and
int4 modes ran faster but fell to cosines 0.699 and 0.649, respectively, and
are rejected for CLM until a stronger quality design passes a broader gate.

## Why the gap exists

The current weight-only int8 decoder expands each int8 row into float32 scratch,
then reads that scratch for a separate dot product. A CPU profile attributed
roughly 64% of sampled inference time to expansion and 32% to the dot kernel.
The gophonic fused Q8 kernel converts values inside the dot loop and writes no
expanded row. The local fast evaluator reuses per-lane activation and KV
storage. Together these changes cut the measured one-token path by about
2.2×, while leaving a roughly 3.9× gap to llama.cpp.

The Qwen3-8B transformer has about 6.946 billion matrix weights per forward
pass, excluding the LM head skipped by last-hidden inference. A cold independent
request must stream roughly 6.95 GB in an 8-bit representation. The 959 ms
baseline corresponds to only about 7.2 GB/s of effective weight throughput;
this is an observed workload figure, not a physical memory-bandwidth limit.

## Experiments and proof gates

1. **Blockwise Q8 integer dot.** [ggml's CPU implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c)
   uses a quantized activation vector and packed quantized weight rows for an
   integer dot. The installed Apple M4 binary uses a four-row `Q8_0` GEMV with
   ARM SDOT. Prototype a block-32 activation quantizer with FP32 activation scales,
   pack weights once, and compare with a scalar block oracle. First keep our
   weights unchanged to isolate added activation error; then test GGUF-style
   blockwise weights against the official reference. If one activation stream changes CLM
   outputs too much, test a second residual int8 stream before rejecting the
   design. No quantized kernel is enabled in the CLM path without hidden-vector
   and full-ranking gates.
2. **Workspace and lifetime reuse.** [ggml's graph allocator](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-alloc.c)
   and [XNNPACK's runtime planner](https://github.com/google/XNNPACK/blob/master/src/runtime.c)
   reserve storage and reuse it across operations. The local evaluator follows
   this with one workspace per concurrent lane. Measure allocations at the
   pretokenized and public text boundaries separately; require zero warmed
   allocations for the former and reduce the latter without changing token IDs.
3. **Packed row and batch experiments.** [ggml repacks quantized rows](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.cpp)
   for its microkernels. Compare one-, four-, and eight-row layouts at the
   real Qwen projection shapes. Keep only the selected packed representation
   and include resident bytes and pack time in the report. For multiple
   candidates or prompt tokens, batch them through each layer so a weight tile
   serves several activations; report latency and throughput separately.
4. **Exact stored-weight alternative.** Retain the official BF16 Qwen weights
   and widen each value in registers. This reads half the bytes of expanded
   FP32 and preserves the checkpoint's stored weights. Compare against fused
   Q8 for speed, resident bytes, and official hidden-vector fidelity.
5. **Prepared repeated actions.** The [CLM embedder](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/embedder.py)
   embeds candidate actions independently. A caller-owned prepared action set
   can reuse exact projected vectors across requests. Measure cold and reused
   paths separately and include model, tokenizer, head, and token IDs in the
   cache identity.

For every retained optimization, use identical model and token IDs where the
format permits, verify finite and adversarial inputs, test actual dispatch and
allocation counts, and benchmark fresh one-token, 16-token, and 128-token
sequences plus a complete state-and-candidates CLM ranking. Report cold load,
resident memory, one-core latency, and only then scaling across cores.
