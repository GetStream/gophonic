# CLM and Qwen3 CPU performance campaign

Measured on 2026-09-24 on an Apple M4 Max, with one CPU thread/core reported
and GPU layers disabled. Model loading is excluded from steady-state timings.
The local Qwen3-8B model is the official safetensors checkpoint; the competitor
uses the official Qwen3-8B Q8_0 GGUF. These weight formats differ: the local
int8 representation has one FP32 scale per row, while GGUF Q8_0 has one FP16
scale per 32 weights. Hidden vectors and CLM rankings must be checked
separately from timing.

| Path | One-token last-hidden latency | Allocation boundary | Notes |
| --- | ---: | ---: | --- |
| GoInfer weight-only int8, existing `HiddenLast` | 959 ms | ~1.8 MB, 2,418 allocs | Same pretokenized input, `GOMAXPROCS=1` |
| gophonic fused Q8 evaluator | 436 ms | 0 B, 0 allocs | Same loaded int8 weights and token ID, pretokenized workspace path |
| gophonic packed Q8 SME, owned encoder | 337.80 ms | 0 B, 0 allocs | One token, isolated 30-call run; packed weights also serve prefill |
| llama.cpp Q8_0 CPU, `llama-bench` | 111.5 ms | not measured | `-p 1 -n 0 -embd 1 -t 1 -ngl 0 -dev none -r 5` |

For the 12-token text `The Moon causes tides by pulling on Earth's oceans.`,
the original token-major evaluator took 5.251 s/op; layer-batched Q8 prefill
then took 3.880 s/op. The 16×64 SME Q8 matrix path averaged 392.27 ms
and 398.42 ms in two isolated 15-call public-API runs. With an owned encoder
using only packed projections, a separate 30-call run measured 382.05 ms
for 12 tokens and 337.80 ms for one token, both with 0 B/op and 0 allocs/op.
A prior 15-call run averaged 501.94 ms, so sub-500 ms has between-process
variance and is not a deterministic guarantee. llama.cpp Q8_0 CPU prefill measured 0.778 s/op (`-p 12 -n 0
-embd 1 -t 1 -ngl 0 -dev none -r 5`). Its benchmark uses different prompt
tokens and its GGUF has a different quantized weight format. The local
12-token SME hidden vector matches the previous int8 decoder with cosine
1.000000000, maximum absolute difference 1.56e-4, and RMS difference 5.73e-6.
The pinned CLM state-and-three-candidates ranking is unchanged.

SME weight packing takes about 5–8 s once. An encoder that owns its newly loaded
decoder discards the canonical Q8 projections after validating that all seven
projections in every layer have packed replacements. In a checkpoint test,
Go heap fell from 14,457 to 7,828 MiB after collection, releasing 6,629 MiB.
External callers constructing a prefill evaluator from their own decoder keep
their canonical weights. Peak loading memory still includes both layouts. A sampled Darwin process held
14,815 MiB RSS before and after compaction, even after `runtime.GC` and
`debug.FreeOSMemory`; the Go heap reduction has not yet translated to a
measured resident-memory reduction.

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

The original GoInfer weight-only int8 decoder expanded each int8 row into
float32 scratch, then read that scratch for a separate dot product. A CPU
profile attributed roughly 64% of its one-token inference time to expansion
and 32% to the dot kernel. Gophonic's fused Q8 GEMV removes that scratch;
its one-token path is about 2.2× faster, while still trailing llama.cpp by
roughly 3.9×.

For 12 tokens, a fresh prompt needs 166.70 billion projection FLOPs. The
layer-batched Q8 SME kernel keeps FP32 activations, widens int8 weights in
registers, and accumulates a 16-token-by-64-output tile. Its full-model
profile attributes 86.74% of sampled inference time to the SME assembly,
4.14% to attention, 3.31% to activation packing, and 1.93% each to row-scale
application and exponential evaluation. The current 100 ms stretch target
would require over 3× more projection throughput than the measured standalone
kernels, and over 69 GB/s of packed weight reads for this independent request.
The full latency and quality gates remain open for that target.

An experimental SME2 `SMOPA` integer tile passed its exact int32 oracle and
warmed zero-allocation tests. Five cache-rotating runs at each 12-token Qwen
projection shape gave a 139 ms estimate for the 36 layers of matrix work alone.
That excludes activation quantization, scaling, normalization, attention, and
KV work. Streaming the roughly 6.95 GB of Q8 matrix bytes in 100 ms requires
at least 69.5 GB/s, versus about 50 GB/s observed in this prototype. This
measurement does not rule out a better kernel, but the current prototype cannot
meet 100 ms end to end. The kernel remains outside the production decoder.

At sampled real Qwen projection inputs, one int8 scale per 32 activation
values had up to 1.36% projection-output relative NRMSE across all 12 positions.
A second int8 residual stream reduced sampled errors to roughly 5e-6–5e-5.
These samples justify a full-model hidden-state and CLM ranking experiment;
they do not establish that integer activation quantization preserves quality.

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
   allocations for both after warmup without changing token IDs.
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
