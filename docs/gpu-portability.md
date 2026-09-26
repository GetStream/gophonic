# Portable GPU execution

Research date: 2026-09-26. Targets: NVIDIA, AMD, and Apple GPUs.
`CGO_ENABLED=0` is a hard build requirement. Native operating-system GPU
drivers are still required. CGO-free FFI to a native library, including
wgpu-native, is allowed; an external runtime is not the same as CGO.

## Decision

Use `github.com/gogpu/wgpu` v0.34.5 for an explicitly selected portable
compute provider. Keep the existing native Metal provider and its optimized
MSL kernels as the Apple default. Use GoGPU's pure Go implementation for
the initial experiments; the advertised Rust alternative has a reproduced
ABI failure described below. Backend choice must be independent of weight
precision. Replacing the binding alone cannot port Apple's SIMD matrix
instructions or unified-memory assumptions.

This is a migration decision, not a claim that WebGPU is faster than native
Metal, CUDA, or HIP. End-to-end measurements decide which provider should be
used on a given device. No NVIDIA or AMD performance result has been measured
on the local Apple machine.

## Candidates reviewed

| Candidate | Fit for gophonic | Decision |
| --- | --- | --- |
| [gogpu/wgpu](https://github.com/gogpu/wgpu/tree/v0.34.5) | Low-level buffers and compute; pure Go Vulkan/Metal/DX12 implementations without CGO. Lets us retain explicit ownership and custom kernels. | Selected portable API, pinned and isolated behind an internal package. |
| [go-webgpu/webgpu](https://github.com/go-webgpu/webgpu/tree/v0.5.5) | Go FFI bindings to wgpu-native v29. CGO-free builds, but a native shared library is still required at runtime. | CGO-free and eligible in principle. The pinned combination failed our real-device smoke test; qualify a corrected binding before deployment. |
| [GoMLX compute](https://github.com/gomlx/compute) and [go-xla](https://github.com/gomlx/go-xla) | Graph compilation can fuse operations and use vendor libraries, but adopting it would replace the execution graph, packing, and cache strategy. XLA also introduces a native runtime. The compute README currently describes its Darwin provider as experimental and broken. | Reconsider for a separate graph-compiler experiment, not this kernel-preserving migration. |
| [oliverbestmann/webgpu](https://github.com/oliverbestmann/webgpu) | Maintained descendant of the Cogent bindings with prebuilt native libraries distributed as separate platform modules. | Alternative binding/distribution model; CGO-free builds and performance would need independent qualification. |
| [rajveermalviya/go-webgpu](https://github.com/rajveermalviya/go-webgpu) / [cogentcore/webgpu](https://github.com/cogentcore/webgpu) | Original is archived; Cogent redirects users to its maintained fork. | Do not introduce either old module. |

## Dependency smoke tests

On the Apple M4 Max, Go 1.27.1 with `CGO_ENABLED=0`, the pinned pure-Go
implementation successfully created a compute pipeline, submitted WGSL, and
read back the exact 256-element integer reference sum, 32896. This proves
basic local compute execution; it does not establish transformer correctness
or cross-vendor support for our workload.

The advertised alternate stack did not pass: `gogpu/wgpu` v0.34.5,
`go-webgpu/webgpu` v0.5.5, and the official macOS arm64 `wgpu-native` v29.0.0.0
release abort inside `CreateBindGroupLayout` while running the upstream
`examples/compute-sum` example. The native v29 `WGPUBindGroupLayoutEntry`
includes `bindingArraySize`; the pinned Go wire structure omits it, changing
subsequent field offsets. Do not promote `-tags rust` as supported until this
ABI incompatibility is resolved and real compute tests pass. Its runtime
dependency is allowed by the project requirements; the reproduced crash is
the reason it is currently unqualified. The failure is in an isolated
research checkout; gophonic's native Metal code does not use
that binding.

A follow-up isolated experiment repaired the missing layout field and got as
far as dispatch, then reproduced another abort: the wrapper calls
`wgpuBufferGetMapState`, which is unimplemented by this native runtime.
Avoiding that query after a successful map allowed the same CGO-free example
to return the exact reference sum. This demonstrates that the native-runtime
approach is repairable. It does not qualify the published bindings, the rest
of their ABI, or their inference performance. These experimental edits remain
outside gophonic and outside the shared module cache.

## Constraints established from source

The existing `internal/metal` implementation supports no-copy wrapping of
mapped weights, CPU-visible numeric buffers, and MSL SIMD matrix operations.
Those are useful capabilities, not portable buffer contracts. A discrete GPU
must upload immutable weights once and keep activations and KV resident.
Host access must use explicit uploads/readbacks.

The pinned GoGPU [Metal adapter](https://github.com/gogpu/wgpu/blob/v0.34.5/hal/metal/api.go)
and [Vulkan adapter](https://github.com/gogpu/wgpu/blob/v0.34.5/hal/vulkan/api.go)
do not advertise the FP16/subgroup features needed to mechanically translate
our optimized MSL. The portable baseline therefore must work without those
features. Packed 16-bit data can be unpacked into FP32 registers without
expanding every weight into a permanent FP32 tensor. Optional kernels must
request and verify their required capabilities.

[GoGPU's compute backend notes](https://github.com/gogpu/wgpu/blob/v0.34.5/docs/COMPUTE-BACKENDS.md)
document Vulkan memory barriers at compute-pass boundaries and incomplete
Metal timestamp support. Dependent kernels need validated GPU ordering;
ending a compute pass does not require a CPU wait. Its interpreted software
backend is not a throughput target or sufficient evidence for the correctness
of shaders that use workgroup barriers.

WebGPU [FP16 and subgroups](https://gpuweb.github.io/gpuweb/wgsl/#enable-extension)
are optional. [Subgroup matrices](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md)
remain a separate draft proposal. Portable WGSL alone is consequently not a
promise of access to every vendor's tensor/matrix hardware. Native optimized
providers remain valid parts of this architecture.

## Driver and runtime requirements

Apple's Metal driver/framework ships with macOS. NVIDIA and AMD execution
requires the vendor or system Vulkan-capable driver; Windows may instead use
DX12 with the GPU driver. The pure Go Vulkan path also needs the system Vulkan
loader. Using Vulkan does not require the CUDA or ROCm toolkit.

The wgpu-native option would add a bundled native runtime on top of those
same system drivers. It can still be invoked with CGO disabled. No binding
library supplies the hardware driver itself.

## Performance gate

Portability is not permission to regress an existing workload. Before porting
the full graph, benchmark the projection kernels against the existing provider.
Keep native optimized kernels wherever they win. Any replacement of a default
must show an end-to-end improvement at the accepted fidelity, including
transfers and synchronization. A slower portable prototype cannot replace the
current fast path. A kernel win alone is insufficient.

Precision must be explicit in every comparison: an exact BF16-storage provider
and the existing Q8B provider have different memory-bandwidth requirements.
Neither a lower precision result nor a higher precision but slower result
establishes an improvement over the current default.

## Kernel strategy

The provider boundary must permit native kernels, not require every provider
express every operation in the same WGSL source. A portable baseline helps
establish correctness and device coverage; it does not set the performance
ceiling. Specialization is by GPU capabilities, operation shape, precision,
and context length. A kernel enters the default path only after representative
model measurements, including small live-call batches.

- **Decode projections:** stream the existing packed weights directly, reuse
  activation loads across several output rows, and tune rows per subgroup
  against register pressure and occupancy. Measure across a complete layer
  sequence that exceeds GPU cache capacity. Preserve scales and quantization.
- **Prefill projections:** use tiled matrix kernels with FP32 accumulation and
  the model's accepted input precision. Retain Apple's native SIMD matrix
  operations; qualify corresponding Vulkan/DX12 matrix capabilities before
  attempting to replace them with generic scalar arithmetic. No FP16,
  subgroup, or cooperative-matrix support may be assumed from API branding.
- **Attention:** keep K/V private to each lane, fuse normalization, RoPE, and
  KV writes where useful, and use online softmax to avoid materializing the
  entire attention score matrix. Tune parallel reduction to actual context
  length rather than paying maximum-context overhead for every token.
- **Execution:** keep intermediate states on the device. Combine compatible
  epilogues and dependent operations, and read back only when the caller
  needs host-visible outputs. Measure final normalization and head execution
  together with decode, not as excluded postprocessing.
- **Concurrency:** batch compatible work at call boundaries when useful, with
  explicit ownership of each lane's scratch and K/V. Immutable weights can be
  reused across a batch without introducing a shared mutable inference state.

For NVIDIA and AMD, the ratified
[`VK_KHR_cooperative_matrix`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
extension provides device queries for supported matrix shapes and types.
A Vulkan specialization can use that contract when the adapter and binding
actually expose it. Apple's
[Metal tensor operations](https://developer.apple.com/documentation/metal/machine-learning-passes)
are another specialization candidate, not an assumed speedup over the
existing SIMD matrix kernel.

Capability detection must choose valid specializations, including subgroup
width and resource limits. Do not assume a 32-lane subgroup on every vendor.
Prefer a small measured kernel table over unbounded runtime autotuning. Record
compilation/load cost separately so warm throughput does not conceal a large
startup or memory penalty.

## Ownership and execution contract

- Models own immutable device weights, compiled pipelines, and immutable
  metadata. There must be no global mutable inference workspace.
- Each execution lane owns its scratch buffers, parameters, upload/readback
  staging, and command state. Each prefix owns its mutable K/V. Concurrent
  lanes cannot reuse each other's mutable buffers.
- Numeric device allocations are explicitly released. Go may track small
  handle objects; it must not retain a second full heap copy of model weights.
  Host staging should use the existing mapped arena where useful.
- Allocate and grow at setup or capacity boundaries. Reuse bounded arenas and
  stable bindings during inference to reduce allocation and fragmentation.
- Batch dependent kernels into GPU submissions; synchronize only when host
  inputs are needed or host-visible outputs are consumed. Do not insert a
  device wait after each layer.
- Validate limits, arithmetic overflow, requested features, buffer ranges,
  and ownership before submission. Unsupported precision/provider combinations
  return errors instead of silently changing precision.
- Keep the existing CPU and native Metal defaults until real workload results
  justify a change. A portable adapter's existence is not model availability.

## Migration sequence

1. Prove the pinned dependency can compile, dispatch, order dependent compute,
   and read back correct values on actual hardware with CGO disabled.
2. Benchmark exact and existing packed-weight projection paths. Stop a slower
   replacement before broad integration; investigate the measured bottleneck.
   Add a backend selector independent of precision and a private compute layer.
   Keep native Metal available and preserve existing default behavior.
3. Port the complete Qwen3LM path: projections, norms, RoPE, causal attention,
   private prefix KV, batched inputs, external embeddings, and optional logits.
   Connect a public model entry point before treating this as a usable provider.
4. Port Qwen3-ASR's separate audio encoder and validate complete transcription.
   Until then, choosing a portable decoder must not make ASR auto-selection
   choose an unavailable GPU encoder. Qwen3-TTS also has CPU codec stages.
5. Tune by measured bottleneck: tiled projections, fused epilogues, attention
   without a full score matrix, persistent bindings, and fewer submissions.
   Consider vendor matrix kernels only with precision and capability checks.
6. Validate NVIDIA/Vulkan, AMD/Vulkan, and Apple/Metal on actual devices before
   promoting portable execution as a default. Windows DX12 needs its own run.

Whisper currently has no GPU provider in this repository. Adding one is a
separate model integration, not something accomplished by replacing Qwen's
Metal binding.

## Acceptance evidence

Correctness needs independent CPU/reference checks for packed weights, norms,
RoPE, attention, KV append/copy/reset, and logits. Include odd dimensions,
capacity edges, two simultaneously active lanes, empty inputs, unavailable
hardware, failed initialization, and repeated release. Test both synthetic
small models and the official checkpoints. Numerical tolerances must be
stated; matching model semantics does not imply bitwise-identical floating
point reductions across GPU vendors.

Performance reports must separate load/compilation, steady-state prefill,
one-token decode including logits, full PCM-to-transcript latency, and
concurrent-call throughput. Record model, precision, driver, GPU, backend,
context length, batch size, warm Go allocations, and resident host/device
memory. Include transfers and synchronization in end-to-end numbers. A kernel
microbenchmark alone does not establish an inference improvement.

Build checks run normal and SIMD Go builds with `CGO_ENABLED=0`, plus Linux
and Windows compile checks. The optional Rust implementation is excluded.
Tests that require hardware must clearly skip when absent; a compile-only or skipped check is
not a GPU correctness result.
