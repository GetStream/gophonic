# Whisper FP32 matrix kernels

`PackedB` stores a reusable right matrix in panels of 16 columns. `Pack`
accepts either row-major `K × N` data or transposed `N × K` weights, including
padded source strides. Construction owns the packed allocation; subsequent
packing and multiplication allocate no memory. Treat packed model weights as
immutable after initialization. Dynamic attention keys and values can reuse a
preallocated `PackedB` after its previous multiplication has completed.

`PackedB.Mul` runs on the calling goroutine. `Executor.Mul` uses persistent
workers with a configurable limit of 1–64, including the caller. Every shard
owns complete output rows, and boundaries stay on four-row tiles. This keeps
the result bit-identical to `PackedB.Mul` for the selected build. Each operation
has one dispatch and completion barrier. Small operations stay on the caller.
Calls on one executor serialize; call `Close` to release its workers.
Dispatch publishes per-worker atomic generations and uses an atomic completion
counter. Idle workers poll briefly, then back off with reusable sleep timers;
an oversubscribed worker budget yields so all shards can make progress.

`Executor.Rows` shares that same worker pool for row operations such as
softmax and GELU. A reused pointer implementing `ApplyRows(start,end int)`
receives disjoint balanced ranges; a caller-selected minimum rows per worker
controls dispatch overhead. The operation must only write its own rows and
must not re-enter the same executor. It can call single-thread `PackedB.Mul`
to combine matrix and elementwise work inside one worker dispatch.

## SME path (Apple M4 and later)

When the CPU reports `FEAT_SME` with 512-bit streaming vectors, `PackedB.Mul`
and `PackedVector.Mul` run streaming-mode kernels written in Go assembly.
Detection is automatic and needs neither cgo nor `GOEXPERIMENT=simd`. Go's
assembler has no SME mnemonics, so `smesrc/` holds the sources and
`python3 smesrc/build.py` regenerates `sme_arm64.s` with clang.

- `PackedB.Mul` transposes up to 32 rows of A through ZA and accumulates 32x32
  output tiles in the four FP32 ZA tiles with `FMOPA`. It measures about
  1 TFLOP/s on one M4 Max core, against roughly 55 GFLOP/s for NEON.
- `PackedVector` stores N-by-K weights for matrix-vector products. It uses FP16
  storage only when every weight converts to FP16 and back to identical FP32
  bits. Each weight widens exactly before an FP32 fused multiply-add, so FP16
  storage halves memory traffic without changing any result. Accumulators
  live in ZA vector groups.
- Every output starts at +0 and adds products in increasing K order with one
  fused FP32 rounding. SME results are therefore the same for any row blocking
  or worker count, including tail rows and partial panels. `FMOPA` returns the
  default NaN instead of propagating NaN payloads.
- Darwin restores only the low 128 bits of each Z register when a signal
  handler returns to a thread in streaming mode. ZA, predicates, and streaming
  mode survive. Each kernel keeps a sentinel in `z31`. After a tile, and after
  each store phase, it checks the sentinel and recomputes the tile if it was
  cleared. Tests cover this with profiling signals, GC, and oversubscription.
- The M4 Max has one SME unit per performance cluster. Aggregate throughput
  saturates around two concurrent streaming threads.

## SIMD paths

With Go 1.27 and `GOEXPERIMENT=simd` on ARM64, dispatch selects a NEON 2 × 32
kernel for pairs of packed panels, with a 4 × 16 kernel for a remaining full
panel. The wider kernel reuses each source broadcast across more columns and
preserves increasing-K FMA order. Final one-to-three-row tails retain their
existing reduction streams, so executor worker counts keep identical results.
AMD64 v3 builds with `GOEXPERIMENT=simd` select an AVX2/FMA 4 × 16 kernel
with 2 × 16 row tails and four independent reduction streams for one-row
decoder work. AMD64 v1/v2
and other builds use the scalar 4 × 4 kernel. All paths are pure Go. The
SIMD kernel uses fused FP32 multiply-add; its single-row path uses four
independent reduction streams. The scalar path accumulates in increasing K
order and permits compiler-fused FP32 multiply-add. Results therefore need
not be bit-identical between builds, architectures, or BLAS implementations.

The tests compare the public path against an independent FP64 oracle reading
unpacked weights. For finite tested inputs, the accepted absolute error is
`4 * (K+1) * 2^-24 * sum(abs(A[k]*B[k]))`. This forward-error bound includes
cancellation; it is not a blanket promise for underflow, overflow, or unbounded
K. Exact dyadic inputs cover every short row/column/reduction tail. Additional
checks cover NaN/Inf, unaligned slices, source/output padding, shape overflow,
zero dimensions, concurrent reads, worker shutdown, and zero warm allocations.
Input/output aliasing is not supported.

`MulVector` reads ordinary row-major weights directly for incremental decoder
projections. It processes four output rows together and reuses each input
load. The ARM64 path uses two four-lane reduction streams per row; AMD64 v3 uses
two eight-lane streams; the scalar path uses two scalar streams. This avoids
a packed copy of the 80 MB
vocabulary matrix. Its tests use the same independent FP64 error bound and
exercise empty inputs, reduction/output tails, special values, and zero warm
allocations. The decoder's vocabulary shards align to four output rows so
worker count does not change its reduction order.

`BenchmarkMul` measures public single-thread dispatch with prepacked weights.
`BenchmarkExecutor` measures the corresponding worker path, and
`BenchmarkPack` reports packing separately. All benchmark iterations write
observable output. `testdata/accelerate_reference.c` is an optional macOS CPU
SGEMM comparison harness using the same shapes and deterministic inputs; it
is never part of the Go runtime. These matrix timings do not establish full
Whisper transcription speed or parity with whisper.cpp. That comparison must
also match model, audio, FP32 mode, thread count, decoding options, and output.
