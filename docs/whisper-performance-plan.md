# Whisper CPU performance plan

September 24, 2026. Target: Apple M4 Max, Go 1.27.0,
`GOEXPERIMENT=simd`, official `tiny.en`, FP32, pure Go with a scalar fallback,
and zero warm allocations at the public transcription boundary.

The user target is **50% lower latency than whisper.cpp**: Go warm median
latency <= 0.5 times the matched C warm median. This is a 2x speedup. Compare
equal CPU budgets and each runtime's best measured CPU configuration.

**The next optimization gate is single-core performance.** The matched
one-worker baseline is **826.660 ms Go versus 435.651 ms optimized C** across
30 timed calls per runtime; Go takes 1.898 times as long. Half that C time is
**217.8 ms**, requiring a 3.80x improvement in Go itself. Improve the hot
kernels before tuning multicore scaling. One-core C sets both `n_threads=1`
and `VECLIB_MAXIMUM_THREADS=1`; actual native-library concurrency remains a
measurement caveat. The [raw alternating samples](benchmarks/whisper/2026-09-24/matched-go-c-1.json)
include a block-bootstrap 95% ratio interval of 1.889–1.955.

The accepted 2×32 GEMM kernel subsequently reduced the uninstrumented
one-worker Go median to **782.065 ms** against **438.468 ms** C in a fresh
alternating 30-call run, a 1.784 ratio. The new half-C target is **219.2 ms**
at that worker budget. [Raw samples](benchmarks/whisper/2026-09-24/matched-go-c-1-wide.json)
and a separate paired kernel ablation distinguish this change from run drift.

Existing matched eight-worker evidence is **210.634 ms Go versus 125.730 ms
optimized C**: Go takes 1.675 times as long. Half that C time is **62.9 ms**,
requiring about a **3.35x improvement in Go itself**. This is a secondary
multicore target, not the next optimization gate. Current evidence does not
establish a credible 50% win under stock Go 1.27 FP32 execution.

At any fixed budget, milestones are parity, <=0.8, <=0.65, and <=0.5 times
matched C latency. Applied to the existing fastest C median, these are
125.7, 100.6, 81.7, and 62.9 ms. At one worker, they are 435.7, 348.5,
283.2, and 217.8 ms. Recompute them whenever the matched workload or best C
configuration changes.

## Reference and measurement caveats

The reference is whisper.cpp at
`a664346ea5c6dddff3e61a2b7b32dd4514613f50`. Measure two CPU configurations:

| Reference | Configuration | What a win establishes |
| --- | --- | --- |
| Native kernels | Metal/CoreML off; Accelerate/BLAS off; native CPU optimizations on | Comparison with the pinned native kernel implementation |
| Optimized CPU | Metal/CoreML off; `GGML_ACCELERATE=ON`, `GGML_BLAS=ON`, `GGML_BLAS_VENDOR=Apple`; native CPU optimizations on | Comparison with the optimized CPU implementation available on this Mac |

Disabling Accelerate/BLAS is useful for kernel diagnosis, but cannot support
an unrestricted claim of beating whisper.cpp's CPU performance. The pinned
`ggml/CMakeLists.txt` defaults Accelerate on and defaults BLAS on with the Apple
vendor on Apple. These are separate paths: `ggml-cpu` uses vDSP vector
operations when Accelerate is enabled, while `ggml-blas/ggml-blas.cpp` calls
`cblas_sgemm` for supported matrices. Confirm which operations actually
dispatch to each backend. Preserve both raw results and use the fastest
quality-matched CPU configuration for the broad 50% claim.

Record CPU backend, compiler flags, tensor precision, and the native math
library's actual concurrency. A BLAS library may manage its own threads;
`n_threads=8` alone does not prove an eight-thread resource budget. Use a
documented library thread control when available and verify its behavior;
otherwise label the native-library CPU budget separately. Accelerate's
internal CPU instruction selection is part of that optimized reference and
must not be assumed to match Go's available NEON instructions.

The pinned C context defaults `flash_attn=true`; the warm harness preserves
that setting. Record it explicitly, including its padded attention lengths.
Its `itype` defaults to FP16 and is used for self/cross KV caches even with
FP32 weights. The C log confirms a 9.44 MB padded cross cache; Go uses FP32
cache elements. Label this reference **FP32 weights with default FP16 KV and
flash attention**, not all-FP32 execution. Keep it as the practical optimized
CPU competitor. A diagnostic with matching intermediate precision is separate
and must not replace the strongest quality-matched reference.

Use the verified FP32 conversion of the same official checkpoint for the
strict comparison. The source/checkpoint pins and numerical contract are in
[whisper-plan.md](whisper-plan.md).

### Formal matched window measurements

Each runtime has 30 timed calls, in alternating blocks with warm calls,
using the same 30-second window and 24-text-token comma transcript:

| Actual inference workers | Go median | Optimized C median | Go/C ratio | 95% block-bootstrap ratio interval |
| --- | ---: | ---: | ---: | ---: |
| 1 | Pending | Pending | Pending | Next optimization gate |
| [8](benchmarks/whisper/2026-09-24/matched-go-c-8.json) | 210.634 ms | 125.730 ms | 1.6753 | 1.636–1.929 |
| [12](benchmarks/whisper/2026-09-24/matched-go-c-12.json) | 247.736 ms | 190.228 ms | 1.3023 | 1.2585–1.3624 |

Eight workers are faster than twelve for both runtimes in these sessions.
The smaller ratio at twelve does not establish a better absolute result.
These are observed deficits; the 50% improvement remains an aspiration.

Earlier exploratory measurements were:

| Measurement | Observed elapsed time |
| --- | ---: |
| Go full-file PCM to text, GOMAXPROCS=8 | 212–245 ms |
| Go full-file PCM to text, GOMAXPROCS=12 | 214–235 ms |
| C CLI total minus load, 8 / 12 threads | about 258 / 227 ms |
| C repeated in-process calls, 8 threads | about 262–286 ms |
| C repeated in-process calls, 12 threads | about 253–273 ms |
| Optimized C with Accelerate/BLAS, FP32 model, 8 threads | about 130 ms, preliminary 10-call result |
| Optimized C with Accelerate/BLAS, FP32 model, 12 threads | about 202 ms, preliminary result |
| Optimized C, audited FP16-storage model, 8 threads | 135.33 ms median; 120.07–165.34 ms range; 12 timed calls after 5 warm calls |
| Go matched window, QKV experiment baseline / candidate | 218.56 / 215.91 ms, 32 pairs; candidate reverted |
| Go encoder, separate stage measurement | about 145 ms |
| Go complete official greedy decoder | 58.65 ms |
| Go `BeginDecode`, separately measured | 11.18 ms |
| Go token loop, separately measured | 45.31 ms |

These ranges do **not** establish a matched end-to-end lead. They contain
different invocation boundaries, run conditions, and potentially token counts.
Separately timed stages need not sum to the complete call.

The optimized C runs set both `n_threads` and `VECLIB_MAXIMUM_THREADS` to the
reported budget. That records the requested library limit; it does not prove
the actual number of threads scheduled internally. Both optimized models
produced the exact comma transcript in their warm harness. The preliminary
FP16-storage run did not outperform the subsequent formal 125.730 ms FP32
result. It does not lower the 62.9 ms secondary target.

The smaller C model was audited against the Go bundle tensor by tensor:
all 167 tensors match FP32 bits after conversion. It contains 67 FP16 tensors
and 100 FP32 tensors; the token embedding is FP16, positional embeddings and
biases are FP32. Only convolution-bias shape notation differs (`[384,1]`
versus `[384]`). Every Go tensor also round-trips through FP16 back to FP32
bit exactly. The model is 77,704,715 bytes with SHA-256
`921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
This proves weight identity, not identity of the C runtime's activation
precision or floating-point reduction order.

The [September 24 benchmark records](benchmarks/whisper/2026-09-24/)
include the tensor audit, optimized C warm samples, matched encoder stage
breakdown, and attention SIMD comparisons. Temporary scratch paths in the
raw records identify how each experiment was run, not durable artifacts.

Two issues were found in the initial comparison setup:

1. `NewTranscriber` used `NewEncoderWorkspace`, which caps the shared
   inference pool at eight workers. Setting GOMAXPROCS=12 therefore did not
   measure twelve inference workers. The warm helper now selects an explicit
   worker budget through `NewTranscriberWithWorkers`; historical measurements
   made through the default constructor retain the eight-worker caveat.
2. Full-file and PCM-padded window transcription are different workloads.
   The official full-file JFK fixture contains 23 text tokens and no comma
   after “you”; the 30-second padded window fixture contains 24 text tokens
   and the comma. The Go warm helper now uses `TranscribeWindowInto` for the
   window comparison. Preserve both oracle contracts and report actual token
   IDs, including the prompt and EOS, before accepting a timing comparison.

The older September 23 sweep in [benchmarks.md](benchmarks.md) is historical
evidence, not the baseline for this plan. A CLI `total - load` estimate is a
diagnostic; repeated in-process wall time is the primary acceptance metric.

## Where the time can go

### Single-core diagnosis: current first priority

Separate instrumented one-worker probes reported:

| Work | Go median | Optimized C median |
| --- | ---: | ---: |
| Complete call | 827.68 ms | 434.03 ms |
| Frontend | 10.61 ms | 13.56 ms |
| Encoder | 668.47 ms | Included below |
| Cross-KV preparation | 57.34 ms | Encoder plus cross-KV: 330.22 ms |
| Token work | 90.55 ms | Decode 75.50 + batched prefix 6.28 + sampling 7.95 ms |

These are diagnostic probes with different stage instrumentation; the
uninstrumented ABBA result is recorded above. Go encoder plus cross-KV takes
about 725.8 ms: over twice C's corresponding 330.2 ms. Go's profile assigns
72.02% of full-call CPU samples directly to `kernel4x16`. The first work is
therefore actual-shape one-core GEMM instruction/throughput analysis, followed
by attention softmax, norms, and activation passes. Decoder-only gains cannot
close this encoder deficit.

Half the provisional C probe is about 217 ms, requiring about 3.8x faster Go.
An illustrative Amdahl calculation that treats the 72% flat sample share as
wall time leaves about 232 ms even if that cost vanishes. CPU samples are not
a precise latency partition, but this warns against a kernel-only 50% plan:
the remaining work also needs material reductions. Recalculate with the
formal one-core baseline and stage ablations.

### Existing multicore diagnosis

The encoder is the largest opportunity. From the fixed tensor dimensions,
its dense multiplies require approximately **36.94 GFLOP** per window:

| Encoder work | Dense arithmetic per window |
| --- | ---: |
| Two stem convolutions | 1.88 GFLOP |
| Q/K/V/output projections across four blocks | 7.08 GFLOP |
| MLP expansion and contraction | 14.16 GFLOP |
| Attention QK and probability/value products | 13.82 GFLOP |

This counts a multiply-add as two operations and excludes elementwise work.
At 145 ms, complete encoder throughput is about 255 GFLOP/s. An encoder
target of 100–110 ms would require approximately 336–369 GFLOP/s including
its other work. Cross-attention cache preparation adds about 3.54 GFLOP.

Improving a 145 ms encoder by 25% saves about 36 ms. Even a 2x improvement
to an 11 ms cache-preparation stage saves only about 5.5 ms. The QKV
and cross-value experiments therefore cannot deliver the 50% goal by
themselves. Current experiment outcomes are:

| Experiment | Outcome | Consequence |
| --- | --- | --- |
| Dynamic worker scheduling | Rejected; no useful reduction in measured tail imbalance | Do not spend another optimization cycle here without new wall-time evidence |
| Direct transposed cross-value projection | Rejected by the current experiment | Do not include its hoped-for gain in a budget |
| Fused encoder QKV | About 1.2% measured benefit; reverted | Insufficient benefit for its complexity or the architectural improvement needed for the target |

These outcomes were reported by the concurrent implementation experiments;
retain their raw records with the final performance report.

A later stage probe measured frontend 11.07 ms, encoder 156.65 ms,
cross-KV preparation 15.30 ms, and token decoding 55.11 ms. Within the encoder,
attention took 61.54 ms and the MLP took 53.52 ms. Those two stages are the
highest-priority compute targets. This separately instrumented probe is not
an additive reconstruction of the 218.56 ms window median.

The decoder has a different constraint. Every vocabulary projection reads
51,864 x 384 FP32 weights, or **79.7 MB**, and performs about 39.8 MFLOP.
Its weight arithmetic intensity is only 0.5 FLOP/byte. The 24-text-token
window requires 26 decoder positions with the current two-token prefix,
so the vocabulary alone streams about 2.07 GB of logical weight data.
Measure its sustainable read throughput separately: dividing those bytes by
measured throughput gives a useful bandwidth floor, not a hardware guarantee.
More FMA unrolling cannot remove that traffic under the FP32 contract.

The current worker experiment found little tail imbalance and rejected dynamic
scheduling. CPU-profile samples in `pthread_cond_wait` do not quantify removable
wall latency. Use stage wall times and worker completion times when deciding
whether to revisit scheduling.

### A throughput check for the secondary 62.9 ms target

An illustrative budget of 5 ms frontend/text work, 4 ms cross-KV preparation,
and 20 ms token decoding leaves **33.9 ms for the encoder**. That requires
about **1.09 TFLOP/s** over its existing 36.94 GFLOP, including its other work:
approximately 4.3–4.6x the observed complete encoder rate. The 4 ms cross-KV
budget independently requires about 885 GFLOP/s. These are required rates,
not measured hardware capability or promised stage latencies.

First measure sustained throughput for the actual matrix shapes, a resident
vector-FMA probe, and an 80 MB streaming GEMV probe on one core. For each
stage, compare arithmetic/sustained compute and bytes/sustained
bandwidth, then add unavoidable serial work. Do not multiply a marketing peak
or single-core kernel number by twelve and call it an achievable encoder rate.
If the measured ceiling leaves no margin for softmax and memory passes, the
strict path needs less work or a toolchain advance; additional fusions cannot
make the 62.9 ms budget real. Do not claim the 50% roadmap is feasible until a
measured stage budget fits below the final deadline with room for variation.
Repeat the ceiling analysis across worker budgets only after the single-core
implementation is competitive; the stage measurements above are multicore
diagnostics and do not predict single-core stage shares.

## Ranked experiments

The first four experiments run at one worker. Single-core stage timing must
replace the earlier multicore experiment budgets before estimating savings.
Those earlier budgets are retained only for context; they are neither
forecasts nor additive gains. Remove candidates without a repeatable benefit
at the complete public boundary and unchanged correctness.

| Rank | Concrete change | Experiment budget | Main risk |
| --- | --- | ---: | --- |
| 1 | Tune encoder GEMM geometry and cache blocking for actual shapes | Earlier multicore budget: 15–30 ms | Vector spills, extra partial-output traffic, regression of another shape |
| 2 | Extend row-local fusion through the encoder MLP | Earlier multicore budget: 5–15 ms | Smaller row chunks may reread more weights |
| 3 | Tune attention score tiles and softmax passes | Earlier multicore budget: 4–10 ms | Summation changes can affect stage tolerance |
| 4 | Improve decoder GEMV reuse and test exact compact weight storage | Profile first; no justified latency budget yet | Conversion or register pressure can exceed memory savings |
| Independent | Skip known-zero frontend tail computation exactly | Earlier multicore budget: 3–7 ms | Boundary and normalization mistakes |
| After single-core gates | Tune row ownership and projection sharding across cores | Earlier multicore budget: 4–12 ms | Dispatch overhead and bandwidth saturation |

### Encoder row locality

After attention completes, every row can independently execute:
output projection, bias, residual addition, MLP LayerNorm, expansion,
bias/GELU, contraction, bias, and residual addition. Submit this chain through
one persistent row operation. Compare whole-worker row shards against chunks
of 16, 32, and 64 rows, preserving four-row GEMM boundaries.

At one worker, this keeps intermediate MLP values closer to their consumer.
Afterward it can parallelize normalization/bias/residual passes and reduce
worker handoffs. Synchronization savings alone are not the hypothesis.
Retain the existing per-row arithmetic order, including bias before residual
addition. Start with the current kernels so the fusion ablation is independent
of kernel changes. Apply the same reasoning to convolution bias/GELU after
the transformer experiment proves useful.

Keep the small QKV result separately attributable. Do not reintroduce the
rejected direct cross-value orientation as part of a combined change that
conceals its measured cost.

### Encoder matrix kernels: first priority

`PackedB` currently packs 16 output columns and `mulPacked` traverses a whole
row shard for each panel. `kernel4x16` uses 16 accumulator vectors and a table
broadcast for each input row and reduction value. The inspected baseline
binary emits vector FMA and table lookups without vector spills in that loop;
the table-index literals are synthesized at every microkernel entry.

Prioritize MLP contraction (`K=1536`) and attention values (`K=1500`), then
MLP expansion and `K=64` attention scores. A long-reduction weight panel alone
occupies about 96 KB. Compare:

- M/K blocking with reusable scratch and explicit accumulate kernels. Keep
  one increasing-K FMA chain per output across blocks; avoid split-K reductions.
- Shape-selected 4x16 versus 8x8 or other register-feasible tiles. Inspect the
  complete emitted loop for spills, bounds checks, loads, and helper calls.
- Amortized/static table-index setup, especially for the short `K=64` kernel.

Measure each real shape and complete encoding. A better GEMM microbenchmark
does not justify a slower attention tile or full window. Preserve the scalar
fallback and validate partial rows, the 1500-column tail, and padded strides.

### Attention scratch traffic

Audio attention already fuses QK, softmax, and the value product per query
tile; it does not materialize a complete score matrix. Its current 32x1500
score tile occupies 192 KB per worker. Compare 16-row and 8-row tiles, which
use 96 KB and 48 KB, while retaining the same complete attention operation
within each worker dispatch.

First vectorize the maximum and final probability scaling while preserving
the current increasing-index sum. Next consider alternate sum reductions
only as a distinct numerical experiment with all existing tolerances intact.
Online softmax is a later experiment: its benefits must exceed the existing
tiling, and its changed accumulation requires independent stage validation.

### Decoder kernels first, projection sharding later

Profile the vocabulary projection, four-block projection GEMVs, attention,
and token selection independently at one worker. Optimize the live GEMV's
weight reads, reduction chains, output tile, and exact compact-storage option
before introducing more workers. Measure complete token sequences as well
as the relevant matrix shapes.

Vocabulary projection is already sharded, and long cross-attention uses up to
six heads concurrently. Most self/cross output projections and MLP GEMVs still
run on the caller. In the later scaling phase, fuse self Q/K/V into one row
operation; test two or four workers for the larger MLP projections with
bias/GELU in their owner shards.
Keep short operations serial when dispatch costs more than it saves.

Preserve each output's reduction order and four-output vocabulary alignment.
Measure complete token sequences, not only one token or one hot GEMV. Do not
expect the six-head attention operation to scale to twelve independent heads.
Long-prompt prefill can later use batched matrix operations; it offers limited
benefit for this two-token JFK prefix and must preserve SOT no-speech logits.

### Exact zero-tail frontend work

Both frontends compute FFT/mel/log values for right-padded PCM that is known
to be zero. A completely zero STFT window produces a pre-normalization log-mel
value of exactly -10. Fill that tail and retain the same global maximum,
max-minus-eight floor, and final scaling.

For the window path, let `used=min(len(pcm),480000)`; the first guaranteed
zero frame is `ceil((used+200)/160)`, clamped to 3000. The full-file path uses
the real PCM length and its own output frame count. Preserve the boundary
frames and reflect padding. For 11-second JFK this removes expensive work
from about 63% of the window's STFT frames, or 73% of the full-file frames;
it does not remove the final normalization or all frontend cost.

Test lengths around hop boundaries, short signals, end impulses, silence,
30-second input, and workspace reuse after longer audio. Keep the full
1500-frame encoder context: positional embeddings and attention make padded
encoder positions meaningful even when their source audio was zero.

## Ablation order and decision gates

1. Freeze both C configurations and run the formal matched **one-worker**
   Go/C baseline. Record native-library thread controls and effective CPU use.
2. Profile frontend, encoder projections/MLP/attention, cross-KV, decoder
   projections/vocabulary/selection at one worker. Measure actual-shape
   compute and streaming-memory ceilings; identify the largest avoidable cost.
3. Tune GEMM/GEMV kernels one shape at a time. Then evaluate encoder row
   locality and attention scratch/softmax. Keep zero-tail frontend work an
   independent ablation. Preserve operation order where the strict gate needs it.
4. Recombine retained candidates and require repeatable full-window one-core
   gains. Evaluate parity, <=0.8, <=0.65, and <=0.5 against one-core optimized C.
5. Pursue the architectural paths below when the measured ceiling shows
   kernel changes cannot reach the target. Keep their work/quality gates explicit.
6. Only then measure 2/4/8/12 workers. Report speedup `T1/Tn`, efficiency
   `T1/(n*Tn)`, and per-stage scaling; investigate the first lost scaling step.
   Tune ownership/sharding where those measurements justify it. Linear scaling
   is a goal to test, not an assumption for bandwidth-bound work.

Retain the existing 8/12-worker records as secondary evidence. They must not
replace the single-core optimization gate. Revert candidates whose public-call
benefit disappears, and preserve unsuccessful variants to avoid repeating them.

## Architectural experiments for the remaining gap

### Strict FP32 model and existing numerical gates

These experiments retain the official graph and weights, FP32 arithmetic,
existing stage tolerances, and exact accepted token sequences. They are
research paths, not an assurance that their combined gains reach 62.9 ms.

| Path | Work it can remove | Bound or uncertainty |
| --- | --- | --- |
| Exact compact weight storage | Use the audited lossless FP16 representation for streaming vocabulary/decoder weights, converting exactly to FP32 immediately before FP32 arithmetic | Halves those weight bytes, not decoder time; conversion cost is unmeasured |
| Constant-tail convolution | Compute the repeated stem output of constant padded mel regions once and fill their interior | Even eliminating both convolutions entirely removes only 1.88 GFLOP, 5.1% of encoder dense arithmetic |
| Blocked or online attention | Keep QK, softmax statistics, and value accumulation in cache; reduce score traffic and serial elementwise passes | Current code is already tiled; quadratic QK/PV arithmetic remains, and reordered sums must pass existing gates |
| FFN cache blocking and restricted fast matrix multiplication | Reuse weights/activations across larger blocks; optionally test one or two Strassen levels on the large MLP shapes with immutable prepacked weight combinations | One Strassen level reduces MLP multiply arithmetic by 12.5%, only about 4.8% of total encoder dense arithmetic before extra sums; cancellation/overhead may reject it |
| Exact greedy speculative verification | Batch draft token candidates through the target model, commit only the verified matching prefix, and reuse weight reads | Needs high held-out acceptance and cheap draft work; cache rollback, masks, EOS, and floating reduction behavior require proof |
| Prefix prefill | Batch long prior-text prompts and omit unused vocabulary projections while preserving required SOT no-speech logits | Important for later windows; small benefit for the two-token JFK prefix |

Lossless half storage is more concrete than ordinary quantization here:
the per-tensor audit found zero differing bits after FP16-to-FP32 round trips.
Start with the 79.7 MB vocabulary matrix. Keep encoder packed weights FP32
when repeated on-the-fly conversion would cost more than memory traffic.
The installed `archsimd` API has no native FP16 vector conversion operation;
test an exact integer-bit SIMD conversion against every half bit pattern
and the real finite weights, including signed zero and subnormals. Require
the existing FP32 operation order after conversion. Do not confuse smaller
storage with permission to round activations to half precision.

Constant-tail stem reuse must retain boundary convolution rows, final reflect
padding, bias/GELU order, and positional embeddings. The optimization ends
before position-dependent transformer work. Truncating padded encoder
positions changes attention keys/values and the model's output; it is not an
exact variable-length implementation of the standard 1500-frame encoder.
Likewise, reusing an overlapping encoder segment is not generally exact when
global attention or its absolute positions change.

### Exact speculative decoder verification

The current decoder has no cheap draft predictor. Begin with blocks of four
positions: one known target token followed by three proposed continuations.
Process the block layer by layer, sharing weight loads across positions;
commit only the prefix accepted by the target model. An n-gram/prompt lookup
can provide a cheap first draft, but its acceptance on unseen speech is unknown.

Required edits are a batched decoder entry point, preallocated block-sized
activations/logits, and batched GEMV kernels. Each row's causal attention must
stop at its own position. Verify using the actual accepted history and all
existing token masks, tie rules, EOS/sample limits, and SOT no-speech handling.
After rejection, reset `nextPos` to the accepted prefix; future tentative KV
slots must be overwritten before use. Do not advance text or segment history
using rejected candidates.

Ordinary packed GEMM changes the present GEMV reduction order. Verifying
against those changed logits does not prove equivalence to the original
decoder. A strict implementation must retain each position's existing
reduction chains and final sum while sharing weight loads across positions,
or independently establish the original token/stage contract on the corpus.
Test forced rejection at every proposal position, EOS inside a block, cache
rollback, history filters, and zero warm allocations.

If each guessed token has acceptance probability `q`, a four-position batch
processes about `A=1+q+q²+q³` useful positions: 1.875 at q=0.5, 2.952 at q=0.8.
Break-even requires `(verify4 + draft + rollback)/A < serial1`. Even ideal
weight sharing yields only about 1.85x at q=0.8 if 20% of baseline token time
is per-position compute: `2.952/(1+3*0.2)`, before draft costs. Measure these
quantities; transcript memorization or repeated-window result caching is invalid.
Speculation alone cannot reach the secondary 62.9 ms target while the measured
full-context encoder takes about 157 ms.

### Optional numerical or model changes

These are separate modes with explicit quality evaluation. They cannot pass
the strict FP32 milestone by relaxing its tests, and their results must be
labeled with their own precision and context settings. Apply the same allowed
accuracy/performance choices to the C competitor.

- **Weight-only int8 or int4:** Most promising initially for vocabulary
  bandwidth. Int8 reduces original FP32 weight bytes by 4x, but by only 2x
  relative to lossless FP16 storage already possible here. Dequantization,
  scales, and outlier handling must be included in live timing. Use FP32
  accumulation first; promote to more aggressive arithmetic only after a
  separate accuracy result.
- **Mixed activation precision:** Quantized projections may reduce matrix
  traffic or arithmetic. The installed `archsimd` surface lacks direct
  SDOT/I8MM/FP16 matrix APIs; widening arithmetic or conversions can consume
  the apparent gain. Do not assume C's quantized kernel speed is expressible
  by this Go API. Keep norms, softmax, and sensitive output stages FP32 unless
  measured quality supports another choice.
  A separate FP16 KV-cache experiment can match C's existing intermediate
  precision more closely, but computed KV values do not share the checkpoint's
  exact half round-trip property and must pass the optional quality gate.
- **Variable audio context or attention approximations:** See the explicit
  context experiment below. Removing padded encoder positions changes the
  graph; VAD segmentation additionally changes boundaries and may remove speech.
- **Distillation or low-rank FFN replacements:** These change the model and
  require their own checkpoint identity, quality report, and comparison. They
  do not establish that the original tiny.en FP32 implementation beat C.

If measured NEON throughput and accepted exact algorithms cannot fit the
62.9 ms budget, the remaining hardware route is a compiler/runtime capability
that exposes CPU matrix instructions. Stock Go 1.27 `archsimd` does not
provide that interface. A compiler fork or future toolchain would be a change
to the toolchain contract, and is not included in a claim about stock Go 1.27.

### Adaptive audio context: substantial work reduction, separate quality mode

Let `L` be encoded audio positions and `r=L/1500`. Dense encoder work is about
`23.114*r + 13.824*r²` GFLOP. JFK's 176,000 samples require at least **L=551**
to retain every STFT frame containing real audio: `ceil((samples+200)/320)`.
Using 550 drops boundary data. A one-second guard gives L=601, or 608 rounded
to a 16-row bucket. Guards and bucket choices must be fixed before held-out tests.
These lengths preserve source coverage, not full-context model equivalence:
removed padding positions have biases/position embeddings and nonzero keys/values.

At L=551, dense encoder work drops from 36.94 to **10.36 GFLOP**. For the
single-core probe, an optimistic model with all attention scaling quadratically
gives `encoder ≈ 407*r + 261*r² ≈ 185 ms`, cross-KV about 21 ms, and token
decoding about 71 ms if only its measured 31 ms cross-attention component
shrinks linearly. Including the frontend gives about **288 ms**, above even
half the full-context C probe. C's same-L latency has not been measured and
is expected to fall as well; 217 ms would no longer be its 50% target.

For comparison with older multicore evidence, the analogous hypothesis is
`encoder ≈ 95.11*r + 61.54*r² ≈ 43.2 ms`, cross-KV about 5.6 ms, and much of
the 55.1 ms token loop unchanged. For a hypothetical 20–30% cross-attention
share of token time, total latency is roughly **98–115 ms** with frontend
variation. Both calculations are optimistic scaling estimates, not measurements
or bounds. Adaptive context alone establishes neither the one-core goal nor a
62.9 ms multicore path. Even an absolute sub-63 ms candidate needs additional
encoder, decoder, and frontend gains and a faster matched C comparison.

Concrete implementation scope:

1. Keep the default 1500-position API and immutable checkpoint unchanged.
   Add an explicit optional context policy and a private encoder operation
   taking live row count plus mel stride; preserve full-feature normalization.
2. Replace live `MelFrames`/`AudioFrames` loop and GEMM dimensions in
   `encoder.go` with `2*L`/`L`, and slice the first L positional embeddings.
   Existing convolution lowering already accepts row counts. Preallocate
   maximum scratch once; do not allocate when the live length changes.
3. Give `audioAttention` and its packed key/value matrices checked live shapes
   within their maximum capacity, clearing padding on length changes.
   `BeginDecode` already accepts 1–1500 encoder rows and tracks `audioFrames`;
   retain its fixed storage strides while changing only live lengths.
4. Start with single utterances and the last short tail of a long file. Retain
   30-second contexts for full windows until a separate segmentation design
   proves complete coverage, history handling, and boundary quality.

**Fair comparison:** pinned C exposes `params.audio_ctx` / `--audio-ctx` and
uses it for convolution, encoder, cross-KV, and decoder context. Give C the
same L, guard, frontend normalization, and input coverage; record default
flash attention's 256-position padding. Recompute the 50% target from the
fastest accepted C result. A Go short-context result versus C's 30-second
context cannot establish an implementation speedup at matched work.

**Long-file seek trap:** C's no-timestamp/single-segment path advances by
30 seconds even when `audio_ctx` is shorter. Setting `-ac 551` on a long file
can therefore skip unencoded speech. Use explicitly bounded chunks with
matching advancement/history, or retain full contexts until the short tail.
Verify coverage independently of whether either transcript happens to look right.

## Accuracy and workload corpus

JFK remains a useful regression fixture, not the optimization dataset or sole
quality gate. Freeze a licensed, hashed manifest before tuning: clean and noisy
English speech, varied accents and speaking rates, quiet speech, silence,
music/background sound, and utterances near word/window boundaries. Cover
1–5 s, 5–15 s, 15–30 s, and multiple-window recordings. Use separate tuning
and held-out subsets; include enough held-out speech for stable WER and slice
comparisons, with an initial target of at least five hours.

Predeclare at least 100 utterances in each short-duration bucket and at least
30 multiwindow recordings. Include final consonants, quiet speech tails,
sample lengths just either side of STFT hops, 29.9/30.1-second inputs, and
words crossing proposed segment boundaries. Check full input coverage and
count omissions/duplications within 250 ms of each boundary. These are corpus
requirements for the proposed experiment; the corpus is not yet assembled.

For strict changes, compare exact token sequences/transcripts against the
pinned FP32 oracle, retain existing stage/logit tolerances, and locate the
first divergent stage or token. Evaluate the baseline on the same corpus
first so pre-existing divergences remain visible. Do not grandfather a new
failure because aggregate WER happens to stay level.

For optional numerical or adaptive-context modes, freeze acceptance thresholds
before tuning. A proposed starting gate is no more than +0.2 absolute WER percentage points
overall and +0.5 points on any predefined speech slice, no added hallucinated
speech in the silence suite, and explicit review of names, numbers, omissions,
and segment-boundary errors. Include paired-bootstrap uncertainty; a point
estimate alone must not hide a likely regression. Require no skipped input,
empty/invalid segments, or new boundary truncation. These are proposed quality
criteria, not existing user-approved error allowances. Keep the strict path available.

Report latency distribution and real-time factor by duration and token-count
bucket, time to first token, peak memory, and warm allocations. For a broad
50% claim, require the predeclared primary workload aggregate to meet the ratio
gate and show every slice, rather than selecting only a short or silent input.

## Exact benchmark and correctness gates

- **Identity:** Record Git revisions and dirty diffs, Go version/experiment,
  C compiler/CMake settings, actual CPU backend, flash attention, CPU model,
  model/audio hashes, GOMAXPROCS, and actual inference workers. Use identical FP32
  weights and the same 176,000-sample JFK PCM.
- **Work:** For the primary comparison, both runtimes must process equivalent
  30-second padded features, greedy English decoding, no timestamps, no
  temperature fallback, and the same 24 text tokens plus matching prompt/EOS
  work. Validate IDs as well as text. Text equality alone is insufficient to
  prove equivalent frontend or decoder work. Retain full-file measurements
  as their own benchmark with their own oracle.
- **Boundary:** Time a reused in-process PCM-to-text call. Include frontend,
  dynamic packing, encoder, cross-KV preparation, token generation, and text
  decoding. Exclude model loading, first weight packing, construction, file
  reading, and printing. C charges some cross-KV work to encoder time; report
  equivalent operations when comparing individual stages.
- **Sampling:** Build first, then run one timing process at a time with agent
  benchmarks and model tests paused. Use at least five untimed warm calls,
  then alternating Go/C blocks of five calls, at least thirty timed calls per
  contender per worker budget. Alternate block order. Save every sample;
  report per-call p50/p95 and a 95% confidence interval for the latency ratio.
  Repeat a separate session to detect drift. Current helpers exclude two
  warm calls; exclude three additional calls before using this protocol.
- **Budgets:** Start at one actual inference worker, GOMAXPROCS=1, C
  `n_threads=1`, and `VECLIB_MAXIMUM_THREADS=1`. Check actual native-library
  CPU use and label any unresolved concurrency. After single-core gains,
  measure 2/4/8/12 workers and report speedup/efficiency, equal-budget ratios,
  and best-Go versus best-C. Increasing GOMAXPROCS alone is not proof that
  the Go worker pool grew; selecting a slow C budget cannot establish a win.
- **Acceptance:** Final median Go/C latency ratio and its upper 95% confidence
  bound must both be <=0.5 at the declared matched budget; the broad CPU
  claim must also use the fastest quality-matched CPU reference. Apply the
  milestones first at one worker. Use <=0.8 and <=0.65 as intermediate gates.
  Investigate a p95 regression even if the median passes. A 62.9 ms JFK result
  alone is not proof of the same speedup across speech durations or token counts.
- **Correctness:** Run independent matrix/vector oracles and existing
  frontend, encoder-stage, prefix/subsequent-logit, greedy-token, full-file,
  silence, and repeated-call checks. Preserve encoder tolerance
  `2e-4 + 2e-4*abs(reference)` and every existing decoder/softmax threshold.
  No oracle edits or threshold relaxation to admit an optimization.
- **Allocations and builds:** Require zero warm bytes and objects at the
  public window and full-file boundaries with sufficient caller capacity.
  Test SIMD and scalar builds, worker counts, tails, and the supported build
  matrix. Use race tests for changes to shared worker state. Keep counters
  and timers out of production paths when disabled.

All Go build/test/tool commands use the project cache wrapper. For example:

```sh
env CODEX_AGENT_ID=whisper-perf \
  CODEX_PROJECT_CACHE_ROOT=/private/tmp/goinfer-build-cache \
  GOPHONIC_WHISPER_MODEL="$PWD/tiny.en.gophonic" \
  GOMAXPROCS=1 GOEXPERIMENT=simd \
  /Users/thesyncim/.codex/bin/project-env \
  go test ./internal/whispergemm ./whisper
```

Use the actual converted model path. Check `go env GOCACHE` through the same
wrapper before a large build; it must derive from the Git common directory
and must not contain `.codex/worktrees`. Repeat relevant correctness gates
with the SIMD experiment disabled. The existing `BenchmarkOfficialTinyENWindow`
and the repeated warm helpers cover the intended public window boundary;
ensure explicit worker selection for a twelve-worker run.

## Hardware and scope limits

The installed Go 1.27 ARM64 `archsimd` surface exposes 128-bit NEON operations.
It has vector `MulAdd`, but no direct public lane-FMA, SVE/SME matrix,
Apple AMX, or prefetch API. Compiler pattern matching may improve instruction
selection, but a plan cannot assume access to those facilities. Current
disassembly uses table broadcasts and vector FMA. Wider register tiles also
compete for a finite register file, so larger source-level tiles can regress.

Compiler intrinsics that express missing hardware instructions are a separate
toolchain research path. They are not a promised gain for the stock Go 1.27
implementation. Lossless FP16 weight storage can stay within the strict FP32
contract only with exact conversion and unchanged FP32 arithmetic. Quantized
weights, reduced activation precision, or shorter attention context require
the separate quality/performance contract described above. Speculative
verification must prove the exact target-model generation behavior before
it can enter the strict path.

For throughput across many requests, shared immutable packing and batching
vocabulary projections may amortize weight traffic. Measure that separately
from single-request latency: the current acceptance target cannot be satisfied
by a higher batch throughput number.
