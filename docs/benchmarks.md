# Benchmarks

Measure the public prediction call with loaded weights and a reused workspace.
Report the model, input boundary, compiler, CPU budget, and allocations together.

## Whisper tiny.en CPU comparison

Whisper transcription is a separate workload from the turn-detector results
below. Use the same official `tiny.en` checkpoint, JFK 16 kHz PCM, FP32
weights, greedy decoding, and CPU budget. Validate the transcript before
comparing latency. The Go benchmark measures a warm `TranscribeInto` call from
PCM to text, including full-file log-mel extraction, encoder, incremental
decoder, and BPE. Loading, construction, first weight packing, file reading,
and command-line JSON output are excluded.

```sh
GOPHONIC_WHISPER_MODEL="$PWD/tiny.en.gophonic" GOMAXPROCS=8 \
  GOEXPERIMENT=simd go test ./whisper -run '^$' \
  -bench '^BenchmarkOfficialTinyENTranscribe$' -benchtime=10x -count=3
```

The encoder-only benchmark receives the same saved `[80,3000]` mel fixture:

```sh
GOPHONIC_WHISPER_MODEL="$PWD/tiny.en.gophonic" GOMAXPROCS=8 \
  GOEXPERIMENT=simd go test ./whisper -run '^$' \
  -bench '^BenchmarkEncoder/workers8$' -benchtime=10x -count=3
```

The pinned CPU-only [whisper.cpp comparison script](../tools/benchmark_whisper_cpp.py)
checks source commit, CMake flags, checkpoint and WAV hashes, and output text.
Its reported `total time` includes model loading; compare warm Go inference
with an explicitly named C++ inference estimate (`total - load`) or use a
matched repeated in-process C++ benchmark. Report encoder and decoder stage
timings separately. Metal, Accelerate, BLAS, and CoreML must be disabled for
the CPU comparison. Run one benchmark process at a time; concurrent builds or
model tests can change results substantially.

### September 23, 2026 M4 Max reference sweep

The [raw Go and C records](benchmarks/whisper/) use Go 1.27.0 with
`GOEXPERIMENT=simd`, the official tiny.en checkpoint, and CPU-only
whisper.cpp at commit `a664346ea5c6dddff3e61a2b7b32dd4514613f50`.
Each Go value is the median of three five-iteration encoder benchmark means;
each C value is the median of three independent CLI runs. The C warm estimate
subtracts its measured model-load time from total time. All Go encoder runs
reported 0 B/op and 0 allocs/op.

| Execution slots | Go encoder, before the September kernel changes | C encoder stage | C total minus load |
| ---: | ---: | ---: | ---: |
| 4 | 319.88 ms | 339.78 ms | 388.15 ms |
| 8 | 211.29 ms | 178.69 ms | 219.85 ms |
| 12 | 202.94 ms | 129.38 ms | 170.31 ms |
| 16 | 221.63 ms | 426.47 ms | 547.60 ms |

The encoder columns are diagnostic, not identical stage boundaries:
whisper.cpp charges some cross-attention key/value preparation to its encoder,
while Go charges that work to `BeginDecode`. The full PCM-to-text result is the
appropriate end-to-end comparison. The 16-thread C slowdown also illustrates
why one worker count cannot represent either runtime.

The retained scalar and SIMD changes have isolated before/after records in
[scalar-ab.txt](benchmarks/whisper/scalar-ab.txt),
[simd-gemm-ab.txt](benchmarks/whisper/simd-gemm-ab.txt), and
[gelu-ab.txt](benchmarks/whisper/gelu-ab.txt). These are paired kernel or
encoder experiments, not an additive estimate of end-to-end speedup. An
end-to-end C win is not yet established; repeat both runtimes in a quiet,
matched window and compare complete warm inference before claiming one.

## TinyMelNet on Apple M4 Max

Recorded September 23, 2026 on macOS 26.3.1, ARM64. Go used **1.27.0** with
`GOEXPERIMENT=simd`. The CPU reference used **ONNX Runtime 1.30.0**, Python
3.14.3, and NumPy 2.5.3. Both execute the
[same pinned TinyMelNet artifact](models.md#convert-tinymelnet).

Each value is the **median of three run means**, with 200 predictions per run.
It is not a per-request p50. The Go columns report **0 B/op and 0 allocs/op**.

| CPU concurrency setting | Go helpers | Go model only | ORT CPU model only | Go PCM → prediction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 6.313 ms | 2.854 ms | 9.987 ms |
| 2 | 1 | 3.371 ms | 2.651 ms | 5.638 ms |
| 4 | 3 | 2.411 ms | 2.641 ms | 4.060 ms |
| 8 | 7 | 2.090 ms | 3.195 ms | 3.553 ms |

Go's setting is `GOMAXPROCS`; its helpers work alongside the caller. ORT's
setting is `intra_op_num_threads`, which includes its calling thread. These
bound the intended operator concurrency; they do not pin work to physical
cores. [ORT thread semantics](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

Go was faster in the four- and eight-thread model-only runs. ORT was faster at
one and two threads. The fastest Go setting used eight execution slots; the
fastest ORT setting used four. These results support a win for this model and
configuration, not a general claim about Go versus C/C++ or other models.

### What the measurements include

**Model only:** both runtimes receive `testdata/tone.mel.f32le`, the same
`[1,80,800]` tensor. Conversion, feature extraction, and weight loading are
excluded. Go includes its public `PredictFeaturesInto` call. ORT includes
Python `session.run` call overhead and output allocation, so this is not a
direct C-API timing.

**PCM → prediction:** Go receives `testdata/tone.pcm.f32le`, eight seconds of
mono 16 kHz audio, and includes audio preparation, feature extraction, and model
execution. File reading, codec decoding, process startup, and JSON encoding are
excluded. ORT audio preprocessing was not measured in this sweep; its model
column is not an end-to-end comparison with the Go PCM column.

Go performs a warm prediction before timing each subcase. ORT performs ten
warmups, uses `CPUExecutionProvider` only, sequential graph execution, all
graph optimizations, and one inter-op thread. Other ORT options keep their
defaults. No GPU or accelerator provider participates.

Raw settings and samples are in the [Go record](benchmarks/m4-max-go-simd.txt)
and [ORT JSON report](benchmarks/m4-max-ort-cpu.json). Thermal state, background
load, and scheduler decisions affect results. The fixed tone workload does not
measure a speech corpus, tail latency, many simultaneous sessions, energy use,
or cold-start cost.

## Reproduce Go results

First [convert the model](models.md#convert-tinymelnet). The full helper sweep
also shows whether smaller pools perform better on your machine:

```sh
GOPHONIC_TEST_TINYMEL_MODEL="$PWD/tinymel.gophonic" \
GOEXPERIMENT=simd go test . -run '^$' \
  -bench '^BenchmarkTinyMelWorkers$' -benchtime=200x -count=3 -cpu=1,2,4,8
```

To measure just the eight-slot, seven-helper configuration:

```sh
GOPHONIC_TEST_TINYMEL_MODEL="$PWD/tinymel.gophonic" \
GOMAXPROCS=8 GOEXPERIMENT=simd go test . -run '^$' \
  -bench '^BenchmarkTinyMelWorkers/(features|audio)/helpers=7$' \
  -benchtime=200x -count=3
```

Keep Go version, experiment flags, fixture, and CPU budget fixed for a before
and after comparison. Run one benchmark process at a time. `ns/op` is elapsed
time per prediction; `B/op` and `allocs/op` describe allocation within the
timed operation.

## Reproduce the ONNX Runtime CPU reference

Use the original TinyMelNet ONNX checkpoint, not the converted Go bundle:

```sh
.venv/bin/python -m pip install numpy onnxruntime
.venv/bin/python docs/benchmark_ort.py model_tinymel_int8.onnx \
  --label "Your CPU model" --output /tmp/gophonic-ort-cpu.json
```

[`benchmark_ort.py`](benchmark_ort.py) verifies the checkpoint SHA-256 and
records runtime versions, feature digest, provider, thread counts, run means,
and the resulting logit/probability. It accepts `--threads`, `--warmups`,
`--batches`, `--iterations`, and `--features`; their defaults reproduce the
method above. Dependencies are confined to the reference environment.

## Smart Turn is a separate performance target

The TinyMelNet numbers do not apply to the larger Smart Turn FP32 graph.
Earlier Go SIMD development runs on the same M4 Max measured roughly
40.65–46.03 ms model-only at `GOMAXPROCS=12`, with zero allocations. An earlier
ORT CPU run measured approximately 9.1 ms with its default pool. Those runs
were not part of the controlled TinyMelNet sweep above, and their thread
settings differ. The current Smart Turn Go path remains slower than that
reference.

```sh
GOPHONIC_TEST_MODEL="$PWD/smart-turn-v3.2.gophonic" \
GOEXPERIMENT=simd go test . -run '^$' \
  -bench '^BenchmarkPredict(Features|Mono16k)$' \
  -benchtime=50x -count=3 -cpu=12
```

`internal/int8probe` measures an experimental quantized Smart Turn matrix
kernel. It is outside the shipped model path and is not evidence of an
end-to-end prediction improvement.
