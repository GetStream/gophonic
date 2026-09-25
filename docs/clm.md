# CLM CPU projection heads

`github.com/GetStream/gophonic/clm` runs the released CLM state/action heads in
pure Go. It converts the PyTorch checkpoint once, then loads a versioned
`.gclm` bundle without Python, cgo, or an inference runtime.

The released `CLM-v0.1-8B` head maps 4096-value Qwen3-8B pooled embeddings
through separate state and action MLPs to 512-value L2-normalized projections.
The model score is `min(exp(logit_scale), 100) * cosine(state, action)`; a
softmax over the candidates produces ranking probabilities. The Go package
implements the head and ranking primitive. The text encoder remains a separate
component: this head was trained against Qwen3-8B's post-final-normalization
last-token embedding, so another encoder can load the weights but will not
reproduce CLM scores.

## Convert the reference checkpoint

Download the checkpoint to a local file, then convert it offline. The
`--expected-sha256` option rejects a checkpoint whose contents do not match the
pinned hash. The converter uses `torch.load(weights_only=True)`, validates every
head tensor name and shape, records per-tensor checksums, and preserves the
source revision and SHA-256 in the output manifest.

```sh
hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --revision 87655cb835bd76fd66c2da78e1e3709f7fa11a94
python3 -m pip install torch numpy
python3 tools/clm_pt_to_gophonic.py CLM_v0.1-8B.pt CLM_v0.1-8B.gclm \
  --expected-sha256 b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5 \
  --source-revision 87655cb835bd76fd66c2da78e1e3709f7fa11a94
```

The converter does not fetch files. Use a separately verified SHA-256 for a
fine-tuned checkpoint, and set `--source-repo`, `--source-revision`, and
`--source-file` to match its origin. The raw checkpoint is not required at
inference time.

## Use embeddings directly

The head API accepts raw 4096-value encoder vectors. It L2-normalizes each
embedding before its MLP, then normalizes each 512-value projection, matching
the reference CLM embedder and heads.

```go
head, err := clm.Load("CLM_v0.1-8B.gclm")
if err != nil {
    return err
}
workspace := head.NewWorkspace() // one workspace per concurrent lane
scores := make([]float32, len(actions))
if err := head.ScoreInto(stateEmbedding, actionEmbeddings, 1, scores, workspace); err != nil {
    return err
}
```

`ScoreInto` returns scaled cosine logits divided by the supplied temperature.
All candidates pass through each head layer as one batched matrix product
(SME FP32 on Apple M4), so the 75 MB head is read once per call rather than
once per candidate.
`Engine.RankInto` also accepts an `Embedder` implementation, calls it once for
the state and once for the action batch, then returns candidates sorted by
softmax probability. The callback receives caller-owned output buffers, and
`RankInto` reuses workspace storage without allocations after setup. Workspace
allocation is capped at 256 MiB. Allocations made inside a concrete embedder are
separate; the local Qwen3-8B encoder is the [`qwen3`](../qwen3) package.

The reference API's HTTP embedder sends `truncate_prompt_tokens` to vLLM and
normalizes the returned vectors. vLLM's default truncation keeps the last
2048 tokens for both state and candidate texts. An embedder must also return
the post-final-normalization Qwen3-8B last-token vector without adding chat
formatting or special tokens to match the CLM training setup.

## Bundle format and verification

A `.gclm` file contains an 8-byte magic, a version, a JSON manifest, and the
state then action tensor payloads as little-endian float32. The manifest holds
the head configuration, capped scale, source checkpoint provenance, tensor
shapes, and SHA-256 hashes for every tensor. `clm.Load` verifies format,
configuration, shape, finite values, per-tensor hashes, and that the file has no
trailing data before exposing the immutable head pair.

The test suite has a small mathematical oracle for activation, normalization,
scoring, and ranking. It also includes an optional gate against the official
PyTorch checkpoint's projected vectors and scores:

```sh
go test ./clm -run TestReferenceHeadOfficialCheckpointOracle   # with models/CLM_v0.1-8B.gclm
```

That oracle pins the reference checkpoint SHA-256. The standalone head package
can be validated without Qwen3 weights; full end-to-end encoder parity is a
separate gate in the example adapter.
