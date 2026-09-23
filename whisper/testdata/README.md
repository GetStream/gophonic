# Encoder softmax reference

`softmax_exp_neon.c` preserves the NEON `ggml_v_expf` implementation from
whisper.cpp commit `a664346ea5c6dddff3e61a2b7b32dd4514613f50`,
`ggml/src/ggml-cpu/vec.h`. Its MIT license is included in the file.

The independent ARM64 executable generated
`softmax_exp_neon.f32le`: 64 records containing four
input float32 values and four output float32 values in little-endian order.
Inputs cover signed zero, normal/subnormal boundaries, underflow, negative
infinity, NaN and deterministic samples in `[-110,0]`.

```sh
clang -O3 -std=c11 whisper/testdata/softmax_exp_neon.c \
  -o /tmp/softmax_exp_neon
/tmp/softmax_exp_neon > whisper/testdata/softmax_exp_neon.f32le
```

Fixture SHA-256:
`d80ca9601cfe78afc049731354eb4b5619c83858429ac6085039a45a9b665423`.

The Go SIMD implementation must reproduce the finite C results bit for bit.
A separate dense-grid test compares it with `math.Exp` across the nonpositive
range, including argument-reduction and subnormal boundaries, with a maximum
error of two float32 ULPs. Encoder stage tolerances are unchanged.
