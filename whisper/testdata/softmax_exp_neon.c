// Independent NEON exponential fixture generator. This is the source-shaped
// ggml_v_expf from ggml/src/ggml-cpu/vec.h at whisper.cpp commit
// a664346ea5c6dddff3e61a2b7b32dd4514613f50, followed by a small input harness.
// Build on ARM64: clang -O3 -std=c11 softmax_exp_neon.c -o softmax_exp_neon
// Run: ./softmax_exp_neon > softmax_exp_neon.f32le
// Each 32-byte record contains four input float32 values, then their outputs.
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

// Adapted upstream from Arm Limited's optimized routine. Upstream states a
// maximum error of 1.45358 plus 0.5 ulps, overflow above 88.38 and zero below
// -103.97. Softmax only needs the nonpositive range and NaN propagation.
static float32x4_t reference_exp(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

int main(void) {
    const float edges[] = {
        0, -0.0f, -0x1p-149f, -0x1p-126f, -0.5f, -1, -10, -40,
        -87, -87.5f, -88, -100, -103.97f, -104, -INFINITY, NAN,
    };
    uint32_t state = 0x19283746;
    for (int base = 0; base < 256; base += 4) {
        float input[4], output[4];
        for (int lane = 0; lane < 4; lane++) {
            if (base + lane < 16) {
                input[lane] = edges[base + lane];
            } else {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                input[lane] = -110.0f * ((float)(state >> 8) / 16777216.0f);
            }
        }
        vst1q_f32(output, reference_exp(vld1q_f32(input)));
        if (fwrite(input, sizeof(float), 4, stdout) != 4 ||
            fwrite(output, sizeof(float), 4, stdout) != 4) return 1;
    }
    return fflush(stdout) != 0;
}
