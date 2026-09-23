// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause
// Optional macOS CPU SGEMM reference; never compiled into the Go package.
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { const char *name; int m, k, n; } shape;
static const shape shapes[] = {
    {"encoder_projection", 1500, 384, 384},
    {"encoder_expand", 1500, 384, 1536},
    {"encoder_contract", 1500, 1536, 384},
    {"attention_scores", 1500, 64, 1500},
    {"attention_values", 1500, 1500, 64},
    {"decoder_projection", 1, 384, 384},
    {"decoder_expand", 1, 384, 1536},
    {"decoder_contract", 1, 1536, 384},
    {"decoder_vocabulary", 1, 384, 51864},
    {"tails", 7, 65, 19},
};
static volatile float sink;
static float next_value(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return (float)((int32_t)(x >> 8) - 8388608) * (1.0f / 8388608.0f);
}
static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void multiply(shape s, const float *a, const float *w, float *c) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                s.m, s.n, s.k, 1.0f, a, s.k, w, s.k, 0.0f, c, s.n);
}
int main(int argc, char **argv) {
    int count = argc > 1 ? atoi(argv[1]) : 20;
    const char *filter = argc > 2 ? argv[2] : "";
    if (count < 1) return 2;
    printf("reference=Accelerate/cblas_sgemm VECLIB_MAXIMUM_THREADS=%s\n",
           getenv("VECLIB_MAXIMUM_THREADS") ? getenv("VECLIB_MAXIMUM_THREADS") : "unset");
    for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); ++si) {
        shape s = shapes[si];
        if (*filter && !strstr(s.name, filter)) continue;
        size_t na = (size_t)s.m*s.k, nw = (size_t)s.n*s.k, nc = (size_t)s.m*s.n;
        float *a = malloc(na*sizeof(float)), *w = malloc(nw*sizeof(float)), *c = malloc(nc*sizeof(float));
        if (!a || !w || !c) return 3;
        uint32_t state = UINT32_C(0xc001cafe);
        for (size_t i = 0; i < na; ++i) a[i] = next_value(&state);
        for (size_t i = 0; i < nw; ++i) w[i] = next_value(&state);
        for (int i = 0; i < 5; ++i) multiply(s, a, w, c);
        double worst = 0;
        for (int sample = 0; sample < 17; ++sample) {
            int r = (int)((int64_t)sample*997%s.m), col = (int)((int64_t)sample*7919%s.n);
            double want = 0, sum_abs = 0;
            for (int p = 0; p < s.k; ++p) {
                double product = (double)a[(size_t)r*s.k+p] * (double)w[(size_t)col*s.k+p];
                want += product; sum_abs += fabs(product);
            }
            double error = fabs((double)c[(size_t)r*s.n+col] - want);
            if (error > worst) worst = error;
            if (!isfinite(c[(size_t)r*s.n+col]) || error > 4.0*(s.k+1)*0x1p-24*sum_abs) {
                fprintf(stderr, "oracle mismatch %s: %.9g != %.9g\n", s.name, c[(size_t)r*s.n+col], want);
                return 4;
            }
        }
        double begin = seconds();
        for (int i = 0; i < count; ++i) multiply(s, a, w, c);
        double elapsed = (seconds()-begin)/count;
        sink = c[nc-1];
        printf("%s M=%d K=%d N=%d count=%d %.6f ms %.3f GFLOP/s max_sample_error=%.3g checksum=%.9g\n",
               s.name, s.m, s.k, s.n, count, elapsed*1000, 2.0*s.m*s.k*s.n/elapsed/1e9, worst, (double)sink);
        free(a); free(w); free(c);
    }
    return 0;
}
