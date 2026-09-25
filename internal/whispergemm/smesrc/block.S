// SME source for smeMulBlock in ../sme_arm64.s. Regenerate the WORD
// encodings with: python3 gen.py block.S > ../sme_block_words.txt
//
// Inputs:
//   x0 = A row pointer, x1 = lda bytes, x2 = rows (1..32), x3 = K (>=1)
//   x4 = packed B panels, x5 = N (>=1), x6 = C pointer, x7 = ldc bytes
//   x8 = scratch [roundup(K,16)][32] float32
//   x15 = physical K pitch between packed panels (>= K)
// Output:
//   x0 = number of tiles recomputed after an asynchronous signal.
//
// Darwin restores only the low 128 bits of each Z register after a signal
// handler returns; ZA, predicates, and streaming mode survive. Packing uses
// only ZA, so it is immune. Every accumulation tile keeps a sentinel in z31
// and is recomputed from zero when its upper lanes were cleared.
	smstart
	ptrue p0.s
	ptrue pn8.s
	mov x17, #0                 // retries

	// ---- Pack A: 32 rows x K into scratch[k][32] through ZA0/ZA1. ----
	mov x9, #0                  // kk (elements)
	mov x16, #16
1:
	zero {za0.s, za1.s}
	whilelt p1.s, x9, x3        // valid k lanes in this chunk
	add x10, x0, x9, lsl #2     // &A[0][kk]
	mov w12, #0
2:
	cmp x12, x2
	b.ge 3f
	cmp x12, #16
	b.ge 21f
	ld1w {za0h.s[w12, 0]}, p1/z, [x10]
	b 22f
21:
	sub w13, w12, #16
	ld1w {za1h.s[w13, 0]}, p1/z, [x10]
22:
	add x10, x10, x1
	add w12, w12, #1
	cmp x12, #32
	b.lt 2b
3:
	// vertical slice j holds 16 rows at k = kk + j
	add x11, x8, x9, lsl #7     // scratch + kk*32*4
	mov w12, #0
4:
	st1w {za0v.s[w12, 0]}, p0, [x11]
	st1w {za1v.s[w12, 0]}, p0, [x11, x16, lsl #2]
	add x11, x11, #128
	add w12, w12, #1
	cmp w12, #16
	b.lt 4b
	add x9, x9, #16
	cmp x9, x3
	b.lt 1b

	// ---- Column panel pairs. ----
	lsl x15, x15, #6            // panel bytes = physical pitch*16*4
	mov x14, #0                 // first column of this pair
5:
	mov x11, x4                 // b0
	add x13, x14, #16
	cmp x13, x5
	csel x10, x4, x4, ge        // odd final panel: reuse b0, discard cols
	add x12, x4, x15
	csel x10, x12, x10, lt      // b1
6:
	zero {za}
	fdup z31.s, #1.0
	mov x9, x8                  // A scratch
	mov x13, x11
	mov x12, x10
	mov x16, x3
	cmp x16, #4
	b.lt 8f
	// Software pipeline over pairs of K steps. Set 0 is z0-z7, set 1 is
	// z8-z15: A {k, k+1} x 32 rows, then two 16-column B panels for k, k+1.
	ld1w {z0.s-z3.s}, pn8/z, [x9]
	ld1w {z4.s-z5.s}, pn8/z, [x13]
	ld1w {z6.s-z7.s}, pn8/z, [x12]
	add x9, x9, #256
	add x13, x13, #128
	add x12, x12, #128
7:
	ld1w {z8.s-z11.s}, pn8/z, [x9]
	ld1w {z12.s-z13.s}, pn8/z, [x13]
	ld1w {z14.s-z15.s}, pn8/z, [x12]
	add x9, x9, #256
	add x13, x13, #128
	add x12, x12, #128
	fmopa za0.s, p0/m, p0/m, z0.s, z4.s
	fmopa za1.s, p0/m, p0/m, z0.s, z6.s
	fmopa za2.s, p0/m, p0/m, z1.s, z4.s
	fmopa za3.s, p0/m, p0/m, z1.s, z6.s
	fmopa za0.s, p0/m, p0/m, z2.s, z5.s
	fmopa za1.s, p0/m, p0/m, z2.s, z7.s
	fmopa za2.s, p0/m, p0/m, z3.s, z5.s
	fmopa za3.s, p0/m, p0/m, z3.s, z7.s
	sub x16, x16, #4
	cmp x16, #4
	b.lt 71f
	ld1w {z0.s-z3.s}, pn8/z, [x9]
	ld1w {z4.s-z5.s}, pn8/z, [x13]
	ld1w {z6.s-z7.s}, pn8/z, [x12]
	add x9, x9, #256
	add x13, x13, #128
	add x12, x12, #128
	fmopa za0.s, p0/m, p0/m, z8.s, z12.s
	fmopa za1.s, p0/m, p0/m, z8.s, z14.s
	fmopa za2.s, p0/m, p0/m, z9.s, z12.s
	fmopa za3.s, p0/m, p0/m, z9.s, z14.s
	fmopa za0.s, p0/m, p0/m, z10.s, z13.s
	fmopa za1.s, p0/m, p0/m, z10.s, z15.s
	fmopa za2.s, p0/m, p0/m, z11.s, z13.s
	fmopa za3.s, p0/m, p0/m, z11.s, z15.s
	b 7b
71:
	fmopa za0.s, p0/m, p0/m, z8.s, z12.s
	fmopa za1.s, p0/m, p0/m, z8.s, z14.s
	fmopa za2.s, p0/m, p0/m, z9.s, z12.s
	fmopa za3.s, p0/m, p0/m, z9.s, z14.s
	fmopa za0.s, p0/m, p0/m, z10.s, z13.s
	fmopa za1.s, p0/m, p0/m, z10.s, z15.s
	fmopa za2.s, p0/m, p0/m, z11.s, z13.s
	fmopa za3.s, p0/m, p0/m, z11.s, z15.s
8:
	cbz x16, 9f
	ld1w {z0.s}, p0/z, [x9]
	ld1w {z1.s}, p0/z, [x9, #1, mul vl]
	ld1w {z2.s}, p0/z, [x13]
	ld1w {z3.s}, p0/z, [x12]
	fmopa za0.s, p0/m, p0/m, z0.s, z2.s
	fmopa za1.s, p0/m, p0/m, z0.s, z3.s
	fmopa za2.s, p0/m, p0/m, z1.s, z2.s
	fmopa za3.s, p0/m, p0/m, z1.s, z3.s
	add x9, x9, #128
	add x13, x13, #64
	add x12, x12, #64
	sub x16, x16, #1
	b 8b
9:
	// Signal check with freshly built predicates.
	ptrue p3.s
	fcmeq p4.s, p3/z, z31.s, #0.0
	ptest p3, p4.b
	b.ne 91f
	cntp x9, p3, p0.s
	cmp x9, #16
	b.ne 91f
	cntp x9, pn8.s, vlx4
	cmp x9, #64
	b.eq 10f
91:
	add x17, x17, #1
	ptrue p0.s
	ptrue pn8.s
	b 6b
10:
	// Store rows with column predicates.
	sub x9, x5, x14             // remaining columns
	mov x16, #16
	whilelt p1.s, xzr, x9
	whilelt p2.s, x16, x9
	add x13, x6, x14, lsl #2    // &C[0][col]
	mov w12, #0
	cmp x2, #16
	csel x9, x2, x16, lt
11:
	cmp x12, x9
	b.ge 12f
	st1w {za0h.s[w12, 0]}, p1, [x13]
	st1w {za1h.s[w12, 0]}, p2, [x13, x16, lsl #2]
	add x13, x13, x7
	add w12, w12, #1
	b 11b
12:
	sub x9, x2, #16
	mov w12, #0
13:
	cmp x12, x9
	b.ge 14f
	st1w {za2h.s[w12, 0]}, p1, [x13]
	st1w {za3h.s[w12, 0]}, p2, [x13, x16, lsl #2]
	add x13, x13, x7
	add w12, w12, #1
	b 13b
14:
	add x14, x14, #32
	add x4, x4, x15, lsl #1     // advance two panels
	cmp x14, x5
	b.lt 5b
	smstop
	mov x0, x17
