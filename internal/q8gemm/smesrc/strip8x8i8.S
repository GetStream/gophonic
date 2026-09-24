// NEON int8 block: up to 8 activation rows by 8 output columns from the
// [K/4][64][4] weight layout, with SDOT by element and exact int32 sums,
// then the same epilogue as the SME kernel: FP32 conversion, column scale,
// row scale.
//
// x0=weights for the block's first column (panel base + 4*column),
// x1=K quads, x2=activations for the block's first row ([K/4][16][4] tile +
// 4*row), x3=dst for (first row, first column), x4=dst row stride in bytes,
// x5=rows to store (1..8), x6=8 FP32 column scales, x7=8 FP32 row scales.
// Accumulators: v16..v31, two per row (columns 0-3 and 4-7).
	movi v16.4s, #0
	movi v17.4s, #0
	movi v18.4s, #0
	movi v19.4s, #0
	movi v20.4s, #0
	movi v21.4s, #0
	movi v22.4s, #0
	movi v23.4s, #0
	movi v24.4s, #0
	movi v25.4s, #0
	movi v26.4s, #0
	movi v27.4s, #0
	movi v28.4s, #0
	movi v29.4s, #0
	movi v30.4s, #0
	movi v31.4s, #0
	mov x8, #256
	mov x9, #64
1:
	cbz x1, 2f
	ld1 {v0.16b, v1.16b}, [x0], x8
	ld1 {v2.16b, v3.16b}, [x2], x9
	sdot v16.4s, v0.16b, v2.4b[0]
	sdot v17.4s, v1.16b, v2.4b[0]
	sdot v18.4s, v0.16b, v2.4b[1]
	sdot v19.4s, v1.16b, v2.4b[1]
	sdot v20.4s, v0.16b, v2.4b[2]
	sdot v21.4s, v1.16b, v2.4b[2]
	sdot v22.4s, v0.16b, v2.4b[3]
	sdot v23.4s, v1.16b, v2.4b[3]
	sdot v24.4s, v0.16b, v3.4b[0]
	sdot v25.4s, v1.16b, v3.4b[0]
	sdot v26.4s, v0.16b, v3.4b[1]
	sdot v27.4s, v1.16b, v3.4b[1]
	sdot v28.4s, v0.16b, v3.4b[2]
	sdot v29.4s, v1.16b, v3.4b[2]
	sdot v30.4s, v0.16b, v3.4b[3]
	sdot v31.4s, v1.16b, v3.4b[3]
	sub x1, x1, #1
	b 1b
2:
	ld1 {v4.4s, v5.4s}, [x6]
	ld1 {v6.4s, v7.4s}, [x7]
	// Row r: convert, multiply by column scales, then by row scale r.
	scvtf v16.4s, v16.4s
	scvtf v17.4s, v17.4s
	fmul v16.4s, v16.4s, v4.4s
	fmul v17.4s, v17.4s, v5.4s
	fmul v16.4s, v16.4s, v6.s[0]
	fmul v17.4s, v17.4s, v6.s[0]
	st1 {v16.4s, v17.4s}, [x3], x4
	cmp x5, #1
	b.le 9f
	scvtf v18.4s, v18.4s
	scvtf v19.4s, v19.4s
	fmul v18.4s, v18.4s, v4.4s
	fmul v19.4s, v19.4s, v5.4s
	fmul v18.4s, v18.4s, v6.s[1]
	fmul v19.4s, v19.4s, v6.s[1]
	st1 {v18.4s, v19.4s}, [x3], x4
	cmp x5, #2
	b.le 9f
	scvtf v20.4s, v20.4s
	scvtf v21.4s, v21.4s
	fmul v20.4s, v20.4s, v4.4s
	fmul v21.4s, v21.4s, v5.4s
	fmul v20.4s, v20.4s, v6.s[2]
	fmul v21.4s, v21.4s, v6.s[2]
	st1 {v20.4s, v21.4s}, [x3], x4
	cmp x5, #3
	b.le 9f
	scvtf v22.4s, v22.4s
	scvtf v23.4s, v23.4s
	fmul v22.4s, v22.4s, v4.4s
	fmul v23.4s, v23.4s, v5.4s
	fmul v22.4s, v22.4s, v6.s[3]
	fmul v23.4s, v23.4s, v6.s[3]
	st1 {v22.4s, v23.4s}, [x3], x4
	cmp x5, #4
	b.le 9f
	scvtf v24.4s, v24.4s
	scvtf v25.4s, v25.4s
	fmul v24.4s, v24.4s, v4.4s
	fmul v25.4s, v25.4s, v5.4s
	fmul v24.4s, v24.4s, v7.s[0]
	fmul v25.4s, v25.4s, v7.s[0]
	st1 {v24.4s, v25.4s}, [x3], x4
	cmp x5, #5
	b.le 9f
	scvtf v26.4s, v26.4s
	scvtf v27.4s, v27.4s
	fmul v26.4s, v26.4s, v4.4s
	fmul v27.4s, v27.4s, v5.4s
	fmul v26.4s, v26.4s, v7.s[1]
	fmul v27.4s, v27.4s, v7.s[1]
	st1 {v26.4s, v27.4s}, [x3], x4
	cmp x5, #6
	b.le 9f
	scvtf v28.4s, v28.4s
	scvtf v29.4s, v29.4s
	fmul v28.4s, v28.4s, v4.4s
	fmul v29.4s, v29.4s, v5.4s
	fmul v28.4s, v28.4s, v7.s[2]
	fmul v29.4s, v29.4s, v7.s[2]
	st1 {v28.4s, v29.4s}, [x3], x4
	cmp x5, #7
	b.le 9f
	scvtf v30.4s, v30.4s
	scvtf v31.4s, v31.4s
	fmul v30.4s, v30.4s, v4.4s
	fmul v31.4s, v31.4s, v5.4s
	fmul v30.4s, v30.4s, v7.s[3]
	fmul v31.4s, v31.4s, v7.s[3]
	st1 {v30.4s, v31.4s}, [x3], x4
9:
