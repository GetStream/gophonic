// residualNormNEON: x0=row x1=dst x2=gamma x3=beta x4=n (multiple of 8, > 0),
// x10=add x11=bias (or 0). row += add + bias, then dst = LayerNorm(row).
// Float64 accumulation over four float64x2 partial sums per pass.
	movi v16.2d, #0
	movi v17.2d, #0
	movi v18.2d, #0
	movi v19.2d, #0
	mov x5, x0
	mov x6, x4
	mov x13, x10
	mov x14, x11
1:
	ldp q0, q1, [x5]
	ldp q8, q9, [x13], #32
	cbz x11, 11f
	ldp q10, q11, [x14], #32
	fadd v8.4s, v8.4s, v10.4s
	fadd v9.4s, v9.4s, v11.4s
11:
	fadd v0.4s, v0.4s, v8.4s
	fadd v1.4s, v1.4s, v9.4s
	stp q0, q1, [x5], #32
	fcvtl v2.2d, v0.2s
	fcvtl2 v3.2d, v0.4s
	fcvtl v4.2d, v1.2s
	fcvtl2 v5.2d, v1.4s
	fadd v16.2d, v16.2d, v2.2d
	fadd v17.2d, v17.2d, v3.2d
	fadd v18.2d, v18.2d, v4.2d
	fadd v19.2d, v19.2d, v5.2d
	subs x6, x6, #8
	b.gt 1b
	fadd v16.2d, v16.2d, v17.2d
	fadd v18.2d, v18.2d, v19.2d
	fadd v16.2d, v16.2d, v18.2d
	faddp d16, v16.2d
	scvtf d20, x4
	fdiv d16, d16, d20
	dup v21.2d, v16.d[0]
	movi v16.2d, #0
	movi v17.2d, #0
	movi v18.2d, #0
	movi v19.2d, #0
	mov x5, x0
	mov x6, x4
2:
	ldp q0, q1, [x5], #32
	fcvtl v2.2d, v0.2s
	fcvtl2 v3.2d, v0.4s
	fcvtl v4.2d, v1.2s
	fcvtl2 v5.2d, v1.4s
	fsub v2.2d, v2.2d, v21.2d
	fsub v3.2d, v3.2d, v21.2d
	fsub v4.2d, v4.2d, v21.2d
	fsub v5.2d, v5.2d, v21.2d
	fmla v16.2d, v2.2d, v2.2d
	fmla v17.2d, v3.2d, v3.2d
	fmla v18.2d, v4.2d, v4.2d
	fmla v19.2d, v5.2d, v5.2d
	subs x6, x6, #8
	b.gt 2b
	fadd v16.2d, v16.2d, v17.2d
	fadd v18.2d, v18.2d, v19.2d
	fadd v16.2d, v16.2d, v18.2d
	faddp d16, v16.2d
	fdiv d16, d16, d20
	movz x7, #0x68f1
	movk x7, #0x88e3, lsl #16
	movk x7, #0xf8b5, lsl #32
	movk x7, #0x3ee4, lsl #48
	fmov d22, x7
	fadd d16, d16, d22
	fsqrt d16, d16
	fmov d23, #1.0
	fdiv d23, d23, d16
	dup v24.2d, v23.d[0]
	mov x6, x4
3:
	ldr q0, [x0], #16
	ldr q1, [x2], #16
	ldr q6, [x3], #16
	fcvtl v2.2d, v0.2s
	fcvtl2 v3.2d, v0.4s
	fcvtl v4.2d, v1.2s
	fcvtl2 v5.2d, v1.4s
	fcvtl v7.2d, v6.2s
	fcvtl2 v6.2d, v6.4s
	fsub v2.2d, v2.2d, v21.2d
	fsub v3.2d, v3.2d, v21.2d
	fmul v2.2d, v2.2d, v24.2d
	fmul v3.2d, v3.2d, v24.2d
	fmla v7.2d, v2.2d, v4.2d
	fmla v6.2d, v3.2d, v5.2d
	fcvtn v0.2s, v7.2d
	fcvtn2 v0.4s, v6.2d
	str q0, [x1], #16
	subs x6, x6, #4
	b.gt 3b
