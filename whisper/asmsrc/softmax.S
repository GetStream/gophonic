// softmaxExpNEON: x0 = x, x1 = n (multiple of 4, >= 4), x2 = constants
// constants: floor, magic, log2e, -ln2hi, -ln2lo, p0, p1, p2, p3, p4, onebits
// Replaces x with exp(x - max(x)) and returns the sum in s0.
	// Pass 1: maximum.
	ldr q16, [x0]
	mov v17.16b, v16.16b
	mov x3, x0
	mov x4, x1
	cmp x4, #8
	b.lt 12f
11:
	ldp q0, q1, [x3], #32
	fmax v16.4s, v16.4s, v0.4s
	fmax v17.4s, v17.4s, v1.4s
	sub x4, x4, #8
	cmp x4, #8
	b.ge 11b
12:
	cbz x4, 13f
	ldr q0, [x3], #16
	fmax v16.4s, v16.4s, v0.4s
13:
	fmax v16.4s, v16.4s, v17.4s
	fmaxv s16, v16.4s
	dup v16.4s, v16.s[0]
	// Constants.
	ld1r {v17.4s}, [x2], #4
	ld1r {v18.4s}, [x2], #4
	ld1r {v19.4s}, [x2], #4
	ld1r {v20.4s}, [x2], #4
	ld1r {v21.4s}, [x2], #4
	ld1r {v22.4s}, [x2], #4
	ld1r {v23.4s}, [x2], #4
	ld1r {v24.4s}, [x2], #4
	ld1r {v25.4s}, [x2], #4
	ld1r {v26.4s}, [x2], #4
	ld1r {v27.4s}, [x2], #4
	movi v28.4s, #0
	movi v29.4s, #0
	mov x3, x0
	mov x4, x1
	cmp x4, #8
	b.lt 22f
21:
	ldp q0, q8, [x3]
	fsub v0.4s, v0.4s, v16.4s
	fsub v8.4s, v8.4s, v16.4s
	fmax v0.4s, v0.4s, v17.4s
	fmax v8.4s, v8.4s, v17.4s
	mov v1.16b, v18.16b
	mov v9.16b, v18.16b
	fmla v1.4s, v0.4s, v19.4s
	fmla v9.4s, v8.4s, v19.4s
	fsub v2.4s, v1.4s, v18.4s
	fsub v10.4s, v9.4s, v18.4s
	mov v3.16b, v0.16b
	mov v11.16b, v8.16b
	fmla v3.4s, v2.4s, v20.4s
	fmla v11.4s, v10.4s, v20.4s
	fmla v3.4s, v2.4s, v21.4s
	fmla v11.4s, v10.4s, v21.4s
	shl v1.4s, v1.4s, #23
	shl v9.4s, v9.4s, #23
	add v1.4s, v1.4s, v27.4s
	add v9.4s, v9.4s, v27.4s
	fmul v4.4s, v3.4s, v3.4s
	fmul v12.4s, v11.4s, v11.4s
	mov v5.16b, v25.16b
	mov v13.16b, v25.16b
	fmla v5.4s, v26.4s, v3.4s
	fmla v13.4s, v26.4s, v11.4s
	mov v6.16b, v23.16b
	mov v14.16b, v23.16b
	fmla v6.4s, v24.4s, v3.4s
	fmla v14.4s, v24.4s, v11.4s
	fmla v6.4s, v5.4s, v4.4s
	fmla v14.4s, v13.4s, v12.4s
	fmul v7.4s, v22.4s, v3.4s
	fmul v15.4s, v22.4s, v11.4s
	fmla v7.4s, v6.4s, v4.4s
	fmla v15.4s, v14.4s, v12.4s
	mov v0.16b, v1.16b
	mov v8.16b, v9.16b
	fmla v0.4s, v7.4s, v1.4s
	fmla v8.4s, v15.4s, v9.4s
	stp q0, q8, [x3], #32
	fadd v28.4s, v28.4s, v0.4s
	fadd v29.4s, v29.4s, v8.4s
	sub x4, x4, #8
	cmp x4, #8
	b.ge 21b
22:
	cbz x4, 23f
	ldr q0, [x3]
	fsub v0.4s, v0.4s, v16.4s
	fmax v0.4s, v0.4s, v17.4s
	mov v1.16b, v18.16b
	fmla v1.4s, v0.4s, v19.4s
	fsub v2.4s, v1.4s, v18.4s
	mov v3.16b, v0.16b
	fmla v3.4s, v2.4s, v20.4s
	fmla v3.4s, v2.4s, v21.4s
	shl v1.4s, v1.4s, #23
	add v1.4s, v1.4s, v27.4s
	fmul v4.4s, v3.4s, v3.4s
	mov v5.16b, v25.16b
	fmla v5.4s, v26.4s, v3.4s
	mov v6.16b, v23.16b
	fmla v6.4s, v24.4s, v3.4s
	fmla v6.4s, v5.4s, v4.4s
	fmul v7.4s, v22.4s, v3.4s
	fmla v7.4s, v6.4s, v4.4s
	mov v0.16b, v1.16b
	fmla v0.4s, v7.4s, v1.4s
	str q0, [x3], #16
	fadd v28.4s, v28.4s, v0.4s
23:
	fadd v28.4s, v28.4s, v29.4s
	faddp v28.4s, v28.4s, v28.4s
	faddp s0, v28.2s
