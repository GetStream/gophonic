// geluNEON: x0 = values, x1 = bias (or 0), x2 = n (multiple of 4), x3 = table
// values[i] = GELU(values[i] + bias[i]) with the geluNormalTable expansion.
	movz w9, #0x4780, lsl #16          // 65536.0
	dup v16.4s, w9
	mov v17.16b, v16.16b               // its bits, subtracted as integers
	fmov v19.4s, #8.0
	fmov v20.4s, #0.5
	movz w9, #0x3e2a, lsl #16          // 1/6
	movk w9, #0xaaab
	dup v21.4s, w9
	fmov v22.4s, #1.0
	movz w9, #1024
	dup v23.4s, w9
1:
	ldr q0, [x0]
	cbz x1, 2f
	ldr q1, [x1], #16
	fadd v0.4s, v0.4s, v1.4s
2:
	fabs v2.4s, v0.4s
	fcmge v3.4s, v2.4s, v19.4s         // |x| >= 8
	fminnm v4.4s, v2.4s, v19.4s        // |x| clamped to 8 (index <= 1024); NaN selects 8
	fadd v5.4s, v4.4s, v16.4s          // z = a + 65536 rounds to 1/128
	sub v6.4s, v5.4s, v17.4s           // table index
	umin v6.4s, v6.4s, v23.4s          // clamp NaN/sNaN-derived bits before gather
	fsub v7.4s, v5.4s, v16.4s          // r
	fsub v8.4s, v4.4s, v7.4s           // d
	umov w9, v6.s[0]
	umov w10, v6.s[1]
	umov w11, v6.s[2]
	umov w12, v6.s[3]
	// Clamp indices as integers: the table has 1025 entries, and a NaN
	// surviving fminnm would otherwise compute an out-of-bounds address.
	mov w13, #1024
	cmp w9, #1024
	csel w9, w9, w13, lo
	cmp w10, #1024
	csel w10, w10, w13, lo
	cmp w11, #1024
	csel w11, w11, w13, lo
	cmp w12, #1024
	csel w12, w12, w13, lo
	add x9, x3, x9, lsl #3
	add x10, x3, x10, lsl #3
	add x11, x3, x11, lsl #3
	add x12, x3, x12, lsl #3
	ld1 {v24.d}[0], [x9]
	ld1 {v24.d}[1], [x10]
	ld1 {v25.d}[0], [x11]
	ld1 {v25.d}[1], [x12]
	uzp1 v26.4s, v24.4s, v25.4s        // Q(r)
	uzp2 v27.4s, v24.4s, v25.4s        // phi(r)
	fmul v28.4s, v7.4s, v7.4s
	fsub v28.4s, v28.4s, v22.4s
	fmul v28.4s, v28.4s, v21.4s        // curvature = (r*r-1)/6
	fmul v29.4s, v7.4s, v20.4s
	fneg v29.4s, v29.4s
	fmla v29.4s, v8.4s, v28.4s         // d*curvature - r/2
	mov v30.16b, v22.16b
	fmla v30.4s, v29.4s, v8.4s         // series
	fmul v31.4s, v27.4s, v8.4s
	fneg v31.4s, v31.4s
	fmla v26.4s, v31.4s, v30.4s        // q = Q(r) - phi(r)*d*series
	bic v26.16b, v26.16b, v3.16b       // q = 0 for |x| >= 8
	fcmgt v3.4s, v0.4s, #0.0
	fsub v31.4s, v22.4s, v26.4s
	bsl v3.16b, v31.16b, v26.16b       // x > 0 ? 1-q : q
	fmul v0.4s, v0.4s, v3.4s
	str q0, [x0], #16
	subs x2, x2, #4
	b.gt 1b
