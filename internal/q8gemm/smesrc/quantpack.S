// Scale FP32 values, round to nearest (ties to even), saturate to int8, and
// scatter each group of four to successive rows of a [K/4][16][4] tile
// (64-byte stride). x0=dst, x1=src, x2=quads (multiple of 4), v0.s[0]=scale.
	dup v31.4s, v0.s[0]
	mov x3, #64
1:
	cmp x2, #4
	b.lt 2f
	ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
	fmul v0.4s, v0.4s, v31.4s
	fmul v1.4s, v1.4s, v31.4s
	fmul v2.4s, v2.4s, v31.4s
	fmul v3.4s, v3.4s, v31.4s
	fcvtns v0.4s, v0.4s
	fcvtns v1.4s, v1.4s
	fcvtns v2.4s, v2.4s
	fcvtns v3.4s, v3.4s
	sqxtn v4.4h, v0.4s
	sqxtn2 v4.8h, v1.4s
	sqxtn v5.4h, v2.4s
	sqxtn2 v5.8h, v3.4s
	sqxtn v6.8b, v4.8h
	sqxtn2 v6.16b, v5.8h
	st1 {v6.s}[0], [x0], x3
	st1 {v6.s}[1], [x0], x3
	st1 {v6.s}[2], [x0], x3
	st1 {v6.s}[3], [x0], x3
	sub x2, x2, #4
	b 1b
2:
