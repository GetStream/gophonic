// Scale FP32 values and round them to FP16 pairs, scattering each pair to
// successive rows of a [K/2][16][2] tile (64-byte stride).
// x0=dst, x1=src, x2=pairs (multiple of 4), v0.s[0]=scale.
	dup v31.4s, v0.s[0]
	mov x3, #64
1:
	cmp x2, #4
	b.lt 2f
	ld1 {v0.4s, v1.4s}, [x1], #32
	fmul v0.4s, v0.4s, v31.4s
	fmul v1.4s, v1.4s, v31.4s
	fcvtn v2.4h, v0.4s
	fcvtn2 v2.8h, v1.4s
	st1 {v2.s}[0], [x0], x3
	st1 {v2.s}[1], [x0], x3
	st1 {v2.s}[2], [x0], x3
	st1 {v2.s}[3], [x0], x3
	sub x2, x2, #4
	b 1b
2:
