// Largest magnitude of n FP32 values (n multiple of 16); NaN propagates.
// x0=src, x1=n. Result in s0.
	movi v0.4s, #0
	movi v1.4s, #0
	movi v2.4s, #0
	movi v3.4s, #0
1:
	cmp x1, #16
	b.lt 2f
	ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
	fabs v4.4s, v4.4s
	fabs v5.4s, v5.4s
	fabs v6.4s, v6.4s
	fabs v7.4s, v7.4s
	fmax v0.4s, v0.4s, v4.4s
	fmax v1.4s, v1.4s, v5.4s
	fmax v2.4s, v2.4s, v6.4s
	fmax v3.4s, v3.4s, v7.4s
	sub x1, x1, #16
	b 1b
2:
	fmax v0.4s, v0.4s, v1.4s
	fmax v2.4s, v2.4s, v3.4s
	fmax v0.4s, v0.4s, v2.4s
	fmaxv s0, v0.4s
