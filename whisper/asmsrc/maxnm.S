// maxNumNEON: x0 = x, x1 = n (multiple of 16, >= 16). Returns max ignoring NaN in s0.
	ldp q0, q1, [x0]
	ldp q2, q3, [x0, #32]
	add x2, x0, #64
	subs x1, x1, #16
	b.le 2f
1:
	ldp q4, q5, [x2], #32
	ldp q6, q7, [x2], #32
	fmaxnm v0.4s, v0.4s, v4.4s
	fmaxnm v1.4s, v1.4s, v5.4s
	fmaxnm v2.4s, v2.4s, v6.4s
	fmaxnm v3.4s, v3.4s, v7.4s
	subs x1, x1, #16
	b.gt 1b
2:
	fmaxnm v0.4s, v0.4s, v1.4s
	fmaxnm v2.4s, v2.4s, v3.4s
	fmaxnm v0.4s, v0.4s, v2.4s
	fmaxnmv s0, v0.4s
