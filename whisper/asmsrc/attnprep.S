// attnPrepNEON: x0 = q, x1 = k, x2 = v, x3 = qbias, x4 = vbias, x5 = n (multiple of 4), s0 = scale
// q = (q + qbias) * scale; k = k * scale; v = v + vbias.
	dup v31.4s, v0.s[0]
1:
	ldr q0, [x0]
	ldr q1, [x3], #16
	fadd v0.4s, v0.4s, v1.4s
	fmul v0.4s, v0.4s, v31.4s
	str q0, [x0], #16
	ldr q2, [x1]
	fmul v2.4s, v2.4s, v31.4s
	str q2, [x1], #16
	ldr q3, [x2]
	ldr q4, [x4], #16
	fadd v3.4s, v3.4s, v4.4s
	str q3, [x2], #16
	subs x5, x5, #4
	b.gt 1b
