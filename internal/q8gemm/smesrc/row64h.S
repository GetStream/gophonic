// One FP16 activation row by 64-column FP16 weight panels, FP32
// accumulation with SME2 multi-vector FDOT (vgx4), which rounds once per K
// pair like the widening FMOPA tile. Each K group of 16 (eight pairs) feeds
// eight ZA vector groups, one per pair, and the groups are summed in FP32
// when the panel is stored.
//
// x0=weight panel ([K/2][64][2] FP16), x1=K/16, x2=panel count,
// x3=activation row (K FP16, contiguous), x4=dst, x5=valid output columns
// from dst, x16=FP32 column scales for the first column, x19=pointer to the
// FP32 row scale (the row's inverse power-of-two scale). x7=panel stride (w8 is the ZA slice index), x9=panel
// index, x10=Q pointer,
// x11=A pointer, x12=remaining K groups, x14/x15=temporaries,
// x17=signal retries. The code falls through to the Go epilogue, which
// stores x0 as the result; it must not return directly.
//
// Darwin restores only the low 128 bits of each Z register on signal return;
// ZA survives. The z31 sentinel is checked after accumulation and after the
// stores, and a cleared sentinel recomputes the panel from zero. Each check
// moves an upper 128-bit block into the signal-preserved low block before
// reading it into a GPR, so signal-clobbered predicates cannot hide a retry.
// Every live predicate and the row scale are rebuilt after the sentinel on
// each panel, including retries; no streaming state crosses that boundary.
	smstart
	mov x17, #0
	lsl x7, x1, #11
	mov x9, #0
	mov w8, #0
1:
	zero {za}
	fdup z31.s, #1.0
	ptrue p0.h
	ptrue p1.s
	ptrue pn8.h
	ld1rw {z20.s}, p1/z, [x19]
	mov x10, x0
	mov x11, x3
	mov x12, x1
2:
	cbz x12, 3f
	ld1rqh {z0.h}, p0/z, [x11]
	ld1rqh {z1.h}, p0/z, [x11, #16]
	ld1h {z4.h-z7.h}, pn8/z, [x10]
	fdot za.s[w8, 0, vgx4], {z4.h-z7.h}, z0.h[0]
	ld1h {z8.h-z11.h}, pn8/z, [x10, #4, mul vl]
	fdot za.s[w8, 1, vgx4], {z8.h-z11.h}, z0.h[1]
	ld1h {z12.h-z15.h}, pn8/z, [x10, #8, mul vl]
	fdot za.s[w8, 2, vgx4], {z12.h-z15.h}, z0.h[2]
	ld1h {z16.h-z19.h}, pn8/z, [x10, #12, mul vl]
	fdot za.s[w8, 3, vgx4], {z16.h-z19.h}, z0.h[3]
	add x10, x10, #1024
	ld1h {z4.h-z7.h}, pn8/z, [x10]
	fdot za.s[w8, 4, vgx4], {z4.h-z7.h}, z1.h[0]
	ld1h {z8.h-z11.h}, pn8/z, [x10, #4, mul vl]
	fdot za.s[w8, 5, vgx4], {z8.h-z11.h}, z1.h[1]
	ld1h {z12.h-z15.h}, pn8/z, [x10, #8, mul vl]
	fdot za.s[w8, 6, vgx4], {z12.h-z15.h}, z1.h[2]
	ld1h {z16.h-z19.h}, pn8/z, [x10, #12, mul vl]
	fdot za.s[w8, 7, vgx4], {z16.h-z19.h}, z1.h[3]
	add x10, x10, #1024
	add x11, x11, #32
	sub x12, x12, #1
	b 2b
3:
	ext z31.b, z31.b, z31.b, #48
	fmov w14, s31
	cbz w14, 8f
	// Sum the eight vector groups: z0-z3 hold columns 0-15, 16-31, 32-47, 48-63.
	mova {z0.d-z3.d}, za.d[w8, 0, vgx4]
	mova {z4.d-z7.d}, za.d[w8, 1, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w8, 2, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w8, 3, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w8, 4, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w8, 5, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w8, 6, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w8, 7, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	add x15, x16, x9, lsl #8
	add x14, x4, x9, lsl #8
	sub x13, x5, x9, lsl #6
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15]
	fmul z0.s, z0.s, z8.s
	fmul z0.s, z0.s, z20.s
	st1w {z0.s}, p2, [x14]
	sub x13, x13, #16
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15, #1, mul vl]
	fmul z1.s, z1.s, z8.s
	fmul z1.s, z1.s, z20.s
	st1w {z1.s}, p2, [x14, #1, mul vl]
	sub x13, x13, #16
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15, #2, mul vl]
	fmul z2.s, z2.s, z8.s
	fmul z2.s, z2.s, z20.s
	st1w {z2.s}, p2, [x14, #2, mul vl]
	sub x13, x13, #16
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15, #3, mul vl]
	fmul z3.s, z3.s, z8.s
	fmul z3.s, z3.s, z20.s
	st1w {z3.s}, p2, [x14, #3, mul vl]
	ext z31.b, z31.b, z31.b, #48
	fmov w14, s31
	cbz w14, 8f
	add x0, x0, x7
	add x9, x9, #1
	cmp x9, x2
	b.lt 1b
	smstop
	mov x0, x17
	b 9f
8:
	// A signal cleared the upper Z lanes: rebuild the panel state and retry.
	add x17, x17, #1
	b 1b
9:
