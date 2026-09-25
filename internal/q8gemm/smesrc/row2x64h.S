// Two contiguous FP16 rows by 64-column FP16 weight panels. Each weight
// vector is loaded once and used by both rows. At SVL=64 bytes, ZA has
// sixteen VGx4 vector groups: row zero owns groups 0..7 and row one 8..15.
// Each row retains smeRowF16's eight independent FP32 partial streams,
// pairwise FDOT rounding, ordered final additions, and scale multiplications.
//
// x0=weights, x1=K/16, x2=panel count, x3=two adjacent K-wide FP16 rows,
// x4=dst, x5=valid columns, x6=destination row stride in bytes,
// x16=column scales, x19=two row inverse scales. x7=weight panel stride,
// w8=0 and w9=8 are ZA row bases, x10=weight cursor, x11/x21=activation
// cursors, x12=remaining K groups, x13..x15=store temporaries, x17=retries,
// x20=panel index, x22=activation row stride in bytes.
//
// As in smeRowF16, every panel rebuilds streaming state after the sentinel.
// A signal-clear check captures upper lanes into the preserved low 128 bits
// before reading a GPR. A retry recomputes both rows; their stores are idempotent.
	smstart
	mov x17, #0
	lsl x7, x1, #11
	mov x20, #0
	mov w8, #0
	mov w9, #8
	lsl x22, x1, #5
1:
	zero {za}
	fdup z31.s, #1.0
	ptrue p0.h
	ptrue p1.s
	ptrue pn8.h
	ld1rw {z20.s}, p1/z, [x19]
	ld1rw {z21.s}, p1/z, [x19, #4]
	mov x10, x0
	mov x11, x3
	add x21, x3, x22
	mov x12, x1
2:
	cbz x12, 3f
	ld1rqh {z0.h}, p0/z, [x11]
	ld1rqh {z1.h}, p0/z, [x11, #16]
	ld1rqh {z2.h}, p0/z, [x21]
	ld1rqh {z3.h}, p0/z, [x21, #16]
	ld1h {z4.h-z7.h}, pn8/z, [x10]
	fdot za.s[w8, 0, vgx4], {z4.h-z7.h}, z0.h[0]
	fdot za.s[w9, 0, vgx4], {z4.h-z7.h}, z2.h[0]
	ld1h {z8.h-z11.h}, pn8/z, [x10, #4, mul vl]
	fdot za.s[w8, 1, vgx4], {z8.h-z11.h}, z0.h[1]
	fdot za.s[w9, 1, vgx4], {z8.h-z11.h}, z2.h[1]
	ld1h {z12.h-z15.h}, pn8/z, [x10, #8, mul vl]
	fdot za.s[w8, 2, vgx4], {z12.h-z15.h}, z0.h[2]
	fdot za.s[w9, 2, vgx4], {z12.h-z15.h}, z2.h[2]
	ld1h {z16.h-z19.h}, pn8/z, [x10, #12, mul vl]
	fdot za.s[w8, 3, vgx4], {z16.h-z19.h}, z0.h[3]
	fdot za.s[w9, 3, vgx4], {z16.h-z19.h}, z2.h[3]
	add x10, x10, #1024
	ld1h {z4.h-z7.h}, pn8/z, [x10]
	fdot za.s[w8, 4, vgx4], {z4.h-z7.h}, z1.h[0]
	fdot za.s[w9, 4, vgx4], {z4.h-z7.h}, z3.h[0]
	ld1h {z8.h-z11.h}, pn8/z, [x10, #4, mul vl]
	fdot za.s[w8, 5, vgx4], {z8.h-z11.h}, z1.h[1]
	fdot za.s[w9, 5, vgx4], {z8.h-z11.h}, z3.h[1]
	ld1h {z12.h-z15.h}, pn8/z, [x10, #8, mul vl]
	fdot za.s[w8, 6, vgx4], {z12.h-z15.h}, z1.h[2]
	fdot za.s[w9, 6, vgx4], {z12.h-z15.h}, z3.h[2]
	ld1h {z16.h-z19.h}, pn8/z, [x10, #12, mul vl]
	fdot za.s[w8, 7, vgx4], {z16.h-z19.h}, z1.h[3]
	fdot za.s[w9, 7, vgx4], {z16.h-z19.h}, z3.h[3]
	add x10, x10, #1024
	add x11, x11, #32
	add x21, x21, #32
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
	add x15, x16, x20, lsl #8
	add x14, x4, x20, lsl #8
	sub x13, x5, x20, lsl #6
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
	// Sum the eight vector groups: z0-z3 hold columns 0-15, 16-31, 32-47, 48-63.
	mova {z0.d-z3.d}, za.d[w9, 0, vgx4]
	mova {z4.d-z7.d}, za.d[w9, 1, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w9, 2, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w9, 3, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w9, 4, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w9, 5, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w9, 6, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	mova {z4.d-z7.d}, za.d[w9, 7, vgx4]
	fadd z0.s, z0.s, z4.s
	fadd z1.s, z1.s, z5.s
	fadd z2.s, z2.s, z6.s
	fadd z3.s, z3.s, z7.s
	add x15, x16, x20, lsl #8
	add x14, x4, x6
	add x14, x14, x20, lsl #8
	sub x13, x5, x20, lsl #6
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15]
	fmul z0.s, z0.s, z8.s
	fmul z0.s, z0.s, z21.s
	st1w {z0.s}, p2, [x14]
	sub x13, x13, #16
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15, #1, mul vl]
	fmul z1.s, z1.s, z8.s
	fmul z1.s, z1.s, z21.s
	st1w {z1.s}, p2, [x14, #1, mul vl]
	sub x13, x13, #16
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15, #2, mul vl]
	fmul z2.s, z2.s, z8.s
	fmul z2.s, z2.s, z21.s
	st1w {z2.s}, p2, [x14, #2, mul vl]
	sub x13, x13, #16
	whilelt p2.s, xzr, x13
	ld1w {z8.s}, p2/z, [x15, #3, mul vl]
	fmul z3.s, z3.s, z8.s
	fmul z3.s, z3.s, z21.s
	st1w {z3.s}, p2, [x14, #3, mul vl]
	ext z31.b, z31.b, z31.b, #48
	fmov w14, s31
	cbz w14, 8f
	add x0, x0, x7
	add x20, x20, #1
	cmp x20, x2
	b.lt 1b
	smstop
	mov x0, x17
	b 9f
8:
	// A signal cleared the upper Z lanes: rebuild the panel state and retry.
	add x17, x17, #1
	b 1b
9:
