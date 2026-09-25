// 16 FP16 activation rows by 64 FP16 weight columns, FP32 accumulation.
// Each ZA FP32 tile accumulates 16 columns with the widening FP16 FMOPA,
// which adds two K products per element and instruction. The main loop
// takes four K pairs per iteration with SME2 multi-vector loads (one for the
// activations, four for the weights), so sixteen independent FMOPAs hide the
// load latency; each tile still accumulates its K pairs in order.
//
// x0=weight panel ([K/2][64][2] FP16), x1=K pairs, x2=panel count,
// x3=[K/2][16][2] FP16 activations, x4=dst, x5=valid output columns from dst,
// x6=real rows (1..16), x7=dst row stride in bytes,
// x16=FP32 column scales for the first column, x19=FP32 row scales.
// x8=packed panel stride, x9=panel index, x10=Q pointer, x11=A pointer,
// x12=remaining K pairs, x13=row, x14/x15/x20=temporaries,
// x17=signal retries. The code falls through to the Go epilogue, which
// stores x0 as the result; it must not return directly.
//
// The sentinel in z31 detects Darwin's signal return behavior: the lower
// 128 bits are restored but upper Z lanes are cleared. ZA survives. The
// sentinel is checked after accumulation and again after the scaled stores,
// because those stores pass through Z registers. On a detected clear, the
// current output panel is recomputed from zero; its stores are idempotent.
	smstart
	ptrue p0.s
	ptrue p2.h
	ptrue pn8.h
	mov x17, #0
	lsl x8, x1, #8
	mov x9, #0
1:
	zero {za}
	fdup z31.s, #1.0
	mov x10, x0
	mov x11, x3
	mov x12, x1
2:
	cmp x12, #4
	b.lt 20f
	ld1h {z0.h-z3.h}, pn8/z, [x11]
	ld1h {z4.h-z7.h}, pn8/z, [x10]
	ld1h {z8.h-z11.h}, pn8/z, [x10, #4, mul vl]
	ld1h {z12.h-z15.h}, pn8/z, [x10, #8, mul vl]
	ld1h {z16.h-z19.h}, pn8/z, [x10, #12, mul vl]
	fmopa za0.s, p2/m, p2/m, z0.h, z4.h
	fmopa za1.s, p2/m, p2/m, z0.h, z5.h
	fmopa za2.s, p2/m, p2/m, z0.h, z6.h
	fmopa za3.s, p2/m, p2/m, z0.h, z7.h
	fmopa za0.s, p2/m, p2/m, z1.h, z8.h
	fmopa za1.s, p2/m, p2/m, z1.h, z9.h
	fmopa za2.s, p2/m, p2/m, z1.h, z10.h
	fmopa za3.s, p2/m, p2/m, z1.h, z11.h
	fmopa za0.s, p2/m, p2/m, z2.h, z12.h
	fmopa za1.s, p2/m, p2/m, z2.h, z13.h
	fmopa za2.s, p2/m, p2/m, z2.h, z14.h
	fmopa za3.s, p2/m, p2/m, z2.h, z15.h
	fmopa za0.s, p2/m, p2/m, z3.h, z16.h
	fmopa za1.s, p2/m, p2/m, z3.h, z17.h
	fmopa za2.s, p2/m, p2/m, z3.h, z18.h
	fmopa za3.s, p2/m, p2/m, z3.h, z19.h
	add x10, x10, #1024
	add x11, x11, #256
	sub x12, x12, #4
	b 2b
20:
	cbz x12, 3f
	ld1h {z0.h}, p2/z, [x11]
	ld1h {z4.h-z7.h}, pn8/z, [x10]
	fmopa za0.s, p2/m, p2/m, z0.h, z4.h
	fmopa za1.s, p2/m, p2/m, z0.h, z5.h
	fmopa za2.s, p2/m, p2/m, z0.h, z6.h
	fmopa za3.s, p2/m, p2/m, z0.h, z7.h
	add x10, x10, #256
	add x11, x11, #64
	sub x12, x12, #1
	b 20b
3:
	ptrue p3.s
	fcmeq p4.s, p3/z, z31.s, #0.0
	ptest p3, p4.b
	b.ne 8f
	cntp x14, p3, p0.s
	cmp x14, #16
	b.ne 8f
	add x15, x16, x9, lsl #8
	sub x14, x5, x9, lsl #6
	whilelt p1.s, xzr, x14
	ld1w {z16.s}, p1/z, [x15]
	sub x14, x14, #16
	whilelt p1.s, xzr, x14
	ld1w {z17.s}, p1/z, [x15, #1, mul vl]
	sub x14, x14, #16
	whilelt p1.s, xzr, x14
	ld1w {z18.s}, p1/z, [x15, #2, mul vl]
	sub x14, x14, #16
	whilelt p1.s, xzr, x14
	ld1w {z19.s}, p1/z, [x15, #3, mul vl]
	mov x13, #0
4:
	cmp x13, x6
	b.ge 7f
	add x20, x19, x13, lsl #2
	ld1rw {z20.s}, p0/z, [x20]
	madd x14, x13, x7, x4
	sub x15, x5, x9, lsl #6
	add x14, x14, x9, lsl #8
	whilelt p1.s, xzr, x15
	mova z1.s, p0/m, za0h.s[w13, 0]
	fmul z1.s, z1.s, z16.s
	fmul z1.s, z1.s, z20.s
	st1w {z1.s}, p1, [x14]
	sub x15, x15, #16
	whilelt p1.s, xzr, x15
	mova z2.s, p0/m, za1h.s[w13, 0]
	fmul z2.s, z2.s, z17.s
	fmul z2.s, z2.s, z20.s
	st1w {z2.s}, p1, [x14, #1, mul vl]
	sub x15, x15, #16
	whilelt p1.s, xzr, x15
	mova z3.s, p0/m, za2h.s[w13, 0]
	fmul z3.s, z3.s, z18.s
	fmul z3.s, z3.s, z20.s
	st1w {z3.s}, p1, [x14, #2, mul vl]
	sub x15, x15, #16
	whilelt p1.s, xzr, x15
	mova z1.s, p0/m, za3h.s[w13, 0]
	fmul z1.s, z1.s, z19.s
	fmul z1.s, z1.s, z20.s
	st1w {z1.s}, p1, [x14, #3, mul vl]
	add x13, x13, #1
	b 4b
7:
	fcmeq p4.s, p3/z, z31.s, #0.0
	ptest p3, p4.b
	b.ne 8f
	add x0, x0, x8
	add x9, x9, #1
	cmp x9, x2
	b.lt 1b
	smstop
	mov x0, x17
	b 9f
8:
	add x17, x17, #1
	b 1b
9:
