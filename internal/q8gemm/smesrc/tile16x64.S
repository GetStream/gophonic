// 16 activation rows by 64 output columns, one FP32 ZA tile per 16 columns.
// x0=q panel, x1=K, x2=panel count, x3=[K][16] FP32 activations,
// x4=dst, x5=N, x6=real rows (1..16).
// x8=packed panel stride, x9=panel index, x10=Q pointer, x11=A pointer,
// x12=remaining K, x13=row, x14=output pointer, x15=remaining N,
// x17=signal retries.
//
// The sentinel in z31 detects Darwin's signal return behavior: the lower
// 128 bits are restored but upper Z lanes are cleared. ZA survives. On a
// detected clear, the current output panel is recomputed from zero.
	smstart
	ptrue p0.s
	mov x17, #0
	lsl x8, x1, #6
	lsl x7, x5, #2
	mov x9, #0
1:
	zero {za}
	fdup z31.s, #1.0
	mov x10, x0
	mov x11, x3
	mov x12, x1
2:
	cbz x12, 3f
	ld1w {z0.s}, p0/z, [x11]
	ld1sb {z4.s}, p0/z, [x10]
	scvtf z4.s, p0/m, z4.s
	fmopa za0.s, p0/m, p0/m, z0.s, z4.s
	add x10, x10, #16
	ld1sb {z4.s}, p0/z, [x10]
	scvtf z4.s, p0/m, z4.s
	fmopa za1.s, p0/m, p0/m, z0.s, z4.s
	add x10, x10, #16
	ld1sb {z4.s}, p0/z, [x10]
	scvtf z4.s, p0/m, z4.s
	fmopa za2.s, p0/m, p0/m, z0.s, z4.s
	add x10, x10, #16
	ld1sb {z4.s}, p0/z, [x10]
	scvtf z4.s, p0/m, z4.s
	fmopa za3.s, p0/m, p0/m, z0.s, z4.s
	add x10, x10, #16
	add x11, x11, #64
	sub x12, x12, #1
	b 2b
3:
	ptrue p3.s
	fcmeq p4.s, p3/z, z31.s, #0.0
	ptest p3, p4.b
	b.ne 8f
	cntp x14, p3, p0.s
	cmp x14, #16
	b.ne 8f
	mov x13, #0
4:
	cmp x13, x6
	b.ge 7f
	madd x14, x13, x7, x4
	sub x15, x5, x9, lsl #6
	add x14, x14, x9, lsl #8
	whilelt p1.s, xzr, x15
	st1w {za0h.s[w13, 0]}, p1, [x14]
	add x14, x14, #64
	sub x15, x15, #16
	whilelt p1.s, xzr, x15
	st1w {za1h.s[w13, 0]}, p1, [x14]
	add x14, x14, #64
	sub x15, x15, #16
	whilelt p1.s, xzr, x15
	st1w {za2h.s[w13, 0]}, p1, [x14]
	add x14, x14, #64
	sub x15, x15, #16
	whilelt p1.s, xzr, x15
	st1w {za3h.s[w13, 0]}, p1, [x14]
	add x13, x13, #1
	b 4b
7:
	add x0, x0, x8
	add x9, x9, #1
	cmp x9, x2
	b.lt 1b
	smstop
	mov x0, x17
	ret
8:
	add x17, x17, #1
	b 1b
