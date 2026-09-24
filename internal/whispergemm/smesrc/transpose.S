// smeTransposePack: pack an N-by-K source (rows N, stride elements apart)
// into PackedB panels [ceil(N/16)][K][16], i.e. dst[p][k][j] = src[(16p+j)*stride + k].
// x0 = src, x1 = stride bytes, x2 = n, x3 = k, x4 = dst
// Uses only ZA, predicates and general registers, so signals cannot corrupt it.
	smstart
	ptrue p0.s
	mov x5, #0                  // panel row base n0
1:
	sub x6, x2, x5
	cmp x6, #16
	mov x7, #16
	csel x6, x6, x7, lt         // rows in this panel
	mov x8, #0                  // k0
2:
	zero {za0.s}
	whilelt p1.s, x8, x3        // valid k in this block
	// load up to 16 source rows as horizontal slices
	madd x9, x5, x1, x0         // &src[n0][0]
	add x9, x9, x8, lsl #2      // + k0
	mov w12, #0
3:
	cmp x12, x6
	b.ge 4f
	ld1w {za0h.s[w12, 0]}, p1/z, [x9]
	add x9, x9, x1
	add w12, w12, #1
	b 3b
4:
	// store vertical slices k0..k0+15 (only valid k)
	sub x10, x3, x8
	cmp x10, #16
	csel x10, x10, x7, lt       // k rows to store
	mul x11, x5, x3             // panel base: n0*K floats
	add x11, x11, x8, lsl #4    // + k0*16 floats
	add x11, x4, x11, lsl #2
	mov w12, #0
5:
	cmp x12, x10
	b.ge 6f
	st1w {za0v.s[w12, 0]}, p0, [x11]
	add x11, x11, #64
	add w12, w12, #1
	b 5b
6:
	add x8, x8, #16
	cmp x8, x3
	b.lt 2b
	add x5, x5, #16
	cmp x5, x2
	b.lt 1b
	smstop
