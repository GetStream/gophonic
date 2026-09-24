// Code generated from asmsrc/gelu.S by internal/whispergemm/smesrc/gen.py; DO NOT EDIT.

//go:build arm64

#include "textflag.h"

// func geluNEON(values, bias *float32, n int, table *[1025][2]float32)
TEXT ·geluNEON(SB), NOSPLIT, $0-32
	MOVD	values+0(FP), R0
	MOVD	bias+8(FP), R1
	MOVD	n+16(FP), R2
	MOVD	table+24(FP), R3
	WORD	$0x52a8f009	// mov w9, #0x47800000 ; =1199570944
	WORD	$0x4e040d30	// dup.4s v16, w9
	WORD	$0x4eb01e11	// mov.16b v17, v16
	WORD	$0x4f01f413	// fmov.4s v19, #8.00000000
	WORD	$0x4f03f414	// fmov.4s v20, #0.50000000
	WORD	$0x52a7c549	// mov w9, #0x3e2a0000 ; =1042939904
	WORD	$0x72955569	// movk w9, #0xaaab
	WORD	$0x4e040d35	// dup.4s v21, w9
	WORD	$0x4f03f616	// fmov.4s v22, #1.00000000
	WORD	$0x52808009	// mov w9, #0x400 ; =1024
	WORD	$0x4e040d37	// dup.4s v23, w9
	WORD	$0x3dc00000	// ldr q0, [x0]
	WORD	$0xb4000061	// cbz x1, 0x3c
	WORD	$0x3cc10421	// ldr q1, [x1], #0x10
	WORD	$0x4e21d400	// fadd.4s v0, v0, v1
	WORD	$0x4ea0f802	// fabs.4s v2, v0
	WORD	$0x6e33e443	// fcmge.4s v3, v2, v19
	WORD	$0x4eb3c444	// fminnm.4s v4, v2, v19
	WORD	$0x4e30d485	// fadd.4s v5, v4, v16
	WORD	$0x6eb184a6	// sub.4s v6, v5, v17
	WORD	$0x6eb76cc6	// umin.4s v6, v6, v23
	WORD	$0x4eb0d4a7	// fsub.4s v7, v5, v16
	WORD	$0x4ea7d488	// fsub.4s v8, v4, v7
	WORD	$0x0e043cc9	// mov.s w9, v6[0]
	WORD	$0x0e0c3cca	// mov.s w10, v6[1]
	WORD	$0x0e143ccb	// mov.s w11, v6[2]
	WORD	$0x0e1c3ccc	// mov.s w12, v6[3]
	WORD	$0x5280800d	// mov w13, #0x400 ; =1024
	WORD	$0x7110013f	// cmp w9, #0x400
	WORD	$0x1a8d3129	// csel w9, w9, w13, lo
	WORD	$0x7110015f	// cmp w10, #0x400
	WORD	$0x1a8d314a	// csel w10, w10, w13, lo
	WORD	$0x7110017f	// cmp w11, #0x400
	WORD	$0x1a8d316b	// csel w11, w11, w13, lo
	WORD	$0x7110019f	// cmp w12, #0x400
	WORD	$0x1a8d318c	// csel w12, w12, w13, lo
	WORD	$0x8b090c69	// add x9, x3, x9, lsl #3
	WORD	$0x8b0a0c6a	// add x10, x3, x10, lsl #3
	WORD	$0x8b0b0c6b	// add x11, x3, x11, lsl #3
	WORD	$0x8b0c0c6c	// add x12, x3, x12, lsl #3
	WORD	$0x0d408538	// ld1.d { v24 }[0], [x9]
	WORD	$0x4d408558	// ld1.d { v24 }[1], [x10]
	WORD	$0x0d408579	// ld1.d { v25 }[0], [x11]
	WORD	$0x4d408599	// ld1.d { v25 }[1], [x12]
	WORD	$0x4e991b1a	// uzp1.4s v26, v24, v25
	WORD	$0x4e995b1b	// uzp2.4s v27, v24, v25
	WORD	$0x6e27dcfc	// fmul.4s v28, v7, v7
	WORD	$0x4eb6d79c	// fsub.4s v28, v28, v22
	WORD	$0x6e35df9c	// fmul.4s v28, v28, v21
	WORD	$0x6e34dcfd	// fmul.4s v29, v7, v20
	WORD	$0x6ea0fbbd	// fneg.4s v29, v29
	WORD	$0x4e3ccd1d	// fmla.4s v29, v8, v28
	WORD	$0x4eb61ede	// mov.16b v30, v22
	WORD	$0x4e28cfbe	// fmla.4s v30, v29, v8
	WORD	$0x6e28df7f	// fmul.4s v31, v27, v8
	WORD	$0x6ea0fbff	// fneg.4s v31, v31
	WORD	$0x4e3ecffa	// fmla.4s v26, v31, v30
	WORD	$0x4e631f5a	// bic.16b v26, v26, v3
	WORD	$0x4ea0c803	// fcmgt.4s v3, v0, #0.0
	WORD	$0x4ebad6df	// fsub.4s v31, v22, v26
	WORD	$0x6e7a1fe3	// bsl.16b v3, v31, v26
	WORD	$0x6e23dc00	// fmul.4s v0, v0, v3
	WORD	$0x3c810400	// str q0, [x0], #0x10
	WORD	$0xf1001042	// subs x2, x2, #0x4
	WORD	$0x54fff96c	// b.gt 0x2c
	RET
