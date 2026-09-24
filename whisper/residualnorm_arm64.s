// Code generated from asmsrc/residualnorm.S by internal/whispergemm/smesrc/gen.py; DO NOT EDIT.

//go:build arm64

#include "textflag.h"

// func residualNormNEON(row, dst, gamma, beta *float32, n int, add, bias *float32)
TEXT ·residualNormNEON(SB), NOSPLIT, $0-56
	MOVD	row+0(FP), R0
	MOVD	dst+8(FP), R1
	MOVD	gamma+16(FP), R2
	MOVD	beta+24(FP), R3
	MOVD	n+32(FP), R4
	MOVD	add+40(FP), R10
	MOVD	bias+48(FP), R11
	WORD	$0x6f00e410	// movi.2d v16, #0000000000000000
	WORD	$0x6f00e411	// movi.2d v17, #0000000000000000
	WORD	$0x6f00e412	// movi.2d v18, #0000000000000000
	WORD	$0x6f00e413	// movi.2d v19, #0000000000000000
	WORD	$0x6f00e41a	// movi.2d v26, #0000000000000000
	WORD	$0x6f00e41b	// movi.2d v27, #0000000000000000
	WORD	$0x6f00e41c	// movi.2d v28, #0000000000000000
	WORD	$0x6f00e41d	// movi.2d v29, #0000000000000000
	WORD	$0xaa0003e5	// mov x5, x0
	WORD	$0xaa0403e6	// mov x6, x4
	WORD	$0xaa0a03ed	// mov x13, x10
	WORD	$0xaa0b03ee	// mov x14, x11
	WORD	$0xad4004a0	// ldp q0, q1, [x5]
	WORD	$0xacc125a8	// ldp q8, q9, [x13], #0x20
	WORD	$0xb400008b	// cbz x11, 0x48
	WORD	$0xacc12dca	// ldp q10, q11, [x14], #0x20
	WORD	$0x4e2ad508	// fadd.4s v8, v8, v10
	WORD	$0x4e2bd529	// fadd.4s v9, v9, v11
	WORD	$0x4e28d400	// fadd.4s v0, v0, v8
	WORD	$0x4e29d421	// fadd.4s v1, v1, v9
	WORD	$0xac8104a0	// stp q0, q1, [x5], #0x20
	WORD	$0x0e617802	// fcvtl v2.2d, v0.2s
	WORD	$0x4e617803	// fcvtl2 v3.2d, v0.4s
	WORD	$0x0e617824	// fcvtl v4.2d, v1.2s
	WORD	$0x4e617825	// fcvtl2 v5.2d, v1.4s
	WORD	$0x4e62d610	// fadd.2d v16, v16, v2
	WORD	$0x4e63d631	// fadd.2d v17, v17, v3
	WORD	$0x4e64d652	// fadd.2d v18, v18, v4
	WORD	$0x4e65d673	// fadd.2d v19, v19, v5
	WORD	$0x4e62cc5a	// fmla.2d v26, v2, v2
	WORD	$0x4e63cc7b	// fmla.2d v27, v3, v3
	WORD	$0x4e64cc9c	// fmla.2d v28, v4, v4
	WORD	$0x4e65ccbd	// fmla.2d v29, v5, v5
	WORD	$0xf10020c6	// subs x6, x6, #0x8
	WORD	$0x54fffd4c	// b.gt 0x30
	WORD	$0x4e71d610	// fadd.2d v16, v16, v17
	WORD	$0x4e73d652	// fadd.2d v18, v18, v19
	WORD	$0x4e72d610	// fadd.2d v16, v16, v18
	WORD	$0x7e70da10	// faddp.2d d16, v16
	WORD	$0x9e620094	// scvtf d20, x4
	WORD	$0x1e741a10	// fdiv d16, d16, d20
	WORD	$0x4e080615	// dup.2d v21, v16[0]
	WORD	$0x4e7bd75a	// fadd.2d v26, v26, v27
	WORD	$0x4e7dd79c	// fadd.2d v28, v28, v29
	WORD	$0x4e7cd75a	// fadd.2d v26, v26, v28
	WORD	$0x7e70db51	// faddp.2d d17, v26
	WORD	$0x1e741a31	// fdiv d17, d17, d20
	WORD	$0x1e700a12	// fmul d18, d16, d16
	WORD	$0x1e723a30	// fsub d16, d17, d18
	WORD	$0xd28d1e27	// mov x7, #0x68f1 ; =26865
	WORD	$0xf2b11c67	// movk x7, #0x88e3, lsl #16
	WORD	$0xf2df16a7	// movk x7, #0xf8b5, lsl #32
	WORD	$0xf2e7dc87	// movk x7, #0x3ee4, lsl #48
	WORD	$0x9e6700f6	// fmov d22, x7
	WORD	$0x1e762a10	// fadd d16, d16, d22
	WORD	$0x1e61c210	// fsqrt d16, d16
	WORD	$0x1e6e1017	// fmov d23, #1.00000000
	WORD	$0x1e701af7	// fdiv d23, d23, d16
	WORD	$0x4e0806f8	// dup.2d v24, v23[0]
	WORD	$0xaa0403e6	// mov x6, x4
	WORD	$0x3cc10400	// ldr q0, [x0], #0x10
	WORD	$0x3cc10441	// ldr q1, [x2], #0x10
	WORD	$0x3cc10466	// ldr q6, [x3], #0x10
	WORD	$0x0e617802	// fcvtl v2.2d, v0.2s
	WORD	$0x4e617803	// fcvtl2 v3.2d, v0.4s
	WORD	$0x0e617824	// fcvtl v4.2d, v1.2s
	WORD	$0x4e617825	// fcvtl2 v5.2d, v1.4s
	WORD	$0x0e6178c7	// fcvtl v7.2d, v6.2s
	WORD	$0x4e6178c6	// fcvtl2 v6.2d, v6.4s
	WORD	$0x4ef5d442	// fsub.2d v2, v2, v21
	WORD	$0x4ef5d463	// fsub.2d v3, v3, v21
	WORD	$0x6e78dc42	// fmul.2d v2, v2, v24
	WORD	$0x6e78dc63	// fmul.2d v3, v3, v24
	WORD	$0x4e64cc47	// fmla.2d v7, v2, v4
	WORD	$0x4e65cc66	// fmla.2d v6, v3, v5
	WORD	$0x0e6168e0	// fcvtn v0.2s, v7.2d
	WORD	$0x4e6168c0	// fcvtn2 v0.4s, v6.2d
	WORD	$0x3c810420	// str q0, [x1], #0x10
	WORD	$0xf10010c6	// subs x6, x6, #0x4
	WORD	$0x54fffdac	// b.gt 0xf0
	RET
