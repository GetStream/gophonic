// Code generated from asmsrc/attnprep.S by internal/whispergemm/smesrc/gen.py; DO NOT EDIT.

//go:build arm64

#include "textflag.h"

// func attnPrepNEON(q, k, v, qbias, vbias *float32, n int, scale float32)
TEXT ·attnPrepNEON(SB), NOSPLIT, $0-52
	MOVD	q+0(FP), R0
	MOVD	k+8(FP), R1
	MOVD	v+16(FP), R2
	MOVD	qbias+24(FP), R3
	MOVD	vbias+32(FP), R4
	MOVD	n+40(FP), R5
	FMOVS	scale+48(FP), F0
	WORD	$0x4e04041f	// dup.4s v31, v0[0]
	WORD	$0x3dc00000	// ldr q0, [x0]
	WORD	$0x3cc10461	// ldr q1, [x3], #0x10
	WORD	$0x4e21d400	// fadd.4s v0, v0, v1
	WORD	$0x6e3fdc00	// fmul.4s v0, v0, v31
	WORD	$0x3c810400	// str q0, [x0], #0x10
	WORD	$0x3dc00022	// ldr q2, [x1]
	WORD	$0x6e3fdc42	// fmul.4s v2, v2, v31
	WORD	$0x3c810422	// str q2, [x1], #0x10
	WORD	$0x3dc00043	// ldr q3, [x2]
	WORD	$0x3cc10484	// ldr q4, [x4], #0x10
	WORD	$0x4e24d463	// fadd.4s v3, v3, v4
	WORD	$0x3c810443	// str q3, [x2], #0x10
	WORD	$0xf10010a5	// subs x5, x5, #0x4
	WORD	$0x54fffe6c	// b.gt 0x4
	RET
