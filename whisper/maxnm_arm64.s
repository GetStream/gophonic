// Code generated from asmsrc/maxnm.S by internal/whispergemm/smesrc/gen.py; DO NOT EDIT.

//go:build arm64

#include "textflag.h"

// func maxNumNEON(x *float32, n int) float32
TEXT ·maxNumNEON(SB), NOSPLIT, $0-20
	MOVD	x+0(FP), R0
	MOVD	n+8(FP), R1
	WORD	$0xad400400	// ldp q0, q1, [x0]
	WORD	$0xad410c02	// ldp q2, q3, [x0, #0x20]
	WORD	$0x91010002	// add x2, x0, #0x40
	WORD	$0xf1004021	// subs x1, x1, #0x10
	WORD	$0x5400012d	// b.le 0x34
	WORD	$0xacc11444	// ldp q4, q5, [x2], #0x20
	WORD	$0xacc11c46	// ldp q6, q7, [x2], #0x20
	WORD	$0x4e24c400	// fmaxnm.4s v0, v0, v4
	WORD	$0x4e25c421	// fmaxnm.4s v1, v1, v5
	WORD	$0x4e26c442	// fmaxnm.4s v2, v2, v6
	WORD	$0x4e27c463	// fmaxnm.4s v3, v3, v7
	WORD	$0xf1004021	// subs x1, x1, #0x10
	WORD	$0x54ffff2c	// b.gt 0x14
	WORD	$0x4e21c400	// fmaxnm.4s v0, v0, v1
	WORD	$0x4e23c442	// fmaxnm.4s v2, v2, v3
	WORD	$0x4e22c400	// fmaxnm.4s v0, v0, v2
	WORD	$0x6e30c800	// fmaxnmv.4s s0, v0
	FMOVS	F0, ret+16(FP)
	RET
