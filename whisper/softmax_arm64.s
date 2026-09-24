// Code generated from asmsrc/softmax.S by internal/whispergemm/smesrc/gen.py; DO NOT EDIT.

//go:build arm64

#include "textflag.h"

// func softmaxExpNEON(x *float32, n int, constants *[11]float32) float32
TEXT ·softmaxExpNEON(SB), NOSPLIT, $0-28
	MOVD	x+0(FP), R0
	MOVD	n+8(FP), R1
	MOVD	constants+16(FP), R2
	WORD	$0x3dc00010	// ldr q16, [x0]
	WORD	$0x4eb01e11	// mov.16b v17, v16
	WORD	$0xaa0003e3	// mov x3, x0
	WORD	$0xaa0103e4	// mov x4, x1
	WORD	$0xf100209f	// cmp x4, #0x8
	WORD	$0x540000eb	// b.lt 0x30
	WORD	$0xacc10460	// ldp q0, q1, [x3], #0x20
	WORD	$0x4e20f610	// fmax.4s v16, v16, v0
	WORD	$0x4e21f631	// fmax.4s v17, v17, v1
	WORD	$0xd1002084	// sub x4, x4, #0x8
	WORD	$0xf100209f	// cmp x4, #0x8
	WORD	$0x54ffff6a	// b.ge 0x18
	WORD	$0xb4000064	// cbz x4, 0x3c
	WORD	$0x3cc10460	// ldr q0, [x3], #0x10
	WORD	$0x4e20f610	// fmax.4s v16, v16, v0
	WORD	$0x4e31f610	// fmax.4s v16, v16, v17
	WORD	$0x6e30fa10	// fmaxv.4s s16, v16
	WORD	$0x4e040610	// dup.4s v16, v16[0]
	WORD	$0x4ddfc851	// ld1r.4s { v17 }, [x2], #4
	WORD	$0x4ddfc852	// ld1r.4s { v18 }, [x2], #4
	WORD	$0x4ddfc853	// ld1r.4s { v19 }, [x2], #4
	WORD	$0x4ddfc854	// ld1r.4s { v20 }, [x2], #4
	WORD	$0x4ddfc855	// ld1r.4s { v21 }, [x2], #4
	WORD	$0x4ddfc856	// ld1r.4s { v22 }, [x2], #4
	WORD	$0x4ddfc857	// ld1r.4s { v23 }, [x2], #4
	WORD	$0x4ddfc858	// ld1r.4s { v24 }, [x2], #4
	WORD	$0x4ddfc859	// ld1r.4s { v25 }, [x2], #4
	WORD	$0x4ddfc85a	// ld1r.4s { v26 }, [x2], #4
	WORD	$0x4ddfc85b	// ld1r.4s { v27 }, [x2], #4
	WORD	$0x4f00041c	// movi.4s v28, #0x0
	WORD	$0x4f00041d	// movi.4s v29, #0x0
	WORD	$0xaa0003e3	// mov x3, x0
	WORD	$0xaa0103e4	// mov x4, x1
	WORD	$0xf100209f	// cmp x4, #0x8
	WORD	$0x5400060b	// b.lt 0x148
	WORD	$0xad402060	// ldp q0, q8, [x3]
	WORD	$0x4eb0d400	// fsub.4s v0, v0, v16
	WORD	$0x4eb0d508	// fsub.4s v8, v8, v16
	WORD	$0x4e31f400	// fmax.4s v0, v0, v17
	WORD	$0x4e31f508	// fmax.4s v8, v8, v17
	WORD	$0x4eb21e41	// mov.16b v1, v18
	WORD	$0x4eb21e49	// mov.16b v9, v18
	WORD	$0x4e33cc01	// fmla.4s v1, v0, v19
	WORD	$0x4e33cd09	// fmla.4s v9, v8, v19
	WORD	$0x4eb2d422	// fsub.4s v2, v1, v18
	WORD	$0x4eb2d52a	// fsub.4s v10, v9, v18
	WORD	$0x4ea01c03	// mov.16b v3, v0
	WORD	$0x4ea81d0b	// mov.16b v11, v8
	WORD	$0x4e34cc43	// fmla.4s v3, v2, v20
	WORD	$0x4e34cd4b	// fmla.4s v11, v10, v20
	WORD	$0x4e35cc43	// fmla.4s v3, v2, v21
	WORD	$0x4e35cd4b	// fmla.4s v11, v10, v21
	WORD	$0x4f375421	// shl.4s v1, v1, #0x17
	WORD	$0x4f375529	// shl.4s v9, v9, #0x17
	WORD	$0x4ebb8421	// add.4s v1, v1, v27
	WORD	$0x4ebb8529	// add.4s v9, v9, v27
	WORD	$0x6e23dc64	// fmul.4s v4, v3, v3
	WORD	$0x6e2bdd6c	// fmul.4s v12, v11, v11
	WORD	$0x4eb91f25	// mov.16b v5, v25
	WORD	$0x4eb91f2d	// mov.16b v13, v25
	WORD	$0x4e23cf45	// fmla.4s v5, v26, v3
	WORD	$0x4e2bcf4d	// fmla.4s v13, v26, v11
	WORD	$0x4eb71ee6	// mov.16b v6, v23
	WORD	$0x4eb71eee	// mov.16b v14, v23
	WORD	$0x4e23cf06	// fmla.4s v6, v24, v3
	WORD	$0x4e2bcf0e	// fmla.4s v14, v24, v11
	WORD	$0x4e24cca6	// fmla.4s v6, v5, v4
	WORD	$0x4e2ccdae	// fmla.4s v14, v13, v12
	WORD	$0x6e23dec7	// fmul.4s v7, v22, v3
	WORD	$0x6e2bdecf	// fmul.4s v15, v22, v11
	WORD	$0x4e24ccc7	// fmla.4s v7, v6, v4
	WORD	$0x4e2ccdcf	// fmla.4s v15, v14, v12
	WORD	$0x4ea11c20	// mov.16b v0, v1
	WORD	$0x4ea91d28	// mov.16b v8, v9
	WORD	$0x4e21cce0	// fmla.4s v0, v7, v1
	WORD	$0x4e29cde8	// fmla.4s v8, v15, v9
	WORD	$0xac812060	// stp q0, q8, [x3], #0x20
	WORD	$0x4e20d79c	// fadd.4s v28, v28, v0
	WORD	$0x4e28d7bd	// fadd.4s v29, v29, v8
	WORD	$0xd1002084	// sub x4, x4, #0x8
	WORD	$0xf100209f	// cmp x4, #0x8
	WORD	$0x54fffa4a	// b.ge 0x8c
	WORD	$0xb4000304	// cbz x4, 0x1a8
	WORD	$0x3dc00060	// ldr q0, [x3]
	WORD	$0x4eb0d400	// fsub.4s v0, v0, v16
	WORD	$0x4e31f400	// fmax.4s v0, v0, v17
	WORD	$0x4eb21e41	// mov.16b v1, v18
	WORD	$0x4e33cc01	// fmla.4s v1, v0, v19
	WORD	$0x4eb2d422	// fsub.4s v2, v1, v18
	WORD	$0x4ea01c03	// mov.16b v3, v0
	WORD	$0x4e34cc43	// fmla.4s v3, v2, v20
	WORD	$0x4e35cc43	// fmla.4s v3, v2, v21
	WORD	$0x4f375421	// shl.4s v1, v1, #0x17
	WORD	$0x4ebb8421	// add.4s v1, v1, v27
	WORD	$0x6e23dc64	// fmul.4s v4, v3, v3
	WORD	$0x4eb91f25	// mov.16b v5, v25
	WORD	$0x4e23cf45	// fmla.4s v5, v26, v3
	WORD	$0x4eb71ee6	// mov.16b v6, v23
	WORD	$0x4e23cf06	// fmla.4s v6, v24, v3
	WORD	$0x4e24cca6	// fmla.4s v6, v5, v4
	WORD	$0x6e23dec7	// fmul.4s v7, v22, v3
	WORD	$0x4e24ccc7	// fmla.4s v7, v6, v4
	WORD	$0x4ea11c20	// mov.16b v0, v1
	WORD	$0x4e21cce0	// fmla.4s v0, v7, v1
	WORD	$0x3c810460	// str q0, [x3], #0x10
	WORD	$0x4e20d79c	// fadd.4s v28, v28, v0
	WORD	$0x4e3dd79c	// fadd.4s v28, v28, v29
	WORD	$0x6e3cd79c	// faddp.4s v28, v28, v28
	WORD	$0x7e30db80	// faddp.2s s0, v28
	FMOVS	F0, ret+24(FP)
	RET
