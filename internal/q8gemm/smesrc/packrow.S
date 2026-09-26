// Same scale/multiply/FCVT sequence as packpairs.S, with contiguous stores.
// x0=dst, x1=src, x2=float32 count (positive multiple of eight), v0.s[0]=scale.
    dup v31.4s, v0.s[0]
1:
    ld1 {v0.4s, v1.4s}, [x1], #32
    fmul v0.4s, v0.4s, v31.4s
    fmul v1.4s, v1.4s, v31.4s
    fcvtn v2.4h, v0.4s
    fcvtn2 v2.8h, v1.4s
    st1 {v2.8h}, [x0], #16
    sub x2, x2, #8
    cbnz x2, 1b
