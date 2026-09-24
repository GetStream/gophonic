#!/usr/bin/env python3
"""Emit the SME GEMV kernels (vector.S) for FP32 and FP16 packed weights.

Inputs (both kernels):
  x0 = x, x1 = K, x2 = packed weights, x3 = N, x4 = y, x5 = Kp (K rounded to 4)
Output: x0 = retries.

Packed layout: [chunk][Kp][G][64] where a chunk holds G<=8 groups of 64
outputs. FP16 groups interleave outputs so FCVT/FCVTLT of two .h vectors
produce outputs 0-15, 16-31, 32-47, 48-63. Accumulators live in ZA vector
groups, which survive Darwin signal delivery; Z registers do not, so a z31
sentinel detects clobbering and the chunk (or its store phase) is redone.
"""
import sys

def block(half, g, j, imm):
    """Load one 64-output group at k and accumulate x[k]*w into ZA group."""
    out = []
    if half:
        a, d = ("z0", "z4") if g % 2 == 0 else ("z2", "z12")
        a2 = "z%d" % (int(a[1:]) + 1)
        dd = int(d[1:])
        out += [f"ld1h {{{a}.h-{a2}.h}}, pn9/z, [x11]",
                "add x11, x11, #128",
                f"fcvt z{dd}.s, p0/m, {a}.h",
                f"fcvtlt z{dd+1}.s, p0/m, {a}.h",
                f"fcvt z{dd+2}.s, p0/m, {a2}.h",
                f"fcvtlt z{dd+3}.s, p0/m, {a2}.h"]
    else:
        dd = 4 if g % 2 == 0 else 12
        out += [f"ld1w {{z{dd}.s-z{dd+3}.s}}, pn8/z, [x11]",
                "add x11, x11, #256"]
    out.append(f"fmla za.s[w8, {imm}, vgx4], {{z{dd}.s-z{dd+3}.s}}, z8.s[{j}]")
    return out

def kernel(half):
    L = []
    e = L.append
    e("smstart")
    e("ptrue p0.s")
    e("ptrue pn8.s")
    e("ptrue pn9.h")
    e("mov x17, #0")
    e("add x13, x3, #63")
    e("lsr x13, x13, #6")          # groups remaining
    e("mov x6, #0")                # output index of chunk
    e("mov x16, x2")               # chunk weight start
    e("1:")                        # chunk loop
    e("mov x7, #8")
    e("cmp x13, #8")
    e("csel x7, x13, x7, lt")      # G
    e("2:")                        # compute (retry target)
    e("zero {za}")
    e("fdup z31.s, #1.0")
    e("mov x9, #0")
    e("mov x10, x0")
    e("mov x11, x16")
    e("mov w8, #0")
    e("cmp x7, #8")
    e("b.ne 5f")
    # full chunk: unrolled
    e("3:")
    e("whilelt p1.s, x9, x1")
    e("ld1rqw {z8.s}, p1/z, [x10]")
    for j in range(4):
        for g in range(8):
            L.extend(block(half, g, j, g))
    e("add x10, x10, #16")
    e("add x9, x9, #4")
    e("cmp x9, x5")
    e("b.lt 3b")
    e("b 7f")
    # tail chunk: loop over groups
    e("5:")
    e("whilelt p1.s, x9, x1")
    e("ld1rqw {z8.s}, p1/z, [x10]")
    for j in range(4):
        e("mov w8, #0")
        e(f"6{j}:")
        L.extend(block(half, 0, j, 0))
        e("add w8, w8, #1")
        e("cmp x8, x7")
        e(f"b.lt 6{j}b")
    e("add x10, x10, #16")
    e("add x9, x9, #4")
    e("cmp x9, x5")
    e("b.lt 5b")
    e("7:")                        # check compute
    e("ptrue p3.s")
    e("fcmeq p4.s, p3/z, z31.s, #0.0")
    e("ptest p3, p4.b")
    e("b.ne 71f")
    e("cntp x9, p3, p0.s")
    e("cmp x9, #16")
    e("b.ne 71f")
    e("cntp x9, pn8.s, vlx4")
    e("cmp x9, #64")
    e("b.ne 71f")
    e("cntp x9, pn9.h, vlx2")
    e("cmp x9, #64")
    e("b.eq 8f")
    e("71:")
    e("add x17, x17, #1")
    e("ptrue p0.s")
    e("ptrue pn8.s")
    e("ptrue pn9.h")
    e("b 2b")
    e("8:")                        # store phase (retry target)
    e("fdup z31.s, #1.0")
    e("mov w8, #0")
    e("mov x9, x6")
    e("add x10, x4, x6, lsl #2")
    e("9:")
    e("mova {z0.s-z3.s}, za.s[w8, 0, vgx4]")
    e("whilelt pn10.s, x9, x3, vlx4")
    e("st1w {z0.s-z3.s}, pn10, [x10]")
    e("add x10, x10, #256")
    e("add x9, x9, #64")
    e("add w8, w8, #1")
    e("cmp x8, x7")
    e("b.lt 9b")
    e("ptrue p3.s")
    e("fcmeq p4.s, p3/z, z31.s, #0.0")
    e("ptest p3, p4.b")
    e("b.eq 10f")
    e("add x17, x17, #1")
    e("b 8b")
    e("10:")
    # advance: weights += Kp*G*64*elem, outputs += G*64
    e("mul x9, x5, x7")
    e("lsl x9, x9, #%d" % (7 if half else 8))
    e("add x16, x16, x9")
    e("add x6, x6, x7, lsl #6")
    e("sub x13, x13, x7")
    e("cbnz x13, 1b")
    e("smstop")
    e("mov x0, x17")
    return "\n".join("\t" + l if not l.endswith(":") else l for l in L) + "\n"

open(sys.argv[1] + "_f32.S", "w").write(kernel(False))
open(sys.argv[1] + "_f16.S", "w").write(kernel(True))
