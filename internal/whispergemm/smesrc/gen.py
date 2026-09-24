#!/usr/bin/env python3
# Assemble an SME source with clang and print Go WORD directives.
import re, subprocess, sys, tempfile

src = sys.argv[1]
with tempfile.NamedTemporaryFile(suffix=".o") as obj:
    subprocess.check_call(["clang", "-c", "-target", "arm64-apple-macos",
                           "-march=armv9-a+sme2", src, "-o", obj.name])
    out = subprocess.check_output(["xcrun", "llvm-objdump", "-d", obj.name]).decode()
for line in out.splitlines():
    m = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(.*)", line)
    if m:
        text = re.sub(r"\s*<.*>", "", re.sub(r"\s+", " ", m.group(2))).strip()
        print("\tWORD\t$0x%s\t// %s" % (m.group(1), text))
