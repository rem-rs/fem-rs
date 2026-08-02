#!/usr/bin/env python3
"""Quick check: compare A/M elim matrices row by row (value multiset, ignoring
column order) between cpp_*_elim.txt and rust_*_elim.txt.

Usage: python compare_ex32_quick.py [A|M]
"""
import re, sys

def read_rows(path):
    rows = {}
    cur = None
    for l in open(path, encoding="utf-8", errors="replace"):
        l = l.strip()
        if not l:
            continue
        m = re.match(r"\[row (\d+)\]", l)
        if m:
            cur = int(m.group(1))
            rows[cur] = []
        for (c, v) in re.findall(r"\((\d+)\s*,\s*([-+0-9.eE]+)\)", l):
            rows[cur].append((int(c), float(v)))
    return rows

def multiset(entries):
    out = {}
    for c, v in entries:
        out[c] = out.get(c, 0.0) + v
    # normalize tiny
    return {c: (0.0 if abs(v) < 1e-14 else v) for c, v in out.items() if abs(v) > 1e-16}

def compare(a, b, name):
    n_mismatch = 0
    n_diag_diff = 0
    first = None
    for r in sorted(set(a) | set(b)):
        va = multiset(a.get(r, []))
        vb = multiset(b.get(r, []))
        if va != vb:
            n_mismatch += 1
            if first is None:
                first = (r, va, vb)
        # diag check
        da = va.get(r, 0.0)
        db = vb.get(r, 0.0)
        if abs(da - db) > 1e-12:
            n_diag_diff += 1
            if n_diag_diff <= 5:
                print(f"  row {r}: diag cpp={da:.12e} rust={db:.12e}")
    print(f"{name}: rows={len(a)}/{len(b)}  value-mismatch-rows={n_mismatch}  diag-diff-rows={n_diag_diff}")
    if first:
        r, va, vb = first
        print(f"  first mismatch row {r}:")
        print(f"    cpp : {va}")
        print(f"    rust: {vb}")

which = sys.argv[1] if len(sys.argv) > 1 else "A"
if which.upper() == "A":
    compare(read_rows("cpp_A_elim.txt"), read_rows("rust_A_elim.txt"), "A_elim")
else:
    compare(read_rows("cpp_M_elim.txt"), read_rows("rust_M_elim.txt"), "M_elim")
