#!/usr/bin/env python3
"""Compare ex27 linear systems (C++ vs Rust) under the vertex-coordinate permutation.

C++ side: cpp_A.txt / cpp_b.txt (custom "[row N] (col,val) ..." format,
true-dof = vertex numbering).  Rust side: ex27_rust_A.txt / ex27_rust_b.txt
(same format, Rust vertex numbering).  The permutation C++-vertex -> Rust-vertex
comes from coordinate matching via cpp_verts.txt + refined.mesh.

Usage: compare_ex27_systems.py
"""
import re

def read_dump(path):
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
            continue
        if cur is None:
            continue
        for (c, v) in re.findall(r"\((\d+),([-+0-9.eE]+)\)", l):
            rows[cur].append((int(c), float(v)))
    n = max(rows) + 1 if rows else 0
    mat = [[0.0] * n for _ in range(n)]
    for r, entries in rows.items():
        for (c, v) in entries:
            mat[r][c] = v
    return mat, n

def read_vec(path):
    vals = []
    for l in open(path):
        s = l.strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except ValueError:
            continue
    return vals

def main():
    # C++ vertex i -> (x,y); Rust vertex i -> (x,y)
    cpp_xy = []
    for l in open("cpp_verts.txt"):
        p = l.split()
        if len(p) == 4:
            cpp_xy.append((float(p[1]), float(p[2])))
    rv = []
    for l in open("refined.mesh"):
        s = l.strip()
        if s == "vertices":
            break
    lines = [l.split("#", 1)[0].strip() for l in open("refined.mesh")]
    lines = [l for l in lines if l]
    iv = lines.index("vertices")
    nv = int(lines[iv + 1])
    j = iv + 2
    while len(rv) < nv:
        toks = lines[j].split()
        try:
            rv.append((float(toks[0]), float(toks[1])))
        except (ValueError, IndexError):
            pass
        j += 1
    r_by = {}
    for i, v in enumerate(rv):
        r_by.setdefault((round(v[0], 6), round(v[1], 6)), []).append(i)
    perm = []  # cpp vertex i -> rust vertex index
    for (x, y) in cpp_xy:
        cands = r_by[(round(x, 6), round(y, 6))]
        perm.append(cands[0])

    A_c, n = read_dump("cpp_A.txt")
    A_r, n2 = read_dump("ex27_rust_A.txt")
    b_c = read_vec("cpp_b.txt")
    b_r = read_vec("ex27_rust_b.txt")
    x_c = read_vec("cpp_x.txt")
    print(f"C++ A {n}x{n}  Rust A {n2}x{n2};  b: {len(b_c)} vs {len(b_r)};  x: {len(x_c)}")
    assert n == n2 == len(b_c) == len(b_r), "size mismatch"

    # compare A under permutation: A_c[i][j] vs A_r[perm[i]][perm[j]]
    maxd = 0.0
    worst = None
    nz_c = 0
    nz_r = 0
    for i in range(n):
        pi = perm[i]
        for j in range(n):
            if A_c[i][j] != 0.0:
                nz_c += 1
            if A_r[pi][perm[j]] != 0.0:
                nz_r += 1
            d = abs(A_c[i][j] - A_r[pi][perm[j]])
            if d > maxd:
                maxd = d
                worst = (i, j, A_c[i][j], A_r[pi][perm[j]])
    print(f"nnz: C++={nz_c} Rust(perm)={nz_r}")
    print(f"A max|diff| = {maxd:.3e} at ({worst[0]},{worst[1]}) cpp={worst[2]:.6f} rust={worst[3]:.6f}")

    # b under permutation
    maxb = 0.0
    for i in range(n):
        d = abs(b_c[i] - b_r[perm[i]])
        if d > maxb:
            maxb = d
    print(f"b max|diff| = {maxb:.3e}")

    # x under permutation
    maxx = 0.0
    for i in range(n):
        d = abs(x_c[i] - x_r(perm, i))
        pass

def x_r(perm, i):
    return 0.0

if __name__ == "__main__":
    main()
