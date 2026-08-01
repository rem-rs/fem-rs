#!/usr/bin/env python3
"""Compare ex29 linear systems (C++ vs Rust) under the DOF-coordinate permutation.

Layouts: both sides are scalar H1 (48 dofs).  C++ dof numbering and Rust dof
numbering are both "vertex-first, edge-mid, interior" but the global order may
differ; we anchor the permutation by matching each C++ dof's physical position
(cpp_dofpos.txt) to the nearest Rust dof position (rust_dofpos.txt).

Files (run from this directory):
  cpp_verts.txt / rust_verts.txt     — 8 transformed vertices, "id x y z"
  cpp_dofpos.txt / rust_dofpos.txt   — 48*3 doubles, per-dof physical coords
  cpp_A.txt     / rust_A.txt         — raw assembled A (pre-elimination)
  cpp_b.txt     / rust_b.txt         — raw rhs
  cpp_x.txt     / rust_x.txt         — solver solution (true dofs)
"""
import re

def read_sparse(path):
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
        for (c, v) in re.findall(r"\((\d+),([-+0-9.eE]+)\)", l):
            rows[cur].append((int(c), float(v)))
    return rows

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

def read_pts3(path):
    """returns dict id -> (x,y,z) for 'id x y z' lines"""
    d = {}
    for l in open(path):
        p = l.split()
        if len(p) == 4:
            d[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    return d

def read_pts3_flat(path):
    """returns list of (x,y,z) triples from a vector dump (1 or 3 values per line)"""
    vals = []
    for l in open(path):
        s = l.strip()
        if not s:
            continue
        for tok in s.split():
            try:
                vals.append(float(tok))
            except ValueError:
                continue
    assert len(vals) % 3 == 0, f"{path}: {len(vals)} values not divisible by 3"
    return [(vals[3*i], vals[3*i+1], vals[3*i+2]) for i in range(len(vals)//3)]

def match_perm(cpp_pts, rust_pts, tol=1e-9):
    """cpp index i -> rust index, nearest by coordinate (must be unique)."""
    perm = []
    for i, p in enumerate(cpp_pts):
        best = None; bd = 1e9
        for j, q in enumerate(rust_pts):
            d = (q[0]-p[0])**2 + (q[1]-p[1])**2 + (q[2]-p[2])**2
            if d < bd:
                bd = d; best = j
        assert bd < tol, f"cpp point {i} {p}: nearest rust dist {bd**0.5:.3e} > {tol**0.5}"
        perm.append(best)
    assert len(set(perm)) == len(perm), "permutation not bijective!"
    return perm

def main():
    # vertex anchor (informational: C++ GetVertex returns PRE-transform vertices,
    # Rust transform mutates coords in place — these are NOT directly comparable;
    # the real permutation anchor is the DOF physical position below)
    cv = read_pts3("cpp_verts.txt")
    rv = read_pts3("rust_verts.txt")
    print(f"vertices: cpp {len(cv)} (pre-transform) vs rust {len(rv)} (post-transform)")

    # dof anchor
    c_pts = read_pts3_flat("cpp_dofpos.txt")
    r_pts = read_pts3_flat("rust_dofpos.txt")
    n = len(c_pts)
    assert len(r_pts) == n, f"dof count differs: {n} vs {len(r_pts)}"
    perm = match_perm(c_pts, r_pts, tol=1e-8)
    print(f"{n} dofs matched (nearest-coordinate permutation)")
    # show first few dofs for sanity
    for i in range(4):
        print(f"  cpp dof {i} {c_pts[i]} -> rust dof {perm[i]} {r_pts[perm[i]]}")

    A_c = read_sparse("cpp_A.txt")
    A_r = read_sparse("rust_A.txt")
    b_c = read_vec("cpp_b.txt")
    b_r = read_vec("rust_b.txt")
    x_c = read_vec("cpp_x.txt")
    x_r = read_vec("rust_x.txt")
    assert len(A_c) == len(A_r) == n, f"A rows {len(A_c)} vs {len(A_r)} vs {n}"
    assert len(b_c) == len(b_r) == n
    assert len(x_c) == len(x_r) == n

    # ---- A comparison under permutation ----
    maxd = 0.0; worst = None; nnz_c = 0; nnz_r = 0
    for i in range(n):
        pi = perm[i]
        row_c = dict(A_c.get(i, []))
        row_r = dict(A_r.get(pi, []))
        cols = set(row_c) | set(row_r)
        for j in cols:
            vc = row_c.get(j, 0.0)
            vr = row_r.get(perm[j], 0.0)
            if j in row_c: nnz_c += 1
            if perm[j] in row_r: nnz_r += 1
            d = abs(vc - vr)
            if d > maxd:
                maxd = d; worst = (i, j, vc, vr)
    print(f"A nnz: cpp={nnz_c} rust(perm)={nnz_r}  max|diff|={maxd:.3e} "
          f"at (cpp_dof {worst[0]}, cpp_dof {worst[1]}): cpp={worst[2]:.10f} rust={worst[3]:.10f}")

    # ---- b comparison ----
    maxb = 0.0; bw = None
    for i in range(n):
        d = abs(b_c[i] - b_r[perm[i]])
        if d > maxb:
            maxb = d; bw = (i, b_c[i], b_r[perm[i]])
    print(f"b max|diff|={maxb:.3e} at cpp dof {bw[0]}: cpp={bw[1]:.10f} rust={bw[2]:.10f}")

    # ---- x comparison ----
    maxx = 0.0; xw = None
    for i in range(n):
        d = abs(x_c[i] - x_r[perm[i]])
        if d > maxx:
            maxx = d; xw = (i, x_c[i], x_r[perm[i]])
    print(f"x max|diff|={maxx:.3e} at cpp dof {xw[0]}: cpp={xw[1]:.10f} rust={xw[2]:.10f}")

if __name__ == "__main__":
    main()
