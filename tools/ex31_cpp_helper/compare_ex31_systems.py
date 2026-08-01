#!/usr/bin/env python3
"""Compare ex31 linear systems (C++ vs Rust) under the DOF-coordinate permutation.

Layouts:
  C++  ND_R2D global dof = [H1 vertex dofs (0..nverts-1) | ND edge dofs (nverts..)]
  Rust example        = [ND dofs (0..nedges-1) | H1 vertex dofs (nedges..)]
So the permutation is anchored by physical position (vertex coord vs edge midpoint).

Files (run from this directory):
  cpp_dofpos.txt / rust_dofpos.txt  — 833*3 doubles, per-dof physical coords
  cpp_A.txt / rust_A.txt            — raw assembled A (pre-elimination)
  cpp_b.txt / rust_b.txt            — raw rhs (LinearForm)
  cpp_elim_A.txt / rust_elim_A.txt  — eliminated system matrix
  cpp_elim_B.txt / rust_elim_B.txt  — eliminated rhs
  cpp_elim_X0.txt / rust_elim_X0.txt— initial X (projected BC values)
  cpp_x.txt / rust_x.txt            — solver solution (true dofs)
  cpp_soldofs.txt / rust_soldofs.txt— projected solution dofs
  cpp_elmat_0.txt / rust_elmat_0.txt— element-0 matrix + dofs + verts
"""
import re
import sys

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
        for (c, v) in re.findall(r"\((\d+)\s*,\s*([-+0-9.eE]+)\)", l):
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

def read_pts3_flat(path):
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

def compare_mat(name, A_c, A_r, perm, n):
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
    print(f"{name}: nnz cpp={nnz_c} rust(perm)={nnz_r}  max|diff|={maxd:.3e} "
          f"at (cpp_dof {worst[0]}, cpp_dof {worst[1]}): cpp={worst[2]:.10f} rust={worst[3]:.10f}")
    return maxd

def compare_vec(name, v_c, v_r, perm):
    maxd = 0.0; bw = None
    for i in range(len(v_c)):
        d = abs(v_c[i] - v_r[perm[i]])
        if d > maxd:
            maxd = d; bw = (i, v_c[i], v_r[perm[i]])
    print(f"{name}: max|diff|={maxd:.3e} at cpp dof {bw[0]}: cpp={bw[1]:.10f} rust={bw[2]:.10f}")
    return maxd

def main():
    c_pts = read_pts3_flat("cpp_dofpos.txt")
    r_pts = read_pts3_flat("rust_dofpos.txt")
    n = len(c_pts)
    assert len(r_pts) == n, f"dof count differs: {n} vs {len(r_pts)}"
    perm = match_perm(c_pts, r_pts, tol=1e-8)
    print(f"{n} dofs matched (nearest-coordinate permutation)")
    # sanity: show a few
    for i in [0, 1, 288, 289, 290, 544, 832]:
        print(f"  cpp dof {i} {c_pts[i]} -> rust dof {perm[i]} {r_pts[perm[i]]}")

    A_c = read_sparse("cpp_A.txt")
    A_r = read_sparse("rust_A.txt")
    b_c = read_vec("cpp_b.txt")
    b_r = read_vec("rust_b.txt")
    assert len(A_c) == len(A_r) == n
    assert len(b_c) == len(b_r) == n

    compare_mat("A (raw)", A_c, A_r, perm, n)
    compare_vec("b (raw)", b_c, b_r, perm)

    # element-0 matrix (verbatim rows; dof mapping not applied)
    print("\n--- element-0 matrix check (raw local rows; see dofs lines) ---")
    for side in ("cpp", "rust"):
        lines = open(f"{side}_elmat_0.txt").read().splitlines()
        print(f"[{side}] " + lines[0])
        print(f"[{side}] " + lines[1])

    # eliminated system
    for f in ("elim_A", "elim_B", "elim_X0"):
        tag = f.split("_")[-1]
        Ac = read_sparse(f"cpp_{f}.txt")
        Ar = read_sparse(f"rust_{f}.txt")
        vc = read_vec(f"cpp_{f}.txt")
        vr = read_vec(f"rust_{f}.txt")
        if tag == "A":
            compare_mat(f"{f} (elim)", Ac, Ar, perm, n)
        else:
            compare_vec(f"{f}", vc, vr, perm)

    x_c = read_vec("cpp_x.txt")
    x_r = read_vec("rust_x.txt")
    compare_vec("x (solution)", x_c, x_r, perm)

if __name__ == "__main__":
    main()
