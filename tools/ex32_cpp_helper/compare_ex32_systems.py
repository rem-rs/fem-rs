#!/usr/bin/env python3
"""Compare ex32 A/M eliminated matrices + ess set + eigenvalues (C++ vs Rust)
under the DOF-coordinate permutation (edge-midpoint anchor, cf. ex31).

Files (run from this directory):
  cpp_dofpos.txt / rust_dofpos.txt — 1752*3 doubles, per-dof physical coords
  cpp_A_elim.txt / rust_A_elim.txt — A after EliminateEssentialBCDiag(1.0)
  cpp_M_elim.txt / rust_M_elim.txt — M after EliminateEssentialBCDiag(min)
  cpp_ess.txt / rust_ess.txt       — essential dof lists
  cpp_eigs.txt / rust_eigs.txt     — eigenvalues (rust side optional)
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

def match_perm(cpp_pts, rust_pts, tol=1e-8):
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
    maxd = 0.0; worst = None; nnz_c = 0; nnz_r = 0; n_extra = 0; n_miss = 0
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
            if vc == 0.0 and vr != 0.0: n_extra += 1
            if vc != 0.0 and vr == 0.0: n_miss += 1
            d = abs(vc - vr)
            if d > maxd:
                maxd = d; worst = (i, j, vc, vr)
    print(f"{name}: nnz cpp={nnz_c} rust(perm)={nnz_r}  max|diff|={maxd:.3e} "
          f"at (cpp_dof {worst[0]}, cpp_dof {worst[1]}): cpp={worst[2]:.10f} rust={worst[3]:.10f}  "
          f"extra={n_extra} missing={n_miss}")
    return maxd

def main():
    c_pts = read_pts3_flat("cpp_dofpos.txt")
    r_pts = read_pts3_flat("rust_dofpos.txt")
    n = len(c_pts)
    assert len(r_pts) == n, f"dof count differs: {n} vs {len(r_pts)}"
    perm = match_perm(c_pts, r_pts)
    print(f"{n} dofs matched (nearest-coordinate permutation)")
    for i in [0, 1, 2, 3, 5, 100, 1000, 1751]:
        print(f"  cpp dof {i} -> rust dof {perm[i]}")

    A_c = read_sparse("cpp_A_elim.txt")
    A_r = read_sparse("rust_A_elim.txt")
    M_c = read_sparse("cpp_M_elim.txt")
    M_r = read_sparse("rust_M_elim.txt")
    assert len(A_c) == len(A_r) == len(M_c) == len(M_r) == n
    compare_mat("A_elim", A_c, A_r, perm, n)
    compare_mat("M_elim", M_c, M_r, perm, n)

    # ess set under permutation
    ess_c = set(int(x) for x in read_vec("cpp_ess.txt"))
    ess_r_rust = set(int(x) for x in read_vec("rust_ess.txt"))
    inv = {perm[i]: i for i in range(n)}  # rust dof -> cpp dof
    ess_r = set(inv[r] for r in ess_r_rust)
    print(f"ess: cpp={len(ess_c)} rust(perm)={len(ess_r)}  "
          f"cpp-only={len(ess_c - ess_r)} rust-only={len(ess_r - ess_c)}")

    # eigenvalues
    eigs_c = read_vec("cpp_eigs.txt")
    print(f"eigenvalues cpp : {['%.8f' % e for e in eigs_c]}")
    import os
    if os.path.exists("rust_eigs.txt"):
        eigs_r = read_vec("rust_eigs.txt")
        print(f"eigenvalues rust: {['%.8f' % e for e in eigs_r]}")
        for a, b in zip(eigs_c, eigs_r):
            print(f"  rel diff = {abs(a-b)/abs(a):.3e}")

if __name__ == "__main__":
    main()
