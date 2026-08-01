#!/usr/bin/env python3
"""Compare ex28 sol.gf (C++ vs Rust) for arbitrary order, using a
position-based DOF permutation (dof positions dumped from both sides).

Files (in the working dir):
  cpp_sol{N}.gf, rust_sol{N}.gf          — FiniteElementSpace-format solutions
  cpp_dofpos.txt                          — flat 2*n_dofs scalars (x,y per dof)
  rust_dofpos.txt                         — n_dofs lines "x y"
Usage: compare_ex28_gf.py <order>
"""
import sys

def read_gf(p):
    vals = []; started = False
    for l in open(p):
        s = l.strip()
        if not s:
            continue
        if s.startswith("FiniteElementSpace"):
            started = True
            continue
        if not started:
            continue
        if s.startswith("FiniteElement") or s.startswith("VDim") or s.startswith("Ordering"):
            continue
        vals.append(float(s))
    return vals

def read_dofpos_flat(p):
    nums = []
    for l in open(p):
        s = l.strip()
        if not s or s.startswith("n_dofs"):
            continue
        nums.append(float(s))
    return [(nums[2*i], nums[2*i+1]) for i in range(len(nums)//2)]

def read_dofpos_pairs(p):
    out = []
    for l in open(p):
        s = l.strip()
        if not s or s.startswith("n_dofs"):
            continue
        toks = s.split()
        out.append((float(toks[0]), float(toks[1])))
    return out

def main():
    order = sys.argv[1] if len(sys.argv) > 1 else "1"
    a = read_gf(f"cpp_sol{order}.gf")
    b = read_gf(f"rust_sol{order}.gf")
    cp = read_dofpos_flat("cpp_dofpos.txt")
    rp = read_dofpos_pairs("rust_dofpos.txt")
    n_scalar = len(cp)
    assert len(a) == len(b) == 2 * n_scalar, (len(a), len(b), n_scalar)
    assert len(rp) == n_scalar, (len(rp), n_scalar)

    # position-based permutation: cpp scalar dof i -> rust scalar dof
    rb = {}
    for j, (x, y) in enumerate(rp):
        rb.setdefault((round(x, 5), round(y, 5)), []).append(j)
    perm = []
    for i, (x, y) in enumerate(cp):
        c = rb.get((round(x, 5), round(y, 5)), [])
        if not c:
            best = None; bd = 1e9
            for j, (xx, yy) in enumerate(rp):
                d = (xx-x)**2 + (yy-y)**2
                if d < bd:
                    bd = d; best = j
            assert bd < 1e-12, (i, (x, y), bd)
            c = [best]
        assert len(c) == 1, (i, (x, y), c)
        perm.append(c[0])

    def tr(v):
        comp, dof = v // n_scalar, v % n_scalar
        return comp * n_scalar + perm[dof]

    worst = 0.0; wi = None
    for i in range(len(a)):
        d = abs(a[i] - b[tr(i)])
        if d > worst:
            worst = d; wi = (i, a[i], b[tr(i)])
    print(f"order-{order}: sol.gf max|diff| = {worst:.3e} at cpp vdof {wi[0]}: "
          f"cpp={wi[1]:.9f} rust={wi[2]:.9f}")

if __name__ == "__main__":
    main()
