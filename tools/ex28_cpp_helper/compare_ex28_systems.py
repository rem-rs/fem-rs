#!/usr/bin/env python3
"""Compare ex28 linear systems (C++ vs Rust) under the vertex-coordinate
permutation, converting C++ byNODES vdofs to Rust byVDIM vdofs.

Layouts:
  C++  : vdof(node, comp) = node*vdim + comp   (byNODES)
  Rust : vdof(node, comp) = comp*n_dofs + node (byVDIM)

Run from this directory (files are cpp_*.txt / rust_*.txt).
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

def read_verts(path):
    v = {}
    for l in open(path):
        p = l.split()
        if len(p) == 3:
            v[int(p[0])] = (float(p[1]), float(p[2]))
    return v

def main():
    cv = read_verts("cpp_verts.txt")
    rv = read_verts("rust_verts.txt")
    nd = len(cv)
    assert len(rv) == nd, f"vertex count differs: {len(cv)} vs {len(rv)}"

    # permutation: cpp vertex i -> rust vertex index (match by coords, tol)
    r_by = {}
    for i, xy in rv.items():
        r_by.setdefault((round(xy[0], 4), round(xy[1], 4)), []).append(i)
    perm = []
    for i in range(nd):
        k = (round(cv[i][0], 4), round(cv[i][1], 4))
        cands = r_by.get(k, [])
        if not cands:
            # tolerance fallback: nearest point
            best = None; bd = 1e9
            for j, xy in rv.items():
                d = (xy[0]-cv[i][0])**2 + (xy[1]-cv[i][1])**2
                if d < bd:
                    bd = d; best = j
            assert bd < 1e-12, f"cpp vertex {i} {cv[i]} no match (nearest d={bd})"
            cands = [best]
        perm.append(cands[0])
    print(f"{nd} vertices matched; perm[0]={perm[0]} perm[1]={perm[1]}")

    A_c = read_sparse("cpp_A.txt")
    A_r = read_sparse("rust_A.txt")
    b_c = read_vec("cpp_b.txt")
    b_r = read_vec("rust_b.txt")
    C_c = read_sparse("cpp_C.txt")
    C_r = read_sparse("rust_C.txt")
    x_c = read_vec("cpp_x.txt")
    x_r = read_vec("rust_x.txt")
    n = len(b_c)
    print(f"A: cpp {len(A_c)}x vs rust {len(A_r)}x;  b: {n} vs {len(b_r)};  "
          f"C rows: cpp {len(C_c)} vs rust {len(C_r)};  x: {len(x_c)} vs {len(x_r)}")

    # convert a C++ vdof to (node, comp); rust vdof = comp*n + node.
    # NOTE: MFEM byNODES is vdof = dof + ndofs*vd  ==  block layout, identical
    # to the Rust VectorH1Space byVDIM layout. So only the vertex numbering
    # differs (perm).
    n_dofs = n // 2
    def cpp_to_rust_vdof(v):
        comp, node = v // n_dofs, v % n_dofs
        return comp * n_dofs + perm[node]

    # ---- A comparison ----
    n_dofs = n // 2
    maxd = 0.0; worst = None; nnz_c = 0; nnz_r = 0
    for i in range(n):
        pi = cpp_to_rust_vdof(i)
        row_c = dict(A_c.get(i, []))
        row_r = dict(A_r.get(pi, []))
        cols = set(row_c) | set(row_r)
        for j in cols:
            vc = row_c.get(j, 0.0); vr = row_r.get(cpp_to_rust_vdof(j), 0.0)
            if j in row_c: nnz_c += 1
            if cpp_to_rust_vdof(j) in row_r: nnz_r += 1
            d = abs(vc - vr)
            if d > maxd:
                maxd = d; worst = (i, j, vc, vr)
    print(f"A nnz: cpp={nnz_c} rust(perm)={nnz_r}  max|diff|={maxd:.3e} "
          f"at vdof({worst[0]},{worst[1]}) cpp={worst[2]:.6f} rust={worst[3]:.6f}")

    # ---- b comparison ----
    maxb = 0.0; bw = None
    for i in range(n):
        d = abs(b_c[i] - b_r[cpp_to_rust_vdof(i)])
        if d > maxb:
            maxb = d; bw = (i, b_c[i], b_r[cpp_to_rust_vdof(i)])
    print(f"b max|diff|={maxb:.3e} at vdof {bw[0]}: cpp={bw[1]:.8f} rust={bw[2]:.8f}")

    # ---- C comparison ----
    print("C rows: cpp=", len(C_c), " rust=", len(C_r))
    # rust rows as dict of rust-vdof -> val; cpp rows mapped to rust vdofs
    cpp_rows = [dict((cpp_to_rust_vdof(c), v) for (c, v) in C_c[r]) for r in sorted(C_c)]
    rust_rows = [dict((c, v) for (c, v) in C_r[r]) for r in sorted(C_r)]
    for r, row in enumerate(cpp_rows):
        s = "  ".join(f"n{vdof % n_dofs}{'x' if vdof < n_dofs else 'y'}={v:.6g}"
                      for vdof, v in sorted(row.items()))
        print(f"  cpp row {r}: {s}")
    for r, row in enumerate(rust_rows):
        s = "  ".join(f"n{vdof % n_dofs}{'x' if vdof < n_dofs else 'y'}={v:.6g}"
                      for vdof, v in sorted(row.items()))
        print(f"  rust row {r}: {s}")

    # ---- x comparison (solution, before any sign flip) ----
    maxx = 0.0; xw = None
    for i in range(n):
        d = abs(x_c[i] - x_r[cpp_to_rust_vdof(i)])
        if d > maxx:
            maxx = d; xw = (i, x_c[i], x_r[cpp_to_rust_vdof(i)])
    print(f"x max|diff|={maxx:.3e} at vdof {xw[0]}: cpp={xw[1]:.8f} rust={xw[2]:.8f}")

if __name__ == "__main__":
    main()
