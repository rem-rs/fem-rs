#!/usr/bin/env python3
"""Compare MFEM ex27 C++ vs Rust solution values vertex-by-coordinate.

C++ side: authoritative dump from ex27_vertex_dump helper, one line per vertex:
    "<vertex_idx> <x> <y> <u>"
Rust side: refined.mesh (flat MFEM v1.0, vertices section) + sol.gf (H1 P1:
dof i <-> mesh vertex i).

The ex27 mesh is periodic: the C++ vertex table reports BOTH the seam and the
center columns at x=0 (an MFEM vertex-averaging artifact), while the Rust mesh
keeps the seam at x=±1.  Vertices in the x=0 column are therefore matched to
the Rust (0,y) center vertex or the (±1,y) seam vertex — whichever value is
closer.

Usage:
  python compare_ex27_gf.py <cpp_dump> <rust_mesh> <rust_gf>
"""
import sys

def parse_rust_mesh_verts(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = [l.split("#", 1)[0].strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    i_vert = lines.index("vertices")
    nv = int(lines[i_vert + 1])
    j = i_vert + 2
    verts = []
    while len(verts) < nv and j < len(lines):
        toks = lines[j].split()
        if len(toks) >= 2:
            try:
                verts.append((float(toks[0]), float(toks[1])))
            except ValueError:
                pass
        j += 1
    return verts

def parse_gf(path):
    vals = []
    started = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("FiniteElementSpace"):
                started = True
                continue
            if s.startswith(("FiniteElementCollection", "VDim", "Ordering")):
                continue
            if started:
                try:
                    vals.append(float(s))
                except ValueError:
                    continue
    return vals

def main():
    cpp_dump, rust_mesh, rust_gf = sys.argv[1:4]
    cpp = []
    for l in open(cpp_dump):
        p = l.split()
        if len(p) == 4:
            cpp.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    rv = parse_rust_mesh_verts(rust_mesh)
    rg = parse_gf(rust_gf)
    print(f"C++ : {len(cpp)} verts | Rust: {len(rv)} verts, {len(rg)} GF values")

    r_by = {}
    for i, v in enumerate(rv):
        r_by.setdefault((round(v[0], 6), round(v[1], 6)), []).append(i)

    n_match = 0
    maxd = 0.0
    worst = None
    hist = {"<1e-6": 0, "<1e-3": 0, "<1e-2": 0, "<0.1": 0, ">=0.1": 0}
    for (vi, x, y, val) in cpp:
        cands = []
        if abs(x) < 1e-9:
            # seam/center column: try the Rust center (0,y) and seam (±1,y)
            cands += r_by.get((0.0, round(y, 6)), [])
            cands += r_by.get((-1.0, round(y, 6)), [])
            cands += r_by.get((1.0, round(y, 6)), [])
        else:
            cands = r_by.get((round(x, 6), round(y, 6)), [])
        if not cands:
            continue
        d_best = min(abs(val - rg[ri]) for ri in cands)
        ri_best = min(cands, key=lambda ri: abs(val - rg[ri]))
        n_match += 1
        if d_best > maxd:
            maxd = d_best
            worst = ((round(x, 6), round(y, 6)), val, rg[ri_best])
        if d_best < 1e-6: hist["<1e-6"] += 1
        elif d_best < 1e-3: hist["<1e-3"] += 1
        elif d_best < 1e-2: hist["<1e-2"] += 1
        elif d_best < 0.1: hist["<0.1"] += 1
        else: hist[">=0.1"] += 1
    print(f"matched {n_match}/{len(cpp)} vertex-value pairs")
    print("diff histogram:", hist)
    if worst:
        print(f"max|diff| = {maxd:.3e} at coord {worst[0]} (cpp={worst[1]:.10f} rust={worst[2]:.10f})")

if __name__ == "__main__":
    main()
