#!/usr/bin/env python3
"""Decode the C++ constraint matrix cpp_C.txt: map each column (vdof, byNODES)
to (vertex_id, component) and attach vertex coordinates."""
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

def main():
    verts = {}
    for l in open("cpp_verts.txt"):
        p = l.split()
        if len(p) == 3:
            verts[int(p[0])] = (float(p[1]), float(p[2]))
    C = read_sparse("cpp_C.txt")
    for r in sorted(C):
        parts = []
        for (c, v) in C[r]:
            node = c // 2
            comp = c % 2
            xy = verts.get(node, ("?", "?"))
            parts.append(f"node{node}({xy[0]},{xy[1]}){'x' if comp==0 else 'y'}={v}")
        print(f"row {r}: " + "  ".join(parts))

if __name__ == "__main__":
    main()
