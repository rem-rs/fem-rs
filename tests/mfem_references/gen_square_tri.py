#!/usr/bin/env python3
"""Create a unit-square triangular mesh with boundary tags matching fem-rs ex1/ex2.

Boundary tags: 1=bottom, 2=right, 3=top, 4=left

Usage: python gen_square_tri.py [n] [output_file]
  n: subdivisions per side (default: 8)
  output_file: path to save mesh (default: square-tri-n.mesh)
"""
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
output = sys.argv[2] if len(sys.argv) > 2 else f"square-tri-{n}.mesh"

np = n + 1  # nodes per side

# Generate MFEM mesh file
lines = []
lines.append("MFEM mesh v1.0")
lines.append("")
lines.append("dimension")
lines.append("2")
lines.append("")
lines.append("elements")
lines.append(str(2 * n * n))  # 2 triangles per quad
for j in range(n):
    for i in range(n):
        v0 = j * np + i
        v1 = j * np + (i + 1)
        v2 = (j + 1) * np + i
        v3 = (j + 1) * np + (i + 1)
        # Two triangles per quad (type=2 = TRIANGLE, attr=1)
        lines.append(f"1 2 {v0} {v1} {v2}")
        lines.append(f"1 2 {v1} {v3} {v2}")

lines.append("")
lines.append("boundary")
lines.append(str(4 * n))  # 4 boundary edges, n per side
# Bottom edge (y=0): tag 1
for i in range(n):
    v0 = i
    v1 = i + 1
    lines.append(f"1 1 {v0} {v1}")
# Right edge (x=1): tag 2
for j in range(n):
    v0 = j * np + n
    v1 = (j + 1) * np + n
    lines.append(f"2 1 {v0} {v1}")
# Top edge (y=1): tag 3
for i in range(n, 0, -1):
    v0 = n * np + i
    v1 = n * np + (i - 1)
    lines.append(f"3 1 {v0} {v1}")
# Left edge (x=0): tag 4
for j in range(n, 0, -1):
    v0 = j * np
    v1 = (j - 1) * np
    lines.append(f"4 1 {v0} {v1}")

lines.append("")
lines.append("vertices")
lines.append(str(np * np))
for j in range(np):
    for i in range(np):
        lines.append(f"{i / n} {j / n}")

with open(output, 'w') as f:
    f.write("\n".join(lines))

print(f"Created {output}: {np*np} nodes, {2*n*n} triangles, {4*n} boundary edges")
