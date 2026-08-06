import re, numpy as np
def read_rows(path):
    rows, cur = {}, None
    for l in open(path):
        l = l.strip()
        if not l: continue
        m = re.match(r"\[row (\d+)\]", l)
        if m: cur = int(m.group(1)); rows[cur] = []
        for (c, v) in re.findall(r"\((\d+)\s*,\s*([-+0-9.eE]+)\)", l):
            rows[cur].append((int(c), float(v)))
    return rows
C = read_rows("rust_C.txt")
G = read_rows("rust_G.txt")
nh1 = max(c for v in G.values() for (c, _) in v) + 1
CG = np.zeros((max(C.keys())+1, nh1))
for f, es in C.items():
    for (e, s) in es:
        for (c, gv) in G.get(e, []):
            CG[f, c] += s * gv
nz = np.abs(CG) > 1e-10
print("C·G nnz:", int(nz.sum()), "max:", np.abs(CG).max())
for f in sorted(C.keys())[:3]:
    print(f"face row {f}: {C[f]}")
    badcols = np.nonzero(np.abs(CG[f]) > 1e-10)[0]
    print(f"  C·G nonzeros at cols {badcols[:8]} vals {CG[f][badcols][:8]}")
