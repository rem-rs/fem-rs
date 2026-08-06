import re, numpy as np
def read_sparse(path, n):
    I, J, V = [], [], []
    cur = None
    for l in open(path):
        l = l.strip()
        if not l: continue
        m = re.match(r"\[row (\d+)\]", l)
        if m: cur = int(m.group(1))
        for (c, v) in re.findall(r"\((\d+)\s*,\s*([-+0-9.eE]+)\)", l):
            I.append(cur); J.append(int(c)); V.append(float(v))
    A = np.zeros((n, n))
    for i, j, v in zip(I, J, V): A[i, j] = v
    return A
n = 276
A = read_sparse("rust_A_elim.txt", n)
# G: n_edges x n_h1
GI, GJ, GV = [], [], []
cur = None
for l in open("rust_G.txt"):
    l = l.strip()
    if not l: continue
    m = re.match(r"\[row (\d+)\]", l)
    if m: cur = int(m.group(1))
    for (c, v) in re.findall(r"\((\d+)\s*,\s*([-+0-9.eE]+)\)", l):
        GI.append(cur); GJ.append(int(c)); GV.append(float(v))
nh1 = max(GJ) + 1
G = np.zeros((n, nh1))
for i, j, v in zip(GI, GJ, GV): G[i, j] = v
print("A", A.shape, "G", G.shape)
AG = A @ G
nz = np.nonzero(np.abs(AG) > 1e-10)
print("A·G nnz:", len(nz[0]), "max:", np.abs(AG).max())
# per-row max residual
rowmax = np.abs(AG).max(axis=1)
bad = np.argsort(rowmax)[-5:][::-1]
print("worst rows:", bad, "vals:", rowmax[bad])
for r in bad:
    print(f"  row {r} (dof {r}): A·G = {AG[r][np.abs(AG[r])>1e-10][:6]}")
# check G row 0: endpoints
print("G row 0:", [(j, G[0, j]) for j in np.nonzero(G[0])[0]])
# dofpos row 0
dp = [l.split() for l in open("rust_dofpos.txt") if l.strip()]
print("dof 0 pos:", dp[0][:3] if dp else "?")
