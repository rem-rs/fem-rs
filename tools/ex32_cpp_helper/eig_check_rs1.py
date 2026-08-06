import re, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
def read_sparse(path):
    rows, cur = {}, None
    for l in open(path):
        l = l.strip()
        if not l: continue
        m = re.match(r"\[row (\d+)\]", l)
        if m: cur = int(m.group(1)); rows[cur] = []
        for (c, v) in re.findall(r"\((\d+)\s*,\s*([-+0-9.eE]+)\)", l):
            rows[cur].append((int(c), float(v)))
    return rows
def to_csr(rows, n):
    I, J, V = [], [], []
    for r, es in rows.items():
        for c, v in es:
            if abs(v) > 1e-300:
                I.append(r); J.append(c); V.append(v)
    return csr_matrix((V, (I, J)), shape=(n, n))
A = to_csr(read_sparse("rust_A_elim.txt"), 456)
M = to_csr(read_sparse("rust_M_elim.txt"), 456)
w = eigsh(A, k=4, M=M, sigma=4.0, which='LM', maxiter=2000, tol=1e-10)
print("rust A/M (rs1) non-zero eigs:", ["%.6f" % e for e in sorted(w[0])])
