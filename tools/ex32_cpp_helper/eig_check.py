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
n = 1752
A = to_csr(read_sparse("rust_A_elim.txt"), n)
M = to_csr(read_sparse("rust_M_elim.txt"), n)
# generalized eigs A x = lam M x, smallest algebraic (sigma=0 shift-invert)
w = eigsh(A, k=5, M=M, sigma=0.0, which='LM', maxiter=2000, tol=1e-10)
print("scipy eigenvalues (sigma=0):", ["%.8f" % e for e in sorted(w[0])])
# also with cpp eigenvalues for reference
print("cpp eigenvalues             :", ['1.27356150','2.88827434','3.28780349','4.63843335','5.07030790'])

# non-zero spectrum: shift sigma away from the nullspace
w2 = eigsh(A, k=5, M=M, sigma=1.0, which='LM', maxiter=4000, tol=1e-10)
print("scipy (sigma=1.0):", ["%.8f" % e for e in sorted(w2[0])])
w3 = eigsh(A, k=5, M=M, sigma=4.0, which='LM', maxiter=4000, tol=1e-10)
print("scipy (sigma=4.0):", ["%.8f" % e for e in sorted(w3[0])])
