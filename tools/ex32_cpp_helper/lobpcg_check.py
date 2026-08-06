import re, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lobpcg
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
A = to_csr(read_sparse("cpp_A_elim.txt"), n)
M = to_csr(read_sparse("cpp_M_elim.txt"), n)
rng = np.random.default_rng(0)
X = rng.standard_normal((n, 5))
w, v = lobpcg(A, X, B=M, largest=False, maxiter=500, tol=1e-8)
print("scipy lobpcg:", ["%.6f" % e for e in np.sort(w)])
print("true        :", ['1.273562','2.888274','3.287803','4.638433','5.070308'])
