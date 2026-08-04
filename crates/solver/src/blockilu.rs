//! Block ILU(0) preconditioner for DG-type matrices (MFEM `BlockILU`).
//!
//! Bit-for-bit-style port of MFEM's `BlockILU` (linalg/solvers.cpp): block
//! size = DOFs per element, block pattern from the sparse matrix, optional
//! minimum-discarded-fill (MDF) block reordering, block LU factorization with
//! partial pivoting (mirroring LAPACK `dgetrf`/`dgetrs` semantics), and a
//! forward/backward block substitution in `Mult`.

use crate::iterative::{diag, dot, fmt_g, into_result_from_cg};
use crate::SolveResult;
use fem_linalg::SolverConfig;

/// Block ILU(0) preconditioner, mirroring MFEM `BlockILU`.
pub struct BlockIlu {
    block_size: usize,
    n_blockrows: usize,
    /// Block permutation (forward).
    p: Vec<usize>,
    /// Block permutation (inverse).
    pinv: Vec<usize>,
    /// Block CSR row pointers (length `n_blockrows + 1`).
    ib: Vec<usize>,
    /// Block CSR column indices (length = number of block entries).
    jb: Vec<usize>,
    /// Block values, each `block_size²` (column-major, like MFEM DenseMatrix).
    ab: Vec<Vec<f64>>,
    /// Position of the diagonal block in each block row.
    id: Vec<usize>,
    /// Diagonal blocks (factorized), each `block_size²`.
    db: Vec<Vec<f64>>,
    /// Pivot indices for the diagonal block factorizations (1-based, LAPACK
    /// style: `ipiv[i]` is the pivot row for step `i`, stored +1).
    ipiv: Vec<usize>,
}

impl BlockIlu {
    /// Construct the block ILU(0) factorization of the sparse matrix `a` with
    /// the given block size (must divide `a.nrows`).
    pub fn new(a: &fem_linalg::CsrMatrix<f64>, block_size: usize) -> Self {
        let mut ilu = BlockIlu {
            block_size,
            n_blockrows: a.nrows / block_size,
            p: Vec::new(),
            pinv: Vec::new(),
            ib: Vec::new(),
            jb: Vec::new(),
            ab: Vec::new(),
            id: Vec::new(),
            db: Vec::new(),
            ipiv: Vec::new(),
        };
        ilu.create_block_pattern(a);
        ilu.factorize();
        ilu
    }

    /// Build the block CSR pattern, the MDF ordering and extract the blocks.
    fn create_block_pattern(&mut self, a: &fem_linalg::CsrMatrix<f64>) {
        let bs = self.block_size;
        let nbr = self.n_blockrows;
        let row_ptr = &a.row_ptr;
        let col = &a.col_idx;
        let val = &a.values;
        let n = a.nrows;
        debug_assert_eq!(n % bs, 0, "block size must divide the matrix size");

        // Unique block columns per block row.
        let mut unique_block_cols: Vec<std::collections::BTreeSet<usize>> =
            (0..nbr).map(|_| Default::default()).collect();
        for iblock in 0..nbr {
            for bi in 0..bs {
                let i = iblock * bs + bi;
                for k in row_ptr[i]..row_ptr[i + 1] {
                    unique_block_cols[iblock].insert(col[k] as usize / bs);
                }
            }
        }
        let nnz: usize = unique_block_cols.iter().map(|s| s.len()).sum();

        // Block graph matrix C (used only for the reordering): C(i,j) =
        // sqrt(sum of A(i,j)² over the block).
        // MFEM builds C as an open SparseMatrix (`C.Add` in ascending
        // `std::set` order → head-inserted linked list → the finalized CSR
        // rows are in *descending* column order).  The MDF weight sums iterate
        // the C row in that order, so we must store `c_rows` descending to be
        // bit-identical (a 1-ulp difference flips WeightMinHeap tie-breaks).
        let mut c_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nbr];
        for iblock in 0..nbr {
            for &jblock in unique_block_cols[iblock].iter().rev() {
                let mut s = 0.0;
                for bi in 0..bs {
                    let i = iblock * bs + bi;
                    for k in row_ptr[i]..row_ptr[i + 1] {
                        let j = col[k] as usize;
                        if j >= jblock * bs && j < (jblock + 1) * bs {
                            s += val[k] * val[k];
                        }
                    }
                }
                c_rows[iblock].push((jblock, s.sqrt()));
            }
        }

        // MDF reordering (MFEM `MinimumDiscardedFillOrdering`).
        let p = mdf_ordering(&c_rows);

        // Inverse permutation.
        let mut pinv = vec![0usize; nbr];
        for (i, &pi) in p.iter().enumerate() {
            pinv[pi] = i;
        }

        // Permuted block columns (in the permuted row order).
        let mut unique_block_cols_perminv: Vec<Vec<usize>> = vec![Vec::new(); nbr];
        for i in 0..nbr {
            let mut cols: Vec<usize> =
                unique_block_cols[p[i]].iter().map(|&j| pinv[j]).collect();
            cols.sort_unstable();
            unique_block_cols_perminv[i] = cols;
        }

        // Allocate block CSR.
        let mut ib = vec![0usize; nbr + 1];
        let mut jb = vec![0usize; nnz];
        let mut ab: Vec<Vec<f64>> = Vec::with_capacity(nnz);
        let mut db: Vec<Vec<f64>> = (0..nbr).map(|_| vec![0.0; bs * bs]).collect();
        let mut id = vec![0usize; nbr];
        let mut counter = 0usize;
        for iblock in 0..nbr {
            let iblock_perm = p[iblock];
            for &jblock in &unique_block_cols_perminv[iblock] {
                let jblock_perm = p[jblock];
                if iblock == jblock {
                    id[iblock] = counter;
                }
                jb[counter] = jblock;
                let mut blk = vec![0.0f64; bs * bs];
                for bi in 0..bs {
                    let i = iblock_perm * bs + bi;
                    for k in row_ptr[i]..row_ptr[i + 1] {
                        let j = col[k] as usize;
                        if j >= jblock_perm * bs && j < (jblock_perm + 1) * bs {
                            let bj = j - jblock_perm * bs;
                            blk[bi + bj * bs] = val[k]; // column-major
                            if iblock == jblock {
                                db[iblock][bi + bj * bs] = val[k];
                            }
                        }
                    }
                }
                ab.push(blk);
                counter += 1;
            }
            ib[iblock + 1] = counter;
        }

        self.p = p;
        self.pinv = pinv;
        self.ib = ib;
        self.jb = jb;
        self.ab = ab;
        self.id = id;
        self.db = db;
        self.ipiv = vec![0usize; bs * nbr];
    }

    /// Block LU factorization (MFEM `BlockILU::Factorize`).
    fn factorize(&mut self) {
        let bs = self.block_size;
        let nbr = self.n_blockrows;
        // Precompute LU of the diagonal blocks.
        for i in 0..nbr {
            lu_factor(&mut self.db[i], &mut self.ipiv[i * bs..(i + 1) * bs]);
        }
        for i in 1..nbr {
            let mut kk = self.ib[i];
            while kk < self.ib[i + 1] {
                let k = self.jb[kk];
                if k == i {
                    break;
                }
                if k > i {
                    panic!("block matrix must be sorted with nonzero diagonal");
                }
                // A_ik = A_ik * D_kk^{-1}
                {
                    let a_ik = &mut self.ab[kk];
                    let dk = &self.db[k];
                    let ip = &self.ipiv[k * bs..(k + 1) * bs];
                    right_solve(bs, bs, a_ik, dk, ip);
                }
                // Modify everything to the right of k in row i.
                let mut jj = kk + 1;
                while jj < self.ib[i + 1] {
                    let j = self.jb[jj];
                    if j <= k {
                        jj += 1;
                        continue;
                    }
                    let mut ll = self.ib[k];
                    while ll < self.ib[k + 1] {
                        let l = self.jb[ll];
                        if l == j {
                            // A_ij = A_ij - A_ik * A_kj
                            let a_ik = self.ab[kk].clone();
                            let a_kj = self.ab[ll].clone();
                            let a_ij = &mut self.ab[jj];
                            add_mult_a(-1.0, &a_ik, &a_kj, a_ij, bs);
                            if j == i {
                                self.db[i].copy_from_slice(&self.ab[jj]);
                                lu_factor(&mut self.db[i], &mut self.ipiv[i * bs..(i + 1) * bs]);
                            }
                            break;
                        }
                        ll += 1;
                    }
                    jj += 1;
                }
                kk += 1;
            }
        }
    }

    /// Apply the preconditioner: `x = BlockILU⁻¹ b` (MFEM `BlockILU::Mult`).
    pub fn apply(&self, b: &[f64], x: &mut [f64]) {        let bs = self.block_size;
        let nbr = self.n_blockrows;
        let mut y = vec![0.0f64; b.len()];
        // Forward substitution: solve L y = b (L has identity on the diagonal).
        for i in 0..nbr {
            let mut yi = vec![0.0f64; bs];
            for ib in 0..bs {
                yi[ib] = b[self.p[i] * bs + ib];
            }
            let mut k = self.ib[i];
            while k < self.id[i] {
                let j = self.jb[k];
                let l_ij = self.ab[k].clone();
                let yj: Vec<f64> = y[j * bs..(j + 1) * bs].to_vec();
                add_mult_a(-1.0, &l_ij, &yj, &mut yi, bs);
                k += 1;
            }
            y[i * bs..(i + 1) * bs].copy_from_slice(&yi);
        }
        // Backward substitution: solve U x = y, then scale by the diagonal
        // block factors.
        for ii in 0..nbr {
            let i = nbr - 1 - ii;
            let mut xi = vec![0.0f64; bs];
            for ib in 0..bs {
                xi[ib] = y[i * bs + ib];
            }
            let mut k = self.id[i] + 1;
            while k < self.ib[i + 1] {
                let j = self.jb[k];
                let u_ij = self.ab[k].clone();
                let xj: Vec<f64> = x[self.p[j] * bs..(self.p[j] + 1) * bs].to_vec();
                add_mult_a(-1.0, &u_ij, &xj, &mut xi, bs);
                k += 1;
            }
            // x_i = D_ii^{-1} x_i (LU solve with pivoting).
            let db = self.db[i].clone();
            let ipiv = self.ipiv[i * bs..(i + 1) * bs].to_vec();
            lu_solve(&db, &ipiv, &mut xi, bs);
            x[self.p[i] * bs..(self.p[i] + 1) * bs].copy_from_slice(&xi);
        }
    }
}

/// MDF (minimum discarded fill) block ordering — bit-for-bit port of MFEM's
/// `MinimumDiscardedFillOrdering` + `WeightMinHeap` (linalg/solvers.cpp:2819,
/// 2907).  `c` is the block graph as rows of `(col, value)` with the value
/// already `sqrt(sum A²)`.
fn mdf_ordering(c: &[Vec<(usize, f64)>]) -> Vec<usize> {
    let n = c.len();
    // Scale rows by the reciprocal of the diagonal, take absolute values.
    let mut v: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for (i, row) in c.iter().enumerate() {
        let diag = row
            .iter()
            .find(|(j, _)| *j == i)
            .map(|(_, val)| *val)
            .unwrap_or(1.0);
        v.push(row.iter().map(|(j, val)| (*j, (val / diag).abs())).collect());
    }
    // Weight = sqrt(sum over missing edges (i,k),(k,j) of (C_ik*C_kj)²).
    // MFEM reads C_ik from *row i* (`for kk=I[i]..I[i+1]: if J[kk]==k`), not
    // from row k — C(i,k) and C(k,i) differ by 1 ulp (float accumulation
    // order), which matters for the tie-breaking of the heap.
    let mut w = vec![0.0f64; n];
    for k in 0..n {
        let mut s = 0.0;
        for &(i, _) in &v[k] {
            // Find value of (i,k): scan row i for column k.
            let c_ik = v[i]
                .iter()
                .find(|(jj, _)| *jj == k)
                .map(|(_, val)| *val)
                .unwrap_or(0.0);
            for &(j, c_kj) in &v[k] {
                if j == k {
                    continue;
                }
                if !v[i].iter().any(|(jj, _)| *jj == j) {
                    s += (c_ik * c_kj).powi(2);
                }
            }
        }
        w[k] = s.sqrt();
    }
    // Tie-breaking matters — the strict `>` / `<` comparisons preserve the
    // insertion order for equal weights, which the linear scan does not.
    let mut heap = WeightMinHeap::new(n);
    for i in 0..n {
        heap.push(&w, i);
    }
    let mut p = Vec::with_capacity(n);
    for _ in 0..n {
        let pi = heap.pop(&w);
        p.push(pi);
        w[pi] = -1.0;
        // Recompute weights of neighbors of pi.
        for &(k, _) in &v[pi] {
            if heap.picked(k) {
                continue;
            }
            let mut s = 0.0;
            for &(i, _) in &v[k] {
                if heap.picked(i) {
                    continue;
                }
                // Find value of (i,k): scan row i for column k (MFEM C_ik).
                let c_ik = v[i]
                    .iter()
                    .find(|(jj, _)| *jj == k)
                    .map(|(_, val)| *val)
                    .unwrap_or(0.0);
                for &(j, c_kj) in &v[k] {
                    if j == k || heap.picked(j) {
                        continue;
                    }
                    if !v[i].iter().any(|(jj, _)| *jj == j) {
                        s += (c_ik * c_kj).powi(2);
                    }
                }
            }
            w[k] = s.sqrt();
            heap.update(&w, k);
        }
    }
    p
}

/// MFEM `WeightMinHeap` (linalg/solvers.cpp:2819) — binary min-heap over the
/// weight array `w` (passed by reference to each operation, mirroring MFEM's
/// external `const std::vector<real_t> &w` member).  `c[pos]` = heap node at
/// `pos` (a node index), `loc[node]` = position of the node in the heap, `-1`
/// if removed.
struct WeightMinHeap {
    c: Vec<usize>,
    loc: Vec<isize>,
}

impl WeightMinHeap {
    fn new(n: usize) -> Self {
        WeightMinHeap {
            c: Vec::with_capacity(n),
            loc: vec![-1; n],
        }
    }

    fn percolate_up(&mut self, w: &[f64], mut pos: usize, val: f64) -> usize {
        while pos > 0 && w[self.c[(pos - 1) / 2]] > val {
            self.c[pos] = self.c[(pos - 1) / 2];
            self.loc[self.c[(pos - 1) / 2]] = pos as isize;
            pos = (pos - 1) / 2;
        }
        pos
    }

    fn percolate_down(&mut self, w: &[f64], mut pos: usize, val: f64) -> usize {
        while 2 * pos + 1 < self.c.len() {
            let left = 2 * pos + 1;
            let right = left + 1;
            let tgt = if right < self.c.len() && w[self.c[right]] < w[self.c[left]] {
                right
            } else {
                left
            };
            if w[self.c[tgt]] < val {
                self.c[pos] = self.c[tgt];
                self.loc[self.c[tgt]] = pos as isize;
                pos = tgt;
            } else {
                break;
            }
        }
        pos
    }

    fn push(&mut self, w: &[f64], i: usize) {
        let val = w[i];
        self.c.push(0);
        let mut pos = self.c.len() - 1;
        pos = self.percolate_up(w, pos, val);
        self.c[pos] = i;
        self.loc[i] = pos as isize;
    }

    fn pop(&mut self, w: &[f64]) -> usize {
        let i = self.c[0];
        let j = *self.c.last().unwrap();
        self.c.pop();
        // Mark as removed.
        self.loc[i] = -1;
        if self.c.is_empty() {
            return i;
        }
        let val = w[j];
        let mut pos = 0;
        pos = self.percolate_down(w, pos, val);
        self.c[pos] = j;
        self.loc[j] = pos as isize;
        i
    }

    fn update(&mut self, w: &[f64], i: usize) {
        let mut pos = self.loc[i] as usize;
        let val = w[i];
        pos = self.percolate_up(w, pos, val);
        pos = self.percolate_down(w, pos, val);
        self.c[pos] = i;
        self.loc[i] = pos as isize;
    }

    fn picked(&self, i: usize) -> bool {
        self.loc[i] < 0
    }
}

/// LU factorization with partial pivoting (LAPACK `dgetf2` semantics; MFEM
/// `LUFactors::Factor` non-LAPACK branch).  `a` is column-major `m×m`,
/// `ipiv` is 1-based pivots.
fn lu_factor(a: &mut [f64], ipiv: &mut [usize]) {
    let m = ipiv.len();
    for i in 0..m {
        // pivoting: max |a[j + i*m]| for j >= i
        let mut piv = i;
        let mut amax = a[piv + i * m].abs();
        for j in i + 1..m {
            let b = a[j + i * m].abs();
            if b > amax {
                amax = b;
                piv = j;
            }
        }
        ipiv[i] = piv + 1;
        if piv != i {
            for j in 0..m {
                a.swap(i + j * m, piv + j * m);
            }
        }
        let a_ii = a[i + i * m];
        if a_ii == 0.0 {
            panic!("singular diagonal block in BlockILU");
        }
        let a_ii_inv = 1.0 / a_ii;
        for j in i + 1..m {
            a[j + i * m] *= a_ii_inv;
        }
        for k in i + 1..m {
            let a_ik = a[i + k * m];
            for j in i + 1..m {
                a[j + k * m] -= a_ik * a[j + i * m];
            }
        }
    }
}

/// Solve `A x = b` given the LU factorization of `A` (LAPACK `dgetrs`
/// semantics; MFEM `LUFactors::Solve` non-LAPACK branch: LSolve + USolve).
/// `x` has `nrhs` columns, column-major, `bs` rows.
fn lu_solve(d: &[f64], ipiv: &[usize], x: &mut [f64], bs: usize) {
    // x <- P x
    for i in 0..bs {
        let pi = ipiv[i] - 1;
        if pi != i {
            x.swap(i, pi);
        }
    }
    // x <- L^{-1} x
    for j in 0..bs {
        let x_j = x[j];
        for i in j + 1..bs {
            x[i] -= d[i + j * bs] * x_j;
        }
    }
    // x <- U^{-1} x
    for j in (0..bs).rev() {
        let x_j = x[j] / d[j + j * bs];
        for i in 0..j {
            x[i] -= d[i + j * bs] * x_j;
        }
        x[j] = x_j;
    }
}

/// Right solve: `x = x · A⁻¹` given the LU factorization of `A` (MFEM
/// `LUFactors::RightSolve` non-LAPACK branch: `x U⁻¹`, `x L⁻¹`, `x P`).
/// `x` is `rows × cols` column-major (MFEM iterates columns with `x += 1`,
/// i.e. column stride 1 in the column-major layout).
fn right_solve(rows: usize, cols: usize, x: &mut [f64], d: &[f64], ipiv: &[usize]) {
    for k in 0..cols {
        let base = k;
        // x <- x U^{-1}
        for j in 0..rows {
            let x_j = x[base + j * cols] / d[j + j * rows];
            x[base + j * cols] = x_j;
            for i in j + 1..rows {
                x[base + i * cols] -= d[j + i * rows] * x_j;
            }
        }
        // x <- x L^{-1}
        for j in (0..rows).rev() {
            let x_j = x[base + j * cols];
            for i in 0..j {
                x[base + i * cols] -= d[j + i * rows] * x_j;
            }
        }
        // x <- x P
        for i in (0..rows).rev() {
            let pi = ipiv[i] - 1;
            x.swap(base + i * cols, base + pi * cols);
        }
    }
}

/// `c = c + alpha * a * b` where `a` is `n×n`, `b` is a vector or matrix.
/// MFEM `DenseMatrix::AddMult_a` (column-major, k summed in order).
fn add_mult_a(alpha: f64, a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    if b.len() == n {
        // b is a vector
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i + k * n] * b[k];
            }
            c[i] += alpha * s;
        }
    } else {
        let m = b.len() / n;
        for j in 0..m {
            for i in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i + k * n] * b[k + j * n];
                }
                c[i + j * n] += alpha * s;
            }
        }
    }
}

/// MFEM `CGSolver::Mult` + `BlockILU` preconditioner: PCG with
/// `iterative_mode = false` (start from zero) and the `(B r, r)` convergence
/// test.
pub fn solve_pcg_blockilu(
    a: &fem_linalg::CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
    block_size: usize,
) -> Result<SolveResult, crate::SolverError> {
    crate::macros::check_dims(a, b, x)?;
    let n = a.nrows;
    let row_ptr = &a.row_ptr;
    let col = &a.col_idx;
    let val = &a.values;

    let prec = BlockIlu::new(a, block_size);

    // r = b (iterative_mode = false => x starts from zero).
    let mut r = b.to_vec();
    for i in 0..n {
        x[i] = 0.0;
    }
    let mut z = vec![0.0f64; n];
    prec.apply(&r, &mut z);
    let mut d = z.clone();
    let mut nom = dot(&d, &r);
    let nom0 = nom;
    let r0 = (nom * cfg.rtol * cfg.rtol).max(cfg.atol * cfg.atol);

    let mut iter = 0usize;
    if cfg.verbose {
        println!("   Iteration : {:3}  (B r, r) = {}", 0, fmt_g(nom));
    }
    if nom <= r0 {
        return Ok(into_result_from_cg(n, iter, nom0, nom, true));
    }

    loop {
        for i in 0..n {
            let mut s = 0.0;
            for k in row_ptr[i]..row_ptr[i + 1] {
                s += val[k] * d[col[k] as usize];
            }
            z[i] = s;
        }
        let den = dot(&z, &d);
        let alpha = nom / den;
        for i in 0..n {
            x[i] += alpha * d[i];
            r[i] -= alpha * z[i];
        }
        prec.apply(&r, &mut z);
        let betanom = dot(&z, &r);
        iter += 1;
        if cfg.verbose {
            println!("   Iteration : {:3}  (B r, r) = {}", iter, fmt_g(betanom));
        }
        if betanom <= r0 || iter >= cfg.max_iter {
            let res = into_result_from_cg(n, iter, nom0, betanom, betanom <= r0);
            if cfg.verbose {
                let avg = (betanom / nom0).powf(0.5 / iter as f64);
                println!("Average reduction factor = {}", fmt_g(avg));
            }
            return Ok(res);
        }
        let beta = betanom / nom;
        for i in 0..n {
            d[i] = z[i] + beta * d[i];
        }
        nom = betanom;
    }
}
