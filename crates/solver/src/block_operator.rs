//! Block-structured linear operators and the MultiphysicsOperator.
//!
//! | Type | Role | MFEM equivalent |
//! |------|------|-----------------|
//! | [`BlockOperator`] | Trait: apply y = A*x | `BlockOperator` |
//! | [`BlockOpMatrix`] | BlockMatrix as a BlockOperator | `BlockMatrix` |
//! | [`SumBlockOp`] | Sum of operators | operator addition |
//! | [`MultiphysicsOperator`] | Coupled operator + coupling generators | `MultiphysicsOperator` |
//! | [`BlockNonlinearForm`] | Auto-assembles block Jacobian | `BlockNonlinearForm` |

use fem_linalg::{BlockMatrix, BlockVector, CooMatrix, CsrMatrix};
use crate::{SolverConfig, SolveResult, SolverError};

// ─── BlockOperator trait ─────────────────────────────────────────────────────

pub trait BlockOperator: Send + Sync {
    fn n_blocks(&self) -> usize;
    fn row_sizes(&self) -> &[usize];
    fn col_sizes(&self) -> &[usize];

    fn apply(&self, x: &BlockVector, y: &mut BlockVector) {
        for bi in 0..self.n_blocks() {
            y.block_mut(bi).fill(0.0);
            for bj in 0..self.n_blocks() {
                if self.has_block(bi, bj) {
                    let xj = x.block(bj);
                    let mut tmp = vec![0.0_f64; self.row_sizes()[bi]];
                    self.apply_block(bi, bj, xj, &mut tmp);
                    for (yi, ti) in y.block_mut(bi).iter_mut().zip(tmp.iter()) {
                        *yi += *ti;
                    }
                }
            }
        }
    }

    fn apply_block(&self, i: usize, j: usize, x: &[f64], y: &mut [f64]);
    fn has_block(&self, i: usize, j: usize) -> bool;
}

// ─── BlockOpMatrix ──────────────────────────────────────────────────────────

pub struct BlockOpMatrix {
    matrix: BlockMatrix,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

impl BlockOpMatrix {
    pub fn new(matrix: BlockMatrix) -> Self {
        let rows = matrix.row_sizes.clone();
        let cols = matrix.col_sizes.clone();
        Self { matrix, rows, cols }
    }
    pub fn matrix(&self) -> &BlockMatrix { &self.matrix }
}

impl BlockOperator for BlockOpMatrix {
    fn n_blocks(&self) -> usize { self.matrix.n_row_blocks() }
    fn row_sizes(&self) -> &[usize] { &self.rows }
    fn col_sizes(&self) -> &[usize] { &self.cols }
    fn apply_block(&self, i: usize, j: usize, x: &[f64], y: &mut [f64]) {
        if let Some(a) = self.matrix.get(i, j) { a.spmv(x, y); }
    }
    fn has_block(&self, i: usize, j: usize) -> bool { self.matrix.get(i, j).is_some() }
}

// ─── SumBlockOp ──────────────────────────────────────────────────────────────

pub struct SumBlockOp {
    a: Box<dyn BlockOperator>,
    b: Box<dyn BlockOperator>,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

impl SumBlockOp {
    pub fn new(a: Box<dyn BlockOperator>, b: Box<dyn BlockOperator>) -> Self {
        assert_eq!(a.n_blocks(), b.n_blocks());
        Self { rows: a.row_sizes().to_vec(), cols: a.col_sizes().to_vec(), a, b }
    }
}

impl BlockOperator for SumBlockOp {
    fn n_blocks(&self) -> usize { self.a.n_blocks() }
    fn row_sizes(&self) -> &[usize] { &self.rows }
    fn col_sizes(&self) -> &[usize] { &self.cols }
    fn apply_block(&self, i: usize, j: usize, x: &[f64], y: &mut [f64]) {
        let mut tmp = vec![0.0_f64; self.rows[i]];
        self.a.apply_block(i, j, x, y);
        self.b.apply_block(i, j, x, &mut tmp);
        for (yi, ti) in y.iter_mut().zip(tmp.iter()) { *yi += *ti; }
    }
    fn has_block(&self, i: usize, j: usize) -> bool {
        self.a.has_block(i, j) || self.b.has_block(i, j)
    }
}

// ─── MultiphysicsOperator ───────────────────────────────────────────────────

/// Coupled multiphysics operator — MFEM's `MultiphysicsOperator` equivalent.
///
/// Stores per-field diagonal CSR matrices and off-diagonal coupling generators.
/// Coupling generators are closures that assemble the coupling Jacobian on demand.
pub struct MultiphysicsOperator {
    n_fields: usize,
    field_sizes: Vec<usize>,
    diag_csr: Vec<Option<CsrMatrix<f64>>>,
    #[allow(clippy::type_complexity)]
    couplings: Vec<(usize, usize, Box<dyn Fn(f64, &BlockVector) -> CsrMatrix<f64> + Send + Sync>)>,
}

impl MultiphysicsOperator {
    pub fn new(field_sizes: Vec<usize>) -> Self {
        let n = field_sizes.len();
        Self { n_fields: n, field_sizes, diag_csr: (0..n).map(|_| None).collect(), couplings: Vec::new() }
    }

    pub fn set_diagonal_csr(&mut self, k: usize, mat: CsrMatrix<f64>) { self.diag_csr[k] = Some(mat); }

    #[allow(clippy::type_complexity)]
    pub fn add_coupling(
        &mut self, i: usize, j: usize,
        gen: Box<dyn Fn(f64, &BlockVector) -> CsrMatrix<f64> + Send + Sync>,
    ) { self.couplings.push((i, j, gen)); }

    pub fn assemble_block_matrix(&self, t: f64, state: &BlockVector) -> BlockMatrix {
        let sizes = self.field_sizes.clone();
        let mut bm = BlockMatrix::new_square(sizes);
        for k in 0..self.n_fields {
            if let Some(ref m) = self.diag_csr[k] { bm.set(k, k, m.clone()); }
        }
        for &(i, j, ref gen) in &self.couplings { bm.set(i, j, gen(t, state)); }
        bm
    }

    pub fn n_fields(&self) -> usize { self.n_fields }
    pub fn field_sizes(&self) -> &[usize] { &self.field_sizes }
}

impl BlockOperator for MultiphysicsOperator {
    fn n_blocks(&self) -> usize { self.n_fields }
    fn row_sizes(&self) -> &[usize] { &self.field_sizes }
    fn col_sizes(&self) -> &[usize] { &self.field_sizes }
    fn apply_block(&self, i: usize, j: usize, x: &[f64], y: &mut [f64]) {
        if i == j { if let Some(ref m) = self.diag_csr[i] { m.spmv(x, y); } }
    }
    fn has_block(&self, i: usize, j: usize) -> bool {
        if i == j { self.diag_csr[i].is_some() }
        else { self.couplings.iter().any(|&(ci, cj, _)| ci == i && cj == j) }
    }
}

// ─── BlockNonlinearForm ─────────────────────────────────────────────────────

/// Block nonlinear form — auto-assembles block Jacobian from per-field integrators.
///
/// `F` is the field context type (e.g. `H1Space`, `VectorH1Space`, or `()`) passed
/// to integrator closures. The block sizes are given explicitly at construction.
pub struct BlockNonlinearForm<F> {
    field_data: Vec<F>,
    block_sizes: Vec<usize>,
    #[allow(clippy::type_complexity)]
    field_integrators: Vec<Vec<Box<dyn Fn(&F) -> CsrMatrix<f64> + Send + Sync>>>,
    #[allow(clippy::type_complexity)]
    coupling_integrators: Vec<Box<dyn Fn(&F, &F, f64, &[f64], &[f64]) -> CsrMatrix<f64> + Send + Sync>>,
}

impl<F: Send + Sync> BlockNonlinearForm<F> {
    pub fn new(field_data: Vec<F>, block_sizes: Vec<usize>) -> Self {
        assert_eq!(field_data.len(), block_sizes.len());
        let n = field_data.len();
        Self {
            field_data,
            block_sizes,
            field_integrators: (0..n).map(|_| Vec::new()).collect(),
            coupling_integrators: Vec::new(),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn add_diagonal_integrator(
        &mut self, k: usize,
        f: Box<dyn Fn(&F) -> CsrMatrix<f64> + Send + Sync>,
    ) { self.field_integrators[k].push(f); }

    #[allow(clippy::type_complexity)]
    pub fn add_coupling_integrator(
        &mut self,
        f: Box<dyn Fn(&F, &F, f64, &[f64], &[f64]) -> CsrMatrix<f64> + Send + Sync>,
    ) { self.coupling_integrators.push(f); }

    /// Assemble block Jacobian at state. Returns (block_sizes, block_matrix).
    pub fn assemble_block_jacobian(&self, t: f64, state: &BlockVector) -> (Vec<usize>, BlockMatrix) {
        let sizes = self.block_sizes.clone();
        let mut jac = BlockMatrix::new_square(sizes.clone());

        for k in 0..self.field_data.len() {
            let mut coo = CooMatrix::new(sizes[k], sizes[k]);
            for f in &self.field_integrators[k] {
                let bk = f(&self.field_data[k]);
                for i in 0..bk.nrows {
                    for p in bk.row_ptr[i]..bk.row_ptr[i+1] {
                        coo.add(i, bk.col_idx[p] as usize, bk.values[p]);
                    }
                }
            }
            let diag = coo.into_csr();
            if diag.nnz() > 0 { jac.set(k, k, diag); }
        }

        for ci in &self.coupling_integrators {
            for i in 0..self.field_data.len() {
                for j in 0..self.field_data.len() {
                    if i == j { continue; }
                    let block = ci(&self.field_data[i], &self.field_data[j], t, state.block(i), state.block(j));
                    jac.set(i, j, block);
                }
            }
        }

        (sizes, jac)
    }

    pub fn block_sizes(&self) -> &[usize] { &self.block_sizes }
    pub fn n_fields(&self) -> usize { self.field_data.len() }
}

// ─── NxN block preconditioners ──────────────────────────────────────────────

/// Per-block solver: `solve(M_i, rhs) -> solution` used by block preconditioners.
pub type BlockSolver = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

/// Block-diagonal preconditioner for NxN block systems.
///
/// Applies per-block solvers independently: `z_i = M_i^{-1} r_i`.
/// Compatible with flat GMRES when wrapped — the callback maps between
/// flat vectors and block views.
pub struct BlockDiagonalPrecondN {
    solvers: Vec<BlockSolver>,
    offsets: Vec<usize>,
}

impl BlockDiagonalPrecondN {
    pub fn new(solvers: Vec<BlockSolver>, block_sizes: Vec<usize>) -> Self {
        let mut offsets = Vec::with_capacity(block_sizes.len() + 1);
        offsets.push(0);
        for s in &block_sizes { offsets.push(offsets.last().unwrap() + s); }
        Self { solvers, offsets }
    }

    fn n_blocks(&self) -> usize { self.solvers.len() }

    /// Apply preconditioner to a flat vector, writing into `z`.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        for i in 0..self.n_blocks() {
            let lo = self.offsets[i];
            let hi = self.offsets[i + 1];
            let zi = self.solvers[i](&r[lo..hi]);
            z[lo..hi].copy_from_slice(&zi);
        }
    }

    /// Return a closure usable as a right-preconditioner in GMRES.
    pub fn as_precond_closure(&self) -> impl Fn(&[f64], &mut [f64]) + Send + Sync + '_ {
        |r, z| self.apply(r, z)
    }
}

/// Block upper-triangular preconditioner for NxN block systems.
///
/// Solves `U z = r` where U is block upper-triangular, by back-substitution:
/// `z_n = M_nn^{-1} r_n`, then `z_{n-1} = M_{n-1,n-1}^{-1}(r_{n-1} - M_{n-1,n} z_n)`, etc.
pub struct BlockTriangularPrecondN {
    solvers: Vec<BlockSolver>,
    /// Off-diagonal coupling blocks: (i, j, matrix) for i < j
    upper_blocks: Vec<(usize, usize, CsrMatrix<f64>)>,
    offsets: Vec<usize>,
}

impl BlockTriangularPrecondN {
    pub fn new(
        solvers: Vec<BlockSolver>,
        upper_blocks: Vec<(usize, usize, CsrMatrix<f64>)>,
        block_sizes: Vec<usize>,
    ) -> Self {
        let mut offsets = Vec::with_capacity(block_sizes.len() + 1);
        offsets.push(0);
        for s in &block_sizes { offsets.push(offsets.last().unwrap() + s); }
        Self { solvers, upper_blocks, offsets }
    }

    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.solvers.len();
        // Back-substitution: solve from last block to first
        for i in (0..n).rev() {
            let lo_i = self.offsets[i];
            let hi_i = self.offsets[i + 1];
            // Compute residual: r_i - sum_{j>i} M_ij * z_j
            let mut residual = r[lo_i..hi_i].to_vec();
            for &(bi, bj, ref mat) in &self.upper_blocks {
                if bi == i && bj > i {
                    let lo_j = self.offsets[bj];
                    let hi_j = self.offsets[bj + 1];
                    let mut tmp = vec![0.0_f64; hi_i - lo_i];
                    mat.spmv(&z[lo_j..hi_j], &mut tmp);
                    for k in 0..residual.len() {
                        residual[k] -= tmp[k];
                    }
                }
            }
            let zi = self.solvers[i](&residual);
            z[lo_i..hi_i].copy_from_slice(&zi);
        }
    }

    pub fn n_blocks(&self) -> usize { self.solvers.len() }

    pub fn as_precond_closure(&self) -> impl Fn(&[f64], &mut [f64]) + Send + Sync + '_ {
        |r, z| self.apply(r, z)
    }
}

/// Convenience: build a block-diagonal solver using diagonal Jacobi per block.
///
/// Each block uses `diag(M_i)^{-1}` as the approximate inverse.
pub fn build_jacobi_block_solvers(block_sizes: &[usize], jac: &BlockMatrix) -> Vec<BlockSolver> {
    let mut solvers = Vec::with_capacity(block_sizes.len());
    for i in 0..block_sizes.len() {
        let diag = jac.get(i, i).map(|m| {
            (0..m.nrows).map(|r| {
                let d = m.get(r, r);
                if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
            }).collect::<Vec<f64>>()
        }).unwrap_or_else(|| vec![1.0; block_sizes[i]]);
        let d_clone: Vec<f64> = diag.clone();
        let n = block_sizes[i];
        let solver: BlockSolver = Box::new(move |r: &[f64]| -> Vec<f64> {
            (0..n).map(|k| d_clone[k] * r[k]).collect()
        });
        solvers.push(solver);
    }
    solvers
}

/// Build block upper-triangular coupling blocks from a BlockMatrix for i < j.
pub fn extract_upper_coupling(jac: &BlockMatrix) -> Vec<(usize, usize, CsrMatrix<f64>)> {
    let mut blocks = Vec::new();
    for i in 0..jac.n_row_blocks() {
        for j in (i + 1)..jac.n_col_blocks() {
            if let Some(m) = jac.get(i, j) {
                blocks.push((i, j, m.clone()));
            }
        }
    }
    blocks
}

/// Solve a block system using GMRES with a block-diagonal preconditioner.
///
/// The system matrix is the flat CSR from `block_matrix_to_csr`, and the
/// preconditioner applies per-block solvers on the flat vector.
///
/// Returns the solution as a flat vector.
pub fn solve_block_precond_gmres(
    jac: &BlockMatrix,
    rhs: &[f64],
    precond: &BlockDiagonalPrecondN,
    restart: usize,
    cfg: &SolverConfig,
) -> Result<(Vec<f64>, SolveResult), SolverError> {
    let flat = block_matrix_to_csr(jac);
    let n = flat.nrows;
    let mut x = vec![0.0_f64; n];
    let precond_fn = precond.as_precond_closure();
    let result = right_preconditioned_gmres(&flat, rhs, &mut x, restart, cfg, precond_fn)?;
    Ok((x, result))
}

/// Right-preconditioned GMRES(m) with a closure preconditioner.
pub fn right_preconditioned_gmres(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    cfg: &SolverConfig,
    precond: impl Fn(&[f64], &mut [f64]),
) -> Result<SolveResult, SolverError> {
    let n = a.nrows;
    let b_norm = norm2(b);
    let tol = (cfg.rtol * b_norm).max(cfg.atol);
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }

    let mut total_iters = 0;

    for _cycle in 0..cfg.max_iter.div_ceil(restart) {
        let mut r = b.to_vec();
        for i in 0..n {
            for p in a.row_ptr[i]..a.row_ptr[i+1] {
                r[i] -= a.values[p] * x[a.col_idx[p] as usize];
            }
        }
        let beta = norm2(&r);
        if beta < tol { return Ok(SolveResult { converged: true, iterations: total_iters, final_residual: beta }); }

        let m = restart;
        let mut v = vec![vec![0.0; n]; m + 1];
        let mut z = vec![vec![0.0; n]; m];
        let mut h = vec![vec![0.0_f64; m]; m + 1];
        let mut cs = vec![0.0_f64; m];
        let mut sn = vec![0.0_f64; m];
        let mut e1 = vec![0.0_f64; m + 1];
        e1[0] = beta;
        for i in 0..n { v[0][i] = r[i] / beta; }

        let mut j = 0;
        while j < m && total_iters < cfg.max_iter {
            precond(&v[j], &mut z[j]);
            let mut w = vec![0.0_f64; n];
            for i in 0..n { for p in a.row_ptr[i]..a.row_ptr[i+1] { w[i] += a.values[p] * z[j][a.col_idx[p] as usize]; } }
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                for k in 0..n { w[k] -= h[i][j] * v[i][k]; }
            }
            h[j+1][j] = norm2(&w);
            if h[j+1][j] > 1e-16 { for i in 0..n { v[j+1][i] = w[i] / h[j+1][j]; } }
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i+1][j];
                h[i+1][j] = -sn[i] * h[i][j] + cs[i] * h[i+1][j];
                h[i][j] = tmp;
            }
            let rv = (h[j][j] * h[j][j] + h[j+1][j] * h[j+1][j]).sqrt();
            cs[j] = h[j][j] / rv; sn[j] = h[j+1][j] / rv;
            h[j][j] = rv; h[j+1][j] = 0.0;
            let tmp = cs[j] * e1[j] + sn[j] * e1[j+1];
            e1[j+1] = -sn[j] * e1[j] + cs[j] * e1[j+1];
            e1[j] = tmp;
            total_iters += 1;
            if e1[j+1].abs() < tol { j += 1; break; }
            j += 1;
        }

        let k = j;
        let mut y = vec![0.0_f64; k];
        for i in (0..k).rev() {
            y[i] = e1[i];
            for jj in i+1..k { y[i] -= h[i][jj] * y[jj]; }
            y[i] /= h[i][i];
        }
        for jj in 0..k { for i in 0..n { x[i] += y[jj] * z[jj][i]; } }

        let mut r_check = b.to_vec();
        for i in 0..n { for p in a.row_ptr[i]..a.row_ptr[i+1] { r_check[i] -= a.values[p] * x[a.col_idx[p] as usize]; } }
        let final_res = norm2(&r_check);
        if final_res < tol || total_iters >= cfg.max_iter {
            return Ok(SolveResult { converged: final_res < tol, iterations: total_iters, final_residual: final_res });
        }
    }
    Ok(SolveResult { converged: false, iterations: total_iters, final_residual: f64::NAN })
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)| x*y).sum() }
fn norm2(v: &[f64]) -> f64 { dot(v, v).sqrt() }

fn block_matrix_to_csr(a: &BlockMatrix) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.total_rows(), a.total_cols());
    let mut row_offset = 0usize;
    for bi in 0..a.n_row_blocks() {
        let mut col_offset = 0usize;
        for bj in 0..a.n_col_blocks() {
            if let Some(b) = a.get(bi, bj) {
                for r in 0..b.nrows {
                    let start = b.row_ptr[r]; let end = b.row_ptr[r+1];
                    for k in start..end {
                        coo.add(row_offset + r, col_offset + b.col_idx[k] as usize, b.values[k]);
                    }
                }
            }
            col_offset += a.col_sizes[bj];
        }
        row_offset += a.row_sizes[bi];
    }
    coo.into_csr()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    #[test]
    fn block_op_matrix_basic() {
        let mut bm = BlockMatrix::new_square(vec![2, 2]);
        let mut coo = CooMatrix::new(2, 2);
        coo.add(0, 0, 1.0); coo.add(1, 1, 1.0);
        bm.set(0, 0, coo.into_csr());
        let mut coo = CooMatrix::new(2, 2);
        coo.add(0, 0, 1.0); coo.add(1, 1, 1.0);
        bm.set(1, 1, coo.into_csr());

        let op = BlockOpMatrix::new(bm);
        assert_eq!(op.n_blocks(), 2);
        assert!(op.has_block(0, 0));
        assert!(!op.has_block(0, 1));
    }

    #[test]
    fn sum_block_op_adds_values() {
        let mut bm1 = BlockMatrix::new_square(vec![1]);
        let mut coo = CooMatrix::new(1, 1); coo.add(0, 0, 2.0);
        bm1.set(0, 0, coo.into_csr());
        let mut bm2 = BlockMatrix::new_square(vec![1]);
        let mut coo = CooMatrix::new(1, 1); coo.add(0, 0, 3.0);
        bm2.set(0, 0, coo.into_csr());

        let sum = SumBlockOp::new(Box::new(BlockOpMatrix::new(bm1)), Box::new(BlockOpMatrix::new(bm2)));
        let mut x = BlockVector::new(vec![1]);
        x.block_mut(0)[0] = 1.0;
        let mut y = BlockVector::new(vec![1]);
        sum.apply(&x, &mut y);
        assert!((y.block(0)[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn multiphysics_operator_assemble_all_blocks() {
        let mut mphys = MultiphysicsOperator::new(vec![1, 1]);
        let mut coo = CooMatrix::new(1, 1); coo.add(0, 0, 2.0);
        mphys.set_diagonal_csr(0, coo.into_csr());
        let mut coo = CooMatrix::new(1, 1); coo.add(0, 0, 3.0);
        mphys.set_diagonal_csr(1, coo.into_csr());

        mphys.add_coupling(0, 1, Box::new(|_t, _st| {
            let mut coo = CooMatrix::new(1, 1); coo.add(0, 0, 1.0); coo.into_csr()
        }));

        let state = BlockVector::new(vec![1, 1]);
        let bm = mphys.assemble_block_matrix(0.0, &state);
        assert!(bm.get(0, 0).is_some());
        assert!(bm.get(1, 1).is_some());
        assert!(bm.get(0, 1).is_some());
    }
}
