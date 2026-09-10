//! Serial port of MFEM `miniapps/solvers/div_free_solver.{hpp,cpp}` (1:1,
//! assembled serial variant — `HypreParMatrix` becomes [`CsrMatrix`]).
//!
//! Ported classes (C++ name → Rust name):
//!
//! | C++                           | Rust                        |
//! |-------------------------------|-----------------------------|
//! | `DFSParameters`               | [`DfsParameters`]           |
//! | `DFSData`                     | [`DfsData`]                 |
//! | `BBTSolver` (CG+AMG on `BBᵀ`) | [`BbtSolver`]               |
//! | `SymDirectSubBlockSolver`     | [`SymDirectSubBlockSolver`] |
//! | `LocalSolver`                 | [`LocalSolver`]             |
//! | `SaddleSchwarzSmoother`       | [`SaddleSchwarzSmoother`]   |
//! | `AuxSpaceSmoother`            | [`AuxSpaceSmoother`]        |
//! | `ProductSolver` (hybrid)      | [`ProductSmoother`]         |
//! | `Multigrid` (V-cycle)         | [`MgVcycle`]                |
//! | `DivFreeSolver`               | [`DivFreeSolver`]           |
//!
//! The space-dependent collector (C++ `DFSSpaces`: per-level RT/L2/H1 spaces,
//! refinement prolongations `P_hdiv`/`P_l2`/`P_hcurl`, aggregate→dof tables,
//! discrete curl `C`, `Q_l2`) belongs to the FE-space layer and lives in the
//! block-solvers miniapp; this module consumes the resulting matrices only.
//!
//! Deviations from C++ (recorded):
//! * hypre `BoomerAMG` → fem-amg with the hypre-default-aligned preset
//!   ([`fem_amg::boomeramg_config`]: Ruge–Stüben coarsening + symmetric
//!   Gauss–Seidel smoothing; iteration counts differ slightly, solution
//!   accuracy is unaffected),
//! * hypre `HypreSmoother` (l1-Jacobi) → [`L1JacobiSmoother`] with a
//!   zero-diagonal guard,
//! * hypre `DropSmallEntries` is not applied (the assembled products are
//!   exact, so dropping tiny entries is a pure memory optimization),
//! * the `Ae` matrices (C with eliminated essential columns) are not stored —
//!   they are unused by `DivFreeSolver` in C++ as well (only
//!   `MLDivFreeSolver` consumes them),
//! * PA / matrix-free variants are **not** ported (gap).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use fem_linalg::{CooMatrix, CsrMatrix};

use crate::darcy_solvers::{BdpMinresSolver, IterSolveParameters, SchurMode};

/// MFEM `blocksolvers::DFSParameters`.
#[derive(Debug, Clone)]
pub struct DfsParameters {
    /// `true`: one coupled GMRES solve; `false`: the particular, div-free and
    /// potential parts are solved one by one.
    pub coupled_solve: bool,
    pub verbose: bool,
    /// Parameters for the coarsest-level [`BdpMinresSolver`].
    pub coarse_solve_param: IterSolveParameters,
    /// Parameters for the `BBᵀ` (potential) CG solver.
    pub bbt_solve_param: IterSolveParameters,
    /// Parameters for the outer solve (GMRES in coupled mode, CG on `CᵀMC`
    /// in decoupled mode).
    pub outer_solve_param: IterSolveParameters,
    /// GMRES restart dimension (coupled mode).
    pub gmres_restart: usize,
    /// Schur preconditioner of the coarsest-level BDP solve.
    pub coarse_schur_mode: SchurMode,
}

impl Default for DfsParameters {
    fn default() -> Self {
        Self {
            coupled_solve: false,
            verbose: false,
            coarse_solve_param: IterSolveParameters::default(),
            bbt_solve_param: IterSolveParameters::default(),
            outer_solve_param: IterSolveParameters::default(),
            gmres_restart: 50,
            // AMG Schur inverse on the coarsest level (C++ hypre BoomerAMG
            // analogue, `boomeramg_config()` preset).  Dense remains available
            // as an exact fallback for small coarse systems.
            coarse_schur_mode: SchurMode::Amg,
        }
    }
}

/// MFEM `blocksolvers::DFSData` — per-level matrices collected by the
/// space-layer collector (C++ `DFSSpaces`).  Level `0` is the coarsest mesh;
/// `P_*[l]` prolongates level `l` → level `l+1`; `l == P_l2.len()` is finest.
pub struct DfsData {
    /// `agg_hdivdof[l]`: level-`l` element → level-`l+1` interior H(div) dofs
    /// (boolean matrix).
    pub agg_hdivdof: Vec<CsrMatrix<f64>>,
    /// `agg_l2dof[l]`: level-`l` element → level-`l+1` L2 dofs (boolean).
    pub agg_l2dof: Vec<CsrMatrix<f64>>,
    /// `P_hdiv[l]`: H(div) prolongation, level `l` → `l+1`.
    pub p_hdiv: Vec<CsrMatrix<f64>>,
    /// `P_l2[l]`: L2 prolongation, level `l` → `l+1`.
    pub p_l2: Vec<CsrMatrix<f64>>,
    /// `P_hcurl[l]` (2-D: H1) prolongation of the kernel space, level `l` →
    /// `l+1`.
    pub p_aux: Vec<CsrMatrix<f64>>,
    /// Coarsest-level essential H(div) dofs.
    pub coarsest_ess_hdivdofs: Vec<usize>,
    /// `C[l]`: discrete curl (aux space → H(div)) at level `l`; `C[0]` is
    /// empty.  Essential columns are eliminated by the collector (hypre
    /// `EliminateCols`).
    pub c: Vec<CsrMatrix<f64>>,
    /// `Q_l2[l] = (P_l2 W P_l2ᵀ)⁻¹_blockwise P_l2ᵀ W` (MFEM `DataFinalize`).
    pub q_l2: Vec<Arc<Ql2Projector>>,
    /// Solver parameters (shared by all solvers built on this data).
    pub param: DfsParameters,
}

impl DfsData {
    /// Single-level data (`par_ref_levels == 0`): [`DivFreeSolver`] then
    /// degenerates to the coarsest BDP solve (see the C++ warning
    /// "DivFree solver is equivalent to BDPMinresSolver when
    /// par_ref_levels == 0").
    pub fn single_level(param: DfsParameters) -> Self {
        Self {
            agg_hdivdof: Vec::new(),
            agg_l2dof: Vec::new(),
            p_hdiv: Vec::new(),
            p_l2: Vec::new(),
            p_aux: Vec::new(),
            coarsest_ess_hdivdofs: Vec::new(),
            c: vec![CooMatrix::<f64>::new(0, 0).into_csr()],
            q_l2: Vec::new(),
            param,
        }
    }
}

/// MFEM `ProductOperator(cW_inv, PTW)` — `Q = cW⁻¹·PᵀW` with blockwise exact
/// `cW⁻¹`:
/// * `Q x  = cW⁻¹ (Pᵀ (W x))`,
/// * `Qᵀ x = W (P (cW⁻¹ x))`.
pub struct Ql2Projector {
    cw_inv: SymDirectSubBlockSolver,
    p: CsrMatrix<f64>,
    w: CsrMatrix<f64>,
}

impl Ql2Projector {
    /// `cW = Pᵀ W P` (level-`l` L2 mass), inverted blockwise over the
    /// level-`l` element→dof table `block_dof`.
    pub fn new(cw: &CsrMatrix<f64>, block_dof: &CsrMatrix<f64>, p: CsrMatrix<f64>, w: CsrMatrix<f64>) -> Self {
        Self {
            cw_inv: SymDirectSubBlockSolver::new(cw, block_dof),
            p,
            w,
        }
    }

    /// `Q x`
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        let n_fine = self.w.nrows;
        debug_assert_eq!(x.len(), n_fine);
        let mut wx = vec![0.0; n_fine];
        self.w.spmv(x, &mut wx);
        let mut ptwx = vec![0.0; self.p.ncols];
        self.p.transpose().spmv(&wx, &mut ptwx);
        self.cw_inv.mult(&ptwx, y);
    }

    /// `Qᵀ x`
    pub fn mult_transpose(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.p.ncols);
        let mut cwinv_x = vec![0.0; x.len()];
        self.cw_inv.mult(x, &mut cwinv_x);
        let mut pcx = vec![0.0; self.p.nrows];
        self.p.spmv(&cwinv_x, &mut pcx);
        self.w.spmv(&pcx, y);
    }
}

/// MFEM `blocksolvers::SymDirectSubBlockSolver`: exact block-diagonal inverse
/// (each block factorized with a dense LU).
pub struct SymDirectSubBlockSolver {
    blocks: Vec<BlockLu>,
}

struct BlockLu {
    dofs: Vec<usize>,
    lu: Vec<f64>,
    piv: Vec<usize>,
}

impl SymDirectSubBlockSolver {
    pub fn new(a: &CsrMatrix<f64>, block_dof: &CsrMatrix<f64>) -> Self {
        assert_eq!(a.nrows, a.ncols);
        let mut blocks = Vec::with_capacity(block_dof.nrows);
        for agg in 0..block_dof.nrows {
            let dofs: Vec<usize> = block_dof.col_idx[block_dof.row_ptr[agg]..block_dof.row_ptr[agg + 1]]
                .iter()
                .map(|&d| d as usize)
                .collect();
            let m = dofs.len();
            let mut lu = vec![0.0; m * m];
            for (li, &di) in dofs.iter().enumerate() {
                for p in a.row_ptr[di]..a.row_ptr[di + 1] {
                    let dj = a.col_idx[p] as usize;
                    if let Some(lj) = dofs.iter().position(|&d| d == dj) {
                        lu[li * m + lj] = a.values[p];
                    }
                }
            }
            let mut piv = vec![0_usize; m];
            fem_linalg::dense::lu_factor(&mut lu, m, &mut piv)
                .unwrap_or_else(|e| panic!("SymDirectSubBlockSolver: singular block (agg {agg}): {e:?}"));
            blocks.push(BlockLu { dofs, lu, piv });
        }
        Self { blocks }
    }

    /// `y ← A_blockdiag⁻¹ x` (MFEM `Mult`; this solver is its own transpose).
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
        for b in &self.blocks {
            let m = b.dofs.len();
            let mut sub = vec![0.0; m];
            for (li, &d) in b.dofs.iter().enumerate() {
                sub[li] = x[d];
            }
            fem_linalg::dense::lu_solve(&b.lu, m, &b.piv, &mut sub);
            for (li, &d) in b.dofs.iter().enumerate() {
                y[d] = sub[li];
            }
        }
    }
}

/// MFEM `blocksolvers::BBTSolver`: CG preconditioned by AMG on `B·Bᵀ`.
///
/// C++ uses hypre BoomerAMG; the serial port uses l1-Jacobi-preconditioned CG
/// (`B·Bᵀ` is a well-conditioned weighted graph Laplacian for RT0-Darcy, so
/// this stays robust without an AMG hierarchy).
pub struct BbtSolver {
    bbt: CsrMatrix<f64>,
    l1: L1JacobiSmoother,
    param: IterSolveParameters,
    last_iters: AtomicUsize,
}

impl BbtSolver {
    pub fn new(b: &CsrMatrix<f64>, param: IterSolveParameters) -> Self {
        let bbt = b.multiply(&b.transpose());
        let l1 = L1JacobiSmoother::new(&bbt);
        Self {
            bbt,
            l1,
            param,
            last_iters: AtomicUsize::new(0),
        }
    }

    /// `y ← (B·Bᵀ)⁻¹ x`; returns the CG iteration count.  A (numerically)
    /// zero rhs skips the solve (`y = 0`).
    pub fn mult(&self, x: &[f64], y: &mut [f64]) -> usize {
        let xn = x.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        if xn < 1e-14 {
            y.fill(0.0);
            self.last_iters.store(0, Ordering::Relaxed);
            return 0;
        }
        let n = self.bbt.nrows;
        let res = crate::solve_pcg_operator_precond(
            n,
            |v, w| self.bbt.spmv(v, w),
            x,
            y,
            |v, w| self.l1.mult(v, w),
            &fem_linalg::SolverConfig {
                rtol: self.param.rel_tol,
                atol: self.param.abs_tol,
                max_iter: self.param.max_iter,
                verbose: false,
                print_level: fem_linalg::PrintLevel::Silent,
            },
        )
        .unwrap_or_else(|e| panic!("BBTSolver: CG on BBᵀ failed: {e:?}"));
        self.last_iters.store(res.iterations, Ordering::Relaxed);
        res.iterations
    }

    pub fn num_iterations(&self) -> usize {
        self.last_iters.load(Ordering::Relaxed)
    }
}

/// MFEM `blocksolvers::LocalSolver`: dense solver of the local saddle system
///
/// ```text
/// [[M_loc, B_locᵀ], [B_loc, −1]]
/// ```
///
/// (row/col `offset` zeroed, `−1` in the corner — the system is then always
/// invertible; the corresponding rhs entry is zeroed inside [`LocalSolver::mult`]).
pub struct LocalSolver {
    lu: Vec<f64>,
    piv: Vec<usize>,
    n: usize,
    offset: usize,
}

impl LocalSolver {
    /// `m_loc`: (n_u × n_u) row-major, `b_loc`: (n_p × n_u) row-major.
    pub fn new(m_loc: &[f64], b_loc: &[f64], n_u: usize, n_p: usize) -> Self {
        let n = n_u + n_p;
        let mut a = vec![0.0; n * n];
        for (i, j, v) in dense_rows(m_loc, n_u, n_u) {
            a[i * n + j] = v;
        }
        // CopyMN(B, offset, 0): rows [n_u..] × cols [0..n_u) = B_loc
        for (i, j, v) in dense_rows(b_loc, n_p, n_u) {
            a[(n_u + i) * n + j] = v;
            // CopyMNt(B, 0, offset): rows [0..n_u) × cols [n_u..] = B_locᵀ
            a[j * n + (n_u + i)] = v;
        }
        let off = n_u;
        for j in 0..n {
            a[off * n + j] = 0.0;
            a[j * n + off] = 0.0;
        }
        a[off * n + off] = -1.0;
        let mut piv = vec![0_usize; n];
        fem_linalg::dense::lu_factor(&mut a, n, &mut piv)
            .expect("LocalSolver: local saddle system is singular");
        Self { lu: a, piv, n, offset: off }
    }

    /// MFEM `LocalSolver::Mult` (temporarily zeroes `x[offset]`).
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        let mut rhs = x.to_vec();
        rhs[self.offset] = 0.0;
        fem_linalg::dense::lu_solve(&self.lu, self.n, &self.piv, &mut rhs);
        y.copy_from_slice(&rhs);
    }
}

/// Iterate the nonzero entries of a dense row-major matrix.
fn dense_rows(a: &[f64], rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
    a.iter().enumerate().filter_map(move |(k, &v)| {
        if v != 0.0 {
            Some((k / cols, k % cols, v))
        } else {
            debug_assert!(rows * cols == a.len());
            None
        }
    })
}

/// `P_l2 ∘ Q_l2` — removes the coarse-range component of the pressure rhs so
/// that the local saddle problems are compatible (MFEM `coarse_l2_projector_`).
struct CoarseL2Projector {
    p: CsrMatrix<f64>,
    q: Arc<Ql2Projector>,
}

impl CoarseL2Projector {
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        let mut qx = vec![0.0; self.q.p.ncols];
        self.q.mult(x, &mut qx);
        self.p.spmv(&qx, y);
    }

    fn mult_transpose(&self, x: &[f64], y: &mut [f64]) {
        let mut ptx = vec![0.0; self.p.ncols];
        self.p.transpose().spmv(x, &mut ptx);
        self.q.mult_transpose(&ptx, y);
    }
}

/// MFEM `blocksolvers::SaddleSchwarzSmoother`: non-overlapping additive
/// Schwarz smoother for `[[M, Bᵀ], [B, 0]]`; local problems are defined on
/// element aggregates, with the coarse-L2 projection applied to the pressure
/// rhs to guarantee solvability.
pub struct SaddleSchwarzSmoother {
    n_u: usize,
    n_p: usize,
    agg_hdivdof: CsrMatrix<f64>,
    agg_l2dof: CsrMatrix<f64>,
    local_solvers: Vec<LocalSolver>,
    coarse_proj: Arc<CoarseL2Projector>,
}

impl SaddleSchwarzSmoother {
    /// `M`: (n_u × n_u) SPD, `B`: (n_p × n_u), `agg_hdivdof`/`agg_l2dof`:
    /// aggregate→dof boolean tables, `P_l2` + `Q_l2` as in the C++
    /// constructor.
    pub fn new(
        m: &CsrMatrix<f64>,
        b: &CsrMatrix<f64>,
        agg_hdivdof: &CsrMatrix<f64>,
        agg_l2dof: &CsrMatrix<f64>,
        p_l2: &CsrMatrix<f64>,
        q_l2: Arc<Ql2Projector>,
    ) -> Self {
        let n_u = m.nrows;
        let n_p = b.nrows;
        let n_agg = agg_l2dof.nrows;
        let mut local_solvers = Vec::with_capacity(n_agg);
        for agg in 0..n_agg {
            let hd = col_indices(agg_hdivdof, agg);
            let ld = col_indices(agg_l2dof, agg);
            let nu = hd.len();
            let np = ld.len();
            let mut m_loc = vec![0.0; nu * nu];
            for (li, &di) in hd.iter().enumerate() {
                for p in m.row_ptr[di]..m.row_ptr[di + 1] {
                    let dj = m.col_idx[p] as usize;
                    if let Some(lj) = hd.iter().position(|&d| d == dj) {
                        m_loc[li * nu + lj] = m.values[p];
                    }
                }
            }
            let mut b_loc = vec![0.0; np * nu];
            for (li, &di) in ld.iter().enumerate() {
                for p in b.row_ptr[di]..b.row_ptr[di + 1] {
                    let dj = b.col_idx[p] as usize;
                    if let Some(lj) = hd.iter().position(|&d| d == dj) {
                        b_loc[li * nu + lj] = b.values[p];
                    }
                }
            }
            local_solvers.push(LocalSolver::new(&m_loc, &b_loc, nu, np));
        }
        Self {
            n_u,
            n_p,
            agg_hdivdof: agg_hdivdof.clone(),
            agg_l2dof: agg_l2dof.clone(),
            local_solvers,
            coarse_proj: Arc::new(CoarseL2Projector { p: p_l2.clone(), q: q_l2 }),
        }
    }

    fn agg_dofs(&self, agg: usize) -> (Vec<usize>, Vec<usize>) {
        (col_indices(&self.agg_hdivdof, agg), col_indices(&self.agg_l2dof, agg))
    }

    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        // Aggregate-wise average-free projection of the pressure part:
        // x_p ← x_p − P_l2 Q_l2 x_p  (transpose side before the local solves).
        let mut xp = x[self.n_u..].to_vec();
        let mut proj = vec![0.0; self.n_p];
        self.coarse_proj.mult_transpose(&xp, &mut proj);
        for (xi, &pi) in xp.iter_mut().zip(&proj) {
            *xi -= pi;
        }

        for (agg, solver) in self.local_solvers.iter().enumerate() {
            let (hd, ld) = self.agg_dofs(agg);
            let nu = hd.len();
            let np = ld.len();
            let mut rhs_loc = vec![0.0; nu + np];
            for (li, &d) in hd.iter().enumerate() {
                rhs_loc[li] = x[d];
            }
            for (li, &d) in ld.iter().enumerate() {
                rhs_loc[nu + li] = xp[d];
            }
            let mut sol_loc = vec![0.0; nu + np];
            solver.mult(&rhs_loc, &mut sol_loc);
            for (li, &d) in hd.iter().enumerate() {
                y[d] += sol_loc[li];
            }
            for (li, &d) in ld.iter().enumerate() {
                y[self.n_u + d] += sol_loc[nu + li];
            }
        }

        // y_p ← y_p − P_l2 Q_l2 y_p
        let mut proj = vec![0.0; self.n_p];
        let yp = y[self.n_u..].to_vec();
        self.coarse_proj.mult(&yp, &mut proj);
        for (yi, &pi) in y[self.n_u..].iter_mut().zip(&proj) {
            *yi -= pi;
        }
    }
}

/// Column indices of row `row` of a boolean CSR table.
fn col_indices(t: &CsrMatrix<f64>, row: usize) -> Vec<usize> {
    t.col_idx[t.row_ptr[row]..t.row_ptr[row + 1]]
        .iter()
        .map(|&d| d as usize)
        .collect()
}

/// `x − a` (elementwise).
fn sub_vec(x: &[f64], a: &[f64]) -> Vec<f64> {
    x.iter().zip(a).map(|(&xi, &ai)| xi - ai).collect()
}

/// l1-Jacobi smoother (hypre `HypreSmoother` stand-in) with a zero-diagonal
/// guard: rows with vanishing l1 norm contribute nothing (hypre instead adds
/// a tiny diagonal).
pub struct L1JacobiSmoother {
    d: Vec<f64>,
}

impl L1JacobiSmoother {
    pub fn new(a: &CsrMatrix<f64>) -> Self {
        let mut d = vec![0.0; a.nrows];
        for i in 0..a.nrows {
            let mut s = 0.0;
            for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                s += a.values[p].abs();
            }
            d[i] = if s.abs() < 1e-300 { 0.0 } else { 1.0 / s };
        }
        Self { d }
    }

    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        for (yi, (&xi, &di)) in y.iter_mut().zip(x.iter().zip(&self.d)) {
            *yi = xi * di;
        }
    }
}

/// MFEM `AuxSpaceSmoother`: `y = C · smooth(Cᵀ M C) · Cᵀ x`, the aux system
/// (`CᵀMC`, zero rows eliminated) smoothed by l1-Jacobi.
pub struct AuxSpaceSmoother {
    c: CsrMatrix<f64>,
    smoother: L1JacobiSmoother,
}

impl AuxSpaceSmoother {
    /// `op`: the (n_u × n_u) block (M), `c`: aux (curl) map at this level.
    pub fn new(op: &CsrMatrix<f64>, c: &CsrMatrix<f64>) -> Self {
        let mc = op.multiply(c);
        let mut aux = c.transpose().multiply(&mc);
        eliminate_zero_rows(&mut aux);
        let smoother = L1JacobiSmoother::new(&aux);
        Self { c: c.clone(), smoother }
    }

    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        let na = self.c.ncols;
        let mut aux_rhs = vec![0.0; na];
        self.c.transpose().spmv(x, &mut aux_rhs);
        let mut aux_sol = vec![0.0; na];
        self.smoother.mult(&aux_rhs, &mut aux_sol);
        self.c.spmv(&aux_sol, y);
    }
}

/// Zero out (numerically) empty rows — MFEM `SparseMatrix::EliminateZeroRows`.
pub fn eliminate_zero_rows(a: &mut CsrMatrix<f64>) {
    for i in 0..a.nrows {
        let row_max = a.values[a.row_ptr[i]..a.row_ptr[i + 1]]
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        if row_max < 1e-14 {
            for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                a.values[p] = 0.0;
            }
        }
    }
}

/// Symmetric saddle block operator `[[M, Bᵀ], [B, 0]]` of one level.
#[derive(Clone)]
pub struct SaddleOp {
    pub m: CsrMatrix<f64>,
    pub b: CsrMatrix<f64>,
    pub bt: CsrMatrix<f64>,
}

impl SaddleOp {
    pub fn new(m: CsrMatrix<f64>, b: CsrMatrix<f64>) -> Self {
        let bt = b.transpose();
        Self { m, b, bt }
    }

    pub fn n_u(&self) -> usize {
        self.m.nrows
    }

    pub fn size(&self) -> usize {
        self.m.nrows + self.b.nrows
    }

    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        let nu = self.n_u();
        self.m.spmv(&x[..nu], &mut y[..nu]);
        let mut bt_xp = vec![0.0; nu];
        self.bt.spmv(&x[nu..], &mut bt_xp);
        for (yi, bi) in y[..nu].iter_mut().zip(&bt_xp) {
            *yi += bi;
        }
        self.b.spmv(&x[..nu], &mut y[nu..]);
    }
}

/// Level smoother of the DFS hierarchies (MFEM `Solver*` in the multigrid).
pub trait BlockSmoother: Send + Sync {
    fn mult(&self, x: &[f64], y: &mut [f64]);
    /// Symmetric smoothers default `mult_transpose = mult`.
    fn mult_transpose(&self, x: &[f64], y: &mut [f64]) {
        self.mult(x, y);
    }
}

impl BlockSmoother for BdpMinresSolver {
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        BdpMinresSolver::mult(self, x, y)
    }
}

impl BlockSmoother for SaddleSchwarzSmoother {
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        SaddleSchwarzSmoother::mult(self, x, y)
    }
}

/// MFEM `BlockDiagonalPreconditioner(ops_offsets_[l])` with only diagonal
/// block 0 set (`AuxSpaceSmoother`); block 1 stays zero, as in C++.
struct ZeroBlockDiag {
    block0: AuxSpaceSmoother,
    n_u: usize,
}

impl BlockSmoother for ZeroBlockDiag {
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        self.block0.mult(&x[..self.n_u], &mut y[..self.n_u]);
        y[self.n_u..].fill(0.0);
    }
}

/// MFEM `ProductSolver`: hybrid smoother `y = S0 x + S1 (I − A S0) x`.
pub struct ProductSmoother {
    a: SaddleOp,
    s0: Box<dyn BlockSmoother>,
    s1: Box<dyn BlockSmoother>,
}

impl ProductSmoother {
    pub fn new(a: SaddleOp, s0: Box<dyn BlockSmoother>, s1: Box<dyn BlockSmoother>) -> Self {
        Self { a, s0, s1 }
    }
}

impl BlockSmoother for ProductSmoother {
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        let n = self.a.size();
        // y = S0 x
        self.s0.mult(x, y);
        // z = x − A y  (mult overwrites z)
        let mut z = vec![0.0; n];
        self.a.mult(y, &mut z);
        let z = sub_vec(x, &z);
        // y += S1 z
        let mut s1z = vec![0.0; n];
        self.s1.mult(&z, &mut s1z);
        for (yi, zi) in y.iter_mut().zip(&s1z) {
            *yi += zi;
        }
    }

    fn mult_transpose(&self, x: &[f64], y: &mut [f64]) {
        let n = self.a.size();
        self.s1.mult_transpose(x, y);
        // z = x − Aᵀ y  (A symmetric)
        let mut ay = vec![0.0; n];
        self.a.mult(y, &mut ay);
        let z = sub_vec(x, &ay);
        let mut s0z = vec![0.0; n];
        self.s0.mult_transpose(&z, &mut s0z);
        for (yi, zi) in y.iter_mut().zip(&s0z) {
            *yi += zi;
        }
    }
}

/// MFEM `Multigrid` with `VCYCLE`, 1 pre + 1 post smoothing step: generic
/// operator/smoother/prolongation hierarchy.  Level `ops.len()-1` is finest.
pub struct MgVcycle {
    ops: Vec<SaddleOp>,
    smoothers: Vec<Box<dyn BlockSmoother>>,
    /// `prolongations[l]`: flat prolongation from level `l` to level `l+1`.
    prolongations: Vec<CsrMatrix<f64>>,
}

impl MgVcycle {
    pub fn new(
        ops: Vec<SaddleOp>,
        smoothers: Vec<Box<dyn BlockSmoother>>,
        prolongations: Vec<CsrMatrix<f64>>,
    ) -> Self {
        assert_eq!(ops.len(), smoothers.len(), "MgVcycle: one smoother per level");
        assert_eq!(ops.len(), prolongations.len() + 1, "MgVcycle: one prolongation per coarse level");
        Self { ops, smoothers, prolongations }
    }

    fn cycle(&self, level: usize, x: &[f64], y: &mut [f64]) {
        if level == 0 {
            // SmoothingStep(0, zero=true): y = S x
            self.smoothers[0].mult(x, y);
            return;
        }
        let n = self.ops[level].size();
        // Pre-smooth: y = S x
        self.smoothers[level].mult(x, y);
        // r = x − A y
        let mut r = vec![0.0; n];
        self.ops[level].mult(y, &mut r);
        r = sub_vec(x, &r);
        // Restrict: x_c = Pᵀ r
        let p = &self.prolongations[level - 1];
        let mut xc = vec![0.0; p.ncols];
        p.transpose().spmv(&r, &mut xc);
        let mut yc = vec![0.0; p.ncols];
        self.cycle(level - 1, &xc, &mut yc);
        // Prolongate and add: y += P y_c
        let mut pc = vec![0.0; n];
        p.spmv(&yc, &mut pc);
        for (yi, pi) in y.iter_mut().zip(&pc) {
            *yi += pi;
        }
        // Post-smooth: y += Sᵀ (x − A y)
        self.ops[level].mult(y, &mut r);
        let r = sub_vec(x, &r);
        let mut z = vec![0.0; n];
        self.smoothers[level].mult_transpose(&r, &mut z);
        for (yi, zi) in y.iter_mut().zip(&z) {
            *yi += zi;
        }
    }

    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        self.cycle(self.ops.len() - 1, x, y);
    }
}

/// Flat block-diagonal prolongation `diag(P_hdiv, P_l2)` (level `l → l+1`).
pub fn flat_block_diagonal(p0: &CsrMatrix<f64>, p1: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n0 = p0.nrows;
    let m0 = p0.ncols;
    let mut coo = CooMatrix::<f64>::new(n0 + p1.nrows, m0 + p1.ncols);
    for i in 0..n0 {
        for p in p0.row_ptr[i]..p0.row_ptr[i + 1] {
            coo.add(i, p0.col_idx[p] as usize, p0.values[p]);
        }
    }
    for i in 0..p1.nrows {
        for p in p1.row_ptr[i]..p1.row_ptr[i + 1] {
            coo.add(n0 + i, m0 + p1.col_idx[p] as usize, p1.values[p]);
        }
    }
    coo.into_csr()
}

/// Galerkin product `Pᵀ A P`.
pub fn galerkin(a: &CsrMatrix<f64>, p: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let ap = a.multiply(p);
    p.transpose().multiply(&ap)
}

/// Multigrid for the decoupled-mode aux hierarchy (`CᵀMC` operators).
struct AuxMg {
    ops: Vec<CsrMatrix<f64>>,
    smoothers: Vec<L1JacobiSmoother>,
    prolongations: Vec<CsrMatrix<f64>>,
}

impl AuxMg {
    fn cycle(&self, level: usize, x: &[f64], y: &mut [f64]) {
        if level == 0 {
            self.smoothers[0].mult(x, y);
            return;
        }
        let n = self.ops[level].nrows;
        self.smoothers[level].mult(x, y);
        let mut r = vec![0.0; n];
        self.ops[level].spmv(y, &mut r);
        r = sub_vec(x, &r);
        let p = &self.prolongations[level - 1];
        let mut xc = vec![0.0; p.ncols];
        p.transpose().spmv(&r, &mut xc);
        let mut yc = vec![0.0; p.ncols];
        self.cycle(level - 1, &xc, &mut yc);
        let mut pc = vec![0.0; n];
        p.spmv(&yc, &mut pc);
        for (yi, pi) in y.iter_mut().zip(&pc) {
            *yi += pi;
        }
        self.ops[level].spmv(y, &mut r);
        let r = sub_vec(x, &r);
        let mut z = vec![0.0; n];
        self.smoothers[level].mult(&r, &mut z);
        for (yi, zi) in y.iter_mut().zip(&z) {
            *yi += zi;
        }
    }

    fn mult(&self, x: &[f64], y: &mut [f64]) {
        self.cycle(self.ops.len() - 1, x, y);
    }
}

/// MFEM `blocksolvers::DivFreeSolver`.
///
/// Exploits a multilevel decomposition of the Raviart-Thomas space to find a
/// particular solution satisfying the divergence constraint, then solves the
/// remaining divergence-free part in the kernel of the discrete divergence
/// operator (Vassilevski 2008, App. F.3; Voronin et al., JCP 373:863-876).
pub struct DivFreeSolver {
    n_u: usize,
    n: usize,
    finest: usize,
    coupled: bool,
    verbose: bool,
    /// All level operators (level 0 = coarsest … `finest` = finest).
    ops: Vec<SaddleOp>,
    /// Level smoothers (moved into the [`MgVcycle`] in coupled mode).
    smoothers: Vec<Box<dyn BlockSmoother>>,
    /// `blk_ps[l]`: flat prolongation level `l → l+1`.
    blk_ps: Vec<CsrMatrix<f64>>,
    bt: CsrMatrix<f64>,
    bbt_solver: BbtSolver,
    /// Coarsest-level BDP solve; the whole solver when `finest == 0`.
    coarse_bdp: Option<BdpMinresSolver>,
    // ── coupled mode ──────────────────────────────────────────────────────
    flat_finest: CsrMatrix<f64>,
    mg: Option<MgVcycle>,
    // ── decoupled mode ────────────────────────────────────────────────────
    c_finest: CsrMatrix<f64>,
    dec_ops: Vec<CsrMatrix<f64>>,
    dec_mg: Option<AuxMg>,
    // parameters
    outer_param: IterSolveParameters,
    gmres_restart: usize,
    last_iters: AtomicUsize,
    dec_cg_iters: AtomicUsize,
}

impl DivFreeSolver {
    /// MFEM `DivFreeSolver::DivFreeSolver(M, B, data)`.
    pub fn new(m: &CsrMatrix<f64>, b: &CsrMatrix<f64>, data: &DfsData) -> Self {
        let param = data.param.clone();
        let n_u = m.nrows;
        let n_p = b.nrows;
        let n = n_u + n_p;
        assert_eq!(m.ncols, n_u, "DivFreeSolver: M must be square");
        assert_eq!(b.ncols, n_u, "DivFreeSolver: B must be (n_p × n_u)");
        let n_levels = data.p_l2.len();
        let finest = n_levels;

        // ops[finest] = [[M, Bᵀ], [B, 0]]; coarser levels are Galerkin products.
        let mut ops: Vec<SaddleOp> = (0..=n_levels)
            .map(|_| SaddleOp::new(CooMatrix::<f64>::new(0, 0).into_csr(), CooMatrix::<f64>::new(0, 0).into_csr()))
            .collect();
        ops[finest] = SaddleOp::new(m.clone(), b.clone());

        let mut smoothers: Vec<Option<Box<dyn BlockSmoother>>> =
            (0..=n_levels).map(|_| None).collect();
        let mut blk_ps: Vec<CsrMatrix<f64>> =
            (0..n_levels).map(|_| CooMatrix::<f64>::new(0, 0).into_csr()).collect();

        let mut coarse_bdp: Option<BdpMinresSolver> = None;

        // Build levels from finest to coarsest (MFEM: `for l = size down to 0`).
        for l in (0..=n_levels).rev() {
            if l == 0 {
                let mut coarse = BdpMinresSolver::new(
                    &ops[0].m,
                    &ops[0].b,
                    param.coarse_solve_param.clone(),
                    param.coarse_schur_mode,
                );
                if n_levels > 0 {
                    coarse.set_ess_zero_dofs(&data.coarsest_ess_hdivdofs);
                }
                if n_levels == 0 {
                    // Single-level: the BDP *is* the solver.
                    coarse_bdp = Some(coarse);
                } else {
                    smoothers[0] = Some(Box::new(coarse));
                }
                break;
            }

            let schwarz = SaddleSchwarzSmoother::new(
                &ops[l].m,
                &ops[l].b,
                &data.agg_hdivdof[l - 1],
                &data.agg_l2dof[l - 1],
                &data.p_l2[l - 1],
                data.q_l2[l - 1].clone(),
            );
            if param.coupled_solve {
                let s1 = ZeroBlockDiag {
                    block0: AuxSpaceSmoother::new(&ops[l].m, &data.c[l]),
                    n_u: ops[l].n_u(),
                };
                smoothers[l] = Some(Box::new(ProductSmoother::new(
                    ops[l].clone(),
                    Box::new(schwarz),
                    Box::new(s1),
                )));
            } else {
                smoothers[l] = Some(Box::new(schwarz));
            }

            // Coarsen: M_c = P_hdivᵀ M P_hdiv, B_c = P_l2ᵀ B P_hdiv.
            let m_c = galerkin(&ops[l].m, &data.p_hdiv[l - 1]);
            let bp = ops[l].b.multiply(&data.p_hdiv[l - 1]);
            let b_c = data.p_l2[l - 1].transpose().multiply(&bp);
            blk_ps[l - 1] = flat_block_diagonal(&data.p_hdiv[l - 1], &data.p_l2[l - 1]);
            ops[l - 1] = SaddleOp::new(m_c, b_c);
        }

        let bbt_solver = BbtSolver::new(b, param.bbt_solve_param.clone());

        let flat_finest = crate::block::BlockSystem {
            a: m.clone(),
            bt: b.transpose(),
            b: b.clone(),
            c: None,
        }
        .to_flat_csr();

        // Coupled mode: GMRES on the finest saddle operator + V-cycle multigrid
        // (the level smoothers move into the cycle; `Mult` in coupled mode
        // never touches `self.smoothers` since `SolveParticular` is
        // decoupled-only).
        let (smoothers_final, mg) = if param.coupled_solve {
            let level_smoothers: Vec<Box<dyn BlockSmoother>> =
                smoothers.into_iter().flatten().collect();
            let mg = if n_levels > 0 {
                Some(MgVcycle::new(ops.clone(), level_smoothers, blk_ps.clone()))
            } else {
                None
            };
            (Vec::new(), mg)
        } else {
            (smoothers.into_iter().flatten().collect::<Vec<_>>(), None)
        };

        // Decoupled mode: CG on CᵀMC with the aux multigrid.
        let mut dec_ops: Vec<CsrMatrix<f64>> = Vec::new();
        let mut dec_mg = None;
        let mut c_finest = CooMatrix::<f64>::new(0, 0).into_csr();
        if !param.coupled_solve && n_levels > 0 {
            c_finest = data.c[finest].clone();
            let mut a_finest = galerkin(&m, &c_finest);
            eliminate_zero_rows(&mut a_finest);
            dec_ops.push(a_finest);
            for l in (0..n_levels).rev() {
                let prev = galerkin(dec_ops.last().unwrap(), &data.p_aux[l]);
                dec_ops.push(prev);
            }
            dec_ops.reverse();
            let dec_smoothers: Vec<L1JacobiSmoother> = dec_ops.iter().map(L1JacobiSmoother::new).collect();
            dec_mg = Some(AuxMg {
                ops: dec_ops.clone(),
                smoothers: dec_smoothers,
                prolongations: data.p_aux.clone(),
            });
        }

        Self {
            n_u,
            n,
            finest,
            coupled: param.coupled_solve,
            verbose: param.verbose,
            ops,
            smoothers: smoothers_final,
            blk_ps,
            bt: b.transpose(),
            bbt_solver,
            coarse_bdp,
            flat_finest,
            mg,
            c_finest,
            dec_ops,
            dec_mg,
            outer_param: param.outer_solve_param,
            gmres_restart: param.gmres_restart,
            last_iters: AtomicUsize::new(0),
            dec_cg_iters: AtomicUsize::new(0),
        }
    }

    /// MFEM `GetNumIterations`.
    pub fn num_iterations(&self) -> usize {
        self.last_iters.load(Ordering::Relaxed)
    }

    /// Size of the system (`n_u + n_p`, MFEM `DarcySolver` height).
    pub fn size(&self) -> usize {
        self.n
    }

    /// Number of multilevel levels (`P_l2.size() + 1`).
    pub fn num_levels(&self) -> usize {
        self.finest + 1
    }

    /// MFEM `DivFreeSolver::Mult`: one application of the DFS preconditioner.
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n, "DivFreeSolver: x size is invalid");
        assert_eq!(y.len(), self.n, "DivFreeSolver: y size is invalid");

        if self.ops.len() == 1 {
            // Equivalent to BDPMinresSolver (par_ref_levels == 0).
            let coarse = self.coarse_bdp.as_ref().expect("single-level coarse BDP");
            coarse.mult(x, y);
            self.last_iters.store(coarse.num_iterations(), Ordering::Relaxed);
            return;
        }

        let nu = self.n_u;
        // resid = x − A y  (y enters as the current iterate)
        let mut resid = vec![0.0; self.n];
        self.ops[self.finest].mult(y, &mut resid);
        resid = sub_vec(x, &resid);

        if self.coupled {
            let mut correction = vec![0.0; self.n];
            let res = crate::block_operator::right_preconditioned_gmres(
                &self.flat_finest,
                &resid,
                &mut correction,
                self.gmres_restart,
                &fem_linalg::SolverConfig {
                    rtol: self.outer_param.rel_tol,
                    atol: self.outer_param.abs_tol,
                    max_iter: self.outer_param.max_iter,
                    verbose: false,
                    print_level: fem_linalg::PrintLevel::Silent,
                },
                |v, w| self.mg.as_ref().expect("coupled multigrid").mult(v, w),
            )
            .expect("DivFreeSolver coupled GMRES failed");
            self.last_iters.store(res.iterations, Ordering::Relaxed);
            for (yi, ci) in y.iter_mut().zip(&correction) {
                *yi += ci;
            }
            if self.verbose {
                println!("Coupled correction found ({} iters).", res.iterations);
            }
        } else {
            // ── 1. Particular solution (div-constraint satisfying part) ───
            let mut correction = vec![0.0; self.n];
            self.solve_particular(&resid, &mut correction);
            for (yi, ci) in y.iter_mut().zip(&correction) {
                *yi += ci;
            }
            if self.verbose {
                let mut r1 = vec![0.0; self.n];
                self.ops[self.finest].mult(y, &mut r1);
                let rn = resid.iter().zip(&r1).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                eprintln!(
                    "[dfs] after particular: ‖resid‖ = {rn:.3e} (pre: {})",
                    resid.iter().map(|v| v * v).sum::<f64>().sqrt()
                );
            }

            // resid = x − A y
            self.ops[self.finest].mult(y, &mut resid);
            resid = sub_vec(x, &resid);

            // ── 2. Divergence-free part ───────────────────────────────────
            let mut corr_u = vec![0.0; nu];
            self.solve_div_free(&resid[..nu], &mut corr_u);
            for (yi, ci) in y[..nu].iter_mut().zip(&corr_u) {
                *yi += ci;
            }
            if self.verbose {
                let mut r2 = vec![0.0; self.n];
                self.ops[self.finest].mult(y, &mut r2);
                let rn = resid.iter().zip(&r2).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                eprintln!("[dfs] after div-free:  ‖resid‖ = {rn:.3e} (cg iters {})", self.dec_cg_iters.load(Ordering::Relaxed));
            }

            // ── 3. Scalar potential ───────────────────────────────────────
            // resid_u -= M·corr_u
            let mut m_corr = vec![0.0; nu];
            self.ops[self.finest].m.spmv(&corr_u, &mut m_corr);
            for (ri, mi) in resid[..nu].iter_mut().zip(&m_corr) {
                *ri -= mi;
            }
            let mut corr_p = vec![0.0; self.n - nu];
            self.solve_potential(&resid[..nu], &mut corr_p);
            for (yi, ci) in y[nu..].iter_mut().zip(&corr_p) {
                *yi += ci;
            }
            if self.verbose {
                let mut r3 = vec![0.0; self.n];
                self.ops[self.finest].mult(y, &mut r3);
                let rn = resid.iter().zip(&r3).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                eprintln!("[dfs] after potential: ‖resid‖ = {rn:.3e} (bbt iters {})", self.bbt_solver.num_iterations());
            }
            self.last_iters.store(self.dec_cg_iters.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    /// MFEM `SolveParticular`: multilevel cycle of the level smoothers.
    fn solve_particular(&self, rhs: &[f64], sol: &mut [f64]) {
        let m = self.ops.len();
        debug_assert_eq!(sol.len(), rhs.len());
        let mut rhss: Vec<Vec<f64>> = vec![Vec::new(); m];
        let mut sols: Vec<Vec<f64>> = vec![Vec::new(); m];
        rhss[m - 1] = rhs.to_vec();
        sols[m - 1] = vec![0.0; rhs.len()];

        for l in (0..m - 1).rev() {
            let nc = self.blk_ps[l].ncols;
            let mut r = vec![0.0; nc];
            self.blk_ps[l].transpose().spmv(&rhss[l + 1], &mut r);
            rhss[l] = r;
            sols[l] = vec![0.0; nc];
        }

        for l in 0..m {
            self.smoothers[l].mult(&rhss[l], &mut sols[l]);
        }

        for l in 0..m - 1 {
            let nr = self.blk_ps[l].nrows;
            let mut p_sol = vec![0.0; nr];
            self.blk_ps[l].spmv(&sols[l], &mut p_sol);
            for (si, pi) in sols[l + 1].iter_mut().zip(&p_sol) {
                *si += pi;
            }
        }

        sol.copy_from_slice(&sols[m - 1]);
    }

    /// MFEM `SolveDivFree`: `sol = C · (CᵀMC)⁻¹ · Cᵀ rhs` (CG + aux multigrid).
    fn solve_div_free(&self, rhs: &[f64], sol: &mut [f64]) {
        let c = &self.c_finest;
        let na = c.ncols;
        sol.fill(0.0);
        let mut rhs_df = vec![0.0; na];
        c.transpose().spmv(rhs, &mut rhs_df);
        let rhs_max = rhs_df.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        if rhs_max < 1e-14 {
            self.dec_cg_iters.store(0, Ordering::Relaxed);
            return;
        }
        let mut potential = vec![0.0; na];
        let mg = self.dec_mg.as_ref().expect("decoupled multigrid missing");
        let res = crate::solve_pcg_operator_precond(
            na,
            |v, w| self.dec_ops.last().unwrap().spmv(v, w),
            &rhs_df,
            &mut potential,
            |v, w| mg.mult(v, w),
            &fem_linalg::SolverConfig {
                rtol: self.outer_param.rel_tol,
                atol: self.outer_param.abs_tol,
                max_iter: self.outer_param.max_iter,
                verbose: false,
                print_level: fem_linalg::PrintLevel::Silent,
            },
        )
        .expect("decoupled CG on CᵀMC failed");
        self.dec_cg_iters.store(res.iterations, Ordering::Relaxed);
        c.spmv(&potential, sol);
    }

    /// MFEM `SolvePotential`: `sol = (B·Bᵀ)⁻¹ · B · rhs` (CG + AMG).
    fn solve_potential(&self, rhs: &[f64], sol: &mut [f64]) {
        // BT = Bᵀ; BTᵀ·rhs = B·rhs.
        let np = self.bt.ncols;
        let mut rhs_p = vec![0.0; np];
        self.bt.transpose().spmv(rhs, &mut rhs_p);
        self.bbt_solver.mult(&rhs_p, sol);
    }

    /// BBT (potential) solver iteration count of the last `mult`.
    pub fn bbt_iterations(&self) -> usize {
        self.bbt_solver.num_iterations()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────
//
// The tests use fem-space/fem-assembly (dev-dependencies) to assemble small
// RT0/L2/H1 systems on unit-square meshes and build a minimal DFSData the
// same way the block-solvers miniapp collector does.

#[cfg(test)]
mod tests {
    use super::*;
    use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
    use fem_assembly::standard::{MassIntegrator, VectorMassIntegrator};
    use fem_assembly::{Assembler, VectorAssembler};
    use fem_element::lagrange::TriP1;
    use fem_element::raviart_thomas::TriRTk;
    use fem_element::reference::{ReferenceElement, VectorReferenceElement};
    use fem_linalg::SolverConfig;
    use fem_mesh::{Mesh, MeshTopology};
    use fem_space::fe_space::FESpace;

    /// Discrete curl `C: H1(order+1) → RT(order)` on a 2-D **triangular**
    /// mesh, via element-local L2 pairing (exact: `curl P_{k+1} ⊂ RT_k`).
    /// This mirrors the miniapp collector's `CurlInterpolator` port.
    ///
    /// Shared face dofs receive the identical exact coefficient from both
    /// adjacent elements; each (column, row) is scattered exactly once.
    fn discrete_curl_h1_rt0_tri(
        h1: &fem_space::H1Space<Mesh<2>>,
        rt: &fem_space::HDivSpace<Mesh<2>>,
    ) -> CsrMatrix<f64> {
        let mesh = rt.mesh();
        let qo = 4_u8;
        let n_rt = rt.n_dofs();
        let n_h1 = h1.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_rt, n_h1);
        // Pass 1: per-element local projections (columns over the element's
        // H1 dofs).  Pass 2: scatter with per-column write-once semantics.
        let mut local: Vec<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>)> = Vec::new(); // (rt dofs, h1 dofs, cols)
        for e in mesh.elem_iter() {
            let rt_ref = TriRTk::new(rt.order() as usize);
            let n_loc = rt_ref.n_dofs();
            let h1_ref = TriP1;
            let nh_loc = h1_ref.n_dofs();
            let qr = rt_ref.quadrature(qo);
            let mut phi = vec![0.0_f64; n_loc * 2];
            let mut dsh = vec![0.0_f64; nh_loc * 2];
            let mut m_e = vec![0.0_f64; n_loc * n_loc];
            let mut g_e = vec![0.0_f64; n_loc * nh_loc];
            let dofs: Vec<usize> = rt.element_dofs(e).iter().map(|&d| d as usize).collect();
            let signs = rt.element_signs(e).to_vec();
            let hdofs: Vec<usize> = h1.element_dofs_u32(e).iter().map(|&d| d as usize).collect();
            for (qi, xi) in qr.points.iter().enumerate() {
                let (jac, _xp) = fem_mesh::element_jacobian_at(mesh, e, xi, 2);
                let det = jac.determinant();
                let w = qr.weights[qi] * det.abs();
                rt_ref.eval_basis_vec(xi, &mut phi);
                h1_ref.eval_grad_basis(xi, &mut dsh);
                for i in 0..n_loc {
                    let s = signs[i];
                    // Piola: φ = J φ̂ / det
                    let px = s * (jac[(0, 0)] * phi[2 * i] + jac[(0, 1)] * phi[2 * i + 1]) / det;
                    let py = s * (jac[(1, 0)] * phi[2 * i] + jac[(1, 1)] * phi[2 * i + 1]) / det;
                    for k in 0..n_loc {
                        let sk = signs[k];
                        let kx = sk * (jac[(0, 0)] * phi[2 * k] + jac[(0, 1)] * phi[2 * k + 1]) / det;
                        let ky = sk * (jac[(1, 0)] * phi[2 * k] + jac[(1, 1)] * phi[2 * k + 1]) / det;
                        m_e[i * n_loc + k] += w * (px * kx + py * ky);
                    }
                    for j in 0..nh_loc {
                        // grad ψ = J⁻ᵀ ∇̂ψ ; curl ψ = (∂y ψ, −∂x ψ)
                        // J⁻ᵀ = 1/det·[[d,−c],[−b,a]], J = [[a,b],[c,d]]
                        let (a, b, c, d) =
                            (jac[(0, 0)], jac[(0, 1)], jac[(1, 0)], jac[(1, 1)]);
                        let gx = (d * dsh[2 * j] - c * dsh[2 * j + 1]) / det;
                        let gy = (a * dsh[2 * j + 1] - b * dsh[2 * j]) / det;
                        g_e[i * nh_loc + j] += w * (px * gy - py * gx);
                    }
                }
            }
        // C_e = M_e⁻¹ G_e (exact projection: curl P1 ⊂ RT0)
        // (pass 1 collects per-element local columns; grouping by GLOBAL
        // H1 column happens below so that shared RT dofs are written once.)
        let mut lu = m_e.clone();
        let mut piv = vec![0_usize; n_loc];
        fem_linalg::dense::lu_factor(&mut lu, n_loc, &mut piv).expect("RT mass singular");
        let cols: Vec<Vec<f64>> = (0..nh_loc)
            .map(|j| {
                let mut col: Vec<f64> = (0..n_loc).map(|i| g_e[i * nh_loc + j]).collect();
                fem_linalg::dense::lu_solve(&lu, n_loc, &piv, &mut col);
                col
            })
            .collect();
        local.push((dofs, hdofs, cols));
        }
        // Pass 2: group by global H1 column, scatter each RT dof once.
        let mut per_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_h1];
        for (dofs, hdofs, cols) in &local {
            for (li, &hcol) in hdofs.iter().enumerate() {
                for (i, &dof) in dofs.iter().enumerate() {
                    if cols[li][i].abs() > 0.0 {
                        per_col[hcol].push((dof, cols[li][i]));
                    }
                }
            }
        }
        for (hcol, entries) in per_col.into_iter().enumerate() {
            let mut written = std::collections::HashSet::<usize>::new();
            for (dof, v) in entries {
                if written.insert(dof) {
                    coo.add(dof, hcol, v);
                }
            }
        }
        coo.into_csr()
    }

    /// `B·C ≈ 0` (columns of C are divergence-free) and `C` removes the
    /// divergence part of a given field.
    #[test]
    fn discrete_curl_columns_are_divergence_free() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let u_sp = fem_space::HDivSpace::new(mesh.clone(), 0);
        let p_sp = fem_space::L2Space::new(mesh.clone(), 0);
        let h1 = fem_space::H1Space::new(mesh.clone(), 1);
        let qo = 3_u8;

        let mut b = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
        for v in &mut b.values {
            *v *= -1.0;
        }
        let m = VectorAssembler::assemble_bilinear(&u_sp, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let c = discrete_curl_h1_rt0_tri(&h1, &u_sp);

        let n_u = u_sp.n_dofs();
        let n_h1 = h1.n_dofs();

        // 1. B·C ≈ 0
        let mut bc = CooMatrix::<f64>::new(p_sp.n_dofs(), n_h1);
        for i in 0..p_sp.n_dofs() {
            for p in b.row_ptr[i]..b.row_ptr[i + 1] {
                let k = b.col_idx[p] as usize;
                for q in c.row_ptr[k]..c.row_ptr[k + 1] {
                    bc.add(i, c.col_idx[q] as usize, b.values[p] * c.values[q]);
                }
            }
        }
        let bc = bc.into_csr();
        let max_bc = bc.values.iter().fold(0.0_f64, |mx, &v| mx.max(v.abs()));
        assert!(max_bc < 1e-11, "‖B·C‖_max = {max_bc:.3e}");

        // 2. projection recovers a given divergence-free part: for
        // u = C·w (random w), the M-orthogonal projection z of u onto
        // range(C) (solving CᵀMC z = CᵀMu) must satisfy C·z = u.
        let mut rng_state = 12345_u64;
        let mut rand = move || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 33) as f64 / (1_u64 << 31) as f64 - 1.0
        };
        let w: Vec<f64> = (0..n_h1).map(|_| rand()).collect();
        let mut u = vec![0.0; n_u];
        c.spmv(&w, &mut u);
        // sanity: B·u ≈ 0 (columns of C are divergence-free)
        let mut bu = vec![0.0; p_sp.n_dofs()];
        b.spmv(&u, &mut bu);
        let scale = b.values.iter().fold(0.0_f64, |mx, &v| mx.max(v.abs()));
        let bu_max = bu.iter().fold(0.0_f64, |mx, &v| mx.max(v.abs()));
        assert!(bu_max < 1e-10 * scale, "div-free field has divergence {bu_max:.3e}");

        let mut mu = vec![0.0; n_u];
        m.spmv(&u, &mut mu);
        let mut cmu = vec![0.0; n_h1];
        c.transpose().spmv(&mu, &mut cmu);
        let a_cm = galerkin(&m, &c);
        let jac = L1JacobiSmoother::new(&a_cm);
        let mut z = vec![0.0; n_h1];
        crate::solve_pcg_operator_precond(
            n_h1,
            |v, w2| a_cm.spmv(v, w2),
            &cmu,
            &mut z,
            |v, w2| jac.mult(v, w2),
            &SolverConfig { rtol: 1e-12, atol: 1e-14, max_iter: 500, ..SolverConfig::default() },
        )
        .expect("CG on CᵀMC failed");
        let mut cz = vec![0.0; n_u];
        c.spmv(&z, &mut cz);
        let err = cz.iter().zip(&u).map(|(a, b2)| (a - b2).powi(2)).sum::<f64>().sqrt();
        let unorm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            err < 1e-9 * unorm.max(1e-30),
            "projection did not recover the div-free part: ‖Cz−u‖ = {err:.3e} (‖u‖ = {unorm:.3e})"
        );
    }

    /// Two-level DFS on a unit-square quad mesh (level 0: 2×2 → level 1: 4×4)
    /// used as a preconditioner inside PCG/GMRES must converge in few
    /// iterations, in both decoupled and coupled modes.
    #[test]
    fn two_level_div_free_solver_saddle_convergence() {
        for coupled in [false, true] {
            let (m, b, data) = build_two_level_quad_case(coupled);
            let solver = DivFreeSolver::new(&m, &b, &data);
            assert_eq!(solver.num_levels(), 2);
            let n = solver.size();

            // Use one DFS application as preconditioner in right-preconditioned
            // GMRES on the saddle system with a random-compatible rhs.
            let mut rng_state = 987654321_u64;
            let mut rand = move || {
                rng_state =
                    rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng_state >> 33) as f64 / (1_u64 << 31) as f64 - 1.0
            };
            let x_true: Vec<f64> = (0..n).map(|_| rand()).collect();
            let mut rhs = vec![0.0; n];
            let flat = crate::block::BlockSystem {
                a: m.clone(),
                bt: b.transpose(),
                b: b.clone(),
                c: None,
            }
            .to_flat_csr();
            flat.spmv(&x_true, &mut rhs);

            let mut x = vec![0.0; n];
            let res = crate::block_operator::right_preconditioned_gmres(
                &flat,
                &rhs,
                &mut x,
                30,
                &SolverConfig { rtol: 1e-8, atol: 1e-12, max_iter: 200, ..SolverConfig::default() },
                |v, w| solver.mult(v, w),
            )
            .expect("DFS-preconditioned GMRES failed");
            assert!(res.converged, "coupled={coupled}: GMRES+DFS did not converge ({})", res.iterations);
            assert!(
                res.iterations <= 30,
                "coupled={coupled}: too many iterations: {}",
                res.iterations
            );
            let err = x.iter().zip(&x_true).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            assert!(err < 1e-6, "coupled={coupled}: solution error {err:.3e}");
        }
    }

    /// Assemble M, B on the 4×4-quad (fine) mesh and build a 2-level DFSData
    /// (level 0 = 2×2 quads, level 1 = fine).  All collector steps (P_hdiv,
    /// P_l2, P_aux, agg tables, C, Q_l2) follow the miniapp collector.
    fn build_two_level_quad_case(coupled: bool) -> (CsrMatrix<f64>, CsrMatrix<f64>, DfsData) {
        const ORDER: u8 = 0;
        let mesh0 = Mesh::<2>::unit_square_quad(2);
        let mesh1 = fem_mesh::refine_uniform(&mesh0);

        let u0 = fem_space::HDivSpace::new(mesh0.clone(), ORDER);
        let u1 = fem_space::HDivSpace::new(mesh1.clone(), ORDER);
        let p0 = fem_space::L2Space::new(mesh0.clone(), ORDER);
        let p1 = fem_space::L2Space::new(mesh1.clone(), ORDER);
        let h1_0 = fem_space::H1Space::new(mesh0.clone(), ORDER + 1);
        let h1_1 = fem_space::H1Space::new(mesh1.clone(), ORDER + 1);
        let qo = 3_u8;

        // Fine-level system.
        let m = VectorAssembler::assemble_bilinear(&u1, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let mut b = assemble_hdiv_l2_mixed(&p1, &u1, &[&HDivL2DivIntegrator], qo);
        for v in &mut b.values {
            *v *= -1.0;
        }

        // ── parents (fine element → coarse element) ───────────────────────
        let ne0 = mesh0.n_elems() as usize;
        let ne1 = mesh1.n_elems() as usize;
        let mut parents = vec![0_usize; ne1];
        for f in 0..ne1 {
            let fe = f as u32;
            let nn = mesh1.element_nodes(fe);
            let cx: f64 = nn.iter().map(|&n| mesh1.node_coords(n)[0]).sum::<f64>() / nn.len() as f64;
            let cy: f64 = nn.iter().map(|&n| mesh1.node_coords(n)[1]).sum::<f64>() / nn.len() as f64;
            parents[f] = (0..ne0)
                .find(|&c| {
                    let ns = mesh0.element_nodes(c as u32);
                    let xs: Vec<f64> = ns.iter().map(|&n| mesh0.node_coords(n)[0]).collect();
                    let ys: Vec<f64> = ns.iter().map(|&n| mesh0.node_coords(n)[1]).collect();
                    cx >= xs.iter().cloned().fold(f64::INFINITY, f64::min) - 1e-12
                        && cx <= xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e-12
                        && cy >= ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1e-12
                        && cy <= ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e-12
                })
                .unwrap_or_else(|| panic!("no parent for fine elem {f}"));
        }

        // ── P_hdiv (patch projection, exact for RT0) ──────────────────────
        let p_hdiv = build_prolongation_hdiv_test(&u0, &u1, &parents);

        // ── P_l2 (L2 dofs are element-local: exact restriction) ───────────
        let n_p0 = p0.n_dofs();
        let n_p1 = p1.n_dofs();
        let mut coo_pl2 = CooMatrix::<f64>::new(n_p1, n_p0);
        for f in 0..ne1 {
            let fd = p1.element_dofs(f as u32);
            let cd = p0.element_dofs(parents[f] as u32);
            // L2_0 per element: local dof k ↔ parent local dof k (Q1→4×Q1)
            for (k, &d) in fd.iter().enumerate() {
                coo_pl2.add(d as usize, cd[k] as usize, 1.0);
            }
        }
        let p_l2 = coo_pl2.into_csr();

        // ── P_aux (H1 P1/Q1: fine nodal values = coarse interpolation) ─────
        let n_h0 = h1_0.n_dofs();
        let n_h1d = h1_1.n_dofs();
        let mut coo_pa = CooMatrix::<f64>::new(n_h1d, n_h0);
        let h1_ref_c: Box<dyn ReferenceElement> = match mesh0.element_type(0) {
            fem_mesh::ElementType::Tri3 => Box::new(fem_element::lagrange::TriP1),
            fem_mesh::ElementType::Quad4 => Box::new(fem_element::lagrange::QuadQ1),
            other => panic!("unsupported element type {other:?}"),
        };
        let mut c_sh = vec![0.0_f64; h1_ref_c.n_dofs()];
        for f in 0..ne1 {
            let c = parents[f] as u32;
            let c_dofs: Vec<usize> = h1_0.element_dofs_u32(c).iter().map(|&d| d as usize).collect();
            let ns = mesh1.element_nodes(f as u32);
            let fd = h1_1.element_dofs_u32(f as u32);
            // For P1/Q1 each local dof sits at local-corner position k.
            for (k, &nd) in ns.iter().enumerate() {
                let xp = mesh1.node_coords(nd);
                let pref = xp_ref(&mesh0, c, &xp);
                h1_ref_c.eval_basis(&pref, &mut c_sh);
                for (l, &cd) in c_dofs.iter().enumerate() {
                    if c_sh[l].abs() > 1e-15 {
                        coo_pa.add(fd[k] as usize, cd, c_sh[l]);
                    }
                }
            }
        }
        let p_aux = coo_pa.into_csr();

        // ── agg tables ─────────────────────────────────────────────────────
        let mut coo_al2 = CooMatrix::<f64>::new(ne0, n_p1);
        for f in 0..ne1 {
            for &d in p1.element_dofs(f as u32) {
                coo_al2.add(parents[f], d as usize, 1.0);
            }
        }
        let agg_l2dof = coo_al2.into_csr();

        // interior H(div) dofs: used by fine elements of a single aggregate
        let n_u1 = u1.n_dofs();
        let mut dof_count = vec![0_usize; n_u1];
        let mut dof_agg = vec![usize::MAX; n_u1];
        for f in 0..ne1 {
            for &d in u1.element_dofs(f as u32) {
                let d = d as usize;
                if dof_agg[d] == usize::MAX {
                    dof_agg[d] = parents[f];
                } else if dof_agg[d] != parents[f] {
                    dof_agg[d] = usize::MAX - 1; // shared across aggregates
                }
                dof_count[d] += 1;
            }
        }
        let mut coo_ah = CooMatrix::<f64>::new(ne0, n_u1);
        for d in 0..n_u1 {
            if dof_count[d] > 0 && dof_agg[d] < ne0 {
                coo_ah.add(dof_agg[d], d, 1.0);
            }
        }
        let agg_hdivdof = coo_ah.into_csr();

        // ── Q_l2 (MFEM DataFinalize with rp = 1) ──────────────────────────
        let w1 = Assembler::assemble_bilinear(&p1, &[&MassIntegrator { rho: 1.0 }], qo);
        let ptw = p_l2.transpose().multiply(&w1);
        let cw = ptw.multiply(&p_l2);
        // el_l2dof[0]: level-0 L2 element→dof boolean table
        let mut coo_el0 = CooMatrix::<f64>::new(ne0, n_p0);
        for c in 0..ne0 {
            for &d in p0.element_dofs(c as u32) {
                coo_el0.add(c, d as usize, 1.0);
            }
        }
        let el_l2dof0 = coo_el0.into_csr();
        let q_l2 = vec![Arc::new(Ql2Projector::new(&cw, &el_l2dof0, p_l2.clone(), w1))];

        // ── C at the fine level (H1 P1 → RT0 on quads) ─────────────────────
        let c1 = discrete_curl_h1_rt0_quad(&h1_1, &u1);
        let mut c = vec![CooMatrix::<f64>::new(0, 0).into_csr(), c1];

        let param = DfsParameters {
            coupled_solve: coupled,
            verbose: false,
            coarse_solve_param: IterSolveParameters { rel_tol: 1e-10, ..IterSolveParameters::default() },
            bbt_solve_param: IterSolveParameters { rel_tol: 1e-10, ..IterSolveParameters::default() },
            outer_solve_param: IterSolveParameters { rel_tol: 1e-10, ..IterSolveParameters::default() },
            gmres_restart: 30,
            coarse_schur_mode: SchurMode::Dense,
        };
        c[1] = zero_ess_cols(&c[1], &[]);

        (
            m,
            b,
            DfsData {
                agg_hdivdof: vec![agg_hdivdof],
                agg_l2dof: vec![agg_l2dof],
                p_hdiv: vec![p_hdiv],
                p_l2: vec![p_l2],
                p_aux: vec![p_aux],
                coarsest_ess_hdivdofs: Vec::new(),
                c,
                q_l2,
                param,
            },
        )
    }

    /// Discrete curl H1(Q1) → RT0 on a quad mesh via element-local pairing.
    fn discrete_curl_h1_rt0_quad(
        h1: &fem_space::H1Space<Mesh<2>>,
        rt: &fem_space::HDivSpace<Mesh<2>>,
    ) -> CsrMatrix<f64> {
        use fem_element::lagrange::QuadQ1;
        use fem_element::raviart_thomas::QuadRTk;
        let mesh = rt.mesh();
        let qo = 4_u8;
        let n_rt = rt.n_dofs();
        let n_h1 = h1.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_rt, n_h1);
        let mut local: Vec<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>)> = Vec::new();
        for e in mesh.elem_iter() {
            let rt_ref = QuadRTk::new(rt.order() as usize);
            let n_loc = rt_ref.n_dofs();
            let h1_ref = QuadQ1;
            let nh_loc = h1_ref.n_dofs();
            let qr = rt_ref.quadrature(qo);
            let mut phi = vec![0.0_f64; n_loc * 2];
            let mut dsh = vec![0.0_f64; nh_loc * 2];
            let mut m_e = vec![0.0_f64; n_loc * n_loc];
            let mut g_e = vec![0.0_f64; n_loc * nh_loc];
            let dofs: Vec<usize> = rt.element_dofs(e).iter().map(|&d| d as usize).collect();
            let signs = rt.element_signs(e).to_vec();
            let hdofs: Vec<usize> = h1.element_dofs_u32(e).iter().map(|&d| d as usize).collect();
            for (qi, xi) in qr.points.iter().enumerate() {
                let (jac, _xp) = fem_mesh::element_jacobian_at(mesh, e, xi, 2);
                let det = jac.determinant();
                let w = qr.weights[qi] * det.abs();
                rt_ref.eval_basis_vec(xi, &mut phi);
                h1_ref.eval_grad_basis(xi, &mut dsh);
                for i in 0..n_loc {
                    let s = signs[i];
                    let px = s * (jac[(0, 0)] * phi[2 * i] + jac[(0, 1)] * phi[2 * i + 1]) / det;
                    let py = s * (jac[(1, 0)] * phi[2 * i] + jac[(1, 1)] * phi[2 * i + 1]) / det;
                    for k in 0..n_loc {
                        let sk = signs[k];
                        let kx = sk * (jac[(0, 0)] * phi[2 * k] + jac[(0, 1)] * phi[2 * k + 1]) / det;
                        let ky = sk * (jac[(1, 0)] * phi[2 * k] + jac[(1, 1)] * phi[2 * k + 1]) / det;
                        m_e[i * n_loc + k] += w * (px * kx + py * ky);
                    }
                    for j in 0..nh_loc {
                        // grad ψ = J⁻ᵀ ∇̂ψ ; curl ψ = (∂y ψ, −∂x ψ)
                        let (a, b, c, d) =
                            (jac[(0, 0)], jac[(0, 1)], jac[(1, 0)], jac[(1, 1)]);
                        let gx = (d * dsh[2 * j] - c * dsh[2 * j + 1]) / det;
                        let gy = (a * dsh[2 * j + 1] - b * dsh[2 * j]) / det;
                        g_e[i * nh_loc + j] += w * (px * gy - py * gx);
                    }
                }
            }
            let mut lu = m_e.clone();
            let mut piv = vec![0_usize; n_loc];
            fem_linalg::dense::lu_factor(&mut lu, n_loc, &mut piv).expect("RT mass singular");
            let cols: Vec<Vec<f64>> = (0..nh_loc)
                .map(|j| {
                    let mut col: Vec<f64> = (0..n_loc).map(|i| g_e[i * nh_loc + j]).collect();
                    fem_linalg::dense::lu_solve(&lu, n_loc, &piv, &mut col);
                    col
                })
                .collect();
            local.push((dofs, hdofs, cols));
        }
        // Scatter grouped by global H1 column (each RT dof written once).
        let mut per_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_h1];
        for (dofs, hdofs, cols) in &local {
            for (li, &hcol) in hdofs.iter().enumerate() {
                for (i, &dof) in dofs.iter().enumerate() {
                    if cols[li][i].abs() > 0.0 {
                        per_col[hcol].push((dof, cols[li][i]));
                    }
                }
            }
        }
        for (hcol, entries) in per_col.into_iter().enumerate() {
            let mut written = std::collections::HashSet::<usize>::new();
            for (dof, v) in entries {
                if written.insert(dof) {
                    coo.add(dof, hcol, v);
                }
            }
        }
        coo.into_csr()
    }

    /// Patch-projection H(div) prolongation (test version, tri/quad RT0).
    ///
    /// For each coarse element (aggregate) the fine children's patch system
    /// `M·P = G` is solved with `M[i][k] = ∫φ_i·φ_k` and `G[i][j] = ∫φ_i·ψ_j`;
    /// the projection is exact for RT0 (parent restriction ∈ child space), so
    /// shared dofs get consistent values from all patches.
    fn build_prolongation_hdiv_test(
        coarse: &fem_space::HDivSpace<Mesh<2>>,
        fine: &fem_space::HDivSpace<Mesh<2>>,
        parents: &[usize],
    ) -> CsrMatrix<f64> {
        let cmesh = coarse.mesh();
        let fmesh = fine.mesh();
        let ne0 = cmesh.n_elems() as usize;
        let ne1 = fmesh.n_elems() as usize;
        let order = coarse.order() as usize;
        let mut coo = CooMatrix::<f64>::new(fine.n_dofs(), coarse.n_dofs());
        // Per-patch local prolongations; scattered grouped by GLOBAL coarse
        // dof column with per-column write-once (shared fine dofs agree
        // across patches because the projection is exact).
        let mut patch_results: Vec<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>)> = Vec::new(); // (patch dofs, coarse dofs, X columns)

        for c in 0..ne0 {
            let ce = c as u32;
            let children: Vec<u32> = (0..ne1)
                .filter(|&f| parents[f] == c)
                .map(|f| f as u32)
                .collect();
            let c_dofs: Vec<usize> = coarse.element_dofs(ce).iter().map(|&d| d as usize).collect();
            let c_signs = coarse.element_signs(ce).to_vec();
            let nc = c_dofs.len();

            // Patch dofs (deduplicated; global fem-rs sign convention used).
            let mut patch_ids: Vec<usize> = Vec::new();
            let mut patch_pos: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            let mut occ: Vec<(u32, usize, usize)> = Vec::new(); // (fine elem, local idx, patch idx)
            for &fe in &children {
                for (i, &d) in fine.element_dofs(fe).iter().enumerate() {
                    let d = d as usize;
                    let next = patch_ids.len();
                    let pi = *patch_pos.entry(d).or_insert(next);
                    if pi == next {
                        patch_ids.push(d);
                    }
                    occ.push((fe, i, pi));
                }
            }
            let np_ = patch_ids.len();
            let mut m_p = vec![0.0_f64; np_ * np_];
            let mut g_p = vec![0.0_f64; np_ * nc];

            let c_ref = tri_or_quad_rt(cmesh, ce, order);
            let n_c_loc = c_ref.n_dofs();
            let mut c_phi = vec![0.0_f64; n_c_loc * 2];

            for &fe in &children {
                let f_ref = tri_or_quad_rt(fmesh, fe, order);
                let nf = f_ref.n_dofs();
                let qr = f_ref.quadrature(4);
                let mut f_phi = vec![0.0_f64; nf * 2];
                let f_signs = fine.element_signs(fe).to_vec();
                for (qi, xi) in qr.points.iter().enumerate() {
                    let (fjac, xp) = fem_mesh::element_jacobian_at(fmesh, fe, xi, 2);
                    let fdet = fjac.determinant();
                    let w = qr.weights[qi] * fdet.abs();
                    f_ref.eval_basis_vec(xi, &mut f_phi);
                    let pref = xp_ref(cmesh, ce, &xp);
                    c_ref.eval_basis_vec(&pref, &mut c_phi);
                    let (cjac, _) = fem_mesh::element_jacobian_at(cmesh, ce, &pref, 2);
                    let cdet = cjac.determinant();

                    // fine basis values at this point (patch-dof indexed)
                    let mut f_val: Vec<(usize, f64, f64)> = Vec::new();
                    for &(_, i, pi) in occ.iter().filter(|(e, _, _)| *e == fe) {
                        let s = f_signs[i];
                        let px = s * (fjac[(0, 0)] * f_phi[2 * i] + fjac[(0, 1)] * f_phi[2 * i + 1]) / fdet;
                        let py = s * (fjac[(1, 0)] * f_phi[2 * i] + fjac[(1, 1)] * f_phi[2 * i + 1]) / fdet;
                        f_val.push((pi, px, py));
                    }
                    // coarse basis values (coarse sign baked in)
                    let mut c_val = vec![(0.0_f64, 0.0_f64); nc];
                    for (j, &s) in c_signs.iter().enumerate() {
                        let px = s * (cjac[(0, 0)] * c_phi[2 * j] + cjac[(0, 1)] * c_phi[2 * j + 1]) / cdet;
                        let py = s * (cjac[(1, 0)] * c_phi[2 * j] + cjac[(1, 1)] * c_phi[2 * j + 1]) / cdet;
                        c_val[j] = (px, py);
                    }

                    for &(pi, px, py) in &f_val {
                        for &(pk, qx, qy) in &f_val {
                            m_p[pi * np_ + pk] += w * (px * qx + py * qy);
                        }
                        for (j, &(cx, cy)) in c_val.iter().enumerate() {
                            g_p[pi * nc + j] += w * (px * cx + py * cy);
                        }
                    }
                }
            }

            // P_patch = M⁻¹ G
            let mut lu = m_p.clone();
            let mut piv = vec![0_usize; np_];
            fem_linalg::dense::lu_factor(&mut lu, np_, &mut piv).expect("patch mass singular");
            let x_cols: Vec<Vec<f64>> = (0..nc)
                .map(|j| {
                    let mut col: Vec<f64> = (0..np_).map(|i| g_p[i * nc + j]).collect();
                    fem_linalg::dense::lu_solve(&lu, np_, &piv, &mut col);
                    col
                })
                .collect();
            patch_results.push((patch_ids, c_dofs, x_cols));
        }
        // Scatter grouped by global coarse dof column.
        let mut per_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); coarse.n_dofs()];
        for (patch_ids, c_dofs, x_cols) in &patch_results {
            for (j, &cd) in c_dofs.iter().enumerate() {
                for (i, &dof) in patch_ids.iter().enumerate() {
                    if x_cols[j][i].abs() > 0.0 {
                        per_col[cd].push((dof, x_cols[j][i]));
                    }
                }
            }
        }
        for (cd, entries) in per_col.into_iter().enumerate() {
            let mut written = std::collections::HashSet::<usize>::new();
            for (dof, v) in entries {
                if written.insert(dof) {
                    coo.add(dof, cd, v);
                }
            }
        }
        coo.into_csr()
    }

    fn tri_or_quad_rt(
        mesh: &Mesh<2>,
        e: u32,
        order: usize,
    ) -> Box<dyn VectorReferenceElement> {
        use fem_element::raviart_thomas::{QuadRTk, TriRTk};
        match mesh.element_type(e) {
            fem_mesh::ElementType::Tri3 => Box::new(TriRTk::new(order)),
            fem_mesh::ElementType::Quad4 => Box::new(QuadRTk::new(order)),
            other => panic!("unsupported element type {other:?}"),
        }
    }

    /// Physical → parent reference coordinates (affine tri / bilinear quad
    /// via Newton).
    fn xp_ref(mesh: &Mesh<2>, e: u32, x: &[f64]) -> [f64; 2] {
        let ns = mesh.element_nodes(e);
        let pts: Vec<[f64; 2]> = ns.iter().map(|&n| mesh.node_coords(n).try_into().unwrap()).collect();
        match mesh.element_type(e) {
            fem_mesh::ElementType::Tri3 => {
                let (x0, y0) = (pts[0][0], pts[0][1]);
                let a11 = pts[1][0] - x0;
                let a12 = pts[2][0] - x0;
                let a21 = pts[1][1] - y0;
                let a22 = pts[2][1] - y0;
                let det = a11 * a22 - a12 * a21;
                let rx = x[0] - x0;
                let ry = x[1] - y0;
                let xi = (a22 * rx - a12 * ry) / det;
                let eta = (-a21 * rx + a11 * ry) / det;
                [xi, eta]
            }
            fem_mesh::ElementType::Quad4 => {
                let mut xi = [0.5_f64, 0.5];
                for _ in 0..50 {
                    let (n0, n1, n2, n3) = (1.0 - xi[0], 1.0 - xi[1], xi[0], xi[1]);
                    let f0 = [n0 * n1, n1 * n2, n2 * n3, n3 * n0];
                    let fx = [
                        f0[0] * pts[0][0] + f0[1] * pts[1][0] + f0[2] * pts[2][0] + f0[3] * pts[3][0] - x[0],
                        f0[0] * pts[0][1] + f0[1] * pts[1][1] + f0[2] * pts[2][1] + f0[3] * pts[3][1] - x[1],
                    ];
                    if fx[0].abs() < 1e-14 && fx[1].abs() < 1e-14 {
                        break;
                    }
                    // dN/dxi = [-(1-η), (1-η), η, -η];
                    // dN/deta = [-(1-ξ), -ξ, ξ, (1-ξ)]
                    let d0 = [-n1, n1, n3, -n3];
                    let d1 = [-n0, -n2, n2, n0];
                    let j11: f64 = (0..4).map(|k| d0[k] * pts[k][0]).sum();
                    let j12: f64 = (0..4).map(|k| d0[k] * pts[k][1]).sum();
                    let j21: f64 = (0..4).map(|k| d1[k] * pts[k][0]).sum();
                    let j22: f64 = (0..4).map(|k| d1[k] * pts[k][1]).sum();
                    let det = j11 * j22 - j12 * j21;
                    xi[0] -= (j22 * fx[0] - j12 * fx[1]) / det;
                    xi[1] -= (-j21 * fx[0] + j11 * fx[1]) / det;
                }
                xi
            }
            other => panic!("unsupported element type {other:?}"),
        }
    }

    fn zero_ess_cols(c: &CsrMatrix<f64>, ess: &[usize]) -> CsrMatrix<f64> {
        if ess.is_empty() {
            return c.clone();
        }
        let mut coo = CooMatrix::<f64>::new(c.nrows, c.ncols);
        let ess: std::collections::HashSet<usize> = ess.iter().copied().collect();
        for i in 0..c.nrows {
            for p in c.row_ptr[i]..c.row_ptr[i + 1] {
                let j = c.col_idx[p] as usize;
                if !ess.contains(&j) {
                    coo.add(i, j, c.values[p]);
                }
            }
        }
        coo.into_csr()
    }
}
