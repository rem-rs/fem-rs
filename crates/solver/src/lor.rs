//! LOR (Low-Order Refined) preconditioner and geometric multigrid.
//!
//! ## LOR-AMG (two-grid Galerkin projection)
//!
//! `LorAmgPrecond` implements `M⁻¹ = P · A_LO⁻¹ · Pᵀ` where
//! `A_LO = Pᵀ · A_HO · P` is the Galerkin projection and AMG is applied to
//! `A_LO`.  P is the prolongation from the low-order (coarse) space to the
//! high-order (fine) space.
//!
//! The **true LOR** approach (as in MFEM) subdivides each high-order element
//! into P1 elements on the *same* mesh and reassembles.  That assembly step
//! lives in `fem-assembly`; this module provides the algebraic preconditioner
//! that consumes the prolongation P once it is built.

use crate::{solve_gmres, solve_pcg_jacobi, SolveResult, SolverConfig, SolverError};
use fem_linalg::{csr_spmm, fem_to_linlvo_csr, CsrMatrix};
pub use linlvo::amg::AmgConfig;
use linlvo::{
    amg::{AmgHierarchy, AmgPrecond},
    core::preconditioner::Preconditioner,
    DenseVec, Scalar as linlvoScalar,
};

/// LOR preconditioner configuration.
#[derive(Debug, Clone)]
pub struct LorPrecond {
    pub smoother_sweeps: usize,
}

impl Default for LorPrecond {
    fn default() -> Self {
        LorPrecond { smoother_sweeps: 2 }
    }
}

impl LorPrecond {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Solve SPD system with LOR-Jacobi preconditioned CG (legacy API stub).
pub fn solve_pcg_lor<T: linlvoScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    _lor: &LorPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_pcg_jacobi(a, b, x, cfg)
}

pub fn solve_gmres_lor<T: linlvoScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    _lor: &LorPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_gmres(a, b, x, restart, cfg)
}

// ─── LOR-AMG ──────────────────────────────────────────────────────────────────

/// Low-Order Refined AMG preconditioner.
///
/// `M⁻¹ = P · A_LO⁻¹ · Pᵀ` where `A_LO = Pᵀ · A_HO · P` and AMG is
/// applied to `A_LO`.  `P` is the prolongation from P1 → high-order.
pub struct LorAmgPrecond {
    prolong: CsrMatrix<f64>, // P:  n_lo → n_hi  (n_hi × n_lo)
    amg: AmgPrecond<f64>,    // AMG on A_LO
    n_lo: usize,
}

/// Build the LOR operator `A_LO = Pᵀ · A_HO · P` as a `CsrMatrix<f64>`.
///
/// # Panics
/// If the matrix dimensions are incompatible.
pub fn build_lor_operator(a_ho: &CsrMatrix<f64>, p: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    assert_eq!(a_ho.nrows, p.nrows, "A_HO rows must match P rows");
    assert_eq!(a_ho.ncols, p.nrows, "A_HO must be square");
    // AP = A_HO · P  (n_hi × n_lo)
    let ap = csr_spmm(a_ho, p);
    // A_LO = Pᵀ · AP  (n_lo × n_lo)
    let pt = p.transpose();
    csr_spmm(&pt, &ap)
}

impl LorAmgPrecond {
    /// Build the LOR-AMG preconditioner from the high‑order matrix and
    /// the prolongation P (maps low‑order → high‑order DOFs).
    pub fn build(a_ho: &CsrMatrix<f64>, p: &CsrMatrix<f64>, amg_cfg: &AmgConfig) -> Self {
        let n_lo = p.ncols;
        let a_lo = build_lor_operator(a_ho, p);
        let la_lo = fem_to_linlvo_csr(&a_lo);
        let hier = AmgHierarchy::build(la_lo, amg_cfg.clone());
        let amg = AmgPrecond::new(hier);
        LorAmgPrecond {
            prolong: p.clone(),
            amg,
            n_lo,
        }
    }
}

/// Helper: compute `y = Pᵀ · x` with a CSR matrix P.
fn apply_prolong_transpose(p: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    // CSR spmv_transpose: for each row i of P, add P[i,j]·x[i] to y[j]
    y.fill(0.0);
    for i in 0..p.nrows {
        let xi = x[i];
        for r in p.row_ptr[i]..p.row_ptr[i + 1] {
            let j = p.col_idx[r] as usize;
            y[j] += p.values[r] * xi;
        }
    }
}

impl Preconditioner for LorAmgPrecond {
    type Vector = DenseVec<f64>;

    fn apply_precond(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
        // 1. Restrict: r_lo = Pᵀ · x_HO
        let mut r_lo = vec![0.0_f64; self.n_lo];
        apply_prolong_transpose(&self.prolong, x.as_slice(), &mut r_lo);

        // 2. AMG on A_LO: z_lo ≈ A_LO⁻¹ · r_lo
        let rhs_lo = DenseVec::from_vec(r_lo);
        let mut z_lo = DenseVec::from_vec(vec![0.0_f64; self.n_lo]);
        self.amg.apply_precond(&rhs_lo, &mut z_lo);

        // 3. Prolong: y_HO = P · z_lo
        self.prolong.spmv(z_lo.as_slice(), y.as_mut_slice());
    }
}

/// Solve `A_HO · x = b` using PCG with the LOR-AMG preconditioner.
///
/// # Arguments
/// * `a_ho` – high‑order system matrix (SPD)
/// * `b`    – right‑hand side
/// * `x`    – initial guess / solution
/// * `lor`  – the LOR-AMG preconditioner (built once)
/// * `cfg`  – convergence parameters
pub fn solve_pcg_lor_amg(
    a_ho: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    lor: &LorAmgPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a_ho.nrows;
    if b.len() != n || x.len() != n {
        return Err(SolverError::DimensionMismatch {
            rows: n,
            cols: n,
            rhs: b.len(),
        });
    }
    // Delegate to `solve_pcg_precond` which takes any Preconditioner.
    crate::solve_pcg_precond(a_ho, b, x, lor, cfg)
}

/// Solve `A_HO · x = b` using GMRES with the LOR-AMG preconditioner.
///
/// Suitable for non‑symmetric high‑order systems when the low‑order
/// operator is a reasonable preconditioner.
pub fn solve_gmres_lor_amg(
    a_ho: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    lor: &LorAmgPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a_ho.nrows;
    if b.len() != n || x.len() != n {
        return Err(SolverError::DimensionMismatch {
            rows: n,
            cols: n,
            rhs: b.len(),
        });
    }
    crate::solve_gmres_precond(a_ho, b, x, restart, lor, cfg)
}

/// Geometric multigrid hierarchy for nested spaces.
///
/// Levels are ordered from fine to coarse. `prolong[l]` maps level `l+1` to
/// level `l` (coarse -> fine).
#[derive(Debug, Clone)]
pub struct GeomMGHierarchy {
    pub levels: Vec<CsrMatrix<f64>>,
    pub prolong: Vec<CsrMatrix<f64>>,
}

impl GeomMGHierarchy {
    pub fn new(levels: Vec<CsrMatrix<f64>>, prolong: Vec<CsrMatrix<f64>>) -> Self {
        assert!(
            levels.len() >= 2,
            "GeomMGHierarchy: need at least two levels"
        );
        assert_eq!(
            prolong.len(),
            levels.len() - 1,
            "GeomMGHierarchy: prolong length mismatch"
        );
        for l in 0..prolong.len() {
            assert_eq!(
                prolong[l].nrows, levels[l].nrows,
                "GeomMGHierarchy: P rows != fine size at level {l}"
            );
            assert_eq!(
                prolong[l].ncols,
                levels[l + 1].nrows,
                "GeomMGHierarchy: P cols != coarse size at level {l}"
            );
        }
        GeomMGHierarchy { levels, prolong }
    }
}

/// Baseline geometric multigrid V-cycle preconditioner.
#[derive(Debug, Clone)]
pub struct GeomMGPrecond {
    pub pre_sweeps: usize,
    pub post_sweeps: usize,
    pub jacobi_omega: f64,
    pub coarse_max_iter: usize,
}

impl Default for GeomMGPrecond {
    fn default() -> Self {
        GeomMGPrecond {
            pre_sweeps: 2,
            post_sweeps: 2,
            jacobi_omega: 0.8,
            coarse_max_iter: 200,
        }
    }
}

impl GeomMGPrecond {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn v_cycle(&self, h: &GeomMGHierarchy, b: &[f64], x: &mut [f64]) {
        self.v_cycle_level(h, 0, b, x);
    }

    fn v_cycle_level(&self, h: &GeomMGHierarchy, lvl: usize, b: &[f64], x: &mut [f64]) {
        let a = &h.levels[lvl];
        if lvl + 1 == h.levels.len() {
            let cfg = SolverConfig {
                rtol: 1e-12,
                atol: 0.0,
                max_iter: self.coarse_max_iter,
                verbose: false,
                ..Default::default()
            };
            let _ = crate::solve_cg(a, b, x, &cfg);
            return;
        }

        jacobi_smooth(a, b, x, self.jacobi_omega, self.pre_sweeps);

        let mut ax = vec![0.0; b.len()];
        a.spmv(x, &mut ax);
        let mut r = vec![0.0; b.len()];
        for i in 0..b.len() {
            r[i] = b[i] - ax[i];
        }

        let p = &h.prolong[lvl];
        let r_c = spmv_transpose(p, &r);
        let mut e_c = vec![0.0; r_c.len()];
        self.v_cycle_level(h, lvl + 1, &r_c, &mut e_c);

        let mut pe = vec![0.0; x.len()];
        p.spmv(&e_c, &mut pe);
        for i in 0..x.len() {
            x[i] += pe[i];
        }

        jacobi_smooth(a, b, x, self.jacobi_omega, self.post_sweeps);
    }
}

/// Solve using repeated geometric multigrid V-cycles.
pub fn solve_vcycle_geom_mg(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    hierarchy: &GeomMGHierarchy,
    mg: &GeomMGPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    if a.nrows != a.ncols || b.len() != a.nrows || x.len() != a.nrows {
        return Err(SolverError::DimensionMismatch {
            rows: a.nrows,
            cols: a.ncols,
            rhs: b.len(),
        });
    }
    if hierarchy.levels[0].nrows != a.nrows {
        return Err(SolverError::DimensionMismatch {
            rows: hierarchy.levels[0].nrows,
            cols: hierarchy.levels[0].ncols,
            rhs: a.nrows,
        });
    }

    let mut ax = vec![0.0; b.len()];
    a.spmv(x, &mut ax);
    let mut r = vec![0.0; b.len()];
    for i in 0..b.len() {
        r[i] = b[i] - ax[i];
    }
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-32);
    let tol = cfg.atol.max(cfg.rtol * b_norm);
    let mut r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r_norm <= tol {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: r_norm,
        });
    }

    for k in 0..cfg.max_iter {
        let mut corr = vec![0.0; x.len()];
        mg.v_cycle(hierarchy, &r, &mut corr);
        for i in 0..x.len() {
            x[i] += corr[i];
        }

        a.spmv(x, &mut ax);
        for i in 0..b.len() {
            r[i] = b[i] - ax[i];
        }
        r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r_norm <= tol {
            return Ok(SolveResult {
                converged: true,
                iterations: k + 1,
                final_residual: r_norm,
            });
        }
    }

    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: r_norm,
    })
}

fn spmv_transpose(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.ncols];
    for i in 0..a.nrows {
        let xi = x[i];
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            y[j] += a.values[p] * xi;
        }
    }
    y
}

fn jacobi_smooth(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], omega: f64, sweeps: usize) {
    if sweeps == 0 {
        return;
    }
    let n = x.len();
    let mut ax = vec![0.0; n];
    let mut diag = vec![1.0; n];
    for i in 0..n {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            if a.col_idx[p] as usize == i {
                diag[i] = a.values[p];
                break;
            }
        }
    }
    for _ in 0..sweeps {
        a.spmv(x, &mut ax);
        for i in 0..n {
            let d = if diag[i].abs() > 1e-14 { diag[i] } else { 1.0 };
            x[i] += omega * (b[i] - ax[i]) / d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    #[test]
    fn solve_pcg_lor_spd_smoke() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 2.0);
        coo.add(1, 1, 3.0);
        let a = coo.into_csr();

        let b = vec![2.0, 3.0];
        let mut x = vec![0.0; 2];
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 200,
            verbose: false,
            ..Default::default()
        };
        let lor = LorPrecond::new();
        let res = solve_pcg_lor(&a, &b, &mut x, &lor, &cfg).expect("solve_pcg_lor failed");

        assert!(res.converged);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solve_gmres_lor_nonsym_smoke() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 3.0);
        coo.add(0, 1, 1.0);
        coo.add(1, 0, 0.0);
        coo.add(1, 1, 2.0);
        let a = coo.into_csr();

        let b = vec![4.0, 2.0];
        let mut x = vec![0.0; 2];
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 200,
            verbose: false,
            ..Default::default()
        };
        let lor = LorPrecond::new();
        let res = solve_gmres_lor(&a, &b, &mut x, 10, &lor, &cfg).expect("solve_gmres_lor failed");

        assert!(res.converged);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    fn lap1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0 {
                coo.add(i, i - 1, -1.0);
            }
            if i + 1 < n {
                coo.add(i, i + 1, -1.0);
            }
        }
        coo.into_csr()
    }

    fn prolong_1d(nf: usize, nc: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(nf, nc);
        // nested odd nodes: coarse node j maps to fine i=2j+1
        for i in 0..nf {
            if i % 2 == 1 {
                let j = (i - 1) / 2;
                if j < nc {
                    coo.add(i, j, 1.0);
                }
            } else {
                // midpoint interpolation between neighboring coarse nodes
                let jr = i / 2;
                if jr > 0 && jr < nc {
                    coo.add(i, jr - 1, 0.5);
                    coo.add(i, jr, 0.5);
                } else if jr == 0 {
                    coo.add(i, 0, 1.0);
                } else {
                    coo.add(i, nc - 1, 1.0);
                }
            }
        }
        coo.into_csr()
    }

    #[test]
    fn geom_mg_vcycle_smoke() {
        let a0 = lap1d(31);
        let a1 = lap1d(15);
        let a2 = lap1d(7);
        let p0 = prolong_1d(31, 15);
        let p1 = prolong_1d(15, 7);
        let h = GeomMGHierarchy::new(vec![a0.clone(), a1, a2], vec![p0, p1]);

        let b = vec![1.0; 31];
        let mut x = vec![0.0; 31];
        let mg = GeomMGPrecond::default();
        let cfg = SolverConfig {
            rtol: 1e-6,
            atol: 0.0,
            max_iter: 80,
            verbose: false,
            ..Default::default()
        };

        let res = solve_vcycle_geom_mg(&a0, &b, &mut x, &h, &mg, &cfg)
            .expect("solve_vcycle_geom_mg failed");
        assert!(
            res.converged,
            "geom mg did not converge: {:.3e}",
            res.final_residual
        );
    }

    // ── LOR-AMG tests ─────────────────────────────────────────────────────

    /// Build P: identity (prolongation = I).  Then A_LO = Pᵀ·A_HO·P = A_HO,
    /// so LOR-AMG reduces to plain AMG — a good baseline check.
    fn identity_prolong(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 1.0);
        }
        coo.into_csr()
    }

    /// Build P: 2×1 (n_hi = 2, n_lo = 1).  Prolongs a scalar into two equal
    /// entries — simple enough to verify the algebra.
    fn simple_prolong() -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(2, 1);
        coo.add(0, 0, 1.0);
        coo.add(1, 0, 1.0);
        coo.into_csr()
    }

    #[test]
    fn build_lor_operator_identity_gives_same_matrix() {
        let n = 5;
        let a_ho = lap1d(n);
        let p = identity_prolong(n);
        let a_lo = build_lor_operator(&a_ho, &p);
        // With P = I, A_LO = Iᵀ · A_HO · I = A_HO
        assert_eq!(a_lo.nrows, n);
        assert_eq!(a_lo.ncols, n);
        for i in 0..n {
            for r in a_lo.row_ptr[i]..a_lo.row_ptr[i + 1] {
                let j = a_lo.col_idx[r] as usize;
                assert!(
                    (a_lo.values[r] - a_ho.get(i, j)).abs() < 1e-15,
                    "A_LO differs at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn build_lor_operator_2x1_verify_manually() {
        // A_HO = [[2,-1],[-1,2]], P = [[1],[1]]
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 2.0);
        coo.add(0, 1, -1.0);
        coo.add(1, 0, -1.0);
        coo.add(1, 1, 2.0);
        let a_ho = coo.into_csr();
        let p = simple_prolong();
        let a_lo = build_lor_operator(&a_ho, &p);
        // A_LO = [1,1]·A_HO·[1;1] = 2+2-1-1 = 2 → 2×2=4... wait
        // Pᵀ·A·P = [1,1]·[[2,-1],[-1,2]]·[1,1]ᵀ
        // = [1,1]·[1;1] = 2
        assert_eq!(a_lo.nrows, 1);
        assert_eq!(a_lo.ncols, 1);
        assert!(
            (a_lo.get(0, 0) - 2.0).abs() < 1e-15,
            "A_LO[0,0] = {} (expected 2)",
            a_lo.get(0, 0)
        );
    }

    #[test]
    fn lor_amg_build_and_apply_smoke() {
        let n = 10;
        let a_ho = lap1d(n);
        let p = identity_prolong(n);
        let amg_cfg = AmgConfig::default();
        let lor = LorAmgPrecond::build(&a_ho, &p, &amg_cfg);
        let x = DenseVec::from_vec(vec![1.0_f64; n]);
        let mut y = DenseVec::from_vec(vec![0.0_f64; n]);
        lor.apply_precond(&x, &mut y);
        assert!(y.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn solve_pcg_lor_amg_with_identity_prolong() {
        // With P = I, LOR-AMG ≡ AMG → should converge rapidly
        let n = 20;
        let a_ho = lap1d(n);
        let p = identity_prolong(n);
        let amg_cfg = AmgConfig::default();
        let lor = LorAmgPrecond::build(&a_ho, &p, &amg_cfg);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 50,
            ..Default::default()
        };
        let res = solve_pcg_lor_amg(&a_ho, &b, &mut x, &lor, &cfg).unwrap();
        assert!(
            res.converged,
            "LOR‑AMG (P=I) did not converge in {} iters (res={:.3e})",
            res.iterations, res.final_residual
        );
        let mut ax = vec![0.0_f64; n];
        a_ho.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "solution error {:.3e}", err);
    }

    #[test]
    fn solve_gmres_lor_amg_with_identity_prolong() {
        let n = 20;
        let a_ho = lap1d(n);
        let p = identity_prolong(n);
        let amg_cfg = AmgConfig::default();
        let lor = LorAmgPrecond::build(&a_ho, &p, &amg_cfg);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 50,
            ..Default::default()
        };
        let res = solve_gmres_lor_amg(&a_ho, &b, &mut x, 30, &lor, &cfg).unwrap();
        assert!(
            res.converged,
            "GMRES+LOR‑AMG (P=I) failed: {} iters res={:.3e}",
            res.iterations, res.final_residual
        );
    }
}
