//! LOR (Low-Order Refined) preconditioner helpers.
//!
//! This is a lightweight entry point for LOR-backed solves. The current
//! implementation delegates to Jacobi-preconditioned CG while preserving an
//! API that can later be wired to AMG/LOR operators.

use crate::{solve_gmres, solve_pcg_jacobi, SolveResult, SolverConfig, SolverError};
use fem_linalg::CsrMatrix;
use linger::Scalar as LingerScalar;

/// LOR preconditioner configuration.
#[derive(Debug, Clone)]
pub struct LorPrecond {
    /// Number of smoother passes (reserved for future AMG backend).
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

/// Solve SPD system with a LOR-style preconditioned CG path.
///
/// Current backend: PCG + Jacobi preconditioner. The `lor` argument is kept
/// for API stability and future backend selection.
pub fn solve_pcg_lor<T: LingerScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    _lor: &LorPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_pcg_jacobi(a, b, x, cfg)
}

/// Solve a (possibly non-symmetric) system with a LOR-style GMRES path.
///
/// Current backend: vanilla GMRES. The `lor` argument is kept for API
/// compatibility and future backend selection.
pub fn solve_gmres_lor<T: LingerScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    _lor: &LorPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_gmres(a, b, x, restart, cfg)
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
        assert!(levels.len() >= 2, "GeomMGHierarchy: need at least two levels");
        assert_eq!(prolong.len(), levels.len() - 1, "GeomMGHierarchy: prolong length mismatch");
        for l in 0..prolong.len() {
            assert_eq!(prolong[l].nrows, levels[l].nrows, "GeomMGHierarchy: P rows != fine size at level {l}");
            assert_eq!(prolong[l].ncols, levels[l + 1].nrows, "GeomMGHierarchy: P cols != coarse size at level {l}");
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
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200, verbose: false, ..Default::default() };
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
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200, verbose: false, ..Default::default() };
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
        assert!(res.converged, "geom mg did not converge: {:.3e}", res.final_residual);
    }
}
