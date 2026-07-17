//! Geometric multigrid preconditioner.
//!
//! Uses nested meshes (uniformly refined) as the multigrid hierarchy.
//! The polynomial order is the same at each level.
//!
//! # Usage
//! ```ignore
//! // Build hierarchy externally (e.g. in example code), then:
//! let mg = GeometricMgPrecond::default();
//! mg.v_cycle(&hierarchy, &b, &mut x);
//! ```

use fem_linalg::CsrMatrix;
use crate::SolverConfig;
use crate::constrained_operator::RectangularConstrainedOperator;

/// Element-by-element stored matrix operator (ConstrainedOperator pattern).
///
/// Precomputed element stiffness matrices applied via
/// gather → dense mat-vec → scatter.  The element matrices come from the
/// **same** integration loop as the CSR matrix (bitwise-identical).
///
/// BC enforcement follows the `ConstrainedOperator` pattern: zero BC DOFs in
/// the input, multiply, zero BC DOFs in the output.
pub struct StoredElementOperator {
    /// Flattened global DOF indices: `elem_dofs[e * ld + i]`.
    pub elem_dofs: Vec<u32>,
    /// Flattened element matrices: `elem_mats[e * ld² + i * ld + j]`.
    pub elem_mats: Vec<f64>,
    /// Local DOFs per element.
    pub ldofs: usize,
    /// Number of elements.
    pub n_elems: usize,
    /// Total number of global DOFs.
    pub n_dofs: usize,
}

impl StoredElementOperator {
    /// Raw element-by-element mat-vec: `y = A x`.  **No BC enforcement.**
    pub fn mult_raw(&self, x: &[f64], y: &mut [f64]) {
        for v in y.iter_mut() { *v = 0.0; }
        let ld = self.ldofs;
        for e in 0..self.n_elems {
            let dof_base = e * ld;
            let mat_base = e * ld * ld;
            let mut xe = [0.0_f64; 64];
            for i in 0..ld {
                xe[i] = x[self.elem_dofs[dof_base + i] as usize];
            }
            for i in 0..ld {
                let mut sum = 0.0;
                let row_off = mat_base + i * ld;
                for j in 0..ld {
                    sum += self.elem_mats[row_off + j] * xe[j];
                }
                y[self.elem_dofs[dof_base + i] as usize] += sum;
            }
        }
    }

    /// ConstrainedOperator-style mat-vec: `y = A x` with BC enforcement.
    pub fn mult_constrained(&self, x: &[f64], y: &mut [f64], bc_dofs: &[u32]) {
        let mut xc = vec![0.0; self.n_dofs];
        xc.copy_from_slice(x);
        for &d in bc_dofs { if (d as usize) < xc.len() { xc[d as usize] = 0.0; } }
        self.mult_raw(&xc, y);
        for &d in bc_dofs { if (d as usize) < y.len() { y[d as usize] = 0.0; } }
    }
}

/// A single level in the geometric multigrid hierarchy.
pub struct GeometricMgLevel {
    /// System matrix at this level.
    pub mat: CsrMatrix<f64>,
    /// Boundary DOF list for this level.
    pub bc_dofs: Vec<u32>,
    /// Optional element-by-element operator.
    /// When present, `mat_vec()` uses `mult_constrained` (ConstrainedOperator
    /// pattern: zero BC in → raw mult → zero BC out).
    pub elem_op: Option<StoredElementOperator>,
    /// Diagonal from the raw (unmodified) matrix, used by Chebyshev smoother.
    pub raw_diag: Vec<f64>,
    /// Inverse of raw diagonal (1/diag[i] for non-zero, 1.0 for zero/BC DOFs).
    pub raw_dinv: Vec<f64>,
}

/// Geometric multigrid hierarchy.
///
/// `levels[0]` = finest, `levels[n-1]` = coarsest.
/// `prolong[l]` maps from level l+1 (coarse) to level l (fine) with BC enforcement.
pub struct GeometricMgHierarchy {
    pub levels: Vec<GeometricMgLevel>,
    pub prolong: Vec<RectangularConstrainedOperator>,
}

impl GeometricMgLevel {
    /// Perform mat-vec with ConstrainedOperator-style BC enforcement.
    ///
    /// When `elem_op` is present, uses element-by-element with BC zeroing
    /// (matching MFEM's `ConstrainedOperator(Mult(PA_operator, ess_tdofs))`).
    ///
    /// When absent, falls back to CSR `spmv` (the modified matrix already has
    /// BC built in via `apply_dirichlet_symmetric`).
    pub fn mat_vec(&self, x: &[f64], y: &mut [f64]) {
        if let Some(ref op) = self.elem_op {
            op.mult_constrained(x, y, &self.bc_dofs);
        } else {
            self.mat.spmv(x, y);
        }
    }
}

impl GeometricMgHierarchy {
    pub fn new(
        levels: Vec<GeometricMgLevel>,
        prolong_mat: Vec<CsrMatrix<f64>>,
    ) -> Self {
        assert_eq!(prolong_mat.len(), levels.len() - 1,
            "GeometricMgHierarchy: need len(prolong) == len(levels) - 1");
        let mut prolong = Vec::with_capacity(prolong_mat.len());
        for l in 0..prolong_mat.len() {
            prolong.push(RectangularConstrainedOperator {
                mat: prolong_mat[l].clone(),
                ess_fine: levels[l].bc_dofs.clone(),
                ess_coarse: levels[l + 1].bc_dofs.clone(),
            });
        }
        GeometricMgHierarchy { levels, prolong }
    }
    pub fn n_levels(&self) -> usize { self.levels.len() }
    pub fn finest_matrix(&self) -> &CsrMatrix<f64> { &self.levels[0].mat }
}

/// Multigrid cycle type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MgCycleType {
    V,
    W,
}

/// Multigrid smoother type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MgSmootherType {
    /// Jacobi with configurable omega (chebyshev_order = 0)
    Jacobi,
    /// Chebyshev polynomial smoother (order 1-3)
    Chebyshev(usize),
    /// SSOR (forward + backward Gauss-Seidel)
    Ssor,
}

/// Geometric multigrid V-cycle configuration.
#[derive(Debug, Clone)]
pub struct GeometricMgConfig {
    pub pre_sweeps: usize,
    pub post_sweeps: usize,
    /// Smoother type (default: Chebyshev(2) for backward compatibility).
    pub smoother: MgSmootherType,
    pub jacobi_omega: f64,
    pub coarse_max_iter: usize,
    pub coarse_rtol: f64,
    /// Override λ_max estimate. None = auto-estimate.
    pub max_eig_override: Option<f64>,
    /// Multigrid cycle type (V or W). Default: V for backward compatibility.
    pub cycle_type: MgCycleType,
}

impl Default for GeometricMgConfig {
    fn default() -> Self {
        GeometricMgConfig {
            pre_sweeps: 2, post_sweeps: 2,
            smoother: MgSmootherType::Chebyshev(2),
            jacobi_omega: 0.8,
            coarse_max_iter: 200, coarse_rtol: 1e-12,
            max_eig_override: None,
            cycle_type: MgCycleType::V,
        }
    }
}

/// Chebyshev smoother for geometric MG.
/// Stores precomputed coefficients and diagonal.
pub struct MgChebyshevSmoother {
    dinv: Vec<f64>,
    coeffs: Vec<f64>,
}

impl MgChebyshevSmoother {
    fn new(a: &CsrMatrix<f64>, bc: &[u32], order: usize, max_eig_override: Option<f64>) -> Self {
        let n = a.nrows;
        let diag = a.diagonal();
        let mut dinv = vec![0.0; n];
        for i in 0..n { dinv[i] = if diag[i].abs() > 1e-30 { 1.0 / diag[i] } else { 1.0 }; }
        for &d in bc { if (d as usize) < n { dinv[d as usize] = 1.0; } }

        // Estimate λ_max(D⁻¹A) via power iteration, or use override
        let max_eig = max_eig_override.unwrap_or_else(|| {
            estimate_max_eigenvalue_simple(a, &dinv, bc)
        });
        if std::env::var("FEM_MG_DEBUG").is_ok() {
            eprintln!("  [mg] Chebyshev smoother: n={n}, λmax(D⁻¹A)≈{max_eig:.6}");
        }

        // MFEM OperatorChebyshevSmoother parameters
        let upper = 1.2 * max_eig;
        let lower = 0.3 * max_eig;
        let theta = 0.5 * (upper + lower);
        let delta = 0.5 * (upper - lower);
        let th2 = theta * theta;
        let d2 = delta * delta;

        let coeffs = match order - 1 {
            0 => vec![1.0 / theta],
            1 => {
                let tmp = 1.0 / (d2 - 2.0 * th2);
                vec![-4.0 * theta * tmp, 2.0 * tmp]
            }
            2 => {
                let t0 = 3.0 * d2;
                let t1 = th2;
                let t2 = 1.0 / (-4.0 * theta * th2 + theta * t0);
                vec![t2 * (t0 - 12.0 * t1), 12.0 / (t0 - 4.0 * t1), -4.0 * t2]
            }
            _ => panic!("MgChebyshevSmoother: order {order} not supported (1-3)"),
        };
        MgChebyshevSmoother { dinv, coeffs }
    }

    fn smooth(&self, level: &GeometricMgLevel, b: &[f64], x: &mut [f64]) {
        let a = &level.mat;
        let n = a.nrows;
        // Residual: r = b - A*x (uses ConstrainedOperator mat-vec when elem_op present)
        let mut r = vec![0.0; n];
        level.mat_vec(x, &mut r);
        for i in 0..n { r[i] = b[i] - r[i]; }

        // Correction: Δ = p(D⁻¹A) D⁻¹ r, matching MFEM OperatorChebyshevSmoother::Mult
        //
        //   y = Σ_{k=0}^{order-1} C[k] · (D⁻¹A)^k · D⁻¹ · r
        let m = self.coeffs.len();
        let mut correction = vec![0.0; n];
        // residual = D⁻¹ · r
        let mut residual = vec![0.0; n];
        for i in 0..n { residual[i] = r[i] * self.dinv[i]; }

        for k in 0..m {
            let c = self.coeffs[k];
            for i in 0..n { correction[i] += c * residual[i]; }
            if k + 1 < m {
                // residual = D⁻¹ · A · residual
                let mut tmp = vec![0.0; n];
                level.mat_vec(&residual, &mut tmp);
                for i in 0..n { residual[i] = tmp[i] * self.dinv[i]; }
            }
        }

        for i in 0..n { x[i] += correction[i]; }
        for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
    }
}

/// Power iteration on D⁻¹A, matching MFEM OperatorChebyshevSmoother.
fn estimate_max_eigenvalue_simple(a: &CsrMatrix<f64>, dinv: &[f64], bc: &[u32]) -> f64 {
    let n = a.nrows;
    // MFEM: PowerMethod on ProductOperator(D⁻¹, A) with 30 iterations, rtol=1e-8, seed=42
    // We operate on D⁻¹A directly (not D⁻¹/² A D⁻¹/²) — same eigenvalues.
    let mut v = vec![1.0; n];
    // Random-ish initial vector matching MFEM seed=42 behavior
    for i in 0..n { v[i] = 1.0 + (i as f64 % 7.0) * 0.1; }
    for &d in bc { if (d as usize) < n { v[d as usize] = 0.0; } }
    let normalize = |w: &mut [f64]| {
        let nrm: f64 = w.iter().map(|x| x*x).sum::<f64>().sqrt().max(1e-30);
        for x in w.iter_mut() { *x /= nrm; }
    };
    normalize(&mut v);
    let mut lambda = 0.0;
    let mut w = vec![0.0; n];
    // w = D⁻¹ A v
    let mut tmp = vec![0.0; n];
    for _iter in 0..30 {
        // tmp = A * v
        a.spmv(&v, &mut tmp);
        // w = D⁻¹ * tmp
        for i in 0..n { w[i] = tmp[i] * dinv[i]; }
        for &d in bc { if (d as usize) < n { w[d as usize] = 0.0; } }
        let rq: f64 = (0..n).map(|i| v[i]*w[i]).sum();
        if (rq - lambda).abs() < 1e-8 * rq.abs().max(1e-30) && _iter > 2 { lambda = rq; break; }
        lambda = rq;
        normalize(&mut w);
        v.copy_from_slice(&w);
    }
    lambda.abs().max(0.1)
}

/// SSOR smoother: forward sweep then backward sweep with relaxation factor omega.
/// Standard SSOR uses omega in (0, 2); omega = 1 gives symmetric Gauss-Seidel.
fn ssor_smooth_level(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], bc: &[u32], omega: f64) {
    let n = a.nrows;
    // Forward sweep (SOR)
    for i in 0..n {
        let mut diag = 0.0;
        let mut sum = 0.0;
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr as usize] as usize;
            if j == i {
                diag = a.values[ptr as usize];
            } else {
                sum += a.values[ptr as usize] * x[j];
            }
        }
        if diag.abs() > 1e-30 {
            // x_i_new = (1-ω) * x_i_old + ω * (b_i - Σ_{j≠i} A_ij * x_j) / A_ii
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sum) / diag;
        }
    }
    // Backward sweep (SOR, reversed)
    for i in (0..n).rev() {
        let mut diag = 0.0;
        let mut sum = 0.0;
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr as usize] as usize;
            if j == i {
                diag = a.values[ptr as usize];
            } else {
                sum += a.values[ptr as usize] * x[j];
            }
        }
        if diag.abs() > 1e-30 {
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sum) / diag;
        }
    }
    // Re-apply BCs
    for &d in bc {
        if (d as usize) < n {
            x[d as usize] = 0.0;
        }
    }
}

/// Geometric multigrid V-cycle preconditioner.
pub struct GeometricMgPrecond {
    pub config: GeometricMgConfig,
    /// Pre-computed Chebyshev smoothers for each level.
    pub smoothers: Vec<MgChebyshevSmoother>,
}

/// Adapter exposing a [`GeometricMgPrecond`] + [`GeometricMgHierarchy`] pair as
/// a linlvo [`linlvo::Preconditioner`], so it can be plugged directly into
/// [`crate::solve_pcg`] (one V-cycle per application, matching MFEM's use of a
/// `Multigrid` as preconditioner in `PCG`).
pub struct GeometricMgAsPrecond<'a> {
    /// The V-cycle engine (smoother data, config).
    pub mg: &'a GeometricMgPrecond,
    /// The level matrices / transfer operators.
    pub hierarchy: &'a GeometricMgHierarchy,
}

impl linlvo::Preconditioner for GeometricMgAsPrecond<'_> {
    type Vector = linlvo::DenseVec<f64>;

    fn apply_precond(&self, x: &linlvo::DenseVec<f64>, y: &mut linlvo::DenseVec<f64>) {
        self.mg.v_cycle(self.hierarchy, x.as_slice(), y.as_mut_slice());
    }
}

impl GeometricMgPrecond {
    pub fn new(config: GeometricMgConfig, h: &GeometricMgHierarchy) -> Self {
        let smoothers = match config.smoother {
            MgSmootherType::Chebyshev(order) => {
                let mut s = Vec::new();
                for level in &h.levels {
                    s.push(MgChebyshevSmoother::new(
                        &level.mat, &level.bc_dofs, order, config.max_eig_override));
                }
                s
            }
            _ => Vec::new(), // Jacobi and SSOR don't need pre-computed smoothers
        };
        GeometricMgPrecond { config, smoothers }
    }

    /// Apply one V/W-cycle: `x ← cycle(levels, prolong, b)` starting from zero.
    pub fn v_cycle(&self, h: &GeometricMgHierarchy, b: &[f64], x: &mut [f64]) {
        // Start from zero (matching MFEM MultigridBase::ArrayMult: *Y(M-1,j) = 0.0)
        for v in x.iter_mut() { *v = 0.0; }
        let w_cycle = self.config.cycle_type == MgCycleType::W;
        self.v_cycle_level_inner(h, 0, b, x, w_cycle);
    }

    /// Core recursive cycle: handles both V-cycle and W-cycle.
    fn v_cycle_level_inner(&self, h: &GeometricMgHierarchy, lvl: usize,
                           b: &[f64], x: &mut [f64], w_cycle: bool) {
        let level = &h.levels[lvl];
        let a = &level.mat;
        let n = a.nrows;

        if lvl + 1 == h.levels.len() {
            // Coarsest level: CG solve
            let cfg = SolverConfig {
                rtol: self.config.coarse_rtol, atol: 0.0,
                max_iter: self.config.coarse_max_iter,
                verbose: false, ..Default::default()
            };
            let _res = crate::solve_cg(a, b, x, &cfg);
            for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
            return;
        }

        // Pre-smooth
        for _ in 0..self.config.pre_sweeps {
            self.smooth_level(lvl, level, b, x);
        }

        // Restrict residual
        let mut ax = vec![0.0; n];
        level.mat_vec(x, &mut ax);
        let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let mut r_c = Vec::new();
        h.prolong[lvl].restrict(&r, &mut r_c);
        let n_c = h.levels[lvl + 1].mat.nrows;

        if w_cycle {
            // W-cycle: two coarse solves where the second starts from
            // the result of the first (non-zero initial guess), giving
            // a more accurate coarse correction than a single V-cycle.
            let mut e_c = vec![0.0; n_c];
            self.v_cycle_level_inner(h, lvl + 1, &r_c, &mut e_c, true);
            self.v_cycle_level_inner(h, lvl + 1, &r_c, &mut e_c, true);
            // Prolongate correction
            let mut corr = vec![0.0; n];
            h.prolong[lvl].prolong(&e_c, &mut corr);
            for i in 0..n { x[i] += corr[i]; }
        } else {
            let mut e_c = vec![0.0; n_c];
            self.v_cycle_level_inner(h, lvl + 1, &r_c, &mut e_c, false);

            // Prolongate correction
            let mut corr = vec![0.0; n];
            h.prolong[lvl].prolong(&e_c, &mut corr);
            for i in 0..n { x[i] += corr[i]; }
        }

        // Post-smooth
        self.smooth_level(lvl, level, b, x);
        for _ in 1..self.config.post_sweeps {
            self.smooth_level(lvl, level, b, x);
        }
        for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
    }

    fn smooth_level(&self, lvl: usize, level: &GeometricMgLevel, b: &[f64],
                    x: &mut [f64]) {
        let a = &level.mat;
        let bc = &level.bc_dofs;
        match self.config.smoother {
            MgSmootherType::Chebyshev(_) => {
                if lvl < self.smoothers.len() {
                    self.smoothers[lvl].smooth(level, b, x);
                } else {
                    // Jacobi fallback
                    self.jacobi_smooth_level(level, b, x);
                }
            }
            MgSmootherType::Ssor => {
                ssor_smooth_level(a, b, x, bc, self.config.jacobi_omega);
            }
            MgSmootherType::Jacobi => {
                self.jacobi_smooth_level(level, b, x);
            }
        }
    }

    /// Plain Jacobi smoothing (omega from config).
    fn jacobi_smooth_level(&self, level: &GeometricMgLevel, b: &[f64],
                           x: &mut [f64]) {
        let a = &level.mat;
        let diag = a.diagonal();
        let mut r = vec![0.0; a.nrows];
        level.mat_vec(x, &mut r);
        for i in 0..r.len() { r[i] = b[i] - r[i]; }
        let omega = self.config.jacobi_omega;
        for i in 0..r.len() {
            if diag[i].abs() > 1e-30 { x[i] += omega * r[i] / diag[i]; }
        }
        for &d in &level.bc_dofs { if (d as usize) < x.len() { x[d as usize] = 0.0; } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    /// Build a 3-level 1D Poisson geometric MG hierarchy (finite difference).
    /// Coarse level ~ n_c DOFs; each refinement doubles DOFs (linear interpolation).
    fn build_1d_poisson_geom_hierarchy(n_coarse: usize) -> GeometricMgHierarchy {
        let n_levels = 3;
        let mut levels = Vec::new();
        let mut prolong_mats = Vec::new();

        let mut n = n_coarse;
        for lvl in 0..n_levels {
            let h = 1.0 / (n - 1) as f64;
            let mut coo = CooMatrix::<f64>::new(n, n);
            for i in 0..n {
                if i > 0 {
                    coo.add(i, i - 1, -1.0 / h);
                }
                coo.add(i, i, 2.0 / h);
                if i + 1 < n {
                    coo.add(i, i + 1, -1.0 / h);
                }
            }
            let mut mat = coo.into_csr();
            let bc_dofs = vec![0u32, (n - 1) as u32];
            let mut dummy = vec![0.0; n];
            for &d in &bc_dofs {
                mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy);
            }
            levels.push(GeometricMgLevel {
                mat,
                bc_dofs: bc_dofs.clone(),
                elem_op: None,
                raw_diag: vec![],
                raw_dinv: vec![],
            });

            // Build prolongation (linear interpolation) to next finer level
            if lvl + 1 < n_levels {
                let n_fine = 2 * n - 1;
                let mut p_coo = CooMatrix::<f64>::new(n_fine, n);
                for i in 0..n_fine {
                    if i % 2 == 0 {
                        p_coo.add(i, i / 2, 1.0);
                    } else {
                        p_coo.add(i, i / 2, 0.5);
                        p_coo.add(i, i / 2 + 1, 0.5);
                    }
                }
                prolong_mats.push(p_coo.into_csr());
            }
            n = 2 * n - 1;
        }

        // Reverse so finest is first
        levels.reverse();
        prolong_mats.reverse();

        GeometricMgHierarchy::new(levels, prolong_mats)
    }

    /// Simple preconditioned Richardson iteration with MG as preconditioner.
    fn richardson_mg(
        mg: &GeometricMgPrecond,
        h: &GeometricMgHierarchy,
        b: &[f64],
        x: &mut [f64],
        max_iter: usize,
        rtol: f64,
    ) -> usize {
        let a = h.finest_matrix();
        let n = a.nrows;
        let b_nrm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        for iter in 0..max_iter {
            let mut ax = vec![0.0; n];
            a.spmv(x, &mut ax);
            let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
            let res = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if res < rtol * b_nrm {
                return iter;
            }
            let mut dx = vec![0.0; n];
            mg.v_cycle(h, &r, &mut dx);
            for i in 0..n {
                x[i] += dx[i];
            }
        }
        max_iter
    }

    #[test]
    fn geometric_mg_w_cycle_converges_faster() {
        let n_coarse = 9; // coarse DOFs = 9, fine DOFs = 33
        let h = build_1d_poisson_geom_hierarchy(n_coarse);
        let n = h.levels[0].mat.nrows;
        eprintln!("  MG hierarchy DOFs per level:");
        for (i, lvl) in h.levels.iter().enumerate() {
            eprintln!("    level {}: {} DOFs", i, lvl.mat.nrows);
        }

        // RHS: all ones (compatible with homogeneous Dirichlet BCs)
        let mut b = vec![1.0_f64; n];
        for &d in &h.levels[0].bc_dofs {
            if (d as usize) < n {
                b[d as usize] = 0.0;
            }
        }

        let max_iter = 200;
        let rtol = 1e-8;

        // V-cycle (Jacobi smoother)
        let cfg_v = GeometricMgConfig {
            cycle_type: MgCycleType::V,
            smoother: MgSmootherType::Jacobi,
            jacobi_omega: 0.67,
            pre_sweeps: 2,
            post_sweeps: 2,
            ..Default::default()
        };
        let mg_v = GeometricMgPrecond::new(cfg_v, &h);
        let mut x_v = vec![0.0; n];
        let iters_v = richardson_mg(&mg_v, &h, &b, &mut x_v, max_iter, rtol);

        // W-cycle (Jacobi smoother)
        let cfg_w = GeometricMgConfig {
            cycle_type: MgCycleType::W,
            smoother: MgSmootherType::Jacobi,
            jacobi_omega: 0.67,
            pre_sweeps: 2,
            post_sweeps: 2,
            ..Default::default()
        };
        let mg_w = GeometricMgPrecond::new(cfg_w, &h);
        let mut x_w = vec![0.0; n];
        let iters_w = richardson_mg(&mg_w, &h, &b, &mut x_w, max_iter, rtol);

        eprintln!("  V-cycle: {iters_v} iters, W-cycle: {iters_w} iters");
        assert!(
            iters_w <= iters_v,
            "W-cycle ({iters_w}) should need <= V-cycle ({iters_v}) iterations"
        );
    }
}

