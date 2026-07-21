//! Advanced preconditioners for saddle-point (Stokes/Navier-Stokes) systems.
//!
//! Provides [`StokesPrecond`]: a block-triangular preconditioner with BFBt
//! Schur complement approximation using the pressure mass matrix.

use crate::block::BlockSystem;
use crate::{solve_cg, solve_gmres, SolverConfig, SolveResult, SolverError};
use fem_linalg::{CooMatrix, CsrMatrix};

/// Build the pressure mass matrix from the divergence operator B.
///
/// `M_p = diag(B * B^T)` — a lumped approximation of the pressure mass matrix.
/// Each entry `M_p[i,i] = Σ_k B_ik^2` (the diagonal of B*B^T).
pub fn build_pressure_mass(b: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n_p = b.nrows;
    let mut coo = CooMatrix::<f64>::new(n_p, n_p);
    for i in 0..n_p {
        let mut diag = 0.0;
        let start = b.row_ptr[i];
        let end = b.row_ptr[i + 1];
        for ptr in start..end {
            let bik = b.values[ptr];
            diag += bik * bik;
        }
        coo.add(i, i, if diag > 0.0 { diag } else { 1.0 });
    }
    coo.into_csr()
}

/// Build `S = B * diag(A)^{-1} * B^T` (BFBt Schur complement approximation) as CSR.
pub fn build_bfbt_schur(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n_p = b.nrows;
    let inv_diag_a: Vec<f64> = (0..a.nrows).map(|i| {
        let d = a.get(i, i);
        if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
    }).collect();

    let mut coo = CooMatrix::<f64>::new(n_p, n_p);
    for i in 0..n_p {
        let start = b.row_ptr[i];
        let end = b.row_ptr[i + 1];
        for ptr in start..end {
            let k = b.col_idx[ptr] as usize;
            let bik = b.values[ptr];
            let aik_inv = inv_diag_a[k];
            let bt_start = b.row_ptr[k];
            let bt_end = b.row_ptr[k + 1];
            for bt_ptr in bt_start..bt_end {
                let j = b.col_idx[bt_ptr] as usize;
                let bjk = b.values[bt_ptr];
                coo.add(i, j, bik * aik_inv * bjk);
            }
        }
    }
    coo.into_csr()
}

// ─── StokesPrecond ───────────────────────────────────────────────────────────

/// Block-triangular preconditioner for Stokes:
/// ```text
/// P = [ A_approx    B^T  ]
///     [ 0          S_approx ]
/// ```
/// where `A_approx^{-1} ≈ diag(A)^{-1}` (Jacobi) and
/// `S_approx = M_p` (pressure mass matrix) or a custom Schur complement.
pub struct StokesPrecond {
    a: CsrMatrix<f64>,
    bt: CsrMatrix<f64>,
    #[allow(dead_code)]
    b: CsrMatrix<f64>,
    mp: CsrMatrix<f64>,
    inv_diag_a: Vec<f64>,
}

impl StokesPrecond {
    /// Create from the block system and pressure mass matrix.
    pub fn new(sys: &BlockSystem, mp: CsrMatrix<f64>) -> Self {
        let inv_diag_a = (0..sys.a.nrows).map(|i| {
            let d = sys.a.get(i, i);
            if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
        }).collect();
        StokesPrecond { a: sys.a.clone(), bt: sys.bt.clone(), b: sys.b.clone(), mp, inv_diag_a }
    }

    /// Create with an externally assembled Schur complement approximation.
    pub fn new_with_schur(sys: &BlockSystem, schur: CsrMatrix<f64>) -> Self {
        Self::new(sys, schur)
    }

    /// Apply the preconditioner:
    ///   zp = M_p^{-1} * rp   (Schur complement solve via CG)
    ///   zu = diag(A)^{-1} * (ru - B^T * zp)
    pub fn apply(&self, ru: &[f64], rp: &[f64], zu: &mut [f64], zp: &mut [f64]) -> Result<(), SolverError> {
        solve_cg(&self.mp, rp, zp, &SolverConfig { rtol: 1e-3, atol: 0.0, max_iter: 50, verbose: false, ..SolverConfig::default() })?;
        let n_u = self.a.nrows;
        let mut bt_zp = vec![0.0; n_u];
        self.bt.spmv(zp, &mut bt_zp);
        for i in 0..ru.len() { zu[i] = self.inv_diag_a[i] * (ru[i] - bt_zp[i]); }
        Ok(())
    }

    /// Solve the Stokes system with GMRES (no preconditioner, flat system).
    pub fn solve_gmres(&self, sys: &BlockSystem, f: &[f64], g: &[f64],
        u: &mut [f64], p: &mut [f64]) -> Result<SolveResult, SolverError> {
        let n = sys.n_u() + sys.n_p();
        let flat = sys.to_flat_csr();
        let mut rhs = vec![0.0; n];
        rhs[..sys.n_u()].copy_from_slice(f);
        rhs[sys.n_u()..].copy_from_slice(g);
        let mut x = vec![0.0; n];
        let res = solve_gmres(&flat, &rhs, &mut x, 50, &SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() })?;
        u.copy_from_slice(&x[..sys.n_u()]);
        p.copy_from_slice(&x[sys.n_u()..]);
        Ok(res)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_stokes() -> (BlockSystem, CsrMatrix<f64>) {
        let mut a_coo = CooMatrix::<f64>::new(2, 2);
        a_coo.add(0, 0, 2.0); a_coo.add(0, 1, -1.0);
        a_coo.add(1, 0, -1.0); a_coo.add(1, 1, 2.0);
        let a = a_coo.into_csr();

        let mut b_coo = CooMatrix::<f64>::new(2, 2);
        b_coo.add(0, 0, 1.0); b_coo.add(1, 1, 1.0);
        let b = b_coo.into_csr();

        let mut bt_coo = CooMatrix::<f64>::new(2, 2);
        bt_coo.add(0, 0, 1.0); bt_coo.add(1, 1, 1.0);
        let bt = bt_coo.into_csr();

        let mut mp_coo = CooMatrix::<f64>::new(2, 2);
        mp_coo.add(0, 0, 1.0); mp_coo.add(1, 1, 1.0);
        let mp = mp_coo.into_csr();

        (BlockSystem { a, bt, b, c: None }, mp)
    }

    #[test]
    fn stokes_precond_apply() {
        let (sys, mp) = make_test_stokes();
        let precond = StokesPrecond::new(&sys, mp);
        let mut zu = vec![0.0; 2];
        let mut zp = vec![0.0; 2];
        let res = precond.apply(&[1.0, 0.0], &[1.0, 0.0], &mut zu, &mut zp);
        assert!(res.is_ok());
        assert!((zp[0] - 1.0).abs() < 1e-6);
        assert!(zp[1].abs() < 1e-10);
    }

    #[test]
    fn bfbt_schur_build_test() {
        let (sys, _) = make_test_stokes();
        let schur = build_bfbt_schur(&sys.a, &sys.b);
        assert_eq!(schur.nrows, 2);
        let d0 = schur.get(0, 0);
        assert!((d0 - 0.5).abs() < 1e-10, "BFBt[0,0]={}, expected 0.5", d0);
    }

    #[test]
    fn pressure_mass_build_test() {
        let (sys, _) = make_test_stokes();
        let mass = build_pressure_mass(&sys.b);
        assert!((mass.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((mass.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stokes_gmres_solve_test() {
        let (sys, mp) = make_test_stokes();
        let precond = StokesPrecond::new(&sys, mp);
        let mut u = vec![0.0; 2];
        let mut p = vec![0.0; 2];
        let res = precond.solve_gmres(&sys, &[1.0, 0.0], &[0.0, 0.0], &mut u, &mut p);
        assert!(res.is_ok(), "solve failed");
        // Solution: A*u + B^T*p = f, B*u = g
        // A = [[2,-1],[-1,2]], B = I, f = [1,0], g = [0,0]
        // u = 0 (B*u = 0 → u = 0)
        // p = f - A*u = f = [1, 0]
        assert!((p[0] - 1.0).abs() < 1e-6, "p[0]={}", p[0]);
        assert!(u[0].abs() < 1e-8, "u[0]={}", u[0]);
    }
}
