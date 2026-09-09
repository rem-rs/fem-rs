//! Bramble–Pasciak solver assembly helpers.
//!
//! 1:1 port of the element-wise mass preconditioner construction of MFEM
//! `miniapps/solvers/bramble_pasciak.{hpp,cpp}` —
//! `BramblePasciakSolver::ConstructMassPreconditioner`.
//!
//! For the Darcy saddle-point system `[[M, Bᵀ], [B, 0]]`, Bramble–Pasciak
//! preconditioning requires an SPD `Q` such that `M − Q` stays SPD.  MFEM
//! builds `Q` from the *element* mass matrices `M_T`:
//!
//! ```text
//!     M_T x = λ · diag(M_T) x          (generalized eigenproblem, per element)
//!     Q_T  = α · λ_min · diag(M_T)     (0 < α < 1, q_scaling)
//! ```
//!
//! so that on every element `Q_T` is a scaled copy of the diagonal of `M_T`.
//! [`element_q_scaling`] computes the per-element factor `α·λ_min` following
//! MFEM's no-LAPACK path exactly:
//! 1. form `M̄ = D^{-1/2} M_T D^{-1/2}` (`DenseMatrix::InvSymmetricScaling`),
//! 2. smallest eigenvalue of `M̄` by the inverse power method on `M̄^{-1}`,
//!    started from `x = Vector::Randomize(696383552 + 779345·elem)` — i.e.
//!    glibc `srand` + `rand()/2³¹` ([`crate::geometric_mg::GlibcRand`]) —
//!    iterating until the relative change of the Rayleigh value is `≤ 1e-12`
//!    (at most 1000 iterations, as MFEM).
//!
//! The assembly of the global `Q` (`qVarf.AssembleElementMatrix(i, Q_i, 1)`)
//! is the responsibility of the caller / bilinear-form layer.

use fem_linalg::dense::{lu_factor, lu_solve};

use crate::geometric_mg::GlibcRand;

/// Per-element scaling factor `q_scaling · λ_min` of MFEM
/// `BramblePasciakSolver::ConstructMassPreconditioner`.
///
/// # Arguments
/// * `m_elem` — row-major `n × n` element mass matrix `M_T` (SPD).
/// * `n`      — element matrix dimension.
/// * `q_scaling` — `α`, must lie in `(0, 1)` (MFEM `MFEM_ASSERT`).
/// * `elem_index` — global element index; seeds the power iteration exactly
///   like MFEM (`696383552 + 779345·elem`).
///
/// # Returns
/// `q_scaling · λ_min` where `λ_min` is the smallest eigenvalue of
/// `M_T x = λ diag(M_T) x`.
///
/// # Panics
/// If `q_scaling` is outside `(0, 1)`, the element matrix is singular, or the
/// inverse power iteration fails to converge in 1000 iterations (MFEM calls
/// `MFEM_ASSERT`/`MFEM_VERIFY` and aborts in the same situations).
pub fn element_q_scaling(m_elem: &[f64], n: usize, q_scaling: f64, elem_index: usize) -> f64 {
    assert!(
        (q_scaling > 0.0) && (q_scaling < 1.0),
        "Invalid Q-scaling factor: q_scaling = {q_scaling}"
    );
    assert_eq!(m_elem.len(), n * n, "element matrix must be n×n row-major");

    // D = diag(M_T); form M̄ = D^{-1/2} · M_T · D^{-1/2} in place
    // (MFEM DenseMatrix::InvSymmetricScaling).
    let mut m = m_elem.to_vec();
    let mut inv_sqrt = vec![0.0f64; n];
    for i in 0..n {
        inv_sqrt[i] = 1.0 / m[i * n + i].sqrt();
    }
    for j in 0..n {
        for i in 0..n {
            m[j * n + i] *= inv_sqrt[i] * inv_sqrt[j];
        }
    }

    // Inverse power iteration for the smallest eigenvalue of M̄, i.e. the
    // largest eigenvalue of M̄^{-1} (MFEM no-LAPACK branch).
    let mut piv = vec![0usize; n];
    lu_factor(&mut m, n, &mut piv)
        .expect("element mass matrix must be non-singular for the power method");

    // x.Randomize(696383552 + 779345 * elem): glibc rand()/2^31 draws.
    let seed = 696_383_552u64.wrapping_add(779_345 * elem_index as u64) as u32;
    let mut rng = GlibcRand::new(seed);
    let mut x: Vec<f64> = (0..n).map(|_| rng.rand_real()).collect();

    const REL_TOL: f64 = 1e-12;
    const MAX_ITER: usize = 1000;
    let mut eval = 0.0f64;
    let mut eval_prev;
    let mut mx = vec![0.0f64; n];
    let mut iter = 0usize;
    let mut converged = false;
    loop {
        eval_prev = eval;
        // M̄^{-1} · x (LU factors of `m` already computed).
        mx.copy_from_slice(&x);
        lu_solve(&m, n, &piv, &mut mx);
        eval = mx.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv_eval = 1.0 / eval;
        for (xi, mxi) in x.iter_mut().zip(mx.iter()) {
            *xi = *mxi * inv_eval;
        }
        iter += 1;
        let rel = (eval - eval_prev).abs() / eval.abs();
        if rel <= REL_TOL {
            converged = true;
            break;
        }
        if iter >= MAX_ITER {
            break;
        }
    }
    assert!(
        converged,
        "Inverse power method did not converge.\n\t iter      = {iter}\n\t \
         eval_i    = {eval}\n\t eval_prev = {eval_prev}\n\t rel change = {}",
        (eval - eval_prev).abs() / eval.abs()
    );

    let lambda_min = 1.0 / eval;
    q_scaling * lambda_min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_scaling_matches_analytic_3x3() {
        // M = [[3,1,0],[1,3,1],[0,1,3]] with D = diag(3,3,3): the generalized
        // eigenproblem M x = λ D x reduces to the plain one for M̄ = M/3 whose
        // eigenvalues are {1, (3±√2)/3}; λ_min = (3−√2)/3.
        let m: [f64; 9] = [3.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 3.0];
        let lam_min = (3.0 - std::f64::consts::SQRT_2) / 3.0;
        let got = element_q_scaling(&m, 3, 0.5, 7);
        let expect = 0.5 * lam_min;
        assert!(
            (got - expect).abs() < 1e-10,
            "scaling {got:.15e} != 0.5·λ_min {expect:.15e}"
        );
        // Same value regardless of the power-iteration seed (element index).
        for elem in 0..16usize {
            let g = element_q_scaling(&m, 3, 0.5, elem);
            assert!(
                (g - expect).abs() < 1e-10,
                "elem {elem}: scaling {g:.15e} != {expect:.15e}"
            );
        }
    }

    #[test]
    fn q_scaling_matches_analytic_2x2() {
        // M = [[2,-0.5],[-0.5,2]], D = diag(2,2): M̄ = M/2 has eigenvalues
        // {1.25, 0.75} → λ_min = 0.75.
        let m: [f64; 4] = [2.0, -0.5, -0.5, 2.0];
        let got = element_q_scaling(&m, 2, 0.5, 0);
        assert!((got - 0.375).abs() < 1e-10, "scaling {got:.15e} != 0.375");
    }

    #[test]
    #[should_panic(expected = "Invalid Q-scaling")]
    fn q_scaling_rejects_out_of_range() {
        let m: [f64; 4] = [2.0, 0.0, 0.0, 2.0];
        let _ = element_q_scaling(&m, 2, 1.5, 0);
    }
}
