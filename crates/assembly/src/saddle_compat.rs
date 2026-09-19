//! Discrete saddle-point compatibility correction (D411).
//!
//! For a mixed system `[K −Gᵀ; G 0]` (Stokes `−νΔu + ∇p = f, ∇·u = g`, Darcy
//! `σ + ∇p = f, ∇·σ = g`, …) the constraint block `G u = g` (divergence rows)
//! is solvable only if the data satisfy the discrete compatibility identity
//!
//! ```text
//!   Σ_q g_q = Σ_q (G·u_bc)_q ,
//! ```
//!
//! the discrete analog of `∫Ω ∇·u = ∫∂Ω u·n` — with a partition-of-unity
//! pressure space (P0/P1: `Σ_q φ_q = 1`) this identity is exact whenever the
//! data come from a compatible continuous problem.  This is the same
//! mechanism as the pure-Neumann Poisson pressure-nullspace condition in
//! MFEM: the constraint rows annihilate the constant vector, so any residual
//! mismatch δ lies in the range of the *transposed* nullspace and the
//! unregularized system is singular.
//!
//! Why regularization is not enough: solving with an εI pressure block
//! (or a dense LU on the ε-regularized matrix) absorbs the mismatch δ as a
//! `δ/ε` pressure constant — for δ ≈ 1.5e-3 of quadrature + interpolation
//! noise and ε = 1e-12 this produced a 1.4e11 pressure constant (observed in
//! the round-48 Stokes–Darcy work, D401).  The exact fix is to make the
//! identity hold: distribute δ uniformly over the constraint rows
//! (`rhs_q += δ/n_q`), which perturbs the datum by the constant δ/Σvol — an
//! O(h²)–O(h⁴) quantity that vanishes under refinement.

use std::collections::BTreeMap;

use fem_linalg::CsrMatrix;

/// Compute the discrete compatibility defect
/// `δ = Σ_q (G·u_bc)_q − Σ_q rhs_q` of a saddle system's constraint block.
///
/// * `mixed_block` — the constraint (divergence) block `G`, rows indexed by
///   constraint dofs `q`, columns by primal dofs.
/// * `rhs_constraint` — the current constraint-row right-hand side `g`.
/// * `essential` — the essential (Dirichlet) primal dofs and their
///   prescribed values; free dofs contribute zero to `Σ_q (G·u_bc)_q`, so the
///   boundary reaction is `Σ_{j,c} G[j,c]·u_bc[c]` over pinned columns `c`.
///
/// Returns the defect with the sign convention
/// `δ = reaction − Σ rhs_q`, i.e. [`correct_saddle_compatibility`] with this
/// δ drives the defect back to zero (to machine precision).
pub fn saddle_compatibility_defect(
    mixed_block: &CsrMatrix<f64>,
    rhs_constraint: &[f64],
    essential: &BTreeMap<usize, f64>,
) -> f64 {
    // reaction = Σ_q (G·u_bc)_q  (free dofs contribute zero to the row sums)
    let mut reaction = 0.0_f64;
    for r in 0..mixed_block.nrows {
        for ptr in mixed_block.row_ptr[r]..mixed_block.row_ptr[r + 1] {
            let c = mixed_block.col_idx[ptr] as usize;
            if let Some(&v) = essential.get(&c) {
                reaction += mixed_block.values[ptr] * v;
            }
        }
    }
    let datum: f64 = rhs_constraint.iter().sum();
    reaction - datum
}

/// Distribute the compatibility defect `delta` uniformly over the constraint
/// rows: `rhs_q += delta / n`.  After the correction,
/// `Σ_q rhs_q == Σ_q (G·u_bc)_q` holds to machine precision, so an εI
/// pressure regularization no longer amplifies the defect by 1/ε.
pub fn correct_saddle_compatibility(rhs_q: &mut [f64], delta: f64) {
    let share = delta / rhs_q.len() as f64;
    for v in rhs_q.iter_mut() {
        *v += share;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G = [1 2 0; 0 1 2], essential u_bc = {0: 1.5, 2: -0.5} ⇒
    /// reaction = 1·1.5 + 0·(−0.5) + 0·1.5 + 2·(−0.5) = 0.5.
    fn tiny_g() -> CsrMatrix<f64> {
        let mut coo = fem_linalg::CooMatrix::<f64>::new(2, 3);
        coo.add(0, 0, 1.0);
        coo.add(0, 1, 2.0);
        coo.add(1, 1, 1.0);
        coo.add(1, 2, 2.0);
        coo.into_csr()
    }

    #[test]
    fn correction_restores_compatibility_to_machine_precision() {
        let g = tiny_g();
        let essential: BTreeMap<usize, f64> = BTreeMap::from([(0, 1.5), (2, -0.5)]);
        let mut rhs = vec![0.3_f64, 0.7];

        let delta = saddle_compatibility_defect(&g, &rhs, &essential);
        // Premise: the raw defect is the O(1e-3)-style mismatch (0.5 − 1.0).
        assert!((delta - (-0.5)).abs() < 1e-15, "defect = {delta}, want -0.5");
        assert!(delta.abs() > 1e-3, "test premise: defect must be nonzero");

        correct_saddle_compatibility(&mut rhs, delta);

        let residual = saddle_compatibility_defect(&g, &rhs, &essential);
        assert!(
            residual.abs() < 1e-15,
            "residual defect {residual} not at machine precision"
        );
        // Uniform split: each row received delta/2 = -0.25.
        assert!((rhs[0] - 0.05).abs() < 1e-15 && (rhs[1] - 0.45).abs() < 1e-15);
    }

    #[test]
    fn zero_defect_is_a_noop() {
        let g = tiny_g();
        let essential: BTreeMap<usize, f64> = BTreeMap::from([(0, 1.5), (2, -0.5)]);
        let mut rhs = vec![0.5_f64, 0.0]; // Σ = 0.5 = reaction ⇒ δ = 0
        assert_eq!(saddle_compatibility_defect(&g, &rhs, &essential), 0.0);
        correct_saddle_compatibility(&mut rhs, 0.0);
        assert_eq!(rhs, vec![0.5, 0.0]);
    }
}
