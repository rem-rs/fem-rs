//! Mixed scalar diffusion integrator — `∫ κ ∇u_trial · ∇v_test dx` between two
//! scalar spaces (the coupling form of MFEM's `DiffusionIntegrator`, used e.g.
//! as the primal block `B0` of a DPG normal-equation system).
//!
//! **Arithmetic follows MFEM `DiffusionIntegrator::AssembleElementMatrix2`**
//! (`fem/bilininteg.cpp`) for the scalar-coefficient case: per quadrature point
//! the weight `w = ip.weight / Weight()` (signed determinant, the assembler's
//! volume convention) is applied to every entry of the trial-side physical
//! gradient row, and the test-side row is then contracted against it with a
//! **single dot product per `(test, trial)` entry** accumulated in dimension
//! order, added once (`AddMultABt`: `elmat(i,j) += Σ_k te(i,k)·(w·tr(j,k))`).
//!
//! Splitting that dot product into per-dimension partial additions
//! (`m += w·g_x·g_x` then `m += w·g_y·g_y`, the shape of the previous
//! example-local integrator) disagrees with MFEM by up to 5.6e-14 relative on
//! the ex8 star-mesh blocks (D864) — small, but it feeds `b = Bᵀ S⁻¹ F` and
//! `A = Bᵀ S⁻¹ B`, so it perturbs the DPG outer PCG trajectory.
//!
//! MFEM's `GetIntegrationRule(trial, test, Trans)` default for this form is
//! `2·max(GetOrder) + OrderW` → 2×2 GL points on `[0,1]` for order-1 test/trial
//! pairs, matching `quad_rule_01(3)`.

use crate::integrator::QpData;
use crate::mixed::MixedBilinearIntegrator;

/// `B(i, j) = ∫ ∇φ_test,i · ∇φ_trial,j dx` (unit coefficient).
pub struct MixedScalarDiffusionIntegrator;

impl MixedBilinearIntegrator for MixedScalarDiffusionIntegrator {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m_elem: &mut [f64]) {
        let n_row = qp_row.n_dofs;
        let n_col = qp_col.n_dofs;
        let dim = qp_col.dim;
        let w = qp_col.weight;
        for i in 0..n_row {
            for j in 0..n_col {
                let mut s = 0.0;
                for k in 0..dim {
                    s += qp_row.grad_phys[i * dim + k] * (w * qp_col.grad_phys[j * dim + k]);
                }
                m_elem[i * n_col + j] += s;
            }
        }
    }
}
