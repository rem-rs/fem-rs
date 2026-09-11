//! SBM2 Neumann boundary condition integrators.
use crate::integrator::{BilinearIntegrator, LinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

pub struct SBM2NeumannLFIntegrator<C: ScalarCoeff = f64> {
    pub g: C,
}

impl<C: ScalarCoeff> LinearIntegrator for SBM2NeumannLFIntegrator<C> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.weight * self.g.eval(&ctx);
        for i in 0..n { f_elem[i] += w * qp.phi[i]; }
    }
}

/// Bilinear part of the SBM Neumann penalty: `∫ κ φᵢ φⱼ dx` on the cut
/// elements — a mass-type *physical volume* form (see
/// `sbm2_dirichlet.rs`, same convention).
///
/// Uses `QpData::phys_weight`; `QpData::weight` is the *DiffusionIntegrator*
/// convention `ip.weight / |det J|` and would scale the matrix by `|det J|⁻²`.
pub struct SBM2NeumannIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for SBM2NeumannIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.phys_weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::standard::MassIntegrator;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, fe_space::FESpace};

    /// D42 regression: see `sbm2_dirichlet.rs` — same mass-type physical
    /// volume form, same `QpData::phys_weight` requirement.
    #[test]
    fn sbm2_neumann_uses_the_physical_measure() {
        let mesh = Mesh::<2>::make_cartesian_2d(3, 2, 2.0, 1.0);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let md = mass.to_dense();
        let k = Assembler::assemble_bilinear(&space, &[&SBM2NeumannIntegrator { coeff: 1.0 }], 3);
        let d = k.to_dense();
        let err = (0..n * n).map(|i| (d[i] - md[i]).abs()).fold(0.0_f64, f64::max);
        assert!(err <= 1e-12, "max|K − Mass| = {err:.6e}");
    }
}
