//! SBM2 Dirichlet boundary condition integrators.
use crate::integrator::{BilinearIntegrator, LinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

pub struct SBM2DirichletLFIntegrator<C: ScalarCoeff = f64> {
    pub g: C,
}

impl<C: ScalarCoeff> LinearIntegrator for SBM2DirichletLFIntegrator<C> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.weight * self.g.eval(&ctx);
        for i in 0..n { f_elem[i] += w * qp.phi[i]; }
    }
}

/// Bilinear part of the SBM Dirichlet penalty: `∫ κ φᵢ φⱼ dx` on the cut
/// elements — a mass-type *physical volume* form.
///
/// It must therefore use `QpData::phys_weight` (`quadrature weight · |det J|`).
/// `QpData::weight` is the *DiffusionIntegrator* convention
/// (`ip.weight / |det J|`) and scaled this matrix by `|det J|⁻²` — measured as
/// `max|K − MassIntegrator| = 1.185e0` on a 3×2 mesh of 2/3 × 1/2 elements
/// (`|det J| = 1/3`).  The face-assembled SBM3 kernels
/// (`sbm3_dirichlet.rs` / `sbm3_neumann.rs`) fold the true face measure in
/// explicitly and are unaffected.
pub struct SBM2DirichletIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for SBM2DirichletIntegrator<C> {
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

    /// D42 regression: the SBM Dirichlet penalty is a mass-type physical
    /// volume form, so `κ = 1` must reproduce the scalar mass matrix exactly.
    /// With `QpData::weight` it was 9× too large on this `|det J| = 1/3` mesh
    /// (`max|K − Mass| = 1.185e0` before the fix).
    #[test]
    fn sbm2_dirichlet_uses_the_physical_measure() {
        let mesh = Mesh::<2>::make_cartesian_2d(3, 2, 2.0, 1.0);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let md = mass.to_dense();
        let k = Assembler::assemble_bilinear(&space, &[&SBM2DirichletIntegrator { coeff: 1.0 }], 3);
        let d = k.to_dense();
        let err = (0..n * n).map(|i| (d[i] - md[i]).abs()).fold(0.0_f64, f64::max);
        assert!(err <= 1e-12, "max|K − Mass| = {err:.6e}");
    }
}
