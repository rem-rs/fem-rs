//! DG convection-diffusion-reaction (CDR) system.
//!
//! Composes SIP-DG diffusion + upwind DG advection + mass reaction.
//!
//! ```text
//! ∂u/∂t + b·∇u − ∇·(κ∇u) + r·u = f
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use crate::assembler::Assembler;
use crate::postproc::coefficient::ConstantVectorCoeff;
use super::dg::DgAssembler;
use super::dg_advection::{DGAdvectionIntegrator, assemble_dg_interior_faces};
use crate::interior_faces::InteriorFaceList;
use crate::standard::{DiffusionIntegrator, MassIntegrator, ConvectionIntegrator};

/// Convenience builder for DG CDR system assembly.
pub struct DgCdrSystem;

impl DgCdrSystem {
    /// Assemble the full CDR system matrix.
    pub fn assemble<S, M>(
        space: &S,
        ifl: &InteriorFaceList,
        kappa: f64,
        velocity: &[f64],
        reaction: f64,
        sigma: f64,
        quad_order: u8,
    ) -> CsrMatrix<f64>
    where
        S: FESpace<Mesh = M>,
        M: MeshTopology,
    {
        let n_dofs = space.n_dofs();

        // Volume: diffusion + advection + reaction(mass)
        let k_diff = Assembler::assemble_bilinear(space, &[
            &DiffusionIntegrator { kappa },
        ], quad_order);

        let k_conv = Assembler::assemble_bilinear(space, &[
            &ConvectionIntegrator { velocity: ConstantVectorCoeff(velocity.to_vec()) },
        ], quad_order);

        let k_mass = Assembler::assemble_bilinear(space, &[
            &MassIntegrator { rho: reaction },
        ], quad_order);

        // Interior faces: SIP diffusion
        let k_sip = DgAssembler::assemble_sip(space, ifl, kappa, sigma, quad_order);

        // Interior faces: upwind advection
        let mut coo_adv = CooMatrix::<f64>::new(n_dofs, n_dofs);
        assemble_dg_interior_faces(&mut coo_adv, space.mesh(), space, ifl, space.order(), quad_order,
            &DGAdvectionIntegrator { velocity: ConstantVectorCoeff(velocity.to_vec()) });
        let k_adv_face = coo_adv.into_csr();

        // Combine via COO
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        let matrices = [&k_diff, &k_conv, &k_mass, &k_sip, &k_adv_face];
        for mat in &matrices {
            for i in 0..n_dofs {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    let v = mat.values[p];
                    if v.abs() > 1e-30 { coo.add(i, j, v); }
                }
            }
        }
        coo.into_csr()
    }

    /// Assemble the RHS vector including volumetric source.
    pub fn assemble_rhs<S, M>(
        space: &S,
        source: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
        quad_order: u8,
    ) -> Vec<f64>
    where
        S: FESpace<Mesh = M>,
        M: MeshTopology,
    {
        use crate::standard::DomainSourceIntegrator;
        Assembler::assemble_linear(space, &[
            &DomainSourceIntegrator::new(source),
        ], quad_order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::L2Space;

    #[test]
    fn dg_cdr_assembles() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let k = DgCdrSystem::assemble(&space, &ifl, 1.0, &[1.0, 0.0], 0.0, 10.0, 3);
        let n = space.n_dofs();
        let mut sum = 0.0;
        for i in 0..n.min(20) {
            for j in 0..n.min(20) { sum += k.get(i, j).abs(); }
        }
        assert!(sum > 0.0, "CDR matrix should have non-zero entries");
    }

    #[test]
    fn dg_cdr_rhs_assembles() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let source = |x: &[f64]| { (std::f64::consts::PI * x[0]).sin() };
        let rhs = DgCdrSystem::assemble_rhs(&space, &source, 3);
        let norm: f64 = rhs.iter().map(|x| x.abs()).sum();
        assert!(norm > 0.0, "CDR RHS should have non-zero entries");
    }
}
