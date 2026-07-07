//! Discrete operators on the **reed** integration path.
//!
//! Scalar **H¹** mass and Poisson matrices use [`crate::assembler::Assembler`] with
//! [`fem_space::H1Space`] and the same [`crate::standard`] integrators as the rest of
//! **fem-rs** (not `reed_cpu::SimplexBasis`), so `FemCeed::{apply_mass_2d, apply_poisson_2d}`
//! / `apply_mass_3d` / `apply_poisson_3d` stay numerically aligned with
//! [`crate::assembler::Assembler::assemble_bilinear`].
//!
//! The ND2→RT2 curl operator delegates to [`crate::DiscreteLinearOperator`] /
//! [`crate::vector_assembler::VectorAssembler`].
//!
//! Legacy quadrature **hints** → reference simplex rule **orders**: [`crate::h1_quad_order_hint`]
//! (`h1_tri_quad_order` / `h1_tet_quad_order` are re-exported below for the `reed::fem_discrete` path).

pub use crate::vector_assembler::TRI_ND2_RT2_MIXED_QUAD_ORDER;

use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::{HCurlSpace, HDivSpace, H1Space};

use crate::assembler::Assembler;
use crate::discrete_op::{DiscreteLinearOperator, DiscreteOpError};
use crate::standard::{DiffusionIntegrator, MassIntegrator};

pub use crate::h1_quad_order_hint::{h1_tet_quad_order, h1_tri_quad_order};

/// Global H¹ mass matrix on a 2D triangular mesh (`ρ = 1`).
pub fn assemble_mass_h1_2d(mesh: &Mesh<2>, poly: u8, quad_order: u8) -> CsrMatrix<f64> {
    let space = H1Space::new(mesh.clone(), poly);
    Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad_order)
}

/// Global H¹ stiffness (Poisson) matrix on a 2D triangular mesh (`κ = 1`).
pub fn assemble_poisson_h1_2d(mesh: &Mesh<2>, poly: u8, quad_order: u8) -> CsrMatrix<f64> {
    let space = H1Space::new(mesh.clone(), poly);
    Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        quad_order,
    )
}

/// Global H¹ mass matrix on a 3D tetrahedral mesh (`ρ = 1`).
pub fn assemble_mass_h1_3d(mesh: &Mesh<3>, poly: u8, quad_order: u8) -> CsrMatrix<f64> {
    let space = H1Space::new(mesh.clone(), poly);
    Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad_order)
}

/// Global H¹ stiffness (Poisson) matrix on a 3D tetrahedral mesh (`κ = 1`).
pub fn assemble_poisson_h1_3d(mesh: &Mesh<3>, poly: u8, quad_order: u8) -> CsrMatrix<f64> {
    let space = H1Space::new(mesh.clone(), poly);
    Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        quad_order,
    )
}

/// Assemble the 2D ND2 → RT2 curl operator `C` (same matrix as [`DiscreteLinearOperator::curl_2d_hdiv`]).
pub fn assemble_curl_hdiv_pairing_2d_nd2_rt2<M: MeshTopology>(
    hcurl_space: &HCurlSpace<M>,
    hdiv_space: &HDivSpace<M>,
) -> Result<CsrMatrix<f64>, DiscreteOpError> {
    DiscreteLinearOperator::curl_2d_hdiv(hcurl_space, hdiv_space)
}

#[cfg(all(test, feature = "reed"))]
mod fem_ceed_kernel_alignment {
    use super::*;
    use crate::FemCeed;
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn mass_poisson_csr_matches_fem_ceed_p1_p2() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        for (poly, q_hint) in [(1usize, 3usize), (2usize, 7usize)] {
            let quad = h1_tri_quad_order(poly, q_hint);
            let space = H1Space::new(mesh.clone(), poly as u8);
            let n = space.n_dofs();
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos()).collect();

            let m_fd = assemble_mass_h1_2d(&mesh, poly as u8, quad);
            let m_fc = FemCeed::new()
                .assemble_mass_2d_csr(&mesh, poly, q_hint)
                .expect("FemCeed mass CSR");
            let mut y_fd = vec![0.0_f64; n];
            let mut y_fc = vec![0.0_f64; n];
            m_fd.spmv(&x, &mut y_fd);
            m_fc.spmv(&x, &mut y_fc);
            assert!(
                max_abs_diff(&y_fd, &y_fc) < 1e-14,
                "mass fem_discrete vs FemCeed poly={poly}"
            );

            let k_fd = assemble_poisson_h1_2d(&mesh, poly as u8, quad);
            let k_fc = FemCeed::new()
                .assemble_poisson_2d_csr(&mesh, poly, q_hint)
                .expect("FemCeed poisson CSR");
            y_fd.fill(0.0);
            y_fc.fill(0.0);
            k_fd.spmv(&x, &mut y_fd);
            k_fc.spmv(&x, &mut y_fc);
            assert!(
                max_abs_diff(&y_fd, &y_fc) < 1e-13,
                "poisson fem_discrete vs FemCeed poly={poly}"
            );
        }
    }

    #[test]
    fn mass_poisson_csr_matches_fem_ceed_3d_p1_p2() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        for (poly, q_hint) in [(1usize, 3usize), (2usize, 7usize)] {
            let quad = h1_tet_quad_order(poly, q_hint);
            let space = H1Space::new(mesh.clone(), poly as u8);
            let n = space.n_dofs();
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();

            let m_fd = assemble_mass_h1_3d(&mesh, poly as u8, quad);
            let m_fc = FemCeed::new()
                .assemble_mass_3d_csr(&mesh, poly, q_hint)
                .expect("FemCeed 3d mass CSR");
            let mut y_fd = vec![0.0_f64; n];
            let mut y_fc = vec![0.0_f64; n];
            m_fd.spmv(&x, &mut y_fd);
            m_fc.spmv(&x, &mut y_fc);
            assert!(
                max_abs_diff(&y_fd, &y_fc) < 1e-13,
                "3d mass fem_discrete vs FemCeed poly={poly}"
            );

            let k_fd = assemble_poisson_h1_3d(&mesh, poly as u8, quad);
            let k_fc = FemCeed::new()
                .assemble_poisson_3d_csr(&mesh, poly, q_hint)
                .expect("FemCeed 3d poisson CSR");
            y_fd.fill(0.0);
            y_fc.fill(0.0);
            k_fd.spmv(&x, &mut y_fd);
            k_fc.spmv(&x, &mut y_fc);
            assert!(
                max_abs_diff(&y_fd, &y_fc) < 1e-12,
                "3d poisson fem_discrete vs FemCeed poly={poly}"
            );
        }
    }
}
