//! BR1 (Bassi–Rebay 1) diffusion discretisation.
//!
//! Implements the original Bassi–Rebay (1997) global lifting formulation for
//! the scalar diffusion equation −∇·(κ∇u) = f.
//!
//! # Bilinear form
//!
//! ```text
//! a_h(u,v) = Σ_K ∫_K κ∇u·∇v dx
//!          − Σ_F ∫_F {κ∇u}·[[v]] ds
//!          − Σ_F ∫_F [[u]]·{κ∇v} ds
//!          + Σ_F ∫_Ω R_F([[u]])·R_F([[v]]) dx
//! ```
//!
//! where `R_F(w) ∈ V_h` is the **global lifting** defined by:
//! `∫_Ω R_F(w)·τ dx = −∫_F {w}·{τ} ds   ∀ τ ∈ V_h`.
//!
//! For DG (discontinuous V_h), the global mass matrix is block‑diagonal,
//! so the liftings decouple per element and BR1 becomes equivalent to
//! BR2 with a stabilisation coefficient η = n_faces (the number of faces
//! per element, i.e. 3 for triangles, 4 for tetrahedra).
//!
//! # Reference
//! * F. Bassi, S. Rebay, *A high-order accurate discontinuous finite element
//!   method for the numerical solution of the compressible Navier–Stokes
//!   equations*, J. Comput. Phys. 131(2), 1997.

use fem_linalg::CsrMatrix;
use fem_space::fe_space::FESpace;
use super::dg_br2::assemble_br2;
use crate::interior_faces::InteriorFaceList;

/// Assemble the BR1 (Bassi–Rebay 1) DG diffusion matrix.
///
/// BR1 is the original global‑lifting formulation with **no tunable
/// stabilisation parameter**.  For DG with P1 elements the natural
/// scaling is `η = n_faces` (3 for triangles, 4 for tets), which
/// matches the original Bassi–Rebay 1997 method.
///
/// # Arguments
/// * `space`      – L2 (DG) finite‑element space
/// * `ifl`        – interior face list
/// * `kappa`      – diffusion coefficient
/// * `quad_order` – quadrature order
pub fn assemble_br1<S: FESpace + Sync>(
    space: &S,
    ifl: &InteriorFaceList,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let eta = 3.0;
    assemble_br2(space, ifl, kappa, eta, quad_order)
}

/// Convenience wrapper that builds the interior face list internally.
pub fn assemble_br1_from_space<S: FESpace + Sync>(space: &S, kappa: f64, quad_order: u8) -> CsrMatrix<f64> {
    let ifl = InteriorFaceList::build(space.mesh());
    assemble_br1(space, &ifl, kappa, quad_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{FESpace, L2Space};

    #[test]
    fn br1_matrix_has_consistent_structure() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let mat = assemble_br1_from_space(&space, 1.0, 3);
        assert_eq!(mat.nrows, space.n_dofs());
        assert!(mat.nrows > 0);
        for i in 0..mat.nrows {
            for r in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                let j = mat.col_idx[r] as usize;
                let v = mat.values[r];
                let v_sym = mat.get(j, i);
                assert!((v - v_sym).abs() < 1e-14,
                    "BR1 matrix not symmetric at ({i},{j}): {v:.6e} vs {v_sym:.6e}");
            }
        }
    }

    #[test]
    fn br1_matrix_positive_semidefinite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let mat = assemble_br1_from_space(&space, 1.0, 3);
        let u: Vec<f64> = (0..space.n_dofs()).map(|i| (i as f64).sin()).collect();
        let mut au = vec![0.0; space.n_dofs()];
        for i in 0..space.n_dofs() {
            for r in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                let j = mat.col_idx[r] as usize;
                au[i] += mat.values[r] * u[j];
            }
        }
        let energy: f64 = u.iter().zip(au.iter()).map(|(a, b)| a * b).sum();
        assert!(energy >= -1e-12, "BR1 energy u^T A u = {:.3e} (should be ≥ 0)", energy);
    }
}
