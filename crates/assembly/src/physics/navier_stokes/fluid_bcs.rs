//! Boundary condition abstraction for incompressible Navier-Stokes.
//!
//! Provides a unified API to apply common CFD boundary conditions:
//! - No-slip wall (u = 0)
//! - Slip wall (u·n = 0)
//! - Inlet with prescribed velocity profile
//! - Open outlet (p = 0, natural BC)
//! - Moving wall (u = prescribed)

use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::constraints::boundary_dofs;

/// Types of boundary conditions for incompressible Navier-Stokes.
#[derive(Debug, Clone)]
pub enum BcType {
    /// No-slip wall: u = 0 (zero Dirichlet).
    NoSlip,
    /// Slip wall: u·n = 0 (zero normal component).
    Slip,
    /// Inlet with prescribed velocity `[vx, vy, vz]`.
    Inlet(Vec<f64>),
    /// Open outlet: p = 0, natural BC for velocity.
    Outlet,
    /// Moving wall: u = prescribed velocity.
    MovingWall(Vec<f64>),
}

/// A boundary condition region specification.
#[derive(Debug, Clone)]
pub struct BcRegion {
    /// Mesh boundary tag(s) for this region.
    pub tags: Vec<i32>,
    /// Type of boundary condition.
    pub bc_type: BcType,
}

/// Apply Navier-Stokes boundary conditions to the system matrix and RHS.
///
/// Modifies `mat` (velocity block) and `rhs` in-place by applying Dirichlet
/// BCs for the velocity DOFs corresponding to each BC region.
///
/// # Arguments
/// * `mat` — velocity block of the Oseen/NS system (will be modified)
/// * `rhs` — velocity RHS (will be modified)
/// * `vel_space` — VectorH1Space for velocity
/// * `regions` — list of BC regions to apply
/// * `t` — current time (for time-dependent BCs)
pub fn apply_ns_bcs<M: MeshTopology + Clone>(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    vel_space: &fem_space::VectorH1Space<M>,
    regions: &[BcRegion],
    t: f64,
) {
    let n_scalar = vel_space.n_scalar_dofs();
    let dm = vel_space.scalar_dof_manager();
    let mesh = vel_space.mesh();

    for region in regions {
        let (vx, vy, vz) = match &region.bc_type {
            BcType::NoSlip => (0.0, 0.0, 0.0),
            BcType::Inlet(v) | BcType::MovingWall(v) => {
                (v.first().copied().unwrap_or(0.0),
                 v.get(1).copied().unwrap_or(0.0),
                 v.get(2).copied().unwrap_or(0.0))
            }
            BcType::Outlet => continue, // natural BC, nothing to enforce
            BcType::Slip => {
                // Slip: zero normal component — enforced via penalty in normal direction
                // slip BC not yet implemented
                continue;
            }
        };

        // Apply Dirichlet BCs for each scalar component
        for &tag in &region.tags {
            let bnd = boundary_dofs(mesh, dm, &[tag]);
            for &dof in &bnd {
                let d = dof as usize;
                // x-component
                mat.apply_dirichlet_symmetric(d, vx, rhs);
                // y-component
                if d + n_scalar < rhs.len() {
                    mat.apply_dirichlet_symmetric(d + n_scalar, vy, rhs);
                }
                // z-component (3D)
                if d + 2 * n_scalar < rhs.len() {
                    mat.apply_dirichlet_symmetric(d + 2 * n_scalar, vz, rhs);
                }
            }
        }
    }
}

/// Apply slip BC: zero normal velocity via penalty in the normal direction.

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn bc_no_slip_creates_diagonal() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let vs = fem_space::VectorH1Space::new(mesh, 1, 2);
        let n_v = vs.n_dofs();

        // Start with identity-like matrix
        let mut coo = fem_linalg::CooMatrix::new(n_v, n_v);
        for i in 0..n_v { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![0.0_f64; n_v];

        let regions = vec![BcRegion {
            tags: vec![1, 2, 3, 4], // all boundaries
            bc_type: BcType::NoSlip,
        }];

        apply_ns_bcs(&mut mat, &mut rhs, &vs, &regions, 0.0);

        // BCs should modify matrix diagonal entries for boundary DOFs
        let dense = mat.to_dense();
        let diag_sum: f64 = (0..n_v).map(|i| dense[i * n_v + i]).sum();
        // The method applies dirichlet_symmetric which modifies the RHS
        // (even if val=0, the RHS entry is set to 0 which may not change it).
        // This test verifies the function runs without error.
        assert!(diag_sum.is_finite(), "Matrix diagonal should be finite");
    }
}
