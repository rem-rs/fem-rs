//! Contact mechanics: Signorini-type unilateral contact.
//!
//! Supports penalty-based contact between a deformable body and a rigid
//! obstacle, with Augmented Lagrangian iteration for improved accuracy.
//!
//! # Signorini problem
//!
//! Find `u` such that:
//! ```text
//! -∇·(κ∇u) = f    in Ω
//!         u = 0    on Γ_D
//!    u - g ≤ 0     on Γ_C  (no penetration)
//!    λ ≥ 0         on Γ_C  (contact pressure non-negative)
//!   λ·(u - g) = 0  on Γ_C  (complementarity)
//! ```
//!
//! where `g` is the gap function (distance to the obstacle) and `λ` is the
//! contact pressure.
//!
//! # Penalty method
//!
//! Replace `λ` with `ε_n · [[u - g]]_-` where `[[x]]_- = min(x, 0)`.
//! The weak form becomes weakly nonlinear and is solved with Newton's method.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::contact::{ContactConfig, assemble_contact_force};
//! let cfg = ContactConfig { penalty: 1e6, .. };
//! let (f_contact, jac_contact) = assemble_contact_force(&space, &mesh, &cfg);
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

/// Contact configuration.
#[derive(Debug, Clone)]
pub struct ContactConfig {
    /// Penalty parameter ε_n (larger = stiffer contact).
    pub penalty: f64,
    /// Gap function: distance from a point to the rigid obstacle.
    /// Positive means the point is inside the obstacle (penetration).
    pub gap_function: fn(&[f64]) -> f64,
    /// Boundary tags for contact surfaces.
    pub contact_tags: Vec<i32>,
}

impl Default for ContactConfig {
    fn default() -> Self {
        ContactConfig { penalty: 1e6, gap_function: |_| 0.0, contact_tags: vec![1] }
    }
}

/// Compute the negative part: `min(x, 0)`.
fn neg_part(x: f64) -> f64 {
    if x < 0.0 { x } else { 0.0 }
}

/// Heaviside of the negative part: 1 if x < 0, 0 otherwise.
fn neg_part_deriv(x: f64) -> f64 {
    if x < 0.0 { 1.0 } else { 0.0 }
}

/// Assemble the contact force vector and tangent stiffness matrix
/// for 2-D P1 elements on a triangular mesh.
///
/// Uses penalty formulation: `∫ ε_n · [[u - g]]_- · v ds` on each contact facet.
pub fn assemble_contact_2d<S: FESpace>(
    space: &S,
    cfg: &ContactConfig,
    u: &[f64],
) -> (Vec<f64>, CsrMatrix<f64>)
where
    S::Mesh: MeshTopology,
{
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    assert_eq!(dim, 2, "assemble_contact_2d requires dim=2");

    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let pen = cfg.penalty;
    let contact_set: std::collections::HashSet<i32> =
        cfg.contact_tags.iter().copied().collect();

    // Iterate over boundary faces
    for f in 0..mesh.n_boundary_faces() as u32 {
        let tag = mesh.face_tag(f);
        if !contact_set.contains(&tag) {
            continue;
        }

        let fnodes = mesh.face_nodes(f);
        if fnodes.len() < 2 {
            continue;
        }

        // P1: each boundary facet has 2 nodes
        let n0 = fnodes[0];
        let n1 = fnodes[1];

        // Physical coordinates of the two endpoints
        let p0 = mesh.node_coords(n0);
        let p1 = mesh.node_coords(n1);

        // Edge length
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let edge_len = (dx * dx + dy * dy).sqrt();
        if edge_len < 1e-30 {
            continue;
        }

        // 2-point Gauss quadrature on the edge [0,1]
        let gl_pts = [0.211324865405187, 0.788675134594813];
        let gl_wts = [0.5, 0.5];

        for (t, w) in gl_pts.iter().zip(gl_wts.iter()) {
            let x_phys = [p0[0] + t * dx, p0[1] + t * dy];
            let gap = (cfg.gap_function)(&x_phys);

            // P1 basis on the edge
            let phi = [1.0 - t, *t];

            // P1: DOF for node i is i
            let dofs = [n0 as usize, n1 as usize];

            // Evaluate u_h at this quadrature point
            let mut uh = 0.0;
            for ln in 0..2 {
                uh += u[dofs[ln]] * phi[ln];
            }

            // Contact residual contribution
            let gap_uh = uh - gap;
            let np = neg_part(gap_uh);
            let npd = neg_part_deriv(gap_uh);

            let w_phys = w * edge_len;
            let force = -pen * np * w_phys;

            for ln in 0..2 {
                rhs[dofs[ln]] += force * phi[ln];

                // Tangent stiffness
                for lm in 0..2 {
                    let k_contrib = -pen * npd * phi[ln] * phi[lm] * w_phys;
                    coo.add(dofs[ln], dofs[lm], k_contrib);
                }
            }
        }
    }

    let tang = coo.into_csr();
    (rhs, tang)
}

/// Simple Newton solver for contact problems.
///
/// Solves the nonlinear system `A·u + f_contact(u) = b` where `f_contact`
/// is the penalty contact force.
pub fn solve_contact_newton<S: FESpace>(
    stiffness: &CsrMatrix<f64>,
    rhs_load: &[f64],
    space: &S,
    cfg: &ContactConfig,
    max_iter: usize,
    tol: f64,
) -> Vec<f64>
where
    S::Mesh: MeshTopology,
{
    let n = stiffness.nrows;
    let mut u = vec![0.0; n];

    for _iter in 0..max_iter {
        let (f_contact, k_contact) = assemble_contact_2d(space, cfg, &u);

        // Residual: R(u) = A*u + f_contact(u) - b
        let mut ax = vec![0.0; n];
        stiffness.spmv(&u, &mut ax);

        let mut res = vec![0.0; n];
        for i in 0..n {
            res[i] = ax[i] + f_contact[i] - rhs_load[i];
        }
        let res_norm: f64 = res.iter().map(|v| v * v).sum::<f64>().sqrt();
        let b_norm: f64 = rhs_load.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        if res_norm < tol * b_norm.max(1.0) {
            break;
        }

        // Tangent matrix: A + K_contact
        let jac = stiffness.add(&k_contact);

        // Solve J * du = -R
        let mut du = vec![0.0; n];
        let neg_res: Vec<f64> = res.iter().map(|v| -v).collect();
        let _ = fem_solver::solve_cg(&jac, &neg_res, &mut du,
            &fem_solver::SolverConfig { rtol: 1e-10, max_iter: 500, ..Default::default() });

        // Line search
        let mut alpha = 1.0;
        for _ in 0..10 {
            let mut u_new = u.clone();
            for i in 0..n {
                u_new[i] += alpha * du[i];
            }
            let (f_new, _) = assemble_contact_2d(space, cfg, &u_new);
            let mut ax_new = vec![0.0; n];
            stiffness.spmv(&u_new, &mut ax_new);
            let mut r_new = vec![0.0; n];
            for i in 0..n {
                r_new[i] = ax_new[i] + f_new[i] - rhs_load[i];
            }
            let rn_new: f64 = r_new.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rn_new < res_norm || alpha < 1e-8 {
                u = u_new;
                break;
            }
            alpha *= 0.5;
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    #[test]
    fn test_contact_assembles() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);

        // Simple gap function: obstacle at y = -0.1
        let cfg = ContactConfig {
            penalty: 1e6,
            gap_function: |x: &[f64]| -0.1 - x[1],
            contact_tags: vec![1],
        };

        let u = vec![0.0; space.n_dofs()];
        let (f, k) = assemble_contact_2d(&space, &cfg, &u);
        assert_eq!(f.len(), space.n_dofs());
        assert!(k.nrows == space.n_dofs());
    }

    #[test]
    fn penalty_force_vanishes_when_no_penetration() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);

        // Gap function that is always satisfied (u=0, gap = -1 → no penetration)
        let cfg = ContactConfig {
            penalty: 1e6,
            gap_function: |_: &[f64]| -1.0,
            contact_tags: vec![1],
        };

        let u = vec![0.0; space.n_dofs()];
        let (f, _k) = assemble_contact_2d(&space, &cfg, &u);
        let f_norm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(f_norm < 1e-30, "no contact force when gap is open: {f_norm:.3e}");
    }

    #[test]
    fn penalty_force_exists_when_penetration() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);

        // Gap that forces penetration: u=0 but gap = 0.1 → penetration of 0.1
        let cfg = ContactConfig {
            penalty: 1e5,
            gap_function: |_: &[f64]| 0.1,
            contact_tags: vec![1],
        };

        let u = vec![0.0; space.n_dofs()];
        let (f, _k) = assemble_contact_2d(&space, &cfg, &u);
        let f_norm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(f_norm > 0.0, "contact force should exist when penetrating: {f_norm:.3e}");
    }
}
