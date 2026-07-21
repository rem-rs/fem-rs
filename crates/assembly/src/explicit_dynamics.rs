//! Explicit dynamics driver for Abaqus/Explicit-style simulations.
//!
//! Combines central difference time integration ([`CentralDifferenceExplicit`])
//! with lumped mass, internal forces, and penalty contact forces.
//!
//! # Typical usage
//!
//! ```rust,ignore
//! use fem_assembly::explicit_dynamics::{ExplicitDynamicsConfig, explicit_step_with_contact_2d};
//! use fem_solver::ode::structural::ExplicitState;
//!
//! // Pre-assemble lumped mass and stiffness
//! let mass_lumped = ...;   // diagonal from LumpedMassOperator
//! let stiff = ...;         // assembled stiffness matrix
//!
//! let cfg = ExplicitDynamicsConfig {
//!     dt: 1e-6,
//!     gamma: 0.5,
//!     bc_dofs: vec![],
//! };
//! let mut u = vec![0.0; n_dofs];
//! let mut state = ExplicitState::new(n_dofs);
//!
//! for step in 0..n_steps {
//!     explicit_step_with_contact_2d(
//!         &mass_lumped, &stiff, &cfg,
//!         &mut u, &mut state,
//!         &f_ext,
//!         slave_mesh, &[1], slave_dofs,
//!         master_mesh, &[1], master_dof_offset,
//!         &contact_cfg,
//!     );
//! }
//! ```

use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::ode::structural::{CentralDifferenceExplicit, ExplicitState};

use crate::contact::{
    N2SContactConfig, build_segment_index, find_closest_segment,
};
use crate::contact::contact_n2s_3d::{Bvh, MasterTriangle, N2SContactConfig3D};

/// Configuration for explicit dynamics.
#[derive(Debug, Clone)]
pub struct ExplicitDynamicsConfig {
    /// Time step size.
    pub dt: f64,
    /// Central difference γ parameter (default 0.5 = 2nd order).
    pub gamma: f64,
    /// Dirichlet DOFs to enforce.
    pub bc_dofs: Vec<u32>,
    /// External force vector (constant; update each step if time-varying).
    pub f_ext: Vec<f64>,
}

impl Default for ExplicitDynamicsConfig {
    fn default() -> Self {
        Self {
            dt: 1e-6,
            gamma: 0.5,
            bc_dofs: vec![],
            f_ext: vec![],
        }
    }
}

impl ExplicitDynamicsConfig {
    /// Create a new config with the given time step and DOF count.
    pub fn new(dt: f64, n_dofs: usize) -> Self {
        Self {
            dt,
            gamma: 0.5,
            bc_dofs: vec![],
            f_ext: vec![0.0; n_dofs],
        }
    }

    /// Set Dirichlet BC DOFs.
    pub fn with_bc(mut self, bc_dofs: Vec<u32>) -> Self {
        self.bc_dofs = bc_dofs;
        self
    }

    /// Set external force vector.
    pub fn with_f_ext(mut self, f_ext: Vec<f64>) -> Self {
        self.f_ext = f_ext;
        self
    }
}

// ─── Force-only contact assembly (no stiffness matrix) ──────────────────────

/// Compute 2D N2S contact force vector only (no stiffness matrix).
///
/// For explicit dynamics, only the force vector is needed — the stiffness
/// matrix is not used since no implicit solve is performed.
///
/// This is a lighter version of [`assemble_n2s_contact_2d`] that skips
/// stiffness assembly.
#[allow(clippy::too_many_arguments)]
pub fn assemble_n2s_contact_force_2d<M: MeshTopology>(
    slave_mesh: &M,
    slave_contact_tags: &[i32],
    slave_dofs: &[usize],
    master_mesh: &M,
    master_contact_tags: &[i32],
    master_dof_offset: usize,
    u: &[f64],
    cfg: &N2SContactConfig,
    n_total_dofs: usize,
) -> Vec<f64> {
    let mut f_contact = vec![0.0_f64; n_total_dofs];

    // Build master segment index
    let segments = crate::contact::build_segment_index(master_mesh, master_contact_tags);
    if segments.is_empty() {
        return f_contact;
    }

    // Iterate over slave boundary faces
    for f in slave_mesh.face_iter() {
        if !slave_contact_tags.contains(&slave_mesh.face_tag(f)) {
            continue;
        }
        let nodes = slave_mesh.face_nodes(f);
        if nodes.len() < 2 {
            continue;
        }

        let n0_coords = slave_mesh.node_coords(nodes[0]);
        let n1_coords = slave_mesh.node_coords(nodes[1]);

        // Add current displacement
        let disp0 = [
            u.get(nodes[0] as usize * 2).copied().unwrap_or(0.0),
            u.get(nodes[0] as usize * 2 + 1).copied().unwrap_or(0.0),
        ];
        let disp1 = [
            u.get(nodes[1] as usize * 2).copied().unwrap_or(0.0),
            u.get(nodes[1] as usize * 2 + 1).copied().unwrap_or(0.0),
        ];

        let x0 = [n0_coords[0] + disp0[0], n0_coords[1] + disp0[1]];
        let x1 = [n1_coords[0] + disp1[0], n1_coords[1] + disp1[1]];
        let xm = [(x0[0] + x1[0]) * 0.5, (x0[1] + x1[1]) * 0.5];

        if let Some((_seg_idx, closest, gap, _xi)) =
            crate::contact::find_closest_segment(&xm, &segments, cfg.search_dist)
        {
            if gap >= 0.0 {
                continue;
            }

            let dx = xm[0] - closest[0];
            let dy = xm[1] - closest[1];
            let dist = (dx * dx + dy * dy).sqrt().max(1e-30);
            let nx = dx / dist;
            let ny = dy / dist;

            let fn_val = -cfg.eps_n * gap;
            let fx = fn_val * nx;
            let fy = fn_val * ny;

            let n_shape = 0.5; // midpoint shape value
            let n0x = nodes[0] as usize * 2;
            let n0y = nodes[0] as usize * 2 + 1;
            let n1x = nodes[1] as usize * 2;
            let n1y = nodes[1] as usize * 2 + 1;

            if n0x < n_total_dofs {
                f_contact[n0x] += n_shape * fx;
            }
            if n0y < n_total_dofs {
                f_contact[n0y] += n_shape * fy;
            }
            if n1x < n_total_dofs {
                f_contact[n1x] += n_shape * fx;
            }
            if n1y < n_total_dofs {
                f_contact[n1y] += n_shape * fy;
            }

            // Distribute reaction to master nodes (action = -reaction)
            if let Some((seg_a, seg_b)) = find_master_segment_nodes(master_mesh, _seg_idx) {
                let xi = _xi;
                let ma = 1.0 - xi;
                let mb = xi;
                let ma_off = master_dof_offset;
                let ma_x = ma_off + seg_a * 2;
                let ma_y = ma_off + seg_a * 2 + 1;
                let mb_x = ma_off + seg_b * 2;
                let mb_y = ma_off + seg_b * 2 + 1;

                if ma_x < n_total_dofs {
                    f_contact[ma_x] -= ma * fx;
                }
                if ma_y < n_total_dofs {
                    f_contact[ma_y] -= ma * fy;
                }
                if mb_x < n_total_dofs {
                    f_contact[mb_x] -= mb * fx;
                }
                if mb_y < n_total_dofs {
                    f_contact[mb_y] -= mb * fy;
                }
            }
        }
    }

    f_contact
}

/// Find the node indices of a master segment by its index.
fn find_master_segment_nodes<M: MeshTopology>(
    mesh: &M,
    segment_idx: usize,
) -> Option<(usize, usize)> {
    let mut seg_count = 0usize;
    for f in mesh.face_iter() {
        let nodes = mesh.face_nodes(f);
        if nodes.len() >= 2 {
            if seg_count == segment_idx {
                return Some((nodes[0] as usize, nodes[1] as usize));
            }
            seg_count += 1;
        }
    }
    None
}

// ─── High-level step functions ──────────────────────────────────────────────

/// Perform one explicit dynamics step with 2D N2S contact.
///
/// This is a convenience wrapper around [`CentralDifferenceExplicit::step`]
/// that assembles internal and contact forces internally.
///
/// # Important
/// The mass matrix must be **lumped** (diagonal) for explicit dynamics to
/// remain efficient. Use [`LumpedMassOperator`](crate::partial::LumpedMassOperator)
/// to pre-assemble it.
#[allow(clippy::too_many_arguments)]
pub fn explicit_step_with_contact_2d<M: MeshTopology>(
    mass_lumped: &[f64],
    stiff: &CsrMatrix<f64>,
    cfg: &ExplicitDynamicsConfig,
    u: &mut [f64],
    state: &mut ExplicitState,
    // Contact parameters (optional: pass None to skip contact)
    contact_params: Option<(
        &M,           // slave_mesh
        &[i32],       // slave_contact_tags
        &[usize],     // slave_dofs
        &M,           // master_mesh
        &[i32],       // master_contact_tags
        usize,        // master_dof_offset
        &N2SContactConfig, // contact config
    )>,
) {
    let cd = CentralDifferenceExplicit { gamma: cfg.gamma };
    let n = u.len();
    let f_ext = &cfg.f_ext;

    cd.step(mass_lumped, cfg.dt, u, state, &cfg.bc_dofs, |u_pred| {
        // Internal force: f_int = K·u_pred  (linear; for nonlinear, use a callback)
        let mut f_int = vec![0.0; n];
        stiff.spmv(u_pred, &mut f_int);

        // Start with f_total = f_ext - f_int
        let mut f_total: Vec<f64> = (0..n).map(|i| f_ext[i] - f_int[i]).collect();

        // Add contact forces at predicted configuration
        if let Some((slv_mesh, slv_tags, slv_dofs, mas_mesh, mas_tags, mas_offset, contact_cfg)) =
            contact_params
        {
            // Compute total DOFs including master body
            let n_total = n.max(mas_offset
                + mas_mesh.n_nodes() as usize * 2);

            let f_contact = assemble_n2s_contact_force_2d(
                slv_mesh,
                slv_tags,
                slv_dofs,
                mas_mesh,
                mas_tags,
                mas_offset,
                u_pred,
                contact_cfg,
                n_total,
            );

            // Add contact force to total (only slave part; master part is already
            // accounted if `master_dof_offset < n_total`)
            for i in 0..n_total.min(n) {
                f_total[i] += f_contact[i];
            }
        }

        f_total
    });
}

/// Perform one explicit dynamics step with 3D N2S contact.
///
/// This is a convenience wrapper around [`CentralDifferenceExplicit::step`]
/// that assembles internal, external and 3D penalty contact forces.
///
/// The master surface is represented as pre-built [`MasterTriangle`]s and a
/// [`Bvh`] for collision detection, which should be constructed once before
/// the time-stepping loop.
///
/// # Precomputation
/// ```rust,ignore
/// use fem_assembly::contact::contact_n2s_3d::{build_master_triangles, MasterTriangle, Bvh};
///
/// let master_tris = build_master_triangles(&master_mesh, &[1]);
/// let bvh = Bvh::new(&master_tris);
/// ```
///
/// # Important
/// The mass matrix must be **lumped** (diagonal). The stiffness matrix is
/// used only for the linear internal force `K·u_pred`; for nonlinear problems
/// pass a callback instead (override the step function).
#[allow(clippy::too_many_arguments)]
pub fn explicit_step_with_contact_3d(
    mass_lumped: &[f64],
    stiff: &CsrMatrix<f64>,
    cfg: &ExplicitDynamicsConfig,
    u: &mut [f64],
    state: &mut ExplicitState,
    contact_params: Option<(
        &[[f64; 3]],                // slave_undeformed_coords (n_slave_nodes × 3)
        &[usize],                   // slave_dof_indices (mapping slave_node→global DOF)
        usize,                      // slave_dof_offset
        &[MasterTriangle],          // master_triangles (pre-built, see build_master_triangles)
        &Bvh,                       // bvh (pre-built from master_triangles)
        &N2SContactConfig3D,        // contact config
    )>,
) {
    let cd = CentralDifferenceExplicit { gamma: cfg.gamma };
    let n = u.len();
    let f_ext = &cfg.f_ext;

    cd.step(mass_lumped, cfg.dt, u, state, &cfg.bc_dofs, |u_pred| {
        // Internal force: f_int = K·u_pred
        let mut f_int = vec![0.0; n];
        stiff.spmv(u_pred, &mut f_int);

        let mut f_total: Vec<f64> = (0..n).map(|i| f_ext[i] - f_int[i]).collect();

        if let Some((slv_coords_0, slv_dof_indices, slv_offset, mas_tris, bvh, n2s_cfg)) =
            contact_params
        {
            // Compute deformed slave node coordinates: x = x₀ + u
            let n_slave = slv_coords_0.len();
            let mut deformed_coords: Vec<[f64; 3]> = Vec::with_capacity(n_slave);

            for (i, &base) in slv_dof_indices.iter().enumerate() {
                let ux = *u_pred.get(base).unwrap_or(&0.0);
                let uy = *u_pred.get(base + 1).unwrap_or(&0.0);
                let uz = *u_pred.get(base + 2).unwrap_or(&0.0);
                deformed_coords.push([
                    slv_coords_0[i][0] + ux,
                    slv_coords_0[i][1] + uy,
                    slv_coords_0[i][2] + uz,
                ]);
            }

            // Compute total DOFs for force vector allocation
            let n_total = n.max(slv_offset + n_slave * 3);

            let f_contact = crate::contact::contact_n2s_3d::assemble_n2s_contact_3d_force_only(
                &deformed_coords,
                slv_offset,
                Some(u_pred),
                mas_tris,
                bvh,
                n2s_cfg,
                n_total,
            );

            for i in 0..n_total.min(n) {
                f_total[i] += f_contact[i];
            }
        }

        f_total
    });
}

/// Obtain an estimate of the critical time step for a 1D wave equation:
///
/// ```text
/// Δt_crit = L_min / c
/// ```
///
/// where `L_min` is the smallest element length and `c = sqrt(E/ρ)` is the
/// wave speed.  Use a safety factor `α ∈ [0.8, 0.98]`.
///
/// For general problems, use the element's CFL condition:
/// ```text
/// Δt_crit = min_elem(L_e / c)
/// ```
/// where `L_e` is the characteristic element length.
pub fn estimate_critical_dt(E: f64, rho: f64, min_elem_size: f64, safety: f64) -> f64 {
    let c = (E / rho).sqrt();
    safety * min_elem_size / c
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_linalg::CooMatrix;
    use fem_solver::ode::structural::ExplicitState;

    #[test]
    fn explicit_step_single_dof() {
        // SDOF: m=1, k=π², free vibration
        let mass_lumped = vec![1.0_f64];
        let mut stiff_coo = CooMatrix::<f64>::new(1, 1);
        stiff_coo.add(0, 0, std::f64::consts::PI * std::f64::consts::PI);
        let stiff = stiff_coo.into_csr();

        let cfg = ExplicitDynamicsConfig::new(0.001, 1);
        let mut u = vec![1.0_f64];
        let mut state = ExplicitState::new(1);
        state.acc[0] = -std::f64::consts::PI * std::f64::consts::PI;

        for _ in 0..100 {
            explicit_step_with_contact_2d::<Mesh<2>>(
                &mass_lumped, &stiff, &cfg,
                &mut u, &mut state, None,
            );
        }

        // After 100 steps × 0.001 = 0.1s, u ≈ cos(π·0.1)
        let exact = (std::f64::consts::PI * 0.1).cos();
        let err = (u[0] - exact).abs();
        assert!(err < 0.01, "error={:.4e}", err);
    }

    #[test]
    fn estimate_cfl() {
        // Steel: E=200e9, ρ=7800, c≈5063 m/s
        let dt = estimate_critical_dt(200e9, 7800.0, 0.01, 0.9);
        let c = (200e9 / 7800.0_f64).sqrt();
        let expected = 0.9 * 0.01 / c;
        assert!((dt - expected).abs() < 1e-12);
    }
}
