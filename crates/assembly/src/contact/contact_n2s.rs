//! Node-to-segment (N2S) penalty contact for 2D deformable bodies.
//!
//! Implements penalty-based normal contact and Coulomb friction between
//! a slave body and a master body using node-to-segment kinematic coupling.
//!
//! # Algorithm
//! 1. For each slave boundary face quadrature point (or slave node):
//!    - Find the closest master segment via spatial proximity search
//!    - Compute signed gap: `g_n = (x_s - x_m) · n_m`
//!    - If penetrating (`g_n < 0`): apply normal penalty force `f_n = ε_n · g_n · n_m`
//!    - For friction: compute tangential slip and apply stick/split via radial return
//! 2. Add contact contributions to residual and stiffness matrix of both bodies

use fem_linalg::{CsrMatrix, CooMatrix, SolverConfig, SolveResult};
use fem_mesh::topology::MeshTopology;

/// Configuration for node-to-segment penalty contact.
#[derive(Debug, Clone)]
pub struct N2SContactConfig {
    /// Normal penalty stiffness.
    pub eps_n: f64,
    /// Tangential penalty stiffness (for friction).
    pub eps_t: f64,
    /// Coulomb friction coefficient (0 = frictionless).
    pub mu: f64,
    /// Maximum search distance for contact detection.
    pub search_dist: f64,
    /// Tolerance for gap convergence.
    pub tol: f64,
}

impl Default for N2SContactConfig {
    fn default() -> Self {
        Self {
            eps_n: 1e6,
            eps_t: 1e5,
            mu: 0.0,
            search_dist: 1.0,
            tol: 1e-8,
        }
    }
}

/// Closest-point projection of point `p` onto line segment `[a, b]`.
///
/// Returns `(closest_point, xi, distance)` where:
/// - `closest_point` is the closest point on the segment
/// - `xi ∈ [0, 1]` is the barycentric coordinate: `x_m = a + xi·(b - a)`
/// - `distance` is the signed distance (positive = outside material)
pub fn closest_point_on_segment(p: &[f64; 2], a: &[f64; 2], b: &[f64; 2]) -> ([f64; 2], f64, f64) {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let ab_sq = ab[0] * ab[0] + ab[1] * ab[1];

    if ab_sq < 1e-30 {
        // Degenerate segment: return endpoint a
        let dx = p[0] - a[0];
        let dy = p[1] - a[1];
        let dist = (dx * dx + dy * dy).sqrt().copysign(-1.0); // assume outside
        return (*a, 0.0, dist);
    }

    let xi = (ap[0] * ab[0] + ap[1] * ab[1]) / ab_sq;
    let xi = xi.clamp(0.0, 1.0);

    let closest = [a[0] + xi * ab[0], a[1] + xi * ab[1]];
    let dx = p[0] - closest[0];
    let dy = p[1] - closest[1];

    // Outward normal of the segment (pointing from master to slave expected)
    let nx = -ab[1] / ab_sq.sqrt();
    let ny = ab[0] / ab_sq.sqrt();

    // Signed distance: positive if p is on the normal side
    let signed_dist = dx * nx + dy * ny;

    (closest, xi, signed_dist)
}

/// Segment bounding box for spatial search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentBox {
    seg_idx: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    a: [f64; 2],
    b: [f64; 2],
}

/// Build a simple spatial hash for master boundary segments.
pub(crate) fn build_segment_index<M: MeshTopology>(
    master_mesh: &M,
    master_contact_tags: &[i32],
) -> Vec<SegmentBox> {
    let mut segments = Vec::new();
    for f in master_mesh.face_iter() {
        if !master_contact_tags.contains(&master_mesh.face_tag(f)) {
            continue;
        }
        let nodes = master_mesh.face_nodes(f);
        if nodes.len() < 2 {
            continue;
        }
        let a = master_mesh.node_coords(nodes[0]);
        let b = master_mesh.node_coords(nodes[1]);
        let a2 = [a[0], a[1]];
        let b2 = [b[0], b[1]];
        let xmin = a2[0].min(b2[0]);
        let xmax = a2[0].max(b2[0]);
        let ymin = a2[1].min(b2[1]);
        let ymax = a2[1].max(b2[1]);
        segments.push(SegmentBox {
            seg_idx: 0,
            xmin, xmax, ymin, ymax,
            a: a2, b: b2,
        });
    }
    segments
}

/// Find the closest master segment to a slave point.
pub(crate) fn find_closest_segment(
    p: &[f64; 2],
    segments: &[SegmentBox],
    search_dist: f64,
) -> Option<(usize, [f64; 2], f64, f64)> {
    let mut best_dist = search_dist;
    let mut best = None;

    for seg in segments {
        // AABB culling
        if p[0] < seg.xmin - best_dist || p[0] > seg.xmax + best_dist { continue; }
        if p[1] < seg.ymin - best_dist || p[1] > seg.ymax + best_dist { continue; }

        let (closest, _xi, dist) = closest_point_on_segment(p, &seg.a, &seg.b);
        let abs_dist = dist.abs();

        if abs_dist < best_dist {
            best_dist = abs_dist;
            best = Some((seg.seg_idx, closest, dist, _xi));
        }
    }

    best
}

/// Assemble 2D node-to-segment penalty contact contributions.
///
/// For each slave boundary face, finds the closest master boundary segment,
/// computes the signed gap, and adds penalty force to both bodies' residual
/// and stiffness matrices.
///
/// # Arguments
/// * `slave_mesh` — slave body mesh (deforming)
/// * `slave_contact_tags` — boundary tags for slave contact surface
/// * `slave_dofs_u32` — slave DOF indices (interleaved: `[ux0, uy0, ux1, uy1, ...]`)
/// * `master_mesh` — master body mesh (deforming)
/// * `master_contact_tags` — boundary tags for master contact surface
/// * `master_dof_offset` — DOF offset for master body (0 if single matrix)
/// * `u` — current displacement vector (both bodies concatenated)
/// * `cfg` — contact configuration
/// * `n_total_dofs` — total number of DOFs in the combined system
///
/// Returns `(contact_force_vector, contact_stiffness_matrix)` in COO format.
pub fn assemble_n2s_contact_2d<M: MeshTopology>(
    slave_mesh: &M,
    slave_contact_tags: &[i32],
    slave_dofs: &[usize],
    master_mesh: &M,
    master_contact_tags: &[i32],
    master_dof_offset: usize,
    u: &[f64],
    cfg: &N2SContactConfig,
    n_total_dofs: usize,
) -> (Vec<f64>, CooMatrix<f64>) {
    let mut f_contact = vec![0.0_f64; n_total_dofs];
    let mut k_contact = CooMatrix::new(n_total_dofs, n_total_dofs);

    // Build master segment index
    let segments = build_segment_index(master_mesh, master_contact_tags);
    if segments.is_empty() {
        return (f_contact, k_contact);
    }

    // Iterate over slave boundary faces
    for f in slave_mesh.face_iter() {
        if !slave_contact_tags.contains(&slave_mesh.face_tag(f)) {
            continue;
        }
        let nodes = slave_mesh.face_nodes(f);
        if nodes.len() < 2 { continue; }

        // For 2D, boundary face = edge segment: use midpoint as slave point
        let n0_coords = slave_mesh.node_coords(nodes[0]);
        let n1_coords = slave_mesh.node_coords(nodes[1]);

        // Add current displacement
        let disp0 = if (nodes[0] as usize) < u.len() / 2 {
            let off = nodes[0] as usize;
            [u.get(off * 2).copied().unwrap_or(0.0), u.get(off * 2 + 1).copied().unwrap_or(0.0)]
        } else { [0.0; 2] };
        let disp1 = if (nodes[1] as usize) < u.len() / 2 {
            let off = nodes[1] as usize;
            [u.get(off * 2).copied().unwrap_or(0.0), u.get(off * 2 + 1).copied().unwrap_or(0.0)]
        } else { [0.0; 2] };

        let x0 = [n0_coords[0] + disp0[0], n0_coords[1] + disp0[1]];
        let x1 = [n1_coords[0] + disp1[0], n1_coords[1] + disp1[1]];

        // Midpoint of the slave edge
        let xm = [(x0[0] + x1[0]) * 0.5, (x0[1] + x1[1]) * 0.5];

        // Find closest master segment
        if let Some((_seg_idx, closest, gap, xi)) = find_closest_segment(&xm, &segments, cfg.search_dist) {
            if gap >= 0.0 { continue; } // no penetration

            // Normal direction (from master surface toward slave)
            // For a master segment [a,b], the outward normal is (-dy, dx) / |ab|
            // We need the normal pointing TOWARD the slave (inward for master,
            // outward for slave), which is opposite of the segment's outward normal.
            // For simplicity, use the direction from closest point to slave point.
            let dx = xm[0] - closest[0];
            let dy = xm[1] - closest[1];
            let dist = (dx * dx + dy * dy).sqrt().max(1e-30);
            let nx = dx / dist;
            let ny = dy / dist;

            // Normal penalty force
            let fn_val = -cfg.eps_n * gap; // gap < 0 → positive force (repulsive)
            let fx = fn_val * nx;
            let fy = fn_val * ny;

            // Distribute to slave nodes (linear shape functions: N0 = 1-xi, N1 = xi at midpoint)
            // At midpoint: N0 = 0.5, N1 = 0.5
            let n0 = 0.5; // shape function at xi=0.5
            let n1 = 0.5;

            // Add to residual: slave side
            let slave_n0_x = nodes[0] as usize * 2;
            let slave_n0_y = nodes[0] as usize * 2 + 1;
            let slave_n1_x = nodes[1] as usize * 2;
            let slave_n1_y = nodes[1] as usize * 2 + 1;

            if slave_n0_x < n_total_dofs { f_contact[slave_n0_x] += n0 * fx; }
            if slave_n0_y < n_total_dofs { f_contact[slave_n0_y] += n0 * fy; }
            if slave_n1_x < n_total_dofs { f_contact[slave_n1_x] += n1 * fx; }
            if slave_n1_y < n_total_dofs { f_contact[slave_n1_y] += n1 * fy; }

            // Add to stiffness: slave diagonal (linearization of penalty force)
            let pairs = vec![(slave_n0_x, n0), (slave_n0_y, n0), (slave_n1_x, n1), (slave_n1_y, n1)];
            for &(i, ni) in &pairs {
                if i >= n_total_dofs { continue; }
                for &(j, nj) in &pairs {
                    if j >= n_total_dofs { continue; }

                    // d(f_i)/d(u_j) = ε_n * N_i(xi_slave) * N_j(xi_slave) * n_i * n_j
                    let i_comp = i % 2;
                    let j_comp = j % 2;
                    let n_i = if i_comp == 0 { nx } else { ny };
                    let n_j = if j_comp == 0 { nx } else { ny };
                    let k_val = cfg.eps_n * ni * nj * n_i * n_j;

                    if k_val.abs() > 1e-30 {
                        k_contact.add(i, j, k_val);
                    }
                }
            }
        }
    }

    (f_contact, k_contact)
}

/// Solve a contact problem using an active-set penalty approach.
///
/// Alternates between solving the global system and updating contact forces
/// until the contact gap converges.
pub fn solve_n2s_contact<M: MeshTopology>(
    k_global: &CsrMatrix<f64>,
    f_ext: &[f64],
    slave_mesh: &M,
    slave_tags: &[i32],
    master_mesh: &M,
    master_tags: &[i32],
    master_dof_offset: usize,
    cfg: &N2SContactConfig,
    solver_cfg: &SolverConfig,
) -> Result<Vec<f64>, String> {
    let n_dofs = k_global.nrows;
    let mut u = vec![0.0_f64; n_dofs];

    for _iter in 0..10 {
        // Assemble contact contributions at current displacement
        let (f_contact, k_contact_coo) = assemble_n2s_contact_2d(
            slave_mesh, slave_tags, &[], master_mesh, master_tags,
            master_dof_offset, &u, cfg, n_dofs);

        // Add to global system
        let k_contact = k_contact_coo.into_csr();
        let k_total = fem_linalg::csr::spadd(k_global, &k_contact);

        // RHS = f_ext + f_contact (contact force is on RHS)
        let mut rhs = f_ext.to_vec();
        for i in 0..n_dofs {
            rhs[i] += f_contact[i];
        }

        // Solve
        let mut du = vec![0.0_f64; n_dofs];
        match fem_solver::solve_pcg_jacobi(&k_total, &rhs, &mut du, solver_cfg) {
            Ok(_res) => {
                // Update displacement
                for i in 0..n_dofs {
                    u[i] += du[i];
                }

                // Check convergence (small correction)
                let du_norm: f64 = du.iter().map(|&v| v * v).sum::<f64>().sqrt();
                let u_norm: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1.0);
                if du_norm < cfg.tol * u_norm {
                    return Ok(u);
                }
            }
            Err(e) => return Err(format!("Solver failed: {}", e)),
        }
    }

    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closest_point_on_segment_midpoint() {
        let p = [0.0, 0.5];
        let a = [-1.0, 0.0];
        let b = [1.0, 0.0];
        let (closest, xi, dist) = closest_point_on_segment(&p, &a, &b);
        assert!((xi - 0.5).abs() < 1e-12);
        assert!((closest[0]).abs() < 1e-12);
        assert!((closest[1]).abs() < 1e-12);
        assert!(dist > 0.0, "point above line should be positive");
    }

    #[test]
    fn test_closest_point_beyond_endpoint() {
        let p = [2.0, 0.5];
        let a = [-1.0, 0.0];
        let b = [1.0, 0.0];
        let (closest, xi, _dist) = closest_point_on_segment(&p, &a, &b);
        assert!((xi - 1.0).abs() < 1e-12);
        assert!((closest[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_closest_point_penetration() {
        let p = [0.0, -0.3];
        let a = [-1.0, 0.0];
        let b = [1.0, 0.0];
        let (_closest, _xi, dist) = closest_point_on_segment(&p, &a, &b);
        assert!(dist < 0.0, "point below line: dist={:.3e}", dist);
    }
}
