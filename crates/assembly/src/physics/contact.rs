//! General-purpose 2D contact mechanics — node-to-segment penalty formulation.
//!
//! Supports frictionless normal contact between a slave surface and a master
//! surface using the penalty method. Each slave node is projected onto the
//! nearest master segment; penetration triggers a restoring penalty force.
//!
//! # Usage
//!
//! ```ignore
//! let pair = ContactPair {
//!     slave_tag: 1,   // slave surface face tag
//!     master_tag: 2,  // master surface face tag
//!     penalty: 1e6,   // normal penalty stiffness
//!     friction: 0.0,
//! };
//! let (f_contact, k_contact) = assemble_contact_2d(
//!     mesh, &u, n_dofs, &dof_map, &[pair],
//! )?;
//! ```

use fem_linalg::{CooMatrix, CsrMatrix, SolverError};
use fem_mesh::topology::MeshTopology;

/// A contact pair between a slave surface and a master surface.
#[derive(Debug, Clone)]
pub struct ContactPair {
    /// Face tag identifying the slave surface.
    pub slave_tag: i32,
    /// Face tag identifying the master surface.
    pub master_tag: i32,
    /// Normal penalty stiffness (εₙ). Larger values reduce penetration.
    pub penalty: f64,
    /// Coulomb friction coefficient (0 = frictionless).
    pub friction: f64,
}

/// Result of a contact assembly.
pub struct ContactResult {
    /// Contact forces assembled into the global RHS.
    pub forces: Vec<f64>,
    /// Contact stiffness matrix (tangent).
    pub stiffness: CsrMatrix<f64>,
    /// Number of active (penetrating) slave nodes.
    pub n_active: usize,
    /// Maximum penetration depth (positive = penetrating).
    pub max_penetration: f64,
}

/// Project a point onto a line segment (2D).
///
/// Given segment from `a` to `b`, find the closest point `p_proj` on the
/// segment and the local coordinate `ξ ∈ [0, 1]`.
fn project_point_to_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> (f64, f64, f64) {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-30 {
        return (ax, ay, 0.0); // degenerate segment
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let proj_x = ax + t * dx;
    let proj_y = ay + t * dy;
    (proj_x, proj_y, t)
}

/// Compute outward normal of a 2D segment (a→b).
/// Returns (nx, ny) — unit normal pointing to the "right" of the direction a→b.
fn segment_normal(ax: f64, ay: f64, bx: f64, by: f64) -> (f64, f64) {
    let dx = bx - ax;
    let dy = by - ay;
    let len = (dx * dx + dy * dy).sqrt().max(1e-30);
    // Right-hand normal: (dy, -dx) / len
    (dy / len, -dx / len)
}

/// Assemble penalty contact forces and tangent stiffness for 2D problems.
///
/// `mesh`: the FE mesh containing both master and slave surfaces.
/// `u`: current displacement vector (interleaved [ux0, uy0, ux1, uy1, ...]).
/// `n_dofs`: total number of displacement DOFs.
/// `space_dim`: spatial dimension (2 for 2D).
/// `pairs`: contact pair definitions.
///
/// Returns contact forces, stiffness matrix, and diagnostic info.
pub fn assemble_contact_2d<M: MeshTopology>(
    mesh: &M,
    u: &[f64],
    n_dofs: usize,
    space_dim: usize,
    pairs: &[ContactPair],
) -> Result<ContactResult, SolverError> {
    let n_nodes = mesh.n_nodes();
    let mut coo = CooMatrix::new(n_dofs, n_dofs);
    let mut forces = vec![0.0; n_dofs];
    let mut n_active = 0;
    let mut max_penetration: f64 = 0.0;

    for pair in pairs {
        let eps_n = pair.penalty;

        // Collect master segment data: for each face with master_tag,
        // store the segment end-node coordinates.
        struct Seg {
            n0: u32, n1: u32,
            ax: f64, ay: f64, bx: f64, by: f64,
            nx: f64, ny: f64, len: f64,
        }
        let mut segments: Vec<Seg> = Vec::new();
        for f in 0..mesh.n_boundary_faces() as u32 {
            let tag = mesh.face_tag(f);
            if tag != pair.master_tag { continue; }
            let nodes = mesh.face_nodes(f);
            if nodes.len() < 2 { continue; }
            let n0 = nodes[0];
            let n1 = nodes[1];
            let c0 = mesh.node_coords(n0);
            let c1 = mesh.node_coords(n1);
            let (nx, ny) = segment_normal(c0[0], c0[1], c1[0], c1[1]);
            let dx = c1[0] - c0[0];
            let dy = c1[1] - c0[1];
            let len = (dx * dx + dy * dy).sqrt().max(1e-30);
            segments.push(Seg { n0, n1, ax: c0[0], ay: c0[1], bx: c1[0], by: c1[1], nx, ny, len });
        }

        // Process slave nodes: for each face with slave_tag, consider
        // each node as a potential contact slave node.
        let mut slave_nodes_done = vec![false; n_nodes];
        for f in 0..mesh.n_boundary_faces() as u32 {
            let tag = mesh.face_tag(f);
            if tag != pair.slave_tag { continue; }
            let nodes = mesh.face_nodes(f);
            for &sn in nodes {
                if sn as usize >= n_nodes { continue; }
                if slave_nodes_done[sn as usize] { continue; }
                slave_nodes_done[sn as usize] = true;

                // Slave node current position
                let sc = mesh.node_coords(sn);
                let sdof_x = (sn as usize) * space_dim;
                let sdof_y = (sn as usize) * space_dim + 1;
                let sx = sc[0] + u.get(sdof_x).copied().unwrap_or(0.0);
                let sy = sc[1] + u.get(sdof_y).copied().unwrap_or(0.0);

                // Find nearest master segment
                let mut min_dist = f64::MAX;
                let mut best_seg: Option<&Seg> = None;
                let mut best_xi = 0.0;
                let mut best_px = 0.0;
                let mut best_py = 0.0;

                for seg in &segments {
                    let (px, py, xi) = project_point_to_segment(
                        sx, sy, seg.ax, seg.ay, seg.bx, seg.by,
                    );
                    let dx = sx - px;
                    let dy = sy - py;
                    let dist = dx * dx + dy * dy;
                    if dist < min_dist {
                        min_dist = dist;
                        best_seg = Some(seg);
                        best_xi = xi;
                        best_px = px;
                        best_py = py;
                    }
                }

                if let Some(seg) = best_seg {
                    // Gap: penetration = (s - p) · n_outward
                    // For contact, we want gap = (s - p) · (-n) when s is inside the master body
                    // The normal points outward from master. Since the slave is on the opposite
                    // side, penetration occurs when (s - p) · n < 0
                    let gx = sx - best_px;
                    let gy = sy - best_py;
                    let gap = gx * seg.nx + gy * seg.ny;

                    if gap < 0.0 {
                        // Penetration detected: apply penalty force
                        n_active += 1;
                        let pen = -gap; // positive penetration depth
                        max_penetration = max_penetration.max(pen);

                        // Penalty force magnitude: F = εₙ · gap (gap is negative)
                        // Force direction: along master normal (outward)
                        let fn_val = -eps_n * gap; // positive = push slave away

                        // Slave node DOFs
                        let mdof_x0 = (seg.n0 as usize) * space_dim;
                        let mdof_y0 = mdof_x0 + 1;
                        let mdof_x1 = (seg.n1 as usize) * space_dim;
                        let mdof_y1 = mdof_x1 + 1;

                        // Force on slave: fn_val * n (in positive normal direction)
                        forces[sdof_x] += fn_val * seg.nx;
                        forces[sdof_y] += fn_val * seg.ny;

                        // Force on master nodes (equal and opposite):
                        // distributed to segment nodes via shape functions N1 = 1-ξ, N2 = ξ
                        let n1 = 1.0 - best_xi;
                        let n2 = best_xi;
                        forces[mdof_x0] -= fn_val * seg.nx * n1;
                        forces[mdof_y0] -= fn_val * seg.ny * n1;
                        forces[mdof_x1] -= fn_val * seg.nx * n2;
                        forces[mdof_y1] -= fn_val * seg.ny * n2;

                        // Tangent stiffness (symmetric contribution)
                        // K_contact = εₙ · n · nᵀ (simplified, neglecting curvature)
                        let k_norm = eps_n;
                        for &(ri, rn) in &[(sdof_x, 1.0), (sdof_y, 1.0),
                                           (mdof_x0, -n1), (mdof_y0, -n1),
                                           (mdof_x1, -n2), (mdof_y1, -n2)] {
                            for &(ci, cn) in &[(sdof_x, 1.0), (sdof_y, 1.0),
                                               (mdof_x0, -n1), (mdof_y0, -n1),
                                               (mdof_x1, -n2), (mdof_y1, -n2)] {
                                let val = k_norm * seg.nx * seg.nx * rn * cn
                                        + k_norm * seg.ny * seg.ny * rn * cn;
                                if val.abs() > 1e-20 {
                                    coo.add(ri, ci, val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let stiffness = coo.into_csr();
    Ok(ContactResult { forces, stiffness, n_active, max_penetration })
}
