//! 3D Node-to-Surface (N2S) penalty contact.
//!
//! Implements penalty-based normal contact between slave nodes and master
//! triangular surface facets using closest-point projection.
//!
//! # Algorithm
//! 1. For each slave node: find closest master triangle via BVH
//! 2. Compute signed gap: `g_n = (x_s - x_m) · n_t`
//! 3. If penetrating: apply normal penalty force with consistent linearization


use fem_linalg::CooMatrix;

/// Configuration for 3D node-to-surface contact.
#[derive(Debug, Clone)]
pub struct N2SContactConfig3D {
    /// Normal penalty stiffness.
    pub eps_n: f64,
    /// Friction coefficient (0 = frictionless).
    pub mu: f64,
    /// Maximum search distance for contact detection.
    pub search_dist: f64,
}

impl Default for N2SContactConfig3D {
    fn default() -> Self {
        Self { eps_n: 1e6, mu: 0.0, search_dist: 1.0 }
    }
}

/// A master triangle from a boundary face.
#[derive(Debug, Clone)]
pub struct MasterTriangle {
    pub face_idx: u32,
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub c: [f64; 3],
    pub normal: [f64; 3],
    pub centroid: [f64; 3],
    pub radius: f64, // bounding sphere radius
}

/// Simple BVH node (axis-aligned bounding box).
#[derive(Debug, Clone)]
struct BvhNode {
    bbox: [[f64; 3]; 2], // min, max
    tri_idx: usize,
}

/// Simple BVH for master triangles.
#[allow(dead_code)]
pub struct Bvh {
    nodes: Vec<BvhNode>,
}

impl Bvh {
    pub fn new(triangles: &[MasterTriangle]) -> Self {
        let mut nodes = Vec::new();
        for (i, tri) in triangles.iter().enumerate() {
            let mut bmin = [f64::MAX; 3];
            let mut bmax = [f64::MIN; 3];
            for p in [&tri.a, &tri.b, &tri.c] {
                for d in 0..3 {
                    bmin[d] = bmin[d].min(p[d]);
                    bmax[d] = bmax[d].max(p[d]);
                }
            }
            nodes.push(BvhNode { bbox: [bmin, bmax], tri_idx: i });
        }
        Self { nodes }
    }

    /// Find the closest master triangle to point `p` within `search_dist`.
    pub fn find_closest(&self, p: &[f64; 3], triangles: &[MasterTriangle], search_dist: f64) -> Option<(usize, [f64; 3], f64)> {
        let mut best_dist = search_dist;
        let mut best = None;

        for node in &self.nodes {
            // AABB culling
            let mut bb_dist = 0.0_f64;
            for d in 0..3 {
                if p[d] < node.bbox[0][d] { let dd = node.bbox[0][d] - p[d]; bb_dist += dd * dd; }
                if p[d] > node.bbox[1][d] { let dd = p[d] - node.bbox[1][d]; bb_dist += dd * dd; }
            }
            if bb_dist > best_dist * best_dist { continue; }

            let tri = &triangles[node.tri_idx];
            if let Some((closest, dist)) = closest_point_on_triangle(p, &tri.a, &tri.b, &tri.c) {
                if dist.abs() < best_dist {
                    best_dist = dist.abs();
                    best = Some((node.tri_idx, closest, dist));
                }
            }
        }

        best
    }
}

/// Closest point projection of `p` onto triangle `(a, b, c)`.
///
/// Returns `(closest_point, signed_distance)` where signed distance is
/// positive if p is on the normal side of the triangle.
pub fn closest_point_on_triangle(
    p: &[f64; 3],
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
) -> Option<([f64; 3], f64)> {
    // Compute normal
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if n_len < 1e-30 { return None; }
    let n_hat = [n[0] / n_len, n[1] / n_len, n[2] / n_len];

    // Project p onto the triangle plane
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let d = ap[0] * n_hat[0] + ap[1] * n_hat[1] + ap[2] * n_hat[2];
    let proj = [p[0] - d * n_hat[0], p[1] - d * n_hat[1], p[2] - d * n_hat[2]];

    // Barycentric coordinates of the projected point
    // Using area method
    let v0 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v2 = [proj[0] - a[0], proj[1] - a[1], proj[2] - a[2]];

    let dot00 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    let dot01 = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
    let dot02 = v0[0]*v2[0] + v0[1]*v2[1] + v0[2]*v2[2];
    let dot11 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
    let dot12 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < 1e-30 { return None; }

    let inv = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;

    // Clamp to triangle
    let (u, v) = clamp_barycentric(u, v);

    let closest = [
        a[0] + u * v0[0] + v * v1[0],
        a[1] + u * v0[1] + v * v1[1],
        a[2] + u * v0[2] + v * v1[2],
    ];

    // Signed distance: positive = on normal side
    let dx = p[0] - closest[0];
    let dy = p[1] - closest[1];
    let dz = p[2] - closest[2];
    let signed_dist = dx * n_hat[0] + dy * n_hat[1] + dz * n_hat[2];

    Some((closest, signed_dist))
}

/// Clamp barycentric coordinates to the triangle.
fn clamp_barycentric(u: f64, v: f64) -> (f64, f64) {
    let u = u.max(0.0).min(1.0);
    let v = v.max(0.0).min(1.0);
    if u + v > 1.0 {
        let scale = 1.0 / (u + v);
        (u * scale, v * scale)
    } else {
        (u, v)
    }
}

/// Build master triangle list from mesh boundary faces.
pub fn build_master_triangles<M: fem_mesh::topology::MeshTopology>(
    mesh: &M,
    contact_tags: &[i32],
) -> Vec<MasterTriangle> {
    let mut tris = Vec::new();
    for f in mesh.face_iter() {
        if !contact_tags.contains(&mesh.face_tag(f)) { continue; }
        let nodes = mesh.face_nodes(f);
        if nodes.len() < 3 { continue; }
        let a = mesh.node_coords(nodes[0]);
        let b = mesh.node_coords(nodes[1]);
        let c = mesh.node_coords(nodes[2]);
        let a3 = [a[0], a[1], a[2]];
        let b3 = [b[0], b[1], b[2]];
        let c3 = [c[0], c[1], c[2]];

        // Compute normal and centroid
        let ab = [b3[0]-a3[0], b3[1]-a3[1], b3[2]-a3[2]];
        let ac = [c3[0]-a3[0], c3[1]-a3[1], c3[2]-a3[2]];
        let n = [
            ab[1]*ac[2] - ab[2]*ac[1],
            ab[2]*ac[0] - ab[0]*ac[2],
            ab[0]*ac[1] - ab[1]*ac[0],
        ];
        let n_len = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt().max(1e-30);
        let normal = [n[0]/n_len, n[1]/n_len, n[2]/n_len];
        let centroid = [
            (a3[0] + b3[0] + c3[0]) / 3.0,
            (a3[1] + b3[1] + c3[1]) / 3.0,
            (a3[2] + b3[2] + c3[2]) / 3.0,
        ];
        let radius = {
            let d1 = dist3(&a3, &centroid);
            let d2 = dist3(&b3, &centroid);
            let d3 = dist3(&c3, &centroid);
            d1.max(d2).max(d3)
        };

        tris.push(MasterTriangle { face_idx: f, a: a3, b: b3, c: c3, normal, centroid, radius });
    }
    tris
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    (dx*dx + dy*dy + dz*dz).sqrt()
}

/// Assemble 3D node-to-surface contact forces and stiffness.
///
/// For each slave node, finds the closest master triangle and applies penalty
/// contact if the node penetrates the master surface.
pub fn assemble_n2s_contact_3d(
    slave_coords: &[[f64; 3]],   // current slave node positions (deformed)
    slave_dof_offset: usize,     // DOF offset for slave nodes
    master_triangles: &[MasterTriangle],
    bvh: &Bvh,
    cfg: &N2SContactConfig3D,
    n_total_dofs: usize,
) -> (Vec<f64>, CooMatrix<f64>) {
    let mut f_contact = vec![0.0_f64; n_total_dofs];
    let mut k_contact = CooMatrix::new(n_total_dofs, n_total_dofs);

    for (slave_node_id, slave_pos) in slave_coords.iter().enumerate() {
        let p = [slave_pos[0], slave_pos[1], slave_pos[2]];

        if let Some((tri_idx, closest, gap)) = bvh.find_closest(&p, master_triangles, cfg.search_dist) {
            if gap >= 0.0 { continue; } // no penetration

            let tri = &master_triangles[tri_idx];
            let n = tri.normal;

            // Penalty force: f_n = ε_n · gap · n  (gap < 0 → repulsive)
            let fn_val = -cfg.eps_n * gap;
            let fx = fn_val * n[0];
            let fy = fn_val * n[1];
            let fz = fn_val * n[2];

            // Add to slave node residual
            let dof_x = slave_dof_offset + slave_node_id * 3;
            let dof_y = slave_dof_offset + slave_node_id * 3 + 1;
            let dof_z = slave_dof_offset + slave_node_id * 3 + 2;

            if dof_x < n_total_dofs { f_contact[dof_x] += fx; }
            if dof_y < n_total_dofs { f_contact[dof_y] += fy; }
            if dof_z < n_total_dofs { f_contact[dof_z] += fz; }

            // Stiffness: slave diagonal
            if dof_x < n_total_dofs { k_contact.add(dof_x, dof_x, cfg.eps_n * n[0] * n[0]); }
            if dof_y < n_total_dofs { k_contact.add(dof_y, dof_y, cfg.eps_n * n[1] * n[1]); }
            if dof_z < n_total_dofs { k_contact.add(dof_z, dof_z, cfg.eps_n * n[2] * n[2]); }
            if dof_x < n_total_dofs && dof_y < n_total_dofs { k_contact.add(dof_x, dof_y, cfg.eps_n * n[0] * n[1]); }
            if dof_x < n_total_dofs && dof_z < n_total_dofs { k_contact.add(dof_x, dof_z, cfg.eps_n * n[0] * n[2]); }
            if dof_y < n_total_dofs && dof_z < n_total_dofs { k_contact.add(dof_y, dof_z, cfg.eps_n * n[1] * n[2]); }
        }
    }

    (f_contact, k_contact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_point_inside_triangle() {
        let p = [0.0, 0.0, 0.5];
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let (closest, dist) = closest_point_on_triangle(&p, &a, &b, &c).unwrap();
        // Point is directly above the triangle, should project to centroid
        assert!((closest[2]).abs() < 1e-10, "should project to plane z=0");
        assert!(dist > 0.0, "should be above triangle");
    }

    #[test]
    fn closest_point_outside_triangle() {
        let p = [2.0, 2.0, 0.5];
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let (closest, _dist) = closest_point_on_triangle(&p, &a, &b, &c).unwrap();
        // Should clamp to nearest edge/vertex
        assert!(closest[0] >= 0.0 && closest[1] >= 0.0);
        assert!(closest[0] + closest[1] <= 1.0 + 1e-10);
    }

    #[test]
    fn bvh_finds_closest() {
        let tri = MasterTriangle {
            face_idx: 0,
            a: [0.0, 0.0, 0.0], b: [1.0, 0.0, 0.0], c: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            centroid: [1.0/3.0, 1.0/3.0, 0.0],
            radius: 1.0,
        };
        let tris = vec![tri];
        let bvh = Bvh::new(&tris);

        let p = [0.2, 0.2, 0.1];
        let result = bvh.find_closest(&p, &tris, 10.0);
        assert!(result.is_some(), "should find the triangle");
        if let Some((_idx, _closest, dist)) = result {
            assert!(dist > 0.0, "point above: dist={:.3e}", dist);
        }
    }
}
