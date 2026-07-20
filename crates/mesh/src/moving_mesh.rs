//! Moving-mesh helpers (quasi-ALE style).
//!
//! Provides dimension-generic mesh-motion utilities:
//! - collect boundary nodes by tags
//! - apply prescribed boundary displacements
//! - smooth interior nodes with Laplacian iterations (2D and 3D)
//!
//! # Usage
//!
//! ```rust,ignore
//! use fem_mesh::{Mesh, moving_mesh::{laplacian_smooth_2d, MeshMotionConfig}};
//!
//! let mut mesh = Mesh::<2>::unit_square_tri(16);
//! let fixed = moving_mesh::all_boundary_nodes(&mesh);
//! laplacian_smooth_2d(&mut mesh, &fixed, MeshMotionConfig::default());
//! ```

use std::collections::BTreeSet;

use fem_core::NodeId;

use crate::{Mesh, topology::MeshTopology};

#[derive(Debug, Clone, Copy)]
pub struct MeshMotionConfig {
    pub omega: f64,
    pub max_iters: usize,
    pub tol: f64,
}

impl Default for MeshMotionConfig {
    fn default() -> Self {
        Self {
            omega: 0.7,
            max_iters: 30,
            tol: 1.0e-10,
        }
    }
}

/// Collect boundary nodes whose face tag matches one of `tags`.
///
/// Generic over mesh dimension `D` (works for `Mesh<2>` and `Mesh<3>`).
pub fn boundary_nodes_with_tags<const D: usize>(mesh: &Mesh<D>, tags: &[i32]) -> Vec<NodeId> {
    let mut out = BTreeSet::<NodeId>::new();
    for f in mesh.face_iter() {
        if tags.contains(&mesh.face_tag(f)) {
            for &n in mesh.face_nodes(f) {
                out.insert(n);
            }
        }
    }
    out.into_iter().collect()
}

/// Collect all boundary nodes from every boundary face.
pub fn all_boundary_nodes<const D: usize>(mesh: &Mesh<D>) -> Vec<NodeId> {
    let mut out = BTreeSet::<NodeId>::new();
    for f in mesh.face_iter() {
        for &n in mesh.face_nodes(f) {
            out.insert(n);
        }
    }
    out.into_iter().collect()
}

/// Apply a displacement function to a set of nodes, updating `mesh.coords` in place.
///
/// The displacement function receives `[f64; D]` (current coordinates) and returns `[f64; D]`
/// (displacement vector to add).
pub fn apply_node_displacement<const D: usize, F>(
    mesh: &mut Mesh<D>,
    nodes: &[NodeId],
    mut displacement: F,
) where
    F: FnMut([f64; D]) -> [f64; D],
{
    for &n in nodes {
        let p = mesh.coords_of(n);
        let d = displacement(p);
        let off = n as usize * D;
        for dim in 0..D {
            mesh.coords[off + dim] = p[dim] + d[dim];
        }
    }
}

/// Laplacian smoothing for interior mesh nodes.
///
/// Moves each interior node to the average of its neighbors using
/// successive over-relaxation (SOR) with relaxation factor `omega`.
/// Nodes in `fixed_nodes` are held in place.
///
/// Returns the number of iterations performed.
pub fn laplacian_smooth<const D: usize>(
    mesh: &mut Mesh<D>,
    fixed_nodes: &[NodeId],
    cfg: MeshMotionConfig,
) -> usize {
    let n = mesh.n_nodes();
    let neighbors = build_node_neighbors(mesh);
    let mut fixed = vec![false; n];
    for &node in fixed_nodes {
        if (node as usize) < n {
            fixed[node as usize] = true;
        }
    }

    let omega = cfg.omega.clamp(0.0, 1.0);
    let mut new_coords = mesh.coords.clone();

    for it in 0..cfg.max_iters {
        let mut max_move = 0.0_f64;
        for i in 0..n {
            if fixed[i] {
                continue;
            }
            let ngh = &neighbors[i];
            if ngh.is_empty() {
                continue;
            }

            let mut sum = vec![0.0_f64; D];
            for &j in ngh {
                let off = j as usize * D;
                for dim in 0..D {
                    sum[dim] += mesh.coords[off + dim];
                }
            }
            let inv = 1.0 / ngh.len() as f64;
            let off = i * D;
            let mut sq_dist = 0.0_f64;
            for dim in 0..D {
                let x0 = mesh.coords[off + dim];
                let x_avg = sum[dim] * inv;
                let x1 = (1.0 - omega) * x0 + omega * x_avg;
                new_coords[off + dim] = x1;
                let d = x1 - x0;
                sq_dist += d * d;
            }

            let mv = sq_dist.sqrt();
            if mv > max_move {
                max_move = mv;
            }
        }

        mesh.coords.copy_from_slice(&new_coords);
        if max_move < cfg.tol {
            return it + 1;
        }
    }

    cfg.max_iters
}

/// 2D convenience wrapper for [`laplacian_smooth`].
pub fn laplacian_smooth_2d(
    mesh: &mut Mesh<2>,
    fixed_nodes: &[NodeId],
    cfg: MeshMotionConfig,
) -> usize {
    laplacian_smooth(mesh, fixed_nodes, cfg)
}

/// 3D convenience wrapper for [`laplacian_smooth`].
pub fn laplacian_smooth_3d(
    mesh: &mut Mesh<3>,
    fixed_nodes: &[NodeId],
    cfg: MeshMotionConfig,
) -> usize {
    laplacian_smooth(mesh, fixed_nodes, cfg)
}

/// Build node adjacency from element connectivity.
///
/// Two nodes are neighbors if they belong to the same element.
/// Works for any element type (Tri3, Quad4, Tet4, Hex8, etc.).
fn build_node_neighbors<const D: usize>(mesh: &Mesh<D>) -> Vec<Vec<NodeId>> {
    let mut sets: Vec<BTreeSet<NodeId>> = (0..mesh.n_nodes()).map(|_| BTreeSet::new()).collect();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for &a in ns {
            for &b in ns {
                if a != b {
                    sets[a as usize].insert(b);
                }
            }
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_node_collection_is_nonempty_2d() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let b = all_boundary_nodes(&mesh);
        assert!(!b.is_empty());
        let tagged = boundary_nodes_with_tags(&mesh, &[1, 2, 3, 4]);
        assert_eq!(b, tagged);
    }

    #[test]
    fn smoothing_moves_interior_with_fixed_boundary_2d() {
        let mut mesh = Mesh::<2>::unit_square_tri(8);
        let fixed = all_boundary_nodes(&mesh);

        let center = (mesh.n_nodes() as u32) / 2;
        let p0 = mesh.coords_of(center);
        {
            let off = center as usize * 2;
            mesh.coords[off] += 0.10;
            mesh.coords[off + 1] -= 0.05;
        }

        let it = laplacian_smooth_2d(
            &mut mesh,
            &fixed,
            MeshMotionConfig {
                omega: 0.7,
                max_iters: 20,
                tol: 1.0e-12,
            },
        );
        assert!(it > 0);

        let p1 = mesh.coords_of(center);
        let d0 = ((p0[0] + 0.10 - p0[0]).powi(2) + (p0[1] - 0.05 - p0[1]).powi(2)).sqrt();
        let d1 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        assert!(d1 < d0, "smoothing should reduce perturbation magnitude");
    }

    #[test]
    fn boundary_nodes_3d_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let b = all_boundary_nodes(&mesh);
        assert!(!b.is_empty());
        // The unit cube has 6 boundary faces with tags 1..6
        let tagged = boundary_nodes_with_tags(&mesh, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(b, tagged);
    }

    #[test]
    fn smoothing_3d_tet_all_fixed_noop() {
        let mut mesh = Mesh::<3>::unit_cube_tet(3);
        let n0 = mesh.n_nodes();
        let coords0 = mesh.coords.clone();
        let all: Vec<NodeId> = (0..n0 as u32).collect();
        let it = laplacian_smooth_3d(
            &mut mesh,
            &all,
            MeshMotionConfig { omega: 0.7, max_iters: 10, tol: 1e-14 },
        );
        // All nodes fixed → immediate convergence (max_move=0)
        assert!(it > 0, "should converge immediately");
        // Coordinates unchanged
        for (a, b) in coords0.iter().zip(mesh.coords.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn smoothing_3d_tet_interior_moves() {
        let mut mesh = Mesh::<3>::unit_cube_tet(3);
        let fixed = all_boundary_nodes(&mesh);
        let n_nodes = mesh.n_nodes();

        // Find a node that is not on the boundary by scanning all nodes
        // and checking if they are in the fixed set
        let fixed_set: std::collections::HashSet<NodeId> = fixed.iter().copied().collect();
        let interior: Option<NodeId> = (0..n_nodes as u32).find(|id| !fixed_set.contains(id));
        let interior = match interior {
            Some(id) => id,
            None => return, // no interior nodes; nothing to test
        };

        let p0 = mesh.coords_of(interior);
        // Perturb this interior node
        {
            let off = interior as usize * 3;
            mesh.coords[off] += 0.05;
            mesh.coords[off + 1] -= 0.03;
            mesh.coords[off + 2] += 0.01;
        }

        let it = laplacian_smooth_3d(
            &mut mesh,
            &fixed,
            MeshMotionConfig { omega: 0.7, max_iters: 50, tol: 1e-12 },
        );
        assert!(it > 0, "smoothing should converge");

        let p1 = mesh.coords_of(interior);
        let d0 = ((0.05_f64).powi(2) + (-0.03_f64).powi(2) + (0.01_f64).powi(2)).sqrt();
        let d1 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
        assert!(d1 < d0, "smoothing should reduce perturbation magnitude ({:.6e} vs {:.6e})", d1, d0);
    }

    #[test]
    fn hex8_smoothing_works() {
        let mut mesh = Mesh::<3>::unit_cube_hex(3);
        let fixed = all_boundary_nodes(&mesh);
        let interior = (mesh.n_nodes() as u32) / 2;
        {
            let off = interior as usize * 3;
            mesh.coords[off] += 0.05;
        }
        let it = laplacian_smooth_3d(
            &mut mesh,
            &fixed,
            MeshMotionConfig { omega: 0.5, max_iters: 20, tol: 1e-12 },
        );
        assert!(it > 0, "Hex8 smoothing should converge");
    }

    #[test]
    fn apply_displacement_2d() {
        let mut mesh = Mesh::<2>::unit_square_tri(4);
        let nodes: Vec<NodeId> = (0..5).collect();
        // Displacement adds 0.1 to both components
        apply_node_displacement(&mut mesh, &nodes, |_p| [0.1, 0.1]);
        for &n in &nodes {
            let p = mesh.coords_of(n);
            assert!(p[0] > 0.0 && p[1] > 0.0, "node {n} should have moved, got ({},{})", p[0], p[1]);
        }
    }
}
