//! 2D moving-mesh helpers (quasi-ALE style).
//!
//! This module provides lightweight mesh-motion utilities for simplex Tri3
//! meshes:
//! - collect boundary nodes by tags
//! - apply prescribed boundary displacements
//! - smooth interior nodes with Laplacian iterations

use std::collections::BTreeSet;

use fem_core::NodeId;

use crate::{SimplexMesh, topology::MeshTopology};

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

pub fn boundary_nodes_with_tags(mesh: &SimplexMesh<2>, tags: &[i32]) -> Vec<NodeId> {
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

pub fn all_boundary_nodes(mesh: &SimplexMesh<2>) -> Vec<NodeId> {
    let mut out = BTreeSet::<NodeId>::new();
    for f in mesh.face_iter() {
        for &n in mesh.face_nodes(f) {
            out.insert(n);
        }
    }
    out.into_iter().collect()
}

pub fn apply_node_displacement<F>(
    mesh: &mut SimplexMesh<2>,
    nodes: &[NodeId],
    mut displacement: F,
) where
    F: FnMut([f64; 2]) -> [f64; 2],
{
    for &n in nodes {
        let p = mesh.coords_of(n);
        let d = displacement(p);
        let off = n as usize * 2;
        mesh.coords[off] = p[0] + d[0];
        mesh.coords[off + 1] = p[1] + d[1];
    }
}

pub fn laplacian_smooth_2d(
    mesh: &mut SimplexMesh<2>,
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

            let mut sx = 0.0;
            let mut sy = 0.0;
            for &j in ngh {
                let off = j as usize * 2;
                sx += mesh.coords[off];
                sy += mesh.coords[off + 1];
            }
            let inv = 1.0 / ngh.len() as f64;
            let ax = sx * inv;
            let ay = sy * inv;

            let off = i * 2;
            let x0 = mesh.coords[off];
            let y0 = mesh.coords[off + 1];
            let x1 = (1.0 - omega) * x0 + omega * ax;
            let y1 = (1.0 - omega) * y0 + omega * ay;
            new_coords[off] = x1;
            new_coords[off + 1] = y1;

            let mv = ((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)).sqrt();
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

fn build_node_neighbors(mesh: &SimplexMesh<2>) -> Vec<Vec<NodeId>> {
    let mut sets: Vec<BTreeSet<NodeId>> = (0..mesh.n_nodes()).map(|_| BTreeSet::new()).collect();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        if ns.len() < 3 {
            continue;
        }
        let a = ns[0] as usize;
        let b = ns[1] as usize;
        let c = ns[2] as usize;
        sets[a].insert(ns[1]);
        sets[a].insert(ns[2]);
        sets[b].insert(ns[0]);
        sets[b].insert(ns[2]);
        sets[c].insert(ns[0]);
        sets[c].insert(ns[1]);
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_node_collection_is_nonempty() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let b = all_boundary_nodes(&mesh);
        assert!(!b.is_empty());
        let tagged = boundary_nodes_with_tags(&mesh, &[1, 2, 3, 4]);
        assert_eq!(b, tagged);
    }

    #[test]
    fn smoothing_moves_interior_with_fixed_boundary() {
        let mut mesh = SimplexMesh::<2>::unit_square_tri(8);
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
}
