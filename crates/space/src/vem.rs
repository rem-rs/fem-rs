//! VEM (Virtual Element Method) finite element space for polygonal meshes.
//!
//! Supports arbitrary polynomial order p ≥ 1 (P1, P2, P3, …).
//! For order p, each polygon element with N vertices has:
//!   - 1 vertex DOF per vertex (function value)
//!   - (p-1) edge DOFs per edge (Legendre moments)
//!   - (p-1)(p-2)/2 internal DOFs (area moments)
//!     Global edge DOFs are shared between adjacent elements.

use std::collections::HashMap;
use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

/// VEM finite element space on a polygonal mesh.
#[derive(Debug, Clone)]
pub struct VEMSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    /// Flat DOF array: element e has DOFs at `dofs_flat[dofs_per_elem[e]..dofs_per_elem[e+1]]`.
    dofs_flat: Vec<DofId>,
    /// Per-element DOF start offsets (length n_elems + 1).
    dofs_per_elem: Vec<usize>,
    /// Number of global DOFs.
    n_dofs_global: usize,
}

/// Key for an edge: sorted (a, b) vertex pair.
type EdgeKey = (u32, u32);

/// Build edge → global edge index map from element connectivity.
fn build_edge_map<M: MeshTopology>(mesh: &M) -> HashMap<EdgeKey, usize> {
    let mut edge_map: HashMap<EdgeKey, usize> = HashMap::new();
    let n_elems = mesh.n_elements();
    for e in 0..n_elems as u32 {
        let nodes = mesh.element_nodes(e);
        let nv = nodes.len();
        for i in 0..nv {
            let a = nodes[i];
            let b = nodes[(i + 1) % nv];
            let key = if a < b { (a, b) } else { (b, a) };
            let n_edges = edge_map.len();
            edge_map.entry(key).or_insert(n_edges);
        }
    }
    edge_map
}

impl<M: MeshTopology> VEMSpace<M> {
    /// Build a VEM space of order `p` (P1, P2, P3, …).
    pub fn new(mesh: M, p: u8) -> Self {
        assert!(p >= 1, "VEM order must be ≥ 1");
        let n_elems = mesh.n_elements();
        let edge_map = build_edge_map(&mesh);
        let n_global_edges = edge_map.len();
        let n_vertices = mesh.n_nodes();

        // Edge DOFs: (p-1) per global edge
        let _n_edge_dofs = n_global_edges * (p as usize).saturating_sub(1);
        let edge_dof_global_offset = n_vertices as DofId;

        // Assign global DOF indices for edge DOFs
        // edge_dof_map[edge_index][k] = global DOF for k-th edge moment
        let mut edge_dof_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        let mut next_edge_dof = edge_dof_global_offset;
        for &key in edge_map.keys() {
            let mut dofs = Vec::with_capacity((p as usize).saturating_sub(1));
            for _ in 0..(p as usize).saturating_sub(1) {
                dofs.push(next_edge_dof);
                next_edge_dof += 1;
            }
            edge_dof_map.insert(key, dofs);
        }

        // Internal DOFs: (p-1)(p-2)/2 per element, element-local
        let n_internal = |p: u8| -> usize {
            let k = p as usize;
            if k < 2 { 0 } else { (k - 1) * (k - 2) / 2 }
        };
        let _n_internal_per_elem = n_internal(p);

        // Build flat DOF array
        let mut dofs_flat = Vec::new();
        let mut dofs_per_elem = Vec::with_capacity(n_elems + 1);
        dofs_per_elem.push(0usize);

        for e in 0..n_elems as u32 {
            let nodes = mesh.element_nodes(e);
            let nv = nodes.len();

            // 1. Vertex DOFs
            for &n in nodes {
                dofs_flat.push(n);
            }

            // 2. Edge DOFs
            for i in 0..nv {
                let a = nodes[i];
                let b = nodes[(i + 1) % nv];
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(edge_dofs) = edge_dof_map.get(&key) {
                    dofs_flat.extend_from_slice(edge_dofs);
                }
            }

            // 3. Internal DOFs (element-local)
            let n_int = n_internal(p);
            for _ in 0..n_int {
                dofs_flat.push(next_edge_dof);
                next_edge_dof += 1;
            }

            dofs_per_elem.push(dofs_flat.len());
        }

        let n_dofs_global = next_edge_dof as usize;

        Self {
            mesh,
            order: p,
            dofs_flat,
            dofs_per_elem,
            n_dofs_global,
        }
    }

    /// Return the number of polygon vertices for element `elem`.
    pub fn element_n_vertices(&self, elem: u32) -> usize {
        self.mesh.element_nodes(elem).len()
    }

    /// Total number of edges in the mesh (global, shared).
    pub fn n_global_edges(&self) -> usize {
        build_edge_map(&self.mesh).len()
    }
}

impl<M: MeshTopology + 'static> FESpace for VEMSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }
    fn n_dofs(&self) -> usize { self.n_dofs_global }
    fn order(&self) -> u8 { self.order }
    fn space_type(&self) -> SpaceType { SpaceType::VEM }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let s = self.dofs_per_elem[elem as usize];
        let e = self.dofs_per_elem[elem as usize + 1];
        &self.dofs_flat[s..e]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.n_dofs_global;
        let mut result = Vector::zeros(n);
        let sl = result.as_slice_mut();

        // Interpolate vertex DOFs (point values)
        let n_vertices = self.mesh.n_nodes();
        for node in 0..n_vertices as u32 {
            let coord = self.mesh.node_coords(node);
            sl[node as usize] = f(coord);
        }

        // Edge and internal DOFs require projection – for a basic interpolate,
        // we set them to zero (they are determined by the projection in the solver).
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::poly_mesh::PolyMesh;

    #[test]
    fn vem_p1_quad_4x4() {
        let mesh = PolyMesh::unit_square_quad(3, 3);
        let space = VEMSpace::new(mesh, 1);
        assert_eq!(space.n_dofs(), 16);
        assert_eq!(space.order(), 1);
        let dofs0 = space.element_dofs(0);
        assert_eq!(dofs0.len(), 4);
    }

    #[test]
    fn vem_p2_quad_2x2_dof_count() {
        // 2×2=4 quad elements, each with 4 vertices, 4 edges (shared).
        // Vertices: 3×3=9
        // Edges (global): 4 horizontal + 4 vertical = 8 interior + 8 boundary = 12 total
        // Edge DOFs per edge: p-1 = 1 → 12 global edge DOFs
        // Internal DOFs: 0 for p=2
        // Total: 9 + 12 = 21
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 2);
        assert_eq!(space.n_dofs(), 21);
        assert_eq!(space.order(), 2);
    }

    #[test]
    fn vem_p2_quad_2x2_element_dofs() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 2);
        // Each quad element has 4 vertices + 4 edges = 8 DOFs
        let dofs0 = space.element_dofs(0);
        assert_eq!(dofs0.len(), 8);
        // First 4 DOFs are vertex DOFs
        assert_eq!(dofs0[0], 0);
        assert_eq!(dofs0[1], 1);
        assert_eq!(dofs0[2], 4); // element 0 has vertices [0,1,4,3]
        assert_eq!(dofs0[3], 3);
        // Next 4 are edge DOFs (shared)
        let dofs1 = space.element_dofs(1);
        assert_eq!(dofs1.len(), 8);
        // Element 0 and 1 share the vertical edge (1,4)
        // In element 0: edge index 1 (vertex 1→4) → dof position 4+1=5
        // In element 1: edge index 3 (vertex 4→1) → dof position 4+3=7
        assert_eq!(dofs0[5], dofs1[7], "shared edge DOF should match");
    }

    #[test]
    fn vem_p3_quad_2x2() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 3);
        // Each quad element: 4 vertices + 2*4 edges + 1 internal = 13 DOFs
        // Global: 9 vertices + 12*2 edges + 4 internal = 9+24+4 = 37
        assert_eq!(space.n_dofs(), 37);
        let dofs0 = space.element_dofs(0);
        assert_eq!(dofs0.len(), 13);
    }

    #[test]
    fn vem_p1_interpolate_constant() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 1);
        let v = space.interpolate(&|_| 3.0);
        for i in 0..space.n_dofs() {
            assert!((v.as_slice()[i] - 3.0).abs() < 1e-14);
        }
    }
}
