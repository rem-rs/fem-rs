//! DOF numbering for Lagrange finite element spaces.
//!
//! Handles both vertex-only DOFs (P1) and vertex+edge DOFs (P2) on simplicial meshes.

use std::collections::HashMap;
use fem_core::types::{DofId, ElemId, NodeId};
use fem_mesh::topology::MeshTopology;

// ─── EdgeKey ─────────────────────────────────────────────────────────────────

/// A canonical (sorted) edge key for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeKey(pub NodeId, pub NodeId);

impl EdgeKey {
    pub fn new(a: NodeId, b: NodeId) -> Self {
        if a < b { EdgeKey(a, b) } else { EdgeKey(b, a) }
    }
}

// ─── FaceKey ─────────────────────────────────────────────────────────────────

/// A canonical (sorted) triangular face key for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceKey(pub NodeId, pub NodeId, pub NodeId);

impl FaceKey {
    pub fn new(a: NodeId, b: NodeId, c: NodeId) -> Self {
        let mut v = [a, b, c];
        v.sort_unstable();
        FaceKey(v[0], v[1], v[2])
    }
}

// ─── DofManager ──────────────────────────────────────────────────────────────

/// Manages the global DOF numbering for a Lagrange FE space.
///
/// Supported orders:
/// - **P1** (`order = 1`): one DOF per mesh node.
/// - **P2** (`order = 2`): one DOF per node plus one per mesh edge.
///   Edge DOFs are indexed `n_nodes .. n_nodes + n_edges`.
///
/// DOF ordering within an element for P2 triangles follows [`fem_element::TriP2`]:
/// vertices first (local indices 0,1,2), then edge midpoints (local 3,4,5).
/// The edge order per triangle is: edge(0→1), edge(1→2), edge(0→2).
pub struct DofManager {
    /// Polynomial order (1 or 2).
    pub order: u8,
    /// Total number of DOFs.
    pub n_dofs: usize,
    /// For each element: flat slice of global DOF indices.
    /// Stored as a single `Vec<DofId>` with per-element stride.
    pub(crate) dofs_flat: Vec<DofId>,
    /// Number of DOFs per element.
    pub(crate) dofs_per_elem: usize,
    /// Coordinates of each DOF node (flat, `n_dofs × dim`).
    pub dof_coords: Vec<f64>,
    /// Spatial dimension.
    pub dim: usize,
    /// Number of mesh nodes (vertex DOFs). For P2, edge DOFs start at this index.
    pub n_vertex_dofs: usize,
    /// Edge-to-DOF mapping (P2 only). Maps canonical edge keys to global DOF IDs.
    /// Empty for P1.
    pub edge_dof_map: HashMap<EdgeKey, DofId>,
}

impl DofManager {
    /// Build the DOF map for a mesh with given polynomial order.
    ///
    /// Currently supports:
    /// - Any mesh with `order = 1` (vertex DOFs).
    /// - 2-D triangular meshes (`Tri3`) with `order = 2`.
    ///
    /// # Panics
    /// Panics if `order > 2` or if `order = 2` is requested on a non-triangular mesh.
    pub fn new<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        match order {
            1 => Self::build_p1(mesh),
            2 => Self::build_p2(mesh),
            _ => panic!("DofManager: order {order} not supported (max 2)"),
        }
    }

    /// Global DOF indices for element `elem`.
    pub fn element_dofs(&self, elem: ElemId) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.dofs_flat[start .. start + self.dofs_per_elem]
    }

    /// Physical coordinates of DOF `dof` (slice of length `dim`).
    pub fn dof_coord(&self, dof: DofId) -> &[f64] {
        let start = dof as usize * self.dim;
        &self.dof_coords[start .. start + self.dim]
    }

    // ─── P1 ──────────────────────────────────────────────────────────────────

    fn build_p1<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        let npe = mesh.element_nodes(0).len(); // nodes per element

        // DOF i = node i directly.
        let mut dofs_flat = Vec::with_capacity(n_elems * npe);
        for e in 0..n_elems as u32 {
            for &n in mesh.element_nodes(e) {
                dofs_flat.push(n);
            }
        }

        // DOF coordinates = node coordinates.
        let mut dof_coords = Vec::with_capacity(n_nodes * dim);
        for n in 0..n_nodes as u32 {
            dof_coords.extend_from_slice(mesh.node_coords(n));
        }

        DofManager { order: 1, n_dofs: n_nodes, dofs_flat, dofs_per_elem: npe, dof_coords, dim, n_vertex_dofs: n_nodes, edge_dof_map: HashMap::new() }
    }

    // ─── P2 ──────────────────────────────────────────────────────────────────

    fn build_p2<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes  = mesh.n_nodes();
        let n_elems  = mesh.n_elements();
        let dim      = mesh.dim() as usize;
        assert_eq!(dim, 2, "P2 DofManager currently only supports 2-D meshes");

        // Edge enumeration: for each element triangle, 3 edges.
        // Edge local ordering matching TriP2: edge(0→1)=3, edge(1→2)=4, edge(0→2)=5
        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_edge_dof = n_nodes as DofId;

        // Pre-allocate DOF lists per element (3 vertices + 3 edges = 6).
        let dofs_per_elem = 6;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 3, "P2 requires at least 3-node elements");
            let (n0, n1, n2) = (ns[0], ns[1], ns[2]);

            // Vertices (first 3 DOFs)
            dofs_flat[e as usize * dofs_per_elem]     = n0;
            dofs_flat[e as usize * dofs_per_elem + 1] = n1;
            dofs_flat[e as usize * dofs_per_elem + 2] = n2;

            // Edge DOFs: edge(n0→n1), edge(n1→n2), edge(n0→n2)
            let edges = [(n0, n1), (n1, n2), (n0, n2)];
            for (k, &(a, b)) in edges.iter().enumerate() {
                let key = EdgeKey::new(a, b);
                let dof = *edge_map.entry(key).or_insert_with(|| {
                    let d = next_edge_dof;
                    next_edge_dof += 1;
                    d
                });
                dofs_flat[e as usize * dofs_per_elem + 3 + k] = dof;
            }
        }

        let n_dofs = next_edge_dof as usize;

        // Build DOF coordinates: vertex coords first, then edge midpoints.
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Vertex coordinates.
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let base = n as usize * dim;
            dof_coords[base .. base + dim].copy_from_slice(c);
        }

        // Edge midpoints.
        for (&EdgeKey(a, b), &dof_id) in &edge_map {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            let base = dof_id as usize * dim;
            for d in 0..dim {
                dof_coords[base + d] = 0.5 * (ca[d] + cb[d]);
            }
        }

        DofManager { order: 2, n_dofs, dofs_flat, dofs_per_elem, dof_coords, dim, n_vertex_dofs: n_nodes, edge_dof_map: edge_map }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn p1_unit_square_dof_count() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dm = DofManager::new(&mesh, 1);
        assert_eq!(dm.n_dofs, mesh.n_nodes());
    }

    #[test]
    fn p1_element_dofs_are_node_ids() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 1);
        for e in 0..mesh.n_elements() as u32 {
            let dofs = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(dofs, nodes, "elem {e}");
        }
    }

    #[test]
    fn p2_unit_square_dof_count() {
        // n×n grid → 2n² triangles; n_nodes = (n+1)², n_edges = 3n² + 2n (internal formula)
        // But we just check the lower bound: n_dofs > n_nodes
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dm = DofManager::new(&mesh, 2);
        assert!(dm.n_dofs > mesh.n_nodes(), "P2 must have more DOFs than nodes");
        assert_eq!(dm.dofs_per_elem, 6);
    }

    #[test]
    fn p2_element_first_three_are_vertex_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 2);
        for e in 0..mesh.n_elements() as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs[..3], nodes, "elem {e}: vertex DOFs mismatch");
        }
    }

    #[test]
    fn p2_edge_dofs_are_shared_between_adjacent_elements() {
        // On a 1×1 unit square with 2 triangles (2×2 mesh, but using 1×1):
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        // Should have exactly 2 triangles sharing the diagonal edge.
        // The two shared edge DOFs should be the same global index.
        let dm = DofManager::new(&mesh, 2);
        assert_eq!(mesh.n_elements(), 2);

        let dofs0 = dm.element_dofs(0).to_vec();
        let dofs1 = dm.element_dofs(1).to_vec();

        // Edge DOFs are at positions 3,4,5 in each element.
        // At least one shared edge DOF must be common between the two elements.
        let shared: Vec<_> = dofs0[3..].iter().filter(|d| dofs1[3..].contains(d)).collect();
        assert!(!shared.is_empty(), "no shared edge DOFs between adjacent triangles");
    }
}
