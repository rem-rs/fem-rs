//! DOF numbering for Lagrange finite element spaces.
//!
//! Handles both vertex-only DOFs (P1) and vertex+edge DOFs (P2) on simplicial meshes.
//! Supports mixed-element meshes (e.g. Tri3+Quad4) via per-element DOF offsets.

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
/// - **P3** (`order = 3`): one DOF per node, two per edge (at 1/3 and 2/3),
///   plus one bubble DOF per element.
///
/// DOF ordering within an element for P2 triangles follows [`fem_element::TriP2`]:
/// vertices first (local indices 0,1,2), then edge midpoints (local 3,4,5).
/// The edge order per triangle is: edge(0→1), edge(1→2), edge(0→2).
///
/// For P3 triangles: vertices (0,1,2), edge DOFs (3..8), bubble (9).
/// Edge ordering same as P2; each edge has 2 DOFs (near first vertex, near second).
///
/// For mixed-element meshes, `dofs_per_elem` is set to 0 and `elem_dof_offsets`
/// provides CSR-like offsets into `dofs_flat`.
pub struct DofManager {
    /// Polynomial order (1, 2, or 3).
    pub order: u8,
    /// Total number of DOFs.
    pub n_dofs: usize,
    /// For each element: flat slice of global DOF indices.
    /// Stored as a single `Vec<DofId>` with per-element stride (uniform) or
    /// variable stride (mixed, when `elem_dof_offsets` is set).
    pub(crate) dofs_flat: Vec<DofId>,
    /// Number of DOFs per element (uniform meshes). Set to 0 for mixed meshes.
    pub(crate) dofs_per_elem: usize,
    /// CSR-like offsets into `dofs_flat` for mixed meshes.
    /// Length = `n_elems + 1`.  `None` for uniform meshes.
    pub(crate) elem_dof_offsets: Option<Vec<usize>>,
    /// Coordinates of each DOF node (flat, `n_dofs × dim`).
    pub dof_coords: Vec<f64>,
    /// Spatial dimension.
    pub dim: usize,
    /// Number of mesh nodes (vertex DOFs). For P2/P3, edge DOFs start at this index.
    pub n_vertex_dofs: usize,
    /// Edge-to-DOF mapping (P2 only). Maps canonical edge keys to global DOF IDs.
    /// Empty for P1 and P3.
    pub edge_dof_map: HashMap<EdgeKey, DofId>,
    /// Edge-to-2-DOF mapping (P3 only). Maps canonical edge keys to two global DOF IDs
    /// ordered [near_first_vertex, near_second_vertex].
    /// Empty for P1 and P2.
    pub edge_dof2_map: HashMap<EdgeKey, [DofId; 2]>,
    /// Index at which bubble DOFs start (P3 only). Equal to `n_dofs` for P1/P2.
    pub bubble_dof_start: usize,
}

impl DofManager {
    /// Build the DOF map for a mesh with given polynomial order.
    ///
    /// Currently supports:
    /// - Any mesh with `order = 1` (vertex DOFs), including mixed-element meshes.
    /// - 2-D triangular meshes (`Tri3`) with `order = 2` or `order = 3`.
    /// - 3-D tetrahedral meshes (`Tet4`) with `order = 2`.
    ///
    /// # Panics
    /// Panics if `order > 3` or if `order = 2,3` is requested on an unsupported mesh type.
    pub fn new<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        match order {
            1 => Self::build_p1(mesh),
            2 => {
                if mesh.dim() == 3 {
                    Self::build_p2_tet(mesh)
                } else {
                    Self::build_p2(mesh)
                }
            }
            3 => Self::build_p3(mesh),
            _ => panic!("DofManager: order {order} not supported (max 3)"),
        }
    }

    /// Global DOF indices for element `elem`.
    pub fn element_dofs(&self, elem: ElemId) -> &[DofId] {
        if let Some(ref offsets) = self.elem_dof_offsets {
            let start = offsets[elem as usize];
            let end = offsets[elem as usize + 1];
            &self.dofs_flat[start..end]
        } else {
            let start = elem as usize * self.dofs_per_elem;
            &self.dofs_flat[start .. start + self.dofs_per_elem]
        }
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

        // Check if all elements have the same number of nodes.
        let first_npe = if n_elems > 0 { mesh.element_nodes(0).len() } else { 0 };
        let is_mixed = (0..n_elems as u32).any(|e| mesh.element_nodes(e).len() != first_npe);

        let mut dofs_flat = Vec::new();
        let mut elem_dof_offsets = if is_mixed { Some(Vec::with_capacity(n_elems + 1)) } else { None };

        if let Some(ref mut offsets) = elem_dof_offsets {
            offsets.push(0);
        }

        for e in 0..n_elems as u32 {
            let nodes = mesh.element_nodes(e);
            for &n in nodes {
                dofs_flat.push(n);
            }
            if let Some(ref mut offsets) = elem_dof_offsets {
                offsets.push(dofs_flat.len());
            }
        }

        // DOF coordinates = node coordinates.
        let mut dof_coords = Vec::with_capacity(n_nodes * dim);
        for n in 0..n_nodes as u32 {
            dof_coords.extend_from_slice(mesh.node_coords(n));
        }

        let dofs_per_elem = if is_mixed { 0 } else { first_npe };

        DofManager {
            order: 1, n_dofs: n_nodes, dofs_flat, dofs_per_elem,
            elem_dof_offsets, dof_coords, dim, n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(),
            edge_dof2_map: HashMap::new(),
            bubble_dof_start: n_nodes,
        }
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

        DofManager {
            order: 2, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes, edge_dof_map: edge_map,
            edge_dof2_map: HashMap::new(),
            bubble_dof_start: n_dofs,
        }
    }

    // ─── P3 ──────────────────────────────────────────────────────────────────

    fn build_p3<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes  = mesh.n_nodes();
        let n_elems  = mesh.n_elements();
        let dim      = mesh.dim() as usize;
        assert_eq!(dim, 2, "P3 DofManager currently only supports 2-D meshes");

        // DOF layout per element (10):
        //   0,1,2   → vertex DOFs (same as node IDs)
        //   3,4     → edge(n0→n1): DOFs at 1/3 (near n0) and 2/3 (near n1)
        //   5,6     → edge(n1→n2): DOFs at 1/3 (near n1) and 2/3 (near n2)
        //   7,8     → edge(n0→n2): DOFs at 1/3 (near n0) and 2/3 (near n2)
        //   9       → bubble DOF (centroid)
        //
        // DOF numbering: vertex 0..n_nodes, then edge 2-DOFs, then bubble DOFs.
        // Two passes: pass 1 assigns edge DOFs; pass 2 assigns bubble DOFs.

        // ── Pass 1: enumerate edges, assign 2 DOFs per unique edge ──────────
        // pair[0] = DOF near canonical-first vertex, pair[1] = near canonical-second.
        let mut edge2_map: HashMap<EdgeKey, [DofId; 2]> = HashMap::new();
        let mut next_edge_dof = n_nodes as DofId;

        let dofs_per_elem = 10;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        // Helper closure (used within the loop below via a function to avoid borrow conflicts).
        // Returns [dof_near_a, dof_near_b] in original a→b orientation.
        fn get_edge_dofs(
            a: NodeId, b: NodeId,
            next: &mut DofId,
            map: &mut HashMap<EdgeKey, [DofId; 2]>,
        ) -> [DofId; 2] {
            let key = EdgeKey::new(a, b);
            let pair = *map.entry(key).or_insert_with(|| {
                let d0 = *next; *next += 1;
                let d1 = *next; *next += 1;
                [d0, d1]  // [near canonical-first = near key.0, near key.1]
            });
            if a == key.0 {
                [pair[0], pair[1]]
            } else {
                [pair[1], pair[0]]
            }
        }

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 3, "P3 requires at least 3-node elements");
            let (n0, n1, n2) = (ns[0], ns[1], ns[2]);

            // Vertices
            let base = e as usize * dofs_per_elem;
            dofs_flat[base]     = n0;
            dofs_flat[base + 1] = n1;
            dofs_flat[base + 2] = n2;

            let [d3, d4] = get_edge_dofs(n0, n1, &mut next_edge_dof, &mut edge2_map);
            dofs_flat[base + 3] = d3;
            dofs_flat[base + 4] = d4;

            let [d5, d6] = get_edge_dofs(n1, n2, &mut next_edge_dof, &mut edge2_map);
            dofs_flat[base + 5] = d5;
            dofs_flat[base + 6] = d6;

            let [d7, d8] = get_edge_dofs(n0, n2, &mut next_edge_dof, &mut edge2_map);
            dofs_flat[base + 7] = d7;
            dofs_flat[base + 8] = d8;
            // Bubble DOF assigned in pass 2.
        }

        // ── Pass 2: assign one bubble DOF per element ────────────────────────
        let bubble_dof_start = next_edge_dof as usize;
        for e in 0..n_elems as u32 {
            let bubble = bubble_dof_start as DofId + e;
            dofs_flat[e as usize * dofs_per_elem + 9] = bubble;
        }

        let n_dofs = bubble_dof_start + n_elems;

        // ── Build DOF coordinates ────────────────────────────────────────────
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Vertex coordinates.
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let base = n as usize * dim;
            dof_coords[base .. base + dim].copy_from_slice(c);
        }

        // Edge DOF coordinates: pair[0] at 1/3 from canonical-first toward second,
        // pair[1] at 2/3 from canonical-first (= 1/3 from canonical-second).
        for (&EdgeKey(a, b), &[d0, d1]) in &edge2_map {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            let base0 = d0 as usize * dim;
            let base1 = d1 as usize * dim;
            for d in 0..dim {
                dof_coords[base0 + d] = (2.0 * ca[d] + cb[d]) / 3.0;
                dof_coords[base1 + d] = (ca[d] + 2.0 * cb[d]) / 3.0;
            }
        }

        // Bubble DOF coordinates: centroid of each element.
        for e in 0..n_elems as u32 {
            let bubble_dof = (bubble_dof_start + e as usize) * dim;
            let ns = mesh.element_nodes(e);
            for d in 0..dim {
                let cx: f64 = ns.iter().take(3).map(|&n| mesh.node_coords(n)[d]).sum::<f64>() / 3.0;
                dof_coords[bubble_dof + d] = cx;
            }
        }

        DofManager {
            order: 3, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(),
            edge_dof2_map: edge2_map,
            bubble_dof_start,
        }
    }

    // ─── P2 (3-D Tet) ─────────────────────────────────────────────────────────

    fn build_p2_tet<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes  = mesh.n_nodes();
        let n_elems  = mesh.n_elements();
        let dim      = mesh.dim() as usize;
        assert_eq!(dim, 3, "build_p2_tet requires a 3-D mesh");

        // DOF layout per element (10):
        //   0,1,2,3  → vertex DOFs (node IDs)
        //   4        → edge(n0→n1) midpoint
        //   5        → edge(n0→n2) midpoint
        //   6        → edge(n0→n3) midpoint
        //   7        → edge(n1→n2) midpoint
        //   8        → edge(n1→n3) midpoint
        //   9        → edge(n2→n3) midpoint
        //
        // Edge order matches TetP2 dof_coords() ordering.

        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_edge_dof = n_nodes as DofId;

        let dofs_per_elem = 10;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 4, "TetP2 requires 4-node tetrahedra");
            let (n0, n1, n2, n3) = (ns[0], ns[1], ns[2], ns[3]);

            let base = e as usize * dofs_per_elem;
            // Vertex DOFs
            dofs_flat[base]     = n0;
            dofs_flat[base + 1] = n1;
            dofs_flat[base + 2] = n2;
            dofs_flat[base + 3] = n3;

            // Edge DOFs (6 edges of a tet)
            let edges = [(n0, n1), (n0, n2), (n0, n3), (n1, n2), (n1, n3), (n2, n3)];
            for (k, &(a, b)) in edges.iter().enumerate() {
                let key = EdgeKey::new(a, b);
                let dof = *edge_map.entry(key).or_insert_with(|| {
                    let d = next_edge_dof;
                    next_edge_dof += 1;
                    d
                });
                dofs_flat[base + 4 + k] = dof;
            }
        }

        let n_dofs = next_edge_dof as usize;

        // Build DOF coordinates: vertices then edge midpoints.
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let base = n as usize * dim;
            dof_coords[base .. base + dim].copy_from_slice(c);
        }

        for (&EdgeKey(a, b), &dof_id) in &edge_map {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            let base = dof_id as usize * dim;
            for d in 0..dim {
                dof_coords[base + d] = 0.5 * (ca[d] + cb[d]);
            }
        }

        DofManager {
            order: 2, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes, edge_dof_map: edge_map,
            edge_dof2_map: HashMap::new(),
            bubble_dof_start: n_dofs,
        }
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

    #[test]
    fn p1_mixed_tri_quad_dofs() {
        use fem_mesh::element_type::ElementType;
        // 5 nodes: 1 quad (0,1,3,2) + 1 tri (1,4,3)
        //  2---3---4
        //  |   | /
        //  0---1
        let mut mesh = SimplexMesh::<2>::uniform(
            vec![0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,  2.0, 1.0],
            vec![0, 1, 3, 2,  1, 4, 3],  // quad then tri
            vec![1, 1],
            ElementType::Quad4,
            vec![], vec![], ElementType::Line2,
        );
        mesh.elem_types = Some(vec![ElementType::Quad4, ElementType::Tri3]);
        mesh.elem_offsets = Some(vec![0, 4, 7]);

        let dm = DofManager::new(&mesh, 1);
        assert_eq!(dm.n_dofs, 5);
        assert!(dm.elem_dof_offsets.is_some(), "mixed mesh should have elem_dof_offsets");
        assert_eq!(dm.element_dofs(0), &[0, 1, 3, 2]);
        assert_eq!(dm.element_dofs(1), &[1, 4, 3]);
    }

    // ─── P3 tests ─────────────────────────────────────────────────────────────

    #[test]
    fn p3_unit_square_dof_count() {
        // P3 on n×n mesh: n_nodes + 2*n_edges + n_elements bubble DOFs.
        // Just verify: n_dofs > P2 dofs > P1 dofs.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dm1 = DofManager::new(&mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let dm2 = DofManager::new(&mesh2, 2);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
        let dm3 = DofManager::new(&mesh3, 3);
        assert!(dm3.n_dofs > dm2.n_dofs, "P3 must have more DOFs than P2");
        assert!(dm2.n_dofs > dm1.n_dofs, "P2 must have more DOFs than P1");
        assert_eq!(dm3.dofs_per_elem, 10, "P3 elements should have 10 DOFs each");
    }

    #[test]
    fn p3_element_first_three_are_vertex_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 3);
        for e in 0..mesh.n_elements() as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs[..3], nodes, "elem {e}: P3 vertex DOFs mismatch");
        }
    }

    #[test]
    fn p3_edge_dofs_are_shared_between_adjacent_elements() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let dm = DofManager::new(&mesh, 3);
        assert_eq!(mesh.n_elements(), 2);

        let dofs0 = dm.element_dofs(0).to_vec();
        let dofs1 = dm.element_dofs(1).to_vec();

        // Edge DOFs are at positions 3..8; bubble at 9.
        // Adjacent triangles share one edge → at least 2 shared edge DOFs.
        let shared: Vec<_> = dofs0[3..9].iter().filter(|d| dofs1[3..9].contains(d)).collect();
        assert!(shared.len() >= 2, "shared edge DOFs between adjacent P3 triangles: {}", shared.len());
    }

    #[test]
    fn p3_bubble_dofs_are_unique_per_element() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 3);
        let n_elems = mesh.n_elements();
        let mut bubble_dofs: Vec<u32> = (0..n_elems as u32)
            .map(|e| dm.element_dofs(e)[9])
            .collect();
        let len_before = bubble_dofs.len();
        bubble_dofs.sort_unstable();
        bubble_dofs.dedup();
        assert_eq!(bubble_dofs.len(), len_before, "bubble DOFs should be unique per element");
    }

    #[test]
    fn p3_dof_coords_in_unit_square() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dm = DofManager::new(&mesh, 3);
        for dof in 0..dm.n_dofs as u32 {
            let c = dm.dof_coord(dof);
            assert_eq!(c.len(), 2);
            assert!(c[0] >= -1e-12 && c[0] <= 1.0 + 1e-12,
                "DOF {dof}: x={} not in [0,1]", c[0]);
            assert!(c[1] >= -1e-12 && c[1] <= 1.0 + 1e-12,
                "DOF {dof}: y={} not in [0,1]", c[1]);
        }
    }

    #[test]
    fn p3_bubble_dof_start_correct() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 3);
        // bubble_dof_start = n_nodes + 2*n_unique_edges
        // Verify all bubble DOFs (one per element, at position 9) are >= bubble_dof_start
        for e in 0..mesh.n_elements() as u32 {
            let bubble = dm.element_dofs(e)[9] as usize;
            assert!(bubble >= dm.bubble_dof_start,
                "elem {e}: bubble dof {bubble} < bubble_dof_start {}", dm.bubble_dof_start);
        }
    }
}
