//! DOF numbering for Lagrange finite element spaces.
//!
//! Handles vertex-only DOFs (P1), vertex+edge DOFs (P2), and arbitrary-order
//! Lagrange spaces (Pk) on simplicial and tensor-product meshes.
//!
//! For arbitrary order `p >= 1`:
//! - Triangles: (p+1)(p+2)/2 DOFs per element
//! - Tetrahedra: (p+1)(p+2)(p+3)/6 DOFs per element
//!
//! DOF ordering within each element follows [`fem_element::TriPk`] / [`fem_element::TetPk`].

use std::collections::HashMap;
use fem_core::types::{DofId, ElemId, NodeId};
use fem_mesh::topology::MeshTopology;
use fem_element::ReferenceElement;

// ─── EdgeKey ─────────────────────────────────────────────────────────────────

/// A canonical (sorted) edge key for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeKey(pub NodeId, pub NodeId);

impl EdgeKey {
    pub fn new(a: NodeId, b: NodeId) -> Self {
        if a < b { EdgeKey(a, b) } else { EdgeKey(b, a) }
    }
}

// ─── Pk helper: edge DOFs ────────────────────────────────────────────────────

/// Get or create `n_dofs` edge DOFs for a canonical edge.
/// Returns DOFs in order from `a`→`b` (reversed if the call arguments are reversed).
fn get_edge_dofs_pk(
    a: NodeId, b: NodeId,
    next: &mut DofId,
    map: &mut HashMap<EdgeKey, Vec<DofId>>,
    n_dofs: usize,
) -> Vec<DofId> {
    let key = EdgeKey::new(a, b);
    let dofs = map.entry(key).or_insert_with(|| {
        (0..n_dofs).map(|_| { let d = *next; *next += 1; d }).collect()
    });
    if a == key.0 { dofs.clone() } else { let mut r = dofs.clone(); r.reverse(); r }
}

/// Get or create `n_dofs` face DOFs for a canonical triangular face.
fn get_face_dofs_pk(
    a: NodeId, b: NodeId, c: NodeId,
    next: &mut DofId,
    map: &mut HashMap<FaceKey, Vec<DofId>>,
    n_dofs: usize,
) -> Vec<DofId> {
    let key = FaceKey::new(a, b, c);
    let dofs = map.entry(key).or_insert_with(|| {
        (0..n_dofs).map(|_| { let d = *next; *next += 1; d }).collect()
    });
    dofs.clone()
}

// ─── FaceKey ─────────────────────────────────────────────────────────────────

/// A canonical (sorted) triangular face key for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceKey(pub NodeId, pub NodeId, pub NodeId);

impl FaceKey {
    pub fn new(a: NodeId, b: NodeId, c: NodeId) -> Self {
        let mut v = [a, b, c];
        v.sort_unstable();
        FaceKey(v[0], v[1], v[2])
    }
}

/// A canonical (sorted) quadrilateral face key for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuadFaceKey(pub NodeId, pub NodeId, pub NodeId, pub NodeId);

impl QuadFaceKey {
    pub fn new(a: NodeId, b: NodeId, c: NodeId, d: NodeId) -> Self {
        let mut v = [a, b, c, d];
        v.sort_unstable();
        QuadFaceKey(v[0], v[1], v[2], v[3])
    }
}

// ─── DofManager ──────────────────────────────────────────────────────────────

/// Manages the global DOF numbering for a Lagrange FE space.
///
/// Supported orders:
/// - **P1** (`order = 1`): one DOF per mesh node.
/// - **P2** (`order = 2`): one DOF per node plus one per mesh edge.
/// - **P3** (`order = 3`): one DOF per node, two per edge, face/volume interior DOFs.
/// - **Pk** (`order >= 4`): vertex, edge, face (3D), and volume interior DOFs
///   in the general pattern.
///
/// DOF ordering within an element follows the factory convention:
/// vertices first, then edge DOFs, then face DOFs (3D), then volume DOFs.
#[derive(Clone)]
pub struct DofManager {
    /// Polynomial order.
    pub order: u8,
    /// Total number of DOFs.
    pub n_dofs: usize,
    /// For each element: flat slice of global DOF indices.
    pub(crate) dofs_flat: Vec<DofId>,
    /// Number of DOFs per element (uniform meshes). 0 for mixed meshes.
    pub(crate) dofs_per_elem: usize,
    /// CSR-like offsets into `dofs_flat` for mixed meshes.
    pub(crate) elem_dof_offsets: Option<Vec<usize>>,
    /// Coordinates of each DOF node (flat, `n_dofs × dim`).
    pub dof_coords: Vec<f64>,
    /// Spatial dimension.
    pub dim: usize,
    /// Number of mesh nodes (vertex DOFs).
    pub n_vertex_dofs: usize,
    /// Edge-to-single-DOF mapping (P2 only). Empty for other orders.
    /// Each canonical edge key maps to its midpoint DOF.
    pub edge_dof_map: HashMap<EdgeKey, DofId>,
    /// Edge-to-2-DOF mapping (P3 only). Empty for other orders.
    /// Ordered [near_first_vertex, near_second_vertex].
    pub edge_dof2_map: HashMap<EdgeKey, [DofId; 2]>,
    /// Edge-to-N-DOFs mapping for general order p.
    /// Each canonical edge key maps to (p-1) DOFs, ordered from near-first-vertex
    /// to near-second-vertex.
    pub edge_pk_map: HashMap<EdgeKey, Vec<DofId>>,
    /// Face-to-N-DOFs mapping for 3D general order p (triangular faces).
    /// For p ≥ 3, each canonical face key maps to (p-1)(p-2)/2 DOFs.
    pub face_pk_map: HashMap<FaceKey, Vec<DofId>>,
    /// Quadrilateral-face-to-N-DOFs mapping for 3D general order p.
    /// For p ≥ 3, each canonical quad face key maps to (p-1)² DOFs.
    pub quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>>,
    /// Index at which bubble DOFs start (P3 only). Equal to `n_dofs` for P1/P2.
    pub bubble_dof_start: usize,
    /// Number of volume-interior DOFs per element (for p ≥ 4 in 3D, p ≥ 3 in 2D).
    pub n_volume_dofs: usize,
    /// Per-element polynomial orders for variable-order p-refinement.
    /// `None` for uniform-order DofManagers, `Some(orders)` for variable order.
    pub elem_orders: Option<Vec<u8>>,
}

impl DofManager {
    /// Build the DOF map for a mesh with given polynomial order.
    ///
    /// Currently supports:
    /// - Any mesh with `order = 1` (vertex DOFs), including mixed-element meshes.
    /// - 2-D triangular meshes (`Tri3`) with `order = 2` or `order = 3`.
    /// - 3-D tetrahedral meshes (`Tet4`) with `order = 2` or `order = 3`.
    /// - Any order `>= 4` on simplicial meshes via the general `build_pk` path.
    ///
    /// # Panics
    /// Panics if the requested order is unsupported for the mesh type.
    pub fn new<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let topo_dim = mesh.topological_dim() as usize;
        match order {
            1 => Self::build_p1(mesh),
            2 => {
                if topo_dim == 3 {
                    if mesh.n_elements() > 0 {
                        let npe = mesh.element_nodes(0).len();
                        match npe {
                            6 => Self::build_p2_prism(mesh),
                            5 => Self::build_p2_pyramid(mesh),
                            8 => Self::build_q2_hex(mesh),
                            _ => Self::build_p2_tet(mesh),
                        }
                    } else {
                        Self::build_p2_tet(mesh)
                    }
                } else if mesh.n_elements() > 0
                    && mesh.element_nodes(0).len() == 4
                    && topo_dim == 2
                {
                    Self::build_q2_quad(mesh)
                } else {
                    Self::build_p2(mesh)
                }
            }
            3 => {
                if topo_dim == 3 && mesh.n_elements() > 0 {
                    let npe = mesh.element_nodes(0).len();
                    if npe == 6 { return Self::build_p3_prism(mesh); }
                    if npe == 5 { return Self::build_p3_pyramid(mesh); }
                }
                // Quad Q3 / Hex Q3 via general pk path
                if mesh.n_elements() > 0 {
                    let npe = mesh.element_nodes(0).len();
                    if npe == 4 && topo_dim == 2 { return Self::build_pk_quad(mesh, order); }
                    if npe == 8 && topo_dim == 3 { return Self::build_pk_hex(mesh, order); }
                }
                Self::build_p3(mesh)
            }
            _ => {
                // General arbitrary-order path for p >= 4
                if mesh.n_elements() > 0 {
                    let npe = mesh.element_nodes(0).len();
                    if npe == 4 && topo_dim == 2 { return Self::build_pk_quad(mesh, order); }
                    if npe == 8 && topo_dim == 3 { return Self::build_pk_hex(mesh, order); }
                    if npe == 6 && topo_dim == 3 { return Self::build_prism_pk(mesh, order); }
                    if npe == 5 && topo_dim == 3 { return Self::build_pyramid_pk(mesh, order); }
                }
                Self::build_pk(mesh, order)
            }
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

    /// Radially project all DOF coordinates beyond `n_vertex_dofs`
    /// (edge midpoints, interior DOFs, …) onto the unit sphere.
    /// Call this after construction when the geometry is a spherical surface.
    pub fn snap_to_sphere(&mut self) {
        for i in self.n_vertex_dofs..self.n_dofs {
            let base = i * self.dim;
            let mut r2 = 0.0_f64;
            for d in 0..self.dim { r2 += self.dof_coords[base + d].powi(2); }
            let r = r2.sqrt().max(1e-30);
            for d in 0..self.dim { self.dof_coords[base + d] /= r; }
        }
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
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_nodes,
            n_volume_dofs: 0,
            elem_orders: None,
        }
    }

    // ─── P2 ──────────────────────────────────────────────────────────────────

    fn build_p2<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes  = mesh.n_nodes();
        let n_elems  = mesh.n_elements();
        let dim      = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 2, "P2 (Tri) DofManager requires 2-D elements");

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
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
        }
    }

    // ─── Q2 (biquadratic quad) ────────────────────────────────────────────────

    /// Build Q2 DOFs for a 2-D Quad4 mesh (9 DOFs per element):
    /// - Positions 0–3: vertex DOFs (same as node IDs)
    /// - Positions 4–7: edge midpoint DOFs (one per edge, shared between adjacent quads)
    ///   order: edge(n0,n1), edge(n1,n2), edge(n2,n3), edge(n3,n0)
    /// - Position 8: interior DOF (one per element, not shared)
    ///
    /// DOF ordering matches [`fem_element::QuadQ2`].
    fn build_q2_quad<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let dim     = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 2, "build_q2_quad requires 2-D elements");

        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_dof = n_nodes as DofId;

        // dofs_per_elem = 9: 4 corners + 4 edges + 1 interior.
        let dofs_per_elem = 9;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        // Phase 1: edge midpoint DOFs (positions 4–7).  MFEM numbers ALL
        // vertex DOFs, then ALL edge DOFs, then ALL interior (face) DOFs —
        // do NOT interleave interior DOFs with edge DOFs (that would shift
        // edge numbering vs MFEM and change the GS-smoother sweep order).
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert_eq!(ns.len(), 4, "build_q2_quad requires Quad4 elements");
            let (n0, n1, n2, n3) = (ns[0], ns[1], ns[2], ns[3]);
            let base = e as usize * dofs_per_elem;

            // Vertex DOFs (positions 0–3)
            dofs_flat[base]     = n0;
            dofs_flat[base + 1] = n1;
            dofs_flat[base + 2] = n2;
            dofs_flat[base + 3] = n3;

            // Edge midpoint DOFs (positions 4–7)
            // Ordering: bottom (n0,n1), right (n1,n2), top (n2,n3), left (n3,n0)
            let edges = [(n0, n1), (n1, n2), (n2, n3), (n3, n0)];
            for (k, &(a, b)) in edges.iter().enumerate() {
                let key = EdgeKey::new(a, b);
                let dof = *edge_map.entry(key).or_insert_with(|| {
                    let d = next_dof; next_dof += 1; d
                });
                dofs_flat[base + 4 + k] = dof;
            }
        }

        // Phase 2: interior (face) DOFs (position 8) — one per element, all
        // numbered after every edge DOF, matching MFEM's vertex→edge→face
        // ordering.
        let n_edge_dofs = edge_map.len();
        let mut interior_dof = (n_nodes + n_edge_dofs) as DofId;
        for e in 0..n_elems as u32 {
            let base = e as usize * dofs_per_elem;
            dofs_flat[base + 8] = interior_dof;
            interior_dof += 1;
        }

        let n_dofs = interior_dof as usize;

        // Build DOF coordinates.
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Vertex coords.
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            dof_coords[n as usize * dim .. n as usize * dim + dim].copy_from_slice(c);
        }

        // Edge midpoints.
        for (&EdgeKey(a, b), &dof_id) in &edge_map {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            let base = dof_id as usize * dim;
            for d in 0..dim { dof_coords[base + d] = 0.5 * (ca[d] + cb[d]); }
        }

        // Interior DOFs: element centroids.
        for e in 0..n_elems as u32 {
            let base = e as usize * dofs_per_elem;
            let interior_dof = dofs_flat[base + 8] as usize;
            let ns = mesh.element_nodes(e);
            let centroid_base = interior_dof * dim;
            for d in 0..dim {
                dof_coords[centroid_base + d] = ns.iter()
                    .map(|&n| mesh.node_coords(n)[d])
                    .sum::<f64>() / ns.len() as f64;
            }
        }

        let n_edge_dofs = edge_map.len();
        DofManager {
            order: 2, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes, edge_dof_map: edge_map,
            edge_dof2_map: HashMap::new(),
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_nodes + n_edge_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
        }
    }

    // ─── Pk (general order) ─────────────────────────────────────────────────

    fn build_p3<M: MeshTopology>(mesh: &M) -> Self {
        let dim = mesh.dim() as usize;
        match dim {
            2 => Self::build_p3_tri(mesh),
            3 => Self::build_p3_tet(mesh),
            _ => panic!("P3 DofManager only supports 2-D and 3-D meshes, got dim={dim}"),
        }
    }

    fn build_p3_tri<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes  = mesh.n_nodes();
        let n_elems  = mesh.n_elements();
        let dim      = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 2, "build_p3_tri requires 2-D elements");

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
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start,
            n_volume_dofs: 0,
            elem_orders: None,
        }
    }

    // ─── P3 (3-D Tet) ─────────────────────────────────────────────────────────

    /// Build a P3 DOF manager for a 3-D tetrahedral mesh.
    ///
    /// 20 DOFs per tet:
    /// - 4 vertex DOFs
    /// - 12 edge DOFs (2 per edge × 6 edges)
    /// - 4 face DOFs (1 per face × 4 faces)
    ///
    /// DOF ordering per element matches [`fem_element::TetP3`].
    fn build_p3_tet<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let dim     = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 3, "build_p3_tet requires 3-D elements");

        // DOF layout per element (20):
        //   0-3    → vertex DOFs (node IDs)
        //   4,5    → edge(v0→v1): near v0, near v1
        //   6,7    → edge(v0→v2): near v0, near v2
        //   8,9    → edge(v0→v3): near v0, near v3
        //   10,11  → edge(v1→v2): near v1, near v2
        //   12,13  → edge(v1→v3): near v1, near v3
        //   14,15  → edge(v2→v3): near v2, near v3
        //   16     → face(v0,v1,v2)
        //   17     → face(v0,v1,v3)
        //   18     → face(v0,v2,v3)
        //   19     → face(v1,v2,v3)
        let dofs_per_elem = 20;

        // ── Pass 1: enumerate edges (2 DOFs each) ───────────────────────────
        fn get_edge_dofs(a: NodeId, b: NodeId, next: &mut DofId, map: &mut HashMap<EdgeKey, [DofId; 2]>) -> [DofId; 2] {
            let key = EdgeKey::new(a, b);
            let pair = *map.entry(key).or_insert_with(|| {
                let d0 = *next; *next += 1;
                let d1 = *next; *next += 1;
                [d0, d1]
            });
            if a == key.0 { [pair[0], pair[1]] } else { [pair[1], pair[0]] }
        }

        let mut edge2_map: HashMap<EdgeKey, [DofId; 2]> = HashMap::new();
        let mut next_dof  = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns  = mesh.element_nodes(e);
            assert!(ns.len() >= 4, "TetP3 requires 4-node tetrahedra");
            let (n0, n1, n2, n3) = (ns[0], ns[1], ns[2], ns[3]);
            let base = e as usize * dofs_per_elem;

            dofs_flat[base]     = n0;
            dofs_flat[base + 1] = n1;
            dofs_flat[base + 2] = n2;
            dofs_flat[base + 3] = n3;

            let [d4,  d5]  = get_edge_dofs(n0, n1, &mut next_dof, &mut edge2_map);
            let [d6,  d7]  = get_edge_dofs(n0, n2, &mut next_dof, &mut edge2_map);
            let [d8,  d9]  = get_edge_dofs(n0, n3, &mut next_dof, &mut edge2_map);
            let [d10, d11] = get_edge_dofs(n1, n2, &mut next_dof, &mut edge2_map);
            let [d12, d13] = get_edge_dofs(n1, n3, &mut next_dof, &mut edge2_map);
            let [d14, d15] = get_edge_dofs(n2, n3, &mut next_dof, &mut edge2_map);

            dofs_flat[base + 4]  = d4;   dofs_flat[base + 5]  = d5;
            dofs_flat[base + 6]  = d6;   dofs_flat[base + 7]  = d7;
            dofs_flat[base + 8]  = d8;   dofs_flat[base + 9]  = d9;
            dofs_flat[base + 10] = d10;  dofs_flat[base + 11] = d11;
            dofs_flat[base + 12] = d12;  dofs_flat[base + 13] = d13;
            dofs_flat[base + 14] = d14;  dofs_flat[base + 15] = d15;
            // Face DOFs assigned in pass 2.
        }

        // ── Pass 2: enumerate faces (1 DOF each) ────────────────────────────
        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();

        for e in 0..n_elems as u32 {
            let ns  = mesh.element_nodes(e);
            let (n0, n1, n2, n3) = (ns[0], ns[1], ns[2], ns[3]);
            let base = e as usize * dofs_per_elem;

            // Faces: (v0,v1,v2), (v0,v1,v3), (v0,v2,v3), (v1,v2,v3)
            let faces = [
                (n0, n1, n2),
                (n0, n1, n3),
                (n0, n2, n3),
                (n1, n2, n3),
            ];
            for (k, &(a, b, c)) in faces.iter().enumerate() {
                let key = FaceKey::new(a, b, c);
                let dof = *face_map.entry(key).or_insert_with(|| {
                    let d = next_dof;
                    next_dof += 1;
                    d
                });
                dofs_flat[base + 16 + k] = dof;
            }
        }

        let n_dofs = next_dof as usize;
        let bubble_dof_start = n_dofs; // no volume bubble for TetP3

        // ── Build DOF coordinates ────────────────────────────────────────────
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Vertex coordinates.
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let base = n as usize * dim;
            dof_coords[base .. base + dim].copy_from_slice(c);
        }

        // Edge DOF coordinates (1/3 and 2/3 along each edge).
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

        // Face DOF coordinates: use face_map + face node lookup for correctness.
        {
            let mut face_nodes_map: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
            for e in 0..n_elems as u32 {
                let ns  = mesh.element_nodes(e);
                let (n0, n1, n2, n3) = (ns[0], ns[1], ns[2], ns[3]);
                for &(a, b, c) in &[(n0,n1,n2),(n0,n1,n3),(n0,n2,n3),(n1,n2,n3)] {
                    face_nodes_map.entry(FaceKey::new(a,b,c)).or_insert([a,b,c]);
                }
            }
            for (&key, &dof_id) in &face_map {
                let nodes = face_nodes_map[&key];
                let base  = dof_id as usize * dim;
                for d in 0..dim {
                    dof_coords[base + d] = nodes.iter().map(|&n| mesh.node_coords(n)[d]).sum::<f64>() / 3.0;
                }
            }
        }

        // Convert face_map (FaceKey -> DofId) into face_pk_map (FaceKey -> Vec<DofId>)
        // so that boundary_dofs() can find face-interior DOFs on boundary faces.
        let face_pk_map: HashMap<FaceKey, Vec<DofId>> = face_map.into_iter()
            .map(|(k, d)| (k, vec![d]))
            .collect();

        DofManager {
            order: 3, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(),
            edge_dof2_map: edge2_map,
            edge_pk_map: HashMap::new(),
            face_pk_map,
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start,
            n_volume_dofs: 0,
            elem_orders: None,
        }
    }

    // ─── P2 (3-D Tet) ─────────────────────────────────────────────────────────

    fn build_p2_tet<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes  = mesh.n_nodes();
        let n_elems  = mesh.n_elements();
        let dim      = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 3, "build_p2_tet requires 3-D elements");

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
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
        }
    }

    // ─── P2 (3-D Prism6) ──────────────────────────────────────────────────────

    /// Q2 (trilinear-biquadratic tensor product) DOFs for a 3-D Hex8 mesh
    /// — 27 DOFs per element: 8 vertices + 12 edge midpoints + 6 face centers
    /// + 1 volume center.
    ///
    /// DOF ordering matches [`fem_element::HexQk`] at order 2:
    ///   [0..8) vertices, [8..20) edges, [20..26) faces, [26] volume,
    /// where the 12 edges and 6 faces use HexQk's enumeration.
    fn build_q2_hex<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 3, "build_q2_hex requires 3-D elements");

        let dofs_per_elem = 27;
        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut qface_map: HashMap<QuadFaceKey, DofId> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        // HexQk edge enumeration (vertex-index pairs), see HexQk::node_to_dof.
        const EDGES: [(usize, usize); 12] = [
            (1, 5), (2, 6), (3, 7), (0, 4), (0, 3), (1, 2),
            (5, 6), (4, 7), (0, 1), (3, 2), (7, 6), (4, 5),
        ];
        // HexQk face enumeration (vertex-index quads), face order:
        // xmin, xmax, ymin, ymax, zmin, zmax.
        const FACES: [[usize; 4]; 6] = [
            [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4],
            [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7],
        ];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert_eq!(ns.len(), 8, "build_q2_hex requires 8-node hexahedra");
            let base = e as usize * dofs_per_elem;

            // Vertices (positions 0..8)
            for (k, &n) in ns.iter().enumerate() {
                dofs_flat[base + k] = n;
            }

            // Edge midpoints (positions 8..20)
            for (k, &(a, b)) in EDGES.iter().enumerate() {
                let key = EdgeKey::new(ns[a], ns[b]);
                let dof = *edge_map.entry(key).or_insert_with(|| {
                    let d = next_dof; next_dof += 1; d
                });
                dofs_flat[base + 8 + k] = dof;
            }

            // Face centers (positions 20..26)
            for (k, quad) in FACES.iter().enumerate() {
                let key = QuadFaceKey::new(ns[quad[0]], ns[quad[1]], ns[quad[2]], ns[quad[3]]);
                let dof = *qface_map.entry(key).or_insert_with(|| {
                    let d = next_dof; next_dof += 1; d
                });
                dofs_flat[base + 20 + k] = dof;
            }

            // Volume center (position 26) — one per element.
            dofs_flat[base + 26] = next_dof;
            next_dof += 1;
        }

        let n_dofs = next_dof as usize;
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Vertex coordinates.
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            dof_coords[n as usize * dim .. n as usize * dim + dim].copy_from_slice(c);
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
        // Face centers (average of the 4 corners).
        {
            let mut face_nodes: HashMap<QuadFaceKey, [NodeId; 4]> = HashMap::new();
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                for quad in FACES.iter() {
                    face_nodes
                        .entry(QuadFaceKey::new(ns[quad[0]], ns[quad[1]], ns[quad[2]], ns[quad[3]]))
                        .or_insert([ns[quad[0]], ns[quad[1]], ns[quad[2]], ns[quad[3]]]);
                }
            }
            for (&key, &dof_id) in &qface_map {
                let nodes = face_nodes[&key];
                let base = dof_id as usize * dim;
                for d in 0..dim {
                    dof_coords[base + d] = nodes.iter().map(|&n| mesh.node_coords(n)[d]).sum::<f64>() / 4.0;
                }
            }
        }
        // Volume centers (average of the 8 corners).
        for e in 0..n_elems as u32 {
            let base = e as usize * dofs_per_elem;
            let vol_dof = dofs_flat[base + 26] as usize;
            let ns = mesh.element_nodes(e);
            let vbase = vol_dof * dim;
            for d in 0..dim {
                dof_coords[vbase + d] = ns.iter().map(|&n| mesh.node_coords(n)[d]).sum::<f64>() / 8.0;
            }
        }

        DofManager {
            order: 2, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: edge_map,
            edge_dof2_map: HashMap::new(),
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
        }
    }

    fn build_p2_prism<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 3, "build_p2_prism requires 3-D elements");

        let dofs_per_elem = 18;
        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 6, "build_p2_prism requires 6-node prisms");
            let (n0,n1,n2,n3,n4,n5) = (ns[0],ns[1],ns[2],ns[3],ns[4],ns[5]);
            let base = e as usize * dofs_per_elem;

            dofs_flat[base]=n0; dofs_flat[base+1]=n1; dofs_flat[base+2]=n2;
            dofs_flat[base+3]=n3; dofs_flat[base+4]=n4; dofs_flat[base+5]=n5;

            let edges = [(n0,n1),(n1,n2),(n0,n2),(n3,n4),(n4,n5),(n3,n5),(n0,n3),(n1,n4),(n2,n5)];
            for (k, &(a,b)) in edges.iter().enumerate() {
                let key = EdgeKey::new(a,b);
                let dof = *edge_map.entry(key).or_insert_with(||{let d=next_dof;next_dof+=1;d});
                dofs_flat[base+6+k]=dof;
            }

            for (k, &(a,b,c)) in [(n0,n1,n2),(n3,n4,n5)].iter().enumerate() {
                let key = FaceKey::new(a,b,c);
                let dof = *face_map.entry(key).or_insert_with(||{let d=next_dof;next_dof+=1;d});
                dofs_flat[base+14+k]=dof;
            }
        }

        let n_dofs = next_dof as usize;
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];
        for n in 0..n_nodes as u32 { let c=mesh.node_coords(n); let b=n as usize*dim; dof_coords[b..b+dim].copy_from_slice(c); }
        for (&EdgeKey(a,b),&dof_id) in &edge_map {
            let ca=mesh.node_coords(a); let cb=mesh.node_coords(b);
            let b=dof_id as usize*dim; for d in 0..dim { dof_coords[b+d]=0.5*(ca[d]+cb[d]); }
        }
        for (&FaceKey(a,b,c),&dof_id) in &face_map {
            let off=dof_id as usize*dim; for d in 0..dim { dof_coords[off+d]=(mesh.node_coords(a)[d]+mesh.node_coords(b)[d]+mesh.node_coords(c)[d])/3.0; }
        }

        DofManager {
            order:2, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets:None, dof_coords, dim,
            n_vertex_dofs:n_nodes,
            edge_dof_map:edge_map, edge_dof2_map:HashMap::new(),
            edge_pk_map:HashMap::new(), face_pk_map:HashMap::new(),
            quad_face_pk_map:HashMap::new(),
            bubble_dof_start:n_dofs, n_volume_dofs:0, elem_orders:None,
        }
    }

    // ─── P2 (3-D Pyramid5) ─────────────────────────────────────────────────────

    fn build_p2_pyramid<M: MeshTopology>(mesh: &M) -> Self {
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        assert_eq!(mesh.topological_dim() as usize, 3, "build_p2_pyramid requires 3-D elements");

        let dofs_per_elem = 14;
        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 5, "build_p2_pyramid requires 5-node pyramids");
            let (n0,n1,n2,n3,n4)=(ns[0],ns[1],ns[2],ns[3],ns[4]);
            let base = e as usize * dofs_per_elem;

            dofs_flat[base]=n0; dofs_flat[base+1]=n1; dofs_flat[base+2]=n2;
            dofs_flat[base+3]=n3; dofs_flat[base+4]=n4;

            let edges = [(n0,n1),(n1,n2),(n2,n3),(n3,n0),(n0,n4),(n1,n4),(n2,n4),(n3,n4)];
            for (k, &(a,b)) in edges.iter().enumerate() {
                let key = EdgeKey::new(a,b);
                let dof = *edge_map.entry(key).or_insert_with(||{let d=next_dof;next_dof+=1;d});
                dofs_flat[base+5+k]=dof;
            }
            dofs_flat[base+13]=next_dof; next_dof+=1;
        }

        let n_dofs = next_dof as usize;
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];
        for n in 0..n_nodes as u32 { let c=mesh.node_coords(n); let b=n as usize*dim; dof_coords[b..b+dim].copy_from_slice(c); }
        for (&EdgeKey(a,b),&dof_id) in &edge_map {
            let ca=mesh.node_coords(a); let cb=mesh.node_coords(b);
            let b=dof_id as usize*dim; for d in 0..dim { dof_coords[b+d]=0.5*(ca[d]+cb[d]); }
        }
        for e in 0..n_elems as u32 {
            let ns=mesh.element_nodes(e);
            let quad_centroid_dof=e as usize*dofs_per_elem+13;
            let dof_id=dofs_flat[quad_centroid_dof] as usize;
            let b=dof_id*dim; for d in 0..dim {
                dof_coords[b+d]=ns[..4].iter().map(|&n|mesh.node_coords(n)[d]).sum::<f64>()/4.0;
            }
        }

        DofManager {
            order:2, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets:None, dof_coords, dim,
            n_vertex_dofs:n_nodes,
            edge_dof_map:edge_map, edge_dof2_map:HashMap::new(),
            edge_pk_map:HashMap::new(), face_pk_map:HashMap::new(),
            quad_face_pk_map:HashMap::new(),
            bubble_dof_start:n_dofs, n_volume_dofs:0, elem_orders:None,
        }
    }

    // ─── P3 (3-D Prism6) — 40 DOFs per element ────────────────────────────────

    fn build_p3_prism<M: MeshTopology>(mesh: &M) -> Self {
        let dim=3usize; let n_nodes=mesh.n_nodes(); let n_elems=mesh.n_elements();
        let dofs_per_elem=40;
        let mut edge2_map:HashMap<EdgeKey,[DofId;2]>=HashMap::new();
        let mut face_map:HashMap<FaceKey,DofId>=HashMap::new();
        let mut qface_map:HashMap<QuadFaceKey,Vec<DofId>>=HashMap::new();
        let mut next_dof=n_nodes as DofId;
        let mut dofs_flat=vec![0u32;n_elems*dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns=mesh.element_nodes(e);
            let (n0,n1,n2,n3,n4,n5)=(ns[0],ns[1],ns[2],ns[3],ns[4],ns[5]);
            let base=e as usize*40;

            // Layer 0 (bottom): TriP3 DOFs 0..9
            dofs_flat[base]=n0; dofs_flat[base+1]=n1; dofs_flat[base+2]=n2;
            for (k,&(a,b)) in [(n0,n1),(n1,n2),(n0,n2)].iter().enumerate() {
                let key=EdgeKey::new(a,b);
                let pair = *edge2_map.entry(key).or_insert_with(||{let d0=next_dof;next_dof+=1;let d1=next_dof;next_dof+=1;[d0,d1]});
                let (d0,d1)=if a==key.0{(pair[0],pair[1])}else{(pair[1],pair[0])};
                dofs_flat[base+3+2*k]=d0; dofs_flat[base+4+2*k]=d1;
            }
            let fk=FaceKey::new(n0,n1,n2);
            dofs_flat[base+9] = *face_map.entry(fk).or_insert_with(||{let d=next_dof;next_dof+=1;d});

            // Layer 1 (ξ=1/3): DOFs 10..19
            for (k,&(a,b)) in [(n0,n3),(n1,n4),(n2,n5)].iter().enumerate() {
                let key=EdgeKey::new(a,b);
                let pair = *edge2_map.entry(key).or_insert_with(||{let d0=next_dof;next_dof+=1;let d1=next_dof;next_dof+=1;[d0,d1]});
                let (d0,_)=if a==key.0{(pair[0],pair[1])}else{(pair[1],pair[0])};
                dofs_flat[base+10+k]=d0;
            }
            for (k,&(a,b,c,d)) in [(n0,n1,n4,n3),(n1,n2,n5,n4),(n2,n0,n3,n5)].iter().enumerate() {
                let qk=QuadFaceKey::new(a,b,c,d);
                let dofs=qface_map.entry(qk).or_insert_with(||{(0..4).map(|_|{let d=next_dof;next_dof+=1;d}).collect()});
                dofs_flat[base+13+k]=dofs[0];
            }
            dofs_flat[base+19]=next_dof; next_dof+=1;

            // Layer 2 (ξ=2/3): DOFs 20..29
            for (k,&(a,b)) in [(n0,n3),(n1,n4),(n2,n5)].iter().enumerate() {
                let key=EdgeKey::new(a,b);
                let pair=*edge2_map.get(&key).unwrap();
                let (_,d1)=if a==key.0{(pair[0],pair[1])}else{(pair[1],pair[0])};
                dofs_flat[base+20+k]=d1;
            }
            for (k,&(a,b,c,d)) in [(n0,n1,n4,n3),(n1,n2,n5,n4),(n2,n0,n3,n5)].iter().enumerate() {
                let qk=QuadFaceKey::new(a,b,c,d);
                let dofs=qface_map.get(&qk).unwrap();
                dofs_flat[base+23+k]=dofs[2];
            }
            dofs_flat[base+29]=next_dof; next_dof+=1;

            // Layer 3 (top): DOFs 30..39
            dofs_flat[base+30]=n3; dofs_flat[base+31]=n4; dofs_flat[base+32]=n5;
            for (k,&(a,b)) in [(n3,n4),(n4,n5),(n3,n5)].iter().enumerate() {
                let key=EdgeKey::new(a,b);
                let pair=*edge2_map.entry(key).or_insert_with(||{let d0=next_dof;next_dof+=1;let d1=next_dof;next_dof+=1;[d0,d1]});
                let (d0,d1)=if a==key.0{(pair[0],pair[1])}else{(pair[1],pair[0])};
                dofs_flat[base+33+2*k]=d0; dofs_flat[base+34+2*k]=d1;
            }
            let fk2=FaceKey::new(n3,n4,n5);
            dofs_flat[base+39] = *face_map.entry(fk2).or_insert_with(||{let d=next_dof;next_dof+=1;d});
        }

        let n_dofs=next_dof as usize;
        let mut dof_coords=vec![0.0_f64;n_dofs*dim];
        for n in 0..n_nodes as u32{let c=mesh.node_coords(n);let b=n as usize*dim;dof_coords[b..b+dim].copy_from_slice(c);}
        for(&EdgeKey(a,b),&[d0,d1])in&edge2_map{let ca=mesh.node_coords(a);let cb=mesh.node_coords(b);
            let b0=d0 as usize*dim;let b1=d1 as usize*dim;for d in 0..dim{dof_coords[b0+d]=(2.0*ca[d]+cb[d])/3.0;dof_coords[b1+d]=(ca[d]+2.0*cb[d])/3.0;}}
        for(&FaceKey(a,b,c),&dof_id)in&face_map{let off=dof_id as usize*dim;for d in 0..dim{dof_coords[off+d]=(mesh.node_coords(a)[d]+mesh.node_coords(b)[d]+mesh.node_coords(c)[d])/3.0;}}
        for(key,dofs)in&qface_map{let n4=[key.0,key.1,key.2,key.3];let c4=[0,1,2,3].map(|i|mesh.node_coords(n4[i]));
            for ix in 0..2{for iy in 0..2{let dof_id=dofs[iy*2+ix];let b=dof_id as usize*dim;
                let tx=(ix+1)as f64/3.0;let ty=(iy+1)as f64/3.0;
                for d in 0..dim{dof_coords[b+d]=(1.0-tx)*(1.0-ty)*c4[0][d]+tx*(1.0-ty)*c4[1][d]+tx*ty*c4[2][d]+(1.0-tx)*ty*c4[3][d];}}}
        }
        for e in 0..n_elems as u32{let ns=mesh.element_nodes(e);let base=e as usize*40;
            let c5=[0,1,2,3,4,5].map(|i|mesh.node_coords(ns[i]));
            for vi in 0..2{let dof_id=dofs_flat[base+19+10*vi]as usize;let xi=(vi+1)as f64/3.0;let b=dof_id*dim;
                for d in 0..dim{let bottom=(c5[0][d]+c5[1][d]+c5[2][d])/3.0;let top=(c5[3][d]+c5[4][d]+c5[5][d])/3.0;
                    dof_coords[b+d]=(1.0-xi)*bottom+xi*top;}}}

        DofManager{order:3,n_dofs,dofs_flat,dofs_per_elem,
            elem_dof_offsets:None,dof_coords,dim,
            n_vertex_dofs:n_nodes,
            edge_dof_map:HashMap::new(),edge_dof2_map:edge2_map,
            edge_pk_map:HashMap::new(),face_pk_map:HashMap::new(),
            quad_face_pk_map:qface_map,
            bubble_dof_start:n_dofs,n_volume_dofs:2,elem_orders:None,}
    }

    // ─── P3 (3-D Pyramid5) — 30 DOFs per element ──────────────────────────────

    fn build_p3_pyramid<M: MeshTopology>(mesh: &M) -> Self {
        let dim=3usize; let n_nodes=mesh.n_nodes(); let n_elems=mesh.n_elements();
        let dofs_per_elem=30;
        let mut edge2_map:HashMap<EdgeKey,[DofId;2]>=HashMap::new();
        let mut qface_map:HashMap<QuadFaceKey,Vec<DofId>>=HashMap::new();
        let mut next_dof=n_nodes as DofId;
        let mut dofs_flat=vec![0u32;n_elems*dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns=mesh.element_nodes(e);
            let (n0,n1,n2,n3,n4)=(ns[0],ns[1],ns[2],ns[3],ns[4]);
            let base=e as usize*dofs_per_elem;

            // Use PyramidPk ref element coordinates to map each DOF position
            let pyr_ref=fem_element::lagrange::PyramidPk::new(3);
            let rc=pyr_ref.dof_coords();

            for ref_i in 0..30 {
                let x=rc[ref_i][0];let y=rc[ref_i][1];let z=rc[ref_i][2];
                let eps=1e-12;
                let on_z0=z.abs()<eps;let on_z1=(z-1.0).abs()<eps;
                let on_x0=x.abs()<eps;let on_x1=(x-1.0+z).abs()<eps;
                let on_y0=y.abs()<eps;let on_y1=(y-1.0+z).abs()<eps;
                let nb=[on_z0,on_z1,on_x0,on_x1,on_y0,on_y1].iter().filter(|&&b|b).count();

                dofs_flat[base+ref_i] = if nb>=3&&on_z1{n4}
                else if nb>=3{match(on_x0,on_y0){(true,true)=>n0,(false,true)=>n1,(false,false)=>n2,_=>n3}}
                else if nb==2&&on_z0{
                    let ek=if on_x0{EdgeKey::new(n3,n0)}else if on_x1{EdgeKey::new(n1,n2)}
                           else if on_y0{EdgeKey::new(n0,n1)}else{EdgeKey::new(n2,n3)};
                    let p2=*edge2_map.entry(ek).or_insert_with(||{let d0=next_dof;next_dof+=1;let d1=next_dof;next_dof+=1;[d0,d1]});
                    let local=(x*3.0).round()as usize;
                    if local==0{p2[0]}else{p2[1]}
                }else if nb==2{
                    let ek=EdgeKey::new(n4,if on_x0&&on_y0{n0}else if on_x1&&on_y0{n1}else if on_x1&&on_y1{n2}else{n3});
                    let p2=*edge2_map.entry(ek).or_insert_with(||{let d0=next_dof;next_dof+=1;let d1=next_dof;next_dof+=1;[d0,d1]});
                    let local=(z*3.0).round()as usize;
                    if local==0{p2[0]}else{p2[1]}
                }else if nb==1&&on_z0{
                    let qk=QuadFaceKey::new(n0,n1,n2,n3);
                    let dofs=qface_map.entry(qk).or_insert_with(||{(0..4).map(|_|{let d=next_dof;next_dof+=1;d}).collect()});
                    let ix=((x*3.0).round()as usize).saturating_sub(1);
                    let iy=((y*3.0).round()as usize).saturating_sub(1);
                    dofs[iy.min(1)*2+ix.min(1)]
                }else{
                    let d=next_dof;next_dof+=1;d
                };
            }
        }

        let n_dofs=next_dof as usize;
        let mut dof_coords=vec![0.0_f64;n_dofs*dim];
        for n in 0..n_nodes as u32{let c=mesh.node_coords(n);let b=n as usize*dim;dof_coords[b..b+dim].copy_from_slice(c);}
        for(&EdgeKey(a,b),&[d0,d1])in&edge2_map{let ca=mesh.node_coords(a);let cb=mesh.node_coords(b);
            let b0=d0 as usize*dim;let b1=d1 as usize*dim;for d in 0..dim{dof_coords[b0+d]=(2.0*ca[d]+cb[d])/3.0;dof_coords[b1+d]=(ca[d]+2.0*cb[d])/3.0;}}
        for(key,dofs)in&qface_map{let n4=[key.0,key.1,key.2,key.3];let c4=[0,1,2,3].map(|i|mesh.node_coords(n4[i]));
            for ix in 0..2{for iy in 0..2{let dof_id=dofs[iy*2+ix];let b=dof_id as usize*dim;
                let tx=(ix+1)as f64/3.0;let ty=(iy+1)as f64/3.0;
                for d in 0..dim{dof_coords[b+d]=(1.0-tx)*(1.0-ty)*c4[0][d]+tx*(1.0-ty)*c4[1][d]+tx*ty*c4[2][d]+(1.0-tx)*ty*c4[3][d];}}}
        }

        DofManager{order:3,n_dofs,dofs_flat,dofs_per_elem,
            elem_dof_offsets:None,dof_coords,dim,
            n_vertex_dofs:n_nodes,
            edge_dof_map:HashMap::new(),edge_dof2_map:edge2_map,
            edge_pk_map:HashMap::new(),face_pk_map:HashMap::new(),
            quad_face_pk_map:qface_map,
            bubble_dof_start:n_dofs,n_volume_dofs:0,elem_orders:None,}
    }

    // ─── Pk for 2-D Quad (tensor-product Qk) ──────────────────────────────────
    //
    // DOF ordering per element (matching QuadQk):
    //   [0..3] = vertices in CCW order
    //   [4..4+(p-1)*4) = edge DOFs: bottom, right, top, left, (p-1) per edge
    //   remaining = interior DOFs in tensor-product (p-1)×(p-1) layout

    fn build_pk_quad<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let p = order as usize;
        assert!(p >= 3, "build_pk_quad: order must be >= 3");
        // Use mesh spatial dimension (3 for surface meshes, 2 for planar)
        let dim = mesh.dim() as usize;
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let edge_dofs_per = p - 1;
        let interior_dofs_per = (p - 1) * (p - 1);
        let n_verts = 4;
        let n_edges = 4;
        let dofs_per_elem = n_verts + n_edges * edge_dofs_per + interior_dofs_per;
        let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 4);
            let base = e as usize * dofs_per_elem;
            dofs_flat[base] = ns[0]; dofs_flat[base + 1] = ns[1];
            dofs_flat[base + 2] = ns[2]; dofs_flat[base + 3] = ns[3];
            let edges = [(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0])];
            let mut off = 4;
            for &(a, b) in &edges {
                let ed = get_edge_dofs_pk(a, b, &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                for (k, &d) in ed.iter().enumerate() { dofs_flat[base + off + k] = d; }
                off += edge_dofs_per;
            }
            for _ in 0..interior_dofs_per {
                dofs_flat[base + off] = next_dof; next_dof += 1; off += 1;
            }
        }

        let n_dofs = next_dof as usize;
        let mut dof_coords = vec![0.0; n_dofs * dim];
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let base = n as usize * dim;
            dof_coords[base..base + dim].copy_from_slice(c);
        }
        for (&EdgeKey(a, b), dofs) in &edge_pk_map {
            let ca = mesh.node_coords(a); let cb = mesh.node_coords(b);
            // Use Gauss-Lobatto-Legendre positions consistent with QuadQk
            let gll_nodes = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1).0;
            for (k, &did) in dofs.iter().enumerate() {
                let t_gll = 0.5 * (gll_nodes[k + 1] + 1.0); // map [-1,1] → [0,1]
                let base = did as usize * dim;
                for d in 0..dim { dof_coords[base + d] = (1.0 - t_gll) * ca[d] + t_gll * cb[d]; }
            }
        }
        // Interior DOFs: tensor-product coordinates using reference element
        if interior_dofs_per > 0 {
            use fem_element::lagrange::factory::{ref_elem, ElemType};
            let factory = ref_elem(ElemType::Quad, order);
            let ref_coords = factory.dof_coords();
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                // Read the actual global DOF ids from dofs_flat: interior DOF
                // ids are NOT a contiguous range (they interleave with edge
                // DOFs created by later elements).
                let ebase = e as usize * dofs_per_elem;
                for k in 0..interior_dofs_per {
                    let did = dofs_flat[ebase + n_verts + n_edges * edge_dofs_per + k];
                    let rc = &ref_coords[n_verts + n_edges * edge_dofs_per + k];
                    let c0 = mesh.node_coords(ns[0]); let c1 = mesh.node_coords(ns[1]);
                    let c2 = mesh.node_coords(ns[2]); let c3 = mesh.node_coords(ns[3]);
                    let u = rc[0]; let v = rc[1];
                    let base = did as usize * dim;
                    for d in 0..dim {
                        // Bilinear mapping on [0,1]² (QuadQk uses GLL nodes on [0,1]²)
                        dof_coords[base + d] = (1.0-u)*(1.0-v)*c0[d]
                            + u*(1.0-v)*c1[d]
                            + u*v*c2[d]
                            + (1.0-u)*v*c3[d];
                    }
                }
            }
        }

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem, elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(), edge_dof2_map: HashMap::new(), edge_pk_map,
            face_pk_map: HashMap::new(), quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_dofs, n_volume_dofs: 0, elem_orders: None,
        }
    }

    // ─── Pk for 3-D Hex (tensor-product Qk) ───────────────────────────────────

    fn build_pk_hex<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let p = order as usize;
        assert!(p >= 3, "build_pk_hex: order must be >= 3");
        let dim = 3usize;
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let edge_dofs_per = p - 1;
        let face_dofs_per = (p - 1) * (p - 1);
        let volume_dofs_per = (p - 1) * (p - 1) * (p - 1);
        let n_verts = 8;
        let n_edges = 12;
        let n_faces = 6;
        let dofs_per_elem = n_verts + n_edges * edge_dofs_per + n_faces * face_dofs_per + volume_dofs_per;
        let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        let mut quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 8);
            let base = e as usize * dofs_per_elem;
            dofs_flat[base..base + 8].copy_from_slice(&ns[..8]);
            let edges: [(usize, usize); 12] = [
                (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];
            let mut off = 8;
            for &(la, lb) in &edges {
                let ed = get_edge_dofs_pk(ns[la], ns[lb], &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                for (k, &d) in ed.iter().enumerate() { dofs_flat[base + off + k] = d; }
                off += edge_dofs_per;
            }
            let quad_faces: [(usize, usize, usize, usize); 6] = [
                (0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)];
            for &(la, lb, lc, ld) in &quad_faces {
                let key = QuadFaceKey::new(ns[la], ns[lb], ns[lc], ns[ld]);
                let fd = quad_face_pk_map.entry(key).or_insert_with(|| {
                    (0..face_dofs_per).map(|_| { let d = next_dof; next_dof += 1; d }).collect()
                });
                for (k, &d) in fd.iter().enumerate() { dofs_flat[base + off + k] = d; }
                off += face_dofs_per;
            }
            for _ in 0..volume_dofs_per {
                dofs_flat[base + off] = next_dof; next_dof += 1; off += 1;
            }
        }

        let n_dofs = next_dof as usize;
        let mut dof_coords = vec![0.0; n_dofs * dim];
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let base = n as usize * dim;
            dof_coords[base..base + dim].copy_from_slice(c);
        }
        for (&EdgeKey(a, b), dofs) in &edge_pk_map {
            let ca = mesh.node_coords(a); let cb = mesh.node_coords(b);
            for (k, &did) in dofs.iter().enumerate() {
                let t = (k + 1) as f64 / (edge_dofs_per + 1) as f64;
                let base = did as usize * dim;
                for d in 0..dim { dof_coords[base + d] = (1.0 - t) * ca[d] + t * cb[d]; }
            }
        }
        // Face + volume DOF coords from factory ref element (trilinear interpolation)
        if p >= 3 {
            use fem_element::lagrange::factory::{ref_elem, ElemType};
            let factory = ref_elem(ElemType::Hex, order);
            let ref_coords = factory.dof_coords();
            let face_vol_start = n_verts + n_edges * edge_dofs_per;
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                let c = [
                    mesh.node_coords(ns[0]), mesh.node_coords(ns[1]),
                    mesh.node_coords(ns[2]), mesh.node_coords(ns[3]),
                    mesh.node_coords(ns[4]), mesh.node_coords(ns[5]),
                    mesh.node_coords(ns[6]), mesh.node_coords(ns[7]),
                ];
                // Read the actual global DOF ids from dofs_flat: face/volume
                // DOF ids are NOT a contiguous range (they interleave with
                // edge DOFs created by later elements).
                let ebase = e as usize * dofs_per_elem;
                for k in 0..(n_faces * face_dofs_per + volume_dofs_per) {
                    let did = dofs_flat[ebase + face_vol_start + k];
                    let ri = n_verts + n_edges * edge_dofs_per + k;
                    let rc = &ref_coords[ri];
                    let (ex, ey, ez) = (rc[0], rc[1], rc[2]);
                    let mut xp = [0.0; 3];
                    for i in 0..8 {
                        let nx = if (i & 1) != 0 { (1.0 + ex) / 2.0 } else { (1.0 - ex) / 2.0 };
                        let ny = if (i & 2) != 0 { (1.0 + ey) / 2.0 } else { (1.0 - ey) / 2.0 };
                        let nz = if (i & 4) != 0 { (1.0 + ez) / 2.0 } else { (1.0 - ez) / 2.0 };
                        let ni = nx * ny * nz;
                        for d in 0..3 { xp[d] += ni * c[i][d]; }
                    }
                    let base = did as usize * dim;
                    dof_coords[base..base + 3].copy_from_slice(&xp);
                }
            }
        }

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem, elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(), edge_dof2_map: HashMap::new(), edge_pk_map,
            face_pk_map: HashMap::new(), quad_face_pk_map,
            bubble_dof_start: n_dofs, n_volume_dofs: 0, elem_orders: None,
        }
    }

    // ─── Pk for Prism (triangular prism) ──────────────────────────────────────

    /// General-order Lagrange DOF manager for triangular prism meshes.
    ///
    /// DOF ordering per element: 6 vertices → 9 edges → 2 tri faces → 3 quad faces → volume.
    fn build_prism_pk<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let p = order as usize;
        assert!(p >= 1, "build_prism_pk: order must be >= 1");
        let dim = 3usize;
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let edge_dofs_per = if p >= 2 { p - 1 } else { 0 };
        let tri_face_dofs_per = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
        let quad_face_dofs_per = if p >= 2 { (p - 1) * (p - 1) } else { 0 };
        let n_verts = 6;
        let n_edges = 9;
        let n_tri_faces = 2;
        let n_quad_faces = 3;
        let surface_dofs = n_verts + n_edges * edge_dofs_per
            + n_tri_faces * tri_face_dofs_per
            + n_quad_faces * quad_face_dofs_per;
        let total_ref = (p + 1) * (p + 1) * (p + 2) / 2;
        let volume_dofs_per = total_ref.saturating_sub(surface_dofs);
        let dofs_per_elem = surface_dofs + volume_dofs_per;

        let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        let mut face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
        let mut quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        // Vertex + edge + tri face DOFs
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 6);
            let base = e as usize * dofs_per_elem;

            dofs_flat[base..base + 6].copy_from_slice(&ns[..6]);
            let mut off = 6;

            if p >= 2 {
                let edges: [(usize, usize); 9] = [
                    (0, 1), (1, 2), (2, 0),
                    (3, 4), (4, 5), (5, 3),
                    (0, 3), (1, 4), (2, 5),
                ];
                for &(la, lb) in &edges {
                    let ed = get_edge_dofs_pk(ns[la], ns[lb], &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                    for (k, &d) in ed.iter().enumerate() { dofs_flat[base + off + k] = d; }
                    off += edge_dofs_per;
                }
            }
            if p >= 3 {
                for &(la, lb, lc) in &[(0, 1, 2), (3, 4, 5)] {
                    let fd = get_face_dofs_pk(ns[la], ns[lb], ns[lc], &mut next_dof, &mut face_pk_map, tri_face_dofs_per);
                    for (k, &d) in fd.iter().enumerate() { dofs_flat[base + off + k] = d; }
                    off += tri_face_dofs_per;
                }
            }
            if p >= 2 {
                let quad_faces: [(usize, usize, usize, usize); 3] = [
                    (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5),
                ];
                let qf_start = off;
                for &(la, lb, lc, ld) in &quad_faces {
                    let key = QuadFaceKey::new(ns[la], ns[lb], ns[lc], ns[ld]);
                    let fd = quad_face_pk_map.entry(key).or_insert_with(|| {
                        (0..quad_face_dofs_per).map(|_| { let d = next_dof; next_dof += 1; d }).collect()
                    });
                    for (k, &d) in fd.iter().enumerate() { dofs_flat[base + off + k] = d; }
                    off += quad_face_dofs_per;
                }
                let _ = qf_start; // quad face region marker
            }
            for _ in 0..volume_dofs_per {
                dofs_flat[base + off] = next_dof;
                next_dof += 1;
                off += 1;
            }
        }

        let n_dofs = next_dof as usize;

        // DOF coordinates
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let b = n as usize * dim;
            dof_coords[b..b + dim].copy_from_slice(c);
        }
        // Edges
        for (&EdgeKey(a, b), dofs) in &edge_pk_map {
            let ca = mesh.node_coords(a); let cb = mesh.node_coords(b);
            for (k, &did) in dofs.iter().enumerate() {
                let t = (k + 1) as f64 / (edge_dofs_per + 1) as f64;
                let base = did as usize * dim;
                for d in 0..dim { dof_coords[base + d] = (1.0 - t) * ca[d] + t * cb[d]; }
            }
        }
        // Tri faces
        if p >= 3 {
            let mut face_nodes: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                for &(a, b, c) in &[(ns[0], ns[1], ns[2]), (ns[3], ns[4], ns[5])] {
                    face_nodes.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
                }
            }
            for (key, dofs) in &face_pk_map {
                let [a, b, c] = face_nodes[key];
                let ca = mesh.node_coords(a); let cb = mesh.node_coords(b); let cc = mesh.node_coords(c);
                for (k, &did) in dofs.iter().enumerate() {
                    let base = did as usize * dim;
                    let t = (k + 1) as f64 / (dofs.len() + 1) as f64;
                    for d in 0..dim {
                        dof_coords[base + d] = (1.0 - t) * ca[d] + t * (cb[d] + cc[d]) / 2.0;
                    }
                }
            }
        }
        // Quad faces: bilinear interpolation
        if quad_face_dofs_per > 0 {
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                let quad_verts: [[&[f64]; 4]; 3] = [
                    [mesh.node_coords(ns[0]), mesh.node_coords(ns[1]), mesh.node_coords(ns[4]), mesh.node_coords(ns[3])],
                    [mesh.node_coords(ns[1]), mesh.node_coords(ns[2]), mesh.node_coords(ns[5]), mesh.node_coords(ns[4])],
                    [mesh.node_coords(ns[2]), mesh.node_coords(ns[0]), mesh.node_coords(ns[3]), mesh.node_coords(ns[5])],
                ];
                for _qf in 0..3 {
                    for row in 0..(p - 1) {
                        let xi = (row + 1) as f64 / p as f64;
                        for col in 0..(p - 1) {
                            let eta = (col + 1) as f64 / p as f64;
                            let elem_base = e as usize * dofs_per_elem;
                            let qf_idx = _qf;
                            let local_offset = 6 + 9 * edge_dofs_per + 2 * tri_face_dofs_per
                                + qf_idx * quad_face_dofs_per + row * (p - 1) + col;
                            let did = dofs_flat[elem_base + local_offset] as usize;
                            let dbase = did * dim;
                            let v = &quad_verts[qf_idx];
                            for d in 0..dim {
                                dof_coords[dbase + d] = (1.0 - xi) * (1.0 - eta) * v[0][d]
                                    + xi * (1.0 - eta) * v[1][d]
                                    + xi * eta * v[2][d]
                                    + (1.0 - xi) * eta * v[3][d];
                            }
                        }
                    }
                }
            }
        }
        // Volume: use PrismPk ref element for accurate coordinates
        if volume_dofs_per > 0 {
            let factory = fem_element::lagrange::PrismPk::new(p);
            let ref_coords = factory.dof_coords();
            let vol_start = n_nodes + edge_pk_map.len() * edge_dofs_per
                + face_pk_map.len() * tri_face_dofs_per
                + quad_face_pk_map.len() * quad_face_dofs_per;
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                let c = [
                    mesh.node_coords(ns[0]), mesh.node_coords(ns[1]), mesh.node_coords(ns[2]),
                    mesh.node_coords(ns[3]), mesh.node_coords(ns[4]), mesh.node_coords(ns[5]),
                ];
                for k in 0..volume_dofs_per {
                    let did = vol_start + e as usize * volume_dofs_per + k;
                    let ri = surface_dofs + k;
                    let rc = &ref_coords[ri];
                    let xi = rc[0]; let eta = rc[1]; let zeta = rc[2];
                    let lam0 = 1.0 - eta - zeta;
                    let dbase = did * dim;
                    for d in 0..dim {
                        let bottom = lam0 * c[0][d] + eta * c[1][d] + zeta * c[2][d];
                        let top = lam0 * c[3][d] + eta * c[4][d] + zeta * c[5][d];
                        dof_coords[dbase + d] = (1.0 - xi) * bottom + xi * top;
                    }
                }
            }
        }

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(), edge_dof2_map: HashMap::new(), edge_pk_map,
            face_pk_map, quad_face_pk_map,
            bubble_dof_start: n_dofs, n_volume_dofs: volume_dofs_per, elem_orders: None,
        }
    }

    // ─── Pk for Pyramid ─────────────────────────────────────────────────────

    /// General-order Lagrange DOF manager for pyramid meshes.
    ///
    /// DOF ordering per element: 5 vertices → 8 edges → 1 quad base → 4 tri sides → volume.
    fn build_pyramid_pk<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let p = order as usize;
        assert!(p >= 1, "build_pyramid_pk: order must be >= 1");
        let dim = 3usize;
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let edge_dofs_per = if p >= 2 { p - 1 } else { 0 };
        let quad_face_dofs_per = if p >= 2 { (p - 1) * (p - 1) } else { 0 };
        let tri_face_dofs_per = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
        let n_verts = 5;
        let n_edges = 8;
        let n_quad_faces = 1;
        let n_tri_faces = 4;
        let surface_dofs = n_verts + n_edges * edge_dofs_per
            + n_quad_faces * quad_face_dofs_per
            + n_tri_faces * tri_face_dofs_per;
        let total_ref = (p + 1) * (p + 2) * (2 * p + 3) / 6;
        let volume_dofs_per = total_ref.saturating_sub(surface_dofs);
        let dofs_per_elem = surface_dofs + volume_dofs_per;

        let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        let mut face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
        let mut quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 5);
            let base = e as usize * dofs_per_elem;

            dofs_flat[base..base + 5].copy_from_slice(&ns[..5]);
            let mut off = 5;

            if p >= 2 {
                let edges: [(usize, usize); 8] = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (0, 4), (1, 4), (2, 4), (3, 4),
                ];
                for &(la, lb) in &edges {
                    let ed = get_edge_dofs_pk(ns[la], ns[lb], &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                    for (k, &d) in ed.iter().enumerate() { dofs_flat[base + off + k] = d; }
                    off += edge_dofs_per;
                }
            }
            if p >= 2 {
                let key = QuadFaceKey::new(ns[0], ns[1], ns[2], ns[3]);
                let fd = quad_face_pk_map.entry(key).or_insert_with(|| {
                    (0..quad_face_dofs_per).map(|_| { let d = next_dof; next_dof += 1; d }).collect()
                });
                for (k, &d) in fd.iter().enumerate() { dofs_flat[base + off + k] = d; }
                off += quad_face_dofs_per;
            }
            if p >= 3 {
                for &(la, lb, lc) in &[(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)] {
                    let fd = get_face_dofs_pk(ns[la], ns[lb], ns[lc], &mut next_dof, &mut face_pk_map, tri_face_dofs_per);
                    for (k, &d) in fd.iter().enumerate() { dofs_flat[base + off + k] = d; }
                    off += tri_face_dofs_per;
                }
            }
            for _ in 0..volume_dofs_per {
                dofs_flat[base + off] = next_dof;
                next_dof += 1;
                off += 1;
            }
        }

        let n_dofs = next_dof as usize;
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let b = n as usize * dim;
            dof_coords[b..b + dim].copy_from_slice(c);
        }
        for (&EdgeKey(a, b), dofs) in &edge_pk_map {
            let ca = mesh.node_coords(a); let cb = mesh.node_coords(b);
            for (k, &did) in dofs.iter().enumerate() {
                let t = (k + 1) as f64 / (edge_dofs_per + 1) as f64;
                let base = did as usize * dim;
                for d in 0..dim { dof_coords[base + d] = (1.0 - t) * ca[d] + t * cb[d]; }
            }
        }
        // Tri faces
        if p >= 3 {
            let mut face_nodes: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                for &(a, b, c) in &[(ns[0],ns[1],ns[4]),(ns[1],ns[2],ns[4]),(ns[2],ns[3],ns[4]),(ns[3],ns[0],ns[4])] {
                    face_nodes.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
                }
            }
            for (key, dofs) in &face_pk_map {
                let [a, b, c] = face_nodes[key];
                let ca = mesh.node_coords(a); let cb = mesh.node_coords(b); let cc = mesh.node_coords(c);
                for (k, &did) in dofs.iter().enumerate() {
                    let base = did as usize * dim;
                    let t = (k + 1) as f64 / (dofs.len() + 1) as f64;
                    for d in 0..dim {
                        dof_coords[base + d] = (1.0 - t) * ca[d] + t * (cb[d] + cc[d]) / 2.0;
                    }
                }
            }
        }
        // Quad base face: bilinear
        if quad_face_dofs_per > 0 {
            let _qfb = n_nodes + edge_pk_map.len() * edge_dofs_per;
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                let v = [mesh.node_coords(ns[0]), mesh.node_coords(ns[1]), mesh.node_coords(ns[2]), mesh.node_coords(ns[3])];
                for row in 0..(p - 1) {
                    let xi = (row + 1) as f64 / p as f64;
                    for col in 0..(p - 1) {
                        let eta = (col + 1) as f64 / p as f64;
                        let local_off = 5 + 8 * edge_dofs_per + row * (p - 1) + col;
                        let elem_base = e as usize * dofs_per_elem;
                        let did = dofs_flat[elem_base + local_off] as usize;
                        let dbase = did * dim;
                        for d in 0..dim {
                            dof_coords[dbase + d] = (1.0 - xi) * (1.0 - eta) * v[0][d]
                                + xi * (1.0 - eta) * v[1][d]
                                + xi * eta * v[2][d]
                                + (1.0 - xi) * eta * v[3][d];
                        }
                    }
                }
            }
        }
        // Volume: use PyramidPk ref element
        if volume_dofs_per > 0 {
            let factory = fem_element::lagrange::PyramidPk::new(p);
            let ref_coords = factory.dof_coords();
            let vol_start = n_nodes + edge_pk_map.len() * edge_dofs_per
                + quad_face_pk_map.len() * quad_face_dofs_per
                + face_pk_map.len() * tri_face_dofs_per;
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                let c = [
                    mesh.node_coords(ns[0]), mesh.node_coords(ns[1]),
                    mesh.node_coords(ns[2]), mesh.node_coords(ns[3]),
                    mesh.node_coords(ns[4]),
                ];
                for k in 0..volume_dofs_per {
                    let did = vol_start + e as usize * volume_dofs_per + k;
                    let ri = surface_dofs + k;
                    let rc = &ref_coords[ri];
                    let (rx, ry, rz) = (rc[0], rc[1], rc[2]);
                    let dbase = did * dim;
                    if (rz - 1.0).abs() < 1e-14 {
                        dof_coords[dbase..dbase + dim].copy_from_slice(c[4]);
                    } else {
                        let iz = 1.0 - rz;
                        let u = rx / iz;
                        let v = ry / iz;
                        for d in 0..dim {
                            let qx = (1.0 - u) * (1.0 - v) * c[0][d] + u * (1.0 - v) * c[1][d]
                                + u * v * c[2][d] + (1.0 - u) * v * c[3][d];
                            dof_coords[dbase + d] = iz * qx + rz * c[4][d];
                        }
                    }
                }
            }
        }

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(), edge_dof2_map: HashMap::new(), edge_pk_map,
            face_pk_map, quad_face_pk_map,
            bubble_dof_start: n_dofs, n_volume_dofs: volume_dofs_per, elem_orders: None,
        }
    }

    // ─── Pk (arbitrary order) ─────────────────────────────────────────────────
    //
    // Builds a general-order Lagrange DOF manager for 2D triangle and 3D tetrahedron
    // meshes. The DOF ordering per element matches TriPk / TetPk from the factory.
    // For prism/pyramid, dispatches to specialized builders.

    fn build_pk<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let dim = mesh.dim() as usize;
        let topo_dim = mesh.topological_dim() as usize;
        let p = order as usize;
        // Prism/pyramid dispatch for general order
        if topo_dim == 3 && mesh.n_elements() > 0 {
            let npe = mesh.element_nodes(0).len();
            if npe == 6 { return Self::build_prism_pk(mesh, order); }
            if npe == 5 { return Self::build_pyramid_pk(mesh, order); }
        }
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();

        assert!(p >= 1, "build_pk: order must be >= 1");
        assert!(topo_dim == 2 || topo_dim == 3, "build_pk: only 2D and 3D elements supported");

        // Entity DOF counts
        let edge_dofs_per = if p >= 2 { p - 1 } else { 0 };
        let face_dofs_per = if topo_dim == 3 && p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
        let volume_dofs_per = if topo_dim == 2 && p >= 3 {
            (p - 1) * (p - 2) / 2
        } else if topo_dim == 3 && p >= 4 {
            (p - 1) * (p - 2) * (p - 3) / 6
        } else { 0 };

        let dofs_per_elem = if topo_dim == 2 {
            (p + 1) * (p + 2) / 2
        } else {
            (p + 1) * (p + 2) * (p + 3) / 6
        };

        let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        let mut face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
        let quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        if topo_dim == 2 {
            // ── 2-D triangles ────────────────────────────────────────────────
            // Local edge definitions matching TriPk:
            //   edge(0→1), edge(1→2), edge(0→2)
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                assert!(ns.len() >= 3, "build_pk 2D requires >= 3-noded elements");
                let (n0, n1, n2) = (ns[0], ns[1], ns[2]);
                let base = e as usize * dofs_per_elem;

                // Vertices (DOFs 0, 1, 2)
                dofs_flat[base]     = n0;
                dofs_flat[base + 1] = n1;
                dofs_flat[base + 2] = n2;

                if p >= 2 {
                    // 3 edges, each with (p-1) DOFs, ordered near-first-vertex to near-second
                    let edges = [(n0, n1), (n1, n2), (n0, n2)];
                    let mut off = 3;
                    for &(a, b) in &edges {
                        let edge_dofs = get_edge_dofs_pk(a, b, &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                        for (k, &dof) in edge_dofs.iter().enumerate() {
                            dofs_flat[base + off + k] = dof;
                        }
                        off += edge_dofs_per;
                    }
                }

                // Face interior (bubble) DOFs for p >= 3
                if volume_dofs_per > 0 {
                    let mut off = 3 + 3 * edge_dofs_per;
                    for _ in 0..volume_dofs_per {
                        dofs_flat[base + off] = next_dof;
                        next_dof += 1;
                        off += 1;
                    }
                }
            }
        } else {
            // ── 3-D tetrahedra ──────────────────────────────────────────────
            // Local edge definitions matching TetPk:
            //   edge(0→1), edge(0→2), edge(0→3), edge(1→2), edge(1→3), edge(2→3)
            // Local face definitions: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                assert!(ns.len() >= 4, "build_pk 3D requires >= 4-noded elements");
                let (n0, n1, n2, n3) = (ns[0], ns[1], ns[2], ns[3]);
                let base = e as usize * dofs_per_elem;

                // Vertices (DOFs 0-3)
                dofs_flat[base]     = n0;
                dofs_flat[base + 1] = n1;
                dofs_flat[base + 2] = n2;
                dofs_flat[base + 3] = n3;

                // 6 edges
                if p >= 2 {
                    let edges = [(n0, n1), (n0, n2), (n0, n3), (n1, n2), (n1, n3), (n2, n3)];
                    let mut off = 4;
                    for &(a, b) in &edges {
                        let edge_dofs = get_edge_dofs_pk(a, b, &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                        for (k, &dof) in edge_dofs.iter().enumerate() {
                            dofs_flat[base + off + k] = dof;
                        }
                        off += edge_dofs_per;
                    }
                }

                // 4 faces
                if p >= 3 {
                    let faces = [(n0, n1, n2), (n0, n1, n3), (n0, n2, n3), (n1, n2, n3)];
                    let mut off = 4 + 6 * edge_dofs_per;
                    for &(a, b, c) in &faces {
                        let face_dofs = get_face_dofs_pk(a, b, c, &mut next_dof, &mut face_pk_map, face_dofs_per);
                        for (k, &dof) in face_dofs.iter().enumerate() {
                            dofs_flat[base + off + k] = dof;
                        }
                        off += face_dofs_per;
                    }
                }

                // Volume interior DOFs for p >= 4
                if volume_dofs_per > 0 {
                    let mut off = 4 + 6 * edge_dofs_per + 4 * face_dofs_per;
                    for _ in 0..volume_dofs_per {
                        dofs_flat[base + off] = next_dof;
                        next_dof += 1;
                        off += 1;
                    }
                }
            }
        }

        let n_dofs = next_dof as usize;

        // ── Build DOF coordinates ────────────────────────────────────────────
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Vertex coordinates.
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let base = n as usize * dim;
            dof_coords[base..base + dim].copy_from_slice(c);
        }

        // Edge DOF coordinates: linear interpolation along each edge.
        // DOF k (0-indexed) at fraction (k+1)/(p) from canonical-a to canonical-b.
        for (&EdgeKey(a, b), dofs) in &edge_pk_map {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            for (k, &dof_id) in dofs.iter().enumerate() {
                let t = (k + 1) as f64 / p as f64;
                let base = dof_id as usize * dim;
                for d in 0..dim {
                    dof_coords[base + d] = (1.0 - t) * ca[d] + t * cb[d];
                }
            }
        }

        // 3D face DOF coordinates: barycentric interpolation from 3 face vertices.
        if topo_dim == 3 && !face_pk_map.is_empty() {
            // Build face→node mapping from element connectivity.
            let mut face_nodes_map: HashMap<FaceKey, [NodeId; 3]> = HashMap::new();
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                for &(a, b, c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                    face_nodes_map.entry(FaceKey::new(a, b, c)).or_insert([a, b, c]);
                }
            }
            for (key, dofs) in &face_pk_map {
                let nodes = face_nodes_map[key];
                let ca = mesh.node_coords(nodes[0]);
                let cb = mesh.node_coords(nodes[1]);
                let cc = mesh.node_coords(nodes[2]);
                let n_face_dofs = dofs.len();
                for (k, &dof_id) in dofs.iter().enumerate() {
                    let base = dof_id as usize * dim;
                    if n_face_dofs == 1 {
                        for d in 0..3 {
                            dof_coords[base + d] = (ca[d] + cb[d] + cc[d]) / 3.0;
                        }
                    } else {
                        // Distribute DOFs along the face using barycentric-like spacing.
                        let t = (k + 1) as f64 / (n_face_dofs + 1) as f64;
                        for d in 0..3 {
                            dof_coords[base + d] = (1.0 - t) * ca[d] + t * (cb[d] + cc[d]) / 2.0;
                        }
                    }
                }
            }
        }

        // Volume/bubble DOF coordinates: use factory reference element for accuracy.
        if volume_dofs_per > 0 {
            use fem_element::lagrange::factory::{ref_elem, ElemType};
            let ft = if topo_dim == 2 { ElemType::Tri } else { ElemType::Tet };
            let factory = ref_elem(ft, order);
            let ref_coords = factory.dof_coords();
            // Volume DOFs in factory are the LAST volume_dofs_per entries.
            let vol_factory_start = dofs_per_elem - volume_dofs_per;
            let vol_start = n_nodes + edge_pk_map.len() * edge_dofs_per;
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                // Vertex coordinates for barycentric interpolation
                let c0 = mesh.node_coords(ns[0]);
                let c1 = mesh.node_coords(ns[1]);
                let c2 = mesh.node_coords(ns[2]);
                let c3 = if dim >= 3 { mesh.node_coords(ns[3]) } else { &[] };
                for k in 0..volume_dofs_per {
                    let dof_id = vol_start + e as usize * volume_dofs_per + k;
                    let base = dof_id * dim;
                    let rc = &ref_coords[vol_factory_start + k];
                    if topo_dim == 2 {
                        let lam0 = 1.0 - rc[0] - rc[1];
                        for d in 0..dim {
                            dof_coords[base + d] = lam0 * c0[d] + rc[0] * c1[d] + rc[1] * c2[d];
                        }
                    } else {
                        let lam0 = 1.0 - rc[0] - rc[1] - rc[2];
                        for d in 0..dim {
                            dof_coords[base + d] = lam0 * c0[d] + rc[0] * c1[d]
                                + rc[1] * c2[d] + rc[2] * c3[d];
                        }
                    }
                }
            }
        }

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(),
            edge_dof2_map: HashMap::new(),
            edge_pk_map,
            face_pk_map,
            quad_face_pk_map,
            bubble_dof_start: n_dofs,
            n_volume_dofs: volume_dofs_per,
            elem_orders: None,
        }
    }

    /// Return the polynomial order for element `elem`.
    /// For uniform-order DofManagers, returns `self.order`.
    /// For variable-order DofManagers, returns the per-element order.
    pub fn element_order(&self, elem: ElemId) -> u8 {
        self.elem_orders.as_ref().map_or(self.order, |orders| orders[elem as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn p1_unit_square_dof_count() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dm = DofManager::new(&mesh, 1);
        assert_eq!(dm.n_dofs, mesh.n_nodes());
    }

    #[test]
    fn p1_element_dofs_are_node_ids() {
        let mesh = Mesh::<2>::unit_square_tri(2);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dm = DofManager::new(&mesh, 2);
        assert!(dm.n_dofs > mesh.n_nodes(), "P2 must have more DOFs than nodes");
        assert_eq!(dm.dofs_per_elem, 6);
    }

    #[test]
    fn p2_element_first_three_are_vertex_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
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
        let mesh = Mesh::<2>::unit_square_tri(1);
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
        let mut mesh = Mesh::<2>::uniform(
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
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dm1 = DofManager::new(&mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let dm2 = DofManager::new(&mesh2, 2);
        let mesh3 = Mesh::<2>::unit_square_tri(4);
        let dm3 = DofManager::new(&mesh3, 3);
        assert!(dm3.n_dofs > dm2.n_dofs, "P3 must have more DOFs than P2");
        assert!(dm2.n_dofs > dm1.n_dofs, "P2 must have more DOFs than P1");
        assert_eq!(dm3.dofs_per_elem, 10, "P3 elements should have 10 DOFs each");
    }

    #[test]
    fn p3_element_first_three_are_vertex_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 3);
        for e in 0..mesh.n_elements() as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs[..3], nodes, "elem {e}: P3 vertex DOFs mismatch");
        }
    }

    #[test]
    fn p3_edge_dofs_are_shared_between_adjacent_elements() {
        let mesh = Mesh::<2>::unit_square_tri(1);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 3);
        // bubble_dof_start = n_nodes + 2*n_unique_edges
        // Verify all bubble DOFs (one per element, at position 9) are >= bubble_dof_start
        for e in 0..mesh.n_elements() as u32 {
            let bubble = dm.element_dofs(e)[9] as usize;
            assert!(bubble >= dm.bubble_dof_start,
                "elem {e}: bubble dof {bubble} < bubble_dof_start {}", dm.bubble_dof_start);
        }
    }

    // ─── TetP3 DOF manager tests ──────────────────────────────────────────────

    #[test]
    fn tet_p3_dof_manager_basic() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let dm = DofManager::new(&mesh, 3);
        // n=2 cube tet: n_nodes = 3×3×3 = 27, n_elements = 6*2³ = 48
        assert_eq!(dm.dofs_per_elem, 20, "TetP3 must have 20 DOFs per element");
        assert!(dm.n_dofs > mesh.n_nodes(), "TetP3 must have more DOFs than nodes");
        // Vertex DOFs: first 4 DOFs of each element should be node IDs.
        for e in 0..mesh.n_elements() as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs[..4], nodes, "elem {e}: first 4 TetP3 DOFs must be vertex node IDs");
            // All 20 DOFs should be in valid range.
            for &d in dofs { assert!((d as usize) < dm.n_dofs, "elem {e}: DOF {d} out of range"); }
        }
    }

    #[test]
    fn tet_p3_dof_coords_in_unit_cube() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let dm = DofManager::new(&mesh, 3);
        for dof in 0..dm.n_dofs as u32 {
            let c = dm.dof_coord(dof);
            for (d, &v) in c.iter().enumerate() {
                assert!(v >= -1e-12 && v <= 1.0 + 1e-12,
                    "TetP3 DOF {dof} coord[{d}] = {v} not in [0,1]");
            }
        }
    }

    #[test]
    fn tet_p3_edge_dofs_shared() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let dm = DofManager::new(&mesh, 3);
        // Elements sharing an edge should share the 2 edge DOFs.
        // Verify edge_dof2_map has consistent entries.
        assert!(!dm.edge_dof2_map.is_empty(), "TetP3 should have non-empty edge_dof2_map");
        // Each edge DOF pair must be unique.
        let mut all_dof_pairs: Vec<[u32; 2]> = dm.edge_dof2_map.values().copied().collect();
        let len = all_dof_pairs.len();
        all_dof_pairs.sort_unstable();
        all_dof_pairs.dedup();
        assert_eq!(all_dof_pairs.len(), len, "TetP3 edge DOF pairs must be unique per edge");
    }

    // ─── Pk (general order) tests ──────────────────────────────────────────

    #[test]
    fn pk4_tri_dof_count() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 4);
        assert_eq!(dm.dofs_per_elem, 15, "P4 triangle should have 15 DOFs per element");
        assert!(dm.n_dofs > mesh.n_nodes(), "P4 must have more DOFs than nodes");
    }

    #[test]
    fn pk4_tri_vertex_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 4);
        for e in 0..mesh.n_elements() as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs[..3], nodes, "elem {e}: first 3 PK DOFs must be vertex node IDs");
        }
    }

    #[test]
    fn pk4_tri_edge_dofs_shared() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let dm = DofManager::new(&mesh, 4);
        assert_eq!(mesh.n_elements(), 2, "1×1 = 2 triangles");
        let dofs0 = dm.element_dofs(0).to_vec();
        let dofs1 = dm.element_dofs(1).to_vec();
        let edge0: Vec<_> = dofs0[3..12].iter().copied().collect();
        let edge1: Vec<_> = dofs1[3..12].iter().copied().collect();
        let shared: Vec<_> = edge0.iter().filter(|d| edge1.contains(d)).copied().collect();
        assert_eq!(shared.len(), 3, "P4 triangles should share 3 edge DOFs, got {}", shared.len());
    }

    #[test]
    fn pk4_tri_bubble_dofs_unique() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 4);
        let n_elems = mesh.n_elements();
        let mut bubbles: Vec<u32> = (0..n_elems as u32)
            .map(|e| dm.element_dofs(e)[14])
            .collect();
        let len = bubbles.len();
        bubbles.sort_unstable();
        bubbles.dedup();
        assert_eq!(bubbles.len(), len, "P4 bubble DOFs should be unique per element");
    }

    #[test]
    fn pk4_tri_n_dofs_increases_with_order() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let prev = DofManager::new(&mesh, 3).n_dofs;
        let cur  = DofManager::new(&mesh, 4).n_dofs;
        assert!(cur > prev, "P4 must have more DOFs than P3");
    }

    #[test]
    fn pk4_tri_dof_coords_in_bounds() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm = DofManager::new(&mesh, 4);
        for dof in 0..dm.n_dofs as u32 {
            let c = dm.dof_coord(dof);
            assert!(c[0] >= -1e-12 && c[0] <= 1.0 + 1e-12,
                "P4 DOF {dof}: x={} out of [0,1]", c[0]);
            assert!(c[1] >= -1e-12 && c[1] <= 1.0 + 1e-12,
                "P4 DOF {dof}: y={} out of [0,1]", c[1]);
        }
    }

    #[test]
    fn pk4_tet_basic() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let dm = DofManager::new(&mesh, 4);
        assert_eq!(dm.dofs_per_elem, 35, "P4 tet should have 35 DOFs per element");
        for e in 0..mesh.n_elements() as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            assert_eq!(&dofs[..4], nodes, "elem {e}: first 4 PK DOFs must be vertex node IDs");
        }
        assert!(!dm.edge_pk_map.is_empty(), "P4 tet should have non-empty edge_pk_map");
    }

    // ─── Prism6 P2 tests ──────────────────────────────────────────────────────

    fn make_prism_mesh() -> Mesh<3> {
        use fem_mesh::element_type::ElementType;
        Mesh::<3>::uniform(
            vec![
                0.,0.,0., 1.,0.,0., 0.,1.,0.,
                0.,0.,1., 1.,0.,1., 0.,1.,1.,
                0.,0.,2., 1.,0.,2., 0.,1.,2.,
            ],
            vec![0,1,2,3,4,5, 3,4,5,6,7,8],
            vec![1,1], ElementType::Prism6, vec![], vec![], ElementType::Tri3,
        )
    }

    #[test] fn prism_p1_basic() {
        let m=make_prism_mesh(); let dm=DofManager::new(&m,1);
        assert_eq!(dm.n_dofs,m.n_nodes());
        for e in 0..m.n_elements() as u32{assert_eq!(dm.element_dofs(e),m.element_nodes(e));}
    }

    #[test] fn prism_p2_basic() {
        let m=make_prism_mesh(); let dm=DofManager::new(&m,2);
        assert_eq!(dm.dofs_per_elem,18); assert!(dm.n_dofs>m.n_nodes());
        for e in 0..m.n_elements() as u32{assert_eq!(&dm.element_dofs(e)[..6],m.element_nodes(e));}
        let d0=dm.element_dofs(0); let d1=dm.element_dofs(1);
        let shared:Vec<_>=d0[6..15].iter().filter(|d|d1[6..15].contains(d)).collect();
        assert!(!shared.is_empty(),"P2 prism adjacent should share edge DOFs");
    }

    #[test] fn prism_p3_basic() {
        let m=make_prism_mesh(); let dm=DofManager::new(&m,3);
        assert_eq!(dm.dofs_per_elem,40);
        assert!(dm.n_dofs>DofManager::new(&m,2).n_dofs);
        // Vertices first in each layer
        for e in 0..m.n_elements() as u32{
            let d=dm.element_dofs(e); let n=m.element_nodes(e);
            assert_eq!(d[0],n[0]); assert_eq!(d[1],n[1]); assert_eq!(d[2],n[2]);
            assert_eq!(d[30],n[3]); assert_eq!(d[31],n[4]); assert_eq!(d[32],n[5]);
        }
    }

    #[test] fn prism_p4_basic() {
        let m=make_prism_mesh(); let dm=DofManager::new(&m,4);
        // P4 prism total DOFs = (4+1)*(4+1)*(4+2)/2 = 5*5*6/2 = 75
        assert_eq!(dm.dofs_per_elem,75);
        assert!(dm.n_dofs>DofManager::new(&m,3).n_dofs);
        for e in 0..m.n_elements() as u32{
            assert_eq!(&dm.element_dofs(e)[..6],m.element_nodes(e));
        }
    }

    #[test] fn prism_p5_basic() {
        let m=make_prism_mesh(); let dm=DofManager::new(&m,5);
        // P5 prism total DOFs = (5+1)*(5+1)*(5+2)/2 = 6*6*7/2 = 126
        assert_eq!(dm.dofs_per_elem,126);
        assert!(dm.n_dofs>DofManager::new(&m,4).n_dofs);
    }

    #[test] fn prism_p6_basic() {
        let m=make_prism_mesh(); let dm=DofManager::new(&m,6);
        // P6 prism total DOFs = (6+1)*(6+1)*(6+2)/2 = 7*7*8/2 = 196
        assert_eq!(dm.dofs_per_elem,196);
        assert!(dm.n_dofs>DofManager::new(&m,5).n_dofs);
        // Verify DOF coords are finite
        for dof in 0..dm.n_dofs {
            let c=dm.dof_coord(dof as u32);
            assert!(c.iter().all(|x|x.is_finite()));
        }
    }

    // ─── Pyramid5 P2 tests ────────────────────────────────────────────────────

    fn make_pyramid_mesh() -> Mesh<3> {
        use fem_mesh::element_type::ElementType;
        Mesh::<3>::uniform(
            vec![0.,0.,0., 1.,0.,0., 1.,1.,0., 0.,1.,0., 0.5,0.5,1., 0.,0.,1.],
            vec![0,1,2,3,4, 3,0,4,5],
            vec![1,1], ElementType::Pyramid5, vec![], vec![], ElementType::Tri3,
        )
    }

    #[test] fn pyramid_p1_basic() {
        let m=make_pyramid_mesh(); let dm=DofManager::new(&m,1);
        assert_eq!(dm.n_dofs,m.n_nodes());
    }

    #[test] fn pyramid_p2_basic() {
        let m=make_pyramid_mesh(); let dm=DofManager::new(&m,2);
        assert_eq!(dm.dofs_per_elem,14); assert!(dm.n_dofs>m.n_nodes());
        for e in 0..m.n_elements() as u32{assert_eq!(&dm.element_dofs(e)[..5],m.element_nodes(e));}
    }

    #[test] fn pyramid_p3_basic() {
        let m=make_pyramid_mesh(); let dm=DofManager::new(&m,3);
        assert_eq!(dm.dofs_per_elem,30);
        assert!(dm.n_dofs>DofManager::new(&m,2).n_dofs);
    }

    #[test] fn pyramid_p4_basic() {
        let m=make_pyramid_mesh(); let dm=DofManager::new(&m,4);
        // P4 pyramid total DOFs = (4+1)(4+2)(2*4+3)/6 = 5*6*11/6 = 55
        assert_eq!(dm.dofs_per_elem,55);
        assert!(dm.n_dofs>DofManager::new(&m,3).n_dofs);
    }

    #[test] fn pyramid_p5_basic() {
        let m=make_pyramid_mesh(); let dm=DofManager::new(&m,5);
        // P5 pyramid total DOFs = (5+1)(5+2)(2*5+3)/6 = 6*7*13/6 = 91
        assert_eq!(dm.dofs_per_elem,91);
        assert!(dm.n_dofs>DofManager::new(&m,4).n_dofs);
    }

    #[test] fn pyramid_p6_basic() {
        let m=make_pyramid_mesh(); let dm=DofManager::new(&m,6);
        // P6 pyramid total DOFs = (6+1)(6+2)(2*6+3)/6 = 7*8*15/6 = 140
        assert_eq!(dm.dofs_per_elem,140);
        assert!(dm.n_dofs>DofManager::new(&m,5).n_dofs);
        for dof in 0..dm.n_dofs {
            let c=dm.dof_coord(dof as u32);
            assert!(c.iter().all(|x|x.is_finite()));
        }
    }
}
