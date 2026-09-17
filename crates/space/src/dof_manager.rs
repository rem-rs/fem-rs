//! DOF numbering for Lagrange finite element spaces.
//!
//! Handles vertex-only DOFs (P1), vertex+edge DOFs (P2), and arbitrary-order
//! Lagrange spaces (Pk) on simplicial and tensor-product meshes.
//!
//! For arbitrary order `p >= 1`:
//! - Triangles: (p+1)(p+2)/2 DOFs per element
//! - Tetrahedra: (p+1)(p+2)(p+3)/6 DOFs per element
//!
//! DOF ordering within each element follows MFEM's `H1_FECollection` entity
//! layout: triangles via [`fem_element::TriPk`]-convention edges with
//! Gauss-Lobatto coordinates (`H1TriPk`), tetrahedra via
//! [`DofManager::build_tet_h1`] (MFEM `H1_TetrahedronElement`, D157), prisms
//! via [`DofManager::build_prism_h1`].

use std::collections::HashMap;
use fem_core::types::{DofId, ElemId, FaceId, NodeId};
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

// ─── D61: periodic-mesh numbering ────────────────────────────────────────────

/// True when `mesh` carries a per-element geometry snapshot whose corner
/// pairing against the folded connectivity reveals **merged (periodic)
/// vertices**: some element corner references a different geometry node than
/// the folded vertex — the signature of `Mesh::make_periodic`, which keeps
/// the pre-merge table as the per-element geometry (order-1 snapshot or a
/// high-order geometry built before the merge).
///
/// Curved *non-periodic* meshes reuse the mesh vertex ids as geometry corner
/// ids (`set_curvature`), so they never trigger; their behaviour stays
/// bit-for-bit unchanged.
fn is_periodic_merged<M: MeshTopology>(mesh: &M) -> bool {
    for e in 0..mesh.n_elements() as u32 {
        let gn = mesh.geometry_nodes(e);
        let fnodes = mesh.element_nodes(e);
        for k in 0..fnodes.len() {
            if gn[k] != fnodes[k] {
                return true;
            }
        }
    }
    false
}

/// Torus-entity signature of one un-merged entity occurrence: the folded
/// vertex set plus the translation-invariant geometric frame that separates
/// distinct periodic images sharing that vertex set (D61).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum EntitySig {
    /// Folded vertex pair + unfolded endpoint displacement along the folded
    /// low→high orientation (quantized).
    Edge { key: (NodeId, NodeId), delta: [i64; 3] },
    /// Sorted folded corners + (folded id, corner offset) of the remaining
    /// corners relative to the lowest folded corner's image.
    TriFace {
        key: (NodeId, NodeId, NodeId),
        offs: [(NodeId, [i64; 3]); 2],
    },
    /// As [`EntitySig::TriFace`], for quadrilateral faces.
    QuadFace {
        key: (NodeId, NodeId, NodeId, NodeId),
        offs: [(NodeId, [i64; 3]); 3],
    },
}

/// The **un-merged** view of a geometrically periodic mesh: element
/// connectivity is the element's own pre-merge geometry corner list (from the
/// order-1 periodic snapshot) and node coordinates are the pre-merge table,
/// so the plain DOF builders number every geometric entity of the covering
/// mesh before the periodic quotient ([`DofManager::build_periodic`]) merges
/// them.
struct UnfoldedPeriodicMesh<'a, M: MeshTopology> {
    inner: &'a M,
}

impl<M: MeshTopology> MeshTopology for UnfoldedPeriodicMesh<'_, M> {
    fn dim(&self) -> u8 { self.inner.dim() }
    fn topological_dim(&self) -> u8 { self.inner.topological_dim() }
    fn n_nodes(&self) -> usize { self.inner.geom_n_nodes() }
    fn n_elements(&self) -> usize { self.inner.n_elements() }
    fn n_boundary_faces(&self) -> usize { self.inner.n_boundary_faces() }
    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        // The element's own pre-merge corners: the geometry snapshot's node
        // list starts with the corners in element vertex order (vertex DOFs
        // reuse the mesh vertices, `set_curvature`), so the FE corner slice
        // is the prefix of the geometry list — also on curved periodic
        // meshes, whose geometry list continues with the high-order nodes.
        let npe = self.inner.element_nodes(elem).len();
        &self.inner.geometry_nodes(elem)[..npe]
    }
    fn element_type(&self, elem: ElemId) -> fem_mesh::ElementType { self.inner.element_type(elem) }
    fn element_tag(&self, elem: ElemId) -> i32 { self.inner.element_tag(elem) }
    fn node_coords(&self, node: NodeId) -> &[f64] { self.inner.geom_coords_of(node) }
    fn face_nodes(&self, face: FaceId) -> &[NodeId] { self.inner.face_nodes(face) }
    fn face_tag(&self, face: FaceId) -> i32 { self.inner.face_tag(face) }
    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>) {
        self.inner.face_elements(face)
    }
    fn geom_order(&self) -> u8 { self.inner.geom_order() }
    fn geometry_nodes(&self, elem: ElemId) -> &[NodeId] { self.inner.geometry_nodes(elem) }
    fn geom_coords_of(&self, node: NodeId) -> &[f64] { self.inner.geom_coords_of(node) }
    fn geom_n_nodes(&self) -> usize { self.inner.geom_n_nodes() }
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
    ///
    /// On meshes with per-element geometry (geometrically periodic) this is
    /// rebuilt from the elements' own geometry nodes with last-writer-wins
    /// semantics (MFEM `ProjectCoefficient`), so a seam DOF holds the
    /// position it has in the *last* element containing it — see
    /// [`DofManager::rebuild_dof_coords_periodic`].
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
    /// Physical mesh node → global vertex DOF id, for NC meshes whose vertex
    /// DOFs follow the MFEM vertex-view order (identity for conforming
    /// meshes).  Constraints (hanging nodes) reference physical node ids;
    /// they must be translated through this map to the global DOF ids.
    pub phys_to_vertex_dof: HashMap<NodeId, DofId>,
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
    /// Variable-order edge DOF variants (MFEM `var_edge_dofs`): per canonical
    /// edge, one `(order, dofs)` pair (ascending order) per distinct adjacent
    /// element order. Empty for uniform-order DofManagers.
    pub edge_variants: HashMap<EdgeKey, Vec<(u8, Vec<DofId>)>>,
    /// Variable-order face DOF variants (3D, MFEM `var_face_dofs`): per
    /// canonical face, one `(order, dofs)` pair per distinct adjacent element
    /// order. Empty for uniform-order DofManagers.
    pub face_variants: HashMap<FaceKey, Vec<(u8, Vec<DofId>)>>,
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
    /// On meshes with per-element geometry (geometrically periodic meshes,
    /// see [`MeshTopology::geometry_nodes`]) the DOF coordinate table is
    /// rebuilt from each element's own geometry nodes after the fold-based
    /// construction (D56) — see [`DofManager::rebuild_dof_coords_periodic`].
    ///
    /// Geometrically periodic meshes (order-1 geometry snapshot, i.e.
    /// `geom_order() == 1 && geom_n_nodes() != n_nodes()`) additionally take
    /// the D61 numbering path
    /// ([`DofManager::build_periodic`]): the plain vertex-pair
    /// [`EdgeKey`]/[`FaceKey`]/[`QuadFaceKey`] dedup under-counts DOFs when a
    /// periodic direction has fewer than three cells (several torus
    /// edges/faces share one folded vertex set), so the numbering is built on
    /// the un-merged connectivity and quotiented by explicit per-entity
    /// geometric identity.
    ///
    /// # Panics
    /// Panics if the requested order is unsupported for the mesh type.
    pub fn new<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let periodic = is_periodic_merged(mesh);
        let mut dm = if periodic {
            Self::build_periodic(mesh, order)
        } else {
            Self::build(mesh, order)
        };
        // D56: a periodic geometry snapshot keeps per-element coordinates, so
        // the fold-based table above places seam DOFs at folded chord
        // positions no element can see.  Rebuild per element (MFEM
        // `ProjectCoefficient` semantics: evaluate at each element's own
        // nodal points, last writer wins for shared DOFs).  D62: this also
        // covers periodic meshes with curved (order >= 2) geometry, whose
        // coordinates are evaluated through the high-order geometry basis.
        if periodic {
            dm.rebuild_dof_coords_periodic(mesh);
        }
        dm
    }

    fn build<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let topo_dim = mesh.topological_dim() as usize;
        match order {
            1 => Self::build_p1(mesh),
            2 => {
                if topo_dim == 3 {
                    if mesh.n_elements() > 0 {
                        let npe = mesh.element_nodes(0).len();
                        match npe {
                            6 => Self::build_prism_h1(mesh, 2),
                            // D191: unified MFEM entity-order pyramid builder.
                            5 => Self::build_pyramid_pk(mesh, 2),
                            8 => Self::build_q2_hex(mesh),
                            _ => Self::build_pk(mesh, 2),
                        }
                    } else {
                        Self::build_pk(mesh, 2)
                    }
                } else if mesh.n_elements() > 0
                    && mesh.element_nodes(0).len() == 4
                    && topo_dim == 2
                {
                    Self::build_q2_quad(mesh)
                } else {
                    Self::build_pk(mesh, 2)
                }
            }
            3 => {
                if topo_dim == 3 && mesh.n_elements() > 0 {
                    let npe = mesh.element_nodes(0).len();
                    if npe == 6 { return Self::build_prism_h1(mesh, 3); }
                    // D191: unified MFEM entity-order pyramid builder.
                    if npe == 5 { return Self::build_pyramid_pk(mesh, 3); }
                }
                // Quad Q3 / Hex Q3 via general pk path
                if mesh.n_elements() > 0 {
                    let npe = mesh.element_nodes(0).len();
                    if npe == 4 && topo_dim == 2 { return Self::build_pk_quad(mesh, order); }
                    if npe == 8 && topo_dim == 3 { return Self::build_pk_hex(mesh, order); }
                }
                Self::build_pk(mesh, 3)
            }
            _ => {
                // General arbitrary-order path for p >= 4
                if mesh.n_elements() > 0 {
                    let npe = mesh.element_nodes(0).len();
                    if npe == 4 && topo_dim == 2 { return Self::build_pk_quad(mesh, order); }
                    if npe == 8 && topo_dim == 3 { return Self::build_pk_hex(mesh, order); }
                    if npe == 6 && topo_dim == 3 { return Self::build_prism_h1(mesh, order); }
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
    ///
    /// On geometrically periodic meshes a shared (seam) DOF has no single
    /// physical position: the returned value is its position in the last
    /// element containing it (MFEM `ProjectCoefficient` last-writer-wins),
    /// see [`DofManager::rebuild_dof_coords_periodic`].
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
            edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_nodes,
            n_volume_dofs: 0,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
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
            edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
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

        // MFEM vertex-view ordering: if the mesh carries an NC vertex view
        // (top-level nodes first, then non-top-level nodes in SFC/leaf order —
        // MFEM UpdateVertices), vertex DOF `d` refers to physical node
        // `view[d]`.  This is how the global DOF ids line up with MFEM on
        // non-conforming meshes.
        let vertex_view: Option<&[NodeId]> = mesh.nc_vertex_view();
        // Number of vertex DOFs = vertex-view length when present.  The mesh
        // node table may contain extra (preserved) nodes that are not part of
        // any element — e.g. edge-midpoint history kept for the NC constraint
        // walk — and those must NOT become DOFs (MFEM: vertex table only
        // covers nodes used by elements).
        let n_vertex = vertex_view.map_or(n_nodes, |v| v.len());
        let node_to_dof: std::collections::HashMap<NodeId, DofId> = match vertex_view {
            Some(view) => view
                .iter()
                .enumerate()
                .map(|(d, &n)| (n, d as DofId))
                .collect(),
            None => (0..n_nodes as NodeId).map(|n| (n, n as DofId)).collect(),
        };
        let vertex_phys = |dof: DofId| -> NodeId {
            match vertex_view {
                Some(view) => view[dof as usize],
                None => dof,
            }
        };

        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_dof = n_vertex as DofId;

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

            // Vertex DOFs (positions 0–3) — via the vertex view so the ids
            // match MFEM's ordering on NC meshes.
            dofs_flat[base]     = node_to_dof[&n0];
            dofs_flat[base + 1] = node_to_dof[&n1];
            dofs_flat[base + 2] = node_to_dof[&n2];
            dofs_flat[base + 3] = node_to_dof[&n3];

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
        let mut interior_dof = (n_vertex + n_edge_dofs) as DofId;
        for e in 0..n_elems as u32 {
            let base = e as usize * dofs_per_elem;
            dofs_flat[base + 8] = interior_dof;
            interior_dof += 1;
        }

        let n_dofs = interior_dof as usize;

        // Build DOF coordinates.
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Vertex coords (via vertex view: DOF d lives at physical node view[d]).
        for d in 0..n_vertex as u32 {
            let phys = vertex_phys(d);
            let c = mesh.node_coords(phys);
            dof_coords[d as usize * dim .. d as usize * dim + dim].copy_from_slice(c);
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
            n_vertex_dofs: n_vertex, edge_dof_map: edge_map,
            edge_dof2_map: HashMap::new(),
            phys_to_vertex_dof: match vertex_view {
                Some(view) => view
                    .iter()
                    .enumerate()
                    .map(|(d, &n)| (n, d as DofId))
                    .collect(),
                None => (0..n_nodes as NodeId)
                    .map(|n| (n, n as DofId))
                    .collect(),
            },
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_nodes + n_edge_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
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
        //   3,4     → edge(n0→n1): DOFs near n0 and near n1 (GLL 1±1/√5 points)
        //   5,6     → edge(n1→n2): DOFs near n1 and near n2
        //   7,8     → edge(n2→n0): DOFs near n2 and near n0
        //   (counter-clockwise ring, matching MFEM/H1TriPk: the last edge
        //   runs from v2 toward v0)
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

            let [d7, d8] = get_edge_dofs(n2, n0, &mut next_edge_dof, &mut edge2_map);
            dofs_flat[base + 7] = d7; // near_n2 (H1TriPk edge-2 runs v2→v0)
            dofs_flat[base + 8] = d8; // near_n0
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

        // Edge DOF coordinates: pair[0] at gll[1] from canonical-first toward
        // second, pair[1] at gll[2] (H1TriPk Gauss-Lobatto closed points,
        // equal to 1/3, 2/3 only when p<=2 — here p=3 so gll = 1±1/√5 scaled).
        let (g_pts, _) = fem_element::quadrature::gauss_lobatto_arbitrary(4);
        let gll = |i: usize| -> f64 { 0.5 * (g_pts[i] + 1.0) };
        for (&EdgeKey(a, b), &[d0, d1]) in &edge2_map {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            let base0 = d0 as usize * dim;
            let base1 = d1 as usize * dim;
            let (t0, t1) = (gll(1), gll(2));
            for d in 0..dim {
                dof_coords[base0 + d] = (1.0 - t0) * ca[d] + t0 * cb[d];
                dof_coords[base1 + d] = (1.0 - t1) * ca[d] + t1 * cb[d];
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
            edge_dof2_map: edge2_map, phys_to_vertex_dof: HashMap::new(), 
            edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start,
            n_volume_dofs: 0,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
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
            edge_dof2_map: edge2_map, phys_to_vertex_dof: HashMap::new(), 
            edge_pk_map: HashMap::new(),
            face_pk_map,
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start,
            n_volume_dofs: 0,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
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
            edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
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
        // This is the element-local dof POSITION order (aligned with the
        // HexQk reference basis, cf. the tmop_form positional tests).
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
        // MFEM's local hex topology (Geometry::Constants<Geometry::CUBE>):
        // Edges[12] and FaceVert[6].  Global dof ids follow MFEM
        // FiniteElementSpace::Construct — vertices, then ALL edge dofs, then
        // ALL face dofs, then ALL volume dofs, with entity ids assigned
        // first-touch in element order scanning each element's entities in
        // MFEM's table order.  Numbering the entities inside the element loop
        // (interleaving faces/volumes with edges of later elements) misnumbers
        // them vs MFEM, which breaks reading MFEM `nodes` grid functions
        // (mesh-optimizer cube.mesh: det(J) -17.4 instead of 0.125).
        const MFEM_EDGES: [(usize, usize); 12] = [
            (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6),
            (7, 6), (4, 7), (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        const MFEM_FACES: [[usize; 4]; 6] = [
            [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
        ];

        // Phase 1: vertex + edge dofs (all edges before any face/volume dof).
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            for &(a, b) in &MFEM_EDGES {
                let key = EdgeKey::new(ns[a], ns[b]);
                edge_map.entry(key).or_insert_with(|| {
                    let d = next_dof; next_dof += 1; d
                });
            }
        }
        // Phase 2: face-center dofs.
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            for quad in &MFEM_FACES {
                let key = QuadFaceKey::new(ns[quad[0]], ns[quad[1]], ns[quad[2]], ns[quad[3]]);
                qface_map.entry(key).or_insert_with(|| {
                    let d = next_dof; next_dof += 1; d
                });
            }
        }
        // Phase 3: volume-center dofs (element order), after all edge/face dofs.
        for e in 0..n_elems as u32 {
            let vol_dof = next_dof;
            next_dof += 1;
            let ns = mesh.element_nodes(e);
            let base = e as usize * dofs_per_elem;

            // Vertices (positions 0..8)
            for (k, &n) in ns.iter().enumerate() {
                dofs_flat[base + k] = n;
            }

            // Edge midpoints (positions 8..20)
            for (k, &(a, b)) in EDGES.iter().enumerate() {
                let key = EdgeKey::new(ns[a], ns[b]);
                dofs_flat[base + 8 + k] = edge_map[&key];
            }

            // Face centers (positions 20..26)
            for (k, quad) in FACES.iter().enumerate() {
                let key = QuadFaceKey::new(ns[quad[0]], ns[quad[1]], ns[quad[2]], ns[quad[3]]);
                dofs_flat[base + 20 + k] = qface_map[&key];
            }

            // Volume center (position 26) — one per element.
            dofs_flat[base + 26] = vol_dof;
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
            edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), edge_pk_map: HashMap::new(),
            face_pk_map: HashMap::new(),
            // Retain the face-center dof of every quad face so boundary-dof
            // collection (boundary_dofs) can find the center dofs of boundary
            // faces (needed for TMOP surface fitting on hex meshes).
            quad_face_pk_map: qface_map.into_iter().map(|(k, d)| (k, vec![d])).collect(),
            bubble_dof_start: n_dofs,
            n_volume_dofs: 0,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
        }
    }

    // ─── H1 Prism (MFEM `H1_WedgeElement` layout, any order) ─────────────────

    /// H¹ DOF manager for triangular prism meshes, any order `p ≥ 1`.
    ///
    /// The per-element slot layout is MFEM `H1_WedgeElement(p)`'s — the same
    /// table [`fem_element::lagrange::H1PrismPk`] evaluates, so the assembler's
    /// reference element pairs slot-for-slot with `element_dofs` (D168 ground
    /// truth, probe `tmp/a34_prism_h1_probe.cpp`): 6 vertices, the 9 edge
    /// blocks ([`PRISM_EDGES`] order, dof `j` at the `j`-th Gauss-Lobatto
    /// point from the edge's *first* vertex, MFEM `SegDofOrd`), the bottom/top
    /// triangular faces (each face's dof list is oriented by the
    /// first-encountering element's triangle order, MFEM `TriDofOrd`), the 3
    /// quadrilateral side faces (dof list in the first-encountering element's
    /// parameterisation, MFEM `QuadDofOrd`), then the element-private interior.
    ///
    /// The old per-order builders (`build_p2_prism`, `build_p3_prism`, the
    /// layer-major `build_prism_pk`) are replaced by this one: p2 wrote its
    /// tri-face dofs into edge slot 14 and left slots 16/17 as dof 0, p3 used
    /// `PrismPk`'s layer order (not MFEM's), and p ≥ 4 mis-sliced
    /// `PrismPk::dof_coords()` for the interior coordinates.
    ///
    /// The **global** dof ids follow MFEM's entity layout (D177, ground truth
    /// `tmp/d177/d177_probe.cpp`, regression
    /// `tests/d177_prism_h1_mfem_numbering.rs`): vertices, then every edge dof
    /// (edge-table order), then every face dof (face-table order, tri and
    /// quad interleaved), then the interiors (element order) — allocation is
    /// phased exactly like `build_tet_h1` / `build_pk_hex` / `build_pk_quad`,
    /// so curved `nodes` files read back through the generic
    /// `DofManager`-numbered arm of `crates/io`'s `build_h1_geometry` land on
    /// the right physical dofs for multi-element meshes too.
    fn build_prism_h1<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        use fem_element::lagrange::{h1_prism_slots, H1PrismPk, H1PrismSlot, PRISM_EDGES};

        let p = order as usize;
        assert!(p >= 1, "build_prism_h1: order must be >= 1");
        assert_eq!(
            mesh.topological_dim() as usize,
            3,
            "build_prism_h1 requires 3-D elements"
        );
        let dim = 3usize;
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let slots = h1_prism_slots(p);
        let dofs_per_elem = slots.len();
        let ne = p - 1;
        let nt = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
        let nq = ne * ne;

        // `H1_TriangleElement`'s interior slot labels: barycentric exponents
        // `(λ0, λ1, λ2)` (λ1 ∝ x, λ2 ∝ y), running (j-outer, i-inner) — the
        // same order the face slots of `H1PrismPk` enumerate.
        let tri_labels: Vec<[usize; 3]> = (1..p)
            .flat_map(|j| (1..(p - j)).map(move |i| [p - i - j, i, j]))
            .collect();
        debug_assert_eq!(tri_labels.len(), nt);
        // MFEM's *bottom* face dof permutation (`fe_h1.cpp:930`: its `FaceVert`
        // list is `(0, 2, 1)`, reversed winding, so the k-th bottom-face dof
        // sits at the triangle interior index
        // `l = j - p + ((2p-1-i)·i)/2` with `(i, j)` the k-th (j-outer,
        // i-inner) pair — `H1PrismPk`'s `TriFace(0, k)` slot position is
        // `H1TriPk` node `3p + l`, i.e. label `tri_labels[l]`).
        let bottom_label_of_k: Vec<[usize; 3]> = (1..p)
            .flat_map(|j| {
                let tri_labels = &tri_labels;
                (1..(p - j)).map(move |i| {
                    let l = (j as i64 - p as i64 + (((2 * p - 1 - i) * i) / 2) as i64) as usize;
                    tri_labels[l]
                })
            })
            .collect();
        debug_assert_eq!(bottom_label_of_k.len(), nt);

        let mut edge_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        // Face entities keep the dof list *and* the first-encountering vertex
        // order that defines the list's orientation.
        let mut tri_map: HashMap<FaceKey, (Vec<DofId>, [NodeId; 3])> = HashMap::new();
        let mut quad_map: HashMap<QuadFaceKey, (Vec<DofId>, [NodeId; 4])> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        // D177: the *global* dof numbering follows MFEM's entity layout —
        // vertices, then every edge dof (edge-table order, contiguous blocks),
        // then every face dof (face-table order, triangular and quadrilateral
        // faces interleaved in one numbering), then the element-private
        // interiors (element order).  The old single-pass first-touch loop
        // interleaved later elements' edge dofs with earlier elements'
        // face/interior dofs, which agreed with MFEM only on single-element
        // meshes and scrambled the non-vertex dofs of every curved `nodes`
        // read-back (and any other consumer comparing against MFEM's tables).
        // The allocation is therefore phased exactly like `build_tet_h1`
        // (D157), `build_pk_hex` and `build_pk_quad`.
        //
        // Phase 1: vertices and edges, element-major, slot order within one
        // element (the element's `PRISM_EDGES` walk is MFEM's edge-table
        // discovery order, so first touch reproduces the edge indices).
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 6, "build_prism_h1 requires 6-node prisms");
            let n6 = [ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]];
            let base = e as usize * dofs_per_elem;
            for (s, slot) in slots.iter().enumerate() {
                match *slot {
                    H1PrismSlot::Vertex(v) => dofs_flat[base + s] = n6[v],
                    H1PrismSlot::Edge(kk, j) => {
                        let (la, lb) = (n6[PRISM_EDGES[kk][0]], n6[PRISM_EDGES[kk][1]]);
                        let key = EdgeKey::new(la, lb);
                        let list = edge_map.entry(key).or_insert_with(|| {
                            (0..ne).map(|_| { let d = next_dof; next_dof += 1; d }).collect()
                        });
                        // `j` counts from the local first vertex; the map is
                        // canonical (ascending vertex id).
                        dofs_flat[base + s] =
                            if la == key.0 { list[j] } else { list[ne - 1 - j] };
                    }
                    _ => {}
                }
            }
        }
        // Phase 2: face DOFs, after ALL edge DOFs.  Within one element the
        // slot walk is [bottom tri, top tri, quad0, quad1, quad2] — MFEM's
        // `Geometry::PRISM` face order — so first-touch allocation from the
        // shared counter interleaves triangular and quadrilateral face blocks
        // exactly as MFEM's face table does.  Each face's list keeps the
        // first-encountering vertex order that defines its orientation.
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            let n6 = [ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]];
            let base = e as usize * dofs_per_elem;
            for (s, slot) in slots.iter().enumerate() {
                match *slot {
                    H1PrismSlot::TriFace(f, k) => {
                        let (a, b, c) =
                            if f == 0 { (n6[0], n6[1], n6[2]) } else { (n6[3], n6[4], n6[5]) };
                        let key = FaceKey::new(a, b, c);
                        let (list, canon) = {
                            let entry = tri_map.entry(key).or_insert_with(|| {
                                let list: Vec<DofId> =
                                    (0..nt).map(|_| { let d = next_dof; next_dof += 1; d }).collect();
                                (list, [a, b, c])
                            });
                            (entry.0.clone(), entry.1)
                        };
                        // Rotate slot `k`'s barycentric label from the local
                        // triangle orientation into the canonical one and look
                        // up the canonical index (MFEM `TriDofOrd`; the bottom
                        // face carries MFEM's own (0,2,1) permutation first).
                        let local = if f == 0 { bottom_label_of_k[k] } else { tri_labels[k] };
                        let local_verts = [a, b, c];
                        let canon_label = [
                            local[local_verts.iter().position(|&v| v == canon[0]).unwrap_or(0)],
                            local[local_verts.iter().position(|&v| v == canon[1]).unwrap_or(0)],
                            local[local_verts.iter().position(|&v| v == canon[2]).unwrap_or(0)],
                        ];
                        let ci = tri_labels
                            .iter()
                            .position(|l| *l == canon_label)
                            .unwrap_or_else(|| {
                                panic!("build_prism_h1: tri dof label {canon_label:?} not in table")
                            });
                        dofs_flat[base + s] = list[ci];
                    }
                    H1PrismSlot::QuadFace(f, i, j) => {
                        // Local side faces `(0,1,4,3) (1,2,5,4) (2,0,3,5)`.
                        let (l0, l1, l2, l3) = match f {
                            2 => (n6[0], n6[1], n6[4], n6[3]),
                            3 => (n6[1], n6[2], n6[5], n6[4]),
                            _ => (n6[2], n6[0], n6[3], n6[5]),
                        };
                        let key = QuadFaceKey::new(l0, l1, l2, l3);
                        let (list, stored) = {
                            let entry = quad_map.entry(key).or_insert_with(|| {
                                let list: Vec<DofId> =
                                    (0..nq).map(|_| { let d = next_dof; next_dof += 1; d }).collect();
                                (list, [l0, l1, l2, l3])
                            });
                            (entry.0.clone(), entry.1)
                        };
                        // Canonical in-face (u, v) of the local in-face (i, j):
                        // transport the local corner directions through the
                        // stored cycle (MFEM `QuadDofOrd`).
                        let grid = [[0usize, 0], [1, 0], [1, 1], [0, 1]];
                        let corner = |v: NodeId| -> usize {
                            stored.iter().position(|&s| s == v).unwrap_or(0)
                        };
                        let (c0, cu, cv) = (corner(l0), corner(l1), corner(l3));
                        let du = [
                            grid[cu][0] as i64 - grid[c0][0] as i64,
                            grid[cu][1] as i64 - grid[c0][1] as i64,
                        ];
                        let dv = [
                            grid[cv][0] as i64 - grid[c0][0] as i64,
                            grid[cv][1] as i64 - grid[c0][1] as i64,
                        ];
                        let pf = p as i64;
                        let u = grid[c0][0] as i64 * pf + i as i64 * du[0] + j as i64 * dv[0];
                        let v = grid[c0][1] as i64 * pf + i as i64 * du[1] + j as i64 * dv[1];
                        debug_assert!(
                            u > 0 && u < pf && v > 0 && v < pf,
                            "build_prism_h1: quad in-face indices ({u}, {v}) out of range"
                        );
                        dofs_flat[base + s] = list[(u as usize - 1) + (v as usize - 1) * ne];
                    }
                    _ => {}
                }
            }
        }
        // Phase 3: element-private interior DOFs, after ALL face DOFs.
        for e in 0..n_elems as u32 {
            let base = e as usize * dofs_per_elem;
            for (s, slot) in slots.iter().enumerate() {
                if matches!(slot, H1PrismSlot::Interior(_)) {
                    dofs_flat[base + s] = next_dof;
                    next_dof += 1;
                }
            }
        }

        let n_dofs = next_dof as usize;

        // DOF coordinates: the linear prism map at `H1PrismPk`'s (Gauss-Lobatto)
        // reference points — the layout and the lattice move together, so the
        // volume dofs need no special case.
        let ref_coords = H1PrismPk::new(p).dof_coords();
        debug_assert_eq!(ref_coords.len(), dofs_per_elem);
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let b = n as usize * dim;
            dof_coords[b..b + dim].copy_from_slice(c);
        }
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            let c: [[f64; 3]; 6] =
                std::array::from_fn(|k| {
                    let x = mesh.node_coords(ns[k]);
                    [x[0], x[1], x[2]]
                });
            let base = e as usize * dofs_per_elem;
            for (s, rc) in ref_coords.iter().enumerate() {
                let did = dofs_flat[base + s] as usize;
                let b = did * dim;
                let (xi, eta, zeta) = (rc[0], rc[1], rc[2]);
                let lam0 = 1.0 - eta - zeta;
                for d in 0..dim {
                    let bottom = lam0 * c[0][d] + eta * c[1][d] + zeta * c[2][d];
                    let top = lam0 * c[3][d] + eta * c[4][d] + zeta * c[5][d];
                    dof_coords[b + d] = (1.0 - xi) * bottom + xi * top;
                }
            }
        }

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(), edge_dof2_map: HashMap::new(),
            phys_to_vertex_dof: HashMap::new(),
            edge_pk_map: edge_map,
            face_pk_map: tri_map.into_iter().map(|(k, (d, _))| (k, d)).collect(),
            quad_face_pk_map: quad_map.into_iter().map(|(k, (d, _))| (k, d)).collect(),
            bubble_dof_start: n_dofs,
            n_volume_dofs: nt * ne,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
        }
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

        // Two-phase DOF assignment, matching MFEM FiniteElementSpace::Construct:
        // first ALL vertex + edge DOFs (in element-traversal order), then ALL
        // interior ("bubble") DOFs — the interior DOF block starts after every
        // edge DOF (ndofs = nvdofs + nedofs + nfdofs + nbdofs).  Assigning the
        // interior DOFs inside the element loop (interleaved with later edge
        // DOFs) misnumbers them vs MFEM at p >= 3 (ex26 P4 regression: interior
        // DOFs at 1373.. instead of 9281..).
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
        }
        // Phase 2: interior (bubble) DOFs, element order, after all edge DOFs.
        for e in 0..n_elems as u32 {
            let base = e as usize * dofs_per_elem;
            let mut off = n_verts + n_edges * edge_dofs_per;
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
            edge_dof_map: HashMap::new(), edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), edge_pk_map,
            face_pk_map: HashMap::new(), quad_face_pk_map: HashMap::new(),
            bubble_dof_start: n_dofs, n_volume_dofs: 0, elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
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

        // The element-local slot layout MUST match the H1 assembly basis the
        // assembler evaluates: `ref_elem_vol_h1` → `HexQk::new(p)` (MFEM
        // `H1_FECollection` GaussLobatto).  Instead of maintaining a
        // hand-written edge/face convention (which drifted from HexQk and
        // broke hex P>=3 assembly + interpolation — same family as the tri
        // Pk>=3 fix), derive the edge/face slot runs directly from
        // HexQk's reference DOF coordinates.
        use fem_element::lagrange::factory::HexQk;
        let ref_coords = HexQk::new(p).dof_coords(); // GLL nodes on [-1,1]^3
        const BND_TOL: f64 = 1e-12;
        // Q1 vertex layout: sign triples of the 8 vertices (bottom ring CCW
        // 0..3, then top ring 4..7).  NOTE: NOT a binary bit encoding.
        const Q1_SIGNS: [[usize; 3]; 8] = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ];

        // ── Edge runs, in HexQk slot order ────────────────────────────────────
        // Each entry is (local a, local b, first slot): its `edge_dofs_per`
        // slots run from near local vertex `a` toward `b` at the GLL
        // parameters gll[1..p].  Slots of one edge are contiguous and share
        // the signature (varying axis, sign bits of the two fixed axes).
        let mut edge_runs: Vec<(usize, usize, usize)> = Vec::new();
        {
            let mut sig2run: HashMap<(usize, u8, u8), usize> = HashMap::new();
            for slot in 8..8 + 12 * edge_dofs_per {
                let rc = &ref_coords[slot];
                let bnd: Vec<usize> = (0..3)
                    .filter(|&d| (rc[d] + 1.0).abs() < BND_TOL || (rc[d] - 1.0).abs() < BND_TOL)
                    .collect();
                debug_assert_eq!(bnd.len(), 2, "HexQk slot {slot} expected on an edge");
                let av = 3 - bnd[0] - bnd[1]; // varying axis (0+1+2 = 3)
                let s0 = if rc[bnd[0]] > 0.0 { 1u8 } else { 0u8 };
                let s1 = if rc[bnd[1]] > 0.0 { 1u8 } else { 0u8 };
                let run = *sig2run.entry((av, s0, s1)).or_insert_with(|| {
                    edge_runs.push((0, 0, slot));
                    edge_runs.len() - 1
                });
                if edge_runs[run].2 == slot {
                    // First slot of the run: it sits at gll[1] along the run
                    // direction, i.e. near the end vertex `a`.
                    let mut slo = [0usize; 3];
                    let mut shi = [0usize; 3];
                    slo[bnd[0]] = s0 as usize; shi[bnd[0]] = s0 as usize;
                    slo[bnd[1]] = s1 as usize; shi[bnd[1]] = s1 as usize;
                    shi[av] = 1;
                    let (lo, hi) = (
                        Q1_SIGNS.iter().position(|&s| s == slo).unwrap(),
                        Q1_SIGNS.iter().position(|&s| s == shi).unwrap(),
                    );
                    if rc[av] < 0.0 {
                        edge_runs[run] = (lo, hi, slot);
                    } else {
                        edge_runs[run] = (hi, lo, slot);
                    }
                }
            }
            assert_eq!(edge_runs.len(), 12, "HexQk: expected 12 edge runs");
        }

        // ── Face runs (HexQk face slot order); orientation-free: face slots
        // are matched across elements by their physical GLL positions. ────────
        // Each entry: (first slot, [4 local vertex indices of the face]).
        let mut face_runs: Vec<(usize, [usize; 4])> = Vec::new();
        {
            let mut sig2run: HashMap<(usize, u8), usize> = HashMap::new();
            for slot in 8 + 12 * edge_dofs_per..8 + 12 * edge_dofs_per + 6 * face_dofs_per {
                let rc = &ref_coords[slot];
                let bnd: Vec<usize> = (0..3)
                    .filter(|&d| (rc[d] + 1.0).abs() < BND_TOL || (rc[d] - 1.0).abs() < BND_TOL)
                    .collect();
                debug_assert_eq!(bnd.len(), 1, "HexQk slot {slot} expected on a face");
                let ax = bnd[0];
                let s = if rc[ax] > 0.0 { 1u8 } else { 0u8 };
                let run = *sig2run.entry((ax, s)).or_insert_with(|| {
                    let verts: Vec<usize> = (0..8)
                        .filter(|&v| Q1_SIGNS[v][ax] == s as usize)
                        .collect();
                    face_runs.push((slot, [verts[0], verts[1], verts[2], verts[3]]));
                    face_runs.len() - 1
                });
                debug_assert!(
                    face_runs[run].0 + face_dofs_per > slot,
                    "HexQk face slots expected contiguous within a run",
                );
            }
            assert_eq!(face_runs.len(), 6, "HexQk: expected 6 face runs");
        }

        // Physical position of a reference point via trilinear mapping of the
        // element's 8 vertex coordinates.  Vertex i sits at the reference
        // corner Q1_SIGNS[i] (ring order — NOT the binary bit pattern of i).
        let phys_pos = |c: &[[f64; 3]], rc: &[f64]| -> [f64; 3] {
            let mut xp = [0.0; 3];
            for i in 0..8 {
                let (sx, sy, sz) = (Q1_SIGNS[i][0], Q1_SIGNS[i][1], Q1_SIGNS[i][2]);
                let nx = if sx == 1 { (1.0 + rc[0]) / 2.0 } else { (1.0 - rc[0]) / 2.0 };
                let ny = if sy == 1 { (1.0 + rc[1]) / 2.0 } else { (1.0 - rc[1]) / 2.0 };
                let nz = if sz == 1 { (1.0 + rc[2]) / 2.0 } else { (1.0 - rc[2]) / 2.0 };
                let ni = nx * ny * nz;
                for d in 0..3 { xp[d] += ni * c[i][d]; }
            }
            xp
        };

        // Phase 1: create edge DOFs (first touch, MFEM block order:
        // all edges before any face/volume DOF).
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            for &(la, lb, _) in &edge_runs {
                get_edge_dofs_pk(ns[la], ns[lb], &mut next_dof, &mut edge_pk_map, edge_dofs_per);
            }
        }

        // Phase 2: create face DOFs + their physical positions (first touch).
        // Canonical face Vec order = the first-touch element's slot order;
        // every element (including the first) resolves its own slots onto the
        // canonical Vec by matching GLL positions, which is orientation- and
        // rotation-proof (unlike vertex-sign conventions).
        let mut face_pos: HashMap<QuadFaceKey, Vec<[f64; 3]>> = HashMap::new();
        let mut elem_coords: Vec<[f64; 3]> = Vec::with_capacity(8);
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            elem_coords.clear();
            for &n in ns.iter().take(8) {
                let c = mesh.node_coords(n);
                elem_coords.push([c[0], c[1], c[2]]);
            }
            for &(slot0, verts) in &face_runs {
                let key = QuadFaceKey::new(ns[verts[0]], ns[verts[1]], ns[verts[2]], ns[verts[3]]);
                face_pos.entry(key).or_insert_with(|| {
                    let dofs: Vec<DofId> = (0..face_dofs_per)
                        .map(|_| { let d = next_dof; next_dof += 1; d })
                        .collect();
                    let pos: Vec<[f64; 3]> = (0..face_dofs_per)
                        .map(|k| phys_pos(&elem_coords, &ref_coords[slot0 + k]))
                        .collect();
                    quad_face_pk_map.insert(key, dofs);
                    pos
                });
            }
        }
        // Phase 3: element slot assignment (vertices → edges → faces →
        // volume), with the edge/face slot order taken from the HexQk runs
        // above.
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 8);
            let base = e as usize * dofs_per_elem;
            dofs_flat[base..base + 8].copy_from_slice(&ns[..8]);
            let mut off = 8;
            // Edges: get_edge_dofs_pk orients the Vec along the call order
            // (ns[la]→ns[lb]), which is exactly the slot-run direction.
            for &(la, lb, _) in &edge_runs {
                let ed = get_edge_dofs_pk(ns[la], ns[lb], &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                for (k, &d) in ed.iter().enumerate() { dofs_flat[base + off + k] = d; }
                off += edge_dofs_per;
            }
            // Faces: match this element's slot GLL positions onto the
            // canonical face Vec (robust to the neighbor's face orientation).
            elem_coords.clear();
            for &n in ns.iter().take(8) {
                let c = mesh.node_coords(n);
                elem_coords.push([c[0], c[1], c[2]]);
            }
            // Scale the match tolerance with the element size so large
            // meshes don't hit absolute-epsilon round-off.
            let scale = elem_coords.iter()
                .fold(1.0_f64, |m, q| m.max(q[0].abs()).max(q[1].abs()).max(q[2].abs()));
            let tol = 1e-9 * scale;
            for &(slot0, verts) in &face_runs {
                let key = QuadFaceKey::new(ns[verts[0]], ns[verts[1]], ns[verts[2]], ns[verts[3]]);
                let fd = &quad_face_pk_map[&key];
                let pos = &face_pos[&key];
                for k in 0..face_dofs_per {
                    let p = phys_pos(&elem_coords, &ref_coords[slot0 + k]);
                    // Exact match up to trilinear-mapping round-off.
                    let j = pos.iter()
                        .position(|&q| (q[0] - p[0]).abs() < tol
                            && (q[1] - p[1]).abs() < tol
                            && (q[2] - p[2]).abs() < tol)
                        .unwrap_or_else(|| panic!(
                            "build_pk_hex: face slot position {p:?} not found on face {key:?}"));
                    dofs_flat[base + off + k] = fd[j];
                }
                off += face_dofs_per;
            }
            // Volume: element-private, created in slot order.
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
        // Edge/face/volume DOF coordinates: the HexQk GLL reference position
        // of each slot, trilinearly mapped through the element.  For edge
        // slots this is the exact linear interpolation at gll[1..p] (MFEM
        // GaussLobatto), NOT the equispaced 1/p..(p-1)/p positions the old
        // code wrote for p >= 3.  Interior DOFs are element-private, and
        // edge/face DOFs map to the same physical point from every incident
        // element, so repeated writes agree.
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            let c: [[f64; 3]; 8] = std::array::from_fn(|i| {
                let ci = mesh.node_coords(ns[i]);
                [ci[0], ci[1], ci[2]]
            });
            let ebase = e as usize * dofs_per_elem;
            for slot in 8..dofs_per_elem {
                let did = dofs_flat[ebase + slot] as usize;
                let xp = phys_pos(&c, &ref_coords[slot]);
                dof_coords[did * dim..did * dim + 3].copy_from_slice(&xp);
            }
        }

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem, elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(), edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), edge_pk_map,
            face_pk_map: HashMap::new(), quad_face_pk_map,
            bubble_dof_start: n_dofs, n_volume_dofs: 0, elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
        }
    }

    // ─── Pk for Pyramid ─────────────────────────────────────────────────────

    /// Reference-grid indices `(i, j, k)` of every H1 pyramid field slot in
    /// **MFEM's entity order** (D191).
    ///
    /// The pyramid field lattice is the collapsed grid whose physical
    /// reference position for lattice node `(i, j, k)` is `(i/p, j/p, k/p)`
    /// (the same points `PyramidPk::dof_coords` enumerates layer-major).  The
    /// slot ORDER reproduced here was dumped from MFEM 4.10
    /// `H1_FECollection(p, 3, GaussLobatto, pyr_type=0)` on a single straight
    /// unit pyramid (probe `tmp/d191/pyr_h1_probe.cpp`, dumps
    /// `$HOME/work/d299/probe_p{2..5}.txt`):
    ///
    /// * slots 0–4: vertices `(0,0,0) (p,0,0) (p,p,0) (0,p,0) (0,0,p)`;
    /// * 8 edge blocks in MFEM's PYRAMID edge-table order and direction
    ///   `(0,1) (1,2) (3,2) (0,3) (0,4) (1,4) (2,4) (3,4)` — note blocks 2/3
    ///   run `v3→v2` and `v0→v3`, **not** `(2,3)/(3,0)`;
    /// * quad base-face block in the H1(quad) interior order: `j` (along
    ///   `v0→v3`) outer, `i` (along `v0→v1`) fastest;
    /// * 4 tri side-face blocks in MFEM face order `(0,1,4) (1,2,4) (2,3,4)
    ///   (3,0,4)`, each in the H1(tri) interior order of that face — rows
    ///   parallel to the base edge for `(0,1,4)/(1,2,4)`, columns across the
    ///   base edge for `(2,3,4)/(3,0,4)`;
    /// * interior block: `k` (layer) outer, then `j`, `i` fastest.
    fn pyramid_entity_slot_grid(p: usize) -> Vec<[usize; 3]> {
        let mut slots: Vec<[usize; 3]> = Vec::with_capacity((p + 1) * (p + 2) * (2 * p + 3) / 6);
        slots.push([0, 0, 0]);
        slots.push([p, 0, 0]);
        slots.push([p, p, 0]);
        slots.push([0, p, 0]);
        slots.push([0, 0, p]);
        if p >= 2 {
            let corners = [[0, 0, 0], [p, 0, 0], [p, p, 0], [0, p, 0], [0, 0, p]];
            let edge_pairs = [[0usize, 1], [1, 2], [3, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]];
            for &[a, b] in &edge_pairs {
                let (ea, eb) = (corners[a], corners[b]);
                for q in 1..p {
                    let pt = [0usize, 1, 2].map(|d| {
                        // Signed lerp: the apex edges decrease a coordinate.
                        let e = (ea[d] as isize) * (p - q) as isize
                            + (eb[d] as isize) * q as isize;
                        (e / p as isize) as usize
                    });
                    slots.push(pt);
                }
            }
            // Quad base face: j (v0→v3) outer, i (v0→v1) fastest.
            for j in 1..p {
                for i in 1..p {
                    slots.push([i, j, 0]);
                }
            }
        }
        if p >= 3 {
            for k in 1..=p - 2 {
                for i in 1..=p - 1 - k {
                    slots.push([i, 0, k]);
                }
            }
            for k in 1..=p - 2 {
                for j in 1..=p - 1 - k {
                    slots.push([p - k, j, k]);
                }
            }
            for i in 1..=p - 2 {
                for k in 1..=p - 1 - i {
                    slots.push([i, p - k, k]);
                }
            }
            for j in 1..=p - 2 {
                for k in 1..=p - 1 - j {
                    slots.push([0, j, k]);
                }
            }
            // Interior: k (layer) outer, j outer, i fastest.
            for k in 1..=p - 2 {
                for j in 1..=p - 1 - k {
                    for i in 1..=p - 1 - k {
                        slots.push([i, j, k]);
                    }
                }
            }
        }
        slots
    }

    /// General-order Lagrange DOF manager for pyramid meshes (D191).
    ///
    /// DOF ordering per element follows MFEM's entity order — see
    /// [`Self::pyramid_entity_slot_grid`] for the slot layout dumped from
    /// MFEM 4.10.  DOF coordinates are evaluated through the *linear*
    /// pyramid transformation at each slot's reference position
    /// `(i/p, j/p, k/p)` (MFEM `SetCurvature`/`ProjectCoefficient`
    /// semantics), so curved-seam rebuilds and coordinates agree for
    /// arbitrary (non-unit) pyramids.
    fn build_pyramid_pk<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        use fem_element::lagrange::pyramid::PyramidPk;
        use fem_element::ReferenceElement;

        let p = order as usize;
        assert!(p >= 1, "build_pyramid_pk: order must be >= 1");
        let dim = 3usize;
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();
        let edge_dofs_per = if p >= 2 { p - 1 } else { 0 };
        let quad_face_dofs_per = if p >= 2 { (p - 1) * (p - 1) } else { 0 };
        let tri_face_dofs_per = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
        let volume_dofs_per = if p >= 3 { (p - 2) * (p - 1) * (2 * p - 3) / 6 } else { 0 };
        let dofs_per_elem = 5 + 8 * edge_dofs_per + quad_face_dofs_per + 4 * tri_face_dofs_per
            + volume_dofs_per;
        let slots = Self::pyramid_entity_slot_grid(p);
        assert_eq!(slots.len(), dofs_per_elem, "pyramid slot table size");

        // Slot ranges (entity blocks) inside one element's dof span.
        let edge_block = 5usize..5 + 8 * edge_dofs_per;
        let quad_block = edge_block.end..edge_block.end + quad_face_dofs_per;
        let tri_block = quad_block.end..quad_block.end + 4 * tri_face_dofs_per;

        let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        let mut face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
        let mut quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        // MFEM PYRAMID edge table: local pairs in block order and direction.
        let edge_pairs = [[0usize, 1], [1, 2], [3, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]];
        // Local tri faces in MFEM face order.
        let tri_faces = [[0usize, 1], [1, 2], [2, 3], [3, 0]];

        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 5);
            let base = e as usize * dofs_per_elem;

            for s in 0..5 {
                dofs_flat[base + s] = ns[s];
            }
            if p >= 2 {
                for (b, &[la, lb]) in edge_pairs.iter().enumerate() {
                    let ed = get_edge_dofs_pk(
                        ns[la], ns[lb], &mut next_dof, &mut edge_pk_map, edge_dofs_per,
                    );
                    let off = edge_block.start + b * edge_dofs_per;
                    for (k, &d) in ed.iter().enumerate() {
                        dofs_flat[base + off + k] = d;
                    }
                }
                let key = QuadFaceKey::new(ns[0], ns[1], ns[2], ns[3]);
                // Allocate the shared vector in slot order (y-outer, x-fast).
                let fd = quad_face_pk_map.entry(key).or_insert_with(|| {
                    (0..quad_face_dofs_per)
                        .map(|_| { let d = next_dof; next_dof += 1; d })
                        .collect()
                });
                for (k, &d) in fd.iter().enumerate() {
                    dofs_flat[base + quad_block.start + k] = d;
                }
            }
            if p >= 3 {
                for (b, &[la, lb]) in tri_faces.iter().enumerate() {
                    let key = FaceKey::new(ns[la], ns[lb], ns[4]);
                    let fd = face_pk_map.entry(key).or_insert_with(|| {
                        (0..tri_face_dofs_per)
                            .map(|_| { let d = next_dof; next_dof += 1; d })
                            .collect()
                    });
                    let off = tri_block.start + b * tri_face_dofs_per;
                    for (k, &d) in fd.iter().enumerate() {
                        dofs_flat[base + off + k] = d;
                    }
                }
                for k in 0..volume_dofs_per {
                    dofs_flat[base + tri_block.end + k] = next_dof;
                    next_dof += 1;
                }
            }
        }

        // Coordinates: every non-vertex slot sits at the linear-pyramid image
        // of its reference position (i/p, j/p, k/p) — the exact analogue of
        // MFEM's `SetCurvature` XYZ projection.
        let n_dofs = next_dof as usize;
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let b = n as usize * dim;
            dof_coords[b..b + dim].copy_from_slice(c);
        }
        let linear = PyramidPk::new(1);
        let mut phi = vec![0.0_f64; 5];
        // D191: `PyramidPk::eval_basis` slots are layer-ordered — the P1
        // element's slot 2/3 carry local vertices 3/2 (see the matching
        // `P1_SLOT_VERTEX` in `Mesh::set_curvature_pyramid5`).
        const P1_SLOT_VERTEX: [usize; 5] = [0, 1, 3, 2, 4];
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            let base = e as usize * dofs_per_elem;
            for (s, g) in slots.iter().enumerate() {
                if s < 5 {
                    continue;
                }
                let theta = [
                    g[0] as f64 / p as f64,
                    g[1] as f64 / p as f64,
                    g[2] as f64 / p as f64,
                ];
                linear.eval_basis(&theta, &mut phi);
                let mut x = [0.0_f64; 3];
                for (k, &phik) in phi.iter().enumerate() {
                    if phik == 0.0 {
                        continue;
                    }
                    let xk = mesh.node_coords(ns[P1_SLOT_VERTEX[k]]);
                    for d in 0..dim {
                        x[d] += phik * xk[d];
                    }
                }
                let did = dofs_flat[base + s] as usize;
                dof_coords[did * dim..did * dim + dim].copy_from_slice(&x[..dim]);
            }
        }

        // `boundary_face_dofs` (assembly/constraints.rs) reads the single
        // mid-edge dof of order-2 spaces from `edge_dof_map`; mirror the
        // (one-entry) edge vectors there.  Orders >= 3 read `edge_pk_map`.
        let edge_dof_map = if p == 2 {
            edge_pk_map.iter().map(|(k, v)| (*k, v[0])).collect()
        } else {
            HashMap::new()
        };

        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map, edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(),
            edge_pk_map,
            face_pk_map, quad_face_pk_map,
            bubble_dof_start: n_dofs, n_volume_dofs: volume_dofs_per, elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
        }
    }

    // ─── Pk (arbitrary order) ─────────────────────────────────────────────────
    //
    // Builds a general-order Lagrange DOF manager for 2D triangle and 3D tetrahedron
    // meshes. The DOF ordering per element matches TriPk (triangles) and — since
    // D157 — MFEM's `H1_TetrahedronElement` (tets, via [`DofManager::build_tet_h1`]).
    // For prism/pyramid, dispatches to specialized builders.

    fn build_pk<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let dim = mesh.dim() as usize;
        let topo_dim = mesh.topological_dim() as usize;
        let p = order as usize;
        // Prism/pyramid/tet dispatch for general order
        if topo_dim == 3 && mesh.n_elements() > 0 {
            let npe = mesh.element_nodes(0).len();
            if npe == 6 { return Self::build_prism_h1(mesh, order); }
            if npe == 5 { return Self::build_pyramid_pk(mesh, order); }
            if npe == 4 { return Self::build_tet_h1(mesh, order); }
        }
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();

        assert!(p >= 1, "build_pk: order must be >= 1");
        assert!(
            topo_dim == 2,
            "build_pk: only 2-D triangles remain (3-D meshes dispatch to \
             build_prism_h1/build_pyramid_pk/build_tet_h1)"
        );

        // Entity DOF counts
        let edge_dofs_per = if p >= 2 { p - 1 } else { 0 };
        let volume_dofs_per = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };

        let dofs_per_elem = (p + 1) * (p + 2) / 2;

        let mut edge_pk_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        // 2-D triangles have no (3-D) face entities; the field stays empty.
        let face_pk_map: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
        let quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        {
            // ── 2-D triangles ────────────────────────────────────────────────
            // Local edge definitions matching TriPk:
            //   edge(0→1), edge(1→2), edge(0→2)
            // Two-phase assignment (MFEM FiniteElementSpace::Construct): all
            // vertex+edge DOFs first, then all interior DOFs in element order.
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
                    // 3 edges, each with (p-1) DOFs, ordered along the element's
                    // local edge direction.  Edge 2 is traversed (v2→v0),
                    // matching the H1 assembly basis H1TriPk / MFEM
                    // H1_TriangleElement, whose edge-3 DOFs run
                    // (cp(p-i), 0) → i.e. from v2 toward v0.
                    let edges = [(n0, n1), (n1, n2), (n2, n0)];
                    let mut off = 3;
                    for &(a, b) in &edges {
                        let edge_dofs = get_edge_dofs_pk(a, b, &mut next_dof, &mut edge_pk_map, edge_dofs_per);
                        for (k, &dof) in edge_dofs.iter().enumerate() {
                            dofs_flat[base + off + k] = dof;
                        }
                        off += edge_dofs_per;
                    }
                }
            }
            // Phase 2: face interior (bubble) DOFs for p >= 3, after ALL edge DOFs.
            if volume_dofs_per > 0 {
                for e in 0..n_elems as u32 {
                    let base = e as usize * dofs_per_elem;
                    let mut off = 3 + 3 * edge_dofs_per;
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
        // 2-D (triangles): the H1 assembly basis is H1TriPk (MFEM
        // GaussLobatto closed points), so DOF k sits at fraction gll[k+1]
        // from canonical-a to canonical-b (identical to equispaced for p<=2).
        {
            let gll_01: Vec<f64> = if p >= 3 {
                let (g, _) = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1);
                g.iter().map(|&x| 0.5 * (x + 1.0)).collect()
            } else {
                Vec::new()
            };
            for (&EdgeKey(a, b), dofs) in &edge_pk_map {
                let ca = mesh.node_coords(a);
                let cb = mesh.node_coords(b);
                for (k, &dof_id) in dofs.iter().enumerate() {
                    let t = if !gll_01.is_empty() { gll_01[k + 1] } else { (k + 1) as f64 / p as f64 };
                    let base = dof_id as usize * dim;
                    for d in 0..dim {
                        dof_coords[base + d] = (1.0 - t) * ca[d] + t * cb[d];
                    }
                }
            }
        }

        // Volume/bubble DOF coordinates: use the H1 assembly reference element
        // for accuracy — 2-D triangles: H1TriPk Gauss-Lobatto nodes — matching
        // the basis the assembler evaluates.
        if volume_dofs_per > 0 {
            let ref_coords: Vec<Vec<f64>> =
                fem_element::lagrange::H1TriPk::new(order as usize).dof_coords();
            // Volume DOFs in the reference element are the LAST volume_dofs_per entries.
            let vol_factory_start = dofs_per_elem - volume_dofs_per;
            let vol_start = n_nodes + edge_pk_map.len() * edge_dofs_per;
            for e in 0..n_elems as u32 {
                let ns = mesh.element_nodes(e);
                // Vertex coordinates for barycentric interpolation
                let c0 = mesh.node_coords(ns[0]);
                let c1 = mesh.node_coords(ns[1]);
                let c2 = mesh.node_coords(ns[2]);
                for k in 0..volume_dofs_per {
                    let dof_id = vol_start + e as usize * volume_dofs_per + k;
                    let base = dof_id * dim;
                    let rc = &ref_coords[vol_factory_start + k];
                    let lam0 = 1.0 - rc[0] - rc[1];
                    for d in 0..dim {
                        dof_coords[base + d] = lam0 * c0[d] + rc[0] * c1[d] + rc[1] * c2[d];
                    }
                }
            }
        }

        let bubble_dof_start = n_nodes + edge_pk_map.len() * edge_dofs_per;
        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(),
            edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(), 
            edge_pk_map,
            face_pk_map,
            quad_face_pk_map,
            bubble_dof_start,
            n_volume_dofs: volume_dofs_per,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
        }
    }

    /// D157: the tetrahedron H¹ DOF numbering of MFEM `H1_FECollection(p, 3)`
    /// — `H1_TetrahedronElement`: the closed **Gauss-Lobatto** node lattice in
    /// MFEM's **entity slot order**.
    ///
    /// Replaces the historical equispaced `factory::TetPk`-convention tet
    /// builder inside [`DofManager::build_pk`].  The two coincide at `p ≤ 2`
    /// (the closed GLL points of `p = 2` are the edge midpoints) and diverge
    /// from `p = 3` (edge/face nodes at `0.2764…, 0.7236…` vs `1/3, 2/3`).
    /// The old builder additionally enumerated the four face blocks in the
    /// factory's face order `(0,1,2) (0,1,3) (0,2,3) (1,2,3)` and ignored the
    /// face orientation of the neighbouring element; MFEM orders the blocks
    /// `TET_FACES = {1,2,3} {0,3,2} {0,1,3} {0,2,1}`, enumerates each block in
    /// `H1_TriangleElement`'s interior order over the face's own vertex order,
    /// and transports shared-face DOFs through `TriDofOrd` — all of which the
    /// barycentric label rotation below reproduces.
    ///
    /// Within one element the slots run `vertices (4) → edges (6·(p−1)) →
    /// faces (4·(p−1)(p−2)/2) → interior ((p−1)(p−2)(p−3)/6)` — exactly the
    /// order of [`fem_element::lagrange::factory::h1_tet_slot_labels`],
    /// verified slot-for-slot against MFEM 4.10 (`tmp/a36_tet_h1_probe.cpp`:
    /// `MakeCartesian3D(2,1,1)` / a 2-tet stack with an orientation-reversed
    /// shared face / a single tet, `p = 2..4`).  Global DOF ids keep this
    /// crate's two-phase creation order (all vertex+edge DOFs element-major,
    /// then all face DOFs, then the element-private interior DOFs) — the same
    /// convention the old builder and the triangle builder use.
    fn build_tet_h1<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        use fem_element::lagrange::factory::{h1_tet_slot_labels, H1TetPk};

        /// MFEM `Tetrahedron::edges` — the element-local edge list and, with
        /// it, the enumeration order of the edge DOF blocks.
        const TET_EDGES: [(usize, usize); 6] =
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        /// MFEM `TET_FACES`: face `f` omits local vertex `f`; the triple is
        /// the face's own vertex order (one slot label per face dof is
        /// `tri_labels[(i, j)]` over exactly this order).
        const TET_FACE_VERTS: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

        let p = order as usize;
        assert!(p >= 1, "build_tet_h1: order must be >= 1");
        assert_eq!(
            mesh.topological_dim() as usize,
            3,
            "build_tet_h1 requires 3-D elements"
        );
        assert_eq!(mesh.dim() as usize, 3, "build_tet_h1 requires a 3-D mesh");
        let dim = 3usize;
        let n_nodes = mesh.n_nodes();
        let n_elems = mesh.n_elements();

        // MFEM's entity slot labels: barycentric exponents `(λ1,λ2,λ3,λ4)`,
        // `λi` the weight of local vertex `i−1`.
        let labels = h1_tet_slot_labels(p);
        let dofs_per_elem = labels.len();
        let ne = p - 1;
        let nt = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
        let nb = if p >= 4 { (p - 1) * (p - 2) * (p - 3) / 6 } else { 0 };

        // `H1_TriangleElement`'s interior slot labels `(λ0,λ1,λ2)` —
        // (j-outer, i-inner) — the in-face enumeration of every tet face
        // block, and the index basis of the canonical face DOF lists.
        let tri_labels: Vec<[usize; 3]> = (1..p)
            .flat_map(|j| (1..(p - j)).map(move |i| [p - i - j, i, j]))
            .collect();
        debug_assert_eq!(tri_labels.len(), nt);

        // One classified slot per element slot.
        enum Slot {
            Vertex(usize),
            Edge { e: usize, k: usize },
            Face { f: usize, i: usize, j: usize },
            Interior,
        }
        let mut slots: Vec<Slot> = Vec::with_capacity(dofs_per_elem);
        for label in &labels {
            let l = [label[0], label[1], label[2], label[3]];
            slots.push(match l.iter().filter(|&&v| v == 0).count() {
                3 => Slot::Vertex(l.iter().position(|&v| v != 0).unwrap()),
                2 => {
                    // The two nonzero entries, ascending local index (a, b):
                    // the label runs (p−k, k) from a to b, so k = l[b] and the
                    // in-edge position is k−1.
                    let nz: Vec<usize> = (0..4).filter(|&i| l[i] != 0).collect();
                    let (a, b) = (nz[0], nz[1]);
                    let e = TET_EDGES
                        .iter()
                        .position(|&(x, y)| x == a && y == b)
                        .expect("build_tet_h1: edge label not in TET_EDGES");
                    Slot::Edge { e, k: l[b] - 1 }
                }
                1 => {
                    let f = l.iter().position(|&v| v == 0).unwrap();
                    let fv = TET_FACE_VERTS[f];
                    Slot::Face { f, i: l[fv[1]], j: l[fv[2]] }
                }
                _ => Slot::Interior,
            });
        }

        let mut edge_map: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
        // Face entities keep the dof list *and* the first-encountering vertex
        // order that defines the list's orientation (the prism builder's
        // convention).
        let mut face_map: HashMap<FaceKey, (Vec<DofId>, [NodeId; 3])> = HashMap::new();
        let mut next_dof = n_nodes as DofId;
        let mut dofs_flat = vec![0u32; n_elems * dofs_per_elem];

        // Phase 1: vertices and edges, element-major, slot order within one
        // element — the same creation order the old builder (and MFEM's
        // vertex/edge phase) used.
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            assert!(ns.len() >= 4, "build_tet_h1 requires 4-noded elements");
            let n4 = [ns[0], ns[1], ns[2], ns[3]];
            let base = e as usize * dofs_per_elem;
            for (s, slot) in slots.iter().enumerate() {
                match slot {
                    Slot::Vertex(v) => dofs_flat[base + s] = n4[*v],
                    Slot::Edge { e: ei, k } => {
                        let (la, lb) = (n4[TET_EDGES[*ei].0], n4[TET_EDGES[*ei].1]);
                        let key = EdgeKey::new(la, lb);
                        let list = edge_map.entry(key).or_insert_with(|| {
                            (0..ne).map(|_| { let d = next_dof; next_dof += 1; d }).collect()
                        });
                        // `k` counts from the local first vertex; the map is
                        // canonical (ascending vertex id).
                        dofs_flat[base + s] =
                            if la == key.0 { list[*k] } else { list[ne - 1 - *k] };
                    }
                    _ => {}
                }
            }
        }
        // Phase 2: face DOFs, after ALL edge DOFs.  The face list is created
        // once per canonical face, indexed by `tri_labels` over the first
        // encounter's vertex order; each element's slot label is rotated from
        // the local face orientation into that canonical one (MFEM
        // `TriDofOrd`).
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            let n4 = [ns[0], ns[1], ns[2], ns[3]];
            let base = e as usize * dofs_per_elem;
            for (s, slot) in slots.iter().enumerate() {
                if let Slot::Face { f, i, j } = *slot {
                    let fv = TET_FACE_VERTS[f];
                    let (a, b, c) = (n4[fv[0]], n4[fv[1]], n4[fv[2]]);
                    let key = FaceKey::new(a, b, c);
                    let (list, canon) = {
                        let entry = face_map.entry(key).or_insert_with(|| {
                            let list: Vec<DofId> =
                                (0..nt).map(|_| { let d = next_dof; next_dof += 1; d }).collect();
                            (list, [a, b, c])
                        });
                        (entry.0.clone(), entry.1)
                    };
                    let local = [p - i - j, i, j];
                    let local_verts = [a, b, c];
                    let canon_label = [
                        local[local_verts.iter().position(|&v| v == canon[0]).unwrap_or(0)],
                        local[local_verts.iter().position(|&v| v == canon[1]).unwrap_or(0)],
                        local[local_verts.iter().position(|&v| v == canon[2]).unwrap_or(0)],
                    ];
                    let ci = tri_labels
                        .iter()
                        .position(|l| *l == canon_label)
                        .unwrap_or_else(|| {
                            panic!("build_tet_h1: face dof label {canon_label:?} not in table")
                        });
                    dofs_flat[base + s] = list[ci];
                }
            }
        }
        // Phase 3: element-private interior DOFs, after ALL face DOFs.
        for e in 0..n_elems as u32 {
            let base = e as usize * dofs_per_elem;
            for (s, slot) in slots.iter().enumerate() {
                if matches!(slot, Slot::Interior) {
                    dofs_flat[base + s] = next_dof;
                    next_dof += 1;
                }
            }
        }

        let n_dofs = next_dof as usize;

        // DOF coordinates: the linear tet map at `H1TetPk`'s (Gauss-Lobatto)
        // reference points — the layout and the lattice move together, so the
        // edge/face/interior dofs need no special case (the prism builder's
        // rule).  Vertex DOFs keep the mesh table bit-for-bit.
        let ref_coords = H1TetPk::new(p).dof_coords();
        debug_assert_eq!(ref_coords.len(), dofs_per_elem);
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];
        for n in 0..n_nodes as u32 {
            let c = mesh.node_coords(n);
            let b = n as usize * dim;
            dof_coords[b..b + dim].copy_from_slice(c);
        }
        for e in 0..n_elems as u32 {
            let ns = mesh.element_nodes(e);
            let c: [[f64; 3]; 4] = std::array::from_fn(|k| {
                let x = mesh.node_coords(ns[k]);
                [x[0], x[1], x[2]]
            });
            let base = e as usize * dofs_per_elem;
            for (s, rc) in ref_coords.iter().enumerate() {
                let did = dofs_flat[base + s] as usize;
                if did < n_nodes {
                    continue; // vertices already exact from the table
                }
                let b = did * dim;
                let (x, y, z) = (rc[0], rc[1], rc[2]);
                let lam0 = 1.0 - x - y - z;
                for d in 0..dim {
                    dof_coords[b + d] = lam0 * c[0][d] + x * c[1][d] + y * c[2][d] + z * c[3][d];
                }
            }
        }

        let bubble_dof_start = n_nodes + edge_map.len() * ne + face_map.len() * nt;
        DofManager {
            order, n_dofs, dofs_flat, dofs_per_elem,
            elem_dof_offsets: None, dof_coords, dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map: HashMap::new(),
            edge_dof2_map: HashMap::new(), phys_to_vertex_dof: HashMap::new(),
            edge_pk_map: edge_map,
            face_pk_map: face_map.into_iter().map(|(k, (d, _))| (k, d)).collect(),
            quad_face_pk_map: HashMap::new(),
            bubble_dof_start,
            n_volume_dofs: nb,
            elem_orders: None,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
        }
    }

    /// Return the polynomial order for element `elem`.
    /// For uniform-order DofManagers, returns `self.order`.
    /// For variable-order DofManagers, returns the per-element order.
    pub fn element_order(&self, elem: ElemId) -> u8 {
        self.elem_orders.as_ref().map_or(self.order, |orders| orders[elem as usize])
    }

    // ─── D56: periodic per-element geometry ────────────────────────────────────

    /// D61: build the DOF numbering of a **geometrically periodic** mesh.
    ///
    /// On a periodic mesh whose connectivity has been vertex-merged, plain
    /// vertex-pair entity keys are no longer injective: when a periodic
    /// direction carries fewer than three cells, distinct torus edges/faces
    /// share one folded vertex set (e.g. on the fully periodic 2×2×2 hex mesh
    /// every element's corner set is the whole folded vertex table, so all 24
    /// torus faces collapse onto 6 [`QuadFaceKey`]s — Q2 would count 34 DOFs
    /// instead of 64).  MFEM avoids the collision the same way on meshes with
    /// ≥3 cells per direction because its vertex-merged topology happens to
    /// keep entity sets distinct; below that it cannot build the mesh at all
    /// (`Mesh::GenerateFaces` rejects the 3-element face), so there is no C++
    /// count to copy and the torus-entity count is the reference.
    ///
    /// The fix mirrors MFEM's design (number DOFs per mesh entity, then merge
    /// through the periodic identification): run the ordinary builder on an
    /// **un-merged** view of the mesh ([`UnfoldedPeriodicMesh`] — each
    /// element's own pre-merge geometry corners, from the order-1 periodic
    /// snapshot), then quotient the resulting entity DOFs to torus entities.
    /// Two entity occurrences are the same torus entity iff their folded
    /// vertex set *and* their translation-invariant geometric frame agree:
    ///
    /// - edges: folded vertex pair + the unfolded endpoint displacement taken
    ///   along the folded low→high orientation.  The displacement separates
    ///   the two half-period arcs between the same vertex pair that appear at
    ///   two cells per direction; it is invariant under the periodic
    ///   translation, so the two covering images of one seam edge unify.
    /// - faces: sorted folded corner ids + the corner offsets relative to the
    ///   lowest folded corner's image (ordered by folded id), which fixes the
    ///   face's phase with respect to the vertex lattice.
    ///
    /// Displacements are quantized against the mesh scale, so covering images
    /// match bit-for-bit after the translation round-trip.
    ///
    /// Final DOF ids are assigned to the quotient classes in order of their
    /// first-touch (un-merged) id, which reproduces the plain builder's
    /// numbering **bit-for-bit** whenever no key collision exists (≥3 cells
    /// per direction) — there the quotient only re-joins the covering images
    /// of seam entities that the folded build merged through the shared key.
    fn build_periodic<M: MeshTopology>(mesh: &M, order: u8) -> Self {
        let wrapper = UnfoldedPeriodicMesh { inner: mesh };
        let uf = Self::build(&wrapper, order);
        let n_nodes = mesh.n_nodes();
        let dim = uf.dim;

        // Unfolded geometry node → folded vertex id (position-wise corner
        // pairing between the pre-merge snapshot and the merged connectivity).
        let n_uf_nodes = wrapper.n_nodes();
        let mut fold = vec![u32::MAX; n_uf_nodes];
        for e in 0..mesh.n_elements() as u32 {
            let gn = mesh.geometry_nodes(e);
            let fnodes = mesh.element_nodes(e);
            for (&g, &f) in gn.iter().zip(fnodes.iter()) {
                fold[g as usize] = f;
            }
        }

        // Quantized displacement of two unfolded points (scale-relative, so
        // the periodic translation round-trip cancels exactly).
        let mut scale = 1.0_f64;
        for g in 0..n_uf_nodes as u32 {
            for &c in wrapper.node_coords(g) {
                scale = scale.max(c.abs());
            }
        }
        let quant = |v: f64| -> i64 { (v / scale * 1.0e10).round() as i64 };
        let pt = |g: NodeId| -> [i64; 3] {
            let c = wrapper.node_coords(g);
            [
                quant(c[0]),
                quant(c[1]),
                if dim > 2 { quant(c[2]) } else { 0 },
            ]
        };

        // One un-merged entity occurrence: its torus signature, its DOFs in
        // stored order, and (edges) whether the stored order runs against the
        // folded low→high orientation.  `base` is the physical image of the
        // reference corner (folded-lowest), used to align multi-DOF members
        // whose creation orientations differ.  `src` identifies the public
        // entity map the occurrence was registered in.
        struct Occ {
            sig: EntitySig,
            dofs: Vec<DofId>,
            flip: bool,
            base: [f64; 3],
            src: u8,
        }
        const SRC_EDGE_DOF: u8 = 0;
        const SRC_EDGE_DOF2: u8 = 1;
        const SRC_EDGE_PK: u8 = 2;
        const SRC_FACE_PK: u8 = 3;
        const SRC_QFACE_PK: u8 = 4;
        let fpt = |g: NodeId| -> [f64; 3] {
            let c = wrapper.node_coords(g);
            [c[0], c[1], if dim > 2 { c[2] } else { 0.0 }]
        };

        let mut occs: Vec<Occ> = Vec::new();
        let push_edge = |a: NodeId, b: NodeId, dofs: Vec<DofId>, src: u8, occs: &mut Vec<Occ>| {
            let (fa, fb) = (fold[a as usize], fold[b as usize]);
            let fkey = EdgeKey::new(fa, fb);
            // Canonical stored order: along folded low → high.
            let flip = fa > fb;
            let delta = if flip {
                [pt(a)[0] - pt(b)[0], pt(a)[1] - pt(b)[1], pt(a)[2] - pt(b)[2]]
            } else {
                [pt(b)[0] - pt(a)[0], pt(b)[1] - pt(a)[1], pt(b)[2] - pt(a)[2]]
            };
            // Reference corner: the endpoint folding to the folded-low id
            // (on a self-loop, both endpoints fold together — use the key's
            // low endpoint consistently).
            let base = if fa <= fb { fpt(a) } else { fpt(b) };
            occs.push(Occ {
                sig: EntitySig::Edge { key: (fkey.0, fkey.1), delta },
                dofs,
                flip,
                base,
                src,
            });
        };
        for &EdgeKey(a, b) in uf.edge_dof_map.keys() {
            let d = uf.edge_dof_map[&EdgeKey(a, b)];
            push_edge(a, b, vec![d], SRC_EDGE_DOF, &mut occs);
        }
        for (&EdgeKey(a, b), dofs) in &uf.edge_dof2_map {
            push_edge(a, b, dofs.to_vec(), SRC_EDGE_DOF2, &mut occs);
        }
        for (&EdgeKey(a, b), dofs) in &uf.edge_pk_map {
            push_edge(a, b, dofs.clone(), SRC_EDGE_PK, &mut occs);
        }
        for (&FaceKey(a, b, c), dofs) in &uf.face_pk_map {
            let mut corners = [
                (fold[a as usize], a, pt(a)),
                (fold[b as usize], b, pt(b)),
                (fold[c as usize], c, pt(c)),
            ];
            corners.sort_by_key(|&(f, _, _)| f);
            let base = fpt(corners[0].1);
            let base_q = corners[0].2;
            let off = |p: [i64; 3]| [p[0] - base_q[0], p[1] - base_q[1], p[2] - base_q[2]];
            occs.push(Occ {
                sig: EntitySig::TriFace {
                    key: (corners[0].0, corners[1].0, corners[2].0),
                    offs: [
                        (corners[1].0, off(corners[1].2)),
                        (corners[2].0, off(corners[2].2)),
                    ],
                },
                dofs: dofs.clone(),
                flip: false,
                base,
                src: SRC_FACE_PK,
            });
        }
        for (&QuadFaceKey(a, b, c, d), dofs) in &uf.quad_face_pk_map {
            let mut corners = [
                (fold[a as usize], a, pt(a)),
                (fold[b as usize], b, pt(b)),
                (fold[c as usize], c, pt(c)),
                (fold[d as usize], d, pt(d)),
            ];
            corners.sort_by_key(|&(f, _, _)| f);
            let base = fpt(corners[0].1);
            let base_q = corners[0].2;
            let off = |p: [i64; 3]| [p[0] - base_q[0], p[1] - base_q[1], p[2] - base_q[2]];
            occs.push(Occ {
                sig: EntitySig::QuadFace {
                    key: (corners[0].0, corners[1].0, corners[2].0, corners[3].0),
                    offs: [
                        (corners[1].0, off(corners[1].2)),
                        (corners[2].0, off(corners[2].2)),
                        (corners[3].0, off(corners[3].2)),
                    ],
                },
                dofs: dofs.clone(),
                flip: false,
                base,
                src: SRC_QFACE_PK,
            });
        }

        // Group occurrences into torus-entity classes.
        occs.sort_by(|x, y| x.sig.cmp(&y.sig));
        let mut i = 0;
        let mut classes: Vec<Vec<Occ>> = Vec::new();
        while i < occs.len() {
            let sig = occs[i].sig.clone();
            let mut members = Vec::new();
            while i < occs.len() && occs[i].sig == sig {
                members.push(std::mem::replace(
                    &mut occs[i],
                    Occ {
                        sig: sig.clone(),
                        dofs: Vec::new(),
                        flip: false,
                        base: [0.0; 3],
                        src: 0,
                    },
                ));
                i += 1;
            }
            classes.push(members);
        }

        // Canonical member of each class = earliest first-touch (its stored
        // order is the one the folded build would have created); final ids are
        // assigned to classes in first-touch order, which reproduces the
        // plain builder's numbering bit-for-bit when no collision exists.
        let mut order_key: Vec<(DofId, usize)> = Vec::with_capacity(classes.len());
        for (ci, members) in classes.iter().enumerate() {
            let first = members.iter().map(|o| o.dofs[0]).min().unwrap();
            order_key.push((first, ci));
        }
        // Volume-interior DOFs (and any other DOF outside the entity maps)
        // are element-private singletons; they join the same first-touch
        // ordering so the global id sequence stays the builder's.
        let mut relabel = vec![u32::MAX; uf.n_dofs];
        for g in 0..n_uf_nodes as u32 {
            if fold[g as usize] != u32::MAX {
                relabel[g as usize] = fold[g as usize];
            }
        }
        let mut singletons: Vec<DofId> = (n_uf_nodes as DofId..uf.n_dofs as DofId)
            .filter(|&d| relabel[d as usize] == u32::MAX)
            .collect();
        let classified: std::collections::HashSet<DofId> = classes
            .iter()
            .flat_map(|ms| ms.iter().flat_map(|o| o.dofs.iter().copied()))
            .collect();
        singletons.retain(|&d| !classified.contains(&d));
        for &d in &singletons {
            order_key.push((d, usize::MAX));
        }
        order_key.sort();
        // Per-map records of the quotient classes: (folded key, final dofs).
        let mut rec_edge_dof: Vec<(EdgeKey, DofId)> = Vec::new();
        let mut rec_edge_dof2: Vec<(EdgeKey, [DofId; 2])> = Vec::new();
        let mut rec_edge_pk: Vec<(EdgeKey, Vec<DofId>)> = Vec::new();
        let mut rec_face_pk: Vec<(FaceKey, Vec<DofId>)> = Vec::new();
        let mut rec_qface_pk: Vec<(QuadFaceKey, Vec<DofId>)> = Vec::new();
        let mut next = n_nodes as DofId;
        for (first, ci) in &order_key {
            if *ci == usize::MAX {
                relabel[*first as usize] = next;
                next += 1;
                continue;
            }
            let members = &classes[*ci];
            // Canonical member: earliest first-touch.  Its (re-oriented) DOF
            // order defines the class vector — the order the folded build
            // would have created.
            let canon = members
                .iter()
                .min_by_key(|o| o.dofs[0])
                .unwrap();
            let canon_orient = |o: &Occ, v: &[DofId]| -> Vec<DofId> {
                if o.flip { v.iter().rev().copied().collect() } else { v.to_vec() }
            };
            let canon_dofs = canon_orient(canon, &canon.dofs);
            for (k, &d) in canon_dofs.iter().enumerate() {
                relabel[d as usize] = next + k as DofId;
            }
            for o in members {
                if std::ptr::eq(o, canon) {
                    continue;
                }
                let member_dofs = canon_orient(o, &o.dofs);
                assert_eq!(member_dofs.len(), canon_dofs.len(), "class size mismatch");
                for (k, &d) in member_dofs.iter().enumerate() {
                    let target = if canon_dofs.len() > 1 {
                        // Align by physical position: the member is a periodic
                        // translate of the canonical occurrence, so its DOF k
                        // sits at the canonical DOF j's position shifted by
                        // t = base_canon − base_member.  (The builder's own
                        // stored order for the two covering images can differ
                        // — e.g. opposite local face templates — so plain
                        // position-wise union would misalign multi-DOF
                        // faces.)
                        let tol = scale * 1.0e-9;
                        let pos = |e: DofId| -> [f64; 3] {
                            let c = &uf.dof_coords[e as usize * dim..e as usize * dim + dim];
                            [c[0], c[1], if dim > 2 { c[2] } else { 0.0 }]
                        };
                        let t = [
                            canon.base[0] - o.base[0],
                            canon.base[1] - o.base[1],
                            canon.base[2] - o.base[2],
                        ];
                        let canon_pos: Vec<[f64; 3]> =
                            canon_dofs.iter().map(|&e| pos(e)).collect();
                        let distinct = canon_pos.iter().all(|&p| {
                            canon_pos.iter().all(|&q| {
                                p == q
                                    || (p[0] - q[0]).abs() > tol
                                        || (p[1] - q[1]).abs() > tol
                                        || (p[2] - q[2]).abs() > tol
                            })
                        });
                        if distinct {
                            let p = pos(d);
                            canon_pos
                                .iter()
                                .position(|&q| {
                                    (q[0] - (p[0] + t[0])).abs() <= tol
                                        && (q[1] - (p[1] + t[1])).abs() <= tol
                                        && (q[2] - (p[2] + t[2])).abs() <= tol
                                })
                                .unwrap_or_else(|| panic!(
                                    "build_periodic: entity DOF at {p:?}+{t:?} not found on the \
                                     canonical image"
                                ))
                        } else {
                            // DOF positions not distinguishable (builder wrote
                            // coincident coordinates): keep stored order — the
                            // same convention as the folded build.
                            k
                        }
                    } else {
                        k
                    };
                    relabel[d as usize] = next + target as DofId;
                }
            }
            // Record the class vector for the public entity map it came from.
            let final_vec: Vec<DofId> =
                (0..canon_dofs.len() as DofId).map(|k| next + k).collect();
            match &members[0].sig {
                EntitySig::Edge { key, .. } => {
                    let fkey = EdgeKey::new(key.0, key.1);
                    match members[0].src {
                        SRC_EDGE_DOF => rec_edge_dof.push((fkey, final_vec[0])),
                        SRC_EDGE_DOF2 => {
                            rec_edge_dof2.push((fkey, [final_vec[0], final_vec[1]]));
                        }
                        _ => rec_edge_pk.push((fkey, final_vec)),
                    }
                }
                EntitySig::TriFace { key, .. } => {
                    rec_face_pk
                        .push((FaceKey::new(key.0, key.1, key.2), final_vec));
                }
                EntitySig::QuadFace { key, .. } => {
                    rec_qface_pk.push((
                        QuadFaceKey::new(key.0, key.1, key.2, key.3),
                        final_vec,
                    ));
                }
            }
            next += canon_dofs.len() as DofId;
        }

        // Rewrite the element tables through the quotient.
        let mut dofs_flat = uf.dofs_flat;
        for v in dofs_flat.iter_mut() {
            let r = relabel[*v as usize];
            assert!(r != u32::MAX, "build_periodic: unlabelled DOF {v}");
            *v = r;
        }

        // Public entity maps carry the quotient classes directly: the class
        // record already holds the folded key and the canonical final vector
        // (folded low→high for edges — the convention `get_edge_dofs_pk`
        // stores vectors in).
        let edge_pk_map: HashMap<EdgeKey, Vec<DofId>> =
            rec_edge_pk.into_iter().collect();
        let edge_dof_map: HashMap<EdgeKey, DofId> =
            rec_edge_dof.into_iter().collect();
        let edge_dof2_map: HashMap<EdgeKey, [DofId; 2]> =
            rec_edge_dof2.into_iter().collect();
        let face_pk_map: HashMap<FaceKey, Vec<DofId>> =
            rec_face_pk.into_iter().collect();
        let quad_face_pk_map: HashMap<QuadFaceKey, Vec<DofId>> =
            rec_qface_pk.into_iter().collect();

        let bubble_dof_start = if uf.bubble_dof_start < uf.n_dofs {
            relabel[uf.bubble_dof_start as usize] as usize
        } else {
            next as usize
        };

        DofManager {
            order: uf.order,
            n_dofs: next as usize,
            dofs_flat,
            dofs_per_elem: uf.dofs_per_elem,
            elem_dof_offsets: uf.elem_dof_offsets,
            dof_coords: vec![0.0; next as usize * dim],
            dim,
            n_vertex_dofs: n_nodes,
            edge_dof_map,
            edge_dof2_map,
            phys_to_vertex_dof: uf.phys_to_vertex_dof,
            edge_pk_map,
            face_pk_map,
            quad_face_pk_map,
            bubble_dof_start,
            n_volume_dofs: uf.n_volume_dofs,
            elem_orders: uf.elem_orders,
            edge_variants: HashMap::new(),
            face_variants: HashMap::new(),
        }
    }

    /// Rebuild the DOF coordinate table from each element's **own** geometry
    /// nodes ([`MeshTopology::geometry_nodes`] / [`MeshTopology::geom_coords_of`]).
    ///
    /// A geometrically periodic mesh carries a per-element geometry snapshot
    /// (MFEM `MakePeriodic` materializes the nodal `Nodes` grid function
    /// *before* merging vertices), so a shared (seam) DOF sits at a
    /// *different* physical point in each of its elements — the folded vertex
    /// table is only one image and the fold-based table built above also
    /// places seam edge/face DOFs at chord midpoints no element can see.
    /// MFEM's `GridFunction::ProjectCoefficient` resolves this by evaluating
    /// the coefficient per element at each local DOF's nodal point under that
    /// element's own transform and letting the last element win for shared
    /// DOFs.  This method produces exactly the coordinate table that a
    /// dof-wise evaluation then reproduces: each element maps its reference
    /// DOF positions through its own geometry and overwrites the global
    /// entries in element order.
    ///
    /// D62: the evaluation runs through the mesh's own **geometry order** —
    /// affine (order-1) snapshots use the trilinear/Q1 basis over the corner
    /// snapshot, curved periodic meshes (order >= 2) use the corresponding
    /// high-order geometry basis over the element's full geometry node list.
    ///
    /// Element slot layouts must match the reference elements the H1
    /// assembler evaluates: the *field* element per shape is QuadQk / HexQk /
    /// H1TriPk / H1TetPk / H1PrismPk, and for pyramids the MFEM entity-order
    /// slot table [`Self::pyramid_entity_slot_grid`] (D191 — `PyramidPk`'s
    /// own `dof_coords` are layer-major and would silently permute the field
    /// slots).  The
    /// *geometry* element follows whatever order
    /// `Mesh::set_curvature_*` writes its table in: QuadQk / HexQk / H1TriPk /
    /// H1TetPk for quad/hex/tri/tet, but **layer-major `PrismPk`** for prisms
    /// (frozen by `crates/mesh/tests/d152_prism_curvature.rs`) and `PyramidPk`
    /// for pyramids.  A slot-count mismatch against these factories is a hard
    /// error (D182): silently keeping fold-based coordinates is exactly the
    /// failure mode the count check used to hide.
    fn rebuild_dof_coords_periodic<M: MeshTopology>(&mut self, mesh: &M) {
        use fem_element::lagrange::factory::{HexQk, H1TetPk, QuadQk};
        use fem_element::lagrange::{H1PrismPk, H1TriPk, PrismPk, PyramidPk};

        let dim = self.dim;
        let topo_dim = mesh.topological_dim() as usize;
        let geom_order = mesh.geom_order() as usize;
        let n_elems = mesh.n_elements();
        for e in 0..n_elems as u32 {
            let p = self.element_order(e) as usize;
            let npe = mesh.element_nodes(e).len();
            // Field-slot reference positions in builder slot order; the
            // geometry element is evaluated at those positions.
            let (ref_dofs, geom_elem): (Vec<[f64; 3]>, Box<dyn ReferenceElement>) =
                match (npe, topo_dim) {
                    (4, 2) => (
                        QuadQk::new(p).dof_coords().iter().map(|c| [c[0], c[1], c.get(2).copied().unwrap_or(0.0)]).collect(),
                        Box::new(QuadQk::new(geom_order)),
                    ),
                    (8, _) => (
                        HexQk::new(p).dof_coords().iter().map(|c| [c[0], c[1], c.get(2).copied().unwrap_or(0.0)]).collect(),
                        Box::new(HexQk::new(geom_order)),
                    ),
                    (3, 2) => (
                        H1TriPk::new(p).dof_coords().iter().map(|c| [c[0], c[1], c.get(2).copied().unwrap_or(0.0)]).collect(),
                        Box::new(H1TriPk::new(geom_order)),
                    ),
                    // D157: the tet *field* slots follow `H1TetPk` (MFEM
                    // entity order, Gauss-Lobatto lattice) and
                    // `set_curvature_tet4` lays the geometry table out in the
                    // same slots — the equispaced `factory::TetPk` disagrees
                    // with both from p = 3 on.
                    (4, _) => (
                        H1TetPk::new(p).dof_coords().iter().map(|c| [c[0], c[1], c.get(2).copied().unwrap_or(0.0)]).collect(),
                        Box::new(H1TetPk::new(geom_order)),
                    ),
                    // D182: the prism *field* slots follow `H1PrismPk`
                    // (MFEM's entity order — the layout `build_prism_h1`
                    // numbers `element_dofs` in), while the *geometry* table
                    // stays layer-major `PrismPk`
                    // (`set_curvature_prism6`'s frozen contract, D152).  The
                    // two share the Gauss-Lobatto lattice and the dof count,
                    // so a wrong field element here passes every count check
                    // and silently permutes the coordinates.
                    (6, _) => (
                        H1PrismPk::new(p).dof_coords().iter().map(|c| [c[0], c[1], c.get(2).copied().unwrap_or(0.0)]).collect(),
                        Box::new(PrismPk::new(geom_order)),
                    ),
                    // D191: the pyramid *field* slots follow the MFEM
                    // entity-order slot table (the layout `build_pyramid_pk`
                    // numbers `element_dofs` in), while the *geometry* table
                    // stays layer-major `PyramidPk`
                    // (`set_curvature_pyramid5`'s frozen contract).
                    (5, _) => (
                        Self::pyramid_entity_slot_grid(p)
                            .iter()
                            .map(|g| {
                                [
                                    g[0] as f64 / p as f64,
                                    g[1] as f64 / p as f64,
                                    g[2] as f64 / p as f64,
                                ]
                            })
                            .collect(),
                        Box::new(PyramidPk::new(geom_order)),
                    ),
                    _ => continue,
                };
            let dofs = self.element_dofs(e).to_vec();
            // D182 hardening: a slot-count mismatch means the field reference
            // element above no longer matches the builder that produced
            // `element_dofs`.  This used to `continue` (keep fold-based
            // coordinates) — which is precisely how the prism arm's *order*
            // mismatch (same count, wrong permutation) stayed invisible.  Any
            // future family split must fail loudly here instead.
            assert_eq!(
                dofs.len(),
                ref_dofs.len(),
                "rebuild_dof_coords_periodic: element {e} (npe {npe}, order {p}) has {} \
                 dofs but the field reference element for this shape has {} — \
                 the builder and the periodic-rebuild field element diverged",
                dofs.len(),
                ref_dofs.len(),
            );
            let gnodes = mesh.geometry_nodes(e);
            let mut phi = vec![0.0_f64; gnodes.len()];
            for (slot, rc) in ref_dofs.iter().enumerate() {
                geom_elem.eval_basis(rc, &mut phi);
                let mut x = [0.0_f64; 3];
                for (k, &phik) in phi.iter().enumerate() {
                    if phik == 0.0 {
                        continue;
                    }
                    let ck = mesh.geom_coords_of(gnodes[k]);
                    for d in 0..dim {
                        x[d] += phik * ck[d];
                    }
                }
                let did = dofs[slot] as usize;
                self.dof_coords[did * dim..did * dim + dim].copy_from_slice(&x[..dim]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    /// D61 pin: on periodic meshes with ≥3 cells per direction (no vertex-set
    /// collisions) the quotient numbering must reproduce the plain folded
    /// build bit-for-bit — same `dofs_flat`, same `n_dofs`, same entity maps.
    fn assert_periodic_matches_folded<M: MeshTopology>(mesh: &M, order: u8) {
        let dm_new = DofManager::new(mesh, order);
        let dm_ref = DofManager::build(mesh, order);
        // Apply the same D56 coordinate rebuild to the reference so both sides
        // carry per-element seam coordinates.
        let mut dm_ref = dm_ref;
        dm_ref.rebuild_dof_coords_periodic(mesh);
        assert_eq!(dm_new.n_dofs, dm_ref.n_dofs, "order {order}: n_dofs");
        assert_eq!(dm_new.dofs_flat, dm_ref.dofs_flat, "order {order}: dofs_flat");
        assert_eq!(dm_new.edge_dof_map, dm_ref.edge_dof_map, "order {order}: edge_dof_map");
        assert_eq!(dm_new.edge_dof2_map, dm_ref.edge_dof2_map, "order {order}: edge_dof2_map");
        assert_eq!(dm_new.edge_pk_map, dm_ref.edge_pk_map, "order {order}: edge_pk_map");
        assert_eq!(dm_new.face_pk_map, dm_ref.face_pk_map, "order {order}: face_pk_map");
        assert_eq!(
            dm_new.quad_face_pk_map, dm_ref.quad_face_pk_map,
            "order {order}: quad_face_pk_map"
        );
        assert_eq!(dm_new.dof_coords, dm_ref.dof_coords, "order {order}: dof_coords");
    }

    #[test]
    fn d61_periodic_numbering_matches_folded_build_hex() {
        for n in [3usize, 4] {
            let base = Mesh::<3>::make_cartesian_3d(
                n, n, n, fem_mesh::ElementType::Hex8, 1.0, 1.0, 1.0, true,
            );
            let mesh = base
                .make_periodic(
                    &[
                        (5, 3, [1.0, 0.0, 0.0]),
                        (2, 4, [0.0, 1.0, 0.0]),
                        (1, 6, [0.0, 0.0, 1.0]),
                    ],
                    1e-10,
                )
                .unwrap();
            for order in [1u8, 2, 3, 4] {
                assert_periodic_matches_folded(&mesh, order);
            }
        }
    }

    #[test]
    fn d61_periodic_numbering_matches_folded_build_2d() {
        // 4x4 quad strip periodic in both directions.
        let base = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
        let mesh = base
            .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
            .unwrap();
        for order in [1u8, 2, 3, 4] {
            assert_periodic_matches_folded(&mesh, order);
        }
        let tri_base = Mesh::<2>::unit_square_tri(4);
        let tri_mesh = tri_base
            .make_periodic(&[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])], 1e-10)
            .unwrap();
        for order in [1u8, 2, 3] {
            assert_periodic_matches_folded(&tri_mesh, order);
        }
    }

    #[test]
    fn pk3_matches_build_p3_tri() {
        // Verify that build_pk(mesh, 3) produces the same DOF ordering as build_p3_tri
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm_pk = DofManager::new(&mesh, 3);
        let dm_p3 = DofManager::build_p3(&mesh);
        assert_eq!(dm_pk.n_dofs, dm_p3.n_dofs, "n_dofs mismatch");
        assert_eq!(dm_pk.dofs_flat, dm_p3.dofs_flat, "dofs_flat mismatch");
        assert_eq!(dm_pk.dof_coords.len(), dm_p3.dof_coords.len(), "dof_coords length mismatch");
        for (a, b) in dm_pk.dof_coords.iter().zip(dof_coords_iter(&dm_p3)) {
            assert!((a - b).abs() < 1e-14, "dof_coords mismatch: {} vs {}", a, b);
        }
    }

    fn dof_coords_iter(dm: &DofManager) -> impl Iterator<Item = f64> + '_ {
        dm.dof_coords.iter().copied()
    }

    #[test]
    fn pk2_matches_build_p2() {
        // Verify that build_pk(mesh, 2) produces the same DOF ordering as build_p2
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm_pk = DofManager::new(&mesh, 2);
        let dm_p2 = DofManager::build_p2(&mesh);
        assert_eq!(dm_pk.n_dofs, dm_p2.n_dofs, "n_dofs mismatch");
        assert_eq!(dm_pk.dofs_flat, dm_p2.dofs_flat, "dofs_flat mismatch");
        assert_eq!(dm_pk.dof_coords, dm_p2.dof_coords, "dof_coords mismatch");
    }

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
        // Verify edge_pk_map has consistent entries (build_pk uses edge_pk_map).
        assert!(!dm.edge_pk_map.is_empty(), "TetP3 should have non-empty edge_pk_map");
        // Each edge DOF pair must be unique.
        let mut all_dof_pairs: Vec<Vec<u32>> = dm.edge_pk_map.values().map(|v| v.clone()).collect();
        let len = all_dof_pairs.len();
        all_dof_pairs.sort();
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
        // MFEM `H1_WedgeElement` slot order: slots 0-5 are the six vertices
        // (bottom triangle 0-2, then top triangle 3-5) — probe
        // `tmp/a34_prism_h1_probe.cpp`, p=3 elem 0 (D168 ground truth; the
        // layer-major layout this test previously pinned put the top vertices
        // at slots 30-32, which is `PrismPk`'s order, not MFEM's).
        for e in 0..m.n_elements() as u32{
            let d=dm.element_dofs(e); let n=m.element_nodes(e);
            assert_eq!(&d[..6],&n[..6]);
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

    /// The prism H¹ slot layout must match MFEM's `H1_WedgeElement` slot by
    /// slot: slot `s`'s dof sits at the linear prism map's image of
    /// `H1PrismPk::dof_coords()[s]` — including the Gauss-Lobatto points from
    /// p = 3 and the rotated-`QuadDofOrd`/`TriDofOrd` orientation handling on
    /// the last (rotated) element (D168; probe
    /// `tmp/a34_prism_h1_probe.cpp`, whose p = 2/3/4 tables this reproduces).
    #[test]
    fn prism_h1_slots_match_mfem_wedge_element() {
        use fem_element::lagrange::{h1_prism_slots, H1PrismPk};
        use fem_element::ReferenceElement;

        // The probe's wedge stack (nphi = 3), last element's vertex list
        // rotated like the probe's mode 1.
        let coords = vec![
            0.,0.,0., 1.,0.,0., 0.,1.,0.,
            0.,0.,1., 1.,0.,1., 0.,1.,1.,
            0.,0.,2., 1.,0.,2., 0.,1.,2.,
            0.,0.,3., 1.,0.,3., 0.,1.,3.,
        ];
        let conn = vec![0,1,2,3,4,5, 3,4,5,6,7,8, 7,8,6,10,11,9];
        let m = Mesh::<3>::uniform(
            coords, conn, vec![1,1,1],
            fem_mesh::ElementType::Prism6, vec![], vec![], fem_mesh::ElementType::Tri3,
        );

        for p in 1..=4u8 {
            let dm = DofManager::new(&m, p);
            let fe = H1PrismPk::new(p as usize);
            let rc = fe.dof_coords();
            assert_eq!(rc.len(), h1_prism_slots(p as usize).len());
            let npe = (p as usize + 1) * (p as usize + 1) * (p as usize + 2) / 2;
            assert_eq!(dm.dofs_per_elem, npe, "p={p}: dofs per element");
            for e in 0..m.n_elements() as u32 {
                let dofs = dm.element_dofs(e);
                let ns = m.element_nodes(e);
                let c: [[f64; 3]; 6] =
                    std::array::from_fn(|k| {
                        let x = m.coords_of(ns[k]);
                        [x[0], x[1], x[2]]
                    });
                for (s, r) in rc.iter().enumerate() {
                    // The linear prism map at slot s's reference point.
                    let (xi, eta, zeta) = (r[0], r[1], r[2]);
                    let lam0 = 1.0 - eta - zeta;
                    let mut want = [0.0_f64; 3];
                    for d in 0..3 {
                        let bottom = lam0 * c[0][d] + eta * c[1][d] + zeta * c[2][d];
                        let top = lam0 * c[3][d] + eta * c[4][d] + zeta * c[5][d];
                        want[d] = (1.0 - xi) * bottom + xi * top;
                    }
                    let got = dm.dof_coord(dofs[s]);
                    let delta: f64 =
                        (0..3).map(|d| (got[d] - want[d]).abs()).fold(0.0, f64::max);
                    assert!(
                        delta < 1e-12,
                        "p={p} elem {e} slot {s}: dof {} at {got:?}, want {want:?} (|Δ|={delta:.3e})",
                        dofs[s]
                    );
                }
                // Vertex slots must be the mesh's own vertices (MFEM's first
                // six dofs) and distinct slots must be distinct dofs.
                assert_eq!(&dofs[..6], &ns[..6], "p={p} elem {e}: vertex slots");
                let mut sorted = dofs.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), dofs.len(), "p={p} elem {e}: duplicate dofs in slot list");
            }
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

