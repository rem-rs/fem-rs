use fem_core::{EdgeId, ElemId, FaceId, FemError, FemResult, NodeId};
use crate::{
    boundary::{BoundaryTag, NamedAttributeRegistry},
    element_type::ElementType,
    topology::MeshTopology,
};

const _MAX_EDGE: u32 = ElemId::MAX;

/// Local face vertex tables for each element type.
fn local_face_verts(dim: usize, elem_type: ElementType) -> Vec<Vec<usize>> {
    match (dim, elem_type) {
        // Triangle faces (edges opposite each vertex)
        (2, ElementType::Tri3 | ElementType::Tri6) => vec![
            vec![1, 2], // opposite v₀
            vec![0, 2], // opposite v₁
            vec![0, 1], // opposite v₂
        ],
        // Quad faces (edges in CCW order)
        (2, ElementType::Quad4) => vec![
            vec![0, 1], // bottom
            vec![1, 2], // right
            vec![2, 3], // top
            vec![3, 0], // left
        ],
        // Tet faces (triangles opposite each vertex)
        (3, ElementType::Tet4 | ElementType::Tet10) => vec![
            vec![1, 2, 3], // opposite v₀
            vec![0, 2, 3], // opposite v₁
            vec![0, 1, 3], // opposite v₂
            vec![0, 1, 2], // opposite v₃
        ],
        // Hex faces
        (3, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27) => vec![
            vec![0, 1, 2, 3], // z=-1 (bottom)
            vec![4, 5, 6, 7], // z= 1 (top)
            vec![0, 1, 5, 4], // y=-1 (near)
            vec![2, 3, 7, 6], // y= 1 (far)
            vec![0, 3, 7, 4], // x=-1 (left)
            vec![1, 2, 6, 5], // x= 1 (right)
        ],
        _ => vec![],
    }
}

/// Local edge vertex pairs (sorted globally, not local index order) for each element type.
/// Returns flat `Vec<(local_node_a, local_node_b)>` for each element.
fn local_element_edges(dim: usize, elem_type: ElementType) -> Vec<[usize; 2]> {
    match (dim, elem_type) {
        (2, ElementType::Tri3 | ElementType::Tri6) => vec![[0, 1], [1, 2], [0, 2]],
        (2, ElementType::Quad4) => vec![[0, 1], [1, 2], [2, 3], [0, 3]],
        (3, ElementType::Tet4 | ElementType::Tet10) => vec![
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
        ],
        (3, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27) => vec![
            [0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7],
        ],
        _ => vec![],
    }
}

/// High-order geometry data for curved meshes (set via [`Mesh::set_curvature`]).
///
/// When present, the mesh's geometry is represented using polynomial order
/// `order` basis functions (isoparametric or superparametric).  The `conn`
/// array maps each element to its high-order geometry nodes, and `coords`
/// stores their physical coordinates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GeometryData {
    /// Geometric polynomial order (1 = linear, 2 = quadratic, …).
    pub order: u8,
    /// High-order geometry connectivity: for element `e`, the geometry-node
    /// indices are `conn[e * nodes_per_elem .. (e+1) * nodes_per_elem]`.
    pub conn: Vec<NodeId>,
    /// Number of geometry nodes per element.
    pub nodes_per_elem: usize,
    /// Coordinates of geometry nodes.  Length = `n_nodes * D`.
    /// Indices 0..n_vertices are the original vertex coordinates; additional
    /// indices beyond that are edge/face/interior geometry nodes.
    pub coords: Vec<f64>,
    /// Total number of geometry nodes (≤ `coords.len() / D`).
    pub n_nodes: usize,
}

/// Unstructured mesh with uniform or mixed element types.
///
/// When all elements share the same type, `elem_type` determines the
/// uniform stride into `conn`.  For mixed-element meshes, the optional
/// `elem_types` and `elem_offsets` fields provide per-element type and
/// connectivity offsets (CSR-like).
///
/// Node coordinates are stored in a flat array: index of node `n`'s
/// first coordinate is `n as usize * D`.
///
/// # Type parameter
/// `D` is the spatial dimension (2 = 2-D, 3 = 3-D).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Mesh<const D: usize> {
    /// Flat node coordinate array.  Length = `n_nodes * D`.
    pub coords: Vec<f64>,
    /// Flat element connectivity (0-based node indices).
    /// Uniform: length = `n_elems * npe`.
    /// Mixed:   length = sum of nodes per element (indexed via `elem_offsets`).
    pub conn: Vec<NodeId>,
    /// Physical group tag per element (e.g. material id). Length = `n_elems`.
    pub elem_tags: Vec<i32>,
    /// Element type (uniform across the mesh, or the "primary" type for mixed).
    pub elem_type: ElementType,
    /// Flat boundary face connectivity (0-based node indices).
    pub face_conn: Vec<NodeId>,
    /// Physical group tag per boundary face (e.g. BC label). Length = `n_faces`.
    pub face_tags: Vec<BoundaryTag>,
    /// Face type (one dimension lower than `elem_type`, or primary face type).
    pub face_type: ElementType,

    // ─── Mixed-element support (None = uniform) ──────────────────────────
    /// Per-element type.  `None` means all elements share `elem_type`.
    pub elem_types: Option<Vec<ElementType>>,
    /// CSR-like start offsets into `conn`.  Length = `n_elems + 1`.
    /// `elem_offsets[e]..elem_offsets[e+1]` are the conn indices for element `e`.
    /// `None` means uniform stride `elem_type.nodes_per_element()`.
    pub elem_offsets: Option<Vec<usize>>,
    /// Per-face type.  `None` means all faces share `face_type`.
    pub face_types: Option<Vec<ElementType>>,
    /// CSR-like start offsets into `face_conn`.  Length = `n_faces + 1`.
    pub face_offsets: Option<Vec<usize>>,

    /// For boundary face `f`, the element that owns it.
    /// `None` until built (lazy construction via [`build_face_to_elem`]).
    pub face_to_elem: Option<Vec<ElemId>>,

    // ─── Edge-level data (lazy) ──────────────────────────────────────────────

    /// Flat array of edge node pairs: `[a0, b0, a1, b1, …]`.
    /// Built by [`build_edge_connectivity`].
    pub edge_conn: Vec<NodeId>,
    /// CSR-like: `edge_to_elem[2*eid]` = first element, `[2*eid+1]` = second or `ElemId::MAX`.
    /// Built by [`build_edge_connectivity`].
    pub edge_to_elem: Vec<ElemId>,

    // ─── High-order geometry (set via set_curvature) ─────────────────────────

    /// High-order geometry data.  `None` means linear (Q1/P1) geometry.
    #[cfg_attr(feature = "serialize", serde(default))]
    #[cfg_attr(feature = "serialize", serde(skip_serializing_if = "Option::is_none"))]
    pub geometry: Option<GeometryData>,
}

impl<const D: usize> Mesh<D> {
    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.coords.len() / D
    }
    /// Number of volume elements.
    pub fn n_elems(&self) -> usize {
        if let Some(ref offsets) = self.elem_offsets {
            offsets.len() - 1
        } else {
            let npe = self.elem_type.nodes_per_element();
            if npe == 0 { 0 } else { self.conn.len() / npe }
        }
    }
    /// Number of boundary faces.
    pub fn n_faces(&self) -> usize {
        if let Some(ref offsets) = self.face_offsets {
            offsets.len() - 1
        } else {
            let npf = self.face_type.nodes_per_element();
            if npf == 0 { 0 } else { self.face_conn.len() / npf }
        }
    }

    /// Geometric type of volume element `e` (mixed meshes: `elem_types`).
    #[inline]
    pub fn element_type_at(&self, e: ElemId) -> ElementType {
        if let Some(ref types) = self.elem_types {
            types[e as usize]
        } else {
            self.elem_type
        }
    }

    /// Geometric type of boundary face `f` (mixed boundaries: `face_types`).
    #[inline]
    pub fn face_type_at(&self, f: FaceId) -> ElementType {
        if let Some(ref types) = self.face_types {
            types[f as usize]
        } else {
            self.face_type
        }
    }

    /// Coordinates of node `n` as a `[f64; D]` array.
    #[inline]
    pub fn coords_of(&self, n: NodeId) -> [f64; D] {
        let off = n as usize * D;
        std::array::from_fn(|i| self.coords[off + i])
    }

    // ─── High-order geometry support ───────────────────────────────────────────

    /// Geometric polynomial order of the mesh (1 = linear, the default).
    pub fn geom_order(&self) -> u8 {
        self.geometry.as_ref().map_or(1, |g| g.order)
    }

    /// Number of geometry nodes (0 if no high-order geometry).
    pub fn n_geom_nodes(&self) -> usize {
        self.geometry.as_ref().map_or(0, |g| g.n_nodes)
    }

    /// Promote the mesh to high-order (curved) geometry of the given order.
    ///
    /// This creates a `GeometryData` entry that maps each element to its
    /// high-order geometry nodes, using the **same nodes** as the mesh vertices
    /// for P1 (order 1). For order > 1, the high-order geometry nodes are
    /// generated at the reference DOF locations, but currently uses the same
    /// coordinates as the linear mesh (callers should snap or project afterward).
    ///
    /// For a sphere mesh, after calling `snap_to_sphere()`, the high-order
    /// geometry nodes will lie on the sphere surface.
    ///
    /// Currently supports `Quad4` element type.  `order = 0` or `1` resets
    /// to linear geometry.
    pub fn set_curvature(&mut self, order: u8) {
        if order <= 1 {
            self.geometry = None;
            return;
        }
        let p = order as usize;
        assert_eq!(D, 3, "set_curvature: surface mesh requires D = 3");

        if self.elem_type == ElementType::Tri3 {
            self.set_curvature_tri3(p);
            return;
        }
        assert!(self.elem_type == ElementType::Quad4,
            "set_curvature: only Tri3 and Quad4 are currently supported");

        use std::collections::HashMap;
        use fem_element::lagrange::factory::QuadQk;
        use fem_element::ReferenceElement;

        let n_elems = self.n_elems();
        let quad = QuadQk::new(p);
        let npe_new = quad.n_dofs(); // (p+1)²
        let n_verts = self.n_nodes();

        // Reference node positions in QuadQk DOF order
        let dof_ref = quad.dof_coords(); // Vec<Vec<f64>>, each of length 2

        // Get vertex coordinates of each element for Q1 interpolation
        let elem_verts: Vec<[NodeId; 4]> = (0..n_elems)
            .map(|e| {
                let n = self.elem_nodes(e as ElemId);
                [n[0], n[1], n[2], n[3]]
            })
            .collect();

        // Build geometry connectivity: each edge is shared via an edge map.
        let mut geom_conn = vec![0u32; n_elems * npe_new];
        let mut edge_map: HashMap<(NodeId, NodeId), Vec<NodeId>> = HashMap::new();
        let mut next_geom = n_verts as NodeId;

        // Geometry coords start with the original vertex coords
        let mut geom_coords = self.coords.clone();

        for e in 0..n_elems {
            let verts = elem_verts[e];
            let base = e * npe_new;

            // Q1 basis functions at reference point (xi, eta) for interpolation:
            // φ₀=(1-ξ)(1-η)/4, φ₁=(1+ξ)(1-η)/4, φ₂=(1+ξ)(1+η)/4, φ₃=(1-ξ)(1+η)/4
            let q1_eval = |xi: f64, eta: f64| -> [f64; 4] {
                [0.25*(1.0-xi)*(1.0-eta), 0.25*(1.0+xi)*(1.0-eta),
                 0.25*(1.0+xi)*(1.0+eta), 0.25*(1.0-xi)*(1.0+eta)]
            };

            // Vertex DOFs: indices 0,1,2,3 are the original vertices
            geom_conn[base..base+4].copy_from_slice(&verts);

            // Edge vertex pairs for the quad: bottom(0→1), right(1→2), top(2→3), left(3→0)
            let edge_verts = [
                (verts[0], verts[1]), // bottom
                (verts[1], verts[2]), // right
                (verts[2], verts[3]), // top
                (verts[3], verts[0]), // left
            ];

            let n_edge_dofs = p - 1; // interior nodes per edge (not counting vertices)
            let mut pos = base + 4;

            // Process each edge
            for ei in 0..4 {
                let (a, b) = edge_verts[ei];
                let key = if a < b { (a, b) } else { (b, a) };
                let ids = edge_map.entry(key).or_insert_with(|| {
                    let ca = self.coords_of(a);
                    let cb = self.coords_of(b);
                    let mut new_ids = Vec::with_capacity(n_edge_dofs);
                    for k in 1..p {
                        let t = k as f64 / p as f64;
                        let mut x = [0.0; D];
                        for d in 0..D { x[d] = (1.0 - t) * ca[d] + t * cb[d]; }
                        geom_coords.extend_from_slice(&x);
                        new_ids.push(next_geom);
                        next_geom += 1;
                    }
                    new_ids
                });
                // DOFs along this edge in QuadQk order
                for id in ids.iter() {
                    geom_conn[pos] = *id;
                    pos += 1;
                }
            }

            // Interior DOFs: (p-1)² — positions from dof_ref
            for idx in 4 + 4 * n_edge_dofs..npe_new {
                let rc = &dof_ref[idx];
                let xi = rc[0];
                let eta = rc[1];
                let q1 = q1_eval(xi, eta);
                let mut x = [0.0; D];
                for v in 0..4 {
                    let vc = self.coords_of(verts[v]);
                    for d in 0..D { x[d] += q1[v] * vc[d]; }
                }
                geom_coords.extend_from_slice(&x);
                geom_conn[pos] = next_geom;
                pos += 1;
                next_geom += 1;
            }
        }

        self.geometry = Some(GeometryData {
            order,
            conn: geom_conn,
            nodes_per_elem: npe_new,
            coords: geom_coords,
            n_nodes: next_geom as usize,
        });
    }

    fn set_curvature_tri3(&mut self, p: usize) {
        use fem_element::lagrange::TriPk;
        use fem_element::ReferenceElement;
        let n_elems = self.n_elems();
        let npe_new = (p + 1) * (p + 2) / 2;

        let tri_pk = TriPk::new(p);
        let dof_coords = tri_pk.dof_coords();

        let mut geo_conn = Vec::with_capacity(n_elems * npe_new);
        let mut geo_coords = self.coords.clone();
        let mut next_id = self.n_nodes() as NodeId;

        for e in 0..n_elems as NodeId {
            let v = self.elem_nodes(e);
            let (x0, x1, x2) = (self.node_coords(v[0]), self.node_coords(v[1]), self.node_coords(v[2]));
            for d in 0..npe_new {
                let xi = &dof_coords[d];
                let is_v0 = xi[0].abs() < 1e-12 && xi[1].abs() < 1e-12;
                let is_v1 = (xi[0]-1.0).abs() < 1e-12;
                let is_v2 = xi[0].abs() < 1e-12 && (xi[1]-1.0).abs() < 1e-12;
                if is_v0 { geo_conn.push(v[0]); }
                else if is_v1 { geo_conn.push(v[1]); }
                else if is_v2 { geo_conn.push(v[2]); }
                else {
                    let x = x0[0]*(1.0-xi[0]-xi[1]) + x1[0]*xi[0] + x2[0]*xi[1];
                    let y = x0[1]*(1.0-xi[0]-xi[1]) + x1[1]*xi[0] + x2[1]*xi[1];
                    let z = x0[2]*(1.0-xi[0]-xi[1]) + x1[2]*xi[0] + x2[2]*xi[1];
                    let inv = 1.0 / (x*x + y*y + z*z).sqrt();
                    geo_conn.push(next_id);
                    geo_coords.push(x*inv); geo_coords.push(y*inv); geo_coords.push(z*inv);
                    next_id += 1;
                }
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geo_conn,
            nodes_per_elem: npe_new,
            coords: geo_coords,
            n_nodes: next_id as usize,
        });
    }

    // ─── Geometric transforms ────────────────────────────────────────────────

    /// Apply a coordinate transform `f` to every mesh node.
    /// The closure receives `[x, y]` (2-D) or `[x, y, z]` (3-D) and returns
    /// the transformed coordinate array.
    ///
    /// If high-order geometry is present (via [`set_curvature`](Self::set_curvature)),
    /// both vertex and geometry-node coordinates are transformed.
    pub fn transform(&mut self, mut f: impl FnMut([f64; D]) -> [f64; D]) {
        // Transform vertex coordinates
        for n in 0..self.n_nodes() {
            let out = f(self.coords_of(n as NodeId));
            let off = n * D;
            self.coords[off..off + D].copy_from_slice(&out);
        }
        // Transform geometry node coordinates (if any)
        if let Some(ref mut geo) = self.geometry {
            let n_geom = geo.coords.len() / D;
            for n in 0..n_geom {
                let off = n * D;
                let mut p = [0.0; D];
                p.copy_from_slice(&geo.coords[off..off + D]);
                let q = f(p);
                geo.coords[off..off + D].copy_from_slice(&q);
            }
        }
    }

    /// Translate all nodes by vector `t`.
    pub fn translate(&mut self, t: [f64; D]) {
        self.transform(|p| std::array::from_fn(|i| p[i] + t[i]));
    }

    /// Uniformly scale all node coordinates about the origin.
    pub fn scale(&mut self, s: f64) {
        self.transform(|p| std::array::from_fn(|i| p[i] * s));
    }

    /// Create a new mesh by displacing each node by a vector field.
    ///
    /// `displacement` is a flat array in **component-major** order
    /// (all x-dofs `0..n_nodes`, then y-dofs, …) matching the
    /// `VectorH1Space` DOF layout.  The first `n_nodes` entries of each
    /// component block are assumed to correspond to the vertex DOFs.
    /// Geometry nodes (high‑order curvature) are displaced identically.
    pub fn apply_displacement(&self, displacement: &[f64], vdim: usize) -> Self {
        let n_nodes = self.n_nodes();
        let mut new_coords = self.coords.clone();
        for n in 0..n_nodes {
            for d in 0..D.min(vdim) {
                let idx = d * n_nodes + n;
                let val = if idx < displacement.len() { displacement[idx] } else { 0.0 };
                new_coords[n * D + d] += val;
            }
        }
        if let Some(ref geo) = self.geometry {
            let n_geom = geo.coords.len() / D;
            let mut new_geo_coords = geo.coords.clone();
            for n in 0..n_geom {
                for d in 0..D.min(vdim) {
                    let idx = d * n_nodes + n;
                    let val = if idx < displacement.len() { displacement[idx] } else { 0.0 };
                    new_geo_coords[n * D + d] += val;
                }
            }
            let mut m = self.clone();
            m.coords = new_coords;
            m.geometry = Some(GeometryData { coords: new_geo_coords, ..geo.clone() });
            m
        } else {
            let mut m = self.clone();
            m.coords = new_coords;
            m
        }
    }

    /// Apply a 3×3 rotation matrix to all nodes (3-D only, panics for D ≠ 3).
    pub fn rotate_3d(&mut self, rot: &[[f64; 3]; 3]) {
        assert_eq!(D, 3, "rotate_3d requires dim=3");
        self.transform(|p| {
            let mut q = [0.0; D];
            for i in 0..3 { for j in 0..3 { q[i] += rot[i][j] * p[j]; } }
            q
        });
    }

    /// Apply a 2×2 rotation matrix to all nodes (2-D only, panics for D ≠ 2).
    pub fn rotate_2d(&mut self, rot: &[[f64; 2]; 2]) {
        assert_eq!(D, 2, "rotate_2d requires dim=2");
        self.transform(|p| {
            let mut q = [0.0; D];
            for i in 0..2 { for j in 0..2 { q[i] += rot[i][j] * p[j]; } }
            q
        });
    }

    /// Node indices of volume element `e`.
    #[inline]
    pub fn elem_nodes(&self, e: ElemId) -> &[NodeId] {
        if let Some(ref offsets) = self.elem_offsets {
            let start = offsets[e as usize];
            let end = offsets[e as usize + 1];
            &self.conn[start..end]
        } else {
            let npe = self.elem_type.nodes_per_element();
            let off = e as usize * npe;
            &self.conn[off..off + npe]
        }
    }

    /// Node indices of boundary face `f`.
    #[inline]
    pub fn bface_nodes(&self, f: FaceId) -> &[NodeId] {
        if let Some(ref offsets) = self.face_offsets {
            let start = offsets[f as usize];
            let end = offsets[f as usize + 1];
            &self.face_conn[start..end]
        } else {
            let npf = self.face_type.nodes_per_element();
            let off = f as usize * npf;
            &self.face_conn[off..off + npf]
        }
    }

    /// Whether this mesh has mixed element types.
    pub fn is_mixed(&self) -> bool {
        self.elem_types.is_some()
    }

    /// Compute the axis-aligned bounding box of the mesh.
    ///
    /// Returns `(min_coords, max_coords)` where each is a `[f64; D]` array.
    ///
    /// # Panics
    /// Panics if the mesh has no nodes.
    pub fn bounding_box(&self) -> ([f64; D], [f64; D]) {
        assert!(self.n_nodes() > 0, "bounding_box: mesh has no nodes");
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        for n in 0..self.n_nodes() as NodeId {
            let c = self.coords_of(n);
            for d in 0..D {
                if c[d] < lo[d] { lo[d] = c[d]; }
                if c[d] > hi[d] { hi[d] = c[d]; }
            }
        }
        (lo, hi)
    }

    /// Return the sorted, deduplicated set of boundary face tags.
    pub fn unique_boundary_tags(&self) -> Vec<BoundaryTag> {
        let mut tags: Vec<BoundaryTag> = self.face_tags.clone();
        tags.sort_unstable();
        tags.dedup();
        tags
    }

    /// Return all element ids that carry the given material tag.
    pub fn element_ids_with_tag(&self, tag: i32) -> Vec<ElemId> {
        let mut out = Vec::new();
        for e in 0..self.n_elems() {
            if self.elem_tags[e] == tag {
                out.push(e as ElemId);
            }
        }
        out
    }

    /// Return all boundary face ids that carry the given boundary tag.
    pub fn face_ids_with_tag(&self, tag: BoundaryTag) -> Vec<FaceId> {
        let mut out = Vec::new();
        for f in 0..self.n_faces() {
            if self.face_tags[f] == tag {
                out.push(f as FaceId);
            }
        }
        out
    }

    /// Query element ids by named attribute set.
    pub fn element_ids_for_named_set(
        &self,
        registry: &NamedAttributeRegistry,
        set_name: &str,
    ) -> FemResult<Vec<ElemId>> {
        let set = registry.get(set_name).ok_or_else(|| {
            FemError::Mesh(format!("named attribute set not found: {set_name}"))
        })?;
        let mut out = Vec::new();
        for e in 0..self.n_elems() {
            if set.has_element_tag(self.elem_tags[e]) {
                out.push(e as ElemId);
            }
        }
        Ok(out)
    }

    /// Query boundary face ids by named attribute set.
    pub fn face_ids_for_named_set(
        &self,
        registry: &NamedAttributeRegistry,
        set_name: &str,
    ) -> FemResult<Vec<FaceId>> {
        let set = registry.get(set_name).ok_or_else(|| {
            FemError::Mesh(format!("named attribute set not found: {set_name}"))
        })?;
        let mut out = Vec::new();
        for f in 0..self.n_faces() {
            if set.has_boundary_tag(self.face_tags[f]) {
                out.push(f as FaceId);
            }
        }
        Ok(out)
    }

    /// Detect periodic boundaries from a `boundary 0` mesh.
    ///
    /// For meshes with no boundary faces (like `periodic-square.mesh`), this
    /// method analyzes element connectivity to find unpaired edges (virtual
    /// boundary edges), detects periodicity by matching opposite edges, and
    /// returns a new mesh with periodic nodes merged.
    ///
    /// This enables `InteriorFaceList` to correctly find all interior faces
    /// on periodic meshes, since merged nodes create shared node keys.
    ///
    /// # Arguments
    /// * `tol` — geometric tolerance for node matching (e.g. 1e-8).
    ///
    /// # Returns
    /// A new mesh with periodic node pairs merged.  The returned mesh has
    /// no boundary faces on the periodic sides.
    ///
    /// # Panics
    /// Panics if the mesh already has boundary faces or is not 2-D.
    pub fn detect_periodic_boundary(&self, tol: f64) -> FemResult<Self> {
        assert_eq!(D, 2, "detect_periodic_boundary requires dim=2");
        if self.n_boundary_faces() > 0 {
            // Mesh already has boundary faces — use make_periodic directly instead.
            return Err(FemError::Mesh(
                "detect_periodic_boundary: mesh already has boundary faces".into()
            ));
        }

        // ── Step 1: find all virtual boundary edges ──────────────────────────
        // An edge is a "virtual boundary edge" if it appears in only one element's
        // connectivity.  We enumerate all element edges and track which elements
        // reference each edge.
        use std::collections::HashMap;

        // key: sorted node pair → (elem_id, local_face_idx, unsorted_nodes)
        let mut edge_map: HashMap<Vec<NodeId>, (ElemId, usize, Vec<NodeId>)> = HashMap::new();

        for e in self.elem_iter() {
            let en = self.elem_nodes(e);
            let npe = en.len();
            let local_faces = match npe {
                3 => vec![vec![0usize, 1], vec![1, 2], vec![0, 2]],
                4 => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
                _ => return Err(FemError::Mesh("unsupported element type".into())),
            };
            for (li, lf) in local_faces.iter().enumerate() {
                let unsorted: Vec<NodeId> = lf.iter().map(|&k| en[k]).collect();
                let mut key: Vec<NodeId> = unsorted.clone();
                key.sort_unstable();
                edge_map.entry(key).or_insert((e, li, unsorted));
                // Note: for boundary 0 closed meshes, ALL edges should appear
                // twice, but since nodes aren't identified across periodic
                // boundaries, some edges appear only once.
            }
        }

        // Edges that appear only once — these are virtual boundary edges
        let boundary_edges: Vec<(ElemId, Vec<NodeId>)> = edge_map
            .into_values()
            .map(|(e, _li, nodes)| (e, nodes))
            .collect();

        if boundary_edges.is_empty() {
            // No boundary edges found — mesh is already topologically closed.
            return Ok(self.clone());
        }

        // ── Step 2: compute edge geometry ────────────────────────────────────
        struct BdrEdge {
            elem: ElemId,
            nodes: Vec<NodeId>,
            mid: [f64; 2],
            normal: [f64; 2],
        }

        let mut bdr: Vec<BdrEdge> = boundary_edges
            .iter()
            .map(|(elem, nodes)| {
                let p0 = self.node_coords(nodes[0]);
                let p1 = self.node_coords(nodes[1]);
                let dx = p1[0] - p0[0];
                let dy = p1[1] - p0[1];
                let len = (dx * dx + dy * dy).sqrt();
                // Left-of-edge normal: (-dy, dx) / len
                let nx = -dy / len;
                let ny = dx / len;
                // Adjust to point outward from the element
                let elem_nodes = self.elem_nodes(*elem);
                let centroid_x: f64 = elem_nodes.iter().map(|&n| self.node_coords(n)[0]).sum::<f64>() / elem_nodes.len() as f64;
                let centroid_y: f64 = elem_nodes.iter().map(|&n| self.node_coords(n)[1]).sum::<f64>() / elem_nodes.len() as f64;
                let mid_x = (p0[0] + p1[0]) / 2.0;
                let mid_y = (p0[1] + p1[1]) / 2.0;
                // Check if normal points from centroid toward midpoint
                let dot = nx * (mid_x - centroid_x) + ny * (mid_y - centroid_y);
                let (nx, ny) = if dot >= 0.0 { (nx, ny) } else { (-nx, -ny) };
                BdrEdge {
                    elem: *elem,
                    nodes: nodes.clone(),
                    mid: [mid_x, mid_y],
                    normal: [nx, ny],
                }
            })
            .collect();

        // ── Step 3: group edges by normal direction ──────────────────────────
        // For a rectangular periodic domain, normals are approx (-1,0), (1,0),
        // (0,-1), (0,1).  Group by which axis component is dominant.
        // left: nx < -0.5, right: nx > 0.5, bottom: ny < -0.5, top: ny > 0.5
        let mut left: Vec<usize> = Vec::new();
        let mut right: Vec<usize> = Vec::new();
        let mut bottom: Vec<usize> = Vec::new();
        let mut top: Vec<usize> = Vec::new();

        for (i, e) in bdr.iter().enumerate() {
            if e.normal[0] < -0.5 {
                left.push(i);
            } else if e.normal[0] > 0.5 {
                right.push(i);
            } else if e.normal[1] < -0.5 {
                bottom.push(i);
            } else if e.normal[1] > 0.5 {
                top.push(i);
            }
        }

        // Sort each group by position along the face (for consistent pairing)
        // left/right: sort by y (increasing)
        left.sort_by(|&a, &b| bdr[a].mid[1].partial_cmp(&bdr[b].mid[1]).unwrap());
        right.sort_by(|&a, &b| bdr[a].mid[1].partial_cmp(&bdr[b].mid[1]).unwrap());
        // bottom/top: sort by x (increasing)
        bottom.sort_by(|&a, &b| bdr[a].mid[0].partial_cmp(&bdr[b].mid[0]).unwrap());
        top.sort_by(|&a, &b| bdr[a].mid[0].partial_cmp(&bdr[b].mid[0]).unwrap());

        // ── Step 4: build periodic pairs ────────────────────────────────────
        // Each pair: (master_edge_list, slave_edge_list, translation)
        let mut pairs_found: Vec<(Vec<Vec<u32>>, Vec<Vec<u32>>, [f64; 2])> = Vec::new();

        // Left ↔ Right
        if !left.is_empty() && !right.is_empty() && left.len() == right.len() {
            let master: Vec<Vec<NodeId>> = left.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let slave: Vec<Vec<NodeId>> = right.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let dx = bdr[right[0]].mid[0] - bdr[left[0]].mid[0];
            let dy = bdr[right[0]].mid[1] - bdr[left[0]].mid[1];
            // For D=2, construct translation as [f64; D]
            let translation = [dx, dy];
            pairs_found.push((master, slave, translation));
        }

        // Bottom ↔ Top
        if !bottom.is_empty() && !top.is_empty() && bottom.len() == top.len() {
            let master: Vec<Vec<NodeId>> = bottom.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let slave: Vec<Vec<NodeId>> = top.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let dx = bdr[top[0]].mid[0] - bdr[bottom[0]].mid[0];
            let dy = bdr[top[0]].mid[1] - bdr[bottom[0]].mid[1];
            let translation = [dx, dy];
            pairs_found.push((master, slave, translation));
        }

        if pairs_found.is_empty() {
            return Err(FemError::Mesh(
                "detect_periodic_boundary: could not pair any boundary edges".into(),
            ));
        }

        // ── Step 5: create boundary faces + call make_periodic ──────────────
        // Clone self and add boundary faces for the detected periodic edges.
        let mut mesh_with_faces = self.clone();
        // Use element tags from the adjacent element as boundary tags
        let mut new_face_conn: Vec<NodeId> = Vec::new();
        let mut new_face_tags: Vec<i32> = Vec::new();
        let mut tag = 1i32;
        let mut make_pairs: Vec<(i32, i32, [f64; D])> = Vec::new();

        for (master_edges, slave_edges, translation) in &pairs_found {
            let tag_a = tag;
            tag += 1;
            let tag_b = tag;
            tag += 1;

            // Master edges → tag_a
            for edge_nodes in master_edges {
                for &n in edge_nodes {
                    new_face_conn.push(n);
                }
                new_face_tags.push(tag_a);
            }
            // Slave edges → tag_b
            for edge_nodes in slave_edges {
                for &n in edge_nodes {
                    new_face_conn.push(n);
                }
                new_face_tags.push(tag_b);
            }

            let mut t = [0.0; D];
            for d in 0..D.min(2) { t[d] = translation[d]; }
            make_pairs.push((tag_a, tag_b, t));
        }

        // Set the boundary faces on the cloned mesh
        mesh_with_faces.face_conn = new_face_conn;
        mesh_with_faces.face_tags = new_face_tags;
        mesh_with_faces.face_type = ElementType::Line2;

        // Now call make_periodic to merge the nodes
        mesh_with_faces.make_periodic(&make_pairs, tol)
    }

    /// Create a periodic mesh by identifying matching node pairs on opposite
    /// boundary faces.
    ///
    /// For each `(tag_a, tag_b)` pair, nodes on boundary `tag_a` are matched
    /// to nodes on boundary `tag_b` using the `translation` vector: a node at
    /// position `x` on side A matches a node at position `x + translation` on
    /// side B (within tolerance `tol`).
    ///
    /// The returned mesh has all "B-side" nodes remapped to their A-side
    /// partners, effectively merging them.  The periodic boundary faces are
    /// removed from the face lists.
    ///
    /// # Arguments
    /// * `pairs` — slice of `(tag_a, tag_b, translation)` triples.
    /// * `tol`   — geometric matching tolerance.
    pub fn make_periodic(
        &self,
        pairs: &[(BoundaryTag, BoundaryTag, [f64; D])],
        tol: f64,
    ) -> FemResult<Self> {
        // 1. Collect boundary nodes per tag
        let mut tag_nodes = std::collections::HashMap::<BoundaryTag, Vec<NodeId>>::new();
        let n_faces = self.n_faces();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            let ns = self.bface_nodes(f);
            for &n in ns {
                tag_nodes.entry(tag).or_default().push(n);
            }
        }
        // Dedup node lists
        for list in tag_nodes.values_mut() {
            list.sort_unstable();
            list.dedup();
        }

        // 2. Build node remap: b_node → a_node
        let mut remap = vec![u32::MAX; self.n_nodes()];
        for (i, r) in remap.iter_mut().enumerate() {
            *r = i as u32;
        }

        let mut periodic_tags = std::collections::HashSet::new();

        for &(tag_a, tag_b, ref translation) in pairs {
            periodic_tags.insert(tag_a);
            periodic_tags.insert(tag_b);

            let nodes_a = tag_nodes.get(&tag_a).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_a={tag_a} not found on boundary"))
            })?;
            let nodes_b = tag_nodes.get(&tag_b).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_b={tag_b} not found on boundary"))
            })?;

            // For each node on B, find matching node on A
            for &nb in nodes_b {
                let cb = self.coords_of(nb);
                let mut matched = false;
                for &na in nodes_a {
                    let ca = self.coords_of(na);
                    let mut dist2 = 0.0;
                    for d in 0..D {
                        let diff = cb[d] - (ca[d] + translation[d]);
                        dist2 += diff * diff;
                    }
                    if dist2.sqrt() < tol {
                        remap[nb as usize] = na;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return Err(FemError::Mesh(format!(
                        "periodic: no match for node {nb} on tag_b={tag_b}"
                    )));
                }
            }
        }

        // 3. Build new compact node numbering (skip merged-away nodes)
        let mut new_id = vec![u32::MAX; self.n_nodes()];
        let mut new_coords = Vec::new();
        let mut next = 0u32;
        for i in 0..self.n_nodes() {
            if remap[i] == i as u32 {
                // This node is kept (not remapped to another)
                new_id[i] = next;
                let off = i * D;
                new_coords.extend_from_slice(&self.coords[off..off + D]);
                next += 1;
            }
        }
        // Map remapped nodes to their target's new ID
        for i in 0..self.n_nodes() {
            if remap[i] != i as u32 {
                let target = remap[i] as usize;
                new_id[i] = new_id[target];
            }
        }

        // 4. Remap element connectivity
        let new_conn: Vec<NodeId> = self.conn.iter().map(|&n| new_id[n as usize]).collect();

        // 5. Filter boundary faces (remove periodic ones)
        let mut new_face_conn = Vec::new();
        let mut new_face_tags = Vec::new();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            if periodic_tags.contains(&tag) {
                continue; // skip periodic boundary faces
            }
            let ns = self.bface_nodes(f);
            for &n in ns {
                new_face_conn.push(new_id[n as usize]);
            }
            new_face_tags.push(tag);
        }

        Ok(Mesh::uniform(
            new_coords,
            new_conn,
            self.elem_tags.clone(),
            self.elem_type,
            new_face_conn,
            new_face_tags,
            self.face_type,
        ))
    }

    /// Make a periodic mesh with affine (rotation + translation) matching.
    ///
    /// Each pair `(tag_a, tag_b, rot, trans)` identifies boundary faces that should
    /// be identified: a node on `tag_b` at position `x_b` is matched to a node on
    /// `tag_a` at position `x_a` if `x_b ≈ rot · x_a + trans`.
    ///
    /// `rot` is a flat `Vec<f64>` of length `D*D` (row-major: `result[i] = Σ rot[i*D + j] * x[j]`).
    pub fn make_periodic_affine(
        &self,
        pairs: &[(BoundaryTag, BoundaryTag, Vec<f64>, [f64; D])],
        tol: f64,
    ) -> FemResult<Self> {
        let mut tag_nodes = std::collections::HashMap::<BoundaryTag, Vec<NodeId>>::new();
        let n_faces = self.n_faces();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            let ns = self.bface_nodes(f);
            for &n in ns { tag_nodes.entry(tag).or_default().push(n); }
        }
        for list in tag_nodes.values_mut() { list.sort_unstable(); list.dedup(); }

        let mut remap = vec![u32::MAX; self.n_nodes()];
        for (i, r) in remap.iter_mut().enumerate() { *r = i as u32; }
        let mut periodic_tags = std::collections::HashSet::new();

        for &(tag_a, tag_b, ref rot, ref trans) in pairs {
            periodic_tags.insert(tag_a);
            periodic_tags.insert(tag_b);
            let nodes_a = tag_nodes.get(&tag_a).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_a={tag_a} not found"))
            })?;
            let nodes_b = tag_nodes.get(&tag_b).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_b={tag_b} not found"))
            })?;
            for &nb in nodes_b {
                let cb = self.coords_of(nb);
                let mut matched = false;
                for &na in nodes_a {
                    let ca = self.coords_of(na);
                    let mut xform = [0.0; D];
                    for i in 0..D { for j in 0..D { xform[i] += rot[i * D + j] * ca[j]; } }
                    for i in 0..D { xform[i] += trans[i]; }
                    let mut dist2 = 0.0;
                    for d in 0..D { let d2 = cb[d] - xform[d]; dist2 += d2 * d2; }
                    if dist2.sqrt() < tol {
                        remap[nb as usize] = na;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return Err(FemError::Mesh(format!(
                        "periodic: no match for node {nb} on tag_b={tag_b}"
                    )));
                }
            }
        }
        let mut new_id = vec![u32::MAX; self.n_nodes()];
        let mut new_coords = Vec::new();
        let mut next = 0u32;
        for i in 0..self.n_nodes() {
            if remap[i] == i as u32 {
                new_id[i] = next;
                let off = i * D;
                new_coords.extend_from_slice(&self.coords[off..off + D]);
                next += 1;
            }
        }
        for i in 0..self.n_nodes() {
            if remap[i] != i as u32 {
                new_id[i] = new_id[remap[i] as usize];
            }
        }
        let new_conn: Vec<NodeId> = self.conn.iter().map(|&n| new_id[n as usize]).collect();
        let mut new_face_conn = Vec::new();
        let mut new_face_tags = Vec::new();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            if periodic_tags.contains(&tag) { continue; }
            let ns = self.bface_nodes(f);
            for &n in ns { new_face_conn.push(new_id[n as usize]); }
            new_face_tags.push(tag);
        }
        Ok(Mesh::<D>::uniform(
            new_coords,
            new_conn,
            self.elem_tags.clone(),
            self.elem_type,
            new_face_conn,
            new_face_tags,
            self.face_type,
        ))
    }

    /// Validate internal consistency.
    pub fn check(&self) -> FemResult<()> {
        let nn = self.n_nodes();
        for (i, &nid) in self.conn.iter().enumerate() {
            if nid as usize >= nn {
                return Err(FemError::Mesh(format!(
                    "element connectivity[{i}] = {nid} exceeds n_nodes = {nn}"
                )));
            }
        }
        for (i, &nid) in self.face_conn.iter().enumerate() {
            if nid as usize >= nn {
                return Err(FemError::Mesh(format!(
                    "face connectivity[{i}] = {nid} exceeds n_nodes = {nn}"
                )));
            }
        }
        Ok(())
    }

    /// Create a uniform (non-mixed) mesh.  Convenience constructor that sets
    /// all mixed-element fields to `None`.
    pub fn uniform(
        coords: Vec<f64>,
        conn: Vec<NodeId>,
        elem_tags: Vec<i32>,
        elem_type: ElementType,
        face_conn: Vec<NodeId>,
        face_tags: Vec<BoundaryTag>,
        face_type: ElementType,
    ) -> Self {
        Mesh {
            coords, conn, elem_tags, elem_type, face_conn, face_tags, face_type,
            elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
            geometry: None,
        }
    }

    // -----------------------------------------------------------------------
    // Face-to-element mapping
    // -----------------------------------------------------------------------

    /// Build the mapping from boundary face → owning element.
    ///
    /// This iterates over all elements, extracts each element's faces, and
    /// records which element owns each boundary face.
    pub fn build_face_to_elem(&mut self) {
        let n_boundary = self.n_faces();
        let mut bface_to_elem = vec![ElemId::MAX; n_boundary];

        // Build element-face-to-boundary-face mapping.
        // For each element, for each of its faces, check if that face
        // corresponds to a boundary face.
        let n_elem = self.n_elems();
        for e in 0..n_elem {
            let verts = self.element_nodes(e as ElemId);
            let dim = D;
            let local_faces = local_face_verts(dim, self.element_type(e as ElemId));
            for fv in &local_faces {
                // Face vertex set as a sorted slice of node indices.
                let mut face_set: Vec<u32> = fv.iter().map(|&i| verts[i]).collect();
                face_set.sort_unstable();

                // Find the corresponding boundary face.
                for bf in 0..n_boundary {
                    let bfv = self.bface_nodes(bf as FaceId);
                    if self.matches_face(bfv, &face_set) {
                        bface_to_elem[bf] = e as ElemId;
                        break;
                    }
                }
            }
        }

        self.face_to_elem = Some(bface_to_elem);
    }

    // ─── Edge connectivity ───────────────────────────────────────────────────

    /// Build the unique edge list and edge→element mapping.
    ///
    /// After calling this, `n_edges() > 0` and `edge_nodes()` / `edge_elements()`
    /// are available.  Idempotent: calling twice is a no-op.
    pub fn build_edge_connectivity(&mut self) {
        if !self.edge_to_elem.is_empty() { return; }
        let dim = D;
        let n_elem = self.n_elems();
        let mut edge_map: std::collections::HashMap<[NodeId; 2], (EdgeId, ElemId, ElemId)> =
            std::collections::HashMap::new();
        let mut next_eid = 0u32;

        for e in 0..n_elem {
            let verts = self.element_nodes(e as ElemId);
            let local_edges = local_element_edges(dim, self.element_type(e as ElemId));
            for &[la, lb] in &local_edges {
                let a = verts[la]; let b = verts[lb];
                let key = if a < b { [a, b] } else { [b, a] };
                let entry = edge_map.entry(key).or_insert((next_eid, e as ElemId, _MAX_EDGE));
                if entry.1 != e as ElemId && entry.2 == _MAX_EDGE {
                    entry.2 = e as ElemId;
                } else if entry.1 == _MAX_EDGE {
                    entry.1 = e as ElemId;
                }
                if entry.0 == next_eid { next_eid += 1; }
            }
        }

        let n = edge_map.len();
        let mut conn = Vec::with_capacity(n * 2);
        let mut e2e = vec![_MAX_EDGE; n * 2];
        for (&key, &(eid, e1, e2)) in &edge_map {
            let i = eid as usize;
            conn.push(key[0]); conn.push(key[1]);
            e2e[2 * i] = e1; e2e[2 * i + 1] = e2;
        }
        self.edge_conn = conn;
        self.edge_to_elem = e2e;
    }

    /// Check if a boundary face's vertex set matches a sorted face set.
    fn matches_face(&self, bf_verts: &[NodeId], sorted_set: &[u32]) -> bool {
        if bf_verts.len() != sorted_set.len() {
            return false;
        }
        let mut bf_sorted: Vec<u32> = bf_verts.to_vec();
        bf_sorted.sort_unstable();
        bf_sorted == sorted_set
    }

    // -----------------------------------------------------------------------
    // Mesh generators
    // -----------------------------------------------------------------------

    /// Generate a uniform triangular mesh on the unit square `[0,1]²`.
    ///
    /// The square is divided into `n × n` sub-squares, each split into 2
    /// triangles by the diagonal from bottom-left to top-right.
    ///
    /// Boundary tag convention:
    /// - 1: bottom edge (y = 0)
    /// - 2: right edge  (x = 1)
    /// - 3: top edge    (y = 1)
    /// - 4: left edge   (x = 0)
    pub fn unit_square_tri(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "unit_square_tri requires D = 2");
        let np = n + 1;               // nodes per side
        let mut coords = Vec::with_capacity(np * np * 2);
        for j in 0..np {
            for i in 0..np {
                coords.push(i as f64 / n as f64); // x
                coords.push(j as f64 / n as f64); // y
            }
        }

        // Node index helper
        let nid = |i: usize, j: usize| -> NodeId { (j * np + i) as NodeId };

        let mut conn      = Vec::with_capacity(2 * n * n * 3);
        let mut elem_tags = Vec::with_capacity(2 * n * n);
        for j in 0..n {
            for i in 0..n {
                let n0 = nid(i,   j  );
                let n1 = nid(i+1, j  );
                let n2 = nid(i+1, j+1);
                let n3 = nid(i,   j+1);
                // lower-left triangle
                conn.extend_from_slice(&[n0, n1, n3]);
                elem_tags.push(1);
                // upper-right triangle
                conn.extend_from_slice(&[n1, n2, n3]);
                elem_tags.push(1);
            }
        }

        // Boundary faces (edges)
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let add_edge = |fc: &mut Vec<NodeId>, ft: &mut Vec<i32>,
                        a: NodeId, b: NodeId, tag: i32| {
            fc.push(a); fc.push(b); ft.push(tag);
        };
        for i in 0..n {
            // bottom (j=0, tag=1)
            add_edge(&mut face_conn, &mut face_tags, nid(i,0), nid(i+1,0), 1);
            // right (i=n, tag=2)
            add_edge(&mut face_conn, &mut face_tags, nid(n,i), nid(n,i+1), 2);
            // top (j=n, tag=3) — reversed for outward normal
            add_edge(&mut face_conn, &mut face_tags, nid(i+1,n), nid(i,n), 3);
            // left (i=0, tag=4)
            add_edge(&mut face_conn, &mut face_tags, nid(0,i+1), nid(0,i), 4);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tri3,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Generate a uniform quadrilateral mesh on the unit square `[0,1]²`.
    ///
    /// The square is divided into `n × n` quadrilateral elements.
    /// Boundary tag convention matches `unit_square_tri`:
    /// - 1: bottom, 2: right, 3: top, 4: left
    pub fn unit_square_quad(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "unit_square_quad requires D = 2");
        let np = n + 1;
        let mut coords = Vec::with_capacity(np * np * 2);
        for j in 0..np {
            for i in 0..np {
                coords.push(i as f64 / n as f64);
                coords.push(j as f64 / n as f64);
            }
        }

        let nid = |i: usize, j: usize| -> NodeId { (j * np + i) as NodeId };

        let mut conn      = Vec::with_capacity(n * n * 4);
        let mut elem_tags = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                // Counter-clockwise: bottom-left, bottom-right, top-right, top-left
                conn.extend_from_slice(&[nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)]);
                elem_tags.push(1);
            }
        }

        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let add_edge = |fc: &mut Vec<NodeId>, ft: &mut Vec<i32>,
                        a: NodeId, b: NodeId, tag: i32| {
            fc.push(a); fc.push(b); ft.push(tag);
        };
        for i in 0..n {
            add_edge(&mut face_conn, &mut face_tags, nid(i,0), nid(i+1,0), 1);
            add_edge(&mut face_conn, &mut face_tags, nid(n,i), nid(n,i+1), 2);
            add_edge(&mut face_conn, &mut face_tags, nid(i+1,n), nid(i,n), 3);
            add_edge(&mut face_conn, &mut face_tags, nid(0,i+1), nid(0,i), 4);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Generate a coaxial cable cross-section mesh (annular region).
    ///
    /// Outer square boundary `[-a, a]²`, inner circular conductor radius `r`.
    /// This is a helper that returns a `Mesh` suitable for the
    /// electrostatics example; requires GMSH for a proper curved mesh.
    /// Here we use a polygonal approximation of the inner conductor.
    pub fn coaxial_annulus_poly(outer_half: f64, inner_r: f64, n_poly: usize, n_radial: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "coaxial_annulus_poly requires D = 2");
        // Build a simple mesh: inner polygon + outer square, triangulated.
        // This is approximate; for production use GMSH.
        use std::f64::consts::PI;

        let mut coords: Vec<f64> = Vec::new();
        let mut conn:   Vec<NodeId> = Vec::new();
        let mut elem_tags: Vec<i32> = Vec::new();

        // Inner polygon nodes
        let inner_start = 0usize;
        for k in 0..n_poly {
            let theta = 2.0 * PI * k as f64 / n_poly as f64;
            coords.push(inner_r * theta.cos());
            coords.push(inner_r * theta.sin());
        }
        // Outer square corners (4 nodes)
        let outer_start = n_poly;
        let corners = [
            [-outer_half, -outer_half],
            [ outer_half, -outer_half],
            [ outer_half,  outer_half],
            [-outer_half,  outer_half],
        ];
        for c in &corners {
            coords.push(c[0]);
            coords.push(c[1]);
        }

        // Triangulate by connecting inner polygon to outer corners naively.
        // For a proper mesh, users should load a GMSH-generated file.
        // Here we just create a minimal ring of triangles from inner to outer.
        let np_inner = n_poly as NodeId;
        let np_outer = 4 as NodeId;
        let _ = (np_inner, np_outer, n_radial); // suppress unused warnings

        // Fan triangles around each inner edge connecting to nearest outer corner
        for k in 0..n_poly {
            let a = (inner_start + k) as NodeId;
            let b = (inner_start + (k + 1) % n_poly) as NodeId;
            // Find nearest outer corner
            let ax = coords[a as usize * 2];
            let ay = coords[a as usize * 2 + 1];
            let mut best_c = outer_start as NodeId;
            let mut best_d = f64::MAX;
            for ci in 0..4usize {
                let cx = corners[ci][0];
                let cy = corners[ci][1];
                let d = (cx - ax).hypot(cy - ay);
                if d < best_d { best_d = d; best_c = (outer_start + ci) as NodeId; }
            }
            conn.extend_from_slice(&[a, b, best_c]);
            elem_tags.push(1);
        }

        let mut face_conn = Vec::new();
        let mut face_tags_v = Vec::new();
        // Inner boundary: tag=1 (conductor surface)
        for k in 0..n_poly {
            let a = (inner_start + k) as NodeId;
            let b = (inner_start + (k + 1) % n_poly) as NodeId;
            face_conn.push(a); face_conn.push(b);
            face_tags_v.push(1i32);
        }
        // Outer boundary: tag=2
        for k in 0..4usize {
            let a = (outer_start + k) as NodeId;
            let b = (outer_start + (k + 1) % 4) as NodeId;
            face_conn.push(a); face_conn.push(b);
            face_tags_v.push(2i32);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tri3,
            face_conn, face_tags_v, ElementType::Line2,
        )
    }

    /// Generate a uniform tetrahedral mesh on the unit cube `[0,1]³`.
    ///
    /// Divides the cube into `n×n×n` sub-cubes, each split into 6 tetrahedra
    /// using a regular decomposition (Freudenthal/Kuhn partition).
    ///
    /// Boundary tag convention (face normals pointing outward):
    /// - 1: z = 0 (bottom)
    /// - 2: z = 1 (top)
    /// - 3: y = 0 (front)
    /// - 4: y = 1 (back)
    /// - 5: x = 0 (left)
    /// - 6: x = 1 (right)
    pub fn unit_cube_tet(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_cube_tet requires D = 3");
        let np = n + 1;
        let mut coords = Vec::with_capacity(np * np * np * 3);
        for k in 0..np {
            for j in 0..np {
                for i in 0..np {
                    coords.push(i as f64 / n as f64);
                    coords.push(j as f64 / n as f64);
                    coords.push(k as f64 / n as f64);
                }
            }
        }

        let nid = |i: usize, j: usize, k: usize| -> NodeId {
            (k * np * np + j * np + i) as NodeId
        };

        // 6 tetrahedra per cube using the Freudenthal decomposition.
        // Each cube (i..i+1, j..j+1, k..k+1) → 6 tets.
        let mut conn      = Vec::new();
        let mut elem_tags = Vec::new();

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let v = [
                        nid(i,   j,   k  ), // 0: (0,0,0)
                        nid(i+1, j,   k  ), // 1: (1,0,0)
                        nid(i+1, j+1, k  ), // 2: (1,1,0)
                        nid(i,   j+1, k  ), // 3: (0,1,0)
                        nid(i,   j,   k+1), // 4: (0,0,1)
                        nid(i+1, j,   k+1), // 5: (1,0,1)
                        nid(i+1, j+1, k+1), // 6: (1,1,1)
                        nid(i,   j+1, k+1), // 7: (0,1,1)
                    ];
                    // Non-degenerate 6-tet cube split along diagonal v0 -> v6.
                    // This avoids coplanar 4-point sets.
                    let tets: [[usize; 4]; 6] = [
                        [0, 1, 2, 6],
                        [0, 2, 3, 6],
                        [0, 3, 7, 6],
                        [0, 7, 4, 6],
                        [0, 4, 5, 6],
                        [0, 5, 1, 6],
                    ];
                    for tet in &tets {
                        conn.extend_from_slice(&[v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]]);
                        elem_tags.push(1i32);
                    }
                }
            }
        }

        // Boundary faces (triangles on the 6 cube faces).
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();

        macro_rules! add_tri {
            ($a:expr, $b:expr, $c:expr, $tag:expr) => {
                face_conn.push($a); face_conn.push($b); face_conn.push($c);
                face_tags.push($tag);
            }
        }

        for j in 0..n {
            for i in 0..n {
                // z=0 (tag=1): outward normal -z → winding n3,n2,n1,n0
                let (a,b,c,d) = (nid(i,j,0), nid(i+1,j,0), nid(i+1,j+1,0), nid(i,j+1,0));
                add_tri!(a, c, b, 1); add_tri!(a, d, c, 1);
                // z=1 (tag=2): outward normal +z
                let (a,b,c,d) = (nid(i,j,n), nid(i+1,j,n), nid(i+1,j+1,n), nid(i,j+1,n));
                add_tri!(a, b, c, 2); add_tri!(a, c, d, 2);
                // y=0 (tag=3): outward normal -y
                let (a,b,c,d) = (nid(i,0,j), nid(i+1,0,j), nid(i+1,0,j+1), nid(i,0,j+1));
                add_tri!(a, b, c, 3); add_tri!(a, c, d, 3);
                // y=1 (tag=4): outward normal +y
                let (a,b,c,d) = (nid(i,n,j), nid(i+1,n,j), nid(i+1,n,j+1), nid(i,n,j+1));
                add_tri!(a, c, b, 4); add_tri!(a, d, c, 4);
                // x=0 (tag=5): outward normal -x
                let (a,b,c,d) = (nid(0,i,j), nid(0,i+1,j), nid(0,i+1,j+1), nid(0,i,j+1));
                add_tri!(a, c, b, 5); add_tri!(a, d, c, 5);
                // x=1 (tag=6): outward normal +x
                let (a,b,c,d) = (nid(n,i,j), nid(n,i+1,j), nid(n,i+1,j+1), nid(n,i,j+1));
                add_tri!(a, b, c, 6); add_tri!(a, c, d, 6);
            }
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tet4,
            face_conn, face_tags, ElementType::Tri3,
        )
    }

    /// Generate a uniform hexahedral mesh on the unit cube `[0,1]³`.
    ///
    /// Divided into `n × n × n` Hex8 elements.  Boundary face (Quad4) tag convention:
    /// - 1: z = 0 (bottom), 2: z = 1 (top), 3: y = 0 (front),
    /// - 4: y = 1 (back),   5: x = 0 (left), 6: x = 1 (right)
    pub fn unit_cube_hex(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_cube_hex requires D = 3");
        let np = n + 1;
        let mut coords = Vec::with_capacity(np * np * np * 3);
        for k in 0..np {
            for j in 0..np {
                for i in 0..np {
                    coords.push(i as f64 / n as f64);
                    coords.push(j as f64 / n as f64);
                    coords.push(k as f64 / n as f64);
                }
            }
        }

        let nid = |i: usize, j: usize, k: usize| -> NodeId {
            (k * np * np + j * np + i) as NodeId
        };

        let mut conn      = Vec::with_capacity(n * n * n * 8);
        let mut elem_tags = Vec::with_capacity(n * n * n);

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    // Bottom face (z=k): CCW from outside (below) → (0,1,2,3)
                    // Top face (z=k+1): CCW from outside (above) → (4,5,6,7)
                    // Standard Hex8 layout:
                    //   (n0, n1, n2, n3) = bottom face CCW
                    //   (n4, n5, n6, n7) = top face, n4 above n0
                    conn.extend_from_slice(&[
                        nid(i,   j,   k  ), // n0
                        nid(i+1, j,   k  ), // n1
                        nid(i+1, j+1, k  ), // n2
                        nid(i,   j+1, k  ), // n3
                        nid(i,   j,   k+1), // n4
                        nid(i+1, j,   k+1), // n5
                        nid(i+1, j+1, k+1), // n6
                        nid(i,   j+1, k+1), // n7
                    ]);
                    elem_tags.push(1i32);
                }
            }
        }

        // Boundary Quad4 faces on the 6 cube faces.
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();

        macro_rules! add_quad {
            ($a:expr, $b:expr, $c:expr, $d:expr, $tag:expr) => {
                face_conn.push($a); face_conn.push($b);
                face_conn.push($c); face_conn.push($d);
                face_tags.push($tag);
            }
        }

        for j in 0..n {
            for i in 0..n {
                // z = 0 bottom face (tag 1), outward normal = -z, CCW when viewed from below
                add_quad!(nid(i,j,0), nid(i,j+1,0), nid(i+1,j+1,0), nid(i+1,j,0), 1);
                // z = n top face (tag 2), outward normal = +z, CCW when viewed from above
                add_quad!(nid(i,j,n), nid(i+1,j,n), nid(i+1,j+1,n), nid(i,j+1,n), 2);
            }
        }
        for k in 0..n {
            for i in 0..n {
                // y = 0 front face (tag 3), outward normal = -y
                add_quad!(nid(i,0,k), nid(i+1,0,k), nid(i+1,0,k+1), nid(i,0,k+1), 3);
                // y = n back face (tag 4), outward normal = +y
                add_quad!(nid(i,n,k), nid(i,n,k+1), nid(i+1,n,k+1), nid(i+1,n,k), 4);
            }
        }
        for k in 0..n {
            for j in 0..n {
                // x = 0 left face (tag 5), outward normal = -x
                add_quad!(nid(0,j,k), nid(0,j,k+1), nid(0,j+1,k+1), nid(0,j+1,k), 5);
                // x = n right face (tag 6), outward normal = +x
                add_quad!(nid(n,j,k), nid(n,j+1,k), nid(n,j+1,k+1), nid(n,j,k+1), 6);
            }
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Hex8,
            face_conn, face_tags, ElementType::Quad4,
        )
    }

    /// Generate an octahedral mesh inscribed in the unit sphere.
    ///
    /// 6 vertices, 8 Tri3 elements, 2D surface in 3D space.
    /// Each face has its own boundary attribute (1..8).
    /// Matching MFEM's ex7 octahedron (inscribed in unit sphere).
    pub fn unit_sphere_octahedron() -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_sphere_octahedron requires D = 3");
        let coords = vec![
            1.0,  0.0,  0.0,  // 0
            0.0,  1.0,  0.0,  // 1
           -1.0,  0.0,  0.0,  // 2
            0.0, -1.0,  0.0,  // 3
            0.0,  0.0,  1.0,  // 4 (north pole)
            0.0,  0.0, -1.0,  // 5 (south pole)
        ];
        let conn: Vec<NodeId> = vec![
            0, 1, 4,  1, 2, 4,  2, 3, 4,  3, 0, 4,
            1, 0, 5,  2, 1, 5,  3, 2, 5,  0, 3, 5,
        ];
        let elem_tags: Vec<i32> = (1..=8).collect();
        let face_conn: Vec<NodeId> = conn.clone();
        let face_tags: Vec<i32> = elem_tags.clone();
        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tri3,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Generate a cube mesh inscribed in the unit sphere.
    ///
    /// 8 vertices, 6 Quad4 elements, 2D surface in 3D space.
    /// Each face has its own boundary attribute (1..6).
    /// Matching MFEM's ex7 cube (inscribed in unit sphere).
    pub fn unit_sphere_cube() -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_sphere_cube requires D = 3");
        let coords = vec![
            -1.0, -1.0, -1.0,  // 0
             1.0, -1.0, -1.0,  // 1
             1.0,  1.0, -1.0,  // 2
            -1.0,  1.0, -1.0,  // 3
            -1.0, -1.0,  1.0,  // 4
             1.0, -1.0,  1.0,  // 5
             1.0,  1.0,  1.0,  // 6
            -1.0,  1.0,  1.0,  // 7
        ];
        let conn: Vec<NodeId> = vec![
            3, 2, 1, 0,  0, 1, 5, 4,  1, 2, 6, 5,
            2, 3, 7, 6,  3, 0, 4, 7,  4, 5, 6, 7,
        ];
        let elem_tags: Vec<i32> = (1..=6).collect();
        let face_conn: Vec<NodeId> = conn.clone();
        let face_tags: Vec<i32> = elem_tags.clone();
        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Snap all mesh nodes to the unit sphere surface.
    ///
    /// Each node's position is normalized to length 1.
    /// (MFEM ex7 SnapNodes on the octahedron/cube mesh.)
    pub fn snap_to_sphere(&mut self)
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "snap_to_sphere requires D = 3");
        for i in 0..self.n_nodes() {
            let base = i * 3;
            let x = self.coords[base];
            let y = self.coords[base + 1];
            let z = self.coords[base + 2];
            let inv_len = 1.0 / (x * x + y * y + z * z).sqrt();
            self.coords[base]     *= inv_len;
            self.coords[base + 1] *= inv_len;
            self.coords[base + 2] *= inv_len;
        }
        // Also snap geometry node coordinates.
        if let Some(ref mut geo) = self.geometry {
            for i in 0..geo.n_nodes {
                let base = i * 3;
                if base + 3 <= geo.coords.len() {
                    let x = geo.coords[base];
                    let y = geo.coords[base + 1];
                    let z = geo.coords[base + 2];
                    let inv_len = 1.0 / (x * x + y * y + z * z).sqrt();
                    geo.coords[base]     *= inv_len;
                    geo.coords[base + 1] *= inv_len;
                    geo.coords[base + 2] *= inv_len;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MeshTopology implementation
// ---------------------------------------------------------------------------

impl<const D: usize> MeshTopology for Mesh<D> {
    fn dim(&self) -> u8 { D as u8 }

    fn topological_dim(&self) -> u8 {
        if self.n_elems() > 0 {
            self.element_type_at(0).dim()
        } else {
            D as u8
        }
    }

    fn n_nodes(&self) -> usize { self.n_nodes() }

    fn n_elements(&self) -> usize { self.n_elems() }

    fn n_boundary_faces(&self) -> usize { self.n_faces() }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] { self.elem_nodes(elem) }

    fn element_type(&self, elem: ElemId) -> ElementType {
        self.element_type_at(elem)
    }

    fn element_tag(&self, elem: ElemId) -> i32 { self.elem_tags[elem as usize] }

    fn geom_order(&self) -> u8 {
        self.geometry.as_ref().map_or(1, |g| g.order)
    }

    fn geometry_nodes(&self, elem: ElemId) -> &[NodeId] {
        if let Some(ref geo) = self.geometry {
            let e = elem as usize;
            let off = e * geo.nodes_per_elem;
            &geo.conn[off..off + geo.nodes_per_elem]
        } else {
            self.element_nodes(elem)
        }
    }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        let off = node as usize * D;
        &self.coords[off..off + D]
    }

    fn geom_coords_of(&self, node: NodeId) -> &[f64] {
        if let Some(ref geo) = self.geometry {
            let n = node as usize;
            if n < geo.n_nodes {
                let off = n * D;
                if off + D <= geo.coords.len() {
                    return &geo.coords[off..off + D];
                }
            }
        }
        self.node_coords(node)
    }

    fn face_nodes(&self, face: FaceId) -> &[NodeId] { self.bface_nodes(face) }

    fn face_tag(&self, face: FaceId) -> i32 { self.face_tags[face as usize] }

    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>) {
        if let Some(ref f2e) = self.face_to_elem {
            let e = f2e[face as usize];
            if e != ElemId::MAX {
                (e, None)
            } else {
                (0, None)
            }
        } else {
            (0, None)
        }
    }

    fn n_edges(&self) -> usize { self.edge_conn.len() / 2 }

    fn edge_nodes(&self, eid: EdgeId) -> &[NodeId] {
        let i = eid as usize * 2;
        &self.edge_conn[i..i + 2]
    }

    fn edge_elements(&self, eid: EdgeId) -> (ElemId, Option<ElemId>) {
        let i = eid as usize * 2;
        let e1 = self.edge_to_elem[i];
        let e2 = self.edge_to_elem[i + 1];
        if e2 == _MAX_EDGE {
            (e1, None)
        } else {
            (e1, Some(e2))
        }
    }

    fn edge_iter(&self) -> std::ops::Range<u32> { 0..self.n_edges() as u32 }
}

/// Compute the volume of a tetrahedral element in a 3-D mesh.
pub fn tet_volume(mesh: &Mesh<3>, elem: u32) -> f64 {
    let ns = mesh.element_nodes(elem);
    let c = |k: usize| -> [f64; 3] {
        let cc = mesh.node_coords(ns[k]);
        [cc[0], cc[1], cc[2]]
    };
    let p = [c(0), c(1), c(2), c(3)];
    // Volume = |det([p1-p0, p2-p0, p3-p0])| / 6
    let (x0, y0, z0) = (p[0][0], p[0][1], p[0][2]);
    let (x1, y1, z1) = (p[1][0], p[1][1], p[1][2]);
    let (x2, y2, z2) = (p[2][0], p[2][1], p[2][2]);
    let (x3, y3, z3) = (p[3][0], p[3][1], p[3][2]);
    let det = (x1 - x0) * ((y2 - y0) * (z3 - z0) - (z2 - z0) * (y3 - y0))
            - (y1 - y0) * ((x2 - x0) * (z3 - z0) - (z2 - z0) * (x3 - x0))
            + (z1 - z0) * ((x2 - x0) * (y3 - y0) - (y2 - y0) * (x3 - x0));
    det.abs() / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedAttributeSet;

    #[test]
    fn unit_square_counts() {
        let n = 4usize;
        let m = Mesh::<2>::unit_square_tri(n);
        assert_eq!(m.n_nodes(), (n + 1) * (n + 1));
        assert_eq!(m.n_elems(), 2 * n * n);
        assert_eq!(m.n_faces(), 4 * n);
        m.check().unwrap();
    }

    #[test]
    fn topology_trait_unit_square() {
        let m = Mesh::<2>::unit_square_tri(3);
        let mt: &dyn MeshTopology = &m;
        assert_eq!(mt.dim(), 2);
        assert_eq!(mt.n_elements(), 18);
        // first element has 3 nodes
        let ns = mt.element_nodes(0);
        assert_eq!(ns.len(), 3);
    }

    #[test]
    fn coords_bottom_left() {
        let m = Mesh::<2>::unit_square_tri(4);
        let c = m.coords_of(0);
        assert!((c[0]).abs() < 1e-14);
        assert!((c[1]).abs() < 1e-14);
    }

    #[test]
    fn face_tags_present() {
        let m = Mesh::<2>::unit_square_tri(4);
        let tags: std::collections::HashSet<i32> = m.face_tags.iter().copied().collect();
        assert!(tags.contains(&1));
        assert!(tags.contains(&3));
    }

    #[test]
    fn bounding_box_unit_square() {
        let m = Mesh::<2>::unit_square_tri(4);
        let (lo, hi) = m.bounding_box();
        assert!((lo[0]).abs() < 1e-14);
        assert!((lo[1]).abs() < 1e-14);
        assert!((hi[0] - 1.0).abs() < 1e-14);
        assert!((hi[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bounding_box_unit_cube() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let (lo, hi) = m.bounding_box();
        for d in 0..3 {
            assert!(lo[d].abs() < 1e-14, "lo[{d}] = {}", lo[d]);
            assert!((hi[d] - 1.0).abs() < 1e-14, "hi[{d}] = {}", hi[d]);
        }
    }

    #[test]
    fn unique_boundary_tags_unit_square() {
        let m = Mesh::<2>::unit_square_tri(4);
        let tags = m.unique_boundary_tags();
        assert_eq!(tags, vec![1, 2, 3, 4]);
    }

    #[test]
    fn unique_boundary_tags_unit_cube() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let tags = m.unique_boundary_tags();
        assert_eq!(tags, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn unit_cube_tet_elements_non_degenerate() {
        let m = Mesh::<3>::unit_cube_tet(1);
        for e in 0..m.n_elems() as ElemId {
            let ns = m.elem_nodes(e);
            assert_eq!(ns.len(), 4);

            let x0 = m.coords_of(ns[0]);
            let x1 = m.coords_of(ns[1]);
            let x2 = m.coords_of(ns[2]);
            let x3 = m.coords_of(ns[3]);

            let j11 = x1[0] - x0[0]; let j12 = x2[0] - x0[0]; let j13 = x3[0] - x0[0];
            let j21 = x1[1] - x0[1]; let j22 = x2[1] - x0[1]; let j23 = x3[1] - x0[1];
            let j31 = x1[2] - x0[2]; let j32 = x2[2] - x0[2]; let j33 = x3[2] - x0[2];

            let det = j11 * (j22 * j33 - j23 * j32)
                - j12 * (j21 * j33 - j23 * j31)
                + j13 * (j21 * j32 - j22 * j31);
            assert!(det.abs() > 1e-12, "degenerate Tet4 at elem {e}, det={det}");
        }
    }

    #[test]
    fn make_periodic_x_direction() {
        // Unit square with tags: 1=bottom, 2=right, 3=top, 4=left.
        // Make periodic in x: pair left (tag=4) with right (tag=2),
        // translation = [1, 0].
        let m = Mesh::<2>::unit_square_tri(4);
        let n_before = m.n_nodes();
        let pm = m.make_periodic(&[(4, 2, [1.0, 0.0])], 1e-10).unwrap();

        // Should have fewer nodes: left boundary nodes merged with right
        // n+1 nodes per side, n-1 interior per side → merge n+1 nodes
        assert!(pm.n_nodes() < n_before,
            "periodic mesh should have fewer nodes: {} vs {}", pm.n_nodes(), n_before);

        // Same number of elements
        assert_eq!(pm.n_elems(), m.n_elems());

        // Periodic boundaries removed: only top and bottom remain
        let tags = pm.unique_boundary_tags();
        assert!(!tags.contains(&2), "right boundary should be removed");
        assert!(!tags.contains(&4), "left boundary should be removed");
        assert!(tags.contains(&1), "bottom should remain");
        assert!(tags.contains(&3), "top should remain");
    }

    #[test]
    fn make_periodic_both_directions() {
        // Make fully periodic (x and y)
        let m = Mesh::<2>::unit_square_tri(3);
        let pm = m.make_periodic(
            &[
                (4, 2, [1.0, 0.0]),  // left → right
                (1, 3, [0.0, 1.0]),  // bottom → top
            ],
            1e-10,
        ).unwrap();

        // No boundary faces should remain
        assert_eq!(pm.n_faces(), 0, "fully periodic mesh should have no boundary faces");
        assert_eq!(pm.n_elems(), m.n_elems());
    }

    #[test]
    fn named_attribute_set_queries_elements_and_faces() {
        let mut m = Mesh::<2>::unit_square_tri(2);
        let n = m.n_elems();
        for i in 0..n {
            m.elem_tags[i] = if i < n / 2 { 7 } else { 9 };
        }

        let mut reg = NamedAttributeRegistry::new();
        reg.insert(
            NamedAttributeSet::new("conductors")
                .with_element_tags([7])
                .with_boundary_tags([1, 3]),
        );

        let elems = m
            .element_ids_for_named_set(&reg, "conductors")
            .expect("missing named set");
        assert!(!elems.is_empty());
        assert!(elems.iter().all(|&e| m.elem_tags[e as usize] == 7));

        let faces = m
            .face_ids_for_named_set(&reg, "conductors")
            .expect("missing named set");
        assert!(!faces.is_empty());
        assert!(faces.iter().all(|&f| {
            let t = m.face_tags[f as usize];
            t == 1 || t == 3
        }));
    }

    #[test]
    fn named_attribute_set_missing_name_errors() {
        let m = Mesh::<2>::unit_square_tri(2);
        let reg = NamedAttributeRegistry::new();
        let err = m
            .element_ids_for_named_set(&reg, "missing")
            .expect_err("expected missing set error");
        let msg = format!("{err}");
        assert!(msg.contains("named attribute set not found"));
    }
}

#[cfg(all(test, feature = "serialize"))]
mod serde_tests {
    use super::*;

    #[test]
    fn simplex_mesh_roundtrip() {
        let m = Mesh::<2>::unit_square_tri(4);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Mesh<2> = serde_json::from_str(&json).unwrap();
        assert_eq!(m.n_nodes(), m2.n_nodes());
        assert_eq!(m.n_elems(), m2.n_elems());
        assert_eq!(m.n_faces(), m2.n_faces());
        assert_eq!(m.coords, m2.coords);
        assert_eq!(m.conn, m2.conn);
        assert_eq!(m.elem_tags, m2.elem_tags);
        assert_eq!(m.elem_type, m2.elem_type);
    }

    #[test]
    fn simplex_mesh_3d_roundtrip() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Mesh<3> = serde_json::from_str(&json).unwrap();
        assert_eq!(m.n_nodes(), m2.n_nodes());
        assert_eq!(m.n_elems(), m2.n_elems());
        assert_eq!(m.coords, m2.coords);
        assert_eq!(m.conn, m2.conn);
    }

    #[test]
    fn simplex_mesh_hex_roundtrip() {
        let m = Mesh::<3>::unit_cube_hex(2);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Mesh<3> = serde_json::from_str(&json).unwrap();
        assert_eq!(m.n_nodes(), m2.n_nodes());
        assert_eq!(m.n_elems(), m2.n_elems());
        assert_eq!(m.coords, m2.coords);
        assert_eq!(m.elem_type, m2.elem_type);
    }
}
