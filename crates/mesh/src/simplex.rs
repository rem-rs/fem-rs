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
        (3, ElementType::Hex8 | ElementType::Hex20) => vec![
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
        (3, ElementType::Hex8 | ElementType::Hex20) => vec![
            [0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7],
        ],
        _ => vec![],
    }
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
pub struct SimplexMesh<const D: usize> {
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
}

impl<const D: usize> SimplexMesh<D> {
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

    // ─── Geometric transforms ────────────────────────────────────────────────

    /// Apply a coordinate transform `f` to every mesh node.
    /// The closure receives `[x, y]` (2-D) or `[x, y, z]` (3-D) and returns
    /// the transformed coordinate array.
    pub fn transform(&mut self, mut f: impl FnMut([f64; D]) -> [f64; D]) {
        for n in 0..self.n_nodes() {
            let out = f(self.coords_of(n as NodeId));
            let off = n * D;
            self.coords[off..off + D].copy_from_slice(&out);
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

        Ok(SimplexMesh::uniform(
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
        Ok(SimplexMesh::<D>::uniform(
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
        SimplexMesh {
            coords, conn, elem_tags, elem_type, face_conn, face_tags, face_type,
            elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
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
        let mut bf_sorted: Vec<u32> = bf_verts.iter().copied().collect();
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

        SimplexMesh::uniform(
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

        SimplexMesh::uniform(
            coords, conn, elem_tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Generate a coaxial cable cross-section mesh (annular region).
    ///
    /// Outer square boundary `[-a, a]²`, inner circular conductor radius `r`.
    /// This is a helper that returns a `SimplexMesh` suitable for the
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

        SimplexMesh::uniform(
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

        SimplexMesh::uniform(
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

        SimplexMesh::uniform(
            coords, conn, elem_tags, ElementType::Hex8,
            face_conn, face_tags, ElementType::Quad4,
        )
    }
}

// ---------------------------------------------------------------------------
// MeshTopology implementation
// ---------------------------------------------------------------------------

impl<const D: usize> MeshTopology for SimplexMesh<D> {
    fn dim(&self) -> u8 { D as u8 }

    fn n_nodes(&self) -> usize { self.n_nodes() }

    fn n_elements(&self) -> usize { self.n_elems() }

    fn n_boundary_faces(&self) -> usize { self.n_faces() }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] { self.elem_nodes(elem) }

    fn element_type(&self, elem: ElemId) -> ElementType {
        self.element_type_at(elem)
    }

    fn element_tag(&self, elem: ElemId) -> i32 { self.elem_tags[elem as usize] }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        let off = node as usize * D;
        &self.coords[off..off + D]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedAttributeSet;

    #[test]
    fn unit_square_counts() {
        let n = 4usize;
        let m = SimplexMesh::<2>::unit_square_tri(n);
        assert_eq!(m.n_nodes(), (n + 1) * (n + 1));
        assert_eq!(m.n_elems(), 2 * n * n);
        assert_eq!(m.n_faces(), 4 * n);
        m.check().unwrap();
    }

    #[test]
    fn topology_trait_unit_square() {
        let m = SimplexMesh::<2>::unit_square_tri(3);
        let mt: &dyn MeshTopology = &m;
        assert_eq!(mt.dim(), 2);
        assert_eq!(mt.n_elements(), 18);
        // first element has 3 nodes
        let ns = mt.element_nodes(0);
        assert_eq!(ns.len(), 3);
    }

    #[test]
    fn coords_bottom_left() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let c = m.coords_of(0);
        assert!((c[0]).abs() < 1e-14);
        assert!((c[1]).abs() < 1e-14);
    }

    #[test]
    fn face_tags_present() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let tags: std::collections::HashSet<i32> = m.face_tags.iter().copied().collect();
        assert!(tags.contains(&1));
        assert!(tags.contains(&3));
    }

    #[test]
    fn bounding_box_unit_square() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let (lo, hi) = m.bounding_box();
        assert!((lo[0]).abs() < 1e-14);
        assert!((lo[1]).abs() < 1e-14);
        assert!((hi[0] - 1.0).abs() < 1e-14);
        assert!((hi[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bounding_box_unit_cube() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let (lo, hi) = m.bounding_box();
        for d in 0..3 {
            assert!(lo[d].abs() < 1e-14, "lo[{d}] = {}", lo[d]);
            assert!((hi[d] - 1.0).abs() < 1e-14, "hi[{d}] = {}", hi[d]);
        }
    }

    #[test]
    fn unique_boundary_tags_unit_square() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let tags = m.unique_boundary_tags();
        assert_eq!(tags, vec![1, 2, 3, 4]);
    }

    #[test]
    fn unique_boundary_tags_unit_cube() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let tags = m.unique_boundary_tags();
        assert_eq!(tags, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn unit_cube_tet_elements_non_degenerate() {
        let m = SimplexMesh::<3>::unit_cube_tet(1);
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
        let m = SimplexMesh::<2>::unit_square_tri(4);
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
        let m = SimplexMesh::<2>::unit_square_tri(3);
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
        let mut m = SimplexMesh::<2>::unit_square_tri(2);
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
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let reg = NamedAttributeRegistry::new();
        let err = m
            .element_ids_for_named_set(&reg, "missing")
            .expect_err("expected missing set error");
        let msg = format!("{err}");
        assert!(msg.contains("named attribute set not found"));
    }
}
