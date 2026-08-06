//! Submesh extraction and parent/child nodal transfer utilities.
//!
//! Current scope:
//! - 2-D `Tri3` meshes
//! - 3-D meshes with mixed element types (Tet4, Hex8, Prism6)
//! - extraction by element tags
//! - nodal-value transfer between parent and submesh

use std::collections::{HashMap, HashSet};

use fem_core::{ElemId, NodeId, FaceId};

use crate::{ElementType, NamedAttributeRegistry, Mesh};

/// Submesh view extracted from a parent mesh.
#[derive(Debug, Clone)]
pub struct SubMesh {
    /// Extracted mesh.
    pub mesh: Mesh<2>,
    /// Parent element ids corresponding to submesh elements.
    pub parent_elem_ids: Vec<ElemId>,
    /// parent_node_of_sub[sub_node_id] = parent_node_id.
    pub parent_node_of_sub: Vec<NodeId>,
}

impl SubMesh {
    /// Transfer nodal values from parent mesh to submesh by direct node mapping.
    pub fn transfer_from_parent(&self, parent_values: &[f64]) -> Vec<f64> {
        self.parent_node_of_sub
            .iter()
            .map(|&pn| parent_values[pn as usize])
            .collect()
    }

    /// Transfer nodal values from submesh back to parent mesh.
    ///
    /// If multiple submesh nodes map to the same parent node (rare for current
    /// extraction strategy), values are averaged.
    pub fn transfer_to_parent(&self, sub_values: &[f64], parent_n_nodes: usize) -> Vec<f64> {
        assert_eq!(
            sub_values.len(),
            self.parent_node_of_sub.len(),
            "transfer_to_parent: sub value length mismatch"
        );

        let mut out = vec![0.0_f64; parent_n_nodes];
        let mut cnt = vec![0usize; parent_n_nodes];

        for (si, &pn) in self.parent_node_of_sub.iter().enumerate() {
            let p = pn as usize;
            out[p] += sub_values[si];
            cnt[p] += 1;
        }

        for i in 0..parent_n_nodes {
            if cnt[i] > 0 {
                out[i] /= cnt[i] as f64;
            }
        }

        out
    }
}

/// Extract a submesh containing elements whose tag belongs to `element_tags`.
///
/// Returns node- and element-remapped mesh plus parent mapping vectors.
pub fn extract_submesh(mesh: &Mesh<2>, element_tags: &[i32]) -> SubMesh {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "extract_submesh: only Tri3 meshes are supported"
    );

    let tag_set: HashSet<i32> = element_tags.iter().copied().collect();
    let mut parent_elem_ids = Vec::<ElemId>::new();
    for e in 0..mesh.n_elems() as ElemId {
        if tag_set.contains(&mesh.elem_tags[e as usize]) {
            parent_elem_ids.push(e);
        }
    }

    let mut parent_nodes_set = HashSet::<NodeId>::new();
    for &e in &parent_elem_ids {
        for &n in mesh.elem_nodes(e) {
            parent_nodes_set.insert(n);
        }
    }

    let mut parent_nodes: Vec<NodeId> = parent_nodes_set.into_iter().collect();
    parent_nodes.sort_unstable();

    let mut sub_of_parent = HashMap::<NodeId, NodeId>::new();
    for (si, &pn) in parent_nodes.iter().enumerate() {
        sub_of_parent.insert(pn, si as NodeId);
    }

    let mut sub_coords = Vec::<f64>::with_capacity(parent_nodes.len() * 2);
    for &pn in &parent_nodes {
        let [x, y] = mesh.coords_of(pn);
        sub_coords.push(x);
        sub_coords.push(y);
    }

    let mut sub_conn = Vec::<NodeId>::new();
    let mut sub_elem_tags = Vec::<i32>::new();
    for &pe in &parent_elem_ids {
        let ns = mesh.elem_nodes(pe);
        sub_conn.push(sub_of_parent[&ns[0]]);
        sub_conn.push(sub_of_parent[&ns[1]]);
        sub_conn.push(sub_of_parent[&ns[2]]);
        sub_elem_tags.push(mesh.elem_tags[pe as usize]);
    }

    // Keep only boundary faces entirely inside selected node set.
    let mut sub_face_conn = Vec::<NodeId>::new();
    let mut sub_face_tags = Vec::<i32>::new();
    for f in 0..mesh.n_faces() {
        let a = mesh.face_conn[2 * f];
        let b = mesh.face_conn[2 * f + 1];
        if let (Some(&sa), Some(&sb)) = (sub_of_parent.get(&a), sub_of_parent.get(&b)) {
            sub_face_conn.push(sa);
            sub_face_conn.push(sb);
            sub_face_tags.push(mesh.face_tags[f]);
        }
    }

    let sub_mesh = Mesh::uniform(
        sub_coords,
        sub_conn,
        sub_elem_tags,
        ElementType::Tri3,
        sub_face_conn,
        sub_face_tags,
        ElementType::Line2,
    );

    SubMesh {
        mesh: sub_mesh,
        parent_elem_ids,
        parent_node_of_sub: parent_nodes,
    }
}

/// Extract submesh by named attribute set.
///
/// The named set is resolved through `registry`, and its element tags are used
/// as extraction tags.
pub fn extract_submesh_by_name(
    mesh: &Mesh<2>,
    registry: &NamedAttributeRegistry,
    set_name: &str,
) -> Result<SubMesh, fem_core::FemError> {
    let set = registry.get(set_name).ok_or_else(|| {
        fem_core::FemError::Mesh(format!("named attribute set not found: {set_name}"))
    })?;
    if set.element_tags.is_empty() {
        return Err(fem_core::FemError::Mesh(format!(
            "named attribute set has no element tags: {set_name}"
        )));
    }
    Ok(extract_submesh(mesh, &set.element_tags))
}

// ─── Boundary SubMesh (MFEM SubMesh::CreateFromBoundary) ─────────────────

/// Boundary submesh extracted from a 3-D parent mesh (MFEM
/// `SubMesh::CreateFromBoundary` equivalent).
///
/// The submesh elements are the boundary faces of the parent mesh whose
/// boundary attribute belongs to `bdr_tags`.  The result is a 2-D surface
/// mesh (Tri3/Quad4 elements with Line2 boundary) embedded in 3-D space.
///
/// Vertex numbering follows MFEM `SubMeshUtils::AddElementsToMesh`
/// (`from_boundary = true`): faces are traversed in parent boundary order and
/// their vertices are numbered by first occurrence.  Element order is the
/// parent boundary-face order filtered by attribute.
#[derive(Debug, Clone)]
pub struct BoundarySubMesh {
    /// 2-D surface mesh (Tri3/Quad4 elements) embedded in 3-D.
    pub mesh: Mesh<3>,
    /// sub element i → parent boundary face id (index into parent face arrays).
    pub parent_face_ids: Vec<FaceId>,
    /// parent_node_of_sub[sub_node_id] = parent_node_id.
    pub parent_node_of_sub: Vec<NodeId>,
}

impl BoundarySubMesh {
    /// Number of submesh (surface) elements.
    pub fn n_elems(&self) -> usize {
        self.mesh.n_elems()
    }

    /// Transfer nodal values from parent mesh to the submesh (by node id).
    pub fn transfer_from_parent(&self, parent_values: &[f64]) -> Vec<f64> {
        self.parent_node_of_sub
            .iter()
            .map(|&pn| parent_values[pn as usize])
            .collect()
    }

    /// Transfer nodal values from submesh back to parent mesh.
    pub fn transfer_to_parent(&self, sub_values: &[f64], parent_n_nodes: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; parent_n_nodes];
        for (si, &pn) in self.parent_node_of_sub.iter().enumerate() {
            out[pn as usize] = sub_values[si];
        }
        out
    }
}

/// Local edge connectivity (vertex indices into the element's nodes) for the
/// boundary-face element types of a 3-D parent mesh.
fn local_edge_vertices_2d(elem_type: ElementType) -> Vec<[usize; 2]> {
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => {
            vec![[0, 1], [1, 2], [0, 2]]
        }
        ElementType::Quad4 | ElementType::Quad9 => {
            vec![[0, 1], [1, 2], [2, 3], [0, 3]]
        }
        _ => vec![],
    }
}

/// Extract a 2-D surface submesh from the boundary faces of a 3-D parent mesh
/// (1:1 with MFEM `SubMesh::CreateFromBoundary`).
///
/// Only boundary faces whose attribute is in `bdr_tags` become submesh
/// elements.  Vertices are numbered by first occurrence while traversing the
/// parent boundary faces in order (MFEM `SubMeshUtils::AddElementsToMesh`).
pub fn extract_boundary_submesh(mesh: &Mesh<3>, bdr_tags: &[i32]) -> BoundarySubMesh {
    let tag_set: HashSet<i32> = bdr_tags.iter().copied().collect();

    // Selected parent boundary faces, in parent boundary order.
    let mut parent_face_ids = Vec::<FaceId>::new();
    for f in 0..mesh.n_faces() as FaceId {
        if tag_set.contains(&(mesh.face_tags[f as usize] as i32)) {
            parent_face_ids.push(f);
        }
    }
    assert!(
        !parent_face_ids.is_empty(),
        "extract_boundary_submesh: no boundary faces match the given tags"
    );

    // Number vertices by first occurrence (face-local vertex order).
    let mut parent_nodes: Vec<NodeId> = Vec::new();
    let mut seen_nodes = HashSet::<NodeId>::new();
    for &f in &parent_face_ids {
        let bfv = mesh.bface_nodes(f);
        for &n in bfv {
            if seen_nodes.insert(n) {
                parent_nodes.push(n);
            }
        }
    }

    let mut sub_of_parent = HashMap::<NodeId, NodeId>::new();
    for (si, &pn) in parent_nodes.iter().enumerate() {
        sub_of_parent.insert(pn, si as NodeId);
    }

    // Submesh coordinates (3-D embedding).
    let mut sub_coords = Vec::<f64>::with_capacity(parent_nodes.len() * 3);
    for &pn in &parent_nodes {
        let [x, y, z] = mesh.coords_of(pn);
        sub_coords.push(x);
        sub_coords.push(y);
        sub_coords.push(z);
    }

    // Elements: one per selected boundary face, using the face's vertex order.
    let mut sub_conn = Vec::<NodeId>::new();
    let mut sub_elem_tags = Vec::<i32>::new();
    let mut sub_elem_types = Vec::<ElementType>::new();
    let mut sub_elem_offsets = Vec::<usize>::new();
    sub_elem_offsets.push(0);
    for &f in &parent_face_ids {
        let ft = mesh.face_type_at(f);
        assert!(
            ft == ElementType::Tri3 || ft == ElementType::Quad4,
            "extract_boundary_submesh: unsupported boundary face type {ft:?}"
        );
        for &n in mesh.bface_nodes(f) {
            sub_conn.push(sub_of_parent[&n]);
        }
        sub_elem_tags.push(mesh.face_tags[f as usize] as i32);
        sub_elem_types.push(ft);
        sub_elem_offsets.push(sub_conn.len());
    }

    // Boundary edges of the submesh: edges of the surface elements that belong
    // to exactly one element.
    let mut edge_counts: HashMap<[NodeId; 2], usize> = HashMap::new();
    for (i, &ft) in sub_elem_types.iter().enumerate() {
        let start = sub_elem_offsets[i];
        let ns: Vec<NodeId> = sub_conn[start..sub_elem_offsets[i + 1]].to_vec();
        for ev in local_edge_vertices_2d(ft) {
            let a = ns[ev[0]];
            let b = ns[ev[1]];
            let key = if a < b { [a, b] } else { [b, a] };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut sub_face_conn = Vec::<NodeId>::new();
    let mut sub_face_tags = Vec::<i32>::new();
    for (key, &cnt) in &edge_counts {
        if cnt == 1 {
            // Boundary edge (Line2), tag 1 (attribute value not used by the
            // H1/L2 port spaces; MFEM SubMesh::AddBoundaryElements assigns
            // attribute max+1 for internal-cut faces).
            sub_face_conn.push(key[0]);
            sub_face_conn.push(key[1]);
            sub_face_tags.push(1);
        }
    }

    let all_uniform = sub_elem_types.iter().all(|&t| t == sub_elem_types[0]);
    let first_type = sub_elem_types[0];

    let sub_mesh = Mesh {
        coords: sub_coords,
        conn: sub_conn,
        elem_tags: sub_elem_tags,
        elem_type: if all_uniform { first_type } else { ElementType::Tri3 },
        face_conn: sub_face_conn,
        face_tags: sub_face_tags.into_iter().map(|t| t as crate::BoundaryTag).collect(),
        face_type: ElementType::Line2,
        elem_types: if all_uniform { None } else { Some(sub_elem_types) },
        elem_offsets: if all_uniform { None } else { Some(sub_elem_offsets) },
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None, nc_vertex_view: None,
    };

    BoundarySubMesh {
        mesh: sub_mesh,
        parent_face_ids,
        parent_node_of_sub: parent_nodes,
    }
}

// ─── 3-D SubMesh ──────────────────────────────────────────────────────────

/// Submesh view extracted from a 3-D parent mesh.
#[derive(Debug, Clone)]
pub struct SubMesh3D {
    /// Extracted mesh (may have mixed element types).
    pub mesh: Mesh<3>,
    /// Parent element ids corresponding to submesh elements.
    pub parent_elem_ids: Vec<ElemId>,
    /// parent_node_of_sub[sub_node_id] = parent_node_id.
    pub parent_node_of_sub: Vec<NodeId>,
    /// sub_face_of_parent[sub_face_id] = parent_face_id, or `FaceId::MAX`.
    pub parent_face_of_sub: Vec<FaceId>,
}

impl SubMesh3D {
    /// Transfer nodal values from parent mesh to submesh.
    pub fn transfer_from_parent(&self, parent_values: &[f64]) -> Vec<f64> {
        self.parent_node_of_sub
            .iter()
            .map(|&pn| parent_values[pn as usize])
            .collect()
    }

    /// Transfer nodal values from submesh back to parent mesh.
    pub fn transfer_to_parent(&self, sub_values: &[f64], parent_n_nodes: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; parent_n_nodes];
        let mut cnt = vec![0usize; parent_n_nodes];
        for (si, &pn) in self.parent_node_of_sub.iter().enumerate() {
            let p = pn as usize;
            out[p] += sub_values[si];
            cnt[p] += 1;
        }
        for i in 0..parent_n_nodes {
            if cnt[i] > 0 {
                out[i] /= cnt[i] as f64;
            }
        }
        out
    }

    /// Transfer DOF values from submesh space to parent space by element mapping.
    ///
    /// `sub_elem_dofs` and `par_elem_dofs` return the DOF list for a given
    /// element index (submesh element or parent element).
    pub fn transfer_dofs_to_parent(
        &self,
        sub_dofs: &[f64],
        n_parent_dofs: usize,
        sub_elem_dofs: &impl Fn(u32) -> Vec<u32>,
        par_elem_dofs: &impl Fn(u32) -> Vec<u32>,
        sub_elem_signs: &impl Fn(u32) -> Vec<f64>,
        par_elem_signs: &impl Fn(u32) -> Vec<f64>,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; n_parent_dofs];
        let mut cnt = vec![0usize; n_parent_dofs];
        for (si, &pe) in self.parent_elem_ids.iter().enumerate() {
            let sub_edofs = sub_elem_dofs(si as u32);
            let par_edofs = par_elem_dofs(pe);
            let sub_signs = sub_elem_signs(si as u32);
            let par_signs = par_elem_signs(pe);
            for k in 0..sub_edofs.len().min(par_edofs.len()) {
                let sd = sub_edofs[k] as usize;
                let pd = par_edofs[k] as usize;
                // MFEM SubMeshUtils::BuildVdofToVdofMap: the sub-domain and
                // parent-domain FE spaces may use different canonical face
                // orientations, so the transfered value must be sign-flipped
                // when the element-local orientations disagree.
                let ss = sub_signs.get(k).copied().unwrap_or(1.0);
                let sp = par_signs.get(k).copied().unwrap_or(1.0);
                out[pd] += sp * ss * sub_dofs[sd];
                cnt[pd] += 1;
            }
        }
        for i in 0..n_parent_dofs {
            if cnt[i] > 0 {
                out[i] /= cnt[i] as f64;
            }
        }
        out
    }
}

/// Helper: vertices of each local face (vertex indices into the element's nodes).
fn local_face_vertices_3d(elem_type: ElementType) -> Vec<Vec<usize>> {
    match elem_type {
        ElementType::Tet4 | ElementType::Tet10 => vec![
            vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2],
        ],
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => vec![
            vec![0, 1, 2, 3], vec![4, 5, 6, 7],
            vec![0, 1, 5, 4], vec![2, 3, 7, 6],
            vec![0, 3, 7, 4], vec![1, 2, 6, 5],
        ],
        ElementType::Prism6 | ElementType::Prism15 => vec![
            vec![0, 1, 2],       // bottom tri
            vec![3, 4, 5],       // top tri
            vec![0, 1, 4, 3],    // quad
            vec![1, 2, 5, 4],    // quad
            vec![0, 2, 5, 3],    // quad
        ],
        _ => vec![],
    }
}

/// Extract a 3-D submesh containing elements whose tag belongs to `element_tags`.
///
/// Supports mixed element types (Tet4, Hex8, Prism6, etc.).
pub fn extract_submesh_3d(mesh: &Mesh<3>, element_tags: &[i32]) -> SubMesh3D {
    let tag_set: HashSet<i32> = element_tags.iter().copied().collect();

    // Collect parent element ids matching the tags.
    let mut parent_elem_ids = Vec::<ElemId>::new();
    for e in 0..mesh.n_elems() as ElemId {
        if tag_set.contains(&mesh.elem_tags[e as usize]) {
            parent_elem_ids.push(e);
        }
    }
    assert!(!parent_elem_ids.is_empty(), "extract_submesh_3d: no elements match the given tags");

    // Collect vertices in MFEM's SubMeshUtils::AddElementsToMesh order:
    // element traversal × local vertex order, first occurrence assigns the
    // submesh id (NOT sorted parent ids — that changes the H1/RT DOF
    // ordering and hence the GS-sweep history).
    let mut parent_nodes: Vec<NodeId> = Vec::new();
    let mut seen_nodes = HashSet::<NodeId>::new();
    for &e in &parent_elem_ids {
        for &n in mesh.elem_nodes(e) {
            if seen_nodes.insert(n) {
                parent_nodes.push(n);
            }
        }
    }

    let mut sub_of_parent = HashMap::<NodeId, NodeId>::new();
    for (si, &pn) in parent_nodes.iter().enumerate() {
        sub_of_parent.insert(pn, si as NodeId);
    }

    // Build submesh coordinates.
    let mut sub_coords = Vec::<f64>::with_capacity(parent_nodes.len() * 3);
    for &pn in &parent_nodes {
        let [x, y, z] = mesh.coords_of(pn);
        sub_coords.push(x);
        sub_coords.push(y);
        sub_coords.push(z);
    }

    // Build element connectivity for submesh (mixed).
    let mut sub_conn = Vec::<NodeId>::new();
    let mut sub_elem_tags = Vec::<i32>::new();
    let mut sub_elem_types = Vec::<ElementType>::new();
    let mut sub_elem_offsets = Vec::<usize>::new();
    sub_elem_offsets.push(0);

    for &pe in &parent_elem_ids {
        let ns = mesh.elem_nodes(pe);
        for &n in ns {
            sub_conn.push(sub_of_parent[&n]);
        }
        sub_elem_tags.push(mesh.elem_tags[pe as usize]);
        let et = mesh.element_type_at(pe);
        sub_elem_types.push(et);
        sub_elem_offsets.push(sub_conn.len());
    }

    // Determine uniform element type (if all same) or keep as mixed.
    let first_type = sub_elem_types[0];
    let all_uniform = sub_elem_types.iter().all(|&t| t == first_type);

    // Build boundary faces for the submesh.
    // A face is on the submesh boundary if it belongs to exactly one submesh element.
    // We build this by collecting all faces of all submesh elements and counting
    // how many elements share each face.

    #[derive(Hash, Eq, PartialEq, Clone)]
    struct FaceKey(Vec<NodeId>);

    // (tag, parent_elem, global_verts, parent_face_id)
    let mut face_counts: HashMap<FaceKey, (i32, ElemId, Vec<NodeId>, FaceId)> = HashMap::new();

    // For each submesh element, iterate its local faces.
    for (&pe, &et) in parent_elem_ids.iter().zip(&sub_elem_types) {
        let ns = mesh.elem_nodes(pe);
        let local_faces = local_face_vertices_3d(et);
        for lfv in &local_faces {
            let mut global_verts: Vec<u32> = lfv.iter().map(|&i| ns[i]).collect();
            global_verts.sort_unstable(); // canonical key
            let key = FaceKey(global_verts.clone());

            let (parent_tag, parent_face_id) = parent_boundary_face(mesh, &global_verts);
            // Submesh-internal faces shared with the rest of the parent mesh
            // (not parent boundary faces) get the MFEM treatment: attribute
            // = max_bdr_attr + 1 (SubMeshUtils::AddBoundaryElements).
            let parent_tag = if parent_face_id == FaceId::MAX {
                mesh.face_tags.iter().copied().max().map_or(1, |m| m + 1) as i32
            } else {
                parent_tag
            };
            face_counts
                .entry(key)
                .and_modify(|(tag, _pe, _nodes, _pfid)| {
                    *tag = -1; // mark as internal
                })
                .or_insert((parent_tag, pe, global_verts, parent_face_id));
        }
    }

    // Keep only faces with tag != -1 (boundary faces).
    let mut sub_face_conn = Vec::<NodeId>::new();
    let mut sub_face_tags = Vec::<i32>::new();
    let mut sub_face_types = Vec::<ElementType>::new();
    let mut sub_face_offsets = Vec::<usize>::new();
    let mut parent_face_of_sub = Vec::<FaceId>::new();
    sub_face_offsets.push(0);

    for (_, &(tag, _pe, ref global_verts, parent_fid)) in &face_counts {
        if tag < 0 { continue; } // internal face, skip

        // Map global vertices to submesh vertices.
        let sub_verts: Vec<u32> = global_verts.iter().map(|&v| sub_of_parent[&v]).collect();

        // Determine face type by vertex count.
        let ftype = match sub_verts.len() {
            3 => ElementType::Tri3,
            4 => ElementType::Quad4,
            _ => continue,
        };

        for &sv in &sub_verts {
            sub_face_conn.push(sv);
        }
        sub_face_tags.push(tag);
        sub_face_types.push(ftype);
        sub_face_offsets.push(sub_face_conn.len());
        parent_face_of_sub.push(parent_fid);
    }

    // Construct the submesh.
    let use_mixed_elem = !all_uniform;
    let use_mixed_face = sub_face_types.len() > 1
        && !sub_face_types.iter().all(|&t| t == sub_face_types[0]);

    let sub_mesh = Mesh {
        coords: sub_coords,
        conn: sub_conn,
        elem_tags: sub_elem_tags,
        elem_type: if all_uniform { first_type } else { ElementType::Tet4 },
        face_conn: sub_face_conn,
        face_tags: sub_face_tags.into_iter().map(|t| t as crate::BoundaryTag).collect(),
        face_type: if sub_face_types.is_empty() { ElementType::Tri3 }
                   else { sub_face_types[0] },
        elem_types: if use_mixed_elem { Some(sub_elem_types) } else { None },
        elem_offsets: if use_mixed_elem { Some(sub_elem_offsets) } else { None },
        face_types: if use_mixed_face { Some(sub_face_types) } else { None },
        face_offsets: if use_mixed_face { Some(sub_face_offsets) } else { None },
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None, nc_vertex_view: None,
    };

    SubMesh3D {
        mesh: sub_mesh,
        parent_elem_ids,
        parent_node_of_sub: parent_nodes,
        parent_face_of_sub,
    }
}

/// Look up the boundary tag and parent face ID of a face (identified by its
/// sorted global vertex set) in the parent mesh.
fn parent_boundary_face(mesh: &Mesh<3>, face_verts: &[u32]) -> (i32, FaceId) {
    for f in 0..mesh.n_faces() as FaceId {
        let bfv = mesh.bface_nodes(f);
        let mut bfv_sorted: Vec<u32> = bfv.iter().copied().collect();
        bfv_sorted.sort_unstable();
        if bfv_sorted.len() == face_verts.len() && bfv_sorted.iter().zip(face_verts).all(|(a, b)| a == b) {
            return (mesh.face_tags[f as usize] as i32, f);
        }
    }
    // Not a parent boundary face → submesh-internal face that became external.
    // Assign tag 1 and FaceId::MAX.
    (1, FaceId::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedAttributeSet;

    /// Two Tets sharing face {0,1,2}; tet0 boundary faces (attr 1):
    /// [1,2,3],[0,2,3],[0,1,3]; tet1 boundary faces (attr 2):
    /// [0,2,4],[1,2,4],[1,0,4].
    fn two_tets_mixed_attr() -> Mesh<3> {
        let coords = vec![
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            0.0, 1.0, 0.0, // 2
            0.0, 0.0, 1.0, // 3
            0.0, 0.0, 2.0, // 4
        ];
        let conn = vec![0u32, 1, 2, 3, 1, 0, 2, 4];
        let elem_tags = vec![1i32, 2];
        let face_conn = vec![1u32, 2, 3, 0, 2, 3, 0, 1, 3, 0, 2, 4, 1, 2, 4, 1, 0, 4];
        let face_tags = vec![1i32, 1, 1, 2, 2, 2];
        Mesh {
            coords,
            conn,
            elem_tags,
            elem_type: ElementType::Tet4,
            face_conn,
            face_tags: face_tags.into_iter().map(|t| t as crate::BoundaryTag).collect(),
            face_type: ElementType::Tri3,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None, nc_vertex_view: None,
        }
    }

    #[test]
    fn extract_boundary_submesh_by_attr() {
        let m = two_tets_mixed_attr();
        let sub = extract_boundary_submesh(&m, &[2]);
        assert_eq!(sub.n_elems(), 3);
        assert_eq!(sub.parent_face_ids, vec![3, 4, 5]);
        // Vertices numbered by first occurrence over faces (0,2,4),(1,2,4),(1,0,4).
        assert_eq!(sub.parent_node_of_sub, vec![0, 2, 4, 1]);
        // Submesh element connectivity (local vertex ids):
        // face3 (0,2,4) -> 0,1,2 ; face4 (1,2,4) -> 3,1,2 ; face5 (1,0,4) -> 3,0,2
        assert_eq!(sub.mesh.conn, vec![0, 1, 2, 3, 1, 2, 3, 0, 2]);
        // Boundary edges: {0,1},{1,2},{0,2} shared-face edges (Line2), 3 edges.
        assert_eq!(sub.mesh.n_faces(), 3);
        assert_eq!(sub.mesh.face_type, ElementType::Line2);
        // Node transfer round-trip.
        let parent_vals: Vec<f64> = (0..m.n_nodes()).map(|i| i as f64 * 10.0).collect();
        let sub_vals = sub.transfer_from_parent(&parent_vals);
        let back = sub.transfer_to_parent(&sub_vals, m.n_nodes());
        assert_eq!(back[0], 0.0);
        assert_eq!(back[2], 20.0);
        assert_eq!(back[4], 40.0);
        assert_eq!(back[1], 10.0);
        assert_eq!(back[3], 0.0);
    }

    #[test]
    fn extract_submesh_by_tag() {
        let mut m = Mesh::<2>::unit_square_tri(2);
        // Mark first half with tag 2, second half with tag 3.
        let half = m.n_elems() / 2;
        for (i, t) in m.elem_tags.iter_mut().enumerate() {
            *t = if i < half { 2 } else { 3 };
        }

        let sub = extract_submesh(&m, &[2]);
        assert!(!sub.parent_elem_ids.is_empty());
        assert!(sub.mesh.n_elems() < m.n_elems());
        assert!(sub.mesh.elem_tags.iter().all(|&t| t == 2));
    }

    #[test]
    fn transfer_parent_sub_parent_roundtrip_on_selected_nodes() {
        let mut m = Mesh::<2>::unit_square_tri(2);
        for (i, t) in m.elem_tags.iter_mut().enumerate() {
            *t = if i % 2 == 0 { 1 } else { 2 };
        }

        let sub = extract_submesh(&m, &[1]);
        let parent_vals: Vec<f64> = (0..m.n_nodes()).map(|i| i as f64).collect();
        let sub_vals = sub.transfer_from_parent(&parent_vals);
        let back = sub.transfer_to_parent(&sub_vals, m.n_nodes());

        for &pn in &sub.parent_node_of_sub {
            let p = pn as usize;
            assert!((back[p] - parent_vals[p]).abs() < 1e-12);
        }
    }

    #[test]
    fn extract_submesh_by_name_works() {
        let mut m = Mesh::<2>::unit_square_tri(2);
        for (i, t) in m.elem_tags.iter_mut().enumerate() {
            *t = if i % 2 == 0 { 4 } else { 8 };
        }

        let mut reg = NamedAttributeRegistry::new();
        reg.insert(NamedAttributeSet::new("fluid").with_element_tags([4]));

        let sub = extract_submesh_by_name(&m, &reg, "fluid").expect("submesh by name failed");
        assert!(!sub.parent_elem_ids.is_empty());
        assert!(sub.mesh.elem_tags.iter().all(|&t| t == 4));
    }

    #[test]
    fn extract_submesh_by_name_missing_set_errors() {
        let m = Mesh::<2>::unit_square_tri(1);
        let reg = NamedAttributeRegistry::new();
        let err = extract_submesh_by_name(&m, &reg, "missing")
            .expect_err("expected missing set error");
        let msg = format!("{err}");
        assert!(msg.contains("named attribute set not found"));
    }
}
