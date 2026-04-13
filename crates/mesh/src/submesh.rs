//! Submesh extraction and parent/child nodal transfer utilities.
//!
//! Current scope:
//! - 2-D `Tri3` meshes
//! - extraction by element tags
//! - nodal-value transfer between parent and submesh

use std::collections::{HashMap, HashSet};

use fem_core::{ElemId, NodeId};

use crate::{ElementType, NamedAttributeRegistry, SimplexMesh};

/// Submesh view extracted from a parent mesh.
#[derive(Debug, Clone)]
pub struct SubMesh {
    /// Extracted mesh.
    pub mesh: SimplexMesh<2>,
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
pub fn extract_submesh(mesh: &SimplexMesh<2>, element_tags: &[i32]) -> SubMesh {
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

    let sub_mesh = SimplexMesh::uniform(
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
    mesh: &SimplexMesh<2>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedAttributeSet;

    #[test]
    fn extract_submesh_by_tag() {
        let mut m = SimplexMesh::<2>::unit_square_tri(2);
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
        let mut m = SimplexMesh::<2>::unit_square_tri(2);
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
        let mut m = SimplexMesh::<2>::unit_square_tri(2);
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
        let m = SimplexMesh::<2>::unit_square_tri(1);
        let reg = NamedAttributeRegistry::new();
        let err = extract_submesh_by_name(&m, &reg, "missing")
            .expect_err("expected missing set error");
        let msg = format!("{err}");
        assert!(msg.contains("named attribute set not found"));
    }
}
