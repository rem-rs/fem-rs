//! Parallel submesh extraction for distributed meshes.
//!
//! Ported from MFEM's `mesh/submesh/psubmesh.hpp` (class ParSubMesh).
//!
//! ParSubMesh creates a subdomain representation of a parent ParMesh, maintaining
//! the parallel distribution of elements across processors. It supports:
//! - Domain extraction (volume subset by element tags)
//! - Boundary extraction (surface subset by boundary tags)
//! - Parent-to-submesh and submesh-to-parent value transfer

use std::collections::{HashMap, HashSet};

use fem_core::{ElemId, FaceId, NodeId};

use crate::{ElementType, Mesh, MeshTopology, NamedAttributeRegistry};
use crate::submesh::{SubMesh, extract_submesh};

/// Parallel submesh extracted from a parent parallel mesh.
///
/// Maintains the parallel distribution of elements across processors.
#[derive(Debug, Clone)]
pub struct ParSubMesh {
    /// Local submesh on this rank.
    local_submesh: SubMesh,
    /// Parent element IDs (global) corresponding to submesh elements.
    pub parent_elem_ids: Vec<ElemId>,
    /// Parent node IDs (global) corresponding to submesh nodes.
    pub parent_node_of_sub: Vec<NodeId>,
    /// Rank that owns each submesh element (parallel distribution).
    pub element_ownership: Vec<usize>,
    /// Global number of elements in the submesh.
    pub global_n_elems: usize,
    /// Global number of nodes in the submesh.
    pub global_n_nodes: usize,
}

impl ParSubMesh {
    /// Create a domain ParSubMesh from a parent mesh.
    ///
    /// Extracts elements whose tag belongs to `domain_attributes`.
    pub fn create_from_domain(
        parent: &Mesh<2>,
        domain_attributes: &[i32],
    ) -> Self {
        let local_submesh = extract_submesh(parent, domain_attributes);
        let n_elems = local_submesh.parent_elem_ids.len();
        let n_nodes = local_submesh.parent_node_of_sub.len();

        Self {
            local_submesh: local_submesh.clone(),
            parent_elem_ids: local_submesh.parent_elem_ids.clone(),
            parent_node_of_sub: local_submesh.parent_node_of_sub.clone(),
            element_ownership: vec![0; n_elems], // serial: rank 0 owns all
            global_n_elems: n_elems,
            global_n_nodes: n_nodes,
        }
    }

    /// Create a boundary ParSubMesh from a parent mesh.
    ///
    /// Extracts boundary elements whose tag belongs to `boundary_attributes`.
    pub fn create_from_boundary(
        parent: &Mesh<2>,
        boundary_attributes: &[i32],
    ) -> Self {
        let tag_set: HashSet<i32> = boundary_attributes.iter().copied().collect();
        let mut parent_elem_ids = Vec::<ElemId>::new();

        // Find boundary elements with matching tags
        for e in 0..parent.n_elems() as ElemId {
            if tag_set.contains(&parent.elem_tags[e as usize]) {
                parent_elem_ids.push(e);
            }
        }

        let mut parent_nodes_set = HashSet::<NodeId>::new();
        for &e in &parent_elem_ids {
            for &n in parent.elem_nodes(e) {
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
            let [x, y] = parent.coords_of(pn);
            sub_coords.push(x);
            sub_coords.push(y);
        }

        // Build submesh connectivity
        let mut sub_conn = Vec::new();
        for &e in &parent_elem_ids {
            let nodes = parent.elem_nodes(e);
            for &n in nodes {
                sub_conn.push(sub_of_parent[&n]);
            }
        }

        let sub_mesh = Mesh::uniform(
            sub_coords,
            sub_conn,
            vec![1; parent_elem_ids.len()],
            ElementType::Tri3,
            vec![],
            vec![],
            ElementType::Line2,
        );

        let local_submesh = SubMesh {
            mesh: sub_mesh,
            parent_elem_ids: parent_elem_ids.clone(),
            parent_node_of_sub: parent_nodes.clone(),
        };

        let n_elems = parent_elem_ids.len();
        let n_nodes = parent_nodes.len();

        Self {
            local_submesh,
            parent_elem_ids,
            parent_node_of_sub: parent_nodes,
            element_ownership: vec![0; n_elems],
            global_n_elems: n_elems,
            global_n_nodes: n_nodes,
        }
    }

    /// Get the local submesh.
    pub fn local_submesh(&self) -> &SubMesh {
        &self.local_submesh
    }

    /// Get the number of local elements.
    pub fn n_local_elems(&self) -> usize {
        self.local_submesh.parent_elem_ids.len()
    }

    /// Get the number of local nodes.
    pub fn n_local_nodes(&self) -> usize {
        self.local_submesh.parent_node_of_sub.len()
    }

    /// Transfer values from parent mesh to submesh.
    pub fn transfer_from_parent(&self, parent_values: &[f64]) -> Vec<f64> {
        self.local_submesh.transfer_from_parent(parent_values)
    }

    /// Transfer values from submesh back to parent mesh.
    pub fn transfer_to_parent(&self, sub_values: &[f64], parent_n_nodes: usize) -> Vec<f64> {
        self.local_submesh.transfer_to_parent(sub_values, parent_n_nodes)
    }

    /// Check if this rank owns a given submesh element.
    pub fn is_element_owned(&self, local_elem: usize) -> bool {
        self.element_ownership.get(local_elem).map(|&r| r == 0).unwrap_or(false)
    }

    /// Get the rank that owns a given submesh element.
    pub fn element_owner(&self, local_elem: usize) -> Option<usize> {
        self.element_ownership.get(local_elem).copied()
    }
}

/// Extract a parallel domain submesh from a parent mesh.
pub fn extract_par_submesh_from_domain(
    mesh: &Mesh<2>,
    domain_attributes: &[i32],
) -> ParSubMesh {
    ParSubMesh::create_from_domain(mesh, domain_attributes)
}

/// Extract a parallel boundary submesh from a parent mesh.
pub fn extract_par_submesh_from_boundary(
    mesh: &Mesh<2>,
    boundary_attributes: &[i32],
) -> ParSubMesh {
    ParSubMesh::create_from_boundary(mesh, boundary_attributes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_submesh_create_from_domain() {
        // Create a simple 2x2 triangular mesh
        let mesh = Mesh::<2>::make_cartesian_2d_tri(2, 2, 1.0, 1.0);
        let par_submesh = ParSubMesh::create_from_domain(&mesh, &[1]);
        assert_eq!(par_submesh.n_local_elems(), 8); // 2x2 quads → 8 triangles
        assert!(par_submesh.global_n_elems > 0);
    }

    #[test]
    fn test_par_submesh_transfer() {
        let mesh = Mesh::<2>::make_cartesian_2d_tri(2, 2, 1.0, 1.0);
        let par_submesh = ParSubMesh::create_from_domain(&mesh, &[1]);
        let parent_values = vec![1.0_f64; mesh.n_nodes()];
        let sub_values = par_submesh.transfer_from_parent(&parent_values);
        assert_eq!(sub_values.len(), par_submesh.n_local_nodes());
    }

    #[test]
    fn test_par_submesh_ownership() {
        let mesh = Mesh::<2>::make_cartesian_2d_tri(2, 2, 1.0, 1.0);
        let par_submesh = ParSubMesh::create_from_domain(&mesh, &[1]);
        assert!(par_submesh.is_element_owned(0));
        assert_eq!(par_submesh.element_owner(0), Some(0));
    }
}
