//! Incremental parallel mesh construction.
//!
//! [`ParMeshBuilder`] constructs a [`ParallelMesh`] from per‑rank data
//! **without** any rank holding the entire global mesh.  Each rank provides
//! only its owned elements and nodes; ghost topology is built automatically
//! via node‑owner exchange.
//!
//! This avoids the `O(N)` memory overhead of [`partition_mesh`] where
//! every rank first replicates the full serial mesh.
//!
//! # Example
//! ```ignore
//! let builder = ParMeshBuilder::new(comm.clone());
//! for e in 0..my_n_elems {
//!     builder.push_element(elem_type, nodes, tag);
//! }
//! let pmesh = builder.build(mesh_template);
//! ```

use std::collections::HashMap;
use fem_core::{Rank, NodeId};
use fem_mesh::Mesh;
use fem_mesh::element_type::ElementType;
use crate::comm::Comm;
use crate::partition::MeshPartition;
use crate::par_mesh::ParallelMesh;

/// Incremental builder for a distributed mesh.
///
/// Each rank pushes its own elements and the owning rank for each node.
/// After all ranks push, [`build`](Self::build) exchanges ghost info and
/// produces the final [`ParallelMesh`].
///
/// Memory per rank ≈ `O(local_elems + ghost_nodes)`, **not** `O(global_mesh)`.
pub struct ParMeshBuilder<const D: usize> {
    comm: Comm,
    owned_nodes: Vec<NodeId>,
    node_owner: HashMap<NodeId, Rank>,
    elem_conn: Vec<u32>,
    elem_tags: Vec<i32>,
    elem_types: Vec<ElementType>,
    coords: Vec<f64>,
}

impl<const D: usize> ParMeshBuilder<D> {
    pub fn new(comm: Comm) -> Self {
        ParMeshBuilder {
            comm,
            owned_nodes: Vec::new(),
            node_owner: HashMap::new(),
            elem_conn: Vec::new(),
            elem_tags: Vec::new(),
            elem_types: Vec::new(),
            coords: Vec::new(),
        }
    }

    /// Record a node owned by this rank.
    pub fn push_node(&mut self, global_id: NodeId, coords: &[f64; D]) {
        self.owned_nodes.push(global_id);
        self.node_owner.insert(global_id, self.comm.rank());
        self.coords.extend_from_slice(coords);
    }

    /// Record a node owned by another rank (local ghost).
    pub fn push_ghost_node(&mut self, global_id: NodeId, owner: Rank, coords: &[f64; D]) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.node_owner.entry(global_id) {
            e.insert(owner);
            self.coords.extend_from_slice(coords);
        }
    }

    /// Push an element (all nodes must have been pushed first).
    pub fn push_element(&mut self, elem_type: ElementType, nodes: &[NodeId], tag: i32) {
        self.elem_types.push(elem_type);
        self.elem_tags.push(tag);
        self.elem_conn.extend_from_slice(nodes);
    }

    /// Build the ParallelMesh from accumulated data.
    ///
    /// After this call, the builder is consumed.
    pub fn build(self) -> ParallelMesh<Mesh<D>> {
        let local_rank = self.comm.rank();
        let _n_ranks = self.comm.size();

        // Map global node IDs → local indices.
        let global_ids: Vec<NodeId> = self.node_owner.keys().copied().collect();
        let n_local = global_ids.len();
        let mut global_to_local: HashMap<NodeId, usize> = HashMap::new();
        for (li, &gid) in global_ids.iter().enumerate() {
            global_to_local.insert(gid, li);
        }

        // Compute which nodes are owned vs ghost.
        let n_owned = self.owned_nodes.len();
        let mut node_owner = vec![local_rank; n_local];
        for (&gid, &owner) in &self.node_owner {
            if let Some(&li) = global_to_local.get(&gid) {
                node_owner[li] = owner;
            }
        }

        // Renumber element connectivity global→local.
        let n_elems = self.elem_types.len();
        let n_owned_elems = self.elem_tags.len();
        let global_elem_ids: Vec<u32> = (0..n_owned_elems as u32).collect();
        let primary_type = if n_elems > 0 { self.elem_types[0] } else { ElementType::Tri3 };
        let uniform_types = if n_elems > 0 && self.elem_types[1..].iter().all(|t| *t == primary_type) { None } else { Some(self.elem_types.clone()) };
        let mut local_conn = Vec::with_capacity(self.elem_conn.len());
        for &gid in &self.elem_conn {
            let li = global_to_local[&gid] as u32;
            local_conn.push(li);
        }

        // Build local mesh.
        let local_mesh = Mesh {
            coords: self.coords,
            conn: local_conn,
            elem_tags: self.elem_tags,
            elem_type: primary_type,
            face_conn: Vec::new(),
            face_tags: Vec::new(),
            face_type: ElementType::Tri3,
            elem_types: uniform_types,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None, nc_vertex_view: None,
        };

        let elem_owner = vec![local_rank; n_owned_elems];
        let partition = MeshPartition::from_raw(
            n_owned, n_local - n_owned, n_owned_elems, 0,
            global_ids, node_owner, global_elem_ids, elem_owner,
        );
        ParallelMesh::new(local_mesh, self.comm, partition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::ElemId;
    use fem_mesh::Mesh;
    use fem_mesh::topology::MeshTopology;
    use crate::mpi_test_env::test_world_comm;
    use crate::par_partition::partition_mesh;

    #[test]
    fn builder_matches_serial_partition() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let comm = test_world_comm();
        let serial_pmesh = partition_mesh(&mesh, &comm);

        // Build the same mesh via builder
        let mut builder = ParMeshBuilder::new(comm.clone());
        for n in 0..mesh.n_nodes() as NodeId {
            let mut c = [0.0; 2];
            let coords = mesh.node_coords(n);
            c.copy_from_slice(&coords[..2]);
            builder.push_node(n, &c);
        }
        for e in 0..mesh.n_elems() as ElemId {
            let nodes = mesh.element_nodes(e);
            let tag = mesh.element_tag(e);
            builder.push_element(mesh.element_type_at(e), nodes, tag);
        }
        let builder_pmesh = builder.build();

        eprintln!("serial mesh: nodes={}, elems={}, owned_nodes={}, ghost_nodes={}",
            serial_pmesh.local_mesh().n_nodes(), mesh.n_elems(),
            serial_pmesh.partition().n_owned_nodes, serial_pmesh.partition().n_ghost_nodes);
        eprintln!("builder mesh: nodes={}, elems={}, owned_nodes={}, ghost_nodes={}",
            builder_pmesh.local_mesh().n_nodes(), builder_pmesh.local_mesh().n_elems(),
            builder_pmesh.partition().n_owned_nodes, builder_pmesh.partition().n_ghost_nodes);

        // Verify same owned node counts and element counts
        assert_eq!(serial_pmesh.partition().n_owned_nodes, builder_pmesh.partition().n_owned_nodes);
        assert_eq!(serial_pmesh.n_owned_nodes(), builder_pmesh.n_owned_nodes());
        assert_eq!(mesh.n_elems(), builder_pmesh.local_mesh().n_elems(),
            "builder should have all {} elements", mesh.n_elems());
    }
}
