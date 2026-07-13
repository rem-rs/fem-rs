//! PUMI-style mesh entity interface.
//!
//! Provides a higher-level mesh abstraction over [`Mesh`] that follows the
//! SCOREC PUMI API patterns:
//!
//! - Entity iteration: `vertices()`, `edges()`, `faces()`, `regions()`
//! - Adjacency queries: `adj(entity, dim) → Vec<Entity>`
//! - Classification: `model_tag(entity) → i32`
//!
//! # Architecture
//!
//! [`PumiMesh<D>`] wraps a [`Mesh<D>`] and caches entity-to-entity
//! adjacency tables (edge→vertex, face→vertex, region→face, region→edge,
//! etc.) at construction time.  After construction all adjacency queries are
//! O(1) lookups (amortised O(1) per entity pair).

use std::collections::HashMap;

use fem_core::NodeId;

use crate::element_type::ElementType;
use crate::simplex::Mesh;

/// Mesh entity dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PumiDim {
    Vertex = 0,
    Edge   = 1,
    Face   = 2,
    Region = 3,
}

/// A mesh entity identified by its dimension and index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub dim: PumiDim,
    pub idx: usize,
}

impl Entity {
    pub fn vertex(idx: usize) -> Self { Entity { dim: PumiDim::Vertex, idx } }
    pub fn edge(idx: usize)   -> Self { Entity { dim: PumiDim::Edge,   idx } }
    pub fn face(idx: usize)   -> Self { Entity { dim: PumiDim::Face,   idx } }
    pub fn region(idx: usize) -> Self { Entity { dim: PumiDim::Region, idx } }
}

/// PUMI-style mesh with cached adjacency.
///
/// After construction the adjacency tables are built once.  All queries are
/// O(1) (the returned vector is a pre-allocated slice).
pub struct PumiMesh<const D: usize> {
    mesh: Mesh<D>,

    // ── entity counts ──
    n_vertices: usize,
    n_edges:    usize,
    n_faces:    usize,
    n_regions:  usize,

    // ── entity→vertex connectivity ──
    /// Edge vertices: flat `[v0,v1]` per edge.
    edge_vertices: Vec<[NodeId; 2]>,
    /// Face vertices: flat per face; length = face_type.npe * n_faces.
    face_vertices: Vec<[NodeId; 4]>, // padded to 4 for Tet/Hex/Quad
    /// Region vertices: same as mesh.conn (element nodes).
    region_vertices: Vec<Vec<NodeId>>,

    // ── down-adjacency (region → faces/edges) ──
    region_faces: Vec<Vec<usize>>,   // face indices per region
    region_edges: Vec<Vec<usize>>,   // edge indices per region

    // ── classification ──
    /// Model tag per entity (Geometric model classification, 0 = unclassified).
    entity_tags: Vec<i32>,
}

impl<const D: usize> PumiMesh<D> {
    /// Build a `PumiMesh` from a `Mesh`, caching all adjacency tables.
    pub fn new(mesh: Mesh<D>) -> Self {
        let n_regions = mesh.n_elems();
        let n_vertices = mesh.n_nodes();
        let elem_type = mesh.elem_type;

        // ── Build edge→vertex set ──
        // Iterate all elements, collect unique {min, max} vertex pairs.
        let mut edge_set: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        let mut edge_vertices: Vec<[NodeId; 2]> = Vec::new();
        let npe = elem_type.nodes_per_element();

        for e in 0..n_regions {
            let conn = if let Some(offsets) = &mesh.elem_offsets {
                &mesh.conn[offsets[e]..offsets[e + 1]]
            } else {
                &mesh.conn[e * npe..(e + 1) * npe]
            };

            // For each pair of nodes that form an edge, add to set.
            let edge_pairs = element_edge_pairs(elem_type, conn);
            for &(a, b) in &edge_pairs {
                let key = if a < b { (a, b) } else { (b, a) };
                edge_set.entry(key).or_insert_with(|| {
                    let idx = edge_vertices.len();
                    edge_vertices.push([a, b]);
                    idx
                });
            }
        }
        let n_edges = edge_vertices.len();

        // ── Build face→vertex set ──
        // For 3D: collect unique face node sets from element faces.
        // For 2D: faces = edges (boundary edges).
        // For 2D: faces = the mesh.face_conn (same as edges for 2D).
        let mut face_vertices: Vec<[NodeId; 4]> = Vec::new();
        let mut face_set: HashMap<Vec<NodeId>, usize> = HashMap::new();

        if D == 3 {
            for e in 0..n_regions {
                let conn = if let Some(offsets) = &mesh.elem_offsets {
                    &mesh.conn[offsets[e]..offsets[e + 1]]
                } else {
                    &mesh.conn[e * npe..(e + 1) * npe]
                };
                let face_lists = element_face_nodes(elem_type, conn);
                for fv in &face_lists {
                    let mut key = fv.clone();
                    key.sort_unstable();
                    face_set.entry(key).or_insert_with(|| {
                        let mut padded = [0u32; 4];
                        let nf = fv.len().min(4);
                        padded[..nf].copy_from_slice(&fv[..nf]);
                        let idx = face_vertices.len();
                        face_vertices.push(padded);
                        idx
                    });
                }
            }
        } else {
            // 2D: faces = edges (boundary edges from the mesh)
            let fnpe = mesh.face_type.nodes_per_element();
            let nf = if fnpe > 0 { mesh.face_conn.len() / fnpe } else { 0 };
            for fi in 0..nf {
                let base = fi * fnpe;
                let p0 = if fnpe > 0 { mesh.face_conn[base] } else { 0 };
                let p1 = if fnpe > 1 { mesh.face_conn[base + 1] } else { 0 };
                face_vertices.push([p0, p1, 0, 0]);
            }
        }
        let n_faces = face_vertices.len();

        // ── Build region→face adjacency ──
        // For 3D: each element's faces map to global face indices.
        // For 2D: each element's edges map to face indices.
        let mut region_faces: Vec<Vec<usize>> = Vec::with_capacity(n_regions);
        let mut region_edges: Vec<Vec<usize>> = Vec::with_capacity(n_regions);

        if D == 3 {
            for e in 0..n_regions {
                let conn = if let Some(offsets) = &mesh.elem_offsets {
                    &mesh.conn[offsets[e]..offsets[e + 1]]
                } else {
                    &mesh.conn[e * npe..(e + 1) * npe]
                };
                let face_lists = element_face_nodes(elem_type, conn);
                let mut rfaces = Vec::with_capacity(face_lists.len());
                for fv in &face_lists {
                    let mut key = fv.clone();
                    key.sort_unstable();
                    if let Some(&fid) = face_set.get(&key) {
                        rfaces.push(fid);
                    }
                }
                region_faces.push(rfaces);

                // Edges per region
                let epairs = element_edge_pairs(elem_type, conn);
                let mut redges = Vec::with_capacity(epairs.len());
                for &(a, b) in &epairs {
                    let key = if a < b { (a, b) } else { (b, a) };
                    if let Some(&eid) = edge_set.get(&key) {
                        redges.push(eid);
                    }
                }
                region_edges.push(redges);
            }
        } else {
            // 2D: face adjacency
            // Compute which edges belong to which element
            for e in 0..n_regions {
                let conn = if let Some(offsets) = &mesh.elem_offsets {
                    &mesh.conn[offsets[e]..offsets[e + 1]]
                } else {
                    &mesh.conn[e * npe..(e + 1) * npe]
                };
                let epairs = element_edge_pairs(elem_type, conn);
                let mut rfaces = Vec::with_capacity(epairs.len());
                for &(a, b) in &epairs {
                    let key = if a < b { (a, b) } else { (b, a) };
                    if let Some(&eid) = edge_set.get(&key) {
                        rfaces.push(eid);
                    }
                }
                region_faces.push(rfaces.clone());
                region_edges.push(rfaces);
            }
        }

        // ── entity tags (classification, default 0) ──
        let total_ents = n_vertices + n_edges + n_faces + n_regions;
        let entity_tags = vec![0i32; total_ents];

        // Build region_vertices
        let mut region_vertices: Vec<Vec<NodeId>> = Vec::with_capacity(n_regions);
        for e in 0..n_regions {
            let conn = if let Some(offsets) = &mesh.elem_offsets {
                &mesh.conn[offsets[e]..offsets[e + 1]]
            } else {
                &mesh.conn[e * npe..(e + 1) * npe]
            };
            region_vertices.push(conn.to_vec());
        }

        PumiMesh {
            mesh,
            n_vertices, n_edges, n_faces, n_regions,
            edge_vertices, face_vertices,
            region_vertices,
            region_faces, region_edges,
            entity_tags,
        }
    }

    // ── entity count queries ──

    /// Number of vertices (mesh nodes).
    pub fn n_vertices(&self) -> usize { self.n_vertices }
    /// Number of unique edges.
    pub fn n_edges(&self) -> usize { self.n_edges }
    /// Number of unique faces (for 3D) or boundary edges (for 2D).
    pub fn n_faces(&self) -> usize { self.n_faces }
    /// Number of regions (elements).
    pub fn n_regions(&self) -> usize { self.n_regions }

    // ── entity→vertex ──

    /// Vertices of a region (element).
    pub fn region_vertices(&self, rid: usize) -> &[NodeId] {
        &self.region_vertices[rid]
    }

    /// Vertices of a face.
    pub fn face_vertices(&self, fid: usize) -> &[NodeId] {
        let p = &self.face_vertices[fid];
        // Return only the non-padding entries
        let n = if D == 2 { 2 } else {
            // Determine face type from number of non-zero entries
            if p[3] != 0 || p[2] != 0 { if p[0] != 0 && p[1] != 0 && p[2] != 0 && p[3] != 0 { 4 } else { 3 } }
            else if p[2] != 0 || (p[0] != 0 && p[1] != 0) { 3 }
            else { 2 }
        };
        &p[..n]
    }

    /// Vertices of an edge.
    pub fn edge_vertices(&self, eid: usize) -> &[NodeId] {
        &self.edge_vertices[eid]
    }

    /// Coordinate of a vertex.
    pub fn vertex_coord(&self, vid: usize) -> &[f64] {
        &self.mesh.coords[vid * D..(vid + 1) * D]
    }

    // ── down-adjacency ──

    /// Faces belonging to a region.
    pub fn region_faces(&self, rid: usize) -> &[usize] {
        &self.region_faces[rid]
    }

    /// Edges belonging to a region.
    pub fn region_edges(&self, rid: usize) -> &[usize] {
        &self.region_edges[rid]
    }

    /// General adjacency: entities of dimension `dim` adjacent to `entity`.
    pub fn adj(&self, entity: &Entity, dim: PumiDim) -> Vec<Entity> {
        match (entity.dim, dim) {
            (PumiDim::Region, PumiDim::Face) => {
                self.region_faces(entity.idx).iter().map(|&f| Entity::face(f)).collect()
            }
            (PumiDim::Region, PumiDim::Edge) => {
                self.region_edges(entity.idx).iter().map(|&e| Entity::edge(e)).collect()
            }
            (PumiDim::Region, PumiDim::Vertex) => {
                self.region_vertices(entity.idx).iter().map(|&v| Entity::vertex(v as usize)).collect()
            }
            (PumiDim::Face, PumiDim::Vertex) => {
                self.face_vertices(entity.idx).to_vec().into_iter().map(|v| Entity::vertex(v as usize)).collect()
            }
            (PumiDim::Edge, PumiDim::Vertex) => {
                self.edge_vertices(entity.idx).to_vec().into_iter().map(|v| Entity::vertex(v as usize)).collect()
            }
            _ => Vec::new(),
        }
    }

    // ── iteration ──

    /// Iterator over all vertices.
    pub fn vertices(&self) -> Vec<Entity> {
        (0..self.n_vertices).map(Entity::vertex).collect()
    }

    /// Iterator over all edges.
    pub fn edges(&self) -> Vec<Entity> {
        (0..self.n_edges).map(Entity::edge).collect()
    }

    /// Iterator over all faces.
    pub fn faces(&self) -> Vec<Entity> {
        (0..self.n_faces).map(Entity::face).collect()
    }

    /// Iterator over all regions.
    pub fn regions(&self) -> Vec<Entity> {
        (0..self.n_regions).map(Entity::region).collect()
    }

    // ── classification ──

    /// Set the geometric model tag for a mesh entity.
    pub fn set_tag(&mut self, entity: &Entity, tag: i32) {
        let idx = self.entity_index(entity);
        self.entity_tags[idx] = tag;
    }

    /// Get the geometric model tag for a mesh entity (0 = unclassified).
    pub fn tag(&self, entity: &Entity) -> i32 {
        let idx = self.entity_index(entity);
        *self.entity_tags.get(idx).unwrap_or(&0)
    }

    /// Access the underlying `Mesh`.
    pub fn mesh(&self) -> &Mesh<D> {
        &self.mesh
    }

    // ── internal ──

    fn entity_index(&self, e: &Entity) -> usize {
        match e.dim {
            PumiDim::Vertex => e.idx,
            PumiDim::Edge   => self.n_vertices + e.idx,
            PumiDim::Face   => self.n_vertices + self.n_edges + e.idx,
            PumiDim::Region => self.n_vertices + self.n_edges + self.n_faces + e.idx,
        }
    }
}

// ─── Element topology helpers ───────────────────────────────────────────────

/// Return the edge-node pairs for an element type.
fn element_edge_pairs(elem_type: ElementType, conn: &[u32]) -> Vec<(u32, u32)> {
    let npe = conn.len();
    match elem_type {
        // Triangle edges (Tri3 = 3 edges, Tri6 = 3 edges + 3 midpoints)
        ElementType::Tri3 | ElementType::Tri6 => {
            if npe >= 3 {
                vec![(conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[0])]
            } else { vec![] }
        }
        // Tet edges (Tet4 = 6 edges)
        ElementType::Tet4 | ElementType::Tet10 => {
            if npe >= 4 {
                vec![
                    (conn[0], conn[1]), (conn[1], conn[2]), (conn[0], conn[2]),
                    (conn[0], conn[3]), (conn[1], conn[3]), (conn[2], conn[3]),
                ]
            } else { vec![] }
        }
        // Hex edges (Hex8 = 12 edges)
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            if npe >= 8 {
                vec![
                    (conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[3]), (conn[3], conn[0]),
                    (conn[4], conn[5]), (conn[5], conn[6]), (conn[6], conn[7]), (conn[7], conn[4]),
                    (conn[0], conn[4]), (conn[1], conn[5]), (conn[2], conn[6]), (conn[3], conn[7]),
                ]
            } else { vec![] }
        }
        // Quad edges
        ElementType::Quad4 | ElementType::Quad9 => {
            if npe >= 4 {
                vec![(conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[3]), (conn[3], conn[0])]
            } else { vec![] }
        }
        // Prism edges (Prism6 = 9 edges)
        ElementType::Prism6 | ElementType::Prism15 => {
            if npe >= 6 {
                vec![
                    (conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[0]),
                    (conn[3], conn[4]), (conn[4], conn[5]), (conn[5], conn[3]),
                    (conn[0], conn[3]), (conn[1], conn[4]), (conn[2], conn[5]),
                ]
            } else { vec![] }
        }
        // Pyramid edges (Pyramid5 = 8 edges)
        ElementType::Pyramid5 | ElementType::Pyramid13 => {
            if npe >= 5 {
                vec![
                    (conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[3]), (conn[3], conn[0]),
                    (conn[0], conn[4]), (conn[1], conn[4]), (conn[2], conn[4]), (conn[3], conn[4]),
                ]
            } else { vec![] }
        }
        _ => vec![],
    }
}

/// Return the face-node lists for an element type (3D only).
fn element_face_nodes(elem_type: ElementType, conn: &[u32]) -> Vec<Vec<u32>> {
    let npe = conn.len();
    match elem_type {
        ElementType::Tet4 | ElementType::Tet10 => {
            if npe >= 4 {
                vec![
                    vec![conn[0], conn[2], conn[1]], // reverse to match outward normal
                    vec![conn[0], conn[1], conn[3]],
                    vec![conn[1], conn[2], conn[3]],
                    vec![conn[2], conn[0], conn[3]],
                ]
            } else { vec![] }
        }
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            if npe >= 8 {
                vec![
                    vec![conn[0], conn[1], conn[2], conn[3]], // bottom
                    vec![conn[4], conn[5], conn[6], conn[7]], // top
                    vec![conn[0], conn[1], conn[5], conn[4]], // front
                    vec![conn[1], conn[2], conn[6], conn[5]], // right
                    vec![conn[2], conn[3], conn[7], conn[6]], // back
                    vec![conn[3], conn[0], conn[4], conn[7]], // left
                ]
            } else { vec![] }
        }
        ElementType::Prism6 | ElementType::Prism15 => {
            if npe >= 6 {
                vec![
                    vec![conn[0], conn[2], conn[1]], // bottom tri (reverse)
                    vec![conn[3], conn[4], conn[5]], // top tri
                    vec![conn[0], conn[1], conn[4], conn[3]], // quad 0
                    vec![conn[1], conn[2], conn[5], conn[4]], // quad 1
                    vec![conn[2], conn[0], conn[3], conn[5]], // quad 2
                ]
            } else { vec![] }
        }
        ElementType::Pyramid5 | ElementType::Pyramid13 => {
            if npe >= 5 {
                vec![
                    vec![conn[0], conn[3], conn[2], conn[1]], // base quad (reverse)
                    vec![conn[0], conn[1], conn[4]], // tri 0
                    vec![conn[1], conn[2], conn[4]], // tri 1
                    vec![conn[2], conn[3], conn[4]], // tri 2
                    vec![conn[3], conn[0], conn[4]], // tri 3
                ]
            } else { vec![] }
        }
        _ => vec![],
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pumi_2d_tri3_vertex_count() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let pumi = PumiMesh::new(mesh);
        assert_eq!(pumi.n_vertices(), 4);
        assert_eq!(pumi.n_regions(), 2);
        assert!(pumi.n_edges() >= 5); // 5 unique edges for a square with 2 tris
    }

    #[test]
    fn pumi_3d_tet4_entity_counts() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let pumi = PumiMesh::new(mesh);
        assert_eq!(pumi.n_vertices(), 8);
        assert_eq!(pumi.n_regions(), 6); // 5 or 6 tets in a unit cube
        assert!(pumi.n_edges() >= 12);   // unit cube has ≥12 unique edges
        assert!(pumi.n_faces() >= 12);   // each face may be split
    }

    #[test]
    fn pumi_region_faces_tet4() {
        // A single tet: 4 faces
        let mesh = Mesh::<3> {
            coords: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            conn: vec![0, 1, 2, 3],
            elem_tags: vec![0],
            elem_type: ElementType::Tet4,
            face_conn: vec![], face_tags: vec![], face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
            geometry: None,
        };
        let pumi = PumiMesh::new(mesh);
        assert_eq!(pumi.n_regions(), 1);
        assert_eq!(pumi.n_faces(), 4);
        assert_eq!(pumi.n_edges(), 6);
        assert_eq!(pumi.region_faces(0).len(), 4);
        assert_eq!(pumi.region_edges(0).len(), 6);
    }

    #[test]
    fn pumi_set_and_get_tag() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let mut pumi = PumiMesh::new(mesh);
        let v = Entity::vertex(0);
        assert_eq!(pumi.tag(&v), 0);
        pumi.set_tag(&v, 42);
        assert_eq!(pumi.tag(&v), 42);
    }

    #[test]
    fn pumi_adj_region_to_vertices() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let pumi = PumiMesh::new(mesh);
        let adj = pumi.adj(&Entity::region(0), PumiDim::Vertex);
        assert_eq!(adj.len(), 3); // Tri3 has 3 vertices
    }

    #[test]
    fn pumi_vertices_iter() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let pumi = PumiMesh::new(mesh);
        let verts = pumi.vertices();
        assert_eq!(verts.len(), 4);
    }

    #[test]
    fn pumi_edges_iter() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let pumi = PumiMesh::new(mesh);
        let edges = pumi.edges();
        assert!(edges.len() >= 5);
    }

    #[test]
    fn pumi_face_vertices_hex() {
        let hex = Mesh::<3> {
            coords: vec![0.0,0.0,0.0, 1.0,0.0,0.0, 0.0,1.0,0.0, 1.0,1.0,0.0,
                         0.0,0.0,1.0, 1.0,0.0,1.0, 0.0,1.0,1.0, 1.0,1.0,1.0],
            conn: vec![0,1,2,3,4,5,6,7],
            elem_tags: vec![0],
            elem_type: ElementType::Hex8,
            face_conn: vec![], face_tags: vec![], face_type: ElementType::Quad4,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
            geometry: None,
        };
        let pumi = PumiMesh::new(hex);
        assert_eq!(pumi.n_faces(), 6);
        assert_eq!(pumi.region_faces(0).len(), 6);
        assert!(pumi.n_edges() >= 12);
    }
}
