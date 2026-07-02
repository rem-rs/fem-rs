//! 2-D polygon mesh for Virtual Element Methods (VEM).
//!
//! Each element is a polygon with a variable number of vertices.
//! Connectivity is stored in CSR format: flat node indices + per-element offsets.

use crate::element_type::ElementType;
use crate::topology::MeshTopology;
use fem_core::{ElemId, FaceId, NodeId};

/// 2-D polygon mesh with CSR connectivity.
#[derive(Debug, Clone)]
pub struct PolyMesh {
    /// Flat node coordinates: length `n_nodes * 2`.
    pub coords: Vec<f64>,
    /// Flat element connectivity (0-based node indices).
    pub conn: Vec<u32>,
    /// Per-element start offset into `conn`.  Length `n_elems + 1`.
    pub offsets: Vec<usize>,
    /// Number of boundary segments (edges on boundary).
    pub n_boundary: usize,
    /// Boundary face nodes: flat array of 2-node edges.
    bdr_conn: Vec<u32>,
    /// Per-boundary-face offset (always 2 per face).
    bdr_offsets: Vec<usize>,
}

impl PolyMesh {
    /// Build from coordinate and connectivity arrays.
    ///
    /// `offsets[e]` and `offsets[e+1]` define `conn[offsets[e]..offsets[e+1]]`.
    pub fn new(
        coords: Vec<f64>,
        conn: Vec<u32>,
        offsets: Vec<usize>,
    ) -> Self {
        let n_elems = offsets.len().saturating_sub(1);
        // Extract boundary edges: any edge (i,i+1) not shared by two elements.
        let mut bdr_conn = Vec::new();
        let mut bdr_offsets = vec![0usize];
        let n_nodes = coords.len() / 2;
        // Build edge-to-element map using sorted (a,b) keys
        let mut edge_map: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
        for e in 0..n_elems {
            let s = offsets[e]; let e2 = offsets[e+1];
            for i in s..e2 {
                let a = conn[i]; let b = conn[if i+1 < e2 { i+1 } else { s }];
                let key = if a < b { (a, b) } else { (b, a) };
                let entry = edge_map.entry(key).or_insert(0);
                *entry += 1;
            }
        }
        for e in 0..n_elems {
            let s = offsets[e]; let e2 = offsets[e+1];
            for i in s..e2 {
                let a = conn[i]; let b = conn[if i+1 < e2 { i+1 } else { s }];
                let key = if a < b { (a, b) } else { (b, a) };
                if edge_map.get(&key) == Some(&1) {
                    bdr_conn.push(a); bdr_conn.push(b);
                    bdr_offsets.push(bdr_conn.len());
                }
            }
        }
        // Deduplicate (each boundary edge appears once per element → appears twice here)
        // We take only unique edges via sorted-key dedup
        let mut unique_bdr: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let mut dedup_conn = Vec::new();
        let mut dedup_offs = vec![0usize];
        for i in 0..bdr_offsets.len() - 1 {
            let s = bdr_offsets[i]; let e = bdr_offsets[i+1];
            if e - s == 2 {
                let a = bdr_conn[s]; let b = bdr_conn[s+1];
                let key = if a < b { (a, b) } else { (b, a) };
                if unique_bdr.insert(key) {
                    dedup_conn.push(a); dedup_conn.push(b);
                    dedup_offs.push(dedup_conn.len());
                }
            }
        }
        Self {
            coords, conn, offsets,
            n_boundary: dedup_offs.len() - 1,
            bdr_conn: dedup_conn,
            bdr_offsets: dedup_offs,
        }
    }

    /// Build a uniform `m × n` quadrilateral mesh (like a grid).
    pub fn unit_square_quad(mx: usize, my: usize) -> Self {
        let nx = mx + 1; let ny = my + 1;
        let coords: Vec<f64> = (0..ny).flat_map(|j| (0..nx).flat_map(move |i| {
            vec![i as f64 / mx as f64, j as f64 / my as f64]
        })).collect();
        let mut conn = Vec::new();
        let mut offsets = vec![0usize];
        for j in 0..my { for i in 0..mx {
            let id = |x: usize, y: usize| (y * nx + x) as u32;
            conn.extend([id(i,j), id(i+1,j), id(i+1,j+1), id(i,j+1)]);
            offsets.push(conn.len());
        }}
        Self::new(coords, conn, offsets)
    }

    /// Build a regular hexagon mesh (6-sided cells).
    pub fn unit_square_hex(mx: usize, my: usize) -> Self {
        let coords: Vec<f64> = (0..=2*my).flat_map(|j| (0..=2*mx).flat_map(move |i| {
            let cx = i as f64 / (2 * mx) as f64;
            let cy = j as f64 / (2 * my) as f64;
            vec![cx, cy]
        })).collect();
        let nx = 2 * mx + 1;
        let mut conn = Vec::new();
        let mut offsets = vec![0usize];
        for j in 0..my { for i in 0..mx {
            let id = |x: usize, y: usize| (y * nx + x) as u32;
            // Hexagon: 6 vertices around the cell
            conn.extend([
                id(2*i, 2*j+1), id(2*i+1, 2*j), id(2*i+2, 2*j),
                id(2*i+2, 2*j+1), id(2*i+1, 2*j+2), id(2*i, 2*j+2),
            ]);
            offsets.push(conn.len());
        }}
        Self::new(coords, conn, offsets)
    }
}

impl MeshTopology for PolyMesh {
    fn dim(&self) -> u8 { 2 }
    fn n_nodes(&self) -> usize { self.coords.len() / 2 }
    fn n_elements(&self) -> usize { self.offsets.len().saturating_sub(1) }
    fn n_boundary_faces(&self) -> usize { self.n_boundary }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        let s = self.offsets[elem as usize];
        let e = self.offsets[elem as usize + 1];
        &self.conn[s..e]
    }

    fn element_type(&self, _elem: ElemId) -> ElementType { ElementType::Polygon }

    fn element_tag(&self, _elem: ElemId) -> i32 { 1 }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        let idx = node as usize * 2;
        &self.coords[idx..idx + 2]
    }

    fn face_nodes(&self, face: FaceId) -> &[NodeId] {
        let s = self.bdr_offsets[face as usize];
        let e = self.bdr_offsets[face as usize + 1];
        &self.bdr_conn[s..e]
    }

    fn face_tag(&self, _face: FaceId) -> i32 { 1 }

    fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) {
        (0, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_mesh_4x4() {
        let m = PolyMesh::unit_square_quad(3, 3);
        assert_eq!(m.n_elements(), 9);
        assert_eq!(m.n_nodes(), 16);
        assert_eq!(m.n_boundary_faces(), 12); // 3*4 boundary edges perim
        for e in 0..9 { assert_eq!(m.element_nodes(e).len(), 4); }
    }

    #[test]
    fn hex_mesh_counts() {
        let m = PolyMesh::unit_square_hex(2, 2);
        assert_eq!(m.n_elements(), 4);
        for e in 0..4 { assert_eq!(m.element_nodes(e).len(), 6); }
    }

    #[test]
    fn poly_mesh_node_coords() {
        let m = PolyMesh::unit_square_quad(1, 1);
        assert_eq!(m.node_coords(0), &[0.0, 0.0]);
        assert_eq!(m.node_coords(1), &[1.0, 0.0]);
        assert_eq!(m.node_coords(2), &[0.0, 1.0]);
    }

    #[test]
    fn element_type_is_polygon() {
        let m = PolyMesh::unit_square_quad(2, 2);
        assert_eq!(m.element_type(0), ElementType::Polygon);
    }
}
