//! Mesh intersection / supermesh construction for 2-D triangular meshes.
//!
//! Given two `Mesh<2>` (Tri3) meshes covering the same domain,
//! computes the intersection supermesh: a conforming triangulation where
//! each element lies entirely within exactly one element from each input mesh.
//!
//! Uses bounding-box overlap detection and Sutherland-Hodgman polygon
//! clipping for triangle-triangle intersection.

use std::collections::HashMap;
use fem_core::{ElemId, NodeId};
use crate::ElementType;
use crate::Mesh;

/// A supermesh element: triangular sub-element of the intersection of
/// `src_elem` from mesh A and `tgt_elem` from mesh B.
#[derive(Debug, Clone)]
pub struct SupermeshElement {
    /// The 3 node indices of this triangle (referring to supermesh coords).
    pub nodes: [NodeId; 3],
    /// Element index from the first input mesh.
    pub src_elem: ElemId,
    /// Element index from the second input mesh.
    pub tgt_elem: ElemId,
}

/// Build the supermesh (intersection) of two Tri3 meshes.
///
/// # Returns
/// `(supermesh, elements)` where `supermesh` is a `Mesh<2>` and
/// `elements` lists each triangle with its source/target element indices.
pub fn build_supermesh(
    mesh_a: &Mesh<2>,
    mesh_b: &Mesh<2>,
) -> (Mesh<2>, Vec<SupermeshElement>) {
    assert_eq!(mesh_a.elem_type, ElementType::Tri3,
        "build_supermesh: mesh_a must be Tri3");
    assert_eq!(mesh_b.elem_type, ElementType::Tri3,
        "build_supermesh: mesh_b must be Tri3");

    // Build bounding boxes for each element in both meshes
    let bboxes_a = build_bboxes(mesh_a);
    let bboxes_b = build_bboxes(mesh_b);

    // Collect all unique nodes and triangle elements
    let mut all_coords: Vec<[f64; 2]> = Vec::new();
    let mut sup_elems: Vec<SupermeshElement> = Vec::new();
    let mut coord_map: HashMap<u64, NodeId> = HashMap::new();

    #[allow(dead_code)]
    fn add_node(c: [f64; 2], coords: &mut Vec<[f64; 2]>, map: &mut HashMap<u64, NodeId>) -> NodeId {
        let x = (c[0] * 1e10).round() as i64;
        let y = (c[1] * 1e10).round() as i64;
        let k = (x.wrapping_mul(314159) ^ y) as u64;
        if let Some(&n) = map.get(&k) { return n; }
        let id = coords.len() as NodeId;
        coords.push(c);
        map.insert(k, id);
        id
    }

    for (ea, &(lo_a, hi_a)) in bboxes_a.iter().enumerate() {
        let ns_a = mesh_a.elem_nodes(ea as ElemId);
        let tri_a = [mesh_a.coords_of(ns_a[0]), mesh_a.coords_of(ns_a[1]), mesh_a.coords_of(ns_a[2])];

        for (eb, &(lo_b, hi_b)) in bboxes_b.iter().enumerate() {
            // Quick AABB overlap test
            if hi_a[0] <= lo_b[0] || hi_b[0] <= lo_a[0] { continue; }
            if hi_a[1] <= lo_b[1] || hi_b[1] <= lo_a[1] { continue; }

            let ns_b = mesh_b.elem_nodes(eb as ElemId);
            let tri_b = [mesh_b.coords_of(ns_b[0]), mesh_b.coords_of(ns_b[1]), mesh_b.coords_of(ns_b[2])];

            // Compute intersection polygon via Sutherland-Hodgman clipping
            if let Some(poly) = clip_triangle(&tri_a, &tri_b) {
                // Triangulate the convex intersection polygon
                let tris = triangulate_convex(&poly, &mut all_coords, &mut coord_map);
                for tri_nodes in tris {
                    sup_elems.push(SupermeshElement {
                        nodes: tri_nodes,
                        src_elem: ea as ElemId,
                        tgt_elem: eb as ElemId,
                    });
                }
            }
        }
    }

    // Build Mesh from collected nodes and elements
    let n_sup = sup_elems.len();
    let mut conn = Vec::with_capacity(n_sup * 3);
    let mut tags = Vec::with_capacity(n_sup);
    for se in &sup_elems {
        conn.push(se.nodes[0]); conn.push(se.nodes[1]); conn.push(se.nodes[2]);
        tags.push(0);
    }
    let coords: Vec<f64> = all_coords.iter().flat_map(|c| { vec![c[0], c[1]] }).collect();

    let mesh = Mesh {
        coords,
        conn,
        elem_tags: tags,
        elem_type: ElementType::Tri3,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
    };

    (mesh, sup_elems)
}

/// Build AABBs for all elements in a Tri3 mesh.
fn build_bboxes(mesh: &Mesh<2>) -> Vec<([f64; 2], [f64; 2])> {
    let mut bboxes = Vec::with_capacity(mesh.n_elems());
    for e in 0..mesh.n_elems() as ElemId {
        let ns = mesh.elem_nodes(e);
        let a = mesh.coords_of(ns[0]);
        let b = mesh.coords_of(ns[1]);
        let c = mesh.coords_of(ns[2]);
        let lo = [a[0].min(b[0]).min(c[0]), a[1].min(b[1]).min(c[1])];
        let hi = [a[0].max(b[0]).max(c[0]), a[1].max(b[1]).max(c[1])];
        bboxes.push((lo, hi));
    }
    bboxes
}

/// Clip triangle A against the half-planes of triangle B using
/// Sutherland-Hodgman algorithm. Returns intersection polygon vertices (0-6).
fn clip_triangle(tri_a: &[[f64; 2]; 3], tri_b: &[[f64; 2]; 3]) -> Option<Vec<[f64; 2]>> {
    let mut input = tri_a.to_vec();

    // Clip against each edge of triangle B
    for i in 0..3 {
        if input.is_empty() { return None; }
        let j = (i + 1) % 3;
        let (p1, p2) = (tri_b[i], tri_b[j]);
        // Edge normal (inward-facing for CCW triangle)
        let nx = p1[1] - p2[1];
        let ny = p2[0] - p1[0];

        let mut output = Vec::new();
        let n = input.len();
        for k in 0..n {
            let curr = input[k];
            let prev = input[(k + n - 1) % n];

            let d_curr = (curr[0] - p1[0]) * nx + (curr[1] - p1[1]) * ny;
            let d_prev = (prev[0] - p1[0]) * nx + (prev[1] - p1[1]) * ny;

            if d_curr >= 0.0 {
                // Current vertex is inside
                if d_prev < 0.0 {
                    // Entering the half-plane: add edge intersection
                    let t = d_prev / (d_prev - d_curr);
                    output.push([
                        prev[0] + t * (curr[0] - prev[0]),
                        prev[1] + t * (curr[1] - prev[1]),
                    ]);
                }
                output.push(curr);
            } else if d_prev >= 0.0 {
                // Leaving the half-plane: add edge intersection
                let t = d_prev / (d_prev - d_curr);
                output.push([
                    prev[0] + t * (curr[0] - prev[0]),
                    prev[1] + t * (curr[1] - prev[1]),
                ]);
            }
            // else: both outside, nothing to add
        }
        input = output;
    }

    if input.len() < 3 { None }
    else { Some(input) }
}

/// Triangulate a convex polygon using ear clipping (fan triangulation).
fn triangulate_convex(
    poly: &[[f64; 2]],
    all_coords: &mut Vec<[f64; 2]>,
    coord_map: &mut HashMap<u64, NodeId>,
) -> Vec<[NodeId; 3]> {
    if poly.len() < 3 { return vec![]; }
    let mut tris = Vec::with_capacity(poly.len() - 2);
    let v0 = add_node_inner(poly[0], all_coords, coord_map);
    for i in 1..(poly.len() - 1) {
        let vi = add_node_inner(poly[i], all_coords, coord_map);
        let vj = add_node_inner(poly[i + 1], all_coords, coord_map);
        tris.push([v0, vi, vj]);
    }
    tris
}

fn add_node_inner(
    c: [f64; 2],
    all_coords: &mut Vec<[f64; 2]>,
    coord_map: &mut HashMap<u64, NodeId>,
) -> NodeId {
    let x = (c[0] * 1e10).round() as i64;
    let y = (c[1] * 1e10).round() as i64;
    let k = (x.wrapping_mul(314159) ^ y) as u64;
    if let Some(&n) = coord_map.get(&k) { return n; }
    let id = all_coords.len() as NodeId;
    all_coords.push(c);
    coord_map.insert(k, id);
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    #[test]
    fn supermesh_inner_square() {
        // Mesh A: full unit square
        let mesh_a = Mesh::<2>::unit_square_tri(2);
        // Mesh B: inner [0.25,0.75]²
        let coords_b = vec![0.25,0.25, 0.75,0.25, 0.75,0.75, 0.25,0.75];
        let conn_b = vec![0u32,1,2, 0,2,3];
        let tags_b = vec![1i32; 2];
        let mesh_b = Mesh {
            coords: coords_b, conn: conn_b, elem_tags: tags_b,
            elem_type: ElementType::Tri3, face_conn: vec![], face_tags: vec![],
            face_type: ElementType::Line2, elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None, face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
        };

        let (super_mesh, elems) = build_supermesh(&mesh_a, &mesh_b);
        assert!(super_mesh.n_elems() > 0, "supermesh should have elements");
        // All nodes within [0.25,0.75]²
        for n in 0..super_mesh.n_nodes() as NodeId {
            let c = super_mesh.coords_of(n);
            assert!(c[0] >= 0.24 && c[0] <= 0.76, "node {} x={} out of range", n, c[0]);
            assert!(c[1] >= 0.24 && c[1] <= 0.76, "node {} y={} out of range", n, c[1]);
        }
        // Every supermesh element belongs to a unique (src,tgt) pair
        for se in &elems {
            assert!(se.src_elem < mesh_a.n_elems() as ElemId);
            assert!(se.tgt_elem < mesh_b.n_elems() as ElemId);
        }
    }

    #[test]
    fn supermesh_same_mesh() {
        let mesh_a = Mesh::<2>::unit_square_tri(4);
        let (super_mesh, _elems) = build_supermesh(&mesh_a, &mesh_a);
        // Same mesh intersection should produce at least as many elements
        assert!(super_mesh.n_elems() >= mesh_a.n_elems(),
            "same mesh intersection: {} vs {} elements",
            super_mesh.n_elems(), mesh_a.n_elems());
        // All nodes within [0,1]×[0,1]
        for n in 0..super_mesh.n_nodes() as NodeId {
            let c = super_mesh.coords_of(n);
            assert!(c[0] >= -1e-12 && c[0] <= 1.0 + 1e-12, "node {} x={} out", n, c[0]);
            assert!(c[1] >= -1e-12 && c[1] <= 1.0 + 1e-12, "node {} y={} out", n, c[1]);
        }
    }

    #[test]
    fn clip_triangle_matches() {
        // Two identical triangles → full triangle
        let tri = [[0.0,0.0],[1.0,0.0],[0.0,1.0]];
        let result = clip_triangle(&tri, &tri);
        assert!(result.is_some());
        assert_eq!(result.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn clip_triangle_disjoint() {
        let a = [[0.0,0.0],[1.0,0.0],[0.0,1.0]];
        let b = [[2.0,2.0],[3.0,2.0],[2.0,3.0]];
        assert!(clip_triangle(&a, &b).is_none());
    }
}
