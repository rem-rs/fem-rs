//! Mesh extrusion: generate 3-D meshes from 2-D cross-section meshes.
//!
//! Supports:
//! - Tri3 → Prism6 extrusion
//! - Tri3 → Tet4 extrusion (each tri split into 3 tets per layer)
//! - Quad4 → Hex8 extrusion

use fem_core::NodeId;
use crate::element_type::ElementType;
use crate::simplex::Mesh;

/// Extrude a 2-D Tri3 mesh into a 3-D Prism6 mesh.
///
/// Each triangle `{a, b, c}` in layer `k` becomes prism
/// `{a_k, b_k, c_k, a_{k+1}, b_{k+1}, c_{k+1}}`.
///
/// Boundary faces: bottom (tag=1), top (tag=2), sides (tag=3).
/// Each triangle extruded into `n_layers` prisms.
pub fn extrude_tri3_to_prisms(
    mesh: &Mesh<2>,
    n_layers: usize,
    height: f64,
) -> Mesh<3> {
    assert_eq!(mesh.elem_type, ElementType::Tri3,
        "extrude_tri3_to_prisms: requires Tri3 mesh");
    assert!(n_layers > 0);

    let nn2 = mesh.n_nodes();
    let ne2 = mesh.n_elems();
    let n_layers_f = n_layers as f64;

    // Coordinates: nn2 nodes per layer × (n_layers + 1) layers
    let n_nodes_3d = nn2 * (n_layers + 1);
    let mut coords_3d = Vec::with_capacity(n_nodes_3d * 3);

    for layer in 0..=n_layers {
        let z = layer as f64 * height / n_layers_f;
        for i in 0..nn2 {
            let coord = mesh.coords_of(i as NodeId);
            coords_3d.push(coord[0]);
            coords_3d.push(coord[1]);
            coords_3d.push(z);
        }
    }

    // Connectivity: each 2D triangle → n_layers prisms
    let n_prism_nodes = 6;
    let n_elems_3d = ne2 * n_layers;
    let mut conn_3d = Vec::with_capacity(n_elems_3d * n_prism_nodes);
    let mut elem_tags_3d = Vec::with_capacity(n_elems_3d);

    for layer in 0..n_layers {
        for e in 0..ne2 as u32 {
            let nodes = mesh.elem_nodes(e);
            // Prism6 nodes: bottom(a,b,c) + top(a',b',c')
            let b0 = nodes[0] + layer as u32 * nn2 as u32;
            let b1 = nodes[1] + layer as u32 * nn2 as u32;
            let b2 = nodes[2] + layer as u32 * nn2 as u32;
            let t0 = b0 + nn2 as u32;
            let t1 = b1 + nn2 as u32;
            let t2 = b2 + nn2 as u32;
            conn_3d.extend_from_slice(&[b0, b1, b2, t0, t1, t2]);
            elem_tags_3d.push(if e == 0 && layer == 0 { 1 } else { 0 });
        }
    }

    // Boundary faces
    let n2_bdy = mesh.n_faces();
    // 3D faces: bottom (ne2 tri faces, tag=1), top (ne2 tri faces, tag=2),
    // sides: 2 per extruded 2D face edge per layer
    let n_faces = ne2 * 2 + n2_bdy * 2 * n_layers;
    let mut face_conn = Vec::with_capacity(n_faces * 3);
    let mut face_tags = Vec::with_capacity(n_faces);

    // Bottom faces (layer=0, tag=1) — reverse orientation for outward normal
    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        face_conn.push(nodes[2]);
        face_conn.push(nodes[1]);
        face_conn.push(nodes[0]);
        face_tags.push(1);
    }

    // Top faces (layer=n_layers, tag=2)
    let top_offset = n_layers as u32 * nn2 as u32;
    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        face_conn.push(nodes[0] + top_offset);
        face_conn.push(nodes[1] + top_offset);
        face_conn.push(nodes[2] + top_offset);
        face_tags.push(2);
    }

    // Side faces (tag=3): for each 2D boundary edge, extrude into 2 quad faces per layer
    // Each 2D boundary face is an edge {a,b}. Extrude → 2 tri faces {a_k, b_k, b_{k+1}} and {a_k, b_{k+1}, a_{k+1}}
    let mut bdy_edges: Vec<(u32, u32)> = Vec::new();
    for b in 0..n2_bdy as u32 {
        let fnodes = mesh.bface_nodes(b);
        if fnodes.len() == 2 {
            bdy_edges.push((fnodes[0], fnodes[1]));
        }
    }

    // Extrude boundary edges into quad faces (2 tri faces per quad)
    for (ea, eb) in &bdy_edges {
        for layer in 0..n_layers {
            let a_bot = *ea + layer as u32 * nn2 as u32;
            let b_bot = *eb + layer as u32 * nn2 as u32;
            let a_top = a_bot + nn2 as u32;
            let b_top = b_bot + nn2 as u32;
            // Tri 1: (a_bot, b_bot, b_top)
            face_conn.push(a_bot);
            face_conn.push(b_bot);
            face_conn.push(b_top);
            face_tags.push(3);
            // Tri 2: (a_bot, b_top, a_top)
            face_conn.push(a_bot);
            face_conn.push(b_top);
            face_conn.push(a_top);
            face_tags.push(3);
        }
    }

    Mesh {
        coords: coords_3d,
        conn: conn_3d,
        elem_tags: elem_tags_3d,
        elem_type: ElementType::Prism6,
        face_conn,
        face_tags: face_tags.into_iter().map(|t| t as crate::boundary::BoundaryTag).collect(),
        face_type: ElementType::Tri3,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
    }
}

/// Extrude a 2-D Quad4 mesh into a 3-D Hex8 mesh.
///
/// Each quad `{a, b, c, d}` in layer `k` becomes hex
/// `{a_k, b_k, c_k, d_k, a_{k+1}, b_{k+1}, c_{k+1}, d_{k+1}}`.
pub fn extrude_quad4_to_hex8(
    mesh: &Mesh<2>,
    n_layers: usize,
    height: f64,
) -> Mesh<3> {
    assert_eq!(mesh.elem_type, ElementType::Quad4,
        "extrude_quad4_to_hex8: requires Quad4 mesh");
    assert!(n_layers > 0);

    let nn2 = mesh.n_nodes();
    let ne2 = mesh.n_elems();

    let n_nodes_3d = nn2 * (n_layers + 1);
    let mut coords_3d = Vec::with_capacity(n_nodes_3d * 3);

    for layer in 0..=n_layers {
        let z = layer as f64 * height / n_layers as f64;
        for i in 0..nn2 {
            let coord = mesh.coords_of(i as NodeId);
            coords_3d.push(coord[0]);
            coords_3d.push(coord[1]);
            coords_3d.push(z);
        }
    }

    let n_hex_nodes = 8;
    let n_elems_3d = ne2 * n_layers;
    let mut conn_3d = Vec::with_capacity(n_elems_3d * n_hex_nodes);
    let mut elem_tags_3d = Vec::with_capacity(n_elems_3d);

    for layer in 0..n_layers {
        for e in 0..ne2 as u32 {
            let nodes = mesh.elem_nodes(e);
            let b0 = nodes[0] + layer as u32 * nn2 as u32;
            let b1 = nodes[1] + layer as u32 * nn2 as u32;
            let b2 = nodes[2] + layer as u32 * nn2 as u32;
            let b3 = nodes[3] + layer as u32 * nn2 as u32;
            let t0 = b0 + nn2 as u32;
            let t1 = b1 + nn2 as u32;
            let t2 = b2 + nn2 as u32;
            let t3 = b3 + nn2 as u32;
            conn_3d.extend_from_slice(&[b0, b1, b2, b3, t0, t1, t2, t3]);
            elem_tags_3d.push(0);
        }
    }

    let n2_bdy = mesh.n_faces();
    let n_faces = ne2 * 2 + n2_bdy * 2 * n_layers;
    let mut face_conn = Vec::with_capacity(n_faces * 4);
    let mut face_tags: Vec<i32> = Vec::with_capacity(n_faces);

    // Bottom quad faces (tag=1)
    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        face_conn.extend_from_slice(&[nodes[3], nodes[2], nodes[1], nodes[0]]);
        face_tags.push(1);
    }

    // Top quad faces (tag=2)
    let top_offset = n_layers as u32 * nn2 as u32;
    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        face_conn.push(nodes[0] + top_offset);
        face_conn.push(nodes[1] + top_offset);
        face_conn.push(nodes[2] + top_offset);
        face_conn.push(nodes[3] + top_offset);
        face_tags.push(2);
    }

    // Side quad faces (tag=3)
    let mut bdy_edges: Vec<(u32, u32)> = Vec::new();
    for b in 0..n2_bdy as u32 {
        let fnodes = mesh.bface_nodes(b);
        if fnodes.len() == 2 {
            bdy_edges.push((fnodes[0], fnodes[1]));
        }
    }
    for (ea, eb) in &bdy_edges {
        for layer in 0..n_layers {
            let a_bot = *ea + layer as u32 * nn2 as u32;
            let b_bot = *eb + layer as u32 * nn2 as u32;
            let a_top = a_bot + nn2 as u32;
            let b_top = b_bot + nn2 as u32;
            face_conn.extend_from_slice(&[a_bot, b_bot, b_top, a_top]);
            face_tags.push(3);
        }
    }

    Mesh {
        coords: coords_3d,
        conn: conn_3d,
        elem_tags: elem_tags_3d,
        elem_type: ElementType::Hex8,
        face_conn,
        face_tags: face_tags.into_iter().map(|t| t as crate::boundary::BoundaryTag).collect(),
        face_type: ElementType::Quad4,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Mesh;

    #[test]
    fn extrude_tri_prism_counts() {
        let m2 = Mesh::<2>::unit_square_tri(2);
        let m3 = extrude_tri3_to_prisms(&m2, 3, 1.0);
        assert_eq!(m3.n_nodes(), m2.n_nodes() * 4); // n_layers+1 = 4
        assert_eq!(m3.n_elems(), m2.n_elems() * 3); // 3 layers
        assert_eq!(m3.elem_type, ElementType::Prism6);
    }

    #[test]
    fn extrude_quad_hex_counts() {
        let m2 = Mesh::<2>::unit_square_quad(2);
        let m3 = extrude_quad4_to_hex8(&m2, 2, 2.0);
        assert_eq!(m3.n_nodes(), m2.n_nodes() * 3);
        assert_eq!(m3.n_elems(), m2.n_elems() * 2);
        assert_eq!(m3.elem_type, ElementType::Hex8);
    }

    #[test]
    fn extrude_prism_geometry() {
        let m2 = Mesh::<2>::unit_square_tri(1);
        let m3 = extrude_tri3_to_prisms(&m2, 1, 5.0);
        // Top layer nodes at z = 5.0
        let top_node = m2.n_nodes() as u32;
        let top_coord = m3.coords_of(top_node);
        assert!((top_coord[2] - 5.0).abs() < 1e-12, "z should be 5, got {}", top_coord[2]);
    }
}
