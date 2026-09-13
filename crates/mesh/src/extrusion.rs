//! Mesh extrusion: generate 3-D meshes from 2-D cross-section meshes.
//!
//! Supports Tri3 → Prism6 and Quad4 → Hex8 extrusion.
//!
//! The node numbering, element attributes, boundary attributes and boundary
//! ordering follow MFEM `Mesh::Extrude2D` (`mesh/mesh.cpp`), so an extruded
//! `Quad4 → Hex8` mesh is section-by-section identical to the C++ output:
//!
//! * vertices: source vertex `i` lifted to layer `j` is node `i * nvz + j`
//!   (`nvz = n_layers + 1`) — *point-major*, not layer-major;
//! * elements: `for e { for j { … } }`, every layer inheriting the **source
//!   element attribute** (MFEM passes `elem->GetAttribute()` to each `AddHex`);
//! * boundary faces, in MFEM's creation order: first the **side** faces (one per
//!   source boundary element per layer, keeping the **source boundary
//!   attribute**), then, per element, its bottom then its top face with
//!   attribute `nba + elem attr` where `nba = max(source boundary attribute)`.
//!
//! `Tri3 → Prism6` boundaries are *mixed* (quad sides + triangular caps) and
//! `Quad4 → Hex8` boundaries are uniform quads, so both paths now write the
//! same face geometry MFEM's `AddBdrQuad` / `AddBdrTriangle` calls produce.

use fem_core::NodeId;
use crate::element_type::ElementType;
use crate::simplex::Mesh;

/// Source-mesh boundary attribute maximum (`Mesh::bdr_attributes.Max()`); `0`
/// when the source mesh has no boundary faces (MFEM's default for an empty
/// `bdr_attributes` array).
fn source_nba(mesh: &Mesh<2>) -> i32 {
    mesh.face_tags.iter().copied().max().unwrap_or(0)
}

/// Extrude a 2-D Tri3 mesh into a 3-D Prism6 mesh.
///
/// Each triangle `{a, b, c}` in layer `k` becomes prism
/// `{a_k, b_k, c_k, a_{k+1}, b_{k+1}, c_{k+1}}`, so each triangle is extruded
/// into `n_layers` prisms.
///
/// Boundary faces: first the sides (one quad per source boundary edge per
/// layer, inheriting the source boundary attribute `1..nba`), then per element
/// its bottom and top triangle with attribute `nba + elem attr`.  The face
/// section is therefore *mixed* (`face_types` / `face_offsets` are set), as in
/// MFEM's `Mesh::Extrude2D`.
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
    let nvz = n_layers + 1;

    // Node `v * nvz + j` is source vertex `v` lifted to layer `j` (MFEM adds
    // the vertices in source-vertex-major order).
    let nid = |v: NodeId, j: usize| -> u32 { v * nvz as u32 + j as u32 };

    let n_nodes_3d = nn2 * nvz;
    let mut coords_3d = Vec::with_capacity(n_nodes_3d * 3);
    for i in 0..nn2 {
        let coord = mesh.coords_of(i as NodeId);
        for j in 0..nvz {
            coords_3d.push(coord[0]);
            coords_3d.push(coord[1]);
            coords_3d.push(height * (j as f64 / n_layers as f64));
        }
    }

    // Connectivity: each 2D triangle → n_layers prisms, element-major (MFEM:
    // `for (i = 0; i < GetNE(); i++) for (j = 0; j < nz; j++) AddWedge`).
    let n_prism_nodes = 6;
    let n_elems_3d = ne2 * n_layers;
    let mut conn_3d = Vec::with_capacity(n_elems_3d * n_prism_nodes);
    let mut elem_tags_3d = Vec::with_capacity(n_elems_3d);

    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        let attr = mesh.elem_tags[e as usize];
        for layer in 0..n_layers {
            conn_3d.extend_from_slice(&[
                nid(nodes[0], layer), nid(nodes[1], layer), nid(nodes[2], layer),
                nid(nodes[0], layer + 1), nid(nodes[1], layer + 1), nid(nodes[2], layer + 1),
            ]);
            elem_tags_3d.push(attr);
        }
    }

    let n2_bdy = mesh.n_faces();
    let nba = source_nba(mesh);
    let n_side = n2_bdy * n_layers;
    let n_faces = n_side + ne2 * 2;
    let mut face_conn: Vec<NodeId> = Vec::with_capacity(n_side * 4 + ne2 * 2 * 3);
    let mut face_tags: Vec<i32> = Vec::with_capacity(n_faces);

    // A prism's lateral face is a quad, so the boundary is *mixed* (quad sides
    // + triangular caps) exactly as MFEM's `AddBdrQuad` / `AddBdrTriangle`
    // produce it — splitting the sides into triangles (the previous behaviour)
    // made the file unloadable by MFEM (`STable3D` cannot find a triangular
    // face of a wedge).  `face_type` is the mixed marker `Line2`, matching what
    // `fem_io::mfem` sets when it reads a mesh with mixed boundary faces.
    let mut face_types: Vec<ElementType> = Vec::with_capacity(n_faces);
    let mut face_offsets: Vec<usize> = Vec::with_capacity(n_faces + 1);
    face_offsets.push(0);

    // (1) Side faces from the 2D boundary, source boundary attribute kept.
    // Each source boundary edge {a,b} extrudes to the quad
    // {a_j, b_j, b_{j+1}, a_{j+1}} (MFEM `AddBdrQuad(qv, attr)`).
    for b in 0..n2_bdy as u32 {
        let fnodes = mesh.bface_nodes(b);
        if fnodes.len() != 2 { continue; }
        let (ea, eb) = (fnodes[0], fnodes[1]);
        let attr = mesh.face_tags[b as usize];
        for layer in 0..n_layers {
            face_conn.extend_from_slice(&[
                nid(ea, layer), nid(eb, layer), nid(eb, layer + 1), nid(ea, layer + 1),
            ]);
            face_tags.push(attr);
            face_types.push(ElementType::Quad4);
            face_offsets.push(face_conn.len());
        }
    }

    // (2) Bottom (layer 0) + top (layer nz) per element, bottom first
    // (MFEM's `TRIANGLE` tail loop with attr = nba + elem attr).
    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        let attr = nba + mesh.elem_tags[e as usize];
        face_conn.push(nid(nodes[0], 0));
        face_conn.push(nid(nodes[2], 0));
        face_conn.push(nid(nodes[1], 0));
        face_tags.push(attr);
        face_types.push(ElementType::Tri3);
        face_offsets.push(face_conn.len());
        face_conn.push(nid(nodes[0], n_layers));
        face_conn.push(nid(nodes[1], n_layers));
        face_conn.push(nid(nodes[2], n_layers));
        face_tags.push(attr);
        face_types.push(ElementType::Tri3);
        face_offsets.push(face_conn.len());
    }

    Mesh {
        coords: coords_3d,
        conn: conn_3d,
        elem_tags: elem_tags_3d,
        elem_type: ElementType::Prism6,
        face_conn,
        face_tags,
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: Some(face_types),
        face_offsets: Some(face_offsets),
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None, nc_vertex_view: None,
    vertex_parents: vec![],
    }
}

/// Extrude a 2-D Quad4 mesh into a 3-D Hex8 mesh.
///
/// Each quad `{a, b, c, d}` in layer `k` becomes hex
/// `{a_k, b_k, c_k, d_k, a_{k+1}, b_{k+1}, c_{k+1}, d_{k+1}}`.
///
/// Boundary faces: first the sides (each source boundary edge per layer,
/// inheriting the source boundary attribute `1..nba`), then per element its
/// bottom and top quad with attribute `nba + elem attr`.
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
    let nvz = n_layers + 1;

    // Node `v * nvz + j` is source vertex `v` lifted to layer `j`.
    let nid = |v: NodeId, j: usize| -> u32 { v * nvz as u32 + j as u32 };

    let n_nodes_3d = nn2 * nvz;
    let mut coords_3d = Vec::with_capacity(n_nodes_3d * 3);
    for i in 0..nn2 {
        let coord = mesh.coords_of(i as NodeId);
        for j in 0..nvz {
            coords_3d.push(coord[0]);
            coords_3d.push(coord[1]);
            coords_3d.push(height * (j as f64 / n_layers as f64));
        }
    }

    let n_hex_nodes = 8;
    let n_elems_3d = ne2 * n_layers;
    let mut conn_3d = Vec::with_capacity(n_elems_3d * n_hex_nodes);
    let mut elem_tags_3d = Vec::with_capacity(n_elems_3d);

    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        let attr = mesh.elem_tags[e as usize];
        for layer in 0..n_layers {
            conn_3d.extend_from_slice(&[
                nid(nodes[0], layer), nid(nodes[1], layer),
                nid(nodes[2], layer), nid(nodes[3], layer),
                nid(nodes[0], layer + 1), nid(nodes[1], layer + 1),
                nid(nodes[2], layer + 1), nid(nodes[3], layer + 1),
            ]);
            elem_tags_3d.push(attr);
        }
    }

    let n2_bdy = mesh.n_faces();
    let nba = source_nba(mesh);
    let n_faces = n2_bdy * n_layers + ne2 * 2;
    let mut face_conn: Vec<NodeId> = Vec::with_capacity(n_faces * 4);
    let mut face_tags: Vec<i32> = Vec::with_capacity(n_faces);

    // (1) Side faces from the 2D boundary: quad {a_j, b_j, b_{j+1}, a_{j+1}}
    // with the source boundary attribute (MFEM `AddBdrQuad(qv, attr)`).
    for b in 0..n2_bdy as u32 {
        let fnodes = mesh.bface_nodes(b);
        if fnodes.len() != 2 { continue; }
        let (ea, eb) = (fnodes[0], fnodes[1]);
        let attr = mesh.face_tags[b as usize];
        for layer in 0..n_layers {
            face_conn.extend_from_slice(&[
                nid(ea, layer), nid(eb, layer), nid(eb, layer + 1), nid(ea, layer + 1),
            ]);
            face_tags.push(attr);
        }
    }

    // (2) Bottom + top per element (MFEM's `SQUARE` tail loop): the bottom quad
    // is vertex-reversed so its normal points out of the extrusion, both carry
    // attribute `nba + elem attr`.
    for e in 0..ne2 as u32 {
        let nodes = mesh.elem_nodes(e);
        let attr = nba + mesh.elem_tags[e as usize];
        face_conn.extend_from_slice(&[
            nid(nodes[0], 0), nid(nodes[3], 0), nid(nodes[2], 0), nid(nodes[1], 0),
        ]);
        face_tags.push(attr);
        face_conn.extend_from_slice(&[
            nid(nodes[0], n_layers), nid(nodes[1], n_layers),
            nid(nodes[2], n_layers), nid(nodes[3], n_layers),
        ]);
        face_tags.push(attr);
    }

    Mesh {
        coords: coords_3d,
        conn: conn_3d,
        elem_tags: elem_tags_3d,
        elem_type: ElementType::Hex8,
        face_conn,
        face_tags,
        face_type: ElementType::Quad4,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None, nc_vertex_view: None,
    vertex_parents: vec![],
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
        // Mixed boundary: one quad per source boundary edge per layer, plus a
        // bottom and a top triangle per element (MFEM `Mesh::Extrude2D`).
        assert_eq!(m3.n_faces(), m2.n_faces() * 3 + m2.n_elems() * 2);
        assert_eq!(m3.face_conn.len(), m2.n_faces() * 3 * 4 + m2.n_elems() * 2 * 3);
        assert!(m3.face_types.as_ref().is_some_and(|t| t.len() == m3.n_faces()));
        assert_eq!(m3.face_offsets.as_ref().map(|o| o.len()), Some(m3.n_faces() + 1));
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
        // Point-major numbering (MFEM `Mesh::Extrude2D`): source vertex `i` at
        // layer `j` is node `i * (n_layers + 1) + j`, so the top of source
        // vertex 0 is node 1.
        let top_node = 1u32;
        let top_coord = m3.coords_of(top_node);
        assert!((top_coord[2] - 5.0).abs() < 1e-12, "z should be 5, got {}", top_coord[2]);
    }

    /// MFEM `Mesh::Extrude2D` puts the source element attribute on every
    /// extruded layer (`AddHex(hv, attr)`); the old code hard-coded `0`, which
    /// made MFEM warn `Non-positive attributes in the domain!` on the extruded
    /// mesh.
    #[test]
    fn extrude_keeps_element_attributes() {
        let mut m2 = Mesh::<2>::unit_square_quad(2);
        for (e, t) in m2.elem_tags.iter_mut().enumerate() {
            *t = (e % 3 + 1) as i32;
        }
        let m3 = extrude_quad4_to_hex8(&m2, 2, 1.0);
        for e in 0..m2.n_elems() {
            // element-major: layers of source element `e` are 2e and 2e+1
            assert_eq!(m3.elem_tags[2 * e], m2.elem_tags[e]);
            assert_eq!(m3.elem_tags[2 * e + 1], m2.elem_tags[e]);
        }
    }

    /// Boundary attribute scheme of `Mesh::Extrude2D`: sides keep the source
    /// boundary attribute (`1..nba`), bottom/top get `nba + elem attr`.  The
    /// original code used the fixed scheme bottom=1 / top=2 / sides=3.
    #[test]
    fn extrude_boundary_attributes_match_mfem() {
        let m2 = Mesh::<2>::unit_square_quad(2);
        let nba = source_nba(&m2);
        assert!(nba >= 1, "the unit square must have boundary attributes");
        // Give the single element a material id distinct from 1.
        let mut m2 = m2;
        for t in m2.elem_tags.iter_mut() { *t = 7; }
        let m3 = extrude_quad4_to_hex8(&m2, 2, 1.0);
        let tags = &m3.face_tags;
        // Sides first: `n_layers` faces per source boundary edge, in source
        // boundary order, each keeping the source boundary attribute.
        let n_side = m2.n_faces() * 2;
        let expected: Vec<i32> = m2
            .face_tags
            .iter()
            .flat_map(|&t| std::iter::repeat(t).take(2))
            .collect();
        assert_eq!(tags[..n_side], expected[..]);
        for &t in &tags[n_side..] {
            assert_eq!(t, nba + 7, "bottom/top attribute must be nba + elem attr");
        }
    }

    /// Node numbering must be point-major (`i * nvz + j`), matching MFEM.
    #[test]
    fn extrude_node_numbering_is_point_major() {
        let m2 = Mesh::<2>::unit_square_quad(2);
        let m3 = extrude_quad4_to_hex8(&m2, 2, 3.0);
        let nvz = 3;
        for i in 0..m2.n_nodes() {
            let c2 = m2.coords_of(i as NodeId);
            for j in 0..nvz {
                let c3 = m3.coords_of((i * nvz + j) as NodeId);
                assert_eq!((c3[0], c3[1]), (c2[0], c2[1]));
                assert!((c3[2] - 3.0 * j as f64 / 2.0).abs() < 1e-12);
            }
        }
    }
}
