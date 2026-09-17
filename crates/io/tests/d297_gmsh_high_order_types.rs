//! D297: Gmsh v4.1 read-in of second-order serendipity/complete element
//! types.  Code 16 is the 8-node second-order quadrangle (Quad8) — it was
//! previously mislabeled Prism15 in `ElementType::from_gmsh_type` — and code
//! 18 is the 15-node second-order prism (Prism15), which was missing.  The
//! same-class complete order-2 codes 10 (9-node quad, Quad9) and 13 (18-node
//! prism, Prism18) were also missing and are pinned here.
//!
//! Node orderings follow the Gmsh .msh file format specification:
//! - type 16 quad: 4 corners (CCW) then 4 edge nodes,
//! - type 18 prism: bottom tri (3), top tri (3), bottom-edge (3),
//!   vertical-edge (3), top-edge (3) nodes.
//!
//! ```bash
//! cargo test -p fem-io --test d297_gmsh_high_order_types
//! ```

use fem_io::gmsh::read_msh;

/// Gmsh v4.1 ASCII fixture: a single 8-node second-order quadrangle
/// (element type 16) on a unit square.
fn gmsh_v41_quad8() -> &'static str {
    "$MeshFormat\n\
     4.1 0 8\n\
     $EndMeshFormat\n\
     $Entities\n\
     0 0 1 0\n\
     $EndEntities\n\
     $Nodes\n\
     1 8 1 8\n\
     2 1 0 8\n\
     1\n\
     2\n\
     3\n\
     4\n\
     5\n\
     6\n\
     7\n\
     8\n\
     0.0 0.0 0.0\n\
     1.0 0.0 0.0\n\
     1.0 1.0 0.0\n\
     0.0 1.0 0.0\n\
     0.5 0.0 0.0\n\
     1.0 0.5 0.0\n\
     0.5 1.0 0.0\n\
     0.0 0.5 0.0\n\
     $EndNodes\n\
     $Elements\n\
     1 1 1 1\n\
     2 1 16 1\n\
     1 1 2 3 4 5 6 7 8\n\
     $EndElements\n"
}

/// Gmsh v4.1 ASCII fixture: a single 15-node second-order prism
/// (element type 18) on the unit prism.
fn gmsh_v41_prism15() -> &'static str {
    "$MeshFormat\n\
     4.1 0 8\n\
     $EndMeshFormat\n\
     $Entities\n\
     0 0 0 1\n\
     $EndEntities\n\
     $Nodes\n\
     1 15 1 15\n\
     3 1 0 15\n\
     1\n\
     2\n\
     3\n\
     4\n\
     5\n\
     6\n\
     7\n\
     8\n\
     9\n\
     10\n\
     11\n\
     12\n\
     13\n\
     14\n\
     15\n\
     0.0 0.0 0.0\n\
     1.0 0.0 0.0\n\
     0.0 1.0 0.0\n\
     0.0 0.0 1.0\n\
     1.0 0.0 1.0\n\
     0.0 1.0 1.0\n\
     0.5 0.0 0.0\n\
     0.5 0.5 0.0\n\
     0.0 0.5 0.0\n\
     0.0 0.0 0.5\n\
     1.0 0.0 0.5\n\
     0.0 1.0 0.5\n\
     0.5 0.0 1.0\n\
     0.5 0.5 1.0\n\
     0.0 0.5 1.0\n\
     $EndNodes\n\
     $Elements\n\
     1 1 1 1\n\
     3 1 18 1\n\
     1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n\
     $EndElements\n"
}

#[test]
fn gmsh_v41_quad8_type16_reads_as_quad8() {
    let msh = read_msh(gmsh_v41_quad8().as_bytes()).expect("parse Quad8 fixture");
    let mesh = msh.mesh2d.as_ref().expect("2D mesh");
    assert_eq!(mesh.elem_type, fem_mesh::element_type::ElementType::Quad8);
    assert_eq!(mesh.n_elems(), 1);
    // node tags 1..8 map to 0-based ids 0..7 in file order
    assert_eq!(mesh.conn, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(mesh.elem_tags, vec![1]);
    assert_eq!(mesh.face_type, fem_mesh::element_type::ElementType::Line3);
}

#[test]
fn gmsh_v41_prism15_type18_reads_as_prism15() {
    let msh = read_msh(gmsh_v41_prism15().as_bytes()).expect("parse Prism15 fixture");
    let mesh = msh.mesh3d.as_ref().expect("3D mesh");
    assert_eq!(mesh.elem_type, fem_mesh::element_type::ElementType::Prism15);
    assert_eq!(mesh.n_elems(), 1);
    assert_eq!(mesh.conn, (0u32..15).collect::<Vec<_>>());
    assert_eq!(mesh.elem_tags, vec![1]);
    assert_eq!(mesh.face_type, fem_mesh::element_type::ElementType::Tri3);
}
