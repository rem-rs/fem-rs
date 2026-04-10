//! Integration tests for the fem-mesh crate.

use fem_mesh::{
    element_type::ElementType,
    topology::MeshTopology,
    SimplexMesh,
};

// ─── Unit-square mesh topology ────────────────────────────────────────────────

#[test]
fn unit_square_tri_node_count() {
    // unit_square_tri(n) produces an (n+1)×(n+1) grid of nodes.
    for n in [4usize, 8, 16] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        assert_eq!(
            mesh.n_nodes(),
            (n + 1) * (n + 1),
            "n={n}: expected {} nodes, got {}",
            (n + 1) * (n + 1),
            mesh.n_nodes()
        );
    }
}

#[test]
fn unit_square_tri_element_count() {
    // unit_square_tri(n) creates 2*n*n triangles (each quad split into 2 tris).
    for n in [4usize, 8, 16] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        assert_eq!(
            mesh.n_elements(),
            2 * n * n,
            "n={n}: expected {} elements, got {}",
            2 * n * n,
            mesh.n_elements()
        );
    }
}

#[test]
fn unit_square_tri_all_elements_are_tri3() {
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    for e in mesh.elem_iter() {
        assert_eq!(
            mesh.element_type(e),
            ElementType::Tri3,
            "element {e} should be Tri3"
        );
    }
}

#[test]
fn unit_square_tri_element_nodes_have_three_nodes() {
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        assert_eq!(
            nodes.len(),
            3,
            "Tri3 element {e} should have 3 nodes, got {}",
            nodes.len()
        );
        // All node indices should be in range
        for &n in nodes {
            assert!(
                (n as usize) < mesh.n_nodes(),
                "element {e}: node index {n} out of range (n_nodes={})",
                mesh.n_nodes()
            );
        }
    }
}

#[test]
fn unit_square_tri_face_count() {
    // A unit-square mesh of n×n tris has 4*n boundary edges.
    for n in [4usize, 8, 16] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        assert_eq!(
            mesh.n_faces(),
            4 * n,
            "n={n}: expected {} boundary faces, got {}",
            4 * n,
            mesh.n_faces()
        );
    }
}

#[test]
fn unit_square_tri_boundary_tags_are_1_to_4() {
    let mesh = SimplexMesh::<2>::unit_square_tri(8);
    let tags = mesh.unique_boundary_tags();
    // All four walls should be present
    for &expected in &[1i32, 2, 3, 4] {
        assert!(
            tags.contains(&expected),
            "boundary tag {expected} should be present, found: {tags:?}"
        );
    }
    // No unexpected tags
    assert_eq!(
        tags.len(),
        4,
        "expected exactly 4 unique boundary tags, got {tags:?}"
    );
}

#[test]
fn unit_square_tri_boundary_tag_physical_location() {
    // Verify that each boundary face tag corresponds to the correct wall.
    // Convention from unit_square_tri:
    //   tag 1 = bottom (y ≈ 0)
    //   tag 2 = right  (x ≈ 1)
    //   tag 3 = top    (y ≈ 1)
    //   tag 4 = left   (x ≈ 0)
    let mesh = SimplexMesh::<2>::unit_square_tri(8);

    for f in mesh.face_iter() {
        let nodes = mesh.face_nodes(f);
        let tag = mesh.face_tag(f);

        for &nd in nodes {
            let c = mesh.node_coords(nd);
            match tag {
                1 => assert!(c[1].abs() < 1e-12, "tag 1 face {f} node {nd}: y={} ≠ 0", c[1]),
                2 => assert!((c[0] - 1.0).abs() < 1e-12, "tag 2 face {f} node {nd}: x={} ≠ 1", c[0]),
                3 => assert!((c[1] - 1.0).abs() < 1e-12, "tag 3 face {f} node {nd}: y={} ≠ 1", c[1]),
                4 => assert!(c[0].abs() < 1e-12, "tag 4 face {f} node {nd}: x={} ≠ 0", c[0]),
                _ => panic!("unexpected boundary tag {tag}"),
            }
        }
    }
}

#[test]
fn unit_square_tri_node_coords_in_unit_square() {
    let mesh = SimplexMesh::<2>::unit_square_tri(8);
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        assert!(c[0] >= -1e-12 && c[0] <= 1.0 + 1e-12,
            "node {n}: x={} not in [0,1]", c[0]);
        assert!(c[1] >= -1e-12 && c[1] <= 1.0 + 1e-12,
            "node {n}: y={} not in [0,1]", c[1]);
    }
}

#[test]
fn unit_square_tri_check_passes() {
    // mesh.check() should return Ok for a well-formed mesh
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    mesh.check().expect("mesh.check() should pass for unit_square_tri");
}

// ─── Mixed mesh ───────────────────────────────────────────────────────────────

#[test]
fn mixed_mesh_elem_type_accessor() {
    // Build a tiny 2-element mixed mesh (Quad4 + Tri3) and verify element_type.
    let coords = vec![0.0f64, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0];
    let conn: Vec<u32> = vec![0, 1, 4, 3, 1, 2, 5, 1, 5, 4];
    let elem_offsets = vec![0usize, 4, 7, 10];
    let elem_types = vec![ElementType::Quad4, ElementType::Tri3, ElementType::Tri3];

    let mesh = SimplexMesh::<2> {
        coords,
        conn,
        elem_tags: vec![0; 3],
        elem_type: ElementType::Tri3,
        face_conn: vec![0, 1, 1, 2, 2, 5, 5, 4, 4, 3, 3, 0],
        face_tags: vec![1i32, 1, 2, 3, 3, 4],
        face_type: ElementType::Line2,
        elem_types: Some(elem_types),
        elem_offsets: Some(elem_offsets),
        face_types: None,
        face_offsets: None,
    };

    assert!(mesh.is_mixed());
    assert_eq!(mesh.n_elements(), 3);
    assert_eq!(mesh.element_type(0), ElementType::Quad4);
    assert_eq!(mesh.element_type(1), ElementType::Tri3);
    assert_eq!(mesh.element_type(2), ElementType::Tri3);
}
