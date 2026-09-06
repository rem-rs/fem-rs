//! Integration tests for the fem-mesh crate.

use fem_mesh::{element_type::ElementType, topology::MeshTopology, Mesh};

// ─── Unit-square mesh topology ────────────────────────────────────────────────

#[test]
fn unit_square_tri_node_count() {
    // unit_square_tri(n) produces an (n+1)×(n+1) grid of nodes.
    for n in [4usize, 8, 16] {
        let mesh = Mesh::<2>::unit_square_tri(n);
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
        let mesh = Mesh::<2>::unit_square_tri(n);
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
    let mesh = Mesh::<2>::unit_square_tri(4);
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
    let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(n);
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
    let mesh = Mesh::<2>::unit_square_tri(8);
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
    let mesh = Mesh::<2>::unit_square_tri(8);

    for f in mesh.face_iter() {
        let nodes = mesh.face_nodes(f);
        let tag = mesh.face_tag(f);

        for &nd in nodes {
            let c = mesh.node_coords(nd);
            match tag {
                1 => assert!(
                    c[1].abs() < 1e-12,
                    "tag 1 face {f} node {nd}: y={} ≠ 0",
                    c[1]
                ),
                2 => assert!(
                    (c[0] - 1.0).abs() < 1e-12,
                    "tag 2 face {f} node {nd}: x={} ≠ 1",
                    c[0]
                ),
                3 => assert!(
                    (c[1] - 1.0).abs() < 1e-12,
                    "tag 3 face {f} node {nd}: y={} ≠ 1",
                    c[1]
                ),
                4 => assert!(
                    c[0].abs() < 1e-12,
                    "tag 4 face {f} node {nd}: x={} ≠ 0",
                    c[0]
                ),
                _ => panic!("unexpected boundary tag {tag}"),
            }
        }
    }
}

#[test]
fn unit_square_tri_node_coords_in_unit_square() {
    let mesh = Mesh::<2>::unit_square_tri(8);
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        assert!(
            c[0] >= -1e-12 && c[0] <= 1.0 + 1e-12,
            "node {n}: x={} not in [0,1]",
            c[0]
        );
        assert!(
            c[1] >= -1e-12 && c[1] <= 1.0 + 1e-12,
            "node {n}: y={} not in [0,1]",
            c[1]
        );
    }
}

#[test]
fn unit_square_tri_check_passes() {
    // mesh.check() should return Ok for a well-formed mesh
    let mesh = Mesh::<2>::unit_square_tri(4);
    mesh.check()
        .expect("mesh.check() should pass for unit_square_tri");
}

// ─── Mixed mesh ───────────────────────────────────────────────────────────────

#[test]
fn mixed_mesh_elem_type_accessor() {
    // Build a tiny 2-element mixed mesh (Quad4 + Tri3) and verify element_type.
    let coords = vec![
        0.0f64, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0,
    ];
    let conn: Vec<u32> = vec![0, 1, 4, 3, 1, 2, 5, 1, 5, 4];
    let elem_offsets = vec![0usize, 4, 7, 10];
    let elem_types = vec![ElementType::Quad4, ElementType::Tri3, ElementType::Tri3];

    let mesh = Mesh::<2> {
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
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    };

    assert!(mesh.is_mixed());
    assert_eq!(mesh.n_elements(), 3);
    assert_eq!(mesh.element_type(0), ElementType::Quad4);
    assert_eq!(mesh.element_type(1), ElementType::Tri3);
    assert_eq!(mesh.element_type(2), ElementType::Tri3);
}

// ─── Face-to-element mapping ──────────────────────────────────────────────────

#[test]
fn face_elements_after_build_all_boundary_faces_have_owner() {
    let mut mesh = Mesh::<2>::unit_square_tri(4);
    mesh.build_face_to_elem();

    let f2e = mesh
        .face_to_elem
        .as_ref()
        .expect("face_to_elem should be built");
    assert_eq!(f2e.len(), mesh.n_boundary_faces());
    for (bf, &owner) in f2e.iter().enumerate() {
        assert!(
            owner < mesh.n_elements() as u32,
            "boundary face {bf} should have an owned element, got {owner}"
        );
        let (elem, neighbor) = mesh.face_elements(bf as u32);
        assert_eq!(
            elem, owner,
            "face_elements returned wrong owner for face {bf}"
        );
        assert!(
            neighbor.is_none(),
            "boundary face {bf} should have no neighbor"
        );
    }
}

#[test]
fn face_elements_3d_cube_each_boundary_face_has_owner() {
    let mut mesh = Mesh::<3>::unit_cube_tet(2);
    mesh.build_face_to_elem();

    let f2e = mesh
        .face_to_elem
        .as_ref()
        .expect("face_to_elem should be built");
    assert_eq!(f2e.len(), mesh.n_boundary_faces());
    for (bf, &owner) in f2e.iter().enumerate() {
        assert!(
            owner < mesh.n_elements() as u32,
            "3-D boundary face {bf} should have an owner, got {owner}"
        );
        let (elem, neighbor) = mesh.face_elements(bf as u32);
        assert_eq!(elem, owner);
        assert!(neighbor.is_none());
    }
}

#[test]
fn face_elements_not_built_returns_zero() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let (elem, _neighbor) = mesh.face_elements(0);
    assert_eq!(
        elem, 0,
        "before build_face_to_elem, should return (0, None)"
    );
}

// ─── make_cartesian_3d (MFEM MakeCartesian3D, verified 1:1 vs MFEM 4.10
//     reference libmfem.a) ────────────────────────────────────────────────────

#[test]
fn make_cartesian_3d_hex_counts_and_tags() {
    // nx=2, ny=3, nz=4, sx=1, sy=2, sz=3
    let m = Mesh::<3>::make_cartesian_3d(2, 3, 4, ElementType::Hex8, 1.0, 2.0, 3.0, true);
    assert_eq!(m.n_nodes(), (2 + 1) * (3 + 1) * (4 + 1)); // 60
    assert_eq!(m.n_elems(), 2 * 3 * 4); // 24
    assert_eq!(m.n_faces(), 2 * (2 * 3 + 2 * 4 + 3 * 4)); // 52
    assert_eq!(m.face_ids_with_tag(1).len(), 2 * 3); // bottom
    assert_eq!(m.face_ids_with_tag(6).len(), 2 * 3); // top
    assert_eq!(m.face_ids_with_tag(5).len(), 3 * 4); // left
    assert_eq!(m.face_ids_with_tag(3).len(), 3 * 4); // right
    assert_eq!(m.face_ids_with_tag(2).len(), 2 * 4); // front
    assert_eq!(m.face_ids_with_tag(4).len(), 2 * 4); // back
}

#[test]
fn make_cartesian_3d_hex_vertex_ordering_matches_mfem() {
    // VTX(x,y,z) = x + (y + z*(ny+1))*(nx+1); sx, sy, sz divide first like MFEM.
    let m = Mesh::<3>::make_cartesian_3d(2, 3, 4, ElementType::Hex8, 1.0, 2.0, 3.0, true);
    let c0 = m.coords_of(0);
    assert_eq!([c0[0], c0[1], c0[2]], [0.0, 0.0, 0.0]);
    let c_last = m.coords_of(59);
    assert!((c_last[0] - 1.0).abs() < 1e-15);
    assert!((c_last[1] - 2.0).abs() < 1e-15);
    assert!((c_last[2] - 3.0).abs() < 1e-15);
    // vertex 3 = VTX(0,1,0) has y = sy/ny = 2/3
    let c3 = m.coords_of(3);
    assert!((c3[1] - 2.0 / 3.0).abs() < 1e-15);
}

#[test]
fn make_cartesian_3d_hex_sfc_element_order_matches_mfem() {
    // MakeCartesian3D defaults to sfc_ordering=true (Hilbert SFC). For the
    // 2×3×4 box the MFEM 4.10 reference library produces the lexicographic
    // hex (0,0,0) first, then (0,0,1) — verified against libmfem.a.
    let m = Mesh::<3>::make_cartesian_3d(2, 3, 4, ElementType::Hex8, 1.0, 2.0, 3.0, true);
    let e0 = m.element_nodes(0);
    assert_eq!(
        &e0[..],
        &[0u32, 1, 4, 3, 12, 13, 16, 15],
        "sfc first hex should be the (x,y,z)=(0,0,0) box (MFEM reference)"
    );
    let e1 = m.element_nodes(1);
    assert_eq!(
        &e1[..],
        &[12u32, 13, 16, 15, 24, 25, 28, 27],
        "sfc second hex is (0,0,1) in the MFEM 4.10 reference library"
    );
    // Lexicographic ordering puts (1,0,0) second instead.
    let ml = Mesh::<3>::make_cartesian_3d(2, 3, 4, ElementType::Hex8, 1.0, 2.0, 3.0, false);
    let l1 = ml.element_nodes(1);
    assert_eq!(&l1[..], &[1u32, 2, 5, 4, 13, 14, 17, 16]);
}

#[test]
fn make_cartesian_3d_tet_split_matches_mfem_reference_lib() {
    // 1×1×1 box → 6 tets.  Vertex order of the MFEM 4.10 reference lib is
    // (vi[6], vi[0], vi[c], vi[b]) per source row (0,b,c,6), empirically
    // verified against libmfem.a (checked-in 4.10-dev sources list the rows
    // as (0,b,c,6); the reference library predates that rewrite).
    let m = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, true);
    assert_eq!(m.n_elems(), 6);
    assert_eq!(m.n_faces(), 12);
    let expected: [[u32; 4]; 6] = [
        [7, 0, 3, 1],
        [7, 0, 1, 5],
        [7, 0, 5, 4],
        [7, 0, 2, 3],
        [7, 0, 6, 2],
        [7, 0, 4, 6],
    ];
    for (i, exp) in expected.iter().enumerate() {
        let v = m.element_nodes(i as u32);
        assert_eq!(&v[..], exp, "tet {i} mismatch");
    }
    // boundary: bottom quad (0,2,3,1) splits as (3,0,2),(0,3,1) in the lib
    let bottom: Vec<Vec<u32>> = m
        .face_ids_with_tag(1)
        .into_iter()
        .map(|f| m.bface_nodes(f).to_vec())
        .collect();
    assert_eq!(bottom, vec![vec![3, 0, 2], vec![0, 3, 1]]);
}

#[test]
fn make_cartesian_3d_tet_larger_counts() {
    let m = Mesh::<3>::make_cartesian_3d(2, 3, 4, ElementType::Tet4, 1.0, 2.0, 3.0, true);
    assert_eq!(m.n_nodes(), 60);
    assert_eq!(m.n_elems(), 6 * 2 * 3 * 4); // 144
    assert_eq!(m.n_faces(), 2 * (2 * 3 + 2 * 4 + 3 * 4) * 2); // 104 tris
    for tag in 1..=6 {
        let per_face = match tag {
            1 | 6 => 2 * 3,
            5 | 3 => 3 * 4,
            _ => 2 * 4,
        };
        assert_eq!(m.face_ids_with_tag(tag).len(), 2 * per_face, "tag {tag}");
    }
}
