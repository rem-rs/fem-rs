//! D472 — pure-pyramid uniform refinement aligned with MFEM (6 Pyramid5 + 4 Tet4).
//!
//! MFEM 4.10 `UniformRefinement3D_base` PYRAMID branch (mesh.cpp:10766-10855)
//! splits each Pyramid5 into **6 Pyramid5 + 4 Tet4** children; the refined mesh
//! is therefore *mixed*.  fem-rs used to split into 16 Tet4 (own semantics);
//! `refine_pyramid5_uniform` now delegates to `refine_mixed_3d`, whose pyramid
//! branch IS the MFEM implementation (same generator as the D114 fichera path).
//!
//! Oracle: MFEM 4.10 probe over `tmp/d472/pyr1.mesh` (single unit pyramid) and
//! `tmp/d472/pyr3.mesh` (two disjoint unit pyramids, attrs 1/2) — outputs kept
//! verbatim in `tmp/d472/probe_pyr1.out` / `probe_pyr3.out`.  Boundary faces
//! are declared outward-oriented (base quad `(3,2,1,0)`), matching MFEM's
//! post-load `fix_orientation` state, so faces pin exactly (order + winding).
//!
//! Vertex numbering (tet-free MFEM path): edge midpoints `oedge + e` in
//! first-touch `pyr_t::Edges` order (5..12), then the base-quad center
//! `oface + f2qf` (13).  No `e2v` re-sort — a pure pyramid mesh has no
//! TETRAHEDRON geometry (`HasGeometry(Geometry::TETRAHEDRON)` gate,
//! mesh.cpp:10452).

use fem_io::mfem::{read_mfem_file, write_mfem_file_3d};
use fem_mesh::element_type::ElementType;
use fem_mesh::{refine_pyramid5_uniform, refine_uniform_3d, Mesh, MeshTopology};

/// Unit pyramid, boundary declared in MFEM's outward orientation (exactly the
/// boundary section of `tmp/d472/pyr1.mesh`).  The boundary mixes one quad
/// with four tris, so the full `Mesh` literal (face_types + face_offsets) is
/// required — `Mesh::uniform` assumes a single uniform face type.
fn unit_pyramid(tag_base: i32) -> Mesh<3> {
    Mesh {
        coords: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        conn: vec![0u32, 1, 2, 3, 4],
        elem_tags: vec![tag_base],
        elem_type: ElementType::Pyramid5,
        // base quad (3,2,1,0) + side tris (0,1,4),(1,2,4),(2,3,4),(3,0,4)
        face_conn: vec![3u32, 2, 1, 0, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4],
        face_tags: vec![tag_base, tag_base + 1, tag_base + 2, tag_base + 3, tag_base + 4],
        face_type: ElementType::Quad4,
        elem_types: None, elem_offsets: None,
        face_types: Some(vec![
            ElementType::Quad4, ElementType::Tri3, ElementType::Tri3,
            ElementType::Tri3, ElementType::Tri3,
        ]),
        face_offsets: Some(vec![0, 4, 7, 10, 13, 16]),
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    }
}

/// Two disjoint unit pyramids at attrs 1/2 (exactly `tmp/d472/pyr3.mesh`).
fn two_pyramids() -> Mesh<3> {
    let mut mesh = unit_pyramid(1);
    mesh.coords.extend_from_slice(&[
        2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0,
    ]);
    mesh.conn.extend_from_slice(&[5u32, 6, 7, 8, 9]);
    mesh.elem_tags.push(2);
    mesh.face_conn
        .extend_from_slice(&[8u32, 7, 6, 5, 5, 6, 9, 6, 7, 9, 7, 8, 9, 8, 5, 9]);
    mesh.face_tags.extend_from_slice(&[6, 7, 8, 9, 10]);
    mesh.face_types.as_mut().unwrap().push(ElementType::Quad4);
    for _ in 0..4 { mesh.face_types.as_mut().unwrap().push(ElementType::Tri3); }
    let fo = mesh.face_offsets.as_mut().unwrap();
    let base = *fo.last().unwrap();
    fo.extend_from_slice(&[base + 4, base + 7, base + 10, base + 13, base + 16]);
    mesh
}

/// Signed tet volume `det[p1-p0, p2-p0, p3-p0] / 6`.
fn tet_vol(p: &[f64], a: u32, b: u32, c: u32, d: u32) -> f64 {
    let g = |n: u32| [p[3 * n as usize], p[3 * n as usize + 1], p[3 * n as usize + 2]];
    let (a, b, c, d) = (g(a), g(b), g(c), g(d));
    let sub = |x: [f64; 3], y: [f64; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let cross = |x: [f64; 3], y: [f64; 3]| {
        [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    };
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    dot(sub(b, a), cross(sub(c, a), sub(d, a))) / 6.0
}

/// Signed volume of one child element (Pyramid5 via its two base tets).
/// Positive children = positively oriented, non-degenerate (det > 0).
fn elem_signed_vol(mesh: &Mesh<3>, e: u32) -> f64 {
    let ns = mesh.elem_nodes(e);
    let p = &mesh.coords;
    match mesh.element_type(e) {
        ElementType::Pyramid5 => {
            tet_vol(p, ns[0], ns[1], ns[2], ns[4]) + tet_vol(p, ns[0], ns[2], ns[3], ns[4])
        }
        ElementType::Tet4 => tet_vol(p, ns[0], ns[1], ns[2], ns[3]),
        other => panic!("unexpected child element type {other:?}"),
    }
}

/// Flatten the boundary of a refined mesh into `(type, verts, tag)` triples.
fn faces(mesh: &Mesh<3>) -> Vec<(ElementType, Vec<u32>, i32)> {
    let mut out = Vec::new();
    for f in 0..mesh.n_faces() {
        let (start, len) = match &mesh.face_offsets {
            Some(off) => (off[f], off[f + 1] - off[f]),
            None => (
                f * mesh.face_type.nodes_per_element(),
                mesh.face_type.nodes_per_element(),
            ),
        };
        let ftype = match &mesh.face_types {
            Some(tys) => tys[f],
            None => mesh.face_type,
        };
        out.push((
            ftype,
            mesh.face_conn[start..start + len].to_vec(),
            mesh.face_tags[f],
        ));
    }
    out
}

const L1_PYR: [[u32; 5]; 6] = [
    [0, 5, 13, 8, 9],
    [5, 1, 6, 13, 10],
    [13, 6, 2, 7, 11],
    [8, 13, 7, 3, 12],
    [9, 10, 11, 12, 4],
    [12, 11, 10, 9, 13],
];
const L1_TET: [[u32; 4]; 4] = [[5, 9, 10, 13], [6, 10, 11, 13], [7, 11, 12, 13], [8, 12, 9, 13]];

/// MFEM probe_pyr1.out L1 — children in MFEM order: 6 corner/apex/inner
/// pyramids, then the 4 base-center tets.
#[test]
fn d472_single_pyramid_l1_children_match_mfem() {
    let mesh = unit_pyramid(1);
    let all: Vec<u32> = (0..mesh.n_elems() as u32).collect();
    let (fine, constraints) = refine_pyramid5_uniform(&mesh, &all);
    assert!(constraints.is_empty(), "uniform refinement creates no constraints");

    assert_eq!(fine.n_elems(), 10, "MFEM: 6 pyramids + 4 tets");
    assert_eq!(fine.n_nodes(), 14, "5 corners + 8 edge midpoints (5..12) + 1 base center");
    assert_eq!(fine.n_faces(), 20, "4 base quads + 16 side tris");

    for (i, want) in L1_PYR.iter().enumerate() {
        assert_eq!(fine.element_type(i as u32), ElementType::Pyramid5, "child {i}");
        assert_eq!(fine.elem_nodes(i as u32), want, "child {i} connectivity (MFEM order)");
    }
    for (k, want) in L1_TET.iter().enumerate() {
        let i = 6 + k;
        assert_eq!(fine.element_type(i as u32), ElementType::Tet4, "child {i}");
        assert_eq!(fine.elem_nodes(i as u32), want, "child {i} connectivity (MFEM order)");
    }

    // Attribute inheritance: every child keeps the parent's tag.
    assert!(fine.elem_tags.iter().all(|&t| t == 1));

    // Boundary: MFEM probe_pyr1.out B0..B19, order + winding + tags.
    let want_faces: Vec<(ElementType, Vec<u32>, i32)> = vec![
        (ElementType::Quad4, vec![3, 7, 13, 8], 1),
        (ElementType::Quad4, vec![7, 2, 6, 13], 1),
        (ElementType::Quad4, vec![13, 6, 1, 5], 1),
        (ElementType::Quad4, vec![8, 13, 5, 0], 1),
        (ElementType::Tri3, vec![0, 5, 9], 2),
        (ElementType::Tri3, vec![10, 9, 5], 2),
        (ElementType::Tri3, vec![5, 1, 10], 2),
        (ElementType::Tri3, vec![9, 10, 4], 2),
        (ElementType::Tri3, vec![1, 6, 10], 3),
        (ElementType::Tri3, vec![11, 10, 6], 3),
        (ElementType::Tri3, vec![6, 2, 11], 3),
        (ElementType::Tri3, vec![10, 11, 4], 3),
        (ElementType::Tri3, vec![2, 7, 11], 4),
        (ElementType::Tri3, vec![12, 11, 7], 4),
        (ElementType::Tri3, vec![7, 3, 12], 4),
        (ElementType::Tri3, vec![11, 12, 4], 4),
        (ElementType::Tri3, vec![3, 8, 12], 5),
        (ElementType::Tri3, vec![9, 12, 8], 5),
        (ElementType::Tri3, vec![8, 0, 9], 5),
        (ElementType::Tri3, vec![12, 9, 4], 5),
    ];
    assert_eq!(faces(&fine), want_faces, "refined boundary (order, winding, tags)");

    // Every child positively oriented and volume-preserving (1/3 exact).
    let vol: f64 = (0..fine.n_elems() as u32).map(|e| elem_signed_vol(&fine, e)).sum();
    for e in 0..fine.n_elems() as u32 {
        assert!(elem_signed_vol(&fine, e) > 1e-12, "child {e} must have det > 0");
    }
    assert!((vol - 1.0 / 3.0).abs() < 1e-12, "children volume {vol:.15} != 1/3");
    fine.check().unwrap();
}

/// The dispatch entry produces exactly the same mesh as the direct call.
#[test]
fn d472_dispatch_and_direct_call_agree() {
    let mesh = unit_pyramid(1);
    let all: Vec<u32> = (0..mesh.n_elems() as u32).collect();
    let (direct, _) = refine_pyramid5_uniform(&mesh, &all);
    let via_dispatch = refine_uniform_3d(&mesh);
    assert_eq!(via_dispatch.n_elems(), direct.n_elems());
    assert_eq!(via_dispatch.n_nodes(), direct.n_nodes());
    assert_eq!(via_dispatch.n_faces(), direct.n_faces());
    for e in 0..direct.n_elems() as u32 {
        assert_eq!(via_dispatch.element_type(e), direct.element_type(e));
        assert_eq!(via_dispatch.elem_nodes(e), direct.elem_nodes(e));
    }
    assert_eq!(faces(&via_dispatch), faces(&direct));
}

/// Two disjoint pyramids (attrs 1, 2 — `tmp/d472/pyr3.mesh`): global vertex
/// numbering is element-order first-touch (parent-0 midpoints 10..17 + center
/// 26, parent-1 midpoints 18..25 + center 27), and the second refinement
/// (through the mixed path, since the children are Pyramid5+Tet4) reproduces
/// MFEM's L2 counts.
#[test]
fn d472_two_pyramids_l1_and_l2_match_mfem() {
    let mesh = two_pyramids();

    let all: Vec<u32> = (0..mesh.n_elems() as u32).collect();
    let (fine, _) = refine_pyramid5_uniform(&mesh, &all);

    // probe_pyr3.out L1
    assert_eq!(fine.n_elems(), 20, "MFEM: 12 pyramids + 8 tets");
    assert_eq!(fine.n_nodes(), 28);
    assert_eq!(fine.n_faces(), 40);

    let mut npyr = 0usize;
    let mut ntet = 0usize;
    for e in 0..fine.n_elems() as u32 {
        match fine.element_type(e) {
            ElementType::Pyramid5 => npyr += 1,
            ElementType::Tet4 => ntet += 1,
            other => panic!("unexpected child type {other:?}"),
        }
    }
    assert_eq!((npyr, ntet), (12, 8));

    // Per-parent vertex blocks (MFEM numbering) — sentinel children.
    assert_eq!(fine.elem_nodes(0), [0, 10, 26, 13, 14], "E0");
    assert_eq!(fine.elem_nodes(6), [10, 14, 15, 26], "E6 (first tet)");
    assert_eq!(fine.elem_nodes(10), [5, 18, 27, 21, 22], "E10 (second parent, center 27)");
    assert_eq!(fine.elem_nodes(16), [18, 22, 23, 27], "E16 (second parent, first tet)");
    // Attribute inheritance per parent block.
    assert!(fine.elem_tags[..10].iter().all(|&t| t == 1));
    assert!(fine.elem_tags[10..].iter().all(|&t| t == 2));

    let vol1: f64 = (0..fine.n_elems() as u32).map(|e| elem_signed_vol(&fine, e)).sum();
    for e in 0..fine.n_elems() as u32 {
        assert!(elem_signed_vol(&fine, e) > 1e-12, "L1 child {e} must have det > 0");
    }
    assert!((vol1 - 2.0 / 3.0).abs() < 1e-12, "L1 volume {vol1:.15} != 2/3");

    // Second refinement — the mesh is now mixed (Pyr+Tet) and routes through
    // refine_mixed_3d; MFEM probe_pyr3.out L2 counts must match exactly.
    let fine2 = refine_uniform_3d(&fine);
    assert_eq!(fine2.n_elems(), 184, "MFEM L2: 72 pyramids + 112 tets");
    assert_eq!(fine2.n_nodes(), 110);
    assert_eq!(fine2.n_faces(), 160);
    let mut npyr2 = 0usize;
    let mut ntet2 = 0usize;
    for e in 0..fine2.n_elems() as u32 {
        match fine2.element_type(e) {
            ElementType::Pyramid5 => npyr2 += 1,
            ElementType::Tet4 => ntet2 += 1,
            other => panic!("unexpected L2 child type {other:?}"),
        }
    }
    assert_eq!((npyr2, ntet2), (72, 112));

    let vol2: f64 = (0..fine2.n_elems() as u32).map(|e| elem_signed_vol(&fine2, e)).sum();
    for e in 0..fine2.n_elems() as u32 {
        assert!(elem_signed_vol(&fine2, e) > 1e-12, "L2 child {e} must have det > 0");
    }
    assert!((vol2 - 2.0 / 3.0).abs() < 1e-12, "L2 volume {vol2:.15} != 2/3");
    fine2.check().unwrap();
}

/// Single pyramid, second refinement (probe_pyr1.out L2): NV 55, NE 92
/// (36 Pyr + 56 Tet), NBE 80 — the L1 tets take the rt-selected octahedron
/// split, but the counts are rt-independent.
#[test]
fn d472_single_pyramid_l2_counts_match_mfem() {
    let mesh = unit_pyramid(1);
    let all: Vec<u32> = (0..mesh.n_elems() as u32).collect();
    let (fine, _) = refine_pyramid5_uniform(&mesh, &all);
    let fine2 = refine_uniform_3d(&fine);

    assert_eq!(fine2.n_elems(), 92);
    assert_eq!(fine2.n_nodes(), 55);
    assert_eq!(fine2.n_faces(), 80);
    let mut npyr = 0usize;
    let mut ntet = 0usize;
    for e in 0..fine2.n_elems() as u32 {
        match fine2.element_type(e) {
            ElementType::Pyramid5 => npyr += 1,
            ElementType::Tet4 => ntet += 1,
            other => panic!("unexpected L2 child type {other:?}"),
        }
    }
    assert_eq!((npyr, ntet), (36, 56));

    let vol: f64 = (0..fine2.n_elems() as u32).map(|e| elem_signed_vol(&fine2, e)).sum();
    for e in 0..fine2.n_elems() as u32 {
        assert!(elem_signed_vol(&fine2, e) > 1e-12, "L2 child {e} must have det > 0");
    }
    assert!((vol - 1.0 / 3.0).abs() < 1e-12, "L2 volume {vol:.15} != 1/3");
    fine2.check().unwrap();
}

/// The refined mixed mesh survives a .mesh write/read cycle (same IO pipeline
/// as the d114 fichera roundtrip).
#[test]
fn d472_refined_mesh_roundtrips_through_io() {
    let mesh = unit_pyramid(1);
    let all: Vec<u32> = (0..mesh.n_elems() as u32).collect();
    let (fine, _) = refine_pyramid5_uniform(&mesh, &all);

    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("d472_refined.mesh");
    write_mfem_file_3d(&out, &fine).expect("write refined mesh");
    let back = read_mfem_file(&out).expect("re-read refined mesh").mesh3d.expect("3D mesh");

    assert_eq!(back.n_elems(), 10, "roundtrip element count");
    assert_eq!(back.n_nodes(), 14, "roundtrip vertex count");
    assert_eq!(back.n_faces(), 20, "roundtrip boundary face count");
    let mut npyr = 0usize;
    let mut ntet = 0usize;
    for e in 0..back.n_elems() as u32 {
        match back.element_type(e) {
            ElementType::Pyramid5 => npyr += 1,
            ElementType::Tet4 => ntet += 1,
            other => panic!("unexpected roundtrip child type {other:?}"),
        }
    }
    assert_eq!((npyr, ntet), (6, 4), "roundtrip child type multiset");
    // IO normalizes element vertex order (see d114), so compare the children
    // as node sets and by volume, not by raw connectivity.
    for e in 0..fine.n_elems() as u32 {
        let mut a = fine.elem_nodes(e).to_vec();
        let mut b = back.elem_nodes(e).to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "child {e} node set");
        assert!(elem_signed_vol(&back, e) > 1e-12, "roundtrip child {e} det > 0");
    }
    let vol: f64 = (0..back.n_elems() as u32).map(|e| elem_signed_vol(&back, e)).sum();
    assert!((vol - 1.0 / 3.0).abs() < 1e-12, "roundtrip volume {vol:.15} != 1/3");
}
