//! NCState3D and NCStateHex refine/derefine roundtrip tests.

use fem_mesh::amr;
use fem_mesh::Mesh;

#[test]
fn nc_state_3d_tet4_refine_derefine_roundtrip() {
    let mut nc = amr::NCState3D::new();
    let mut mesh = Mesh::<3>::unit_cube_tet(2);
    let orig_nodes = mesh.n_nodes();
    let orig_elems = mesh.n_elems();
    let marked = vec![0u32];
    let (new_mesh, constraints, _mm, hanging_faces) = nc.refine(&mesh, &marked);
    assert!(new_mesh.n_elems() > orig_elems);
    assert!(new_mesh.n_nodes() > orig_nodes);
    assert!(!constraints.is_empty() || !hanging_faces.is_empty());
    assert!(nc.can_derefine());
    let (recovered, _c, _hf) = nc.derefine_last().unwrap();
    assert_eq!(recovered.n_elems(), orig_elems);
    assert_eq!(recovered.n_nodes(), orig_nodes);
    assert!(!nc.can_derefine());
}

#[test]
fn nc_state_hex8_refine_derefine_roundtrip() {
    let mut nc = amr::NCStateHex::new();
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let orig_nodes = mesh.n_nodes();
    let orig_elems = mesh.n_elems();
    let marked = vec![0u32];
    let (new_mesh, constraints, _fc, _mm) = nc.refine(&mesh, &marked);
    assert!(new_mesh.n_elems() > orig_elems);
    assert!(new_mesh.n_nodes() > orig_nodes);
    assert!(!constraints.is_empty());
    let (recovered, _c, _fc) = nc.derefine_last().unwrap();
    assert_eq!(recovered.n_elems(), orig_elems);
    assert_eq!(recovered.n_nodes(), orig_nodes);
}

#[test]
fn prolongate_restrict_p1_quad_roundtrip() {
    let mut nc = amr::NCStateQuad::new();
    let mesh = Mesh::<2>::unit_square_quad(2);
    let n0 = mesh.n_nodes();
    let u0: Vec<f64> = (0..n0).map(|i| (i as f64).sin()).collect();
    let marked = vec![0u32];
    let (fine, _c, midpoint_map) = nc.refine(&mesh, &marked, 0);
    let u_fine = amr::prolongate_p1(&u0, fine.n_nodes(), &midpoint_map);
    let u_recovered = amr::restrict_to_coarse_p1(&u_fine, n0);
    for i in 0..n0 {
        assert!((u_recovered[i] - u0[i]).abs() < 1e-14);
    }
}

#[test]
fn refine_nonconforming_tri3_public_api() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let marked = vec![0u32, 1];
    let (new_mesh, constraints) = amr::refine_nonconforming(&mesh, &marked, None);
    assert!(new_mesh.n_elems() > mesh.n_elems());
    assert!(!constraints.is_empty());
    new_mesh.check().unwrap();
}

#[test]
fn refine_nonconforming_3d_tet4_public_api() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let marked = vec![0u32];
    let (new_mesh, edge_c, face_c) = amr::refine_nonconforming_3d(&mesh, &marked, None);
    assert!(new_mesh.n_elems() > mesh.n_elems());
    assert!(!edge_c.is_empty() || !face_c.is_empty());
    new_mesh.check().unwrap();
}

#[test]
fn refine_nonconforming_quad_public_api() {
    let mesh = Mesh::<2>::unit_square_quad(2);
    let marked = vec![0u32];
    let (new_mesh, edge_c) = amr::refine_nonconforming_quad(&mesh, &marked, None);
    assert!(new_mesh.n_elems() > mesh.n_elems());
    assert!(!edge_c.is_empty());
    new_mesh.check().unwrap();
}

#[test]
fn tet4_nc_refine_mesh_valid() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let marked = vec![0u32, 1, 2];
    let (new_mesh, _c, _tf) = amr::refine_nonconforming_3d(&mesh, &marked, None);
    new_mesh.check().unwrap();
}

#[test]
fn hex8_nc_refine_mesh_valid() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let marked = vec![0u32, 1];
    let (new_mesh, _c, _qf, _mm) = amr::refine_nonconforming_hex(&mesh, &marked, None);
    new_mesh.check().unwrap();
}

#[test]
fn p_refine_tri3_to_tri6_roundtrip() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let n_orig = mesh.n_nodes();
    let (p2, midpoint_map) = amr::p_refine_tri3_to_tri6(&mesh, &[0, 1, 2]);
    assert!(p2.n_nodes() > n_orig);
    assert_eq!(p2.n_elems(), mesh.n_elems());
    assert!(!midpoint_map.is_empty());
    if let Some(ref types) = p2.elem_types {
        assert_eq!(types[0], fem_mesh::element_type::ElementType::Tri6);
    }
    p2.check().unwrap();
}

#[test]
fn p_refine_tet4_to_tet10() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let n_orig = mesh.n_nodes();
    let (p2, midpoint_map) = amr::p_refine_tet4_to_tet10(&mesh, &[0, 1]);
    assert!(p2.n_nodes() > n_orig);
    assert_eq!(p2.n_elems(), mesh.n_elems());
    assert!(!midpoint_map.is_empty());
    p2.check().unwrap();
}

#[test]
fn p_refine_quad4_to_quad9() {
    let mesh = Mesh::<2>::unit_square_quad(2);
    let n_orig = mesh.n_nodes();
    let (p2, midpoint_map) = amr::p_refine_quad4_to_quad9(&mesh, &[0, 1]);
    assert!(p2.n_nodes() > n_orig);
    assert_eq!(p2.n_elems(), mesh.n_elems());
    assert!(!midpoint_map.is_empty());
    p2.check().unwrap();
}

#[test]
fn p_refine_hex8_to_hex20() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let n_orig = mesh.n_nodes();
    let (p2, midpoint_map) = amr::p_refine_hex8_to_hex20(&mesh, &[0]);
    assert!(p2.n_nodes() > n_orig);
    assert_eq!(p2.n_elems(), mesh.n_elems());
    assert!(!midpoint_map.is_empty());
    p2.check().unwrap();
}

#[test]
fn hanging_node_constraint_p1() {
    let hc = amr::HangingNodeConstraint::new_p1(10, 3, 7);
    assert_eq!(hc.constrained, 10);
    assert_eq!(hc.parent_a, 3);
    assert_eq!(hc.parent_b, 7);
    assert_eq!(hc.coeff_a, 0.5);
    assert_eq!(hc.coeff_b, 0.5);
    let parents: Vec<_> = hc.parents().collect();
    assert_eq!(parents, vec![(3, 0.5), (7, 0.5)]);
}

#[test]
fn hanging_face_constraint_basic() {
    let hc = amr::HangingFaceConstraint {
        constrained: 10, parent_a: 0, parent_b: 1, parent_c: 2,
    };
    assert_eq!(hc.constrained, 10);
}

#[test]
fn hanging_quad_face_constraint_basic() {
    let hc = amr::HangingQuadFaceConstraint {
        constrained: 20, parent_a: 0, parent_b: 1, parent_c: 2, parent_d: 3,
    };
    assert_eq!(hc.constrained, 20);
}
