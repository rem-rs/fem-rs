//! D602: `HCurlSpace::face_pair_storage_map` — the cross-element frame view
//! of shared triangular faces, pinned against MFEM 4.10's
//! `ND_DofTransformation` T table on the D559 6-tet unit cube (ND2, 74 DOFs).
//!
//! Ground truth (MFEM 4.10, `tmp/d559/` + `tmp/d602/`):
//!
//! * MFEM anchors each face's canonical frame at `FaceInfo::Elem1No` — the
//!   face-creating (first-encounter) element — with orientation **0**
//!   (`Mesh::AddTriangleFaceElement`: `Elem1Inf = 64*lf`, "orientation 0").
//!   fem-rs anchors the same way, so the `.gf` *file* storage maps are
//!   identical (identity); what differs per element is only the *local
//!   frame* of the second element, which this API exposes.
//! * On the 6-tet cube, every interior face's second element has face
//!   orientation Fo = 5 (D559 dump `d559_mfem_dump.txt` FACEORI lines), and
//!   `ND_DofTransformation` T-data (doftrans.cpp:168-176) has
//!   `T(5) = [[0,1],[1,0]]` — a pure swap.  Hence every interior face map
//!   here must be `T(5)⁻¹ = [[0,1],[1,0]]`.

use fem_io::mfem::read_mfem;
use fem_space::{FaceKey, HCurlSpace};
use fem_mesh::topology::MeshTopology;

/// `tmp/d559/d559_tet6.mesh` — MFEM `MakeCartesian3D(1,1,1, TETRAHEDRON)`
/// saved by MFEM 4.10 (unit cube split into 6 tets, MFEM `AddHexAsTets`).
const TET6_MESH: &str = "MFEM mesh v1.0

dimension
3

elements
6
1 4 7 0 3 1
1 4 7 0 1 5
1 4 7 0 5 4
1 4 7 0 2 3
1 4 7 0 6 2
1 4 7 0 4 6

boundary
12
1 2 3 0 2
1 2 0 3 1
6 2 7 4 5
6 2 4 7 6
5 2 6 0 4
5 2 0 6 2
3 2 7 1 3
3 2 1 7 5
2 2 5 0 1
2 2 0 5 4
4 2 7 2 6
4 2 2 7 3

vertices
8
3
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
";

/// One tet's four local faces (any rotation works — `FaceKey` sorts).
const TET_FACES_TEST: [(usize, usize, usize); 4] =
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];

fn mul2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

#[test]
fn interior_face_maps_match_mfem_t5() {
    let file = read_mfem(TET6_MESH.as_bytes()).expect("read tet6 mesh");
    let mesh = file.mesh3d.expect("3-D mesh");
    let space = HCurlSpace::new(mesh, 2);
    assert_eq!(space.n_dofs(), 74, "tet ND2 cube vsize (D559)");
    let topo = space.mesh_topology();

    // Enumerate every unique tri face by scanning the elements.
    let mut faces: Vec<FaceKey> = Vec::new();
    for e in topo.elem_iter() {
        let verts = topo.element_nodes(e).to_vec();
        for &(a, b, c) in &TET_FACES_TEST {
            let key = FaceKey::new(verts[a], verts[b], verts[c]);
            if !faces.contains(&key) {
                faces.push(key);
            }
        }
    }
    assert_eq!(faces.len(), 18, "12 boundary + 6 interior faces");

    let mut n_interior = 0usize;
    let mut n_boundary = 0usize;
    for key in &faces {
        // Every face of a tet NDk (k >= 2) space carries shared DOFs.
        assert!(space.face_dof(*key).is_some(), "face {key:?} registered");
        let map = space.face_pair_storage_map(*key);
        match map {
            None => n_boundary += 1,
            Some(r) => {
                n_interior += 1;
                // MFEM T(5) = [[0,1],[1,0]] (D559 FACEORI dump: Fo(Elem2) = 5
                // on all six interior faces of this cube).
                let t5 = [[0.0, 1.0], [1.0, 0.0]];
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            (r[i][j] - t5[i][j]).abs() < 1e-12,
                            "face {key:?} map {r:?} != T(5)^-1 {t5:?}"
                        );
                    }
                }
                // T(5) is an involution, so R² must be the identity.
                let r2 = mul2(r, r);
                assert!((r2[0][0] - 1.0).abs() < 1e-12 && r2[0][1].abs() < 1e-12);
                assert!(r2[1][0].abs() < 1e-12 && (r2[1][1] - 1.0).abs() < 1e-12);
            }
        }
    }
    assert_eq!(n_interior, 6, "six interior faces (D559)");
    assert_eq!(n_boundary, 12, "twelve boundary triangles");
}

#[test]
fn unknown_and_low_order_faces_yield_none() {
    let file = read_mfem(TET6_MESH.as_bytes()).expect("read tet6 mesh");
    let mesh = file.mesh3d.expect("3-D mesh");

    let topo_box = mesh.clone_mesh();
    let topo: &dyn MeshTopology = topo_box.as_ref();
    let ns = topo.face_nodes(0).to_vec();
    let boundary_key = FaceKey::new(ns[0], ns[1], ns[2]);

    // ND1 has no face DOFs at all.
    let nd1 = HCurlSpace::new(mesh.clone_mesh(), 1);
    assert!(nd1.face_pair_storage_map(boundary_key).is_none());

    let nd2 = HCurlSpace::new(mesh.clone_mesh(), 2);
    // A boundary face has a single writer → no cross-element relation.
    assert!(nd2.face_pair_storage_map(boundary_key).is_none());
    // An unregistered face key (duplicated vertex → not a real face).
    let bogus = FaceKey::new(0, 0, 1);
    assert!(nd2.face_pair_storage_map(bogus).is_none());
}
