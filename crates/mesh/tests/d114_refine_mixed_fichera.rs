//! D114 regression — uniform refinement of the mixed hex/prism/pyramid mesh
//! `data/fichera-mixed-16.mesh`.
//!
//! History (debt since round 29): `refine_uniform_3d` panicked on this mesh.
//! Two compounding causes in the mixed-3D path (`refine_mixed_3d` /
//! `mark_tet_mesh_for_refinement`):
//! 1. The shared edge tables skipped `Pyramid5` entirely (`_ => continue` /
//!    `_ => &[]`), so a pyramid-owned boundary-triangle edge was never
//!    registered: `mark_tet_mesh_for_refinement` panicked at load on any
//!    tet+pyramid mesh (`em[&edge_key(..)]` — key not found), and the mixed
//!    refinement's boundary rebuild panicked with
//!    `rebuild_3d_boundary: midpoint of edge (a,b) missing from refinement maps`.
//! 2. The child-generation match silently dropped `Pyramid5` parents.
//!
//! MFEM 4.10 semantics pinned here (`Mesh::UniformRefinement3D_base`,
//! `Element::PYRAMID` branch, mesh.cpp:10766-10855): each Pyramid5 →
//! **6 Pyramid5 + 4 Tet4** children (4 corner pyramids, the apex pyramid,
//! the inverted inner pyramid, 4 base-center tets), sharing the global
//! edge-midpoint vertex block plus the base-quad face-center block (MFEM
//! `pyr_t::Edges`, `pyr_t::FaceVert[0]`); a pyramid has no body center.
//!
//! Parent mesh (MFEM `data/fichera-mixed-16.mesh`): 1 Hex8 + 6 Prism6 +
//! 9 Pyramid5 = 16 elements, 39 boundary faces (9 Quad4 + 30 Tri3).
//! Expected children: 8 Hex8 + 48 Prism6 + 54 Pyramid5 + 36 Tet4 = 146
//! elements, boundary ×4 → 36 Quad4 + 120 Tri3 = 156 faces.

use fem_core::{ElemId, FaceId};
use fem_io::mfem::{read_mfem_file, write_mfem_file_3d};
use fem_mesh::element_type::ElementType;
use fem_mesh::transformation::element_jacobian_at;
use fem_mesh::{refine_uniform_3d, Mesh};

const MESH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/fichera-mixed-16.mesh");

fn load_fichera_mixed_16() -> Mesh<3> {
    read_mfem_file(MESH).expect("read fichera-mixed-16").mesh3d.expect("3D mesh")
}

fn child_type_counts(mesh: &Mesh<3>) -> (usize, usize, usize, usize, usize) {
    let (mut n_hex, mut n_prism, mut n_pyr, mut n_tet, mut n_other) = (0, 0, 0, 0, 0);
    for e in 0..mesh.n_elems() {
        match mesh.element_type_at(e as ElemId) {
            ElementType::Hex8 => n_hex += 1,
            ElementType::Prism6 => n_prism += 1,
            ElementType::Pyramid5 => n_pyr += 1,
            ElementType::Tet4 => n_tet += 1,
            _ => n_other += 1,
        }
    }
    (n_hex, n_prism, n_pyr, n_tet, n_other)
}

/// The mixed refinement must keep every parent element: MFEM's per-type
/// child multiplicities on `fichera-mixed-16.mesh`.
#[test]
fn d114_mixed_fichera_refines_all_children() {
    let parent = load_fichera_mixed_16();
    assert_eq!(parent.n_elems(), 16, "parent element count");
    assert_eq!(parent.n_faces(), 39, "parent boundary face count");

    let fine = refine_uniform_3d(&parent);

    let (n_hex, n_prism, n_pyr, n_tet, n_other) = child_type_counts(&fine);
    assert_eq!(n_other, 0, "unexpected child element type");
    // 1 hex → 8, 6 prisms → 8 each, 9 pyramids → (6 Pyramid5 + 4 Tet4) each.
    assert_eq!(n_hex, 8, "Hex8 children");
    assert_eq!(n_prism, 48, "Prism6 children");
    assert_eq!(n_pyr, 54, "Pyramid5 children");
    assert_eq!(n_tet, 36, "Tet4 children");
    assert_eq!(fine.n_elems(), 146, "total children");
}

/// The rebuilt boundary is the parent boundary ×4 (tri → 4 tris,
/// quad → 4 quads), with every pyramid-owned triangle resolved.
#[test]
fn d114_mixed_fichera_boundary_rebuilt() {
    let parent = load_fichera_mixed_16();
    let fine = refine_uniform_3d(&parent);

    let (mut tri, mut quad, mut other) = (0usize, 0usize, 0usize);
    for f in 0..fine.n_faces() {
        match fine.face_type_at(f as FaceId) {
            ElementType::Tri3 => tri += 1,
            ElementType::Quad4 => quad += 1,
            _ => other += 1,
        }
    }
    assert_eq!(other, 0, "unexpected boundary face type");
    assert_eq!(tri, 120, "Tri3 boundary children (30 × 4)");
    assert_eq!(quad, 36, "Quad4 boundary children (9 × 4)");
    assert_eq!(fine.n_faces(), 156, "total boundary faces");
}

/// Affine children (Tet4/Hex8/Prism6) keep positive determinant Jacobians —
/// spot-check at each family's reference centroid.
#[test]
fn d114_mixed_fichera_children_positive_det() {
    let parent = load_fichera_mixed_16();
    let fine = refine_uniform_3d(&parent);

    let mut checked = 0usize;
    for e in 0..fine.n_elems() as u32 {
        let xi: [f64; 3] = match fine.element_type_at(e) {
            ElementType::Tet4 => [0.25, 0.25, 0.25],
            ElementType::Hex8 => [0.0, 0.0, 0.0],
            ElementType::Prism6 => [1.0 / 3.0, 1.0 / 3.0, 0.5],
            _ => continue,
        };
        let (jac, _) = element_jacobian_at(&fine, e, &xi, 3);
        let det = jac.determinant();
        assert!(det > 0.0, "element {e} ({:?}) has det {det}", fine.element_type_at(e));
        checked += 1;
    }
    assert_eq!(checked, 8 + 48 + 36, "affine children checked");
}

/// `Mesh::Save` → re-read: the refined mixed mesh survives a full file
/// round trip (the reader re-runs `mark_tet_mesh_for_refinement` because the
/// children include tets — the pyramid edges must resolve there too).
#[test]
fn d114_mixed_fichera_roundtrip() {
    let parent = load_fichera_mixed_16();
    let fine = refine_uniform_3d(&parent);

    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("d114_refined.mesh");
    write_mfem_file_3d(&out, &fine).expect("write refined mesh");
    let back = read_mfem_file(&out).expect("re-read refined mesh").mesh3d.expect("3D mesh");

    assert_eq!(back.n_elems(), 146, "roundtrip element count");
    assert_eq!(back.n_faces(), 156, "roundtrip boundary face count");
    let (n_hex, n_prism, n_pyr, n_tet, n_other) = child_type_counts(&back);
    assert_eq!((n_hex, n_prism, n_pyr, n_tet, n_other), (8, 48, 54, 36, 0));
}
