//! D393 — mixed-cell HDiv spaces must size face blocks by the face shape and
//! match the assembly reference elements' per-element dof counts.
//!
//! Debt: `HDivSpace::build_mixed` allocated `order + 1` dofs on *every* face
//! and hard-coded tet/hex interior counts (`k=1 → 2`, else 6 / `k>=1 → 12`),
//! so mixed meshes at order >= 1 diverged from MFEM's layout on three counts:
//!
//! 1. triangular faces carry `(k+1)(k+2)/2` dofs (MFEM `RT_TriangleElement(p)`
//!    trace), quadrilateral faces `(k+1)^2` (`RT_QuadrilateralElement(p)`
//!    trace) — `fe_coll.cpp:2531` (`RT_FECollection` ctor) and the per-shape
//!    face tables of `RT_TetrahedronElement` (`fe/fe_rt.cpp:899`) /
//!    `RT_HexahedronElement` (`fe/fe_rt.cpp:326`);
//! 2. tet interiors are `k(k+1)(k+2)/2` and hex interiors `3k(k+1)^2` (the
//!    same formulas the pure-mesh builders `build_3d_tet`/`build_3d_hex` use,
//!    D377/D392-verified against MFEM `GetVSize`);
//! 3. the space's per-element slot count must equal the assembly reference
//!    element's `n_dofs` (hex `HexRTk(k)`, tet `TetRTk(k)`), otherwise the
//!    vector assembler pairs slots with the wrong basis functions / panics on
//!    a count mismatch.
//!
//! Oracle (MFEM 4.10 closed formula on `data/d393_mixed_hex_tet.mesh` —
//! 1 hex + 2 tets; 6 quad + 7 tri unique faces, 6+6 of them on the boundary;
//! archived probe runs `tmp/d392/probe49_run2.out`):
//!
//! ```text
//! vsize(k) = 7·(k+1)(k+2)/2 + 6·(k+1)² + 2·k(k+1)(k+2)/2 + 3k(k+1)²
//! ess(k)   = 6·(k+1)(k+2)/2 + 6·(k+1)²
//!   RT0  13/12    RT1  63/42    RT2 174/90    RT3 370/156
//! ```

use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::{FESpace, HDivSpace};

fn load(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

/// `(k, MFEM GetVSize, MFEM GetBoundaryTrueDofs count)` on the mixed
/// hex + bipyramid (two tets) mesh.
const ORACLE: &[(u8, usize, usize)] = &[
    (0, 13, 12),
    (1, 63, 42),
    (2, 174, 90),
    (3, 370, 156),
];

#[test]
fn rt0_numbering_is_first_encounter_entity_major() {
    // RT0 bit-identity pin (D393): at k = 0 the rebuilt entity-major
    // mixed builder must reproduce the historical single-pass numbering —
    // dof ids in first-encounter face order, tet faces in MFEM FaceVert
    // canon order.
    let mesh = load("data/d393_mixed_hex_tet.mesh");
    let space = HDivSpace::new(mesh.clone(), 0);
    // Elem 0 = hex: its 6 faces are first-seen → dofs 0..6 in HEX_FACES order.
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3, 4, 5]);
    // Elem 1 = tet (verts 8,9,10,11): TET_FACES_CANON rows {9,10,11}, {8,11,10},
    // {8,9,11}, {8,10,9} — all first-seen → 6, 7, 8, 9.
    assert_eq!(space.element_dofs(1), &[6, 7, 8, 9]);
    // Elem 2 = tet (verts 9,8,10,12): three new faces → 10, 11, 12; the last
    // canon row {9,10,8} is the shared face — elem 1's fourth canon face
    // (dof 9).
    assert_eq!(space.element_dofs(2), &[10, 11, 12, 9]);
}

#[test]
fn vsize_and_ess_match_mfem_on_mixed_hex_tet_mesh() {
    let mesh = load("data/d393_mixed_hex_tet.mesh");
    let all_tags = mesh.unique_boundary_tags();
    for &(k, vsize, ess) in ORACLE {
        let space = HDivSpace::new(mesh.clone(), k);
        assert_eq!(space.n_dofs(), vsize, "k={k}: GetVSize mismatch");
        let dofs = boundary_dofs_hdiv(space.mesh(), &space, &all_tags);
        assert_eq!(dofs.len(), ess, "k={k}: GetBoundaryTrueDofs count mismatch");
        assert!(
            dofs.windows(2).all(|w| w[0] < w[1]),
            "k={k}: essential list not strictly sorted"
        );
    }
}

#[test]
fn face_blocks_are_sized_by_face_shape() {
    let mesh = load("data/d393_mixed_hex_tet.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);

    // Element layout: 0 = hex (verts 0..7), 1 = tet (8,9,10,11),
    // 2 = tet (9,8,10,12).  The two tets share the triangular face {8,9,10}.
    let shared_tri = fem_space::dof_manager::FaceKey::new(8, 9, 10);
    let block = space.face_dofs(shared_tri).expect("shared tri face missing");
    assert_eq!(block.len(), 3, "tri face must carry (k+1)(k+2)/2 = 3 dofs at k=1");

    // Every hex face is a boundary quad: face 0 of the boundary table is
    // verts (1-based 1 4 3 2) = {0, 3, 2, 1}.  The builder stores the quad
    // under the sorted-4 first-3 key, i.e. {0, 1, 2}.
    let quad_block = space
        .face_dofs(fem_space::dof_manager::FaceKey::new(0, 1, 2))
        .expect("hex quad face missing");
    assert_eq!(quad_block.len(), 4, "quad face must carry (k+1)^2 = 4 dofs at k=1");

    // A boundary tet tri face: boundary entry 7 is (1-based 10 11 12).
    let btri = space
        .face_dofs(fem_space::dof_manager::FaceKey::new(9, 10, 11))
        .expect("boundary tri face missing");
    assert_eq!(btri.len(), 3);
}

#[test]
fn element_slot_counts_match_reference_elements() {
    use fem_element::raviart_thomas::{HexRTk, TetRTk};
    use fem_element::VectorReferenceElement;

    let mesh = load("data/d393_mixed_hex_tet.mesh");
    for k in 0..=3u8 {
        let space = HDivSpace::new(mesh.clone(), k);
        let hex_ref = HexRTk::new(k as usize);
        let tet_ref = TetRTk::new(k as usize);
        for e in 0..mesh.n_elements() as u32 {
            let expect = match mesh.element_type(e) {
                fem_mesh::element_type::ElementType::Hex8 => hex_ref.n_dofs(),
                _ => tet_ref.n_dofs(),
            };
            assert_eq!(
                space.element_dofs(e).len(),
                expect,
                "k={k} elem {e}: space slot count must equal the assembly element n_dofs"
            );
        }
    }
}

#[test]
fn shared_tet_face_is_conforming() {
    let mesh = load("data/d393_mixed_hex_tet.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);

    let shared_tri = fem_space::dof_manager::FaceKey::new(8, 9, 10);
    let block = space.face_dofs(shared_tri).expect("shared face missing");
    let mut a: Vec<_> = space.element_dofs(1).to_vec();
    let mut b: Vec<_> = space.element_dofs(2).to_vec();
    a.sort_unstable();
    b.sort_unstable();

    // Both tets reference exactly the shared face block in common — no more
    // (no accidental extra sharing) and no less (no duplicate face dofs).
    let mut inter: Vec<_> = a.iter().copied().collect();
    inter.retain(|d| b.binary_search(d).is_ok());
    let mut block_sorted = block.clone();
    block_sorted.sort_unstable();
    assert_eq!(inter, block_sorted, "shared face dofs must be exactly the common set");

    // The whole block lies inside each element's slot table.
    for &d in &block {
        assert!(a.binary_search(&d).is_ok(), "tet 1 misses shared dof {d}");
        assert!(b.binary_search(&d).is_ok(), "tet 2 misses shared dof {d}");
    }
}
