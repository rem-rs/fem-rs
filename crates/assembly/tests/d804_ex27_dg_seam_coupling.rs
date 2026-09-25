//! D799-1 (round 74) — the ex27 seam: DG coupling across a *folded periodic*
//! seam, and the periodic-merged mesh machinery on a quad mesh.
//!
//! MFEM ex27 `GenerateSerialMesh` stitches the x=±1 ends of its flat mesh
//! (`v2v` + `RemoveUnusedVertices`), so the seam edges are ordinary interior
//! faces and the DG (L2) assembly couples across them.  fem-rs mirrors the
//! stitch with `Mesh::make_periodic` (D62 periodic-quotient machinery): the
//! connectivity is merged, the seam boundary faces are dropped, and the
//! per-element high-order geometry keeps its pre-merge corners (MFEM's
//! discontinuous nodal GF semantics — `geometry_nodes != element_nodes`).
//!
//! Before D799-1 the ex27 mesh stayed unfolded with the seam as bdr tags
//! 5/6: the DG interior-face list never saw the seam ⇒ zero coupling, and
//! the `-dg` first residual sat at ~0.6× the C++ value at every `-rs`.

use fem_assembly::dg::dg_base::ref_elem_vol;
use fem_assembly::{DgAssembler, InteriorFaceList};
use fem_mesh::{ElementType, Mesh, topology::MeshTopology};
use fem_space::L2Space;

/// 3×1 strip of quads whose x=0 and x=3 ends carry seam tags 5/6.  After
/// `make_periodic` the ends merge into ONE interior face that couples the
/// first and last quad — a pair no other face couples.
fn strip_mesh() -> Mesh<2> {
    let c: Vec<f64> = [
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],
    ].iter().flat_map(|p| p.iter().copied()).collect();
    let q: Vec<u32> = [
        [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
    ].iter().flat_map(|q| q.iter().copied()).collect();
    // bottom tag 1, top tag 2, left seam tag 5, right seam tag 6
    let bf: Vec<([u32; 2], i32)> = vec![
        ([0, 1], 1), ([1, 2], 1), ([2, 3], 1),
        ([7, 6], 2), ([6, 5], 2), ([5, 4], 2),
        ([0, 4], 5), ([3, 7], 6),
    ];
    let fc: Vec<u32> = bf.iter().flat_map(|(e, _)| e.iter().copied()).collect();
    let ft: Vec<i32> = bf.iter().map(|(_, t)| *t).collect();
    Mesh::<2>::uniform(c, q, vec![1; 3], ElementType::Quad4, fc, ft, ElementType::Line2)
}

#[test]
fn d804_periodic_merge_folds_the_seam_into_an_interior_face() {
    let mesh = strip_mesh()
        .make_periodic(&[(5, 6, [3.0, 0.0])], 1e-9)
        .expect("make_periodic");
    // 8 nodes − 2 merged = 6; boundary faces: 6 (tags 1/2), the two seam
    // faces are gone.
    assert_eq!(mesh.n_nodes(), 6, "merged node count");
    assert_eq!(mesh.n_boundary_faces(), 6, "seam faces dropped");

    // The periodic-quotient H1 count matches the merged connectivity (used
    // by the ex27 H1 path); the DG-relevant part is the interior-face list:
    // the middle edges {1,5}/{2,6} plus the folded seam {0,4}.
    let ifl = InteriorFaceList::build(&mesh);
    assert_eq!(ifl.faces.len(), 3, "two middle edges + the folded seam");
    // Post-compaction ids: kept nodes old 0,1,2,4,5,6 renumber to 0..5, so the
    // folded seam edge (old {0,4}) is {0,3} in the merged numbering.
    let seam = ifl.faces.iter().find(|f| {
        let mut n = f.face_nodes.clone();
        n.sort_unstable();
        n == vec![0, 3]
    });
    assert!(seam.is_some(), "the seam edge must be an interior face");
    // The seam couples the FIRST and LAST quad — the pair no other face
    // couples (the middle edges couple (0,1) and (1,2)).
    let seam = seam.unwrap();
    let pair = (seam.elem_left.min(seam.elem_right), seam.elem_left.max(seam.elem_right));
    assert_eq!(pair, (0, 2), "the seam couples quad 0 and quad 2");
}

#[test]
fn d804_dg_assembles_nonzero_coupling_across_the_folded_seam() {
    let mesh = strip_mesh()
        .make_periodic(&[(5, 6, [3.0, 0.0])], 1e-9)
        .expect("make_periodic");
    let space = L2Space::new(mesh.clone(), 1);
    let ifl = InteriorFaceList::build(&mesh);
    let mat = DgAssembler::assemble_dg(&space, &ifl, 1.0, -1.0, 4.0, 2, Some(&[0]));

    // L2 Q1 on 3 quads: 12 dofs, 4 per element (element-local blocks).
    let el = |e: usize, space: &L2Space<Mesh<2>>| -> Vec<usize> {
        space.element_dofs(e as u32).iter().map(|&d| d as usize).collect()
    };
    let e0 = el(0, &space);
    let e2 = el(2, &space);

    // Count entries coupling quad 0 and quad 2 through the seam face.
    let mut seam_entries = 0usize;
    let mut seam_max_abs = 0.0f64;
    for &dr in e0.iter() {
        for &dc in e2.iter() {
            let v = mat.get(dr, dc);
            if v != 0.0 { seam_entries += 1; seam_max_abs = seam_max_abs.max(v.abs()); }
            let v = mat.get(dc, dr);
            if v != 0.0 { seam_entries += 1; seam_max_abs = seam_max_abs.max(v.abs()); }
        }
    }
    assert!(seam_entries > 0, "DG must couple across the folded seam");
    assert!(seam_max_abs > 1e-3, "seam coupling must be O(1), got {seam_max_abs}");
}

#[test]
fn d804_unfolded_seam_has_no_interior_face() {
    // The pre-fix topology: the same strip WITHOUT the merge — the seam is
    // two bdr faces and no interior face couples quad 0 to quad 2 (the red
    // state of the ex27 `-dg` assembly).
    let mesh = strip_mesh();
    let ifl = InteriorFaceList::build(&mesh);
    assert_eq!(ifl.faces.len(), 2, "only the two middle edges");
    assert!(ifl.faces.iter().all(|f| {
        let (a, b) = (f.elem_left.min(f.elem_right), f.elem_left.max(f.elem_right));
        (a, b) != (0, 2)
    }), "no quad0–quad2 coupling without the merge");
}

/// Keep the import honest: the reference-element hook the DG assembler uses
/// (same one the ex27 DG path exercises through `ref_elem_vol`).
#[test]
fn d804_reference_element_available_for_quad() {
    let re = ref_elem_vol(ElementType::Quad4, 1);
    assert!(re.n_dofs() >= 4);
}
