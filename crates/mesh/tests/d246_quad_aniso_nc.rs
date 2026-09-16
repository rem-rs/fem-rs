//! D246 — Quad4 ANISOTROPIC nonconforming refinement (D229) and the
//! "MFEM NC mesh v1.0" writer (D231) vs MFEM 4.10.
//!
//! References probed with `$HOME/work/d246/d246_aniso_probe.cpp` (WSL tree
//! `mfem410_ser`, counts recorded 2026-09-15): an 8×4×… Cartesian Quad4 mesh
//! refined with `Mesh::GeneralRefinement(refs, -1, nclimit)` where the
//! marking **and the per-element ref_type** are order-independent geometric
//! predicates on the centroid (`cx, cy` scaled by 32 are exact integers):
//!
//! ```text
//! mark iff (ix + iy) % 3 == 0,  k = ((ix - iy) % 3 + 3) % 3,
//! rtype = 7 (iso) if k == 0, 1 (X) if k == 1, 2 (Y) if k == 2
//! ```
//!
//! 8×8 grid, 8 rounds (`$HOME/work/d246/unit_aniso8_ncl1.txt`):
//! ```text
//! nclimit = 1:  NE 64 -> 103 -> 133 -> 208 -> 313 -> 469 -> 868 -> 1879 -> 4774
//!               NV 81 -> 150 -> 198 -> 279 -> 414 -> 612 -> 1137 -> 2418 -> 5823
//! nclimit = 0:  NE 64 -> 103 -> 121 -> 139 -> 157 -> 193 -> 265 -> 409 -> 697
//!               NV 81 -> 150 -> 186 -> 222 -> 258 -> 330 -> 474 -> 762 -> 1338
//! ```
//! 4×4 grid (`unit_aniso_ncl1.txt`):
//! ```text
//! nclimit = 1:  NE 16 -> 23 -> 27 -> 29,  NV 25 -> 38 -> 44 -> 46
//! ```
//!
//! The writer is validated end-to-end by the toys: `mandel`/`mondrian` (iso
//! and `-a`) save `MFEM NC mesh v1.0` files **byte-identical** to MFEM
//! `Mesh::Save` (mandel iso 926,121 B / -a 808,514 B; mondrian iso 6,827 B /
//! -a 5,885 B), and MFEM 4.10 reads them back without warnings.

use fem_mesh::amr::nc_quad_tree::NcQuadTree;
use fem_mesh::amr::{general_refinement_quad_aniso, limit_nc_level_quad_aniso, refine_uniform};
use fem_mesh::Mesh;

/// Geometric marking rule of the C++ probe: mark iff `(ix+iy)%3 == 0`
/// (works on any quad mesh whose coordinates are dyadic multiples of 1/32).
fn probe_marked(mesh: &Mesh<2>) -> Vec<u32> {
    let mut out = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for &n in ns {
            let c = mesh.coords_of(n);
            sx += c[0];
            sy += c[1];
        }
        let cx = sx / 4.0;
        let cy = sy / 4.0;
        let ix = (cx * 32.0).round() as i64;
        let iy = (cy * 32.0).round() as i64;
        if (ix + iy) % 3 == 0 {
            out.push(e);
        }
    }
    out
}

/// Geometric marking + ref_type rule of the C++ probe (mixed X/Y/XY).
fn probe_marks(mesh: &Mesh<2>) -> Vec<(u32, u8)> {
    probe_marked(mesh)
        .into_iter()
        .map(|e| {
            let ns = mesh.elem_nodes(e);
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            for &n in ns {
                let c = mesh.coords_of(n);
                sx += c[0];
                sy += c[1];
            }
            let ix = (sx / 4.0 * 32.0).round() as i64;
            let iy = (sy / 4.0 * 32.0).round() as i64;
            let k = ((ix - iy) % 3 + 3) % 3;
            (e, if k == 0 { 7u8 } else if k == 1 { 1u8 } else { 2u8 })
        })
        .collect()
}

/// One probe round through the `NcQuadTree` (MFEM `NCMesh` replica):
/// returns the extracted leaf mesh after the batch.
fn tree_round(tree: &mut NcQuadTree, nclimit: u32) -> Mesh<2> {
    let mesh = tree.extract_mesh();
    let marks = probe_marks(&mesh);
    let marks: Vec<(usize, u8)> =
        marks.iter().map(|&(e, rt)| (e as usize, rt)).collect();
    tree.general_refinement(&marks, nclimit);
    tree.extract_mesh()
}

/// One probe round through the stateless batch API, threading the `Iso` flag.
fn stateless_round(
    mesh: &Mesh<2>,
    iso: bool,
    nclimit: u32,
) -> (Mesh<2>, bool) {
    let marks = probe_marks(mesh);
    let (m, iso, _) = general_refinement_quad_aniso(mesh, &marks, nclimit, iso, None);
    (m, iso)
}

#[test]
fn d246_aniso_tree_counts_8x8_nclimit_1() {
    let mesh = Mesh::<2>::make_cartesian_2d(8, 8, 1.0, 1.0);
    let mut tree = NcQuadTree::from_mesh(&mesh);
    assert_eq!(tree.leaf_count(), 64);
    assert!(tree.is_iso());

    // MFEM 4.10 d246_aniso_probe -nx 8 -ny 8 (nclimit = 1).
    let ne_ref = [103, 133, 208, 313, 469, 868, 1879, 4774];
    let nv_ref = [150, 198, 279, 414, 612, 1137, 2418, 5823];

    for r in 0..8 {
        let m = tree_round(&mut tree, 1);
        assert_eq!(m.n_elems(), ne_ref[r], "NE mismatch at round {r}");
        assert_eq!(
            tree.leaf_vertex_count(),
            nv_ref[r],
            "NV mismatch at round {r}"
        );
    }
    // A batch with X/Y refinements happened in round 0 (k == 1/2 marks):
    // MFEM `NCMesh::Iso` must be cleared.
    assert!(!tree.is_iso());
}

#[test]
fn d246_aniso_stateless_counts_8x8() {
    let mesh = Mesh::<2>::make_cartesian_2d(8, 8, 1.0, 1.0);

    // nclimit = 1 (with LimitNCLevel propagation, directional once Iso broke).
    let ne_ref = [103, 133, 208, 313, 469, 868, 1879, 4774];
    let nv_ref = [150, 198, 279, 414, 612, 1137, 2418, 5823];
    let (mut current, mut iso) = (mesh.clone(), true);
    for r in 0..8 {
        let (m, i) = stateless_round(&current, iso, 1);
        current = m;
        iso = i;
        assert_eq!(current.n_elems(), ne_ref[r], "NE mismatch at round {r}");
        assert_eq!(current.n_nodes(), nv_ref[r], "NV mismatch at round {r}");
    }
    assert!(!iso);

    // nclimit = 0 (no propagation).
    let ne_ref0 = [103, 121, 139, 157, 193, 265, 409, 697];
    let nv_ref0 = [150, 186, 222, 258, 330, 474, 762, 1338];
    let (mut current, mut iso) = (mesh.clone(), true);
    for r in 0..8 {
        let (m, i) = stateless_round(&current, iso, 0);
        current = m;
        iso = i;
        assert_eq!(current.n_elems(), ne_ref0[r], "NE mismatch at round {r} (ncl 0)");
        assert_eq!(current.n_nodes(), nv_ref0[r], "NV mismatch at round {r} (ncl 0)");
    }
}

#[test]
fn d246_aniso_tree_counts_4x4_nclimit_1() {
    let mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    let mut tree = NcQuadTree::from_mesh(&mesh);

    // MFEM 4.10 d246_aniso_probe (4×4, nclimit = 1) — the sequence saturates
    // after three rounds (marked = 0).
    let ne_ref = [23, 27, 29];
    let nv_ref = [38, 44, 46];
    for r in 0..3 {
        let m = tree_round(&mut tree, 1);
        assert_eq!(m.n_elems(), ne_ref[r], "NE mismatch at round {r}");
        assert_eq!(tree.leaf_vertex_count(), nv_ref[r]);
    }
    // Saturated: an empty batch must leave the tree untouched (MFEM
    // `NonconformingRefinement` early-return).
    let before = tree.leaf_count();
    tree.general_refinement(&[], 1);
    assert_eq!(tree.leaf_count(), before);
}

/// Y splits produce MFEM's two children (bottom/top halves) and the
/// directional limit sees the deep vertical split of the coarse neighbour.
#[test]
fn d246_aniso_split_shapes_and_directional_limit() {
    use fem_mesh::amr::nc_quad_tree::REF_Y;

    let mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
    // 2×1 grid: two 0.5-wide cells; the shared edge is VERTICAL.
    let mut tree = NcQuadTree::from_mesh(&mesh);

    // Y-split the left element: children [n0, n1, m12, m30] (bottom) and
    // [m30, m12, n2, n3] (top) — MFEM RefineElement Y branch.  The 2×1 grid
    // cells are 0.5×1.0, so the children are 0.5×0.5.
    tree.refine(&[(0, REF_Y)]);
    assert_eq!(tree.leaf_count(), 3);
    let m = tree.extract_mesh();
    assert_eq!(m.n_elems(), 3);
    let mut halves = 0;
    let mut wholes = 0;
    for e in 0..m.n_elems() as u32 {
        let ns = m.elem_nodes(e);
        let ys: Vec<f64> = ns.iter().map(|&n| m.coords_of(n)[1]).collect();
        let h = ys.iter().cloned().fold(f64::MIN, f64::max)
            - ys.iter().cloned().fold(f64::MAX, f64::min);
        if (h - 0.5).abs() < 1e-12 {
            halves += 1;
        } else if (h - 1.0).abs() < 1e-12 {
            wholes += 1;
        }
    }
    assert_eq!((halves, wholes), (2, 1));

    // Y-split the two children again: the shared vertical edge now carries a
    // level-2 split chain on the right element → its `splits[1]` exceeds
    // nc_limit = 1 and the (Iso-clearing) directional limit must queue a
    // Y (ref_type 2) split for exactly that element.
    let marks: Vec<(usize, u8)> = vec![(0usize, REF_Y), (1usize, REF_Y)];
    tree.refine(&marks);
    let refs = tree.get_limit_refinements(1);
    assert!(!tree.is_iso());
    assert_eq!(refs.len(), 1, "expected exactly the coarse neighbour: {refs:?}");
    assert_eq!(refs[0].1, 2, "violation is vertical → Y split, got {refs:?}");
}

/// Directional limit propagation on the stateless API: after X-splitting one
/// element and (in a second batch) its two children, the coarse neighbour
/// violates the limit **vertically** — `limit_nc_level_quad_aniso` must emit
/// ref_type 2 (Y) for it, and 7 while the mesh is still iso.
#[test]
fn d246_aniso_stateless_directional_limit() {
    // 2×1 grid, Y-split element 0 twice so the shared vertical edge of
    // element 1 carries a level-2 split chain.
    let mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
    let (m1, iso1, _) = general_refinement_quad_aniso(
        &mesh,
        &[(0, 2)], // Y-split element 0
        0,         // no limit propagation
        true,
        None,
    );
    assert!(!iso1, "a Y split must clear Iso");

    // Children of element 0 are the leaves with y-extent 0.5 (the 2×1 grid
    // cells are 0.5×1.0; the untouched neighbour is 0.5×1.0).
    let mut children = Vec::new();
    for e in 0..m1.n_elems() as u32 {
        let ns = m1.elem_nodes(e);
        let ys: Vec<f64> = ns.iter().map(|&n| m1.coords_of(n)[1]).collect();
        let h = ys.iter().cloned().fold(f64::MIN, f64::max)
            - ys.iter().cloned().fold(f64::MAX, f64::min);
        if (h - 0.5).abs() < 1e-12 {
            children.push(e);
        }
    }
    assert_eq!(children.len(), 2);
    let (m2, _, _) = general_refinement_quad_aniso(
        &m1,
        &children.iter().map(|&e| (e, 2u8)).collect::<Vec<_>>(),
        0, // no limit propagation
        iso1,
        None,
    );

    // Now the right element violates nc_limit = 1 vertically.
    let refs = limit_nc_level_quad_aniso(&m2, 1, false);
    assert_eq!(refs.len(), 1, "expected exactly the coarse neighbour: {refs:?}");
    assert_eq!(refs[0].1, 2, "violation is vertical → Y split, got {refs:?}");

    // While Iso is still true the same violation forces an iso (7) split.
    let refs_iso = limit_nc_level_quad_aniso(&m2, 1, true);
    assert_eq!(refs_iso.len(), 1);
    assert_eq!(refs_iso[0].1, 7);
}

/// D231: the writer emits the "MFEM NC mesh v1.0" sections with MFEM's exact
/// framing (see `NCMesh::Print`, ncmesh.cpp:6349, and the `mfem_mesh_end`
/// trailer of `Mesh::Printer`).
#[test]
fn d246_nc_v10_writer_structure() {
    let mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    let mut tree = NcQuadTree::from_mesh(&mesh);
    // One aniso batch: 2 X-splits, 1 Y-split, 1 iso split (leaves 16 → 21).
    tree.refine(&[(0, 1), (1, 2), (2, 7), (3, 3)]);

    let s = tree.print_mfem_nc_v10();
    assert!(s.starts_with("MFEM NC mesh v1.0\n\n"));
    assert!(s.contains("\n# NCMesh supported geometry types:\n"));
    assert!(s.contains("\ndimension\n2\n"));
    assert!(s.contains("\n# rank attr geom ref_type nodes/children\nelements\n"));
    assert!(s.contains("\n# attr geom nodes\nboundary\n"));
    assert!(s.contains("\n# vert_id p1 p2\nvertex_parents\n"));
    assert!(s.contains("\n# root element orientation\nroot_state\n"));
    assert!(s.contains("\n# top-level node coordinates\ncoordinates\n25\n2\n"));
    assert!(s.ends_with("\nmfem_mesh_end\n"));

    // X/Y splits cleared the Iso flag; a refined element carries ref_type 1/2.
    let mut has_x = false;
    for line in s.lines() {
        if line.starts_with("-1 1 3 1 ") {
            has_x = true;
        }
    }
    assert!(has_x, "expected a ref_type 1 (X) tree element");

    // Leaf attribute updates flow into the elements section
    // (leaf line = "0 <attr> 3 0 <nodes>").
    tree.set_leaf_attribute(0, 42);
    let s = tree.print_mfem_nc_v10();
    assert!(s.contains("\n0 42 3 0 "), "leaf attr not propagated");
}

/// D160 guard: the iso path through the tree reproduces the round-39 counts
/// (`1024, 2254, 5884, 16006` for the mandel start grid is exercised by the
/// toy; here the equivalent uniform-refined 4×4 probe of d160 stays intact
/// through the stateless iso API, which the tree must agree with).
#[test]
fn d246_iso_counts_unchanged() {
    let mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);

    // d160 reference (iso-only marking, nclimit = 1).
    let ne_ref = [31, 91, 244, 553, 1228, 3046];
    let mut current = mesh.clone();
    for r in 0..6 {
        let marked = probe_marked(&current);
        // All-iso batch (ref_type 7): Iso stays true.
        let (m, iso, _) =
            general_refinement_quad_aniso(&current, &marked.iter().map(|&e| (e, 7u8)).collect::<Vec<_>>(), 1, true, None);
        current = m;
        assert!(iso, "iso-only batches must not clear Iso");
        assert_eq!(current.n_elems(), ne_ref[r], "iso NE mismatch at round {r}");
    }

    // The tree agrees with the stateless path on the same iso batches.
    let mut tree = NcQuadTree::from_mesh(&Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0));
    for r in 0..6 {
        let m = tree.extract_mesh();
        let marked = probe_marked(&m);
        let refs: Vec<(usize, u8)> =
            marked.iter().map(|&e| (e as usize, 7u8)).collect();
        tree.general_refinement(&refs, 1);
        let m = tree.extract_mesh();
        assert_eq!(m.n_elems(), tree.leaf_count());
        assert_eq!(tree.leaf_count(), ne_ref[r], "tree iso NE mismatch at round {r}");
    }
    assert!(tree.is_iso(), "iso-only batches must keep Iso true");
}

/// The initial 3-level uniform refinement the toys perform must yield the
/// same root grid MFEM's `UniformRefinement` produces (16 → 1024 quads,
/// 1089 vertices), since the tree roots inherit its element/vertex order.
#[test]
fn d246_root_grid_after_uniform_refinement() {
    let mut mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    for _ in 0..3 {
        mesh = refine_uniform(&mesh);
    }
    assert_eq!(mesh.n_elems(), 1024);
    assert_eq!(mesh.n_nodes(), 1089);
    let tree = NcQuadTree::from_mesh(&mesh);
    assert_eq!(tree.leaf_count(), 1024);
    assert_eq!(tree.leaf_vertex_count(), 1089);
    // Root coordinates section = the 1089 top-level vertices.
    let s = tree.print_mfem_nc_v10();
    assert!(s.contains("\ncoordinates\n1089\n2\n"));
}
