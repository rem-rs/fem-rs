//! D840-1 (round 123 lane_per): the tet4 / prism6 refinement paths must
//! transport `make_periodic`'s per-element own-side geometry snapshot.
//!
//! # Witness (round-121, pro-fluid Cyclic cut, recorded in
//! `docs/fluid-openfoam-parity-plan.md` Appendix G)
//!
//! After a `make_periodic` fold, seam-adjacent elements' geometry measured
//! volume 1.25 ≠ 1.0 with flipped Jacobians.  At this round's HEAD the live
//! form of the debt is one level deeper: the *fold itself* is healthy (the
//! 2026-09-10 order-1 snapshot, `simplex.rs::periodic_geometry_snapshot`,
//! keeps every seam element its own side — pinned by
//! `make_periodic_preserves_per_element_geometry`), but the tet/prism
//! refinement kernels did not recognize the snapshot's *shared-row* layout
//! (`l2_p1_tet_geometry`/`l2_p1_prism_geometry` only match the fresh
//! element-major identity rows a file read produces), so a refined periodic
//! tet mesh dropped the snapshot and rebuilt child geometry from the folded
//! vertex frame: on a 2×2×2 tet cube folded on all three axes, **192 of 384
//! children had det J = −1/64** (pre-fix evidence, red below).
//!
//! # MFEM 4.10 reference (probe `tmp/rr123/lane_per/probe_periodic_geom.cpp`,
//! full dumps WSL `~/work/rr123/mfem_{tetx,prismz,quad}.txt`)
//!
//! `Mesh::MakePeriodic` (mesh.cpp:6205) copies the mesh, materializes the
//! nodal `Nodes` grid function **before** the `v2v` renumbering
//! (`SetCurvature(order, /*discont=*/true)` → `L2_T1_<dim>D_P1`,
//! `NDofs = NE·nve`), renumbers the element/boundary vertices, and removes
//! the unused ones — the geometry is per-element own-side forever:
//!
//! * `tetx` (2×2×2 tet cube, x-fold): coarse `L2_T1_3D_P1 ND 192`, all 48
//!   `det J = +0.125`; after `UniformRefinement`: `NE 384`,
//!   `L2_T1_3D_P1 ND 1536`, **all 384 children `det J = +0.015625`**; own-side
//!   Σx over the coarse/fine geometry rows = 96 / 768.  fem-rs matches these
//!   bitwise (`tet_x_fold_refine_matches_mfem_tetx`).
//! * `quad` (4×4, x-fold): coarse/fine `L2_T1_2D_P1 ND 64/256`, all
//!   `det J = +1/16, +1/64`.
//!
//! Every Cartesian-wedge fold probed (`prismz`: 1×1×2 column, z-fold) aborts
//! inside MFEM itself — upstream face-hash collisions — so the prism pin
//! holds the transport to the source-level semantics without a C++ numeric
//! oracle.  The same is true of multi-axis folds of coarse boxes (tet/hex);
//! fem-rs accepts those and the pin checks the transport invariants.

use fem_io::mfem::{read_mfem, write_mfem_nodes, NodesSpace};
use fem_mesh::extrusion::extrude_tri3_to_prisms;
use fem_mesh::{element_jacobian_at, Mesh, MeshTopology};

/// det(J) at every element's reference center, through the per-element
/// geometry table (the seam elements' own side).
fn center_dets_3d(m: &Mesh<3>) -> Vec<f64> {
    let topo: &dyn MeshTopology = m;
    (0..m.n_elems() as u32)
        .map(|e| {
            let (jac, _) = element_jacobian_at(topo, e, &[0.5, 0.5, 0.5], 3);
            jac[(0, 0)] * (jac[(1, 1)] * jac[(2, 2)] - jac[(1, 2)] * jac[(2, 1)])
                - jac[(0, 1)] * (jac[(1, 0)] * jac[(2, 2)] - jac[(1, 2)] * jac[(2, 0)])
                + jac[(0, 2)] * (jac[(1, 0)] * jac[(2, 1)] - jac[(1, 1)] * jac[(2, 0)])
        })
        .collect()
}

/// Σ of each coordinate over every element's own-side geometry row.
fn geometry_coord_sums(m: &Mesh<3>) -> [f64; 3] {
    let g = m.geometry.as_ref().expect("own-side geometry snapshot");
    let mut s = [0.0_f64; 3];
    for &d in &g.conn {
        let c = &g.coords[d as usize * 3..d as usize * 3 + 3];
        for k in 0..3 {
            s[k] += c[k];
        }
    }
    s
}

fn assert_det(det: f64, want: f64, what: &str) {
    assert!(
        (det - want).abs() <= 1e-15 && det > 0.0,
        "{what}: det J = {det:.17e}, want +{want:.17e}"
    );
}

/// x-folded 2×2×2 tet cube — the MFEM `tetx` oracle case.
///
/// Boundary attrs of `unit_cube_tet`: x=0 → 5, x=1 → 6 (y: 3/2, z: 1/4 —
/// probe `periodic_pairs_3d` classification, round-123 scratch probe).
#[test]
fn tet_x_fold_refine_matches_mfem_tetx() {
    let cube = Mesh::<3>::unit_cube_tet(2);
    assert_eq!(cube.n_elems(), 48);
    let pm = cube
        .make_periodic(&[(5, 6, [1.0, 0.0, 0.0])], 1e-10)
        .expect("x-fold");

    // Coarse: MFEM `L2_T1_3D_P1 ND 192` — the snapshot's connectivity is the
    // same element-major row count (48·4) addressing the *pre-merge* 27-node
    // table; every det J = +0.125, Σ own-side rows = (96, 96, 96).
    assert_eq!(pm.n_elems(), 48);
    let g = pm.geometry.as_ref().expect("periodic snapshot");
    assert_eq!((g.order, g.nodes_per_elem), (1, 4));
    assert_eq!(g.conn.len(), 48 * 4, "row count = MFEM L2 NDofs");
    assert_eq!(g.n_nodes, 27, "rows address the pre-merge node table");
    for (e, det) in center_dets_3d(&pm).into_iter().enumerate() {
        assert_det(det, 0.125, &format!("coarse elem {e}"));
    }
    let s = geometry_coord_sums(&pm);
    assert_eq!(s, [96.0, 96.0, 96.0], "coarse own-side Σ (MFEM tetx)");

    // The folded dof (vertex) table only keeps x ∈ {0, 0.5}.
    for n in 0..pm.n_nodes() {
        let c = pm.coords_of(n as u32);
        assert!(c[0] <= 0.5 + 1e-12, "folded vertex x = {}", c[0]);
    }

    // Fine: MFEM `NE 384, L2_T1_3D_P1 ND 1536`, all 384 children
    // det J = +0.015625, Σ own-side rows = (768, 768, 768).
    let fine = fem_mesh::refine_uniform_3d(&pm);
    assert_eq!(fine.n_elems(), 384);
    let gf = fine.geometry.as_ref().expect("refined own-side table");
    assert_eq!((gf.order, gf.nodes_per_elem, gf.n_nodes), (1, 4, 384 * 4));
    for (e, det) in center_dets_3d(&fine).into_iter().enumerate() {
        assert_det(det, 0.015625, &format!("fine elem {e}"));
    }
    let sf = geometry_coord_sums(&fine);
    assert_eq!(sf, [768.0, 768.0, 768.0], "fine own-side Σ (MFEM tetx)");
}

/// The round-121 witness family: 2×2×2 tet cube folded on **all three axes**.
/// Pre-fix: 192/384 refined children flipped (det J = −1/64).  MFEM itself
/// aborts on this fold (upstream face-hash collision), so the pin checks the
/// transport invariants: every child own-side and positive, total volume 1.
#[test]
fn tet_3axis_fold_refine_stays_own_side() {
    let cube = Mesh::<3>::unit_cube_tet(2);
    // tag 5 at x=0, 6 at x=1, 3 at y=0, 4 at y=1, 1 at z=0, 2 at z=1
    // (probe pairing, round-123 scratch probe).
    let pm = cube
        .make_periodic(
            &[
                (5, 6, [1.0, 0.0, 0.0]),
                (3, 4, [0.0, 1.0, 0.0]),
                (1, 2, [0.0, 0.0, 1.0]),
            ],
            1e-10,
        )
        .expect("3-axis fold");
    assert_eq!(pm.n_elems(), 48);
    assert!(pm.geometry.is_some());
    for (e, det) in center_dets_3d(&pm).into_iter().enumerate() {
        assert_det(det, 0.125, &format!("3-axis coarse elem {e}"));
    }

    let fine = fem_mesh::refine_uniform_3d(&pm);
    assert_eq!(fine.n_elems(), 384);
    let dets = center_dets_3d(&fine);
    for (e, det) in dets.iter().copied().enumerate() {
        assert_det(det, 0.015625, &format!("3-axis fine elem {e}"));
    }
    let volume: f64 = dets.iter().map(|d| d / 6.0).sum();
    assert!((volume - 1.0).abs() < 1e-12, "torus volume {volume}");
}

/// A *partial* (non-conforming) refinement of a folded tet mesh must keep the
/// own-side rows too — the transport is not gated on the uniform case.
#[test]
fn tet_partial_refine_child_rows_own_side() {
    let cube = Mesh::<3>::unit_cube_tet(2);
    let pm = cube
        .make_periodic(&[(5, 6, [1.0, 0.0, 0.0])], 1e-10)
        .expect("x-fold");
    let (fine, _edge_c, _face_c) = fem_mesh::refine_nonconforming_3d(&pm, &[0], None);
    assert_eq!(fine.n_elems(), 48 - 1 + 8);
    assert!(fine.geometry.is_some(), "partial refine keeps the snapshot");
    // Refined element's 8 children: det J = +1/64; the 39 carried elements:
    // +1/8.  Total volume must stay 1.
    let mut n_fine = 0usize;
    let mut volume = 0.0_f64;
    for (e, det) in center_dets_3d(&fine).into_iter().enumerate() {
        if (det - 0.015625).abs() < 1e-15 {
            n_fine += 1;
        } else {
            assert_det(det, 0.125, &format!("carried elem {e}"));
        }
        assert!(det > 0.0, "elem {e} flipped: {det}");
        volume += det / 6.0;
    }
    assert_eq!(n_fine, 8, "eight children of the marked element");
    assert!((volume - 1.0).abs() < 1e-12, "volume {volume}");
}

/// z-folded 1×1×2 wedge column.
///
/// No MFEM oracle exists for this case: every Cartesian-wedge fold we probed
/// aborts inside MFEM 4.10 itself (the prismz probe — even the accepted-shape
/// 1×1 base — collides folded vertical quad-face hashes: "Interior
/// quadrilateral face found connecting elements 0, 1 and 2").  The pin holds
/// the transport to the same source-level MFEM semantics as the tet oracle
/// (`MakePeriodic` = `SetCurvature(order, /*discont=*/true)` *before* the
/// `v2v` renumbering, mesh.cpp:6205): every element keeps its own-side row
/// through the fold and through refinement, every Jacobian stays positive,
/// and the torus volume is conserved.  Wedge det J reference: reference-wedge
/// volume 1/2, so a unit-triangle-base wedge of height 1 has det J = 1.
#[test]
fn prism_z_fold_refine_keeps_own_side() {
    let column = extrude_tri3_to_prisms(&Mesh::<2>::unit_square_tri(1), 2, 2.0);
    assert_eq!(column.n_elems(), 4);
    let mut retagged = column;
    for f in 0..retagged.n_faces() {
        let nodes = retagged.bface_nodes(f as u32);
        let all = |z: f64| nodes.iter().all(|&n| (retagged.coords_of(n)[2] - z).abs() < 1e-12);
        if all(0.0) {
            retagged.face_tags[f] = 5;
        } else if all(2.0) {
            retagged.face_tags[f] = 6;
        }
    }
    let pm = retagged
        .make_periodic(&[(5, 6, [0.0, 0.0, 2.0])], 1e-10)
        .expect("z-fold");

    // Coarse: 4 wedges, per-element own-side rows over the pre-merge table.
    assert_eq!(pm.n_elems(), 4);
    let g = pm.geometry.as_ref().expect("periodic snapshot");
    assert_eq!((g.order, g.nodes_per_elem), (1, 6));
    assert_eq!(g.conn.len(), 4 * 6, "row count = MFEM L2 NDofs");
    assert_eq!(g.n_nodes, 12, "rows address the pre-merge node table");
    for (e, det) in center_dets_3d(&pm).into_iter().enumerate() {
        assert_det(det, 1.0, &format!("coarse prism {e}"));
    }
    let s = geometry_coord_sums(&pm);
    // Own-side Σ: base triangles own the x/y lattice, the z layers keep their
    // own heights (layer L wedge: Σz = 6L+3 → layers (0,0,1,1) give 24).
    assert_eq!(s, [12.0, 12.0, 24.0], "coarse own-side Σ");

    // Fine: 32 children, every det J = +0.125, torus volume 2 conserved.
    let fine = fem_mesh::refine_uniform_3d(&pm);
    assert_eq!(fine.n_elems(), 32);
    let gf = fine.geometry.as_ref().expect("refined own-side table");
    assert_eq!((gf.order, gf.nodes_per_elem, gf.n_nodes), (1, 6, 32 * 6));
    let mut volume = 0.0_f64;
    for (e, det) in center_dets_3d(&fine).into_iter().enumerate() {
        assert_det(det, 0.125, &format!("fine prism {e}"));
        volume += det * (1.0 / 2.0);
    }
    assert!((volume - 2.0).abs() < 1e-12, "torus volume {volume}");
}

/// The folded mesh must survive the MFEM file round trip: the writer's
/// default `NodesSpace::Continuous` correctly refuses the discontinuous
/// geometry (D805-3), and the explicit `Discontinuous` encoding
/// (`L2_T1_2D_P1` — what MFEM's own `Mesh::Printer` writes for a
/// `MakePeriodic` mesh) reads back with the own-side geometry intact.
#[test]
fn periodic_quad4_io_roundtrip_keeps_own_side() {
    let m = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    let pm = m
        .make_periodic(&[(4, 2, [1.0, 0.0])], 1e-10)
        .expect("x-fold");

    let mut bytes = Vec::new();
    write_mfem_nodes(&mut bytes, &pm, None, NodesSpace::Discontinuous)
        .expect("write the L2_T1 nodes section");
    let file = read_mfem(&bytes[..]).expect("read back");
    let back = file.mesh2d.expect("2-D mesh");

    assert_eq!(back.n_elems(), 16);
    let topo: &dyn MeshTopology = &back;
    for e in 0..16u32 {
        let (jac, _) = element_jacobian_at(topo, e, &[0.5, 0.5], 2);
        let det = jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)];
        assert_det(det, 0.0625, &format!("roundtrip elem {e}"));
    }
    // Own-side Σx over the geometry rows (MFEM quad probe: 32).
    let g = back.geometry.as_ref().expect("L2 own-side geometry");
    let sx: f64 = g.conn.iter().map(|&d| g.coords[d as usize * 2]).sum();
    assert_eq!(sx, 32.0, "roundtrip own-side Σx (MFEM quad)");
}
