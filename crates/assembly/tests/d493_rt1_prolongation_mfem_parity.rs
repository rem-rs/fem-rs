//! D494/D461 — RT1 H(div) prolongation on quads and hexes against MFEM probe
//! truth, plus the D493 pyramid RT0 oracle.
//!
//! The quad/hex truth tables were dumped from MFEM 4.10
//! (`tmp/d493/probe.cpp`: `UniformRefinement` + `fes.Update()` with
//! `MFEM_SPARSEMAT` update operator, unit-vector column extraction).  The
//! fine meshes are built from the probe's exact dumped connectivity
//! (`*_FCONN`), so the (element, slot) dof pairing resolves on both sides —
//! the same recipe as the tet RT1 test in `d468_hdiv_prolongation_mfem_parity`.
//!
//! RT1 has interior (bubble) dofs whose prolongation rows the legacy builder
//! left at zero; the MFEM-exact path fills them from the same
//! `LocalInterpolation_RT` formula (D461).  These tests pin the full matrix,
//! bubble rows included.

use std::collections::{HashMap, HashSet};

use fem_assembly::transfer::build_prolongation_hdiv;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::HDivSpace;

// Generated from tmp/d493/d493_quad_o1.txt / d493_hex_o1.txt by
// tmp/d493/gen_truth.py — do not edit.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d493/quad_o1_truth.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d493/hex_o1_truth.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d493/pyramid_o0_truth.rs"));


// ── mesh fixtures (MFEM's exact connectivity from the probe dump) ────────────

fn q(v: f64) -> i64 {
    (v * 1e9).round() as i64
}

/// MFEM `MakeCartesian2D(1, 1, QUADRILATERAL)` (coarse) and its uniform
/// refinement (fine), built from the dumped C_CONN/F_CONN/C_VERT/F_VERT.
fn mfem_quad_meshes() -> (Mesh<2>, Mesh<2>) {
    // Truth tables carry 3-component rows (z padded with 0); a 2-D mesh
    // consumes (x, y) pairs.
    let flat = |tab: &[&[f64]]| -> Vec<f64> {
        tab.iter().flat_map(|c| c[..2].to_vec()).collect()
    };
    let conn = |tab: &[&[u32]]| -> Vec<u32> { tab.iter().flat_map(|c| c.to_vec()).collect() };
    // Boundary edges of the 2x2 grid, counterclockwise (orientation only
    // feeds the face table; HDivSpace derives canonical signs from the
    // element scan).
    let face_conn: Vec<u32> = [0u32, 4, 4, 1, 1, 5, 5, 3, 3, 6, 6, 2, 2, 7, 7, 0]
        .iter()
        .copied()
        .collect();
    let coarse = Mesh::<2> {
        coords: flat(MFEM_QUAD_O1_CVERT),
        conn: conn(MFEM_QUAD_O1_CCONN),
        vertex_parents: vec![],
        elem_tags: vec![1],
        elem_type: ElementType::Quad4,
        face_conn: vec![0, 1, 1, 3, 3, 2, 2, 0],
        face_tags: vec![1; 4],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    };
    let fine = Mesh::<2> {
        coords: flat(MFEM_QUAD_O1_FVERT),
        conn: conn(MFEM_QUAD_O1_FCONN),
        vertex_parents: vec![],
        elem_tags: vec![1; 4],
        elem_type: ElementType::Quad4,
        face_conn,
        face_tags: vec![1; 8],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    };
    (coarse, fine)
}

/// MFEM `MakeCartesian3D(1, 1, 1, HEXAHEDRON)` (coarse) and its uniform
/// refinement (fine), built from the dumped C_CONN/F_CONN/C_VERT/F_VERT.
fn mfem_hex_meshes() -> (Mesh<3>, Mesh<3>) {
    const FACES: [(usize, usize, usize, usize); 6] = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
    ];
    let flat = |tab: &[&[f64]]| -> Vec<f64> { tab.iter().flat_map(|c| c.to_vec()).collect() };
    let conn = |tab: &[&[u32]]| -> Vec<u32> { tab.iter().flat_map(|c| c.to_vec()).collect() };
    // Boundary quads = faces encountered exactly once (first registrant's
    // orientation; only the vertex set matters for the space).
    let mut count: HashMap<[u32; 4], [u32; 4]> = HashMap::new();
    for t in MFEM_HEX_O1_FCONN {
        for &(a, b, c, d) in &FACES {
            let mut f = [t[a], t[b], t[c], t[d]];
            f.sort_unstable();
            count.entry(f).or_insert(f);
        }
    }
    // faces appearing once are boundary
    let mut all: HashMap<[u32; 4], usize> = HashMap::new();
    for t in MFEM_HEX_O1_FCONN {
        for &(a, b, c, d) in &FACES {
            let mut f = [t[a], t[b], t[c], t[d]];
            f.sort_unstable();
            *all.entry(f).or_insert(0) += 1;
        }
    }
    let mut face_conn = Vec::new();
    for (f, n) in all.iter() {
        if *n == 1 {
            let canon = count[f];
            face_conn.extend_from_slice(&canon);
        }
    }
    let n_bfaces = face_conn.len() / 4;
    let coarse = Mesh::<3> {
        coords: flat(MFEM_HEX_O1_CVERT),
        conn: conn(MFEM_HEX_O1_CCONN),
        vertex_parents: vec![],
        elem_tags: vec![1],
        elem_type: ElementType::Hex8,
        face_conn: vec![
            0, 1, 2, 3, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5,
        ],
        face_tags: vec![1; 6],
        face_type: ElementType::Quad4,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    };
    let fine = Mesh::<3> {
        coords: flat(MFEM_HEX_O1_FVERT),
        conn: conn(MFEM_HEX_O1_FCONN),
        vertex_parents: vec![],
        elem_tags: vec![1; 8],
        elem_type: ElementType::Hex8,
        face_conn,
        face_tags: vec![1; n_bfaces],
        face_type: ElementType::Quad4,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    };
    (coarse, fine)
}

// ── truth join (element, slot) ────────────────────────────────────────────────

/// Resolves an MFEM (element, slot) dof table to fem-rs dof ids.  The meshes
/// are built with MFEM's exact connectivity, so element order and slot
/// layouts match; a layout mismatch would surface as value mismatches below.
fn resolve_dof_map<M: MeshTopology>(
    space: &HDivSpace<M>,
    dofmap: &[(u32, u32)],
    label: &str,
) -> Vec<u32> {
    let mut out = vec![u32::MAX; dofmap.len()];
    for (d, &(e, slot)) in dofmap.iter().enumerate() {
        let dof = space.element_dofs(e)[slot as usize];
        assert!(
            out[d as usize] == u32::MAX,
            "{label}: mfem dof {d} resolved twice"
        );
        out[d as usize] = dof;
    }
    out
}

fn assert_parity(
    p: &fem_linalg::CsrMatrix<f64>,
    c2r: &[u32],
    f2r: &[u32],
    truth: &[(u32, u32, f64)],
    label: &str,
) {
    let mut checked = 0usize;
    let mut max_err = 0.0_f64;
    for &(i, j, v) in truth {
        let fi = f2r[i as usize] as usize;
        let cj = c2r[j as usize] as usize;
        let got = (p.row_ptr[fi]..p.row_ptr[fi + 1])
            .map(|k| (p.col_idx[k] as usize, p.values[k]))
            .find(|&(c, _)| c == cj)
            .unwrap_or_else(|| panic!("{label}: missing P[{fi},{cj}] for MFEM {v}"));
        max_err = max_err.max((got.1 - v).abs());
        assert!((got.1 - v).abs() <= 1e-12, "{label}: P[{fi},{cj}] = {} vs MFEM {v}", got.1);
        checked += 1;
    }
    eprintln!("{label}: {checked} MFEM entries matched, max|delta| = {max_err:.3e}");
}

/// The exact path must serve every fine dof (legacy fallback would leave the
/// bubble rows zero and report fewer located dofs).
fn assert_exact_path_served(p: &fem_linalg::CsrMatrix<f64>, n_fine_dofs: usize) {
    assert_eq!(p.nrows, n_fine_dofs);
    for r in 0..p.nrows {
        assert!(p.row_ptr[r + 1] > p.row_ptr[r], "empty prolongation row {r}");
    }
}

// ── RT1 parity (D494/D461) ────────────────────────────────────────────────────

#[test]
fn d493_quad_rt1_matches_mfem() {
    let (coarse_mesh, fine_mesh) = mfem_quad_meshes();
    let coarse_space = HDivSpace::new(coarse_mesh.clone(), 1);
    let fine_space = HDivSpace::new(fine_mesh.clone(), 1);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(stats.located_count, fine_space.n_dofs());
    assert_exact_path_served(&p, fine_space.n_dofs());
    let c2r = resolve_dof_map(&coarse_space, MFEM_QUAD_O1_CMAP, "quad o1 coarse");
    let f2r = resolve_dof_map(&fine_space, MFEM_QUAD_O1_FMAP, "quad o1 fine");
    assert_parity(&p, &c2r, &f2r, MFEM_QUAD_O1_P, "quad RT1");
}

#[test]
fn d493_hex_rt1_matches_mfem() {
    let (coarse_mesh, fine_mesh) = mfem_hex_meshes();
    let coarse_space = HDivSpace::new(coarse_mesh.clone(), 1);
    let fine_space = HDivSpace::new(fine_mesh.clone(), 1);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(stats.located_count, fine_space.n_dofs());
    assert_exact_path_served(&p, fine_space.n_dofs());
    let c2r = resolve_dof_map(&coarse_space, MFEM_HEX_O1_CMAP, "hex o1 coarse");
    let f2r = resolve_dof_map(&fine_space, MFEM_HEX_O1_FMAP, "hex o1 fine");
    assert_parity(&p, &c2r, &f2r, MFEM_HEX_O1_P, "hex RT1");
}

// ── constant-field semantics (D461 bubble columns) ────────────────────────────

/// A constant field is exactly representable in RT1, so the prolongation of
/// its coarse projection must equal the fine projection dof-for-dof — the
/// bubble/interior columns included (D461).  This is the semantic companion
/// to the bitwise matrix parity above.  Green on hexes: fem-rs's hex RT1 dof
/// values coincide with MFEM `Project_RT` (empirically `HexRTk(1)` is
/// sample-dual), so the MFEM-exact matrix is also the fem-rs-consistent
/// operator.
#[test]
fn d493_hex_rt1_constant_field_prolongs_exactly() {
    let c3 = [0.9_f64, 0.4, -1.1];
    let (coarse_mesh, fine_mesh) = mfem_hex_meshes();
    let coarse_space = HDivSpace::new(coarse_mesh, 1);
    let fine_space = HDivSpace::new(fine_mesh, 1);
    let (p, _) = build_prolongation_hdiv(&coarse_space, &fine_space);
    let x_c = coarse_space.interpolate_vector(&|_| c3.to_vec());
    let x_f = fine_space.interpolate_vector(&|_| c3.to_vec());
    let mut y = vec![0.0_f64; fine_space.n_dofs()];
    p.spmv(x_c.as_slice(), &mut y);
    for i in 0..fine_space.n_dofs() {
        assert!(
            (y[i] - x_f.as_slice()[i]).abs() <= 1e-12,
            "hex RT1: constant-flux dof {i}: P·x_c = {} vs fine projection {}",
            y[i],
            x_f.as_slice()[i]
        );
    }
}

/// Quad RT1 semantic companion — currently divergent, kept as the open
/// oracle.  The assembled matrix is bitwise MFEM (see
/// `d493_quad_rt1_matches_mfem`), but `HDivSpace::interpolate_vector`'s
/// nodal engine solves the REFERENCE dual system `W·c = samples` for quad
/// dofs, and `QuadRTk(1)` is NOT sample-dual (its interior dofs are
/// integral-moment functionals, W ≠ I) — unlike tri/tet where D33/D34 made
/// W the identity, and unlike `HexRTk(1)`.  Fem-rs quad RT1 dof values are
/// therefore `W⁻¹·(MFEM nodal samples)`, so the MFEM-exact matrix applied
/// to fem-rs dof vectors does not reproduce the fem-rs fine projection
/// (measured: dof 2: P·x_c = 0.8879165124598852 vs x_f = 0.65 for the
/// constant field (1.3, −0.7) on the refined unit square).  Closing this
/// needs the space-side flip of the quad RT1 dual convention (hdiv.rs /
/// element quad RTk dual), out of scope while `crates/space` is read-only.
#[test]
#[ignore = "quad RT1 fem-rs dof values are W^-1-samples (QuadRTk(1) not sample-dual); space-side dual flip pending (D461 sliver class, hdiv.rs read-only round 54)"]
fn d493_quad_rt1_constant_field_prolongs_exactly() {
    let c2 = [1.3_f64, -0.7];
    let (coarse_mesh, fine_mesh) = mfem_quad_meshes();
    let coarse_space = HDivSpace::new(coarse_mesh, 1);
    let fine_space = HDivSpace::new(fine_mesh, 1);
    let (p, _) = build_prolongation_hdiv(&coarse_space, &fine_space);
    let x_c = coarse_space.interpolate_vector(&|_| c2.to_vec());
    let x_f = fine_space.interpolate_vector(&|_| c2.to_vec());
    let mut y = vec![0.0_f64; fine_space.n_dofs()];
    p.spmv(x_c.as_slice(), &mut y);
    for i in 0..fine_space.n_dofs() {
        assert!(
            (y[i] - x_f.as_slice()[i]).abs() <= 1e-12,
            "quad RT1: constant-flux dof {i}: P·x_c = {} vs fine projection {}",
            y[i],
            x_f.as_slice()[i]
        );
    }
}

/// Tet RT1 on fem-rs's own uniform refinement (whose historical vertex order
/// makes corner children 1/3 mirrored — see `mfem_tet_refine` in the d468
/// test): the exact path must serve the mirrored children through the
/// order-1 slot remap instead of falling back to the legacy builder, and
/// must stay exact for fields RT1 represents exactly (constant and linear).
#[test]
fn d493_tet_rt1_own_mesh_exact_path_and_exact_fields() {
    let coarse_mesh = Mesh::<3>::unit_cube_tet(1);
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh.clone());
    let coarse_space = HDivSpace::new(coarse_mesh, 1);
    let fine_space = HDivSpace::new(fine_mesh, 1);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    // the exact path writes every fine dof exactly once; the legacy fallback
    // would leave interior rows empty and report fewer located dofs
    assert_eq!(stats.located_count, fine_space.n_dofs());
    assert_exact_path_served(&p, fine_space.n_dofs());
    for (name, f) in [
        (
            "constant",
            Box::new(|_: &[f64]| vec![0.9_f64, 0.4, -1.1]) as Box<dyn Fn(&[f64]) -> Vec<f64>>,
        ),
        (
            "linear",
            Box::new(|x: &[f64]| {
                vec![1.2 * x[0] - 0.4 * x[1] + 2.0, -0.5 * x[2] + 0.3, 0.7 * x[1]]
            }) as Box<dyn Fn(&[f64]) -> Vec<f64>>,
        ),
    ] {
        let x_c = coarse_space.interpolate_vector(&f);
        let x_f = fine_space.interpolate_vector(&f);
        let mut y = vec![0.0_f64; fine_space.n_dofs()];
        p.spmv(x_c.as_slice(), &mut y);
        for i in 0..fine_space.n_dofs() {
            assert!(
                (y[i] - x_f.as_slice()[i]).abs() <= 1e-12,
                "tet RT1 own mesh ({name}): dof {i}: P·x_c = {} vs fine projection {}",
                y[i],
                x_f.as_slice()[i]
            );
        }
    }
}

// ── D493: pyramid RT0 oracle (open — see the ignore reason) ───────────────────

/// Pyramid RT0 MFEM oracle over the mixed 6Pyr+4Tet refinement.
///
/// The truth table (`tmp/d493/d493_pyramid_o0.txt`, 89 entries) is joined by
/// sorted face-vertex coordinates like the prism oracle.  It is kept as an
/// `#[ignore]`d oracle because two independent blockers stand between it and
/// a closable parity test (round 54 route-3 findings):
///
/// 1. **MFEM upstream bug on the tet children.**  In
///    `RefinementMatrix_main` the four interior tet children of a pyramid
///    parent are keyed by their *own* geometry: they receive
///    `localP[TETRAHEDRON](emb.matrix = 0)` — the *tet identity* refinement
///    rows — mapped onto the first four coarse pyramid dofs, instead of a
///    cross-geometry interpolation of the parent pyramid field (the
///    pyr_children point matrices 6..9 that MFEM tabulates for exactly this
///    purpose are never consulted).  On top, each tet row's fifth column
///    picks up a stale value from the previous (pyramid) element's row
///    buffer: `Vector row` is refilled to length 4 by `lP.GetRow` while
///    `SetRow` iterates the 5 coarse dofs, reading `srow(4)` past the end —
///    empirically the leftover of localP[PYRAMID](5) row 4 (0.15625 = the
///    assembled P[28,4]).  Bitwise parity would mean reproducing an
///    out-of-bounds read; the pyr-child rows (0..28) are well-defined, the
///    4 tet rows (29..32) are not.
/// 2. **fem-rs pyramid dof-value convention.**  `HDivSpace::interpolate_vector`
///    routes meshes containing pyramids to the legacy canonical-moment
///    engine, while MFEM's Fuentes RT0 dofs are nodal samples — the same
///    diagonal-convention gap D33/D34 closed for tri/tet (that closure was a
///    `crates/space` change; hdiv.rs was read-only this round).  Until the
///    space convention flips, an MFEM-equal matrix would still not be the
///    fem-rs-consistent prolongation operator.
#[test]
#[ignore = "D493 open: MFEM upstream tet-child rows are tet-identity + stale-buffer artifact (probe_lf/probe_lp evidence, tmp/d493/); fem-rs pyramid dof convention (moment vs Fuentes-nodal) needs the space-side flip first"]
fn d493_pyramid_rt0_mfem_oracle() {
    // The oracle body is intentionally minimal: the truth table is kept
    // machine-readable in tmp/d493/pyramid_o0_truth.rs for the follow-up
    // round; wiring it here requires the two blockers above to clear.
    assert!(MFEM_PYRAMID_O0.len() == 89);
}

/// Sanity pin (kept active): fem-rs's own uniform pyramid refinement is
/// construction-identical to MFEM's PYRAMID branch — children, order and
/// vertex orders (`refine_pyramid5_uniform` vs mesh.cpp:10766-10855).  The
/// prolongation parity for pyramids can therefore reuse fem-rs's refined
/// meshes once the blockers above clear.
#[test]
fn d493_pyramid_refinement_matches_mfem_construction() {
    let coarse = Mesh::<3> {
        coords: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0],
        conn: vec![0, 1, 2, 3, 4],
        vertex_parents: vec![],
        elem_tags: vec![1],
        elem_type: ElementType::Pyramid5,
        face_conn: vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 0, 3, 2, 1],
        face_tags: vec![1; 5],
        face_type: ElementType::Tri3,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    };
    // MFEM's fine element table (from the probe dump, F_CONN):
    // corner pyramids (v0/v1/v2/v3), apex pyramid, inner inverted pyramid,
    // then the four base-center tets — in vertex ids over the uniform grid.
    let expected: Vec<Vec<u32>> = vec![
        vec![0, 5, 13, 8, 9],
        vec![5, 1, 6, 13, 10],
        vec![13, 6, 2, 7, 11],
        vec![8, 13, 7, 3, 12],
        vec![9, 10, 11, 12, 4],
        vec![12, 11, 10, 9, 13],
        vec![5, 9, 10, 13],
        vec![6, 10, 11, 13],
        vec![7, 11, 12, 13],
        vec![8, 12, 9, 13],
    ];
    // fem-rs's refinement: renumber fine vertices by (quantized) coordinate
    // into MFEM's dump numbering — corners keep ids 0..4, midpoints 5..12,
    // base center 13.
    let fine = fem_mesh::refine_uniform_3d(&coarse);
    let mut coords: Vec<[f64; 3]> = Vec::new();
    let mut id_of: HashMap<[i64; 3], u32> = HashMap::new();
    for v in 0..fine.n_nodes() as u32 {
        let c = fine.node_coords(v);
        let key = [q(c[0]), q(c[1]), q(c[2])];
        let next = coords.len() as u32;
        let id = *id_of.entry(key).or_insert_with(|| {
            coords.push([c[0], c[1], c[2]]);
            next
        });
        let _ = id;
    }
    // Map fem-rs fine ids to MFEM ids via coordinates: build fem-rs id →
    // coordinate-key → MFEM id (corners 0..4 then first-appearance).
    let mut mfem_of_key: HashMap<[i64; 3], u32> = HashMap::new();
    for (i, v) in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 1.0)]
        .iter()
        .enumerate()
    {
        mfem_of_key.insert([q(v.0), q(v.1), q(v.2)], i as u32);
    }
    // remaining 9 fine vertices in MFEM dump id order 5..13
    let midpoints: Vec<[f64; 3]> = vec![
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.25, 0.25, 0.5],
        [0.75, 0.25, 0.5],
        [0.75, 0.75, 0.5],
        [0.25, 0.75, 0.5],
        [0.5, 0.5, 0.0],
    ];
    for (i, v) in midpoints.iter().enumerate() {
        mfem_of_key.insert([q(v[0]), q(v[1]), q(v[2])], (5 + i) as u32);
    }
    let got: Vec<Vec<u32>> = (0..fine.n_elements() as usize)
        .map(|e| {
            fine.element_nodes(e as u32)
                .iter()
                .map(|&n| {
                    let c = fine.node_coords(n);
                    *mfem_of_key.get(&[q(c[0]), q(c[1]), q(c[2])]).expect("known vertex")
                })
                .collect()
        })
        .collect();
    assert_eq!(
        got, expected,
        "pyramid refinement children/order/vertex order must match MFEM"
    );
    let _ = (coords, id_of);
    // mixed child geometries: 6 pyramids + 4 tets
    assert_eq!(fine.element_type(0), ElementType::Pyramid5);
    assert_eq!(fine.element_type(5), ElementType::Pyramid5);
    assert_eq!(fine.element_type(6), ElementType::Tet4);
    assert_eq!(fine.element_type(9), ElementType::Tet4);
    let _ = HashSet::<u32>::new();
}
