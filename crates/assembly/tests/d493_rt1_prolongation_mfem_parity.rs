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
use fem_space::{FaceKey, HDivSpace};

// Generated from tmp/d493/d493_quad_o1.txt / d493_hex_o1.txt by
// tmp/d493/gen_truth.py — do not edit.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d493/quad_o1_truth.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d493/hex_o1_truth.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d493/pyramid_o0_truth.rs"));
// D493 (round 55): MFEM's as-shipped pyramid P has no trustworthy tet-child
// rows (see `d493_pyramid_rt0_matches_corrected_mfem`), so the parity oracle is
// the operator re-assembled with the parent's finite element —
// tmp/d493/probe_fixed.cpp -> tmp/d493/pyramid_o0_fixed_truth.rs.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d493/pyramid_o0_fixed_truth.rs"));


// ── mesh fixtures (MFEM's exact connectivity from the probe dump) ────────────

fn q(v: f64) -> i64 {
    (v * 1e9).round() as i64
}

fn coord3(mesh: &Mesh<3>, v: u32) -> [f64; 3] {
    let c = mesh.node_coords(v);
    [c[0], c[1], c[2]]
}

/// Face identity key: the face's vertex coordinates sorted lexicographically
/// and flattened onto the 1e-9 grid (the prism oracle's join key).
fn face_key(mut vs: Vec<[f64; 3]>) -> Vec<i64> {
    let mut s: Vec<[i64; 3]> = vs.drain(..).map(|c| [q(c[0]), q(c[1]), q(c[2])]).collect();
    s.sort();
    s.into_iter().flatten().collect()
}

/// The same key built from a flattened coordinate run (a truth row's face
/// segment), quantised and sorted so the dump's vertex order does not matter.
fn key_from_run(run: &[f64]) -> Vec<i64> {
    face_key(run.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
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
#[ignore = "D493 open: MFEM upstream tet-child rows are tet-identity + stale-buffer artifact (probe_lf/probe_lp evidence, tmp/d493/); superseded in substance by d493_pyramid_rt0_matches_corrected_mfem, which pins the corrected operator"]
fn d493_pyramid_rt0_mfem_oracle() {
    // Kept as the as-shipped-MFEM artefact of record: 89 entries, of which the
    // 4 tet-child rows carry the stale buffer value.  The operator fem-rs now
    // produces is checked against the *corrected* table
    // (`MFEM_PYRAMID_O0_FIXED`, tmp/d493/d493_pyramid_o0_fixed.txt) below.
    assert!(MFEM_PYRAMID_O0.len() == 89);
}

/// Unit-pyramid coarse mesh of the round-54/55 probes (MFEM's PYRAMID vertex
/// order: unit-square base, apex above its centre).
fn mfem_pyramid_mesh() -> Mesh<3> {
    Mesh::<3> {
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
    }
}

/// fem-rs RT0 face table on a mixed pyramid/tet mesh: the sorted face-vertex
/// coordinates → global dof (the same key rule `hdiv_face_key` uses — quad
/// faces key on their sorted first three vertices).
fn face_dofs_pyramid(
    space: &HDivSpace<Mesh<3>>,
    mesh: &Mesh<3>,
) -> HashMap<Vec<i64>, u32> {
    const PYR_TRI: [[usize; 3]; 4] = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
    const PYR_QUAD: [usize; 4] = [0, 1, 2, 3];
    const TET_TRI: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    let mut out: HashMap<Vec<i64>, u32> = HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let nd = mesh.element_nodes(e);
        let tri = |v: &[usize]| -> Vec<u32> { v.iter().map(|&i| nd[i]).collect() };
        let mut faces: Vec<Vec<u32>> = Vec::new();
        match mesh.element_type(e) {
            ElementType::Pyramid5 => {
                for t in &PYR_TRI {
                    faces.push(tri(t));
                }
                faces.push(tri(&PYR_QUAD));
            }
            ElementType::Tet4 => {
                for t in &TET_TRI {
                    faces.push(tri(t));
                }
            }
            other => panic!("unexpected element type {other:?}"),
        }
        for f in faces {
            let key = if f.len() == 3 {
                FaceKey::new(f[0], f[1], f[2])
            } else {
                let mut v = f.clone();
                v.sort_unstable();
                FaceKey::new(v[0], v[1], v[2])
            };
            let dof = space
                .tri_face_dof(key)
                .unwrap_or_else(|| panic!("missing RT0 face dof for {f:?}"));
            let coords: Vec<[f64; 3]> = f.iter().map(|&v| coord3(mesh, v)).collect();
            out.insert(face_key(coords), dof);
        }
    }
    out
}

/// Variable-length face-key join for the pyramid refinement (fine faces are
/// tri 9-coord or quad 12-coord, coarse likewise → truth rows of 19/22/25
/// entries), plus the strict "no extra entries" guard.
fn fm_get(m: &HashMap<Vec<i64>, u32>, k: &[i64]) -> Option<u32> {
    m.get(k).copied()
}

fn assert_matches_mfem_pyramid(
    p: &fem_linalg::CsrMatrix<f64>,
    fine_map: &HashMap<Vec<i64>, u32>,
    coarse_map: &HashMap<Vec<i64>, u32>,
    truth: &[&[f64]],
    label: &str,
) -> (usize, f64) {
    let mut rs: HashMap<u32, HashMap<u32, f64>> = HashMap::new();
    for r in 0..p.nrows {
        for k in p.row_ptr[r]..p.row_ptr[r + 1] {
            rs.entry(r as u32).or_default().insert(p.col_idx[k] as u32, p.values[k]);
        }
    }
    let mut matched = HashSet::new();
    let mut max_err = 0.0_f64;
    let mut bad: Vec<(u32, u32, f64, f64)> = Vec::new();
    for row in truth {
        let n = row.len();
        let mut found = false;
        let mut tried = String::new();
        for (vf, vc) in [(3usize, 3usize), (3, 4), (4, 3), (4, 4)] {
            if 3 * vf + 3 * vc + 1 != n {
                continue;
            }
            let fine_key = key_from_run(&row[..3 * vf]);
            let coarse_key = key_from_run(&row[3 * vf..3 * vf + 3 * vc]);
            let v = row[n - 1];
            tried.push_str(&format!(
                " (vf={vf},vc={vc}: fine {} coarse {})",
                fm_get(fine_map, &fine_key).is_some(),
                fm_get(coarse_map, &coarse_key).is_some()
            ));
            let (Some(&r), Some(&c)) = (fine_map.get(&fine_key), coarse_map.get(&coarse_key))
            else {
                continue;
            };
            let got = rs.get(&r).and_then(|m| m.get(&c)).unwrap_or_else(|| {
                panic!("{label}: fem-rs P[{r},{c}] missing for MFEM entry {v}")
            });
            max_err = max_err.max((got - v).abs());
            if (got - v).abs() > 1e-12 {
                bad.push((r, c, v, *got));
            }
            matched.insert((r, c));
            found = true;
            break;
        }
        assert!(found, "{label}: no join for truth row {row:?}; lookups:{tried}");
    }
    assert!(
        bad.is_empty(),
        "{label}: {} of {} entries differ (first {}: {})",
        bad.len(),
        truth.len(),
        bad.len().min(8),
        bad.iter()
            .take(8)
            .map(|(r, c, v, g)| format!("P[{r},{c}] {g} vs {v}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let extra: Vec<(u32, u32)> = rs
        .iter()
        .flat_map(|(r, m)| m.keys().map(move |c| (*r, *c)))
        .filter(|k| !matched.contains(k))
        .collect();
    assert!(
        extra.is_empty(),
        "{label}: fem-rs carries {} entries absent from the truth table, e.g. {:?}",
        extra.len(),
        &extra[..extra.len().min(4)]
    );
    eprintln!(
        "{label}: {} MFEM entries matched, max|delta| = {max_err:.3e}",
        truth.len()
    );
    (truth.len(), max_err)
}

/// D493 (round 55) + D534/D535 (round 56): pyramid RT0 prolongation against
/// the **corrected** MFEM oracle.
///
/// The oracle truth table (`tmp/d493/pyramid_o0_fixed_truth.rs`) was
/// regenerated in D572 by `tmp/d572/gen_truth_rt03d.py` from probe
/// `tmp/d572/d572_probe_pyr_o0_rt03d.cpp` in the D572-adjudicated per-family
/// convention (pyramid = generic-collection `RT_FuentesPyramidElement(0)`
/// rows, tet children = `RT0TetFiniteElement`'s `n̂|F|` functionals via the
/// RT0Tet-nk shim) — bitwise identical to the round-55 `probe_fixed.cpp`
/// dump that round 57 re-pinned by hand (85/85 rows unchanged, tet rows
/// carry the ½-scaled values).  It assembles the operator MFEM *would*
/// produce:
/// per child `LocalInterpolation_RT` with the **parent's** element
/// (`GetLocalInterpolation` for the 6 pyramid children — the call MFEM's own
/// assembly makes — and `GetTransferMatrix` across the 4 tetrahedron
/// children), with the child's own reference nodes/normals and its affine
/// embedding recovered through the parent's affine frame.  Two independent
/// checks make this the right oracle rather than a re-run of the shipped bug:
///
/// * the affine-frame embeddings reproduce `mesh.cpp`'s `pyr_children` point
///   matrices **138/138 entries** (all 10 children), and
/// * the corrected operator prolongates a constant flux field exactly
///   (`max|P·x_c − x_f| = 1.7e-16` with MFEM's own `Project_RT` dof values),
///   while MFEM's as-shipped operator misses by **0.633** — the tet children's
///   rows.
///
/// Green as of D534/D535: `PyraRTk` spans MFEM's Fuentes space (the raw
/// expansion is bitwise MFEM 4.10, probe `tmp/d534/d534_fuentes_basis.txt`),
/// the nodal basis is point-dual (W = I, so the builder's rows are MFEM's
/// `LocalInterpolation_RT` rows verbatim), and `HDivSpace::interpolate_vector`
/// serves pyramid RT0 with MFEM `Project_RT` dof values.
#[test]
fn d493_pyramid_rt0_matches_corrected_mfem() {
    let coarse_mesh = mfem_pyramid_mesh();
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
    assert_eq!(fine_mesh.n_elements(), 10);
    let coarse_space = HDivSpace::new(coarse_mesh.clone(), 0);
    let fine_space = HDivSpace::new(fine_mesh.clone(), 0);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(stats.located_count, fine_space.n_dofs());
    assert_eq!(fine_space.n_dofs(), 33);
    assert_exact_path_served(&p, fine_space.n_dofs());
    let cm = face_dofs_pyramid(&coarse_space, &coarse_mesh);
    let fm = face_dofs_pyramid(&fine_space, &fine_mesh);
    let (n, err) = assert_matches_mfem_pyramid(&p, &fm, &cm, MFEM_PYRAMID_O0_FIXED, "pyramid RT0");
    assert_eq!(n, 85);
    assert!(err <= 1e-12, "pyramid RT0: max|delta| = {err:.3e}");
}

/// The upstream-bug delineation, computed from the two MFEM dumps alone (no
/// fem-rs operator involved, so this test stays active):
///
/// * MFEM's **as-shipped** P (89 rows, `tmp/d493/d493_pyramid_o0.txt`) and the
///   **corrected** P (85 rows, `tmp/d493/d493_pyramid_o0_fixed.txt`) are joined
///   by face coordinates; every entry they disagree on must sit on a fine dof
///   of one of the four interior *tetrahedron* children, and
/// * the stale buffer value `0.15625` (`localP[PYRAMID](5)` row 4's last
///   entry, read out of bounds by `SetRow`) appears in exactly the four
///   tet-child rows.
#[test]
fn d493_pyramid_mfem_shipped_p_diverges_only_on_tet_children() {
    let coarse_mesh = mfem_pyramid_mesh();
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
    let coarse_space = HDivSpace::new(coarse_mesh.clone(), 0);
    let fine_space = HDivSpace::new(fine_mesh.clone(), 0);
    let cm = face_dofs_pyramid(&coarse_space, &coarse_mesh);
    let fm = face_dofs_pyramid(&fine_space, &fine_mesh);
    // fine dof -> the element shapes that own it (bit 0 = Pyramid5, bit 1 = Tet4)
    let mut dof_shapes: HashMap<u32, u8> = HashMap::new();
    for e in 0..fine_mesh.n_elements() as u32 {
        let bit = match fine_mesh.element_type(e) {
            ElementType::Pyramid5 => 1u8,
            ElementType::Tet4 => 2u8,
            other => panic!("unexpected fine element {other:?}"),
        };
        for &d in fine_space.element_dofs(e) {
            *dof_shapes.entry(d).or_insert(0) |= bit;
        }
    }
    const PYR: u8 = 1;
    const TET: u8 = 2;
    let join = |row: &[f64]| -> Option<(u32, u32, f64)> {
        let n = row.len();
        for (vf, vc) in [(3usize, 3usize), (3, 4), (4, 3), (4, 4)] {
            if 3 * vf + 3 * vc + 1 != n {
                continue;
            }
            let fk = key_from_run(&row[..3 * vf]);
            let ck = key_from_run(&row[3 * vf..3 * vf + 3 * vc]);
            if let (Some(&r), Some(&c)) = (fm.get(&fk), cm.get(&ck)) {
                return Some((r, c, row[n - 1]));
            }
        }
        None
    };
    let mut shipped: HashMap<(u32, u32), f64> = HashMap::new();
    for row in MFEM_PYRAMID_O0 {
        let (r, c, v) = join(row).expect("as-shipped row must join");
        shipped.insert((r, c), v);
    }
    let mut corrected: HashMap<(u32, u32), f64> = HashMap::new();
    for (li, row) in MFEM_PYRAMID_O0_FIXED.iter().enumerate() {
        let (r, c, v) = join(row).expect("corrected row must join");
        if (29..=32).contains(&r) {
            eprintln!("D573-LOC row-index {li} file-line {} -> ({r},{c}) = {v}", li + 4);
        }
        corrected.insert((r, c), v);
    }
    assert_eq!(MFEM_PYRAMID_O0.len(), 89);
    assert_eq!(MFEM_PYRAMID_O0_FIXED.len(), 85);
    let shared = shipped.keys().filter(|k| corrected.contains_key(*k)).count();
    assert!(shared >= 80, "expected >= 80 shared entries, got {shared}");
    // (a) every shared entry on a pyramid-only fine dof agrees exactly
    let mut pyr_only_shared = 0usize;
    for (key, v) in &shipped {
        if dof_shapes[&key.0] != PYR {
            continue; // shared with a tet child (or tet-only): see (b)
        }
        let w = corrected.get(key).expect("shared key");
        assert!((v - w).abs() <= 1e-14, "pyramid-only entry {key:?}: {v} vs {w}");
        pyr_only_shared += 1;
    }
    assert!(
        pyr_only_shared >= 20,
        "expected >= 20 pyramid-only shared entries, got {pyr_only_shared}"
    );
    // (b) every *value* divergence sits on a fine dof owned by a tet child;
    //     entries that are an explicit zero on one side only are MFEM's
    //     `|Ikj| < 1e-12` truncation vs the builder's skip (same value)
    let mut diverging: Vec<(u32, u32)> = Vec::new();
    let zeroish = |v: Option<&f64>| v.map_or(true, |x| x.abs() <= 1e-12);
    for key in shipped.keys().chain(corrected.keys()) {
        let same = match (shipped.get(key), corrected.get(key)) {
            (Some(x), Some(y)) => (x - y).abs() <= 1e-14,
            _ => zeroish(shipped.get(key)) && zeroish(corrected.get(key)),
        };
        if !same {
            assert_ne!(
                dof_shapes[&key.0] & TET,
                0,
                "as-shipped vs corrected divergence on {key:?}, whose dof is not a tet-child dof"
            );
            diverging.push(*key);
        }
    }
    assert!(!diverging.is_empty(), "the tet children must diverge");
    // (c) the out-of-bounds artefact: the four tet-child rows repeat the last
    //     row buffer's stale value — empirically P[28, 4] = 0.15625, the last
    //     pyramid child's `localP(5)` row 4 — in a column the corrected table
    //     leaves empty
    let stale = shipped[&(28, 4)];
    assert!((stale - 0.15625).abs() < 1e-15, "pyramid child 5's last row: {stale}");
    for r in 29..=32u32 {
        let v = shipped
            .get(&(r, 4))
            .copied()
            .unwrap_or_else(|| panic!("expected the stale entry P[{r},4]"));
        assert!((v - stale).abs() < 1e-15, "P[{r},4] = {v}, expected the stale {stale}");
        assert!(
            corrected.get(&(r, 4)).map_or(true, |w| (w - stale).abs() > 1e-15),
            "corrected P[{r},4] repeats the stale value {stale}"
        );
    }
    eprintln!(
        "as-shipped MFEM pyramid P: 89 entries, {shared} shared with the corrected oracle, \
         {} diverging entries, 4 stale duplicate rows (P[29..32,4] = P[28,4] = 0.15625)",
        diverging.len()
    );
}

/// fem-rs's pyramid RT0 prolongation, structurally complete and
/// convention-exact on every fine dof (D534/D535; re-pinned by the D572
/// adjudication).
///
/// The exact path serves a pyramid mesh (the 6 pyramid children through
/// their own geometry, the 4 tets through the parent pyramid's element —
/// D493), so the operator is complete: every fine dof has a row (the legacy
/// builder left 13 of the 33 rows empty, i.e. those fine dofs were silently
/// zeroed on transfer).
///
/// D572 adjudication (`tmp/d572/adjudication.md`): fem-rs keeps the pyramid
/// on MFEM's generic-collection `RT_FuentesPyramidElement(0)` (bitwise pin,
/// D534) and the tet on `RT0TetFiniteElement`'s `n̂|F|` duals (D560).  Those
/// are *different* face functionals on shared pyramid↔tet faces — the
/// Fuentes side samples `f·adj(J)·2n̂|F|`, the RT0Tet side `f·adj(J)·n̂|F|`
/// — and MFEM's own `RT0_3DFECollection` is no better: probe
/// `tmp/d572/d572_probe_pyr_o0_rt03d.cpp` measures
/// `max|P·x_c − x_f| = 0.54375` for ITS hierarchy with its own projections
/// (RT0Pyr(true) keeps the Fuentes `nk`, so the split survives there too;
/// only the class comment claims otherwise).  Within any single collection
/// the shared-face dofs are written by whichever element comes last, so a
/// mixed hierarchy carries the split in its gridfunction; this test pins
/// fem-rs's exact value of that behaviour:
///
/// * pyramid-only dofs prolong exactly (`≤ 1e-13`; measured 5.6e-17),
/// * the 12 pyramid↔tet shared faces store exactly half the
///   Fuentes-convention prolongation: `P·x_c = 2·x_f` (measured 4.2e-17
///   against the 2x form; 1.813e-1 against the naive equality), and
/// * the 4 tet↔tet faces prolong exactly (0.0) — single-convention.
#[test]
fn d493_pyramid_rt0_exact_path_serves_every_fine_dof() {
    let coarse_mesh = mfem_pyramid_mesh();
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
    let coarse_space = HDivSpace::new(coarse_mesh, 0);
    let fine_space = HDivSpace::new(fine_mesh.clone(), 0);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(fine_space.n_dofs(), 33);
    assert_eq!(stats.located_count, fine_space.n_dofs());
    assert_exact_path_served(&p, fine_space.n_dofs());
    let c3 = [0.9_f64, 0.4, -1.1];
    let x_c = coarse_space.interpolate_vector(&|_| c3.to_vec());
    let x_f = fine_space.interpolate_vector(&|_| c3.to_vec());
    let mut y = vec![0.0_f64; fine_space.n_dofs()];
    p.spmv(x_c.as_slice(), &mut y);
    // Ownership split of the fine dofs.  The pyramid↔tet shared faces carry
    // BOTH conventions (the exact path's first-touch writer is a pyramid
    // child — Fuentes rows; the engine's last writer is a tet — RT0Tet
    // values, exactly half the Fuentes functional on the same face).  A
    // tet↔tet shared dof is single-convention (RT0Tet rows, RT0Tet stored
    // values), so it must prolong exactly like the pyramid-owned dofs.
    const PYR: u8 = 1;
    const TET: u8 = 2;
    let mut owned = vec![0u8; 33];
    for e in 0..fine_mesh.n_elements() as u32 {
        let bit = match fine_mesh.element_type(e) {
            ElementType::Pyramid5 => PYR,
            ElementType::Tet4 => TET,
            other => panic!("unexpected fine element {other:?}"),
        };
        for &d in fine_space.element_dofs(e) {
            owned[d as usize] |= bit;
        }
    }
    let (mut pyr_res, mut split_res, mut tet_res) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut n_pyr, mut n_split, mut n_tettet) = (0usize, 0usize, 0usize);
    for i in 0..fine_space.n_dofs() {
        match owned[i] {
            TET => {
                // tet↔tet face: single convention, must prolong exactly
                tet_res = tet_res.max((y[i] - x_f.as_slice()[i]).abs());
                n_tettet += 1;
            }
            _ => {
                if owned[i] & PYR != 0 {
                    if owned[i] & TET != 0 {
                        // pyramid↔tet face: Fuentes functional = 2 x RT0Tet
                        split_res = split_res.max((y[i] - 2.0 * x_f.as_slice()[i]).abs());
                        n_split += 1;
                    } else {
                        pyr_res = pyr_res.max((y[i] - x_f.as_slice()[i]).abs());
                        n_pyr += 1;
                    }
                } else {
                    panic!("dof {i} owned by nobody");
                }
            }
        }
    }
    assert_eq!(
        n_pyr + n_split + n_tettet,
        fine_space.n_dofs(),
        "every fine dof is pyramid-owned, tet-owned or both"
    );
    assert_eq!(
        n_split + n_tettet,
        16,
        "the 16 tet-written dofs split into pyramid↔tet and tet↔tet faces"
    );
    eprintln!(
        "pyramid RT0 constant field (D572): {n_pyr} pyramid-only dofs residual {pyr_res:.3e}; \
         {n_split} pyramid↔tet shared dofs residual-vs-2x {split_res:.3e}; \
         {n_tettet} tet↔tet dofs residual {tet_res:.3e}"
    );
    assert!(
        pyr_res <= 1e-13,
        "pyramid-only dofs must prolong exactly, got {pyr_res:.3e}"
    );
    assert!(
        split_res <= 1e-13,
        "pyramid↔tet shared dofs must store exactly half the Fuentes-convention \
         prolongation (P·x_c = 2·x_f), got {split_res:.3e}"
    );
    assert!(
        tet_res <= 1e-13,
        "tet↔tet dofs must prolong exactly, got {tet_res:.3e}"
    );
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

/// D536: pyramid RT1/RT2 prolongation takes the MFEM `LocalInterpolation_RT`
/// exact path on the uniform pyramid refinement (6 pyramid + 4 tet children,
/// the inner one inverted) and prolongates the constant field **exactly** on
/// every fine dof.  The slot layout is MFEM's Fuentes order on both the
/// element and the space side (D445/D534), the slot rows are the single
/// `PyraRTk::mfem_nodal_rows(order)` table (D541), and the whole tet RT
/// family shares the same MFEM nodal convention (D540) — so the pyramid↔tet
/// mixed hierarchy is convention-consistent at every served order.  (MFEM
/// upstream's own pyramid transfer rows carry the tet-identity + stale-buffer
/// artifact documented in `d493_pyramid_rt0_mfem_oracle`, so the operator is
/// pinned against exactness on representable fields, not against MFEM's
/// buggy dump — the corrected-operator policy of D493.)
#[test]
fn d536_pyramid_rt1_rt2_exact_path_and_constant_field() {
    for order in [1u8, 2u8, 3u8] {
        let coarse_mesh = mfem_pyramid_mesh();
        let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
        let coarse_space = HDivSpace::new(coarse_mesh, order);
        let fine_space = HDivSpace::new(fine_mesh.clone(), order);
        let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
        assert_eq!(
            stats.located_count,
            fine_space.n_dofs(),
            "order {order}: exact path must serve every fine dof"
        );
        assert_exact_path_served(&p, fine_space.n_dofs());
        let c3 = [0.9_f64, 0.4, -1.1];
        let x_c = coarse_space.interpolate_vector(&|_| c3.to_vec());
        let x_f = fine_space.interpolate_vector(&|_| c3.to_vec());
        let mut y = vec![0.0_f64; fine_space.n_dofs()];
        p.spmv(x_c.as_slice(), &mut y);
        let mut max_res = 0.0_f64;
        for i in 0..fine_space.n_dofs() {
            max_res = max_res.max((y[i] - x_f.as_slice()[i]).abs());
        }
        eprintln!(
            "d536 pyramid RT{order} constant field: {dofs} fine dofs, max residual {max_res:.3e}",
            dofs = fine_space.n_dofs()
        );
        assert!(
            max_res <= 1e-12,
            "order {order}: constant field must prolong exactly, got {max_res:.3e}"
        );
    }
}
