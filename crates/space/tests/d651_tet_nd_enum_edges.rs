//! D651 — tet ND1 global edge-dof numbering vs MFEM `EnumEdges` order.
//!
//! MFEM numbers global edge dofs by first-encounter order over elements × the
//! geometry local-edge table (`Mesh::GetVertexToVertexTable` walking
//! `Element::GetEdgeVertices(j)`, `mesh.cpp:8538`).  This test pins the
//! resulting `GetElementDofs` signed tables for the first refined beam-tet
//! elements against the MFEM 4.10 probe dump
//! (`$HOME/work/d651/probe`, source `tmp/d651/probe.cpp`):
//!
//! ```text
//! NE0=48 NV0=36 ref_levels=3
//! NE=24576 NV=5265 NEdges=32016 NFaces=51328 NBdrElem=4352
//! VSize=32016
//! E0: 0 1 2 3 4 -6
//! E1: -7 7 8 9 10 -12
//! ...
//! ess_tdof count=6528
//! ```
//!
//! Signed encoding: `dofs[j] >= 0` → (dof, +1); `dofs[j] < 0` → (-1-dofs[j], -1).

use fem_io::mfem::read_mfem_file;
use fem_mesh::{amr::refine_uniform_3d, Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs_hcurl;
use fem_space::{FESpace, HCurlSpace};

fn load_beam_tet_ref3() -> Mesh<3> {
    let path = format!("{}/../../data/beam-tet.mesh", env!("CARGO_MANIFEST_DIR"));
    let mfem = read_mfem_file(&path).expect("failed to read beam-tet.mesh");
    let mesh = mfem.mesh3d.expect("beam-tet must be a 3-D mesh");
    // ex3 ref_levels = floor(log(50000/48)/log(2)/3) = 3
    let n_ref = ((50000.0 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / 3.0).floor() as usize;
    let mut mesh = mesh;
    for _ in 0..n_ref {
        mesh = refine_uniform_3d(&mesh);
    }
    mesh
}

/// Signed MFEM-style table from the fem-rs space (slot-major, dof ids + signs).
fn signed_table(space: &HCurlSpace<Mesh<3>>, e: u32) -> Vec<i32> {
    let dofs = space.element_dofs(e);
    let signs = space.element_signs(e);
    (0..dofs.len())
        .map(|j| {
            if signs[j] > 0.0 {
                dofs[j] as i32
            } else {
                -1 - dofs[j] as i32
            }
        })
        .collect()
}

/// MFEM probe dump (first 16 elements, beam-tet ND1, refined 3 levels).
const MFEM_ELEMENT_DOFS: [&[i32]; 16] = [
    &[0, 1, 2, 3, 4, -6],
    &[-7, 7, 8, 9, 10, -12],
    &[12, -14, 14, -16, 16, 17],
    &[18, 19, -21, 21, -23, -24],
    &[-5, 5, 24, 3, 7, 12],
    &[5, 19, 24, 14, 12, -17],
    &[19, 18, 24, -22, -17, 11],
    &[18, -5, 24, -9, 11, 7],
    &[25, 26, 27, 28, 29, -31],
    &[-32, 32, 33, 34, 35, -37],
    &[-38, -39, 39, -41, 41, 42],
    &[-44, 44, -46, 46, -48, -49],
    &[-30, 30, -50, 28, 32, -38],
    &[30, 44, -50, 39, -38, -42],
    &[44, -44, -50, -47, -42, 36],
    &[-44, -30, -50, -34, 36, 32],
];

#[test]
fn d651_beam_tet_nd1_element_dofs_match_mfem() {
    let mesh = load_beam_tet_ref3();
    assert_eq!(mesh.n_elems(), 24576, "refined element count");
    let space = HCurlSpace::new(mesh, 1);
    assert_eq!(space.n_dofs(), 32016, "ND1 vsize = edge count");

    let mut mismatches = Vec::new();
    for (i, golden) in MFEM_ELEMENT_DOFS.iter().enumerate() {
        let got = signed_table(&space, i as u32);
        if got != *golden {
            mismatches.push((i, *golden, got));
        }
    }
    if !mismatches.is_empty() {
        for (i, golden, got) in &mismatches {
            eprintln!("E{}: MFEM {:?}\n    fem-rs {:?}", i, golden, got);
        }
        panic!(
            "{}/16 element dof tables differ from the MFEM 4.10 probe dump",
            mismatches.len()
        );
    }
}

/// Global edge id -> (min vertex, max vertex) for the first 64 edges, from the
/// same MFEM probe (`Mesh::GetEdgeVertices`, sorted pairs).
const MFEM_EDGE_PAIRS: [(u32, u32); 64] = [
    (28, 996), (28, 1001), (28, 1000), (996, 1001), (996, 1000), (1000, 1001),
    (324, 996), (996, 3122), (996, 3121), (324, 3122), (324, 3121), (3121, 3122),
    (1001, 3122), (329, 1001), (1001, 3153), (329, 3122), (3122, 3153), (329, 3153),
    (1000, 3121), (1000, 3153), (328, 1000), (3121, 3153), (328, 3121), (328, 3153),
    (1000, 3122), (324, 1114), (324, 3128), (324, 3127), (1114, 3128), (1114, 3127),
    (3127, 3128), (42, 1114), (1114, 1125), (1114, 1124), (42, 1125), (42, 1124),
    (1124, 1125), (1125, 3128), (427, 3128), (3128, 3727), (427, 1125), (1125, 3727),
    (427, 3727), (1124, 3127), (3127, 3727), (426, 3127), (1124, 3727), (426, 1124),
    (426, 3727), (1125, 3127), (329, 3161), (329, 1922), (329, 3163), (1922, 3161),
    (3161, 3163), (1922, 3163), (427, 3161), (1924, 3161), (3161, 3732), (427, 1924),
    (427, 3732), (1924, 3732), (1922, 1924), (124, 1922),
];

/// MFEM `GetEssentialTrueDofs` head (count = 6528), same probe.
const MFEM_ESS_COUNT: usize = 6528;
const MFEM_ESS_HEAD: [u32; 64] = [
    1, 2, 5, 13, 14, 17, 19, 20, 23, 51, 52, 55, 63, 64, 67, 69, 70, 73, 76, 77,
    80, 88, 89, 92, 94, 95, 98, 112, 114, 115, 164, 165, 166, 170, 171, 172, 176,
    177, 178, 189, 190, 191, 195, 196, 197, 201, 202, 203, 214, 215, 216, 220,
    221, 222, 226, 227, 228, 250, 252, 254, 261, 262, 265, 273,
];

/// Global edge dofs 0..63 must carry the same physical edges as MFEM
/// (dof numbering == first-encounter EnumEdges order on the identically
/// refined mesh).
#[test]
fn d651_beam_tet_nd1_edge_pairs_match_mfem() {
    let mesh = load_beam_tet_ref3();
    let space = HCurlSpace::new(mesh, 1);
    let m = space.mesh();
    const TET_EDGES: [(usize, usize); 6] =
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut pairs: Vec<(u32, u32)> = vec![(u32::MAX, u32::MAX); space.n_dofs()];
    for e in 0..m.n_elements() as u32 {
        let verts = m.element_nodes(e);
        for &(li, lj) in &TET_EDGES {
            let (gi, gj) = (verts[li], verts[lj]);
            let key = fem_space::dof_manager::EdgeKey::new(gi, gj);
            if let Some(dof) = space.edge_dof(key) {
                if pairs[dof as usize].0 == u32::MAX {
                    pairs[dof as usize] = (gi.min(gj), gi.max(gj));
                }
            }
        }
    }
    for (i, &(a, b)) in MFEM_EDGE_PAIRS.iter().enumerate() {
        assert_eq!(
            (pairs[i].0, pairs[i].1),
            (a, b),
            "edge dof {i} carries the wrong physical edge"
        );
    }
}

/// The essential-boundary true dof list must match MFEM's head and count.
#[test]
fn d651_beam_tet_nd1_ess_dofs_match_mfem() {
    let mesh = load_beam_tet_ref3();
    let space = HCurlSpace::new(mesh, 1);
    let m = space.mesh();
    let tags = m.unique_boundary_tags();
    let ess = boundary_dofs_hcurl(m, &space, &tags);
    assert_eq!(ess.len(), MFEM_ESS_COUNT, "ess tdof count");
    for (i, &d) in MFEM_ESS_HEAD.iter().enumerate() {
        assert_eq!(ess[i], d, "ess head[{i}] differs from MFEM");
    }
}

/// Dump mode: `cargo test -p fem-space --test d651_tet_nd_enum_edges -- --nocapture d651_dump`
/// writes `tmp/d651/rs_dump.txt` with the same sections as the C++ probe
/// (element dofs head, edge vertex pairs by dof id, ess dofs head).
#[test]
fn d651_dump_femrs_tables() {
    if std::env::var("D651_DUMP").is_err() {
        return;
    }
    // One-level refinement: first parent's 8 children + rt (D651 debugging).
    let path = format!("{}/../../data/beam-tet.mesh", env!("CARGO_MANIFEST_DIR"));
    let mfem = read_mfem_file(&path).unwrap();
    let one = refine_uniform_3d(&mfem.mesh3d.clone().unwrap());
    println!("one-level NE={} NV={}", one.n_elems(), one.n_nodes());
    for e in 0..9u32 {
        println!("child {e}: {:?}", one.element_nodes(e));
    }
    let two = refine_uniform_3d(&one);
    println!("two-level NE={} NV={}", two.n_elems(), two.n_nodes());
    println!("two-level elem0: {:?}", two.element_nodes(0));
    let three = refine_uniform_3d(&two);
    println!("three-level NE={} NV={}", three.n_elems(), three.n_nodes());
    // elements 0..8 = the children of level-2 element 0 (28, 324, 329, 328).
    for e in 0..9u32 {
        println!("three child {e}: {:?}", three.element_nodes(e));
    }

    let mesh = load_beam_tet_ref3();
    let n_verts = mesh.node_coords(0).len();
    let space = HCurlSpace::new(mesh, 1);
    let m = space.mesh();

    let mut out = String::new();
    out.push_str(&format!(
        "NE={} NEdges(dofs)={}\n",
        m.n_elements(),
        space.n_dofs()
    ));
    for i in 0..16u32 {
        out.push_str(&format!("E{i}: {:?}\n", signed_table(&space, i)));
    }

    // Reproduce the discovery walk: dof id -> (global vertex pair, local pair
    // direction) on first insertion, mirroring hcurl pass 1.  Local tet edge
    // table = hcurl `TET_EDGES` (MFEM `Geometry::Constants<TETRAHEDRON>::Edges`).
    const TET_EDGES: [(usize, usize); 6] =
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut pairs: Vec<(u32, u32)> = vec![(u32::MAX, u32::MAX); space.n_dofs()];
    for e in 0..m.n_elements() as u32 {
        let verts = m.element_nodes(e);
        for &(li, lj) in &TET_EDGES {
            let (gi, gj) = (verts[li], verts[lj]);
            let key = fem_space::dof_manager::EdgeKey::new(gi, gj);
            if let Some(dof) = space.edge_dof(key) {
                if pairs[dof as usize].0 == u32::MAX {
                    pairs[dof as usize] = (gi, gj);
                }
            }
        }
    }
    for (dof, (a, b)) in pairs.iter().take(64).enumerate() {
        out.push_str(&format!("EDGE {dof} : {a} {b}\n"));
    }

    let tags = m.unique_boundary_tags();
    let ess = boundary_dofs_hcurl(m, &space, &tags);
    out.push_str(&format!("ess_tdof count={} head:", ess.len()));
    for d in ess.iter().take(64) {
        out.push_str(&format!(" {d}"));
    }
    out.push('\n');
    out.push_str(&format!("n_verts(node_coords len check)={n_verts}\n"));

    let path = format!("{}/../../tmp/d651/rs_dump.txt", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(&path, out).unwrap();
    println!("wrote {path}");
}
