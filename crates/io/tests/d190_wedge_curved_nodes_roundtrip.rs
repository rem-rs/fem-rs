//! D190 + D295: curved multi-prism `nodes` meshes must round-trip — the
//! loader's per-element geometry tables have to put the file's dof rows back
//! in the slot order every in-memory consumer evaluates, and `write_mfem`
//! must re-emit the file byte-for-byte like C++ `Mesh::Save`.
//!
//! Fixture: `tests/data/wedge_curved2.mesh` — serial MFEM 4.10 on a 2-prism
//! split of the unit cube (`Finalize` → `SetCurvature(2)` → `Save(out, 16)`),
//! the round-37 D177 probe artifact (`$HOME/work/d177/wedge_curved2.mesh`,
//! md5 `72eeb68849d5fb03fa99a66315f64788`).  H1 3-D P2 on 2 prisms:
//! 8 vertex dofs + 14 edge dofs + 5 quad-face dofs = **27 dofs, 19 of them
//! non-vertex**, the smallest curved wedge fixture that exercises shared
//! edges *and* a shared face between two elements.  The toroid wedges
//! (`toroid_wedge_o3.mesh`, `toroid_wedge_o3_r1.mesh`, `fem-mesh` test data)
//! add the order-3 triangular-face/interior blocks and the 8-child refined
//! mesh.
//!
//! Truth pairing: MFEM's `GetElementDofs` tables printed by the round-37
//! harness (`$HOME/work/d177/probe_111.txt`, `order 2 VSize=27`):
//!
//! ```text
//! elem 0 dofs(18): 0 1 3 4 5 7 8 9 10 11 12 13 14 15 16 22 23 24
//! elem 1 dofs(18): 0 3 2 4 7 6 10 17 18 13 19 20 14 16 21 24 25 26
//! ```
//!
//! Element 0's slots are the file dofs in table order, element 1 runs the
//! shared edges (edge 0-3 = dofs 10, edge 3-7 = dofs 13, edge 0-4 = dofs 14)
//! and the shared quad face (dofs 16/24) in its own local orientations —
//! exactly the entity-ordered H1 wedge numbering `DofManager::build_prism_h1`
//! reproduces (D177, pinned in `crates/space/tests/
//! d177_prism_h1_mfem_numbering.rs`).
//!
//! **Slot order (D295).**  The dof *ids* above are MFEM's file numbering, but
//! the row layout of `Mesh::geometry` is contracted to `PrismPk`'s
//! **layer-major** order — `Mesh::set_curvature`'s frozen prism table
//! (`crates/mesh/tests/d152_prism_curvature.rs`), and the layout
//! `element_jacobian`, `geo_ref_elem`, `curved::CurvedMesh`, AMR's
//! `curved_prism` and fem-io's `prism_nodes_dof_values` all evaluate.  Until
//! D295 the reader stored the H1 entity order, which made the read path and
//! the write path mutually exclusive (the writer's shared-dof consistency
//! check rejected every curved wedge read from a file).  The reader now
//! re-lays each row through `H1PrismPk::layer_perm` — `perm[m]` is the layer
//! slot holding H1 slot `m`'s dof — so elem 0's row becomes
//! `[0 1 3 | 8 9 10 | 14 15 16 | 22 23 24 | 4 5 7 | 11 12 13]`
//! (bottom layer, middle layer, top layer).
//!
//! Acceptance: every geometry slot lands on its file dof row (all
//! coordinates being multiples of 1/2 here, compared bit-for-bit), and the
//! written file equals C++ `Mesh::Save(out, 16)` (the round-31 probe
//! `tmp/round31_spot_save.cpp`, re-runs archived in
//! `tests/data/d314_cpp_save/`) **byte for byte** — which also proves the
//! writer's MFEM numbering off the permuted table.

use std::path::PathBuf;

use fem_element::lagrange::H1PrismPk;
use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem_nodes, NodesSpace};
use fem_mesh::simplex::Mesh;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/wedge_curved2.mesh");

/// MFEM's per-element `nodes` dof tables (probe_111.txt, see module docs).
const ELEM_DOFS: [[usize; 18]; 2] = [
    [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24],
    [0, 3, 2, 4, 7, 6, 10, 17, 18, 13, 19, 20, 14, 16, 21, 24, 25, 26],
];

/// The fixture's `nodes` rows (`VDim: 3, Ordering: 1` — one `x y z` line per
/// dof, in dof order).
fn parse_node_rows(text: &str) -> Vec<[f64; 3]> {
    let mut rows = Vec::new();
    let mut in_values = false;
    for line in text.lines() {
        if line.starts_with("Ordering:") {
            in_values = true;
            continue;
        }
        if in_values {
            let t = line.split_whitespace().collect::<Vec<_>>();
            if t.len() == 3 {
                rows.push([t[0].parse().unwrap(), t[1].parse().unwrap(), t[2].parse().unwrap()]);
            }
        }
    }
    rows
}

/// The loader's row layout: H1 slot `m`'s dof lands at layer slot `perm[m]`.
///
/// Spot-checked against the layer-major lattice: slot 0..2 are the middle
/// layer's triangle vertices (the vertical-edge dofs 14/15/16), slots 9..11
/// the middle layer's triangle-edge dofs (the shared quad faces 22/23/24).
#[test]
fn layer_permutation_lays_h1_slots_out_layer_major() {
    let prism = H1PrismPk::new(2);
    let perm = prism.layer_perm();
    assert_eq!(perm.len(), 18);
    // `perm[m]` is a bijection onto 0..18.
    let mut seen = [false; 18];
    for &p in perm.iter() {
        assert!(!seen[p], "perm repeats layer slot {p}");
        seen[p] = true;
    }
    // H1 slot 0 (vertex 0) stays at layer slot 0; the bottom e01 edge dof
    // (H1 slot 6) sits at layer 0's third triangle dof; the vertical edge
    // 0-3 dof (H1 slot 12) sits at the middle layer's vertex-0 slot.
    assert_eq!(perm[0], 0);
    assert_eq!(perm[6], 3);
    assert_eq!(perm[12], 6);
}

#[test]
fn reader_reconstructs_mfem_wedge_geometry_table() {
    let text = std::fs::read_to_string(FIXTURE).unwrap();
    let rows = parse_node_rows(&text);
    assert_eq!(rows.len(), 27, "8 vertices + 14 edge + 5 quad-face dofs");

    let f = read_mfem_file(FIXTURE).expect("read fixture");
    let mesh = f.mesh3d.expect("dimension 3 mesh");
    assert_eq!(mesh.n_elems(), 2);
    assert_eq!(mesh.geom_order(), 2, "curved geometry must be picked up");

    let g = mesh.geometry.as_ref().expect("high-order geometry table");
    assert_eq!(g.order, 2);
    assert_eq!(g.nodes_per_elem, 18, "H1 P2 wedge slot count");
    assert_eq!(g.n_nodes, 27, "geometry node space = the file's dof space");

    // The slot order is `PrismPk`'s (layer-major), carrying MFEM's dof ids.
    let prism = H1PrismPk::new(2);
    let perm = prism.layer_perm();
    for (e, dofs) in ELEM_DOFS.iter().enumerate() {
        for (m, &dof) in dofs.iter().enumerate() {
            let slot = perm[m];
            let n = g.conn[e * 18 + slot] as usize;
            assert_eq!(
                n, dof,
                "elem {e} H1 slot {m} (layer slot {slot}): must hold the file dof"
            );
            // Every slot's geometry coordinate must be the file's dof row,
            // bit-for-bit (all coordinates are exact binary fractions here).
            for c in 0..3 {
                assert_eq!(
                    g.coords[3 * n + c], rows[dof][c],
                    "elem {e} H1 slot {m} (layer slot {slot}, file dof {dof}) component {c}"
                );
            }
        }
    }
    // The layer-major row of element 0 spelled out (module doc): bottom
    // layer, middle layer, top layer.
    let want0 = [
        0, 1, 3, 8, 9, 10, 14, 15, 16, 22, 23, 24, 4, 5, 7, 11, 12, 13,
    ];
    for (slot, &dof) in want0.iter().enumerate() {
        assert_eq!(g.conn[slot] as usize, dof, "elem 0 layer slot {slot}");
    }
}

/// The two elements share four edges and one quad face: the truth tables put
/// the same file dof at different element slots (each element walks the
/// entity in its own local orientation), and the loader's tables must
/// address those dofs through the *same* geometry nodes — that is what keeps
/// a shared entity's curved shape from being duplicated.  The slot pairs are
/// derived from the truth tables instead of being spelled out, so a typo in
/// one table breaks both tests loudly.
#[test]
fn shared_entities_resolve_to_single_geometry_nodes() {
    let f = read_mfem_file(FIXTURE).expect("read fixture");
    let mesh = f.mesh3d.expect("dimension 3 mesh");
    let g = mesh.geometry.as_ref().expect("geometry table");

    // File dofs claimed by both elements: 4 vertices + the 4 shared edge
    // dofs + the shared quad-face dof = 9.
    let shared: Vec<(usize, usize, usize)> = (0..18)
        .flat_map(|s0| {
            (0..18)
                .filter(move |&s1| ELEM_DOFS[0][s0] == ELEM_DOFS[1][s1])
                .map(move |s1| (ELEM_DOFS[0][s0], s0, s1))
        })
        .collect();
    assert_eq!(shared.len(), 9, "4 vertices + 5 shared entity dofs");
    let prism = H1PrismPk::new(2);
    let perm = prism.layer_perm();
    for (dof, s0, s1) in shared {
        let n0 = g.conn[perm[s0]] as usize;
        let n1 = g.conn[18 + perm[s1]] as usize;
        assert_eq!(n0, n1, "file dof {dof}: shared entity must be one geometry node");
        assert_eq!(
            &g.coords[3 * n0..3 * n0 + 3],
            &g.coords[3 * n1..3 * n1 + 3],
        );
    }
    // And the slot space resolves to exactly the file's dof space: the 36
    // slots collapse to 27 distinct geometry nodes (2 × 18 − 9 shared).
    let ids: std::collections::HashSet<usize> =
        g.conn.iter().map(|&n| n as usize).collect();
    assert_eq!(ids.len(), 27, "slot space must collapse to the 27 file dofs");
}

// ── D295: read → write byte-parity with C++ `Mesh::Save` ────────────────────

/// (curved prism fixture, C++ `Mesh::Save(out, 16)` re-save).  The
/// `wedge_curved2` fixture lives in this crate's test data, the two toroid
/// wedges in the `fem-mesh` test data.
const ROUNDTRIP_CASES: &[(&str, &str, bool)] = &[
    ("wedge_curved2.mesh", "cpp_wedge_curved2.mesh", false),
    ("toroid_wedge_o3.mesh", "cpp_toroid_wedge_o3.mesh", true),
    ("toroid_wedge_o3_r1.mesh", "cpp_toroid_wedge_o3_r1.mesh", true),
];

fn fixture_path(name: &str, mesh_data: bool) -> PathBuf {
    if mesh_data {
        [env!("CARGO_MANIFEST_DIR"), "..", "mesh", "tests", "data"]
            .iter()
            .collect::<PathBuf>()
            .join(name)
    } else {
        [env!("CARGO_MANIFEST_DIR"), "tests", "data"]
            .iter()
            .collect::<PathBuf>()
            .join(name)
    }
}

fn reference_path(name: &str) -> PathBuf {
    [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "data",
        "d314_cpp_save",
    ]
    .iter()
    .collect::<PathBuf>()
    .join(name)
}

/// Read a curved prism fixture and write it back with the continuous
/// `nodes` writer.
fn roundtrip_text(fixture: &str) -> String {
    let file = read_mfem_file(fixture).unwrap_or_else(|e| panic!("read {fixture}: {e}"));
    let mesh = file.mesh3d.expect("3-D wedge mesh");
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(
        &mut buf,
        &Mesh::<2>::unit_square_tri(1), // ignored by the 3-D write path
        Some(&mesh),
        NodesSpace::Continuous,
    )
    .unwrap_or_else(|e| panic!("write {fixture}: {e}"));
    String::from_utf8(buf).expect("written mesh is ASCII")
}

/// Until D295 this write failed outright: the reader laid the geometry rows
/// out in MFEM's H1 entity order while the writer evaluated them in
/// `PrismPk`'s layer-major order, so the shared-dof consistency check saw
/// "dof 10 shared by two elements with different coordinates" (and friends).
/// With the reader permuted, the written file must be exactly MFEM's own
/// `Save(…, 16)` of the same fixture — every section, including the
/// geometry-type comment block (D294).
#[test]
fn curved_prism_write_matches_cpp_save_byte_for_byte() {
    for (fixture, reference, mesh_data) in ROUNDTRIP_CASES {
        let fixture = &fixture_path(fixture, *mesh_data);
        let fixture_name = fixture.display().to_string();
        let got = roundtrip_text(&fixture_name);
        let want = std::fs::read_to_string(reference_path(reference))
            .unwrap_or_else(|e| panic!("{reference}: {e}"));
        assert!(
            got == want,
            "{fixture_name} vs C++ re-save {reference}: {} byte mismatch\nfirst difference: {:?}",
            if got.len() != want.len() {
                format!("{} vs {} ", got.len(), want.len())
            } else {
                String::new()
            },
            got.bytes()
                .zip(want.bytes())
                .position(|(a, b)| a != b)
                .map(|i| &got[i.saturating_sub(40)..(i + 40).min(got.len())]),
        );
    }
}

/// The write is a formatting fixed point: read(write(read(f))) writes back
/// the same bytes — C++'s own 16-digit save satisfies the same property.
#[test]
fn curved_prism_write_read_write_is_byte_stable() {
    for (fixture, _, mesh_data) in ROUNDTRIP_CASES {
        let fixture = fixture_path(fixture, *mesh_data);
        let fixture_name = fixture.display().to_string();
        let text1 = roundtrip_text(&fixture_name);
        let again = read_mfem(std::io::Cursor::new(text1.as_bytes().to_vec()))
            .expect("read_mfem must accept the written text");
        let mesh = again.mesh3d.expect("3-D wedge mesh");
        let mut buf: Vec<u8> = Vec::new();
        write_mfem_nodes(
            &mut buf,
            &Mesh::<2>::unit_square_tri(1),
            Some(&mesh),
            NodesSpace::Continuous,
        )
        .expect("second write must succeed");
        let text2 = String::from_utf8(buf).unwrap();
        assert_eq!(
            text1, text2,
            "{fixture_name}: the second write must reproduce the first byte for byte"
        );
    }
}
