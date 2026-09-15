//! D190: a **multi-prism** mesh with an MFEM high-order `nodes` section must
//! round-trip — the loader's reconstructed per-element geometry tables have to
//! put the file's dof rows back in MFEM's H1 wedge slot order.
//!
//! Fixture: `tests/data/wedge_curved2.mesh` — serial MFEM 4.10 on a 2-prism
//! split of the unit cube (`Finalize` → `SetCurvature(2)` → `Save(out, 16)`),
//! the round-37 D177 probe artifact (`$HOME/work/d177/wedge_curved2.mesh`,
//! md5 `72eeb68849d5fb03fa99a66315f64788`).  H1 3-D P2 on 2 prisms:
//! 8 vertex dofs + 14 edge dofs + 5 quad-face dofs = **27 dofs, 19 of them
//! non-vertex**, the smallest curved wedge fixture that exercises shared
//! edges *and* a shared face between two elements.
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
//! d177_prism_h1_mfem_numbering.rs`).  The read here therefore must be exact:
//! every geometry slot lands on its file dof row, all coordinates being
//! multiples of 1/2 (straight-sided cube), compared bit-for-bit.

use fem_io::mfem::read_mfem_file;

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

    for (e, dofs) in ELEM_DOFS.iter().enumerate() {
        // Vertex slots must reuse the mesh vertices, in connectivity order
        // (elem 0: 0 1 3 4 5 7, elem 1: 0 3 2 4 7 6 — the file's elements).
        for (s, &dof) in dofs.iter().enumerate().take(6) {
            assert_eq!(
                g.conn[e * 18 + s] as usize, dof,
                "elem {e} slot {s}: vertex slot must be the file dof (mesh vertex)"
            );
        }
        // Every slot's geometry coordinate must be the file's dof row,
        // bit-for-bit (all coordinates are exact binary fractions here).
        for (s, &dof) in dofs.iter().enumerate() {
            let n = g.conn[e * 18 + s] as usize;
            for c in 0..3 {
                assert_eq!(
                    g.coords[3 * n + c], rows[dof][c],
                    "elem {e} slot {s} (file dof {dof}) component {c}"
                );
            }
        }
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
    for (dof, s0, s1) in shared {
        let n0 = g.conn[s0] as usize;
        let n1 = g.conn[18 + s1] as usize;
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
