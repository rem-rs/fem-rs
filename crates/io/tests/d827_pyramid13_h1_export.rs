//! D827-3 / D827-4 — the **continuous** (`H1_3D_P2`) `nodes` export of a
//! Pyramid13 mesh, and the probed dead end of curved rows.
//!
//! MFEM's own continuous pyramid container is the `H1_FuentesPyramidElement(2)`
//! — 15 dofs/element, `p(p²+3)+1` (`fe_coll.cpp:1978-1985`; probe
//! `tmp/d85b/probe_pyr_gen_h1.txt`).  The entity numbering and the straight
//! P1-map synthesis of the two rowless dofs are pinned against real MFEM 4.10
//! by `tmp/d86b/probe_pyr_h1.cpp`:
//!
//! ```text
//! D827_DUMP_DIR=tmp/d86b cargo test -p fem-io --test d827_pyramid13_h1_export
//! wsl -e bash -lc 'cd /mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/d86b && \
//!   g++ -std=c++17 -O2 -I$HOME/mfem410_ser probe_pyr_h1.cpp \
//!       $HOME/mfem410_ser/libmfem.a -o probe_pyr_h1 && \
//!   ./probe_pyr_h1 gen     && \
//!   ./probe_pyr_h1 check rs_pyrh1.mesh rs_pyrh1.want'
//! ```
//!
//! `gen` dumps `GetElementDofs` (= `0..15` on the single element) and checks
//! MFEM's generated values equal the straight P1 map at the H1 Fuentes nodal
//! points **with max deviation 0.0** (including the base-face dof at
//! reference `(½,½,0)` and the interior dof at `(¼,¼,½)` — the two slots a
//! 13-node row does not carry); `check` loads the fem-rs export, compares
//! every `nodes(GetElementDofs(e)[s])` against the writer's slot table, and
//! re-saves for the byte comparison.
//!
//! Curved rows stay refused in **both** spaces (D827-4): the probed verdict is
//! that MFEM defines no curved-pyramid geometry a 13-node row could be
//! measured against — its Gmsh reader has no code-19 (type 19 is absent from
//! the element-type table, `Unknown Gmsh element type`, `mesh/gmsh.cpp:677`;
//! probe `tmp/d86b/probe_pyr_curved_g19.txt`, exit 134) and its nominal
//! type-14 path is defective (14 refiner-stump values written into the 15-dof
//! ClosedUniform space, mispairing every dof and leaving the interior dof at
//! uninitialised memory — `tmp/d86b/probe_pyr_curved_g14_straight.txt`), while
//! the `.mesh` route is circular.

use fem_io::mfem::{write_mfem_nodes, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;

// ─── helpers ────────────────────────────────────────────────────────────────

/// Dump the written file (and the writer's per-slot want table) for the WSL
/// probe when `D827_DUMP_DIR` is set.
fn dump(name: &str, bytes: &[u8], want: &[Vec<f64>]) {
    if let Ok(dir) = std::env::var("D827_DUMP_DIR") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).expect("dump dir created");
        std::fs::write(dir.join(format!("rs_{name}.mesh")), bytes).expect("dump written file");
        let want_text: String = want
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join(format!("rs_{name}.want")), want_text).expect("dump want file");
    }
}

/// The `x (y z)` lines of the `nodes` payload, everything after `Ordering: 1`.
fn dof_lines(text: &str) -> Vec<Vec<f64>> {
    let start = text.find("Ordering: 1\n").expect("nodes header") + "Ordering: 1\n".len();
    text[start..]
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|v| v.parse::<f64>().unwrap()).collect())
        .collect()
}

fn assert_xyz(got: &[Vec<f64>], want: &[(f64, f64, f64)], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: dof count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let comps = [w.0, w.1, w.2];
        assert!(
            (1..=3).contains(&g.len())
                && (0..g.len()).all(|c| (g[c] - comps[c]).abs() < 1e-13),
            "{what}: dof {i} = {g:?}, want {w:?}"
        );
    }
}

/// The probe's asymmetric straight pyramid (positive orientation: base CCW
/// seen from above, apex above — the winding MFEM's byte-stable round-trip
/// keeps).
const PYR_CORNERS: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.7, 0.0],
    [0.1, 0.9, 0.0],
    [0.3, 0.4, 1.2],
];

/// The Gmsh type-19 midsides of a row whose corners are `corners`: base cycle
/// (0,1),(1,2),(2,3),(3,0) then the four laterals, each at its exact midpoint.
fn midsides(corners: &[[f64; 3]; 5]) -> [[f64; 3]; 8] {
    const MIDS: [[usize; 2]; 8] =
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]];
    std::array::from_fn(|k| {
        let [a, b] = MIDS[k];
        std::array::from_fn(|d| 0.5 * (corners[a][d] + corners[b][d]))
    })
}

/// A straight Pyramid13 row from 5 corner coordinates and a corner-node-id
/// table plus per-slot midsides node ids.
fn pyr_row(corner_ids: &[u32; 5], mid_ids: &[u32; 8]) -> Vec<u32> {
    let mut row = corner_ids.to_vec();
    row.extend_from_slice(mid_ids);
    row
}

// ─── single pyramid: the gen oracle ─────────────────────────────────────────

/// MFEM's own `SetCurvature(2, false, 3, byVDIM)` values for the probe
/// pyramid — the 15 H1 Fuentes dofs in entity order (`probe_pyr_h1_gen.txt`,
/// max |nodes − P1_map| = 0.0).  Slots 5-12 run `Constants<PYRAMID>::Edges`
/// (the row's midsides order: `(3,2)`/`(0,3)` are the row's `(2,3)`/`(3,0)`);
/// slots 13/14 are the two rowless dofs, the synthesised base-face and
/// interior values.
const PYR_H1_ORACLE: [(f64, f64, f64); 15] = [
    (0.0, 0.0, 0.0), // 0 v0
    (1.0, 0.0, 0.0), // 1 v1
    (1.0, 0.7, 0.0), // 2 v2
    (0.1, 0.9, 0.0), // 3 v3
    (0.3, 0.4, 1.2), // 4 apex
    (0.5, 0.0, 0.0), // 5 e(0,1)
    (1.0, 0.35, 0.0), // 6 e(1,2)
    (0.55, 0.8, 0.0), // 7 e(3,2) = the row's e(2,3)
    (0.05, 0.45, 0.0), // 8 e(0,3) = the row's e(3,0)
    (0.15, 0.2, 0.6), // 9 e(0,4)
    (0.65, 0.2, 0.6), // 10 e(1,4)
    (0.65, 0.55, 0.6), // 11 e(2,4)
    (0.2, 0.65, 0.6), // 12 e(3,4)
    (0.525, 0.4, 0.0), // 13 base quad face (synthesised)
    (0.4125, 0.4, 0.6), // 14 interior (synthesised)
];

#[test]
fn d827_pyramid13_exports_fuentes_h1() {
    let mids = midsides(&PYR_CORNERS);
    let mut coords: Vec<f64> = PYR_CORNERS.iter().flat_map(|c| *c).collect();
    coords.extend(mids.iter().flat_map(|c| *c));
    let mesh = Mesh::<3>::uniform(
        coords,
        (0..13).collect(),
        vec![1],
        ElementType::Pyramid13,
        // The base quad, written outward (normal −z): the reversed base cycle.
        vec![0, 3, 2, 1],
        vec![1],
        ElementType::Quad4,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Continuous)
        .expect("a straight pyramid exports its 15-dof H1 Fuentes container");
    let want: Vec<Vec<f64>> = PYR_H1_ORACLE
        .iter()
        .map(|(x, y, z)| vec![*x, *y, *z])
        .collect();
    dump("pyrh1", &bytes, &want);
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("elements\n1\n1 7 0 1 2 3 4\n"),
        "PYRAMID corner row:\n{text}"
    );
    assert!(
        text.contains("boundary\n1\n1 3 0 3 2 1\n"),
        "base quad boundary row:\n{text}"
    );
    assert!(text.contains("vertices\n5\n"), "compacted corner count:\n{text}");
    assert!(
        text.contains("FiniteElementCollection: H1_3D_P2"),
        "collection:\n{text}"
    );
    assert!(text.contains("VDim: 3\nOrdering: 1\n"), "section header:\n{text}");
    let got = dof_lines(&text);
    assert_eq!(got.len(), 15, "the H1 Fuentes container has 15 dofs/element");
    assert_xyz(&got, &PYR_H1_ORACLE.to_vec().as_slice(), "H1 Fuentes oracle");
}

// ─── inverted pair: shared base quad face + the four shared base edges ──────

/// The probe's `gen2` topology (`probe_pyr_h1_gen2.txt`): pyramid A base
/// (0,1,2,3) apex 4 above; pyramid B base (0,3,2,1) apex 5 below.
/// `NDofs = 6 + 12 + 1 + 2 = 21`; `GetElementDofs` =
/// `[0 1 2 3 4 | 6..13 | 18 | 19]` and `[0 3 2 1 5 | 9 8 7 6 14..17 | 18 | 20]`
/// — the four base-edge dofs and the base-face dof are shared, and the shared
/// face dof's synthesised value (the base-corner average, the P1 map at
/// reference `(½,½,0)`) agrees from both sides.
#[test]
fn d827_pyramid13_h1_inverted_pair_shares_face_and_base_edges() {
    let mut corners: Vec<[f64; 3]> = PYR_CORNERS.to_vec();
    corners.push([0.3, 0.4, -1.1]); // 5: apex of the inverted twin
    let a_mids = midsides(&PYR_CORNERS);
    // B's lateral midsides (its base midsides are A's, its base cycle runs
    // (0,3),(3,2),(2,1),(1,0) — the same four edges in reverse slots), so the
    // laterals run (0,5),(3,5),(2,5),(1,5) in the row's cycle order.
    let b_apex = corners[5];
    let b_cycle: [usize; 4] = [0, 3, 2, 1];
    let b_lat: [[f64; 3]; 4] = std::array::from_fn(|k| {
        std::array::from_fn(|d| 0.5 * (PYR_CORNERS[b_cycle[k]][d] + b_apex[d]))
    });
    let mut coords: Vec<f64> = Vec::new();
    for c in corners.iter().take(6) {
        coords.extend_from_slice(c);
    }
    for c in a_mids.iter() {
        coords.extend_from_slice(c);
    }
    for c in b_lat.iter() {
        coords.extend_from_slice(c);
    }
    // node ids: 0-5 corners, 6-13 A's midsides, 14-17 B's laterals.
    let a_row = pyr_row(&[0, 1, 2, 3, 4], &[6, 7, 8, 9, 10, 11, 12, 13]);
    let b_row = pyr_row(&[0, 3, 2, 1, 5], &[9, 8, 7, 6, 14, 15, 16, 17]);
    let conn: Vec<u32> = [a_row, b_row].concat();
    let mesh = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1],
        ElementType::Pyramid13,
        vec![0, 3, 2, 1],
        vec![1],
        ElementType::Quad4,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Continuous)
        .expect("export");
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("elements\n2\n1 7 0 1 2 3 4\n1 7 0 3 2 1 5\n"),
        "PYRAMID corner rows:\n{text}"
    );
    assert!(text.contains("vertices\n6\n"), "compacted corner count:\n{text}");
    // The 21 global dofs in id order: 6 vertices, 12 edges (A's eight first,
    // then B's four laterals), the shared base-face dof, the two interiors.
    let face = (
        0.25 * (PYR_CORNERS[0][0] + PYR_CORNERS[1][0] + PYR_CORNERS[2][0] + PYR_CORNERS[3][0]),
        0.25 * (PYR_CORNERS[0][1] + PYR_CORNERS[1][1] + PYR_CORNERS[2][1] + PYR_CORNERS[3][1]),
        0.0,
    );
    let interior = |apex: [f64; 3]| {
        (
            0.5 * face.0 + 0.5 * apex[0],
            0.5 * face.1 + 0.5 * apex[1],
            0.5 * 0.0 + 0.5 * apex[2],
        )
    };
    let a_int = interior(PYR_CORNERS[4]);
    let b_int = interior(corners[5]);
    let mut want: Vec<(f64, f64, f64)> = Vec::new();
    for c in corners.iter().take(6) {
        want.push((c[0], c[1], c[2]));
    }
    for c in a_mids.iter() {
        want.push((c[0], c[1], c[2]));
    }
    for c in b_lat.iter() {
        want.push((c[0], c[1], c[2]));
    }
    want.push(face);
    want.push(a_int);
    want.push(b_int);
    let got = dof_lines(&text);
    assert_eq!(got.len(), 21, "NDofs = 6 + 12 + 1 + 2:\n{text}");
    assert_xyz(&got, &want, "inverted pair: shared face dof 18, edges 6..17");
}

// ─── side-by-side pair: shared base edge (1,2), distinct bases ──────────────

/// The probe's `gen3` topology (`probe_pyr_h1_gen3.txt`): pyramid A base
/// (0,1,2,3) apex 4; pyramid B base (2,1,5,6) apex 7 — sharing only the base
/// edge (1,2).  `NDofs = 8 + 15 + 2 + 2 = 27`; `GetElementDofs` =
/// `[0 1 2 3 4 | 8..15 | 23 | 25]` and `[2 1 5 6 7 | 9 16..22 | 24 | 26]`.
#[test]
fn d827_pyramid13_h1_side_by_side_shares_base_edge() {
    let mut corners: Vec<[f64; 3]> = PYR_CORNERS.to_vec();
    corners.push([1.9, 0.4, 0.0]); // 5
    corners.push([1.5, 1.2, 0.0]); // 6
    corners.push([1.4, 0.8, 1.1]); // 7 apex of B
    let a_mids = midsides(&PYR_CORNERS);
    let b_base: [[f64; 3]; 5] = [corners[2], corners[1], corners[5], corners[6], corners[7]];
    let b_mids = midsides(&b_base);
    let mut coords: Vec<f64> = Vec::new();
    for c in &corners {
        coords.extend_from_slice(c);
    }
    for c in a_mids.iter() {
        coords.extend_from_slice(c);
    }
    // B's midsides: slot 0 (edge (2,1)) is A's e(1,2) midside — shared node;
    // slots 1..8 are B-only edges.
    for c in b_mids.iter().skip(1) {
        coords.extend_from_slice(c);
    }
    // node ids: 0-7 corners, 8-15 A's midsides, 16-22 B's own midsides.
    let a_row = pyr_row(&[0, 1, 2, 3, 4], &[8, 9, 10, 11, 12, 13, 14, 15]);
    let b_row = pyr_row(&[2, 1, 5, 6, 7], &[9, 16, 17, 18, 19, 20, 21, 22]);
    let conn: Vec<u32> = [a_row, b_row].concat();
    let mesh = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1],
        ElementType::Pyramid13,
        vec![0, 3, 2, 1],
        vec![1],
        ElementType::Quad4,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Continuous)
        .expect("export");
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("elements\n2\n1 7 0 1 2 3 4\n1 7 2 1 5 6 7\n"),
        "PYRAMID corner rows:\n{text}"
    );
    assert!(text.contains("vertices\n8\n"), "compacted corner count:\n{text}");
    // The 27 global dofs: 8 vertices, 15 edges (A's eight, then B's seven
    // new ones — its base edge (2,1) is A's e(1,2)), the two base-face dofs,
    // the two interiors.
    let avg4 = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]| {
        (
            0.25 * (a[0] + b[0] + c[0] + d[0]),
            0.25 * (a[1] + b[1] + c[1] + d[1]),
            0.25 * (a[2] + b[2] + c[2] + d[2]),
        )
    };
    let a_face = avg4(corners[0], corners[1], corners[2], corners[3]);
    let b_face = avg4(corners[2], corners[1], corners[5], corners[6]);
    let interior = |f: (f64, f64, f64), apex: [f64; 3]| {
        (
            0.5 * f.0 + 0.5 * apex[0],
            0.5 * f.1 + 0.5 * apex[1],
            0.5 * f.2 + 0.5 * apex[2],
        )
    };
    let mut want: Vec<(f64, f64, f64)> = Vec::new();
    for c in &corners {
        want.push((c[0], c[1], c[2]));
    }
    for c in a_mids.iter() {
        want.push((c[0], c[1], c[2]));
    }
    for c in b_mids.iter().skip(1) {
        want.push((c[0], c[1], c[2]));
    }
    want.push(a_face);
    want.push(b_face);
    want.push(interior(a_face, corners[4]));
    want.push(interior(b_face, corners[7]));
    let got = dof_lines(&text);
    assert_eq!(got.len(), 27, "NDofs = 8 + 15 + 2 + 2:\n{text}");
    assert_xyz(&got, &want, "side-by-side: distinct base-face dofs 23/24");
}

// ─── curved rows: refused in both spaces (D827-4) ───────────────────────────

/// A curved Pyramid13 row (a midsides node off its edge midpoint) has no
/// MFEM-defined payload in either space — the probed verdict (D827-4) is that
/// MFEM 4.10 cannot ingest a curved pyramid at all: no code-19 in the Gmsh
/// reader, the type-14 path defective, the `.mesh` route circular.
#[test]
fn d827_pyramid13_curved_row_refused_both_spaces() {
    let mids = midsides(&PYR_CORNERS);
    let mut coords: Vec<f64> = PYR_CORNERS.iter().flat_map(|c| *c).collect();
    coords.extend(mids.iter().flat_map(|c| *c));
    let mesh = Mesh::<3>::uniform(
        coords,
        (0..13).collect(),
        vec![1],
        ElementType::Pyramid13,
        vec![0, 3, 2, 1],
        vec![1],
        ElementType::Quad4,
    );
    let mut curved = mesh;
    // Move the lateral (0,4) midsides node (id 9) off its midpoint.
    curved.coords[9 * 3 + 2] = 0.65;
    let scratch = Mesh::<2>::unit_square_tri(1);
    for space in [NodesSpace::Discontinuous, NodesSpace::Continuous] {
        let mut bytes = Vec::new();
        let err = write_mfem_nodes(&mut bytes, &scratch, Some(&curved), space)
            .expect_err("a curved pyramid row must be refused in both spaces");
        let msg = err.to_string();
        assert!(
            msg.contains("curved Pyramid13") && msg.contains("D827-4"),
            "refusal must name the cause: {msg}"
        );
        assert!(
            msg.contains("code-19") && msg.contains("type-14"),
            "refusal must carry the D827-4 probe evidence: {msg}"
        );
        assert!(bytes.is_empty(), "a refused write must emit nothing");
    }
}
