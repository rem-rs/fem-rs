//! D825-1 / D825-2 — the **continuous** (`H1_*D_P2`) `nodes` numbering for the
//! remaining row-geometry cells (Line3/Tri6/Tet10/Hex27/Prism18) and the
//! Pyramid13 export (PYRAMID base row + the 27-dof `L2_T1_3D_P2` Fuentes
//! container).
//!
//! MFEM numbers an order-2 H1 space `[vertices | edges | faces | interiors]`;
//! the entity enumeration (element traversal × local entity order, first
//! encounter wins) and the compacted-vertex numbering are pinned against real
//! MFEM 4.10 by `tmp/d85b/probe_h1_truth.cpp` (cartesian meshes per family,
//! `SetCurvature(2, false, …)`, full `GetElementDofs` dumps in
//! `probe_h1_{line3,tri6,tet10,hex27,prism18}.txt`), and the fem-rs exports
//! are re-verified end-to-end by `tmp/d85b/probe_h1_check.cpp` (loads each
//! dumped file, compares every `nodes(dof(slot))` against the writer's row
//! coordinates, Save/Load round-trip, re-save byte comparison):
//!
//! ```text
//! D825_DUMP_DIR=tmp/d85b cargo test -p fem-io --test d825_row_geometry_h1
//! wsl -e bash -lc 'cd /mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/d85b && \
//!   g++ -std=c++17 -O2 -I$HOME/mfem410_ser probe_h1_check.cpp \
//!       $HOME/mfem410_ser/libmfem.a -o probe_h1_check && \
//!   for f in line3 tri6 tet10 hex27 prism18; do \
//!     ./probe_h1_check rs_$f.mesh rs_$f.want; done'
//! ```
//!
//! The Pyramid13 container is pinned against MFEM's own generator
//! (`tmp/d85b/probe_pyr_gen.txt`): `SetCurvature(2, true, 3, byVDIM)` on a
//! straight asymmetric pyramid fills the 27 Fuentes dofs with the **straight
//! P1 pyramid map evaluated at the L2 Fuentes nodal points** — max deviation
//! 5.6e-17; the oracle table below is that dump verbatim.

use fem_io::mfem::{read_mfem, write_mfem_nodes, write_mfem_nodes_1d, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;

// ─── helpers ────────────────────────────────────────────────────────────────

/// Dump the written file (and the writer's per-slot want table) for the WSL
/// probe when `D825_DUMP_DIR` is set.
fn dump(name: &str, bytes: &[u8], want: &[Vec<f64>]) {
    if let Ok(dir) = std::env::var("D825_DUMP_DIR") {
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

/// The probe's want table: for every element row, the coordinate each *slot*
/// carries (the checker compares it against `nodes(GetElementDofs(e)[s])`, so
/// a wrong slot→dof map fails on exactly the swapped slots).
fn slot_want(coords: &[f64], conn: &[u32], stride: usize, sdim: usize) -> Vec<Vec<f64>> {
    let mut want = Vec::with_capacity(conn.len() / stride);
    for row in conn.chunks(stride) {
        for &n in row {
            want.push((0..sdim).map(|c| coords[n as usize * sdim + c]).collect());
        }
    }
    want
}

/// The dof-order values of the single-prism fixture's H1 payload (also the
/// probe's want table in the H1 wedge's entity slot order — see the prism18
/// test).
const PRISM18_H1_DOFS: [(f64, f64, f64); 18] = [
    (0.0, 0.0, 0.0), // 0 v0
    (2.0, 0.0, 0.0), // 1 v1
    (0.0, 2.0, 0.0), // 2 v2
    (0.0, 0.0, 2.0), // 3 v3
    (2.0, 0.0, 2.0), // 4 v4
    (0.0, 2.0, 2.0), // 5 v5
    (1.0, 0.0, 0.0), // 6 bottom e01
    (1.0, 1.0, 0.0), // 7 bottom e12
    (0.0, 1.0, 0.0), // 8 bottom e20
    (1.0, 0.0, 2.0), // 9 top e34
    (1.0, 1.0, 2.0), // 10 top e45
    (0.0, 1.0, 2.0), // 11 top e53
    (0.0, 0.0, 1.0), // 12 vertical edge at v0
    (2.0, 0.0, 1.0), // 13 vertical edge at v1
    (0.0, 2.0, 1.0), // 14 vertical edge at v2
    (1.0, 0.0, 1.0), // 15 quad face y=0
    (1.0, 1.0, 1.0), // 16 quad face hypotenuse
    (0.0, 1.0, 1.0), // 17 quad face x=0
];

// ─── Line3 (1-D): H1_1D_P2 ──────────────────────────────────────────────────

/// Probe truth (`probe_h1_line3.txt`): three cartesian segments give
/// `NDofs = NV + NE = 4 + 3 = 7` with `GetElementDofs` = `[0 1 4] [1 2 5]
/// [2 3 6]` — `[v0 | v1 | NV + e]`, the segment's edge dof private.  The
/// fem-rs fixture is the same topology over `0..3` with midside nodes.
#[test]
fn d825_line3_exports_segment_h1() {
    // fem-rs Line3 row = the Gmsh type-8 order [v0, v1, mid].
    let coords: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    let conn: Vec<u32> = vec![0, 1, 4, 1, 2, 5, 2, 3, 6];
    let mesh = Mesh::<1>::uniform(
        coords,
        conn,
        vec![1, 1, 1],
        ElementType::Line3,
        vec![0, 3],
        vec![1, 2],
        ElementType::Point1,
    );
    let mut bytes = Vec::new();
    write_mfem_nodes_1d(&mut bytes, &mesh, NodesSpace::Continuous).expect("export");
    dump("line3", &bytes, &slot_want(&mesh.coords, &mesh.conn, 3, 1));
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("FiniteElementCollection: H1_1D_P2"),
        "collection:\n{text}"
    );
    assert!(
        text.contains("elements\n3\n1 1 0 1\n1 1 1 2\n1 1 2 3\n"),
        "SEGMENT corner rows:\n{text}"
    );
    assert!(text.contains("vertices\n4\n"), "compacted corner count:\n{text}");
    let got = dof_lines(&text);
    assert_xyz(
        &got,
        &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0),
          (0.5, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 0.0, 0.0)],
        "line3 H1 dofs: [v0 | v1 | NV+e]",
    );
}

// ─── Tri6 (2-D): H1_2D_P2 ───────────────────────────────────────────────────

/// Probe truth (`probe_h1_tri6.txt`): the two triangles of a 2×1 grid share
/// their diagonal edge as ONE dof (`elem0 = 0 4 3 6 7 8`, `elem1 = 4 0 1 6 9
/// 10`).  The fem-rs fixture: A(0,0) B(2,0) C(0,2) D(2,2), T0 = A-B-C, T1 =
/// B-D-C, sharing the B-C edge — `NDofs = NV + NE = 4 + 5 = 9`, T1's e20 slot
/// carrying the shared edge dof.
#[test]
fn d825_tri6_exports_triangle_h1() {
    let coords: Vec<f64> = vec![
        0.0, 0.0, // 0 A
        2.0, 0.0, // 1 B
        0.0, 2.0, // 2 C
        2.0, 2.0, // 3 D
        1.0, 0.0, // 4 mid AB
        1.0, 1.0, // 5 mid BC (shared)
        0.0, 1.0, // 6 mid CA
        2.0, 1.0, // 7 mid BD
        1.0, 2.0, // 8 mid DC
    ];
    let mesh = Mesh::<2>::uniform(
        coords,
        // Tri6 row = [v0 v1 v2 | e01 e12 e20].
        vec![0, 1, 2, 4, 5, 6, 1, 3, 2, 7, 8, 5],
        vec![1, 1],
        ElementType::Tri6,
        vec![0, 1, 1, 3, 3, 2, 2, 0],
        vec![1, 2, 3, 4],
        ElementType::Line2,
    );
    let mut bytes = Vec::new();
    write_mfem_nodes(&mut bytes, &mesh, None, NodesSpace::Continuous).expect("export");
    dump("tri6", &bytes, &slot_want(&mesh.coords, &mesh.conn, 6, 2));
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("FiniteElementCollection: H1_2D_P2"),
        "collection:\n{text}"
    );
    assert!(
        text.contains("elements\n2\n1 2 0 1 2\n1 2 1 3 2\n"),
        "TRIANGLE corner rows:\n{text}"
    );
    assert!(text.contains("vertices\n4\n"), "compacted corner count:\n{text}");
    let got = dof_lines(&text);
    assert_eq!(got.len(), 9, "NDofs = NV + NE = 9:\n{text}");
    assert_xyz(
        &got,
        &[
            (0.0, 0.0, 0.0), // 0 A
            (2.0, 0.0, 0.0), // 1 B
            (0.0, 2.0, 0.0), // 2 C
            (2.0, 2.0, 0.0), // 3 D
            (1.0, 0.0, 0.0), // 4 e(A,B) — T0's e01
            (1.0, 1.0, 0.0), // 5 e(B,C) — T0's e12 and T1's e20: ONE dof
            (0.0, 1.0, 0.0), // 6 e(C,A)
            (2.0, 1.0, 0.0), // 7 e(B,D)
            (1.0, 2.0, 0.0), // 8 e(D,C)
        ],
        "tri6 H1 dofs with the shared diagonal edge single",
    );
}

// ─── Tet10 (3-D): H1_3D_P2 ──────────────────────────────────────────────────

/// A single positive-orientation tet: `NDofs = 4 + 6 = 10` (no face or
/// interior dofs at P2 — probe `probe_h1_tet10.txt`, where even the 6-tet
/// cube only ever adds edge dofs).  Slots 4..10 run the `TET::Edges` order
/// [e01 e02 e03 e12 e13 e23].
#[test]
fn d825_tet10_exports_tetrahedron_h1() {
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, // 0
        2.0, 0.0, 0.0, // 1
        0.0, 2.0, 0.0, // 2
        0.0, 0.0, 2.0, // 3
        1.0, 0.0, 0.0, // 4 e01
        0.0, 1.0, 0.0, // 5 e02
        0.0, 0.0, 1.0, // 6 e03
        1.0, 1.0, 0.0, // 7 e12
        1.0, 0.0, 1.0, // 8 e13
        0.0, 1.0, 1.0, // 9 e23
    ];
    let mesh = Mesh::<3>::uniform(
        coords,
        (0..10).collect(),
        vec![1],
        ElementType::Tet10,
        // The four faces, written outward: (0,2,1) base, (0,3,2) x=0,
        // (0,1,3) y=0, (1,2,3) hypotenuse.
        vec![0, 2, 1, 0, 3, 2, 0, 1, 3, 1, 2, 3],
        vec![1, 2, 3, 4],
        ElementType::Tri3,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Continuous).expect("export");
    dump("tet10", &bytes, &slot_want(&mesh.coords, &mesh.conn, 10, 3));
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("FiniteElementCollection: H1_3D_P2"),
        "collection:\n{text}"
    );
    assert!(
        text.contains("elements\n1\n1 4 0 1 2 3\n"),
        "TETRAHEDRON corner row:\n{text}"
    );
    assert!(text.contains("vertices\n4\n"), "compacted corner count:\n{text}");
    let got = dof_lines(&text);
    assert_eq!(got.len(), 10, "NDofs = 4 + 6 edges:\n{text}");
    assert_xyz(
        &got,
        &[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ],
        "tet10 H1 dofs: vertices then the TET::Edges edge block",
    );
}

// ─── Hex27 (3-D): H1_3D_P2 ──────────────────────────────────────────────────

/// A single hex: `NDofs = 8 + 12 + 6 + 1 = 27`, slots = the `HexQk(2)` H1
/// topological order [vertices | `CUBE::Edges` blocks | `CUBE::FaceVert`
/// blocks | centre] (`probe_h1_hex27.txt` pins the same order on a 2-hex
/// grid, where the shared face's edge dofs 13/17/21/22 and face dof 34 are
/// single).
#[test]
fn d825_hex27_exports_hexahedron_h1() {
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, // 0  v0
        2.0, 0.0, 0.0, // 1  v1
        2.0, 2.0, 0.0, // 2  v2
        0.0, 2.0, 0.0, // 3  v3
        0.0, 0.0, 2.0, // 4  v4
        2.0, 0.0, 2.0, // 5  v5
        2.0, 2.0, 2.0, // 6  v6
        0.0, 2.0, 2.0, // 7  v7
        1.0, 0.0, 0.0, // 8  e01
        2.0, 1.0, 0.0, // 9  e12
        1.0, 2.0, 0.0, // 10 e32
        0.0, 1.0, 0.0, // 11 e03
        1.0, 0.0, 2.0, // 12 e45
        2.0, 1.0, 2.0, // 13 e56
        1.0, 2.0, 2.0, // 14 e76
        0.0, 1.0, 2.0, // 15 e47
        0.0, 0.0, 1.0, // 16 e04
        2.0, 0.0, 1.0, // 17 e15
        2.0, 2.0, 1.0, // 18 e26
        0.0, 2.0, 1.0, // 19 e37
        1.0, 1.0, 0.0, // 20 f z0
        1.0, 0.0, 1.0, // 21 f y0
        2.0, 1.0, 1.0, // 22 f x1
        1.0, 2.0, 1.0, // 23 f y1
        0.0, 1.0, 1.0, // 24 f x0
        1.0, 1.0, 2.0, // 25 f z1
        1.0, 1.0, 1.0, // 26 centre
    ];
    let want: Vec<(f64, f64, f64)> = (0..27)
        .map(|i| (coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]))
        .collect();
    let mesh = Mesh::<3>::uniform(
        coords,
        (0..27).collect(),
        vec![1],
        ElementType::Hex27,
        // The six faces in `CUBE::FaceVert` order (outward).
        vec![0, 3, 2, 1, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7, 4, 5, 6, 7],
        vec![1, 2, 3, 4, 5, 6],
        ElementType::Quad4,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Continuous).expect("export");
    dump("hex27", &bytes, &slot_want(&mesh.coords, &mesh.conn, 27, 3));
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("FiniteElementCollection: H1_3D_P2"),
        "collection:\n{text}"
    );
    assert!(
        text.contains("elements\n1\n1 5 0 1 2 3 4 5 6 7\n"),
        "CUBE corner row:\n{text}"
    );
    assert!(text.contains("vertices\n8\n"), "compacted corner count:\n{text}");
    let got = dof_lines(&text);
    assert_eq!(got.len(), 27, "NDofs = 8 + 12 + 6 + 1:\n{text}");
    assert_xyz(&got, &want, "hex27 H1 dofs: [v | e | f | centre]");
}

// ─── Prism18 (3-D): H1_3D_P2 ────────────────────────────────────────────────

/// A single prism (corners at the layer-major `PrismPk(2)` slots [0, 1, 2,
/// 12, 13, 14] — D820-1's probe-pinned table): `NDofs = 6 + 9 + 3 = 18`
/// (quad-face blocks only; the two triangular faces carry no interior dofs at
/// P2 — probe `probe_h1_prism18.txt`).
#[test]
fn d825_prism18_exports_wedge_h1() {
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, // 0  v0 (bottom tri)
        2.0, 0.0, 0.0, // 1  v1
        0.0, 2.0, 0.0, // 2  v2
        1.0, 0.0, 0.0, // 3  e01
        1.0, 1.0, 0.0, // 4  e12
        0.0, 1.0, 0.0, // 5  e20
        0.0, 0.0, 1.0, // 6  vertical-edge dof at v0
        2.0, 0.0, 1.0, // 7  vertical-edge dof at v1
        0.0, 2.0, 1.0, // 8  vertical-edge dof at v2
        1.0, 0.0, 1.0, // 9  quad-face dof (y = 0 side)
        1.0, 1.0, 1.0, // 10 quad-face dof (hypotenuse side)
        0.0, 1.0, 1.0, // 11 quad-face dof (x = 0 side)
        0.0, 0.0, 2.0, // 12 v3 (top tri)
        2.0, 0.0, 2.0, // 13 v4
        0.0, 2.0, 2.0, // 14 v5
        1.0, 0.0, 2.0, // 15 e34
        1.0, 1.0, 2.0, // 16 e45
        0.0, 1.0, 2.0, // 17 e53
    ];
    let mesh = Mesh::<3>::uniform(
        coords,
        (0..18).collect(),
        vec![1],
        ElementType::Prism18,
        // The three quadrilateral sides, outward (the two triangular caps are
        // left out of the boundary section — MFEM does not require it, and
        // keeping the section uniform lets the single `Quad4` face type stand).
        vec![0, 1, 13, 12, 1, 2, 14, 13, 2, 0, 12, 14],
        vec![1, 2, 3],
        ElementType::Quad4,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Continuous).expect("export");
    // The probe's want table must be in MFEM's `H1_WedgeElement` *entity* slot
    // order (what `GetElementDofs` returns), which differs from the mesh's
    // layer-major `PrismPk` row order — the file's global dof values are
    // order-free, so the table below doubles as the writer's oracle.
    let want: Vec<Vec<f64>> = [
        (0.0, 0.0, 0.0), // 0 v0
        (2.0, 0.0, 0.0), // 1 v1
        (0.0, 2.0, 0.0), // 2 v2
        (0.0, 0.0, 2.0), // 3 v3
        (2.0, 0.0, 2.0), // 4 v4
        (0.0, 2.0, 2.0), // 5 v5
        (1.0, 0.0, 0.0), // 6 bottom e01
        (1.0, 1.0, 0.0), // 7 bottom e12
        (0.0, 1.0, 0.0), // 8 bottom e20
        (1.0, 0.0, 2.0), // 9 top e34
        (1.0, 1.0, 2.0), // 10 top e45
        (0.0, 1.0, 2.0), // 11 top e53
        (0.0, 0.0, 1.0), // 12 vertical edge at v0
        (2.0, 0.0, 1.0), // 13 vertical edge at v1
        (0.0, 2.0, 1.0), // 14 vertical edge at v2
        (1.0, 0.0, 1.0), // 15 quad face y=0
        (1.0, 1.0, 1.0), // 16 quad face hypotenuse
        (0.0, 1.0, 1.0), // 17 quad face x=0
    ]
    .iter()
    .map(|(x, y, z)| vec![*x, *y, *z])
    .collect();
    dump("prism18", &bytes, &want);
    let text = String::from_utf8(bytes).unwrap();

    assert!(
        text.contains("FiniteElementCollection: H1_3D_P2"),
        "collection:\n{text}"
    );
    assert!(
        text.contains("elements\n1\n1 6 0 1 2 3 4 5\n"),
        // The emitted corner ids are the *compacted* corners: the row's
        // corner slots [0, 1, 2, 12, 13, 14] hold node ids [0, 1, 2, 12, 13,
        // 14], whose compacted ranks are [0, 1, 2, 3, 4, 5].
        "PRISM corner row:\n{text}"
    );
    assert!(text.contains("vertices\n6\n"), "compacted corner count:\n{text}");
    let got = dof_lines(&text);
    assert_eq!(got.len(), 18, "NDofs = 6 + 9 + 3:\n{text}");
    assert_xyz(&got, &PRISM18_H1_DOFS, "prism18 H1 dofs: [v | edges | quad faces]");
}

// ─── Pyramid13 (3-D): PYRAMID row + the 27-dof L2_T1_3D_P2 Fuentes container ─

/// The corners of the probe's asymmetric straight pyramid
/// (`tmp/d85b/probe_pyramid.cpp`), whose MFEM-generated Fuentes payload this
/// test pins verbatim (probe output `probe_pyr_gen.txt`, max deviation of
/// MFEM's own generator from the straight P1 map: 5.6e-17).
const PYR_CORNERS: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.7, 0.0],
    [0.1, 0.9, 0.0],
    [0.3, 0.4, 1.2],
];

/// MFEM's own `SetCurvature(2, true, 3, byVDIM)` oracle for that pyramid —
/// the 27 Fuentes dofs in file order (`o = k(p+1)² + j(p+1) + i`).
const PYR_ORACLE: [(f64, f64, f64); 27] = [
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.05, 0.45, 0.0),
    (0.525, 0.4, 0.0),
    (1.0, 0.35, 0.0),
    (0.1, 0.9, 0.0),
    (0.55, 0.8, 0.0),
    (1.0, 0.7, 0.0),
    (0.13309475019311126, 0.17745966692414836, 0.53237900077244504),
    (0.41127016653792581, 0.17745966692414836, 0.53237900077244504),
    (0.68944558288274038, 0.17745966692414836, 0.53237900077244504),
    (0.16091229182759273, 0.42781754163448149, 0.53237900077244504),
    (0.42517893735516654, 0.4, 0.53237900077244504),
    (0.68944558288274038, 0.37218245836551855, 0.53237900077244504),
    (0.18872983346207417, 0.6781754163448146, 0.53237900077244504),
    (0.43908770817240728, 0.62254033307585166, 0.53237900077244504),
    (0.68944558288274038, 0.56690524980688872, 0.53237900077244504),
    (0.26618950038622252, 0.35491933384829671, 1.0647580015448901),
    (0.32254033307585167, 0.35491933384829671, 1.0647580015448901),
    (0.37889116576548082, 0.35491933384829671, 1.0647580015448901),
    (0.27182458365518541, 0.40563508326896297, 1.0647580015448901),
    (0.32535787471033312, 0.4, 1.0647580015448901),
    (0.37889116576548082, 0.39436491673103713, 1.0647580015448901),
    (0.27745966692414836, 0.45635083268962917, 1.0647580015448901),
    (0.32817541634481456, 0.44508066615170339, 1.0647580015448901),
    (0.37889116576548082, 0.43381049961377749, 1.0647580015448901),
];

/// The straight Pyramid13 fixture: 5 corners + the 8 Gmsh type-19 midsides
/// (base cycle then laterals), all at their exact edge midpoints.
fn pyramid13_fixture() -> Mesh<3> {
    let mut coords: Vec<f64> = Vec::with_capacity(13 * 3);
    for c in PYR_CORNERS {
        coords.extend_from_slice(&c);
    }
    // Gmsh type-19 midsides: base (0,1),(1,2),(2,3),(3,0), laterals to apex.
    const MIDS: [[usize; 2]; 8] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4],
    ];
    for &[a, b] in MIDS.iter() {
        for d in 0..3 {
            coords.push(0.5 * (PYR_CORNERS[a][d] + PYR_CORNERS[b][d]));
        }
    }
    Mesh::<3>::uniform(
        coords,
        (0..13).collect(),
        vec![1],
        ElementType::Pyramid13,
        // The base quad, written outward (normal −z): the reversed base cycle.
        vec![0, 3, 2, 1],
        vec![1],
        ElementType::Quad4,
    )
}

/// D825-2: the Pyramid13 row exports as MFEM's own pyramid curvature
/// container — PYRAMID corner rows plus a per-element 27-dof `L2_T1_3D_P2`
/// section whose values are the straight P1 map at the Fuentes nodal points
/// (byte-level oracle: `tmp/d85b/probe_pyr_gen.txt`; container load +
/// round-trip: `./probe_pyramid check rs_pyr.mesh rs_pyr.want`).
#[test]
fn d825_pyramid13_exports_fuentes_l2() {
    let mesh = pyramid13_fixture();
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Discontinuous)
        .expect("export");
    let want: Vec<Vec<f64>> = PYR_ORACLE
        .iter()
        .map(|(x, y, z)| vec![*x, *y, *z])
        .collect();
    dump("pyr", &bytes, &want);
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
        text.contains("FiniteElementCollection: L2_T1_3D_P2"),
        "collection:\n{text}"
    );
    assert!(text.contains("VDim: 3\nOrdering: 1\n"), "section header:\n{text}");
    let got = dof_lines(&text);
    assert_eq!(got.len(), 27, "the Fuentes container has 27 dofs/element");
    assert_xyz(&got, &PYR_ORACLE.to_vec().as_slice(), "Fuentes oracle");
}

/// A *curved* Pyramid13 row (a midsides node off its edge midpoint) has no
/// derived Fuentes synthesis — the 27-dof payload is derived from the five
/// corners of a straight pyramid only (D827-4).
#[test]
fn d825_pyramid13_curved_row_refused() {
    let mesh = pyramid13_fixture();
    let mut curved = mesh;
    // Move the base (0,1) midsides node (id 5) off its midpoint.
    curved.coords[5 * 3 + 2] = 0.01;
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    let err = write_mfem_nodes(&mut bytes, &scratch, Some(&curved), NodesSpace::Discontinuous)
        .expect_err("a curved pyramid row must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("curved Pyramid13") && msg.contains("D827-4"),
        "refusal must name the cause: {msg}"
    );
    assert!(bytes.is_empty(), "a refused write must emit nothing");
}

/// The *continuous* space originally stayed refused for pyramids (round 85:
/// MFEM's own continuous container is the H1 Fuentes element with 15 dofs per
/// element, probe `tmp/d85b/probe_pyr_gen_h1.txt` — a payload the 13-node row
/// does not fill).  **D827-3 superseded the refusal**: the two rowless dofs
/// (base-face `(½,½,0)`, interior `(¼,¼,½)`) are the straight P1 map at the
/// H1 Fuentes nodal points — MFEM's own generator produces exactly that with
/// max deviation 0.0 (probe `tmp/d86b/probe_pyr_h1_gen.txt`) — so a straight
/// row now exports.  This test keeps the smoke pin; the full pin (entity
/// numbering on shared-edge/shared-face pairs, the byte oracle, the curved
/// refusal in both spaces) lives in `d827_pyramid13_h1_export`.
#[test]
fn d825_pyramid13_continuous_exported() {
    let mesh = pyramid13_fixture();
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Continuous)
        .expect("a straight pyramid exports its 15-dof H1 Fuentes container");
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        text.contains("FiniteElementCollection: H1_3D_P2"),
        "collection:\n{text}"
    );
    assert!(
        text.contains("elements\n1\n1 7 0 1 2 3 4\n"),
        "PYRAMID corner row:\n{text}"
    );
    let got = dof_lines(&text);
    assert_eq!(got.len(), 15, "15 H1 Fuentes dofs:\n{text}");
}

// ─── reader smoke: the written files round-trip through fem-rs's own reader ─

/// The continuous Line3 export read back: the base SEGMENT topology plus the
/// `H1_1D_P2` payload (the reader keeps the written vertex count verbatim,
/// D600).
#[test]
fn d825_line3_h1_read_roundtrip() {
    let coords: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    let mesh = Mesh::<1>::uniform(
        coords,
        vec![0, 1, 4, 1, 2, 5, 2, 3, 6],
        vec![1, 1, 1],
        ElementType::Line3,
        vec![0, 3],
        vec![1, 2],
        ElementType::Point1,
    );
    let mut bytes = Vec::new();
    write_mfem_nodes_1d(&mut bytes, &mesh, NodesSpace::Continuous).expect("export");
    let file = read_mfem(std::io::Cursor::new(&bytes)).expect("read back");
    let m1 = file.mesh1d.expect("1-D mesh");
    assert_eq!(m1.n_elems(), 3, "three SEGMENT rows");
    assert_eq!(m1.conn, vec![0, 1, 1, 2, 2, 3], "base corner rows");
    let geo = m1.geometry.as_ref().expect("H1 table built");
    assert_eq!(geo.order, 2, "order-2 table from the H1_1D_P2 section");
    assert_eq!(geo.nodes_per_elem, 3, "3 dofs per segment");
    // The reader stores 1-D geometry tables in its *ascending-lattice* slot
    // order [first, mid, last] (the convention `row_geometry_slots` contrasts
    // with the Gmsh-type-8 row order), so element e's slots carry dofs
    // [e, NV+e, e+1].
    assert_eq!(geo.conn, vec![0, 4, 1, 1, 5, 2, 2, 6, 3], "slot -> dof table");
}
