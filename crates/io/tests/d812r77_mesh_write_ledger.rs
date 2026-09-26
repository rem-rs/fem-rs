//! D812-1 — byte-golden ledger for **every** `.mesh` write path.
//!
//! `nodes_dof_values` used to emit a `nodes` section only when
//! `Mesh::geom_order() > 1`, while MFEM's `Mesh::Printer` writes it whenever
//! `Nodes != NULL` (`mesh/mesh.cpp:7214` is the only place a `Nodes` field is
//! dropped).  A mesh whose geometry table has order 1 therefore lost its
//! per-element geometry on write — for a **discontinuous** order-1 table
//! (`L2_T1_*_P1`, how `data/periodic-*.mesh` store folded vertices) that is
//! silent geometry corruption, not just a formatting loss.
//!
//! Fixing the emission rule touches the single writer every `.mesh` consumer
//! goes through, so this ledger pins the whole corpus:
//!
//! * **Straight meshes** (`geometry == None`) must stay byte-identical.  The
//!   pre-fix equivalence is exact by construction: the D812-1 diff's only
//!   behavioural hunk is the gate itself, and for `geometry == None` the new
//!   gate returns `Ok(None)` exactly where the old `geom_order() <= 1` did — so
//!   every byte after it is the same code.  These digests are the forward
//!   regression guard: any later edit of the write path has to move them
//!   deliberately.
//! * **Geometry-bearing meshes** must carry a `nodes` section in their output
//!   (and survive a read → write → read round trip bit for bit — see
//!   `d812r77_order1_geometry_round_trips`).
//! * Meshes the reader or the writer legitimately refuses are pinned as
//!   refusals (*that* they are refused, and at which stage), so a later change
//!   cannot silently start dropping a payload that used to be rejected loudly.
//!
//! The digests below were taken **with the D812-1 fix in the tree**; the file
//! header says so explicitly, so nobody later mistakes them for a pre-fix
//! capture.  The digest is a local SHA-256 (no new dependency).  Regenerate
//! with `D812R77_WRITE_LEDGER=regen cargo test --release -p fem-io --test
//! d812r77_mesh_write_ledger` — never regenerate to silence a failure without
//! first explaining the diff.

use fem_io::mfem::{
    read_mfem, read_mfem_file, write_mfem_nodes, write_mfem_nodes_1d, NodesSpace,
};
use fem_mesh::simplex::Mesh;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Ledger header — states the provenance of every digest in the file.
const LEDGER_HEADER: &str = "\
# D812-1 .mesh write-path byte ledger.\n\
#\n\
# Captured WITH the D812-1 fix in the tree (`nodes_dof_values` gated on the\n\
# geometry table's presence instead of `geom_order() > 1`).  The pre-fix\n\
# equivalence for straight meshes is structural: that diff's only behavioural\n\
# hunk is the gate, and `geometry == None` returns `Ok(None)` exactly where the\n\
# old `geom_order() <= 1` did, so every following byte is unchanged code.\n\
#\n\
# Columns: name<TAB>status<TAB>sha256<TAB>bytes<TAB>lines<TAB>geometry<TAB>\n\
#          geom_order<TAB>nodes<TAB>note\n\
# status: ok | read_err | write_err | skip_1d\n\
# note:   why a refusal happens (documentation; not asserted)\n\
# Asserted: the status; the digest of every geometry-free `ok` row (byte\n\
# golden); and that every geometry-bearing `ok` row has a `nodes` section.\n";

// ─── ledger format ──────────────────────────────────────────────────────────

/// One ledger row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Row {
    /// File name inside `data/`.
    name: String,
    /// `ok` | `read_err` | `write_err` | `skip_1d`
    status: String,
    /// SHA-256 of the continuous-writer output (empty when not written).
    digest: String,
    /// Byte length of that output.
    bytes: usize,
    /// Line count of that output.
    lines: usize,
    /// The mesh carries a geometry table (MFEM `Nodes != NULL`).
    geometry: bool,
    /// That table's order.
    geom_order: u8,
    /// The output contains a `nodes` section header.
    nodes_section: bool,
    /// Why the row is refused (documentation only).
    note: String,
}

impl Row {
    fn render(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.name,
            self.status,
            self.digest,
            self.bytes,
            self.lines,
            u8::from(self.geometry),
            self.geom_order,
            u8::from(self.nodes_section),
            self.note,
        )
    }

    fn parse(line: &str) -> Row {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 9, "malformed ledger row: {line:?}");
        Row {
            name: f[0].to_string(),
            status: f[1].to_string(),
            digest: f[2].to_string(),
            bytes: f[3].parse().unwrap(),
            lines: f[4].parse().unwrap(),
            geometry: f[5] == "1",
            geom_order: f[6].parse().unwrap(),
            nodes_section: f[7] == "1",
            note: f[8].to_string(),
        }
    }
}

fn ledger_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/d812r77_write_ledger.txt")
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

// ─── measurement ────────────────────────────────────────────────────────────

/// Render a mesh the way `write_mfem_file` does, into memory.
fn render(mesh_d: &Mesh<2>, mesh_3d: Option<&Mesh<3>>) -> Result<Vec<u8>, String> {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut buf, mesh_d, mesh_3d, NodesSpace::Continuous)
        .map(|()| buf)
        .map_err(|e| e.to_string())
}

fn blank(name: &str) -> Row {
    Row {
        name: name.to_string(),
        status: "ok".into(),
        digest: String::new(),
        bytes: 0,
        lines: 0,
        geometry: false,
        geom_order: 1,
        nodes_section: false,
        note: "-".into(),
    }
}

/// A short tag naming *why* a row is refused — documentation for the corpus.
fn refusal_note(msg: &str) -> String {
    let tag = if msg.contains("ragged") {
        "ragged_geo (D627)"
    } else if msg.contains("not continuous") {
        "disc_geo (continuous writer refuses a folded table)"
    } else if msg.contains("NC mesh") {
        "nc_mesh (not a conforming MFEM mesh)"
    } else if msg.contains("NURBS NC-patch") {
        "nurbs_nc_patch"
    } else if msg.contains("dim=1") {
        "dim1 (no 1-D reader)"
    } else if msg.contains("patches") {
        "nurbs_patch_mesh"
    } else if msg.contains("no `nodes`") || msg.contains("not supported") {
        "unsupported_nodes_numbering"
    } else {
        "other"
    };
    tag.to_string()
}

fn measure(path: &Path) -> Row {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let file = match read_mfem_file(path) {
        Ok(f) => f,
        Err(e) => {
            if std::env::var("D812R77_DIAG").is_ok() {
                eprintln!("[diag] {name}: READ {e}");
            }
            return Row {
                status: "read_err".into(),
                note: refusal_note(&e.to_string()),
                ..blank(&name)
            };
        }
    };
    // 1-D containers are written through their own D813-4 entry point
    // (`write_mfem_nodes_1d`); the *discontinuous* space is the faithful one
    // for the folded `L2_T1_1D_P*` fixtures, and a table-free mesh ignores the
    // choice (`nodes_dof_values` returns `Ok(None)` either way).
    let (geometry, geom_order, rendered) = match (&file.mesh2d, &file.mesh3d, &file.mesh1d) {
        (Some(m), _, _) => (m.geometry.is_some(), m.geom_order(), render(m, None)),
        (None, Some(m), _) => {
            let scratch = Mesh::<2>::unit_square_tri(1);
            (m.geometry.is_some(), m.geom_order(), render(&scratch, Some(m)))
        }
        (None, None, Some(m)) => {
            let mut buf: Vec<u8> = Vec::new();
            (
                m.geometry.is_some(),
                m.geom_order(),
                write_mfem_nodes_1d(&mut buf, m, NodesSpace::Discontinuous)
                    .map(|()| buf)
                    .map_err(|e| e.to_string()),
            )
        }
        (None, None, None) => {
            return Row {
                status: "skip_1d".into(),
                note: "read-only 1-D container".into(),
                ..blank(&name)
            }
        }
    };
    let bytes = match rendered {
        Ok(b) => b,
        Err(e) => {
            if std::env::var("D812R77_DIAG").is_ok() {
                eprintln!("[diag] {name}: WRITE {e}");
            }
            return Row {
                status: "write_err".into(),
                geometry,
                geom_order,
                note: refusal_note(&e.to_string()),
                ..blank(&name)
            };
        }
    };
    let text = String::from_utf8_lossy(&bytes).to_string();
    Row {
        status: "ok".into(),
        digest: sha256_hex(&bytes),
        bytes: bytes.len(),
        lines: text.lines().count(),
        geometry,
        geom_order,
        nodes_section: text.lines().any(|l| l.trim() == "nodes"),
        ..blank(&name)
    }
}

// ─── the ledger test ────────────────────────────────────────────────────────

#[test]
fn d812r77_mesh_write_ledger_matches_the_recorded_golden() {
    let ledger = ledger_path();
    let mut files: Vec<PathBuf> = std::fs::read_dir(data_dir())
        .expect("data/ directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "mesh"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 100,
        "expected the full data/ corpus (106 meshes), saw {}",
        files.len()
    );

    let measured: Vec<Row> = files.iter().map(|p| measure(p)).collect();

    if std::env::var("D812R77_WRITE_LEDGER").as_deref() == Ok("regen") {
        let mut out = String::from(LEDGER_HEADER);
        for r in &measured {
            writeln!(out, "{}", r.render()).unwrap();
        }
        std::fs::write(&ledger, out).expect("write ledger");
        panic!("d812r77: ledger regenerated at {}", ledger.display());
    }
    assert!(ledger.exists(), "ledger fixture missing: {}", ledger.display());

    let text = std::fs::read_to_string(&ledger).expect("read ledger");
    let recorded: Vec<Row> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(Row::parse)
        .collect();
    assert_eq!(
        recorded.len(),
        measured.len(),
        "ledger has {} rows, data/ has {} meshes — regenerate deliberately",
        recorded.len(),
        measured.len()
    );

    let mut straight_moved: Vec<String> = Vec::new();
    let mut geometry_wrong: Vec<String> = Vec::new();
    let mut status_moved: Vec<String> = Vec::new();

    for (rec, got) in recorded.iter().zip(&measured) {
        assert_eq!(rec.name, got.name, "ledger is sorted by file name");
        if rec.status != got.status {
            status_moved.push(format!(
                "{}: {} ({} -> {})",
                rec.name, rec.status, rec.note, got.note
            ));
            continue;
        }
        // A straight mesh must be byte-identical: this is the regression guard
        // over the whole write corpus.
        if !rec.geometry && rec.status == "ok" && rec.digest != got.digest {
            straight_moved.push(format!(
                "{}: {} B/{} lines -> {} B/{} lines",
                rec.name, rec.bytes, rec.lines, got.bytes, got.lines
            ));
        }
        // A geometry-bearing mesh must carry its geometry: after D812-1 the
        // output has a `nodes` section (MFEM: `Nodes != NULL` ⇒ write it).
        if rec.geometry && rec.status == "ok" && !got.nodes_section {
            assert_eq!(
                rec.geom_order, got.geom_order,
                "{}: geometry order moved",
                rec.name
            );
            geometry_wrong.push(format!(
                "{}: geometry order {} written WITHOUT a nodes section",
                rec.name, got.geom_order
            ));
        }
    }

    assert!(
        status_moved.is_empty(),
        "the writer's accept/reject decision moved for {} file(s):\n  {}",
        status_moved.len(),
        status_moved.join("\n  ")
    );
    assert!(
        straight_moved.is_empty(),
        "{} straight (geometry-free) mesh(es) changed bytes — the D812-1 fix must not \
         move them:\n  {}",
        straight_moved.len(),
        straight_moved.join("\n  ")
    );
    assert!(
        geometry_wrong.is_empty(),
        "{} geometry-bearing mesh(es) still lose their `nodes` section:\n  {}",
        geometry_wrong.len(),
        geometry_wrong.join("\n  ")
    );
}

/// The `nodes` payload of a mesh read from a discontinuous order-1 `nodes`
/// section must survive a read → write → read round trip bit for bit: that is
/// the difference between "the section was dropped" (silent corruption, the
/// element geometry is then the folded vertex coordinates) and "the section was
/// re-emitted".
#[test]
fn d812r77_order1_geometry_round_trips() {
    for name in [
        "periodic-hexagon.mesh",
        "periodic-square.mesh",
        "periodic-segment.mesh",
        "periodic-cube.mesh",
    ] {
        let path = data_dir().join(name);
        if !path.exists() {
            eprintln!("d812r77: {name} absent, skipped");
            continue;
        }
        let first = match read_mfem_file(&path) {
            Ok(f) => f,
            Err(e) => {
                // All four fixtures read since D813-4 gave the reader a dim=1
                // branch; a read error is a regression, not a limitation.
                panic!("d812r77: {name} is not readable ({e})");
            }
        };
        // 1-D container: the dedicated D813-4 write path (the round-77 ledger
        // skipped this fixture before the dim=1 reader existed).
        if let Some(m1) = &first.mesh1d {
            let probe_geo = m1
                .geometry
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: the 1-D container lost the geometry table"));
            let (order, npe) = (probe_geo.order, probe_geo.nodes_per_elem);
            let (conn, coords) = (probe_geo.conn.clone(), probe_geo.coords.clone());
            let mut buf: Vec<u8> = Vec::new();
            write_mfem_nodes_1d(&mut buf, m1, NodesSpace::Discontinuous)
                .unwrap_or_else(|e| panic!("{name}: the 1-D writer refused ({e})"));
            let text = String::from_utf8_lossy(&buf).to_string();
            assert!(
                text.lines().any(|l| l.trim() == "nodes"),
                "{name}: the written file has no `nodes` section — the order-{order} \
                 ({npe} nodes/elem) geometry table was dropped"
            );
            let back = read_mfem(std::io::Cursor::new(buf.clone())).expect("read back");
            let m1b = back.mesh1d.unwrap_or_else(|| panic!("{name}: 1-D container lost"));
            let g = m1b
                .geometry
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: geometry table lost on the round trip"));
            assert_eq!(g.order, order, "{name}: geometry order");
            assert_eq!(g.nodes_per_elem, npe, "{name}: nodes per element");
            assert_eq!(g.conn, conn, "{name}: geometry connectivity");
            assert_eq!(
                g.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}: geometry coordinates"
            );
            continue;
        }
        let (m2, m3) = (first.mesh2d.clone(), first.mesh3d.clone());
        let Some(probe_geo) = m2
            .as_ref()
            .and_then(|m| m.geometry.as_ref())
            .or_else(|| m3.as_ref().and_then(|m| m.geometry.as_ref()))
        else {
            panic!("{name}: the reader produced no geometry table — the fixture's `nodes` section was dropped at read time");
        };
        let (order, npe) = (probe_geo.order, probe_geo.nodes_per_elem);
        let (conn, coords) = (probe_geo.conn.clone(), probe_geo.coords.clone());

        let (mesh_d, mesh_3d) = match (&m2, &m3) {
            (Some(m), _) => (m.clone(), None),
            (None, Some(m)) => (Mesh::<2>::unit_square_tri(1), Some(m.clone())),
            _ => continue,
        };
        // The discontinuous arm is the faithful one for an L2 table; fall back
        // to the continuous arm only if it is the one that accepts.
        let mut buf: Vec<u8> = Vec::new();
        let mut err = match write_mfem_nodes(
            &mut buf,
            &mesh_d,
            mesh_3d.as_ref(),
            NodesSpace::Discontinuous,
        ) {
            Ok(()) => None,
            Err(_) => {
                let mut b2: Vec<u8> = Vec::new();
                let r = write_mfem_nodes(
                    &mut b2,
                    &mesh_d,
                    mesh_3d.as_ref(),
                    NodesSpace::Continuous,
                );
                buf = b2;
                r.err().map(|e| format!("disc={e}"))
            }
        };
        if let Some(e) = err.take() {
            panic!("{name}: no writer accepts the order-{order} geometry table ({e})");
        }
        let text = String::from_utf8_lossy(&buf).to_string();
        assert!(
            text.lines().any(|l| l.trim() == "nodes"),
            "{name}: the written file has no `nodes` section — the order-{order} \
             ({npe} nodes/elem) geometry table was dropped"
        );
        // Read back and compare the geometry table bit for bit.
        let back = read_mfem(std::io::Cursor::new(buf.clone())).expect("read back");
        let (b_order, b_npe, b_conn, b_coords) = match (&back.mesh2d, &back.mesh3d) {
            (Some(m), _) => geom_of_2d(m, name),
            (None, Some(m)) => geom_of_3d(m, name),
            _ => panic!("{name}: the round trip produced no container"),
        };
        assert_eq!(b_order, order, "{name}: geometry order");
        assert_eq!(b_npe, npe, "{name}: nodes per element");
        assert_eq!(b_conn, conn, "{name}: geometry connectivity");
        assert_eq!(
            b_coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}: geometry coordinates"
        );
    }
}

// ─── geometry-table extraction (one helper per container type) ──────────────

type GeomBits = (u8, usize, Vec<u32>, Vec<f64>);

fn geom_of_2d(m: &Mesh<2>, name: &str) -> GeomBits {
    let g = m
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{name}: geometry table lost on the round trip"));
    (g.order, g.nodes_per_elem, g.conn.clone(), g.coords.clone())
}

fn geom_of_3d(m: &Mesh<3>, name: &str) -> GeomBits {
    let g = m
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{name}: geometry table lost on the round trip"));
    (g.order, g.nodes_per_elem, g.conn.clone(), g.coords.clone())
}

// ─── SHA-256 (no new dependency) ────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in [a, b, c, d, e, f, g, hh].iter().enumerate() {
            h[slot] = h[slot].wrapping_add(*v);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

#[test]
fn sha256_known_answers() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
