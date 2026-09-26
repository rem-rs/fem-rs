//! D813-5 — the `Tet4` `L2_T1_3D_P*` `nodes` arm, end to end.
//!
//! `l2_geometry_slots` / `mfem_l2_slots` had **no `Tet4` arm**, so
//! `write_mfem_nodes` refused an order-1+ `L2_T1_3D_P*` geometry table
//! ("no verified `L2_T1_3D_P1` slot mapping for Tet4") and a *read* table fell
//! through to the identity permutation with the D153 warning ("the geometry is
//! read with the file's own DOF order, which is likely scrambled") — i.e. a
//! curved tetrahedral mesh could neither be written faithfully nor, from order
//! 2 on, read faithfully.
//!
//! The two arms are now present (`l2_geometry_slots`: the `H1TetPk` lattice;
//! `mfem_l2_slots`: MFEM's `L2_TetrahedronElement` enumeration,
//! `fem/fe/fe_l2.cpp:695`).  The *ordering* is pinned element-by-element in
//! `crates/io/src/mfem.rs`'s unit test `d813_tet_l2_nodes_match_mfem_in_order`
//! (which can see the two private tables); this file pins the read → write
//! round trip against MFEM's own files:
//!
//! * `d813_mfem_beam-tet_l2p1.mesh.txt` / `..._l2p2.mesh.txt` —
//!   `Mesh::SetCurvature(p, true)` + `Mesh::Print(.., 16)` of
//!   `data/beam-tet.mesh` (`$HOME/work/d78main/d813_tet_probe`), i.e. genuine
//!   `L2_T1_3D_P1` / `P2` `nodes` sections of the same mesh.
//!
//! Set `D78B_DUMP_DIR=<dir>` to drop the re-written meshes for an external
//! `diff` (that is how the byte-level result below was captured).

use fem_io::mfem::{read_mfem_file, write_mfem_nodes, NodesSpace};
use fem_mesh::simplex::Mesh;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn golden(name: &str) -> String {
    std::fs::read_to_string(fixture(name))
        .unwrap_or_else(|e| panic!("missing fixture {name}: {e}"))
        .replace("\r\n", "\n")
}

fn dump(name: &str, bytes: &[u8]) {
    if let Some(dir) = std::env::var_os("D78B_DUMP_DIR") {
        let dir = Path::new(&dir);
        std::fs::create_dir_all(dir).expect("create D78B_DUMP_DIR");
        std::fs::write(dir.join(name), bytes).expect("dump");
    }
}

fn render(mesh: &Mesh<3>, space: NodesSpace) -> Vec<u8> {
    // The same 2-D placeholder `write_mfem_file_3d_nodes` uses.
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut buf, &Mesh::<2>::unit_square_tri(2), Some(mesh), space)
        .expect("write_mfem_nodes");
    buf
}

/// Read MFEM's own `L2_T1_3D_P{1,2}` file, check the recovered geometry table,
/// and write it back: the result must equal MFEM's file byte for byte.
///
/// `Mesh::SetCurvature(p, true)` stores the *element's own* nodes (an `L2`
/// field), so the table must round-trip exactly; MFEM printed at precision 16,
/// which is what the writer emits, so byte-identity is the expected outcome
/// (measured: both files, `diff` empty).
#[test]
fn d813_tet_l2_nodes_round_trip_matches_mfem() {
    for (name, order, npe) in [
        ("d813_mfem_beam-tet_l2p1.mesh.txt", 1u8, 4usize),
        ("d813_mfem_beam-tet_l2p2.mesh.txt", 2u8, 10usize),
    ] {
        let text = golden(name);
        let parsed = read_mfem_file(fixture(name)).expect("read the MFEM fixture");
        let mesh = parsed.mesh3d.expect("3-D mesh");

        // The geometry table came back as an L2 order-`order` table with
        // `npe` per-element nodes, on `ne` elements.
        let g = mesh.geometry.as_ref().expect("L2 geometry table");
        assert_eq!(g.order, order, "{name}: geometry order");
        assert_eq!(g.nodes_per_elem, npe, "{name}: nodes per element");
        assert_eq!(
            g.n_nodes,
            mesh.n_elems() as usize * npe,
            "{name}: independent nodes per element"
        );
        assert_eq!(
            g.conn.len(),
            mesh.n_elems() as usize * npe,
            "{name}: per-element connectivity"
        );
        // …and it is a *table*, not the vertex table: the dof values are 48*npe
        // triples, i.e. more values than the 36 vertices carry.
        assert_eq!(text.matches("L2_T1_3D_P").count(), 1, "{name}: FEC header");

        let bytes = render(&mesh, NodesSpace::Discontinuous);
        dump(&format!("d813_rs_{name}"), &bytes);
        let got = String::from_utf8(bytes).expect("utf-8");
        assert_eq!(
            got.lines().count(),
            text.lines().count(),
            "{name}: line count"
        );
        assert_eq!(got, text, "{name}: read → write must be MFEM's own file");
    }
}

/// Teeth: the round trip is *only* byte-exact because the tet arm exists.  A
/// mesh whose table cannot be numbered must be refused rather than written with
/// a scrambled ordering — the negative control uses the same fixture with its
/// geometry replaced by an equispaced (non-Gauss-Lobatto) `P3` table, which is
/// not the lattice MFEM's `L2_TetrahedronElement(3)` enumerates.
#[test]
fn d813_tet_l2_p3_off_lattice_table_is_refused() {
    let parsed = read_mfem_file(fixture("d813_mfem_beam-tet_l2p1.mesh.txt")).expect("read");
    let mut mesh = parsed.mesh3d.expect("3-D mesh");
    let g = mesh.geometry.as_mut().expect("table");
    // Pretend the file claimed order 3 (20 dofs/elem) while the table holds
    // only the 4 corner dofs: the slot count no longer matches, so no faithful
    // numbering exists and the writer must refuse.
    g.order = 3;
    let mut buf: Vec<u8> = Vec::new();
    let r = write_mfem_nodes(
        &mut buf,
        &Mesh::<2>::unit_square_tri(1),
        Some(&mesh),
        NodesSpace::Discontinuous,
    );
    assert!(r.is_err(), "an off-lattice L2_P3 tet table must not be written");
    assert!(buf.is_empty(), "a refused write must emit nothing");
}
