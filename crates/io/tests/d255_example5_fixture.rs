//! D255 (round 41): generate a minimal **2-D** VisIt collection named
//! `Example5` and pin it through `load_visit_collection`.
//!
//! The repo ships no `Example5` fixture, but the compare-dc miniapp's default
//! run (`compare-dc -r0 Example5 -r1 Example5`, MFEM's own sample invocation)
//! needs one.  This test both
//!
//! * always pins the round-trip: save a 2-D mesh + one H1 field with the
//!   `VisItCollection` writer, reload through `load_visit_collection`, and
//!   check the field DOFs come back (proving `read_visit_root`'s
//!   metadata-only contract is compensated by the slice reads), and
//! * when `D255_FIXTURE_DIR` is set, additionally writes the same collection
//!   into that directory so the manual byte-parity run can be done:
//!   `cd <D255_FIXTURE_DIR> && cargo run --release --example compare-dc --
//!   -r0 Example5 -r1 Example5`.  A one-hex 3-D twin (named `Example23`,
//!   exercising the `VisitMesh::Mesh3d` loader arm) is written to the same
//!   directory for `-r0 Example23 -r1 Example23`.

use fem_io::data_collection::{DcField, VisItCollection};
use fem_io::data_collection_load::load_visit_collection;
use std::path::PathBuf;

/// Two-quad 2-D MFEM mesh slice (v1.0 text).
const MESH_TXT: &str = concat!(
    "MFEM mesh v1.0\n",
    "\n",
    "dimension\n",
    "2\n",
    "\n",
    "elements\n",
    "2\n",
    "1 3 0 1 2 3\n",
    "2 3 1 4 5 2\n",
    "\n",
    "boundary\n",
    "6\n",
    "1 1 0 1\n",
    "1 1 1 4\n",
    "1 1 4 5\n",
    "2 1 5 2\n",
    "2 1 2 3\n",
    "2 1 3 0\n",
    "\n",
    "vertices\n",
    "6\n",
    "2\n",
    "0 0\n",
    "1 0\n",
    "1 1\n",
    "0 1\n",
    "2 0\n",
    "2 1\n",
);

/// Six nodal values (P1 on 6 vertices), deterministic.
const U_VALS: [f64; 6] = [0.25, -1.5, 2.0, 4.0, -0.125, 3.75];

/// One-hex 3-D MFEM mesh slice (v1.0 text, with boundary quads) — written
/// alongside the 2-D fixture (as `Example23`) so the manual compare-dc
/// regression can also cover the 3-D loader arm:
/// `cd <D255_FIXTURE_DIR> && <bin>/miniapp_compare_dc -r0 Example23 -r1 Example23`.
const HEX_MESH_TXT: &str = concat!(
    "MFEM mesh v1.0\n",
    "\n",
    "dimension\n",
    "3\n",
    "\n",
    "elements\n",
    "1\n",
    "1 5 0 1 2 3 4 5 6 7\n",
    "\n",
    "boundary\n",
    "6\n",
    "1 3 0 1 2 3\n",
    "1 3 1 5 6 2\n",
    "1 3 0 4 5 1\n",
    "1 3 3 7 6 2\n",
    "1 3 0 4 7 3\n",
    "1 3 4 5 6 7\n",
    "\n",
    "vertices\n",
    "8\n",
    "3\n",
    "0 0 0\n",
    "1 0 0\n",
    "1 1 0\n",
    "0 1 0\n",
    "0 0 1\n",
    "1 0 1\n",
    "1 1 1\n",
    "0 1 1\n",
);

fn build_collection(prefix: &str) -> VisItCollection {
    let mut dc = VisItCollection::new("Example5");
    dc.set_prefix_path(prefix);
    dc.set_cycle(0);
    dc.spatial_dim = 2;
    dc.topo_dim = 2;
    dc.register_field(DcField::nodes(
        "pressure",
        "H1_2D_P1",
        1,
        1,
        U_VALS.to_vec(),
    ));
    dc
}

/// The 3-D twin of [`build_collection`], named `Example23` (MFEM's own ex23
/// saves its VisIt collection under that name).
fn build_hex_collection(prefix: &str) -> VisItCollection {
    let mut dc = VisItCollection::new("Example23");
    dc.set_prefix_path(prefix);
    dc.set_cycle(0);
    dc.spatial_dim = 3;
    dc.topo_dim = 3;
    dc.register_field(DcField::nodes(
        "solution",
        "H1_3D_P1",
        1,
        1,
        [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .map(|v| v * 0.5)
            .to_vec(),
    ));
    dc
}

#[test]
fn example5_fixture_loads() {
    let out_dir = std::env::var("D255_FIXTURE_DIR").ok().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("d255_example5")
    });
    let _ = std::fs::remove_dir_all(&out_dir);
    let dc = build_collection(out_dir.to_str().unwrap());
    dc.save(0, MESH_TXT).expect("save failed");
    // Optional 3-D twin for the manual compare-dc 3-D regression.
    if std::env::var("D255_FIXTURE_DIR").is_ok() {
        build_hex_collection(out_dir.to_str().unwrap())
            .save(0, HEX_MESH_TXT)
            .expect("save 3-D fixture failed");
    }

    let root = out_dir.join("Example5_000000.mfem_root");
    let (cycle, mesh_txt, fields) =
        load_visit_collection(&root).expect("reload failed");
    assert_eq!(cycle, 0);
    assert!(mesh_txt.contains("MFEM mesh v1.0"));
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].0, "pressure");
    assert_eq!(fields[0].1, "H1_2D_P1");
    assert_eq!(fields[0].2, 1);
    // The writer prints at ostream precision 6, so the DOFs round-trip at
    // C++-observable precision only.
    let expect: Vec<f64> = U_VALS.iter().map(|v| format!("{v:.6}").parse().unwrap()).collect();
    for (got, want) in fields[0].3.iter().zip(expect.iter()) {
        assert!((got - want).abs() < 5e-7 * want.abs().max(1.0), "{got} vs {want}");
    }
    assert_eq!(fields[0].3.len(), 6);
}
