//! D602: the MFEM `.gf` reader (`read_mfem_gf`) must parse the native
//! `FiniteElementSpace` format written by `GridFunction::Save`, and the
//! values must be interpreted **directly** as fem-rs canonical dof values.
//!
//! Fixtures: `d602_mfem_tet6_a.gf` / `_b.gf` are MFEM 4.10
//! `GridFunction::ProjectCoefficient` + `Save` outputs on the D559 6-tet
//! ND2 cube (74 DOFs, two affine probe fields).  D602 re-measurement proved
//! the file values equal fem-rs `HCurlSpace::interpolate_vector` output on
//! all 74 dofs (the round-59 "12 differing dofs" compared per-element RAW
//! writer values, which MFEM never stores — `tmp/d602/`).

use fem_io::mfem::{read_mfem_gf, read_mfem_gf_file, write_mfem_gf_file};

const TET6_A: &str = include_str!("fixtures/d602_mfem_tet6_a.gf");
const TET6_B: &str = include_str!("fixtures/d602_mfem_tet6_b.gf");

#[test]
fn parses_mfem_saved_gf() {
    let gf = read_mfem_gf(TET6_A.as_bytes()).expect("parse a");
    assert_eq!(gf.collection, "ND_3D_P2");
    assert_eq!(gf.vdim, 1);
    assert_eq!(gf.ordering, 0);
    assert_eq!(gf.values.len(), 74, "tet ND2 cube vsize");
    // First values of the MFEM file, bit-for-bit (text is %.16g).
    assert_eq!(gf.values[0], 2.324573519457059);
    assert_eq!(gf.values[1], 8.675426480542942);

    let gf_b = read_mfem_gf(TET6_B.as_bytes()).expect("parse b");
    assert_eq!(gf_b.values.len(), 74);
    assert_eq!(gf_b.values[0], 1.47927405783631);
}

#[test]
fn reads_from_disk() {
    // Round-trip through the writer: read one fixture, rewrite it at
    // precision 16 (%.16g, the writer's MFEM-identical formatting), and
    // re-parse — the values must agree to rounding of the 16-digit text.
    let path = std::env::temp_dir().join("d602_roundtrip_a.gf");
    let gf = read_mfem_gf(TET6_A.as_bytes()).expect("parse a");
    write_mfem_gf_file(&path, 3, &gf.values, "ND", 2, gf.vdim, 16).expect("write");
    let gf2 = read_mfem_gf_file(&path).expect("re-read");
    assert_eq!(gf2.collection, "ND_3D_P2");
    assert_eq!(gf2.values.len(), 74);
    let max_dev = gf
        .values
        .iter()
        .zip(&gf2.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_dev < 1e-13, "%.16g round-trip deviation {max_dev}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn rejects_non_fes_payload() {
    let err = read_mfem_gf("MFEM grid function v1.0\n".as_bytes()).unwrap_err();
    assert!(format!("{err}").contains("FiniteElementSpace"), "{err}");
    let err = read_mfem_gf("FiniteElementSpace\nbroken header\n".as_bytes()).unwrap_err();
    assert!(format!("{err}").contains("collection"), "{err}");
}
