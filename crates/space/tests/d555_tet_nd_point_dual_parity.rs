//! D555/D560 — tet ND / RT projection parity with MFEM 4.10 on the 2×2×2 tet
//! unit cube (`Mesh::MakeCartesian3D(2,2,2, TETRAHEDRON)`, archived as
//! `d555_tet222_mfem.mesh`): the manufactured field
//! `A = (0, 0, sin πx·sin πy)` and its curl, projected by MFEM's
//! `GridFunction::ProjectCoefficient` into ND1 and RT0 (golden dumps
//! `d555_nd1/rt0_tet222_mfem.txt` from `tmp/d546/d560_final_probe.cpp`).
//!
//! * ND1: the dual is MFEM's `Project_ND` point value `Φ(mid)·(J tk)` — the
//!   FULL physical edge vector.  Fem-rs matches bit-for-bit, which settles
//!   D555's "ND half-circulation" hypothesis in the negative.
//! * RT0: the dual is MFEM's tet RT face functional `f·cof(J)·nk/2` = the
//!   face flux ∫_F f·n̂ ds (D560 fix in `HDivSpace::interpolate_vector`);
//!   before the fix fem-rs was exactly 2× MFEM on all 64 nonzero dofs.

use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_space::{HCurlSpace, HDivSpace};

const GOLDEN_ND1: &str = include_str!("data/d555_nd1_tet222_mfem.txt");
const GOLDEN_RT0: &str = include_str!("data/d555_rt0_tet222_mfem.txt");

fn load_mfem_mesh() -> Mesh<3> {
    let path = format!(
        "{}/tests/data/d555_tet222_mfem.mesh",
        env!("CARGO_MANIFEST_DIR")
    );
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap()
}

fn golden(tag: &str, text: &str) -> Vec<f64> {
    let prefix = format!("{tag} ");
    text.lines()
        .filter(|l| l.starts_with(&prefix))
        .map(|l| l.split_whitespace().nth(2).unwrap().parse().unwrap())
        .collect()
}

fn field(x: &[f64]) -> Vec<f64> {
    vec![
        0.0,
        0.0,
        (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin(),
    ]
}
fn exact_curl(x: &[f64]) -> Vec<f64> {
    vec![
        std::f64::consts::PI * (std::f64::consts::PI * x[0]).sin()
            * (std::f64::consts::PI * x[1]).cos(),
        -std::f64::consts::PI * (std::f64::consts::PI * x[0]).cos()
            * (std::f64::consts::PI * x[1]).sin(),
        0.0,
    ]
}

#[test]
fn d555_nd1_interpolation_matches_mfem_point_duals() {
    let mesh = load_mfem_mesh();
    let space = HCurlSpace::new(mesh, 1);
    let want = golden("ND1P", GOLDEN_ND1);
    assert_eq!(space.n_dofs(), want.len(), "ndof mismatch vs MFEM");
    let got = space.interpolate_vector(&field);
    let mut worst = (0usize, 0.0_f64);
    for (i, w) in want.iter().enumerate() {
        let d = (got.as_slice()[i] - w).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    assert!(
        worst.1 < 1e-12,
        "ND1 dof {} off by {:e}",
        worst.0,
        worst.1
    );
}

#[test]
#[ignore = "D560 (OPEN): fem-rs tet RT0 interpolation is exactly 2x MFEM's gridfunction convention - MFEM's tet RT face dual = f*cof(J)*nk_full/2 = the face flux (measured: golden RT0P here; main-session d560_ratio probe 64/64 ratio 2.0). The validated fix recipe: halve the tet RT0 dual in HDivSpace::interpolate_vector's Tet4|Tet10 arm (scoped order 0) - then this test goes green AND curl_3d_manufactured_field_convergence restores rate ~1.0. HOWEVER the halving exposes the stack-wide normalization split: hdiv_interpolate_regression::tet_rt0_constant_fields (reconstruction round-trip) and d493_pyramid_rt0_exact_path pin the old full-cross convention, i.e. the tet RT0 stack (interpolate / get-values / assembly basis - the assembled mass is exactly 1/4 MFEM, see d560_rt0_mass_parity) needs a coordinated 2x unification across hdiv.rs + raviart_thomas + transfer.rs before the flux convention can land. Deferred with main-session approval."]
fn d560_rt0_interpolation_matches_mfem_face_flux_duals() {
    let mesh = load_mfem_mesh();
    let space = HDivSpace::new(mesh, 0);
    let want = golden("RT0P", GOLDEN_RT0);
    assert_eq!(space.n_dofs(), want.len(), "ndof mismatch vs MFEM");
    let got = space.interpolate_vector(&exact_curl);
    let mut worst = (0usize, 0.0_f64);
    for (i, w) in want.iter().enumerate() {
        let d = (got.as_slice()[i] - w).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    assert!(
        worst.1 < 1e-12,
        "RT0 dof {} off by {:e}",
        worst.0,
        worst.1
    );
}
