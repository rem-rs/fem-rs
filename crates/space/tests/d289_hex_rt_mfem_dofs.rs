//! D289 regression: hex H(div) dof semantics of `HDivSpace::interpolate_vector`.
//!
//! `fill_dual_matrix` must pair the hex interpolation rows with the SAME
//! reference basis the assembly/get-values stack uses for these dofs — MFEM's
//! default `RT_FECollection` nodal GaussLegendre variant (`vec_ref_elem` since
//! D245), not the LOR-pinned IntegratedGLL variant.  The nodal-GL basis is
//! exactly point-dual to the Gauss sample rows (`W = (1/4)·I`, k = 0,1,2), so
//! the solve returns the MFEM `Project_RT` dof values themselves
//! (`nk·adj(J)·u` at the dof nodes) — the values an MFEM VisIt/DC file carries.
//! With the IntegratedGLL dual (`W = I/16` at k = 0, generally dense above)
//! the stored values were `W^{-1} d` — exactly 4x MFEM at RT0.
//!
//! Ground truth: MFEM 4.10 single hex `1.0 x 1.2 x 0.8`,
//! `tmp/d309/probe_rt_dof_div.cpp` (Project_RT dofs %.17e + GetDivergence at
//! reference points), archived in `tmp/d309/d309_mfem_truth.txt`.

use fem_element::raviart_thomas::HexRTk;
use fem_element::reference::VectorReferenceElement;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::{fe_space::FESpace, HDivSpace};

/// Single affine hex `1.0 x 1.2 x 0.8` (the D225/D236 archive geometry, MFEM
/// CUBE vertex layout as produced by `unit_cube_hex`).
fn hex_mesh() -> Mesh<3> {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    mesh.transform(|p| [p[0], 1.2 * p[1], 0.8 * p[2]]);
    mesh
}

/// Parse a whitespace-separated f64 table.
fn table(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .map(|t| t.parse::<f64>().expect("token parses"))
        .collect()
}

/// MFEM 4.10 `Project_RT` dofs of u = (1, 2, 3) on the single hex, RT0
/// (6 dofs, FE-local order == fem-rs slot order).
const MFEM_RT0_DOFS: &str = "
-3.59999999999999964e+00 -1.60000000000000009e+00 9.59999999999999964e-01
1.60000000000000009e+00 -9.59999999999999964e-01 3.59999999999999964e+00
";

/// MFEM 4.10 `Project_RT` dofs of u = (x+2y+3z+1, 0, 0), RT1 (36 dofs).
const MFEM_RT1_DOFS: &str = "
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 2.89378497978710225e+00
4.22400000000000020e+00 4.22400000000000020e+00 5.55421502021289726e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 -3.26399999999999979e+00 -1.93378497978710207e+00
-4.59421502021289729e+00 -3.26399999999999979e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
2.41378497978710183e+00 3.74399999999999933e+00 3.74399999999999977e+00
5.07421502021289772e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
";

/// Physical divergence of the projected field at the reference points
/// (MFEM `GridFunction::GetDivergence`; rows = [RT0 p1, RT0 p2, RT0 p3,
/// RT1 p1, RT1 p2, RT1 p3]).
const MFEM_DIV: &str = "
4.62592926927148591e-16 4.62592926927148591e-16 4.62592926927148690e-16
1.00000000000000133e+00 9.99999999999997224e-01 9.99999999999998002e-01
";

/// The three probe points in `[-1,1]^3` reference coordinates (as used by the
/// C++ probe; identical frames on both sides).  Probe 2 is the hex centroid
/// (the `compute_element_divergence` sample point).
const PROBE_XI: [[f64; 3]; 3] = [
    [0.30000000000000004, 0.39999999999999991, 0.49999999999999978],
    [0.0, 0.0, 0.0],
    [0.0, -0.16666666666666663, 0.25],
];

/// `det J` of the `[-1,1]^3` isoparametric map of the hex.
const DET_RS: f64 = 0.5 * 0.6 * 0.4;

fn interp_dofs(order: u8, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    let space = HDivSpace::new(hex_mesh(), order);
    space
        .interpolate_vector(f)
        .as_slice()
        .to_vec()
}

/// `GridFunction::GetDivergence` counterpart: div u_phys(xi)
/// = (sum_i c_i s_i div_hat_i(xi)) / det J  ([-1,1] isoparametric frame).
fn physical_div(space: &HDivSpace<Mesh<3>>, g: &[f64], xi: &[f64; 3]) -> f64 {
    let re = HexRTk::new_gauss_legendre(space.order() as usize);
    let n = re.n_dofs();
    let mut dv = vec![0.0_f64; n];
    re.eval_div(xi, &mut dv);
    let dofs = space.element_dofs(0);
    let signs = space.element_signs(0);
    let mut d = 0.0_f64;
    for i in 0..n {
        d += g[dofs[i] as usize] * signs[i] * dv[i];
    }
    d / DET_RS
}

fn assert_close(tag: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * (1.0 + want.abs()),
        "{tag}: fem-rs {got:.17e} vs mfem {want:.17e}"
    );
}

/// RT0: interpolated dofs == MFEM `Project_RT` dofs (the pre-D289 fix stored
/// exactly 4x these values: [-14.4, -6.4, 3.84, 6.4, -3.84, 14.4]).
#[test]
fn rt0_interpolated_dofs_match_mfem_project_rt() {
    let got = interp_dofs(0, &|_| vec![1.0, 2.0, 3.0]);
    let want = table(MFEM_RT0_DOFS);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(&format!("rt0 dof {i}"), g, w, 1e-15);
    }
}

/// RT1: interpolated dofs == MFEM `Project_RT` dofs (36 dofs, FE-local order).
#[test]
fn rt1_interpolated_dofs_match_mfem_project_rt() {
    let got = interp_dofs(1, &|x| {
        vec![x[0] + 2.0 * x[1] + 3.0 * x[2] + 1.0, 0.0, 0.0]
    });
    let want = table(MFEM_RT1_DOFS);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(&format!("rt1 dof {i}"), g, w, 1e-14);
    }
}

/// div of the interpolated field at the two probe points == MFEM
/// `GridFunction::GetDivergence` (RT0: div 0 field; RT1: div 1 field).
#[test]
fn interpolated_field_divergence_matches_mfem_get_divergence() {
    let space0 = HDivSpace::new(hex_mesh(), 0);
    let g0 = space0.interpolate_vector(&|_| vec![1.0, 2.0, 3.0]);
    let space1 = HDivSpace::new(hex_mesh(), 1);
    let g1 = space1.interpolate_vector(&|x| {
        vec![x[0] + 2.0 * x[1] + 3.0 * x[2] + 1.0, 0.0, 0.0]
    });
    let want = table(MFEM_DIV);
    for (p, &xi) in PROBE_XI.iter().enumerate() {
        let d0 = physical_div(&space0, g0.as_slice(), &xi);
        let d1 = physical_div(&space1, g1.as_slice(), &xi);
        assert_close(&format!("rt0 div probe {p}"), d0, want[p], 1e-13);
        assert_close(&format!("rt1 div probe {p}"), d1, want[3 + p], 1e-14);
    }
}

/// External-dof scenario (DC file): MFEM dofs injected directly must recover
/// the divergence exactly through the GL evaluation — and the interpolated
/// dofs must agree with them (interpolate -> store -> evaluate round trip).
#[test]
fn external_mfem_dofs_give_mfem_divergence() {
    let space = HDivSpace::new(hex_mesh(), 1);
    let external = table(MFEM_RT1_DOFS);
    let interp = interp_dofs(1, &|x| {
        vec![x[0] + 2.0 * x[1] + 3.0 * x[2] + 1.0, 0.0, 0.0]
    });
    for (p, &xi) in PROBE_XI.iter().enumerate() {
        let d_ext = physical_div(&space, &external, &xi);
        let d_int = physical_div(&space, &interp, &xi);
        assert_close(&format!("external div probe {p}"), d_ext, 1.0, 1e-13);
        assert_close(
            &format!("interp vs external div probe {p}"),
            d_int,
            d_ext,
            1e-14,
        );
    }
}

/// LOR-facing contract (`lor.rs::pair_sign` reads only constant-field *signs*
/// of `interpolate_vector`): on hexes the interpolated constant-field dofs in
/// the six face blocks carry the outward-flux sign of their face (MFEM slot
/// order bottom, front, right, back, left, top — the C++ RT0 truth above
/// fixes the pattern [-,-,+,+,-,+] for u = (1,2,3)).  This sign pattern is
/// basis-independent, so the D289 dual-basis switch cannot perturb the LOR
/// dof pairing; the LOR assemblers additionally pin IntegratedGLL explicitly
/// and never consume interpolated values.
#[test]
fn hex_constant_field_face_block_signs_are_basis_independent() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    // per face block: expected sign of (u · outward normal), u = (1,2,3)
    const FACE_SIGNS: [f64; 6] = [-3.0, -2.0, 1.0, 2.0, -1.0, 3.0];
    for order in 0..=2u8 {
        let space = HDivSpace::new(mesh.clone(), order);
        let g = space.interpolate_vector(&|_| vec![1.0, 2.0, 3.0]);
        let m = (order as usize + 1) * (order as usize + 1);
        for e in 0..mesh.n_elements() as u32 {
            let dofs = space.element_dofs(e);
            let signs = space.element_signs(e);
            for (block, &expected) in FACE_SIGNS.iter().enumerate() {
                for i in 0..m {
                    let v = g[dofs[block * m + i] as usize];
                    let want = expected.signum() * signs[block * m + i];
                    assert_eq!(
                        v.signum(),
                        want,
                        "order {order} elem {e} face block {block} slot {i}: {v}"
                    );
                }
            }
        }
    }
}
