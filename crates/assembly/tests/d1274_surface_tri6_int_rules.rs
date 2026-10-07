//! D1274: SurfaceTri6 integration-rule parity pins (vs MFEM 4.10 probe).
//!
//! Ground truth: `probe_d1274` (WSL `~/work/rr123/probe_d1274.cpp`), which
//! mirrors `mfem410/examples/ex7.cpp` (elem_type=0, order=2, ref_levels=2,
//! snap at end) and dumps
//!   (a) `IntRules.Get(Geometry::TRIANGLE, 2/4/6)` point/weight tables,
//!   (b) element-0 local matrices of `DiffusionIntegrator`, `MassIntegrator`
//!       and `DomainLFIntegrator` (defaults oa=2, ob=0),
//! with every value in `%.17g` + `%a` (the `%a` forms are copied verbatim so
//! the expected doubles are exact).

use fem_assembly::boundary::surface::{
    SurfaceTri6BilinearIntegrator, SurfaceTri6LinearIntegrator,
};
use fem_assembly::boundary::surface_tri6::{
    SurfaceTri6DiffusionIntegrator, SurfaceTri6DomainSourceIntegrator, SurfaceTri6MassIntegrator,
};
use fem_element::quadrature::tri_rule_mfem_order;

/// Exact double from a C `%a` hex literal.
fn h(s: &str) -> f64 {
    // Rust parses hexadecimal float literals? Not on stable — use a tiny
    // converter via format! trick: parse as f64 from hex string.
    let s = s.trim();
    let neg = s.starts_with('-');
    let body = s.trim_start_matches('-');
    let body = body.strip_prefix("0x").expect("hex literal");
    let (mant, exp) = match body.split_once('p') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap()),
        None => (body, 0),
    };
    let (int_part, frac_part) = match mant.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mant, ""),
    };
    let mut v = 0.0f64;
    for c in int_part.chars() {
        v = v * 16.0 + c.to_digit(16).unwrap() as f64;
    }
    let mut scale = 1.0 / 16.0;
    for c in frac_part.chars() {
        v += c.to_digit(16).unwrap() as f64 * scale;
        scale /= 16.0;
    }
    let v = v * (exp as f64).exp2();
    if neg { -v } else { v }
}

fn assert_bitwise(tag: &str, i: usize, j: usize, got: f64, want_hex: &str, want_g: f64) {
    let want = h(want_hex);
    assert!(
        got.to_bits() == want.to_bits(),
        "{tag}[{i}][{j}]: got {:.17e} (0x{:x}) want {:.17e} ({want_hex} = {:.17e})",
        got,
        got.to_bits(),
        want_g,
        want,
    );
}

// ── (a) quadrature rule tables (probe_d1274 BEGIN_RULE blocks) ─────────────

#[test]
fn tri_rule_mfem_order_matches_probe() {
    // tri2: 3 points, w = 1/6 — AddTriPoints3(0, 1./6., 1./6.)
    let r2 = tri_rule_mfem_order(2);
    assert_eq!(r2.points.len(), 3);
    let a = "0x1.5555555555555p-3"; // 1/6
    let b = "0x1.5555555555556p-1"; // 1.-2.*(1./6.) = 2/3
    let want2 = [(a, a), (a, b), (b, a)];
    for (q, &(x, y)) in want2.iter().enumerate() {
        assert_eq!(h(x), r2.points[q][0], "tri2 q{q} x");
        assert_eq!(h(y), r2.points[q][1], "tri2 q{q} y");
        assert_eq!(h(a), r2.weights[q], "tri2 q{q} w");
    }

    // tri4: 6 points
    let r4 = tri_rule_mfem_order(4);
    assert_eq!(r4.points.len(), 6);
    let x4 = [
        "0x1.c8a6b8a0bd0dap-2", "0x1.baca3afa1793p-4",
        "0x1.77189ea1db0c8p-4", "0x1.a239d857893cep-1",
    ];
    let w4 = ["0x1.c97c4971907ccp-4", "0x1.c25cc272345bep-5"];
    let want4 = [
        (x4[0], x4[0], w4[0]),
        (x4[0], x4[1], w4[0]),
        (x4[1], x4[0], w4[0]),
        (x4[2], x4[2], w4[1]),
        (x4[2], x4[3], w4[1]),
        (x4[3], x4[2], w4[1]),
    ];
    for (q, &(x, y, w)) in want4.iter().enumerate() {
        assert_eq!(h(x), r4.points[q][0], "tri4 q{q} x");
        assert_eq!(h(y), r4.points[q][1], "tri4 q{q} y");
        assert_eq!(h(w), r4.weights[q], "tri4 q{q} w");
    }

    // tri6: 12 points
    let r6 = tri_rule_mfem_order(6);
    assert_eq!(r6.points.len(), 12);
    let a1 = "0x1.0269a05fa55d4p-4";
    let b1 = "0x1.bf6597e816a8bp-1";
    let a2 = "0x1.fe8a0c8eaecap-3";
    let b2 = "0x1.00baf9b8a89bp-1";
    let a3 = "0x1.45e3a7d318ec1p-1";
    let b3 = "0x1.3dcd086db1b5ep-2";
    let c3 = "0x1.b35d3f60e39p-5";
    let w6a = "0x1.a0857f40e72e6p-6";
    let w6b = "0x1.de5b492ddcee2p-5";
    let w6c = "0x1.535ba6438268p-5";
    let want6 = [
        (a1, a1, w6a), (a1, b1, w6a), (b1, a1, w6a),
        (a2, a2, w6b), (a2, b2, w6b), (b2, a2, w6b),
        (a3, b3, w6c), (b3, a3, w6c), (a3, c3, w6c), (c3, a3, w6c), (b3, c3, w6c), (c3, b3, w6c),
    ];
    for (q, &(x, y, w)) in want6.iter().enumerate() {
        assert_eq!(h(x), r6.points[q][0], "tri6 q{q} x");
        assert_eq!(h(y), r6.points[q][1], "tri6 q{q} y");
        assert_eq!(h(w), r6.weights[q], "tri6 q{q} w");
    }
}

// ── (b) element-0 local integrators on the bitwise ex7 mesh ────────────────

/// Element 0 of the ex7 tri6 mesh after 2 uniform refinements + end snap —
/// the six row coordinates in MFEM's H1 element-dof order (probe_d1274
/// ELEM0 dofs [0, 18, 20, 66, 67, 68]; coordinates from the D1273 probe
/// `r2snap` dump, bitwise vs `mfem410` `UniformRefinement` + `SnapNodes`).
fn elem0_nodes() -> [[f64; 3]; 6] {
    [
        [h("0x1p+0"), h("0x0p+0"), h("0x0p+0")],                  // dof 0
        [h("0x1.e5b9d136c6d96p-1"), h("0x1.43d136248490fp-2"), h("0x0p+0")], // dof 18
        [h("0x1.e5b9d136c6d96p-1"), h("0x0p+0"), h("0x1.43d136248490fp-2")], // dof 20
        [h("0x1.fadaa8f7eed52p-1"), h("0x1.21a1851ff630ap-3"), h("0x0p+0")], // dof 66
        [h("0x1.f2581ddd9b73p-1"), h("0x1.4c3abe93bcf75p-3"), h("0x1.4c3abe93bcf75p-3")], // dof 67
        [h("0x1.fadaa8f7eed52p-1"), h("0x0p+0"), h("0x1.21a1851ff630ap-3")], // dof 68
    ]
}

#[test]
fn elem0_local_matrices_match_probe() {
    let x = elem0_nodes();

    let mut ke = [0.0f64; 36];
    SurfaceTri6DiffusionIntegrator.add_to_element_matrix(&x, &mut ke);
    // probe elem0_diffusion (row-major)
    let want_d = [
        ["0x1.cbbe019447456p-1", "0x1.13049415a4a77p-3", "0x1.13049415a4a6fp-3", "-0x1.25c6aa2dcffaap-1", "-0x1.365ee86f3477p-6", "-0x1.25c6aa2dcffabp-1"],
        ["0x1.13049415a4a77p-3", "0x1.e855482ea5745p-2", "0x1.43612d0adf177p-5", "-0x1.ef33c2615d38ep-2", "-0x1.708a4bf18c977p-3", "0x1.a6a60fe9fb3dp-7"],
        ["0x1.13049415a4a6fp-3", "0x1.43612d0adf177p-5", "0x1.e855482ea5742p-2", "0x1.a6a60fe9fb39p-7", "-0x1.708a4bf18c977p-3", "-0x1.ef33c2615d38cp-2"],
        ["-0x1.25c6aa2dcffaap-1", "-0x1.ef33c2615d38ep-2", "0x1.a6a60fe9fb39p-7", "0x1.3442be885f85ap+1", "-0x1.21a3f0b34f574p+0", "-0x1.dbf4967022f72p-3"],
        ["-0x1.365ee86f3477p-6", "-0x1.708a4bf18c977p-3", "-0x1.708a4bf18c977p-3", "-0x1.21a3f0b34f574p+0", "0x1.5221f8025f534p+1", "-0x1.21a3f0b34f577p+0"],
        ["-0x1.25c6aa2dcffabp-1", "0x1.a6a60fe9fb3dp-7", "-0x1.ef33c2615d38cp-2", "-0x1.dbf4967022f72p-3", "-0x1.21a3f0b34f577p+0", "0x1.3442be885f85ap+1"],
    ];
    for i in 0..6 {
        for j in 0..6 {
            assert_bitwise("diffusion", i, j, ke[i * 6 + j], want_d[i][j], 0.0);
        }
    }

    let mut km = [0.0f64; 36];
    SurfaceTri6MassIntegrator.add_to_element_matrix(&x, &mut km);
    let want_m = [
        ["0x1.5c00311e3a691p-10", "-0x1.065556d2074ffp-12", "-0x1.065556d2074f8p-12", "-0x1.55fdd6c5fddd7p-12", "-0x1.58fc4394aa14cp-10", "-0x1.55fdd6c5fddcdp-12"],
        ["-0x1.065556d2074ffp-12", "0x1.06a75ffae97dap-9", "-0x1.9b18184606436p-12", "0x1.f442472862512p-13", "0x1.086f52c570b8bp-13", "-0x1.1d4c839e7a7cfp-10"],
        ["-0x1.065556d2074f8p-12", "-0x1.9b18184606436p-12", "0x1.06a75ffae97d5p-9", "-0x1.1d4c839e7a7cdp-10", "0x1.086f52c570b85p-13", "0x1.f442472862514p-13"],
        ["-0x1.55fdd6c5fddd7p-12", "0x1.f442472862512p-13", "-0x1.1d4c839e7a7cdp-10", "0x1.1e03b84ae9ab4p-7", "0x1.3b2463999248dp-8", "0x1.1d4c839e7a7cfp-8"],
        ["-0x1.58fc4394aa14cp-10", "0x1.086f52c570b8bp-13", "0x1.086f52c570b85p-13", "0x1.3b2463999248dp-8", "0x1.5a118936b6c9p-7", "0x1.3b2463999248dp-8"],
        ["-0x1.55fdd6c5fddcdp-12", "-0x1.1d4c839e7a7cfp-10", "0x1.f442472862514p-13", "0x1.1d4c839e7a7cfp-8", "0x1.3b2463999248dp-8", "0x1.1e03b84ae9ab3p-7"],
    ];
    for i in 0..6 {
        for j in 0..6 {
            assert_bitwise("mass", i, j, km[i * 6 + j], want_m[i][j], 0.0);
        }
    }

    let rhs = |p: &[f64; 3]| {
        let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        7.0 * p[0] * p[1] / r2
    };
    let src = SurfaceTri6DomainSourceIntegrator { f: &rhs };
    let mut fe = [0.0f64; 6];
    src.add_to_element_vector(&x, &mut fe);
    let want_f = [
        "-0x1.2ff33aaab5228p-9",
        "0x1.2fcf2029c6544p-8",
        "-0x1.d27b1f056bfbp-10",
        "0x1.d73721e5197aap-7",
        "0x1.10e26cd1502aap-6",
        "0x1.dc133f4d1a833p-8",
    ];
    for (i, wh) in want_f.iter().enumerate() {
        assert_bitwise("lf", i, i, fe[i], wh, 0.0);
    }
}
