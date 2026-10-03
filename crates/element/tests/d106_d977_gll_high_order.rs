//! D977: `gauss_lobatto_1d` beyond the old n ≤ 5 table — the Gauss-Lobatto points
//! must equal MFEM's `Poly_1D::GetPoints(order, BasisType::ClosedGL)` values
//! (probe `tmp/d106kernel/d977_gll_probe.cpp`, MFEM 4.10 serial, output kept
//! verbatim in `tmp/d106kernel/d977_mfem_gll.txt`).  This is the end-to-end
//! enabler for `grad_div -o 5` (RT4 → 6 GLL nodes), which panicked before
//! D977 while the C++ miniapp ran fine (179 it, L2 1.7589e-07,
//! `tmp/d105spde/cpp_ref/grad_div_star_o5.log`).
//!
//! The n ≥ 6 path is the MFEM `QuadratureFunctions1D::GaussLobatto` Newton
//! port already pinned to the last bit for n = 6..9 in `d275_gll_mfem_ulp.rs`;
//! here the check runs through the `gauss_lobatto_1d`/`gauss_lobatto_01`
//! entry points (the ones the n ≤ 5 cap used to abort) and extends to n = 12.
//!
//! Tolerance 1e-13 on the `[0,1]` image (the probe stores %.17g of MFEM's
//! doubles; both sides are the same iteration, so agreement is ~1 ulp, but
//! the 0.5·(x+1) mapping in `gauss_lobatto_01` may differ by 1 ulp from
//! MFEM's stored `z`).

use fem_element::quadrature::gauss_lobatto_01;
/// MFEM `Poly_1D::GetPoints(order, BasisType::GaussLobatto)` on `[0,1]`, n = 2..12
/// (probe dump verbatim).
const MFEM_CLOSED_GL: [&[&str]; 11] = [
    &[
        "0",
        "1",
    ],
    &[
        "0",
        "0.5",
        "1",
    ],
    &[
        "0",
        "0.27639320225002106",
        "0.72360679774997894",
        "1",
    ],
    &[
        "0",
        "0.17267316464601146",
        "0.5",
        "0.82732683535398854",
        "1",
    ],
    &[
        "0",
        "0.11747233803526766",
        "0.35738424175967742",
        "0.64261575824032258",
        "0.88252766196473231",
        "1",
    ],
    &[
        "0",
        "0.084888051860716532",
        "0.26557560326464291",
        "0.5",
        "0.73442439673535709",
        "0.91511194813928343",
        "1",
    ],
    &[
        "0",
        "0.064129925745196686",
        "0.20414990928342885",
        "0.39535039104876057",
        "0.60464960895123943",
        "0.79585009071657109",
        "0.93587007425480329",
        "1",
    ],
    &[
        "0",
        "0.050121002294269933",
        "0.16140686024463113",
        "0.31844126808691092",
        "0.5",
        "0.68155873191308913",
        "0.83859313975536887",
        "0.94987899770573003",
        "1",
    ],
    &[
        "0",
        "0.040233045916770585",
        "0.13061306744724746",
        "0.26103752509477773",
        "0.4173605211668065",
        "0.58263947883319345",
        "0.73896247490522227",
        "0.86938693255275257",
        "0.95976695408322943",
        "1",
    ],
    &[
        "0",
        "0.032999284795970446",
        "0.10775826316842778",
        "0.21738233650189751",
        "0.35212093220653029",
        "0.5",
        "0.64787906779346971",
        "0.78261766349810247",
        "0.89224173683157226",
        "0.96700071520402953",
        "1",
    ],
    &[
        "0",
        "0.027550363888558894",
        "0.090360339177996657",
        "0.18356192348406966",
        "0.30023452951732554",
        "0.43172353357253623",
        "0.56827646642746377",
        "0.69976547048267446",
        "0.81643807651593037",
        "0.90963966082200332",
        "0.97244963611144108",
        "1",
    ],
];
#[test]
fn d977_gauss_lobatto_01_matches_mfem_closed_gl_to_n12() {
    for (n, want) in MFEM_CLOSED_GL.iter().enumerate() {
        let n = n + 2;
        let (pts, wts) = gauss_lobatto_01(n);
        assert_eq!(pts.len(), n, "n = {n}: point count");
        assert_eq!(wts.len(), n, "n = {n}: weight count");
        for (i, w) in want.iter().enumerate() {
            let expected: f64 = w.parse().unwrap_or_else(|_| {
                panic!("n = {n}, i = {i}: MFEM literal {w:?} must parse");
            });
            let scale = expected.abs().max(1.0);
            assert!(
                (pts[i] - expected).abs() <= 1e-13 * scale,
                "n = {n}, i = {i}: point {} vs MFEM {expected} (diff {:.3e})",
                pts[i],
                (pts[i] - expected).abs()
            );
        }
        // Weights sum to 1 on [0,1] (up to roundoff).
        let sum: f64 = wts.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-13,
            "n = {n}: weight sum {sum} != 1"
        );
        // Symmetry.
        for i in 0..n / 2 {
            let s = wts[i] - wts[n - 1 - i];
            assert!(s.abs() < 1e-13, "n = {n}: weight symmetry broken at {i}");
        }
    }
}

/// Quadrature exactness sanity: the seg lobatto rule integrates x^k exactly
/// through its stated degree (order → n = (order+4)/2).
#[test]
fn d977_seg_lobatto_rule_exact_beyond_n5() {
    use fem_element::quadrature::seg_lobatto_rule;
    for order in [7u8, 9, 11, 13] {
        let rule = seg_lobatto_rule(order);
        // ∫_0^1 x^order dx = 1/(order+1).
        let mut integral = 0.0_f64;
        for (xi, wi) in rule.points.iter().zip(&rule.weights) {
            integral += wi * xi[0].powi(order as i32);
        }
        let exact = 1.0 / (order as f64 + 1.0);
        assert!(
            (integral - exact).abs() < 1e-13,
            "order {order}: ∫x^p = {integral} vs exact {exact}"
        );
    }
}
