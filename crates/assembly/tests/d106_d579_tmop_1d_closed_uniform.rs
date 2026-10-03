//! D579: the 1-D piece of the TMOP path — `quadrature_functions_1d_closed_uniform`
//! (MFEM `IntegrationRules` with `Quadrature1D::ClosedUniform`, the tmop
//! miniapps' `-qt 1` / `QuadratureFunctions1D::ClosedUniform`) pinned against
//! the full MFEM 4.10 segment table (`tmp/d106kernel/d579_tmop_1d_probe.cpp`,
//! output verified 2026-10-02).
//!
//! Premise correction (why this is only a rule pin): the coverage matrix's
//! "1-D 重心式 LAT" reading of *metric* semantics on 1×1 Jacobians is
//! **refuted against MFEM itself** — `TMOP_Metric_001::EvalW` on a 1×1
//! `DenseMatrix` corrupts the MFEM heap (`InvariantsEvaluator2D` reads the
//! 2×2 block from the 1-entry buffer; probe crashed with
//! `malloc(): corrupted top size`).  MFEM 4.10 defines no 1-D TMOP metric
//! semantics, so there is nothing to align there; the 1-D TMOP capability is
//! exactly this quadrature rule (plus the 2-D/3-D metric zoo already pinned
//! by d102).

use fem_assembly::tmop_form::quadrature_functions_1d_closed_uniform;

/// MFEM `IntegrationRules(0, Quadrature1D::ClosedUniform).Get(SEGMENT, qo)`
/// for qo = 1..8: `(x, w)` per point, %.17g.
const MFEM_SEG_NC_CLOSED: [[(&str, &str); 9]; 8] = [
    [("0.5", "1"), ("", ""), ("", ""), ("", ""), ("", ""), ("", ""), ("", ""), ("", ""), ("", "")],
    [
        ("0", "0.16666666666666663"),
        ("0.5", "0.66666666666666663"),
        ("1", "0.16666666666666663"),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
    ],
    [
        ("0", "0.16666666666666663"),
        ("0.5", "0.66666666666666663"),
        ("1", "0.16666666666666663"),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
    ],
    [
        ("0", "0.077777777777777793"),
        ("0.25", "0.35555555555555551"),
        ("0.5", "0.13333333333333328"),
        ("0.75", "0.35555555555555551"),
        ("1", "0.077777777777777793"),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
    ],
    [
        ("0", "0.077777777777777793"),
        ("0.25", "0.35555555555555551"),
        ("0.5", "0.13333333333333328"),
        ("0.75", "0.35555555555555551"),
        ("1", "0.077777777777777793"),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
    ],
    [
        ("0", "0.048809523809523796"),
        ("0.16666666666666666", "0.2571428571428569"),
        ("0.33333333333333331", "0.032142857142857167"),
        ("0.5", "0.32380952380952366"),
        ("0.66666666666666663", "0.032142857142857195"),
        ("0.83333333333333337", "0.25714285714285695"),
        ("1", "0.048809523809523837"),
        ("", ""),
        ("", ""),
    ],
    [
        ("0", "0.048809523809523796"),
        ("0.16666666666666666", "0.2571428571428569"),
        ("0.33333333333333331", "0.032142857142857167"),
        ("0.5", "0.32380952380952366"),
        ("0.66666666666666663", "0.032142857142857195"),
        ("0.83333333333333337", "0.25714285714285695"),
        ("1", "0.048809523809523837"),
        ("", ""),
        ("", ""),
    ],
    [
        ("0", "0.034885361552028218"),
        ("0.125", "0.20768959435626111"),
        ("0.25", "-0.032733686067019575"),
        ("0.375", "0.37022927689594376"),
        ("0.5", "-0.16014109347442695"),
        ("0.625", "0.37022927689594376"),
        ("0.75", "-0.032733686067019402"),
        ("0.875", "0.20768959435626094"),
        ("1", "0.034885361552028267"),
    ],
];

#[test]
fn d579_closed_uniform_matches_mfem_segment_table() {
    // fem-rs takes the node COUNT np; MFEM's `Get(SEGMENT, qo)` returns the
    // rule with npts = qo|1 points (odd qo shares the qo-1 rule), so
    // np = qo if qo is odd, else qo + 1.
    for (qo, want) in MFEM_SEG_NC_CLOSED.iter().enumerate() {
        let qo = qo + 1;
        let np = if qo % 2 == 1 { qo } else { qo + 1 };
        let (xs, ws) = quadrature_functions_1d_closed_uniform(np);
        let npts = want.iter().filter(|(x, _)| !x.is_empty()).count();
        assert_eq!(xs.len(), npts, "qo = {qo}: point count");
        for (i, (wx, ww)) in want.iter().enumerate() {
            if wx.is_empty() {
                break;
            }
            let ex: f64 = wx.parse().unwrap();
            let ew: f64 = ww.parse().unwrap();
            assert!(
                (xs[i] - ex).abs() <= 1e-13,
                "qo = {qo}, i = {i}: x {} vs MFEM {ex}",
                xs[i]
            );
            assert!(
                (ws[i] - ew).abs() <= 1e-13,
                "qo = {qo}, i = {i}: w {} vs MFEM {ew}",
                ws[i]
            );
        }
        // Weights sum to 1 (MFEM comment: `SegmentIntegrationRule` closed NC).
        let sum: f64 = ws.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "qo = {qo}: weight sum {sum}");
    }
}
