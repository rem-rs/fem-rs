//! D369: pin the TMOP `TmopQuadType::ClosedUniform` quadrature (`-qt 3`,
//! MFEM `IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform)`)
//! against the MFEM 4.10 oracle.
//!
//! All numbers below are verbatim output of the C++ probe
//! `tmp/d369/qt3_probe.cpp` (source kept next to this repo at
//! `../tmp/d369/qt3_probe.cpp`), compiled against MFEM 4.10
//! (`g++ -std=c++17 -O2 -I$HOME/mfem410_ser qt3_probe.cpp
//! $HOME/mfem410_ser/libmfem.a`) and run 2026-09-18.
//!
//! MFEM facts pinned here (fem/intrules.cpp, MFEM 4.10):
//! - `QuadratureFunctions1D::ClosedUniform` (:856): equally spaced nodes
//!   `x_i = i/(np-1)` on `[0,1]`, endpoints included; weights from
//!   `CalculateUniformWeights` (:964) = exact Gauss-Legendre integration of
//!   the nodal Lagrange basis (classic Newton-Cotes, negative interior
//!   weights allowed from np = 9 on).
//! - `IntegrationRules::SegmentIntegrationRule` (:1029): ClosedUniform
//!   segment rule uses `n = Order | 1` points (always odd).
//! - SQUARE/CUBE rules (:1861/:2533) are the tensor products of that 1-D
//!   rule; TRIANGLE/TETRAHEDRON/PRISM do **not** depend on `quad_type`
//!   (Witherden-Vincent rules, identical under `-qt 2`/`-qt 3` — the
//!   `tri=`/`tet=`/`pri=` columns equal the `qt2` probe section).
use fem_assembly::tmop_form::quadrature_functions_1d_closed_uniform;

/// `(order, seg_np, tri, sqr, sqr_w_sum, tet, hex, hex_w_sum, pri)` straight
/// from the probe's `qt3 order N ...` lines.  `tri`/`tet`/`pri` are listed
/// for provenance only (they are the quad_type-independent W&V rules); the
/// assertions cover `seg_np` and the tensor products `sqr`/`hex` with their
/// weight sums.
const PROBE: [(usize, usize, usize, usize, f64, usize, usize, f64, usize); 11] = [
    (2, 3, 3, 9, 0.99999999999999978, 4, 27, 0.99999999999999978, 9),
    (3, 3, 6, 9, 0.99999999999999978, 8, 27, 0.99999999999999978, 18),
    (4, 5, 6, 25, 0.99999999999999956, 14, 125, 0.99999999999999911, 30),
    (5, 5, 7, 25, 0.99999999999999956, 14, 125, 0.99999999999999911, 35),
    (6, 7, 12, 49, 0.99999999999999911, 24, 343, 0.99999999999999956, 84),
    (7, 7, 15, 49, 0.99999999999999911, 35, 343, 0.99999999999999956, 105),
    (8, 9, 16, 81, 1.0000000000000002, 46, 729, 1.0000000000000009, 144),
    (9, 9, 19, 81, 1.0000000000000002, 59, 729, 1.0000000000000009, 171),
    (10, 11, 25, 121, 1.0000000000000011, 81, 1331, 1.0000000000000004, 275),
    (11, 11, 28, 121, 1.0000000000000011, 96, 1331, 1.0000000000000004, 308),
    (12, 13, 33, 169, 1.0000000000000024, 123, 2197, 1.0000000000000053, 429),
];

/// Probe's `nc1d` lines: full 1-D ClosedUniform (Newton-Cotes) rules.
/// The np = 9 weights are MFEM-verbatim, including the negative interior
/// weights of the classic closed Newton-Cotes formula and the tiny
/// asymmetry between the first/last endpoint weights (GL-evaluation
/// round-off in MFEM itself).
const NC1D: [(usize, &[f64], &[f64]); 4] = [
    (
        3,
        &[0.0, 0.5, 1.0],
        &[
            0.16666666666666663,
            0.66666666666666663,
            0.16666666666666663,
        ],
    ),
    (
        5,
        &[0.0, 0.25, 0.5, 0.75, 1.0],
        &[
            0.077777777777777793,
            0.35555555555555551,
            0.13333333333333328,
            0.35555555555555551,
            0.077777777777777793,
        ],
    ),
    (
        7,
        &[
            0.0,
            0.16666666666666666,
            0.33333333333333331,
            0.5,
            0.66666666666666663,
            0.83333333333333337,
            1.0,
        ],
        &[
            0.048809523809523796,
            0.2571428571428569,
            0.032142857142857167,
            0.32380952380952366,
            0.032142857142857195,
            0.25714285714285695,
            0.048809523809523837,
        ],
    ),
    (
        9,
        &[
            0.0,
            0.125,
            0.25,
            0.375,
            0.5,
            0.625,
            0.75,
            0.875,
            1.0,
        ],
        &[
            0.034885361552028218,
            0.20768959435626111,
            -0.032733686067019575,
            0.37022927689594376,
            -0.16014109347442695,
            0.37022927689594376,
            -0.032733686067019402,
            0.20768959435626094,
            0.034885361552028267,
        ],
    ),
];

/// 1-D rules match the probe bit-for-bit on points and to ~1 ulp on weights
/// (MFEM evaluates the Lagrange basis through `Poly_1D::Basis`, this port
/// uses the direct product form; both integrate the same exact rule).
#[test]
fn nc1d_rules_match_probe() {
    for (np, pts, wts) in NC1D {
        let (xs, ws) = quadrature_functions_1d_closed_uniform(np);
        assert_eq!(xs.len(), np, "np={np}");
        assert_eq!(ws.len(), np, "np={np}");
        for (i, (&x, &px)) in xs.iter().zip(pts.iter()).enumerate() {
            assert_eq!(x, px, "point {i} of np={np}");
        }
        for (i, (&w, &pw)) in ws.iter().zip(wts.iter()).enumerate() {
            assert!(
                (w - pw).abs() <= 1e-15 * pw.abs().max(1.0),
                "weight {i} of np={np}: {w} vs {pw}"
            );
        }
    }
}

/// Segment `n = Order | 1` mapping and the SQUARE/CUBE tensor products:
/// point counts and weight sums across orders 2..=12.
#[test]
fn segment_and_tensor_counts_match_probe() {
    for &(order, seg_np, _tri, sqr, sqr_w, tet, hex, hex_w, _pri) in PROBE.iter() {
        // `IntegrationRules::SegmentIntegrationRule` ClosedUniform arm.
        let n = order | 1;
        assert_eq!(n, seg_np, "segment np at order {order}");
        let (xs, ws) = quadrature_functions_1d_closed_uniform(n);
        let seg_sum: f64 = ws.iter().sum();
        assert!(
            (seg_sum - 1.0).abs() < 1e-13,
            "segment weight sum at order {order}: {seg_sum}"
        );

        // SQUARE = tensor product (probe: `sqr=... sqrw=...`).
        assert_eq!(xs.len() * xs.len(), sqr, "square np at order {order}");
        let mut sqr_sum = 0.0;
        for wi in &ws {
            for wj in &ws {
                sqr_sum += wi * wj;
            }
        }
        assert!(
            (sqr_sum - sqr_w).abs() < 1e-13,
            "square weight sum at order {order}: {sqr_sum} vs {sqr_w}"
        );

        // CUBE = triple tensor product (probe: `hex=... hexw=...`).
        assert_eq!(xs.len().pow(3), hex, "cube np at order {order}");
        let mut hex_sum = 0.0;
        for wi in &ws {
            for wj in &ws {
                for wk in &ws {
                    hex_sum += wi * wj * wk;
                }
            }
        }
        assert!(
            (hex_sum - hex_w).abs() < 1e-12,
            "cube weight sum at order {order}: {hex_sum} vs {hex_w}"
        );
        // Tet counts stay quad-type-independent (provenance).
        assert!(tet > 0 && hex > 0);
    }
}

/// Endpoint structure at order 8 (`-qo 8`, the mesh-optimizer default):
/// the square rule starts at (0,0) and ends at (1,1) with the squared 1-D
/// endpoint weights, matching the probe's `sqr first=/last=` lines to ~1 ulp
/// (MFEM's own first/last weights differ in the last digits).
#[test]
fn square_endpoints_order8_match_probe() {
    let n = 8 | 1;
    let (xs, ws) = quadrature_functions_1d_closed_uniform(n);
    let w0 = ws[0];
    let wn = ws[n - 1];
    let (first, last) = (w0 * w0, wn * wn);
    assert!((first - 0.0012169884506157288).abs() < 1e-15, "{first}");
    assert!((last - 0.001216988450615732).abs() < 1e-15, "{last}");
    assert_eq!(xs[0], 0.0);
    assert_eq!(xs[n - 1], 1.0);
}
