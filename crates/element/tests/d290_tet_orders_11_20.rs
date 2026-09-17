//! D290: pin of the fem-rs tetrahedron positive-weight rules of orders 11-20
//! against the MFEM 4.10 `IntegrationRules::TetrahedronIntegrationRule`
//! tables (Witherden-Vincent 11-13, Chuluunbaatar et al. 14-20).  Before
//! D290 these orders fell through to the Grundmann-Moller fallback
//! (negative weights); the tabulated replacement has all-positive weights.
//!
//! Permanent checks per order (11..=20): MFEM point count, all weights
//! strictly positive, weight sum = 1/6, and the minimum weight **bit-equal**
//! to the MFEM table value (order-independent quantity parsed from the
//! %.17e probe dump — full 17-digit round-trip).  Together with the
//! in-crate monomial exactness test (`tet_pos_orders_11_to_20_exact_to_degree`)
//! any transcription typo is caught.
//!
//! Point-by-point (%.17e) comparison against the C++ tables is done via
//! dump files: run with `D290_DUMP_DIR=<dir>` to emit `tet_rule_rust_d290.txt`
//! in the same sorted format as `tmp/d319/tet_rule_probe_d290.cpp`, then
//! diff field-by-field (see `tmp/d319/EVIDENCE.md`).
//!
//! ```bash
//! cargo test -p fem-element --test d290_tet_orders_11_20 -- --nocapture
//! ```

use fem_element::quadrature::tet_rule;

/// MFEM 4.10 minimum weights per order 11-20 (sorted %.17e probe dump
/// `tmp/d319/tet_rule_mfem_d290.txt`); bit-exact pins.
const MFEM_WMIN: [(u8, f64); 10] = [
    (11, 1.95152894059845476e-04),
    (12, 1.42695998696545142e-04),
    (13, 1.11328544338301012e-04),
    (14, 3.74818592914694638e-05),
    (15, 6.94170071688395132e-05),
    (16, 5.70730781338430580e-05),
    (17, 3.37712309004839736e-05),
    (18, 2.96638661291863846e-05),
    (19, 2.06414634761428247e-05),
    (20, 7.60496594911488061e-06),
];

/// MFEM 4.10 point counts per order 11-20.
const MFEM_NPTS: [(u8, usize); 10] = [
    (11, 96), (12, 123), (13, 145), (14, 175), (15, 209),
    (16, 248), (17, 284), (18, 343), (19, 383), (20, 441),
];

#[test]
fn tet_rules_11_20_point_counts_and_positive_weights() {
    for (order, npts) in MFEM_NPTS {
        let r = tet_rule(order);
        assert_eq!(r.points.len(), npts, "order={order}");
        assert_eq!(r.weights.len(), npts, "order={order}");
        for (i, &w) in r.weights.iter().enumerate() {
            assert!(w > 0.0, "order={order}: weight[{i}]={w} not positive");
            let (x, y, z) = (r.points[i][0], r.points[i][1], r.points[i][2]);
            // MFEM's order-11 table contains a degenerate S211 orbit whose
            // `cb = 1 - 2a - bc` evaluates to exactly 0.0, so coordinates of
            // exactly 0 (points on the reference-tet boundary) are legitimate.
            assert!(
                x.is_finite() && y.is_finite() && z.is_finite()
                    && x >= 0.0 && y >= 0.0 && z >= 0.0
                    && x + y + z <= 1.0 + 1e-12,
                "order={order}: point[{i}] = ({x},{y},{z}) outside reference tet"
            );
        }
        let wsum: f64 = r.weights.iter().sum();
        assert!(
            (wsum - 1.0 / 6.0).abs() < 1e-14,
            "order={order}: weight sum {wsum:.17e} != 1/6"
        );
    }
}

#[test]
fn tet_rules_11_20_min_weight_bit_match_mfem() {
    for (order, wmin) in MFEM_WMIN {
        let r = tet_rule(order);
        let wmin_rs = r.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        assert_eq!(
            wmin_rs, wmin,
            "order={order}: min weight {wmin_rs:.17e} != MFEM {wmin:.17e}"
        );
    }
}

/// Emit the sorted (x, y, z, w) dump for orders 3-20 in the C++ probe
/// format when `D290_DUMP_DIR` is set (point-by-point diff evidence).
#[test]
fn dump_rules_for_mfem_diff() {
    let dir = match std::env::var("D290_DUMP_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => return, // silent no-op in normal test runs
    };
    let mut out = String::new();
    for order in 3u8..=20 {
        let r = tet_rule(order);
        let mut rows: Vec<[f64; 4]> = r
            .points
            .iter()
            .zip(r.weights.iter())
            .map(|(p, &w)| [p[0], p[1], p[2], w])
            .collect();
        rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let wsum: f64 = r.weights.iter().sum();
        let wmin = r.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        out.push_str(&format!(
            "RULE order={order} npts={} wsum={wsum:.17e} wmin={wmin:.17e}\n",
            r.points.len()
        ));
        for p in &rows {
            out.push_str(&format!(
                "  {:+.17e} {:+.17e} {:+.17e} {:+.17e}\n",
                p[0], p[1], p[2], p[3]
            ));
        }
    }
    let path = std::path::Path::new(&dir).join("tet_rule_rust_d290.txt");
    std::fs::write(&path, out).expect("write dump");
}
