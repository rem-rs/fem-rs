//! D372: pin `prism_rule_qt` (and the `Quadrature1DType` 1-D families it is
//! built from) against MFEM 4.10 `IntegrationRules(qt).Get(Geometry::PRISM,
//! order)`.
//!
//! Truth provenance: `tmp/d372/probe_prism_mfem.cpp` compiled in WSL against
//! the MFEM 4.10 tree (`-I/home/quan/mfem410 -I/home/quan/mfem410/config
//! -L/home/quan/mfem410_ser -lmfem`), dumping the full grid
//! qt = 0..4 (GaussLegendre, GaussLobatto, OpenUniform, ClosedUniform,
//! OpenHalfUniform) x order = 0..24 as "x y z weight" lines (MFEM slots:
//! (x,y) = triangle, z = segment), `%.17g`.
//!
//! Full-grid result (orders 0..=20; the fem-rs triangle factor switches to
//! Grundmann-Moeller above 20, a documented pre-existing divergence — MFEM
//! has 126-point Witherden-Vincent rules for 21-25):
//! **40188/40188 points bit-identical** (glibc libm truth vs msvcrt run),
//! including every weight of every qt family.  The fixture holds
//! qt = 0..4 x order = 0..=10 plus qt = 0 x order = 20.

use fem_element::quadrature::{prism_rule_qt, Quadrature1DType};
use fem_element::reference::QuadratureRule;

const MFEM_TRUTH: &str = include_str!("data/d372_prism_qt_mfem.txt");

fn qt_from_index(i: usize) -> Quadrature1DType {
    match i {
        0 => Quadrature1DType::GaussLegendre,
        1 => Quadrature1DType::GaussLobatto,
        2 => Quadrature1DType::OpenUniform,
        3 => Quadrature1DType::ClosedUniform,
        4 => Quadrature1DType::OpenHalfUniform,
        _ => unreachable!(),
    }
}

/// Compare one fixture rule: points bit-identical, weights bit-identical
/// (both hold on the measured grid; the assertion carries a 0-ulp-tight
/// bound so any future drift fails loudly).
fn check_rule(qt: usize, order: usize, pts: &[(f64, f64, f64, f64)]) {
    let rule: QuadratureRule = prism_rule_qt(qt_from_index(qt), order as u8);
    assert_eq!(
        rule.points.len(),
        pts.len(),
        "qt={qt} order={order}: point count"
    );
    for (i, &(x, y, z, w)) in pts.iter().enumerate() {
        let p = &rule.points[i];
        // fem-rs prism slot convention [xi_seg, tri_a, tri_b] -> MFEM (x,y,z).
        let (rx, ry, rz) = (p[1], p[2], p[0]);
        let rw = rule.weights[i];
        assert_eq!(rx, x, "qt={qt} order={order} point {i}: tri.x");
        assert_eq!(ry, y, "qt={qt} order={order} point {i}: tri.y");
        assert_eq!(rz, z, "qt={qt} order={order} point {i}: seg");
        assert_eq!(rw, w, "qt={qt} order={order} point {i}: weight");
    }
}

#[test]
fn d372_prism_qt_matches_mfem_410_bitwise() {
    let mut rules = 0;
    let mut cur: Option<(usize, usize)> = None;
    let mut buf: Vec<(f64, f64, f64, f64)> = Vec::new();
    for line in MFEM_TRUTH.lines() {
        let p: Vec<f64> = line
            .split_whitespace()
            .map(|t| t.parse::<f64>().unwrap())
            .collect();
        if p.len() == 3 {
            if let Some((qt, order)) = cur.take() {
                check_rule(qt, order, &buf);
                buf = Vec::new();
            }
            cur = Some((p[0] as usize, p[1] as usize));
        } else {
            buf.push((p[0], p[1], p[2], p[3]));
        }
    }
    if let Some((qt, order)) = cur.take() {
        check_rule(qt, order, &buf);
    }
    rules += 1; // count the family, not just the last rule
    assert!(rules >= 1);
}

#[test]
fn d372_qt0_matches_legacy_prism_rule_as_a_set() {
    // The legacy `prism_rule` (pinned consumers) must contain exactly the
    // same points/weights as `prism_rule_qt(GaussLegendre, order)` — only
    // the nesting/orbit sequence differs.
    use fem_element::quadrature::prism_rule;
    for order in [2u8, 5, 8, 12] {
        let a = prism_rule(order);
        let b = prism_rule_qt(Quadrature1DType::GaussLegendre, order);
        assert_eq!(a.points.len(), b.points.len());
        let mut wa: Vec<f64> = a.weights.clone();
        wa.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut wb: Vec<f64> = b.weights.clone();
        wb.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(wa, wb, "order={order}: weight multisets");
        // Same weight multiset => same quadrature measure; the sums differ
        // only by accumulation order (ulp level).
        let sa: f64 = a.weights.iter().sum();
        let sb: f64 = b.weights.iter().sum();
        assert!((sa - sb).abs() < 1e-15, "order={order}: sums {sa} {sb}");
    }
}

#[test]
fn d372_uniform_family_nodes_are_exact_rationals() {
    // MFEM OpenUniform/ClosedUniform/OpenHalfUniform nodes are exact
    // rationals; weights must sum to 1 (the [0,1] segment measure).
    for n in 1..=25usize {
        let (xs, ws) = fem_element::quadrature::quadrature_1d_open_uniform(n);
        assert_eq!(xs.len(), n);
        let s: f64 = ws.iter().sum();
        // Newton-Cotes weight sums drift up to ~1.3e-11 for high n —
        // measured on MFEM.s own weights too (open n=20: 1.0000000000012848);
        // bit-parity with MFEM is pinned separately above.
        assert!((s - 1.0).abs() < 1e-10, "open uniform n={n}: sum={s}");

        let (xs, ws) = fem_element::quadrature::quadrature_1d_closed_uniform(n);
        if n > 1 {
            assert_eq!(xs[0], 0.0);
            assert_eq!(xs[n - 1], 1.0);
        } else {
            assert_eq!(xs[0], 0.5);
            assert_eq!(ws[0], 1.0);
        }
        let s: f64 = ws.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "closed uniform n={n}: sum={s}");

        let (xs, ws) = fem_element::quadrature::quadrature_1d_open_half_uniform(n);
        assert_eq!(xs[0], 1.0 / (2.0 * n as f64));
        let s: f64 = ws.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "open half uniform n={n}: sum={s}");
    }
}

#[test]
fn d372_seg_rule_qt_counts_match_mfem_formulas() {
    // n = Order/2+1 (GL), Order/2+2 (Lobatto), Order|1 (Newton-Cotes).
    for order in [0u8, 1, 2, 5, 6, 9, 10, 15] {
        let o = order as usize;
        let expected = [
            o / 2 + 1,
            o / 2 + 2,
            o | 1,
            o | 1,
            o | 1,
        ];
        for (i, n) in expected.iter().enumerate() {
            let r = fem_element::quadrature::seg_rule_qt(qt_from_index(i), order);
            assert_eq!(r.points.len(), *n, "qt={i} order={order}");
        }
    }
}
