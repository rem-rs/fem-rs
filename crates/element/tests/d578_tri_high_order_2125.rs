//! D578: the MFEM 4.10 triangle rules for orders 21-25 — the single shared
//! 126-point rule of `fem/intrules.cpp` case `21: 22: 23: 24: 25:` — are
//! ported and [`fem_element::quadrature::tri_rule`] dispatches to them
//! (previously orders >= 21 fell back to Grundmann-Möller, which diverged
//! from MFEM on this segment).
//!
//! Truth provenance: `tmp/d580/probe/probe_tri_2125.cpp` compiled in WSL
//! against the MFEM 4.10 tree (`g++ -std=c++17 -O2 -I$HOME/mfem410_ser x.cpp
//! $HOME/mfem410_ser/libmfem.a`), dumping `IntRules.Get(TRIANGLE, 21..=25)`:
//! * `data/d578_tri_2125_mfem.txt` — "order npts rule_order" header +
//!   "x y weight" lines, `%.17g` (exact f64 round-trip),
//! * `data/d578_tri_2125_mfem_bits.txt` — the same values as raw f64 bit
//!   patterns (`%016llx`), pinning the port at the bit level.
//!
//! The prism-side consequence (the same triangle factor entering
//! `prism_rule_qt` at orders 21-25) is pinned by the d372 extension test
//! `d372_prism_qt_matches_mfem_410_bitwise_o21_25`.

use fem_element::quadrature::tri_rule;

/// Parse the probe text fixture into (order, npts, rule_order, points+weights)
/// blocks.  Headers ("order npts rule_order") are all-integer lines; point
/// lines are "x y weight" floats.
fn parse_tri_fixture(text: &str) -> Vec<(usize, usize, usize, Vec<(f64, f64, f64)>)> {
    let mut blocks = Vec::new();
    let mut cur: Option<(usize, usize, usize)> = None;
    let mut buf: Vec<(f64, f64, f64)> = Vec::new();
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let ints: Option<Vec<usize>> =
            tokens.iter().map(|t| t.parse::<usize>().ok()).collect();
        if let Some(h) = ints {
            assert_eq!(h.len(), 3, "header line: {line}");
            if let Some(header) = cur.take() {
                blocks.push((header.0, header.1, header.2, std::mem::take(&mut buf)));
            }
            cur = Some((h[0], h[1], h[2]));
        } else {
            let p: Vec<f64> = tokens.iter().map(|t| t.parse().unwrap()).collect();
            assert_eq!(p.len(), 3, "point line: {line}");
            buf.push((p[0], p[1], p[2]));
        }
    }
    if let Some(header) = cur.take() {
        blocks.push((header.0, header.1, header.2, buf));
    }
    blocks
}

/// Every order 21..=25 reproduces the MFEM rule bit-for-bit (text level,
/// 0-ulp assert), and MFEM's rule carries order 25 for all five requests.
#[test]
fn d578_tri_rule_21_25_matches_mfem_410_bitwise() {
    let blocks = parse_tri_fixture(include_str!("data/d578_tri_2125_mfem.txt"));
    assert_eq!(blocks.len(), 5, "orders 21..=25");
    for (order, npts, rule_order, pts) in &blocks {
        assert!((21..=25).contains(order));
        assert_eq!(*npts, 126, "order {order}: 126 points");
        assert_eq!(*rule_order, 25, "order {order}: MFEM SetOrder(25)");

        let rule = tri_rule(*order as u8);
        assert_eq!(rule.points.len(), *npts, "order {order}: point count");
        for (i, &(x, y, w)) in pts.iter().enumerate() {
            assert_eq!(rule.points[i][0], x, "order {order} point {i}: x");
            assert_eq!(rule.points[i][1], y, "order {order} point {i}: y");
            assert_eq!(rule.weights[i], w, "order {order} point {i}: weight");
        }
    }
}

/// Bit-level version of the same pin via the raw f64 bit-pattern dump —
/// `to_bits` equality for every coordinate and weight.
#[test]
fn d578_tri_rule_21_25_matches_mfem_410_bits() {
    fn finish(
        cur: &mut Option<(usize, usize)>,
        buf: &mut Vec<(u64, u64, u64)>,
        checked: &mut usize,
    ) {
        if let Some((order, npts)) = cur.take() {
            let rule = tri_rule(order as u8);
            assert_eq!(buf.len(), npts);
            for (i, &(xb, yb, wb)) in buf.iter().enumerate() {
                assert_eq!(rule.points[i][0].to_bits(), xb, "order {order} point {i}: x bits");
                assert_eq!(rule.points[i][1].to_bits(), yb, "order {order} point {i}: y bits");
                assert_eq!(rule.weights[i].to_bits(), wb, "order {order} point {i}: weight bits");
            }
            *checked += 1;
            buf.clear();
        }
    }

    let mut cur: Option<(usize, usize)> = None;
    let mut buf: Vec<(u64, u64, u64)> = Vec::new();
    let mut checked = 0usize;
    for line in include_str!("data/d578_tri_2125_mfem_bits.txt").lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() == 2 {
            finish(&mut cur, &mut buf, &mut checked);
            cur = Some((
                tokens[0].parse().unwrap(),
                tokens[1].parse().unwrap(),
            ));
        } else {
            assert_eq!(tokens.len(), 3, "bit line: {line}");
            buf.push((
                u64::from_str_radix(tokens[0], 16).unwrap(),
                u64::from_str_radix(tokens[1], 16).unwrap(),
                u64::from_str_radix(tokens[2], 16).unwrap(),
            ));
        }
    }
    finish(&mut cur, &mut buf, &mut checked);
    assert_eq!(checked, 5, "five order blocks pinned at the bit level");
}

/// MFEM serves ONE rule object for orders 21-25 (`TriangleIntRules[21] = ...
/// = [25] = ir`): the five fem-rs requests must be the identical rule too.
#[test]
fn d578_tri_21_25_share_one_rule() {
    let r21 = tri_rule(21);
    for order in 22..=25u8 {
        let r = tri_rule(order);
        assert_eq!(r.points, r21.points, "order {order}");
        assert_eq!(r.weights, r21.weights, "order {order}");
    }
}

/// Sanity on the shared rule: positive weights, interior points, total mass
/// = the reference-triangle area 0.5.
#[test]
fn d578_tri_21_25_positive_interior_mass_half() {
    let r = tri_rule(23);
    let total: f64 = r.weights.iter().sum();
    assert!((total - 0.5).abs() < 1e-13, "weights sum to {total}");
    for p in &r.points {
        let (x, y) = (p[0], p[1]);
        assert!(x > 0.0 && y > 0.0 && x + y < 1.0, "interior: ({x}, {y})");
    }
    assert!(r.weights.iter().all(|&w| w > 0.0));
}
