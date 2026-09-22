//! D603: the Grundmann-Möller fallback of [`tri_rule`] / `tri_rule_mfem_order`
//! (triangle orders > 25) must round the order up exactly like MFEM 4.10's
//! `IntegrationRules::TriangleIntegrationRule` default branch:
//!
//! ```cpp
//! int i = (Order / 2) * 2 + 1;   // closest odd >= Order
//! ir->GrundmannMollerSimplexRule(i/2, 2);
//! ```
//!
//! i.e. GM level `s = order / 2` (integer division) for BOTH parities, so an
//! even order is served by the rule of the NEXT odd order (order 26 -> s = 13,
//! 560 points = C(16,3), exact degree 27).  fem-rs used `s = (order-1)/2`,
//! which served even orders with the PREVIOUS odd order's rule (order 26 ->
//! s = 12, 455 points = C(15,3), exact degree 25 < 26 — wrong exactness).
//!
//! Truth provenance: `tmp/d603/tri_gm_probe.cpp` (archived), compiled in WSL
//! against the MFEM 4.10 tree
//! (`g++ -std=c++17 -O2 -I$HOME/mfem410_ser tri_gm_probe.cpp
//!   $HOME/mfem410_ser/libmfem.a -o tri_gm_probe`), dumping
//! `IntRules.Get(TRIANGLE, 26..=33)` and `IntRules.Get(PRISM, {26, 28})`:
//! * `data/d603_tri_gm_mfem.txt` — "order npts rule_order" header + "x y
//!   weight" lines, `%.17g` (exact f64 round-trip),
//! * `data/d603_tri_gm_mfem_bits.txt` — the same values as raw f64 bit
//!   patterns (`%016llx`), pinning points (and weights, see below) at the
//!   bit level.
//!
//! MFEM ground truth (probe headers): order 26/27 -> 560 pts, rule order 27;
//! 28/29 -> 680/29; 30/31 -> 816/31; 32/33 -> 969/33.  PRISM 26 -> 7840 =
//! 560 x 14 (Gauss-Legendre segment factor), PRISM 28 -> 10200 = 680 x 15.

use fem_element::quadrature::{prism_rule_qt, tri_rule, Quadrature1DType};

/// One expected block of the text fixture: (order, npts, rule_order, points).
type Block = (usize, usize, usize, Vec<(f64, f64, f64)>);

fn parse_text_fixture(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut cur: Option<(usize, usize, usize)> = None;
    let mut buf: Vec<(f64, f64, f64)> = Vec::new();
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if let Ok(order) = tokens[0].parse::<usize>() {
            assert_eq!(tokens.len(), 3, "header line: {line}");
            if let Some(header) = cur.take() {
                blocks.push((header.0, header.1, header.2, std::mem::take(&mut buf)));
            }
            cur = Some((order, tokens[1].parse().unwrap(), tokens[2].parse().unwrap()));
        } else {
            assert_eq!(tokens.len(), 3, "point line: {line}");
            buf.push((
                tokens[0].parse().unwrap(),
                tokens[1].parse().unwrap(),
                tokens[2].parse().unwrap(),
            ));
        }
    }
    if let Some(header) = cur.take() {
        blocks.push((header.0, header.1, header.2, buf));
    }
    blocks
}

/// One expected block of the bits fixture: (order, npts, (x, y, w) bit trios).
type BitsBlock = (usize, usize, Vec<(u64, u64, u64)>);

fn parse_bits_fixture(text: &str) -> Vec<BitsBlock> {
    let mut blocks = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    let mut buf: Vec<(u64, u64, u64)> = Vec::new();
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if let Ok(order) = tokens[0].parse::<usize>() {
            assert_eq!(tokens.len(), 2, "header line: {line}");
            if let Some(header) = cur.take() {
                blocks.push((header.0, header.1, std::mem::take(&mut buf)));
            }
            cur = Some((order, tokens[1].parse().unwrap()));
        } else {
            assert_eq!(tokens.len(), 3, "bit line: {line}");
            buf.push((
                u64::from_str_radix(tokens[0], 16).unwrap(),
                u64::from_str_radix(tokens[1], 16).unwrap(),
                u64::from_str_radix(tokens[2], 16).unwrap(),
            ));
        }
    }
    if let Some(header) = cur.take() {
        blocks.push((header.0, header.1, buf));
    }
    blocks
}

/// RED-first evidence for D603: every order 26..=33 must reproduce the MFEM
/// point count and rule order.  Before the fix this fails at order 26 with
/// 455 points (s = 12, exact degree 25) where MFEM has 560 (s = 13, degree
/// 27); all even orders of this window were one GM level low.
#[test]
fn d603_tri_gm_fallback_point_counts_and_orders_match_mfem() {
    let blocks = parse_text_fixture(include_str!("data/d603_tri_gm_mfem.txt"));
    assert_eq!(blocks.len(), 8, "orders 26..=33");
    for (order, npts, rule_order, _) in &blocks {
        let rule = tri_rule(*order as u8);
        assert_eq!(rule.points.len(), *npts, "order {order}: point count");
        assert_eq!(rule.weights.len(), *npts, "order {order}: weight count");
        // GM rule of level s = order/2 is exact for degree 2s+1 = MFEM's
        // rule_order (27/29/31/33).
        assert_eq!(
            2 * (*order as usize / 2) + 1,
            *rule_order,
            "order {order}: MFEM rule order formula"
        );
    }
}

/// The moment test: the served rule integrates every monomial x^a y^b with
/// a + b <= 27 exactly (degree of the s = 13 GM rule), which the pre-fix
/// s = 12 rule (exact degree 25) cannot do for order 26.  Exact triangle
/// moments: int x^a y^b = a! b! / (a+b+2)!.
#[test]
fn d603_tri_order26_rule_exact_through_degree_27() {
    let rule = tri_rule(26);
    let mut fact = [1.0f64; 31];
    for k in 1..31 {
        fact[k] = fact[k - 1] * k as f64;
    }
    let max_deg = 27usize;
    for a in 0..=max_deg {
        for b in 0..=max_deg - a {
            let exact = fact[a] * fact[b] / fact[a + b + 2];
            // Cancellation-aware scale: the GM weights alternate sign, so the
            // honest round-off yardstick is the sum of |terms|, not `exact`.
            let scale: f64 = rule
                .points
                .iter()
                .zip(&rule.weights)
                .map(|(p, &w)| w.abs() * p[0].powi(a as i32) * p[1].powi(b as i32))
                .sum();
            let sum: f64 = rule
                .points
                .iter()
                .zip(&rule.weights)
                .map(|(p, &w)| w * p[0].powi(a as i32) * p[1].powi(b as i32))
                .sum();
            assert!(
                (sum - exact).abs() <= 1e-13 * scale,
                "monomial x^{a} y^{b}: got {sum}, exact {exact}, scale {scale}"
            );
        }
    }
    // (The s=12 pre-fix rule is already killed by the degree-26/27 moment
    // checks above — its exactness stops at 25.  A "NOT exact at degree 28"
    // probe was measured and dropped: for every a+b = 28 monomial the exact
    // GM defect is below f64 round-off, so no honest negative assertion
    // exists there; bit-level multiset equality with MFEM is the real pin.)
}

/// Every order 26..=33 reproduces the MFEM rule points bit-for-bit; the
/// weights are compared per GM level (both implementations store levels
/// contiguously, level i having C(s-i+2, 2) points) and must be within 2 ulp
/// of the MFEM fixture values.  fem-rs now evaluates MFEM's closed form
/// `w_i = ±2^-2s · m^(2s+1) / (i! · (2s+d+1-i)!)` (D603; the previous
/// moment-system solve diverged from it by up to 2.3e5 relative at s ≥ 13,
/// wrong signs included).  The residual ≤ 2 ulp slack exists because MFEM's
/// `pow(m, 2s+1)` goes through libm pow, whose last bit can differ across
/// C libraries; measured deviation on this platform is printed.
#[test]
fn d603_tri_gm_fallback_points_bits_and_weights_match_mfem() {
    let blocks = parse_text_fixture(include_str!("data/d603_tri_gm_mfem.txt"));
    let bits = parse_bits_fixture(include_str!("data/d603_tri_gm_mfem_bits.txt"));
    assert_eq!(blocks.len(), 8);
    assert_eq!(bits.len(), 8);

    // Level point counts: level i (k = s - i) has C(k+2, 2) points.
    let level_npts = |s: usize| -> Vec<usize> {
        (0..=s)
            .map(|i| {
                let k = s - i;
                (k + 1) * (k + 2) / 2
            })
            .collect()
    };

    for ((order, npts, _, _), (_, _, bpts)) in blocks.iter().zip(&bits) {
        let rule = tri_rule(*order as u8);
        assert_eq!(rule.points.len(), *npts);
        assert_eq!(bpts.len(), *npts);

        // MFEM enumerates the beta compositions of each GM level with beta[0]
        // fastest; fem-rs enumerates the same level set with the first
        // barycentric component slowest.  Same multiset, different sequence:
        // compare as sorted-by-bits multisets.
        let mut got_pts: Vec<[u64; 2]> = rule
            .points
            .iter()
            .map(|p| [p[0].to_bits(), p[1].to_bits()])
            .collect();
        got_pts.sort_unstable();
        let mut want_pts: Vec<[u64; 2]> = bpts.iter().map(|&(x, y, _)| [x, y]).collect();
        want_pts.sort_unstable();
        assert_eq!(got_pts, want_pts, "order {order}: point bit multisets");

        // Per-level weights: constant-weight runs on both sides.
        let s = *order as usize / 2;
        let counts = level_npts(s);
        assert_eq!(counts.iter().sum::<usize>(), *npts, "order {order}: level partition");

        let mut max_ulp = 0u64;
        let mut off = 0usize;
        for (level, &cnt) in counts.iter().enumerate() {
            // MFEM fixture run must be constant-weight (level structure).
            let w0 = bpts[off].2;
            for r in off..off + cnt {
                assert_eq!(bpts[r].2, w0, "order {order} level {level}: MFEM weight run");
                assert_eq!(
                    rule.weights[r].to_bits(),
                    rule.weights[off].to_bits(),
                    "order {order} level {level}: fem-rs weight run"
                );
            }
            let g = f64::from_bits(rule.weights[off].to_bits());
            let w = f64::from_bits(w0);
            assert_eq!(
                g.is_sign_positive(),
                w.is_sign_positive(),
                "order {order} level {level}: weight sign"
            );
            let d = g.abs().to_bits().abs_diff(w.abs().to_bits());
            max_ulp = max_ulp.max(d);
            off += cnt;
        }
        println!("order {order}: max per-level weight deviation vs MFEM = {max_ulp} ulp");
        assert!(
            max_ulp <= 2,
            "order {order}: GM level weights deviate {max_ulp} ulp from MFEM"
        );
    }
}

/// Odd orders 27..=33 and the whole 0..=25 segment are untouched by the fix:
/// odd orders were already correct (`s = (o-1)/2 = o/2` for odd o) and the
/// even-order change starts at 26.  Pin the odd GM orders bit-wise here too
/// (same multiset assertion as above, via the odd-order fixture blocks).
#[test]
fn d603_tri_odd_gm_orders_27_to_33_multiset_matches_mfem() {
    let blocks = parse_text_fixture(include_str!("data/d603_tri_gm_mfem.txt"));
    let bits = parse_bits_fixture(include_str!("data/d603_tri_gm_mfem_bits.txt"));
    for ((order, npts, _, _), (_, _, bpts)) in blocks.iter().zip(&bits) {
        if *order % 2 == 0 {
            continue;
        }
        let rule = tri_rule(*order as u8);
        assert_eq!(rule.points.len(), *npts, "order {order}");
        let mut got_pts: Vec<[u64; 2]> = rule
            .points
            .iter()
            .map(|p| [p[0].to_bits(), p[1].to_bits()])
            .collect();
        got_pts.sort_unstable();
        let mut want_pts: Vec<[u64; 2]> = bpts.iter().map(|&(x, y, _)| [x, y]).collect();
        want_pts.sort_unstable();
        assert_eq!(got_pts, want_pts, "order {order}: point bit multisets");
    }
}

/// The triangle factor of `prism_rule_qt` shares `tri_rule_mfem_order` and
/// benefits automatically: PRISM order 26 = 560 tri pts x 14 GL seg pts =
/// 7840 (MFEM probe header), order 28 = 680 x 15 = 10200.  The probe's first
/// three prism points per order (MFEM's kp = ks*nt + kt nesting: segment
/// slowest, triangle fastest; IntegrationPoint (x, y) = triangle, z = segment)
/// must appear in the fem-rs rule with bit-identical coordinates and weight —
/// the weight is `ipt.weight * ips.weight` (triangle factor first) in both
/// implementations.
#[test]
fn d603_prism_rule_qt_even_orders_tri_factor_benefits() {
    // (order, npts, x, y, z, w) — verbatim from the probe's PRISM section
    // (archived in tmp/d603/tri_mfem_dump.txt).
    let probe_points: [(u8, usize, [(f64, f64, f64, f64); 3]); 2] = [
        (
            26,
            7840,
            [
                (
                    0.034482758620689655,
                    0.034482758620689655,
                    0.0068580956515938291,
                    0.090353519431519702,
                ),
                (
                    0.10344827586206896,
                    0.034482758620689655,
                    0.0068580956515938291,
                    0.090353519431519702,
                ),
                (
                    0.17241379310344829,
                    0.034482758620689655,
                    0.0068580956515938291,
                    0.090353519431519702,
                ),
            ],
        ),
        (
            28,
            10200,
            [
                (
                    0.032258064516129031,
                    0.032258064516129031,
                    0.0060037409897572922,
                    0.12373380942729291,
                ),
                (
                    0.096774193548387094,
                    0.032258064516129031,
                    0.0060037409897572922,
                    0.12373380942729291,
                ),
                (
                    0.16129032258064516,
                    0.032258064516129031,
                    0.0060037409897572922,
                    0.12373380942729291,
                ),
            ],
        ),
    ];
    for (order, npts, points) in probe_points {
        let rule = prism_rule_qt(Quadrature1DType::GaussLegendre, order);
        assert_eq!(rule.points.len(), npts, "order {order}: prism point count");

        // fem-rs stores [xi_seg, tri_a, tri_b]; MFEM stores (x, y) = triangle,
        // z = segment.  Every probed MFEM point must be present bit-for-bit.
        for &(tx, ty, seg, w) in &points {
            let hit = rule
                .points
                .iter()
                .zip(&rule.weights)
                .find(|(p, _)| {
                    p[0].to_bits() == seg.to_bits()
                        && p[1].to_bits() == tx.to_bits()
                        && p[2].to_bits() == ty.to_bits()
                })
                .map(|(_, &w_got)| w_got)
                .unwrap_or_else(|| panic!("order {order}: probe point ({tx}, {ty}, {seg}) not found"));
            assert_eq!(
                hit.to_bits(),
                w.to_bits(),
                "order {order}: weight at ({tx}, {ty}, {seg})"
            );
        }
    }
}
