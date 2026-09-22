//! D617: the Grundmann-Möller fallback of [`tet_rule`] (tetrahedron orders
//! > 20, MFEM `IntegrationRules::TetrahedronIntegrationRule`'s `default:`
//! branch) must reproduce MFEM 4.10 **bit for bit**:
//!
//! ```cpp
//! int i = (Order / 2) * 2 + 1;   // closest odd >= Order
//! ir->GrundmannMollerSimplexRule(i/2, 3);
//! ```
//!
//! i.e. GM level `s = order / 2` (integer division) for BOTH parities, exact
//! degree `i = 2s+1` (21/23/25/27 for orders 21..=27).
//!
//! Status note (round 61): the **weights** were already fixed to MFEM's
//! closed form by the D603 work (round 60, `grundmann_moller_simplex`'s
//! `pow(2,-2s)·m^(2s+1)/i!/(2s+d+1-i)!`); this fixture pins the tetrahedron
//! segment, which the tri-only D603 test did not cover.  The red state is
//! documented without a code change: the PRE-D603 weight algorithm (monomial
//! moment system + Gaussian elimination) collapses on the tet range too —
//! replicated offline against this fixture it deviates from MFEM by
//! 1.0e18..2.5e24 relative at s = 10..13 with garbage magnitudes
//! (`tmp/d614/red_old_gm_tet.log`, script `tmp/d614/red_old_gm_tet.py`).
//!
//! Truth provenance: `tmp/d614/tet_gm_probe.cpp`, compiled in WSL against the
//! MFEM 4.10 tree (`g++ -std=c++17 -O2 -I$HOME/mfem410_ser tet_gm_probe.cpp
//! $HOME/mfem410_ser/libmfem.a -o tet_gm_probe`), dumping
//! `IntRules.Get(TETRAHEDRON, 21..=27)`:
//! * `data/d617_tet_gm_mfem.txt` — "TET order npts rule_order" header +
//!   "x y z weight" lines, `%.17g` (exact f64 round-trip),
//! * `data/d617_tet_gm_mfem_bits.txt` — the same values as raw f64 bit
//!   patterns (`%016llx`), pinning points and weights at the bit level.
//!
//! MFEM ground truth (probe headers, `tmp/d614/tet_gm_probe_headers.log`):
//! order 21 -> 1001 pts (rule order 21); 22/23 -> 1365/23; 24/25 -> 1820/25;
//! 26/27 -> 2380/27.  Point counts = C(s+4, 4) (GM tet level sums).

use fem_element::quadrature::tet_rule;

/// One expected block of the text fixture: (order, npts, rule_order, points).
type Block = (usize, usize, usize, Vec<(f64, f64, f64, f64)>);

fn parse_text_fixture(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut cur: Option<(usize, usize, usize)> = None;
    let mut buf: Vec<(f64, f64, f64, f64)> = Vec::new();
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if let Ok(order) = tokens[0].parse::<usize>() {
            assert_eq!(tokens.len(), 3, "header line: {line}");
            if let Some(header) = cur.take() {
                blocks.push((header.0, header.1, header.2, std::mem::take(&mut buf)));
            }
            cur = Some((order, tokens[1].parse().unwrap(), tokens[2].parse().unwrap()));
        } else {
            assert_eq!(tokens.len(), 4, "point line: {line}");
            buf.push((
                tokens[0].parse().unwrap(),
                tokens[1].parse().unwrap(),
                tokens[2].parse().unwrap(),
                tokens[3].parse().unwrap(),
            ));
        }
    }
    if let Some(header) = cur.take() {
        blocks.push((header.0, header.1, header.2, buf));
    }
    blocks
}

/// One expected block of the bits fixture: (order, npts, (x, y, z, w) bits).
type BitsBlock = (usize, usize, Vec<(u64, u64, u64, u64)>);

fn parse_bits_fixture(text: &str) -> Vec<BitsBlock> {
    let mut blocks = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    let mut buf: Vec<(u64, u64, u64, u64)> = Vec::new();
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if let Ok(order) = tokens[0].parse::<usize>() {
            assert_eq!(tokens.len(), 2, "header line: {line}");
            if let Some(header) = cur.take() {
                blocks.push((header.0, header.1, std::mem::take(&mut buf)));
            }
            cur = Some((order, tokens[1].parse().unwrap()));
        } else {
            assert_eq!(tokens.len(), 4, "bit line: {line}");
            buf.push((
                u64::from_str_radix(tokens[0], 16).unwrap(),
                u64::from_str_radix(tokens[1], 16).unwrap(),
                u64::from_str_radix(tokens[2], 16).unwrap(),
                u64::from_str_radix(tokens[3], 16).unwrap(),
            ));
        }
    }
    if let Some(header) = cur.take() {
        blocks.push((header.0, header.1, buf));
    }
    blocks
}

/// RED-first evidence for D617 (kept as a live pin): every order 21..=27 must
/// reproduce the MFEM point count and rule order.
#[test]
fn d617_tet_gm_fallback_point_counts_and_orders_match_mfem() {
    let blocks = parse_text_fixture(include_str!("data/d617_tet_gm_mfem.txt"));
    assert_eq!(blocks.len(), 7, "orders 21..=27");
    for (order, npts, rule_order, _) in &blocks {
        let rule = tet_rule(*order as u8);
        assert_eq!(rule.points.len(), *npts, "order {order}: point count");
        assert_eq!(rule.weights.len(), *npts, "order {order}: weight count");
        // GM rule of level s = order/2 is exact for degree 2s+1 = MFEM's
        // rule_order (21/23/25/27).
        assert_eq!(
            2 * (*order as usize / 2) + 1,
            *rule_order,
            "order {order}: MFEM rule order formula"
        );
    }
}

/// Moment test on the served order-21 rule (s = 10, exact degree 21): it must
/// integrate every monomial x^a y^b z^c with a + b + c <= 21 exactly.
/// Exact tetrahedron moments: int x^a y^b z^c = a! b! c! / (a+b+c+3)!.
/// The PRE-D603 weight algorithm fails this window outright (weights ~1e11
/// instead of ~0.43 — see the module header); the closed-form weights pass.
#[test]
fn d617_tet_order21_rule_exact_through_degree_21() {
    let rule = tet_rule(21);
    let mut fact = [1.0f64; 31];
    for k in 1..31 {
        fact[k] = fact[k - 1] * k as f64;
    }
    let max_deg = 21usize;
    for a in 0..=max_deg {
        for b in 0..=max_deg - a {
            for c in 0..=max_deg - a - b {
                let exact = fact[a] * fact[b] * fact[c] / fact[a + b + c + 3];
                // Cancellation-aware scale: the GM weights alternate sign, so
                // the honest round-off yardstick is the sum of |terms|.
                let scale: f64 = rule
                    .points
                    .iter()
                    .zip(&rule.weights)
                    .map(|(p, &w)| {
                        w.abs() * p[0].powi(a as i32) * p[1].powi(b as i32) * p[2].powi(c as i32)
                    })
                    .sum();
                let sum: f64 = rule
                    .points
                    .iter()
                    .zip(&rule.weights)
                    .map(|(p, &w)| {
                        w * p[0].powi(a as i32) * p[1].powi(b as i32) * p[2].powi(c as i32)
                    })
                    .sum();
                assert!(
                    (sum - exact).abs() <= 1e-12 * scale,
                    "monomial x^{a} y^{b} z^{c}: got {sum}, exact {exact}, scale {scale}"
                );
            }
        }
    }
}

/// Every order 21..=27 reproduces the MFEM rule points bit-for-bit (sorted-bit
/// multiset — MFEM enumerates beta compositions beta[0]-fastest, fem-rs the
/// same levels with the first barycentric component slowest); the weights are
/// compared per GM level (constant-weight runs, level i of s having
/// C(s-i+3, 3) points) and must be within 2 ulp of the MFEM fixture values
/// (MFEM's `pow(m, 2s+1)` goes through libm pow, whose last bit can differ
/// across C libraries; measured deviation on this platform is printed).
#[test]
fn d617_tet_gm_fallback_points_bits_and_weights_match_mfem() {
    let blocks = parse_text_fixture(include_str!("data/d617_tet_gm_mfem.txt"));
    let bits = parse_bits_fixture(include_str!("data/d617_tet_gm_mfem_bits.txt"));
    assert_eq!(blocks.len(), 7);
    assert_eq!(bits.len(), 7);

    // Level point counts: level i (k = s - i) has C(k+3, 3) points.
    let level_npts = |s: usize| -> Vec<usize> {
        (0..=s)
            .map(|i| {
                let k = s - i;
                (k + 1) * (k + 2) * (k + 3) / 6
            })
            .collect()
    };

    let mut global_max_ulp = 0u64;
    for ((order, npts, _, _), (_, _, bpts)) in blocks.iter().zip(&bits) {
        let rule = tet_rule(*order as u8);
        assert_eq!(rule.points.len(), *npts);
        assert_eq!(bpts.len(), *npts);

        // Same multiset, different sequence: compare as sorted-by-bits multisets.
        let mut got_pts: Vec<[u64; 3]> = rule
            .points
            .iter()
            .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
            .collect();
        got_pts.sort_unstable();
        let mut want_pts: Vec<[u64; 3]> = bpts.iter().map(|&(x, y, z, _)| [x, y, z]).collect();
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
            let w0 = bpts[off].3;
            for r in off..off + cnt {
                assert_eq!(bpts[r].3, w0, "order {order} level {level}: MFEM weight run");
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
        global_max_ulp = global_max_ulp.max(max_ulp);
    }
    println!("d617: global max per-level weight deviation = {global_max_ulp} ulp over orders 21..=27");
}
