//! D305 — `pyramid_rule` fidelity against MFEM 4.10
//! `IntegrationRules::PyramidIntegrationRule` (`fem/intrules.cpp:2460`).
//!
//! MFEM's pyramid rule is the **cube** rule of `order` — the tensor product of
//! the 1-D Gauss-Legendre `[0,1]` rule with `n = (order|1)/2 + 1 = order/2 + 1`
//! points (`GetSegmentRealOrder`) — mapped by the Duffy transform
//! `(x,y,z) = (r(1-t), s(1-t), t)`, weight `((wx·wy)·wz)·(1-t)²`, point order
//! `iz·n² + iy·n + ix` (x fastest).  MFEM has **no point-count cap**; the
//! historical fem-rs `clamp(2, 4)` under-integrated every order above 7.
//!
//! The reference data is C++ MFEM's own output at `%.17e`:
//! * `tests/data/d339_pyramid_rule_mfem.txt` — `RULE order=K npts=M` + `N idx x
//!   y z w` (MFEM's natural point order) for `order = 0..14`, dumped by
//!   `tmp/d339/pyr_rule_probe_d305.cpp` (WSL, linked against `mfem410_ser`).
//! * `tests/data/d339_pyramid_rule_frozen_n23.txt` — the pre-D305 small-`n`
//!   rule (`n = 2, 3`) re-implemented independently in Python from the old
//!   body, used to pin the bit-frozen orders 2..5.
//!
//! Regimes (see `pyramid_rule`'s doc comment):
//! * orders 0/1 — MFEM's `npts == 1` special case, bit-identical;
//! * orders 2..5 (`n ≤ 3`) — **bit-frozen** legacy values: these are exactly
//!   the quadrature orders of the p ≤ 2 assembly paths, whose results are
//!   pinned by `d304_pyramid_h1_mass` / `d191_pyramid_h1_mfem_layout`.  MFEM's
//!   own `n = 2, 3` table entries differ from them by ≤ 1 ulp (D339 §"n ≤ 3");
//! * orders 6..14 (`n = 4..8`) — MFEM's rule **bit-identical**, including the
//!   point sequence.  `n = 4, 5` come from the `[0,1]` table, `n ≥ 6` from
//!   `gauss_legendre_01_newton_mfem`, the 1:1 port of
//!   `QuadratureFunctions1D::GaussLegendre` (`fem/intrules.cpp:620`) that
//!   replaced the generic O'Donnell iteration (D339).

use fem_element::quadrature::pyramid_rule;

const MFEM_DUMP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/d339_pyramid_rule_mfem.txt"
);
const FROZEN_DUMP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/d339_pyramid_rule_frozen_n23.txt"
);

/// One `RULE` block of a reference dump.
struct RuleRef {
    key: usize,
    npts: usize,
    rows: Vec<[f64; 4]>,
}

/// Parse a `RULE <key>=K npts=M` block dump (`key` = `order` or `n`).
fn parse_dump(path: &str, key: &str) -> Vec<RuleRef> {
    let text = std::fs::read_to_string(path).expect("read reference dump");
    let prefix = format!("{key}=");
    let mut rules = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("RULE") => {
                let mut k = None;
                let mut npts = None;
                for field in line.split_whitespace() {
                    if let Some(v) = field.strip_prefix(&prefix) {
                        k = Some(v.parse::<usize>().expect("rule key"));
                    }
                    if let Some(v) = field.strip_prefix("npts=") {
                        npts = Some(v.parse::<usize>().expect("npts"));
                    }
                }
                rules.push(RuleRef {
                    key: k.expect("RULE block without key"),
                    npts: npts.expect("RULE block without npts"),
                    rows: Vec::new(),
                });
            }
            Some("N") => {
                let vals: Vec<f64> = it
                    .skip(1) // index column
                    .take(4)
                    .map(|v| v.parse::<f64>().expect("dump number"))
                    .collect();
                assert_eq!(vals.len(), 4, "malformed dump row: {line}");
                rules.last_mut().expect("N row before RULE").rows.push([
                    vals[0], vals[1], vals[2], vals[3],
                ]);
            }
            _ => {}
        }
    }
    for r in &rules {
        assert_eq!(r.rows.len(), r.npts, "dump block {}=%d: row count", r.key);
    }
    rules
}

fn fem_rs_rows(order: u8) -> Vec<[f64; 4]> {
    let r = pyramid_rule(order);
    assert_eq!(r.points.len(), r.weights.len(), "order={order}");
    r.points
        .iter()
        .zip(r.weights.iter())
        .map(|(p, &w)| [p[0], p[1], p[2], w])
        .collect()
}

/// Relative difference between two equal-length row sequences (all components
/// of a pyramid rule are strictly positive); also returns the index of the
/// worst component.
fn max_rel_diff(a: &[[f64; 4]], b: &[[f64; 4]]) -> (f64, usize, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = (0.0_f64, 0, 0);
    for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        for j in 0..4 {
            assert!(rb[j] > 0.0, "reference component {j} of row {i} is not positive");
            let d = (ra[j] - rb[j]).abs() / rb[j];
            if d > worst.0 {
                worst = (d, i, j);
            }
        }
    }
    worst
}

/// `∫_pyramid x^a y^b z^c dV`:
/// `1/((a+1)(b+1)) · ∫₀¹ z^c (1-z)^{a+b+2} dz`
/// = `1/((a+1)(b+1)) · c!(a+b+2)!/(a+b+c+3)!`.
fn exact_monomial(a: u32, b: u32, c: u32) -> f64 {
    fn fact(n: u32) -> f64 {
        (1..=n as u64).map(|k| k as f64).product::<f64>().max(1.0)
    }
    fact(a + b + 2) * fact(c) / (fact(a + b + c + 3) * (a + 1) as f64 * (b + 1) as f64)
}

fn integrate(rows: &[[f64; 4]], a: u32, b: u32, c: u32) -> f64 {
    rows.iter()
        .map(|r| r[3] * r[0].powi(a as i32) * r[1].powi(b as i32) * r[2].powi(c as i32))
        .sum()
}

// ─── regimes ──────────────────────────────────────────────────────────────────

/// Orders 0/1: MFEM's hand-tuned single point `(3/8, 3/8, 1/4)`, weight `1/3`
/// (a one-point Duffy rule cannot integrate the `(1-t)²` factor, so MFEM picks
/// the point that makes the rule exact for `1, x, y, z`).
#[test]
fn orders_0_and_1_use_mfems_single_point_rule() {
    let mfem = parse_dump(MFEM_DUMP, "order");
    let r0 = mfem.iter().find(|r| r.key == 0).expect("order 0");
    let r1 = mfem.iter().find(|r| r.key == 1).expect("order 1");
    assert_eq!(r0.npts, 1);
    assert_eq!(r0.rows[0], [0.375, 0.375, 0.25, 1.0 / 3.0]);
    for order in [0u8, 1] {
        assert_eq!(fem_rs_rows(order), r0.rows, "order={order}");
        // Reference value sanity: MFEM's own dump for order 1 is identical.
        assert_eq!(r1.rows, r0.rows);
    }
    // The special point integrates 1, x, y, z exactly (∫x = 1/8, ∫z = 1/12).
    let rows = fem_rs_rows(0);
    assert!((integrate(&rows, 0, 0, 0) - 1.0 / 3.0).abs() < 1e-16);
    assert!((integrate(&rows, 1, 0, 0) - 1.0 / 8.0).abs() < 1e-17);
    assert!((integrate(&rows, 0, 0, 1) - 1.0 / 12.0).abs() < 1e-17);
}

/// Orders 2..5 (`n ≤ 3`) are bit-frozen at the pre-D305 values: every p ≤ 2
/// assembly result stays bit-identical (D304/D191 pins).  Odd orders reuse the
/// same rule as their even twin (`n = order/2 + 1`).
#[test]
fn orders_2_to_5_are_bit_frozen_at_the_legacy_rule() {
    let frozen = parse_dump(FROZEN_DUMP, "n");
    for (order, n) in [(2u8, 2usize), (3, 2), (4, 3), (5, 3)] {
        let want = &frozen.iter().find(|r| r.key == n).expect("frozen block").rows;
        let got = fem_rs_rows(order);
        assert_eq!(got.len(), want.len(), "order={order}");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "order={order} row {i}: {g:?} != {w:?} (bit-frozen)");
        }
    }
}

/// Orders 6..9 (`n = 4, 5`) are MFEM's rule **bit-for-bit**, including the
/// point sequence (`iz·n² + iy·n + ix`).
#[test]
fn orders_6_to_9_are_bit_identical_to_mfem() {
    let mfem = parse_dump(MFEM_DUMP, "order");
    for order in 6u8..=9 {
        let want = &mfem.iter().find(|r| r.key == order as usize).expect("mfem block");
        let got = fem_rs_rows(order);
        assert_eq!(got.len(), want.npts, "order={order} npts");
        for (i, (g, w)) in got.iter().zip(want.rows.iter()).enumerate() {
            assert_eq!(g, w, "order={order} point {i}: {g:?} != {w:?}");
        }
    }
}

/// Orders 10..14 (`n >= 6`) are MFEM's rule **bit-for-bit** after D339: the
/// 1-D nodes come from [`gauss_legendre_01_newton_mfem`], the 1:1 port of
/// `QuadratureFunctions1D::GaussLegendre` (`fem/intrules.cpp:620`), not the
/// generic O'Donnell/Newton solver that used to sit behind
/// `gauss_legendre_01_arbitrary` (which was up to 2.7e-14 off).
///
/// [`gauss_legendre_01_newton_mfem`]: fem_element::quadrature::gauss_legendre_01_newton_mfem
#[test]
fn orders_10_to_14_are_bit_identical_to_mfem() {
    let mfem = parse_dump(MFEM_DUMP, "order");
    for order in 10u8..=14 {
        let want = &mfem.iter().find(|r| r.key == order as usize).expect("mfem block");
        let got = fem_rs_rows(order);
        assert_eq!(got.len(), want.npts, "order={order} npts");
        let (d, i, j) = max_rel_diff(&got, &want.rows);
        eprintln!("D339 pyramid_rule order={order}: max rel diff = {d:.3e} (row {i} comp {j})");
        for (k, (g, w)) in got.iter().zip(want.rows.iter()).enumerate() {
            assert_eq!(g, w, "order={order} point {k}: {g:?} != {w:?}");
        }
    }
}

/// The port and the `n <= 5` tables agree bit-for-bit — the cross-check MFEM's
/// own source implies (`GaussLegendre` hard-codes `case 1/2/3` and iterates the
/// rest, while the segment rule of `n = 4, 5` is the same iteration).  This is
/// what makes it safe for `pyramid_rule` to take its `n = 4, 5` nodes from
/// [`gauss_legendre_01`] and its `n >= 6` nodes from the port.
#[test]
fn mfem_port_agrees_with_the_1d_tables_up_to_five() {
    use fem_element::quadrature::{gauss_legendre_01, gauss_legendre_01_newton_mfem};
    for n in 1usize..=5 {
        let (a, wa) = gauss_legendre_01_newton_mfem(n);
        let (b, wb) = gauss_legendre_01(n);
        assert_eq!(a, b, "n={n} nodes");
        assert_eq!(wa, wb, "n={n} weights");
    }
}

/// Point counts for every order 0..14 are MFEM's (`1, 1, 8, 8, 27, 27, 64,
/// 64, 125, 125, 216, 216, 343, 343, 512`) — the D305 cap is gone.
#[test]
fn point_counts_match_mfem_for_all_orders() {
    let mfem = parse_dump(MFEM_DUMP, "order");
    for order in 0u8..=14 {
        let want = &mfem.iter().find(|r| r.key == order as usize).expect("mfem block");
        let r = pyramid_rule(order);
        assert_eq!(r.points.len(), want.npts, "order={order}");
        let n = order as usize / 2 + 1;
        assert_eq!(r.points.len(), n * n * n, "order={order}: n³");
        let wsum: f64 = r.weights.iter().sum();
        assert!((wsum - 1.0 / 3.0).abs() < 1e-14, "order={order}: Σw = {wsum}");
        for w in &r.weights {
            assert!(*w > 0.0, "order={order}: non-positive weight");
        }
    }
}

/// Exactness: the rule integrates every monomial with `a, b ≤ 2n−1` and
/// `a + b + c ≤ 2n−3` exactly (the Duffy-mapped integrand
/// `(1-t)²·x^a y^b z^c` is a tensor polynomial of degree `2n−1` in each
/// variable).  Orders 0/1 are exempt: MFEM's single-point rule is only exact
/// for `1, x, y, z` (tested in `orders_0_and_1_use_mfems_single_point_rule`).
#[test]
fn rules_integrate_monomials_exactly_to_the_rule_degree() {
    for order in 2u8..=14 {
        let n = order as usize / 2 + 1;
        let rows = fem_rs_rows(order);
        let deg1d = 2 * n - 1; // exact in r / s / t per variable
        let mut worst = (0.0_f64, 0u32, 0u32, 0u32);
        for a in 0..=deg1d {
            for b in 0..=deg1d {
                if a + b > 2 * n - 3 {
                    continue; // no `c ≥ 0` satisfies `a + b + c ≤ 2n−3`
                }
                let cmax = (2 * n - 3 - a - b).min(12);
                for c in 0..=cmax {
                    let exact = exact_monomial(a as u32, b as u32, c as u32);
                    let got = integrate(&rows, a as u32, b as u32, c as u32);
                    let rel = (got - exact).abs() / exact.abs();
                    if rel > worst.0 {
                        worst = (rel, a as u32, b as u32, c as u32);
                    }
                }
            }
        }
        eprintln!(
            "D305 exactness order={order} (n={n}): worst rel = {:.3e} at x^{} y^{} z^{}",
            worst.0, worst.1, worst.2, worst.3
        );
        assert!(
            worst.0 < 1e-12,
            "order={order}: monomial x^{} y^{} z^{} off by {:.3e}",
            worst.1,
            worst.2,
            worst.3,
            worst.0
        );
    }
    // The uncapped orders would have failed before D305: order 10 (n = 6) is
    // exact for `a + b + c ≤ 9`, while the old cap gave n = 4 (exact only for
    // `a + b + c ≤ 5`) — `x⁵z⁴` is exact now, grossly wrong before.
    let rows = fem_rs_rows(10);
    let exact = exact_monomial(5, 0, 4);
    let got = integrate(&rows, 5, 0, 4);
    assert!((got - exact).abs() / exact < 1e-13, "order 10 x^5 z^4");
}

// ─── evidence generator ───────────────────────────────────────────────────────

/// Emit the sorted `(x, y, z, w)` pyramid-rule dump plus the 1-D segment rules
/// in the C++ probe format when `D339_DUMP_DIR` is set.
#[test]
fn dump_rules_for_mfem_diff() {
    let dir = match std::env::var("D339_DUMP_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => return, // silent no-op in normal test runs
    };
    let mut out = String::new();
    out.push_str("# fem-rs pyramid_rule, sorted by (x,y,z)\n");
    for order in 0u8..=14 {
        let r = pyramid_rule(order);
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
                "P {:.17e} {:.17e} {:.17e} {:.17e}\n",
                p[0], p[1], p[2], p[3]
            ));
        }
    }
    std::fs::write(
        std::path::Path::new(&dir).join("pyramid_rule_rust_d339.txt"),
        out,
    )
    .expect("write pyramid dump");

    let mut out = String::new();
    out.push_str("# fem-rs pyramid_rule natural point order (orders 0..14)\n");
    for order in 0u8..=14 {
        let r = pyramid_rule(order);
        let wsum: f64 = r.weights.iter().sum();
        out.push_str(&format!(
            "RULE order={order} npts={} wsum={wsum:.17e}\n",
            r.points.len()
        ));
        for (k, (p, &w)) in r.points.iter().zip(r.weights.iter()).enumerate() {
            out.push_str(&format!(
                "N {k} {:.17e} {:.17e} {:.17e} {:.17e}\n",
                p[0], p[1], p[2], w
            ));
        }
    }
    std::fs::write(
        std::path::Path::new(&dir).join("pyramid_rule_rust_natural_d339.txt"),
        out,
    )
    .expect("write natural-order dump");

    let mut out = String::new();
    out.push_str("# fem-rs gauss_legendre_01, n = 1..11\n");
    for n in 1usize..=11 {
        let (xs, ws) = fem_element::quadrature::gauss_legendre_01(n);
        let wsum: f64 = ws.iter().sum();
        out.push_str(&format!("RULE n={n} npts={} wsum={wsum:.17e}\n", xs.len()));
        for (x, w) in xs.iter().zip(ws.iter()) {
            out.push_str(&format!("P {x:.17e} {w:.17e}\n"));
        }
    }
    std::fs::write(std::path::Path::new(&dir).join("seg_rule_rust_d339.txt"), out)
        .expect("write seg dump");
}
