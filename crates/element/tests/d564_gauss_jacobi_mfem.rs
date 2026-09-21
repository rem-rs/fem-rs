//! D564/D565: pin `gauss_jacobi` (now a 1:1 port of MFEM 4.10
//! `QuadratureFunctions1D::GaussJacobi`, `fem/intrules.cpp:488`) against the
//! MFEM 4.10 probe truth.
//!
//! Truth provenance: `tmp/d550/probe_gj.cpp` compiled against the MFEM 4.10
//! serial tree (`-I/home/quan/mfem410 -L/home/quan/mfem410_ser -lmfem`, WSL),
//! dumping `QuadratureFunctions1D::GaussJacobi(n, alpha, beta)` converted to
//! the `[-1,1]` convention (`xi = 2t - 1`, `w_xi = w_t * 2^(alpha+beta+1)`),
//! `%.17g` per node.  Full 240-combination grid (`(alpha,beta)` in
//! {(0,0),(1,0),(0,1),(2,0),(0,2),(1,1),(3,1),(4,4),(0.5,-0.5),(-0.5,0.5)}
//! x n = 1..24): **0 panics** (the pre-D564 Golub-Welsch route panicked on
//! 161/240), 1674/3000 rows bit-identical, worst |dx| = 2.22e-16, worst
//! |dw| = 8.33e-15 (2.9e-14 relative) — cross-platform (glibc libm truth vs
//! msvcrt run), so the tolerances below carry margin.
//!
//! The fixture holds the complete n = 1..12 truth for the two representative
//! families (1,0) (the Stroud-consumed pair, bit-identical here) and
//! (0.5,-0.5) (the face the old implementation got outright wrong:
//! n = 2 collapsed to nodes {0,0} with weights {pi, 0}).

use fem_element::quadrature::gauss_jacobi;

const MFEM_TRUTH: &str = include_str!("data/d564_gj_mfem410_subset.txt");

#[test]
fn d564_matches_mfem_410_truth() {
    let mut rows = 0;
    for line in MFEM_TRUTH.lines() {
        let p: Vec<f64> = line
            .split_whitespace()
            .map(|t| t.parse::<f64>().unwrap())
            .collect();
        let (a, b, n, i) = (p[0], p[1], p[2] as usize, p[3] as usize);
        let (x_truth, w_truth) = (p[4], p[5]);
        let (xs, ws) = gauss_jacobi(n, a, b);
        assert_eq!(xs.len(), n, "row {line}");
        let dx = (xs[i] - x_truth).abs();
        let dw = (ws[i] - w_truth).abs();
        assert!(dx <= 3e-16, "node deviates: {line} -> dx={dx:e}");
        assert!(dw <= 1e-13, "weight deviates: {line} -> dw={dw:e}");
        rows += 1;
    }
    assert_eq!(rows, 156);
}

#[test]
fn d564_stroud_pair_is_bit_identical() {
    // (1,0) measured bit-identical to the glibc MFEM run for all n = 1..24
    // (MFEM 4.10 uses (0,0)/(1,0)/(2,0) inside the Stroud tri/tet rules).
    for n in 1..=24usize {
        let (xs, ws) = gauss_jacobi(n, 1.0, 0.0);
        for line in MFEM_TRUTH.lines() {
            let p: Vec<f64> = line
                .split_whitespace()
                .map(|t| t.parse::<f64>().unwrap())
                .collect();
            if p[0] == 1.0 && p[1] == 0.0 && p[2] as usize == n {
                let i = p[3] as usize;
                assert_eq!(xs[i], p[4], "n={n} i={i}");
                assert_eq!(ws[i], p[5], "n={n} i={i}");
            }
        }
    }
}

#[test]
fn d564_weights_sum_to_mu0() {
    // mu0 = 2^(a+b+1) * B(a+1, b+1), computed with independent ln_gamma here.
    fn ln_gamma(x: f64) -> f64 {
        let p = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let g = 7.0;
        let x = x - 1.0;
        let mut a = p[0];
        let t = x + g + 0.5;
        for i in 1..p.len() {
            a += p[i] / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (t).ln() * (x + 0.5) - t + a.ln()
    }
    let cases = [(0.0, 0.0), (1.0, 0.0), (3.0, 1.0), (4.0, 4.0), (0.5, -0.5), (-0.5, 0.5)];
    for (a, b) in cases {
        let mu0 = 2.0_f64.powf(a + b + 1.0)
            * ((ln_gamma(a + 1.0) + ln_gamma(b + 1.0) - ln_gamma(a + b + 2.0)).exp());
        for n in 1..=24usize {
            let (_, ws) = gauss_jacobi(n, a, b);
            let sum: f64 = ws.iter().sum();
            assert!(
                (sum - mu0).abs() <= 1e-12 * mu0.abs(),
                "({a},{b}) n={n}: sum={sum:e} mu0={mu0:e}"
            );
        }
    }
}

#[test]
fn d564_half_power_face_has_distinct_nodes() {
    // The pre-D564 defect: (0.5,-0.5) returned collapsed nodes {0,0} with
    // weights {pi, 0} at n = 2.
    let (xs, ws) = gauss_jacobi(2, 0.5, -0.5);
    assert!((xs[0] + 0.80901699437494745).abs() < 1e-15, "xs={xs:?}");
    assert!((xs[1] - 0.30901699437494745).abs() < 1e-15, "xs={xs:?}");
    assert!((ws[0] - 2.2732777998989708).abs() < 1e-13, "ws={ws:?}");
    assert!((ws[1] - 0.86831485369082395).abs() < 1e-13, "ws={ws:?}");
}

#[test]
#[should_panic(expected = "only defined for alpha > -1")]
fn d564_panics_below_minus_one() {
    gauss_jacobi(4, -1.0, 0.0);
}

#[test]
#[should_panic(expected = "only tested for alpha <= 4")]
fn d564_panics_above_four() {
    gauss_jacobi(4, 0.0, 5.0);
}
