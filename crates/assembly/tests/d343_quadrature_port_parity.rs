//! D339 cross-crate parity: the two copies of MFEM's
//! `QuadratureFunctions1D::GaussLegendre(np, ir)` port must stay identical.
//!
//! The port exists twice because `fem-element` cannot depend on `fem-assembly`:
//!
//! * `fem_element::quadrature::gauss_legendre_01_newton_mfem` — added in D339 so
//!   `pyramid_rule`, `seg_rule` and the tensor rules can use MFEM's 1-D nodes;
//! * `fem_assembly::postproc::plbound::mfem_gauss_legendre_01` — the D259 copy
//!   used by the NURBS/bounds machinery (its own bit-exactness test is
//!   `plbound::tests::mfem_gauss_legendre_matches_cpp_dump`).
//!
//! This test pins them equal for `np = 1..=20` so a future edit to either copy
//! cannot silently drift.  (The identical situation for Gauss-Lobatto is
//! documented in `fem_element::quadrature::gauss_lobatto_01_newton_mfem`.)

use fem_assembly::postproc::plbound::mfem_gauss_legendre_01;
use fem_element::quadrature::gauss_legendre_01_newton_mfem;

#[test]
fn the_two_mfem_gauss_legendre_ports_are_identical() {
    for np in 1usize..=20 {
        let (xa, wa) = gauss_legendre_01_newton_mfem(np);
        let (xb, wb) = mfem_gauss_legendre_01(np);
        assert_eq!(xa, xb, "np={np}: nodes differ between the two ports");
        assert_eq!(wa, wb, "np={np}: weights differ between the two ports");
        // Sanity: the rule stays on [0,1], ascending, with weights summing to 1.
        let wsum: f64 = wa.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-14, "np={np}: sum of weights = {wsum}");
        for k in 1..np {
            assert!(xa[k - 1] < xa[k], "np={np}: nodes not ascending at {k}");
        }
    }
}

/// The D339 fix itself: `pyramid_rule`'s `n >= 6` nodes are now MFEM's, and the
/// 5-point case still comes from the `[0,1]` table (unchanged for every
/// consumer that relies on it, e.g. `ex41`'s BlockILU MDF tie-breaks).
///
/// The oracle is MFEM 4.10's own `%.17g` output for
/// `QuadratureFunctions1D::GaussLegendre(n)` on `[0,1]`
/// (`tmp/d343/probe_pyr_d339.cpp` -> `$HOME/work/d343/pyr_d339.txt`, the `GL`
/// section); the same file's `PYR` section reproduces the committed
/// `crates/element/tests/data/d339_pyramid_rule_mfem.txt` bit for bit.
#[test]
fn pyramid_rule_high_orders_use_the_mfem_1d_rule() {
    use fem_element::quadrature::{gauss_legendre_01, pyramid_rule};

    // MFEM 4.10 `GaussLegendre(6)`, %.17g, from the probe dump.
    let mfem_n6_x = [
        0.033765242898423996,
        0.16939530676686773,
        0.38069040695840156,
        0.61930959304159838,
        0.83060469323313224,
        0.96623475710157603,
    ];
    let mfem_n6_w = [
        0.085662246189585234,
        0.1803807865240693,
        0.23395696728634552,
        0.23395696728634552,
        0.1803807865240693,
        0.085662246189585234,
    ];
    let (xs, ws) = gauss_legendre_01_newton_mfem(6);
    assert_eq!(xs, mfem_n6_x.to_vec(), "n = 6 nodes vs MFEM 4.10");
    assert_eq!(ws, mfem_n6_w.to_vec(), "n = 6 weights vs MFEM 4.10");
    // `gauss_legendre_01` routes `n > 5` through the same port.
    assert_eq!(gauss_legendre_01(6), mfem_gauss_legendre_01(6));

    // order 10 -> n = 6: the Duffy map of the port's 6-point rule.
    let rule = pyramid_rule(10);
    assert_eq!(rule.points.len(), 216);
    let mut k = 0;
    for (zk, wz) in xs.iter().zip(ws.iter()) {
        let t = *zk;
        let omt = 1.0 - t;
        for (yj, wy) in xs.iter().zip(ws.iter()) {
            for (xi, wx) in xs.iter().zip(ws.iter()) {
                assert_eq!(rule.points[k], vec![xi * omt, yj * omt, t], "order 10 point {k}");
                assert_eq!(rule.weights[k], wx * wy * wz * (omt * omt), "order 10 weight {k}");
                k += 1;
            }
        }
    }
    // n = 5 (orders 8, 9) still comes from the `[0,1]` table, which the port
    // reproduces bit-for-bit (that is why the two paths are interchangeable).
    let (t5, w5) = gauss_legendre_01(5);
    assert_eq!(t5, gauss_legendre_01_newton_mfem(5).0, "n=5 must agree with the table");
    assert_eq!(w5, gauss_legendre_01_newton_mfem(5).1, "n=5 must agree with the table");
}
