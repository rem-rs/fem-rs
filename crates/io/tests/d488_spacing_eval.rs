//! D488a: `NurbsSpacingRecord::eval` — MFEM `SpacingFunction::EvalAll` for the
//! eight `SpacingType`s of the v1.1 `spacing` section (`mesh/spacing.cpp`).
//!
//! The reference is an MFEM 4.10 probe (`tmp/d487/spacing_probe.cpp`, output
//! `tmp/d487/spacing_truth.txt`): each instance is built through the same
//! entry point `NURBSExtension::Load` uses when reading a `spacing` section
//! (`GetSpacingFunction(type, ipar, dpar)`) and `EvalAll` is dumped with 17
//! significant digits.  The instances exercise every type's iterative path
//! (GEOMETRIC/GAUSSIAN Newton, BELL's interior-spacing scheme, the
//! PIECEWISE/PARTIAL composites) and the `reverse` flag.

use fem_io::nurbs_mesh::NurbsSpacingRecord;

fn record(spacing_type: i32, int_params: &[i32], real_params: &[f64]) -> NurbsSpacingRecord {
    NurbsSpacingRecord {
        knotvector: 0,
        spacing_type,
        int_params: int_params.to_vec(),
        real_params: real_params.to_vec(),
    }
}

/// `0 UNIFORM: n=8`.
const UNIFORM: [f64; 8] = [0.125; 8];

/// `1 LINEAR: n=10, reverse=1, scale=1, s=0.02`.
const LINEAR: [f64; 10] = [
    0.17999999999999999,
    0.16222222222222221,
    0.14444444444444443,
    0.12666666666666665,
    0.1088888888888889,
    0.091111111111111115,
    0.073333333333333334,
    0.055555555555555552,
    0.037777777777777778,
    0.02,
];

/// `2 GEOMETRIC: n=12, reverse=0, scale=1, s=0.01` (growing, r > 1).
const GEOMETRIC: [f64; 12] = [
    0.01,
    0.01347096914861033,
    0.018146700980281132,
    0.024445364905442399,
    0.032930275646773623,
    0.044360272729292154,
    0.059757586536023477,
    0.080499260462218419,
    0.10844030541724918,
    0.14607960087416452,
    0.19678337966171811,
    0.26508628363822778,
];

/// `2 GEOMETRIC: n=9, reverse=0, scale=1, s=0.2` (shrinking, r < 1).
const GEOMETRIC_SHRINK: [f64; 9] = [
    0.20000000000000001,
    0.16860035554015224,
    0.14213039944132871,
    0.11981617939435936,
    0.10100525222675825,
    0.08514760718427096,
    0.07177958422330652,
    0.060510317102868909,
    0.051010304886955252,
];

/// `3 BELL: n=16, reverse=0, scale=1, s0=0.2, s1=0.1`.
const BELL: [f64; 16] = [
    0.20000000000000001,
    0.11266207931784405,
    0.075803912093865611,
    0.056677703721537742,
    0.045659208611579627,
    0.038991875135518828,
    0.034968557061973216,
    0.032752990350576106,
    0.031944779698984904,
    0.032408293948829825,
    0.034220641400240726,
    0.037700865019870911,
    0.043541211752507691,
    0.053145924237506637,
    0.069521957649164356,
    0.099999999999999978,
];

/// `3 BELL: n=11, reverse=1, scale=1, s0=0.25, s1=0.05`.
const BELL_REV: [f64; 11] = [
    0.050000000000000044,
    0.048967449822681974,
    0.049472126019369567,
    0.051579535303711399,
    0.055574919126665789,
    0.062051356231083377,
    0.072118216258399537,
    0.087873932969124158,
    0.11356026889464077,
    0.15880219537432316,
    0.25,
];

/// `4 GAUSSIAN: n=16, reverse=0, scale=1, s0=0.05, s1=0.15`.
const GAUSSIAN: [f64; 16] = [
    0.049999999999999996,
    0.04545818469368338,
    0.042335672108873408,
    0.040388071326822109,
    0.039468630365601234,
    0.039509659119207854,
    0.040514155919719151,
    0.042556175693443568,
    0.045790003514134224,
    0.050469737521601304,
    0.056982788887669694,
    0.065903524533829957,
    0.078077494318435872,
    0.094753525775457828,
    0.11779237442159433,
    0.15000000000000002,
];

/// `5 LOGARITHMIC: n=14, reverse=0, sym=1, base=10`.
const LOG_SYM: [f64; 14] = [
    0.021638638576285427,
    0.030066790806117353,
    0.041777670355359751,
    0.058049884724178574,
    0.080660053273126114,
    0.11207678059890608,
    0.1557301816660267,
    0.1557301816660267,
    0.11207678059890608,
    0.080660053273126114,
    0.058049884724178574,
    0.041777670355359751,
    0.030066790806117353,
    0.021638638576285427,
];

/// `5 LOGARITHMIC: n=13, reverse=1, sym=0, base=2`.
const LOG_NONSYM_REV: [f64; 13] = [
    0.10384497132165693,
    0.09845308228725913,
    0.093341153533934795,
    0.088494648828003619,
    0.083899786693171663,
    0.079543501221648771,
    0.07541340492005566,
    0.071497753484459858,
    0.067785412404381518,
    0.064265825300801716,
    0.0609289839081395,
    0.057765399614840174,
    0.054766076481646664,
];

/// `6 PIECEWISE: n=12, np=2, reverse=0, relN=[2,1], partition=[0.4],
/// pieces = [UNIFORM, LINEAR(s=0.3)]`.
const PIECEWISE: [f64; 12] = [
    0.050000000000000003,
    0.050000000000000003,
    0.050000000000000003,
    0.050000000000000003,
    0.050000000000000003,
    0.050000000000000003,
    0.050000000000000003,
    0.050000000000000003,
    0.17999999999999999,
    0.16,
    0.13999999999999999,
    0.12,
];

/// `7 PARTIAL: n=6, reverse=0, first=2, num=3, num_full=8, full = GEOMETRIC
/// (n=8, s=0.01, scale=1)` — window [4..10) of the refined full table.
const PARTIAL: [f64; 6] = [
    0.097042589792698614,
    0.11788015913184599,
    0.14319209685802134,
    0.17393916629908673,
    0.21128843167105651,
    0.25665755624729075,
];

/// All ten probe instances with their record parameters.
fn instances() -> Vec<(&'static str, NurbsSpacingRecord, &'static [f64])> {
    vec![
        ("uniform", record(0, &[8], &[]), &UNIFORM),
        ("linear", record(1, &[10, 1, 1], &[0.02]), &LINEAR),
        ("geometric", record(2, &[12, 0, 1], &[0.01]), &GEOMETRIC),
        (
            "geometric-shrink",
            record(2, &[9, 0, 1], &[0.2]),
            &GEOMETRIC_SHRINK,
        ),
        ("bell", record(3, &[16, 0, 1], &[0.2, 0.1]), &BELL),
        ("bell-rev", record(3, &[11, 1, 1], &[0.25, 0.05]), &BELL_REV),
        ("gaussian", record(4, &[16, 0, 1], &[0.05, 0.15]), &GAUSSIAN),
        ("log-sym", record(5, &[14, 0, 1], &[10.0]), &LOG_SYM),
        (
            "log-nonsym-rev",
            record(5, &[13, 1, 0], &[2.0]),
            &LOG_NONSYM_REV,
        ),
        (
            "piecewise",
            // ipar = [n, np, reverse, relN…, {type, nip, nrp, ipar…} per piece]
            record(
                6,
                &[12, 2, 0, 2, 1, 0, 1, 0, 1, 1, 3, 1, 1, 0, 0],
                &[0.4, 0.3],
            ),
            &PIECEWISE,
        ),
        (
            "partial",
            // ipar = [n, reverse, first, num, num_full, type_full, nip, nrp,
            //         full ipar…]
            record(7, &[6, 0, 2, 3, 8, 2, 3, 1, 8, 0, 1], &[0.01]),
            &PARTIAL,
        ),
    ]
}

/// Every instance matches the MFEM 4.10 probe at 1e-15 relative tolerance.
#[test]
fn eval_matches_mfem_on_all_eight_types() {
    let mut max_rel = 0.0_f64;
    for (name, rec, want) in instances() {
        let got = rec.eval().unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.len(), want.len(), "{name}: width count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let rel = (*g - w).abs() / w.abs();
            assert!(rel < 1e-15, "{name}[{i}]: got {g:e}, want {w:e} (rel {rel:e})");
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("d488 spacing eval: max relative deviation {max_rel:e}");
}

/// Widths sum to 1 by design — except GAUSSIAN, whose fit leaves the sum
/// short by its Newton residual tolerance (probe: 1 - sum = 1.7999e-9), so
/// the bound is 1e-8 there.
#[test]
fn widths_sum_to_one() {
    for (name, rec, _) in instances() {
        let sum: f64 = rec.eval().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-8, "{name}: widths sum {sum:e}");
    }
}

/// `MFEM_VERIFY` failures surface as errors (the base-class parameter checks).
#[test]
fn invalid_parameters_are_rejected() {
    // LINEAR with s outside (0,1).
    assert!(record(1, &[10, 0, 0], &[1.5]).eval().is_err());
    // BELL with s0 + s1 >= 1.
    assert!(record(3, &[8, 0, 0], &[0.6, 0.6]).eval().is_err());
    // PARTIAL with n not a multiple of num_elems.
    assert!(record(7, &[7, 0, 2, 3, 8, 2, 3, 1, 8, 0, 1], &[0.01]).eval().is_err());
    // Unknown spacing type.
    assert!(record(9, &[8], &[]).eval().is_err());
}
