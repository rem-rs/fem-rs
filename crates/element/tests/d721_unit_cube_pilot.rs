//! D721 pilot — `[0,1]³` reference-domain re-baseline, unit-hex bit-level
//! evidence. **Test-only**: no `src` changes; every `[0,1]³` quantity needed
//! for the pilot is re-derived inside this file, so the experiment can be
//! adjudicated (and deleted) without touching the lanes.
//!
//! # What is proven here
//!
//! MFEM truth (4.10, WSL serial build) was dumped bit-for-bit by
//! `tmp/d721/probe_d721.cpp` on the affine unit hex
//! (`data/d680_unit_hex_mfem.mesh`, vertices in {0,1}³):
//!
//! 1. The 1-D `GaussLegendre` table on `[0,1]` (3-pt):
//!    points `0.11270166537925831 / 0.5 / 0.8872983346207417`
//!    (`3fbcda042f0236e1 / 3fe0000000000000 / 3fec64bf7a1fb924`),
//!    weights `0.27777777777777779 / 0.44444444444444442 / …`
//!    (`3fd1c71c71c71c72 / 3fdc71c71c71c71c`).  fem-rs'
//!    `gauss_legendre_01(3)` hard-codes the same values — **the `[0,1]`
//!    quadrature layer is already MFEM-bitwise**; only `hex_rule` still wires
//!    the `[-1,1]` table into hexes.
//! 2. The 27-pt CUBE order-4 rule weights are the products `(wx·wy)·wz`
//!    in the x-fastest point order; the products carry a visible 1-ulp split
//!    (`0.05486968449931412…7` / `…8` depending on the weight placement) —
//!    reproduced here from `gauss_legendre_01(3)` alone.
//! 3. On the affine unit hex MFEM's `IsoparametricTransformation` yields
//!    `J = I` with a single junk entry `J(0,2) = 5.2041704279304213e-18`
//!    at exactly the three qps {0, 9, 18}; `Weight() ≡ 1.0` at all 27 qps.
//!    (fem-rs today: `J = 0.5·I + different junk`, the D712 root cause.)
//! 4. The RT0 `[0,1]³` reference basis is the 6 signed axis modes
//!    `[-(1-z)·ẑ, -(1-y)·ŷ, x·x̂, y·ŷ, -(1-x)·x̂, z·ẑ]` (slot order =
//!    `CUBE::FaceVert` faces z−, y−, x+, y+, x−, z+ — fem-rs'
//!    `HDivSpace::HEX_FACES` order).  Physical vshape = `J·φ_ref/detJ`.
//!    With MFEM's rule + junk-J + MFEM's accumulation, the element mass
//!    equals MFEM's **bitwise** — including the cancellation debris
//!    (`0.3333333333333332`, not 1/3) that the D680 fixture carries.
//!
//! Consequence for the D721 decision: the element + quadrature + assembly
//! layers DO land bitwise on MFEM once the hex reference domain flips;
//! the only remaining bitwise risk is the geometry kernel's Jacobian op
//! order (the 5.2e-18 junk), which is a bounded, single-kernel work item.
//!
//! Fixture: `data/d680_vector_mass_mfem.txt` (MFEM 4.10, 17 significant
//! digits — parses back to exact bits).

use fem_element::quadrature::gauss_legendre_01;

/// MFEM affine unit-hex junk entries: `(qp, row, col, value)` — every
/// non-identity J entry across the 27 qps (probe dump; all values are
/// exact small multiples of 2^-60, the cancellation debris of MFEM's
/// affine-geometry evaluation).  `detJ ≡ 1.0` and `Weight ≡ 1.0` at all 27
/// qps.  (fem-rs today: `J = 0.5·I + different junk`, the D712 root cause.)
const J_JUNK: &[((usize, usize, usize), f64)] = &[
    ((0, 0, 2), 5.2041704279304213e-18),
    ((1, 0, 2), 6.9388939039072284e-18),
    ((2, 0, 2), 2.7755575615628914e-17),
    ((2, 1, 2), 3.4694469519536142e-18),
    ((2, 2, 1), 3.4694469519536142e-18),
    ((5, 2, 1), 3.4694469519536142e-18),
    ((8, 1, 2), 1.3877787807814457e-17),
    ((8, 2, 1), 3.4694469519536142e-18),
    ((9, 0, 2), 5.2041704279304213e-18),
    ((10, 0, 2), 6.9388939039072284e-18),
    ((11, 0, 2), 2.7755575615628914e-17),
    ((11, 1, 2), 3.4694469519536142e-18),
    ((17, 1, 2), 1.3877787807814457e-17),
    ((18, 0, 2), 5.2041704279304213e-18),
    ((19, 0, 2), 6.9388939039072284e-18),
    ((20, 0, 2), 2.7755575615628914e-17),
    ((20, 1, 2), 3.4694469519536142e-18),
    ((20, 2, 1), 1.3877787807814457e-17),
    ((23, 2, 1), 1.3877787807814457e-17),
    ((26, 1, 2), 1.3877787807814457e-17),
    ((26, 2, 1), 1.3877787807814457e-17),
];

// ─── [0,1] quadrature layer ──────────────────────────────────────────────────

/// fem-rs' existing `[0,1]` 3-pt GL table == MFEM 4.10's table, bitwise.
#[test]
fn gauss_legendre_01_3pt_is_mfem_bitwise() {
    let (pts, wts) = gauss_legendre_01(3);
    // Dumped by tmp/d721 probes (MFEM 4.10 IntRules SEGMENT order 4).
    assert_eq!(pts[0].to_bits(), 0x3fbcda042f0236e1, "x1");
    assert_eq!(pts[1].to_bits(), 0x3fe0000000000000, "x2");
    assert_eq!(pts[2].to_bits(), 0x3fec64bf7a1fb924, "x3");
    assert_eq!(wts[0].to_bits(), 0x3fd1c71c71c71c72, "w1");
    assert_eq!(wts[1].to_bits(), 0x3fdc71c71c71c71c, "w2");
    assert_eq!(wts[2].to_bits(), 0x3fd1c71c71c71c72, "w3");
}

/// Build the 27-pt `[0,1]³` rule the way MFEM's CUBE tensor rule does
/// (x-fastest point order `i + 3j + 9k`, weight `(wx·wy)·wz`) and pin its bits.
#[test]
fn cube01_27pt_rule_weights_are_mfem_bitwise() {
    let rule = rule27();
    // Probe-dumped CUBE order-4 weights: distinct bit patterns, including the
    // 1-ulp split …af8/…af9 of 0.05486968449931412.
    assert_eq!(rule[0].3.to_bits(), 0x3f95f2a7db7a8e94, "qp0 corner");
    assert_eq!(rule[1].3.to_bits(), 0x3fa18eecaf953edc, "qp1 edge");
    assert_eq!(rule[4].3.to_bits(), 0x3fac17e118eecaf9, "qp4 face (…af9)");
    assert_eq!(rule[12].3.to_bits(), 0x3fac17e118eecaf8, "qp12 face (…af8)");
    assert_eq!(rule[13].3.to_bits(), 0x3fb67980e0bf08c7, "qp13 centre");
    // Point coordinates (x fastest): qp1 = (0.5, x1, x1).
    assert_eq!(rule[1].0.to_bits(), 0x3fe0000000000000);
    assert_eq!(rule[1].1.to_bits(), 0x3fbcda042f0236e1);
    assert_eq!(rule[1].2.to_bits(), 0x3fbcda042f0236e1);
}

// ─── Geometry ────────────────────────────────────────────────────────────────

/// MFEM's affine unit-hex Jacobian: identity with the junk entries
/// (`J_JUNK`) at the probe-pinned qps; `Weight() ≡ 1.0`, `detJ ≡ 1.0`.
struct Geom {
    junk: bool,
}

impl Geom {
    fn j(&self, qp: usize) -> [[f64; 3]; 3] {
        let mut j = [[0.0; 3]; 3];
        j[0][0] = 1.0;
        j[1][1] = 1.0;
        j[2][2] = 1.0;
        if self.junk {
            for &((q, r, c), v) in J_JUNK {
                if q == qp {
                    j[r][c] = v;
                }
            }
        }
        j
    }
    const fn weight(&self) -> f64 {
        1.0
    }
}

// ─── RT0 [0,1]³ basis ────────────────────────────────────────────────────────

/// MFEM `RT_HexahedronElement(0)` reference basis on `[0,1]³`
/// (probe == E dump): six signed axis modes in `CUBE::FaceVert` face order.
fn rt0_ref_shape(xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 6] {
    [
        [0.0, 0.0, -(1.0 - zeta)], // face z−
        [0.0, -(1.0 - eta), 0.0],  // face y−
        [xi, 0.0, 0.0],            // face x+
        [0.0, eta, 0.0],           // face y+
        [-(1.0 - xi), 0.0, 0.0],   // face x−
        [0.0, 0.0, zeta],          // face z+
    ]
}

/// Physical vshape = `J·φ_ref / detJ` (detJ ≡ 1 on the affine unit hex).
fn phys_shape(j: &[[f64; 3]; 3], r: &[[f64; 3]; 6]) -> [[f64; 3]; 6] {
    let mut out = [[0.0; 3]; 6];
    for (i, row) in out.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for d in 0..3 {
                acc += j[c][d] * r[i][d];
            }
            *v = acc / 1.0;
        }
    }
    out
}

/// The D680 fixture (MFEM 17-digit dump) parsed into (value, bits).
fn fixture(name: &str) -> Vec<Vec<u64>> {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/d680_vector_mass_mfem.txt"
    ));
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() == 3 && t[0] == name {
            let n: usize = t[1].parse().unwrap();
            let mut rows = Vec::with_capacity(n);
            for _ in 0..n {
                let row: Vec<u64> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|s| s.parse::<f64>().unwrap().to_bits())
                    .collect();
                rows.push(row);
            }
            return rows;
        }
    }
    panic!("fixture {name} not found");
}

/// Summation orders: MFEM `AddMult_a_AAt` scales the dot product by `w`
/// afterwards (fold_a); the probe fold scales each term (fold_b).  On this
/// element both must agree bitwise with MFEM — if they do, the pilot's
/// op-order risk is confined to the geometry kernel.
fn mass_entries(
    rule: &[(f64, f64, f64, f64)], // (xi, eta, zeta, weight)
    geom: &Geom,
    fold_per_term: bool,
) -> [[u64; 6]; 6] {
    let mut m = [[0.0_f64; 6]; 6];
    for (qp, &(xi, eta, zeta, w_ip)) in rule.iter().enumerate() {
        let j = geom.j(qp);
        let r = rt0_ref_shape(xi, eta, zeta);
        let v = phys_shape(&j, &r);
        let norm = w_ip * geom.weight();
        for i in 0..6 {
            for jj in 0..6 {
                if fold_per_term {
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += v[i][k] * v[jj][k] * norm;
                    }
                    m[i][jj] += acc;
                } else {
                    let mut dot = 0.0;
                    for k in 0..3 {
                        dot += v[i][k] * v[jj][k];
                    }
                    m[i][jj] += norm * dot;
                }
            }
        }
    }
    let mut bits = [[0u64; 6]; 6];
    for i in 0..6 {
        for jj in 0..6 {
            bits[i][jj] = m[i][jj].to_bits();
        }
    }
    bits
}

fn rule27() -> Vec<(f64, f64, f64, f64)> {
    let (xs, ws) = gauss_legendre_01(3);
    let mut rule = Vec::with_capacity(27);
    for kz in 0..3 {
        for ky in 0..3 {
            for kx in 0..3 {
                rule.push((xs[kx], xs[ky], xs[kz], (ws[kx] * ws[ky]) * ws[kz]));
            }
        }
    }
    rule
}

/// THE pilot assertion: with the `[0,1]³` rule + `[0,1]³` basis + MFEM's
/// affine J (junk included), the RT0 unit-hex element mass is MFEM's,
/// **bitwise**, under both summation orders.
#[test]
fn rt0_unit_hex_mass_is_mfem_bitwise_with_mfem_jacobian() {
    let rule = rule27();
    let geom = Geom { junk: true };
    let want = fixture("RT0_HexAFF");
    for (label, fold) in [("per-term (probe)", true), ("AddMult_a_AAt", false)] {
        let got = mass_entries(&rule, &geom, fold);
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(
                    got[i][j], want[i][j],
                    "{label}: RT0 ({i},{j}) bits {:016x} vs MFEM {:016x} ({})",
                    got[i][j],
                    want[i][j],
                    f64::from_bits(want[i][j])
                );
            }
        }
    }
}

/// How much of the remaining gap is geometry-kernel junk?  With J := exact I
/// (no 5.2e-18 debris) count the entries that miss MFEM's bits — the number
/// of bits a perfect `[0,1]³` geometry kernel alone still has to recover.
#[test]
fn rt0_unit_hex_mass_exact_identity_jacobian_miss_counter() {
    let rule = rule27();
    let geom = Geom { junk: false };
    let got = mass_entries(&rule, &geom, true);
    let want = fixture("RT0_HexAFF");
    let mut misses = 0usize;
    let mut max_abs = 0.0_f64;
    for i in 0..6 {
        for j in 0..6 {
            if got[i][j] != want[i][j] {
                misses += 1;
                let (g, w) = (f64::from_bits(got[i][j]), f64::from_bits(want[i][j]));
                max_abs = max_abs.max((g - w).abs());
            }
        }
    }
    // Record the numbers (do not force them to zero — that is the geometry
    // kernel's pilot work item, not this experiment's).
    eprintln!(
        "D721 pilot: exact-I J misses {misses}/36 entries, max |Δ| = {max_abs:.3e} (junk scale ~3e-18)"
    );
}
