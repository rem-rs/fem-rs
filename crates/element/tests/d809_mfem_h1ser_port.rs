//! D809 (round 75) — the 2-D serendipity arm is now MFEM 4.10's
//! `H1Ser_QuadrilateralElement` entry for entry.
//!
//! `crates/element/src/serendipity.rs::QuadSerendipityPk` used to be a
//! *different* element (D768's audit note: equispaced lattice, `4p` nodal
//! DOFs, the `{xⁱyʲ : i ∈ {0,p} ∨ j ∈ {0,p}}` tensor span).  D809 ports MFEM's
//! construction: the Gauss-Lobatto closed lattice, `(p²+3p+6)/2` DOFs, the real
//! serendipity space `S_p` (superlinear degree ≤ p) with non-nodal interior
//! Legendre bubbles from `p ≥ 4`, and MFEM's vertex → south/east/north/west →
//! interior slot order.
//!
//! Truth: `tests/data/d809_mfem_h1ser_truth.txt`, dumped by
//! `tmp/d809/d809_ser_probe.cpp` (WSL, MFEM 4.10 serial:
//! `H1Ser_QuadrilateralElement(p)` for `p = 1..5`: DOF count, node table,
//! `CalcShape`/`CalcDShape` at seven sample points, and the Kronecker /
//! partition-of-unity residuals at the nodes).
//!
//! `p = 1` is compared in a **reduced** form: MFEM's
//! `H1Ser_QuadrilateralElement(1)` is degenerate — the DOF formula gives 5, the
//! constructor's edge loop is empty (`for i < p-1`), so the fifth shape
//! function is identically zero and the element is not even nodal
//! (`kron p=1 = 1.0` in the dump).  fem-rs keeps the 4-DOF bilinear for `p = 1`
//! (MFEM's ordinary `H1_FECollection(1)`, `BiLinear2DFiniteElement` — already
//! pinned bit-for-bit by `d768_quad_serendipity.rs`), and this file pins the
//! *reason*: the degenerate fifth shape.

use fem_element::ReferenceElement;
use fem_element::serendipity::QuadSerendipityPk;

const TRUTH: &str = include_str!("data/d809_mfem_h1ser_truth.txt");

const PTS: [[f64; 2]; 7] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [0.3, 0.7],
    [0.25, 0.75],
];

struct Truth {
    /// `p` → DOF count.
    dofs: std::collections::HashMap<usize, usize>,
    /// `(p, k)` → node coordinate.
    nodes: std::collections::HashMap<(usize, usize), [f64; 2]>,
    /// `(p, k, sample index)` → value.
    shape: std::collections::HashMap<(usize, usize, usize), f64>,
    /// `(p, k, sample index)` → gradient.
    grad: std::collections::HashMap<(usize, usize, usize), [f64; 2]>,
    /// `p` → Kronecker residual at the node lattice.
    kron: std::collections::HashMap<usize, f64>,
    /// `(p, k)` → `∫_□ φ_k`.
    intphi: std::collections::HashMap<(usize, usize), f64>,
    /// `p` → `Σ_k ∫_□ φ_k` (1 iff the element reproduces constants).
    sumint: std::collections::HashMap<usize, f64>,
    /// `p` → max partition-of-unity residual at the sample points.
    pou_pts: std::collections::HashMap<usize, f64>,
}

fn truth() -> Truth {
    let mut t = Truth {
        dofs: Default::default(),
        nodes: Default::default(),
        shape: Default::default(),
        grad: Default::default(),
        kron: Default::default(),
        intphi: Default::default(),
        sumint: Default::default(),
        pou_pts: Default::default(),
    };
    for line in TRUTH.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("dofs") => {
                t.dofs.insert(f[1].parse().unwrap(), f[2].parse().unwrap());
            }
            Some("kron") => {
                t.kron.insert(f[1].parse().unwrap(), f[2].parse().unwrap());
            }
            Some("sumint") => {
                t.sumint.insert(f[1].parse().unwrap(), f[2].parse().unwrap());
            }
            Some("pou_pts") => {
                t.pou_pts.insert(f[1].parse().unwrap(), f[2].parse().unwrap());
            }
            Some("intphi") => {
                t.intphi.insert(
                    (f[1].parse().unwrap(), f[2].parse().unwrap()),
                    f[3].parse().unwrap(),
                );
            }
            Some("node") => {
                t.nodes.insert(
                    (f[1].parse().unwrap(), f[2].parse().unwrap()),
                    [f[3].parse().unwrap(), f[4].parse().unwrap()],
                );
            }
            Some("shape") => {
                let s = sample_index(f[3].parse().unwrap(), f[4].parse().unwrap());
                t.shape.insert(
                    (f[1].parse().unwrap(), f[2].parse().unwrap(), s),
                    f[5].parse().unwrap(),
                );
            }
            Some("grad") => {
                let s = sample_index(f[3].parse().unwrap(), f[4].parse().unwrap());
                t.grad.insert(
                    (f[1].parse().unwrap(), f[2].parse().unwrap(), s),
                    [f[5].parse().unwrap(), f[6].parse().unwrap()],
                );
            }
            _ => {}
        }
    }
    t
}

fn sample_index(x: f64, y: f64) -> usize {
    PTS.iter()
        .position(|p| p[0] == x && p[1] == y)
        .unwrap_or_else(|| panic!("sample point ({x},{y}) not in PTS"))
}

/// The DOF counts are MFEM's `(p² + 3p + 6)/2` — `5/8/12/17/23` for `p = 1..5`
/// (the `p = 1` member is the degenerate one documented above; fem-rs's
/// bilinear has 4).
#[test]
fn d809_dof_counts_are_mfems_serendipity_formula() {
    let t = truth();
    for p in 1..=5usize {
        let mfem = *t.dofs.get(&p).unwrap();
        assert_eq!(mfem, (p * p + 3 * p + 6) / 2, "MFEM formula at p={p}");
        let ours = QuadSerendipityPk::new(p).n_dofs();
        if p == 1 {
            assert_eq!(ours, 4, "p=1 is the 4-DOF bilinear (MFEM H1_FECollection(1))");
        } else {
            assert_eq!(ours, mfem, "p={p} DOF count");
        }
    }
}

/// MFEM's `H1Ser_QuadrilateralElement(1)` is degenerate: five DOFs, the fifth
/// shape identically zero at every sample point, and `kron p=1 = 1.0` (not
/// nodal) — this is why fem-rs keeps the bilinear at `p = 1` instead.
#[test]
fn d809_mfem_h1ser_order_1_is_degenerate() {
    let t = truth();
    assert_eq!(*t.dofs.get(&1).unwrap(), 5);
    for s in 0..PTS.len() {
        assert_eq!(*t.shape.get(&(1, 4, s)).unwrap(), 0.0, "5th shape at sample {s}");
        assert_eq!(*t.grad.get(&(1, 4, s)).unwrap(), [0.0, 0.0]);
    }
    assert!(t.kron[&1] > 0.5, "p=1 is not nodal (kron = {})", t.kron[&1]);
}

/// The node table is MFEM's Gauss-Lobatto closed lattice in MFEM's slot order
/// (vertices, south → east → north → west, then the interior `Sr_DOF_MAP`
/// slots) — entry for entry, `p = 2..=5`.
#[test]
fn d809_nodes_match_mfem_gauss_lobatto_lattice() {
    let t = truth();
    for p in 2..=5usize {
        let fe = QuadSerendipityPk::new(p);
        let nd = fe.n_dofs();
        let coords = fe.dof_coords();
        assert_eq!(coords.len(), nd);
        let mut worst = 0.0_f64;
        for k in 0..nd {
            let want = t.nodes[&(p, k)];
            worst = worst.max((coords[k][0] - want[0]).abs());
            worst = worst.max((coords[k][1] - want[1]).abs());
        }
        assert!(worst < 1e-15, "p={p}: node table deviates by {worst:.3e}");
    }
    // Sanity on the lattice itself: p = 2's GLL interior node *is* the midpoint
    // (the 3-point Gauss-Lobatto rule), p = 3's is not (0.2764/0.7236).
    let c2 = QuadSerendipityPk::new(2).dof_coords();
    assert!((c2[4][0] - 0.5).abs() < 1e-15, "p=2 south node is the midpoint");
    let c3 = QuadSerendipityPk::new(3).dof_coords();
    assert!((c3[4][0] - 0.27639320225002106).abs() < 1e-15, "p=3 GLL node");
}

/// `CalcShape` against MFEM's dump, entry for entry, `p = 2..=5` (all seven
/// sample points × all slots).
#[test]
fn d809_shapes_match_mfem_entrywise() {
    let t = truth();
    let mut worst = 0.0_f64;
    let mut wloc = None;
    for p in 2..=5usize {
        let fe = QuadSerendipityPk::new(p);
        let nd = fe.n_dofs();
        let mut vals = vec![0.0_f64; nd];
        for (s, pt) in PTS.iter().enumerate() {
            fe.eval_basis(pt, &mut vals);
            for k in 0..nd {
                let want = t.shape[&(p, k, s)];
                let d = (vals[k] - want).abs();
                if d > worst {
                    worst = d;
                    wloc = Some((p, k, s, vals[k], want));
                }
            }
        }
    }
    assert!(worst < 1e-14, "worst |Δshape| = {worst:.3e} at {wloc:?}");
}

/// `CalcDShape` against MFEM's dump, entry for entry, `p = 2..=5`.
#[test]
fn d809_gradients_match_mfem_entrywise() {
    let t = truth();
    let mut worst = 0.0_f64;
    let mut wloc = None;
    for p in 2..=5usize {
        let fe = QuadSerendipityPk::new(p);
        let nd = fe.n_dofs();
        let mut g = vec![0.0_f64; nd * 2];
        for (s, pt) in PTS.iter().enumerate() {
            fe.eval_grad_basis(pt, &mut g);
            for k in 0..nd {
                let want = t.grad[&(p, k, s)];
                for d in 0..2 {
                    let dd = (g[2 * k + d] - want[d]).abs();
                    if dd > worst {
                        worst = dd;
                        wloc = Some((p, k, s, d, g[2 * k + d], want[d]));
                    }
                }
            }
        }
    }
    assert!(worst < 1e-13, "worst |Δgrad| = {worst:.3e} at {wloc:?}");
}

/// `p = 2, 3` are nodal on the GLL lattice (MFEM's own `kron` residual is
/// 0.000e+00 there) and reproduce constants; from `p = 4` the interior bubbles
/// are non-nodal (MFEM's `kron`/`pu` residuals are O(1) — reproduced, not
/// "fixed").
#[test]
fn d809_nodality_follows_mfem() {
    let t = truth();
    let fe2 = QuadSerendipityPk::new(2);
    let fe3 = QuadSerendipityPk::new(3);
    for fe in [&fe2, &fe3] {
        let nd = fe.n_dofs();
        let mut vals = vec![0.0_f64; nd];
        let coords = fe.dof_coords();
        let mut worst = 0.0_f64;
        for k in 0..nd {
            let pt = [coords[k][0], coords[k][1]];
            fe.eval_basis(&pt, &mut vals);
            for j in 0..nd {
                let want = if j == k { 1.0 } else { 0.0 };
                worst = worst.max((vals[j] - want).abs());
            }
        }
        assert!(worst < 1e-13, "p={}: not nodal on the GLL lattice ({worst:.3e})", fe.order());
        assert!(t.kron[&(fe.order() as usize)] < 1e-13, "MFEM agrees");
    }
    // p = 4: MFEM is non-nodal there; assert the deviation is NOT small, so a
    // future "fix" that silently makes it nodal would fail this test.
    assert!(t.kron[&4] > 0.5, "p=4 bubbles are non-nodal in MFEM ({})", t.kron[&4]);
}

/// `p = 1` is still bit-for-bit MFEM's `BiLinear2DFiniteElement` (the D768 pin;
/// re-asserted here so the port cannot have regressed it).
#[test]
fn d809_p1_bilinear_is_untouched() {
    let fe = QuadSerendipityPk::new(1);
    assert_eq!(fe.n_dofs(), 4);
    let mut vals = vec![0.0_f64; 4];
    fe.eval_basis(&[0.25, 0.75], &mut vals);
    assert_eq!(vals, [0.75 * 0.25, 0.25 * 0.25, 0.75 * 0.75, 0.25 * 0.75]);
}

/// `∫_□ φ_k` reproduces MFEM's dump for `p = 2, 3`, where the element *is* a
/// partition of unity (`Σ∫φ = 1`): for `p = 2` the vertex integrals are
/// **negative** (`−1/12`) and the edge ones `1/3`, which is a property of
/// MFEM's construction, not of the old lattice element (whose integrals were
/// `1/4` each).
#[test]
fn d809_integrals_match_mfem_for_the_pou_orders() {
    let t = truth();
    for p in 2..=3usize {
        let fe = QuadSerendipityPk::new(p);
        let nd = fe.n_dofs();
        let q = fe.quadrature(10);
        let mut phi = vec![0.0_f64; nd];
        let mut iint = vec![0.0_f64; nd];
        for (qi, pt) in q.points.iter().enumerate() {
            fe.eval_basis(pt, &mut phi);
            for k in 0..nd {
                iint[k] += q.weights[qi] * phi[k];
            }
        }
        let mut worst = 0.0_f64;
        for k in 0..nd {
            worst = worst.max((iint[k] - t.intphi[&(p, k)]).abs());
        }
        assert!(worst < 1e-13, "p={p}: ∫φ deviates by {worst:.3e}");
        let sum: f64 = iint.iter().sum();
        assert!((sum - 1.0).abs() < 1e-13, "p={p}: Σ∫φ = {sum}");
        assert!((sum - t.sumint[&p]).abs() < 1e-13);
    }
    // p = 2's signature: negative vertex integrals.
    assert!(t.intphi[&(2, 0)] < -0.08 && t.intphi[&(2, 0)] > -0.09);
    assert!((t.intphi[&(2, 4)] - 1.0 / 3.0).abs() < 1e-15);
}

/// **Finding (D809-1)**: MFEM's own `H1Ser_QuadrilateralElement` stops
/// reproducing constants from `p = 4` — `Σ∫φ = 1 + 1/36` and the sample-point
/// partition-of-unity residual is `6.25e-2` (the interior Legendre bubbles are
/// not fixed up for the vertex/edge corrections).  The port reproduces this
/// faithfully rather than "fixing" it: the test asserts the deviation is
/// **present** in MFEM's dump *and* in our element, so neither can drift
/// silently.
#[test]
fn d809_high_order_constants_quirk_is_reproduced() {
    let t = truth();
    for p in 4..=5usize {
        assert!(
            (t.sumint[&p] - 1.0).abs() > 1e-3,
            "p={p}: MFEM Σ∫φ = {} — expected the documented deviation",
            t.sumint[&p]
        );
        assert!(t.pou_pts[&p] > 1e-3, "p={p}: MFEM POU residual {}", t.pou_pts[&p]);

        let fe = QuadSerendipityPk::new(p);
        let nd = fe.n_dofs();
        let mut phi = vec![0.0_f64; nd];
        // Our element must deviate the same way (same order of magnitude, and
        // matching MFEM's value to quadrature accuracy).
        let q = fe.quadrature(10);
        let mut sum = 0.0_f64;
        for (qi, pt) in q.points.iter().enumerate() {
            fe.eval_basis(pt, &mut phi);
            sum += q.weights[qi] * phi.iter().sum::<f64>();
        }
        assert!(
            (sum - t.sumint[&p]).abs() < 1e-12,
            "p={p}: our Σ∫φ = {sum} vs MFEM {}",
            t.sumint[&p]
        );
        let mut pou = 0.0_f64;
        for pt in PTS.iter() {
            fe.eval_basis(pt, &mut phi);
            pou = pou.max((phi.iter().sum::<f64>() - 1.0).abs());
        }
        assert!(
            (pou - t.pou_pts[&p]).abs() < 1e-12,
            "p={p}: our POU residual {pou} vs MFEM {}",
            t.pou_pts[&p]
        );
    }
}
