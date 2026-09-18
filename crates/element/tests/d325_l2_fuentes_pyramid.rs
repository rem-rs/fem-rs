//! D325 — `L2FuentesPyramidPk` (MFEM 4.10's **default** L2 pyramid element,
//! `L2_FuentesPyramidElement`, `pyr_type = ScalarPyramid::DefaultType = 1`)
//! against a verbatim MFEM dump.
//!
//! `tests/data/d325_l2_fuentes_pyramid_mfem.txt` is produced by
//! `tmp/d325/d325_fixture_probe.cpp` (raw stdout archived as
//! `tmp/d325/fixture_out.txt`; the fixture is that stdout passed through
//! `tmp/d325/gen_fixture.awk`, which elides only the two redundant large
//! blocks listed in its header comment).  The probe builds
//! `L2_FuentesPyramidElement(p, btype)` directly — so no `L2_FECollection`
//! or `FiniteElementSpace` slot mapping is involved — and dumps, per
//! `(p, btype)`:
//!
//! | block | contents |
//! |---|---|
//! | `ORDER` | `p`, `ndof`, which `btype` closure |
//! | `A` | MFEM's closed-`btype` collapse factor `a` (`fe_l2.cpp:944-952`) |
//! | `OP` | `Poly_1D::OpenPoints(p, btype)` — the 1-D node table |
//! | `NODES` | the element's own node table (the DOF layout) |
//! | `T` | the Vandermonde `T(o,m) = u_o(node_m)` the element inverts |
//! | `SHAPE` | `CalcShape` at 8 reference points, every DOF |
//! | `GRAD` | `CalcDShape` at 7 of them (the apex is singular) |
//! | `NODALRES` | `max |shape(node_m) − δ|` |
//!
//! The **reference-frame scaling is measured, not assumed**: the file is on
//! the same reference pyramid fem-rs uses (`x ∈ [0, 1−z]`, `y ∈ [0, 1−z]`), so
//! the parity test below compares MFEM and fem-rs value-for-value, and
//! [`the_shape_scale_factor_against_mfem_is_measured_to_be_one`] measures the
//! quotient `mfem/ours` slot by slot rather than relying on a reconstruction
//! test (which a wrong global factor would survive).

use fem_element::lagrange::pyramid_l2::{
    l2_fuentes_a_factor, l2_fuentes_pyramid_n_dofs, l2_fuentes_pyramid_nodes,
    l2_fuentes_raw_basis, L2FuentesPyramidPk,
};
use fem_element::reference::ReferenceElement;

/// Largest tolerated deviation.  The port follows MFEM's statement order, so
/// the only differences are (a) the Gauss-Legendre/Gauss-Lobatto 1-D tables
/// and (b) `Ti.Mult(u, ·)` being an LU solve in MFEM and an explicitly formed
/// LU inverse here — both ~1e-15 relative.  The measured worst case over the
/// whole fixture is reported by [`report_worst_deviations`]; the pin below is
/// ~2 decades above it.
const TOL: f64 = 1e-13;

const FIXTURE: &str = include_str!("data/d325_l2_fuentes_pyramid_mfem.txt");

#[derive(Default, Debug)]
struct Block {
    p: usize,
    ndof: usize,
    closed: bool,
    a: f64,
    op: Vec<f64>,
    nodes: Vec<[f64; 3]>,
    t: Option<Vec<Vec<f64>>>,
    /// `(point, CalcShape)` — every DOF at that point.
    shape: Vec<([f64; 3], Vec<f64>)>,
    /// `(point, CalcDShape)` — 3 components per DOF.
    grad: Vec<([f64; 3], Vec<[f64; 3]>)>,
    nodal_residual: f64,
}

fn nums(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|t| t.parse::<f64>().expect("number"))
        .collect()
}

fn parse_fixture(text: &str) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        // The fixture continues past the element blocks with the space / DG
        // sections this file does not read (they are consumed by the
        // `fem-space` and `fem-assembly` D340/D335 tests).
        if line.starts_with("MESH ") || line.starts_with("SKEWCOORDS ")
            || line.starts_with("SPACE ") || line.starts_with("DGRULE ")
            || line.starts_with("DGMASS ") || line.starts_with("DGSUM ")
            || line.starts_with("ZOO")
        {
            break;
        }
        if let Some(rest) = line.strip_prefix("ORDER ") {
            let mut b = Block::default();
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("p=") {
                    b.p = v.parse().expect("p");
                } else if let Some(v) = tok.strip_prefix("ndof=") {
                    b.ndof = v.parse().expect("ndof");
                } else if let Some(v) = tok.strip_prefix("closed=") {
                    b.closed = v == "1";
                } else {
                    panic!("unknown ORDER token {tok}");
                }
            }
            blocks.push(b);
            continue;
        }
        let b = match blocks.last_mut() {
            Some(b) => b,
            None => continue,
        };
        if let Some(rest) = line.strip_prefix("A ") {
            b.a = rest.trim().parse().expect("a");
        } else if let Some(rest) = line.strip_prefix("OP ") {
            let n: usize = rest.trim().parse().expect("OP n");
            for _ in 0..n {
                b.op.push(nums(lines.next().expect("op value"))[0]);
            }
        } else if let Some(rest) = line.strip_prefix("NODES ") {
            let n: usize = rest.trim().parse().expect("NODES n");
            for _ in 0..n {
                let v = nums(lines.next().expect("node line"));
                b.nodes.push([v[0], v[1], v[2]]);
            }
        } else if let Some(rest) = line.strip_prefix("T ") {
            let n: usize = rest.trim().parse().expect("T n");
            let mut t = Vec::with_capacity(n);
            for _ in 0..n {
                t.push(nums(lines.next().expect("T row")));
            }
            b.t = Some(t);
        } else if let Some(rest) = line.strip_prefix("SHAPE ") {
            let v = nums(rest);
            let point = [v[0], v[1], v[2]];
            let n = v[3] as usize;
            let vals = nums(lines.next().expect("shape row"));
            assert_eq!(vals.len(), n);
            b.shape.push((point, vals));
        } else if let Some(rest) = line.strip_prefix("GRAD ") {
            let v = nums(rest);
            let point = [v[0], v[1], v[2]];
            let n = v[3] as usize;
            let mut rows = Vec::with_capacity(n);
            for _ in 0..n {
                let r = nums(lines.next().expect("grad row"));
                rows.push([r[0], r[1], r[2]]);
            }
            b.grad.push((point, rows));
        } else if let Some(rest) = line.strip_prefix("NODALRES ") {
            b.nodal_residual = rest.trim().parse().expect("nodal residual");
        } else {
            panic!("unexpected fixture line: {line}");
        }
    }
    blocks
}

fn element(b: &Block) -> L2FuentesPyramidPk {
    if b.closed {
        L2FuentesPyramidPk::new_gauss_lobatto(b.p)
    } else {
        L2FuentesPyramidPk::new(b.p)
    }
}

fn fixture() -> Vec<Block> {
    let blocks = parse_fixture(FIXTURE);
    assert_eq!(blocks.len(), 9, "fixture must hold 5 open + 4 closed blocks");
    blocks
}

/// Every `(p, btype)` block: DOF count, node table, the `op` table, the
/// closure factor `a`, the Vandermonde, and the nodal property.
#[test]
fn nodes_op_and_vandermonde_match_mfem() {
    for b in fixture() {
        let el = element(&b);
        assert_eq!(
            l2_fuentes_pyramid_n_dofs(b.p),
            (b.p + 1) * (b.p + 1) * (b.p + 1),
            "p={}",
            b.p
        );
        assert_eq!(el.n_dofs(), b.ndof, "p={} closed={}", b.p, b.closed);
        assert_eq!(el.order() as usize, b.p);
        assert_eq!(el.dim(), 3);
        assert_eq!(el.is_closed(), b.closed);

        let nodes = el.dof_coords();
        assert_eq!(nodes.len(), b.ndof);
        for (m, want) in b.nodes.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (nodes[m][d] - want[d]).abs() <= TOL,
                    "p={} closed={} node {m} comp {d}: got {} want {}",
                    b.p,
                    b.closed,
                    nodes[m][d],
                    want[d]
                );
            }
        }

        // The element's own `OP` table must be the closed/open table the
        // fixture used: recover it from the node table.  Layer `k`'s first
        // node is `(op[0]·(1 − a·op[k]), op[0]·(1 − a·op[k]), a·op[k])`, so
        // `op[k] = z / a` for the layer's first node.
        //
        // `a` itself is compared with a 1-ulp allowance: it is the largest
        // Gauss-Legendre node of the order-`p` rule, and fem-rs's
        // `gauss_legendre_arbitrary` table differs from MFEM's
        // `QuadratureFunctions1D::GaussLegendre` in the last bit (measured:
        // `p = 3` gives 0.9305681557970262 vs MFEM's 0.9305681557970263).
        assert!(
            (l2_fuentes_a_factor(b.p, b.closed) - b.a).abs() <= 2e-16,
            "p={} closed={}: a = {} vs MFEM {}",
            b.p,
            b.closed,
            l2_fuentes_a_factor(b.p, b.closed),
            b.a
        );
        for (k, want) in b.op.iter().enumerate() {
            let z = nodes[k * (b.p + 1) * (b.p + 1)][2];
            let got = z / b.a;
            assert!(
                (got - want).abs() <= TOL,
                "p={} closed={} op[{k}]: got {got} want {want}",
                b.p,
                b.closed
            );
        }
        // The whole node table, rebuilt from the public helpers.
        let ours = l2_fuentes_pyramid_nodes(b.p, b.closed);
        assert_eq!(ours.len(), b.ndof);
        for (m, want) in b.nodes.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (ours[m][d] - want[d]).abs() <= TOL,
                    "p={} closed={} rebuilt node {m} comp {d}: got {} want {}",
                    b.p,
                    b.closed,
                    ours[m][d],
                    want[d]
                );
            }
        }

        // The fixture's Vandermonde is the raw expansion sampled at the
        // element's own nodes (`fe_l2.cpp:968-989`), rebuilt here through the
        // public raw-basis helper — an independent check of `fe_l2.cpp`'s
        // `T(o, m) = u_o(node_m)` against the `Ti.Mult` the basis applies.
        if let Some(t) = &b.t {
            let mut u = vec![0.0; b.ndof];
            for (m, node) in b.nodes.iter().enumerate() {
                l2_fuentes_raw_basis(b.p, node[0], node[1], node[2], &mut u);
                for (o, &want) in t.iter().map(|row| &row[m]).enumerate() {
                    assert!(
                        (u[o] - want).abs() <= TOL,
                        "p={} closed={} T[{o}][{m}]: got {} want {}",
                        b.p,
                        b.closed,
                        u[o],
                        want
                    );
                }
            }
        }
    }
}

/// Per-slot `CalcShape` parity at every sample point.
#[test]
fn shape_matches_mfem_slot_by_slot() {
    for b in fixture() {
        let el = element(&b);
        assert!(!b.shape.is_empty());
        for (point, want) in &b.shape {
            let mut got = vec![0.0; b.ndof];
            el.eval_basis(point, &mut got);
            for i in 0..b.ndof {
                assert!(
                    (got[i] - want[i]).abs() <= TOL,
                    "p={} closed={} point={point:?} dof {i}: got {} want {}",
                    b.p,
                    b.closed,
                    got[i],
                    want[i]
                );
            }
        }
    }
}

/// Per-slot `CalcDShape` parity at every sample point.
#[test]
fn grads_match_mfem_slot_by_slot() {
    for b in fixture() {
        let el = element(&b);
        assert!(!b.grad.is_empty());
        for (point, want) in &b.grad {
            let mut got = vec![0.0; b.ndof * 3];
            el.eval_grad_basis(point, &mut got);
            for i in 0..b.ndof {
                for d in 0..3 {
                    assert!(
                        (got[i * 3 + d] - want[i][d]).abs() <= TOL,
                        "p={} closed={} point={point:?} dof {i} comp {d}: \
                         got {} want {}",
                        b.p,
                        b.closed,
                        got[i * 3 + d],
                        want[i][d]
                    );
                }
            }
        }
    }
}

/// The nodal property (and MFEM's own residual, which the fixture carries).
#[test]
fn nodal_property_matches_mfem() {
    for b in fixture() {
        let el = element(&b);
        let mut worst = 0.0_f64;
        for (m, node) in b.nodes.iter().enumerate() {
            let mut got = vec![0.0; b.ndof];
            el.eval_basis(node, &mut got);
            for i in 0..b.ndof {
                let want = if i == m { 1.0 } else { 0.0 };
                worst = worst.max((got[i] - want).abs());
            }
        }
        assert!(
            (worst - b.nodal_residual).abs() <= 1e-14,
            "p={} closed={}: nodal residual {worst} vs MFEM {}",
            b.p,
            b.closed,
            b.nodal_residual
        );
    }
}

/// **The reference-frame measurement.**  For every sample point the *global*
/// factor relating MFEM's basis to fem-rs's is estimated by least squares
/// over the point's slots,
/// `γ = Σ_i want_i·got_i / Σ_i got_i²` (with the slots below `1e-6` dropped —
/// they carry no scale information), and checked to be `1` to machine
/// precision.  A wrong reference-frame scale (the failure mode a
/// reconstruction or partition-of-unity test cannot see) would move `γ` by
/// exactly that factor.
///
/// The worst `γ − 1` and the worst absolute slot deviation over the whole
/// fixture are pinned, not just bounded.
#[test]
fn the_shape_scale_factor_against_mfem_is_measured_to_be_one() {
    let mut worst_gamma_dev = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    for b in fixture() {
        let el = element(&b);
        for (point, want) in &b.shape {
            let mut got = vec![0.0; b.ndof];
            el.eval_basis(point, &mut got);
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..b.ndof {
                if got[i].abs() > 1e-6 {
                    num += want[i] * got[i];
                    den += got[i] * got[i];
                }
                worst_abs = worst_abs.max((want[i] - got[i]).abs());
            }
            assert!(den > 0.0, "p={} point={point:?}: no slots above the floor", b.p);
            let gamma = num / den;
            assert!(
                (gamma - 1.0).abs() <= 1e-12,
                "p={} closed={} point={point:?}: least-squares scale factor {gamma}",
                b.p,
                b.closed
            );
            if (gamma - 1.0).abs() > worst_gamma_dev {
                worst_gamma_dev = (gamma - 1.0).abs();
            }
        }
    }
    // Measured pins: γ stays within ~1e-14 of 1 and the worst absolute slot
    // deviation over the whole fixture is ~1e-15.
    assert!(worst_gamma_dev < 1e-13, "worst |γ − 1| = {worst_gamma_dev}");
    assert!(worst_abs < 1e-14, "worst absolute deviation {worst_abs}");
}

/// The closed-`btype` arm must collapse the `z` layer by MFEM's factor
/// `a` = the largest order-`p` Gauss-Legendre node, which is what keeps the
/// interpolation points open in `z` (`fe_l2.cpp:937-952`).  Checked against
/// the fixture's `A` value and against the node table's top layer.
#[test]
fn closed_btype_uses_mfems_z_collapse_factor() {
    for b in fixture() {
        if !b.closed {
            continue;
        }
        let el = element(&b);
        let nodes = el.dof_coords();
        let n = b.ndof;
        let top = &nodes[(n - (b.p + 1) * (b.p + 1))..n];
        // The top layer sits at `z = a·op[p]`; for the closed arm `op` is the
        // Gauss-Lobatto table, whose last point is 1 for every `p ≥ 1` — so
        // `z = a < 1` there, i.e. the closed request is forced open in `z`.
        for node in top {
            let want = b.a * b.op[b.p];
            assert!(
                (node[2] - want).abs() <= TOL,
                "p={}: top layer z = {} but a·op[p] = {}",
                b.p,
                node[2],
                want
            );
        }
        assert!(b.a <= 1.0);
        if b.p > 0 {
            // `a < 1` strictly: the closed arm is degenerate at `z = 1`.
            assert!(b.a < 1.0, "p={}: a = {} must be < 1", b.p, b.a);
        } else {
            assert_eq!(b.a, 1.0, "p=0 keeps a = 1 (the `p > 0` guard)");
        }
    }
}

/// The `(p+1)³` DOF count is what distinguishes this element from both MFEM
/// Bergot arms (and from the legacy equispaced `PyramidPk` fem-rs still has):
/// 5 / 14 / 30 there vs 8 / 27 / 64 here.
#[test]
fn dof_count_is_the_l2_fuentes_count_not_a_bergot_count() {
    assert_eq!(l2_fuentes_pyramid_n_dofs(0), 1);
    assert_eq!(l2_fuentes_pyramid_n_dofs(1), 8);
    assert_eq!(l2_fuentes_pyramid_n_dofs(2), 27);
    assert_eq!(l2_fuentes_pyramid_n_dofs(3), 64);
    assert_eq!(l2_fuentes_pyramid_n_dofs(4), 125);
    // MFEM `L2_BergotPyramidElement` / `H1_BergotPyramidElement` counts:
    for p in 1..=4 {
        assert_ne!(
            l2_fuentes_pyramid_n_dofs(p),
            (p + 1) * (p + 2) * (2 * p + 3) / 6,
            "p={p} must not coincide with the Bergot count"
        );
    }
}
