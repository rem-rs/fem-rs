//! D335 — `DGMassInverse` on a pyramid L² space.
//!
//! `crates/assembly/src/dgmassinv.rs` is a 1:1 port of MFEM's
//! `fem/dgmassinv.hpp`/`.cpp`.  It was **unreachable** for pyramids twice over:
//!
//! 1. `L2Space::new` panicked on any 5-node mesh (D340), so
//!    `DGMassInverse::new` could not even be called; and
//! 2. `l2_ref_elem`'s fallback `dg_base::ref_elem_vol` panics on `Pyramid5`
//!    (`dg_base.rs:52`), so the element lookup inside `assemble()` would have
//!    panicked next.
//!
//! D340 + the pyramid arm in `l2_ref_elem` close both.  The reference values
//! are MFEM 4.10's **dense** `MassIntegrator::AssembleElementMatrix` for
//! `L2_FECollection(p, 3)`'s pyramid element, from the same probe as the D325
//! fixture (`tmp/d325/d325_fixture_probe.cpp`, blocks `DGRULE`/`DGMASS`/
//! `DGSUM`); `tmp/d325/d335_dginv_probe.cpp` records the separate finding that
//! MFEM's *own* `DGMassInverse` aborts on this element.
//!
//! ## MFEM cross-check: MFEM cannot run its own `DGMassInverse` here
//!
//! `DGMassInverse::Update` -> `BilinearForm::Assemble` at
//! `AssemblyLevel::PARTIAL` -> `MassIntegrator::AssemblePA`
//! (`fem/integ/bilininteg_mass_pa.cpp:29`) ->
//! `el.GetDofToQuad(*ir, DofToQuad::TENSOR)`.  `L2_FuentesPyramidElement` is a
//! `NodalFiniteElement`, whose `GetDofToQuad` forwards non-`FULL` modes to
//! `FiniteElement::GetDofToQuad`, which verifies `mode == DofToQuad::FULL`.
//! Measured (`tmp/d325/dginv2_out.txt`):
//!
//! ```text
//! ---- GetDofToQuad(TENSOR) ----
//! Verification failed: (mode == DofToQuad::FULL) is false:
//!  --> invalid mode requested
//!  ... in function: virtual const mfem::DofToQuad& mfem::FiniteElement::GetDofToQuad(...) const
//!  ... in file: fem/fe/fe_base.cpp:377
//! ```
//!
//! So the only MFEM-side reference for this path is the dense element matrix,
//! which is exactly what the fem-rs port assembles.  (The same situation is
//! already documented for the simplex L² bases in `dgmassinv.rs`'s header.)

use fem_assembly::dgmassinv::DGMassInverse;
use fem_element::quadrature::pyramid_rule;
use fem_mesh::{ElementType, Mesh};
use fem_space::{L2Basis, L2Space};

const FIXTURE: &str = include_str!("../../element/tests/data/d325_l2_fuentes_pyramid_mfem.txt");

fn nums(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|t| t.parse::<f64>().expect("number"))
        .collect()
}

#[derive(Default, Debug)]
struct DgBlock {
    p: usize,
    ndof: usize,
    /// `(point, weight)` of the rule MFEM integrated the mass matrix with.
    rule: Vec<([f64; 3], f64)>,
    /// Full dense element mass matrix (empty when the fixture elided it).
    mass: Vec<Vec<f64>>,
    /// `Σ_j M_ij` — the normalisation-sensitive invariant, kept for every p.
    rowsum: Vec<f64>,
}

fn parse_dg(text: &str) -> Vec<DgBlock> {
    let mut out: Vec<DgBlock> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("DGRULE p=") {
            let p: usize = rest.trim().parse().expect("p");
            let hdr = lines.next().expect("DGRULEPTS header");
            let n: usize = hdr
                .strip_prefix("DGRULEPTS ")
                .expect("DGRULEPTS")
                .trim()
                .parse()
                .expect("npts");
            let mut rule = Vec::with_capacity(n);
            for _ in 0..n {
                let v = nums(lines.next().expect("rule point"));
                rule.push(([v[0], v[1], v[2]], v[3]));
            }
            out.push(DgBlock { p, rule, ..Default::default() });
        } else if let Some(rest) = line.strip_prefix("DGMASS p=") {
            let t: Vec<&str> = rest.split_whitespace().collect();
            let p: usize = t[0].parse().expect("p");
            let n: usize = t[1].parse().expect("ndof");
            let b = out.iter_mut().find(|b| b.p == p).expect("block");
            b.ndof = n;
            for _ in 0..n {
                b.mass.push(nums(lines.next().expect("mass row")));
            }
        } else if let Some(rest) = line.strip_prefix("DGSUM p=") {
            let t: Vec<&str> = rest.split_whitespace().collect();
            let p: usize = t[0].parse().expect("p");
            let n: usize = t[1].parse().expect("ndof");
            let b = out.iter_mut().find(|b| b.p == p).expect("block");
            // `DGMASS`'s rows may have been elided for size (`# DGMASS rows
            // elided (ndof=64, ...)`), so the DOF count is taken from here too.
            b.ndof = n;
            b.rowsum = nums(lines.next().expect("rowsum"));
        }
    }
    // The elided `DGMASS` rows leave the header with the elision comment on
    // the same line; drop that marker so the parse stays strict.
    assert_eq!(out.len(), 3, "fixture must hold DGRULE p = 1..3");
    out
}

fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `rho = 1`, `quad_order = 2p + 2` (the exact rule, and the one the fixture
/// integrated with).
fn dg(p: u8) -> DGMassInverse<'static, L2Space<Mesh<3>>, f64> {
    let space: &'static L2Space<Mesh<3>> = Box::leak(Box::new(L2Space::new(unit_pyramid(), p)));
    DGMassInverse::new(space, 1.0, 2 * p + 2)
}

/// The pyramid arm exists at all — the two panics D335 recorded are gone.
#[test]
fn dgmassinv_is_reachable_on_a_pyramid_mesh() {
    for p in 1..=3u8 {
        let op = dg(p);
        assert_eq!(op.n_dofs(), (p as usize + 1).pow(3));
        assert_eq!(op.dofs_per_element(), (p as usize + 1).pow(3));
    }
}

/// Per-slot comparison of the assembled pyramid mass block against MFEM's
/// dense `MassIntegrator::AssembleElementMatrix` on the unit pyramid.
#[test]
fn pyramid_mass_block_matches_mfem_dense_element_matrix() {
    for b in parse_dg(FIXTURE) {
        if b.mass.is_empty() {
            continue;  // p = 3's rows were elided from the fixture (size only)
        }
        let op = dg(b.p as u8);
        let got = op.element_mass(0);
        assert_eq!(got.len(), b.ndof * b.ndof);
        for i in 0..b.ndof {
            for j in 0..b.ndof {
                let want = b.mass[i][j];
                let g = got[i * b.ndof + j];
                // Absolute tolerance scaled by the block's magnitude: the
                // entries range over ~5 decades between the vertex and the
                // interior modes.
                let tol = 1e-13 * want.abs().max(1e-3);
                assert!(
                    (g - want).abs() <= tol,
                    "p={} M[{i}][{j}]: got {g} want {want}",
                    b.p
                );
            }
        }
    }
}

/// The row sums `Σ_j M_ij = ∫_e φ_i` for every order — the invariant the
/// fixture keeps for `p = 3` too, and the one a wrong basis or a wrong
/// quadrature order moves immediately.
#[test]
fn pyramid_mass_row_sums_match_mfem() {
    for b in parse_dg(FIXTURE) {
        let op = dg(b.p as u8);
        let m = op.element_mass(0);
        for i in 0..b.ndof {
            let got: f64 = (0..b.ndof).map(|j| m[i * b.ndof + j]).sum();
            let want = b.rowsum[i];
            let tol = 1e-12 * want.abs().max(1e-4);
            assert!(
                (got - want).abs() <= tol,
                "p={} row {i}: got {got} want {want}",
                b.p
            );
        }
    }
}

/// The quadrature rule `DGMassInverse` uses is MFEM's pyramid rule of the same
/// order (`IntRules.Get(Geometry::PYRAMID, 2p + 2)`) — same point set and
/// weights.  Compared as a *set* keyed by the point coordinates: the two
/// implementations enumerate the collapsed tensor rule in different orders,
/// which is irrelevant to the mass kernel (a sum over the same pairs) and is
/// pinned separately by `d339_pyramid_rule_mfem`.
#[test]
fn dgmassinv_uses_mfems_pyramid_quadrature_rule() {
    for b in parse_dg(FIXTURE) {
        let r = pyramid_rule(2 * b.p as u8 + 2);
        assert_eq!(r.n_points(), b.rule.len(), "p={}", b.p);
        let mut ours: Vec<([i64; 3], f64)> = (0..r.n_points())
            .map(|q| {
                (
                    [
                        (r.points[q][0] * 1e15).round() as i64,
                        (r.points[q][1] * 1e15).round() as i64,
                        (r.points[q][2] * 1e15).round() as i64,
                    ],
                    r.weights[q],
                )
            })
            .collect();
        let mut theirs: Vec<([i64; 3], f64)> = b
            .rule
            .iter()
            .map(|(p, w)| {
                (
                    [
                        (p[0] * 1e15).round() as i64,
                        (p[1] * 1e15).round() as i64,
                        (p[2] * 1e15).round() as i64,
                    ],
                    *w,
                )
            })
            .collect();
        ours.sort_by_key(|e| e.0);
        theirs.sort_by_key(|e| e.0);
        for (q, ((op, ow), (tp, tw))) in ours.iter().zip(theirs.iter()).enumerate() {
            assert_eq!(op, tp, "p={} rule point {q}", b.p);
            assert!(
                (ow - tw).abs() <= 1e-15 * tw.abs().max(1e-3),
                "p={} rule weight {q}: got {ow} want {tw}",
                b.p
            );
        }
    }
}

/// `mult` solves `M u = b` element by element (the operator's contract).
#[test]
fn mult_inverts_the_pyramid_mass_blocks() {
    for p in 1..=3u8 {
        let op = dg(p);
        let n = op.n_dofs();
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();
        let mut u = vec![0.0; n];
        op.mult(&b, &mut u);
        // Residual per element, in the element's own block.
        let d = op.dofs_per_element();
        for e in 0..1 {
            let m = op.element_mass(e);
            for i in 0..d {
                let mut acc = 0.0;
                for j in 0..d {
                    acc += m[i * d + j] * u[e * d + j];
                }
                let want = b[e * d + i];
                assert!(
                    (acc - want).abs() <= 1e-10 * want.abs().max(1.0),
                    "p={p} dof {i}: (M u)_i = {acc} vs b_i = {want}"
                );
            }
        }
    }
}

/// The `GaussLobatto` L² basis arm reaches the closed-btype pyramid element
/// (same `(p+1)³` count, MFEM's closed node table).
#[test]
fn gauss_lobatto_l2_pyramid_reaches_the_closed_element() {
    for p in 1..=2u8 {
        let space: &'static L2Space<Mesh<3>> = Box::leak(Box::new(
            L2Space::new_with_basis(unit_pyramid(), p, L2Basis::GaussLobatto),
        ));
        let op = DGMassInverse::new(space, 1.0, 2 * p + 2);
        assert_eq!(op.dofs_per_element(), (p as usize + 1).pow(3));
        // The closed arm's mass row sums differ from the open arm's (it is a
        // different point set for p >= 1).
        let open = dg(p);
        let mut differs = false;
        for i in 0..op.dofs_per_element() {
            let a: f64 = (0..op.dofs_per_element())
                .map(|j| op.element_mass(0)[i * op.dofs_per_element() + j])
                .sum();
            let b: f64 = (0..open.dofs_per_element())
                .map(|j| open.element_mass(0)[i * open.dofs_per_element() + j])
                .sum();
            if (a - b).abs() > 1e-12 {
                differs = true;
            }
        }
        assert!(differs, "p={p}: the closed L2 pyramid arm produced the open matrix");
    }
}
