//! D324: `H1FuentesPyramidPk` (MFEM 4.10's **default** pyramid H¹ element,
//! `H1_FuentesPyramidElement`, `pyr_type = 1`) against a verbatim MFEM dump.
//!
//! `tests/data/fuentes_pyramid_h1_mfem.txt` is produced by
//! `tmp/d324/fuentes_probe.cpp`, which builds
//! `H1_FECollection(p, 3, BasisType::GaussLobatto)` — whose pyramid element
//! *is* the Fuentes one, `ScalarPyramid::DefaultType = 1` — and dumps, in the
//! element's **own** DOF order (`fe->GetNodes()`, `fe->CalcShape`,
//! `fe->CalcDShape`, so no `FiniteElementSpace` slot mapping is involved):
//!
//! | block | contents |
//! |---|---|
//! | `ORDER` | `p`, `ndof` |
//! | `NODES` | the element's node table (the DOF layout) |
//! | `MASS` | element mass matrix, collapsed tensor Gauss `n = 2p+4` (exact): `M_ij = Σ_q w_q φ_i φ_j (1−z_q)²`; entry-wise for `p ≤ 3` only |
//! | `SHAPE` | `CalcShape` at 5 points (interior, `x = 0` face, near-apex, exact apex) |
//! | `GRAD` | `CalcDShape` at the same 5 points |
//! | `INVAR` / `ROWSUM` / `DIAG` | `trace / ‖M‖_F / Σ M`, `Σ_j M_ij`, `M_ii` |
//!
//! Orders 4 and 5 are pinned by the pointwise tables (every DOF value and
//! gradient component at every sample point) plus the mass invariants — an
//! entry-wise `141×141` dump would add ~220 kB to the fixture.
//!
//! The family comparison this test closes (`Fuentes` vs the Bergot family
//! fem-rs has always used, `H1PyramidPk`) is in `tmp/d324/cmp_family.txt`.

use fem_element::lagrange::pyramid_fuentes::{
    fuentes_pyramid_n_dofs, h1_fuentes_pyramid_nodes, H1FuentesPyramidPk,
};
use fem_element::quadrature::gauss_legendre_01_arbitrary;
use fem_element::reference::ReferenceElement;

/// Largest tolerated deviation: the port follows MFEM's expression tree
/// statement by statement, so only the GLL points, the Gauss–Legendre rule of
/// the mass matrix and `libm`'s `pow` differ at all.
const TOL: f64 = 1e-14;

const FIXTURE: &str = include_str!("data/fuentes_pyramid_h1_mfem.txt");

#[derive(Default, Debug)]
struct Block {
    p: usize,
    ndof: usize,
    nodes: Vec<[f64; 3]>,
    mass: Vec<Vec<f64>>,
    trace: f64,
    fro: f64,
    total: f64,
    rowsum: Vec<f64>,
    diag: Vec<f64>,
    /// `(point, CalcShape)`
    shape: Vec<([f64; 3], Vec<f64>)>,
    /// `(point, CalcDShape)`, 3 components per DOF
    grad: Vec<([f64; 3], Vec<[f64; 3]>)>,
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
        if let Some(rest) = line.strip_prefix("ORDER ") {
            let mut b = Block::default();
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("p=") {
                    b.p = v.parse().expect("p");
                } else if let Some(v) = tok.strip_prefix("ndof=") {
                    b.ndof = v.parse().expect("ndof");
                }
            }
            blocks.push(b);
            continue;
        }
        let b = match blocks.last_mut() {
            Some(b) => b,
            None => continue,
        };
        if let Some(rest) = line.strip_prefix("NODES ") {
            let n: usize = rest.parse().expect("NODES n");
            for _ in 0..n {
                let l = lines.next().expect("node line");
                let v = nums(l);
                b.nodes.push([v[1], v[2], v[3]]);
            }
        } else if let Some(rest) = line.strip_prefix("MASS ") {
            let n: usize = rest.parse().expect("MASS n");
            for _ in 0..n {
                let l = lines.next().expect("mass row");
                b.mass.push(nums(l));
            }
        } else if let Some(rest) = line.strip_prefix("INVAR ") {
            for tok in rest.split_whitespace() {
                let (k, v) = tok.split_once('=').expect("key=value");
                let v: f64 = v.parse().expect("invariant");
                match k {
                    "trace" => b.trace = v,
                    "fro" => b.fro = v,
                    "total" => b.total = v,
                    other => panic!("unknown INVAR key {other}"),
                }
            }
        } else if let Some(rest) = line.strip_prefix("ROWSUM ") {
            let n: usize = rest.parse().expect("ROWSUM n");
            b.rowsum = nums(lines.next().expect("rowsum values"));
            assert_eq!(b.rowsum.len(), n);
        } else if let Some(rest) = line.strip_prefix("DIAG ") {
            let n: usize = rest.parse().expect("DIAG n");
            b.diag = nums(lines.next().expect("diag values"));
            assert_eq!(b.diag.len(), n);
        } else if let Some(rest) = line.strip_prefix("SHAPE ") {
            let v = nums(rest);
            let point = [v[0], v[1], v[2]];
            let vals = nums(lines.next().expect("shape values"));
            assert_eq!(vals.len(), b.ndof);
            b.shape.push((point, vals));
        } else if let Some(rest) = line.strip_prefix("GRAD ") {
            let v = nums(rest);
            let point = [v[0], v[1], v[2]];
            let mut rows = Vec::with_capacity(b.ndof);
            for _ in 0..b.ndof {
                let r = nums(lines.next().expect("grad row"));
                rows.push([r[0], r[1], r[2]]);
            }
            b.grad.push((point, rows));
        }
    }
    blocks
}

fn fixture() -> Vec<Block> {
    let blocks = parse_fixture(FIXTURE);
    assert_eq!(blocks.len(), 5, "fixture must hold p = 1..5");
    assert_eq!(blocks[0].p, 1);
    blocks
}

/// The collapsed-tensor Gauss rule the fixture's `MASS` blocks use:
/// `n = 2p + 4` points per collapsed axis, exactly integrating degree
/// `2p + 2` per variable.
fn collapsed_gauss(p: usize) -> (Vec<f64>, Vec<f64>) {
    gauss_legendre_01_arbitrary(2 * p + 4)
}

/// Element mass matrix in the element's own DOF order, with the fixture's
/// rule and summation order.
fn element_mass(elem: &H1FuentesPyramidPk, p: usize) -> Vec<Vec<f64>> {
    let (xs, ws) = collapsed_gauss(p);
    let n = elem.n_dofs();
    let mut m = vec![vec![0.0; n]; n];
    let mut phi = vec![0.0; n];
    for (a, &z) in xs.iter().enumerate() {
        for (b, &r) in xs.iter().enumerate() {
            for (c, &s) in xs.iter().enumerate() {
                elem.eval_basis(&[r * (1.0 - z), s * (1.0 - z), z], &mut phi);
                let w = ws[a] * ws[b] * ws[c] * (1.0 - z) * (1.0 - z);
                for i in 0..n {
                    if phi[i] == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        m[i][j] += w * phi[i] * phi[j];
                    }
                }
            }
        }
    }
    m
}

/// DOF counts `p(p²+3)+1` — 5 / 15 / 37 / 77 / 141 — and the node-table size.
#[test]
fn d324_fuentes_dof_counts_match_mfem() {
    let want = [5usize, 15, 37, 77, 141];
    for (b, &w) in fixture().iter().zip(want.iter()) {
        assert_eq!(b.ndof, w, "fixture dof count p={}", b.p);
        assert_eq!(fuentes_pyramid_n_dofs(b.p), w, "p={}", b.p);
        assert_eq!(H1FuentesPyramidPk::new(b.p).n_dofs(), w, "p={}", b.p);
        assert_eq!(h1_fuentes_pyramid_nodes(b.p).len(), w, "p={}", b.p);
    }
}

/// The DOF layout itself: every slot's reference position, i.e. the Fuentes
/// node table (`fe_h1.cpp:1064-1153`) — vertices, the 8 edge blocks, the base
/// quad face (`j` reversed), the four triangular faces and the `(p−1)³`
/// interior bubble grid.
#[test]
fn d324_fuentes_nodes_match_mfem() {
    for b in fixture() {
        let got = h1_fuentes_pyramid_nodes(b.p);
        assert_eq!(got.len(), b.ndof, "p={}", b.p);
        let mut worst = 0.0_f64;
        let mut at = (0usize, 0usize);
        for (m, w) in b.nodes.iter().enumerate() {
            for d in 0..3 {
                let e = (got[m][d] - w[d]).abs();
                if e > worst {
                    worst = e;
                    at = (m, d);
                }
            }
        }
        assert!(
            worst <= TOL,
            "p={}: node {}/axis {} off by {worst:e}",
            b.p,
            at.0,
            at.1
        );
        // The element's `dof_coords` must be the same table.
        let coords = H1FuentesPyramidPk::new(b.p).dof_coords();
        assert_eq!(coords.len(), b.ndof);
        for (m, c) in coords.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (c[d] - got[m][d]).abs() <= TOL,
                    "p={} slot {m}: dof_coords differs from the node table",
                    b.p
                );
            }
        }
    }
}

/// Element mass matrix vs MFEM: entry-wise for `p = 2, 3`, and through
/// `trace / ‖M‖_F / ΣM`, the row sums `∫φ_i` and the diagonal for `p = 4, 5`.
#[test]
fn d324_fuentes_mass_matches_mfem() {
    for b in fixture() {
        let elem = H1FuentesPyramidPk::new(b.p);
        let m = element_mass(&elem, b.p);
        let mut total = 0.0;
        let mut fro2 = 0.0;
        let mut trace = 0.0;
        for i in 0..b.ndof {
            trace += m[i][i];
            for j in 0..b.ndof {
                total += m[i][j];
                fro2 += m[i][j] * m[i][j];
            }
        }
        let fro = fro2.sqrt();
        assert!(
            (trace - b.trace).abs() <= TOL,
            "p={}: trace {} vs {}",
            b.p,
            trace,
            b.trace
        );
        assert!(
            (fro - b.fro).abs() <= TOL,
            "p={}: |M|_F {fro} vs {}",
            b.p,
            b.fro
        );
        assert!(
            (total - b.total).abs() <= TOL,
            "p={}: total {total} vs {}",
            b.p,
            b.total
        );
        for i in 0..b.ndof {
            let rs: f64 = m[i].iter().sum();
            assert!(
                (rs - b.rowsum[i]).abs() <= TOL,
                "p={}: row sum {i} = {rs} vs {}",
                b.p,
                b.rowsum[i]
            );
            assert!(
                (m[i][i] - b.diag[i]).abs() <= TOL,
                "p={}: diag {i} = {} vs {}",
                b.p,
                m[i][i],
                b.diag[i]
            );
        }
        if !b.mass.is_empty() {
            let mut worst = 0.0_f64;
            let mut at = (0usize, 0usize);
            for i in 0..b.ndof {
                for j in 0..b.ndof {
                    let e = (m[i][j] - b.mass[i][j]).abs();
                    if e > worst {
                        worst = e;
                        at = (i, j);
                    }
                }
            }
            assert!(
                worst <= TOL,
                "p={}: mass entry {}/{} off by {worst:e}",
                b.p,
                at.0,
                at.1
            );
        }
    }
}

/// `CalcShape` at all five sample points — every DOF value, including the
/// apex (`z = 1`) and the `x = 0` triangular face.
#[test]
fn d324_fuentes_shape_matches_mfem() {
    for b in fixture() {
        let elem = H1FuentesPyramidPk::new(b.p);
        let mut phi = vec![0.0; b.ndof];
        for (pt, want) in &b.shape {
            elem.eval_basis(pt, &mut phi);
            let mut worst = 0.0_f64;
            let mut at = 0usize;
            for (m, w) in want.iter().enumerate() {
                let e = (phi[m] - w).abs();
                if e > worst {
                    worst = e;
                    at = m;
                }
            }
            assert!(
                worst <= TOL,
                "p={} at {pt:?}: dof {at} off by {worst:e}",
                b.p
            );
            // Partition of unity, as a sanity check on the same values.
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() <= TOL, "p={} at {pt:?}: sum = {s}", b.p);
        }
    }
}

/// `CalcDShape` at all five sample points — every gradient component.
///
/// The criterion is relative to the **gradient scale of the sample point**
/// (`|Δ| ≤ TOL · max(1, max|∇φ|)`): the Fuentes gradients reach ~28 on the
/// graded node lattice, and `T⁻¹` (MFEM's LU of the Vandermonde) amplifies the
/// last-ulp differences of the raw expansion by its condition number, so a
/// pure absolute `1e-14` is below one ulp there.  Measured: absolute worst
/// `1.46e-13` at `p = 5`, i.e. `5.2e-15` relative to the point's scale;
/// `p ≤ 3` stay below `4.3e-15` absolute, and `p = 1` is exact.
#[test]
fn d324_fuentes_grad_matches_mfem() {
    for b in fixture() {
        let elem = H1FuentesPyramidPk::new(b.p);
        let mut g = vec![0.0; b.ndof * 3];
        for (pt, want) in &b.grad {
            elem.eval_grad_basis(pt, &mut g);
            let scale = want
                .iter()
                .flat_map(|w| w.iter())
                .fold(1.0_f64, |a, &x| a.max(x.abs()));
            let mut worst = 0.0_f64;
            let mut at = (0usize, 0usize);
            for (m, w) in want.iter().enumerate() {
                for d in 0..3 {
                    let e = (g[m * 3 + d] - w[d]).abs() / scale;
                    if e > worst {
                        worst = e;
                        at = (m, d);
                    }
                }
            }
            assert!(
                worst <= TOL,
                "p={} at {pt:?}: grad {}/{} off by {worst:e} (of scale {scale})",
                b.p,
                at.0,
                at.1
            );
        }
    }
}

/// The nodal property `φ_m(node_l) = δ_ml` and partition of unity /
/// constant-annihilating gradients, which together with the fixture pin the
/// Vandermonde inversion (MFEM's `Ti.Factor(T)`).
#[test]
fn d324_fuentes_nodal_and_pou() {
    for p in 1..=5usize {
        let elem = H1FuentesPyramidPk::new(p);
        let coords = elem.dof_coords();
        let n = elem.n_dofs();
        let mut phi = vec![0.0; n];
        for (l, node) in coords.iter().enumerate() {
            elem.eval_basis(node, &mut phi);
            for (m, v) in phi.iter().enumerate() {
                let target = if l == m { 1.0 } else { 0.0 };
                assert!(
                    (v - target).abs() < 1e-12,
                    "p={p}: φ_{m}(node {l}) = {v}"
                );
            }
        }
        let mut g = vec![0.0; n * 3];
        for &(x, y, z) in [(0.1, 0.2, 0.3), (0.05, 0.4, 0.5), (0.2, 0.0, 0.3)].iter() {
            elem.eval_basis(&[x, y, z], &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "p={p}: sum = {s}");
            elem.eval_grad_basis(&[x, y, z], &mut g);
            for d in 0..3 {
                let sd: f64 = (0..n).map(|i| g[i * 3 + d]).sum();
                assert!(sd.abs() < 1e-12, "p={p} at ({x},{y},{z}): grad sum {d} = {sd}");
            }
        }
    }
}

/// Gradient finite differences against `CalcShape` on the interior.
#[test]
fn d324_fuentes_gradient_fd() {
    let h = 1e-7;
    for p in 1..=5usize {
        let elem = H1FuentesPyramidPk::new(p);
        let n = elem.n_dofs();
        let (mut vc, mut vx, mut vy, mut vz, mut g) = (
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n * 3],
        );
        for &(x, y, z) in [(0.1, 0.2, 0.3), (0.05, 0.4, 0.5)].iter() {
            elem.eval_basis(&[x, y, z], &mut vc);
            elem.eval_basis(&[x + h, y, z], &mut vx);
            elem.eval_basis(&[x, y + h, z], &mut vy);
            elem.eval_basis(&[x, y, z + h], &mut vz);
            elem.eval_grad_basis(&[x, y, z], &mut g);
            for i in 0..n {
                let fd = [
                    (vx[i] - vc[i]) / h,
                    (vy[i] - vc[i]) / h,
                    (vz[i] - vc[i]) / h,
                ];
                for d in 0..3 {
                    // O(h) truncation: the second derivatives on this node
                    // lattice reach O(10²), so `h·|φ''|/2 ≈ 5e-6`.
                    let tol = 1e-4 * (1.0 + g[i * 3 + d].abs());
                    assert!(
                        (g[i * 3 + d] - fd[d]).abs() < tol,
                        "p={p} at ({x},{y},{z}) dof {i} dir {d}: {} vs {}",
                        g[i * 3 + d],
                        fd[d]
                    );
                }
            }
        }
    }
}
