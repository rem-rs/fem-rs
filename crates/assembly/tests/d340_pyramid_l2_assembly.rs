//! D340 — the L² **assembly** paths on a pyramid space.
//!
//! `L2Space` (D340) and `DGMassInverse` (D335) were wired for pyramids first;
//! the *generic* assembly dispatch was the remaining hole:
//! `ref_elem_vol_l2` fell through to `ref_elem_vol` for `Pyramid5`, which
//! returns the legacy equispaced `PyramidPk` with `(p+1)(p+2)(2p+3)/6` DOFs
//! against the space's `(p+1)³`, so `Assembler::assemble_bilinear` panicked
//! (`index out of bounds: the len is 5 but the index is 5`,
//! `standard/mass.rs:22`).  The arbitration for that file (granted in round 47)
//! added the two pyramid arms:
//!
//! * `ref_elem_vol_l2` → `P0Pyr` at order 0, else
//!   `fem_space::l2::l2_pyramid_element(o, L2Basis::GaussLegendre)`;
//! * `ref_elem_vol_for_space`'s GaussLobatto branch →
//!   `fem_space::l2::l2_pyramid_element(o, L2Basis::GaussLobatto)`.
//!
//! Both call the **same function** `L2Space::build_pyramid` numbers the
//! space's DOFs from, so the numbering and the evaluated element cannot
//! disagree.
//!
//! MFEM reference for the mass matrices is the D325 fixture's `DGMASS` /
//! `DGSUM` blocks (dense `MassIntegrator::AssembleElementMatrix` on the unit
//! pyramid, `IntRules.Get(Geometry::PYRAMID, 2p+2)`), the same dump the D335
//! test uses for `DGMassInverse`.

use fem_assembly::assembler::ref_elem_vol_l2;
use fem_assembly::postproc::grid_function::{
    compute_coeff_l2_norm, compute_coeff_l2_norm_first_n,
};
use fem_assembly::standard::MassIntegrator;
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::{Assembler, GridFunction};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::l2::l2_pyramid_element;
use fem_space::{L2Basis, L2Space};

const FIXTURE: &str =
    include_str!("../../element/tests/data/d325_l2_fuentes_pyramid_mfem.txt");

fn nums(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|t| t.parse::<f64>().expect("number"))
        .collect()
}

/// `(p, ndof, dense MFEM mass block)` — the `DGMASS` blocks of the fixture.
fn mfem_dense_mass() -> Vec<(usize, usize, Vec<Vec<f64>>)> {
    let mut out: Vec<(usize, usize, Vec<Vec<f64>>)> = Vec::new();
    let mut lines = FIXTURE.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("DGMASS p=") {
            let t: Vec<&str> = rest.split_whitespace().collect();
            let p: usize = t[0].parse().expect("p");
            let n: usize = t[1].parse().expect("ndof");
            let mut rows = Vec::with_capacity(n);
            for _ in 0..n {
                rows.push(nums(lines.next().expect("mass row")));
            }
            out.push((p, n, rows));
        }
    }
    assert_eq!(
        out.len(),
        2,
        "the fixture holds the entry-wise DGMASS blocks for p = 1..2; p = 3's \
         rows are elided there for size (its `DGSUM` row sums are pinned by \
         crates/assembly/tests/d335_pyramid_dgmassinv.rs)"
    );
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

/// The assembled L² pyramid mass matrix equals MFEM's dense
/// `MassIntegrator::AssembleElementMatrix`, entry for entry.
#[test]
fn pyramid_l2_mass_assembly_matches_mfem_dense_element_matrix() {
    for (p, n, want) in mfem_dense_mass() {
        let space = L2Space::new(unit_pyramid(), p as u8);
        assert_eq!(space.n_dofs(), n, "p={p}");
        let m = Assembler::assemble_bilinear(
            &space,
            &[&MassIntegrator { rho: 1.0 }],
            2 * p as u8 + 2,
        );
        assert_eq!(m.nrows, n, "p={p}");
        assert_eq!(m.ncols, n, "p={p}");
        for i in 0..n {
            for j in 0..n {
                let got = m.get(i, j);
                let e = want[i][j];
                assert!(
                    (got - e).abs() <= 1e-13 * e.abs().max(1e-3),
                    "p={p} M[{i}][{j}]: got {got} want {e}"
                );
            }
        }
    }
}

/// The total mass of the unit pyramid is its volume, at every order
/// (including the order-0 `P0Pyr` arm added by the arbitration).
#[test]
fn pyramid_l2_mass_totals_the_pyramid_volume() {
    let mass = |order: u8| -> (f64, usize) {
        let space = L2Space::new(unit_pyramid(), order);
        let m = Assembler::assemble_bilinear(
            &space,
            &[&MassIntegrator { rho: 1.0 }],
            2 * order + 2,
        );
        let total: f64 = (0..m.nrows)
            .map(|i| (0..m.ncols).map(|j| m.get(i, j)).sum::<f64>())
            .sum();
        (total, m.nrows)
    };
    for (order, want) in [(0u8, 1usize), (1, 8), (2, 27), (3, 64)] {
        let (total, n) = mass(order);
        assert_eq!(n, want, "order {order}: DOF count");
        assert!(
            (total - 1.0 / 3.0).abs() < 1e-13,
            "order {order}: unit-pyramid L2 mass {total} != 1/3"
        );
    }
}

/// The GaussLobatto pyramid L² space takes the *closed*-`btype` element arm
/// (`L2_FuentesPyramidElement(p, GaussLobatto)`), not the GaussLegendre one:
/// the two node tables differ, so the mass blocks differ.
#[test]
fn pyramid_l2_gauss_lobatto_space_assembles_its_own_element() {
    for p in 1..=2u8 {
        let gl = L2Space::new(unit_pyramid(), p);
        let gll = L2Space::new_with_basis(unit_pyramid(), p, L2Basis::GaussLobatto);
        let q = 2 * p + 2;
        let m_gl = Assembler::assemble_bilinear(&gl, &[&MassIntegrator { rho: 1.0 }], q);
        let m_gll = Assembler::assemble_bilinear(&gll, &[&MassIntegrator { rho: 1.0 }], q);
        assert_eq!(m_gl.nrows, m_gll.nrows);
        let mut differs = false;
        for i in 0..m_gl.nrows {
            for j in 0..m_gl.ncols {
                if (m_gl.get(i, j) - m_gll.get(i, j)).abs() > 1e-12 {
                    differs = true;
                }
            }
        }
        assert!(differs, "p={p}: GLL arm produced the GL mass matrix");
        // Both describe the same volume.
        for m in [&m_gl, &m_gll] {
            let total: f64 = (0..m.nrows)
                .map(|i| (0..m.ncols).map(|j| m.get(i, j)).sum::<f64>())
                .sum();
            assert!((total - 1.0 / 3.0).abs() < 1e-13, "p={p}: total {total}");
        }
    }
}

/// `ref_elem_vol_l2` returns exactly the element `L2Space` numbers its DOFs
/// from — one source of truth, for both `btype` arms.
#[test]
fn ref_elem_vol_l2_pyramid_arm_is_the_space_element() {
    for p in 1..=3u8 {
        let space = L2Space::new(unit_pyramid(), p);
        let re = ref_elem_vol_l2(ElementType::Pyramid5, p);
        assert_eq!(
            re.n_dofs(),
            space.element_dofs(0).len(),
            "p={p}: ref_elem_vol_l2 vs the space's DOFs per element"
        );
        // The node tables are the same, node for node.
        let want = l2_pyramid_element(p as usize, L2Basis::GaussLegendre).dof_coords();
        let got = re.dof_coords();
        assert_eq!(got.len(), want.len());
        for (k, w) in want.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (got[k][d] - w[d]).abs() <= 1e-15,
                    "p={p} node {k} comp {d}: {} vs {}",
                    got[k][d],
                    w[d]
                );
            }
        }
    }
    // The closed arm too.
    let gll = l2_pyramid_element(2, L2Basis::GaussLobatto).dof_coords();
    let gl = l2_pyramid_element(2, L2Basis::GaussLegendre).dof_coords();
    assert!(gll.iter().zip(gl.iter()).any(|(a, b)| (a[2] - b[2]).abs() > 1e-6));
}

// ─── the L2-error path ──────────────────────────────────────────────────────

/// The exact field both L²-error tests use.
///
/// A **transcendental** field on purpose: measured, the Fuentes pyramid L²
/// interpolation reproduces every polynomial of degree ≤ p *exactly* at
/// `p = 1, 2, 3` (lin/quad/cubic reference errors 2.3e-16 / 8.2e-17 / 1.2e-16),
/// so a polynomial would make the oracle read ~0 and prove nothing.  This one
/// gives 1.35e-1 / 1.49e-2 / 7.16e-3 at `p = 1/2/3` on the unit pyramid.
fn exact_field(x: &[f64]) -> f64 {
    (3.0 * x[0] + x[1]).sin() + x[2] * x[2]
}

/// The skewed pyramid of the D325 fixture (`(0,0,0) (1,0,0) (1.2,1,0)
/// (0,0.8,0) (0.3,0.45,1)`), so the L² error path is exercised on a
/// non-affine cell as well as on the reference one.
fn skew_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.2, 1.0, 0.0, //
            0.0, 0.8, 0.0, //
            0.3, 0.45, 1.0,
        ],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// The determinant the affine contraction of `simplex_jacobian` produces for a
/// 5-node element (`(3, 5)` → `_ => &[1, 2, 3]`, so
/// `J = [v1 − v0, v2 − v0, v3 − v0]`): three vectors lying in the (planar)
/// base, hence rank ≤ 2 for *every* pyramid.
///
/// This is the fact the dedicated pyramid branch in `compute_l2_error_owned`
/// exists to avoid — with it the measure was 0, so the L² error was silently
/// exactly `0.0` for every field, and `phys_coords` mapped every quadrature
/// point into the base plane.
fn base_triple_det(mesh: &Mesh<3>) -> f64 {
    let ns = mesh.element_nodes(0);
    let c = |k: usize| mesh.node_coords(ns[k]);
    let (v0, v1, v2, v3) = (c(0), c(1), c(2), c(3));
    let col = |k: usize| [v1[k] - v0[k], v2[k] - v0[k], v3[k] - v0[k]];
    let (a, b, d) = (col(0), col(1), col(2));
    a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0])
        + a[2] * (b[0] * d[1] - b[1] * d[0])
}

/// The independent oracle: the same integral computed from `pyramid_rule` +
/// `element_jacobian_at` — i.e. the linear pyramid map the **assembler** and
/// `L2Space::build_pyramid` use (D331's layer-slot permutation).
///
/// A second oracle exists in the test below: `geo_ref_elem_from_mesh` →
/// `GeoPyrP1` + `isoparametric_jacobian`, an independent *encoding* of the same
/// map.  Both must agree with the metric.
fn reference_l2_error(
    mesh: &Mesh<3>,
    space: &L2Space<Mesh<3>>,
    dofs: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
    q: u8,
) -> f64 {
    let re = l2_pyramid_element(space.order() as usize, L2Basis::GaussLegendre);
    let quad = re.quadrature(q);
    let mut err2 = 0.0_f64;
    for (qi, xi) in quad.points.iter().enumerate() {
        let (jac, xp) = fem_mesh::transformation::element_jacobian_at(mesh, 0, xi, 3);
        let w = quad.weights[qi] * jac.determinant().abs();
        let mut phi = vec![0.0; re.n_dofs()];
        re.eval_basis(xi, &mut phi);
        let uh: f64 = (0..re.n_dofs()).map(|i| dofs[i] * phi[i]).sum();
        err2 += w * (uh - exact(&xp)).powi(2);
    }
    err2.max(0.0).sqrt()
}

/// The second, independent oracle: the assembler's own isoparametric route
/// (`vector_assembler::geo_ref_elem_from_mesh` picks `GeoPyrP1` for a straight
/// pyramid — the D304 element — and `isoparametric_jacobian` evaluates it
/// against the vertex table).
fn reference_l2_error_iso(
    mesh: &Mesh<3>,
    space: &L2Space<Mesh<3>>,
    dofs: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
    q: u8,
) -> f64 {
    let re = l2_pyramid_element(space.order() as usize, L2Basis::GaussLegendre);
    let quad = re.quadrature(q);
    let ge = geo_ref_elem_from_mesh(mesh, 0).expect("a geometry element for the pyramid");
    // `compute_l2_error_owned`'s own rule: the geometry *table* for high-order
    // geometry, the vertex table otherwise.
    let gn: Vec<u32> = if mesh.geom_order() > 1 {
        mesh.geometry_nodes(0).to_vec()
    } else {
        mesh.element_nodes(0).to_vec()
    };
    let mut err2 = 0.0_f64;
    for (qi, xi) in quad.points.iter().enumerate() {
        let (_j, det, xp) = isoparametric_jacobian(mesh, &gn, ge.as_ref(), xi, 3);
        let w = quad.weights[qi] * det.abs();
        let mut phi = vec![0.0; re.n_dofs()];
        re.eval_basis(xi, &mut phi);
        let uh: f64 = (0..re.n_dofs()).map(|i| dofs[i] * phi[i]).sum();
        err2 += w * (uh - exact(&xp)).powi(2);
    }
    err2.max(0.0).sqrt()
}

/// **The fix, against two independent oracles.**
///
/// `compute_l2_error` used to return exactly `0.0` here (every quadrature
/// weight was 0 — see [`base_triple_det`]); the dedicated pyramid branch routes
/// the metric through `element_jacobian_at`, the same map the space places its
/// DOFs with.  The test requires, for both a reference and a non-affine
/// (skewed) pyramid at `p = 1..3`:
///
/// * the error is **not** zero and is the interpolant's genuine error,
/// * it equals [`reference_l2_error`] (the assembler's map),
/// * it equals [`reference_l2_error_iso`] (the independently encoded
///   `GeoPyrP1` map) — so a wrong map cannot pass,
/// * and `compute_l2_error_owned` returns the same value as
///   `compute_l2_error` for the full element set (`compute_l2_error` just
///   forwards `n_elems`, `grid_function.rs:1248-1256`).
#[test]
fn pyramid_l2_error_matches_both_independent_oracles() {
    let f = exact_field;
    for (name, mesh) in [("unit", unit_pyramid()), ("skew", skew_pyramid())] {
        // The mechanism the branch avoids: the affine contraction is singular
        // for a pyramid, so any path through `simplex_jacobian` gets a zero
        // measure (this is why the branch exists, and why the error used to be
        // identically 0.0).
        assert_eq!(
            base_triple_det(&mesh),
            0.0,
            "{name}: the planar base makes simplex_jacobian's (3,5) J singular"
        );
        for p in 1..=3u8 {
            let q = 2 * p + 2;
            let space = L2Space::new(mesh.clone(), p);
            let dofs = space.interpolate(&f);
            let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

            let got = gf.compute_l2_error(&f, q);
            let owned = gf.compute_l2_error_owned(&f, q, mesh.n_elements() as u32);
            let oracle = reference_l2_error(&mesh, &space, dofs.as_slice(), &f, q);
            let oracle_iso = reference_l2_error_iso(&mesh, &space, dofs.as_slice(), &f, q);

            assert!(
                got > 1e-3,
                "{name} p={p}: the error must no longer be identically 0, got {got:e}"
            );
            assert!(
                (got - oracle).abs() <= 1e-13 * oracle,
                "{name} p={p}: compute_l2_error {got:e} != oracle {oracle:e}"
            );
            assert!(
                (got - oracle_iso).abs() <= 1e-13 * oracle_iso,
                "{name} p={p}: compute_l2_error {got:e} != GeoPyrP1 oracle {oracle_iso:e}"
            );
            assert_eq!(
                got, owned,
                "{name} p={p}: compute_l2_error_owned must agree with compute_l2_error"
            );
        }

        // The `phys_coords` side (condition 6).  `phys_coords(x0, J, ξ)` is the
        // affine contraction `x0 + J·ξ`; with `simplex_jacobian`'s singular
        // base-edge `J` its third column is a *base* vector (z = 0), so every
        // quadrature point landed in the base plane.  The map the metric now
        // uses keeps the cell's height — and the two oracles above agreeing
        // with the metric is what pins the physical points to it.
        let ns = mesh.element_nodes(0);
        let x0 = mesh.node_coords(ns[0]);
        let quad = l2_pyramid_element(1, L2Basis::GaussLegendre).quadrature(4);
        let xi = quad
            .points
            .iter()
            .max_by(|a, b| a[2].partial_cmp(&b[2]).unwrap())
            .unwrap();
        let (j, xp) = fem_mesh::transformation::element_jacobian_at(&mesh, 0, xi, 3);
        assert!(
            j.determinant().abs() > 1e-12,
            "{name}: the pyramid Jacobian must be regular"
        );
        assert!(
            xp[2] > 0.5,
            "{name}: the map on the pyramid path must keep the cell height, z = {}",
            xp[2]
        );
        assert_eq!(
            x0[2], 0.0,
            "{name}: the base is in the z = 0 plane, so the affine surrogate \
             `x0 + J·ξ` (base-edge columns) can only ever produce z = 0"
        );
    }
}

/// The **closed-form** oracle.
///
/// The Fuentes pyramid L² interpolant reproduces every polynomial of degree
/// ≤ p exactly (measured in §7.3 of the evidence file: 2.3e-16 / 8.2e-17 /
/// 1.2e-16 for lin/quad/cubic), so for a *linear* exact field the true L² error
/// is **zero** — and the metric can only return zero if its physical points
/// `xp` and its weights `w` are both right.  A wrong (but non-degenerate) map
/// would integrate a non-zero `|u_h − u(xp)|²`, and no map at all gives the
/// `0.0` the branch replaced.  Together with the oracle test above (which
/// requires a *non*-zero value on a non-polynomial field) this pins both
/// directions.
#[test]
fn pyramid_l2_error_of_a_representable_field_is_zero() {
    let lin = |x: &[f64]| 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2] + 1.25;
    for (name, mesh) in [("unit", unit_pyramid()), ("skew", skew_pyramid())] {
        for p in 1..=3u8 {
            let q = 2 * p + 2;
            let space = L2Space::new(mesh.clone(), p);
            let dofs = space.interpolate(&lin);
            let gf = GridFunction::new(&space, dofs.as_slice().to_vec());
            let got = gf.compute_l2_error(&lin, q);
            assert!(
                got <= 1e-13,
                "{name} p={p}: a representable (linear) field must integrate to \
                 zero error, got {got:e} — a wrong pyramid map shows up here"
            );
        }
    }
}

/// Order convergence, with the measured values pinned as ranges: the
/// non-polynomial field's error must drop with `p` on both pyramids.
///
/// Measured: unit 1.3526e-1 → 1.4931e-2 → 7.1649e-3, skew 1.2814e-1 →
/// 2.5575e-2 → 6.1424e-3 for `p = 1/2/3`.
#[test]
fn pyramid_l2_error_converges_with_order() {
    let f = exact_field;
    for (name, mesh) in [("unit", unit_pyramid()), ("skew", skew_pyramid())] {
        let mut errs = Vec::new();
        for p in 1..=3u8 {
            let space = L2Space::new(mesh.clone(), p);
            let dofs = space.interpolate(&f);
            let gf = GridFunction::new(&space, dofs.as_slice().to_vec());
            errs.push(gf.compute_l2_error(&f, 2 * p + 2));
        }
        assert!(
            errs[0] > errs[1] && errs[1] > errs[2],
            "{name}: errors must decrease with the order, got {errs:?}"
        );
        // Pinned magnitudes (see the doc comment).
        let want = if name == "unit" {
            [1.3526495958998882e-1, 1.4931002802957881e-2, 7.164853451387289e-3]
        } else {
            [1.2814328972030531e-1, 2.557503748080528e-2, 6.142393030268551e-3]
        };
        for (k, w) in want.iter().enumerate() {
            assert!(
                (errs[k] - w).abs() <= 1e-12 * w,
                "{name} p={}: got {:e} want {w:e}",
                k + 1,
                errs[k]
            );
        }
    }
}

// ─── the curved pyramid ─────────────────────────────────────────────────────
/// The genuinely **curved** pyramid fixture: `set_curvature(2)` plus a real
/// second-order feature — the Fuentes geometry table's base-quad-centre node
/// (slot 13) displaced 0.1 out of the base plane.  No vertex moves, so a
/// vertex-only (P1) map cannot see it at all; the cell's measure moves off
/// `1/3` accordingly.  This is the D306/D339 recipe
/// (`crates/assembly/tests/d339_curved_pyramid_geometry.rs`).
fn curved_pyramid() -> Mesh<3> {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);
    let nodes = mesh.geometry_nodes(0).to_vec();
    let moved = nodes[13];
    {
        let g = mesh.geometry.as_mut().expect("set_curvature(2) built a geometry table");
        let off = moved as usize * 3;
        g.coords[off + 2] += 0.1;
    }
    mesh.invalidate_locators();
    mesh
}

/// `∫ |det J|` over element 0 with the map under test.
fn curved_volume(mesh: &Mesh<3>, q: u8) -> (f64, f64) {
    let rule = fem_element::quadrature::pyramid_rule(q);
    let mut via_helper = 0.0_f64;
    let mut via_mesh = 0.0_f64;
    for (xi, w) in rule.points.iter().zip(rule.weights.iter()) {
        via_helper += w
            * fem_mesh::transformation::element_jacobian_at(mesh, 0, xi, 3)
                .0
                .determinant()
                .abs();
        via_mesh += w * mesh.element_jacobian(0, xi).1.abs();
    }
    (via_helper, via_mesh)
}

/// The curved case, to the same standard as the straight one.
///
/// `compute_l2_error` panicked on a curved pyramid L² space before this
/// (`ref_elem_vol: unsupported (element_type=Pyramid5, order=2)`, the
/// file-local table in `grid_function.rs`); it now has a pyramid arm that
/// delegates to
/// `fem_element::lagrange::h1_pyramid_element(g, PyramidBasisType::default())`
/// — the same call `crates/mesh/src/transformation.rs::curved_pyramid_geometry`
/// (the D334 arm of `element_jacobian_at`) and
/// `geo_ref_elem_from_mesh`'s curved arm make.
///
/// What is cross-checked here:
/// * the **geometry source** itself: `element_jacobian_at` agrees with
///   `Mesh::element_jacobian` (an independent implementation of the same
///   isoparametric Jacobian, `crates/mesh/src/simplex.rs`) to ≤ 1e-15, and the
///   curved cell's volume is off `1/3` (so the map really sees the displaced
///   node);
/// * the **metric**: equal to both oracles and to
///   `compute_l2_error_owned`, for `p = 1..3`.
///
/// What is **not** independent of the metric: for the curved case the metric,
/// `oracle_jac` and `oracle_iso` are three *call paths* that all end in the
/// same delegation (`h1_pyramid_element(g, default())` over
/// `mesh.geometry_nodes`), so they cross-check the pairing/plumbing, not the
/// family choice; the family itself is pinned against MFEM by D347/D339's own
/// tests (`d339_curved_pyramid_geometry`, `d347_pyramid_fuentes_wiring`).
#[test]
fn curved_pyramid_l2_error_matches_both_independent_oracles() {
    let mesh = curved_pyramid();
    assert_eq!(mesh.geom_order(), 2, "the fixture must be high-order geometry");
    assert_eq!(mesh.geom_n_nodes(), 15, "Fuentes pyramid geometry table at g = 2");
    let (via_helper, via_mesh) = curved_volume(&mesh, 6);
    assert!(
        (via_helper - via_mesh).abs() <= 1e-15,
        "element_jacobian_at {} != Mesh::element_jacobian {}",
        via_helper,
        via_mesh
    );
    assert!(
        (via_helper - 1.0 / 3.0).abs() > 1e-6,
        "the fixture must be genuinely curved, volume {} is still 1/3",
        via_helper
    );

    let f = exact_field;
    for p in 1..=3u8 {
        let q = 2 * p + 2;
        let space = L2Space::new(mesh.clone(), p);
        let dofs = space.interpolate(&f);
        let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

        let got = gf.compute_l2_error(&f, q);
        let owned = gf.compute_l2_error_owned(&f, q, mesh.n_elements() as u32);
        let oracle = reference_l2_error(&mesh, &space, dofs.as_slice(), &f, q);
        let oracle_iso = reference_l2_error_iso(&mesh, &space, dofs.as_slice(), &f, q);

        assert!(
            got > 1e-3 && got.is_finite(),
            "curved p={p}: the error must be non-trivial and finite, got {got:e}"
        );
        assert!(
            (got - oracle).abs() <= 1e-13 * oracle,
            "curved p={p}: compute_l2_error {got:e} != oracle {oracle:e}"
        );
        assert!(
            (got - oracle_iso).abs() <= 1e-13 * oracle_iso,
            "curved p={p}: compute_l2_error {got:e} != iso-route oracle {oracle_iso:e}"
        );
        assert_eq!(
            got, owned,
            "curved p={p}: compute_l2_error_owned must agree with compute_l2_error"
        );
    }
}

/// The curved case's **closed-form** oracle, and its honest limit.
///
/// The L² pyramid space holds the polynomials of degree ≤ p *in reference
/// coordinates*.  With an order-2 geometry map `x(ξ)`, a linear field `f(x)`
/// composes to a **quadratic** in `ξ`, so for `p ≥ 2` the interpolant is exact
/// in exact arithmetic — measured 2.77e-16 (`p = 2`) and 4.01e-16 (`p = 3`),
/// asserted ≤ 1e-13.  At `p = 1` the space holds only linears in `ξ`, so the
/// error is the genuine interpolation error of that composition — measured
/// 4.4836580097054922e-3, which the test requires to match the oracle (i.e. it
/// is a real error, not a geometry artefact).
#[test]
fn curved_pyramid_l2_error_of_a_linear_field_is_exact_from_p2() {
    let lin = |x: &[f64]| 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2] + 1.25;
    let mesh = curved_pyramid();
    for p in 1..=3u8 {
        let q = 2 * p + 2;
        let space = L2Space::new(mesh.clone(), p);
        let dofs = space.interpolate(&lin);
        let gf = GridFunction::new(&space, dofs.as_slice().to_vec());
        let got = gf.compute_l2_error(&lin, q);
        let oracle = reference_l2_error(&mesh, &space, dofs.as_slice(), &lin, q);
        assert!(
            (got - oracle).abs() <= 1e-13 * oracle.max(1e-15),
            "curved p={p}: linear field error {got:e} != oracle {oracle:e}"
        );
        if p >= 2 {
            assert!(
                got <= 1e-13,
                "curved p={p}: f∘x is quadratic in ξ and the space holds it \
                 from p = 2, so the error must vanish; got {got:e}"
            );
        } else {
            assert!(
                (got - 4.4836580097054922e-3).abs() <= 1e-5,
                "curved p=1: the composition is quadratic, so the error is the \
                 genuine p = 1 interpolation error ~4.483658e-3; got {got:e}"
            );
        }
    }
}

// ─── D353: the 3-D isoparametric cells now integrate ────────────────────────

/// **D353 regression** (found as the "FOURTH FINDING" of
/// `tmp/d325/ARBITRATION_REQUEST.md`, fixed in round 48).
///
/// `fem_assembly::postproc::grid_function::compute_coeff_l2_norm` used to
/// return **`0.0` for every 3-D isoparametric cell** — Hex8, Prism6 and
/// Pyramid5 — while Quad4 and Tet4 were correct.  Cause: its per-element
/// geometry helper `element_jacobian` (`postproc/grid_function.rs`) listed
/// those types in `needs_iso` and then built the geometry element as
/// `ref_elem_vol(ElementType::Quad4, 1)` for every non-quad type — a **2-D**
/// basis for a 3-D cell, so the third column of `J` stayed 0, `det J ≡ 0` and
/// the norm collapsed to zero.
///
/// The helper now delegates to `fem_mesh::transformation::element_jacobian_at`
/// (the mesh crate's single source of truth for geometry Jacobians), so the
/// geometry element is the element's own type and the norm is the true cell
/// measure.  Measured (unit mesh of each type, `coeff = 1`, `quad_order = 6`):
///
/// | element | value | expected |
/// |---|---|---|
/// | Quad4 | 9.9999999999999978e-1 | 1 |
/// | Tet4 | 4.0824829046386302e-1 | `sqrt(1/6)` |
/// | Hex8 | 1 | 1 |
/// | Prism6 | 7.0710678118654757e-1 | `sqrt(1/2)` |
/// | Pyramid5 | 5.7735026918962584e-1 | `sqrt(1/3)` |
///
/// Everything except the pyramid is a *pre-existing* zero, not a round-47
/// regression — that is why the canary pinned the `0.0` values before.
#[test]
fn coeff_l2_norm_on_3d_iso_cells() {
    let one = |_: &[f64]| 1.0;
    let q = 6;
    let quad = Mesh::<2>::unit_square_quad(1);
    assert!(
        (compute_coeff_l2_norm(&quad, &one, q) - 1.0).abs() < 1e-13,
        "Quad4 must stay correct"
    );
    let tet = Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Tet4,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    assert!(
        (compute_coeff_l2_norm(&tet, &one, q) - (1.0 / 6.0f64).sqrt()).abs() < 1e-13,
        "Tet4 (non-iso branch) must stay correct"
    );
    for (name, mesh, want) in [
        ("hex8", Mesh::<3>::unit_cube_hex(1), 1.0),
        (
            "prism6",
            Mesh::<3>::uniform(
                vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1.],
                vec![0, 1, 2, 3, 4, 5],
                vec![1],
                ElementType::Prism6,
                vec![],
                vec![],
                ElementType::Tri3,
            ),
            0.5f64.sqrt(),
        ),
        ("pyramid5", unit_pyramid(), (1.0 / 3.0f64).sqrt()),
    ] {
        let got = compute_coeff_l2_norm(&mesh, &one, q);
        assert!(
            (got - want).abs() < 1e-12,
            "{name}: D353 regression — got {got} for the unit cell, expected the \
             exact cell measure {want}"
        );
    }
}

/// The same cell measures through the `ComputeLpNorm(2.0, coeff, mesh)`
/// entry point that reports over the *first n* elements (the parallel
/// partition entry point) — both call the same helper, so both were zero.
#[test]
fn coeff_l2_norm_first_n_on_3d_iso_cells() {
    let one = |_: &[f64]| 1.0;
    let q = 6;
    for (name, mesh, want) in [
        ("hex8", Mesh::<3>::unit_cube_hex(1), 1.0),
        ("pyramid5", unit_pyramid(), (1.0 / 3.0f64).sqrt()),
    ] {
        let n = mesh.n_elements() as usize;
        let got = compute_coeff_l2_norm_first_n(&mesh, &one, q, n);
        assert!(
            (got - want).abs() < 1e-12,
            "{name}: D353 regression in compute_coeff_l2_norm_first_n — got {got}, \
             expected {want}"
        );
    }
}

/// **D353 against the C++ oracle.**  MFEM 4.10's
/// `ComputeLpNorm(2.0, coeff, mesh, irs)` (`fem/coefficient.cpp:1751`
/// `LpNormLoop`) on the same four fixtures plus a scaled hex, with
/// `irs[geom] = IntRules.Get(geom, 6)`.
///
/// The probe is `tmp/d353_probe.cpp` (kept in fem-pro); the printed values
/// below are its output, reproduced verbatim:
///
/// ```text
/// quad4    order=6 npoints=16 box=[0,1]x[0,1]       n1=0.99999999999999978 n2=0.44721359549995787
/// hex8     order=6 npoints=64 box=[0,1]x[0,1]x[0,1] n1=0.99999999999999944 n2=0.44721359549995787
/// prism6   order=6 npoints=48 box=[0,1]x[0,1]x[0,1] n1=0.70710678118654757 n2=0.18257418583505539
/// pyramid5 order=6 npoints=64 box=[0,1]x[0,1]x[0,1] n1=0.57735026918962584 n2=0.16903085094570333
/// hex8_scaled order=6                                n1=4.8989794855663531  n2=8.7635609200826554
/// ```
///
/// `coeff = 1` checks the geometry map (`n1²` is the cell measure);
/// `coeff = x²` checks the quadrature points as well, because
/// `ComputeLpNorm(2.0, f)` returns `(∫|f|²)^{1/2}`, i.e. `(∫x⁴)^{1/2}` —
/// `sqrt(1/5)` on the unit quad/hex, `sqrt(1/30)` on the prism and
/// `sqrt(1/35)` on this pyramid.  The scaled `2×3×4` hex pins a non-unit map
/// (`n1 = sqrt(24)`).
///
/// Tolerances are `1e-14` relative rather than bit-exact: the two sides
/// traverse the same rule only if fem-rs's per-element rule table matches
/// MFEM's `IntRules` entry for entry, which is a separate (already
/// calibrated) concern; what this test pins is the *integral*.
#[test]
fn coeff_l2_norm_matches_the_cpp_compute_lp_norm_oracle() {
    let one = |_: &[f64]| 1.0;
    let xsquared = |x: &[f64]| x[0] * x[0];
    let q = 6;

    let prism6 = Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1.],
        vec![0, 1, 2, 3, 4, 5],
        vec![1],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    let scaled_hex = Mesh::<3>::make_cartesian_3d(
        1, 1, 1, ElementType::Hex8, 2.0, 3.0, 4.0, false,
    );

    // (name, mesh, n1 oracle, n2 oracle) — C++ output above.
    let cases: Vec<(&str, Mesh<3>, f64, f64)> = vec![
        ("hex8", Mesh::<3>::unit_cube_hex(1), 0.99999999999999944, 0.44721359549995787),
        ("prism6", prism6, 0.70710678118654757, 0.18257418583505539),
        ("pyramid5", unit_pyramid(), 0.57735026918962584, 0.16903085094570333),
        ("hex8_scaled", scaled_hex, 4.8989794855663531, 8.7635609200826554),
    ];
    for (name, mesh, want1, want2) in cases {
        let got1 = compute_coeff_l2_norm(&mesh, &one, q);
        let got2 = compute_coeff_l2_norm(&mesh, &xsquared, q);
        let rel = |got: f64, want: f64| ((got - want) / want).abs();
        assert!(
            rel(got1, want1) < 1e-14,
            "{name}: |1|_L2 = {got1}, C++ {want1}"
        );
        assert!(
            rel(got2, want2) < 1e-14,
            "{name}: |x²|_L2 = {got2}, C++ {want2}"
        );
    }

    // The 2-D Quad4 fixture lives on the same rule table path.
    let quad = Mesh::<2>::unit_square_quad(1);
    let got1 = compute_coeff_l2_norm(&quad, &one, q);
    let got2 = compute_coeff_l2_norm(&quad, &xsquared, q);
    assert!(
        ((got1 - 0.99999999999999978) / 0.99999999999999978).abs() < 1e-14,
        "quad4: |1|_L2 = {got1}, C++ 0.99999999999999978"
    );
    assert!(
        ((got2 - 0.44721359549995787) / 0.44721359549995787).abs() < 1e-14,
        "quad4: |x²|_L2 = {got2}, C++ 0.44721359549995787"
    );
}
