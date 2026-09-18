//! D331 — straight-pyramid geometry in `element_jacobian_at` /
//! `geometry_jacobian` (`crates/mesh/src/transformation.rs`).
//!
//! `PyramidPk(1)`'s layer slots carry the shape functions of the mesh
//! vertices *permuted* by `P1_SLOT_VERTEX = [0, 1, 3, 2, 4]`: MFEM's pyramid
//! vertex order is `(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1)` while the
//! layer order enumerates the base as `(0,0,0), (1,0,0), (0,1,0), (1,1,0),
//! (0,0,1)` (D191, `fem_element::lagrange::h1_pyramid_slot_labels`).  A
//! straight mesh stores its geometry nodes in **vertex** order, so the
//! layer-ordered basis must be evaluated over the permuted node list — the
//! `GeoPyrP1` convention the assembler has used since D304
//! (`assembler::geo_ref_elem`, `vector_assembler::geo_ref_elem_from_mesh`,
//! `standard/bbar`).
//!
//! Before the fix the two helpers in this file paired the layer-ordered
//! `PyramidPk(1)` with the raw vertex list, swapping the two base corners of
//! every straight pyramid: measured `∫|det J| = 0.173755809543588` on the unit
//! pyramid instead of the exact `1/3` (−47.9 %), while the element's own
//! vertices are reproduced (slot `s` → vertex `s`) so nothing looked wrong.
//!
//! The expectations below come from an *independent* replica — this file's own
//! `P1_SLOT_VERTEX` table applied to `PyramidPk(1)`'s basis from the
//! `fem-element` crate (pinned against MFEM 4.10 by D191/D304) — plus the
//! analytic determinants of the reference pyramid and of its affine images,
//! never from the code under test.

use fem_element::lagrange::pyramid::PyramidPk;
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::transformation::{element_jacobian_at, geometry_jacobian};
use fem_mesh::{Mesh, MeshTopology};

/// `PyramidPk(1)`'s layer slot `k` carries the shape function of mesh vertex
/// `P1_SLOT_VERTEX[k]` (D191; the same table `Mesh::set_curvature_pyramid5`
/// and `DofManager::build_pyramid_pk` use).
const P1_SLOT_VERTEX: [usize; 5] = [0, 1, 3, 2, 4];

/// MFEM/Gmsh pyramid vertex order in the reference domain: the base quad
/// `(0,0,0), (1,0,0), (1,1,0), (0,1,0)` plus the apex `(0,0,1)`.
const REF_VERTICES: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

const UNIT_VERTICES_FLAT: [f64; 15] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
];

fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        UNIT_VERTICES_FLAT.to_vec(),
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// The sheared/scaled/translated image `x = A·v + b` of the reference pyramid
/// (`A` upper triangular, `det A = 9`).
const AFFINE_A: [[f64; 3]; 3] = [[2.0, 0.3, 0.0], [0.0, 1.5, 0.2], [0.0, 0.0, 3.0]];
const AFFINE_B: [f64; 3] = [1.0, -2.0, 0.5];
const AFFINE_DET: f64 = 9.0; // 2 * 1.5 * 3 (A is triangular)

fn affine_pyramid() -> Mesh<3> {
    let mut coords = Vec::with_capacity(15);
    for v in REF_VERTICES {
        for i in 0..3 {
            coords.push((0..3).map(|j| AFFINE_A[i][j] * v[j]).sum::<f64>() + AFFINE_B[i]);
        }
    }
    Mesh::<3>::uniform(
        coords,
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Independent replica of `element_jacobian_at` for a **straight** pyramid:
/// evaluate `PyramidPk(1)`'s (layer-ordered) basis at `xi` over the geometry
/// nodes permuted into layer slots, i.e. over `X[l] = x(P1_SLOT_VERTEX[l])`.
/// Returns `(J, x)` with `J[i][j] = ∂x_i/∂ξ_j`.
fn replica_jacobian(mesh: &Mesh<3>, e: u32, xi: &[f64]) -> ([[f64; 3]; 3], [f64; 3]) {
    let gnodes = mesh.geometry_nodes(e);
    let layer: Vec<&[f64]> = P1_SLOT_VERTEX
        .iter()
        .map(|&v| mesh.geom_coords_of(gnodes[v]))
        .collect();
    let lin = PyramidPk::new(1);
    let mut phi = [0.0_f64; 5];
    let mut grad = [0.0_f64; 15];
    lin.eval_basis(xi, &mut phi);
    lin.eval_grad_basis(xi, &mut grad);
    let mut jac = [[0.0_f64; 3]; 3];
    let mut xp = [0.0_f64; 3];
    for k in 0..5 {
        for i in 0..3 {
            xp[i] += layer[k][i] * phi[k];
            for j in 0..3 {
                jac[i][j] += layer[k][i] * grad[k * 3 + j];
            }
        }
    }
    (jac, xp)
}

fn max_abs(a: &nalgebra::DMatrix<f64>, b: &[[f64; 3]; 3]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[(i, j)] - b[i][j]).abs());
        }
    }
    worst
}

/// Reference points inside the pyramid domain `{x, y >= 0, x, y <= 1 - z}`:
/// four interior points plus the four base corners.  The apex `(0,0,1)` is
/// excluded: the collapsed rational basis has a degenerate (zero-determinant)
/// Jacobian there, so `geometry_jacobian` — which inverts `J` — panics by
/// design; `probe_points_with_apex` covers it for the non-inverting helper.
fn probe_points() -> Vec<[f64; 3]> {
    let mut pts = vec![
        [0.25, 0.25, 0.25],
        [0.5, 0.25, 0.1],
        [0.1, 0.5, 0.2],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ];
    pts.extend(&REF_VERTICES[..4]);
    pts
}

fn probe_points_with_apex() -> Vec<[f64; 3]> {
    let mut pts = probe_points();
    pts.push(REF_VERTICES[4]);
    pts
}

/// `∫|det J|` over the element, using the pyramid quadrature rule (its points
/// are already in the pyramid domain and its weights carry the collapsed
/// coordinate Jacobian `(1-t)²`, so `Σ w = |Δ| = 1/3` for the reference
/// pyramid).
fn measure(mesh: &Mesh<3>, e: u32, order: u8) -> f64 {
    let rule = PyramidPk::new(1).quadrature(order);
    let mut total = 0.0_f64;
    for (q, xi) in rule.points.iter().enumerate() {
        let (jac, _x) = element_jacobian_at(mesh, e, xi, 3);
        total += rule.weights[q] * jac.determinant().abs();
    }
    total
}

/// The straight unit pyramid is mapped by the identity: `det J ≡ 1` at every
/// point and `∫|det J| = 1/3` (the pyramid volume) to round-off.
#[test]
fn straight_unit_pyramid_det_is_identity_and_measure_is_one_third() {
    let mesh = unit_pyramid();
    let mut worst: (f64, usize, Vec<f64>) = (0.0, 0, vec![0.0; 3]);
    for order in [2u8, 4, 6, 8] {
        let rule = PyramidPk::new(1).quadrature(order);
        for (q, xi) in rule.points.iter().enumerate() {
            let (jac, xp) = element_jacobian_at(&mesh, 0, xi, 3);
            let det = jac.determinant();
            if (det - 1.0).abs() > worst.0 {
                worst = ((det - 1.0).abs(), q, xi.clone());
            }
            // The map is the identity, so the physical point is the reference
            // point as well.
            for d in 0..3 {
                assert!(
                    (xp[d] - xi[d]).abs() < 1e-13,
                    "order {order} qp {q}: x({xi:?}) = {xp:?} is not the identity"
                );
            }
        }
        let total = measure(&mesh, 0, order);
        assert!(
            (total - 1.0 / 3.0).abs() <= 1e-15,
            "order {order}: ∫|det J| = {total:.17} instead of 1/3 = {:.17}",
            1.0 / 3.0
        );
    }
    assert!(
        worst.0 < 1e-13,
        "pointwise det J off the analytic 1 by {:.3e} at qp {} ({:?}) — the layer-ordered \
         `PyramidPk(1)` basis is not paired with the layer-slot node table (D331)",
        worst.0,
        worst.1,
        worst.2
    );
}

/// An affine image of the reference pyramid has a constant Jacobian equal to
/// `det A` everywhere and measure `det A / 3`.
#[test]
fn affine_pyramid_det_equals_det_a_everywhere() {
    let mesh = affine_pyramid();
    for order in [2u8, 6] {
        let rule = PyramidPk::new(1).quadrature(order);
        for (q, xi) in rule.points.iter().enumerate() {
            let (jac, xp) = element_jacobian_at(&mesh, 0, xi, 3);
            let det = jac.determinant();
            assert!(
                (det - AFFINE_DET).abs() < 1e-12,
                "order {order} qp {q} ({xi:?}): det J = {det} instead of det A = {AFFINE_DET}"
            );
            let want = (0..3)
                .map(|i| (0..3).map(|j| AFFINE_A[i][j] * xi[j]).sum::<f64>() + AFFINE_B[i])
                .collect::<Vec<_>>();
            for d in 0..3 {
                assert!(
                    (xp[d] - want[d]).abs() < 1e-13,
                    "order {order} qp {q}: x = {xp:?} instead of {:?}",
                    want
                );
            }
        }
        let total = measure(&mesh, 0, order);
        assert!(
            (total - AFFINE_DET / 3.0).abs() <= 1e-15,
            "order {order}: ∫|det J| = {total:.17} instead of det A/3 = {:.17}",
            AFFINE_DET / 3.0
        );
    }
}

/// `element_jacobian_at` must agree with the independent replica (permuted
/// layer-slot node table) at every probe point, for the unit and for the
/// affine pyramid.
#[test]
fn element_jacobian_at_matches_permuted_layer_replica() {
    for (name, mesh) in [("unit", unit_pyramid()), ("affine", affine_pyramid())] {
        for xi in probe_points_with_apex() {
            let (jac, xp) = element_jacobian_at(&mesh, 0, &xi, 3);
            let (rj, rx) = replica_jacobian(&mesh, 0, &xi);
            let err = max_abs(&jac, &rj);
            assert!(
                err < 1e-13,
                "{name} pyramid at {xi:?}: J differs from the layer-slot replica by {err:.3e}\n\
                 got  {jac:?}\nwant {rj:?}"
            );
            for d in 0..3 {
                assert!(
                    (xp[d] - rx[d]).abs() < 1e-13,
                    "{name} pyramid at {xi:?}: x = {xp:?} vs replica {rx:?}"
                );
            }
        }
    }
}

/// `geometry_jacobian` (the same node-table selection, returning `J^{-T}`)
/// must follow the same convention.
#[test]
fn geometry_jacobian_matches_permuted_layer_replica() {
    for (name, mesh) in [("unit", unit_pyramid()), ("affine", affine_pyramid())] {
        for xi in probe_points() {
            let (det, jit) = geometry_jacobian(&mesh, 0, &xi, 3);
            let (rj, _rx) = replica_jacobian(&mesh, 0, &xi);
            let want = nalgebra::DMatrix::from_row_slice(3, 3, &[
                rj[0][0], rj[0][1], rj[0][2],
                rj[1][0], rj[1][1], rj[1][2],
                rj[2][0], rj[2][1], rj[2][2],
            ])
            .try_inverse()
            .expect("replica Jacobian is invertible")
            .transpose();
            let mut err = (det - rj[0][0] * (rj[1][1] * rj[2][2] - rj[1][2] * rj[2][1])
                + rj[0][1] * (rj[1][2] * rj[2][0] - rj[1][0] * rj[2][2])
                + rj[0][2] * (rj[1][0] * rj[2][1] - rj[1][1] * rj[2][0]))
                .abs();
            assert!(
                err <= 1e-13 * AFFINE_DET.max(1.0),
                "{name} pyramid at {xi:?}: det J = {det} vs replica {err:.3e}"
            );
            for i in 0..3 {
                for j in 0..3 {
                    err = err.max((jit[(i, j)] - want[(i, j)]).abs());
                }
            }
            assert!(
                err < 1e-12,
                "{name} pyramid at {xi:?}: J^-T differs from the layer-slot replica by {err:.3e}"
            );
        }
    }
}
