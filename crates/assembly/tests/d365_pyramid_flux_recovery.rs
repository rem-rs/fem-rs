//! D365/D366 — the pyramid arm of the ZZ flux-recovery pipeline.
//!
//! Round 48 (D353) gave `postproc/flux_recovery.rs` hex/prism arms where the
//! basis and the geometry share one frame ("basis = geometry family"); the
//! pyramid cell was left out, so `ref_elem_vol` refused `Pyramid5` outright
//! and the whole pyramid ZZ error-estimation path panicked:
//!
//! ```text
//! ref_elem_vol: unsupported (element_type=Pyramid5, order=1)
//! ```
//!
//! Fix (round-49 ruling): the flux sampler is the **H¹ solution basis family**
//! — `fem_space::ref_elem::h1_pyramid_slots(order, PyramidBasisType::default())`
//! (Fuentes, entity slot order — the numbering `DofManager::build_pyramid_pk`
//! gives the H¹ space's `element_dofs`) — and the geometry Jacobian reuses the
//! mesh crate's `element_jacobian_at` (the `PYR_P1_SLOT_VERTEX` straight-pyramid
//! slot permutation, D331; the curved-pyramid order-`g` element, D334), exactly
//! the delegation `postproc/grid_function.rs` already makes.
//!
//! Oracles (closed-form, no C++ needed):
//! * `u = x + 2y + 3z` is affine, hence exactly representable in H¹(P1): the
//!   recovered flux must equal `(1,2,3)` at **every** flux DOF.
//! * a constant flux difference `v` has energy `κ·|v|²·|K|` with `|K| = ⅓` per
//!   unit pyramid.
//! * `zz_estimator_mfem_nc` (the `ThresholdRefiner` entry point) on the affine
//!   field must report `total_error < 1e-12`.
//!
//! Test anti-self-deception (round-48 lesson): a **one-element** mesh makes the
//! ZZ difference vanish identically for any field, and a z-stacked mesh makes
//! z-independent fields agree across cells by construction — so the estimator
//! runs on a **two-element bipyramid** (two unit pyramids sharing their base
//! quad) and the degeneracy probe uses `sin(x)·y + z²`, which varies across
//! both cells.

use fem_assembly::postproc::flux_recovery::{zz_estimator_mfem_nc, FluxRecovery};
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::GridFunction;
use fem_element::lagrange::{h1_pyramid_element, PyramidBasisType};
use fem_element::quadrature::pyramid_rule;
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Two unit pyramids glued base-to-base: the shared base quad
/// `(0,0,0),(1,0,0),(1,1,0),(0,1,0)` with apices at `z = ±1`.  Two cells share
/// (at p = 1) the four base DOFs, so the ZZ average has something to disagree
/// with; each cell keeps the unit-pyramid volume `|K| = 1/3`.
fn bipyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., // 0
            1., 0., 0., // 1
            1., 1., 0., // 2
            0., 1., 0., // 3
            0.5, 0.5, 1.0, // 4 — apex of the upper pyramid
            0.5, 0.5, -1.0, // 5 — apex of the lower pyramid
        ],
        vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 5],
        vec![1, 1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `u = x + 2y + 3z` — affine, hence exactly representable in H¹(P1), with the
/// constant gradient `(1, 2, 3)`.
fn u_lin(x: &[f64]) -> f64 {
    x[0] + 2.0 * x[1] + 3.0 * x[2]
}

/// Flux-space DOF coordinates: the H¹ pyramid element of the default family
/// (Fuentes) in entity slot order — the same slots the H¹ space numbers its
/// `element_dofs` in.
fn flux_dof_coords(order: u8) -> Vec<Vec<f64>> {
    h1_pyramid_element(order as usize, PyramidBasisType::default()).dof_coords()
}

/// The recovered flux of an affine field is the field's own gradient at every
/// flux DOF.  Before the D365 pyramid arm this test panicked in `ref_elem_vol`.
#[test]
fn d365_pyramid_flux_recovery_is_exact_on_an_affine_field() {
    let mesh = bipyramid();
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
    let coords = flux_dof_coords(1);
    let dim = 3usize;

    for e in 0..mesh.n_elements() as u32 {
        let flux = integrator.compute_element_flux(&mesh, gf.space(), e, gf.dofs(), &coords);
        assert_eq!(flux.len(), coords.len() * dim);
        for (i, f) in flux.chunks(dim).enumerate() {
            for (d, &got) in f.iter().enumerate() {
                let want = [1.0, 2.0, 3.0][d];
                assert!(
                    (got - want).abs() < 1e-12,
                    "element {e} flux dof {i} component {d}: got {got}, want {want}"
                );
            }
        }
    }
}

/// The energy of a constant flux difference `v` is `κ·|v|²·|K|` — per unit
/// pyramid with `v = (1,2,3)` and `κ = 2`: `2·(1+4+9)·(1/3) = 28/3`.  This
/// exercises `compute_flux_energy`'s pyramid quadrature (`pyramid_rule`,
/// weight sum 1/3) and its `det J` weight (the second use of the pyramid
/// geometry Jacobian).
#[test]
fn d365_pyramid_flux_energy_matches_the_closed_form() {
    let mesh = bipyramid();
    let integrator = DiffusionIntegrator::<f64> { kappa: 2.0 };
    let n = flux_dof_coords(1).len();
    let diff = vec![1.0, 2.0, 3.0].repeat(n);
    let want = 2.0 * 14.0 / 3.0;

    for e in 0..mesh.n_elements() as u32 {
        let got = integrator.compute_flux_energy(&mesh, e, &diff);
        assert!(
            (got - want).abs() < 1e-12,
            "element {e}: energy {got}, want {want} (a 0 here means det J == 0)"
        );
    }
}

/// End-to-end: the estimator `amr_refiner::ThresholdRefiner` calls on a
/// pyramid mesh with an exactly-representable (affine) field — every element
/// indicator vanishes up to rounding.  Before the D365 arm this entry point
/// panicked with
/// `ref_elem_vol: unsupported (element_type=Pyramid5, order=1)`.
#[test]
fn d365_zz_estimator_mfem_nc_pyramid_affine_total_error_below_1e_minus_12() {
    let mesh = bipyramid();
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };

    let ind = zz_estimator_mfem_nc(&gf, &integrator, &[]);
    assert_eq!(ind.eta.len(), mesh.n_elements());
    assert!(
        ind.total_error < 1e-12,
        "ZZ total error {} must vanish for an exactly-representable (affine) field",
        ind.total_error
    );
}

/// The estimator must not be degenerate: for a field the H¹(P1) space does
/// *not* represent exactly — one that also varies across the two cells — every
/// indicator is strictly positive.  `z²` is what keeps the two apices' cells
/// from carrying identical fluxes; a z-independent field would make the ZZ
/// difference vanish by construction.
#[test]
fn d365_zz_estimator_pyramid_is_not_degenerate_on_a_varying_field() {
    let mesh = bipyramid();
    let h1 = H1Space::new(mesh.clone(), 1);
    let u = |x: &[f64]| x[0].sin() * x[1] + x[2] * x[2];
    let gf = GridFunction::new(&h1, h1.interpolate(&u).as_slice().to_vec());
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };

    let ind = zz_estimator_mfem_nc(&gf, &integrator, &[]);
    assert!(
        ind.total_error > 1e-6,
        "ZZ total error {} is not distinguishable from a silently zero estimator",
        ind.total_error
    );
    for (e, &eta) in ind.eta.iter().enumerate() {
        assert!(
            eta > 0.0,
            "element {e} indicator is {eta}; a unit cell carrying a varying \
             field must not have an exactly-zero indicator"
        );
    }
}

/// D366: the `fe_order` inference behind `compute_flux_energy` must resolve
/// the pyramid flux-vector length to the same order the flux space used.  A
/// quadratic flux `g` (ζ-free, so `|g|²` has reference degree 4 and a
/// degree-4 rule is exact) makes the energy `κ·∫|g|²dΩ` exact only when the
/// inferred order is ≥ 2 (`quad_order = 2·fe_order` — a wrong `14/15 → 1`
/// inference would underintegrate the degree-4 integrand).  Expected values
/// are computed with an independent high-order reference rule and the mesh's
/// own geometry Jacobian, so the assertion pins the inference, not the
/// quadrature family.
#[test]
fn d366_pyramid_fe_order_inference_matches_the_flux_space_at_p1_p2() {
    let mesh = bipyramid();
    let dim = 3usize;
    // g(ξ) quadratic in the reference coordinates, ζ-free.
    let g = |xi: &[f64]| [1.0 + xi[0], 2.0 + xi[1], 3.0 + xi[0] * xi[1]];

    // Only p = 2 discriminates: there `g` lies in the Fuentes span, so the
    // nodal interpolant *is* `g` and the closed form applies; at p = 1 the
    // interpolant of a quadratic through the five corner dofs is not `g`, so
    // no closed form exists to compare against.  (Per-type × per-order
    // inference coverage is the `d366_fe_order_table` unit test inside
    // `flux_recovery.rs`.)
    for p in [2u8] {
        let elem = h1_pyramid_element(p as usize, PyramidBasisType::default());
        let n = elem.n_dofs();
        // flux_diff = g sampled at the flux DOF coordinates.
        let mut diff = vec![0.0; n * dim];
        for (j, xi) in elem.dof_coords().iter().enumerate() {
            for (d, v) in g(xi).iter().enumerate() {
                diff[j * dim + d] = *v;
            }
        }

        // Expected: κ·∫|g|²dΩ with an independent (high-order) rule.
        let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
        let want = {
            let quad = pyramid_rule(16);
            let mut acc = 0.0;
            for (q, xi) in quad.points.iter().enumerate() {
                let (j, _) = fem_mesh::transformation::element_jacobian_at(&mesh, 0, xi, 3);
                let gv = g(xi);
                let gv2: f64 = gv.iter().map(|v| v * v).sum();
                acc += quad.weights[q] * gv2 * j.determinant().abs();
            }
            acc
        };

        let got = integrator.compute_flux_energy(&mesh, 0, &diff);
        // Tolerance: the correct inference (order-4 rule) still differs from
        // the order-16 reference by ~4.0e-5 — the degree-4 |g|² times the
        // straight-pyramid (1-ζ)² measure is total degree 6, beyond the
        // order-4 rule's exactness.  A wrong inference (order-2 rule) misses
        // by ~8e-2, three orders of magnitude more.
        assert!(
            (got - want).abs() < 1e-4,
            "p={p}: energy {got}, want {want} — the fe_order inference mapped \
             n_flux_dofs={n} to the wrong order (its quadrature underintegrates \
             the degree-4 integrand)"
        );
    }
}
