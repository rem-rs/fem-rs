//! D353 sweep: the *same class* of defect as the `compute_coeff_l2_norm` zero
//! — a geometry reference element of the wrong dimension, or a Jacobian built
//! from the wrong node columns, so `det J` collapses and the quantity that
//! multiplies it silently becomes `0.0`.
//!
//! `postproc/grid_function.rs::element_jacobian` built `ref_elem_vol(Quad4, 1)`
//! for every non-quad type (a 2-D basis for a 3-D cell).  This file covers the
//! second copy of the pattern:
//! `postproc/flux_recovery.rs::geom_jacobian` falls back to
//! "simplex-like (nodes 0..dim)" for **any** element type it does not special
//! case.  For a hex the three corners `nodes[1]`, `nodes[2]`, `nodes[3]` are
//! two base edges and the base *diagonal*, so the three columns of `J` are
//! linearly dependent and `det J == 0`; `compute_element_flux` then does
//! `jac.try_inverse().unwrap_or_default()` — a zero matrix — and returns an
//! identically zero flux.
//!
//! MFEM's `DiffusionIntegrator::ComputeElementFlux` evaluates `κ·∇u_h`
//! through `ElementTransformation::SetIntPoint` (an isoparametric map of the
//! element's own geometry), so on the affine unit box the flux is exactly the
//! constant gradient.  That is the oracle used below.

use fem_assembly::postproc::flux_recovery::FluxRecovery;
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::GridFunction;
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Flux-space DOF coordinates: MFEM's ZZ estimator recovers into a
/// `FiniteElementSpace` of the *same* order and geometry, so the H¹(T, 1) DOF
/// coordinates are the right sample set.
fn flux_dof_coords(elem_type: ElementType) -> Vec<Vec<f64>> {
    match elem_type {
        ElementType::Hex8 | ElementType::Hex20 => {
            fem_element::lagrange::HexQk::new(1).dof_coords()
        }
        ElementType::Prism6 | ElementType::Prism15 => {
            fem_element::lagrange::PrismPk::new(1).dof_coords()
        }
        other => panic!("no flux sample set wired for {other:?}"),
    }
}

/// Unit cube, 2×1×1 hexes (the `d250_error_geometry` HEXBOX fixture).
fn hex_box() -> Mesh<3> {
    Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false)
}

/// Unit right prism, vertices as in the D353 oracle probe:
/// (0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1).
fn unit_prism() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1.],
        vec![0, 1, 2, 3, 4, 5],
        vec![1],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Two unit prisms stacked along `z`, sharing the face `(3,4,5)`.
///
/// A **one-element** mesh cannot exercise the ZZ estimator on a P1 field: the
/// recovered flux equals the single element's own flux, so the difference is
/// identically zero whatever the field is.  The estimator needs at least two
/// cells for the recovered flux to have something to disagree with.
fn stacked_prisms() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 0., 1., 0., // 0..2
            0., 0., 1., 1., 0., 1., 0., 1., 1., // 3..5
            0., 0., 2., 1., 0., 2., 0., 1., 2., // 6..8
        ],
        vec![0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8],
        vec![1, 1],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `u = x + 2y + 3z` — affine, hence exactly representable in H¹(P1) and with
/// the constant gradient `(1, 2, 3)`.
fn u_lin(x: &[f64]) -> f64 {
    x[0] + 2.0 * x[1] + 3.0 * x[2]
}

/// The recovered flux of an affine field is the field's own gradient, at every
/// flux DOF, on any mesh whose geometry the recovery can represent.
///
/// This is the oracle for the D353 sweep: a geometry basis that disagrees with
/// the solution basis (wrong dimension, wrong reference frame, wrong node slot
/// order) makes the recovered gradient wrong, and a singular `J` makes it
/// identically zero.
#[test]
fn diffusion_flux_recovery_is_exact_on_an_affine_cell_field() {
    for (name, mesh) in [("hex8", hex_box()), ("prism6", unit_prism())] {
        let elem_type = mesh.element_type(0);
        let h1 = H1Space::new(mesh.clone(), 1);
        let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());

        let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
        let dof_coords = flux_dof_coords(elem_type);
        let dim = 3usize;

        for e in 0..mesh.n_elements() as u32 {
            let flux =
                integrator.compute_element_flux(&mesh, gf.space(), e, gf.dofs(), &dof_coords);
            assert_eq!(flux.len(), dof_coords.len() * dim);
            for (i, f) in flux.chunks(dim).enumerate() {
                for (d, &got) in f.iter().enumerate() {
                    let want = [1.0, 2.0, 3.0][d];
                    assert!(
                        (got - want).abs() < 1e-12,
                        "{name} element {e} flux dof {i} component {d}: got {got}, want \
                         {want} (an all-zero flux means det J == 0 — D353 sweep)"
                    );
                }
            }
        }
    }
}

/// The estimator must not be *degenerate*: the affine case above is the easy
/// one, and an estimator that returned zero indicators for every input would
/// pass it.  For a field the H¹(P1) space does **not** represent exactly, every
/// indicator must be strictly positive, and the flux-difference energy must be
/// concentrated where the field varies — a zero total here would mean the
/// recovered flux is identically zero (`det J == 0`), which is exactly the
/// defect this file guards.
#[test]
fn zz_estimator_mfem_nc_is_not_degenerate_on_a_hex_mesh() {
    use fem_assembly::postproc::flux_recovery::zz_estimator_mfem_nc;

    for (name, mesh) in [("hex8", hex_box()), ("prism6", stacked_prisms())] {
        let h1 = H1Space::new(mesh.clone(), 1);
        // Not in the P1 span, and *varying across the cells of both fixtures*:
        // the hex box splits in `x`, the prism stack in `z`, so `z²` is what
        // makes the two stacked prisms carry different fluxes (a z-independent
        // field would give them identical fluxes, and the ZZ difference would
        // vanish by construction rather than because of a defect).
        let u = |x: &[f64]| x[0].sin() * x[1] + x[2] * x[2];
        let gf = GridFunction::new(&h1, h1.interpolate(&u).as_slice().to_vec());
        let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };

        let ind = zz_estimator_mfem_nc(&gf, &integrator, &[]);
        assert!(
            ind.total_error > 1e-6,
            "{name}: ZZ total error {} is not distinguishable from a silently \
             zero estimator",
            ind.total_error
        );
        for (e, &eta) in ind.eta.iter().enumerate() {
            assert!(
                eta > 0.0,
                "{name}: element {e} indicator is {eta}; a mesh of unit cells \
                 carrying the same field must not have an exactly-zero indicator"
            );
        }
    }
}

/// The flux-difference energy of a *zero* difference is zero; of a constant
/// difference it is `κ·|v|²·|K|`.  On the affine unit box that is checkable in
/// closed form, and it exercises `compute_flux_energy`'s `det J` weight — the
/// second use of the same Jacobian in this file.
#[test]
fn diffusion_flux_energy_uses_a_nonsingular_det_j() {
    // (name, mesh, |K|): a 0.5×1×1 half-box per unit hex, and a unit prism
    // (base 1/2 × height 1 = 1/2).
    for (name, mesh, cell_volume) in
        [("hex8", hex_box(), 0.5), ("prism6", unit_prism(), 0.5)]
    {
        let integrator = DiffusionIntegrator::<f64> { kappa: 2.0 };
        let n = flux_dof_coords(mesh.element_type(0)).len();

        let diff = vec![1.0, 2.0, 3.0].repeat(n);
        let want_per_elem = 2.0 * (1.0 + 4.0 + 9.0) * cell_volume;
        for e in 0..mesh.n_elements() as u32 {
            let got = integrator.compute_flux_energy(&mesh, e, &diff);
            assert!(
                (got - want_per_elem).abs() < 1e-12,
                "{name} element {e}: energy {got}, want {want_per_elem} (a 0 here \
                 means det J == 0 — D353 sweep)"
            );
        }
    }
}

/// **End-to-end**: the estimator `amr_refiner::ThresholdRefiner` actually calls
/// (`zz_estimator_mfem_nc`) on a 3-D hex mesh.  `ref_elem_vol` refused Hex8
/// before this round, so the whole 3-D-hex dynamic-AMR path panicked; the
/// per-element pieces above are only useful if this entry point runs.
///
/// With a field the H¹(P1) space represents *exactly* (affine), the recovered
/// flux equals the true flux, so every element indicator is zero up to
/// rounding — MFEM's `ZienkiewiczZhuEstimator` on `hex-box` with a linear
/// `u` reports the same.  A non-zero total here means the flux recovery is
/// wrong, not merely imprecise.
#[test]
fn zz_estimator_mfem_nc_runs_on_a_hex_mesh_and_is_exact_for_an_affine_field() {
    use fem_assembly::postproc::flux_recovery::zz_estimator_mfem_nc;

    for (name, mesh) in [("hex8", hex_box()), ("prism6", stacked_prisms())] {
        let h1 = H1Space::new(mesh.clone(), 1);
        let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
        let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };

        let ind = zz_estimator_mfem_nc(&gf, &integrator, &[]);
        assert_eq!(ind.eta.len(), mesh.n_elements());
        assert!(
            ind.total_error < 1e-12,
            "{name}: ZZ total error {} must vanish for an exactly-representable \
             (affine) field; before round 48 this entry point panicked \
             (`ref_elem_vol: unsupported (element_type=Hex8, order=1)`)",
            ind.total_error
        );
    }
}
