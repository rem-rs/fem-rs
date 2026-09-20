//! D462 — `zz_estimator_mfem` must honour `FESpace::pyramid_basis()`.
//!
//! `postproc/flux_recovery.rs` picked the *solution* reference element with
//! `ref_elem_vol(elem_type, order)`, whose pyramid arm is unconditionally the
//! default family (Fuentes).  A Bergot pyramid solution space
//! (`H1Space::with_pyramid_basis(.., PyramidBasisType::Bergot)`, MFEM
//! `H1_FECollection(p, 3, .., pyr_type = 0)`) numbers its `element_dofs` in
//! the Bergot slot order — 14 dofs at p = 2, not Fuentes' 15 — so the
//! estimator read a 15-DOF Fuentes basis (and Fuentes dof coordinates)
//! against a 14-DOF element-dof table.
//!
//! Oracles (closed-form, the d365 pattern):
//! * `u = x + 2y + 3z` is affine, hence exactly representable: the flux
//!   recovered at *Bergot* dof coordinates must equal `(1,2,3)` everywhere and
//!   `zz_estimator_mfem` must report `total_error < 1e-12`.
//! * a field the space cannot represent exactly must give strictly positive
//!   indicators (anti-degeneracy).

use fem_assembly::postproc::flux_recovery::{zz_estimator_mfem, FluxRecovery};
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::GridFunction;
use fem_element::lagrange::{h1_pyramid_element, PyramidBasisType};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Two unit pyramids glued base-to-base (the d365 mesh): shared base quad
/// with apices at `z = ±1`.
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

fn u_lin(x: &[f64]) -> f64 {
    x[0] + 2.0 * x[1] + 3.0 * x[2]
}

/// Bergot flux-space DOF coordinates: the H¹ pyramid element of the *Bergot*
/// family — the slots the Bergot-managed H¹ space numbers its `element_dofs`
/// in.
fn bergot_dof_coords(order: u8) -> Vec<Vec<f64>> {
    h1_pyramid_element(order as usize, PyramidBasisType::Bergot).dof_coords()
}

/// The recovered flux of an affine field is the field's own gradient at every
/// *Bergot* flux DOF.  Before D462 the sampler was built as the Fuentes
/// default (15 dofs) against the Bergot element-dof table (14 slots) and the
/// gradient loop read past the table.
#[test]
fn d462_compute_element_flux_honours_the_bergot_solution_space() {
    let mesh = bipyramid();
    let h1 = H1Space::with_pyramid_basis(mesh.clone(), 2, PyramidBasisType::Bergot);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
    let coords = bergot_dof_coords(2);
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

/// End-to-end: `zz_estimator_mfem` on a Bergot pyramid space with an
/// exactly-representable (affine) field — every indicator vanishes up to
/// rounding.  Before D462 the estimator sampled Fuentes dof coordinates
/// against the Bergot `element_dofs` table.
#[test]
fn d462_zz_estimator_mfem_bergot_affine_total_error_below_1e_minus_12() {
    let mesh = bipyramid();
    let h1 = H1Space::with_pyramid_basis(mesh.clone(), 2, PyramidBasisType::Bergot);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };

    let ind = zz_estimator_mfem(&gf, &integrator);
    assert_eq!(ind.eta.len(), mesh.n_elements());
    assert!(
        ind.total_error < 1e-12,
        "ZZ total error {} must vanish for an exactly-representable (affine) field",
        ind.total_error
    );
}

/// Anti-degeneracy: on a field the Bergot p = 2 space does not represent
/// exactly — one that varies across the two cells — every indicator must be
/// strictly positive (the D365 lesson: a silently-zero estimator is a defect,
/// not a pass).
#[test]
fn d462_zz_estimator_mfem_bergot_is_not_degenerate_on_a_varying_field() {
    let mesh = bipyramid();
    let h1 = H1Space::with_pyramid_basis(mesh.clone(), 2, PyramidBasisType::Bergot);
    let u = |x: &[f64]| x[0].sin() * x[1] + x[2] * x[2];
    let gf = GridFunction::new(&h1, h1.interpolate(&u).as_slice().to_vec());
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };

    let ind = zz_estimator_mfem(&gf, &integrator);
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
