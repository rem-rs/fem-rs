//! D793-1 — `postproc::flux_recovery::geom_jacobian`'s **simplex arm** built an
//! affine map from the element's *vertex* coordinates, so on a curved
//! triangle/tetrahedron `compute_element_flux` / `compute_flux_energy` still
//! differentiated against the straight corner geometry while the field itself
//! was placed with the curved one.
//!
//! The fix routes that arm through `fem_mesh::transformation::element_jacobian_at`
//! — family-aware since D787 — exactly like the hex/prism/pyramid arms of the
//! same function already did.
//!
//! # Fixture and truth
//!
//! Same fixture as `crates/mesh/tests/d787_consumer_l2_error.rs`: the unit
//! cell + `set_curvature(2)` + D319 quadratic warp applied to every geometry
//! node and vertex.  MFEM's side is `tmp/d793/d793_probe.cpp` (a copy of the
//! D787 probe with the flux/`ComputeLpNorm` arms added), built and run in WSL:
//!
//! ```text
//! g++ -std=c++17 -O2 -I$HOME/mfem410_ser d793_probe.cpp \
//!     $HOME/mfem410_ser/libmfem.a -o $HOME/work/d793/d793_probe
//! $HOME/work/d793/d793_probe tet 2 > tmp/d793/mfem_d793_tet_p2.txt
//! $HOME/work/d793/d793_probe tri 2 > tmp/d793/mfem_d793_tri_p2.txt
//! ```
//!
//! The probe calls MFEM's own
//! `DiffusionIntegrator::ComputeElementFlux(el(1), Trans, u_local, fluxelem =
//! el(1), flux, with_coef = false)` — the estimator's H¹(1) flux space, the
//! element's own `ElementTransformation` (curved) — and
//! `ComputeFluxEnergy(el, Trans, diff)` on a constant difference vector.  The
//! H¹(1) field is the interpolant of the affine `u = 1 + x + 2y (+ 3z)`, whose
//! `∇u_h` is *not* `(1,2,3)` on a curved cell (the reference gradient maps
//! through `J^{-T}`), so the comparison has full sensitivity to the geometry
//! map.  Layout: MFEM stores the flux component-major (`flux(nd*j+i)`), the
//! fem-rs API dof-major (`flux[i*dim+d]`); the dumped rows are dof-major.

use fem_assembly::postproc::flux_recovery::FluxRecovery;
use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// MFEM 4.10 `mfem_d793_tet_p2.txt`: curved tet P2, H¹(1) flux space,
/// `u = 1 + x + 2y + 3z`, flux rows in dof-major order.
const MFEM_TET_P2_FLUX: [[f64; 3]; 4] = [
    [1.2, 2.0, 3.0],
    [0.85714285714285754, 1.708542713567839, 2.9145728643216078],
    [0.90909090909090917, 2.0, 2.9090909090909092],
    [1.1055276381909547, 1.8894472361809043, 2.9999999999999996],
];
/// MFEM 4.10: `ComputeFluxEnergy(el, Trans, diff)` with `diff = (1,2,3)` per
/// flux dof (dof-major), `κ = 1`.
const MFEM_TET_P2_ENERGY_DIFF: f64 = 2.561777807341997;

/// MFEM 4.10 `mfem_d793_tri_p2.txt`: curved tri P2, `u = 1 + x + 2y`.
const MFEM_TRI_P2_FLUX: [[f64; 2]; 3] = [
    [1.2, 2.0],
    [0.85714285714285765, 1.8231292517006807],
    [1.0000000000000002, 2.0],
];
/// MFEM 4.10: same energy with `diff = (1,2)` per flux dof.
const MFEM_TRI_P2_ENERGY_DIFF: f64 = 2.9666666666666677;

const TOL: f64 = 1e-13;

fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn g2(x: [f64; 2]) -> [f64; 2] {
    [
        x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1],
        x[1] + 0.05 * x[0] * x[1],
    ]
}

/// Warp every geometry node and vertex — the D787 fixture.
fn warp_mesh<const D: usize>(mesh: &mut Mesh<D>, f: fn([f64; D]) -> [f64; D]) {
    let n_geom = mesh.geometry.as_ref().expect("curved mesh keeps a table").coords.len() / D;
    {
        let geo = mesh.geometry.as_mut().expect("curved table");
        for k in 0..n_geom {
            let mut x = [0.0_f64; D];
            x.copy_from_slice(&geo.coords[k * D..(k + 1) * D]);
            let y = f(x);
            geo.coords[k * D..(k + 1) * D].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; D];
        x.copy_from_slice(&mesh.coords[k * D..(k + 1) * D]);
        let y = f(x);
        mesh.coords[k * D..(k + 1) * D].copy_from_slice(&y);
    }
}

fn curved_unit_tet() -> Mesh<3> {
    let mut m = Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Tet4,
        vec![0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        vec![1; 4],
        ElementType::Tri3,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g3);
    m
}

fn curved_unit_tri() -> Mesh<2> {
    let mut m = Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 0],
        vec![1; 3],
        ElementType::Line2,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g2);
    m
}

#[test]
fn d793_flux_recovery_curved_tet_matches_mfem() {
    let mesh = curved_unit_tet();
    assert_eq!(mesh.geom_order(), 2);
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2]);
    let dofs = dofs.as_slice();
    let flux_dof_coords = fem_space::ref_elem::h1_simplex_slots(ElementType::Tet4, 1).dof_coords();
    assert_eq!(flux_dof_coords.len(), 4);

    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
    let flux = integrator.compute_element_flux(&mesh, &space, 0, dofs, &flux_dof_coords);
    assert_eq!(flux.len(), 4 * 3);
    let mut worst = 0.0_f64;
    for i in 0..4 {
        for d in 0..3 {
            let got = flux[i * 3 + d];
            let want = MFEM_TET_P2_FLUX[i][d];
            worst = worst.max((got - want).abs());
            assert!(
                (got - want).abs() <= TOL * want.abs().max(1.0),
                "curved tet P2 flux dof {i} component {d}: got {got:.17e}, \
                 MFEM {want:.17e} — geometry map mismatch"
            );
        }
    }
    eprintln!("D793-1 curved tet P2 flux: worst |Δ| = {worst:.3e} (MFEM 4.10)");

    // MFEM's `ComputeFluxEnergy` of the constant difference vector (1,2,3).
    let diff: Vec<f64> = (0..4).flat_map(|_| [1.0, 2.0, 3.0]).collect();
    let energy = integrator.compute_flux_energy(&mesh, 0, &diff);
    eprintln!("D793-1 curved tet P2 energy((1,2,3)) = {energy:.17e}");
    assert!(
        (energy - MFEM_TET_P2_ENERGY_DIFF).abs() <= 1e-12,
        "curved tet P2 flux energy: got {energy:.17e}, MFEM {MFEM_TET_P2_ENERGY_DIFF:.17e}"
    );
}

#[test]
fn d793_flux_recovery_curved_tri_matches_mfem() {
    let mesh = curved_unit_tri();
    assert_eq!(mesh.geom_order(), 2);
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&|x: &[f64]| 1.0 + x[0] + 2.0 * x[1]);
    let dofs = dofs.as_slice();
    let flux_dof_coords = fem_space::ref_elem::h1_simplex_slots(ElementType::Tri3, 1).dof_coords();
    assert_eq!(flux_dof_coords.len(), 3);

    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
    let flux = integrator.compute_element_flux(&mesh, &space, 0, dofs, &flux_dof_coords);
    assert_eq!(flux.len(), 3 * 2);
    for i in 0..3 {
        for d in 0..2 {
            let got = flux[i * 2 + d];
            let want = MFEM_TRI_P2_FLUX[i][d];
            assert!(
                (got - want).abs() <= TOL * want.abs().max(1.0),
                "curved tri P2 flux dof {i} component {d}: got {got:.17e}, \
                 MFEM {want:.17e} — geometry map mismatch"
            );
        }
    }
    let diff: Vec<f64> = (0..3).flat_map(|_| [1.0, 2.0]).collect();
    let energy = integrator.compute_flux_energy(&mesh, 0, &diff);
    eprintln!("D793-1 curved tri P2 energy((1,2)) = {energy:.17e}");
    assert!(
        (energy - MFEM_TRI_P2_ENERGY_DIFF).abs() <= 1e-12,
        "curved tri P2 flux energy: got {energy:.17e}, MFEM {MFEM_TRI_P2_ENERGY_DIFF:.17e}"
    );
}
