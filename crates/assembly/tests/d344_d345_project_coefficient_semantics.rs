//! D344 / D345 — the `project_*_coefficient` family must implement MFEM's
//! `GridFunction::ProjectCoefficient`, i.e. the **nodal interpolant**, not an
//! L² mass solve.
//!
//! MFEM's `GridFunction::ProjectCoefficient(coeff)` calls `FiniteElement::Project`
//! on the collection's element, which for a nodal space is `dof_i = f(x_i)` at
//! the element's DOF nodes (for H(div)/H(curl) the dual/nodal *interpolants*
//! `Project_RT` / `Project_ND`).  Only the interpolant commutes with the
//! discrete differential operators, which is why the mixed solves in ex24 land
//! on their exact-projection legs.
//!
//! * D344: `project_hcurl_coefficient{,_2d}` claimed `ProjectCoefficient` and
//!   solved `M u = b`.  Both helpers are **unused** in-tree (verified by
//!   `grep`), so the change is behaviour-neutral for every existing result.
//! * D345: the scalar `project_coefficient` (7 consumers — see
//!   `tmp/d343/EVIDENCE.md` §6) had the same defect; ex24 `-p 0` printed
//!   0.00222497 against C++ 0.0022257 for that reason
//!   (`tmp/d343/EVIDENCE.md` §2.5).
//!
//! The L² operators are kept where they are *documented*: `GridFunction::
//! from_projection` / `GridFunction::project_coefficient` (test 2 below).

use fem_assembly::postproc::grid_function::{
    project_coefficient, project_hcurl_coefficient, project_hcurl_coefficient_2d, GridFunction,
};
use fem_assembly::standard::{DomainSourceIntegrator, MassIntegrator, VectorMassIntegrator};
use fem_assembly::{Assembler, VectorAssembler};
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, H1Space};

/// Independent replica of the historical `M c = b` L² projection (never calls
/// the code under test).
fn l2_reference<S: FESpace>(space: &S, f: &(dyn Fn(&[f64]) -> f64 + Send + Sync), qo: u8) -> Vec<f64> {
    let m = Assembler::assemble_bilinear(space, &[&MassIntegrator { rho: 1.0 }], qo);
    let b = Assembler::assemble_linear(space, &[&DomainSourceIntegrator::new(f)], qo);
    let mut x = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-14, atol: 1e-30, max_iter: 10_000, ..Default::default() };
    solve_cg(&m, &b, &mut x, &cfg).expect("L2 reference solve");
    x
}

fn f_sin(x: &[f64]) -> f64 {
    (2.0 * x[0]).sin() * (3.0 * x[1]).cos()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(u, v)| (u - v).abs()).fold(0.0, f64::max)
}

/// D345: the free helper is `FESpace::interpolate` (MFEM's
/// `ProjectCoefficient`), and it is measurably different from the L² solve.
#[test]
fn d345_project_coefficient_is_the_nodal_interpolant() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    for order in 1..=3u8 {
        let space = H1Space::new(mesh.clone(), order);
        let qo = (2 * order + 1).max(3) as u8;
        let got = project_coefficient(&space, &f_sin, qo);
        let want_interp = space.interpolate(&f_sin).into_vec();
        let want_l2 = l2_reference(&space, &f_sin, qo);
        let d_interp = max_abs_diff(&got, &want_interp);
        let d_l2 = max_abs_diff(&got, &want_l2);
        eprintln!(
            "D345 P{order}: |helper - interpolate| = {d_interp:.3e}, |helper - L²| = {d_l2:.3e}"
        );
        assert_eq!(d_interp, 0.0, "order {order}: the helper must BE the interpolant");
        assert!(d_l2 > 1e-8, "order {order}: expected a measurable L² gap, got {d_l2:.3e}");
    }
}

/// The L² semantics survive where they are documented.
#[test]
fn d345_from_projection_keeps_the_l2_projection() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let order = 2u8;
    let space = H1Space::new(mesh.clone(), order);
    let qo = (2 * order + 1).max(3) as u8;
    let gf = GridFunction::from_projection(&space, &f_sin, qo);
    let want_l2 = l2_reference(&space, &f_sin, qo);
    let d = max_abs_diff(gf.dofs(), &want_l2);
    eprintln!("D345 from_projection vs L² reference: {d:.3e}");
    assert!(d < 1e-10, "from_projection must stay an L² projection ({d:.3e})");

    // ... and the in-place method likewise.
    let mut gf2 = GridFunction::new(&space, vec![0.0; space.n_dofs()]);
    gf2.project_coefficient(&f_sin, qo);
    let d2 = max_abs_diff(gf2.dofs(), &want_l2);
    assert!(d2 < 1e-10, "GridFunction::project_coefficient must stay L² ({d2:.3e})");
}

fn v_coeff(x: &[f64], out: &mut [f64]) {
    out[0] = (2.0 * x[0]).sin() * x[1];
    out[1] = x[0] * (3.0 * x[1]).cos();
}

/// D344 (2-D): `project_hcurl_coefficient_2d` is `HCurlSpace::interpolate_vector`
/// (`Project_ND`), and differs from the mass solve it used to perform.
#[test]
fn d344_hcurl_2d_is_project_nd() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    for order in 1..=2u8 {
        let nd = HCurlSpace::new(mesh.clone(), order);
        let qo = (2 * order + 1).max(3) as u8;
        let got = project_hcurl_coefficient_2d(&nd, &v_coeff, qo);
        let want = nd
            .interpolate_vector(&|x: &[f64]| {
                let mut v = vec![0.0; 2];
                v_coeff(x, &mut v);
                v
            })
            .into_vec();
        assert_eq!(got, want, "order {order}: must delegate to interpolate_vector");

        // The historical L² operator (replicated here) is measurably different.
        let m = VectorAssembler::assemble_bilinear(&nd, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let rhs = VectorAssembler::assemble_linear(
            &nd,
            &[&fem_assembly::standard::VectorDomainLFIntegrator {
                f: fem_assembly::coefficient::FnVectorCoeff(Box::new(|x: &[f64], out: &mut [f64]| {
                    v_coeff(x, out)
                })),
            }],
            qo,
        );
        let mut l2 = vec![0.0; nd.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 5000, verbose: false, ..Default::default() };
        solve_cg(&m, &rhs, &mut l2, &cfg).expect("hcurl L2 reference");
        let d = max_abs_diff(&got, &l2);
        eprintln!("D344 2-D P{order}: |interpolant - L²| = {d:.3e}");
        assert!(d > 1e-9, "order {order}: the operator change must be observable");
    }
}

/// D344 (3-D): the same for the 3-D helper (hex, orders 1..=2).
#[test]
fn d344_hcurl_3d_is_project_nd() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    for order in 1..=2u8 {
        let nd = HCurlSpace::new(mesh.clone(), order);
        let qo = (2 * order + 1).max(3) as u8;
        let got = project_hcurl_coefficient(&nd, &v_coeff, qo);
        let want = nd
            .interpolate_vector(&|x: &[f64]| {
                let mut v = vec![0.0; 3];
                v_coeff(x, &mut v);
                v
            })
            .into_vec();
        assert_eq!(got, want, "order {order}: must delegate to interpolate_vector");
        eprintln!("D344 3-D P{order}: delegating ({} dofs)", got.len());
    }
    // The prism/pyramid families are served at order 1 only (ND1 is edge-only).
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    let nd = HCurlSpace::new(mesh.clone(), 1);
    assert_eq!(project_hcurl_coefficient(&nd, &v_coeff, 3).len(), nd.n_dofs());
    let _ = &mut mesh;
}

/// The mesh used above must really be a hex mesh (guards against a fixture
/// change silently turning these tests into simplex runs).
#[test]
fn d344_fixture_is_a_hex_mesh() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    assert_eq!(mesh.n_elements(), 1);
    assert_eq!(mesh.element_type(0), fem_mesh::ElementType::Hex8);
}
