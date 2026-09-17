//! D265: the `postproc/postprocess.rs` kernels on **hexahedral** meshes.
//!
//! Before D265 the scalar table (`ref_elem_vol`) had no Hex8 arm — every
//! kernel below panicked on a hex — and the curl/div kernels evaluated the
//! corner-difference "simplex" Jacobian, whose hex corner triple (nodes
//! 0,1,2 — all on the bottom face) is coplanar, i.e. degenerate.  After D265:
//!
//! * the scalar table serves `HexQk` ([-1,1]³ GLL slots, MFEM
//!   `H1_HexahedronElement` order), matching `grid_function.rs`;
//! * all kernels route quad/hex elements through the isoparametric geometry
//!   (`geo_ref_elem_from_mesh` + `isoparametric_jacobian`, the D242/D250
//!   recipe) and evaluate at the hex frame origin instead of the simplex
//!   centroid (1/4,1/4,1/4);
//! * `recover_gradient_nodal` weights hexes by the true volume ∫|det J| and
//!   averages over all 8 corners.
//!
//! Acceptance: exact analytic values on affine and warped hex meshes
//! (= the C++ / analytic truth the MFEM-parity bases guarantee).

use fem_assembly::postprocess::{
    compute_element_curl, compute_element_divergence, compute_element_gradients,
    compute_h1_error, recover_gradient_nodal,
};
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, H1Space, HDivSpace};

const TOL: f64 = 1e-12;

fn u_lin3(x: &[f64]) -> f64 {
    1.0 + 2.0 * x[0] - 3.0 * x[1] + 5.0 * x[2]
}
fn grad_lin3(x: &[f64]) -> Vec<f64> {
    let _ = x;
    vec![2.0, -3.0, 5.0]
}
fn u_hexq2(x: &[f64]) -> f64 {
    1.0 + x[0] * x[0] - 3.0 * x[1] * x[2]
}
fn grad_hexq2(x: &[f64]) -> Vec<f64> {
    vec![2.0 * x[0], -3.0 * x[2], -3.0 * x[1]]
}

fn hex_box() -> Mesh<3> {
    Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false)
}

/// Single hexahedron with a warped (trilinear, non-affine) geometry.
fn hex_warped() -> Mesh<3> {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    for (nv, d) in [
        (6usize, [0.25, -0.2, 0.3]),
        (5, [0.1, 0.05, 0.2]),
        (7, [-0.05, 0.15, -0.1]),
    ] {
        for c in 0..3 {
            mesh.coords[nv * 3 + c] += d[c];
        }
    }
    mesh
}

#[test]
fn d265_hex_element_gradients_linear_exact() {
    let mesh = hex_box();
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&u_lin3).as_slice().to_vec();
    let grads = compute_element_gradients(&space, &dofs);
    assert_eq!(grads.len(), mesh.n_elements());
    for (e, g) in grads.iter().enumerate() {
        assert!(
            (g[0] - 2.0).abs() < TOL && (g[1] + 3.0).abs() < TOL && (g[2] - 5.0).abs() < TOL,
            "elem {e}: ∇u = {:?}, expected [2, -3, 5]",
            g
        );
    }
}

#[test]
fn d265_hex_warped_element_gradients_linear_exact() {
    // The trilinear map reproduces affine fields exactly; only the
    // isoparametric geometry captures the point-dependent Jacobian (the
    // pre-D265 corner difference cannot — and its hex corner triple is
    // coplanar besides).
    let mesh = hex_warped();
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&u_lin3).as_slice().to_vec();
    let grads = compute_element_gradients(&space, &dofs);
    for (e, g) in grads.iter().enumerate() {
        assert!(
            (g[0] - 2.0).abs() < TOL && (g[1] + 3.0).abs() < TOL && (g[2] - 5.0).abs() < TOL,
            "elem {e}: warped ∇u = {:?}, expected [2, -3, 5]",
            g
        );
    }
}

#[test]
fn d265_hex_h1_seminorm_exact_fields_zero() {
    // P1 affine field.
    let mesh = hex_box();
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&u_lin3).as_slice().to_vec();
    let err = compute_h1_error(&space, &dofs, &grad_lin3, 5);
    assert!(err < TOL, "P1 affine: H1 semi = {err:e}");

    // P2 quadratic field.
    let space2 = H1Space::new(mesh.clone(), 2);
    let dofs2 = space2.interpolate(&u_hexq2).as_slice().to_vec();
    let err2 = compute_h1_error(&space2, &dofs2, &grad_hexq2, 7);
    assert!(err2 < TOL, "P2 quadratic: H1 semi = {err2:e}");
}

#[test]
fn d265_hex_recover_gradient_nodal_exact() {
    for (tag, mesh) in [("box", hex_box()), ("warped", hex_warped())] {
        let space = H1Space::new(mesh.clone(), 1);
        let dofs = space.interpolate(&u_lin3).as_slice().to_vec();
        let grad = recover_gradient_nodal(&space, &dofs);
        assert_eq!(grad.len(), 3);
        for node in 0..mesh.n_nodes() {
            assert!(
                (grad[0][node] - 2.0).abs() < TOL
                    && (grad[1][node] + 3.0).abs() < TOL
                    && (grad[2][node] - 5.0).abs() < TOL,
                "{tag} node {node}: recovered ∇u = {:?}, expected [2, -3, 5]",
                [grad[0][node], grad[1][node], grad[2][node]]
            );
        }
    }
}

#[test]
fn d265_hex_curl_nd1_exact() {
    // A = (0, -z, y) = (1,0,0)×x lies in every lowest-order Nédélec space and
    // has curl A = (2, 0, 0).
    let mesh = hex_box();
    let space = HCurlSpace::new(mesh.clone(), 1);
    let a = space.interpolate_vector(&|x| vec![0.0, -x[2], x[1]]);
    let curls = compute_element_curl(&space, a.as_slice());
    assert_eq!(curls.len(), mesh.n_elements());
    for (e, c) in curls.iter().enumerate() {
        assert_eq!(c.len(), 3, "3-D curl must have 3 components");
        assert!(
            (c[0] - 2.0).abs() < TOL && c[1].abs() < TOL && c[2].abs() < TOL,
            "elem {e}: curl = {:?}, expected [2, 0, 0]",
            c
        );
    }
}

#[test]
fn d265_hex_div_rt0_exact() {
    // The hex RT0 x-component spans {a + b·ξ}, so the physical field
    // F = (x − 1/2, 0, 0) lies in the space on every axis-aligned element and
    // has div F = 1.  Note (x, 0, 0) is *not* in RT0 — the constant offset
    // matters.
    //
    // The coefficients are built by an L² mass projection through the
    // assembler, whose hex H(div) basis is the MFEM-default GaussLegendre
    // variant (D245) — the same variant [`compute_element_divergence`]
    // evaluates, so the pairing is normalization-consistent.  (The
    // `interpolate_vector` dual is still IntegratedGLL-normalized — hdiv.rs
    // `fill_dual_matrix`, deliberately untouched in D236 §6.4 — and mixing
    // those dofs with the GL shapes rescales the result; tracked as D289.)
    use fem_assembly::standard::VectorMassIntegrator;
    use fem_assembly::vector_assembler::VectorAssembler;
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};

    struct VectorRhs(fn(&[f64]) -> Vec<f64>);
    impl VectorLinearIntegrator for VectorRhs {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
            let ue = (self.0)(qp.x_phys);
            for i in 0..qp.n_dofs {
                let mut dot = 0.0;
                for c in 0..qp.dim {
                    dot += qp.phi_vec[i * qp.dim + c] * ue[c];
                }
                f_elem[i] += qp.weight * dot;
            }
        }
    }

    let mesh = hex_box();
    let space = HDivSpace::new(mesh.clone(), 0);
    let f_field: fn(&[f64]) -> Vec<f64> = |x| vec![x[0] - 0.5, 0.0, 0.0];

    let mass = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: 1.0 }], 6);
    let rhs = VectorAssembler::assemble_linear(&space, &[&VectorRhs(f_field)], 6);
    let mut dofs = vec![0.0_f64; space.n_dofs()];
    fem_solver::solve_pcg_jacobi(
        &mass,
        &rhs,
        &mut dofs,
        &fem_solver::SolverConfig {
            rtol: 1e-13,
            max_iter: 10_000,
            verbose: false,
            ..fem_solver::SolverConfig::default()
        },
    )
    .unwrap();

    let divs = compute_element_divergence(&space, &dofs);
    assert_eq!(divs.len(), mesh.n_elements());
    for (e, &d) in divs.iter().enumerate() {
        assert!(
            (d - 1.0).abs() < TOL,
            "elem {e}: div = {d}, expected 1"
        );
    }
}

