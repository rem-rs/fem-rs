//! D39 regression at miniapp level: the statically condensed (`-sc`) solve of
//! the real `DpgWeakForm` must reproduce the uncondensed solve.
//!
//! The weak forms built here are 1:1 with `miniapps/dpg/dpg_poisson_2d.rs`
//! (MFEM `miniapps/dpg/diffusion.cpp`) and with the complex Helmholtz problem
//! of `miniapps/dpg/dpg_acoustics_2d.rs`, including the essential-BC dof set
//! and the PCG/block-Gauss–Seidel solve.
//!
//! History: before D39 the real static-condensation path was inconsistent with
//! the uncondensed one because (a) the essential-BC reduction never wrote the
//! prescribed values back into the reduced right-hand side (MFEM
//! `S->PartMult`) and (b) `lu_solve`'s forward substitution used the raw
//! permuted right-hand side instead of the already-substituted components,
//! which silently corrupted every solve with more than two eliminated dofs per
//! element (i.e. every trial order `p >= 2`).

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgDomainLFIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedVectorGradientIntegrator,
    DpgMixedVectorWeakDivergenceIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTraceIntegrator, DpgTVectorFEMassIntegrator, DpgVectorFEDivergenceIntegrator,
    DpgVectorFEMassIntegrator,
};
use fem_assembly::dpg_weakform::{DpgBlockGs, DpgSystem, DpgWeakForm};
use fem_assembly::ComplexDPGWeakForm;
use fem_linalg::CsrMatrix;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::{solve_pcg_operator_precond, SolverConfig};

/// `-(σ, τ)` adapter — MFEM `TransposeIntegrator(VectorFEMassIntegrator(-1))`
/// (copied from the miniapp; the integrator is not part of the public API).
struct NegVectorMass;
impl fem_assembly::dpg::dpg_integrators::DpgBilinear2 for NegVectorMass {
    fn assemble2(
        &self,
        ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
        trial: &fem_assembly::dpg::dpg_basis::VolVals,
        test: &fem_assembly::dpg::dpg_basis::VolVals,
        m: &mut [f64],
    ) {
        let d = ctx.dim;
        let nt = test.n_scalar;
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        for k in 0..nt {
            for c in 0..d {
                for j in 0..nsc {
                    m[k * nc + c * nsc + j] -= ctx.w * test.phi[k * d + c] * trial.phi[j];
                }
            }
        }
    }
}

const PI: f64 = std::f64::consts::PI;

/// Ultraweak DPG Poisson form, 1:1 with `miniapps/dpg/dpg_poisson_2d.rs`.
fn build_poisson_2d(mesh: &Mesh<2>, p: u8) -> (DpgWeakForm<Mesh<2>>, [usize; 4]) {
    let test_order = p + 1; // delta_order = 1
    let mut a: DpgWeakForm<Mesh<2>> = DpgWeakForm::new(mesh.clone());
    let u = a.add_trial_scalar_space(p - 1);
    let sig = a.add_trial_vector_space(p - 1, 2);
    let hatu = a.add_trial_trace_space_h1(p);
    let hatsig = a.add_trial_trace_space(p - 1);
    let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
    let v = a.add_test_space(VolKind::Scalar, test_order);

    a.add_trial_integrator(
        Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
        u,
        tau,
    );
    a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
    a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
    a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
    a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);

    a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
    a.add_test_integrator(Box::new(DpgVectorFEMassIntegrator { q: 1.0 }), tau, tau);
    a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
    a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
    a.add_domain_lf_integrator(
        Box::new(DpgDomainLFIntegrator {
            f: |x: &[f64]| 2.0 * PI * PI * (PI * (x[0] + x[1])).sin(),
        }),
        v,
    );
    (a, [u, sig, hatu, hatsig])
}

/// Miniapp essential dofs: every dof of every boundary face of `hatu`, set to
/// the manufactured solution `sin(π(x+y))`.
fn poisson_bcs(a: &DpgWeakForm<Mesh<2>>, hatu: usize) -> (Vec<usize>, Vec<f64>) {
    let base = a.trial_offsets()[hatu];
    let sk = a.skeleton(hatu);
    let mut ess = Vec::new();
    let mut x = vec![0.0_f64; a.size()];
    for f in 0..sk.n_faces() {
        if !sk.is_boundary_face(f) {
            continue;
        }
        // `face_dof_list` names the shared vertex dofs of the H1-trace mode
        // (the contiguous `face_dofs` range does not).
        for (k, &d) in sk.face_dof_list(f).iter().enumerate() {
            ess.push(base + d);
            let pt = a.face_dof_point(&sk, f, k);
            x[base + d] = (PI * (pt[0] + pt[1])).sin();
        }
    }
    (ess, x)
}

fn solve_pcg(sys: &DpgSystem, b: &[f64], x0: &[f64], rtol: f64) -> Vec<f64> {
    let offsets = match sys {
        DpgSystem::Full { offsets, .. } | DpgSystem::Condensed { offsets, .. } => offsets.clone(),
    };
    let precond = DpgBlockGs::new(sys, &offsets);
    let mat: &CsrMatrix<f64> = sys.matrix();
    let n = mat.nrows;
    let mut xs = x0.to_vec();
    let apply = |x: &[f64], y: &mut [f64]| mat.spmv(x, y);
    let pc = move |r: &[f64], z: &mut [f64]| precond.apply(r, z);
    let cfg = SolverConfig {
        rtol,
        max_iter: 4000,
        ..SolverConfig::default()
    };
    let res = solve_pcg_operator_precond(n, apply, b, &mut xs, pc, &cfg).expect("PCG solve failed");
    assert!(
        res.iterations < cfg.max_iter,
        "PCG did not converge ({} iterations)",
        res.iterations
    );
    xs
}

/// PCG on the doubled real operator of a `ComplexDpgSystem` — the miniapp
/// (`dpg_acoustics_2d`) preconditioner layout: one block per (half, block).
fn solve_pcg_complex(
    sys: &fem_assembly::ComplexDpgSystem,
    b: &[f64],
    x0: &[f64],
    rtol: f64,
) -> Vec<f64> {
    let big = sys.to_real_block_csr();
    let half = sys.n_complex();
    let nb = sys.offsets.len() - 1;
    let mut off2: Vec<usize> = sys.offsets.clone();
    for k in 1..=nb {
        off2.push(half + sys.offsets[k]);
    }
    let precond = DpgBlockGs::from_matrix(&big, &off2);
    let n = big.nrows;
    let mut xs = x0.to_vec();
    let apply = |x: &[f64], y: &mut [f64]| big.spmv(x, y);
    let pc = move |r: &[f64], z: &mut [f64]| precond.apply(r, z);
    let cfg = SolverConfig {
        rtol,
        max_iter: 4000,
        ..SolverConfig::default()
    };
    let res = solve_pcg_operator_precond(n, apply, b, &mut xs, pc, &cfg).expect("PCG solve failed");
    assert!(
        res.iterations < cfg.max_iter,
        "complex PCG did not converge ({} iterations)",
        res.iterations
    );
    xs
}

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// Miniapp level: `dpg_poisson_2d -o p -sc` == `dpg_poisson_2d -o p`.
#[test]
fn poisson_2d_sc_matches_uncondensed() {
    for p in [1u8, 2, 3] {
        let mesh = Mesh::<2>::unit_square_quad(2);

        let (mut a1, ids1) = build_poisson_2d(&mesh, p);
        a1.assemble();
        let (ess, x_full) = poisson_bcs(&a1, ids1[2]);
        let (sys1, xs1, b1) = a1.form_linear_system(&ess, &x_full, false);
        let x1 = solve_pcg(&sys1, &b1, &xs1, 1e-13);
        let sol1 = a1.recover_fem_solution(&x1);

        let (mut a2, ids2) = build_poisson_2d(&mesh, p);
        a2.enable_static_condensation();
        a2.assemble();
        let (ess2, x_full2) = poisson_bcs(&a2, ids2[2]);
        assert_eq!(ess, ess2);
        let (sys2, xs2, b2) = a2.form_linear_system(&ess2, &x_full2, false);
        assert!(
            sys2.size() < sys1.size(),
            "p={p}: the condensed system must be smaller"
        );
        let x2 = solve_pcg(&sys2, &b2, &xs2, 1e-13);
        let sol2 = a2.recover_fem_solution(&x2);

        let rel = max_rel_diff(&sol1, &sol2);
        eprintln!(
            "poisson_2d p={p}: condenser size {} vs full {}, -sc vs uncondensed \
             relative solution difference {rel:.3e}",
            sys2.size(),
            sys1.size()
        );
        assert!(
            rel <= 1e-10,
            "p={p}: static condensation != uncondensed, relative solution \
             difference {rel:.3e}"
        );
    }
}

/// Complex path (reference implementation, `dpg_acoustics_2d` shape):
/// `-sc` == uncondensed as well.
#[test]
fn complex_helmholtz_2d_sc_matches_uncondensed() {
    let omega = 2.0 * PI; // rnum = 1
    let mesh = Mesh::<2>::unit_square_quad(2);
    let p = 1u8;
    let test_order = p + 1;

    let build = |mesh: &Mesh<2>| -> (ComplexDPGWeakForm<Mesh<2>>, [usize; 4]) {
        let mut a: ComplexDPGWeakForm<Mesh<2>> = ComplexDPGWeakForm::new(mesh.clone());
        let ps = a.add_trial_scalar_space(p - 1);
        let us = a.add_trial_vector_space(p - 1, 2);
        // p̂ ∈ H^{1/2}: vertex-continuous H1 trace (MFEM H1_Trace_FECollection)
        let hatp = a.add_trial_trace_space_h1(p);
        // û ∈ H^{-1/2}: per-edge discontinuous RT trace
        let hatu = a.add_trial_trace_space(p - 1);
        let q = a.add_test_space(VolKind::Scalar, test_order);
        let v = a.add_test_space(VolKind::HDiv, test_order - 1);
        a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: omega })), ps, q);
        a.add_trial_integrator(Some(Box::new(DpgTGradientIntegrator { q: -1.0 })), None, us, q);
        a.add_trial_integrator(
            Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 })),
            None,
            ps,
            v,
        );
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: omega })),
            us,
            v,
        );
        a.add_trace_integrator(Some(Box::new(DpgNormalTraceIntegrator)), None, hatp, v);
        a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hatu, q);
        // adjoint graph norm
        a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, q, q);
        a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q, q);
        a.add_test_integrator(Some(Box::new(DpgDivDivIntegrator { q: 1.0 })), None, v, v);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, v, v);
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorGradientIntegrator {
                q: vec![vec![-omega, 0.0], vec![0.0, -omega]],
            })),
            v,
            q,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator {
                q: vec![vec![-omega, 0.0], vec![0.0, -omega]],
            })),
            q,
            v,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: omega * omega })),
            None,
            v,
            v,
        );
        a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: omega * omega })), None, q, q);
        a.add_test_integrator(
            None,
            Some(Box::new(DpgVectorFEDivergenceIntegrator { q: -omega })),
            q,
            v,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: -omega })),
            v,
            q,
        );
        (a, [ps, us, hatp, hatu])
    };

    // plane wave p = exp(i β (x+y)), β = ω/√2 — p̂ = p on the boundary.
    let beta = omega / 2.0f64.sqrt();
    let pbc = |x: f64, y: f64| -> (f64, f64) {
        let t = beta * (x + y);
        (t.cos(), t.sin())
    };
    let bcs = |a: &ComplexDPGWeakForm<Mesh<2>>, hatp: usize| -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        let base = a.trial_offsets()[hatp];
        let sk = a.skeleton(hatp);
        let mut ess = Vec::new();
        let mut xr = vec![0.0_f64; a.size()];
        let mut xi = vec![0.0_f64; a.size()];
        for f in 0..sk.n_faces() {
            if !sk.is_boundary_face(f) {
                continue;
            }
            for (k, &dof) in sk.face_dof_list(f).iter().enumerate() {
                ess.push(base + dof);
                let nodes = sk.face_nodes(f);
                // face dof point (miniapp `face_point` convention: node k at
                // parameter k/p along the face)
                let p_param = k as f64 / p as f64;
                let (n0, n1) = (nodes[0], nodes[1]);
                let c0 = a.mesh().node_coords(n0);
                let c1 = a.mesh().node_coords(n1);
                let x = (1.0 - p_param) * c0[0] + p_param * c1[0];
                let y = (1.0 - p_param) * c0[1] + p_param * c1[1];
                let (pr, pi) = pbc(x, y);
                xr[base + dof] = pr;
                xi[base + dof] = pi;
            }
        }
        (ess, xr, xi)
    };

    let (mut a1, ids1) = build(&mesh);
    a1.assemble();
    let (ess, xr, xi) = bcs(&a1, ids1[2]);
    let (sys1, xs1, b1) = a1.form_linear_system(&ess, &xr, &xi);
    let x1 = solve_pcg_complex(&sys1, &b1, &xs1, 1e-13);
    let (r1, i1) = a1.recover_fem_solution(&x1);

    let (mut a2, ids2) = build(&mesh);
    a2.enable_static_condensation();
    a2.assemble();
    let (ess2, xr2, xi2) = bcs(&a2, ids2[2]);
    assert_eq!(ess, ess2);
    let (sys2, xs2, b2) = a2.form_linear_system(&ess2, &xr2, &xi2);
    assert!(sys2.n_complex() < sys1.n_complex());
    let x2 = solve_pcg_complex(&sys2, &b2, &xs2, 1e-13);
    let (r2, i2) = a2.recover_fem_solution(&x2);

    let mut scale = 0.0f64;
    for v in r1.iter().chain(i1.iter()) {
        scale = scale.max(v.abs());
    }
    let mut worst = 0.0f64;
    for (x, y) in r1.iter().zip(&r2).chain(i1.iter().zip(&i2)) {
        worst = worst.max((x - y).abs());
    }
    let rel = worst / scale;
    eprintln!(
        "complex_helmholtz_2d: -sc vs uncondensed relative solution difference {rel:.3e}"
    );
    assert!(
        rel <= 1e-10,
        "complex -sc != uncondensed, relative solution difference {rel:.3e}"
    );
}
