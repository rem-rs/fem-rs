//! d104 regression pins for the ultraweak DPG convection-diffusion miniapp
//! (`miniapps/dpg/pconvection_diffusion.rs`, MFEM 4.10
//! `miniapps/dpg/pconvection-diffusion.cpp`).
//!
//! Every constant below is the **C++ MPI oracle** stdout value
//! (`mpirun -np 1 ~/pconv_cpp -no-vis …`, build line in
//! `tmp/d104pconv/REPORT.md`), reproduced by the fem-rs miniapp to all printed
//! digits (`--ranks 1`):
//!
//! ```text
//! -theta 0.0:            113  1.033e+00  9.290e-01   →   417  5.172e-01  4.817e-01  (-1.06 / -1.01)
//! -theta 0.0 -eps 1e-2:  113  2.523e-01  3.101e-01
//! ```
//!
//! The pins cover the pieces this round added: the vector-coefficient
//! `-(βu,∇v)` trial block, the tensor `ββᵀ` test-norm block and the L²(P0)
//! weights `c₁, c₂` (`setup_test_norm_coeffs`), plus the residual used for AMR
//! marking.  The PCG iteration counts are NOT pinned (D963: the C++
//! preconditions with Hypre AMG/AMS, fem-rs with block symmetric GS).

use std::sync::{Arc, Mutex};

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDiffusionIntegrator, DpgDiffusionSpatialIntegrator, DpgDivDivIntegrator,
    DpgDomainLFIntegrator, DpgMassSpatialIntegrator, DpgMixedScalarWeakGradientIntegrator,
    DpgMixedScalarWeakDivergenceSpatialIntegrator, DpgNormalTraceIntegrator,
    DpgTGradientIntegrator, DpgTraceIntegrator, DpgTVectorFEMassIntegrator,
    DpgVectorFEMassScalarSpatialIntegrator,
};
use fem_assembly::dpg_weakform::DpgBlockGs;
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_dpg_weakform::ParDpgWeakForm;
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_solve_pcg_precond;
use fem_parallel::WorkerConfig;
use fem_solver::SolverConfig;

const PI: f64 = std::f64::consts::PI;

fn beta_function(beta_const: &[f64], _x: &[f64], out: &mut [f64]) {
    out[0] = beta_const[0];
    out[1] = beta_const[1];
}

fn exact_u(x: &[f64]) -> f64 {
    (PI * (x[0] + x[1])).sin()
}

fn exact_gradu(x: &[f64]) -> [f64; 2] {
    let a = PI * (x[0] + x[1]);
    [PI * a.cos(), PI * a.cos()]
}

fn exact_sigma(x: &[f64], eps: f64) -> [f64; 2] {
    let g = exact_gradu(x);
    [eps * g[0], eps * g[1]]
}

fn exact_laplacian_u(x: &[f64]) -> f64 {
    -PI * PI * (PI * (x[0] + x[1])).sin() * 2.0
}

fn f_exact(x: &[f64], eps: f64, beta_const: &[f64]) -> f64 {
    let du = exact_gradu(x);
    -eps * exact_laplacian_u(x) + beta_const[0] * du[0] + beta_const[1] * du[1]
}

/// C++ `std::scientific` with 3 digits (`1.033e+00`).
fn cpp_sci3(v: f64) -> String {
    if v == 0.0 {
        return "0.000e+00".to_string();
    }
    let neg = v < 0.0;
    let a = v.abs();
    let mut exp = a.log10().floor() as i32;
    let mut mant = a / 10f64.powi(exp);
    if mant >= 10.0 {
        mant /= 10.0;
        exp += 1;
    }
    if mant < 1.0 {
        mant *= 10.0;
        exp -= 1;
    }
    let mut s = format!("{mant:.3}");
    if s.parse::<f64>().unwrap_or(mant) >= 10.0 {
        s = format!("{:.3}", mant / 10.0);
        exp += 1;
    }
    format!(
        "{}{}e{}{:02}",
        if neg { "-" } else { "" },
        s,
        if exp < 0 { "-" } else { "+" },
        exp.abs()
    )
}

/// MFEM `Mesh::GetElementVolume` (`IntRules.Get(geom, OrderJ())` — the
/// one-point rule for straight-sided elements, exact on the affine
/// inline-quad family).
fn element_volume(mesh: &Mesh<2>, e: u32) -> f64 {
    use fem_assembly::dpg::dpg_basis::vol_quadrature;
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};

    let (qpts, qwts) = vol_quadrature(mesh.element_type(e), 1);
    let mut vol = 0.0_f64;
    for (q, xi) in qpts.iter().enumerate() {
        let geo = geo_ref_elem_from_mesh(mesh, e).expect("d104: geo elem");
        let gnodes = mesh.geometry_nodes(e).to_vec();
        let (_, det, _) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
        vol += qwts[q] * det.abs();
    }
    vol
}

/// `setup_test_norm_coeffs`: `c1 = min(ε/vol, 1)`, `c2 = min(1/ε, 1/vol)`.
fn setup_test_norm_coeffs(mesh: &Mesh<2>, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let ne = mesh.n_elements();
    let mut c1 = Vec::with_capacity(ne);
    let mut c2 = Vec::with_capacity(ne);
    for e in 0..ne as u32 {
        let volume = element_volume(mesh, e);
        c1.push((eps / volume).min(1.0));
        c2.push((1.0 / eps).min(1.0 / volume));
    }
    (c1, c2)
}

/// One `-theta 0`-style level: assemble + solve + postprocess, returning the
/// printed `(dofs, l2, residual)` triple.
fn solve_level(mesh: &Mesh<2>, order: u8, delta_order: u8, eps: f64) -> (usize, f64, f64) {
    let p: u8 = order;
    let test_order: u8 = order + delta_order;
    let result = Arc::new(Mutex::new(None::<(usize, f64, f64)>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::new(mesh.clone());

    let launcher = ThreadLauncher::new(WorkerConfig::new(1));
    launcher.launch(move |comm| {
        let par_mesh = partition_mesh_identity(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();

        let (c1, c2) = setup_test_norm_coeffs(&local_mesh, eps);
        let (c1, c2) = (Arc::new(c1), Arc::new(c2));
        let beta_const: Arc<Vec<f64>> = Arc::new(vec![1.0, 0.0]);

        let mut a = ParDpgWeakForm::new(local_mesh, partition, comm.clone());
        a.set_quad_order((2 * test_order as usize).min(255) as u8);
        a.set_face_quad_order(test_order + p - 1);

        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space_h1(p);
        let hatf = a.add_trial_trace_space(p - 1);
        let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
        let v = a.add_test_space(VolKind::Scalar, test_order);

        // -(βu , ∇v)
        let bv = Arc::clone(&beta_const);
        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakDivergenceSpatialIntegrator {
                beta: Box::new(
                    move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
                          out: &mut [f64]| {
                        beta_function(&bv, &ctx.x, out);
                    },
                ),
            }),
            u,
            v,
        );
        // (σ, ∇v)
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
        // (u , ∇⋅τ)
        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakGradientIntegrator { q: -1.0 }),
            u,
            tau,
        );
        // 1/ε (σ, τ)
        a.add_trial_integrator(
            Box::new(DpgTVectorFEMassIntegrator { q: 1.0 / eps }),
            sig,
            tau,
        );
        //  <û, τ⋅n>   /   <f̂ , v>
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatf, v);

        // c1 (v, δv)   /   ε (∇v, ∇δv)   /   (β·∇v, β·∇δv)
        let c1_v = Arc::clone(&c1);
        a.add_test_integrator(
            Box::new(DpgMassSpatialIntegrator {
                q: Box::new(
                    move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx| {
                        c1_v[ctx.elem as usize]
                    },
                ),
            }),
            v,
            v,
        );
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: eps }), v, v);
        let bb = Arc::clone(&beta_const);
        a.add_test_integrator(
            Box::new(DpgDiffusionSpatialIntegrator {
                q: Box::new(
                    move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
                          out: &mut [f64]| {
                        let mut b = [0.0_f64; 2];
                        beta_function(&bb, &ctx.x, &mut b);
                        out[0] = b[0] * b[0];
                        out[1] = b[0] * b[1];
                        out[2] = b[1] * b[0];
                        out[3] = b[1] * b[1];
                    },
                ),
            }),
            v,
            v,
        );
        // c2 (τ, δτ)   /   (∇⋅τ, ∇⋅δτ)
        let c2_v = Arc::clone(&c2);
        a.add_test_integrator(
            Box::new(DpgVectorFEMassScalarSpatialIntegrator {
                q: Box::new(
                    move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx| {
                        c2_v[ctx.elem as usize]
                    },
                ),
            }),
            tau,
            tau,
        );
        a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);

        // (f, v)
        let bf = Arc::clone(&beta_const);
        a.add_domain_lf_integrator(
            Box::new(DpgDomainLFIntegrator {
                f: move |x: &[f64]| f_exact(x, eps, &bf),
            }),
            v,
        );

        a.store_matrices(true);
        a.assemble();

        // û = -u on the whole boundary; no essential f̂.
        let pairs = a.trace_boundary_dofs(hatu);
        let merged = a.merge_dof_points(&pairs);
        let ess_ids: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();
        let mut x_local = vec![0.0_f64; a.local().size()];
        a.fill_essential_values(&mut x_local, &merged, &|pt: &[f64]| -exact_u(pt));

        let (sys, x0, _) = a.form_linear_system(&ess_ids, &x_local);
        let sub = a.owned_matrix(&sys.a);
        let precond = DpgBlockGs::from_matrix(&sub, a.owned_block_offsets());
        let pc = move |r: &[f64], z: &mut [f64]| precond.apply(r, z);
        let mut xv = fem_parallel::ParVector::from_local_raw(
            x0,
            sys.n_owned,
            a.ghost_exchange_arc(),
            comm.clone(),
        );
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = par_solve_pcg_precond(&sys.a, &sys.b, &mut xv, &pc, &cfg)
            .expect("d104: PCG failed");
        assert!(
            res.iterations < 2000,
            "d104: PCG did not converge — the pinned table is meaningless"
        );
        let x_owned = xv.as_slice()[..sys.n_owned].to_vec();
        let x_full = a.recover_fem_solution(&x_owned);
        let residual = a.global_residual_norm(&x_full);

        // MFEM ComputeL2Error (both overloads, 2*fe order + 3 rule).
        use fem_assembly::dpg::dpg_basis::{scalar_ref_elem, vol_quadrature};
        use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
        let mesh = a.local().mesh();
        let offsets = a.local().trial_offsets();
        let fe = scalar_ref_elem(mesh.element_type(0), p - 1);
        let (qpts, qwts) = vol_quadrature(mesh.element_type(0), 2 * (p - 1) + 3);
        let n = fe.n_dofs();
        let mut phi = vec![0.0_f64; n];
        let (mut error_u, mut error_s) = (0.0_f64, 0.0_f64);
        for e in 0..mesh.n_elements() as u32 {
            let mut elem_u = 0.0_f64;
            let mut elem_s = 0.0_f64;
            for (q, xi) in qpts.iter().enumerate() {
                let geo = geo_ref_elem_from_mesh(mesh, e).expect("d104: geo elem");
                let gnodes = mesh.geometry_nodes(e).to_vec();
                let (_, det, xp) =
                    isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
                let w = qwts[q] * det.abs();
                fe.eval_basis(xi, &mut phi);
                let base = offsets[u] + e as usize * n;
                let mut uh = 0.0;
                for (i, &pv) in phi.iter().enumerate() {
                    uh += x_full[base + i] * pv;
                }
                let a_err = uh - exact_u(&xp);
                elem_u += w * a_err * a_err;
                let sbase = offsets[sig] + e as usize * n * 2;
                let ex = exact_sigma(&xp, eps);
                let mut d0 = 0.0;
                let mut d1 = 0.0;
                for (i, &pv) in phi.iter().enumerate() {
                    d0 += x_full[sbase + i] * pv;
                    d1 += x_full[sbase + n + i] * pv;
                }
                d0 -= ex[0];
                d1 -= ex[1];
                let err = (d0 * d0 + d1 * d1).sqrt();
                elem_s += w * (err * err);
            }
            error_u += elem_u.abs();
            error_s += elem_s.abs();
        }
        let u_err = error_u.sqrt();
        let sig_err = error_s.sqrt();
        let l2 = (u_err * u_err + sig_err * sig_err).sqrt();

        if comm.rank() == 0 {
            *result_slot.lock().expect("d104 mutex") =
                Some((a.n_global_trial_dofs(), l2, residual));
        }
    });

    let out = result
        .lock()
        .expect("d104 mutex after launch")
        .take()
        .expect("d104: rank 0 did not publish");
    out
}

#[test]
fn d104_pconv_cpp_pins_theta0_eps1_two_levels() {
    let mfem = fem_io::mfem::read_mfem_file("../../data/inline-quad.mesh")
        .expect("d104: inline-quad.mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("d104: 2-D mesh");

    // C++ `-no-vis -theta 0.0` row 0: `113 | 1.033e+00 | 9.290e-01`.
    let (dofs0, l20, res0) = solve_level(&mesh, 1, 1, 1.0);
    assert_eq!(dofs0, 113);
    assert_eq!(cpp_sci3(l20), "1.033e+00", "L2 level 0 vs C++");
    assert_eq!(cpp_sci3(res0), "9.290e-01", "DPG residual level 0 vs C++");

    // Uniform refinement (the `-theta 0.0` mark-all set), row 1:
    // `417 | 5.172e-01 | 4.817e-01 | -1.06 | -1.01`.
    mesh = refine_uniform(&mesh);
    let (dofs1, l21, res1) = solve_level(&mesh, 1, 1, 1.0);
    assert_eq!(dofs1, 417);
    assert_eq!(cpp_sci3(l21), "5.172e-01", "L2 level 1 vs C++");
    assert_eq!(cpp_sci3(res1), "4.817e-01", "DPG residual level 1 vs C++");
    let rate_err = 2.0 * (l20 / l21).ln() / ((dofs0 as f64) / (dofs1 as f64)).ln();
    let rate_res = 2.0 * (res0 / res1).ln() / ((dofs0 as f64) / (dofs1 as f64)).ln();
    assert_eq!(format!("{rate_err:.2}"), "-1.06", "L2 rate vs C++");
    assert_eq!(format!("{rate_res:.2}"), "-1.01", "residual rate vs C++");
}

#[test]
fn d104_pconv_cpp_pins_theta0_eps1e2() {
    let mfem = fem_io::mfem::read_mfem_file("../../data/inline-quad.mesh")
        .expect("d104: inline-quad.mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("d104: 2-D mesh");

    // C++ `-no-vis -theta 0.0 -eps 1e-2` row 0: `113 | 2.523e-01 | 3.101e-01`.
    // This exercises the coefficient-weighted test norm: c1 = min(ε/|e|,1) =
    // 0.16 and c2 = min(1/ε, 1/|e|) = 16 on the level-0 mesh.
    let (dofs0, l20, res0) = solve_level(&mesh, 1, 1, 1e-2);
    assert_eq!(dofs0, 113);
    assert_eq!(cpp_sci3(l20), "2.523e-01", "L2 level 0 (eps=1e-2) vs C++");
    assert_eq!(cpp_sci3(res0), "3.101e-01", "DPG residual (eps=1e-2) vs C++");
}
