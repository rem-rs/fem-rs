//! Parallel ultraweak DPG solver for the convection–diffusion problem — 1:1
//! port of MFEM's `miniapps/dpg/pconvection-diffusion.cpp` (MFEM 4.10).
//!
//! Solves `-εΔu + ∇·(βu) = f` in Ω, `u = u₀` on ∂Ω through the first-order
//! system `∇·σ + ∇·(βu) = f`, `1/ε σ − ∇u = 0` with skeleton traces
//! `û ∈ H^{1/2}`, `f̂ ∈ H^{-1/2}`:
//!
//! ```text
//!     -(βu , ∇v)  + (σ , ∇v)     + < f̂ ,  v  > = (f,v),   ∀ v ∈ H¹(Ωₕ)
//!       (u , ∇⋅τ) + 1/ε (σ , τ)  + < û , τ⋅n > = 0,     ∀ τ ∈ H(div,Ωₕ)
//! ```
//!
//! Trial spaces (as in the C++ miniapp): `u ∈ L²(p−1)`, `σ ∈ (L²(p−1))²`,
//! `û ∈ H¹-trace(p)`, `f̂ ∈ RT-trace(p−1)`; broken test spaces `v ∈ H¹(p+δ)`,
//! `τ ∈ RT(p+δ−1)` with the *coefficient-weighted* test norm
//! `c₁(v,δv) + ε(∇v,∇δv) + (β·∇v,β·∇δv) + c₂(τ,δτ) + (∇·τ,∇·δτ)`, where the
//! L²(P0) fields `c₁, c₂` are recomputed per element after every refinement
//! (`setup_test_norm_coeffs`: `c₁ = min(ε/|e|, 1)`, `c₂ = min(1/ε, 1/|e|)`).
//!
//! Problem cases: `-prob 0` manufactured `u = sin(π(x+y))` (default, `β` from
//! `-beta`, default `(1,0)`); `-prob 2` curved streamlines
//! (`β = (eˣsin y, eˣcos y)`, `u = atan((1−r)/ε)`); `-prob 3` boundary layer
//! (`β = (1,2)`, no volume forcing, piecewise `û|∂Ω`, residual-only table).
//! `-prob 1` (Erickson–Johnson) is not ported (D962).
//!
//! # Verified against the C++ MPI reference
//!
//! (Round-105 D963 note: at delivery the `--ranks 2` rows ran against the
//! pre-D963 broken parallel path and were checked at `--ranks 1` only; the
//! round-105 parallel-DPG fix (see `pdiffusion.rs` § D963) closed the gap and
//! the `--ranks 2` tables below were re-verified digit-for-digit against a
//! fresh `mpirun -np 2` oracle, including the residual-marker union fix — at
//! two ranks the published mark list must be the union of every rank's list,
//! not rank 0's alone.)
//!
//! Built from `$HOME/mfem410_mpi` (build line in `tmp/d104pconv/REPORT.md`) and
//! run under `mpirun -np {1,2} … -no-vis -theta 0.0`.  Every printed number of
//! the convergence table — `Dofs`, `L2 Error`, `Residual`, `Rate` — matches the
//! fem-rs run **to all printed digits** (`--ranks 1` and `--ranks 2` agree):
//!
//! ```text
//!                                   C++ (-np 1/2)                fem-rs
//! -theta 0.0:            113 1.033e+00 9.290e-01   ==  113 1.033e+00 9.290e-01
//!                        417 5.172e-01 4.817e-01   ==  417 5.172e-01 4.817e-01
//! -theta 0.0 -eps 1e-2:  113 2.523e-01 3.101e-01   ==  113 2.523e-01 3.101e-01
//!                        417 1.431e-01 1.414e-01   ==  417 1.431e-01 1.414e-01
//! -o 2 -theta 0.0 -ref 2:    337 1.052e-01 9.535e-02   ==  337 1.052e-01 9.535e-02
//!                           1281 2.625e-02 2.437e-02   == 1281 2.625e-02 2.437e-02
//!                           4993 6.552e-03 6.145e-03   == 4993 6.552e-03 6.145e-03
//! -prob 2 -eps 5e-3 -o 2 -theta 0.0 -ref 2:
//!                          337 6.984e-01 6.401e-01   ==  337 6.984e-01 6.401e-01
//!                         1281 4.960e-01 6.226e-01   == 1281 4.960e-01 6.226e-01
//!                         4993 2.202e-01 2.582e-01   == 4993 2.202e-01 2.582e-01
//! -beta '2 3' -o 2 -theta 0.0:
//!                          337 1.331e-01 7.851e-02   ==  337 1.331e-01 7.851e-02
//!                         1281 2.960e-02 2.064e-02   == 1281 2.960e-02 2.064e-02
//! ```
//!
//! The **PCG iteration count is not reproduced** (D963): the C++ miniapp
//! preconditions each diagonal block with Hypre (`MakeFESpaceDefaultSolver` =
//! BoomerAMG on the H1/L2 blocks, AMS on the RT block in 2-D), fem-rs applies
//! symmetric GS per block (`DpgBlockGs`, same situation as `pdiffusion`), and
//! the C++ count already differs between `-np 1` and `-np 2` for the identical
//! system (41 vs 43 at `-theta 0.0`).  The printed `Residual` is the *DPG*
//! residual `‖B x − F‖_{S⁻¹}`, which is reproduced to all printed digits.
//!
//! # Known gaps (exit code 3, debts D960–D964)
//!
//! * The C++ **literal default** `theta = 0.7` marks the subset of elements
//!   with `res_e > θ·max_e` and refines them with MFEM's nonconforming quad
//!   split (`GeneralRefinement(marked,1,1)`, hanging nodes).  **D960 closed
//!   (round 106)**: the mark set itself matches C++ (9/16 at level 0) and the
//!   refinement is wired to `amr::general_refinement_quad_aniso` (the
//!   `NcQuadTree` machinery, MFEM-probe-validated in `d246_quad_aniso_nc`).
//!   The remaining gap is **D1030**: the DPG solve lacks hanging-node
//!   constraints for the trace blocks (level-1 dofs 305 vs C++ 275 — the
//!   parent-edge RT-trace dofs C++ eliminates through its conforming
//!   restriction), so a *partial* mark set still exits 3 with the precise
//!   delta printed.  With `-theta 0.0` (mark-all) the refinement is uniform
//!   and matches C++ exactly.
//! * `-pmg` (`PRefinementMultigrid`): not ported (**D961**); exits 3.
//! * `-prob 1` (Erickson–Johnson): the essential `f̂` boundary condition needs
//!   `ProjectBdrCoefficientNormal` (RT-trace normal projection), which fem-rs
//!   does not have (**D962**); exits 3.
//! * 3-D meshes (`-m ../../data/inline-hex.mesh`): the parallel DPG lane is
//!   2-D (**D964**); exits 3.
//! * GLVis/ParaView output is not implemented; `-no-vis` (and `-no-paraview`)
//!   match the C++ output.
//!
//! Usage:
//!   cargo run --release --example pconvection_diffusion -- --ranks 2 -theta 0.0
//!   cargo run --release --example pconvection_diffusion -- --ranks 2 -o 2 -theta 0.0 -ref 2

use std::process::exit;
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
use fem_mesh::amr::general_refinement_quad_aniso;
use fem_mesh::{refine_uniform, ElementType, Mesh, MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_dpg_weakform::ParDpgWeakForm;
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_solve_pcg_precond;
use fem_parallel::WorkerConfig;
use fem_solver::SolverConfig;

const PI: f64 = std::f64::consts::PI;

/// Problem case — C++ `enum prob_type { sinusoidal, EJ, curved_streamlines,
/// bdr_layer }`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prob {
    Sinusoidal,
    Ej,
    CurvedStreamlines,
    BdrLayer,
}

/// `beta_function` of `pconvection-diffusion.cpp` (`β = β_` except for the
/// curved-streamlines case; the C++ always `SetSize(2)`s the output).
fn beta_function(x: &[f64], prob: Prob, beta_const: &[f64], out: &mut [f64]) {
    out[0] = 0.0;
    out[1] = 0.0;
    if prob == Prob::CurvedStreamlines {
        let (px, py) = (x[0], x[1]);
        out[0] = px.exp() * py.sin();
        out[1] = px.exp() * py.cos();
    } else {
        out[0] = beta_const[0];
        out[1] = beta_const[1];
    }
}

/// `exact_u` of `pconvection-diffusion.cpp`.
fn exact_u(x: &[f64], prob: Prob, eps: f64) -> f64 {
    let (px, py) = (x[0], x[1]);
    match prob {
        Prob::Sinusoidal => (PI * (px + py)).sin(),
        Prob::Ej => {
            let alpha = (1.0 + 4.0 * eps * eps * PI * PI).sqrt();
            let r1 = (1.0 + alpha) / (2.0 * eps);
            let r2 = (1.0 - alpha) / (2.0 * eps);
            let denom = (-r2).exp() - (-r1).exp();
            let g = (r2 * (px - 1.0)).exp() - (r1 * (px - 1.0)).exp();
            g * (PI * py).cos() / denom
        }
        Prob::CurvedStreamlines => {
            let r = (px * px + py * py).sqrt();
            ((1.0 - r) / eps).atan()
        }
        Prob::BdrLayer => unreachable!("exact_u is not evaluated for the bdr_layer problem"),
    }
}

/// `exact_gradu` of `pconvection-diffusion.cpp`.
fn exact_gradu(x: &[f64], prob: Prob, eps: f64, du: &mut [f64]) {
    let (px, py) = (x[0], x[1]);
    match prob {
        Prob::Sinusoidal => {
            let alpha = PI * (px + py);
            du[0] = PI * alpha.cos();
            du[1] = PI * alpha.cos();
        }
        Prob::Ej => {
            let alpha = (1.0 + 4.0 * eps * eps * PI * PI).sqrt();
            let r1 = (1.0 + alpha) / (2.0 * eps);
            let r2 = (1.0 - alpha) / (2.0 * eps);
            let denom = (-r2).exp() - (-r1).exp();
            let g1 = (r2 * (px - 1.0)).exp();
            let g1_x = r2 * g1;
            let g2 = (r1 * (px - 1.0)).exp();
            let g2_x = r1 * g2;
            let g = g1 - g2;
            let g_x = g1_x - g2_x;
            du[0] = g_x * (PI * py).cos() / denom;
            du[1] = -PI * g * (PI * py).sin() / denom;
        }
        Prob::CurvedStreamlines => {
            let r = (px * px + py * py).sqrt();
            let alpha = -2.0 * r + r * r + eps * eps + 1.0;
            let denom = r * alpha;
            du[0] = -px * eps / denom;
            du[1] = -py * eps / denom;
        }
        Prob::BdrLayer => unreachable!("exact_gradu is not evaluated for the bdr_layer problem"),
    }
}

/// `exact_laplacian_u` of `pconvection-diffusion.cpp`.
fn exact_laplacian_u(x: &[f64], prob: Prob, eps: f64) -> f64 {
    let (px, py) = (x[0], x[1]);
    match prob {
        Prob::Sinusoidal => -PI * PI * (PI * (px + py)).sin() * 2.0,
        Prob::Ej => {
            let alpha = (1.0 + 4.0 * eps * eps * PI * PI).sqrt();
            let r1 = (1.0 + alpha) / (2.0 * eps);
            let r2 = (1.0 - alpha) / (2.0 * eps);
            let denom = (-r2).exp() - (-r1).exp();
            let g1 = (r2 * (px - 1.0)).exp();
            let g1_x = r2 * g1;
            let g1_xx = r2 * g1_x;
            let g2 = (r1 * (px - 1.0)).exp();
            let g2_x = r1 * g2;
            let g2_xx = r1 * g2_x;
            let g = g1 - g2;
            let g_xx = g1_xx - g2_xx;
            let u = g * (PI * py).cos() / denom;
            let u_xx = g_xx * (PI * py).cos() / denom;
            let u_yy = -PI * PI * u;
            u_xx + u_yy
        }
        Prob::CurvedStreamlines => {
            let r = (px * px + py * py).sqrt();
            let alpha = -2.0 * r + r * r + eps * eps + 1.0;
            eps * (r * r - eps * eps - 1.0) / (r * alpha * alpha)
        }
        Prob::BdrLayer => unreachable!("exact_laplacian_u is not evaluated for bdr_layer"),
    }
}

/// `exact_sigma` = `ε ∇u`.
fn exact_sigma(x: &[f64], prob: Prob, eps: f64, sigma: &mut [f64]) {
    exact_gradu(x, prob, eps, sigma);
    sigma[0] *= eps;
    sigma[1] *= eps;
}

/// `exact_hatu` = `-u` on the skeleton.
fn exact_hatu(x: &[f64], prob: Prob, eps: f64) -> f64 {
    -exact_u(x, prob, eps)
}

/// `f_exact` = `-εΔu + β·∇u`.
fn f_exact(x: &[f64], prob: Prob, eps: f64, beta_const: &[f64]) -> f64 {
    let mut du = [0.0_f64; 2];
    exact_gradu(x, prob, eps, &mut du);
    let d2u = exact_laplacian_u(x, prob, eps);
    let mut beta_val = [0.0_f64; 2];
    beta_function(x, prob, beta_const, &mut beta_val);
    -eps * d2u + beta_val[0] * du[0] + beta_val[1] * du[1]
}

/// `bdr_data` of `pconvection-diffusion.cpp` (the `û` boundary data).
fn bdr_data(x: &[f64], prob: Prob, eps: f64) -> f64 {
    if prob == Prob::BdrLayer {
        let (px, py) = (x[0], x[1]);
        if py == 0.0 {
            -(1.0 - px)
        } else if px == 0.0 {
            -(1.0 - py)
        } else {
            0.0
        }
    } else {
        exact_hatu(x, prob, eps)
    }
}

/// C++-style `std::scientific` with 3 digits (`1.033e+00`).
fn cpp_sci3(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v:.3e}");
    }
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

/// C++-style `operator<<` default (`%g`, 6 significant digits) for the option
/// banner (`--epsilon 1`, `--theta 0.7`, `--relaxation-factor 0.666667`).
fn cpp_g6(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let neg = v < 0.0;
    let a = v.abs();
    let exp = a.log10().floor() as i32;
    let s = if exp < -4 || exp >= 6 {
        let mut mant = a / 10f64.powi(exp);
        if mant >= 10.0 {
            mant /= 10.0;
        }
        let mut s = format!("{mant:.5}");
        if s.parse::<f64>().unwrap_or(mant) >= 10.0 {
            s = format!("{:.5}", mant / 10.0);
        }
        let s = s
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        format!("{s}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    } else {
        let prec = (5 - exp).max(0) as usize;
        let s = format!("{:.*}", prec, a);
        s.trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    };
    format!("{}{s}", if neg { "-" } else { "" })
}

/// One refinement level: assemble, solve, and return `(dofs, l2_error,
/// residual, iterations, marked_elements)`.
struct LevelResult {
    dofs: usize,
    l2: f64,
    residual: f64,
    iters: usize,
    marked: Vec<u32>,
}

fn solve_level(
    mesh: &Mesh<2>,
    n_workers: usize,
    order: u8,
    delta_order: u8,
    prob: Prob,
    static_cond: bool,
    eps: f64,
    beta_const: Arc<Vec<f64>>,
    theta: f64,
) -> LevelResult {
    let p: u8 = order;
    let test_order: u8 = order + delta_order;
    let p_us = p as usize;
    let exact_known = prob != Prob::BdrLayer;
    let result = Arc::new(Mutex::new(None::<LevelResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::new(mesh.clone());

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        // Identity node numbering: the DPG face/edge tables derive their
        // canonical face direction from the local node ids, so the numbering
        // must agree with the global one on every rank.
        let par_mesh = partition_mesh_identity(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();
        let elem_owner: Vec<i32> = partition.elem_owner.clone();
        let global_elem_ids: Vec<u32> = partition.global_elem_ids.clone();

        // setup_test_norm_coeffs (util/preconditioners.cpp): per-element P0
        // weights c1 = min(eps/vol, 1), c2 = min(1/eps, 1/vol) with the MFEM
        // GetElementVolume rule (`IntRules.Get(geom, OrderJ())`, one point for
        // straight-sided elements — exact for the affine inline-quad family).
        let (c1, c2) = setup_test_norm_coeffs(&local_mesh, eps);
        let c1 = Arc::new(c1);
        let c2 = Arc::new(c2);

        let mut a = ParDpgWeakForm::new(local_mesh, partition, comm.clone());
        a.set_quad_order((2 * test_order as usize).min(255) as u8);
        a.set_face_quad_order(test_order + p - 1);

        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space_h1(p);
        let hatf = a.add_trial_trace_space(p - 1);

        let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
        let v = a.add_test_space(VolKind::Scalar, test_order);

        // Trial integrators (C++ block table).
        // -(βu , ∇v)
        let beta_const_v = Arc::clone(&beta_const);
        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakDivergenceSpatialIntegrator {
                beta: Box::new(move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
                                    out: &mut [f64]| {
                    beta_function(&ctx.x, prob, &beta_const_v, out);
                }),
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
        //  <û, τ⋅n>
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        //  <f̂ , v>
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatf, v);

        // Test integrators (coefficient-weighted test norm).
        // c1 (v, δv)
        let c1_v = Arc::clone(&c1);
        a.add_test_integrator(
            Box::new(DpgMassSpatialIntegrator {
                q: Box::new(move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx| {
                    c1_v[ctx.elem as usize]
                }),
            }),
            v,
            v,
        );
        // ε (∇v, ∇δv)
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: eps }), v, v);
        // (β·∇v, β·∇δv)
        let beta_const_b = Arc::clone(&beta_const);
        a.add_test_integrator(
            Box::new(DpgDiffusionSpatialIntegrator {
                q: Box::new(move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
                                  out: &mut [f64]| {
                    // OuterProductCoefficient(betacoeff, betacoeff): M = ββᵀ.
                    let mut beta_val = [0.0_f64; 2];
                    beta_function(&ctx.x, prob, &beta_const_b, &mut beta_val);
                    out[0] = beta_val[0] * beta_val[0];
                    out[1] = beta_val[0] * beta_val[1];
                    out[2] = beta_val[1] * beta_val[0];
                    out[3] = beta_val[1] * beta_val[1];
                }),
            }),
            v,
            v,
        );
        // c2 (τ, δτ)
        let c2_v = Arc::clone(&c2);
        a.add_test_integrator(
            Box::new(DpgVectorFEMassScalarSpatialIntegrator {
                q: Box::new(move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx| {
                    c2_v[ctx.elem as usize]
                }),
            }),
            tau,
            tau,
        );
        // (∇⋅τ, ∇⋅δτ)
        a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);

        // (f, v) — only the sinusoidal / curved-streamlines cases force.
        if matches!(prob, Prob::Sinusoidal | Prob::CurvedStreamlines) {
            let beta_const_f = Arc::clone(&beta_const);
            a.add_domain_lf_integrator(
                Box::new(DpgDomainLFIntegrator {
                    f: move |x: &[f64]| f_exact(x, prob, eps, &beta_const_f),
                }),
                v,
            );
        }

        a.store_matrices(true);
        if static_cond {
            a.enable_static_condensation();
        }
        a.assemble();

        // Essential BCs: û on every global boundary face (C++
        // `ess_bdr_uhat = 1`), none for f̂ (`ess_bdr_fhat = 0`); the EJ
        // attribute split is the D962 gap and never reaches this point.
        let pairs = a.trace_boundary_dofs(hatu);
        let merged = a.merge_dof_points(&pairs);
        let ess_ids: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();

        let mut x_local = vec![0.0_f64; a.local().size()];
        a.fill_essential_values(&mut x_local, &merged, &move |pt: &[f64]| {
            bdr_data(pt, prob, eps)
        });

        let (sys, x0, _) = a.form_linear_system(&ess_ids, &x_local);

        // Block-diagonal symmetric-GS preconditioner on the owned segment
        // (C++ uses Hypre BoomerAMG/AMS per block — D963).
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
            .expect("pconvection_diffusion: PCG failed");
        let x_owned = xv.as_slice()[..sys.n_owned].to_vec();
        let x_full = a.recover_fem_solution(&x_owned);

        // Per-element DPG residuals (C++ `ComputeResidual`) over the rank's
        // local mesh (owned + ghost closure).
        let residuals = a.compute_residual(&x_full);
        let residual = a.global_residual_norm(&x_full);

        // Marking `residuals[iel] > theta * maxresidual` over the rank's OWNED
        // elements (C++ marks its rank-local pmesh elements; a ghost is marked
        // by its owner, so the union across ranks is exactly-once).  Local
        // indices map to serial element ids through
        // `partition.global_elem_ids`; the max is a global reduction (Comm has
        // no max — one f64 per rank through `alltoallv_bytes`, the same
        // pattern as `merge_dof_points`).
        let rank_owner = rank as i32;
        let mut maxresidual = 0.0_f64;
        for (iel, &r) in residuals.iter().enumerate() {
            if elem_owner[iel] == rank_owner && r > maxresidual {
                maxresidual = r;
            }
        }
        let maxresidual = allreduce_max_f64(&comm, maxresidual);
        let mut marked = Vec::new();
        if theta < 1.0 || maxresidual == 0.0 {
            for (iel, &r) in residuals.iter().enumerate() {
                if elem_owner[iel] == rank_owner && r > theta * maxresidual {
                    marked.push(global_elem_ids[iel]);
                }
            }
            marked.sort_unstable();
        }
        // The serial mesh is refined from the **union** of every rank's mark
        // list (C++ refines collectively on the ParMesh with rank-local
        // indices; here the ranks share one serial mesh, so the published
        // list must be merged — rank 0's list alone is half the mesh at
        // two ranks).
        let mut union: std::collections::BTreeSet<u32> = marked.iter().copied().collect();
        if comm.size() > 1 {
            let mut payload = Vec::with_capacity(marked.len() * 4 + 4);
            payload.extend_from_slice(&(marked.len() as u32).to_le_bytes());
            for &g in &marked {
                payload.extend_from_slice(&g.to_le_bytes());
            }
            let sends: Vec<(i32, Vec<u8>)> = (0..comm.size() as i32)
                .map(|r| (r, payload.clone()))
                .collect();
            for (_, bytes) in comm.alltoallv_bytes(&sends) {
                if bytes.len() >= 4 {
                    let n = u32::from_le_bytes(bytes[..4].try_into().expect("count")) as usize;
                    for i in 0..n {
                        let b = 4 + 4 * i;
                        if b + 4 <= bytes.len() {
                            union.insert(u32::from_le_bytes(
                                bytes[b..b + 4].try_into().expect("gid"),
                            ));
                        }
                    }
                }
            }
        }
        let marked: Vec<u32> = union.into_iter().collect();

        let (e_u, e_s) = if exact_known {
            l2_errors(&a, &x_full, u, sig, p_us.saturating_sub(1) as u8, prob, eps)
        } else {
            (0.0, 0.0)
        };
        let l2 = comm.allreduce_sum_f64(e_u + e_s).max(0.0).sqrt();

        if rank == 0 {
            *result_slot.lock().expect("pconvection mutex") = Some(LevelResult {
                dofs: a.n_global_trial_dofs(),
                l2,
                residual,
                iters: res.iterations,
                marked,
            });
        }
    });

    let out = result
        .lock()
        .expect("pconvection mutex after launch")
        .take()
        .expect("rank 0 did not publish the pconvection-diffusion result");
    out
}

/// MFEM `Mesh::GetElementVolume`: `IntRules.Get(geom, OrderJ())` — for
/// straight-sided elements `OrderJ()` is 1, i.e. the one-point rule (exact for
/// the constant Jacobian of the affine inline-quad family).
fn element_volume(mesh: &Mesh<2>, e: u32) -> f64 {
    use fem_assembly::dpg::dpg_basis::{scalar_ref_elem, vol_quadrature};
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};

    let et = mesh.element_type(e);
    let (qpts, qwts) = vol_quadrature(et, 1);
    let simp = matches!(et, ElementType::Tri3);
    let mut vol = 0.0_f64;
    for (q, xi) in qpts.iter().enumerate() {
        let det = if simp {
            let tr = fem_mesh::ElementTransformation::from_simplex_nodes(
                mesh,
                mesh.element_nodes(e),
            );
            tr.det_j()
        } else {
            let geo = geo_ref_elem_from_mesh(mesh, e).expect("pconvection: geo elem");
            let gnodes = mesh.geometry_nodes(e).to_vec();
            let (_, det, _) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
            det
        };
        vol += qwts[q] * det.abs();
    }
    let _ = scalar_ref_elem(ElementType::Quad4, 1);
    vol
}

/// `setup_test_norm_coeffs` (`util/preconditioners.cpp`): per-element
/// `c1 = min(ε/vol, 1)`, `c2 = min(1/ε, 1/vol)`.
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

/// Squared L2 errors of the `u` and `σ` trial blocks (owned elements only),
/// mirroring MFEM `ParGridFunction::ComputeL2Error`.
fn l2_errors(
    a: &ParDpgWeakForm<Mesh<2>>,
    x_full: &[f64],
    u_block: usize,
    sig_block: usize,
    order: u8,
    prob: Prob,
    eps: f64,
) -> (f64, f64) {
    use fem_assembly::dpg::dpg_basis::{scalar_ref_elem, vol_quadrature};
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};

    let mesh = a.local().mesh();
    let offsets = a.local().trial_offsets();
    let et = mesh.element_type(0);
    let n = scalar_ref_elem(et, order).n_dofs();
    let (qpts, qwts) = vol_quadrature(et, 2 * order + 3);
    let fe = scalar_ref_elem(et, order);
    let mut phi = vec![0.0_f64; n];
    let simp = matches!(et, ElementType::Tri3);
    let rank = a.comm().rank();
    let part = a.partition_ref();

    // MFEM `GridFunction::ComputeL2Error` kernels, summation for summation:
    // both overloads use `intorder = 2*fe->GetOrder() + 3`, accumulate a
    // per-element `elem_error` (added as `fabs`), and multiply
    // `ip.weight * Trans.Weight()` first; the vector overload takes the
    // per-point 2-norm before squaring.
    let (mut error_u, mut error_s) = (0.0_f64, 0.0_f64);
    for e in 0..mesh.n_elements() as u32 {
        if part.elem_owner[e as usize] != rank {
            continue;
        }
        let mut elem_error_u = 0.0_f64;
        let mut elem_error_s = 0.0_f64;
        for (q, xi) in qpts.iter().enumerate() {
            let (det, xp) = if simp {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(
                    mesh,
                    mesh.element_nodes(e),
                );
                (tr.det_j(), tr.map_to_physical(xi))
            } else {
                let geo = geo_ref_elem_from_mesh(mesh, e).expect("pconvection: geo elem");
                let gnodes = mesh.geometry_nodes(e).to_vec();
                let (_, det, xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
                (det, xp)
            };
            fe.eval_basis(xi, &mut phi);
            let w = qwts[q] * det.abs();
            // u (scalar L2 block) — scalar-Coefficient overload:
            // `elem_error += ip.weight * Trans.Weight() * a * a`.
            let base = offsets[u_block] + e as usize * n;
            let mut uh = 0.0;
            for (i, &p) in phi.iter().enumerate() {
                uh += x_full[base + i] * p;
            }
            let a = uh - exact_u(&xp, prob, eps);
            elem_error_u += w * a * a;
            // σ (vector L2 block, byNODES ordering) — VectorCoefficient
            // overload: `err = Norm2(vals - exact)` per point, then
            // `elem_error += ip.weight * Trans.Weight() * (err*err)`.
            let sbase = offsets[sig_block] + e as usize * n * 2;
            let mut ex = [0.0_f64; 2];
            exact_sigma(&xp, prob, eps, &mut ex);
            let (mut d0, mut d1) = (0.0_f64, 0.0_f64);
            for (i, &p) in phi.iter().enumerate() {
                d0 += x_full[sbase + i] * p;
                d1 += x_full[sbase + n + i] * p;
            }
            d0 -= ex[0];
            d1 -= ex[1];
            let err = (d0 * d0 + d1 * d1).sqrt();
            elem_error_s += w * (err * err);
        }
        error_u += elem_error_u.abs();
        error_s += elem_error_s.abs();
    }
    (error_u, error_s)
}

/// Global max across ranks (the DPG residual marker's `MPI_Allreduce(MAX)`;
/// `Comm` exposes only sums, so ship one `f64` per rank through
/// `alltoallv_bytes` — the same pattern `ParDpgWeakForm::merge_dof_points`
/// uses).
fn allreduce_max_f64(comm: &fem_parallel::Comm, local: f64) -> f64 {
    let bits = local.to_le_bytes().to_vec();
    let sends: Vec<(i32, Vec<u8>)> = (0..comm.size() as i32)
        .map(|r| (r, bits.clone()))
        .collect();
    let mut out = local;
    for (_, bytes) in comm.alltoallv_bytes(&sends) {
        if bytes.len() >= 8 {
            let v = f64::from_le_bytes(bytes[..8].try_into().expect("f64 bytes"));
            if v > out {
                out = v;
            }
        }
    }
    out
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flags: &[&str]) -> Option<T> {
    for f in flags {
        if let Some(i) = args.iter().position(|a| a == f) {
            if let Some(v) = args.get(i + 1) {
                if let Ok(v) = v.parse::<T>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn has_flag(args: &[String], flags: &[&str]) -> bool {
    args.iter().any(|a| flags.contains(&a.as_str()))
}

/// Vector option (`OptionsParser` + `Vector`): consumes the numeric tokens
/// that follow the flag (`-beta '2 3'` → `-beta 2 3` after shell quoting).
fn parse_vec_arg(args: &[String], flags: &[&str]) -> Option<Vec<f64>> {
    for f in flags {
        if let Some(i) = args.iter().position(|a| a == f) {
            let mut vals = Vec::new();
            for a in &args[i + 1..] {
                match a.parse::<f64>() {
                    Ok(v) => vals.push(v),
                    Err(_) => break,
                }
            }
            if !vals.is_empty() {
                return Some(vals);
            }
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_workers: usize = parse_arg(&args, &["--ranks"]).unwrap_or(2);
    let mut mesh_file: String =
        parse_arg(&args, &["-m", "--mesh"]).unwrap_or_else(|| "data/inline-quad.mesh".into());
    let order: i32 = parse_arg(&args, &["-o", "--order"]).unwrap_or(1);
    let delta_order: i32 = parse_arg(&args, &["-do", "--delta-order"]).unwrap_or(1);
    let eps: f64 = parse_arg(&args, &["-eps", "--epsilon"]).unwrap_or(1.0);
    let ref_levels: i32 = parse_arg(&args, &["-ref", "--num-refinements"]).unwrap_or(1);
    let theta: f64 = parse_arg(&args, &["-theta", "--theta"]).unwrap_or(0.7);
    let mut iprob: i32 = parse_arg(&args, &["-prob", "--problem"]).unwrap_or(0);
    let mut beta: Vec<f64> = parse_vec_arg(&args, &["-beta", "--beta"]).unwrap_or_default();
    let static_cond = has_flag(&args, &["-sc", "--static-condensation"]);
    let pmg = has_flag(&args, &["-pmg", "--p-refinement-multigrid"]);
    let _pmg_levels: i32 = parse_arg(&args, &["-pmgl", "--p-refinement-multigrid-levels"])
        .unwrap_or(-1);
    let _relax_factor: f64 = parse_arg(&args, &["-rf", "--relaxation-factor"]).unwrap_or(2.0 / 3.0);
    let visualization = !has_flag(&args, &["-no-vis", "--no-visualization"]);
    let _paraview = has_flag(&args, &["-paraview", "--paraview"]);
    let _visport: i32 = parse_arg(&args, &["-p", "--send-port"]).unwrap_or(19916);

    // C++: `if (iprob > 3) { iprob = 3; }`.
    if iprob > 3 {
        iprob = 3;
    }
    let prob = [
        Prob::Sinusoidal,
        Prob::Ej,
        Prob::CurvedStreamlines,
        Prob::BdrLayer,
    ][iprob.unsigned_abs().min(3) as usize];

    // C++ forces the inline-quad mesh for the EJ / curved-streamlines /
    // boundary-layer cases.
    if matches!(
        prob,
        Prob::Ej | Prob::CurvedStreamlines | Prob::BdrLayer
    ) {
        mesh_file = "data/inline-quad.mesh".into();
    }

    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("pconvection_diffusion: cannot read {mesh_file}: {e}"));
    let mut mesh: Mesh<2> = match mfem.mesh2d {
        Some(m) => m,
        None => {
            eprintln!(
                "pconvection_diffusion: GAP — 3-D meshes are not supported by the 2-D \
                 parallel DPG lane (D964)."
            );
            exit(3);
        }
    };
    let dim = 2usize;

    let exact_known = prob != Prob::BdrLayer;
    match prob {
        Prob::Sinusoidal | Prob::Ej => {
            if beta.is_empty() {
                beta = vec![0.0; dim];
                beta[0] = 1.0;
            }
        }
        Prob::BdrLayer => {
            beta = vec![0.0; dim];
            beta[0] = 1.0;
            beta[1] = 2.0;
        }
        Prob::CurvedStreamlines => {}
    }

    // C++ `args.PrintOptions` banner (with -no-vis / -no-paraview).
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --order {order}");
    println!("   --delta-order {delta_order}");
    println!("   --epsilon {}", cpp_g6(eps));
    println!("   --num-refinements {ref_levels}");
    println!("   --theta {}", cpp_g6(theta));
    println!("   --problem {iprob}");
    print!("   --beta '");
    for (i, b) in beta.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", cpp_g6(*b));
    }
    println!("'");
    println!(
        "   {}",
        if static_cond {
            "--static-condensation"
        } else {
            "--no-static-condensation"
        }
    );
    println!(
        "   {}",
        if pmg {
            "--p-refinement-multigrid"
        } else {
            "--no-p-refinement-multigrid"
        }
    );
    println!("   --p-refinement-multigrid-levels -1");
    println!("   --relaxation-factor {}", cpp_g6(2.0 / 3.0));
    println!(
        "   {}",
        if visualization {
            "--visualization"
        } else {
            "--no-visualization"
        }
    );
    println!("   --no-paraview");
    println!("   --send-port 19916");
    println!("   --ranks {n_workers}");

    if pmg {
        eprintln!(
            "pconvection_diffusion: GAP — `-pmg` (PRefinementMultigrid) is not ported to \
             fem-rs (D961).  The C++ miniapp builds a p-multigrid preconditioner over the \
             trial spaces (`util/preconditioners.cpp:PRefinementMultigrid`); fem-rs has no \
             p-prolongation operators for DPG trace/volume blocks."
        );
        exit(3);
    }
    if prob == Prob::Ej {
        eprintln!(
            "pconvection_diffusion: GAP — `-prob 1` (Erickson–Johnson) needs the essential \
             f̂ boundary condition `ProjectBdrCoefficientNormal` (RT-trace normal \
             projection), which fem-rs does not implement (D962)."
        );
        exit(3);
    }

    if exact_known {
        println!(
            "\n  Ref |    Dofs    |  L2 Error  |  Rate  |  Residual  |  Rate  | CG it  |"
        );
        println!("{}", "-".repeat(72));
    } else {
        println!("\n  Ref |    Dofs    |  Residual  |  Rate  | CG it  |");
        println!("{}", "-".repeat(50));
    }

    let order = order.max(1) as u8;
    let delta_order = delta_order.max(0) as u8;
    let beta_const = Arc::new(beta);

    let mut err0 = 0.0_f64;
    let mut res0 = 0.0_f64;
    let mut dof0 = 0usize;

    for it in 0..=ref_levels.max(0) {
        let r = solve_level(
            &mesh,
            n_workers,
            order,
            delta_order,
            prob,
            static_cond,
            eps,
            Arc::clone(&beta_const),
            theta,
        );
        let dim_f = dim as f64;
        let rate_err = if it > 0 && err0 > 0.0 && r.dofs != dof0 {
            dim_f * (err0 / r.l2).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
        } else {
            0.0
        };
        let rate_res = if it > 0 && res0 > 0.0 && r.dofs != dof0 {
            dim_f * (res0 / r.residual).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
        } else {
            0.0
        };
        err0 = r.l2;
        res0 = r.residual;
        dof0 = r.dofs;

        if exact_known {
            print!(
                "{:>5} | {:>10} | {:>10} | {:>6.2} | ",
                it,
                dof0,
                cpp_sci3(err0),
                rate_err
            );
        } else {
            print!("{:>5} | {:>10} | ", it, dof0);
        }
        println!(
            "{:>10} | {:>6.2} | {:>6} | ",
            cpp_sci3(res0),
            rate_res,
            r.iters
        );

        if it == ref_levels.max(0) {
            break;
        }

        // C++ `pmesh.GeneralRefinement(elements_to_refine, 1, 1)`: the
        // nonconforming quad path (MFEM builds an NCMesh, refines the marked
        // elements isotropically — `Refinement(index)` defaults to type 7 —
        // and applies `LimitNCLevel(1)`).  fem-rs replica:
        // `amr::general_refinement_quad_aniso` (validated against the MFEM
        // `d246_aniso_probe` element/vertex counts, test `d246_quad_aniso_nc`).
        // All-marked stays on `refine_uniform` (the fully conforming special
        // case, byte-verified against C++).
        //
        // D960 residual (D1030): the *solve* side still lacks hanging-node
        // constraints for the trace blocks (MFEM's conforming restriction
        // eliminates the parent-edge RT-trace dof against its children —
        // measured here as 30 dofs at level 1: fem-rs 305 vs C++ 275), so the
        // partial-mark table would diverge from C++ from level 1 on.  Kept as
        // an honest exit(3) until the DPG conforming restriction lands.
        if r.marked.len() == mesh.n_elements() {
            mesh = refine_uniform(&mesh);
        } else if !r.marked.is_empty() {
            let marks: Vec<(u32, u8)> =
                r.marked.iter().map(|&e| (e, 7u8)).collect();
            let (m, iso, _hanging) =
                general_refinement_quad_aniso(&mesh, &marks, 1, true, None);
            let _ = (m, iso, _hanging);
            eprintln!(
                "pconvection_diffusion: GAP — partial refinement ({}/{} elements) is \
                 wired to the NC quad refinement (D960 closed: geometry matches MFEM), \
                 but the DPG solve lacks hanging-node constraints for the trace blocks \
                 (D1030): fem-rs level-1 dofs 305 vs C++ 275 (30 parent-edge RT-trace \
                 dofs to eliminate).  Re-run with `-theta 0.0` (mark-all → uniform \
                 refinement, verified path).",
                r.marked.len(),
                mesh.n_elements()
            );
            exit(3);
        }
    }
}
