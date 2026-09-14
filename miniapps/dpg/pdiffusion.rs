//! Parallel ultraweak DPG solver for the Poisson problem — 1:1 port of MFEM's
//! `miniapps/dpg/pdiffusion.cpp` (MFEM 4.10).
//!
//! Solves `-Δu = f` in Ω, `u = u₀` on ∂Ω through the first-order system
//! `∇u − σ = 0`, `−∇·σ = f` with skeleton traces `û ∈ H^{1/2}`, `σ̂ ∈
//! H^{-1/2}`:
//!
//! ```text
//!     -(u, ∇·τ) - (σ, τ)  + < û, τ·n > = 0,     ∀ τ ∈ H(div)
//!      (σ, ∇v)           + < σ̂, v    > = (f,v), ∀ v ∈ H¹
//! ```
//!
//! Trial spaces (C++ `pdiffusion.cpp`): `u ∈ L²(p−1)`, `σ ∈ (L²(p−1))²`,
//! `û ∈ H¹-trace(p)`, `σ̂ ∈ RT-trace(p−1)`; broken test spaces `τ ∈
//! RT(p+δ−1)`, `v ∈ H¹(p+δ)` with the space-induced test norm
//! `‖∇·τ‖² + ‖τ‖² + ‖∇v‖² + ‖v‖²`, solved with `ParDPGWeakForm`
//! ([`fem_parallel::par_dpg_weakform::ParDpgWeakForm`]).
//!
//! Problem cases: `-prob 0` (manufactured, default) `u = sin(π(x+y))`,
//! `f = −Δu`, `û|∂Ω = u`; `-prob 1` (L-shape benchmark, `f = 0`,
//! `u = r^{2/3} sin(2φ/3)` on `data/l-shape.mesh` rotated as in the C++
//! miniapp).
//!
//! Printed table matches the C++ miniapp:
//! `Ref | Dofs | L2 Error | Rate | Residual | Rate | PCG it`.
//!
//! # Verified against the C++ MPI reference
//!
//! Built from `$HOME/mfem410_mpi` (`mpicxx … pdiffusion.cpp util/*.cpp
//! ../common/*_extras.cpp libmfem.a -lHYPRE -lmetis -lrt`) and run under
//! `mpirun -np {1,2} … -theta 0.0 -no-vis`.  Every printed number of the
//! convergence table — `Dofs`, `L2 Error`, `Residual` — matches the fem-rs run
//! **to all printed digits**:
//!
//! ```text
//!                                     C++                fem-rs
//! -prob 0 -sref 0 -np1:  113  1.021e+00  9.951e-01   ==  113  1.021e+00  9.951e-01
//! -prob 0 -sref 0 -np2:  113  1.021e+00  9.951e-01   ==  113  1.021e+00  9.951e-01
//! -prob 0 -sref 1 -np1:  417  5.149e-01  5.115e-01   ==  417  5.149e-01  5.115e-01
//! -prob 0 -sref 1 -np2:  417  5.149e-01  5.115e-01   ==  417  5.149e-01  5.115e-01
//! -prob 0 -sref 0 -sc -np1: 113 1.021e+00 9.951e-01   ==  113  1.021e+00  9.951e-01
//! -prob 0 -sref 0 -sc -np2: 113 1.021e+00 9.951e-01   ==  113  1.021e+00  9.951e-01
//! -prob 1 -sref 0 -np1:   27  4.755e-01  5.539e-01   ==   27  4.755e-01  5.539e-01
//! -prob 1 -sref 0 -np2:   27  4.755e-01  5.539e-01   ==   27  4.755e-01  5.539e-01
//! -prob 1 -sref 1 -np1:   89  2.780e-01  3.240e-01   ==   89  2.780e-01  3.240e-01
//! -prob 1 -sref 1 -np2:   89  2.780e-01  3.240e-01   ==   89  2.780e-01  3.240e-01
//! ```
//!
//! `-prob 0` is the manufactured `u = sin(π(x+y))` problem on
//! `data/inline-quad.mesh`, `-prob 1` the L-shape benchmark on
//! `data/l-shape.mesh` (rotated exactly as the C++ miniapp does).
//!
//! The **PCG iteration count is not reproduced**: MFEM applies a Hypre
//! `GSSmoother` per block of the *parallel* matrix (including the off-diagonal
//! ghost columns), fem-rs applies symmetric GS to the owned block of each
//! diagonal block only, and MFEM's own count already differs between `-np 1`
//! and `-np 2` (29 vs 32) for the identical system, so it is not a
//! partition-independent quantity.  The residual printed by the miniapp is the
//! *DPG* residual `‖B x − F‖_{S⁻¹}`, which is reproduced exactly.
//!
//! # Known gaps (exit code 3)
//!
//! * `-pref > 0` (parallel AMR driven by the DPG residual indicator): the
//!   non-conforming parallel refinement + repartition + `ParDPGWeakForm::Update`
//!   path is not wired (`fem_parallel` has `par_refine_marked*`, but the
//!   identity-node partitioning the DPG trace numbering requires is not
//!   rebuilt by it).  The run prints the gap and exits 3.
//! * `-pmg` (P-refinement multigrid): `PRefinementMultigrid` is not ported;
//!   exits 3.
//! * `-paraview`: no ParaView writer for this path; ignored with a notice.
//! * GLVis visualization is not implemented (C++ `-vis`); `-no-vis` matches.
//!
//! Usage:
//!   cargo run --release --example pdiffusion -- --ranks 2 -sref 1
//!   cargo run --release --example pdiffusion -- --ranks 1 -m data/inline-quad.mesh

use std::sync::{Arc, Mutex};

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgDomainLFIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTraceIntegrator, DpgVectorFEMassIntegrator,
};
use fem_assembly::dpg_weakform::DpgBlockGs;
use fem_mesh::{refine_uniform, ElementType, Mesh, MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_dpg_weakform::ParDpgWeakForm;
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_solve_pcg_precond;
use fem_parallel::WorkerConfig;
use fem_solver::SolverConfig;

const PI: f64 = std::f64::consts::PI;

/// Problem case — C++ `enum prob_type { manufactured, lshape }`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prob {
    Manufactured,
    Lshape,
}

impl Prob {
    fn name(self) -> &'static str {
        match self {
            Prob::Manufactured => "manufactured",
            Prob::Lshape => "lshape",
        }
    }
}

fn sum_x(x: &[f64]) -> f64 {
    x.iter().sum()
}

/// `exact_u` of `pdiffusion.cpp`.
fn exact_u(x: &[f64], prob: Prob) -> f64 {
    match prob {
        Prob::Lshape => {
            let (px, py) = (x[0], x[1]);
            let r = (px * px + py * py).sqrt();
            let alpha = 2.0 / 3.0;
            let mut phi = py.atan2(px);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            r.powf(alpha) * (alpha * phi).sin()
        }
        Prob::Manufactured => (PI * sum_x(x)).sin(),
    }
}

/// `exact_gradu` of `pdiffusion.cpp`.
fn exact_gradu(x: &[f64], prob: Prob) -> Vec<f64> {
    match prob {
        Prob::Lshape => {
            let (px, py) = (x[0], x[1]);
            let r = (px * px + py * py).sqrt();
            let alpha = 2.0 / 3.0;
            let mut phi = py.atan2(px);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            let r_x = px / r;
            let r_y = py / r;
            let phi_x = -py / (r * r);
            let phi_y = px / (r * r);
            let beta = alpha * r.powf(alpha - 1.0);
            let s = (alpha * phi).sin();
            let c = (alpha * phi).cos();
            vec![
                beta * (r_x * s + r * phi_x * c),
                beta * (r_y * s + r * phi_y * c),
            ]
        }
        Prob::Manufactured => {
            let a = PI * sum_x(x);
            vec![PI * a.cos(); x.len()]
        }
    }
}

/// `exact_sigma` = `∇u`.
fn exact_sigma(x: &[f64], prob: Prob) -> Vec<f64> {
    exact_gradu(x, prob)
}

/// `exact_laplacian_u`.
fn exact_laplacian_u(x: &[f64], prob: Prob) -> f64 {
    match prob {
        Prob::Manufactured => {
            let a = PI * sum_x(x);
            -PI * PI * (a.sin()) * x.len() as f64
        }
        Prob::Lshape => unreachable!("f_exact is not called for the L-shape problem"),
    }
}

fn f_exact(x: &[f64], prob: Prob) -> f64 {
    -exact_laplacian_u(x, prob)
}

/// `-(σ, τ)` adapter — MFEM `TransposeIntegrator(VectorFEMassIntegrator(-1))`
/// on (σ: vector-L2 trial, τ: H(div) test).
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

/// C++-style `std::scientific` with 3 digits (`1.021e+00`, `9.951e-01`).
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

/// One refinement level: assemble, solve, return `(dofs, l2_error, residual,
/// iterations)`.
struct LevelResult {
    dofs: usize,
    l2: f64,
    residual: f64,
    iters: usize,
}

fn solve_level(
    mesh: &Mesh<2>,
    n_workers: usize,
    order: u8,
    delta_order: u8,
    prob: Prob,
    static_cond: bool,
) -> LevelResult {
    let p: u8 = order;
    let test_order: u8 = order + delta_order;
    let p_us = p as usize;
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

        let mut a = ParDpgWeakForm::new(local_mesh, partition, comm.clone());
        a.set_quad_order((2 * test_order as usize).min(255) as u8);
        a.set_face_quad_order(test_order + p - 1);

        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space_h1(p);
        let hatsig = a.add_trial_trace_space(p - 1);

        let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
        let v = a.add_test_space(VolKind::Scalar, test_order);

        // Trial integrators (MFEM pdiffusion.cpp block table).
        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
            u,
            tau,
        ); // -(u, ∇·τ)
        a.add_trial_integrator(Box::new(NegVectorMass), sig, tau); // -(σ, τ)
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v); // (σ, ∇v)
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau); // <û, τ·n>
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v); // -<σ̂, v>

        // Test integrators (space-induced norm for H(div) × H1).
        a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
        a.add_test_integrator(Box::new(DpgVectorFEMassIntegrator { q: 1.0 }), tau, tau);
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
        a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);

        if prob == Prob::Manufactured {
            a.add_domain_lf_integrator(
                Box::new(DpgDomainLFIntegrator {
                    f: move |x: &[f64]| f_exact(x, prob),
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
        // `hatu_fes->GetEssentialTrueDofs(ess_bdr, ...)` with all boundary
        // attributes).  `trace_boundary_dofs` returns absolute global ids
        // plus the physical DOF points; the merged list is consistent on all
        // ranks because a global boundary face is a boundary face on every
        // rank holding it.
        let pairs = a.trace_boundary_dofs(hatu);
        let merged = a.merge_dof_points(&pairs);
        let ess_ids: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();

        let mut x_local = vec![0.0_f64; a.local().size()];
        a.fill_essential_values(&mut x_local, &merged, &move |pt: &[f64]| exact_u(pt, prob));

        let (sys, x0, _) = a.form_linear_system(&ess_ids, &x_local);

        // Block-diagonal symmetric-GS preconditioner on the owned segment
        // (MFEM `BlockDiagonalPreconditioner` of block `GSSmoother`s).
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
            .expect("pdiffusion: PCG failed");
        let x_owned = xv.as_slice()[..sys.n_owned].to_vec();
        let x_full = a.recover_fem_solution(&x_owned);

        // Residual ‖residuals‖₂ over the owned elements (global reduction).
        let residual = a.global_residual_norm(&x_full);

        // L2 error of (u, σ) over the owned elements.
        let (e_u, e_s) = l2_errors(&a, &x_full, u, sig, p_us.saturating_sub(1) as u8, prob);
        let l2 = comm.allreduce_sum_f64(e_u + e_s).max(0.0).sqrt();

        if rank == 0 {
            *result_slot.lock().expect("pdiffusion mutex") = Some(LevelResult {
                dofs: a.n_global_trial_dofs(),
                l2,
                residual,
                iters: res.iterations,
            });
        }
    });

    let out = result
        .lock()
        .expect("pdiffusion mutex after launch")
        .take()
        .expect("rank 0 did not publish the pdiffusion result");
    out
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

    let (mut su, mut ss) = (0.0_f64, 0.0_f64);
    for e in 0..mesh.n_elements() as u32 {
        if part.elem_owner[e as usize] != rank {
            continue;
        }
        for (q, xi) in qpts.iter().enumerate() {
            let (det, xp) = if simp {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, mesh.element_nodes(e));
                (tr.det_j(), tr.map_to_physical(xi))
            } else {
                let geo = geo_ref_elem_from_mesh(mesh, e).expect("pdiffusion: geo elem");
                let gnodes = mesh.geometry_nodes(e).to_vec();
                let (_, det, xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
                (det, xp)
            };
            fe.eval_basis(xi, &mut phi);
            let w = qwts[q] * det.abs();
            // u (scalar L2 block).
            let base = offsets[u_block] + e as usize * n;
            let mut uh = 0.0;
            for (i, &p) in phi.iter().enumerate() {
                uh += x_full[base + i] * p;
            }
            su += w * (uh - exact_u(&xp, prob)).powi(2);
            // σ (vector L2 block, byNODES ordering).
            let sbase = offsets[sig_block] + e as usize * n * 2;
            let ex = exact_sigma(&xp, prob);
            for c in 0..2 {
                let mut sh = 0.0;
                for (i, &p) in phi.iter().enumerate() {
                    sh += x_full[sbase + c * n + i] * p;
                }
                ss += w * (sh - ex[c]).powi(2);
            }
        }
    }
    (su, ss)
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_workers: usize = parse_arg(&args, &["--ranks"]).unwrap_or(2);
    let mut mesh_file: String =
        parse_arg(&args, &["--mesh-path"]).unwrap_or_else(|| "data/inline-quad.mesh".into());
    let order: i32 = parse_arg(&args, &["-o", "--order"]).unwrap_or(1);
    let delta_order: i32 = parse_arg(&args, &["-do", "--delta_order"]).unwrap_or(1);
    let sref: i32 = parse_arg(&args, &["-sref", "--num-serial-refinements"]).unwrap_or(0);
    let pref: i32 = parse_arg(&args, &["-pref", "--num-parallel-refinements"]).unwrap_or(0);
    let iprob: i32 = parse_arg(&args, &["-prob", "--problem"]).unwrap_or(0);
    let static_cond = has_flag(&args, &["-sc", "--static-condensation"]);
    let pmg = has_flag(&args, &["-pmg", "--p-refinement-multigrid"]);
    if let Some(i) = args.iter().position(|a| a == "-m" || a == "--mesh") {
        if let Some(v) = args.get(i + 1) {
            mesh_file = v.clone();
        }
    }
    let prob = if iprob > 0 { Prob::Lshape } else { Prob::Manufactured };
    if prob == Prob::Lshape {
        mesh_file = "data/l-shape.mesh".into();
    }

    println!("=== fem-rs pdiffusion: parallel ultraweak DPG for the Poisson problem ===");
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --order {order}");
    println!("   --delta_order {delta_order}");
    println!("   --num-serial-refinements {sref}");
    println!("   --num-parallel-refinements {pref}");
    println!("   --problem {iprob} ({})", prob.name());
    println!("   --theta-factor 0 (uniform serial refinement only)");
    println!("   --ranks {n_workers}");
    println!("   --no-visualization");

    if pmg {
        eprintln!(
            "pdiffusion: GAP — `-pmg` (PRefinementMultigrid) is not ported to fem-rs. \
             The C++ miniapp builds a p-multigrid preconditioner over the trial spaces \
             (`util/preconditioners.cpp:PRefinementMultigrid`); fem-rs has no \
             p-prolongation operators for DPG trace/volume blocks."
        );
        std::process::exit(3);
    }
    if pref > 0 {
        eprintln!(
            "pdiffusion: GAP — `-pref {pref}` (parallel AMR driven by the DPG residual \
             indicator) is not ported.  `ParDPGWeakForm::Update()` + \
             `ParMesh::GeneralRefinement` require a non-conforming parallel refinement \
             that keeps the identity node numbering the DPG trace numbering depends on; \
             `fem_parallel::par_refine_marked*` rebuilds the partition with compact node \
             ids.  Re-run with `-pref 0` (the C++ default) for the verified path."
        );
        std::process::exit(3);
    }

    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("pdiffusion: cannot read {mesh_file}: {e}"));
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("pdiffusion: mesh must be 2-D");
    if prob == Prob::Lshape {
        // C++ rotates the L-shape mesh: (x, y) → (2y − 1, −2x + 1).
        for v in 0..mesh.n_nodes() {
            let (x, y) = (mesh.coords[2 * v], mesh.coords[2 * v + 1]);
            mesh.coords[2 * v] = 2.0 * y - 1.0;
            mesh.coords[2 * v + 1] = -2.0 * x + 1.0;
        }
    }
    for _ in 0..sref {
        mesh = refine_uniform(&mesh);
    }

    if prob == Prob::Manufactured {
        println!("\n  Ref |    Dofs    |  L2 Error  |  Rate  |  Residual  |  Rate  | PCG it |");
        println!("{}", "-".repeat(72));
    }
    let order = order.max(1) as u8;
    let delta_order = delta_order.max(0) as u8;

    let mut err0 = 0.0_f64;
    let mut res0 = 0.0_f64;
    let mut dof0 = 0usize;

    for it in 0..=pref.max(0) {
        let r = solve_level(&mesh, n_workers, order, delta_order, prob, static_cond);
        let rate_err = if it > 0 && err0 > 0.0 && r.dofs != dof0 {
            2.0 * (err0 / r.l2).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
        } else {
            0.0
        };
        let rate_res = if it > 0 && res0 > 0.0 && r.dofs != dof0 {
            2.0 * (res0 / r.residual).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
        } else {
            0.0
        };
        err0 = r.l2;
        res0 = r.residual;
        dof0 = r.dofs;
        println!(
            "{:>5} | {:>10} | {:>10} | {:>6.2} | {:>10} | {:>6.2} | {:>6} | ",
            it,
            r.dofs,
            cpp_sci3(r.l2),
            rate_err,
            cpp_sci3(r.residual),
            rate_res,
            r.iters
        );
    }
}
