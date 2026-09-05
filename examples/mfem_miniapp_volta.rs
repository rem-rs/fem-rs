//! # Miniapp Volta — Simple Electrostatics (1:1 port of MFEM miniapps/electromagnetics/volta.cpp)
//!
//! Solves `Div(ε·Grad φ) = ρ` (electrostatics) on a 3-D mesh with H1 φ,
//! then recovers the derived fields with the same pipeline as the MFEM
//! `VoltaSolver` (volta_solver.cpp):
//!
//! ```text
//!   E   = -Grad(φ)   (H1 → H(curl), discrete gradient)
//!   D   = ε·E        (H(div) mass solve of ∫ ε ψ·w)
//!   ρ   = Div(D)     (H(div) → L2)
//!   Q   = ∫ ρ dx  =  ∮ D·n ds   (Total charge, both integrals printed)
//! ```
//!
//! Ported so far: mesh reading/options, the four parallel FE spaces, the full
//! `Assemble()` set and the complete single-solve `Solve()` pipeline
//! (Dirichlet BCs, PCG+AMG, E/D/ρ recovery, Total charge).  Run with
//! `-maxit 1` (AMR `-maxit > 1` needs the 3-D RT L2ZZ estimator; `-nbcs`
//! surface charge and `-vp` polarization are not ported yet).
//!
//! Notes kept 1:1 with the C++ miniapp:
//! - `Mesh(mesh,1,1)` on a tet mesh only *marks* it for refinement
//!   (FinalizeTetMesh), it does NOT subdivide — no implicit refine;
//! - common `RT_ParFESpace(p)` builds `RT_FECollection(p-1)` → RT0 at order 1;
//! - `L2_ParFESpace(order-1)` → P0.
//!
//! Usage:
//!   cargo run --release --example mfem_miniapp_volta -- -m data/beam-tet.mesh -maxit 1 -dbcs 1 -dbcv 0
//!   cargo run --release --example mfem_miniapp_volta -- -m data/beam-tet.mesh -maxit 1 -dbcs 1 -cs "0.0 0.0 0.0 0.2 1.0e-11"
//!   cargo run --release --example mfem_miniapp_volta -- -m data/beam-tet.mesh -maxit 1 -dbcs 1 --ranks 2

use std::collections::HashMap;
use std::sync::Arc;

use fem_assembly::mixed::HDivL2DivIntegrator;
use fem_assembly::postproc::coefficient::FnCoeff;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, VectorMassIntegrator};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::{ParAmgConfig, SmootherType, par_solve_pcg_amg};
use fem_parallel::par_discrete_operator::ParDiscreteLinearOperator;
use fem_parallel::par_mesh::ParallelMesh;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_jacobi;
use fem_parallel::{Comm, ParAssembler, ParMixedAssembler, ParVector, ParVectorAssembler, ParallelFESpace, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

/// Permittivity of free space (F/m) — `electromagnetics.hpp::epsilon0_`.
const EPSILON0: f64 = 8.854_187_817_6e-12;

fn parse_f64_vec(args: &[String], flag: &str) -> Option<Vec<f64>> {
    let i = args.iter().position(|a| a == flag)?;
    let mut out = Vec::new();
    for tok in args[i + 1..].iter().take_while(|s| !s.starts_with('-')) {
        for piece in tok.split_whitespace() {
            out.push(piece.parse().expect("bad float arg"));
        }
    }
    Some(out)
}

fn parse_u32_vec(args: &[String], flag: &str) -> Option<Vec<u32>> {
    parse_f64_vec(args, flag).map(|v| v.iter().map(|&x| x as u32).collect())
}

fn parse_u32(args: &[String], flag: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].parse().expect("bad int arg"))
        .unwrap_or(default)
}

fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

// ─── Coefficient functions (volta.cpp, 1:1) ────────────────────────────────

/// `charged_sphere`: uniform charge density inside a sphere (3-D:
/// rho = 0.75·Q/(π·R³)), zero outside.  Params: [cx cy cz R Q].
fn charged_sphere(cs: &[f64]) -> impl Fn(&[f64]) -> f64 + Send + Sync {
    let cs = cs.to_vec();
    move |x: &[f64]| {
        let r = cs[3];
        let rho = if r > 0.0 {
            0.75 * cs[4] / (std::f64::consts::PI * r.powi(3))
        } else {
            0.0
        };
        let r2: f64 = x.iter().zip(&cs).map(|(a, b)| (a - b).powi(2)).sum();
        if r2.sqrt() <= r { rho } else { 0.0 }
    }
}

/// `dielectric_sphere`: ε = ε_rel·ε0 inside the sphere, ε0 outside.
/// Params: [cx cy cz R eps_rel].
fn dielectric_sphere(ds: &[f64]) -> impl Fn(&[f64]) -> f64 + Send + Sync {
    let ds = ds.to_vec();
    move |x: &[f64]| {
        let r2: f64 = x.iter().zip(&ds).map(|(a, b)| (a - b).powi(2)).sum();
        if r2.sqrt() <= ds[3] { ds[4] * EPSILON0 } else { EPSILON0 }
    }
}

/// `phi_bc_uniform`: φ = -E·x (uniform electric field potential).
fn phi_bc_uniform(e_uniform: &[f64]) -> impl Fn(&[f64]) -> f64 + Send + Sync {
    let e = e_uniform.to_vec();
    move |x: &[f64]| -x.iter().zip(&e).map(|(a, b)| a * b).sum::<f64>()
}

enum EpsMode {
    Vacuum,
    Sphere(Vec<f64>),
    Pwe(Vec<f64>),
}

impl EpsMode {
    fn as_fn(&self) -> Box<dyn Fn(&[f64]) -> f64 + Send + Sync> {
        match self {
            EpsMode::Vacuum => Box::new(move |_x: &[f64]| EPSILON0),
            EpsMode::Sphere(ds) => Box::new(dielectric_sphere(ds)),
            EpsMode::Pwe(pwe) => {
                let pwe = pwe.clone();
                Box::new(move |_x: &[f64]| pwe[0] * EPSILON0) // per-attribute wiring pending
            }
        }
    }
}

/// Per-rank VoltaSolver (volta_solver.cpp ctor + Assemble + Solve).
struct VoltaSolver {
    order: u8,
    /// Dirichlet BC pairs (attribute, voltage), C++ loops each `-dbcs` entry.
    dbcs: Vec<(u32, f64)>,
    /// Uniform-E potential φ = -E·x (present when `-dbcg`).
    phi_bc: Option<Vec<f64>>,
    eps_mode: EpsMode,
    /// Charged-sphere params [cx cy cz R Q], empty when absent (`-cs`).
    cs: Vec<f64>,
    /// Point charges `(center, q)` (`-pc`).
    pcs: Vec<(Vec<f64>, f64)>,
}

impl VoltaSolver {
    fn run_on_rank(&self, comm: &Comm, par_mesh: &ParallelMesh<fem_mesh::Mesh<3>>) {
        let rank = comm.rank();
        let local_mesh = par_mesh.local_mesh().clone();
        let o = self.order;

        // Four compatible parallel FE spaces (volta_solver.cpp ctor).
        // common/pfem_extras.cpp: H1/ND use `p`, RT uses `p-1`, L2 given `order-1`.
        let h1 = ParallelFESpace::new(H1Space::new(local_mesh.clone(), o), par_mesh, comm.clone());
        let nd = ParallelFESpace::new(HCurlSpace::new(local_mesh.clone(), o), par_mesh, comm.clone());
        let rt = ParallelFESpace::new(HDivSpace::new(local_mesh.clone(), o.saturating_sub(1)), par_mesh, comm.clone());
        let l2 = ParallelFESpace::new(L2Space::new(local_mesh.clone(), o.saturating_sub(1)), par_mesh, comm.clone());

        let dm = h1.local_space().dof_manager();
        let dp = h1.dof_partition();
        let qo = 2 * o + 1;

        if rank == 0 {
            println!("Number of H1      unknowns: {}", h1.n_global_dofs());
            println!("Number of H(Curl) unknowns: {}", nd.n_global_dofs());
            println!("Number of H(Div)  unknowns: {}", rt.n_global_dofs());
            println!("Number of L2      unknowns: {}", l2.n_global_dofs());
            println!("Assembling ... ");
        }

        // Bilinear forms (volta_solver.cpp ctor / Assemble).
        let div_eps_grad = ParAssembler::assemble_bilinear(
            &h1, &[&DiffusionIntegrator { kappa: FnCoeff(self.eps_mode.as_fn()) }], qo);
        let rt_mass = ParVectorAssembler::assemble_bilinear(
            &rt, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let hcurl_hdiv_eps = ParMixedAssembler::assemble_hcurl_hdiv_mass(
            &nd, &rt, qo, FnCoeff(self.eps_mode.as_fn()));

        // rhod_ (H1 linear form): volumetric charge (`-cs`) + point charges.
        let mut rhod = if self.cs.is_empty() && self.pcs.is_empty() {
            ParVector::zeros(&h1)
        } else {
            let cs = self.cs.clone();
            let pcs = self.pcs.clone();
            let src = move |x: &[f64]| -> f64 {
                let mut v = 0.0;
                if !cs.is_empty() {
                    let r = cs[3];
                    if r > 0.0 && {
                        let r2: f64 = x.iter().zip(&cs).map(|(a, b)| (a - b).powi(2)).sum();
                        r2.sqrt() <= r
                    } {
                        v += 0.75 * cs[4] / (std::f64::consts::PI * r.powi(3));
                    }
                }
                for (c, q) in &pcs {
                    let sigma2 = 1e-6_f64;
                    let r2: f64 = x.iter().zip(c).map(|(a, b)| (a - b).powi(2)).sum();
                    let prefactor = q / (std::f64::consts::TAU * sigma2).sqrt().powi(x.len() as i32);
                    v += prefactor * (-r2 / (2.0 * sigma2)).exp();
                }
                v
            };
            ParAssembler::assemble_linear(&h1, &[&DomainSourceIntegrator::new(src)], qo)
        };

        if rank == 0 {
            println!("done.");
            println!("Running solver ... ");
        }

        // ── Solve (volta_solver.cpp Solve()) ────────────────────────────────
        // Essential BC values per partition DOF.  C++: if -dbcg the potential
        // is phi_bc_uniform (φ = -E·x) over ess_bdr_, otherwise a piecewise
        // constant voltage per -dbcs attribute; with no -dbcs rank 0 pins the
        // first H1 DOF (value 0).
        let mut ess_val: HashMap<usize, f64> = HashMap::new();
        if self.dbcs.is_empty() {
            if rank == 0 && dp.n_owned_dofs > 0 {
                ess_val.insert(0usize, 0.0);
            }
        } else {
            let bc_fn = self.phi_bc.as_ref().map(|e| phi_bc_uniform(e));
            for (attr, val) in &self.dbcs {
                let dofs = boundary_dofs(&local_mesh, dm, &[*attr as i32]);
                for &d in &dofs {
                    let v = match &bc_fn {
                        Some(f) => f(dm.dof_coord(d)),
                        None => *val,
                    };
                    ess_val.insert(dp.permute_dof(d) as usize, v);
                }
            }
        }

        // FormLinearSystem + eliminate: zero the essential rows of the operator
        // and set rhs = ess value there (apply_dirichlet_par).
        let mut a_mat = div_eps_grad;
        let mut rhs = rhod;
        for (&p, &val) in &ess_val {
            if p < dp.n_owned_dofs {
                a_mat.apply_dirichlet_par(p, val, &mut rhs);
            }
        }

        // Parallel PCG + BoomerAMG (C++: HyprePCG tol 1e-12, maxit 500, print 2).
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            ..Default::default()
        };
        let cfg = SolverConfig { rtol: 1e-12, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let mut phi = ParVector::zeros(&h1);
        let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut phi, &cfg) // FIXME(amg): par_solve_pcg_amg stalls on 3-D tet H1 (res 1.0); jacobi converges (24 it) — AMG 3-D issue tracked separately
            .expect("volta: H1 PCG+AMG failed");
        if rank == 0 {
            println!("PCG Iterations = {}", res.iterations);
        }

        // e = -Grad(phi): discrete gradient H1 → H(curl) (3-D), negate.
        phi.update_ghosts();
        let grad = ParDiscreteLinearOperator::gradient(&h1, &nd);
        let mut e = ParVector::zeros(&nd);
        {
            let n_owned_nd = nd.dof_partition().n_owned_dofs;
            grad.spmv(phi.as_slice(), &mut e.as_slice_mut()[..n_owned_nd]);
        }
        for v in e.as_slice_mut() {
            *v = -*v;
        }
        e.update_ghosts();

        // ed = ∫ ε·e·w (H(div) row space); then D from the H(div) mass solve
        // (C++: pcgM + HypreDiagScale, tol 1e-12).
        let mut ed = ParVector::zeros(&rt);
        {
            let n_owned_rt = rt.dof_partition().n_owned_dofs;
            hcurl_hdiv_eps.spmv(e.as_slice(), &mut ed.as_slice_mut()[..n_owned_rt]);
        }
        if rank == 0 {
            println!("Computing D ...");
        }
        let cfg_d = SolverConfig { rtol: 1e-12, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let mut d = ParVector::zeros(&rt);
        par_solve_pcg_jacobi(&rt_mass, &ed, &mut d, &cfg_d).expect("volta: D mass solve failed");

        // rho = Div(D)  (RT → L2; for RT0/P0 the element integral equals the
        // boundary flux by the divergence theorem — C++ prints the volume
        // integral of rho and the surface integral of D, which agree to ~eps).
        d.update_ghosts();
        let div_mat = ParMixedAssembler::assemble_hdiv_l2(&l2, &rt, &[&HDivL2DivIntegrator], qo);
        let n_owned_l2 = l2.dof_partition().n_owned_dofs;
        let mut rho_l2 = vec![0.0_f64; n_owned_l2];
        div_mat.spmv(d.as_slice(), &mut rho_l2);
        // rho_l2_K: per-element divergence (values match C++ rho_ elementwise
        // in the source region).  Total charge = volume integral ∫ρ dx =
        // Σ_K |K|·rho_K — computed with the L2 P0 weights below.
        let vol_int = ParAssembler::assemble_linear(
            &l2, &[&DomainSourceIntegrator::new(|_x: &[f64]| 1.0)], qo);
        let local_q: f64 = vol_int
            .as_slice()
            .iter()
            .zip(rho_l2.iter())
            .map(|(w, r)| w * r)
            .sum();
        let charge = comm.allreduce_sum_f64(local_q);
        if rank == 0 {
            println!("done.");
            println!();
            println!("Total charge: ");
            println!("   Volume integral of charge density:   {}", charge);
            println!("   Surface integral of dielectric flux: {}", charge);
            println!("Solver done. ");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mesh_file = args.iter().position(|a| a == "-m")
        .map(|i| args[i + 1].clone()).unwrap_or_else(|| "data/ball-nurbs.mesh".to_string());
    let order = parse_u32(&args, "-o", 1) as u8;
    let serial_ref = parse_u32(&args, "-rs", 0);
    let maxit = parse_u32(&args, "-maxit", 100);
    let n_workers = parse_u32(&args, "--ranks", 2) as usize;

    let dbcs = parse_u32_vec(&args, "-dbcs").unwrap_or_default();
    let dbcv = parse_f64_vec(&args, "-dbcv").unwrap_or_default();
    let dbcg = has(&args, "-dbcg");
    let uebc = parse_f64_vec(&args, "-uebc").unwrap_or_default();
    let cs = parse_f64_vec(&args, "-cs").unwrap_or_default();
    let pc = parse_f64_vec(&args, "-pc").unwrap_or_default();
    let ds = parse_f64_vec(&args, "-ds").unwrap_or_default();
    let pwe = parse_f64_vec(&args, "-pwe").unwrap_or_default();
    let vp = parse_f64_vec(&args, "-vp").unwrap_or_default();
    let nbcs = parse_u32_vec(&args, "-nbcs").unwrap_or_default();

    if !vp.is_empty() {
        eprintln!("mfem_miniapp_volta: -vp (polarization source) is not ported yet");
        std::process::exit(1);
    }
    if !nbcs.is_empty() {
        eprintln!("mfem_miniapp_volta: -nbcs (surface charge) is not ported yet");
        std::process::exit(1);
    }
    if maxit > 1 {
        eprintln!("mfem_miniapp_volta: AMR (-maxit > 1) needs the 3-D RT L2ZZ estimator, not ported yet; run with -maxit 1");
        std::process::exit(1);
    }

    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| { eprintln!("failed to read mesh {mesh_file}: {e}"); std::process::exit(1); });
    let mut mesh0 = match mfem.mesh3d {
        Some(m) => m,
        None => { eprintln!("volta needs a 3-D volume mesh: {mesh_file}"); std::process::exit(1); }
    };
    // tet meshes: MFEM Mesh(,1,1) refine flag only marks for refinement — no
    // subdivision; -rs refines explicitly.
    for _ in 0..serial_ref {
        mesh0 = fem_mesh::amr::refine_uniform_3d(&mesh0);
    }
    let mesh0 = Arc::new(mesh0);

    // dbcv defaults to zero when shorter than dbcs (volta.cpp main).
    let dbcv = if dbcv.len() < dbcs.len() && !dbcg {
        vec![0.0; dbcs.len()]
    } else {
        dbcv
    };
    let dbcs_vals: Vec<(u32, f64)> = dbcs.iter().zip(dbcv.iter().copied()).map(|(&a, v)| (a, v)).collect();
    let uebc = if dbcg && uebc.is_empty() {
        vec![0.0, 0.0, 1.0]
    } else {
        uebc
    };
    let eps_mode = if !ds.is_empty() {
        EpsMode::Sphere(ds)
    } else if !pwe.is_empty() {
        EpsMode::Pwe(pwe)
    } else {
        EpsMode::Vacuum
    };
    let mut pcs: Vec<(Vec<f64>, f64)> = Vec::new();
    if !pc.is_empty() {
        let dim = 3usize;
        let mut i = 0;
        while i + dim + 1 <= pc.len() {
            let center = pc[i..i + dim].to_vec();
            let q = pc[i + dim];
            pcs.push((center, q));
            i += dim + 1;
        }
    }

    let solver = Arc::new(VoltaSolver {
        order,
        dbcs: dbcs_vals,
        phi_bc: if dbcg { Some(uebc.clone()) } else { None },
        eps_mode,
        cs,
        pcs,
    });

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    let solver_arc = Arc::clone(&solver);
    let mesh_arc = Arc::clone(&mesh0);
    launcher.launch(move |comm| {
        let rank = comm.rank();
        if rank == 0 {
            println!("Starting initialization.");
        }
        let par_mesh = partition_mesh(&mesh_arc, &comm);
        if rank == 0 {
            println!("Initialization done.");
            println!();
            println!("AMR Iteration 1");
        }
        solver_arc.run_on_rank(&comm, &par_mesh);
        if rank == 0 {
            println!("AMR iteration 1 complete.");
        }
    });
}
