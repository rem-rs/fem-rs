//! # Miniapp Volta — Simple Electrostatics (1:1 port of MFEM miniapps/electromagnetics/volta.cpp)
//!
//! Solves `Div(ε·Grad φ) = ρ` (electrostatics) on a 3-D mesh with H1 φ,
//! then recovers the derived fields with the same pipeline as the MFEM
//! `VoltaSolver` (volta_solver.cpp):
//!
//! ```text
//!   E   = -Grad(φ)                    (H1 → H(curl), discrete gradient)
//!   D   = ε·E   (+ P)                 (H(div) mass solve of ∫ ε ψ·w)
//!   ρ   = Div(D)                      (H(div) → L2)
//!   Q   = ∫ ρ dx  =  ∮ D·n ds         (Total charge, both integrals printed)
//! ```
//!
//! Command line options mirror `volta.cpp`.  Ported so far: mesh reading +
//! refinement (`-m/-rs`, default `ball-nurbs.mesh`), the four parallel FE
//! spaces (`-o`) and the full `Assemble()` set (H1 diffusion with a possibly
//! spatially-varying ε, H(div) mass, mixed H(curl)×H(div) mass, H1 source
//! linear form from `-cs`/`-pc`).  Still in progress (next rounds): the
//! `Solve()` step (Dirichlet BCs `-dbcs/-dbcv/-dbcg/-uebc`, PCG+AMG, E/D/ρ
//! recovery, Total charge), `-nbcs`, `-vp`, and the AMR loop (`-maxit > 1`,
//! needs the 3-D RT L2ZZ estimator).  Run with `-maxit 1` for the exact C++
//! single-solve path.
//!
//! Rust runs the partition with `ThreadLauncher` (multi-rank threads in one
//! process, like the pex examples).  Usage:
//!   cargo run --release --example mfem_miniapp_volta -- -m data/beam-tet.mesh -maxit 1 -no-vis -no-visit

use std::sync::Arc;

use fem_assembly::postproc::coefficient::FnCoeff;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, VectorMassIntegrator};
use fem_mesh::amr::refine_uniform_3d;
use fem_mesh::{Mesh, refine_uniform};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_mesh::ParallelMesh;
use fem_parallel::{Comm, ParAssembler, ParMixedAssembler, ParVector, ParVectorAssembler, ParallelFESpace, WorkerConfig};
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

/// Permittivity of free space (F/m) — `electromagnetics.hpp::epsilon0_`.
const EPSILON0: f64 = 8.854_187_817_6e-12;

fn parse_f64_vec(args: &[String], flag: &str) -> Option<Vec<f64>> {
    let i = args.iter().position(|a| a == flag)?;
    let vals: Vec<f64> = args[i + 1..]
        .iter()
        .take_while(|s| !s.starts_with('-'))
        .map(|s| s.parse().expect("bad float arg"))
        .collect();
    Some(vals)
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

// ─── Source/boundary coefficient functions (volta.cpp, 1:1) ────────────────

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

/// `dielectric_sphere`: permittivity is `eps_rel·ε0` inside the sphere,
/// ε0 outside.  Params: [cx cy cz R eps_rel].
fn dielectric_sphere(ds: &[f64]) -> impl Fn(&[f64]) -> f64 + Send + Sync {
    let ds = ds.to_vec();
    move |x: &[f64]| {
        let r2: f64 = x.iter().zip(&ds).map(|(a, b)| (a - b).powi(2)).sum();
        if r2.sqrt() <= ds[3] {
            ds[4] * EPSILON0
        } else {
            EPSILON0
        }
    }
}

/// Point charge (fem-rs `DeltaCoefficient` Gaussian approx, σ² = 1e-6):
/// q·δ(x−x₀) → narrow Gaussian with total integral `q`.
fn point_charge_gauss(center: &[f64], scale: f64) -> impl Fn(&[f64]) -> f64 + Send + Sync {
    let center = center.to_vec();
    move |x: &[f64]| {
        let dim = x.len();
        let sigma2 = 1e-6_f64;
        let r2: f64 = x.iter().zip(&center).map(|(a, b)| (a - b).powi(2)).sum();
        let prefactor = scale / (std::f64::consts::TAU * sigma2).sqrt().powi(dim as i32);
        prefactor * (-r2 / (2.0 * sigma2)).exp()
    }
}

/// `phi_bc_uniform`: φ = -E·x (uniform electric field potential).
fn phi_bc_uniform(e_uniform: &[f64]) -> impl Fn(&[f64]) -> f64 + Send + Sync {
    let e = e_uniform.to_vec();
    move |x: &[f64]| -x.iter().zip(&e).map(|(a, b)| a * b).sum::<f64>()
}

/// Permittivity model: vacuum, dielectric sphere (`-ds`) or per-attribute
/// piecewise constant (`-pwe`, attribute wiring pending → treated as the
/// first value for now, see main()).
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
                Box::new(move |_x: &[f64]| pwe[0] * EPSILON0)
            }
        }
    }
}

/// Per-rank VoltaSolver-equivalent (volta_solver.cpp ctor + Assemble()).
struct VoltaSolver {
    order: u8,
    /// Dirichlet BC pairs (attribute, voltage) — C++ loops each `-dbcs` entry.
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
    /// Assemble everything (divEpsGrad, hDivMass, hCurlHDivEps, rhod_).
    /// The `Solve()` step (BCs, PCG+AMG, field recovery, Total charge) is
    /// added in the next round.
    fn assemble_on_rank(&self, comm: &Comm, par_mesh: &ParallelMesh<Mesh<3>>) {
        let rank = comm.rank();
        let local_mesh = par_mesh.local_mesh().clone();
        let o = self.order;

        // Four compatible parallel FE spaces (volta_solver.cpp ctor).
        let h1 = ParallelFESpace::new(H1Space::new(local_mesh.clone(), o), par_mesh, comm.clone());
        let nd = ParallelFESpace::new(HCurlSpace::new(local_mesh.clone(), o), par_mesh, comm.clone());
        let rt = ParallelFESpace::new(HDivSpace::new(local_mesh.clone(), o), par_mesh, comm.clone());
        let l2 = ParallelFESpace::new(L2Space::new(local_mesh.clone(), o.saturating_sub(1)), par_mesh, comm.clone());

        // PrintSizes (C++ GlobalTrueVSize, no hanging constraints at -maxit 1).
        if rank == 0 {
            println!("Number of H1      unknowns: {}", h1.n_global_dofs());
            println!("Number of H(Curl) unknowns: {}", nd.n_global_dofs());
            println!("Number of H(Div)  unknowns: {}", rt.n_global_dofs());
            println!("Number of L2      unknowns: {}", l2.n_global_dofs());
        }

        let qo = 2 * o + 1;
        if rank == 0 {
            println!("Assembling ... ");
        }

        // Bilinear forms (volta_solver.cpp ctor; Assemble() = assemble +
        // finalize per form, done lazily by the assemblers).
        let div_eps_grad = ParAssembler::assemble_bilinear(
            &h1, &[&DiffusionIntegrator { kappa: FnCoeff(self.eps_mode.as_fn()) }], qo);
        let rt_mass = ParVectorAssembler::assemble_bilinear(
            &rt, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let hcurl_hdiv_eps = ParMixedAssembler::assemble_hcurl_hdiv_mass(
            &nd, &rt, qo, FnCoeff(self.eps_mode.as_fn()));

        // rhod_ (H1 linear form): volumetric charge (`-cs`) + point charges.
        let rhod: ParVector = if self.cs.is_empty() && self.pcs.is_empty() {
            ParVector::zeros(&h1)
        } else {
            let cs = self.cs.clone();
            let pcs = self.pcs.clone();
            let src = move |x: &[f64]| -> f64 {
                let mut v = 0.0;
                if !cs.is_empty() {
                    let r = cs[3];
                    let rho = if r > 0.0 {
                        0.75 * cs[4] / (std::f64::consts::PI * r.powi(3))
                    } else {
                        0.0
                    };
                    let r2: f64 = x.iter().zip(&cs).map(|(a, b)| (a - b).powi(2)).sum();
                    if r2.sqrt() <= r { v += rho; }
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
        }
        // Keep the assembled operators alive for the next round's Solve():
        let _ = (&div_eps_grad, &rt_mass, &hcurl_hdiv_eps, &rhod, &l2, &nd, &rt);
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

    // Read the mesh (volta.cpp: `Mesh(mesh_file, 1, 1)` = generate edges +
    // one uniform refinement), before partitioning.
    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| { eprintln!("failed to read mesh {mesh_file}: {e}"); std::process::exit(1); });
    let mut mesh0 = match mfem.mesh3d {
        Some(m) => m,
        None => { eprintln!("volta needs a 3-D volume mesh: {mesh_file}"); std::process::exit(1); }
    };
    // NURBS meshes take the C++ `NURBSext` path (extra refine + SetCurvature)
    // which fem-io does not parse yet — pass a plain 3-D mesh (-m beam-tet.mesh
    // / fichera.mesh) for the 1:1 comparison.
    mesh0 = refine_uniform_3d(&mesh0);
    for _ in 1..serial_ref {
        mesh0 = refine_uniform_3d(&mesh0);
    }
    let mesh0 = Arc::new(mesh0);

    // dbcv defaults to zero when shorter than dbcs (volta.cpp main).
    let dbcv = if dbcv.len() < dbcs.len() && !dbcg {
        vec![0.0; dbcs.len()]
    } else {
        dbcv
    };
    let dbcs_vals: Vec<(u32, f64)> = dbcs.iter().zip(dbcv.iter().copied()).map(|(&a, v)| (a, v)).collect();
    // dbcg: default E = (0, 0, 1).
    let uebc = if dbcg && uebc.is_empty() {
        vec![0.0, 0.0, 1.0]
    } else {
        uebc
    };
    // eps coefficient: -ds > -pwe > vacuum.
    let eps_mode = if !ds.is_empty() {
        EpsMode::Sphere(ds)
    } else if !pwe.is_empty() {
        // NOTE: C++ uses PWConstCoefficient per element attribute; the
        // attribute-to-value wiring is a follow-up (mesh attrs vs -pwe list).
        EpsMode::Pwe(pwe)
    } else {
        EpsMode::Vacuum
    };
    // Point charges, chunked by dim + 1 (3-D).
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
        solver_arc.assemble_on_rank(&comm, &par_mesh);
        if rank == 0 {
            println!("AMR iteration 1 complete.");
        }
    });
}
