//! # Parallel Example 16 — Nonlinear heat equation  [1:1 translation of MFEM ex16p]
//!
//! Solves `du/dt = M⁻¹(−Ku)` with nonlinear diffusion `K = ∇·((κ + αu)∇)` on
//! `star.mesh`, pure Neumann BCs, SDIRK33 implicit time integration (the
//! diffusion operator is linearized with the lagged solution, so every stage
//! is one linear solve of `T = M + dt·K` with CG + Jacobi).
//!
//! Parallel layout follows the pex14/pex36 template: serial mesh construction +
//! refinement, `partition_mesh`, `par_uniform_refine` (2-D), DofManager-aware
//! P2 parallel space, local assembly → permute → `from_local_matrix`, and
//! parallel CG with the MFEM `(B r, r)` convergence criterion.
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex16_parallel_nonlinear_heat -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex16_parallel_nonlinear_heat -- --ranks 4 -no-vis
//! ```

use std::path::Path;
use std::sync::Arc;

use fem_assembly::postproc::coefficient::GridFunctionCoeff;
use fem_io::mfem::read_mfem_file;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_mesh::Mesh;
use fem_parallel::{
    ParAssembler, ParCsrMatrix, ParVector, ParallelFESpace, WorkerConfig,
    launcher::native::ThreadLauncher, par_partition::partition_mesh,
    par_refine::par_uniform_refine,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace};

// ─── Initial temperature (C++ ex16.cpp:364-374) ────────────────────────────

fn initial_temperature(x: &[f64]) -> f64 {
    if x[0] * x[0] + x[1] * x[1] < 0.25 { 2.0 } else { 1.0 }
}

// ─── SDIRK33 coefficients (MFEM ode.cpp SDIRK33Solver::Step) ───────────────

const SDIRK33_A: f64 = 0.435866521508458999416019;
const SDIRK33_B: f64 = 1.20849664917601007033648;
const SDIRK33_C: f64 = 0.717933260754229499708010;

// ─── Parallel CG + Jacobi with the MFEM (B r, r) criterion ──────────────────
//
// C++ ex16p uses CGSolver + HypreSmoother(Jacobi) with rel_tol 1e-8,
// iterative_mode=false.  MFEM PCG wraps the tolerance: the inner test is
// (B r, r) <= rtol²·(B r0, r0).

struct PcgJacobiResult {
    converged: bool,
    iterations: usize,
}

fn par_solve_cg_jacobi_mfem(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    rtol: f64,
    max_iter: usize,
) -> PcgJacobiResult {
    let n = a.n_owned();
    let inv_diag: Vec<f64> = a
        .diagonal()
        .iter()
        .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
        .collect();

    // r = b − A·x ; z = D⁻¹ r ; d = z ; nom = (r, z)
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    let mut r = b.clone_vec();
    r.axpy(-1.0, &ax);
    let mut z = ParVector::zeros_like(b);
    for i in 0..n {
        z.as_slice_mut()[i] = r.as_slice()[i] * inv_diag[i];
    }
    let mut d = z.clone_vec();
    let mut nom = r.global_dot(&z);
    let nom0 = nom;
    let target = (nom * rtol * rtol).max(0.0);

    let mut ap = ParVector::zeros_like(b);
    let mut iter = 0usize;
    if nom <= target {
        return PcgJacobiResult { converged: true, iterations: 0 };
    }

    loop {
        // ap = A·d
        a.spmv(&mut d, &mut ap);
        let pap = d.global_dot(&ap);
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = nom / pap;

        // x += alpha·d ; r −= alpha·ap
        x.axpy(alpha, &d);
        r.axpy(-alpha, &ap);

        // z = D⁻¹ r
        for i in 0..n {
            z.as_slice_mut()[i] = r.as_slice()[i] * inv_diag[i];
        }
        let nom_new = r.global_dot(&z);
        let beta = nom_new / nom;
        nom = nom_new;

        // d = z + beta·d
        for i in 0..n {
            d.as_slice_mut()[i] = z.as_slice()[i] + beta * d.as_slice()[i];
        }

        iter += 1;
        if nom <= target || iter >= max_iter {
            return PcgJacobiResult { converged: nom <= target, iterations: iter };
        }
    }
    PcgJacobiResult { converged: false, iterations: iter }
}

// ─── Parallel conduction operator ───────────────────────────────────────────

struct ParConductionOperator {
    ps: ParallelFESpace<H1Space<Mesh<2>>>,
    m_mat: ParCsrMatrix,
    k_mat: ParCsrMatrix,
    t_mat: Option<ParCsrMatrix>,
    current_dt: f64,
    alpha: f64,
    kappa: f64,
    quad_order: u8,
    z: ParVector,
}

impl ParConductionOperator {
    fn new(
        ps: ParallelFESpace<H1Space<Mesh<2>>>,
        alpha: f64,
        kappa: f64,
        u_par: &ParVector,
        comm: &fem_parallel::Comm,
        quad_order: u8,
    ) -> Self {
        // M = ∫ v·u  (C++: MassIntegrator, Assemble(0) keeps the sparsity
        // pattern identical to K's so M + dt·K is a structure-preserving add).
        let m_integ = MassIntegrator { rho: 1.0 };
        let m_mat = ParAssembler::assemble_bilinear(&ps, &[&m_integ], quad_order);

        let n_owned = ps.dof_partition().n_owned_dofs;
        let ghost = ps.dof_ghost_exchange_arc();
        let k_mat = ParCsrMatrix::from_local_matrix(
            &fem_linalg::CsrMatrix::new_empty(n_owned, n_owned),
            n_owned,
            ghost,
            comm.clone(),
        );
        let mut oper = ParConductionOperator {
            ps,
            m_mat,
            k_mat,
            t_mat: None,
            current_dt: 0.0,
            alpha,
            kappa,
            quad_order,
            z: ParVector::zeros_like(u_par),
        };
        oper.set_parameters(u_par);
        oper
    }

    /// Build T = M + dt·K (structure-preserving add on diag + offd blocks).
    fn implicit_solve(&mut self, dt: f64, u: &ParVector, k: &mut ParVector) {
        if self.t_mat.is_none() {
            let m = &self.m_mat;
            let kk = &self.k_mat;
            let diag = m.diag_block().axpby(1.0, kk.diag_block(), dt);
            let offd = m.offd_block().axpby(1.0, kk.offd_block(), dt);
            let n_owned = m.n_owned();
            let ghost = m.ghost_exchange_arc();
            let comm = m.comm().clone();
            let t = ParCsrMatrix::from_blocks(
                diag,
                offd,
                n_owned,
                m.n_ghost(),
                ghost,
                comm,
            );
            self.t_mat = Some(t);
            self.current_dt = dt;
        }
        assert!(
            (dt - self.current_dt).abs() < 1e-15,
            "ImplicitSolve: dt changed ({:.4e} vs {:.4e})",
            dt,
            self.current_dt
        );

        // RHS = −K·u
        let mut u_mut = u.clone_vec();
        self.k_mat.spmv(&mut u_mut, &mut self.z);
        let n = self.ps.dof_partition().n_owned_dofs;
        for i in 0..n {
            self.z.as_slice_mut()[i] = -self.z.as_slice()[i];
        }

        // Solve T·k = RHS (C++: T_solver, CG + Jacobi, rel_tol 1e-8, max_iter 100)
        let t = self.t_mat.as_ref().unwrap();
        let _ = par_solve_cg_jacobi_mfem(t, &self.z, k, 1e-8, 100);
    }

    /// K = ∇·((κ + αu)∇) with the lagged solution u (C++ SetParameters).
    fn set_parameters(&mut self, u_par: &ParVector) {
        // u_alpha = κ + α·u as a nodal grid-function coefficient in DofManager
        // order (GridFunctionCoeff indexes element_dofs into this vector).
        let mut u_synced = u_par.clone_vec();
        u_synced.update_ghosts();
        let dp = self.ps.dof_partition();
        let n_dm = dp.n_total_dofs();
        let mut u_dm = vec![0.0; n_dm];
        for p in 0..n_dm {
            u_dm[dp.unpermute_dof(p as u32) as usize] = u_synced.as_slice()[p];
        }
        let u_alpha: Vec<f64> = u_dm.iter().map(|&v| self.kappa + self.alpha * v).collect();
        let u_coeff = GridFunctionCoeff::new(u_alpha);
        let k_integ = DiffusionIntegrator { kappa: u_coeff };
        self.k_mat = ParAssembler::assemble_bilinear(&self.ps, &[&k_integ], self.quad_order);
        self.t_mat = None;
    }
}

// ─── MFEM-mesh reader (same as serial ex16) ─────────────────────────────────

fn read_star_mesh() -> Mesh<2> {
    let path = {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"));
        p.parent().unwrap().parent().unwrap().join("data/star.mesh")
    };
    let mfem = read_mfem_file(&path).expect("failed to read data/star.mesh");
    mfem.mesh2d.expect("star.mesh must be 2D")
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ser_ref: usize = parse_arg(&args, "-rs").unwrap_or(2) as usize;
    let par_ref: usize = parse_arg(&args, "-rp").unwrap_or(1) as usize;
    let order: u8 = parse_arg(&args, "-o").unwrap_or(2) as u8;
    let t_final: f64 = parse_arg_f64(&args, "-tf").unwrap_or(0.5);
    let dt: f64 = parse_arg_f64(&args, "-dt").unwrap_or(1.0e-2);
    let alpha: f64 = parse_arg_f64(&args, "-a").unwrap_or(1.0e-2);
    let kappa: f64 = parse_arg_f64(&args, "-k").unwrap_or(0.5);
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(1) as usize;
    let visualization = !args.iter().any(|a| a == "-no-vis" || a == "--no-visualization");

    println!("Options used:");
    println!("   --mesh ../data/star.mesh");
    println!("   --refine-serial {}", ser_ref);
    println!("   --refine-parallel {}", par_ref);
    println!("   --order {}", order);
    println!("   --ode-solver 23");
    println!("   --t-final {}", cpp_fmt(t_final));
    println!("   --time-step {}", cpp_fmt(dt));
    println!("   --alpha {}", cpp_fmt(alpha));
    println!("   --kappa {}", cpp_fmt(kappa));
    println!("   {}", if visualization { "--visualization" } else { "--no-visualization" });

    // Serial mesh + refinement (C++ steps 3-5).
    let mesh = read_star_mesh();
    let mesh = if ser_ref > 0 {
        let mut m = mesh;
        for _ in 0..ser_ref {
            m = fem_mesh::refine_uniform(&m);
        }
        m
    } else {
        mesh
    };
    let mesh = Arc::new(mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let is_root = rank == 0;

        // ── Partition + parallel refinement (2-D: supported) ───────────────
        let mut par_mesh = partition_mesh(&mesh, &comm);
        for _ in 0..par_ref {
            par_mesh = par_uniform_refine(&par_mesh);
        }
        let local_mesh = par_mesh.local_mesh().clone();

        // ── H1 space (P2 needs the DofManager-aware parallel path) ──────────
        let local_space = H1Space::new(local_mesh, order);
        let dm = local_space.dof_manager().clone();
        let ps = ParallelFESpace::new_with_dof_manager(
            local_space, &par_mesh, &dm, comm.clone(),
        );
        let dp = ps.dof_partition().clone();
        let n_owned = dp.n_owned_dofs;
        let fe_size = ps.n_global_dofs();
        if is_root {
            println!("Number of temperature unknowns: {}", fe_size);
        }

        // ── Initial condition u₀ (dm order) → partition order ──────────────
        let dim = 2;
        let mut u0_dm = vec![0.0; dm.n_dofs];
        for i in 0..dm.n_dofs {
            let x = dm.dof_coord(i as u32);
            u0_dm[i] = initial_temperature(&x[..dim]);
        }
        let u0_part = fem_parallel::par_assembler::permute_vec(&u0_dm, &dp);
        let mut u_par = ParVector::from_local_raw(
            u0_part, n_owned, ps.dof_ghost_exchange_arc(), comm.clone(),
        );

        // ── Conduction operator (C++: ConductionOperator oper(fespace, ...)) ──
        let quad_order = 2 * order;
        let mut oper = ParConductionOperator::new(ps, alpha, kappa, &u_par, &comm, quad_order);

        // ── Time integration (SDIRK33; C++ ex16p.cpp:290-331) ──────────────
        let n = n_owned;
        let mut t = 0.0;
        let vis_steps = 5;
        let mut last_step = false;
        let mut ti = 1usize;
        while !last_step {
            if t + dt >= t_final - dt / 2.0 {
                last_step = true;
            }

            let a = SDIRK33_A;
            let b = SDIRK33_B;
            let c = SDIRK33_C;

            // k = ImplicitSolve(a·dt, x)
            let mut k = ParVector::zeros_like(&u_par);
            oper.implicit_solve(a * dt, &u_par, &mut k);

            // y = x + (c−a)·dt·k ; x += b·dt·k
            let mut y = ParVector::zeros_like(&u_par);
            y.copy_from(&u_par);
            y.axpy((c - a) * dt, &k);
            u_par.axpy(b * dt, &k);

            // k = ImplicitSolve(a·dt, y) ; x += (1−a−b)·dt·k
            oper.implicit_solve(a * dt, &y, &mut k);
            u_par.axpy((1.0 - a - b) * dt, &k);

            // k = ImplicitSolve(a·dt, x) ; x += a·dt·k
            oper.implicit_solve(a * dt, &u_par, &mut k);
            u_par.axpy(a * dt, &k);

            t += dt;

            if last_step || ti % vis_steps == 0 {
                if is_root {
                    println!("step {}, t = {}", ti, cpp_fmt(t));
                }
            }

            // Update K with the new solution (lagged linearization).
            oper.set_parameters(&u_par);

            ti += 1;
        }

        // ── Final metrics (partition-invariant) ─────────────────────────────
        let local_norm: f64 = u_par.owned_slice().iter().map(|v| v * v).sum();
        let sol_norm = comm.allreduce_sum_f64(local_norm).sqrt();
        let local_checksum: f64 = (0..n)
            .map(|p| (dp.global_dof(p as u32) as f64 + 1.0) * u_par.as_slice()[p])
            .sum();
        let checksum = comm.allreduce_sum_f64(local_checksum);

        // ── Optional dump of the final solution in DofManager order (np1:
        //    equals the MFEM H1_2D_P2 dof order) for C++ comparison ─────────
        if std::env::var("PEX16_DUMP").as_deref() == Ok("1") && rank == 0 {
            let mut u_synced = u_par.clone_vec();
            u_synced.update_ghosts();
            let n_dm = dp.n_total_dofs();
            let mut u_dm = vec![0.0; n_dm];
            for p in 0..n_dm {
                u_dm[dp.unpermute_dof(p as u32) as usize] = u_synced.as_slice()[p];
            }
            let mut buf = String::new();
            for v in &u_dm {
                buf.push_str(&format!("{:.15e}\n", v));
            }
            std::fs::write("output/pex16_rust_np1.sol", buf).expect("write dump");
        }

        if is_root {
            println!();
            println!("=== Comparison Metrics ===");
            println!("DOFs: {}", fe_size);
            println!("Steps: {}", ti - 1);
            println!("Final t: {:.6e}", t);
            println!("L2norm = {:.6e}", sol_norm);
            println!("chksum = {:.6e}", checksum);
            println!("kappa = {:.3}", kappa);
            println!("alpha = {:.3}", alpha);
            println!("order = {}", order);
            println!("dt = {:.4e}", dt);
            println!("t_final = {:.4e}", t_final);
            println!("=========================");
            println!("\nDone.");
        }
    });
}

fn parse_arg(args: &[String], name: &str) -> Option<i64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_arg_f64(args: &[String], name: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

/// C++ `cout.precision(8)` defaultfloat formatting (trailing zeros stripped).
fn cpp_fmt(v: f64) -> String {
    let p = 8usize;
    let sci = format!("{:.7e}", v);
    let (mant, exp) = sci.split_once('e').expect("sci format");
    let exp: i32 = exp.parse().expect("exp");
    let neg = mant.starts_with('-');
    let mant = mant.trim_start_matches('-');
    let mut digits: Vec<char> = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    while digits.len() > 1 && digits[digits.len() - 1] == '0' {
        digits.pop();
    }
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if exp >= -4 && exp < p as i32 {
        if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if int_len >= digits.len() {
                out.push_str(&digits.iter().collect::<String>());
                out.push_str(&"0".repeat(int_len - digits.len()));
            } else {
                out.push_str(&digits[..int_len].iter().collect::<String>());
                out.push('.');
                out.push_str(&digits[int_len..].iter().collect::<String>());
            }
        } else {
            out.push('0');
            out.push('.');
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits.iter().collect::<String>());
        }
    } else {
        out.push(digits[0]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..].iter().collect::<String>());
        }
        out.push('e');
        if exp < 0 {
            out.push('-');
        } else {
            out.push('+');
        }
        let e = exp.abs();
        if e < 10 {
            out.push('0');
        }
        out.push_str(&e.to_string());
    }
    out
}
