//! # Parallel LOR Solvers Miniapp — port of MFEM `miniapps/solvers/plor_solvers.cpp`
//!
//! Definite Helmholtz problem `u − Δu = f` with the manufactured solution of
//! `lor_mms.hpp` (`u = sin πx sin πy`, inhomogeneous Dirichlet data `u|∂Ω`),
//! discretised with an `H1_FECollection(order, dim, GaussLobatto)` space and
//! preconditioned by the **low-order refined** operator: `P1` on the
//! Gauss-Lobatto refined mesh (`Mesh::MakeRefined`), solved by AMG
//! (MFEM: `LORSolver<HypreBoomerAMG>`), applied through the assumed-constraint
//! LOR → HO permutation (`M⁻¹ = Π · A_LOR⁻¹ · Πᵀ`).
//!
//! 1:1 with the C++ driver:
//!
//! | C++ | here |
//! |---|---|
//! | `Mesh serial_mesh(mesh_file, 1, 1)` + `-rs` refinements | `read_mfem_file` + `refine_uniform` × (`-rs` + `-rp`) |
//! | `ParMesh mesh(MPI_COMM_WORLD, serial_mesh)` + `-rp` | `partition_mesh` (fem-parallel has no parallel uniform refinement, so the refinements run serially up front — same global mesh) |
//! | `ParFiniteElementSpace fes(&mesh, fec)` | `ParallelFESpace::new_with_dof_manager` |
//! | `fes.GlobalTrueVSize()` | `ParallelFESpace::n_global_dofs()` |
//! | `fes.GetBoundaryTrueDofs` | `boundary_dofs` on the rank-local mesh |
//! | `MassIntegrator + DiffusionIntegrator`, `DomainLFIntegrator(f)`, `UseFastAssembly` | `ParAssembler` with the same MMS `f` |
//! | `FormLinearSystem` | MFEM `DIAG_KEEP` elimination (`apply_dirichlet_par_keep_diag` + `apply_ghost_ess_columns`) |
//! | `LORSolver<HypreBoomerAMG>` | `LorH1` + `Assembler` on the refined mesh + `ParAmgHierarchy` |
//! | `CGSolver` (rtol 1e-12, 500 iters) | [`fem_solver::solve_pcg_par_lor`] (same tolerances, plus a *true* residual restart) |
//! | `x.ComputeL2Error(u)` | `GridFunction::compute_l2_error_owned` + `allreduce` |
//!
//! ## Rank count
//!
//! `--ranks N` runs `N` ranks as threads inside one process (`ThreadLauncher`,
//! the repository's convention for the parallel examples); `mpirun -np N` with
//! the `mpi` feature works the same way through `MpiLauncher` — the driver only
//! sees a [`Comm`].
//!
//! ## Sample runs
//!
//! ```text
//! cargo run --release --example plor_solvers -- --ranks 1
//! cargo run --release --example plor_solvers -- --ranks 4 -m data/star.mesh
//! cargo run --release --example plor_solvers -- --ranks 2 -m data/inline-quad.mesh -o 2 -rs 1 -rp 0
//! ```
//!
//! ## Not ported (honest gaps)
//!
//! * `-fe n` / `-fe r` (H(curl) / H(div)): MFEM's `LORSolver<HypreAMS>` /
//!   `<HypreADS>` needs *distributed* AMS/ADS.  fem-rs has the serial LOR-AMS /
//!   LOR-ADS machinery (`fem_assembly::lor_factory`) and linger has serial
//!   `AmsPrecond`/`AdsPrecond`, but no parallel auxiliary-space solver, so the
//!   ND/RT legs have no parallel preconditioner to build here.  Reported and
//!   refused, not faked.
//! * `-fe l` (L² DG): fem-rs has no parallel DG face integrators
//!   (`DGDiffusionIntegrator` interior/boundary face terms).
//! * Partial assembly: `ParAssembler` assembles fully; MFEM's `PA` path is an
//!   implementation detail of the operator action (identical matrix).
//! * GLVis/ParaView output.

use std::cell::RefCell;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::{ParAmgConfig, ParAmgHierarchy};
use fem_parallel::par_assembler::permute_csr;
use fem_parallel::par_csr::ParCsrMatrix;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_vector::ParVector;
use fem_parallel::{Comm, ParAssembler, ParallelFESpace, WorkerConfig};
use fem_solver::par_lor::{solve_pcg_par_lor, ParOperator, ParPrecond};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::dof_manager::DofManager;
use fem_space::lor::LorH1;
use fem_space::{FESpace, H1Space};

/// `lor_mms.hpp`: `u(x) = sin(πx) sin(πy)`.
fn u_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

/// `lor_mms.hpp`: `f = mass_coeff·u + 2π²·u` for the 2-D definite Helmholtz
/// problem `mass_coeff·u − Δu = f` with `mass_coeff = 1`.
fn f_rhs(x: &[f64]) -> f64 {
    let s = (PI * x[0]).sin() * (PI * x[1]).sin();
    s + 2.0 * PI * PI * s
}

#[derive(Clone)]
struct RunArgs {
    mesh_file: String,
    ser_ref_levels: usize,
    par_ref_levels: usize,
    order: u8,
    fe: String,
    ranks: usize,
}

struct RunResult {
    global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    l2_err: f64,
    avg_reduction: f64,
}

/// `A` seen as a [`ParOperator`]: keeps its `ParVector` scratch so the driver's
/// slice interface costs one `O(n)` copy per product, not an allocation.
struct ParOp {
    a: ParCsrMatrix,
    x: RefCell<ParVector>,
    y: RefCell<ParVector>,
}

impl ParOp {
    fn new(a: ParCsrMatrix) -> Self {
        let n = a.n_owned() + a.n_ghost();
        ParOp {
            x: RefCell::new(ParVector::from_local_raw(
                vec![0.0; n],
                a.n_owned(),
                a.ghost_exchange_arc(),
                a.comm().clone(),
            )),
            y: RefCell::new(ParVector::from_local_raw(
                vec![0.0; n],
                a.n_owned(),
                a.ghost_exchange_arc(),
                a.comm().clone(),
            )),
            a,
        }
    }
}

impl ParOperator for ParOp {
    fn n_owned(&self) -> usize {
        self.a.n_owned()
    }
    fn n_total(&self) -> usize {
        self.a.n_owned() + self.a.n_ghost()
    }
    fn is_root(&self) -> bool {
        self.a.comm().is_root()
    }
    fn spmv(&self, x: &[f64], y: &mut [f64]) {
        let mut xb = self.x.borrow_mut();
        xb.as_slice_mut().copy_from_slice(x);
        let mut yb = self.y.borrow_mut();
        self.a.spmv(&mut xb, &mut yb);
        y.copy_from_slice(yb.as_slice());
    }
    fn global_dot(&self, x: &[f64], y: &[f64]) -> f64 {
        let mut xb = self.x.borrow_mut();
        xb.as_slice_mut().copy_from_slice(x);
        let mut yb = self.y.borrow_mut();
        yb.as_slice_mut().copy_from_slice(y);
        xb.global_dot(&yb)
    }
    fn global_norm(&self, x: &[f64]) -> f64 {
        let mut xb = self.x.borrow_mut();
        xb.as_slice_mut().copy_from_slice(x);
        xb.global_norm()
    }
}

/// `M⁻¹`: a V-cycle of the AMG hierarchy built on the LOR operator.
struct ParLorAmg {
    hier: ParAmgHierarchy,
    r: RefCell<ParVector>,
    z: RefCell<ParVector>,
}

impl ParLorAmg {
    fn new(hier: ParAmgHierarchy, reference: &ParCsrMatrix) -> Self {
        let n = reference.n_owned() + reference.n_ghost();
        let mk = || {
            ParVector::from_local_raw(
                vec![0.0; n],
                reference.n_owned(),
                reference.ghost_exchange_arc(),
                reference.comm().clone(),
            )
        };
        ParLorAmg {
            hier,
            r: RefCell::new(mk()),
            z: RefCell::new(mk()),
        }
    }
}

impl ParPrecond for ParLorAmg {
    fn apply_precond(&self, r: &[f64], z: &mut [f64]) {
        let mut rb = self.r.borrow_mut();
        rb.as_slice_mut().copy_from_slice(r);
        let mut zb = self.z.borrow_mut();
        // `vcycle(b, x)` returns `x + M⁻¹(b − A x)`: the result is the V-cycle's
        // action only when it starts from zero.  A stale buffer makes the
        // preconditioner an affine (non-symmetric) map, which silently breaks
        // PCG's conjugacy.
        zb.as_slice_mut().fill(0.0);
        self.hier.vcycle(&rb, &mut zb);
        z.copy_from_slice(zb.as_slice());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut run = RunArgs {
        mesh_file: "data/star.mesh".to_string(),
        ser_ref_levels: 1,
        par_ref_levels: 1,
        order: 3,
        fe: "h".to_string(),
        ranks: 1,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--mesh" => {
                i += 1;
                run.mesh_file = args[i].clone();
            }
            "-rs" | "--refine-serial" => {
                i += 1;
                run.ser_ref_levels = args[i].parse().unwrap_or(1);
            }
            "-rp" | "--refine-parallel" => {
                i += 1;
                run.par_ref_levels = args[i].parse().unwrap_or(1);
            }
            "-o" | "--order" => {
                i += 1;
                run.order = args[i].parse().unwrap_or(3);
            }
            "-fe" | "--fe-type" => {
                i += 1;
                run.fe = args[i].clone();
            }
            "-np" | "--ranks" | "--np" => {
                i += 1;
                run.ranks = args[i].parse().unwrap_or(1);
            }
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            "-d" | "--device" => {
                i += 1; // accepted and ignored (CPU only)
            }
            other => eprintln!("plor_solvers: ignoring unknown option '{other}'"),
        }
        i += 1;
    }

    if run.fe != "h" {
        eprintln!(
            "plor_solvers: -fe {} is not implemented in fem-rs (only 'h'). The \
             parallel H(curl)/H(div) legs need a distributed AMS/ADS \
             (MFEM LORSolver<HypreAMS/HypreADS>); fem-rs has the serial LOR-AMS/ADS \
             kernels and linger only serial AmsPrecond/AdsPrecond.",
            run.fe
        );
        std::process::exit(2);
    }
    if run.order < 2 {
        eprintln!("plor_solvers: -o {} — a low-order space needs no LOR (use -o >= 2)", run.order);
        std::process::exit(2);
    }

    // Serial input mesh: `-rs` serial + `-rp` "parallel" refinements.  fem-parallel
    // has no parallel uniform refinement, so both run serially before the
    // partition; the global mesh is the one C++ sees after its ParMesh
    // refinements.
    let mfem = read_mfem_file(&run.mesh_file)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", run.mesh_file));
    let mut mesh: Mesh<2> = mfem
        .mesh2d
        .unwrap_or_else(|| panic!("{} must be a 2-D mesh", run.mesh_file));
    for _ in 0..(run.ser_ref_levels + run.par_ref_levels) {
        mesh = refine_uniform(&mesh);
    }

    println!(
        "=== fem-rs plor_solvers (H1 P{}) — {} ranks, {} elems ===",
        run.order,
        run.ranks,
        mesh.n_elems()
    );
    if run.ranks == 1 {
        println!("Device configuration: cpu");
    }

    let mesh = Arc::new(mesh);
    let slot: Arc<Mutex<Option<RunResult>>> = Arc::new(Mutex::new(None));
    let slot_out = Arc::clone(&slot);

    let run_closure = Arc::new(run);
    ThreadLauncher::new(WorkerConfig::new(run_closure.ranks)).launch(move |comm| {
        let res = run_rank(&mesh, &comm, &run_closure);
        if comm.is_root() {
            println!("Number of DOFs: {}", res.global_dofs);
            *slot_out.lock().expect("plor result mutex poisoned") = Some(res);
        }
    });

    let outcome = slot.lock().expect("plor result mutex poisoned").take();
    if let Some(res) = outcome {
        println!("Average reduction factor = {:.6}", res.avg_reduction);
        if res.converged {
            println!("PCG converged in {} iterations.", res.iterations);
        } else {
            println!(
                "PCG did NOT converge: {} iterations, true residual = {:.3e}",
                res.iterations, res.final_residual
            );
        }
        println!("L2 error: {:.6e}", res.l2_err);
    }
}

fn run_rank(mesh: &Mesh<2>, comm: &Comm, run: &RunArgs) -> RunResult {
    let order = run.order;
    let pmesh = partition_mesh(mesh, comm);
    let local_mesh = pmesh.local_mesh().clone();
    let dm = DofManager::new(&local_mesh, order);
    let local_space = H1Space::new(local_mesh.clone(), order);
    let par_space = ParallelFESpace::new_with_dof_manager(local_space, &pmesh, &dm, comm.clone());
    let dof_part = par_space.dof_partition();
    let n_owned = dof_part.n_owned_dofs;
    let n_total = dof_part.n_total_dofs();
    assert_eq!(
        n_total,
        par_space.local_space().n_dofs(),
        "the dof partition must cover every rank-local dof"
    );
    let ghost_arc = par_space.dof_ghost_exchange_arc();

    // ── high-order system: M + K, then the MMS load ──────────────────────────
    let qo = (2 * order + 1) as u8;
    let mass = MassIntegrator { rho: 1.0 };
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&mass, &diff], qo);

    let src = DomainSourceIntegrator::new(|x: &[f64]| f_rhs(x));
    // The load rule is the one that reproduces MFEM's result: with `2·order + 2`
    // the L2 error matches the C++ run to 4–6 significant digits (star.mesh P3:
    // 2.502523e-5 vs MFEM 2.50252e-05; inline-quad Q2: 1.930630e-3 vs
    // 1.93092e-3), while the collocated rule `order + 1` moves it to
    // 2.017379e-5 / 1.565363e-3.  The remaining last-digit difference is this
    // rule against MFEM's partial-assembly rule for the load.
    let mut rhs = ParAssembler::assemble_linear(&par_space, &[&src], (2 * order + 2) as u8);

    // ── essential (boundary) dofs, with the MMS Dirichlet data u|∂Ω ──────────
    //
    // `boundary_dofs` runs on the rank-local mesh, and the one-layer ghost
    // overlap does not always contain the boundary face a rank needs: an
    // **owned** edge-interior dof can have its boundary edge only on a
    // neighbour (the owner then leaves the dof free), and a **ghost** one can
    // be missed locally (its column then survives the elimination).  The flags
    // are therefore exchanged in both directions — reverse
    // (`accumulate_ghosts`, ghost → owner) and forward (`update_ghosts`,
    // owner → ghost) — which is what `GetBoundaryTrueDofs` gives MFEM for
    // free.  Measured without it: 2.0e-3 solution drift between np = 1 and
    // np = 2, and 9 un-eliminated LOR couplings at np = 4.
    let bdr_tags = local_mesh.unique_boundary_tags();
    let bc_dofs = boundary_dofs(&local_mesh, &dm, &bdr_tags);
    let mk_mask = |slots: &[u32]| {
        let mut v = ParVector::from_local_raw(
            vec![0.0_f64; n_total],
            n_owned,
            ghost_arc.clone(),
            comm.clone(),
        );
        for &d in slots {
            v.as_slice_mut()[dof_part.permute_dof(d) as usize] = 1.0;
        }
        v
    };
    let ess_dm: Vec<u32> = {
        let mut detected = mk_mask(&bc_dofs);
        detected.accumulate_ghosts();
        let mut owned = mk_mask(&[]);
        {
            let (flags, src) = (owned.as_slice_mut(), detected.as_slice());
            for pid in 0..n_owned {
                if src[pid] > 0.5 {
                    flags[pid] = 1.0;
                }
            }
        }
        owned.update_ghosts();
        let (det, own) = (detected.as_slice(), owned.as_slice());
        (0..n_total)
            .filter(|&i| det[i] > 0.5 || own[i] > 0.5)
            .map(|i| dof_part.unpermute_dof(i as u32))
            .collect()
    };
    let mut owned_ess: Vec<(usize, f64)> = Vec::new();
    let mut ghost_ess: Vec<(usize, f64)> = Vec::new();
    for &d in &ess_dm {
        let val = u_exact(dm.dof_coord(d));
        let pid = dof_part.permute_dof(d);
        if (pid as usize) < n_owned {
            owned_ess.push((pid as usize, val));
        } else {
            ghost_ess.push(((pid as usize) - n_owned, val));
        }
    }

    // ── LOR operator: P1 mass + diffusion on the Gauss-Lobatto refined mesh ──
    let lor = LorH1::<2>::new(&local_mesh, order)
        .unwrap_or_else(|e| panic!("LOR H1 (order {order}) failed: {e}"));
    assert_eq!(
        lor.n_ho(),
        par_space.local_space().n_dofs(),
        "LOR/HO dof counts must agree for the assumed-constraint map"
    );
    let lor_space = H1Space::new(lor.lor_mesh().clone(), 1);
    assert_eq!(
        lor_space.n_dofs(),
        lor.n_lor(),
        "the P1 space on the refined mesh must number the LOR dofs"
    );
    let a_lor_lor = Assembler::assemble_bilinear(&lor_space, &[&mass, &diff], 3);
    // Recover the HO dof numbering (`Π A_LOR Πᵀ`) so the LOR matrix shares the
    // HO distribution and can go through the same partition + halo.
    let mut a_lor_ho = lor.ho_numbering(&a_lor_lor);
    // The LOR preconditioner acts on the eliminated system (MFEM's `LORSolver`
    // receives the essential dofs and its `FormSystemMatrix` does
    // `EliminateRowColDiag` with the diagonal set to 1, keeping it SPD for CG).
    // The elimination runs on the *local* matrix, so the essential columns of
    // ghost dofs are zeroed before the `[owned | ghost]` split — the same
    // effect as `ParCsrMatrix::eliminate_diag_symmetric_with_ghost`.
    for &d in &ess_dm {
        a_lor_ho.eliminate_essential_bc_diag_symmetric(d as usize, 1.0);
    }
    let a_lor_perm = permute_csr(&a_lor_ho, dof_part);
    let a_lor_par =
        ParCsrMatrix::from_local_matrix(&a_lor_perm, n_owned, ghost_arc.clone(), comm.clone());

    // ── eliminate the essential dofs from the HO system ──────────────────────
    for &(pid, val) in &owned_ess {
        a_mat.apply_dirichlet_par_keep_diag(pid, val, &mut rhs);
    }
    if !ghost_ess.is_empty() {
        a_mat.apply_ghost_ess_columns(&ghost_ess, &mut rhs);
    }

    // ── initial guess ────────────────────────────────────────────────────────
    // MFEM's `x.ProjectCoefficient(u_coeff)` fills `x` with the interpolant of
    // the exact solution, but `ParBilinearForm::FormLinearSystem` overrides the
    // serial default with `copy_interior = 0` and keeps only the essential
    // entries (`X.SetSubVectorComplement(ess_tdof_list, 0.0)`).  The C++ run
    // therefore starts from the *boundary values only* — reproduced here, since
    // starting from the interpolant would make r₀ ~10⁵ times smaller and the
    // iteration counts incomparable (measured: rel. r₀ 8.7e-6 vs 0.89).
    let mut x =
        ParVector::from_local_raw(vec![0.0; n_total], n_owned, ghost_arc.clone(), comm.clone());
    for &(pid, val) in &owned_ess {
        x.as_slice_mut()[pid] = val;
    }
    // Ghost entries are refreshed by the first `spmv`'s halo exchange.

    // ── LOR-AMG preconditioned PCG ───────────────────────────────────────────
    // The default (rank-local) aggregation is used: the ghost-aware
    // `ParAmgHierarchy::build_global` was measured *worse* here (star.mesh P3:
    // 91 / 100 iterations at np = 2 / 4 against 69 / 74 for the local one).
    // MFEM's BoomerAMG coarsens across ranks without that penalty (24 → 26),
    // so the residual np-growth of the fem-rs numbers is the inner AMG, not the
    // LOR transfer — `plor_h1_iterations_do_not_grow_with_the_mesh` gates the
    // mesh independence that LOR is responsible for.
    let amg_cfg = ParAmgConfig::default();
    let hier = ParAmgHierarchy::build(&a_lor_par, comm, amg_cfg);
    let a_op = ParOp::new(a_mat.clone_vec());
    let m_op = ParLorAmg::new(hier, &a_lor_par);

    // Diagnostics: the magnitudes the C++ run prints indirectly through its
    // `(B r, r)` column.  Note the collectives are executed on *every* rank and
    // only the printing is root-guarded.
    let (b_n, x_n, ax_n, r_n) = {
        let b = rhs.as_slice();
        let mut ax0 = vec![0.0_f64; n_total];
        a_op.spmv(x.as_slice(), &mut ax0);
        let mut r0 = vec![0.0_f64; n_total];
        for i in 0..n_total {
            r0[i] = b[i] - ax0[i];
        }
        (
            a_op.global_norm(b),
            a_op.global_norm(x.as_slice()),
            a_op.global_norm(&ax0),
            a_op.global_norm(&r0),
        )
    };
    if comm.is_root() {
        println!(
            "|b| = {b_n:.6e}  |x0| = {x_n:.6e}  |A x0| = {ax_n:.6e}  |r0|/|b| = {:.6e}",
            r_n / b_n
        );
    }

    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };
    let first_pr = std::cell::Cell::new(f64::NAN);
    let last_pr = std::cell::Cell::new(f64::NAN);
    let root = comm.is_root();
    let b = rhs.as_slice().to_vec();
    let mut xv = x.as_slice().to_vec();
    let res = {
        let mut it_count = 0usize;
        solve_pcg_par_lor(&a_op, &m_op, &b, &mut xv, &cfg, |k, pr, true_res| {
            if it_count == 0 {
                first_pr.set(pr);
            }
            last_pr.set(pr);
            it_count = k;
            if root {
                // MFEM prints `(B r, r)` with its stream's default 6 significant
                // digits; `{:e}` keeps the small tail values readable (MFEM's
                // `<<` would show `4.5e-17` too).
                println!("   Iteration : {k:3}  (B r, r) = {pr:.6e}   true res = {true_res:.3e}");
            }
        })
        .expect("parallel LOR PCG failed")
    };
    let (first_pr, last_pr) = (first_pr.get(), last_pr.get());
    x.as_slice_mut().copy_from_slice(&xv);

    // ── L2 error of the numerical solution (owned elements only) ─────────────
    x.update_ghosts();
    let mut dm_data = vec![0.0_f64; par_space.local_space().n_dofs()];
    for pid in 0..n_total {
        dm_data[dof_part.unpermute_dof(pid as u32) as usize] = x.as_slice()[pid];
    }
    let n_owned_elems = pmesh.partition().n_owned_elems;
    let local_err2 = {
        let gf = GridFunction::make_ref(par_space.local_space(), &mut dm_data);
        // The rule that reproduces MFEM's `ComputeL2Error` value on these
        // meshes (measured: `2·order` gives 1.607049e-3 / 2.007773e-5 against
        // MFEM's 1.93092e-3 / 2.50252e-5; `2·order + 2` gives 1.930630e-3 /
        // 2.502523e-5).
        let e = gf.compute_l2_error_owned(
            &|x: &[f64]| u_exact(x),
            (2 * order + 2) as u8,
            n_owned_elems as u32,
        );
        e * e
    };
    let l2_err = comm.allreduce_sum_f64(local_err2).sqrt();

    // MFEM's "Average reduction factor" over the preconditioned residual.
    let avg_reduction = if res.iterations > 0 && first_pr.is_finite() && first_pr > 0.0 {
        (last_pr / first_pr).powf(1.0 / res.iterations as f64)
    } else {
        f64::NAN
    };

    RunResult {
        global_dofs: par_space.n_global_dofs(),
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        l2_err,
        avg_reduction,
    }
}
