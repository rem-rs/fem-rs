//! Block-solvers miniapp — serial cut (1:1 with MFEM `miniapps/solvers/block-solvers.cpp`)
//!
//! Compares solvers for the mixed-Darcy saddle-point system of ex5p:
//!
//! ```text
//!     k·u + ∇p = f,  −∇·u = g  in Ω,  −p = p̄ on ∂Ω
//!     [ M  Bᵀ ] [u] = [f]
//!     [ B   0 ] [p]   [g]
//! ```
//! RT H(div) velocity, L₂ pressure.  Exact solution
//! `u = (−eˣ sin y, −eˣ cos y)`, `p = eˣ sin y`.
//!
//! Solvers (C++ block-solvers.cpp compares five; this serial cut currently
//! wires the Bramble–Pasciak ones):
//! * **BPCG** (`BramblePasciakSolver(use_bpcg=true)` — `fem_solver::bpcg::solve_bpcg`)
//! * regular PCG on the transformed operator (`use_bpcg=false`), planned
//!
//! Bramble–Pasciak preconditioning: SPD `Q` with `M − Q` SPD (`Q_T =
//! q_scaling·λ_min·diag(M_T)` element-wise — MFEM `ConstructMassPreconditioner`,
//! assembled by `assemble_element_q_diag` with element mass matrices of the
//! physical RT space), `N = diag(invQ, 0)`, particular preconditioner
//! `P = cpc·tri` with `cpc = diag(invQ, M1)`, `tri = [[I,0],[B·invQ,−I]]` and
//! `M1` a solver on the Schur complement `S = B·diag(M)⁻¹·Bᵀ`.

use std::time::Instant;

use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{MassIntegrator, VectorMassIntegrator};
use fem_assembly::{Assembler, VectorAssembler};
use fem_element::reference::{ReferenceElement, VectorReferenceElement};
use fem_linalg::{CooMatrix, CsrMatrix, PrintLevel, SolverConfig};
use fem_mesh::transformation::geometry_jacobian;
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_solver::darcy_solvers::{BdpMinresSolver, IterSolveParameters, SchurMode};
use fem_solver::div_free_solver::{
    DfsData, DfsParameters, DivFreeSolver, Ql2Projector,
};
use fem_solver::block::BlockSystem;
use fem_solver::bpcg::solve_bpcg;
use fem_solver::bramble_pasciak::element_q_scaling;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HDivSpace, L2Space};

/// Exact solutions and data (block-solvers.cpp `u_exact`/`p_exact`, 2D).
fn u_exact(x: &[f64]) -> [f64; 2] {
    [-(x[0].exp() * x[1].sin()), -(x[0].exp() * x[1].cos())]
}
fn p_exact(x: &[f64]) -> f64 {
    x[0].exp() * x[1].sin()
}

fn main() {
    let args = parse_args();
    let mesh = fem_io::mfem::read_mfem_file(args.mesh.as_deref().unwrap_or("../data/star.mesh"))
        .expect("mesh read failed")
        .mesh2d
        .unwrap();
    let mut mesh0 = mesh;
    for _ in 0..args.ser_ref_levels {
        mesh0 = refine_uniform(&mesh0);
    }
    // DFS refinement levels (block-solvers.cpp `par_ref_levels`): all solvers
    // act on the mesh refined `rp` more times; the DFS hierarchy coarsens back
    // to `mesh0`.  (C++ default is 1; the serial cut keeps 0 so the BPCG
    // baseline reproduces the historical run verbatim.)
    let mut mesh = mesh0.clone();
    for _ in 0..args.dfs_ref_levels {
        mesh = refine_uniform(&mesh);
    }

    let u_sp = HDivSpace::new(mesh.clone(), args.order);
    let p_sp = L2Space::new(mesh, args.order);
    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    let n = n_u + n_p;

    println!("***********************************************************");
    println!("dim(R) = {n_u}");
    println!("dim(W) = {n_p}");
    println!("dim(R+W) = {n}");
    println!("***********************************************************");

    // ── Assemble (identical to examples/mfem_ex5_mixed_darcy.rs) ──────────
    let qo = (2 * args.order as usize + 1).max(2) as u8;
    // M = ∫ u·v dx  (k = 1 coefficient)
    let m_csr = VectorAssembler::assemble_bilinear(
        &u_sp,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        qo,
    );
    // B = −∫ div(u) q dx
    let mut b_csr = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut b_csr.values {
        *v *= -1.0;
    }

    // RHS: natural BC −p = p_exact → ∫ (−p_exact)(v·n) ds ;  g = 0 (2D)
    let tags: Vec<i32> = u_sp.mesh().unique_boundary_tags();
    let fu = if !tags.is_empty() {
        assemble_bdr_rhs(&u_sp, &tags, args.order as usize + 1, &|x: &[f64]| -p_exact(x))
    } else {
        vec![0.0; n_u]
    };
    let gp = vec![0.0; n_p];
    let mut rhs = Vec::with_capacity(n);
    rhs.extend_from_slice(&fu);
    rhs.extend_from_slice(&gp);

    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter: 1000,
        verbose: false,
        print_level: PrintLevel::Iterations,
    };

    // Flat saddle operator [[M, Bᵀ], [B, 0]] (and its C++-style size print).
    let flat = BlockSystem {
        a: m_csr.clone(),
        bt: b_csr.transpose(),
        b: b_csr.clone(),
        c: None,
    }
    .to_flat_csr();

    let mut x: Vec<f64> = vec![0.0; n];

    match args.solver.as_str() {
        // ── Bramble–Pasciak CG (MFEM BramblePasciakSolver, use_bpcg=true) ────
        "bpcg" => {
            solve_bpcg_mode(&args, &u_sp, &p_sp, &m_csr, &b_csr, &rhs, &cfg);
        }
        // ── Block-diagonal-preconditioned MINRES (BDPMinresSolver) ───────────
        "bdp" => {
            // Diagnostic: dump the assembled Schur complement S = B·diag(M)⁻¹·Bᵀ
            // as index-1-based COO text (same triple format as MatrixMarket
            // coordinate, without the header) for offline spectral analysis.
            if let Some(path) = &args.dump_schur {
                let bt = b_csr.transpose();
                let mut minvbt = CooMatrix::<f64>::new(n_u, n_p);
                for i in 0..n_u {
                    let inv_d = 1.0 / m_csr.get(i, i).max(1e-300);
                    for ptr in bt.row_ptr[i]..bt.row_ptr[i + 1] {
                        let j = bt.col_idx[ptr] as usize;
                        minvbt.add(i, j, bt.values[ptr] * inv_d);
                    }
                }
                let s = b_csr.multiply(&minvbt.into_csr());
                let mut text = String::with_capacity(s.values.len() * 32);
                for i in 0..s.nrows {
                    for ptr in s.row_ptr[i]..s.row_ptr[i + 1] {
                        text.push_str(&format!(
                            "{} {} {:.17e}\n",
                            i + 1,
                            s.col_idx[ptr] as usize + 1,
                            s.values[ptr]
                        ));
                    }
                }
                std::fs::write(path, text).expect("dump-schur write failed");
                eprintln!("[dump-schur] wrote {n_p}x{n_p} S to {path}");
            }
            let schur = match args.schur.as_str() {
                "amg" => SchurMode::Amg,
                "diag" => SchurMode::Diag,
                _ => SchurMode::Dense,
            };
            let param = IterSolveParameters {
                print_level: if args.verbose { 1 } else { 0 },
                max_iter: cfg.max_iter,
                abs_tol: cfg.atol,
                rel_tol: cfg.rtol,
            };
            let line = "*".repeat(58);
            println!("{line}");
            println!("Block-diagonal-preconditioned MINRES solver:");
            let start = Instant::now();
            let solver = BdpMinresSolver::new(&m_csr, &b_csr, param, schur);
            let setup = start.elapsed().as_secs_f64();
            let start = Instant::now();
            solver.mult(&rhs, &mut x);
            let solve = start.elapsed().as_secs_f64();
            if !solver.converged() {
                eprintln!("BDP MINRES did not converge ({} iters)", solver.num_iterations());
            }
            println!("   Setup time: {setup:.6}s.");
            println!("   Solve time: {solve:.6}s.");
            println!("   Total time: {:.6}s.", setup + solve);
            println!("   Iteration count: {}", solver.num_iterations());
            print_errors(&u_sp, &p_sp, &x, args.order);
        }
        // ── Divergence free solver (decoupled / coupled) ─────────────────────
        s_mode @ ("dfs-dec" | "dfs-coupled") => {
            let coupled = s_mode == "dfs-coupled";
            let param = DfsParameters {
                coupled_solve: coupled,
                verbose: args.verbose,
                coarse_solve_param: IterSolveParameters {
                    max_iter: cfg.max_iter,
                    abs_tol: cfg.atol,
                    rel_tol: cfg.rtol,
                    ..IterSolveParameters::default()
                },
                bbt_solve_param: IterSolveParameters {
                    max_iter: cfg.max_iter,
                    abs_tol: cfg.atol,
                    rel_tol: cfg.rtol,
                    ..IterSolveParameters::default()
                },
                outer_solve_param: IterSolveParameters {
                    max_iter: cfg.max_iter,
                    abs_tol: cfg.atol,
                    rel_tol: cfg.rtol,
                    ..IterSolveParameters::default()
                },
                gmres_restart: args.restart,
                coarse_schur_mode: match args.schur.as_str() {
                    "amg" => SchurMode::Amg,
                    "diag" => SchurMode::Diag,
                    _ => SchurMode::Dense,
                },
            };
            let line = "*".repeat(58);
            println!("{line}");
            println!(
                "Divergence free solver ({} mode):",
                if coupled { "coupled" } else { "decoupled" }
            );
            let start = Instant::now();
            let dfs_data = if args.dfs_ref_levels == 0 {
                DfsData::single_level(param)
            } else {
                collect_dfs_data(&mesh0, args.dfs_ref_levels, args.order, param)
            };
            if args.dfs_ref_levels > 0 {
                let dim_ker = dfs_data.c.last().map(|c| c.ncols).unwrap_or(0);
                println!("Dimension of the divergence free subspace: {dim_ker}");
            }
            let solver = DivFreeSolver::new(&m_csr, &b_csr, &dfs_data);
            let setup = start.elapsed().as_secs_f64();
            let start = Instant::now();
            // C++ applies the DFS Mult once as the solver; to reach the same
            // discrete-solution accuracy as the other solvers we run FGMRES
            // with the DFS application as (variable) preconditioner.
            let precond = DfsPreconditioner { dfs: &solver, n };
            let res = fem_solver::solve_fgmres_precond(
                &flat,
                &rhs,
                &mut x,
                args.restart,
                &precond,
                &SolverConfig {
                    rtol: cfg.rtol,
                    atol: cfg.atol,
                    max_iter: cfg.max_iter,
                    verbose: false,
                    print_level: PrintLevel::Silent,
                },
            )
            .map_err(|e| format!("{e:?}"))
            .unwrap();
            let solve = start.elapsed().as_secs_f64();
            println!("   Setup time: {setup:.6}s.");
            println!("   Solve time: {solve:.6}s.");
            println!("   Total time: {:.6}s.", setup + solve);
            println!(
                "   Iteration count: {} (last DFS inner count: {})",
                res.iterations,
                solver.num_iterations()
            );
            if !res.converged {
                eprintln!("DFS-preconditioned FGMRES did not converge");
            }
            print_errors(&u_sp, &p_sp, &x, args.order);
        }
        other => {
            eprintln!("unknown -solver '{other}' (expected bpcg|bdp|dfs-dec|dfs-coupled)");
            std::process::exit(2);
        }
    }
}

/// linlvo `Preconditioner` adapter around one [`DivFreeSolver`] application
/// (always started from a zero iterate — a true fixed linear-in/linear-out
/// operator application, as a Krylov preconditioner requires).
struct DfsPreconditioner<'a> {
    dfs: &'a DivFreeSolver,
    n: usize,
}

impl fem_solver::linlvoPreconditioner for DfsPreconditioner<'_> {
    type Vector = fem_solver::DenseVec<f64>;

    fn apply_precond(&self, x: &Self::Vector, y: &mut Self::Vector) {
        let mut z = vec![0.0; self.n];
        self.dfs.mult(x.as_slice(), &mut z);
        let slice = y.as_mut_slice();
        slice.copy_from_slice(&z);
    }
}

/// L² error print (block-solvers.cpp `ShowError` line format).
fn print_errors(u_sp: &HDivSpace<Mesh<2>>, p_sp: &L2Space<Mesh<2>>, x: &[f64], order: u8) {    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    let order_quad = std::cmp::max(2, 2 * order as usize + 1) as u8;
    let p_gf = GridFunction::new(p_sp, x[n_u..].to_vec());
    let ep = p_gf.compute_l2_error(&p_exact, order_quad);
    let p_zero = GridFunction::new(p_sp, vec![0.0; n_p]);
    let np = p_zero.compute_l2_error(&p_exact, order_quad);
    let eu = compute_hdiv_l2_error_2d(u_sp, &x[..n_u], &u_exact);
    let nu = compute_hdiv_l2_error_2d(u_sp, &vec![0.0; n_u], &u_exact);
    println!("|| u_h - u_ex || / || u_ex || = {:.6e}", eu / nu.max(1e-32));
    println!("|| p_h - p_ex || / || p_ex || = {:.6e}", ep / np.max(1e-32));
}

/// The original serial BPCG cut (kept bit-identical to the historical run).
fn solve_bpcg_mode(
    args: &Args,
    u_sp: &HDivSpace<Mesh<2>>,
    p_sp: &L2Space<Mesh<2>>,
    m_csr: &CsrMatrix<f64>,
    b_csr: &CsrMatrix<f64>,
    rhs: &[f64],
    _cfg: &SolverConfig,
) {
    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    let n = n_u + n_p;
    let m_csr = m_csr.clone();
    let b_csr = b_csr.clone();
    // Element-wise mass preconditioner Q (MFEM `ConstructMassPreconditioner`):
    // per element Q_T = q_scaling·λ_min(M_T, diag(M_T))·diag(M_T), assembled
    // into a global diagonal.  This guarantees M − Q SPD (a global
    // q_scaling·diag(M) does NOT for RT0 — BPCG δ<0 breakdown observed).
    let diag_m: Vec<f64> = (0..n_u).map(|i| m_csr.get(i, i).max(1e-30)).collect();
    let q_diag = assemble_element_q_diag(&u_sp, args.order, args.q_scaling);
    let inv_q: Vec<f64> = q_diag.iter().map(|q| 1.0 / q.max(1e-300)).collect();

    // Schur complement S = B·diag(M)⁻¹·Bᵀ and its Jacobi preconditioner M1.
    let bt = b_csr.transpose();
    let mut minvbt_coo = CooMatrix::<f64>::new(n_u, n_p);
    for i in 0..n_u {
        let inv_d = 1.0 / diag_m[i];
        for ptr in bt.row_ptr[i]..bt.row_ptr[i + 1] {
            let j = bt.col_idx[ptr] as usize;
            minvbt_coo.add(i, j, bt.values[ptr] * inv_d);
        }
    }
    let s_csr = b_csr.multiply(&minvbt_coo.into_csr());
    let s_diag: Vec<f64> = (0..n_p).map(|i| s_csr.get(i, i).max(1e-30)).collect();

    // M1: Schur solver for the p block. Default `diag` = diag(S)⁻¹ (weak);
    // `-m1 dense` = exact S⁻¹ (experiment to isolate M1 strength; C++ uses
    // hypre BoomerAMG on S).
    let m1: Box<dyn Fn(&[f64], &mut [f64])> = if args.m1_mode == "dense" {
        let mut a_d = vec![0.0; n_p * n_p];
        for i in 0..n_p {
            for p in s_csr.row_ptr[i]..s_csr.row_ptr[i + 1] {
                a_d[i * n_p + s_csr.col_idx[p] as usize] = s_csr.values[p];
            }
        }
        let mut lu = a_d.clone();
        let mut piv = vec![0usize; n_p];
        fem_linalg::dense::lu_factor(&mut lu, n_p, &mut piv).expect("Schur S singular");
        let mut inv = vec![vec![0.0; n_p]; n_p];
        for c in 0..n_p {
            let mut e = vec![0.0; n_p];
            e[c] = 1.0;
            fem_linalg::dense::lu_solve(&lu, n_p, &piv, &mut e);
            for r in 0..n_p {
                inv[r][c] = e[r];
            }
        }
        Box::new(move |wp: &[f64], zp: &mut [f64]| {
            for r in 0..n_p {
                zp[r] = (0..n_p).map(|c| inv[r][c] * wp[c]).sum();
            }
        })
    } else {
        Box::new(move |wp: &[f64], zp: &mut [f64]| {
            for r in 0..n_p {
                zp[r] = wp[r] / s_diag[r];
            }
        })
    };

    // Flat saddle operator [[M, Bᵀ], [B, 0]].
    let b_for_p = b_csr.clone(); // kept for apply_p (BlockSystem takes b_csr)
    let flat = BlockSystem {
        a: m_csr,
        bt: b_csr.transpose(),
        b: b_csr,
        c: None,
    }
    .to_flat_csr();
    // Diagnostic: flat symmetry and M-Q SPD.
    {
        let mut sym_err = 0.0f64;
        for i in 0..n {
            for ptr in flat.row_ptr[i]..flat.row_ptr[i+1] {
                let j = flat.col_idx[ptr] as usize;
                let aij = flat.values[ptr];
                let mut aji = 0.0;
                for q in flat.row_ptr[j]..flat.row_ptr[j+1] {
                    if flat.col_idx[q] as usize == i { aji = flat.values[q]; break; }
                }
                sym_err = sym_err.max((aij - aji).abs());
            }
        }
        let mut m_q_min = f64::INFINITY;
        for i in 0..n_u {
            let m_ii = diag_m[i];
            let q_ii = q_diag[i];
            m_q_min = m_q_min.min(m_ii - q_ii);
        }
        eprintln!("[diag] flat |A-Aᵀ|_max = {sym_err:.3e}, min(diag(M-Q)) = {m_q_min:.6}");
    }
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter: 1000,
        verbose: false,
        print_level: PrintLevel::Iterations,
    };

    // ── Solve with Bramble–Pasciak CG ─────────────────────────────────────
    // apply_a: y = A·x (flat [[M,Bᵀ],[B,0]])
    // apply_n: y = N·x = (invQ·x_u, 0)
    // apply_p: y = P·x = cpc·tri·x, cpc = diag(invQ, M1), M1 = diag(S)⁻¹
    let start = Instant::now();
    let mut x = vec![0.0; n];
    let res = solve_bpcg(
        n,
        |v, w| flat.spmv(v, w),
        |v, w| {
            // tri·v = (v_u, B·invQ·v_u − v_p)
            let inv_q_u: Vec<f64> = inv_q
                .iter()
                .zip(&v[..n_u])
                .map(|(q, x)| q * x)
                .collect();
            let mut bp = vec![0.0; n_p];
            b_for_p.spmv(&inv_q_u, &mut bp);
            w[..n_u].copy_from_slice(&v[..n_u]);
            for k in 0..n_p {
                w[n_u + k] = bp[k] - v[n_u + k];
            }
            // cpc·(tri·v) = (invQ·(tri·v)_u, M1·(tri·v)_p)
            for i in 0..n_u {
                w[i] *= inv_q[i];
            }
            // cpc p block: zp = M1·(tri·v)_p
            let wp: Vec<f64> = w[n_u..].to_vec();
            m1(&wp, &mut w[n_u..]);
        },
        |v, w| {
            for i in 0..n_u {
                w[i] = inv_q[i] * v[i];
            }
            for k in 0..n_p {
                w[n_u + k] = 0.0;
            }
        },
        &rhs,
        &mut x,
        &cfg,
    );
    let elapsed = start.elapsed();
    println!("\nBPCG solver took {:.4}s.", elapsed.as_secs_f64());

    let res = match res {
        Ok(r) => r,
        Err(e) => {
            eprintln!("BPCG failed: {e}");
            return;
        }
    };
    if !res.converged {
        eprintln!("BPCG did not converge ({})", res.iterations);
    }

    // ── L² errors (same conventions as ex5) ───────────────────────────────
    print_errors(u_sp, p_sp, &x, args.order);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DFS data collector — serial port of MFEM `DFSSpaces` (div_free_solver.cpp):
// per-level RT/L2/H1 spaces, refinement prolongations, aggregates, discrete
// curl `C` and `Q_l2` (DataFinalize).  All transfer matrices are exact
// projections (the parent restriction lies in the child space for RT_k/L2_k/
// P_k under uniform refinement), so shared dofs get consistent values from
// adjacent patches and each (column, row) is scattered exactly once.
// ═══════════════════════════════════════════════════════════════════════════════

/// Affine map from a fine (child) element's reference coordinates to its
/// parent element's reference coordinates, derived from the standard 1:4
/// midpoint refinement pattern (child vertices are parent corners, edge
/// midpoints, or the quad center).  Exact for `refine_uniform` output.
#[derive(Debug)]
enum ChildToParentRef {
    /// parent ξ = P0 + ξ_child·(P1−P0) + η_child·(P2−P0), where Pk are the
    /// parent reference positions of the child's local vertices (tri).
    Tri([[f64; 2]; 3]),
    /// parent ξ = offset + 0.5·child ξ (quad).
    Quad([f64; 2]),
}

impl ChildToParentRef {
    fn map(&self, xi: &[f64]) -> [f64; 2] {
        match self {
            ChildToParentRef::Quad(off) => [off[0] + 0.5 * xi[0], off[1] + 0.5 * xi[1]],
            ChildToParentRef::Tri(p) => {
                let l1 = xi[0];
                let l2 = xi[1];
                let l0 = 1.0 - l1 - l2;
                [
                    l0 * p[0][0] + l1 * p[1][0] + l2 * p[2][0],
                    l0 * p[0][1] + l1 * p[1][1] + l2 * p[2][1],
                ]
            }
        }
    }
}

/// Build the child→parent reference map by matching the child's corner nodes
/// against the parent's reference landmark positions (corners, edge
/// midpoints, center).
fn child_to_parent_ref(
    cmesh: &Mesh<2>,
    ce: u32,
    fmesh: &Mesh<2>,
    fe: u32,
) -> ChildToParentRef {
    match cmesh.element_type(ce) {
        fem_mesh::ElementType::Quad4 => {
            // Parent reference landmarks: 4 corners, 4 edge midpoints, center.
            let pns = cmesh.element_nodes(ce);
            let pc: Vec<[f64; 2]> = pns
                .iter()
                .map(|&n| cmesh.node_coords(n).try_into().unwrap())
                .collect();
            let landmarks: [([f64; 2], [f64; 2]); 9] = [
                (pc[0], [0.0, 0.0]),
                (pc[1], [1.0, 0.0]),
                (pc[2], [1.0, 1.0]),
                (pc[3], [0.0, 1.0]),
                ([(pc[0][0] + pc[1][0]) * 0.5, (pc[0][1] + pc[1][1]) * 0.5], [0.5, 0.0]),
                ([(pc[1][0] + pc[2][0]) * 0.5, (pc[1][1] + pc[2][1]) * 0.5], [1.0, 0.5]),
                ([(pc[2][0] + pc[3][0]) * 0.5, (pc[2][1] + pc[3][1]) * 0.5], [0.5, 1.0]),
                ([(pc[3][0] + pc[0][0]) * 0.5, (pc[3][1] + pc[0][1]) * 0.5], [0.0, 0.5]),
                ([(pc[0][0] + pc[1][0] + pc[2][0] + pc[3][0]) * 0.25,
                  (pc[0][1] + pc[1][1] + pc[2][1] + pc[3][1]) * 0.25],
                 [0.5, 0.5]),
            ];
            let cns = fmesh.element_nodes(fe);
            let ccoords: Vec<[f64; 2]> =
                cns.iter().map(|&n| fmesh.node_coords(n).try_into().unwrap()).collect();
            const REF_CORNERS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            for off in [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]] {
                let mut ok = true;
                for (k, r) in REF_CORNERS.iter().enumerate() {
                    let target = [off[0] + 0.5 * r[0], off[1] + 0.5 * r[1]];
                    match landmarks.iter().find(|(_, lr)| {
                        (lr[0] - target[0]).abs() < 1e-12 && (lr[1] - target[1]).abs() < 1e-12
                    }) {
                        Some((p, _)) => {
                            if (p[0] - ccoords[k][0]).abs() > 1e-9
                                || (p[1] - ccoords[k][1]).abs() > 1e-9
                            {
                                ok = false;
                                break;
                            }
                        }
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    return ChildToParentRef::Quad(off);
                }
            }
            panic!("child_to_parent_ref: quad child {fe} does not match midpoint refinement");
        }
        fem_mesh::ElementType::Tri3 => {
            let pns = cmesh.element_nodes(ce);
            let pts: Vec<[f64; 2]> =
                pns.iter().map(|&n| cmesh.node_coords(n).try_into().unwrap()).collect();
            let mut landmarks: Vec<([f64; 2], [f64; 2])> = pts
                .clone()
                .into_iter()
                .zip([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                .collect();
            landmarks.push((
                [(pts[0][0] + pts[1][0]) * 0.5, (pts[0][1] + pts[1][1]) * 0.5],
                [0.5, 0.0],
            ));
            landmarks.push((
                [(pts[1][0] + pts[2][0]) * 0.5, (pts[1][1] + pts[2][1]) * 0.5],
                [0.5, 0.5],
            ));
            landmarks.push((
                [(pts[2][0] + pts[0][0]) * 0.5, (pts[2][1] + pts[0][1]) * 0.5],
                [0.0, 0.5],
            ));
            let cns = fmesh.element_nodes(fe);
            let mut pref: [[f64; 2]; 3] = [[0.0; 2]; 3];
            for (k, &cn) in cns.iter().enumerate() {
                let cp: [f64; 2] = fmesh.node_coords(cn).try_into().unwrap();
                let hit = landmarks
                    .iter()
                    .find(|(p, _)| (p[0] - cp[0]).abs() < 1e-9 && (p[1] - cp[1]).abs() < 1e-9)
                    .unwrap_or_else(|| {
                        panic!("child_to_parent_ref: tri child vertex not a landmark")
                    });
                pref[k] = hit.1;
            }
            ChildToParentRef::Tri(pref)
        }
        other => panic!("child_to_parent_ref: unsupported element type {other:?}"),
    }
}

/// Crossing-number point-in-polygon test (works for convex and concave
/// simple polygons; used as the containment pre-filter).
fn point_in_polygon(pts: &[[f64; 2]], x: &[f64]) -> bool {
    let n = pts.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (pts[i][0], pts[i][1]);
        let (xj, yj) = (pts[j][0], pts[j][1]);
        if ((yi > x[1]) != (yj > x[1]))
            && (x[0] < (xj - xi) * (x[1] - yi) / (yj - yi + 1e-300) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Locate the element containing a physical point (bbox + crossing test;
/// the reference coordinates are not needed by the collector).
fn locate_in_mesh(mesh: &Mesh<2>, x: &[f64]) -> Option<(u32, [f64; 2])> {
    for e in mesh.elem_iter() {
        let ns = mesh.element_nodes(e);
        let (mut xmin, mut ymin) = (f64::INFINITY, f64::INFINITY);
        let (mut xmax, mut ymax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &n in ns {
            let c = mesh.node_coords(n);
            xmin = xmin.min(c[0]);
            ymin = ymin.min(c[1]);
            xmax = xmax.max(c[0]);
            ymax = ymax.max(c[1]);
        }
        if x[0] < xmin - 1e-9 || x[0] > xmax + 1e-9 || x[1] < ymin - 1e-9 || x[1] > ymax + 1e-9 {
            continue;
        }
        let poly: Vec<[f64; 2]> = ns
            .iter()
            .map(|&n| mesh.node_coords(n).try_into().expect("2D coords"))
            .collect();
        if point_in_polygon(&poly, x) {
            return Some((e, [0.0, 0.0]));
        }
    }
    None
}

/// Reference element accessor (scalar RT family, 2-D tri/quad).
fn rt_ref_elem(mesh: &Mesh<2>, e: u32, order: usize) -> Box<dyn VectorReferenceElement> {
    use fem_element::raviart_thomas::{QuadRTk, TriRTk};
    match mesh.element_type(e) {
        fem_mesh::ElementType::Tri3 => Box::new(TriRTk::new(order)),
        fem_mesh::ElementType::Quad4 => Box::new(QuadRTk::new(order)),
        other => panic!("rt_ref_elem: unsupported element type {other:?}"),
    }
}

/// Nodal scalar reference element (H1 family) on the SAME reference domains
/// as the mesh geometry (tri [0,1]², quad [0,1]² row-major x-fastest).
/// NOTE: `lagrange::QuadQ1/QuadQ2` live on [-1,1]² and must NOT be mixed with
/// the [0,1]² quad geometry of `fem_mesh`.
fn h1_ref_elem(mesh: &Mesh<2>, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::factory::{QuadQk, TriPk};
    let p = order.max(1) as usize;
    match mesh.element_type(0) {
        fem_mesh::ElementType::Tri3 => Box::new(TriPk::new(p)),
        fem_mesh::ElementType::Quad4 => Box::new(QuadQk::new(p)),
        other => panic!("h1_ref_elem: unsupported element type {other:?}"),
    }
}

/// L2 constant (order-0) reference element: `P0` basis, value ≡ 1.
struct L2P0;
impl ReferenceElement for L2P0 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], values: &mut [f64]) { values[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        grads[0] = 0.0;
        grads[1] = 0.0;
    }
    fn quadrature(&self, _order: u8) -> fem_element::reference::QuadratureRule {
        fem_element::reference::QuadratureRule {
            points: vec![vec![0.5, 0.5]],
            weights: vec![1.0],
        }
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.5, 0.5]] }
}

/// Scalar reference element for the L2 space (P0 for order 0, else the
/// complete/tensor polynomial space matching `L2Space`'s span).
fn l2_ref_elem(mesh: &Mesh<2>, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::factory::{QuadL2GL, TriPk};
    if order == 0 {
        return Box::new(L2P0);
    }
    match mesh.element_type(0) {
        fem_mesh::ElementType::Tri3 => Box::new(TriPk::new(order as usize)),
        fem_mesh::ElementType::Quad4 => Box::new(QuadL2GL::new(order as usize)),
        other => panic!("l2_ref_elem: unsupported element type {other:?}"),
    }
}

/// Patch-projection H(div) prolongation coarse→fine (MFEM `P_hdiv`).
fn build_p_hdiv(
    coarse: &HDivSpace<Mesh<2>>,
    fine: &HDivSpace<Mesh<2>>,
    parents: &[usize],
) -> CsrMatrix<f64> {
    let cmesh = coarse.mesh();
    let fmesh = fine.mesh();
    let ne0 = cmesh.n_elems() as usize;
    let ne1 = fmesh.n_elems() as usize;
    let order = coarse.order() as usize;
    let mut coo = CooMatrix::<f64>::new(fine.n_dofs(), coarse.n_dofs());
    let mut patch_results: Vec<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>)> = Vec::new();

    for c in 0..ne0 {
        let ce = c as u32;
        let children: Vec<u32> = (0..ne1)
            .filter(|&f| parents[f] == c)
            .map(|f| f as u32)
            .collect();
        let c_dofs: Vec<usize> = coarse.element_dofs(ce).iter().map(|&d| d as usize).collect();
        let c_signs = coarse.element_signs(ce).to_vec();
        let nc = c_dofs.len();

        // Patch dofs (deduplicated within the patch; global sign convention).
        let mut patch_ids: Vec<usize> = Vec::new();
        let mut patch_pos: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut occ: Vec<(u32, usize, usize)> = Vec::new();
        for &fe in &children {
            for (i, &d) in fine.element_dofs(fe).iter().enumerate() {
                let d = d as usize;
                let next = patch_ids.len();
                let pi = *patch_pos.entry(d).or_insert(next);
                if pi == next {
                    patch_ids.push(d);
                }
                occ.push((fe, i, pi));
            }
        }
        let np_ = patch_ids.len();
        let mut m_p = vec![0.0_f64; np_ * np_];
        let mut g_p = vec![0.0_f64; np_ * nc];

        let c_ref = rt_ref_elem(cmesh, ce, order);
        let n_c_loc = c_ref.n_dofs();
        let mut c_phi = vec![0.0_f64; n_c_loc * 2];

        for &fe in &children {
            let f_ref = rt_ref_elem(fmesh, fe, order);
            let nf = f_ref.n_dofs();
            let qr = f_ref.quadrature(4);
            let mut f_phi = vec![0.0_f64; nf * 2];
            let f_signs = fine.element_signs(fe).to_vec();
            let c2p = child_to_parent_ref(cmesh, ce, fmesh, fe);
            for (qi, xi) in qr.points.iter().enumerate() {
                let (fjac, _xp) = fem_mesh::element_jacobian_at(fmesh, fe, xi, 2);
                let fdet = fjac.determinant();
                let w = qr.weights[qi] * fdet.abs();
                f_ref.eval_basis_vec(xi, &mut f_phi);
                let pref = c2p.map(xi);
                let (cjac, _) = fem_mesh::element_jacobian_at(cmesh, ce, &pref, 2);
                let cdet = cjac.determinant();
                c_ref.eval_basis_vec(&pref, &mut c_phi);

                let mut f_val: Vec<(usize, f64, f64)> = Vec::new();
                for &(_, i, pi) in occ.iter().filter(|(e2, _, _)| *e2 == fe) {
                    let s = f_signs[i];
                    let px = s * (fjac[(0, 0)] * f_phi[2 * i] + fjac[(0, 1)] * f_phi[2 * i + 1]) / fdet;
                    let py = s * (fjac[(1, 0)] * f_phi[2 * i] + fjac[(1, 1)] * f_phi[2 * i + 1]) / fdet;
                    f_val.push((pi, px, py));
                }
                let mut c_val = vec![(0.0_f64, 0.0_f64); nc];
                for (j, &s) in c_signs.iter().enumerate() {
                    let px = s * (cjac[(0, 0)] * c_phi[2 * j] + cjac[(0, 1)] * c_phi[2 * j + 1]) / cdet;
                    let py = s * (cjac[(1, 0)] * c_phi[2 * j] + cjac[(1, 1)] * c_phi[2 * j + 1]) / cdet;
                    c_val[j] = (px, py);
                }
                for &(pi, px, py) in &f_val {
                    for &(pk, qx, qy) in &f_val {
                        m_p[pi * np_ + pk] += w * (px * qx + py * qy);
                    }
                    for (j, &(cx, cy)) in c_val.iter().enumerate() {
                        g_p[pi * nc + j] += w * (px * cx + py * cy);
                    }
                }
            }
        }

        let mut lu = m_p.clone();
        let mut piv = vec![0_usize; np_];
        fem_linalg::dense::lu_factor(&mut lu, np_, &mut piv).expect("P_hdiv patch mass singular");
        let x_cols: Vec<Vec<f64>> = (0..nc)
            .map(|j| {
                let mut col: Vec<f64> = (0..np_).map(|i| g_p[i * nc + j]).collect();
                fem_linalg::dense::lu_solve(&lu, np_, &piv, &mut col);
                col
            })
            .collect();
        patch_results.push((patch_ids, c_dofs, x_cols));
    }

    // Scatter grouped by global coarse dof column (per-column write-once).
    let mut per_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); coarse.n_dofs()];
    for (patch_ids, c_dofs, x_cols) in &patch_results {
        for (j, &cd) in c_dofs.iter().enumerate() {
            for (i, &dof) in patch_ids.iter().enumerate() {
                if x_cols[j][i].abs() > 0.0 {
                    per_col[cd].push((dof, x_cols[j][i]));
                }
            }
        }
    }
    for (cd, entries) in per_col.into_iter().enumerate() {
        let mut written = std::collections::HashSet::<usize>::new();
        for (dof, v) in entries {
            if written.insert(dof) {
                coo.add(dof, cd, v);
            }
        }
    }
    coo.into_csr()
}

/// L2 prolongation coarse→fine (element-local exact restriction/projection;
/// L2 dofs are element-local so there are no shared-dof conflicts).
fn build_p_l2(
    coarse: &L2Space<Mesh<2>>,
    fine: &L2Space<Mesh<2>>,
    parents: &[usize],
) -> CsrMatrix<f64> {
    let cmesh = coarse.mesh();
    let fmesh = fine.mesh();
    let ne0 = cmesh.n_elems() as usize;
    let ne1 = fmesh.n_elems() as usize;
    let mut coo = CooMatrix::<f64>::new(fine.n_dofs(), coarse.n_dofs());

    for c in 0..ne0 {
        let ce = c as u32;
        let children: Vec<u32> = (0..ne1)
            .filter(|&f| parents[f] == c)
            .map(|f| f as u32)
            .collect();
        let c_dofs: Vec<usize> = coarse.element_dofs(ce).iter().map(|&d| d as usize).collect();
        let nc = c_dofs.len();
        let c_ref = l2_ref_elem(cmesh, coarse.order());
        let n_c_loc = c_ref.n_dofs();
        let mut c_sh = vec![0.0_f64; n_c_loc];

        // Fine patch dofs (L2 dofs are exclusive to their element).
        let mut patch_ids: Vec<usize> = Vec::new();
        let mut occ: Vec<(u32, usize, usize)> = Vec::new();
        for &fe in &children {
            for (i, &d) in fine.element_dofs(fe).iter().enumerate() {
                patch_ids.push(d as usize);
                occ.push((fe, i, patch_ids.len() - 1));
            }
        }
        let np_ = patch_ids.len();
        let mut m_p = vec![0.0_f64; np_ * np_];
        let mut g_p = vec![0.0_f64; np_ * nc];

        for &fe in &children {
            let f_ref = l2_ref_elem(fmesh, fine.order());
            let nf = f_ref.n_dofs();
            let qr = f_ref.quadrature(4);
            let mut f_sh = vec![0.0_f64; nf];
            let c2p = child_to_parent_ref(cmesh, ce, fmesh, fe);
            for (qi, xi) in qr.points.iter().enumerate() {
                let (fjac, _xp) = fem_mesh::element_jacobian_at(fmesh, fe, xi, 2);
                let w = qr.weights[qi] * fjac.determinant().abs();
                let pref = c2p.map(xi);
                c_ref.eval_basis(&pref, &mut c_sh);
                f_ref.eval_basis(xi, &mut f_sh);
                for &(_, i, pi) in occ.iter().filter(|(e2, _, _)| *e2 == fe) {
                    for &(_, k, pk) in occ.iter().filter(|(e2, _, _)| *e2 == fe) {
                        m_p[pi * np_ + pk] += w * f_sh[i] * f_sh[k];
                    }
                    for (j, &csv) in c_sh.iter().enumerate() {
                        g_p[pi * nc + j] += w * f_sh[i] * csv;
                    }
                }
            }
        }

        let mut lu = m_p.clone();
        let mut piv = vec![0_usize; np_];
        fem_linalg::dense::lu_factor(&mut lu, np_, &mut piv).expect("P_l2 patch mass singular");
        for (j, &cd) in c_dofs.iter().enumerate() {
            let mut col: Vec<f64> = (0..np_).map(|i| g_p[i * nc + j]).collect();
            fem_linalg::dense::lu_solve(&lu, np_, &piv, &mut col);
            for (i, &dof) in patch_ids.iter().enumerate() {
                if col[i].abs() > 0.0 {
                    coo.add(dof, cd, col[i]);
                }
            }
        }
    }
    coo.into_csr()
}

/// H1 prolongation coarse→fine: fine nodal values = coarse interpolation
/// evaluated through the parent geometry (exact for P_k/Q_k under uniform
/// refinement).  Shared dofs are written once (identical values everywhere).
fn build_p_h1(
    coarse: &H1Space<Mesh<2>>,
    fine: &H1Space<Mesh<2>>,
    parents: &[usize],
) -> CsrMatrix<f64> {
    let cmesh = coarse.mesh();
    let fmesh = fine.mesh();
    let ne1 = fmesh.n_elems() as usize;
    let mut coo = CooMatrix::<f64>::new(fine.n_dofs(), coarse.n_dofs());
    let c_ref = h1_ref_elem(cmesh, coarse.order());
    let n_c_loc = c_ref.n_dofs();
    let mut c_sh = vec![0.0_f64; n_c_loc];
    let mut written = std::collections::HashSet::<usize>::new();

    for f in 0..ne1 {
        let fe = f as u32;
        let c = parents[f] as u32;
        let c_dofs: Vec<usize> = coarse.element_dofs_u32(c).iter().map(|&d| d as usize).collect();
        let f_dofs: Vec<usize> = fine.element_dofs_u32(fe).iter().map(|&d| d as usize).collect();
        let f_ref = h1_ref_elem(fmesh, fine.order());
        let f_dcoords = f_ref.dof_coords();
        let c2p = child_to_parent_ref(cmesh, c, fmesh, fe);
        for (k, &gd) in f_dofs.iter().enumerate() {
            if !written.insert(gd) {
                continue; // shared dof already interpolated (identical value)
            }
            // Fine dof k sits at reference position f_dcoords[k] (DofManager
            // P1/Q1 dof order == reference basis order); map it to the parent
            // reference domain.
            let pref = c2p.map(&f_dcoords[k]);
            c_ref.eval_basis(&pref, &mut c_sh);
            for (l, &cd) in c_dofs.iter().enumerate() {
                if c_sh[l].abs() > 1e-15 {
                    coo.add(gd, cd, c_sh[l]);
                }
            }
        }
    }
    coo.into_csr()
}

/// Discrete curl `C: H1(order+1) → RT(order)` (MFEM `CurlInterpolator`,
/// 2-D: `curl ψ = (∂yψ, −∂xψ)`).  Element-local L2 pairing; exact because
/// `curl P_{k+1} ⊂ RT_k`.  Shared RT dofs are written exactly once.
fn build_curl(h1: &H1Space<Mesh<2>>, rt: &HDivSpace<Mesh<2>>) -> CsrMatrix<f64> {
    let mesh = rt.mesh();
    let qo = (2 * h1.get_order() as usize + 3).max(4) as u8;
    let mut coo = CooMatrix::<f64>::new(rt.n_dofs(), h1.n_dofs());
    let mut local: Vec<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>)> = Vec::new();

    for e in mesh.elem_iter() {
        let rt_ref = rt_ref_elem(mesh, e, rt.order() as usize);
        let n_loc = rt_ref.n_dofs();
        let h1_ref = h1_ref_elem(mesh, h1.get_order());
        let nh_loc = h1_ref.n_dofs();
        let qr = rt_ref.quadrature(qo);
        let mut phi = vec![0.0_f64; n_loc * 2];
        let mut dsh = vec![0.0_f64; nh_loc * 2];
        let mut m_e = vec![0.0_f64; n_loc * n_loc];
        let mut g_e = vec![0.0_f64; n_loc * nh_loc];
        let dofs: Vec<usize> = rt.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = rt.element_signs(e).to_vec();
        let hdofs: Vec<usize> = h1.element_dofs_u32(e).iter().map(|&d| d as usize).collect();
        for (qi, xi) in qr.points.iter().enumerate() {
            let (jac, _xp) = fem_mesh::element_jacobian_at(mesh, e, xi, 2);
            let det = jac.determinant();
            let w = qr.weights[qi] * det.abs();
            rt_ref.eval_basis_vec(xi, &mut phi);
            h1_ref.eval_grad_basis(xi, &mut dsh);
            let (a, b, c, d) = (jac[(0, 0)], jac[(0, 1)], jac[(1, 0)], jac[(1, 1)]);
            for i in 0..n_loc {
                let s = signs[i];
                let px = s * (a * phi[2 * i] + b * phi[2 * i + 1]) / det;
                let py = s * (c * phi[2 * i] + d * phi[2 * i + 1]) / det;
                for k in 0..n_loc {
                    let sk = signs[k];
                    let kx = sk * (a * phi[2 * k] + b * phi[2 * k + 1]) / det;
                    let ky = sk * (c * phi[2 * k] + d * phi[2 * k + 1]) / det;
                    m_e[i * n_loc + k] += w * (px * kx + py * ky);
                }
                for j in 0..nh_loc {
                    // ∇ψ = J⁻ᵀ∇̂ψ  (J = [[a,b],[c,d]] ⇒ J⁻ᵀ = [[d,−c],[−b,a]]/det)
                    let gx = (d * dsh[2 * j] - c * dsh[2 * j + 1]) / det;
                    let gy = (a * dsh[2 * j + 1] - b * dsh[2 * j]) / det;
                    g_e[i * nh_loc + j] += w * (px * gy - py * gx);
                }
            }
        }
        let mut lu = m_e.clone();
        let mut piv = vec![0_usize; n_loc];
        fem_linalg::dense::lu_factor(&mut lu, n_loc, &mut piv).expect("RT mass singular");
        let cols: Vec<Vec<f64>> = (0..nh_loc)
            .map(|j| {
                let mut col: Vec<f64> = (0..n_loc).map(|i| g_e[i * nh_loc + j]).collect();
                fem_linalg::dense::lu_solve(&lu, n_loc, &piv, &mut col);
                col
            })
            .collect();
        local.push((dofs, hdofs, cols));
    }
    // Scatter grouped by global H1 column (per-column write-once per dof).
    let mut per_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); h1.n_dofs()];
    for (dofs, hdofs, cols) in &local {
        for (li, &hcol) in hdofs.iter().enumerate() {
            for (i, &dof) in dofs.iter().enumerate() {
                if cols[li][i].abs() > 0.0 {
                    per_col[hcol].push((dof, cols[li][i]));
                }
            }
        }
    }
    for (hcol, entries) in per_col.into_iter().enumerate() {
        let mut written = std::collections::HashSet::<usize>::new();
        for (dof, v) in entries {
            if written.insert(dof) {
                coo.add(dof, hcol, v);
            }
        }
    }
    coo.into_csr()
}

/// Element ancestry for one refinement level: `parents[f]` = coarse element
/// containing the centroid of fine element `f`.
fn compute_parents(coarse: &Mesh<2>, fine: &Mesh<2>) -> Vec<usize> {
    let ne1 = fine.n_elems() as usize;
    let mut parents = vec![0_usize; ne1];
    for f in 0..ne1 {
        let fe = f as u32;
        let nn = fine.element_nodes(fe);
        let cx: f64 = nn.iter().map(|&n| fine.node_coords(n)[0]).sum::<f64>() / nn.len() as f64;
        let cy: f64 = nn.iter().map(|&n| fine.node_coords(n)[1]).sum::<f64>() / nn.len() as f64;
        parents[f] = locate_in_mesh(coarse, &[cx, cy])
            .map(|(e, _)| e as usize)
            .unwrap_or_else(|| {
                eprintln!(
                    "[dbg] fine elem {f} centroid = ({cx:.6e}, {cy:.6e}), n_coarse = {}",
                    coarse.n_elems()
                );
                let e0 = 0;
                let ns = coarse.element_nodes(e0);
                eprintln!(
                    "[dbg] coarse elem 0 nodes = {:?} coords = {:?}",
                    ns,
                    ns.iter().map(|&n| coarse.node_coords(n)).collect::<Vec<_>>()
                );
                panic!("compute_parents: no parent for fine elem {f}")
            });
    }
    parents
}

/// `agg_l2dof[l]` (level-`l` element → level-`l+1` L2 dofs) and
/// `agg_hdivdof[l]` (level-`l` element → level-`l+1` interior H(div) dofs —
/// dofs used by fine elements of a single aggregate only; C++
/// `AggToInteriorDof`; essential dofs are excluded, here the serial cut uses
/// natural BCs on the whole boundary so the ess set is empty).
fn build_agg_tables(
    ne0: usize,
    hdiv_fine: &HDivSpace<Mesh<2>>,
    l2_fine: &L2Space<Mesh<2>>,
    parents: &[usize],
    ess_dofs: &[usize],
) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let ne1 = parents.len();
    let n_u1 = hdiv_fine.n_dofs();

    let mut coo_al2 = CooMatrix::<f64>::new(ne0, l2_fine.n_dofs());
    for f in 0..ne1 {
        for &d in l2_fine.element_dofs(f as u32) {
            coo_al2.add(parents[f], d as usize, 1.0);
        }
    }
    let agg_l2dof = coo_al2.into_csr();

    // dof → set of aggregates containing it (via fine element→dof incidence).
    let mut dof_agg = vec![usize::MAX; n_u1];
    let mut dof_used = vec![false; n_u1];
    let ess: std::collections::HashSet<usize> = ess_dofs.iter().copied().collect();
    for f in 0..ne1 {
        for &d in hdiv_fine.element_dofs(f as u32) {
            let d = d as usize;
            if dof_agg[d] == usize::MAX {
                dof_agg[d] = parents[f];
            } else if dof_agg[d] != parents[f] {
                dof_agg[d] = usize::MAX - 1; // shared across aggregates
            }
            dof_used[d] = true;
        }
    }
    let mut coo_ah = CooMatrix::<f64>::new(ne0, n_u1);
    for d in 0..n_u1 {
        if dof_used[d] && dof_agg[d] < ne0 && !ess.contains(&d) {
            coo_ah.add(dof_agg[d], d, 1.0);
        }
    }
    let agg_hdivdof = coo_ah.into_csr();
    (agg_hdivdof, agg_l2dof)
}

/// Element→dof boolean table (used for the `SymDirectSubBlockSolver` blocks).
fn elem_dof_table(space_dofs: impl Fn(u32) -> Vec<usize>, n_elems: usize, n_dofs: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_elems, n_dofs);
    for e in 0..n_elems {
        for d in space_dofs(e as u32) {
            coo.add(e, d, 1.0);
        }
    }
    coo.into_csr()
}

/// Serial port of `DFSSpaces` data collection (`CollectDFSData` +
/// `MakeDofRelationTables` + `DataFinalize`): builds the per-level refinement
/// prolongations, aggregate tables, discrete curl and `Q_l2` operators for
/// `rp` refinement levels above `mesh0`.
fn collect_dfs_data(
    mesh0: &Mesh<2>,
    rp: usize,
    order: u8,
    param: DfsParameters,
) -> DfsData {
    // Per-level meshes and spaces (level 0 = coarsest, level rp = finest).
    let mut meshes: Vec<Mesh<2>> = Vec::with_capacity(rp + 1);
    meshes.push(mesh0.clone());
    for _ in 0..rp {
        let last = meshes.last().unwrap().clone();
        meshes.push(refine_uniform(&last));
    }
    let hdiv: Vec<HDivSpace<Mesh<2>>> =
        meshes.iter().map(|m| HDivSpace::new(m.clone(), order)).collect();
    let l2: Vec<L2Space<Mesh<2>>> =
        meshes.iter().map(|m| L2Space::new(m.clone(), order)).collect();
    let h1: Vec<H1Space<Mesh<2>>> =
        meshes.iter().map(|m| H1Space::new(m.clone(), order + 1)).collect();
    let qo_dbg = (2 * order as usize + 3).max(4) as u8;

    let mut p_hdiv = Vec::with_capacity(rp);
    let mut p_l2 = Vec::with_capacity(rp);
    let mut p_aux = Vec::with_capacity(rp);
    let mut agg_hdivdof = Vec::with_capacity(rp);
    let mut agg_l2dof = Vec::with_capacity(rp);
    let mut cs: Vec<CsrMatrix<f64>> = vec![CooMatrix::<f64>::new(0, 0).into_csr()];
    let mut parents_all = Vec::with_capacity(rp);
    let verbose = param.verbose;

    for l in 0..rp {
        let parents = compute_parents(&meshes[l], &meshes[l + 1]);
        p_hdiv.push(build_p_hdiv(&hdiv[l], &hdiv[l + 1], &parents));
        p_l2.push(build_p_l2(&l2[l], &l2[l + 1], &parents));
        p_aux.push(build_p_h1(&h1[l], &h1[l + 1], &parents));
        let (ah, al2) = build_agg_tables(meshes[l].n_elems() as usize, &hdiv[l + 1], &l2[l + 1], &parents, &[]);
        agg_hdivdof.push(ah);
        agg_l2dof.push(al2);
        cs.push(build_curl(&h1[l + 1], &hdiv[l + 1]));
        parents_all.push(parents);
        if !verbose {
            continue;
        }
        // ── validation (exactness of the collected operators) ─────────────
        let mf = VectorAssembler::assemble_bilinear(
            &hdiv[l + 1],
            &[&VectorMassIntegrator { alpha: 1.0 }],
            qo_dbg,
        );
        let mc = VectorAssembler::assemble_bilinear(
            &hdiv[l],
            &[&VectorMassIntegrator { alpha: 1.0 }],
            qo_dbg,
        );
        let mcg = fem_solver::div_free_solver::galerkin(&mf, &p_hdiv[l]);
        let mut dmax = 0.0_f64;
        for i in 0..mc.nrows {
            for j in 0..mc.ncols {
                dmax = dmax.max((mc.get(i, j) - mcg.get(i, j)).abs());
            }
        }
        // B·C ≈ 0 (columns of C are divergence-free)
        let bf = {
            let mut bb = assemble_hdiv_l2_mixed(&l2[l + 1], &hdiv[l + 1], &[&HDivL2DivIntegrator], qo_dbg);
            for v in &mut bb.values {
                *v *= -1.0;
            }
            bb
        };
        let bc = bf.multiply(&cs[l + 1]);
        let bcmax = bc.values.iter().fold(0.0_f64, |mx, &v| mx.max(v.abs()));
        // P_l2 exactness
        let wf = Assembler::assemble_bilinear(&l2[l + 1], &[&MassIntegrator { rho: 1.0 }], qo_dbg);
        let wc = Assembler::assemble_bilinear(&l2[l], &[&MassIntegrator { rho: 1.0 }], qo_dbg);
        let wcg = fem_solver::div_free_solver::galerkin(&wf, &p_l2[l]);
        let mut dl2 = 0.0_f64;
        for i in 0..wc.nrows {
            for j in 0..wc.ncols {
                dl2 = dl2.max((wc.get(i, j) - wcg.get(i, j)).abs());
            }
        }
        // P_aux (H1) exactness via nodal interpolation of a quadratic? just Galerkin on mass:
        let hf = Assembler::assemble_bilinear(&h1[l + 1], &[&MassIntegrator { rho: 1.0 }], qo_dbg);
        let hc = Assembler::assemble_bilinear(&h1[l], &[&MassIntegrator { rho: 1.0 }], qo_dbg);
        let hcg = fem_solver::div_free_solver::galerkin(&hf, &p_aux[l]);
        let mut dh1 = 0.0_f64;
        for i in 0..hc.nrows {
            for j in 0..hc.ncols {
                dh1 = dh1.max((hc.get(i, j) - hcg.get(i, j)).abs());
            }
        }
        eprintln!(
            "[dfs-collect l={l}] ‖PᵀMP−M‖ = {dmax:.3e}, ‖B·C‖ = {bcmax:.3e}, ‖PᵀWP−W‖ = {dl2:.3e}, ‖PᵀHP−H‖ = {dh1:.3e}"
        );
    }
    let _ = parents_all;

    // Q_l2 (DataFinalize): W runs from the finest L2 mass down to level 1.
    let qo = (2 * order as usize + 1).max(2) as u8;
    let mut w_mass = Assembler::assemble_bilinear(&l2[rp], &[&MassIntegrator { rho: 1.0 }], qo);
    let mut q_l2 = vec![None; rp];
    for l in (0..rp).rev() {
        let p = &p_l2[l];
        let ptw = p.transpose().multiply(&w_mass);
        let cw = ptw.multiply(p);
        // el_l2dof[l]: level-l L2 element→dof table
        let el_l2dof = elem_dof_table(
            |e| l2[l].element_dofs(e).iter().map(|&d| d as usize).collect(),
            meshes[l].n_elems() as usize,
            l2[l].n_dofs(),
        );
        q_l2[l] = Some(std::sync::Arc::new(Ql2Projector::new(
            &cw,
            &el_l2dof,
            p.clone(),
            w_mass.clone(),
        )));
        w_mass = cw;
    }
    let q_l2: Vec<std::sync::Arc<Ql2Projector>> = q_l2.into_iter().map(Option::unwrap).collect();

    DfsData {
        agg_hdivdof,
        agg_l2dof,
        p_hdiv,
        p_l2,
        p_aux,
        // Natural BCs on the whole boundary in this serial cut ⇒ no
        // essential H(div) dofs (C++: `GetEssentialTrueDofs(ess_attr)`).
        coarsest_ess_hdivdofs: Vec::new(),
        c: cs,
        q_l2,
        param,
    }
}

/// Assemble the global diagonal of the element-wise Bramble–Pasciak mass
/// preconditioner (MFEM `ConstructMassPreconditioner`):
/// `Q[dof] += q_scaling·λ_min(M_T, diag(M_T))·diag(M_T)[dof]` per element.
///
/// Supports Tri3 (RTk via `TriRTk`) and Quad4 (`QuadRTk`, reference domain
/// [0,1]²) elements; the per-element mass matrix is integrated with the
/// isoparametric geometry (`geometry_jacobian`, valid for both element
/// types) under the contravariant Piola map.
fn assemble_element_q_diag(space: &HDivSpace<Mesh<2>>, order: u8, q_scaling: f64) -> Vec<f64> {
    use fem_element::raviart_thomas::{QuadRTk, TriRTk};
    use fem_mesh::{ElementType, MeshTopology};

    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let mut q = vec![0.0; n_dofs];
    // Quad4 geometry is bilinear, so keep a safe quadrature order.
    let q_order = 2 * order + 4;

    for (ei, e) in mesh.elem_iter().enumerate() {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        match mesh.element_type(e) {
            ElementType::Tri3 => {
                add_element_q(mesh, e, &dofs, &TriRTk::new(order as usize), q_order, q_scaling, ei, &mut q);
            }
            ElementType::Quad4 => {
                add_element_q(mesh, e, &dofs, &QuadRTk::new(order as usize), q_order, q_scaling, ei, &mut q);
            }
            other => panic!("assemble_element_q_diag: unsupported element type {other:?}"),
        }
    }
    q
}

/// One element's contribution to the global Q diagonal.
fn add_element_q<RE: VectorReferenceElement>(
    mesh: &Mesh<2>,
    e: u32,
    dofs: &[usize],
    ref_elem: &RE,
    q_order: u8,
    q_scaling: f64,
    ei: usize,
    q: &mut [f64],
) {
    let n_ld = ref_elem.n_dofs();
    assert_eq!(
        dofs.len(),
        n_ld,
        "RT element DOF count mismatch (elem {e}: {} DOFs vs {} basis)",
        dofs.len(),
        n_ld
    );
    let qr = ref_elem.quadrature(q_order);
    let mut phi = vec![0.0; n_ld * 2];
    let mut phys = vec![0.0; n_ld * 2];

    // Element RT mass matrix M_T[j][k] = ∫ φ_j·φ_k dx (contravariant Piola:
    // φ_phys = J·φ̂/detJ), integrated with the isoparametric geometry.
    let mut me = vec![0.0; n_ld * n_ld];
    for (qi, xi) in qr.points.iter().enumerate() {
        ref_elem.eval_basis_vec(xi, &mut phi);
        // geometry_jacobian returns (detJ, J^{-T}); recover J = (J^{-T})ᵀ⁻¹.
        let (det, ji) = geometry_jacobian(mesh, e, xi, 2);
        let j = ji.transpose().try_inverse().expect("singular element Jacobian");
        let w = qr.weights[qi] * det.abs();
        for i in 0..n_ld {
            phys[2 * i] = (j[(0, 0)] * phi[2 * i] + j[(0, 1)] * phi[2 * i + 1]) / det;
            phys[2 * i + 1] = (j[(1, 0)] * phi[2 * i] + j[(1, 1)] * phi[2 * i + 1]) / det;
        }
        for jj in 0..n_ld {
            for k in 0..n_ld {
                me[jj * n_ld + k] +=
                    w * (phys[2 * jj] * phys[2 * k] + phys[2 * jj + 1] * phys[2 * k + 1]);
            }
        }
    }
    let scaling = element_q_scaling(&me, n_ld, q_scaling, ei);
    for j in 0..n_ld {
        q[dofs[j]] += scaling * me[j * n_ld + j];
    }
}

fn compute_hdiv_l2_error_2d<F>(space: &HDivSpace<Mesh<2>>, u: &[f64], ex: &F) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
{
    use fem_element::raviart_thomas::{QuadRTk, TriRTk};
    use fem_element::reference::VectorReferenceElement;
    use fem_mesh::{element_jacobian_at, ElementType, MeshTopology};

    let order = space.order() as usize;
    let mut e2 = 0.0;
    // Quadrature exact enough for the RT_k physical integrand (degree ≤ k+1).
    let q_order = (2 * order + 4).min(9) as u8;

    for e in space.mesh().elem_iter() {
        let ref_elem: Box<dyn VectorReferenceElement> = match space.mesh().element_type(e) {
            ElementType::Tri3 => Box::new(TriRTk::new(order)),
            ElementType::Quad4 => Box::new(QuadRTk::new(order)),
            other => panic!("compute_hdiv_l2_error_2d: unsupported element type {other:?}"),
        };
        let n_ldofs = ref_elem.n_dofs();
        let q = ref_elem.quadrature(q_order);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let mut ref_phi = vec![0.0_f64; n_ldofs * 2];

        for (qi, xi) in q.points.iter().enumerate() {
            let (jac, xp) = element_jacobian_at(space.mesh(), e, xi, 2);
            let det = jac.determinant();
            let w = q.weights[qi] * det.abs();
            ref_elem.eval_basis_vec(xi, &mut ref_phi);
            let mut fh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let r0 = ref_phi[i * 2];
                let r1 = ref_phi[i * 2 + 1];
                let px = s * (jac[(0, 0)] * r0 + jac[(0, 1)] * r1) / det;
                let py = s * (jac[(1, 0)] * r0 + jac[(1, 1)] * r1) / det;
                fh[0] += u[dofs[i]] * px;
                fh[1] += u[dofs[i]] * py;
            }
            let exact = ex(&xp);
            e2 += w * ((fh[0] - exact[0]).powi(2) + (fh[1] - exact[1]).powi(2));
        }
    }
    e2.sqrt()
}

fn assemble_bdr_rhs(
    space: &HDivSpace<Mesh<2>>,
    tags: &[i32],
    nd: usize,
    g: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    use fem_mesh::MeshTopology;
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];

    // Gauss–Legendre nodes/weights on [0,1]; RT_k has (k+1) edge DOFs, so the
    // boundary-flux RHS ∫₀¹ g·φ_k dξ ≈ w_k·g(ξ_k) uses nd = k+1 points.
    // (RT0: 1 point; RT1: the 2-point rule of ex5; higher orders as needed.)
    let (xi, wts): (Vec<f64>, Vec<f64>) = match nd {
        1 => (vec![0.5], vec![1.0]),
        2 => (
            vec![
                0.5 * (1.0 - 1.0 / 3.0f64.sqrt()),
                0.5 * (1.0 + 1.0 / 3.0f64.sqrt()),
            ],
            vec![0.5, 0.5],
        ),
        _ => {
            // 3/4-point rules on [0,1].
            let (r, w): (&[f64], &[f64]) = match nd {
                3 => (
                    &[-0.7745966692414834, 0.0, 0.7745966692414834],
                    &[0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
                ),
                _ => (
                    &[
                        -0.8611363115940526,
                        -0.3399810435848563,
                        0.3399810435848563,
                        0.8611363115940526,
                    ],
                    &[
                        0.3478548451374538,
                        0.6521451548625461,
                        0.6521451548625461,
                        0.3478548451374538,
                    ],
                ),
            };
            (
                r.iter().map(|v| 0.5 * (1.0 + v)).collect(),
                w.iter().map(|v| 0.5 * v).collect(),
            )
        }
    };
    assert_eq!(xi.len(), nd);

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) {
            continue;
        }
        let nodes = mesh.face_nodes(f);
        if nodes.len() < 2 {
            continue;
        }
        let pa = mesh.node_coords(nodes[0]);
        let pb = mesh.node_coords(nodes[1]);
        let (a, b) = (nodes[0], nodes[1]);
        let key = if a < b { (a, b) } else { (b, a) };
        let Some(first) = space.edge_face_dof(fem_space::dof_manager::EdgeKey::new(key.0, key.1))
        else {
            continue;
        };
        let first = first as usize;
        let face_forward = a < b;
        let cor = if face_forward { 1 } else { -1 };
        for k in 0..nd {
            let t = xi[k];
            let xp = [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
            // Reversed edge orientation mirrors the DOF order (RT_k edge DOFs
            // are ordered along the edge) and flips the sign.
            let global = if cor > 0 { first + k } else { first + (nd - 1 - k) };
            let sgn = if cor > 0 { 1.0 } else { -1.0 };
            rhs[global] += sgn * wts[k] * (g)(&xp);
        }
    }
    rhs
}

struct Args {
    mesh: Option<String>,
    order: u8,
    ser_ref_levels: usize,
    /// DFS (hierarchy) refinement levels — block-solvers.cpp `par_ref_levels`.
    dfs_ref_levels: usize,
    solver: String,
    q_scaling: f64,
    m1_mode: String,
    /// Schur preconditioner of the BDP/DFS coarse solve: amg|dense|diag.
    /// Default `amg` (hypre BoomerAMG analogue, `fem_amg::boomeramg_config`);
    /// `dense` (exact inverse) remains as a fallback for small problems.
    schur: String,
    /// Optional path: dump the assembled Schur complement S (COO text).
    dump_schur: Option<String>,
    /// GMRES restart for the coupled DFS mode.
    restart: usize,
    verbose: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        order: 0,
        ser_ref_levels: 2,
        dfs_ref_levels: 0,
        solver: "bpcg".to_string(),
        q_scaling: 0.5,
        m1_mode: "diag".to_string(),
        // C++ block-solvers uses hypre BoomerAMG on the Schur complement
        // (BDPMinresSolver); AMG is the default, Dense is the fallback.
        schur: "amg".to_string(),
        dump_schur: None,
        restart: 50,
        verbose: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "-rs" | "--refine-serial" => {
                a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2);
            }
            "-rp" | "--refine-dfs" => {
                a.dfs_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-solver" | "--solver" => {
                a.solver = it.next().unwrap_or_else(|| "bpcg".to_string());
            }
            "-q" | "--q-scaling" => {
                a.q_scaling = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5);
            }
            "-m1" | "--m1-mode" => {
                a.m1_mode = it.next().unwrap_or_else(|| "diag".to_string());
            }
            "-schur" | "--schur-mode" => {
                a.schur = it.next().unwrap_or_else(|| "dense".to_string());
            }
            "-dump-schur" | "--dump-schur" => {
                a.dump_schur = it.next();
            }
            "-restart" | "--gmres-restart" => {
                a.restart = it.next().and_then(|v| v.parse().ok()).unwrap_or(50);
            }
            "-vis" | "--verbose" => a.verbose = true,
            _ => {}
        }
    }
    a
}
