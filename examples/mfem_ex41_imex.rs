//! mfem_ex41_imex — 1:1 port of MFEM Example 41.
//!
//! Solves the time-dependent advection-diffusion equation
//! `du/dt + v·grad(u) − a·div(grad(u)) = 0` on a (possibly geometrically
//! periodic) mesh, using a Discontinuous Galerkin (DG) discretization
//! (`NonconservativeDGTraceIntegrator` + `DGDiffusionIntegrator`) and an IMEX
//! ODE time integrator (`M du/dt + (C + K) u = 0`; `-s 64` = IMEX_DIRK_RK3).
//!
//! Defaults match MFEM ex41: `-m ../data/periodic-square.mesh -p 0 -r 2 -o 3
//! -s 64 -tf 10 -dt 0.01 -dc 0.01` (DG).  `-cg` switches to continuous
//! Galerkin (H1).  The `-s` ODE-solver selector is honoured for 61 (IMEX
//! Euler), 62 (IMEXRK2(2,2,2)), 63 (IMEXRK2(2,3,2)) and 64 (IMEX_DIRK_RK3,
//! default); only 64 is implemented as a bit-for-bit MFEM port.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::ConstantVectorCoeff,
    dg::dg_imex::{assemble_ex41_bdr_faces, assemble_ex41_interior_faces, build_bdr_face_locs, build_face_locs, MfemHeadInsert},
    standard::{ConvectionIntegrator, DiffusionIntegrator, MassIntegrator},
};
use fem_core::types::DofId;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, amr::refine_uniform, topology::MeshTopology};
use fem_solver::{
    ImexDirkRk3, ImexExpImplEuler, ImexOperator, ImexRk2_222, ImexRk2_232, SolverConfig,
    solve_pcg_blockilu, solve_pcg_dsmoother,
};
use fem_space::{L2Basis, L2Space, H1Space, fe_space::FESpace};

// ─── Coefficient functions (problem 0..3) ──────────────────────────────────

struct Args {
    mesh_file: String,
    problem: usize,
    ref_levels: usize,
    order: usize,
    ode_solver_type: usize,
    t_final: f64,
    dt: f64,
    diffusion_term: f64,
    cg: bool,
    vis_steps: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/periodic-square.mesh".into(),
        problem: 0,
        ref_levels: 2,
        order: 3,
        ode_solver_type: 64,
        t_final: 10.0,
        dt: 0.01,
        diffusion_term: 0.01,
        cg: false,
        vis_steps: 50,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_default();
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = val(),
            "-p" | "--problem" => a.problem = val().parse().unwrap_or(a.problem),
            "-r" | "--refine" => a.ref_levels = val().parse().unwrap_or(a.ref_levels),
            "-o" | "--order" => a.order = val().parse().unwrap_or(a.order),
            "-s" | "--ode-solver" => a.ode_solver_type = val().parse().unwrap_or(a.ode_solver_type),
            "-tf" | "--t-final" => a.t_final = val().parse().unwrap_or(a.t_final),
            "-dt" | "--time-step" => a.dt = val().parse().unwrap_or(a.dt),
            "-dc" | "--diffusion-coeff" => a.diffusion_term = val().parse().unwrap_or(a.diffusion_term),
            "-vs" | "--visualization-steps" => a.vis_steps = val().parse().unwrap_or(a.vis_steps),
            "-cg" | "--continuous-galerkin" => a.cg = true,
            "-dg" | "--discontinuous-galerkin" => a.cg = false,
            _ => {}
        }
    }
    a
}

/// Velocity coefficient (MFEM `velocity_function`).
fn velocity_function(problem: usize, bb_min: &[f64], bb_max: &[f64], x: &[f64], v: &mut [f64]) {
    let dim = x.len();
    let mut xr = vec![0.0; dim];
    for i in 0..dim {
        let center = (bb_min[i] + bb_max[i]) * 0.5;
        xr[i] = 2.0 * (x[i] - center) / (bb_max[i] - bb_min[i]);
    }
    match problem {
        0 => {
            match dim {
                1 => v[0] = 1.0,
                2 => {
                    v[0] = (2.0f64 / 3.0f64).sqrt();
                    v[1] = (1.0f64 / 3.0f64).sqrt();
                }
                3 => {
                    v[0] = (3.0f64 / 6.0f64).sqrt();
                    v[1] = (2.0f64 / 6.0f64).sqrt();
                    v[2] = (1.0f64 / 6.0f64).sqrt();
                }
                _ => {}
            }
        }
        1 | 2 => {
            let w = PI / 2.0;
            match dim {
                1 => v[0] = 1.0,
                2 => {
                    v[0] = w * xr[1];
                    v[1] = -w * xr[0];
                }
                3 => {
                    v[0] = w * xr[1];
                    v[1] = -w * xr[0];
                    v[2] = 0.0;
                }
                _ => {}
            }
        }
        3 => {
            let w = PI / 2.0;
            let d = (xr[0] + 1.0).max(0.0) * (1.0 - xr[0]).max(0.0)
                * (xr[1] + 1.0).max(0.0) * (1.0 - xr[1]).max(0.0);
            let d = d * d;
            match dim {
                1 => v[0] = 1.0,
                2 => {
                    v[0] = d * w * xr[1];
                    v[1] = -d * w * xr[0];
                }
                3 => {
                    v[0] = d * w * xr[1];
                    v[1] = -d * w * xr[0];
                    v[2] = 0.0;
                }
                _ => {}
            }
        }
        _ => {}
    }
}

/// Initial condition (MFEM `u0_function`).
fn u0_function(problem: usize, bb_min: &[f64], bb_max: &[f64], x: &[f64]) -> f64 {
    let dim = x.len();
    let mut xr = vec![0.0; dim];
    for i in 0..dim {
        let center = (bb_min[i] + bb_max[i]) * 0.5;
        xr[i] = 2.0 * (x[i] - center) / (bb_max[i] - bb_min[i]);
    }
    match problem {
        0 | 1 => match dim {
            1 => (-40.0 * (xr[0] - 0.5).powi(2)).exp(),
            _ => {
                let rx = 0.45;
                let ry = 0.25;
                let cx = 0.0;
                let cy = -0.2;
                let w = 10.0;
                let (mut rx, mut ry) = (rx, ry);
                if dim == 3 {
                    let s = 1.0 + 0.25 * (2.0 * PI * xr[2]).cos();
                    rx *= s;
                    ry *= s;
                }
                (libm::erfc(w * (xr[0] - cx - rx)) * libm::erfc(-w * (xr[0] - cx + rx))
                    * libm::erfc(w * (xr[1] - cy - ry)) * libm::erfc(-w * (xr[1] - cy + ry)))
                    / 16.0
            }
        },
        2 => {
            let rho = (xr[0] * xr[0] + xr[1] * xr[1]).sqrt();
            let phi = xr[1].atan2(xr[0]);
            (PI * rho).sin().powi(2) * (3.0 * phi).sin()
        }
        3 => {
            let f = PI;
            (f * xr[0]).sin() * (f * xr[1]).sin()
        }
        _ => 0.0,
    }
}

// ─── Evolution operator (MFEM IMEX_Evolution) ─────────────────────────────

/// MFEM `SparseMatrix::operator+=` semantics: `out = primary + scale*secondary`
/// with the *row-internal column order* of MFEM's open SparseMatrix.
///
/// MFEM keeps one linked list per row (`SearchRow` head-inserts new columns,
/// existing columns are accumulated in place) and `Finalize()` does NOT sort —
/// so `A = s; A *= dt; A += m` keeps `s`'s column order and only *head-inserts*
/// `m`'s columns that are missing from `s`.  `primary`'s column order is
/// preserved verbatim; `secondary`'s columns already present in the row are
/// accumulated in place, its new columns are head-inserted (in reverse of the
/// `secondary` row order, matching the linked-list head-insert).
///
/// The Rust `CooMatrix::into_csr()` merge (sort by (row,col) then reverse by
/// insertion index) would re-order columns, so ex41 must build M/S/K/A with
/// this function instead of `coo.into_csr()` to be bit-identical to MFEM.
/// Insert a domain (element) matrix in MFEM's `AddSubMatrix(vdofs, vdofs)`
/// order: for each element `e`, rows `dofs[i]` then columns `dofs[j]` (i and
/// j ascending) — the head-insert turns this into MFEM's column order.
fn domain_insert(
    hi: &mut MfemHeadInsert,
    space: &fem_space::L2Space<fem_mesh::Mesh<2>>,
    mat: &CsrMatrix<f64>,
    scale: f64,
) {
    let n_elem = space.mesh().n_elements();
    for e in 0..n_elem as u32 {
        let dofs = space.element_dofs(e);
        let nd = dofs.len();
        for i in 0..nd {
            let r = dofs[i] as usize;
            for j in 0..nd {
                let c = dofs[j] as usize;
                let v = mat.find_entry(r, c).map(|p| mat.values[p]).unwrap_or(0.0);
                hi.add(r, c, scale * v);
            }
        }
    }
}

fn merge_csr_mfem_plus_eq(
    primary: &CsrMatrix<f64>,
    secondary: &CsrMatrix<f64>,
    scale_secondary: f64,
) -> CsrMatrix<f64> {
    let n = primary.nrows;
    debug_assert_eq!(n, secondary.nrows);
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 0..n {
        // primary row i, in order.
        let mut row: Vec<(u32, f64)> = Vec::new();
        for k in primary.row_ptr[i]..primary.row_ptr[i + 1] {
            row.push((primary.col_idx[k], primary.values[k]));
        }
        // secondary row i: accumulate existing columns, collect new ones.
        let mut new_cols: Vec<(u32, f64)> = Vec::new();
        for k in secondary.row_ptr[i]..secondary.row_ptr[i + 1] {
            let j = secondary.col_idx[k];
            if let Some(p) = row.iter().position(|(c, _)| *c == j) {
                row[p].1 += scale_secondary * secondary.values[k];
            } else {
                new_cols.push((j, scale_secondary * secondary.values[k]));
            }
        }
        // Head-insert new columns (MFEM SearchRow: node->Prev = Rows[row]).
        for (j, v) in new_cols.into_iter().rev() {
            row.insert(0, (j, v));
        }
        for (j, v) in row {
            col_idx.push(j);
            values.push(v);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { nrows: n, ncols: primary.ncols, row_ptr, col_idx, values }
}

/// Scale every value of a CSR matrix (structure unchanged).
fn scale_csr(a: &CsrMatrix<f64>, s: f64) -> CsrMatrix<f64> {
    let values = a.values.iter().map(|v| s * v).collect();
    CsrMatrix { nrows: a.nrows, ncols: a.ncols, row_ptr: a.row_ptr.clone(), col_idx: a.col_idx.clone(), values }
}

/// `M du/dt = K u - S u + b`, explicit part `M⁻¹K u`, implicit part solves
/// `(M + dt·S) k = -S·u` — bit-for-bit MFEM `IMEX_Evolution`.
struct ImexEvolution {
    m: CsrMatrix<f64>,
    k: CsrMatrix<f64>,
    s: CsrMatrix<f64>,
}

impl ImexEvolution {
    fn spmv(csr: &CsrMatrix<f64>, x: &[f64], out: &mut [f64]) {
        csr.spmv(x, out);
    }
}

impl ImexOperator for ImexEvolution {
    fn explicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        // Mult1: y = M^{-1} (K u + b), b = 0.  CG + DSmoother (rtol 1e-9,
        // max 100, iterative_mode = false).
        let n = u.len();
        let mut kx = vec![0.0f64; n];
        Self::spmv(&self.k, u, &mut kx);
        let cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_dsmoother(&self.m, &kx, out, &cfg).expect("Mult1 CG solve failed");
    }

    fn implicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        // Not used by ImexDirkRk3 (which calls implicit_solve directly), but
        // provided for completeness: f_I = -M^{-1} S u.
        let n = u.len();
        let mut sx = vec![0.0f64; n];
        Self::spmv(&self.s, u, &mut sx);
        let cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..SolverConfig::default()
        };
        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            rhs[i] = -sx[i];
        }
        solve_pcg_dsmoother(&self.m, &rhs, out, &cfg).expect("implicit CG solve failed");
    }

    fn jac_implicit(&self, _t: f64, _u: &[f64]) -> CsrMatrix<f64> {
        // Not needed by the ex41 operator (implicit_solve is overridden).
        let n = self.m.nrows;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.add(i, i, 0.0);
        }
        coo.into_csr()
    }

    fn implicit_solve(&self, dt: f64, x: &[f64], k: &mut [f64]) {
        // ImplicitSolve2: (M + dt·S) k = -S·x, CG + BlockILU (rtol 1e-9,
        // max 100, iterative_mode = false).
        let n = x.len();
        let mut sx = vec![0.0f64; n];
        Self::spmv(&self.s, x, &mut sx);
        for i in 0..n {
            sx[i] = -sx[i];
        }
        // A = M + dt·S — MFEM builds it as `A = S; A *= dt; A += M` (per-entry
        // summation order `dt*S + M`, `S`'s column order preserved by the
        // open-matrix SearchRow semantics — see merge_csr_mfem_plus_eq).
        let scaled_s = scale_csr(&self.s, dt);
        let a = merge_csr_mfem_plus_eq(&scaled_s, &self.m, 1.0);
        let cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_blockilu(&a, &sx, k, &cfg, 16).expect("ImplicitSolve2 CG solve failed");
    }
}

// ─── main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // 2. Read the mesh (geometrically periodic meshes supported).
    let mf = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let base = mf.mesh2d.expect("expected a 2D mesh");

    // 4. Refine the mesh.
    let mut mesh = base;
    for _ in 0..args.ref_levels {
        mesh = refine_uniform(&mesh);
    }
    // NURBS meshes would call SetCurvature(max(order,1)) here; the reader
    // does not support NURBS, so nothing to do.
    let dim = 2usize;
    let _ = dim;

    // Bounding box (MFEM GetBoundingBox(bb_min, bb_max, max(order,1))).
    let mut bb_min = vec![f64::INFINITY; 2];
    let mut bb_max = vec![f64::NEG_INFINITY; 2];
    for e in 0..mesh.n_elements() {
        let gn = mesh.geometry_nodes(e as u32);
        for &n in gn {
            let c = mesh.geom_coords_of(n);
            for d in 0..2 {
                bb_min[d] = bb_min[d].min(c[d]);
                bb_max[d] = bb_max[d].max(c[d]);
            }
        }
    }

    // 5. FE space.
    let kappa = (args.order + 1) * (args.order + 1);
    let n_dofs;
    let mut u: Vec<f64>;
    let m: CsrMatrix<f64>;
    let k: CsrMatrix<f64>;
    let s: CsrMatrix<f64>;
    let dof_coords: Vec<f64>;

    if args.cg {
        // Continuous Galerkin (-cg): H1 space, no face integrators.
        let space = H1Space::new(mesh, args.order as u8);
        n_dofs = space.n_dofs();
        let qo = 2 * args.order as u8;
        m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], qo);
        s = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa: args.diffusion_term }],
            qo,
        );
        let vel = {
            let bb_min = bb_min.clone();
            let bb_max = bb_max.clone();
            let problem = args.problem;
            fem_assembly::postproc::coefficient::FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                velocity_function(problem, &bb_min, &bb_max, x, out);
            })
        };
        let k_vol = Assembler::assemble_bilinear(
            &space,
            &[&ConvectionIntegrator { velocity: vel }],
            qo,
        );
        // K = alpha * Convection with alpha = -1.
        let mut k_coo = CooMatrix::new(n_dofs, n_dofs);
        for i in 0..n_dofs {
            for p in k_vol.row_ptr[i]..k_vol.row_ptr[i + 1] {
                k_coo.add(i, k_vol.col_idx[p] as usize, -k_vol.values[p]);
            }
        }
        k = k_coo.into_csr();
        // u0 at H1 dofs.
        let dm = space.dof_manager();
        u = vec![0.0f64; n_dofs];
        for i in 0..n_dofs {
            let x = dm.dof_coord(i as DofId);
            u[i] = u0_function(args.problem, &bb_min, &bb_max, x);
        }
        dof_coords = vec![];
    } else {
        let space = L2Space::new_with_basis(mesh.clone(), args.order as u8, L2Basis::GaussLobatto);
        n_dofs = space.n_dofs();
        dof_coords = space.dof_coords().to_vec();

        // 6. Assemble M, K, S.
        let qo = 2 * args.order as u8;
        m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], qo);
        let s_vol = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa: args.diffusion_term }],
            qo,
        );
        let k_vol = Assembler::assemble_bilinear(
            &space,
            &[&ConvectionIntegrator {
                velocity: ConstantVectorCoeff(vec![
                    ((2.0f64) / 3.0f64).sqrt(),
                    ((1.0f64) / 3.0f64).sqrt(),
                ]),
            }],
            qo,
        );

        // Face integrators: NonconservativeDGTraceIntegrator(vel, -1) and
        // DGDiffusionIntegrator(diff, sigma=-1, kappa).
        let faces = build_face_locs(&mesh);
        let vel = |_x: f64, _y: f64| -> [f64; 2] {
            let mut v = [0.0f64; 2];
            velocity_function(args.problem, &bb_min, &bb_max, &[_x, _y], &mut v);
            v
        };
        let alpha = -1.0;
        let sigma = -1.0;
        // MFEM assembles K/S into ONE open SparseMatrix in this order:
        // domain integrators (Convection/Diffusion, all elements) first, then
        // interior faces, then boundary faces.  `MfemHeadInsert` reproduces
        // the resulting head-insert column ordering bit-for-bit — BlockILU's
        // MDF block reordering sums A² over the CSR column order, so a
        // different column order changes the whole preconditioner.
        let mut hi_k = MfemHeadInsert::new(n_dofs);
        let mut hi_s = MfemHeadInsert::new(n_dofs);
        domain_insert(&mut hi_k, &space, &k_vol, -1.0); // K = -Convection
        domain_insert(&mut hi_s, &space, &s_vol, 1.0);
        assemble_ex41_interior_faces(
            &mut hi_k,
            &mesh,
            &space,
            &faces,
            &vel,
            alpha,
            0.0,
            sigma,
            0.0,
        );
        let zero_vel = |_: f64, _: f64| -> [f64; 2] { [0.0, 0.0] };
        assemble_ex41_interior_faces(
            &mut hi_s,
            &mesh,
            &space,
            &faces,
            &zero_vel,
            0.0,
            args.diffusion_term,
            sigma,
            kappa as f64,
        );
        // Boundary faces (MFEM AddBdrFaceIntegrator, ndof2==0 branch).
        let bdr_faces = build_bdr_face_locs(&mesh);
        if !bdr_faces.is_empty() {
            assemble_ex41_bdr_faces(
                &mut hi_k,
                &mesh,
                &space,
                &bdr_faces,
                &vel,
                alpha,
                0.0,
                sigma,
                0.0,
            );
            assemble_ex41_bdr_faces(
                &mut hi_s,
                &mesh,
                &space,
                &bdr_faces,
                &zero_vel,
                0.0,
                args.diffusion_term,
                sigma,
                kappa as f64,
            );
        }
        k = hi_k.into_csr();
        s = hi_s.into_csr();

        // 7. Initial conditions.
        u = vec![0.0f64; n_dofs];
        for (i, ch) in dof_coords.chunks(2).enumerate() {
            u[i] = u0_function(args.problem, &bb_min, &bb_max, ch);
        }
    }

    println!("Number of unknowns: {n_dofs}");

    // 9. Time integration.
    let adv = ImexEvolution { m, k, s };

    let mut t = 0.0f64;
    let mut u_vec = u;
    let mut ti = 0usize;
    let mut done = false;
    let dirk = ImexDirkRk3;
    let euler_imex = ImexExpImplEuler;
    let rk2_222 = ImexRk2_222;
    let rk2_232 = ImexRk2_232;
    while !done {
        let dt_real = args.dt.min(args.t_final - t);
        // ODE solver selector: 61 = IMEXExpImplEuler, 62 = IMEXRK2(2,2,2),
        // 63 = IMEXRK2_3StageExplicit, 64 = IMEX_DIRK_RK3 — all 1:1 MFEM ports.
        match args.ode_solver_type {
            64 => dirk.step(&adv, &mut t, dt_real, &mut u_vec),
            61 => euler_imex.step(&adv, &mut t, dt_real, &mut u_vec),
            62 => rk2_222.step(&adv, &mut t, dt_real, &mut u_vec),
            63 => rk2_232.step(&adv, &mut t, dt_real, &mut u_vec),
            _ => {
                dirk.step(&adv, &mut t, dt_real, &mut u_vec);
            }
        }
        ti += 1;
        done = t >= args.t_final - 1e-8 * args.dt;
        if done || ti % args.vis_steps == 0 {
            // MFEM prints `time: t` with the default ostream precision (6
            // significant digits).
            let norm = u_vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let sum: f64 = u_vec.iter().sum();
            println!("time step: {ti}, time: {}, ||u|| = {norm:.6e}, sum = {sum:.6e}", fem_solver::fmt_g(t));
        }
        // Dump first 10 dofs at final step for cross-checking
        if done {
            eprintln!("[ex41] FINAL sol (first 10 dofs):");
            for i in 0..10.min(n_dofs) {
                eprintln!("  u[{i}] = {:.10e}", u_vec[i]);
            }
        }
    }
}
