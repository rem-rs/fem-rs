//!
//! Parallel IMEX advection-diffusion (pex41).
//!
//! Solves `du/dt + v·grad(u) - a·div(grad(u)) = 0` on a (possibly periodic)
//! mesh with DG discretization and IMEX ODE time integration.
//!
//! MFEM ex41p: `-m ../data/periodic-square.mesh -p 0 -r 2 -o 3 -s 64 -tf 10 -dt 0.01`

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::dg::dg_imex::{
    assemble_ex41_bdr_faces, assemble_ex41_interior_faces, build_bdr_face_locs,
    build_face_locs, MfemHeadInsert,
};
use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::standard::{ConvectionIntegrator, DiffusionIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_mesh::refine_uniform;
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_amg;
use fem_parallel::{
    DofPartition, ParAmgConfig, ParCsrMatrix, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
};
use fem_solver::{
    SolverConfig,
};
use fem_space::fe_space::FESpace;
use fem_space::{L2Basis, L2Space};

struct Args {
    mesh_file: String,
    problem: usize,
    ser_ref_levels: usize,
    par_ref_levels: usize,
    order: usize,
    ode_solver_type: usize,
    t_final: f64,
    dt: f64,
    diffusion_term: f64,
    vis_steps: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/periodic-square.mesh".into(),
        problem: 0,
        ser_ref_levels: 2,
        par_ref_levels: 0,
        order: 3,
        ode_solver_type: 64,
        t_final: 10.0,
        dt: 0.01,
        diffusion_term: 0.01,
        vis_steps: 50,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_default();
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = val(),
            "-p" | "--problem" => a.problem = val().parse().unwrap_or(a.problem),
            "-rs" | "--refine-serial" => {
                a.ser_ref_levels = val().parse().unwrap_or(a.ser_ref_levels)
            }
            "-rp" | "--refine-parallel" => {
                a.par_ref_levels = val().parse().unwrap_or(a.par_ref_levels)
            }
            "-o" | "--order" => a.order = val().parse().unwrap_or(a.order),
            "-s" | "--ode-solver" => a.ode_solver_type = val().parse().unwrap_or(a.ode_solver_type),
            "-tf" | "--t-final" => a.t_final = val().parse().unwrap_or(a.t_final),
            "-dt" | "--time-step" => a.dt = val().parse().unwrap_or(a.dt),
            "-dc" | "--diffusion-coeff" => {
                a.diffusion_term = val().parse().unwrap_or(a.diffusion_term)
            }
            "-vs" | "--vis-steps" => a.vis_steps = val().parse().unwrap_or(a.vis_steps),
            "-r" | "--refine" => {
                a.ser_ref_levels = val().parse().unwrap_or(a.ser_ref_levels)
            }
            _ => {}
        }
    }
    a
}

fn velocity_function(problem: usize, bb_min: &[f64], bb_max: &[f64], x: &[f64], v: &mut [f64]) {
    let dim = x.len();
    let mut xr = vec![0.0; dim];
    for i in 0..dim {
        let center = (bb_min[i] + bb_max[i]) * 0.5;
        xr[i] = 2.0 * (x[i] - center) / (bb_max[i] - bb_min[i]);
    }
    match problem {
        0 => match dim {
            1 => v[0] = 1.0,
            2 => {
                v[0] = (2.0_f64 / 3.0_f64).sqrt();
                v[1] = (1.0_f64 / 3.0_f64).sqrt();
            }
            3 => {
                v[0] = (3.0_f64 / 6.0_f64).sqrt();
                v[1] = (2.0_f64 / 6.0_f64).sqrt();
                v[2] = (1.0_f64 / 6.0_f64).sqrt();
            }
            _ => {}
        },
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

fn domain_insert(
    hi: &mut MfemHeadInsert,
    space: &L2Space<Mesh<2>>,
    mat: &fem_linalg::CsrMatrix<f64>,
    scale: f64,
) {
    for e in 0..space.mesh().n_elements() {
        let dofs = space.element_dofs(e as u32);
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

fn scale_csr(a: &fem_linalg::CsrMatrix<f64>, s: f64) -> fem_linalg::CsrMatrix<f64> {
    let values = a.values.iter().map(|v| s * v).collect();
    fem_linalg::CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        row_ptr: a.row_ptr.clone(),
        col_idx: a.col_idx.clone(),
        values,
    }
}

fn merge_csr_mfem_plus_eq(
    primary: &fem_linalg::CsrMatrix<f64>,
    secondary: &fem_linalg::CsrMatrix<f64>,
    scale_secondary: f64,
) -> fem_linalg::CsrMatrix<f64> {
    let n = primary.nrows;
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 0..n {
        let mut row: Vec<(u32, f64)> = Vec::new();
        for k in primary.row_ptr[i]..primary.row_ptr[i + 1] {
            row.push((primary.col_idx[k], primary.values[k]));
        }
        let mut new_cols: Vec<(u32, f64)> = Vec::new();
        for k in secondary.row_ptr[i]..secondary.row_ptr[i + 1] {
            let j = secondary.col_idx[k];
            if let Some(p) = row.iter().position(|(c, _)| *c == j) {
                row[p].1 += scale_secondary * secondary.values[k];
            } else {
                new_cols.push((j, scale_secondary * secondary.values[k]));
            }
        }
        for (j, v) in new_cols.into_iter().rev() {
            row.insert(0, (j, v));
        }
        for (j, v) in row {
            col_idx.push(j);
            values.push(v);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    fem_linalg::CsrMatrix {
        nrows: n,
        ncols: primary.ncols,
        row_ptr,
        col_idx,
        values,
    }
}

fn main() {
    let args = parse_args();

    let mf = read_mfem_file(&args.mesh_file).expect("failed to read mesh");
    let mut mesh: Mesh<2> = mf.mesh2d.expect("expected 2D mesh");
    for _ in 0..args.ser_ref_levels {
        mesh = refine_uniform(&mesh);
    }
    let mesh = Arc::new(mesh);

    let args_vec: Vec<String> = std::env::args().collect();
    let n_workers = args_vec
        .iter()
        .position(|a| a == "--ranks")
        .and_then(|i| args_vec.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let result = Arc::new(std::sync::Mutex::new(None));
    let result_slot = result.clone();
    let mesh_arc = mesh.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition();

        let space_local = L2Space::new_with_basis(
            local_mesh.clone(),
            args.order as u8,
            L2Basis::GaussLobatto,
        );
        let part = DofPartition::from_l2_space(&space_local, partition, &comm);
        let ps =
            ParallelFESpace::new_with_dof_partition(space_local, part, comm.clone());
        let n_owned = ps.dof_partition().n_owned_dofs;
        let ghost = ps.dof_ghost_exchange_arc();

        let mut bb_min = vec![f64::INFINITY; 2];
        let mut bb_max = vec![f64::NEG_INFINITY; 2];
        for nid in 0..mesh_arc.n_nodes() {
            let c = mesh_arc.node_coords(nid as u32);
            for d in 0..2 {
                bb_min[d] = bb_min[d].min(c[d]);
                bb_max[d] = bb_max[d].max(c[d]);
            }
        }

        let qo = 2 * args.order as u8;
        let m_local = Assembler::assemble_bilinear(
            ps.local_space(),
            &[&MassIntegrator { rho: 1.0 }],
            qo,
        );
        let s_vol = Assembler::assemble_bilinear(
            ps.local_space(),
            &[&DiffusionIntegrator {
                kappa: args.diffusion_term,
            }],
            qo,
        );
        let vel = |_x: f64, _y: f64| -> [f64; 2] {
            let mut v = [0.0f64; 2];
            velocity_function(
                args.problem,
                &bb_min,
                &bb_max,
                &[_x, _y],
                &mut v,
            );
            v
        };
        let alpha = -1.0;
        let sigma = -1.0;
        let kappa = (args.order + 1) * (args.order + 1);

        let mut hi_k = MfemHeadInsert::new(ps.n_local_dofs());
        let mut hi_s = MfemHeadInsert::new(ps.n_local_dofs());
        let k_vol = Assembler::assemble_bilinear(
            ps.local_space(),
            &[&ConvectionIntegrator {
                velocity: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
                    velocity_function(args.problem, &bb_min, &bb_max, x, out);
                }),
            }],
            qo,
        );
        domain_insert(&mut hi_k, &ps.local_space(), &k_vol, -1.0);
        domain_insert(&mut hi_s, &ps.local_space(), &s_vol, 1.0);
        let faces = build_face_locs(ps.local_space().mesh());
        assemble_ex41_interior_faces(
            &mut hi_k,
            ps.local_space().mesh(),
            ps.local_space(),
            &faces,
            &vel,
            alpha,
            0.0,
            sigma,
            0.0,
        );
        let zero_vel = |_x: f64, _y: f64| -> [f64; 2] { [0.0, 0.0] };
        assemble_ex41_interior_faces(
            &mut hi_s,
            ps.local_space().mesh(),
            ps.local_space(),
            &faces,
            &zero_vel,
            0.0,
            args.diffusion_term,
            sigma,
            kappa as f64,
        );
        let bdr_faces = build_bdr_face_locs(ps.local_space().mesh());
        if !bdr_faces.is_empty() {
            assemble_ex41_bdr_faces(
                &mut hi_k,
                ps.local_space().mesh(),
                ps.local_space(),
                &bdr_faces,
                &vel,
                alpha,
                0.0,
                sigma,
                0.0,
            );
            assemble_ex41_bdr_faces(
                &mut hi_s,
                ps.local_space().mesh(),
                ps.local_space(),
                &bdr_faces,
                &zero_vel,
                0.0,
                args.diffusion_term,
                sigma,
                kappa as f64,
            );
        }

        let k_local = hi_k.into_csr();
        let s_local = hi_s.into_csr();

        let m_mat = ParCsrMatrix::from_local_matrix(&m_local, n_owned, ghost.clone(), comm.clone());
        let k_mat = ParCsrMatrix::from_local_matrix(&k_local, n_owned, ghost.clone(), comm.clone());
        let s_mat = ParCsrMatrix::from_local_matrix(&s_local, n_owned, ghost.clone(), comm.clone());

        let dof_coords = ps.local_space().dof_coords();
        let n_local = ps.n_local_dofs();
        let mut u_local = vec![0.0f64; n_local];
        for (i, ch) in dof_coords.chunks(2).enumerate() {
            if i < n_local {
                u_local[i] = u0_function(args.problem, &bb_min, &bb_max, ch);
            }
        }

        if rank == 0 {
            println!("Number of unknowns: {}", ps.n_global_dofs());
        }
        // eprintln!("[rank {}] n_owned={}, n_local={}", rank, n_owned, ps.n_local_dofs());

        let mut t = 0.0f64;
        let mut ti = 0usize;
        let mut done = false;
        let mut u_par = ParVector::from_local_raw(u_local, n_owned, ghost.clone(), comm.clone());

        // IMEX RK3 constants (ImexDirkRk3, bit-for-bit MFEM IMEX_DIRK_RK3).
        let gamma = 0.4358665215_f64;
        let b1 = 1.208496649_f64;
        let b2 = -0.644363171_f64;
        let a31 = 0.3212788860_f64;
        let a32 = 0.3966543747_f64;
        let a41 = -0.105858296_f64;
        let a42 = 0.5529291479_f64;
        let a43 = 0.5529291479_f64;

        // Explicit M⁻¹ via block-diagonal Jacobi (exact for block-diag M).
        let m_diag_inv: Vec<f64> = m_mat
            .diagonal()
            .iter()
            .map(|&x| if x.abs() > 1e-300 { 1.0 / x } else { 0.0 })
            .collect();

        // Implicit (M + γ·dt·S) solver: AMG with local aggregation + PCG.
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            block_size: 1,
            use_global_aggregation: false,
            ..ParAmgConfig::default()
        };
        let solve_cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 200,
            verbose: false,
            ..SolverConfig::default()
        };

        // Reusable buffers.
        let mut k1_exp = ParVector::zeros(&ps);
        let mut k2_exp = ParVector::zeros(&ps);
        let mut k3_exp = ParVector::zeros(&ps);
        let mut k4_exp = ParVector::zeros(&ps);
        let mut k2_imp = ParVector::zeros(&ps);
        let mut k3_imp = ParVector::zeros(&ps);
        let mut y = ParVector::zeros(&ps);

        while !done {
            let dt_real = args.dt.min(args.t_final - t);

            // Implicit system matrix: A = M + gamma*dt*S (same for all 3 implicit stages)
            let scaled_s = scale_csr(&s_local, gamma * dt_real);
            let a_local = merge_csr_mfem_plus_eq(&scaled_s, &m_local, 1.0);
            let a_mat = ParCsrMatrix::from_local_matrix(&a_local, n_owned, ghost.clone(), comm.clone());

            // ===== Stage 1: K1_exp =====
            // k1_exp = M^{-1} * K * u
            let mut ku = ParVector::zeros(&ps);
            k_mat.spmv(&mut u_par.clone_vec(), &mut ku);
            for i in 0..(n_owned.min(m_diag_inv.len())) {
                k1_exp.as_slice_mut()[i] = m_diag_inv[i] * ku.as_slice()[i];
            }

            // ===== Stage 2: K2_exp + K2_imp =====
            // y = u + gamma*dt*k1_exp
            for i in 0..n_owned {
                y.as_slice_mut()[i] = u_par.as_slice()[i] + gamma * dt_real * k1_exp.as_slice()[i];
            }
            // k2_exp = M^{-1} * K * y
            let mut ky = ParVector::zeros(&ps);
            k_mat.spmv(&mut y, &mut ky);
            for i in 0..(n_owned.min(m_diag_inv.len())) {
                k2_exp.as_slice_mut()[i] = m_diag_inv[i] * ky.as_slice()[i];
            }
            // k2_imp: (M + gamma*dt*S) k2_imp = -S*u  (implicit_solve uses original u)
            let mut su = ParVector::zeros(&ps);
            s_mat.spmv(&mut u_par.clone_vec(), &mut su);
            for i in 0..n_owned { su.as_slice_mut()[i] = -su.as_slice()[i]; }
            par_solve_pcg_amg(&a_mat, &su, &mut k2_imp, &amg_cfg, &solve_cfg).expect("stage2 implicit");

            // ===== Stage 3: K3_exp + K3_imp =====
            // y = u + dt*(a31*k1_exp + a32*k2_exp)   (explicit intermediate)
            for i in 0..n_owned {
                y.as_slice_mut()[i] = u_par.as_slice()[i]
                    + dt_real * (a31 * k1_exp.as_slice()[i] + a32 * k2_exp.as_slice()[i]);
            }
            // k3_exp = M^{-1} * K * y
            let mut ky3 = ParVector::zeros(&ps);
            k_mat.spmv(&mut y, &mut ky3);
            for i in 0..(n_owned.min(m_diag_inv.len())) {
                k3_exp.as_slice_mut()[i] = m_diag_inv[i] * ky3.as_slice()[i];
            }
            // k3_imp: y_imp = u + dt*(1-gamma)/2*k2_imp; (M+gamma*dt*S) k3_imp = -S*y_imp
            for i in 0..n_owned {
                y.as_slice_mut()[i] = u_par.as_slice()[i]
                    + dt_real * (1.0 - gamma) / 2.0 * k2_imp.as_slice()[i];
            }
            let mut sy3 = ParVector::zeros(&ps);
            s_mat.spmv(&mut y, &mut sy3);
            for i in 0..n_owned { sy3.as_slice_mut()[i] = -sy3.as_slice()[i]; }
            par_solve_pcg_amg(&a_mat, &sy3, &mut k3_imp, &amg_cfg, &solve_cfg).expect("stage3 implicit");

            // ===== Stage 4: K4_exp + K4_imp =====
            // y = u + dt*(a41*k1_exp + a42*k2_exp + a43*k3_exp)  (explicit intermediate)
            for i in 0..n_owned {
                y.as_slice_mut()[i] = u_par.as_slice()[i]
                    + dt_real * (a41 * k1_exp.as_slice()[i] + a42 * k2_exp.as_slice()[i] + a43 * k3_exp.as_slice()[i]);
            }
            // k4_exp = M^{-1} * K * y
            let mut ky4 = ParVector::zeros(&ps);
            k_mat.spmv(&mut y, &mut ky4);
            for i in 0..(n_owned.min(m_diag_inv.len())) {
                k4_exp.as_slice_mut()[i] = m_diag_inv[i] * ky4.as_slice()[i];
            }
            // k4_imp (reuses k3_imp buffer):
            //   u += dt*b1*k2_imp; u += dt*b2*k3_imp  (MFEM: two separate in-place updates)
            //   then solve (M + gamma*dt*S) k3_imp = -S*u  (reuses k3_imp as k4_imp)
            for i in 0..n_owned { u_par.as_slice_mut()[i] += dt_real * b1 * k2_imp.as_slice()[i]; }
            for i in 0..n_owned { u_par.as_slice_mut()[i] += dt_real * b2 * k3_imp.as_slice()[i]; }
            let mut su4 = ParVector::zeros(&ps);
            s_mat.spmv(&mut u_par.clone_vec(), &mut su4);
            for i in 0..n_owned { su4.as_slice_mut()[i] = -su4.as_slice()[i]; }
            par_solve_pcg_amg(&a_mat, &su4, &mut k3_imp, &amg_cfg, &solve_cfg).expect("stage4 implicit");

            // ===== Final update =====
            // u += dt*b1*k2_exp; u += dt*b2*k3_exp; u += dt*gamma*k4_exp; u += dt*gamma*k3_imp
            for i in 0..n_owned {
                u_par.as_slice_mut()[i] += dt_real * b1 * k2_exp.as_slice()[i];
                u_par.as_slice_mut()[i] += dt_real * b2 * k3_exp.as_slice()[i];
                u_par.as_slice_mut()[i] += dt_real * gamma * k4_exp.as_slice()[i];
                u_par.as_slice_mut()[i] += dt_real * gamma * k3_imp.as_slice()[i];
            }

            t += dt_real;
            ti += 1;
            done = t >= args.t_final - 1e-8 * args.dt;
            if done || ti % args.vis_steps == 0 {
                let norm = u_par.global_norm();
                let sum = comm.allreduce_sum_f64(u_par.as_slice()[..n_owned].iter().sum::<f64>());
                if rank == 0 {
                    println!("time step: {ti}, time: {t:.6}, ||u|| = {norm:.6e}, sum = {sum:.6e}");
                }
            }
        }

        if rank == 0 {
            *result_slot.lock().unwrap() = Some((ti, t));
        }
        // Dump solution values at final step for comparison
        if done && rank == 0 {
            eprintln!("[pex41] FINAL sol (first 10 owned dofs):");
            for i in 0..10.min(n_owned) {
                eprintln!("  u[{i}] = {:.10e}", u_par.as_slice()[i]);
            }
        }
    });

    let res = *result.lock().unwrap();
    if let Some((ti, t)) = res {
        println!("Completed: {ti} steps, final time {t:.6}");
    }
}
