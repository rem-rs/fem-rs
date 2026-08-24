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
use fem_linalg::CooMatrix;
use fem_mesh::refine_uniform;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::ParAmgHierarchy;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{
    DofPartition, ParAmgConfig, ParCsrMatrix, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
};
use fem_solver::{
    ImexOperator, SolverConfig, solve_pcg_blockilu, solve_pcg_dsmoother,
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

#[allow(dead_code)]
struct ImexEvolution {
    m: fem_linalg::CsrMatrix<f64>,
    k: fem_linalg::CsrMatrix<f64>,
    s: fem_linalg::CsrMatrix<f64>,
}

impl ImexOperator for ImexEvolution {
    fn explicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        let n = u.len();
        let mut kx = vec![0.0f64; n];
        self.k.spmv(u, &mut kx);
        let cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_dsmoother(&self.m, &kx, out, &cfg).expect("Mult1 CG failed");
    }

    fn implicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        let n = u.len();
        let mut sx = vec![0.0f64; n];
        self.s.spmv(u, &mut sx);
        let cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..SolverConfig::default()
        };
        for i in 0..n {
            sx[i] = -sx[i];
        }
        solve_pcg_dsmoother(&self.m, &sx, out, &cfg).expect("implicit CG failed");
    }

    fn jac_implicit(&self, _t: f64, _u: &[f64]) -> fem_linalg::CsrMatrix<f64> {
        let n = self.m.nrows;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.add(i, i, 0.0);
        }
        coo.into_csr()
    }

    fn implicit_solve(&self, dt: f64, x: &[f64], k: &mut [f64]) {
        let n = x.len();
        let mut sx = vec![0.0f64; n];
        self.s.spmv(x, &mut sx);
        for i in 0..n {
            sx[i] = -sx[i];
        }
        let scaled_s = scale_csr(&self.s, dt);
        let a = merge_csr_mfem_plus_eq(&scaled_s, &self.m, 1.0);
        let cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_blockilu(&a, &sx, k, &cfg, 16).expect("ImplicitSolve CG failed");
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

        let _m_mat = ParCsrMatrix::from_local_matrix(&m_local, n_owned, ghost.clone(), comm.clone());
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

        let mut t = 0.0f64;
        let mut ti = 0usize;
        let mut done = false;
        let mut u_par = ParVector::from_local_raw(u_local, n_owned, ghost.clone(), comm.clone());

        let dt_repr = args.dt;
        let scaled_s = scale_csr(&s_local, dt_repr);
        let a_local = merge_csr_mfem_plus_eq(&scaled_s, &m_local, 1.0);
        let a_mat = ParCsrMatrix::from_local_matrix(&a_local, n_owned, ghost.clone(), comm.clone());

        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            block_size: 1,
            use_global_aggregation: true,
            ..ParAmgConfig::default()
        };
        let amg = ParAmgHierarchy::build_global(&a_mat, &comm, amg_cfg);

        while !done {
            let dt_real = args.dt.min(args.t_final - t);
            let mut ku = ParVector::zeros(&ps);
            k_mat.spmv(&mut u_par.clone_vec(), &mut ku);
            ku.update_ghosts();
            let mut rhs = ParVector::zeros(&ps);
            amg.vcycle(&ku, &mut rhs);
            rhs.update_ghosts();

            let mut su = ParVector::zeros(&ps);
            s_mat.spmv(&mut u_par.clone_vec(), &mut su);
            su.update_ghosts();
            let mut rhs_implicit = ParVector::zeros(&ps);
            for i in 0..n_owned {
                rhs_implicit.as_slice_mut()[i] = -su.as_slice()[i] + rhs.as_slice()[i];
            }
            let mut k_vec = ParVector::zeros(&ps);
            amg.vcycle(&rhs_implicit, &mut k_vec);
            k_vec.update_ghosts();

            for i in 0..n_owned {
                u_par.as_slice_mut()[i] += dt_real * k_vec.as_slice()[i];
            }
            u_par.update_ghosts();

            t += dt_real;
            ti += 1;
            done = t >= args.t_final - 1e-8 * args.dt;
            if done || ti % args.vis_steps == 0 {
                if rank == 0 {
                    println!("time step: {ti}, time: {t:.6}");
                }
            }
        }

        if rank == 0 {
            *result_slot.lock().unwrap() = Some((ti, t));
        }
    });

    let res = *result.lock().unwrap();
    if let Some((ti, t)) = res {
        println!("Completed: {ti} steps, final time {t:.6}");
    }
}
