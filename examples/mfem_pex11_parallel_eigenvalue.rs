//! # Parallel eigenvalue (pex11) — LOBPCG + AMG (analogous to MFEM ex11p)
//!
//! Solves `-Δu = λu` with homogeneous Dirichlet BC using LOBPCG
//! preconditioned by AMG-CG, in a thread-parallel setting.
//!
//! **Current scope**: parallel assembly + serial eigenvalue solve on rank 0.
//! A distributed solver [`fem_parallel::par_lobpcg::par_lobpcg`] exists but
//! its Rayleigh–Ritz step needs robustness improvements (singular B_proj
//! fallback) before it can replace the rank‑0 serial path.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_pex11_parallel_eigenvalue -- --n 8 --ranks 2
//! ```

use std::sync::{Arc, Mutex};

use fem_solver::amg::{AmgConfig, solve_amg_cg};
use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_linalg::SolverConfig;
use fem_mesh::Mesh;
use fem_solver::LobpcgConfig;
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace,
    par_partition::partition_mesh,
    launcher::native::ThreadLauncher, WorkerConfig,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = args.iter().position(|x| x == "--n")
        .and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let r = args.iter().position(|x| x == "--ranks")
        .and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);

    let mesh = Arc::new(Mesh::<2>::unit_square_tri(n));
    let result = Arc::new(Mutex::new(None));
    let result_clone = Arc::clone(&result);

    ThreadLauncher::new(WorkerConfig::new(r)).launch(move |comm| {
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let ps = ParallelFESpace::new(H1Space::new(lm, 1), &pm, comm.clone());
        let n_global = ps.n_global_dofs();

        // ── Parallel assembly ──────────────────────────────────────────────
        let mut stiff_p = ParAssembler::assemble_bilinear(&ps, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let _mass_p = ParAssembler::assemble_bilinear(&ps, &[&MassIntegrator { rho: 1.0 }], 3);

        // ── Apply Dirichlet BC on parallel stiffness matrix ───────────────
        let bd = boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(),
            &ps.local_space().mesh().unique_boundary_tags());
        let dp = ps.dof_partition();
        let mut tmp = ParVector::zeros(&ps);
        for &d in &bd {
            let p = dp.permute_dof(d) as usize;
            if p < dp.n_owned_dofs { stiff_p.apply_dirichlet_par(p, 1.0, &mut tmp); }
        }

        // ── Rank 0: serial reference solve with AMG‑CG preconditioner ────
        if comm.rank() == 0 {
            let ser_mesh = Mesh::<2>::unit_square_tri(n);
            let ser_space = H1Space::new(ser_mesh, 1);
            let quad = 3;

            let mut a = Assembler::assemble_bilinear(&ser_space, &[&DiffusionIntegrator { kappa: 1.0 }], quad);
            let b = Assembler::assemble_bilinear(&ser_space, &[&MassIntegrator { rho: 1.0 }], quad);
            let bnd = boundary_dofs(ser_space.mesh(), ser_space.dof_manager(),
                &ser_space.mesh().unique_boundary_tags());
            let zero = vec![0.0; bnd.len()];
            fem_space::constraints::apply_dirichlet(&mut a, &mut vec![0.0; ser_space.n_dofs()], &bnd, &zero);

            eprintln!("  [pex11] dofs={n_global} ranks={} ser_dofs={} (serial solve on rank 0)",
                      comm.size(), ser_space.n_dofs());

            let amg_cfg = AmgConfig::default();
            let pcg_cfg = SolverConfig {
                rtol: 1e-2, atol: 1e-4, max_iter: 5, verbose: false,
                ..SolverConfig::default()
            };

            let precond = |r: &nalgebra::DMatrix<f64>| -> nalgebra::DMatrix<f64> {
                let nrows = r.nrows();
                let k = r.ncols();
                let mut z = nalgebra::DMatrix::zeros(nrows, k);
                for j in 0..k {
                    let rhs = r.column(j).to_owned();
                    let rhs_slice = rhs.as_slice();
                    let mut x = vec![0.0; nrows];
                    let _ = solve_amg_cg(&a, rhs_slice, &mut x, &amg_cfg, &pcg_cfg);
                    z.set_column(j, &nalgebra::DVector::from_vec(x));
                }
                z
            };

            let cfg = LobpcgConfig {
                max_iter: 100, tol: 1e-8, verbose: true,
                ..LobpcgConfig::default()
            };

            use fem_solver::lobpcg_constrained_preconditioned;
            match lobpcg_constrained_preconditioned(
                &a, Some(&b), 5,
                &nalgebra::DMatrix::zeros(0, 0), precond, &cfg,
            ) {
                Ok(eig) => {
                    *result_clone.lock().unwrap() = Some((eig.eigenvalues, eig.iterations, n_global));
                }
                Err(e) => eprintln!("  LOBPCG failed: {e}"),
            }
        }
    });

    let taken = result.lock().unwrap().take();
    if let Some((vals, iters, dof)) = taken {
        println!("\npex11: dofs={dof} iters={iters}");
        for (i, v) in vals.iter().enumerate() {
            println!("  lambda[{}] = {:.10e}", i, v);
        }
        // Verify: 5 positive eigenvalues, sorted ascending
        let ok = vals.len() == 5 && vals.iter().all(|&v| v > 0.0)
            && (1..vals.len()).all(|i| vals[i-1] <= vals[i]);
        println!("  {}", if ok { "PASS" } else { "FAIL" });
    }
}
