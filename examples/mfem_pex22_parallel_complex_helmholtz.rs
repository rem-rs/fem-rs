//! # Parallel Example 22 鈥?Complex Helmholtz  (1:1 with MFEM pex22p)
//!
//! Solves `-Div(a Grad u) - w^2 b u + i w c u = f` with damping.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_pex22_parallel_complex_helmholtz -- --ranks 2
//! cargo run --example mfem_pex22_parallel_complex_helmholtz -- --ranks 1 --freq 2.0
//! ```

use std::sync::{Arc, Mutex};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator};
use fem_linalg::{CooMatrix, SolverConfig};
use fem_mesh::Mesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace,
    par_partition::partition_mesh,
    launcher::native::ThreadLauncher, WorkerConfig,
};
use fem_space::{H1Space, constraints::boundary_dofs};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = a.iter().position(|x| x == "--n").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(16);
    let r = a.iter().position(|x| x == "--ranks").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let freq = a.iter().position(|x| x == "--freq").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let omega = 2.0 * std::f64::consts::PI * freq;

    let mesh = Arc::new(Mesh::<2>::unit_square_tri(n));
    let result = Arc::new(Mutex::new(None));
    let r2 = Arc::clone(&result);

    ThreadLauncher::new(WorkerConfig::new(r)).launch(move |comm| {
        let rank = comm.rank();
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let ps = ParallelFESpace::new(H1Space::new(lm.clone(), 1), &pm, comm.clone());
        let dp = ps.dof_partition();
        let n_owned = dp.n_owned_dofs;
        let qo = 3u8;

        // Real part: K_re = -div(grad) - w^2 * mass
        let mut k_re = ParAssembler::assemble_bilinear(&ps, &[
            &DiffusionIntegrator { kappa: 0.0 },
            &MassIntegrator { rho: -omega * omega },
        ], qo);

        // Imaginary part: K_im = w * c * mass (damping, c=1)
        let mut k_im = ParAssembler::assemble_bilinear(&ps, &[
            &MassIntegrator { rho: omega * 20.0 },
        ], qo);

        // RHS: Gaussian source
        let f = |x: &[f64]| { let r2 = (x[0]-0.5).powi(2)+(x[1]-0.5).powi(2); (-10.0*r2).exp() };
        let mut rhs_re = ParAssembler::assemble_linear(&ps, &[&DomainSourceIntegrator::new(&f)], qo);
        let mut rhs_im = ParVector::zeros(&ps);

        // BC: u=0 on all boundaries
        let bd = boundary_dofs(&lm, ps.local_space().dof_manager(), &lm.unique_boundary_tags());
        for &d in &bd {
            let p = dp.permute_dof(d) as usize;
            if p < n_owned {
                k_re.apply_dirichlet_par(p, 1.0, &mut rhs_re);
                k_im.apply_dirichlet_par(p, 0.0, &mut rhs_im);
            }
        }

        // Build 2x2 block system on local diag: [K_re  -K_im; K_im  K_re]
        let d_re = k_re.diag_block();
        let d_im = k_im.diag_block();
        let n2 = 2 * n_owned;
        let mut coo = CooMatrix::new(n2, n2);
        for i in 0..n_owned {
            let s = d_re.row_ptr[i]; let e = d_re.row_ptr[i+1];
            for k in s..e {
                let j = d_re.col_idx[k] as usize; let v = d_re.values[k];
                if v != 0.0 { coo.add(i, j, v); coo.add(n_owned+i, n_owned+j, v); }
            }
        }
        for i in 0..n_owned {
            let s = d_im.row_ptr[i]; let e = d_im.row_ptr[i+1];
            for k in s..e {
                let j = d_im.col_idx[k] as usize; let v = d_im.values[k];
                if v != 0.0 { coo.add(i, n_owned+j, -v); coo.add(n_owned+i, j, v); }
            }
        }
        let a_block = coo.into_csr();

        let rhs_s = rhs_re.as_slice();
        let im_s  = rhs_im.as_slice();
        let mut b = vec![0.0; n2];
        for i in 0..n_owned { b[i] = rhs_s[i]; b[n_owned+i] = im_s[i]; }

        let mut x = vec![0.0; n2];
        let res = fem_solver::solve_gmres_ilu0(&a_block, &b, &mut x, 50,
            &SolverConfig { rtol:1e-6, max_iter:2000, verbose:rank==0, ..Default::default() });

        match res {
            Ok(info) => {
                if rank == 0 {
                    println!("pex22: dofs={} conv={} iters={} res={:.3e}",
                        ps.n_global_dofs(), info.converged, info.iterations, info.final_residual);
                }
                *r2.lock().unwrap() = Some((info.converged, info.iterations, ps.n_global_dofs()));
            }
            Err(e) => { eprintln!("rank {rank}: GMRES failed: {e:?}"); }
        }
    });
    let _g = result.lock().unwrap();
}
