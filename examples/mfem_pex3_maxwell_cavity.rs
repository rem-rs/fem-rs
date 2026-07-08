//! # Parallel Example 3 — Maxwell cavity  (1:1 with MFEM pex3)
//!
//! Solves `∇×(∇×E) + E = f` with PEC BC, in parallel.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_pex3_maxwell_cavity -- -m data/star.mesh --ranks 4
//! cargo run --example mfem_pex3_maxwell_cavity -- --n 16 --ranks 4
//! cargo run --example mfem_pex3_maxwell_cavity -- -m data/star.mesh --ranks 2 -r 2
//! ```

use std::f64::consts::PI;
use std::io::Write;
use std::sync::{Arc, Mutex};

use fem_assembly::{
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{Mesh, amr::refine_uniform};
use fem_parallel::{
    ParVectorAssembler, ParVector, ParallelFESpace,
    par_partition::partition_mesh, par_solve_pcg_jacobi,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::{HCurlSpace, fe_space::FESpace, constraints::boundary_dofs_hcurl};

struct Src { kappa: f64 }
impl VectorLinearIntegrator for Src {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
        let k = self.kappa;
        let c = 1.0 + k * k;
        let fx = c * (k * qp.x_phys[1]).sin();
        let fy = c * (k * qp.x_phys[0]).sin();
        for i in 0..qp.n_dofs {
            fe[i] += qp.weight * (qp.phi_vec[i*2]*fx + qp.phi_vec[i*2+1]*fy);
        }
    }
}

#[allow(dead_code)]
fn exact_e(x: &[f64], kappa: f64) -> [f64; 2] {
    [(kappa * x[1]).sin(), (kappa * x[0]).sin()]
}

#[allow(unused_variables, unused_assignments)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file: Option<String> = None;
    let mut n = 16usize;
    let mut order = 1u8;
    let mut ranks = 2usize;
    let mut ref_levels = 2usize;
    let mut freq = 1.0_f64;
    let mut visualization = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--mesh" => { i += 1; mesh_file = Some(args[i].clone()); }
            "--n" => { i += 1; n = args[i].parse().unwrap_or(16); }
            "-o" | "--order" => { i += 1; order = args[i].parse().unwrap_or(1); }
            "--ranks" => { i += 1; ranks = args[i].parse().unwrap_or(2); }
            "-r" | "--refine" => { i += 1; ref_levels = args[i].parse().unwrap_or(0); }
            "-f" | "--frequency" => { i += 1; freq = args[i].parse().unwrap_or(1.0); }
            "-vis" | "--visualization" => { visualization = true; }
            "-no-vis" | "--no-visualization" => { visualization = false; }
            _ => {}
        }
        i += 1;
    }

    let base_mesh: Mesh<2> = if let Some(ref path) = mesh_file {
        read_mfem_file(path).expect("failed to read MFEM mesh")
            .mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(n)
    };
    let mesh = Arc::new(if ref_levels > 0 {
        let mut m = base_mesh;
        for _ in 0..ref_levels { m = refine_uniform(&m); }
        m
    } else { base_mesh });

    let kappa = freq * PI;
    let quad_order = order as u8 * 2 + 2;
    let result = Arc::new(Mutex::new(None));
    let r2 = result.clone();

    ThreadLauncher::new(WorkerConfig::new(ranks)).launch(move |comm| {
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let ps = ParallelFESpace::new_for_edge_space(HCurlSpace::new(lm, order), &pm, comm.clone());
        let n_global = ps.n_global_dofs();

        if comm.rank() == 0 {
            println!("Options: mesh={} order={order} quad_order={quad_order} ranks={ranks}", mesh_file.as_deref().unwrap_or("built-in"));
            println!("Number of finite element unknowns: {n_global}");
        }

        let mut stiff = ParVectorAssembler::assemble_bilinear(&ps, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], quad_order);
        let mut rhs = ParVectorAssembler::assemble_linear(&ps, &[&Src { kappa }], quad_order);

        // PEC BC — zero tangential field on all boundaries
        let bdr = boundary_dofs_hcurl(ps.local_space().mesh(), ps.local_space(), &[1]);
        let dp = ps.dof_partition();
        for &d in &bdr {
            let p = dp.permute_dof(d) as usize;
            if p < dp.n_owned_dofs { stiff.apply_dirichlet_par(p, 0.0, &mut rhs); }
        }

        let mut u = ParVector::zeros(&ps);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 10000, verbose: false, ..Default::default() };
        let res = par_solve_pcg_jacobi(&stiff, &rhs, &mut u, &cfg)
            .expect("PCG solve failed");

        if comm.rank() == 0 {
            println!("PCG Iterations = {}", res.iterations);
            println!("Final PCG Relative Residual Norm = {:.6e}", res.final_residual);
        }

        // Save mesh and solution per rank (matching MFEM pex3 format).
        {
            let mesh_name = format!("mesh.{:06}", comm.rank());
            let sol_name = format!("sol.{:06}", comm.rank());
            let mut mesh_f = std::fs::File::create(&mesh_name)
                .expect("cannot create mesh file");
            write_mfem(&mut mesh_f, ps.local_space().mesh(), None)
                .expect("mesh write failed");
            let mut sol_f = std::fs::File::create(&sol_name)
                .expect("cannot create sol file");
            for &v in u.owned_slice() {
                writeln!(sol_f, "{:.14e}", v).expect("sol write failed");
            }
        }
        if comm.rank() == 0 {
            eprintln!("  Wrote mesh.XXXXXX and sol.XXXXXX per rank");
        }

        *r2.lock().unwrap() = Some((n_global, res.iterations, res.final_residual));
    });

    let taken = result.lock().unwrap().take();
    if let Some((dofs, iters, res)) = taken {
        println!("pex3: dofs={dofs} iters={iters} residual={res:.3e}");
    }
}
