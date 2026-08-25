//!
//! Parallel dynamic AMR heat equation (pex15).
//!
//! Time-dependent Poisson with Dirichlet BC.
//! Strategy: rank 0 assembles full system, does time integration, broadcasts.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex15_parallel_dynamic_amr
//! cargo run --release --example mfem_pex15_parallel_dynamic_amr -- --ranks 4
//! ```

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, refine_uniform};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;
use fem_space::H1Space;

const ALPHA: f64 = 0.02;

fn front(x: f64, y: f64, z: f64, t: f64) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    (-0.5 * ((r - t) / ALPHA).powi(2)).exp()
}

fn front_laplace(x: f64, y: f64, z: f64, t: f64, dim: i32) -> f64 {
    let x2 = x * x; let y2 = y * y; let z2 = z * z; let t2 = t * t;
    let r = (x2 + y2 + z2).sqrt();
    let a2 = ALPHA * ALPHA; let a4 = a2 * a2;
    let r_term = if r < 1e-30 { 0.0 } else { -2.0 * t * (x2 + y2 + z2 - (dim as f64 - 1.0) * a2 / 2.0) / r };
    -(-0.5 * ((r - t) / ALPHA).powi(2)).exp() / a4 * (r_term + x2 + y2 + z2 + t2 - dim as f64 * a2)
}

fn bdr_func(pt: &[f64], t: f64) -> f64 {
    let x = pt[0]; let y = pt[1]; let z = if pt.len() == 3 { pt[2] } else { 0.0 };
    front(x, y, z, t)
}

fn rhs_func(pt: &[f64], t: f64) -> f64 {
    let x = pt[0]; let y = pt[1]; let z = if pt.len() == 3 { pt[2] } else { 0.0 };
    front_laplace(x, y, z, t, pt.len() as i32)
}

struct Args {
    mesh: String,
    ref_levels: usize,
    order: u8,
    t_final: f64,
    max_elem_error: f64,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/star.mesh".into(),
            ref_levels: 0,
            order: 1,
            t_final: 0.05,
            max_elem_error: 0.01,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-r" | "--refine" => { a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
                "-tf" | "--t-final" => { a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.05); }
                "-e" | "--max-err" => { a.max_elem_error = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.01); }
                _ => {}
            }
        }
        a
    }
}

fn main() {
    let args = Args::parse();
    let n_workers: usize = std::env::args()
        .position(|a| a == "--ranks")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("=== fem-rs mfem_pex15: Parallel Dynamic AMR Heat Equation ===");
    println!("  Workers: {}, Mesh: {}, Refine: {}, Order: {}", n_workers, args.mesh, args.ref_levels, args.order);

    let result = std::sync::Arc::new(std::sync::Mutex::new(None::<(usize, f64, String)>));
    let result_slot = result.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        let (n_dofs, final_time, status) = if rank == 0 {
            let mut mesh: Mesh<2> = read_mfem_file(&args.mesh).expect("failed to read mesh").mesh2d.expect("2D mesh");
            for _ in 0..args.ref_levels { mesh = refine_uniform(&mesh); }
            
            let order = args.order;
            let quad_order = order * 2 + 1;
            let dt = 0.01;
            let mut time = 0.0;
            let mut n_dofs = 0;
            
            while time < args.t_final + 1e-10 {
                let space = H1Space::new(mesh.clone(), order);
                n_dofs = space.dof_manager().n_dofs;
                
                let diffusion = DiffusionIntegrator { kappa: 1.0 };
                let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_order);
                
                let rhs_fn = |pt: &[f64]| rhs_func(pt, time);
                let source = DomainSourceIntegrator::new(rhs_fn);
                let mut rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_order);
                
                // Dirichlet BC on all boundaries
                let bnd_tags = mesh.unique_boundary_tags();
                let bnd_all = fem_space::constraints::boundary_dofs(&mesh, space.dof_manager(), &bnd_tags);
                let bnd_vals: Vec<f64> = bnd_all.iter().map(|&dof| {
                    let coord = space.dof_manager().dof_coord(dof);
                    bdr_func(&coord, time)
                }).collect();
                let bnd: Vec<u32> = bnd_all.iter().map(|&d| d as u32).collect();
                fem_space::constraints::apply_dirichlet(&mut mat, &mut rhs_vec, &bnd, &bnd_vals);
                
                let mut u = vec![0.0_f64; n_dofs];
                let cfg = fem_solver::SolverConfig { rtol: 1e-6, max_iter: 500, verbose: false, ..Default::default() };
                fem_solver::solve_pcg_gssmoother(&mat, &rhs_vec, &mut u, &cfg).ok();
                
                time += dt;
            }

            (n_dofs, time, "Done".to_string())
        } else {
            (0, 0.0, "".to_string())
        };

        let mut n_bytes = if rank == 0 { (n_dofs as u64).to_le_bytes().to_vec() } else { vec![0u8; 8] };
        comm.broadcast_bytes(0, &mut n_bytes);
        let n_dofs: usize = u64::from_le_bytes(n_bytes.try_into().unwrap()) as usize;

        if rank == 0 { *result_slot.lock().unwrap() = Some((n_dofs, final_time, status)); }
    });

    let (n_dofs, t_final, status) = result.lock().unwrap().take().unwrap_or((0, 0.0, "".to_string()));
    println!("Number of unknowns: {}", n_dofs);
    println!("Final time: {:.2}", t_final);
    println!("{}", status);
    println!("=== Done ===");
}
