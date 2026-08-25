//!
//! Parallel hyperelastic dynamics (pex10).
//!
//! Nonlinear elasticity with NeoHookean model, Forward Euler time integration.
//! Strategy: rank 0 assembles full system, does time integration, broadcasts.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex10_parallel_hyperelastic
//! cargo run --release --example mfem_pex10_parallel_hyperelastic -- --ranks 4
//! ```

use fem_assembly::{
    Assembler,
    standard::{VectorDiffusionIntegrator, VectorH1MassIntegrator},
    HyperelasticModel, HyperelasticityForm,
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{Mesh, refine_uniform};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;
use fem_space::VectorH1Space;

struct Args {
    mesh: String,
    ref_levels: usize,
    order: u8,
    t_final: f64,
    dt: f64,
    viscosity: f64,
    mu: f64,
    K: f64,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/beam-quad.mesh".into(),
            ref_levels: 0,
            order: 1,
            t_final: 0.5,
            dt: 0.1,
            viscosity: 1e-2,
            mu: 0.25,
            K: 5.0,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-r" | "--refine" => { a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
                "-tf" | "--t-final" => { a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5); }
                "-dt" | "--time-step" => { a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.1); }
                _ => {}
            }
        }
        a
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum() }

fn main() {
    let args = Args::parse();
    let n_workers: usize = std::env::args()
        .position(|a| a == "--ranks")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("=== fem-rs mfem_pex10: Parallel Hyperelastic Dynamics ===");
    println!("  Workers: {}, Mesh: {}, Refine: {}, Order: {}", n_workers, args.mesh, args.ref_levels, args.order);

    let result = std::sync::Arc::new(std::sync::Mutex::new(None::<(usize, f64, f64)>));
    let result_slot = result.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        let (n_total, ee_final, ke_final) = if rank == 0 {
            let mut mesh: Mesh<2> = read_mfem_file(&args.mesh).expect("failed to read mesh").mesh2d.expect("2D mesh");
            for _ in 0..args.ref_levels { mesh = refine_uniform(&mesh); }
            let dim = 2usize;
            let space = VectorH1Space::new(mesh.clone(), args.order, dim as u8);
            let n_total = space.n_scalar_dofs() * dim;
            let quad_order = args.order * 2 + 1;

            let m_integ = VectorH1MassIntegrator { kappa: 1.0 };
            let m = Assembler::assemble_bilinear(&space, &[&m_integ], quad_order);
            let s_integ = VectorDiffusionIntegrator { kappa: args.viscosity };
            let s = Assembler::assemble_bilinear(&space, &[&s_integ], quad_order);

            let model = HyperelasticModel::MfemNeoHookean { mu: args.mu, bulk_modulus: args.K };
            let hyper = HyperelasticityForm::new(space, model, vec![], quad_order);

            let n = n_total;
            let mut vx = vec![0.0; 2 * n];
            let (v_block, x_block) = vx.split_at_mut(n);
            
            // Initial velocity (parabolic profile)
            let n_scalar = n / dim;
            for i in 0..n {
                let c = i / n_scalar; // component (0=x, 1=y)
                let s = i % n_scalar; // scalar DOF index
                // Approximate x-coordinate from DOF index (beam is [0,8]x[0,1])
                let x = (s as f64) / (n_scalar as f64) * 8.0;
                v_block[i] = 0.0;
                if c == 0 {
                    v_block[i] = -0.1 / 64.0 * x * x;
                } else if c == 1 {
                    v_block[i] = 0.1 / 64.0 * x * x * (8.0 - x);
                }
            }

            let mut t = 0.0;
            let mut step = 0;
            let dt = args.dt;
            while t < args.t_final {
                let dt_real = dt.min(args.t_final - t);
                step += 1;
                let mut rhs = vec![0.0; n];
                hyper.raw_residual(x_block, &mut rhs);
                let mut sv = vec![0.0; n];
                s.spmv(v_block, &mut sv);
                for i in 0..n { rhs[i] += sv[i]; }
                for i in 0..n { rhs[i] = -rhs[i]; }
                let mut dv = vec![0.0; n];
                let cfg = fem_solver::SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 30, verbose: false, ..Default::default() };
                fem_solver::solve_pcg_jacobi(&m, &rhs, &mut dv, &cfg).ok();
                for i in 0..n {
                    v_block[i] += dt_real * dv[i];
                    x_block[i] += dt_real * v_block[i];
                }
                t += dt_real;
            }

            let ee = hyper.elastic_energy(x_block);
            let mut mv_tmp = vec![0.0; n];
            m.spmv(v_block, &mut mv_tmp);
            let ke = 0.5 * dot(v_block, &mv_tmp);
            println!("step {step}, t = {t:.1}, EE = {ee:.6}, KE = {ke:.6}");
            (n_total, ee, ke)
        } else {
            (0, 0.0, 0.0)
        };

        let mut n_bytes = if rank == 0 { (n_total as u64).to_le_bytes().to_vec() } else { vec![0u8; 8] };
        comm.broadcast_bytes(0, &mut n_bytes);
        let n_total: usize = u64::from_le_bytes(n_bytes.try_into().unwrap()) as usize;

        if rank == 0 { *result_slot.lock().unwrap() = Some((n_total, ee_final, ke_final)); }
    });

    let (n_total, ee, ke) = result.lock().unwrap().take().unwrap_or((0, 0.0, 0.0));
    println!("Number of unknowns: {}", n_total);
    println!("EE = {:.6}, KE = {:.6}", ee, ke);
    println!("=== Done ===");
}
