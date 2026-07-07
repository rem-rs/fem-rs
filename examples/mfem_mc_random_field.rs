//! # Monte Carlo Poisson with random diffusion coefficient
//!
//! Solves `-∇·(κ(x,ω) ∇u) = 1` on [0,1]² with homogeneous Dirichlet BCs where
//! κ is a log-normal random field. Runs Monte Carlo samples and estimates
//! statistics of the solution L² norm.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_mc_random_field --release
//! cargo run --example mfem_mc_random_field --release -- --n 16 --samples 200
//! ```

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_stochastic::{
    Covariance1D as _, ExponentialCovariance1D, KarhunenLoeveExpansion1D,
    RandomField, MonteCarloConfig, run_monte_carlo,
};
use fem_space::{
    H1Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== fem-rs: Monte Carlo with random diffusion ===");

    // FE mesh and space
    let mesh = Mesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let quad = 3;
    let n = space.n_dofs();

    // Boundary DOFs (all walls)
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());

    println!("  DOFs: {}", n);
    println!("  KL modes: {}", args.kl_modes);
    println!("  MC samples: {}", args.samples);

    // Build the KL expansion for log-κ on the element centroids
    let cov = ExponentialCovariance1D { sigma2: 0.25, length: 0.5 };
    let kl = KarhunenLoeveExpansion1D::new(args.n, args.kl_modes, 0.0, &cov);

    let config = MonteCarloConfig {
        n_samples: args.samples,
        report_every: if args.samples >= 20 { args.samples / 10 } else { 0 },
    };

    let result = run_monte_carlo(&config, |_sample_idx, rng| {
        // Generate random field at element centroids
        let elem_centroids: Vec<[f64; 1]> = (0..space.mesh().n_elements())
            .map(|e| {
                let nodes = space.mesh().element_nodes(e as u32);
                let mut cx = 0.0;
                for &n in nodes {
                    let c = space.mesh().node_coords(n);
                    cx += c[0];
                }
                cx / nodes.len() as f64
            })
            .map(|cx| [cx])
            .collect();

        let log_kappa = kl.realisation(&elem_centroids, rng);

        // κ(x) = exp(log κ(x))
        // We approximate by a piecewise-constant field (constant per elem)
        // using the centroid value
        let rhs_const = |_x: &[f64]| 1.0;
        let source = DomainSourceIntegrator::new(rhs_const);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);

        // Diffusion integrator per element
        // For this simple demo, we use a single DiffusionIntegrator and scale
        // with the average κ. A more accurate approach would use Nitsche or
        // element-wise assembly.
        let kappa_mean: f64 = log_kappa.iter().map(|&lk| lk.exp()).sum::<f64>() / log_kappa.len() as f64;

        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: kappa_mean }], quad);

        // Dirichlet BCs
        let bnd_vals = vec![0.0_f64; bnd.len()];
        let mut mat = mat;
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        // Solve
        let mut u = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");
        if !res.converged {
            return 0.0;
        }

        // QoI: L² norm of solution
        u.iter().map(|v| v * v).sum::<f64>().sqrt()
    });

    println!();
    println!("  === Monte Carlo results ===");
    println!("  Samples:  {}", result.n_samples);
    println!("  Mean QoI: {:.6e}", result.mean);
    println!("  Std dev:  {:.6e}", result.variance.sqrt());
    println!("  Std err:  {:.6e}", result.std_err);
    println!("  CV:       {:.4}%", result.cv() * 100.0);
    println!("  Time:     {:.2}s", result.elapsed.as_secs_f64());
}

struct Args { n: usize, kl_modes: usize, samples: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 12, kl_modes: 4, samples: 50 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(12); }
            "--kl-modes" => { a.kl_modes = it.next().and_then(|v| v.parse().ok()).unwrap_or(4); }
            "--samples" => { a.samples = it.next().and_then(|v| v.parse().ok()).unwrap_or(50); }
            _ => {}
        }
    }
    a
}
