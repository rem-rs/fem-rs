//! # Parallel Example 3 — Maxwell cavity  (analogous to MFEM pex3)
//!
//! Solves the vector curl-curl + mass problem in parallel:
//!
//! ```text
//!   ∇×(∇×E) + E = f    in Ω = [0,1]²
//!          n×E = 0    on ∂Ω
//! ```
//!
//! Uses H(curl) Nédélec ND1 edge elements, partitioned across workers.
//!
//! ## Usage
//! ```
//! cargo run --example pex3_maxwell
//! cargo run --example pex3_maxwell -- --n 16 --ranks 4
//! ```

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::{
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    ParVectorAssembler, ParVector, ParallelFESpace,
    par_simplex::partition_simplex,
    par_solve_pcg_jacobi,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::{HCurlSpace, fe_space::FESpace};
use fem_space::constraints::boundary_dofs_hcurl;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let n_workers = parse_arg(&args, "--ranks").unwrap_or(2);
    let mesh_n = parse_arg(&args, "--n").unwrap_or(16);

    println!("=== fem-rs pex3: Parallel Maxwell (ND1) ===");
    println!("  Workers: {n_workers}, Mesh: {mesh_n}x{mesh_n}");

    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(mesh_n));

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition mesh.
        let par_mesh = partition_simplex(&mesh, &comm);

        // 2. Build parallel H(curl) space.
        let local_space = HCurlSpace::new(par_mesh.local_mesh().clone(), 1);
        let par_space = ParallelFESpace::new_for_edge_space(
            local_space, &par_mesh, comm.clone(),
        );

        if rank == 0 {
            println!("  Global DOFs: {}", par_space.n_global_dofs());
        }

        // 3. Assemble (∇×∇× + I).
        let curl_curl = CurlCurlIntegrator { mu: 1.0 };
        let vec_mass = VectorMassIntegrator { alpha: 1.0 };
        let mut a_mat = ParVectorAssembler::assemble_bilinear(
            &par_space, &[&curl_curl, &vec_mass], 4,
        );

        // 4. Assemble RHS: f = ((1+π²)sin(πy), (1+π²)sin(πx)).
        let source = MaxwellSource;
        let mut rhs = ParVectorAssembler::assemble_linear(&par_space, &[&source], 4);

        // 5. Apply n×E = 0 on all boundary edges.
        let bnd = boundary_dofs_hcurl(
            par_space.local_space().mesh(),
            par_space.local_space(),
            &[1, 2, 3, 4],
        );
        let dof_part = par_space.dof_partition();
        for &d in &bnd {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs {
                a_mat.apply_dirichlet_par(pid, 0.0, &mut rhs);
            }
        }

        // 6. Solve with parallel PCG + Jacobi.
        let mut u = ParVector::zeros(&par_space);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
        let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

        if rank == 0 {
            println!(
                "  PCG: {} iters, residual = {:.3e}, converged = {}",
                res.iterations, res.final_residual, res.converged
            );
            let h = 1.0 / mesh_n as f64;
            println!("  h = {h:.4e}  (expected O(h) error for ND1)");
            println!("=== Done ===");
        }
    });
}

// ─── Manufactured source ────────────────────────────────────────────────────

struct MaxwellSource;

impl VectorLinearIntegrator for MaxwellSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        let coeff = 1.0 + PI * PI;
        let fx = coeff * (PI * x[1]).sin();
        let fy = coeff * (PI * x[0]).sin();

        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
            f_elem[i] += qp.weight * dot;
        }
    }
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
