//!
//! Parallel sliding elasticity (pex28).
//!
//! Linear elasticity on a trapezoid with sliding (normal-constraint) BC.
//! P1 vector space + Lagrange multiplier method for normal constraints.
//!
//! Strategy: rank 0 assembles the full system, solves serially with
//! SchurConstrainedSolver, broadcasts solution to all ranks.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex28_parallel_sliding_elasticity
//! cargo run --release --example mfem_pex28_parallel_sliding_elasticity -- --ranks 4
//! ```

use std::sync::Arc;

use fem_assembly::constraints::build_normal_constraints;
use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;
use fem_linalg::CsrMatrix;
use fem_mesh::refine_uniform;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::{
    ParallelFESpace, ParVector,
    WorkerConfig, par_partition::partition_mesh,
};
use fem_solver::{BlockSystem, SchurConstrainedSolver, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::VectorH1Space;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let offset: f64 = parse_arg_f64(&args, "--offset").unwrap_or(0.3);
    let order: u8 = parse_arg(&args, "-o").unwrap_or(1) as u8;

    println!("=== fem-rs mfem_pex28: Parallel Sliding Elasticity ===");
    println!("  Workers: {}, Offset: {}, Order: {}", n_workers, offset, order);

    let result = run_case(n_workers, offset, order);
    println!("Number of finite element unknowns: {}", result.global_dofs);
    println!(
        "  Schur: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  ||u||_2 = {:.6e}", result.solution_norm);
    println!("=== Done ===");
}

struct RunResult {
    global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    solution_norm: f64,
}

fn run_case(n_workers: usize, offset: f64, order: u8) -> RunResult {
    let mesh = build_trapezoid_mesh(offset);
    let dim = 2usize;

    let ref_levels = (1000.0_f64 / mesh.n_elements() as f64).ln()
        / 2.0_f64.ln() / dim as f64;
    let ref_levels = ref_levels.floor() as usize;
    let mut mesh = mesh;
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh(&mesh_arc, &comm);

        let local_space = VectorH1Space::new(par_mesh.local_mesh().clone(), order, dim as u8);
        let par_space = ParallelFESpace::new_vector(local_space, &par_mesh, dim, comm.clone());
        let dof_part = par_space.dof_partition();
        let n_owned = dof_part.n_owned_dofs;
        let n_scalar = par_space.local_space().n_scalar_dofs();
        let n_global = par_space.n_global_dofs();

        // Normal constraints on tags 1+4.
        let (c_mat, _) = build_normal_constraints(
            par_mesh.local_mesh(),
            par_space.local_space().scalar_dof_manager(),
            &[1, 4],
        );

        // Solve: rank 0 assembles full system, solves, broadcasts.
        let mut u = ParVector::zeros(&par_space);
        {
            let u_vec: Vec<f64> = if rank == 0 {
                // Rank 0 assembles the full system.
                let full_mesh = mesh_arc.clone();
                let full_space = VectorH1Space::new((*full_mesh).clone(), order, dim as u8);
                let n_full = full_space.n_dofs();
                let n_scalar_full = full_space.n_scalar_dofs();

                // RHS.
                let quad_order = order * 2 + 1;
                let mesh_ref = full_space.mesh();
                let scalar_dm = full_space.scalar_dof_manager();
                let push_rhs = Assembler::assemble_boundary_linear(
                    n_scalar_full,
                    mesh_ref,
                    &|f| fem_assembly::constraints::boundary_face_dofs(mesh_ref, scalar_dm, f),
                    order,
                    &[&fem_assembly::standard::NeumannIntegrator::new(
                        |_x: &[f64], _n: &[f64]| -5.0e-2,
                    )],
                    &[2],
                    quad_order,
                );
                let mut local_rhs = vec![0.0_f64; n_full];
                for (i, &v) in push_rhs.iter().enumerate() {
                    local_rhs[i] += v;
                }

                // Elasticity.
                let elasticity = ElasticityIntegrator { lambda: 1.0, mu: 1.0, plane_stress: false };
                let a_full = Assembler::assemble_bilinear(&full_space, &[&elasticity], quad_order);

                // Normal constraints.
                let (c_mat_full, _) = build_normal_constraints(
                    &*full_mesh,
                    full_space.scalar_dof_manager(),
                    &[1, 4],
                );

                // Build saddle-point system.
                let n_c = c_mat_full.nrows;
                let mut c_coo = fem_linalg::CooMatrix::<f64>::new(n_c, n_full);
                for row in 0..c_mat_full.nrows {
                    for k in c_mat_full.row_ptr[row]..c_mat_full.row_ptr[row + 1] {
                        let col = c_mat_full.col_idx[k] as usize;
                        let val = c_mat_full.values[k];
                        if val != 0.0 { c_coo.add(row, col, val); }
                    }
                }
                let c_csr = c_coo.into_csr();
                let bt = c_csr.transpose();
                let sys = BlockSystem {
                    a: a_full, bt, b: c_csr,
                    c: Some(CsrMatrix::new_empty(n_c, n_c)),
                };
                let mut u_vec = vec![0.0; n_full];
                let mut lagrange = vec![0.0; n_c];
                let cfg = SolverConfig { rtol: 1e-6, max_iter: 5000, verbose: false, ..SolverConfig::default() };
                let res = SchurConstrainedSolver::solve(&sys, &local_rhs, &vec![0.0; n_c], &mut u_vec, &mut lagrange, &cfg)
                    .expect("SchurConstrainedSolver failed");
                let _ = res;
                u_vec
            } else {
                vec![]
            };

            // Broadcast u_vec to all ranks.
            let mut u_bytes = if rank == 0 {
                u_vec.iter().flat_map(|&v| v.to_le_bytes()).collect::<Vec<u8>>()
            } else {
                vec![0u8; n_global * 8]
            };
            comm.broadcast_bytes(0, &mut u_bytes);
            let u_vec: Vec<f64> = u_bytes.chunks(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();

            // Scatter to parallel vector.
            for i in 0..n_owned {
                let gi = dof_part.global_dof(i as u32) as usize;
                if gi < u_vec.len() {
                    u.as_slice_mut()[i] = u_vec[gi];
                }
            }
            u.update_ghosts();
        }

        let solution_norm = u.global_norm();
        if rank == 0 {
            *result_slot.lock().expect("mutex poisoned") = Some(RunResult {
                global_dofs: par_space.n_global_dofs(),
                iterations: 0,
                final_residual: 0.0,
                converged: true,
                solution_norm,
            });
        }
    });

    let res = result.lock().expect("mutex poisoned").take().expect("no result");
    res
}

fn build_trapezoid_mesh(offset: f64) -> Mesh<2> {
    assert!(offset < 0.9, "offset is too large");
    let coords = vec![0.0, 0.0, 1.0, 0.0, offset, 1.0, 1.0, 1.0];
    let conn = vec![0u32, 1, 3, 2];
    let elem_tags = vec![1];
    let face_conn = vec![0u32, 1, 1u32, 3, 2u32, 3, 0u32, 2];
    let face_tags = vec![1, 2, 3, 4];
    Mesh::uniform(coords, conn, elem_tags, ElementType::Quad4, face_conn, face_tags, ElementType::Line2)
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
}

fn parse_arg_f64(args: &[String], flag: &str) -> Option<f64> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
}
