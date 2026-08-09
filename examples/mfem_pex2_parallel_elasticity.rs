//! # Parallel Example 2 — Parallel Linear Elasticity (1:1 port of MFEM pex2 / ex2p.cpp)
//!
//! Multi-material cantilever beam: `-div(sigma(u)) = 0` with
//! `sigma(u) = lambda div(u) I + mu (grad u + u grad)`, Lame coefficients
//! piecewise-constant per material (boundary/material attribute 1:
//! lambda = mu = 50, attribute 2: lambda = mu = 1).
//! BC: `u = 0` on boundary attribute 1 (fixed end); `sigma(u).n = (0, -1e-2)`
//! on boundary attribute 2 (downward pull), zero elsewhere.
//!
//! Matches MFEM `examples/ex2p.cpp` defaults: `beam-tri.mesh` (16 triangles),
//! serial `ref_levels` (≤1000 elems → 2) then 1 parallel uniform refinement
//! (4 total children per element, the last refinement done on the partitioned
//! mesh via [`fem_parallel::par_refine::par_uniform_refine`] — the distributed
//! counterpart of C++ `ParMesh::UniformRefinement`), H1 order 1, byNODES
//! vector ordering (C++ `-nodes` variant; Rust `VectorH1Space` is the byNODES
//! block layout), PCG + AMG V-cycle (C++ uses HyprePCG + BoomerAMG systems).
//!
//! Usage:
//!   cargo run --release --example mfem_pex2_parallel_elasticity
//!   cargo run --release --example mfem_pex2_parallel_elasticity -- --ranks 4
//!   cargo run --release --example mfem_pex2_parallel_elasticity -- --dump-sol output/pex2_sol.txt

use std::sync::Arc;

use fem_assembly::assembler::Assembler;
use fem_assembly::face_dofs_p1;
use fem_assembly::postproc::coefficient::PWConstCoeff;
use fem_assembly::standard::{ElasticityIntegrator, NeumannIntegrator};
use fem_mesh::{Mesh, refine_uniform};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::{
    ParAmgConfig, ParAssembler, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
    par_partition::partition_mesh, par_refine::par_uniform_refine, par_solve_pcg_amg,
};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let dump_sol = parse_arg_str(&args, "--dump-sol");

    println!("=== fem-rs mfem_pex2: Parallel Linear Elasticity ===");
    println!("  Workers: {}, Mesh: beam-tri.mesh x4 (1:1 MFEM ex2p)", n_workers);

    let result = run_case(n_workers, dump_sol);
    println!("Number of finite element unknowns: {}", result.global_dofs);
    println!(
        "  PCG: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    let avg = result
        .final_residual
        .powf(1.0 / (result.iterations.max(1) as f64));
    println!("  Average reduction factor = {:.6}", avg);
    println!(
        "  ||u||_2 = {:.6e}, sum = {:.8e}, checksum = {:.8e}",
        result.solution_norm, result.solution_sum, result.solution_checksum
    );
    println!("=== Done ===");
}

struct RunResult {
    global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    solution_norm: f64,
    solution_sum: f64,
    solution_checksum: f64,
}

fn run_case(n_workers: usize, dump_sol: Option<String>) -> RunResult {
    // 1:1 with MFEM ex2p.cpp: beam-tri.mesh + serial ref_levels (≤1000 elems:
    // floor(log(1000/16)/log(2)/2) = 2) + 1 *parallel* uniform refinement
    // (ParMesh::UniformRefinement) = 3 total refinements, the last one done
    // after partitioning exactly like C++.
    let mfem = fem_io::mfem::read_mfem_file("data/beam-tri.mesh")
        .expect("failed to read data/beam-tri.mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("beam-tri.mesh must be 2-D");
    for _ in 0..2 {
        mesh = refine_uniform(&mesh);
    }
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        // 1. Partition the refined mesh.
        let mut par_mesh = partition_mesh(&mesh_arc, &comm);
        let rank = comm.rank();

        // 2. Parallel uniform refinement (C++ ParMesh::UniformRefinement).
        par_mesh = par_uniform_refine(&par_mesh);

        // 2. Vector H1 FE space (dim=2), byNODES block layout.
        let order: u8 = 1;
        let dim = 2usize;
        let local_space = VectorH1Space::new(par_mesh.local_mesh().clone(), order, dim as u8);
        let par_space = ParallelFESpace::new_vector(local_space, &par_mesh, dim, comm.clone());
        let dof_part = par_space.dof_partition();
        let n_scalar = par_space.local_space().n_scalar_dofs();

        // 3. Essential BCs: u = 0 on boundary attribute 1 (both components).
        let mesh_ref = par_space.local_space().mesh();
        let bnd_scalar = boundary_dofs(mesh_ref, par_space.local_space().scalar_dof_manager(), &[1]);
        let mut clamped: Vec<u32> = Vec::with_capacity(bnd_scalar.len() * 2);
        for &d in &bnd_scalar {
            clamped.push(d);                   // x-component DOF
            clamped.push(d + n_scalar as u32); // y-component DOF
        }

        // 4. RHS: downward traction on boundary attribute 2.
        //    f = (0, -1e-2): scalar Neumann integral over tag 2 → y block.
        let quad_order = order * 2 + 1;
        let fdofs = face_dofs_p1(mesh_ref);
        let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
        let traction_y = Assembler::assemble_boundary_linear(
            n_scalar,
            mesh_ref,
            &fdofs,
            order,
            &[&neumann],
            &[2],
            quad_order,
        );
        let mut local_rhs = vec![0.0_f64; par_space.local_space().n_dofs()];
        for (i, &v) in traction_y.iter().enumerate() {
            local_rhs[n_scalar + i] += v;
        }
        // Permute DofManager (space) ordering → partition [owned|ghost] ordering.
        let n_total = dof_part.n_total_dofs();
        let mut rhs_data = vec![0.0_f64; n_total];
        for (i, &v) in local_rhs.iter().enumerate() {
            rhs_data[dof_part.permute_dof(i as u32) as usize] = v;
        }
        let mut rhs = ParVector::from_local_raw(
            rhs_data,
            dof_part.n_owned_dofs,
            par_space.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // 5. Stiffness: piecewise-constant lambda/mu per element attribute.
        let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
        let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
        let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
        let mut a_mat =
            ParAssembler::assemble_bilinear(&par_space, &[&elasticity], quad_order);

        // 6. Eliminate essential BCs.
        for &d in &clamped {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs {
                a_mat.apply_dirichlet_par(pid, 0.0, &mut rhs);
            }
        }

        // 7. Solve: PCG + AMG V-cycle (C++: HyprePCG + BoomerAMG systems,
        //    tol 1e-8, max_iter 500).
        let mut u = ParVector::zeros(&par_space);
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 5000,
            verbose: false,
            ..SolverConfig::default()
        };
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            ..Default::default()
        };
        let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg).unwrap();

        // 8. Global norms + optional solution dump (like MFEM sol.XXXXXX).
        let n_owned = dof_part.n_owned_dofs;
        let solution_norm = u.global_norm();
        let solution_sum =
            comm.allreduce_sum_f64(u.as_slice()[..n_owned].iter().sum::<f64>());
        let local_checksum: f64 = (0..n_owned)
            .map(|pid| {
                let gid = dof_part.global_dof(pid as u32) as f64 + 1.0;
                gid * u.as_slice()[pid]
            })
            .sum();
        let solution_checksum = comm.allreduce_sum_f64(local_checksum);

        if let Some(ref path) = dump_sol {
            let owned: Vec<(u32, f64)> = (0..n_owned)
                .map(|pid| (dof_part.global_dof(pid as u32), u.as_slice()[pid]))
                .collect();
            if rank == 0 {
                let mut all: Vec<(u32, f64)> = owned.clone();
                for src in 1..comm.size() as i32 {
                    let gids: Vec<u32> = comm.recv(src, 81);
                    let vals: Vec<f64> = comm.recv(src, 82);
                    all.extend(gids.into_iter().zip(vals));
                }
                all.sort_unstable_by_key(|&(g, _)| g);
                let mut s = String::new();
                for (_, v) in all {
                    // MFEM ex2p.cpp saves the inverted solution (x *= -1), so
                    // dump -u to match sol.XXXXXX output.
                    s.push_str(&format!("{:.10e}\n", -v));
                }
                std::fs::write(path, s).expect("failed to write solution dump");
            } else {
                let gids: Vec<u32> = owned.iter().map(|&(g, _)| g).collect();
                let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                comm.send(0, 81, &gids);
                comm.send(0, 82, &vals);
            }
        }

        if rank == 0 {
            *result_slot.lock().expect("pex2 result mutex poisoned") = Some(RunResult {
                global_dofs: par_space.n_global_dofs(),
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                solution_norm,
                solution_sum,
                solution_checksum,
            });
        }
    });

    let final_result = result
        .lock()
        .expect("pex2 result mutex poisoned after launch")
        .take()
        .expect("rank 0 did not publish pex2 result");
    final_result
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_arg_str<'a>(args: &'a [String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
