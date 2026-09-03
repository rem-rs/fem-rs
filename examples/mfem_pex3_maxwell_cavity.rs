//! # Parallel Example 3 — Maxwell cavity  (1:1 with MFEM pex3 / ex3p.cpp)
//!
//! Solves `curl curl E + E = f` (electromagnetic diffusion) with PEC BC
//! (`E × n = 0`, non-homogeneous: projected exact tangential values), in
//! parallel.  Matches MFEM examples/ex3p.cpp defaults: star.mesh, serial
//! ref_levels (≤1000 elems → 2) then 2 parallel uniform refinements
//! (4 total, done serially before partitioning), ND1 (ND_FECollection(1,2))
//! H(curl) space, muinv = sigma = 1, exact solution
//! `E = (sin(κy), sin(κx))`, `f = (1+κ²)·E`, κ = π.  C++ uses HyprePCG + HypreAMS;
//! Rust uses PCG + AMG (DIAG_KEEP symmetric elimination, matching C++ FormLinearSystem).
//!
//! ## Usage
//! ```bash
//! cargo run --release --example mfem_pex3_maxwell_cavity
//! cargo run --release --example mfem_pex3_maxwell_cavity -- --ranks 4
//! cargo run --release --example mfem_pex3_maxwell_cavity -- --n 16 --ranks 4
//! ```

use std::f64::consts::PI;
use std::io::Write;
use std::sync::{Arc, Mutex};

use fem_assembly::{
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_element::{VectorReferenceElement, nedelec::{TriNDk, QuadNDk}};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_io::glvis::GlVisSocket;
use fem_mesh::{ElementType, Mesh, MeshTopology, amr::refine_uniform};
use fem_parallel::{
    ParVectorAssembler, ParVector, ParallelFESpace,
    par_partition::partition_mesh,
    WorkerConfig, DofPartition, ParAmgConfig, SmootherType, par_solve_pcg_amg,
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

    // 1:1 default: star.mesh + 4 uniform refinements (2 serial ref_levels +
    // 2 parallel refinements equivalent; done serially before partitioning).
    // `--n N` keeps the unit-square triangle self-test path.
    let base_mesh: Mesh<2> = if let Some(ref path) = mesh_file {
        read_mfem_file(path).expect("failed to read MFEM mesh")
            .mesh2d.expect("MFEM mesh must be 2D")
    } else if n != 16 {
        Mesh::<2>::unit_square_tri(n)
    } else {
        let mfem = read_mfem_file("data/star.mesh")
            .expect("failed to read data/star.mesh");
        let m = mfem.mesh2d.expect("star.mesh must be 2-D");
        let mut m = m;
        for _ in 0..4 {
            m = refine_uniform(&m);
        }
        m
    };
    let mesh = Arc::new(if ref_levels > 0 && n != 16 {
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

        // PEC BC — zero tangential field on all boundaries.  Non-homogeneous
        // (MFEM ex3p: x.ProjectCoefficient(E)): DIAG_KEEP elimination (symmetric
        // system, matching C++ FormLinearSystem) + PCG + AMG (C++: HyprePCG + HypreAMS).
        let bdr = boundary_dofs_hcurl(ps.local_space().mesh(), ps.local_space(), &[1]);
        let dp = ps.dof_partition();
        let n_owned = dp.n_owned_dofs;

        // Initial guess x = projection of E_exact (dm order → partition order).
        let x0 = ps.local_space().interpolate_vector(&|p| exact_e(p, kappa).to_vec());
        let x0_perm = fem_parallel::par_assembler::permute_vec(x0.as_slice(), dp);
        let mut u = ParVector::from_local_raw(
            x0_perm,
            n_owned,
            ps.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // Collect global IDs of locally-essential DOFs for cross-rank exchange
        // (matching C++ GetEssentialTrueDofs + parallel distribution).
        let local_bnd_global: Vec<u32> = bdr
            .iter()
            .map(|&d| dp.global_dof(dp.permute_dof(d)))
            .collect();
        let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
        for r in 0..comm.size() as i32 {
            if r == comm.rank() { continue; }
            let mut bytes = Vec::with_capacity(local_bnd_global.len() * 4);
            for &g in &local_bnd_global {
                bytes.extend_from_slice(&g.to_le_bytes());
            }
            sends.push((r, bytes));
        }
        let incoming = comm.alltoallv_bytes(&sends);
        let mut all_bnd: std::collections::HashSet<u32> = local_bnd_global.iter().copied().collect();
        for (_, bytes) in incoming {
            for chunk in bytes.chunks_exact(4) {
                all_bnd.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        // DIAG_KEEP elimination for owned essential DOFs (symmetric, keeps diag).
        let clamped: Vec<usize> = (0..dp.n_owned_dofs)
            .filter(|&pid| all_bnd.contains(&dp.global_dof(pid as u32)))
            .collect();
        for &pid in &clamped {
            let bc_val = u.owned_slice()[pid];
            stiff.apply_dirichlet_par_keep_diag(pid, bc_val, &mut rhs);
        }

        let cfg = SolverConfig { rtol: 1e-8, max_iter: 10000, verbose: false, ..Default::default() };
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            block_size: 1,
            use_global_aggregation: false,
            ..ParAmgConfig::default()
        };
        let res = par_solve_pcg_amg(&stiff, &rhs, &mut u, &amg_cfg, &cfg)
            .expect("PCG+AMG solve failed");

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

        // GLVis visualization (parallel mode).
        if visualization {
            let lm = ps.local_space().mesh();
            let n_nodes = lm.n_nodes() as usize;
            let ref_elem = TriNDk::new(1);
            let n_ldofs = ref_elem.n_dofs();
            let ref_verts: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

            let mut sum_x = vec![0.0_f64; n_nodes];
            let mut sum_y = vec![0.0_f64; n_nodes];
            let mut count = vec![0u32; n_nodes];
            let mut ref_phi = vec![0.0_f64; n_ldofs * 2];

            let n_owned_elems = pm.partition().n_owned_elems;
            for e in 0..n_owned_elems as u32 {
                let nodes = lm.element_nodes(e);
                let dofs: Vec<usize> = ps.local_space().element_dofs(e)
                    .iter().map(|&d| d as usize).collect();
                let signs = ps.local_space().element_signs(e);

                let x0 = lm.node_coords(nodes[0]);
                let x1 = lm.node_coords(nodes[1]);
                let x2 = lm.node_coords(nodes[2]);

                let j00 = x1[0] - x0[0]; let j01 = x2[0] - x0[0];
                let j10 = x1[1] - x0[1]; let j11 = x2[1] - x0[1];
                let inv_det = 1.0 / (j00 * j11 - j01 * j10);
                let (jit00, jit01) = ( j11 * inv_det, -j10 * inv_det);
                let (jit10, jit11) = (-j01 * inv_det,  j00 * inv_det);

                for vi in 0..3 {
                    ref_elem.eval_basis_vec(&ref_verts[vi], &mut ref_phi);

                    let mut eh_x = 0.0_f64;
                    let mut eh_y = 0.0_f64;
                    for i in 0..n_ldofs {
                        let px = jit00 * ref_phi[i * 2] + jit01 * ref_phi[i * 2 + 1];
                        let py = jit10 * ref_phi[i * 2] + jit11 * ref_phi[i * 2 + 1];
                        let val = u.as_slice()[dofs[i]];
                        eh_x += signs[i] * val * px;
                        eh_y += signs[i] * val * py;
                    }

                    let nid = nodes[vi] as usize;
                    sum_x[nid] += eh_x;
                    sum_y[nid] += eh_y;
                    count[nid] += 1;
                }
            }

            let mut e_node_x = vec![0.0_f64; n_nodes];
            let mut e_node_y = vec![0.0_f64; n_nodes];
            for i in 0..n_nodes {
                if count[i] > 0 {
                    let inv = 1.0 / count[i] as f64;
                    e_node_x[i] = sum_x[i] * inv;
                    e_node_y[i] = sum_y[i] * inv;
                }
            }

            let n_ranks = pm.comm().size();
            let my_rank = pm.comm().rank() as usize;
            match GlVisSocket::connect("localhost", 19916) {
                Ok(mut vis) => {
                    vis.send_parallel_solution_2d_vector(
                        n_ranks, my_rank, lm, &e_node_x, &e_node_y, "E",
                    ).ok();
                }
                Err(e) => {
                    if comm.rank() == 0 {
                        eprintln!("  GLVis not available: {e}");
                    }
                }
            }
        }

        // Compute L² error on owned elements (ghost elements excluded).
        // Note: u.as_slice() returns DOFs in partition ordering, but
        // element_dofs() returns DOFs in space (DM) ordering.  We must
        // permute via the DofPartition and apply sign corrections.
        let lm = ps.local_space().mesh();
        let dp = ps.dof_partition();
        let n_owned_elems = pm.partition().n_owned_elems;
        // Refresh ghost DOFs and convert partition order → dm order (with
        // sign correction back, see pex4 notes), then use the library L2
        // error function (dm order, verified against serial ex3).
        let mut u_full = u.clone_vec();
        u_full.update_ghosts();
        let n_dm = ps.local_space().n_dofs();
        let mut u_dm = vec![0.0_f64; n_dm];
        {
            let needs_sign = dp.needs_sign_correction();
            for pid in 0..dp.n_total_dofs() {
                let dm = dp.unpermute_dof(pid as u32) as usize;
                let s = if needs_sign {
                    dp.sign_correction(dm as u32)
                } else {
                    1.0
                };
                u_dm[dm] = u_full.as_slice()[pid] * s;
            }
        }
        let local_err = if n_owned_elems > 0 {
            fem_examples::maxwell::l2_error_hcurl_exact_owned(
                ps.local_space(),
                &u_dm,
                |x| exact_e(x, kappa),
                &|e: u32| pm.partition().elem_owner[e as usize] == comm.rank(),
            )
        } else {
            0.0
        };
        let global_err = comm.allreduce_sum_f64(local_err * local_err).sqrt();
        if comm.rank() == 0 {
            println!("\n|| E_h - E ||_{{L^2}} = {:.14e}\n", global_err);
        }

        *r2.lock().unwrap() = Some((n_global, res.iterations, res.final_residual));
    });

    let taken = result.lock().unwrap().take();
    if let Some((dofs, iters, res)) = taken {
        println!("pex3: dofs={dofs} iters={iters} residual={res:.3e}");
    }
}
