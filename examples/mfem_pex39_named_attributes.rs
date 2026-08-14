//! Parallel Example 39 — Named attribute sets (1:1 translation of MFEM ex39p)
//!
//! Solves `−∇·(κ ∇u) = 1_s` on `compass.msh` where:
//! - the source region `s` is given by name (`-src`, default `Rose Even`),
//! - the diffusion coefficient is a sum of three pieces:
//!   `1e-6` everywhere + `1.0` on `Base` + `2.0` on `Rose Even`
//!   (markers come from named attribute sets read from the GMSH file),
//! - homogeneous Dirichlet BCs on the named boundary set `-ess` (default
//!   `Boundary`).
//!
//! Parallel layout (following MFEM ex39p):
//! - serial mesh read once (GMSH `compass.msh`, with the `refine=1` triangle
//!   rotation via `mark_tri_mesh_for_refinement`), serial uniform
//!   refinements capped at **10000** elements (ex39p, not 50000 like ex39);
//! - named attribute sets (element + boundary) are built **once** on the main
//!   thread from the global mesh; only the resulting marker arrays are shared
//!   with the workers (attribute numbers are global, so markers must be sized
//!   against the global max attribute);
//! - `partition_mesh` + 2 parallel uniform refinements, H1 space, marked
//!   assembly via `ParAssembler::assemble_{linear,bilinear}_marked`
//!   (the partition migrates `elem_tags`/`face_tags`, so the local marked
//!   assembly over owned + ghost elements is the restriction of the global
//!   one);
//! - essential BCs: boundary-vertex DOFs are detected locally and their
//!   global ids exchanged so every rank clamps the same global set (a
//!   boundary vertex may be owned by a rank that never sees the boundary
//!   face, which migrates to the adjacent-element owner);
//! - symmetric essential-BC elimination (`eliminate_diag_symmetric`,
//!   MFEM `FormLinearSystem`), PCG + AMG V-cycle (C++: HypreCG + BoomerAMG,
//!   rtol 1e-12, max 2000).
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex39_named_attributes -- --ranks 1
//! cargo run --release --example mfem_pex39_named_attributes -- --ranks 4
//! cargo run --release --example mfem_pex39_named_attributes -- --ranks 2 -ess "Southern Boundary"
//! ```

use std::sync::{Arc, Mutex};

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_io::read_msh_file;
use fem_mesh::Mesh;
use fem_parallel::{
    ParAmgConfig, ParAssembler, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
    launcher::native::ThreadLauncher, par_partition::partition_mesh,
    par_refine::par_uniform_refine, par_solve_pcg_amg,
};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2) as usize;
    let mesh_file: String = parse_arg_str(&args, "-m")
        .unwrap_or_else(|| "data/compass.msh".to_string());
    let order: i32 = parse_arg(&args, "-o").unwrap_or(1);
    let source_name: String = parse_arg_str(&args, "-src")
        .unwrap_or_else(|| "Rose Even".to_string());
    let ess_name: String = parse_arg_str(&args, "-ess")
        .unwrap_or_else(|| "Boundary".to_string());
    let dump_sol = parse_arg_str(&args, "--dump-sol");

    // ── 1. Read the mesh + build the named attribute sets (main thread) ─────
    let msh = read_msh_file(&mesh_file).unwrap_or_else(|e| panic!("read {mesh_file}: {e}"));
    let mut registry = msh.named_attribute_registry();
    let mut bdr_registry = registry.clone();
    let mut mesh: Mesh<2> = msh.into_2d().expect("compass.msh must be 2-D");

    let dim = 2usize;
    let ne = mesh.n_elems();
    // ex39p: largest ref_levels with at most 10,000 elements (note: ex39 uses
    // 50,000 — the parallel example's cap differs, so the meshes differ).
    let ref_levels = ((10000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize;
    // MFEM Mesh(filename, 1, 1): `refine=1` rotates each triangle so its
    // longest edge is (0,1) BEFORE uniform refinement (affects edge-midpoint
    // numbering of every refinement level).
    fem_mesh::amr::mark_tri_mesh_for_refinement(&mut mesh);
    for _ in 0..ref_levels {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }

    // ── 2. Build new element sets (ex39p step 6b) ───────────────────────────
    let n_even = registry.element_set("N Even").to_vec();
    let n_odd = registry.element_set("N Odd").to_vec();
    let s_even = registry.element_set("S Even").to_vec();
    let s_odd = registry.element_set("S Odd").to_vec();
    let e_even = registry.element_set("E Even").to_vec();
    let e_odd = registry.element_set("E Odd").to_vec();
    let w_even = registry.element_set("W Even").to_vec();
    let w_odd = registry.element_set("W Odd").to_vec();

    registry.set_attribute_set("North", &n_even);
    registry.add_to_attribute_set("North", &n_odd);
    registry.set_attribute_set("South", &s_even);
    registry.add_to_attribute_set("South", &s_odd);
    registry.set_attribute_set("East", &e_even);
    registry.add_to_attribute_set("East", &e_odd);
    registry.set_attribute_set("West", &w_even);
    registry.add_to_attribute_set("West", &w_odd);
    registry.set_attribute_set("Rose Even", &n_even);
    registry.add_to_attribute_set("Rose Even", &s_even);
    registry.add_to_attribute_set("Rose Even", &e_even);
    registry.add_to_attribute_set("Rose Even", &w_even);
    registry.set_attribute_set("Rose Odd", &n_odd);
    registry.add_to_attribute_set("Rose Odd", &s_odd);
    registry.add_to_attribute_set("Rose Odd", &e_odd);
    registry.add_to_attribute_set("Rose Odd", &w_odd);
    let rose_even = registry.element_set("Rose Even").to_vec();
    let rose_odd = registry.element_set("Rose Odd").to_vec();
    registry.set_attribute_set("Rose", &rose_even);
    registry.add_to_attribute_set("Rose", &rose_odd);

    // ── 3. Build new boundary sets (ex39p step 6c) ──────────────────────────
    let nne = bdr_registry.boundary_set("NNE").to_vec();
    let nnw = bdr_registry.boundary_set("NNW").to_vec();
    let ene = bdr_registry.boundary_set("ENE").to_vec();
    let ese = bdr_registry.boundary_set("ESE").to_vec();
    let sse = bdr_registry.boundary_set("SSE").to_vec();
    let ssw = bdr_registry.boundary_set("SSW").to_vec();
    let wnw = bdr_registry.boundary_set("WNW").to_vec();
    let wsw = bdr_registry.boundary_set("WSW").to_vec();

    bdr_registry.set_boundary_attribute_set("Northern Boundary", &nne);
    bdr_registry.add_to_boundary_attribute_set("Northern Boundary", &nnw);
    bdr_registry.set_boundary_attribute_set("Southern Boundary", &sse);
    bdr_registry.add_to_boundary_attribute_set("Southern Boundary", &ssw);
    bdr_registry.set_boundary_attribute_set("Eastern Boundary", &ene);
    bdr_registry.add_to_boundary_attribute_set("Eastern Boundary", &ese);
    bdr_registry.set_boundary_attribute_set("Western Boundary", &wnw);
    bdr_registry.add_to_boundary_attribute_set("Western Boundary", &wsw);

    let n_bound = bdr_registry.boundary_set("Northern Boundary").to_vec();
    let s_bound = bdr_registry.boundary_set("Southern Boundary").to_vec();
    let e_bound = bdr_registry.boundary_set("Eastern Boundary").to_vec();
    let w_bound = bdr_registry.boundary_set("Western Boundary").to_vec();
    bdr_registry.set_boundary_attribute_set("Boundary", &n_bound);
    bdr_registry.add_to_boundary_attribute_set("Boundary", &s_bound);
    bdr_registry.add_to_boundary_attribute_set("Boundary", &e_bound);
    bdr_registry.add_to_boundary_attribute_set("Boundary", &w_bound);

    // ── 4. Attribute set names (C++ prints on root rank) ────────────────────
    let mut elem_names: Vec<String> = registry
        .names()
        .into_iter()
        .filter(|n| !registry.element_set(n).is_empty())
        .map(|s| s.to_string())
        .collect();
    elem_names.sort();
    print!("Element Attribute Set Names: ");
    for n in &elem_names {
        print!(" \"{n}\"");
    }
    println!();

    let mut bdr_names: Vec<String> = bdr_registry
        .names()
        .into_iter()
        .filter(|n| !bdr_registry.boundary_set(n).is_empty())
        .map(|s| s.to_string())
        .collect();
    bdr_names.sort();
    print!("Boundary Attribute Set Names: ");
    for n in &bdr_names {
        print!(" \"{n}\"");
    }
    println!();

    // ── 5. Global markers (sized against the global max attribute) ──────────
    let max_elem_attr = mesh.elem_tags.iter().copied().max().unwrap_or(0);
    let max_bdr_attr = mesh.face_tags.iter().copied().max().unwrap_or(0);
    let source_marker = registry.element_set_marker(&source_name, max_elem_attr);
    let base_marker = registry.element_set_marker("Base", max_elem_attr);
    let rose_marker = registry.element_set_marker("Rose Even", max_elem_attr);
    let ess_marker = bdr_registry.boundary_set_marker(&ess_name, max_bdr_attr);
    let ess_tags: Vec<i32> = (0..ess_marker.len() as i32)
        .filter(|&i| ess_marker[i as usize] == 1)
        .map(|i| i + 1)
        .collect();

    let mesh = Arc::new(mesh);
    let source_marker = Arc::new(source_marker);
    let base_marker = Arc::new(base_marker);
    let rose_marker = Arc::new(rose_marker);
    let ess_tags = Arc::new(ess_tags);

    let result = Arc::new(Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);
    let sm = Arc::clone(&source_marker);
    let bm = Arc::clone(&base_marker);
    let rm = Arc::clone(&rose_marker);
    let et = Arc::clone(&ess_tags);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // ── 6. Partition + parallel uniform refinements (C++: 2) ─────────────
        let mut par_mesh = partition_mesh(&mesh_arc, &comm);
        for _ in 0..2 {
            par_mesh = par_uniform_refine(&par_mesh);
        }

        // ── 7. H1 space ───────────────────────────────────────────────────────
        let local_mesh = par_mesh.local_mesh().clone();
        let space = H1Space::new(local_mesh, order as u8);
        let ps = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let dof_part = ps.dof_partition();
        let n_owned = dof_part.n_owned_dofs;
        let ghost = ps.dof_ghost_exchange_arc();

        // ── 8. RHS: ∫ 1_s v with the source marker ───────────────────────────
        let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
        let qo = (2 * order + 1) as u8;
        let mut rhs = ParAssembler::assemble_linear_marked(
            &ps,
            &[(&source, Some(source_marker.as_slice()))],
            qo,
        );

        // ── 9. A: κ = 1e-6 everywhere + 1.0 on Base + 2.0 on Rose Even ───────
        let default_coef = DiffusionIntegrator { kappa: 1.0e-6 };
        let base_coef = DiffusionIntegrator { kappa: 1.0 };
        let rose_coef = DiffusionIntegrator { kappa: 2.0 };
        let mut a = ParAssembler::assemble_bilinear_marked(
            &ps,
            &[
                (&default_coef, None),
                (&base_coef, Some(base_marker.as_slice())),
                (&rose_coef, Some(rose_marker.as_slice())),
            ],
            qo,
        );

        // ── 10. Essential BCs (FormLinearSystem): symmetric row+col
        //     elimination.  Boundary vertices can be owned by a different
        //     rank than the element adjacent to their boundary face (the
        //     face migrates to the adjacent-element owner).  A vertex is then
        //     detected as essential only by the rank holding the face — where
        //     it may be a ghost DOF — while its owner never sees the face.
        //     Exchange the global ids of locally-detected boundary DOFs and
        //     clamp every owned DOF whose global id appears in the union.
        let mesh_ref = ps.local_space().mesh();
        let bnd_local = boundary_dofs(mesh_ref, ps.local_space().dof_manager(), &ess_tags);
        let local_bnd_global: Vec<u32> = bnd_local
            .iter()
            .map(|&d| dof_part.global_dof(dof_part.permute_dof(d)))
            .collect();
        let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
        for r in 0..comm.size() as i32 {
            if r == rank {
                continue;
            }
            let mut bytes = Vec::with_capacity(local_bnd_global.len() * 4);
            for &g in &local_bnd_global {
                bytes.extend_from_slice(&g.to_le_bytes());
            }
            sends.push((r, bytes));
        }
        let incoming = comm.alltoallv_bytes(&sends);
        let mut all_bnd: std::collections::HashSet<u32> =
            local_bnd_global.iter().copied().collect();
        for (_, bytes) in incoming {
            for chunk in bytes.chunks_exact(4) {
                all_bnd.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        let clamped: Vec<usize> = (0..n_owned)
            .filter(|&pid| all_bnd.contains(&dof_part.global_dof(pid as u32)))
            .collect();
        a.eliminate_diag_symmetric(&clamped, 1.0);
        for &pid in &clamped {
            rhs.as_slice_mut()[pid] = 0.0;
        }

        // ── 11. Solve: PCG + AMG (C++: HypreCG + BoomerAMG, rtol 1e-12) ──────
        let mut u = ParVector::zeros(&ps);
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            block_size: 1,
            // κ jumps by 1e6 between the rose/Base regions and the background;
            // the default local aggregation converges fine here (the
            // high-contrast regions stay inside one rank's partition).
            use_global_aggregation: false,
            ..ParAmgConfig::default()
        };
        let res = par_solve_pcg_amg(&a, &rhs, &mut u, &amg_cfg, &cfg).expect("PCG+AMG failed");

        // ── 12. Metrics + optional dump (global dof order) ───────────────────
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
                    let gids: Vec<u32> = comm.recv(src, 91);
                    let vals: Vec<f64> = comm.recv(src, 92);
                    all.extend(gids.into_iter().zip(vals));
                }
                all.sort_unstable_by_key(|&(g, _)| g);
                let mut s = String::new();
                for (_, v) in all {
                    s.push_str(&format!("{:.10e}\n", v));
                }
                std::fs::write(path, s).expect("failed to write solution dump");
            } else {
                let gids: Vec<u32> = owned.iter().map(|&(g, _)| g).collect();
                let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                comm.send(0, 91, &gids);
                comm.send(0, 92, &vals);
            }
        }

        if rank == 0 {
            *result_slot.lock().expect("pex39 result mutex poisoned") = Some(RunResult {
                global_dofs: ps.n_global_dofs(),
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                solution_norm,
                solution_sum,
                solution_checksum,
            });
        }
    });

    let r = result.lock().expect("pex39 result mutex after launch").take();
    match r {
        Some(r) => {
            println!("Number of finite element unknowns: {}", r.global_dofs);
            println!(
                "  PCG: {} iters, residual = {:.3e}, converged = {}",
                r.iterations, r.final_residual, r.converged
            );
            let avg = r
                .final_residual
                .powf(1.0 / (r.iterations.max(1) as f64));
            println!("  Average reduction factor = {:.6}", avg);
            println!(
                "  ||u||_2 = {:.6e}, sum = {:.8e}, checksum = {:.8e}",
                r.solution_norm, r.solution_sum, r.solution_checksum
            );
            println!("=== Done ===");
        }
        None => panic!("pex39: no result from workers"),
    }
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

fn parse_arg(args: &[String], name: &str) -> Option<i32> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_arg_str(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
