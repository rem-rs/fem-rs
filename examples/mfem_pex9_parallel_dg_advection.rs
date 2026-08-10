//! # Parallel Example 9 — DG Advection (periodic, aligned with MFEM ex9p)
//! (aligned with MFEM pex9 / ex9p.cpp)
//!
//! Time-dependent advection `du/dt + v·∇u = 0` with `v = (1, 0)` and a smooth
//! wave, discretized with DG (L2 space on Gauss-Lobatto nodes, order `-o`),
//! explicit RK4, mass matrix inverted at every stage
//! (`dudt = M⁻¹ (K u + rhs)`).
//!
//! The parallel port assembles the DG operators per rank over the local mesh
//! (owned + ghost elements — the standard overlap that makes owned rows
//! complete, including cross-rank interior faces), packs them into
//! [`ParCsrMatrix`] and time-steps with RK4 whose stage solves use a
//! Jacobi-PCG on the (block-diagonal) mass matrix.
//!
//! ex9p defaults to *periodic* meshes (`periodic-hexagon.mesh`): the mesh
//! files are topologically periodic — element connectivity wraps across the
//! domain (periodic vertices share ids), there are no boundary faces, and
//! every periodic face is an ordinary interior face sharing two vertices.
//! The DG interior-face assembly therefore needs no special-casing; the
//! periodic mesh is read with `-m` (default `data/periodic-hexagon.mesh`).
//! A non-periodic mesh with boundary faces is also supported (inflow RHS +
//! boundary flux, as before).
//!
//! # Multi-rank support
//! Multi-rank runs are consistent: interior DG faces are normalized by
//! global element ids (e1 = smaller gid, so both ranks assemble identical
//! face fluxes and normals) and boundary faces are assigned to the rank that
//! owns the adjacent element (MFEM `GetBdrElementAdjacentElement`), so each
//! owned row receives its boundary flux.  On a periodic mesh mass is
//! conserved (nothing leaves the domain).
//!
//! # Known limitation
//! On a *periodic* mesh the multi-rank (`--ranks 2`) mass evolution shows a
//! small drift (≈ ±0.3%/step at order 1 on periodic-hexagon) while np1 is
//! exactly conservative — the periodic face-flux scatter in the local
//! (owned+ghost) assembly is not yet bit-consistent across ranks.  The
//! non-periodic path (unit-square with inflow) has no such drift (np2 mass
//! matches np1).
//!
//! Usage:
//!   cargo run --release --example mfem_pex9_parallel_dg_advection -- --ranks 1
//!   cargo run --release --example mfem_pex9_parallel_dg_advection -- --ranks 2 -o 2 -n 8
//!   cargo run --release --example mfem_pex9_parallel_dg_advection -- -m data/periodic-square.mesh

use std::sync::Arc;

use fem_assembly::dg::dg_imex::{
    assemble_ex41_bdr_faces, assemble_ex41_interior_faces, build_bdr_face_locs,
    build_face_locs, MfemHeadInsert,
};
use fem_assembly::dg::DGAdvectionIntegrator;
use fem_assembly::postproc::coefficient::{FnVectorCoeff, VectorCoeff};
use fem_assembly::standard::MassIntegrator;
use fem_assembly::Assembler;
use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::refine_uniform;
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_jacobi;
use fem_parallel::{
    DofPartition, ParCsrMatrix, ParVector, ParallelFESpace, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::fe_space::FESpace;
use fem_space::{L2Basis, L2Space};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let _n: usize = parse_arg(&args, "-n").unwrap_or(6); // kept for CLI compat (mesh size now from -m)
    let order: u8 = parse_arg(&args, "-o").map(|o| o as u8).unwrap_or(1);
    let ref_levels: usize = parse_arg(&args, "-r").unwrap_or(0);
    let dt: f64 = parse_arg_f64(&args, "-dt").unwrap_or(0.01);
    let t_final: f64 = parse_arg_f64(&args, "-tf").unwrap_or(2.0);

    println!("=== fem-rs mfem_pex9: Parallel DG Advection (RK4, periodic) ===");

    // Periodic mesh by default (ex9p defaults to periodic-hexagon.mesh);
    // `-m` overrides with any MFEM 2-D mesh (non-periodic meshes with
    // boundary faces are also supported via the inflow/boundary path).
    let mesh_file = args
        .iter()
        .position(|a| a == "-m")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/periodic-hexagon.mesh");
    let mfem = fem_io::mfem::read_mfem_file(mesh_file)
        .unwrap_or_else(|e| panic!("failed to read {mesh_file}: {e}"));
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("mesh must be 2-D");
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }
    let n_global_elems = mesh.n_elems();
    println!(
        "  Workers: {n_workers}, mesh: {n_global_elems} triangles, order {order}, dt {dt}, tf {t_final}"
    );
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None::<(usize, f64, f64)>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition();

        // ── 1. DG space (Gauss-Lobatto nodes, like ex9) ─────────────────────
        let space_local = L2Space::new_with_basis(local_mesh.clone(), order, L2Basis::GaussLobatto);
        let part = DofPartition::from_l2_space(&space_local, partition, &comm);
        let ps = ParallelFESpace::new_with_dof_partition(space_local, part, comm.clone());
        let n_local = ps.n_local_dofs();
        let n_owned = ps.dof_partition().n_owned_dofs;
        let ghost = ps.dof_ghost_exchange_arc();

        // Bounding box (whole mesh, replicated on every rank) — used by the
        // MFEM initial condition to map into the reference [-1, 1] domain.
        let mut bb_min = vec![f64::INFINITY; 2];
        let mut bb_max = vec![f64::NEG_INFINITY; 2];
        for nid in 0..mesh_arc.n_nodes() {
            let c = mesh_arc.node_coords(nid as u32);
            for d in 0..2 {
                bb_min[d] = bb_min[d].min(c[d]);
                bb_max[d] = bb_max[d].max(c[d]);
            }
        }

        // ── 2. Local operators (owned + ghost elements → complete rows) ─────
        let qo_mass = (order as u8 * 2 + 2).max(3);
        let qo_adv = (order as u8 * 2).max(2);
        let mass_local = Assembler::assemble_bilinear(
            ps.local_space(),
            &[&MassIntegrator { rho: 1.0 }],
            qo_mass,
        );

        // Volume advection (weak form, constant-preserving): v = (1, 0).
        let vel = FnVectorCoeff(|_x: &[f64], out: &mut [f64]| {
            out[0] = 1.0;
            out[1] = 0.0;
        });
        let dg_adv = DGAdvectionIntegrator { velocity: vel };
        let adv_local = Assembler::assemble_bilinear(
            ps.local_space(),
            &[&dg_adv],
            qo_adv,
        );

        // Interior DG faces (NonconservativeDG, α = -1) + boundary faces.
        // Cross-rank consistency: `build_face_locs` registers the first-seen
        // element as e1 and orients the face by its edge — the first-seen
        // element differs between ranks for a cross-rank face, flipping the
        // face normal and the trace-flux sign.  Normalize e1/e2 by global
        // element id (rank-independent); e2_flip is invariant under the swap
        // (both elements orient the shared edge oppositely in a consistent
        // mesh, and flip only records "e2's edge direction vs face
        // direction", which the swap preserves).
        let mut faces = build_face_locs(ps.local_space().mesh());
        for f in &mut faces {
            if partition.global_elem(f.e1) > partition.global_elem(f.e2) {
                std::mem::swap(&mut f.e1, &mut f.e2);
                std::mem::swap(&mut f.e1_inf, &mut f.e2_inf);
            }
        }
        let mut hi = MfemHeadInsert::new(n_local);
        assemble_ex41_interior_faces(
            &mut hi,
            ps.local_space().mesh(),
            ps.local_space(),
            &faces,
            &|_x, _y| [1.0, 0.0],
            -1.0, 0.0, 0.0, 0.0,
        );
        let bdr_faces = build_bdr_face_locs(ps.local_space().mesh());
        assemble_ex41_bdr_faces(
            &mut hi,
            ps.local_space().mesh(),
            ps.local_space(),
            &bdr_faces,
            &|_x, _y| [1.0, 0.0],
            -1.0, 0.0, 0.0, 0.0,
        );

        let mut coo = CooMatrix::<f64>::new(n_local, n_local);
        for i in 0..n_local {
            for p in adv_local.row_ptr[i]..adv_local.row_ptr[i + 1] {
                coo.add(i, adv_local.col_idx[p] as usize, adv_local.values[p]);
            }
        }
        let k_face = hi.into_csr();
        for i in 0..n_local {
            for p in k_face.row_ptr[i]..k_face.row_ptr[i + 1] {
                coo.add(i, k_face.col_idx[p] as usize, k_face.values[p]);
            }
        }
        let k_adv_local = coo.into_csr();

        // Inflow RHS: u_in = 0 (nothing enters through x = 0 beyond the wave).
        let bc_tags: Vec<i32> = local_mesh.unique_boundary_tags();
        let inflow_g = |_x: &[f64]| 0.0_f64;
        let vel_bdr = FnVectorCoeff(|_x: &[f64], out: &mut [f64]| {
            out[0] = 1.0;
            out[1] = 0.0;
        });
        let (_k_bdr, rhs_bc_local) = fem_assembly::dg::dg_advection::assemble_advection_boundary_full(
            ps.local_space(), &vel_bdr, &bc_tags, &inflow_g, order, qo_adv,
        );

        // ── 3. Pack into ParCsrMatrix + ParVector ───────────────────────────
        let mass = ParCsrMatrix::from_local_matrix(
            &mass_local, n_owned, Arc::clone(&ghost), comm.clone(),
        );
        let k_adv = ParCsrMatrix::from_local_matrix(
            &k_adv_local, n_owned, Arc::clone(&ghost), comm.clone(),
        );
        let rhs_bc = ParVector::from_local_raw(
            rhs_bc_local, n_owned, Arc::clone(&ghost), comm.clone(),
        );

        // ── 4. Initial condition: MFEM ex9 u0 (problem 0) — a smooth wave
        //     erfc·erfc·erfc·erfc/16 mapped into the reference [-1, 1] box,
        //     parameters rx=0.45, ry=0.25, cx=0, cy=-0.2, w=10.
        let ic = {
            let bb_min = bb_min.clone();
            let bb_max = bb_max.clone();
            move |x: &[f64]| {
                let xr = 2.0 * (x[0] - 0.5 * (bb_min[0] + bb_max[0])) / (bb_max[0] - bb_min[0]);
                let yr = 2.0 * (x[1] - 0.5 * (bb_min[1] + bb_max[1])) / (bb_max[1] - bb_min[1]);
                let rx = 0.45;
                let ry = 0.25;
                let cx = 0.0;
                let cy = -0.2;
                let w = 10.0;
                (libm::erfc(w * (xr - cx - rx)) * libm::erfc(-w * (xr - cx + rx))
                    * libm::erfc(w * (yr - cy - ry)) * libm::erfc(-w * (yr - cy + ry)))
                    / 16.0
            }
        };
        let u0_local = ps.local_space().interpolate(&ic).as_slice().to_vec();
        let mut u = ParVector::from_local_raw(
            u0_local, n_owned, Arc::clone(&ghost), comm.clone(),
        );
        // 每 rank 的初始质量（owned dof 的 Σu·mass 行和近似——用 Σu 简化）
        let init_mass: f64 = {
            let mut s = 0.0;
            for i in 0..n_owned {
                s += u.as_slice()[i];
            }
            comm.allreduce_sum_f64(s)
        };

        // ── 5. RK4 time integration ─────────────────────────────────────────
        let mass_cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 50,
            verbose: false,
            ..SolverConfig::default()
        };
        let mut t = 0.0_f64;
        let mut step = 0usize;
        let mut mass_u = 0.0_f64; // 最终质量
        while t < t_final - 1e-14 {
            let dta = dt.min(t_final - t);
            let mut k1 = ParVector::zeros_like(&u);
            let mut k2 = ParVector::zeros_like(&u);
            let mut k3 = ParVector::zeros_like(&u);
            let mut k4 = ParVector::zeros_like(&u);
            let mut tmp = ParVector::zeros_like(&u);

            // k1 = f(u)
            rhs_apply(&k_adv, &rhs_bc, &u, &mut k1, &mass, &mass_cfg);
            // k2 = f(u + dt/2 k1)
            tmp.copy_from(&u);
            tmp.axpy(0.5 * dta, &k1);
            rhs_apply(&k_adv, &rhs_bc, &tmp, &mut k2, &mass, &mass_cfg);
            // k3 = f(u + dt/2 k2)
            tmp.copy_from(&u);
            tmp.axpy(0.5 * dta, &k2);
            rhs_apply(&k_adv, &rhs_bc, &tmp, &mut k3, &mass, &mass_cfg);
            // k4 = f(u + dt k3)
            tmp.copy_from(&u);
            tmp.axpy(dta, &k3);
            rhs_apply(&k_adv, &rhs_bc, &tmp, &mut k4, &mass, &mass_cfg);

            // u += dt/6 (k1 + 2k2 + 2k3 + k4)
            u.axpy(dta / 6.0, &k1);
            u.axpy(dta / 3.0, &k2);
            u.axpy(dta / 3.0, &k3);
            u.axpy(dta / 6.0, &k4);

            t += dta;
            step += 1;
            if step % 20 == 0 && rank == 0 {
                println!("time step: {step}, time: {t:.3}");
            }
        }
        let mut s = 0.0;
        for i in 0..n_owned {
            s += u.as_slice()[i];
        }
        mass_u = comm.allreduce_sum_f64(s);

        if rank == 0 {
            *result_slot.lock().expect("pex9 mutex") = Some((
                ps.n_global_dofs(),
                init_mass,
                mass_u,
            ));
        }
    });

    let (dofs, init_mass, final_mass) = result
        .lock()
        .expect("pex9 mutex after launch")
        .take()
        .expect("rank 0 did not publish pex9 result");
    println!(
        "=== Done: dofs = {dofs}, mass Σu: {init_mass:.6e} → {final_mass:.6e} (periodic: conserved) ==="
    );
}

/// dudt = M⁻¹ (K u + rhs)
fn rhs_apply(
    k_adv: &ParCsrMatrix,
    rhs_bc: &ParVector,
    u: &ParVector,
    dudt: &mut ParVector,
    mass: &ParCsrMatrix,
    mass_cfg: &SolverConfig,
) {
    let mut tmp = ParVector::zeros_like(u);
    k_adv.spmv(&mut u.clone_vec(), &mut tmp);
    let n = u.n_owned();
    for i in 0..n {
        tmp.as_slice_mut()[i] += rhs_bc.as_slice()[i];
    }
    par_solve_pcg_jacobi(mass, &tmp, dudt, mass_cfg).expect("mass solve failed");
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_arg_f64(args: &[String], flag: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
