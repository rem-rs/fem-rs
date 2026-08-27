//! # Parallel Example 26 — Geometric Multigrid (1:1 with MFEM ex26p)
//!
//! Parallel Poisson −Δu = 1 with homogeneous Dirichlet BC, solved with PCG
//! preconditioned by a geometric multigrid (p-refinement hierarchy P1→P2→P4,
//! matching MFEM ex26p defaults: `-or 2`, no geometric refinements).
//!
//! C++ ex26p flow:
//!   1. serial mesh → uniform refine so NE ≤ 1000
//!   2. ParMesh = partition(serial mesh), then 2 parallel uniform refinements
//!   3. ParFiniteElementSpaceHierarchy: coarse P1 + order refinements P2, P4
//!   4. DiffusionMultigrid: coarse = AMG + CG; fine = Chebyshev smoothers
//!   5. CGSolver(rtol 1e-12, max 2000) with the MG as preconditioner
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex26_parallel_geom_mg -- --ranks 2
//! cargo run --release --example mfem_pex26_parallel_geom_mg -- --ranks 4 -or 1
//! ```

use std::sync::{Arc, Mutex};

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_parallel::ParAssembler;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_refine::par_uniform_refine;
use fem_parallel::par_assembler::permute_vec;
use fem_parallel::{
    DofPartition, ParCsrMatrix, ParVector, ParallelFESpace,
    WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};
use fem_space::build_h1_prolongation_matrix;

/// One level's prolongation/restriction pair (local CSR, partition order).
struct MgTransfer {
    /// Prolongation P: fine_owned × coarse_total (fine ← coarse).
    prolong_local: fem_linalg::CsrMatrix<f64>,
    /// Restriction R: coarse_total × fine_total (coarse ← fine; ghost
    /// columns are zero — the owning rank computes them).
    restrict_local: fem_linalg::CsrMatrix<f64>,
    coarse_owned: usize,
    /// Coarse-space DofPartition (for global-id cross-rank sum).
    coarse_dp_for_global: DofPartition,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ranks: usize = arg(&args, "--ranks").unwrap_or(2);
    let order_refs: usize = arg(&args, "-or").unwrap_or(2);
    let par_refs: usize = arg(&args, "-rp").unwrap_or(2);

    // 1. Serial mesh (star.mesh — ex26p default) + uniform refine ≤ 1000 elems.
    let mfem = read_mfem_file("data/star.mesh").expect("failed to read data/star.mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("star.mesh must be 2-D");
    let dim = 2;
    let ne = mesh.n_elems();
    let auto_ser = if ne > 0 {
        ((1000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
    } else { 0 };
    let ser_refs = if args.iter().any(|a| a == "-rs") { arg(&args, "-rs").unwrap() } else { auto_ser };
    for _ in 0..ser_refs { mesh = fem_mesh::refine_uniform(&mesh); }
    let mesh = Arc::new(mesh);

    let result = Arc::new(Mutex::new(None::<String>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    ThreadLauncher::new(WorkerConfig::new(ranks)).launch(move |comm| {
        let rank = comm.rank();
        // 2. Partition + parallel uniform refinements.
        let mut pm = partition_mesh(&mesh_arc, &comm);
        for _ in 0..par_refs { pm = par_uniform_refine(&pm); }
        let local_mesh = pm.local_mesh().clone();

        // 3. p-refinement hierarchy: orders 1, 2, 4, ...
        let mut orders: Vec<u8> = vec![1];
        for k in 1..=order_refs { orders.push(1u8 << k); }
        let n_levels = orders.len();

        // Parallel spaces per level (same local mesh, different order).
        let mut spaces: Vec<ParallelFESpace<H1Space<Mesh<2>>>> = Vec::new();
        for &o in &orders {
            let dm = fem_space::dof_manager::DofManager::new(&local_mesh, o);
            spaces.push(ParallelFESpace::new_with_dof_manager(
                H1Space::new(local_mesh.clone(), o), &pm, &dm, comm.clone(),
            ));
        }
        let n_global = spaces.last().unwrap().n_global_dofs();
        if rank == 0 { println!("Number of finite element unknowns: {n_global}"); }

        // 4. Boundary (all) → homogeneous Dirichlet.
        let bnd_tags: Vec<i32> = local_mesh.unique_boundary_tags();
        let qo = |o: u8| (2 * o + 1).max(3) as u8;

        // 5. Assemble diffusion matrix per level + symmetric BC elimination.
        //    Boundary DOFs owned by other ranks appear as ghost slots on this
        //    rank; their offd *column* contributions must be zeroed too, or
        //    the operator becomes asymmetric across ranks (A_ij ≠ A_ji for
        //    boundary pairs) and PCG/CG stagnate on np > 1.
        let mut mats: Vec<ParCsrMatrix> = Vec::with_capacity(n_levels);
        for (i, sp) in spaces.iter().enumerate() {
            let mut m = ParAssembler::assemble_bilinear(sp, &[&DiffusionIntegrator { kappa: 1.0 }], qo(orders[i]));
            let bc = boundary_dofs(sp.local_space().mesh(), sp.local_space().dof_manager(), &bnd_tags);
            let owned_ess: Vec<usize> = bc.iter()
                .map(|&d| sp.dof_partition().permute_dof(d as u32) as usize)
                .filter(|&p| p < sp.dof_partition().n_owned_dofs)
                .collect();
            let ghost_ess: Vec<usize> = bc.iter()
                .map(|&d| sp.dof_partition().permute_dof(d as u32) as usize)
                .filter(|&p| p >= sp.dof_partition().n_owned_dofs && p < sp.dof_partition().n_total_dofs())
                .map(|p| p - sp.dof_partition().n_owned_dofs)
                .collect();
            m.eliminate_diag_symmetric_with_ghost(&owned_ess, &ghost_ess, 1.0);
            mats.push(m);
        }

        // 6. Build MgTransfer per level boundary (fine ← coarse).
        let mut transfers: Vec<MgTransfer> = Vec::with_capacity(n_levels - 1);
        for l in 0..n_levels - 1 {
            let c_dm = fem_space::dof_manager::DofManager::new(&local_mesh, orders[l]);
            let f_dm = fem_space::dof_manager::DofManager::new(&local_mesh, orders[l + 1]);
            let p_dm = build_h1_prolongation_matrix(&local_mesh, &c_dm, &local_mesh, &f_dm);
            let fine_dp = spaces[l + 1].dof_partition().clone();
            let coarse_dp = spaces[l].dof_partition().clone();
            let (prolong_local, restrict_local) = to_partition_order(
                &p_dm, &fine_dp, &coarse_dp,
            );
            if rank == 0 {
                println!(
                    "transfer {} -> {}: prolong {}x{}, restrict {}x{}, coarse_owned {}",
                    orders[l], orders[l + 1],
                    prolong_local.nrows, prolong_local.ncols,
                    restrict_local.nrows, restrict_local.ncols,
                    spaces[l].dof_partition().n_owned_dofs,
                );
                println!(
                    "level {}: mat_ghost {} dp_ghost {}",
                    l, mats[l].n_ghost(), spaces[l].dof_partition().n_ghost_dofs,
                );
            }
            transfers.push(MgTransfer {
                prolong_local,
                restrict_local,
                coarse_owned: spaces[l].dof_partition().n_owned_dofs,
                coarse_dp_for_global: spaces[l].dof_partition().clone(),
            });
        }

        // 7. RHS on the finest space.
        let fine_sp = spaces.last().unwrap();
        let fine_order = *orders.last().unwrap();
        let mut rhs_local = Assembler::assemble_linear(
            fine_sp.local_space(), &[&DomainSourceIntegrator::new(|_| 1.0)], qo(fine_order),
        );
        let bc_fine = boundary_dofs(
            fine_sp.local_space().mesh(), fine_sp.local_space().dof_manager(), &bnd_tags,
        );
        for &d in &bc_fine { rhs_local[d as usize] = 0.0; }
        let dp_fine = fine_sp.dof_partition();
        let rhs_perm = permute_vec(&rhs_local, dp_fine);
        let mut rhs = ParVector::from_local_raw(
            rhs_perm, dp_fine.n_owned_dofs, fine_sp.dof_ghost_exchange_arc(), comm.clone(),
        );
        rhs.update_ghosts();
        let mut u = ParVector::zeros(fine_sp);

        // 8. MG preconditioner (PCG wrapper: r → z via V-cycle).
        let n_owned_fine = dp_fine.n_owned_dofs;
        let ghost_arc = fine_sp.dof_ghost_exchange_arc();
        let comm2 = comm.clone();
        let mats_arc = Arc::new(mats);
        let trans_arc = Arc::new(transfers);
        let mats_pc = Arc::clone(&mats_arc);
        let trans_pc = Arc::clone(&trans_arc);
        let precond = move |r: &[f64], z: &mut [f64]| {
            let n_ghost_fine = mats_pc[n_levels - 1].n_ghost();
            let mut rv = ParVector::zeros_raw(n_owned_fine, n_ghost_fine, ghost_arc.clone(), comm2.clone());
            rv.owned_slice_mut().copy_from_slice(r);
            rv.update_ghosts();
            let mut zv = ParVector::zeros_like(&rv);
            v_cycle(&trans_pc, &mats_pc, n_levels - 1, &rv, &mut zv, &comm2);
            z[..n_owned_fine].copy_from_slice(&zv.as_slice()[..n_owned_fine]);
        };
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..Default::default() };
        let res = fem_parallel::par_solve_pcg_precond(
            &mats_arc[n_levels - 1], &rhs, &mut u, &precond, &cfg,
        ).expect("PCG failed");
        if rank == 0 {
            *result_slot.lock().unwrap() = Some(format!(
                "pex26: dofs={n_global} converged={} iters={} residual={:.3e}",
                res.converged, res.iterations, res.final_residual,
            ));
        }
    });

    let taken = result.lock().expect("pex26 result mutex").take();
    if let Some(s) = taken { println!("{s}"); }
}

/// Convert a dm-order prolongation P (fine_dm × coarse_dm) into partition-order
/// local CSR matrices:
///   - P_local: fine_owned × (coarse_owned + coarse_ghost)
///   - R_local: coarse_total × fine_total (rows cover coarse owned + ghost;
///     columns only the fine-owned slots — ghost columns are left zero because
///     the owning rank computes those contributions, avoiding double count)
fn to_partition_order(
    p_dm: &fem_linalg::CsrMatrix<f64>,
    fine_dp: &DofPartition,
    coarse_dp: &DofPartition,
) -> (fem_linalg::CsrMatrix<f64>, fem_linalg::CsrMatrix<f64>) {
    let n_fine_total = fine_dp.n_total_dofs();
    let n_coarse_total = coarse_dp.n_total_dofs();
    let n_fine_owned = fine_dp.n_owned_dofs;

    let mut p_coo = fem_linalg::CooMatrix::<f64>::new(n_fine_owned, n_coarse_total);
    // Restriction covers ALL coarse slots (owned + ghost): each rank's local
    // fine→coarse contributions go into owned *and* ghost coarse rows, then
    // the V-cycle accumulates them by global id.
    let mut r_coo = fem_linalg::CooMatrix::<f64>::new(n_coarse_total, n_fine_total);

    for fdm in 0..p_dm.nrows {
        let fpart = fine_dp.permute_dof(fdm as u32) as usize;
        if fpart >= n_fine_owned { continue; }
        for k in p_dm.row_ptr[fdm]..p_dm.row_ptr[fdm + 1] {
            let cdm = p_dm.col_idx[k] as usize;
            let cpart = coarse_dp.permute_dof(cdm as u32) as usize;
            let v = p_dm.values[k];
            if v.abs() < 1e-30 { continue; }
            p_coo.add(fpart, cpart, v);
            // Row cpart may be owned or ghost; all contributions are kept
            // locally and accumulated across ranks via global_sum_by_dof.
            r_coo.add(cpart, fpart, v);
        }
    }
    (p_coo.into_csr(), r_coo.into_csr())
}

/// Recursive V-cycle on the p-refinement hierarchy.
/// `lvl` = matrix level (0 = P1, 1 = P2, ...); `b`/`x` live in the `lvl` space.
/// `transfers[l]` carries P_{l+1} ← P_l (fine ← coarse).
fn v_cycle(
    transfers: &[MgTransfer],
    mats: &[ParCsrMatrix],
    lvl: usize,
    b: &ParVector,
    x: &mut ParVector,
    comm: &fem_parallel::Comm,
) {
    let n_own = mats[lvl].n_owned();

    // Coarsest level (P1): parallel CG with loose tolerance (the small P1
    // system converges quickly; a few iterations suffice for the MG cycle).
    if lvl == 0 {
        let cfg = SolverConfig { rtol: 1e-4, atol: 0.0, max_iter: 30, verbose: false, ..Default::default() };
        let mut xc = x.clone_vec();
        xc.owned_slice_mut().fill(0.0);
        let _ = fem_parallel::par_solve_cg(&mats[lvl], b, &mut xc, &cfg);
        x.owned_slice_mut().copy_from_slice(&xc.as_slice()[..n_own]);

        return;
    }

    let omega = 0.8;
    // Pre-smooth: damped Jacobi.
    let diag: Vec<f64> = mats[lvl].diag_block().diagonal();
    for i in 0..n_own {
        let d = if diag[i].abs() > 1e-30 { diag[i] } else { 1.0 };
        x.as_slice_mut()[i] = omega * b.as_slice()[i] / d;
    }

    x.update_ghosts();

    // Residual: r = b - A*x
    let mut ax = ParVector::zeros_like(b);
    mats[lvl].spmv(x, &mut ax);

    let mut r = ParVector::zeros_like(b);
    for i in 0..n_own { r.as_slice_mut()[i] = b.as_slice()[i] - ax.as_slice()[i]; }
    r.update_ghosts();

    // Restrict: r_c = R_{l-1} * r  (ALL coarse rows owned+ghost × fine cols;
    // fine ghost columns are zero by construction — the owning rank computes
    // them — so local_spmv needs the full ghosted fine residual).
    let tr = &transfers[lvl - 1];
    let n_coarse_own = tr.coarse_owned;
    let n_coarse_total = tr.restrict_local.nrows;
    let mut r_c = vec![0.0_f64; n_coarse_total];
    local_spmv(&tr.restrict_local, r.as_slice(), &mut r_c);
    // Cross-rank sum of the coarse residual by GLOBAL dof id.  Each rank's
    // local restriction fills owned AND ghost coarse rows: a coarse dof's
    // fine contributions come from the ranks owning its fine dofs, and those
    // ranks may hold the coarse dof only as a ghost slot.  Broadcasting
    // every local slot by gid and summing per gid is exact (no double
    // counting: fine-owned columns are non-zero only on the owning rank).
    let coarse_dp = &transfers[lvl - 1].coarse_dp_for_global;
    let global_r_c = global_sum_by_dof(&r_c, coarse_dp, comm);
    let n_coarse_ghost = mats[lvl - 1].n_ghost();
    let mut rv_c = ParVector::zeros_raw(
        n_coarse_own, n_coarse_ghost,
        mats[lvl - 1].ghost_exchange_arc(), comm.clone(),
    );
    rv_c.owned_slice_mut().copy_from_slice(&global_r_c[..n_coarse_own]);
    rv_c.update_ghosts();
    let mut ev_c = ParVector::zeros_like(&rv_c);
    v_cycle(transfers, mats, lvl - 1, &rv_c, &mut ev_c, comm);

    // Prolong: x += P_{l-1} * e_c  (fine_owned × coarse_total)
    // Sync the coarse correction to ghost slots first (the recursive coarse
    // solve only fills owned slots; prolongation needs ghost values too).
    ev_c.update_ghosts();
    let mut corr = vec![0.0_f64; n_own];
    local_spmv(&tr.prolong_local, ev_c.as_slice(), &mut corr);
    for i in 0..n_own { x.as_slice_mut()[i] += corr[i]; }
    x.update_ghosts();

    // Post-smooth: damped Jacobi on (b - A*x) residual update.
    let mut ax2 = ParVector::zeros_like(b);
    mats[lvl].spmv(x, &mut ax2);
    let diag2: Vec<f64> = mats[lvl].diag_block().diagonal();
    for i in 0..n_own {
        let d = if diag2[i].abs() > 1e-30 { diag2[i] } else { 1.0 };
        x.as_slice_mut()[i] += omega * (b.as_slice()[i] - ax2.as_slice()[i]) / d;
    }
    x.update_ghosts();
}

/// Local CSR SpMV (diag-only; the operator's column range covers the input).
fn local_spmv(a: &fem_linalg::CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows.min(y.len()) {
        let mut s = 0.0;
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[k] as usize;
            if c < x.len() { s += a.values[k] * x[c]; }
        }
        y[i] = s;
    }
}

/// Sum `local` (indexed by ALL local coarse slots — owned + ghost) across
/// ranks by GLOBAL dof id.  Each rank broadcasts every local slot's
/// (gid, value); every rank accumulates incoming values into the slot for
/// that gid.  Contributions from different ranks are disjoint (a fine-owned
/// restriction column is non-zero only on the rank owning that fine dof),
/// so the sum is the exact global restriction: every slot ends up holding
/// the full global value for its gid.
fn global_sum_by_dof(
    local: &[f64],
    coarse_dp: &DofPartition,
    comm: &fem_parallel::Comm,
) -> Vec<f64> {
    let n_total = coarse_dp.n_total_dofs();
    let n_owned = coarse_dp.n_owned_dofs;
    debug_assert!(local.len() >= n_total);
    let n_ranks = comm.size() as i32;
    let rank = comm.rank();

    // Broadcast (gid, value) of every local slot (owned + ghost) to every
    // other rank.
    let mut send_bundles: Vec<(i32, Vec<u8>)> = Vec::new();
    for r in 0..n_ranks {
        if r == rank { continue; }
        let mut payload = Vec::with_capacity(n_total * 12);
        for i in 0..n_total {
            let gid = coarse_dp.global_dof(i as u32);
            payload.extend_from_slice(&gid.to_le_bytes());
            payload.extend_from_slice(&local[i].to_le_bytes());
        }
        send_bundles.push((r, payload));
    }
    let recv = comm.alltoallv_bytes(&send_bundles);

    // Start with our own contributions (owned + ghost slots).
    let mut sum = local[..n_total].to_vec();
    // Accumulate every incoming (gid, value) into that gid's local slot.
    for (_src, bytes) in recv {
        for chunk in bytes.chunks_exact(12) {
            let gid = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let val = f64::from_le_bytes(chunk[4..12].try_into().unwrap());
            if let Some(lid) = coarse_dp.local_dof(gid) {
                sum[lid as usize] += val;
            }
        }
    }
    sum
}

fn arg(args: &[String], key: &str) -> Option<usize> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
}
