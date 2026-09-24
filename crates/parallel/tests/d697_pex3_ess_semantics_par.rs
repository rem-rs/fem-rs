//! D697: pex3 non-homogeneous PEC elimination — the core-level semantics pin.
//!
//! Round-67 bisection (worktree probes `tmp/d697/`, both ranks, release):
//! the pex3 regression `‖E−E_h‖ = 2.7005305e-2 → 1.6238508e0` flips exactly at
//! `25c4c99` (D124): parent `a5307e06` reproduces the round-30 ledger values
//! bit-for-bit (ranks1 87 it / 2.70053048394657e-2, ranks2 102 it /
//! 2.70053055689122e-2) while `25c4c99` already yields the broken values
//! (271/2085 it, 1.62385083e0) that HEAD still shows.
//!
//! Root cause (this file pins the semantics that locate it): MFEM
//! `GetEssentialTrueDofs` (pfespace.cpp:1165) provides **indices only** — the
//! elimination **values** flow from the projected solution `x` inside
//! `FormLinearSystem(ess_tdof_list, x, b, A, X, B)` (ex3p.cpp:202
//! `x.ProjectCoefficient(E)`, :223).  The D124 pex3 rewiring adopted the new
//! distributed index entry but passed `0.0` as every essential value,
//! silently homogenizing the non-homogeneous PEC boundary (`E·t = O(1)` on
//! the boundary for the exact field `(sin κy, sin κx)`).  The crates-side
//! chain — `essential_true_dofs` (halo-synced set),
//! `ParCsrMatrix::apply_dirichlet_par_keep_diag` (value-bearing),
//! `ParCsrMatrix::apply_ghost_ess_columns` (value-bearing) — is correct:
//!
//! 1. `essential_true_dofs` equals the pre-D124 manual detection (per-rank
//!    `boundary_dofs_hcurl` + global-id alltoallv union, restricted to owned)
//!    at 1/2/4 ranks, with the global count matching the serial set — the
//!    distributed entry introduced no set defect;
//! 2. eliminating with the **projected values** reproduces them exactly at
//!    the solved vector's essential entries (the DIAG_KEEP row is
//!    `A[i,i]·v`), np1 and np2;
//! 3. eliminating with `0.0` (the regression) moves the solution O(1) away
//!    from the values-solution — the in-core signature of the pex3 flip.
//!
//! The pex3 example-side restoration (projected values in place of the
//! hardcoded zeros) is registered as D705 and owned by the examples lane;
//! with it pex3 returns to 87/102 it, ‖E−E_h‖ = 2.7005e-2 (C++ 4.10:
//! 17/19 it, 0.0270053 — the iteration gap is the documented PCG+AMG vs
//! HypreAMS solver-stack deviation, round-30 adjudication).

use std::collections::HashSet;

use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_mesh::Mesh;
use fem_mesh::amr::refine_uniform;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_assembler::permute_vec;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::par_vector::ParVector;
use fem_parallel::{
    Comm, ParAmgConfig, ParVectorAssembler, SmootherType, WorkerConfig, par_solve_pcg_amg,
};
use fem_space::constraints::boundary_dofs_hcurl;
use fem_space::fe_space::FESpace;
use fem_space::HCurlSpace;
use fem_solver::SolverConfig;

const KAPPA: f64 = std::f64::consts::PI;

/// The pex3 manufactured family phase-shifted by π/2: `(cos κy, cos κx)`.
/// pex3's own `(sin κy, sin κx)` has an identically-zero tangential trace on
/// axis-aligned boundaries (E_x vanishes at y∈{0,1}, E_y at x∈{0,1}), so the
/// homogenization regression is invisible there; on star.mesh's oblique
/// boundary edges the trace is O(1), and the phase shift reproduces exactly
/// that on the programmatic square while satisfying the same PDE
/// `(curl curl + 1)E = (1+κ²)E` (divergence-free, ΔE = −κ²E).
fn exact_e(x: &[f64]) -> [f64; 2] {
    [(KAPPA * x[1]).cos(), (KAPPA * x[0]).cos()]
}

struct Src;
impl VectorLinearIntegrator for Src {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
        let c = 1.0 + KAPPA * KAPPA;
        let fx = c * (KAPPA * qp.x_phys[1]).cos();
        let fy = c * (KAPPA * qp.x_phys[0]).cos();
        for i in 0..qp.n_dofs {
            fe[i] += qp.weight * (qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy);
        }
    }
}

type Pex3Space = ParallelFESpace<HCurlSpace<Mesh<2>>>;

/// The pex3-like mesh: ex3p folds par_ref_levels=2 into the serial
/// refinement; the same folding here (2 levels on the programmatic base)
/// keeps the family/order/BC shape of the pex3 configuration on a mesh that
/// needs no data files.
fn pex3_mesh() -> Mesh<2> {
    let base = Mesh::<2>::unit_square_tri(4);
    refine_uniform(&refine_uniform(&base))
}

fn build_pex3_space(comm: &Comm) -> Pex3Space {
    let mesh = pex3_mesh();
    let pm = partition_mesh(&mesh, comm);
    let lm = pm.local_mesh().clone();
    ParallelFESpace::new_for_edge_space(HCurlSpace::new(lm, 1), &pm, comm.clone())
}

/// The pre-D124 essential-set semantics (the old pex3 in-example workaround):
/// per-rank serial detection, gid alltoallv union, owned restriction.
fn manual_union_owned_gids(ps: &Pex3Space) -> Vec<u32> {
    let comm = ps.comm();
    let dp = ps.dof_partition();
    let local: Vec<u32> = boundary_dofs_hcurl(ps.local_space().mesh(), ps.local_space(), &[1])
        .iter()
        .map(|&d| dp.global_dof(dp.permute_dof(d)))
        .collect();
    let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
    for r in 0..comm.size() as i32 {
        if r == comm.rank() {
            continue;
        }
        let mut bytes = Vec::with_capacity(local.len() * 4);
        for &g in &local {
            bytes.extend_from_slice(&g.to_le_bytes());
        }
        sends.push((r, bytes));
    }
    let incoming = comm.alltoallv_bytes(&sends);
    let mut all_bnd: HashSet<u32> = local.iter().copied().collect();
    for (_, bytes) in incoming {
        for chunk in bytes.chunks_exact(4) {
            all_bnd.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
    }
    let mut owned: Vec<u32> = (0..dp.n_owned_dofs)
        .filter(|&pid| all_bnd.contains(&dp.global_dof(pid as u32)))
        .map(|pid| dp.global_dof(pid as u32))
        .collect();
    owned.sort_unstable();
    owned
}

/// The D124 distributed entry, restricted to owned dofs, as global ids.
fn entry_owned_gids(ps: &Pex3Space) -> Vec<u32> {
    let dp = ps.dof_partition();
    let mut gids: Vec<u32> = ps
        .essential_true_dofs(&[1])
        .iter()
        .filter_map(|&d| {
            let pid = dp.permute_dof(d) as usize;
            (pid < dp.n_owned_dofs).then_some(dp.global_dof(pid as u32))
        })
        .collect();
    gids.sort_unstable();
    gids.dedup();
    gids
}

/// Max-reduction (the Comm facade only offers sum): exchange one f64 per
/// rank over the alltoallv byte channel and take the max.
fn allreduce_max_f64(comm: &Comm, local: f64) -> f64 {
    let sends: Vec<(i32, Vec<u8>)> = (0..comm.size() as i32)
        .filter(|&r| r != comm.rank())
        .map(|r| (r, local.to_le_bytes().to_vec()))
        .collect();
    let incoming = comm.alltoallv_bytes(&sends);
    incoming.iter().fold(local, |m, (_, bytes)| {
        let v = f64::from_le_bytes(bytes[..8].try_into().unwrap());
        m.max(v)
    })
}

fn amg_cfg() -> ParAmgConfig {
    ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        n_pre_smooth: 2,
        n_post_smooth: 2,
        smoothed_prolongation: true,
        block_size: 1,
        use_global_aggregation: false,
        ..ParAmgConfig::default()
    }
}

fn solver_cfg() -> SolverConfig {
    SolverConfig { rtol: 1e-8, max_iter: 10000, verbose: false, ..Default::default() }
}

/// Solve the pex3 system; `homogenize` selects the BC value
/// (false = MFEM semantics, projected exact values; true = the D124 pex3
/// regression's hardcoded zeros).  Returns the solved owned slice.
fn solve_pex3_like(ps: &Pex3Space, u0: &ParVector, homogenize: bool) -> Vec<f64> {
    let comm = ps.comm().clone();
    let dp = ps.dof_partition();

    let mut stiff = ParVectorAssembler::assemble_bilinear(
        ps,
        &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
        4,
    );
    let mut rhs = ParVectorAssembler::assemble_linear(ps, &[&Src], 4);

    let mut owned_ess: Vec<(usize, f64)> = Vec::new();
    let mut ghost_ess: Vec<(usize, f64)> = Vec::new();
    for &d in ps.essential_true_dofs(&[1]).iter() {
        let pid = dp.permute_dof(d) as usize;
        let v = if homogenize { 0.0 } else { u0.as_slice()[pid] };
        if pid < dp.n_owned_dofs {
            owned_ess.push((pid, v));
        } else {
            ghost_ess.push((pid - dp.n_owned_dofs, v));
        }
    }
    for &(pid, v) in &owned_ess {
        stiff.apply_dirichlet_par_keep_diag(pid, v, &mut rhs);
    }
    stiff.apply_ghost_ess_columns(&ghost_ess, &mut rhs);

    let mut u = ParVector::from_local_raw(
        vec![0.0; dp.n_total_dofs()],
        dp.n_owned_dofs,
        ps.dof_ghost_exchange_arc(),
        comm,
    );
    par_solve_pcg_amg(&stiff, &rhs, &mut u, &amg_cfg(), &solver_cfg())
        .expect("PCG+AMG solve failed");
    u.owned_slice().to_vec()
}

// ── 1. the distributed entry equals the manual gid union ────────────────────

#[test]
fn d697_ess_set_matches_manual_gid_union() {
    for np in [1usize, 2, 4] {
        ThreadLauncher::new(WorkerConfig::new(np)).launch(move |comm| {
            let ps = build_pex3_space(&comm);
            let manual = manual_union_owned_gids(&ps);
            let entry = entry_owned_gids(&ps);
            assert_eq!(
                manual, entry,
                "np{}: essential_true_dofs != manual gid union (owned)",
                comm.rank()
            );
            // Global count invariant: the sum over ranks of the owned parts
            // equals the serial whole-mesh set size.
            let serial_len =
                boundary_dofs_hcurl(&pex3_mesh(), &HCurlSpace::new(pex3_mesh(), 1), &[1])
                    .len();
            let total_owned = comm.allreduce_sum_f64(entry.len() as f64) as usize;
            assert_eq!(
                total_owned, serial_len,
                "np{}: global essential count {} != serial set size {serial_len}",
                comm.rank(),
                total_owned
            );
            // np1 bitwise guarantee (the documented single-rank case): the
            // entry's local ids equal the serial collector's.
            if np == 1 {
                let serial = boundary_dofs_hcurl(
                    ps.local_space().mesh(),
                    ps.local_space(),
                    &[1],
                );
                let bitwise: Vec<u32> =
                    ps.essential_true_dofs(&[1]).iter().map(|&d| d as u32).collect();
                let serial_ids: Vec<u32> = serial.iter().map(|&d| d as u32).collect();
                assert_eq!(bitwise, serial_ids, "np1 entry != serial collector");
            }
        });
    }
}

// ── 2/3. values-chain fidelity + homogenization signature ───────────────────

#[test]
fn d697_value_chain_fidelity_and_zero_regression_signature() {
    for np in [1usize, 2] {
        ThreadLauncher::new(WorkerConfig::new(np)).launch(move |comm| {
            let ps = build_pex3_space(&comm);
            let dp = ps.dof_partition();

            // Projected exact field in partition order (BC values and the
            // fidelity reference — MFEM's `x.ProjectCoefficient(E)`).  The
            // ghost slots carry the same interpolated values as their owners
            // (deterministic interpolation), so they feed
            // `apply_ghost_ess_columns` directly.
            let x0 = ps
                .local_space()
                .interpolate_vector(&|p| exact_e(p).to_vec());
            let x0_perm = permute_vec(x0.as_slice(), dp);
            let u0 = ParVector::from_local_raw(
                x0_perm,
                dp.n_owned_dofs,
                ps.dof_ghost_exchange_arc(),
                comm.clone(),
            );

            // The mathematical premise: the exact field's projected boundary
            // trace is O(1) — homogenizing it is a real defect, not a no-op.
            let max_bc_local = (0..dp.n_owned_dofs)
                .map(|pid| u0.as_slice()[pid].abs())
                .fold(0.0_f64, f64::max);
            let max_bc = allreduce_max_f64(&comm, max_bc_local);
            assert!(max_bc > 0.05, "np{np}: projected BC trace ~0 ({max_bc})");

            let u_vals = solve_pex3_like(&ps, &u0, false);
            let u_zero = solve_pex3_like(&ps, &u0, true);

            // Fidelity: with MFEM's value-bearing elimination the solved
            // essential entries ARE the projected values (DIAG_KEEP row is
            // A[i,i]·v — the solve returns v exactly).
            let mut fid = 0.0_f64;
            for &d in ps.essential_true_dofs(&[1]).iter() {
                let pid = dp.permute_dof(d) as usize;
                if pid < dp.n_owned_dofs {
                    fid = fid.max((u_vals[pid] - u0.as_slice()[pid]).abs());
                }
            }
            assert!(
                fid < 1e-10,
                "np{np}: value-chain fidelity broken (max|Δ| = {fid})"
            );

            // Regression signature: the hardcoded-zeros variant solves a
            // DIFFERENT problem — O(1) away from the values solution.
            let drift_local: f64 = u_vals
                .iter()
                .zip(u_zero.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let drift = allreduce_max_f64(&comm, drift_local);
            assert!(
                drift > 0.1,
                "np{np}: homogenized BC did not move the solution ({drift})"
            );
        });
    }
}
