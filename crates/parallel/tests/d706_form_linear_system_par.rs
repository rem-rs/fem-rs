//! D706: the MFEM `FormLinearSystem` shape core entry — the root fix of the
//! D697 accident surface.
//!
//! Round-67 D697 root cause chain: pex3 hand-derived the essential-BC
//! elimination (owned/ghost split, value plumbing) at the example layer and a
//! refactor (`25c4c99`) hardcoded the values to 0.0, silently homogenizing
//! the non-homogeneous PEC boundary.  The root fix is a **core entry** with
//! MFEM's one-stop signature —
//! `form_linear_system(a, par_space, ess_tdof_list, x, b, policy) -> X`
//! (MFEM `ParBilinearForm::FormLinearSystem`, pbilinearform.cpp:475, with
//! `FormSystemMatrix` = the row/col elimination under `diag_policy` and
//! `EliminateBC(Ae, ess, x, b)`, hypre.cpp:2461, = the `b -= Ae·x` reactions
//! plus the ess-row overwrite `B(r) = A(r,r)·x(r)`) — so the values can no
//! longer be decoupled from the indices.
//!
//! Pins (the pex3 example-side migration gate is the 87/102 it +
//! 2.70053048394657e-2 / 2.70053055702279e-2 bitwise run, verified in the
//! round-68 session):
//!
//! 1. the new entry reproduces the former two-step path
//!    (`apply_dirichlet_par_keep_diag` + `apply_ghost_ess_columns`)
//!    **bitwise** on the eliminated matrix and RHS, and returns
//!    `X = x` bitwise (conforming restriction, `copy_interior = 1`) — the
//!    migration-safety proof;
//! 2. a direct value-level reference of MFEM `EliminateBC` semantics
//!    (`B = b − Ae·x`, ess rows `B(r) = A(r,r)·x(r)`), entrywise exact on
//!    every owned row, with genuine cross-rank ghost reactions at np = 2;
//! 3. the `DIAG_ONE` strategy: identity rows (diag = 1, off-diagonals zero in
//!    both blocks), exact value fidelity at the solved essential entries, and
//!    the homogenization signature (zeros ≠ projected values) so the policy
//!    path is value-bearing too.

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
    Comm, ElimPolicy, ParAmgConfig, ParCsrMatrix, ParVectorAssembler, SmootherType, WorkerConfig,
    par_solve_pcg_amg,
};
use fem_solver::SolverConfig;
use fem_space::fe_space::FESpace;
use fem_space::HCurlSpace;

const KAPPA: f64 = std::f64::consts::PI;

type Pex3Space = ParallelFESpace<HCurlSpace<Mesh<2>>>;

/// The d697 phase-shifted pex3 family `(cos κy, cos κx)`: O(1) tangential
/// trace on axis-aligned boundaries (the homogenization regression is O(1)
/// visible), same PDE `(curl curl + 1)E = (1+κ²)E`.
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

/// The pex3-like fixture: unit square, 2 folded refinement levels, ND1,
/// quad order 4 (ex3p folds par_ref_levels = 2 into the serial refinement).
fn build_pex3_space(comm: &Comm) -> Pex3Space {
    let base = Mesh::<2>::unit_square_tri(4);
    let mesh = refine_uniform(&refine_uniform(&base));
    let pm = partition_mesh(&mesh, comm);
    let lm = pm.local_mesh().clone();
    ParallelFESpace::new_for_edge_space(HCurlSpace::new(lm, 1), &pm, comm.clone())
}

/// The projected exact field in partition order (MFEM's
/// `x.ProjectCoefficient(E)` restricted to the true dofs).
fn projected_solution(ps: &Pex3Space, comm: &Comm) -> ParVector {
    let dp = ps.dof_partition();
    let x0 = ps.local_space().interpolate_vector(&|p| exact_e(p).to_vec());
    let x0_perm = permute_vec(x0.as_slice(), dp);
    ParVector::from_local_raw(
        x0_perm,
        dp.n_owned_dofs,
        ps.dof_ghost_exchange_arc(),
        comm.clone(),
    )
}

fn assemble(ps: &Pex3Space) -> (ParCsrMatrix, ParVector) {
    let stiff = ParVectorAssembler::assemble_bilinear(
        ps,
        &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
        4,
    );
    let rhs = ParVectorAssembler::assemble_linear(ps, &[&Src], 4);
    (stiff, rhs)
}

/// The owned/ghost (slot, value) split of the essential set, dm order.
fn split_ess(
    ps: &Pex3Space,
    ess: &[fem_core::types::DofId],
    x: &ParVector,
) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    let dp = ps.dof_partition();
    let mut owned = Vec::new();
    let mut ghost = Vec::new();
    for &d in ess {
        let pid = dp.permute_dof(d) as usize;
        let v = x.as_slice()[pid];
        if pid < dp.n_owned_dofs {
            owned.push((pid, v));
        } else {
            ghost.push((pid - dp.n_owned_dofs, v));
        }
    }
    (owned, ghost)
}

fn amg_cfg() -> ParAmgConfig {
    ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        n_pre_smooth: 2,
        n_post_smooth: 2,
        smoothed_prolongation: true,
        block_size: 1,
        ..ParAmgConfig::default()
    }
}

fn solver_cfg() -> SolverConfig {
    SolverConfig { rtol: 1e-8, max_iter: 10000, verbose: false, ..Default::default() }
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

// ── 1. migration safety: the entry IS the former two-step path, bitwise ─────

#[test]
fn d706_entry_equals_former_two_step_path_bitwise() {
    for np in [1usize, 2] {
        ThreadLauncher::new(WorkerConfig::new(np)).launch(move |comm| {
            let ps = build_pex3_space(&comm);
            let ess = ps.essential_true_dofs(&[1]);
            let x = projected_solution(&ps, &comm);

            // New entry path.
            let (mut a_new, mut b_new) = assemble(&ps);
            let x_new = ParVectorAssembler::form_linear_system(
                &mut a_new, &ps, &ess, &x, &mut b_new, ElimPolicy::DiagKeep,
            );

            // Former example-side two-step path (the pre-D706 pex3 block).
            let (mut a_old, mut b_old) = assemble(&ps);
            let (owned_ess, ghost_ess) = split_ess(&ps, &ess, &x);
            for &(pid, v) in &owned_ess {
                a_old.apply_dirichlet_par_keep_diag(pid, v, &mut b_old);
            }
            a_old.apply_ghost_ess_columns(&ghost_ess, &mut b_old);

            // Eliminated operators bitwise.
            let m_new = a_new.to_local_matrix();
            let m_old = a_old.to_local_matrix();
            assert_eq!(m_new.row_ptr, m_old.row_ptr, "np{np}: pattern drifted");
            assert_eq!(m_new.col_idx, m_old.col_idx, "np{np}: pattern drifted");
            for (i, (&n, &o)) in m_new.values.iter().zip(m_old.values.iter()).enumerate() {
                assert_eq!(
                    n.to_bits(), o.to_bits(),
                    "np{np} rank {}: eliminated matrix bit-drift at nnz #{i}: {n} vs {o}",
                    comm.rank()
                );
            }

            // Eliminated RHS bitwise (owned rows are the true-dof B).
            for (i, (&n, &o)) in b_new
                .owned_slice()
                .iter()
                .zip(b_old.owned_slice().iter())
                .enumerate()
            {
                assert_eq!(
                    n.to_bits(), o.to_bits(),
                    "np{np} rank {}: eliminated rhs bit-drift at row {i}: {n} vs {o}",
                    comm.rank()
                );
            }

            // X = x bitwise (conforming restriction, copy_interior = 1).
            for (i, (&n, &o)) in x_new
                .as_slice()
                .iter()
                .zip(x.as_slice().iter())
                .enumerate()
            {
                assert_eq!(
                    n.to_bits(), o.to_bits(),
                    "np{np} rank {}: X bit-drift at slot {i}",
                    comm.rank()
                );
            }

            // The ghost path must be genuinely exercised at np > 1.
            if np > 1 {
                let ghost_total = comm.allreduce_sum_f64(ghost_ess.len() as f64) as usize;
                assert!(ghost_total > 0, "np{np}: no cross-rank ess ghosts — test vacuous");
            }
        });
    }
}

// ── 2. direct EliminateBC value reference (incl. cross-rank reactions) ──────

#[test]
fn d706_eliminated_rhs_matches_eliminate_bc_reference() {
    for np in [1usize, 2] {
        ThreadLauncher::new(WorkerConfig::new(np)).launch(move |comm| {
            let ps = build_pex3_space(&comm);
            let ess = ps.essential_true_dofs(&[1]);
            let x = projected_solution(&ps, &comm);
            let (owned_ess, ghost_ess) = split_ess(&ps, &ess, &x);

            // Reference: from the UNeliminated snapshot,
            //   B[j] = b[j] − Σ_k A[j,k]·v_k − Σ_g Aoffd[j,g]·v_g   (k, g ess),
            //   B[r] = A(r,r)·v_r                                    (owned ess r),
            // the accumulation order matching the in-place elimination
            // (ess-list order, owned pass then ghost pass).
            let (a_ref, b_ref) = assemble(&ps);
            let diag = a_ref.diag_block();
            let offd = a_ref.offd_block();
            let n_owned = a_ref.n_owned();
            let mut expect = b_ref.owned_slice().to_vec();
            for row in 0..n_owned {
                for &(k, v) in &owned_ess {
                    if k == row {
                        continue;
                    }
                    let a_jk = diag.get(row, k);
                    if a_jk != 0.0 {
                        expect[row] -= a_jk * v;
                    }
                }
                for &(g, v) in &ghost_ess {
                    let s = offd.row_ptr[row];
                    let e = offd.row_ptr[row + 1];
                    let mut found = 0.0_f64;
                    for p in s..e {
                        if offd.col_idx[p] as usize == g {
                            found = offd.values[p];
                            break;
                        }
                    }
                    if found != 0.0 {
                        expect[row] -= found * v;
                    }
                }
            }
            for &(r, v) in &owned_ess {
                expect[r] = diag.get(r, r) * v;
            }

            // The entry, from the same snapshot.
            let (mut a, mut b) = assemble(&ps);
            let _x = ParVectorAssembler::form_linear_system(
                &mut a, &ps, &ess, &x, &mut b, ElimPolicy::DiagKeep,
            );
            for (i, (&got, &want)) in b.owned_slice().iter().zip(expect.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(), want.to_bits(),
                    "np{np} rank {}: B[{i}] = {got} != EliminateBC reference {want}",
                    comm.rank()
                );
            }
        });
    }
}

// ── 3. the DIAG_ONE strategy: identity rows + fidelity + value bearing ─────

#[test]
fn d706_diag_one_identity_rows_fidelity_and_value_bearing() {
    for np in [1usize, 2] {
        ThreadLauncher::new(WorkerConfig::new(np)).launch(move |comm| {
            let ps = build_pex3_space(&comm);
            let ess = ps.essential_true_dofs(&[1]);
            let x = projected_solution(&ps, &comm);
            let (owned_ess, _ghost) = split_ess(&ps, &ess, &x);

            // Eliminate with DIAG_ONE + projected values.
            let (mut a1, mut b1) = assemble(&ps);
            let _ = ParVectorAssembler::form_linear_system(
                &mut a1, &ps, &ess, &x, &mut b1, ElimPolicy::DiagOne,
            );

            // Every owned ess row is an identity row: diag = 1, all
            // off-diagonals (diag block AND offd block) zero.
            let diag = a1.diag_block();
            let offd = a1.offd_block();
            for &(r, _) in &owned_ess {
                for p in diag.row_ptr[r]..diag.row_ptr[r + 1] {
                    let c = diag.col_idx[p] as usize;
                    let want = if c == r { 1.0 } else { 0.0 };
                    assert_eq!(
                        diag.values[p], want,
                        "np{np} rank {}: DiagOne row {r} col {c} = {} != {want}",
                        comm.rank(),
                        diag.values[p]
                    );
                }
                for p in offd.row_ptr[r]..offd.row_ptr[r + 1] {
                    assert_eq!(
                        offd.values[p], 0.0,
                        "np{np} rank {}: DiagOne offd row {r} not zeroed",
                        comm.rank()
                    );
                }
                // B(r) = x(r) exactly.
                assert_eq!(
                    b1.owned_slice()[r].to_bits(),
                    x.owned_slice()[r].to_bits(),
                    "np{np} rank {}: DiagOne B[{r}] != x[{r}]",
                    comm.rank()
                );
            }

            // Solve both policies from zero; the essential entries of the
            // DIAG_ONE solve are the projected values exactly (identity
            // rows), and the interiors agree — same PDE, different
            // elimination shape.
            let mut u1 = ParVector::zeros(&ps);
            par_solve_pcg_amg(&a1, &b1, &mut u1, &amg_cfg(), &solver_cfg())
                .expect("DIAG_ONE solve failed");

            let (mut a2, mut b2) = assemble(&ps);
            let _ = ParVectorAssembler::form_linear_system(
                &mut a2, &ps, &ess, &x, &mut b2, ElimPolicy::DiagKeep,
            );
            let mut u2 = ParVector::zeros(&ps);
            par_solve_pcg_amg(&a2, &b2, &mut u2, &amg_cfg(), &solver_cfg())
                .expect("DIAG_KEEP solve failed");

            for &(r, v) in &owned_ess {
                // Row-residual bound: DiagOne's identity row leaves
                // |u[r]−v| = |r_f[r]| (no diag amplification), so the
                // fidelity is bounded by the global rtol only; DiagKeep's
                // A(r,r) factor amplifies it (d697's 1e-10).
                assert!(
                    (u1.owned_slice()[r] - v).abs() < 5e-7,
                    "np{np} rank {}: DiagOne fidelity broken at {r}",
                    comm.rank()
                );
                assert!(
                    (u2.owned_slice()[r] - v).abs() < 1e-10,
                    "np{np} rank {}: DiagKeep fidelity broken at {r}",
                    comm.rank()
                );
            }
            let diff: f64 = u1
                .owned_slice()
                .iter()
                .zip(u2.owned_slice().iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max);
            // Sanity (not the pin): both solves sit at rtol 1e-8 and the two
            // policies scale the ess diagonals differently, so the PCG paths
            // differ — the interiors agree within the rtol band.
            assert!(
                diff < 1e-6,
                "np{np} rank {}: DiagOne/DiagKeep interiors disagree ({diff})",
                comm.rank()
            );

            // Value bearing under the new policy: hardcoded zeros move the
            // solution O(1) — the D697 signature must be unreachable through
            // any policy.
            let mut x_zero = x.clone_vec();
            for slot in x_zero.as_slice_mut().iter_mut() {
                *slot = 0.0;
            }
            let (mut a3, mut b3) = assemble(&ps);
            let _ = ParVectorAssembler::form_linear_system(
                &mut a3, &ps, &ess, &x_zero, &mut b3, ElimPolicy::DiagOne,
            );
            let mut u3 = ParVector::zeros(&ps);
            par_solve_pcg_amg(&a3, &b3, &mut u3, &amg_cfg(), &solver_cfg())
                .expect("homogenized solve failed");
            let drift_local: f64 = u1
                .owned_slice()
                .iter()
                .zip(u3.owned_slice().iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max);
            // Global: at np = 2 the owned ess dofs may sit on a single rank,
            // so the O(1) signature must be measured across the union.
            let drift = allreduce_max_f64(&comm, drift_local);
            assert!(
                drift > 0.1,
                "np{np} rank {}: homogenized DiagOne did not move the solution ({drift})",
                comm.rank()
            );
        });
    }
}
