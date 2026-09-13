//! Parallel (distributed) low-order-refined solvers — the driver shared by the
//! per-rank LOR machinery.
//!
//! # Where ParLOR lives in fem-rs
//!
//! MFEM's `miniapps/solvers/plor_solvers.cpp` builds a
//! `ParFiniteElementSpace`, a `LORSolver<HypreBoomerAMG>` (H¹ / L²),
//! `LORSolver<HypreAMS>` (H(curl), 2-D H(div)) or `LORSolver<HypreADS>`
//! (3-D H(div)) and hands the pair to a parallel `CGSolver`.  The LOR
//! preconditioner is `M⁻¹ = Π · A_LOR⁻¹ · Πᵀ`, with `Π` the assumed-constraint
//! permutation `LOR → HO` (`LORBase::GetDofPermutation`) and `A_LOR` the
//! low-order operator assembled on the Gauss-Lobatto refined mesh.
//!
//! In fem-rs that splits along the crate boundary:
//!
//! * the **distributed layout** — partitioning, halo exchange, the per-rank LOR
//!   matrix in `[owned | ghost]` ordering and its AMG hierarchy — is built from
//!   `fem-parallel` (`ParMesh`, `ParallelFESpace`, `ParAssembler`,
//!   `ParCsrMatrix`, `ParAmgHierarchy`) by whoever owns a `ParallelFESpace`;
//! * the **solver driver** lives here: the PCG that consumes those objects and
//!   — unlike the vendor PCG — stops on the **true** residual `‖b − A x‖/‖b‖`.
//!
//! The driver is generic over two small traits so that `fem-solver` needs no
//! dependency on `fem-parallel` (which depends on `fem-solver`, so a direct
//! edge would be a cycle).
//!
//! # What the caller must supply (and two traps)
//!
//! 1. **`M⁻¹` must be linear.** `ParAmgHierarchy::vcycle(b, x)` returns
//!    `x + M⁻¹(b − A x)`: an adapter that reuses its output buffer without
//!    zeroing it turns the preconditioner into an *affine* map.  PCG then loses
//!    conjugacy silently — the preconditioned residual falls for ~8 iterations
//!    and then *rises* while the true residual stalls at ~3e-2 (measured).
//!    Zero the target vector before every V-cycle.
//! 2. **The essential dofs must be exchanged both ways.** The rank-local
//!    `boundary_dofs` is not the distributed essential set: the one-layer ghost
//!    overlap can hide the boundary face of an *owned* edge-interior dof (the
//!    owner leaves it free; measured 2.0e-3 solution drift between np = 1 and
//!    np = 2 on a 4x4 Q2 mesh), and a *ghost* dof can be missed locally (its
//!    column then survives the elimination; measured 9 un-eliminated LOR
//!    couplings at np = 4).  Run the flags through
//!    `ParVector::accumulate_ghosts` (ghost → owner) **and**
//!    `ParVector::update_ghosts` (owner → ghost), and eliminate the union —
//!    `crates/solver/tests/par_lor_h1.rs` and
//!    `miniapps/solvers/plor_solvers.rs` carry the recipe.
//!
//! # Status of the legs
//!
//! | leg | state |
//! |---|---|
//! | H¹ (`-fe h`) | **done**: `miniapps/solvers/plor_solvers.rs` runs at np = 1, 2, 4; the L2 error agrees with MFEM 4.10 to 4–6 significant digits and is bit-stable across np |
//! | L²/DG (`-fe l`) | not ported: fem-rs has no parallel DG face integrators |
//! | H(curl) / H(div) (`-fe n` / `-fe r`) | **blocked**: MFEM's `LORSolver<HypreAMS/HypreADS>` needs a *distributed* AMS/ADS.  fem-rs has the serial LOR-ND/RT kernels (`fem_assembly::lor_factory`, `LorNd`/`LorRt`) and linger has `AmsPrecond`/`AdsPrecond`, but neither offers a parallel auxiliary-space solve (`linger`'s `parallel_dist` ships `DistCsrMatrix` only), so there is no parallel preconditioner to build the leg on.  `plor_solvers -fe n\|r` reports this and exits instead of faking it. |
//!
//! # The distribution invariant of the per-rank LOR matrix
//!
//! The H¹ LOR dof map is a *bijection* between refined-mesh (LOR) dofs and
//! high-order dofs.  Conjugating the LOR matrix into the HO numbering
//! (`LorH1::ho_numbering`) therefore yields a matrix that can be permuted with
//! the **HO** [`DofPartition`](https://docs.rs/fem-parallel) and split into
//! `[owned | ghost]` exactly like the HO matrix, with the same halo.  This is
//! valid because ownership is derived from mesh entities in both discretisations:
//! a lattice point shared between ranks (a vertex, an edge point or a face
//! point) is owned by the rank owning that entity, and the refined-mesh twin of
//! that dof lies on the same entity.  Two consequences make the construction
//! self-checking, and both are gated by
//! `crates/solver/tests/par_lor_h1.rs`:
//!
//! 1. the LOR matrix at a given *global* dof is identical for np = 1, 2, 4;
//! 2. the LOR-preconditioned solution at a given global dof is identical for
//!    np = 1, 2, 4 (up to the AMG's rank-local coarsening).
//!
//! # Why the true residual matters
//!
//! linger's `ConjugateGradient` stops on its recurrence residual, which on
//! preconditioned LOR systems can fall below `rtol` while `‖b − A x‖/‖b‖` is
//! still orders of magnitude larger (the false convergence reported in D72).
//! [`solve_pcg_par_lor`] therefore (a) reports the residual computed from an
//! explicit `b − A x` and (b) restarts the inner PCG from that true residual,
//! exactly like the serial [`crate::solve_pcg_lor_amg`].

use crate::{SolveResult, SolverConfig, SolverError};

/// A distributed operator seen through the slices of its local vector.
///
/// `x` and `y` have the local length `n_total()` (owned entries followed by
/// ghost entries); only the first `n_owned()` entries of an operator's *output*
/// are meaningful — the rest are the halo values the next `spmv` needs.
pub trait ParOperator {
    /// Number of DOFs this rank owns.
    fn n_owned(&self) -> usize;
    /// Number of local DOFs (owned + ghost).
    fn n_total(&self) -> usize;
    /// True on rank 0 (keeps printing single-rooted).
    fn is_root(&self) -> bool;
    /// `y = A x` (halo-consistent: ghost entries of `y` need not be filled).
    fn spmv(&self, x: &[f64], y: &mut [f64]);
    /// MPI-reduced dot product (every owned entry counted once globally).
    fn global_dot(&self, x: &[f64], y: &[f64]) -> f64;
    /// MPI-reduced 2-norm.
    fn global_norm(&self, x: &[f64]) -> f64;
}

/// A distributed preconditioner: `z = M⁻¹ r`.
pub trait ParPrecond {
    /// Apply the preconditioner to the local vector `r`, writing `z`.
    fn apply_precond(&self, r: &[f64], z: &mut [f64]);
}

/// PCG for a distributed system, driven by the **true** residual.
///
/// `a` is the (already eliminated) system operator, `m` the LOR preconditioner,
/// `b` the right-hand side and `x` the initial guess — all local slices of
/// length `a.n_total()`.
///
/// The inner CG runs on the recurrence residual; the outer loop recomputes
/// `b − A x` from scratch and restarts the inner CG from it (at most
/// `MAX_ROUNDS` times, `cfg.max_iter` inner iterations in total).  The returned
/// [`SolveResult::final_residual`] is therefore a *measured* relative residual
/// of the returned `x`.
///
/// `on_iter(total_iterations, (B r, r), ‖r‖/‖b‖)` runs after every inner
/// iteration, mirroring MFEM's `CGSolver::Mult` print level 1 (the
/// `(B r, r) = …` line) with the true relative residual appended.
pub fn solve_pcg_par_lor<A, M>(
    a: &A,
    m: &M,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
    mut on_iter: impl FnMut(usize, f64, f64),
) -> Result<SolveResult, SolverError>
where
    A: ParOperator,
    M: ParPrecond,
{
    const MAX_ROUNDS: usize = 8;
    let n = a.n_total();
    if b.len() != n || x.len() != n {
        return Err(SolverError::DimensionMismatch { rows: n, cols: n, rhs: b.len() });
    }

    let tol = cfg.rtol.max(0.0);
    let atol = cfg.atol.max(0.0);
    let b_norm = a.global_norm(b);
    if b_norm <= 1e-300 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }
    let b_den = b_norm;

    let mut total_iters = 0usize;
    let mut converged = false;

    let mut ax = vec![0.0_f64; n];
    let mut r = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut p = vec![0.0_f64; n];
    let mut ap = vec![0.0_f64; n];
    let mut dx = vec![0.0_f64; n];

    for _round in 0..MAX_ROUNDS {
        a.spmv(x, &mut ax);
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
        let r_norm = a.global_norm(&r);
        if r_norm <= tol * b_den + atol {
            converged = true;
            break;
        }
        if total_iters >= cfg.max_iter {
            break;
        }

        // Inner PCG on A·dx = r, dx₀ = 0.
        dx.iter_mut().for_each(|v| *v = 0.0);
        m.apply_precond(&r, &mut z);
        let mut rz = a.global_dot(&r, &z);
        if !rz.is_finite() || rz <= 0.0 {
            break; // M⁻¹ is not SPD on this r; nothing left to do.
        }
        p.copy_from_slice(&z);
        loop {
            if total_iters >= cfg.max_iter {
                break;
            }
            a.spmv(&p, &mut ap);
            let pap = a.global_dot(&p, &ap);
            if !pap.is_finite() || pap <= 0.0 {
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                dx[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            total_iters += 1;
            let r_norm = a.global_norm(&r);
            on_iter(total_iters, rz, r_norm / b_den);
            if r_norm <= tol * b_den + atol {
                break;
            }
            m.apply_precond(&r, &mut z);
            let rz_new = a.global_dot(&r, &z);
            if !rz_new.is_finite() || rz_new <= 0.0 {
                break;
            }
            let beta = rz_new / rz;
            rz = rz_new;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }
        for i in 0..n {
            x[i] += dx[i];
        }
    }

    // Final measurement from an explicit residual: the reported number is the
    // true one even if the inner CG broke down early.
    a.spmv(x, &mut ax);
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
    let final_residual = a.global_norm(&r) / b_den;
    if final_residual <= tol + atol / b_den {
        converged = true;
    }

    Ok(SolveResult { converged, iterations: total_iters, final_residual })
}
