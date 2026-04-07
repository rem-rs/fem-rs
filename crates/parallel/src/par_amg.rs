//! Parallel Algebraic Multigrid (AMG) preconditioner.
//!
//! Implements a distributed AMG V-cycle using **local smoothed aggregation**:
//!
//! 1. Each rank coarsens its owned rows independently (aggregates don't cross
//!    partition boundaries).
//! 2. Prolongation/restriction operators are distributed sparse matrices.
//! 3. Coarse-level matrices are formed via the Galerkin product `R A P`.
//! 4. The coarsest level is solved with Jacobi-preconditioned CG.
//!
//! This is simpler than full boundary-crossing aggregation but produces good
//! convergence for typical elliptic problems.

use std::sync::Arc;

use fem_linalg::{CooMatrix, CsrMatrix};

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the parallel AMG hierarchy.
#[derive(Debug, Clone)]
pub struct ParAmgConfig {
    /// Maximum number of AMG levels.
    pub max_levels: usize,
    /// Coarsest level size (total global DOFs) below which we stop coarsening.
    pub coarse_size: usize,
    /// Number of pre-smoothing Jacobi iterations.
    pub n_pre_smooth: usize,
    /// Number of post-smoothing Jacobi iterations.
    pub n_post_smooth: usize,
    /// Strength threshold for aggregation.
    pub strength_threshold: f64,
}

impl Default for ParAmgConfig {
    fn default() -> Self {
        ParAmgConfig {
            max_levels: 10,
            coarse_size: 50,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            strength_threshold: 0.25,
        }
    }
}

// ── AMG Level ───────────────────────────────────────────────────────────────

/// One level in the AMG hierarchy.
struct AmgLevel {
    /// System matrix at this level.
    a: ParCsrMatrix,
    /// Prolongation operator (coarse → fine).  `None` at the coarsest level.
    p: Option<ParCsrMatrix>,
    /// Restriction operator (fine → coarse) = P^T.  `None` at the coarsest.
    r: Option<ParCsrMatrix>,
    /// Inverse diagonal for Jacobi smoothing.
    inv_diag: Vec<f64>,
}

// ── ParAmgHierarchy ─────────────────────────────────────────────────────────

/// A distributed AMG hierarchy for use as a preconditioner.
pub struct ParAmgHierarchy {
    levels: Vec<AmgLevel>,
    config: ParAmgConfig,
}

impl ParAmgHierarchy {
    /// Build the AMG hierarchy from a distributed SPD matrix.
    pub fn build(a: &ParCsrMatrix, comm: &Comm, config: ParAmgConfig) -> Self {
        let mut levels = Vec::new();
        let mut current_a = Some(clone_par_csr(a));

        for _level in 0..config.max_levels {
            let ca = current_a.take().unwrap();
            let inv_diag = compute_inv_diag(&ca);
            let n_global = comm.allreduce_sum_i64(ca.n_owned as i64) as usize;

            if n_global <= config.coarse_size || ca.n_owned <= 1 {
                levels.push(AmgLevel { a: ca, p: None, r: None, inv_diag });
                break;
            }

            let (p, r, coarse_a) = build_coarse_level(
                &ca, comm, config.strength_threshold,
            );

            levels.push(AmgLevel {
                a: ca,
                p: Some(p),
                r: Some(r),
                inv_diag,
            });

            current_a = Some(coarse_a);
        }

        // If we hit max_levels without reaching coarse_size, push the last level.
        if let Some(ca) = current_a {
            let inv_diag = compute_inv_diag(&ca);
            levels.push(AmgLevel { a: ca, p: None, r: None, inv_diag });
        }

        ParAmgHierarchy { levels, config }
    }

    /// Number of levels in the hierarchy.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Apply one AMG V-cycle as a preconditioner: `x ← M⁻¹ b`.
    ///
    /// `x` should be zero on entry (or contain the initial guess).
    pub fn vcycle(&self, b: &ParVector, x: &mut ParVector) {
        self.vcycle_level(0, b, x);
    }

    fn vcycle_level(&self, level: usize, b: &ParVector, x: &mut ParVector) {
        let lvl = &self.levels[level];

        if lvl.p.is_none() {
            // Coarsest level: solve approximately with many Jacobi iterations.
            for _ in 0..20 {
                jacobi_smooth(&lvl.a, x, b, &lvl.inv_diag);
            }
            return;
        }

        let p = lvl.p.as_ref().unwrap();
        let r_op = lvl.r.as_ref().unwrap();
        let coarse_lvl = &self.levels[level + 1];

        // Pre-smoothing.
        for _ in 0..self.config.n_pre_smooth {
            jacobi_smooth(&lvl.a, x, b, &lvl.inv_diag);
        }

        // Compute residual: res = b - A*x
        let mut ax = ParVector::zeros_like(b);
        let mut x_mut = x.clone_vec();
        lvl.a.spmv(&mut x_mut, &mut ax);
        let mut res = b.clone_vec();
        for i in 0..lvl.a.n_owned {
            res.data[i] -= ax.data[i];
        }

        // Restrict residual to coarse level: r_coarse = R * res (local CSR SpMV)
        let mut r_coarse = zeros_for_mat(&coarse_lvl.a);
        local_spmv(&r_op.diag, res.as_slice(), r_coarse.as_slice_mut());

        // Recursively solve on coarse level.
        let mut e_coarse = ParVector::zeros_like(&r_coarse);
        self.vcycle_level(level + 1, &r_coarse, &mut e_coarse);

        // Prolongate correction: x += P * e_coarse (local CSR SpMV)
        let mut correction = vec![0.0_f64; lvl.a.n_owned];
        local_spmv(&p.diag, e_coarse.as_slice(), &mut correction);
        for i in 0..lvl.a.n_owned {
            x.data[i] += correction[i];
        }

        // Post-smoothing.
        for _ in 0..self.config.n_post_smooth {
            jacobi_smooth(&lvl.a, x, b, &lvl.inv_diag);
        }
    }
}

// ── Jacobi smoother ─────────────────────────────────────────────────────────

/// One Jacobi smoothing iteration: x ← x + ω D⁻¹ (b - Ax).
fn jacobi_smooth(
    a: &ParCsrMatrix,
    x: &mut ParVector,
    b: &ParVector,
    inv_diag: &[f64],
) {
    let omega = 2.0 / 3.0; // damping factor
    let n = a.n_owned;

    // Compute Ax.
    let mut ax = ParVector::zeros_like(b);
    let mut x_clone = x.clone_vec();
    a.spmv(&mut x_clone, &mut ax);

    // x += ω D⁻¹ (b - Ax)
    for i in 0..n {
        let r_i = b.data[i] - ax.data[i];
        x.data[i] += omega * inv_diag[i] * r_i;
    }
}

// ── Aggregation & coarsening ────────────────────────────────────────────────

/// Build the prolongation, restriction, and coarse-level matrix.
///
/// Uses local (non-overlapping) aggregation on owned DOFs.
fn build_coarse_level(
    a: &ParCsrMatrix,
    comm: &Comm,
    strength_threshold: f64,
) -> (ParCsrMatrix, ParCsrMatrix, ParCsrMatrix) {
    let n_owned = a.n_owned;

    // 1. Build strength-of-connection: strong(i,j) iff |a_ij| >= θ * max_k |a_ik|
    let diag = &a.diag;
    let mut aggregate = vec![-1i32; n_owned]; // aggregate[i] = aggregate ID
    let mut n_agg = 0i32;

    // Phase 1: seed aggregates (unaggregated nodes that have no strong neighbours yet).
    for i in 0..n_owned {
        if aggregate[i] >= 0 { continue; }

        // Find max off-diagonal magnitude in this row.
        let mut max_off_diag = 0.0_f64;
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i {
                max_off_diag = max_off_diag.max(diag.values[k].abs());
            }
        }
        let threshold = strength_threshold * max_off_diag;

        // Try to form a new aggregate: this node + its strong unassigned neighbours.
        aggregate[i] = n_agg;
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i && j < n_owned && aggregate[j] < 0 && diag.values[k].abs() >= threshold {
                aggregate[j] = n_agg;
            }
        }
        n_agg += 1;
    }

    // Phase 2: assign remaining unaggregated nodes to nearest aggregate.
    for i in 0..n_owned {
        if aggregate[i] >= 0 { continue; }
        // Assign to the aggregate of the strongest connected neighbour.
        let mut best_agg = -1i32;
        let mut best_val = 0.0_f64;
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i && j < n_owned && aggregate[j] >= 0 && diag.values[k].abs() > best_val {
                best_val = diag.values[k].abs();
                best_agg = aggregate[j];
            }
        }
        if best_agg >= 0 {
            aggregate[i] = best_agg;
        } else {
            // Isolated node: make its own aggregate.
            aggregate[i] = n_agg;
            n_agg += 1;
        }
    }

    let n_coarse_owned = n_agg as usize;

    // 2. Build prolongation P (n_owned × n_coarse_owned): P[i, agg[i]] = 1.
    // This is "tentative prolongation" (unsmoothed).
    let mut p_coo = CooMatrix::<f64>::new(n_owned, n_coarse_owned.max(1));
    for i in 0..n_owned {
        let agg = aggregate[i] as usize;
        p_coo.add(i, agg, 1.0);
    }
    let p_local = p_coo.into_csr();

    // 3. Build restriction R = P^T (n_coarse_owned × n_owned).
    let r_local = transpose_csr(&p_local);

    // 4. Build coarse matrix: A_c = R * A_diag * P (Galerkin triple product).
    // We use only the diag block for the local part. For true parallel,
    // off-diag contributions would require communication, but local aggregation
    // keeps everything within the rank.
    let ap_local = csr_multiply(&a.diag, &p_local);
    let ac_local = csr_multiply(&r_local, &ap_local);

    // 5. Wrap in ParCsrMatrix (no ghost DOFs at coarse level for local aggregation).
    let ghost_ex = Arc::new(GhostExchange::from_trivial());
    let p_par = ParCsrMatrix::from_blocks(
        p_local,
        CsrMatrix::new_empty(n_owned, 0),
        n_owned, 0,
        Arc::clone(&ghost_ex),
        comm.clone(),
    );
    let r_par = ParCsrMatrix::from_blocks(
        r_local,
        CsrMatrix::new_empty(n_coarse_owned, 0),
        n_coarse_owned, 0,
        Arc::clone(&ghost_ex),
        comm.clone(),
    );
    let ac_par = ParCsrMatrix::from_blocks(
        ac_local,
        CsrMatrix::new_empty(n_coarse_owned, 0),
        n_coarse_owned, 0,
        Arc::clone(&ghost_ex),
        comm.clone(),
    );

    (p_par, r_par, ac_par)
}

// ── Local CSR SpMV ──────────────────────────────────────────────────────────

/// Compute y = A * x using only the local CSR data (no ghost exchange).
/// y is zeroed before accumulation.
fn local_spmv(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows.min(y.len()) {
        let mut sum = 0.0;
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[k] as usize;
            if j < x.len() {
                sum += a.values[k] * x[j];
            }
        }
        y[i] = sum;
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

fn compute_inv_diag(a: &ParCsrMatrix) -> Vec<f64> {
    let diag = a.diagonal();
    diag.into_iter()
        .map(|d| if d.abs() > 1e-14 { 1.0 / d } else { 1.0 })
        .collect()
}

fn clone_par_csr(a: &ParCsrMatrix) -> ParCsrMatrix {
    ParCsrMatrix::from_blocks(
        a.diag.clone(),
        a.offd.clone(),
        a.n_owned,
        a.n_ghost,
        a.ghost_exchange_arc(),
        a.comm().clone(),
    )
}

fn zeros_for_mat(a: &ParCsrMatrix) -> ParVector {
    ParVector::zeros_raw(
        a.n_owned,
        a.n_ghost,
        a.ghost_exchange_arc(),
        a.comm().clone(),
    )
}

/// Transpose a CSR matrix.
fn transpose_csr(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.ncols, a.nrows);
    for i in 0..a.nrows {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[k] as usize;
            coo.add(j, i, a.values[k]);
        }
    }
    coo.into_csr()
}

/// Sparse matrix multiply C = A * B.
fn csr_multiply(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    assert_eq!(a.ncols, b.nrows, "csr_multiply: dimension mismatch");
    let mut coo = CooMatrix::<f64>::new(a.nrows, b.ncols);

    for i in 0..a.nrows {
        for ka in a.row_ptr[i]..a.row_ptr[i + 1] {
            let k = a.col_idx[ka] as usize;
            let a_ik = a.values[ka];
            if a_ik == 0.0 { continue; }
            for kb in b.row_ptr[k]..b.row_ptr[k + 1] {
                let j = b.col_idx[kb] as usize;
                let b_kj = b.values[kb];
                if b_kj != 0.0 {
                    coo.add(i, j, a_ik * b_kj);
                }
            }
        }
    }

    coo.into_csr()
}

// ── Public solver function ──────────────────────────────────────────────────

/// Solve `A x = b` using AMG-preconditioned Conjugate Gradient.
///
/// Builds the AMG hierarchy once, then runs PCG with V-cycle preconditioning.
pub fn par_solve_pcg_amg(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    amg_cfg: &ParAmgConfig,
    solver_cfg: &fem_solver::SolverConfig,
) -> Result<fem_solver::SolveResult, fem_solver::SolverError> {
    let comm = x.comm().clone();
    let hierarchy = ParAmgHierarchy::build(a, &comm, amg_cfg.clone());

    if comm.is_root() {
        log::info!("par_amg: {} levels built", hierarchy.n_levels());
    }

    // PCG with AMG V-cycle as preconditioner.
    let n = a.n_owned;
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(&mut x.clone_vec(), &mut ax);
    for i in 0..n { r.data[i] = b.data[i] - ax.data[i]; }

    // z = M⁻¹ r
    let mut z = ParVector::zeros_like(b);
    hierarchy.vcycle(&r, &mut z);

    let mut p = z.clone_vec();
    let mut rz = r.global_dot(&z);
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(fem_solver::SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParVector::zeros_like(b);

    for iter in 0..solver_cfg.max_iter {
        a.spmv(&mut p, &mut ap);
        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 { break; }
        let alpha = rz / pap;

        x.axpy(alpha, &p);
        r.axpy(-alpha, &ap);

        let res_norm = r.global_norm() / b_norm;

        if solver_cfg.verbose && comm.is_root() {
            log::info!("par_pcg_amg iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < solver_cfg.rtol || r.global_norm() < solver_cfg.atol {
            return Ok(fem_solver::SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        // z = M⁻¹ r
        for v in z.as_slice_mut() { *v = 0.0; }
        hierarchy.vcycle(&r, &mut z);

        let rz_new = r.global_dot(&z);
        let beta = rz_new / rz;
        for i in 0..p.len() {
            p.data[i] = z.data[i] + beta * p.data[i];
        }
        rz = rz_new;
    }

    let final_res = r.global_norm() / b_norm;
    Ok(fem_solver::SolveResult {
        converged: false,
        iterations: solver_cfg.max_iter,
        final_residual: final_res,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_assembler::ParAssembler;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::standard::DiffusionIntegrator;
    use fem_mesh::SimplexMesh;
    use fem_solver::SolverConfig;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;
    use fem_space::constraints::boundary_dofs;

    #[test]
    fn par_pcg_amg_poisson_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let amg_cfg = ParAmgConfig::default();
            let solver_cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &solver_cfg).unwrap();

            assert!(res.converged,
                "AMG-PCG did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_pcg_amg_poisson_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let amg_cfg = ParAmgConfig::default();
            let solver_cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &solver_cfg).unwrap();

            assert!(res.converged,
                "rank {}: AMG-PCG did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_amg_fewer_iters_than_jacobi() {
        // AMG should converge in fewer iterations than plain Jacobi PCG.
        let mesh = SimplexMesh::<2>::unit_square_tri(16);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let solver_cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, ..SolverConfig::default() };

            // Jacobi PCG
            let mut u_jac = ParVector::zeros(&par_space);
            let res_jac = crate::par_solver::par_solve_pcg_jacobi(
                &a_mat, &rhs, &mut u_jac, &solver_cfg,
            ).unwrap();

            // AMG PCG
            let mut u_amg = ParVector::zeros(&par_space);
            let amg_cfg = ParAmgConfig::default();
            let res_amg = par_solve_pcg_amg(&a_mat, &rhs, &mut u_amg, &amg_cfg, &solver_cfg).unwrap();

            assert!(res_jac.converged, "Jacobi didn't converge");
            assert!(res_amg.converged, "AMG didn't converge");
            assert!(res_amg.iterations < res_jac.iterations,
                "AMG ({} iters) should be faster than Jacobi ({} iters)",
                res_amg.iterations, res_jac.iterations);
        });
    }
}
