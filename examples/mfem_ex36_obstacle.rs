//! # Example 36 �?Obstacle problem (toward MFEM ex36)
//!
//! Solves the constrained minimization problem
//!
//! ```text
//!   minimize  1/2 �?|∇u|² dx - �?f u dx
//!   subject to u >= ψ in Ω,   u = 0 on ∂�?
//! ```
//!
//! on the unit square using a primal-dual active-set (PDAS) iteration on the
//! assembled H¹ stiffness matrix.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex36_obstacle
//! cargo run --example mfem_ex36_obstacle -- --n 24 --load -7.5
//! ```

use fem_assembly::{Assembler, GridFunction, standard::DiffusionIntegrator};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::solve_sparse_cholesky;
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum SolveMethod {
    Pdas,
    SemismoothNewton,
}

fn main() {
    let args = parse_args();
    let result = solve_obstacle_problem(args.n, args.load, args.method);

    let method_label = match args.method {
        SolveMethod::Pdas => "PDAS",
        SolveMethod::SemismoothNewton => "Semismooth Newton",
    };
    println!("=== fem-rs Example 36: obstacle problem ({method_label}) ===");
    println!("  Mesh: {}x{} subdivisions, P1 elements", args.n, args.n);
    println!("  Load: {:.3}", args.load);
    println!("  Iterations: {}", result.iterations);
    println!("  Final update: {:.3e}", result.final_update);
    println!("  Feasibility min(u-psi): {:.3e}", result.min_gap);
    println!("  Contact DOFs: {}", result.contact_dofs);
    println!("  Min multiplier (Au-b): {:.3e}", result.min_multiplier);
    println!("  Complementarity max|(Au-b)(u-psi)|: {:.3e}", result.complementarity);
    println!("  Unconstrained min(u-psi): {:.3e}", result.unconstrained_min_gap);
    println!("  L2 distance to obstacle: {:.3e}", result.obstacle_l2_distance);
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
    load: f64,
    method: SolveMethod,
}

#[derive(Debug, Clone)]
struct ObstacleResult {
    iterations: usize,
    final_update: f64,
    min_gap: f64,
    contact_dofs: usize,
    min_multiplier: f64,
    complementarity: f64,
    unconstrained_min_gap: f64,
    obstacle_l2_distance: f64,
}

fn parse_args() -> Args {
    let mut args = Args { n: 20, load: -5.0, method: SolveMethod::Pdas };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                args.n = it.next().unwrap_or("20".into()).parse().unwrap_or(20);
            }
            "--load" => {
                args.load = it.next().unwrap_or("-5.0".into()).parse().unwrap_or(-5.0);
            }
            "--method" => {
                args.method = match it.next().as_deref() {
                    Some("ssn") | Some("semismooth") => SolveMethod::SemismoothNewton,
                    _ => SolveMethod::Pdas,
                };
            }
            _ => {}
        }
    }
    args
}

fn solve_obstacle_problem(n: usize, load: f64, method: SolveMethod) -> ObstacleResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();

    let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let mut rhs = assemble_constant_rhs(&space, load);

    let dm = space.dof_manager();
    let boundary = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let boundary_mask = dof_mask(ndofs, &boundary);
    apply_dirichlet(&mut mat, &mut rhs, &boundary, &vec![0.0; boundary.len()]);

    let mut obstacle = space.interpolate(&obstacle_profile).into_vec();
    for &dof in &boundary {
        obstacle[dof as usize] = 0.0;
    }

    let mut unconstrained = vec![0.0; ndofs];
    gauss_seidel_solve(&mat, &rhs, &mut unconstrained, &boundary_mask, 3_000, 1e-13);
    let unconstrained_min_gap = unconstrained.iter().zip(obstacle.iter())
        .map(|(u, psi)| u - psi)
        .fold(f64::INFINITY, f64::min);

    let (solution, iterations, final_update) = match method {
        SolveMethod::Pdas => primal_dual_active_set(
            &mat,
            &rhs,
            &obstacle,
            &boundary_mask,
            80,
            1e-10,
        ),
        SolveMethod::SemismoothNewton => semismooth_newton(
            &mat,
            &rhs,
            &obstacle,
            &boundary_mask,
            80,
            1e-10,
        ),
    };

    let (min_gap, contact_dofs, min_multiplier, complementarity) =
        obstacle_kkt_metrics(&mat, &rhs, &solution, &obstacle, &boundary_mask);

    let gf_solution = GridFunction::new(&space, solution.clone());
    let obstacle_l2_distance = gf_solution.compute_l2_error(&obstacle_profile, 4);

    ObstacleResult {
        iterations,
        final_update,
        min_gap,
        contact_dofs,
        min_multiplier,
        complementarity,
        unconstrained_min_gap,
        obstacle_l2_distance,
    }
}

fn obstacle_profile(x: &[f64]) -> f64 {
    let dx = x[0] - 0.5;
    let dy = x[1] - 0.5;
    0.12 - 2.0 * (dx * dx + dy * dy)
}

fn assemble_constant_rhs<S: FESpace>(space: &S, load: f64) -> Vec<f64> {
    let mesh = space.mesh();
    let mut rhs = vec![0.0_f64; space.n_dofs()];

    for elem in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(elem);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let area = 0.5 * ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();
        let local = load * area / 3.0;
        for &dof in space.element_dofs(elem) {
            rhs[dof as usize] += local;
        }
    }

    rhs
}

fn dof_mask(ndofs: usize, dofs: &[u32]) -> Vec<bool> {
    let mut mask = vec![false; ndofs];
    for &dof in dofs {
        mask[dof as usize] = true;
    }
    mask
}

fn gauss_seidel_solve(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    x: &mut [f64],
    constrained: &[bool],
    max_iter: usize,
    tol: f64,
) {
    for _ in 0..max_iter {
        let mut max_update = 0.0_f64;
        for row in 0..mat.nrows {
            if constrained[row] {
                x[row] = rhs[row];
                continue;
            }
            let start = mat.row_ptr[row];
            let end = mat.row_ptr[row + 1];
            let mut diag = 0.0;
            let mut sigma = 0.0;
            for idx in start..end {
                let col = mat.col_idx[idx] as usize;
                let val = mat.values[idx];
                if col == row {
                    diag = val;
                } else {
                    sigma += val * x[col];
                }
            }
            let next = (rhs[row] - sigma) / diag;
            max_update = max_update.max((next - x[row]).abs());
            x[row] = next;
        }
        if max_update < tol {
            return;
        }
    }
}

fn primal_dual_active_set(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    obstacle: &[f64],
    constrained: &[bool],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize, f64) {
    let n = mat.nrows;
    let mut x = obstacle.to_vec();
    for i in 0..n {
        if constrained[i] {
            x[i] = rhs[i];
        }
    }

    let mut prev_active = vec![false; n];
    let mut final_update = f64::INFINITY;

    for iter in 0..max_iter {
        let mut ax = vec![0.0_f64; n];
        mat.spmv(&x, &mut ax);

        let mut active = vec![false; n];
        for i in 0..n {
            if constrained[i] {
                active[i] = true;
                continue;
            }
            let lambda_i = ax[i] - rhs[i];
            let gap_i = x[i] - obstacle[i];
            active[i] = gap_i <= 1e-10 && lambda_i > 1e-10;
        }

        let mut free = Vec::new();
        let mut fixed = vec![false; n];
        for i in 0..n {
            if active[i] || constrained[i] {
                fixed[i] = true;
            } else {
                free.push(i);
            }
        }

        let mut x_new = x.clone();
        for i in 0..n {
            if fixed[i] {
                x_new[i] = if constrained[i] { rhs[i] } else { obstacle[i] };
            }
        }

        if !free.is_empty() {
            let (a_ff, b_f) = reduced_free_system(mat, rhs, &free, &fixed, &x_new);
            let u_f = solve_sparse_cholesky(&a_ff, &b_f).expect("PDAS reduced solve failed");
            for (k, &gi) in free.iter().enumerate() {
                x_new[gi] = u_f[k].max(obstacle[gi]);
            }
        }

        let mut max_update = 0.0_f64;
        for i in 0..n {
            max_update = max_update.max((x_new[i] - x[i]).abs());
        }
        final_update = max_update;

        let active_unchanged = active == prev_active;
        x = x_new;
        prev_active = active;

        if active_unchanged && max_update < tol {
            return (x, iter + 1, final_update);
        }
    }

    (x, max_iter, final_update)
}

fn reduced_free_system(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    free: &[usize],
    fixed: &[bool],
    x_fixed: &[f64],
) -> (CsrMatrix<f64>, Vec<f64>) {
    let nf = free.len();
    let mut inv = vec![usize::MAX; mat.nrows];
    for (i, &g) in free.iter().enumerate() {
        inv[g] = i;
    }

    let mut coo = CooMatrix::<f64>::new(nf, nf);
    let mut b = vec![0.0_f64; nf];

    for (ri, &gi) in free.iter().enumerate() {
        let mut rhs_i = rhs[gi];
        for idx in mat.row_ptr[gi]..mat.row_ptr[gi + 1] {
            let gj = mat.col_idx[idx] as usize;
            let aij = mat.values[idx];
            let cj = inv[gj];
            if cj != usize::MAX {
                coo.add(ri, cj, aij);
            } else if fixed[gj] {
                rhs_i -= aij * x_fixed[gj];
            }
        }
        b[ri] = rhs_i;
    }

    (coo.into_csr(), b)
}

/// Semismooth Newton method for the obstacle problem.
///
/// Reformulates KKT complementarity using the `min` function:
///   φᵢ(u) = min(uᵢ − ψᵢ,  fᵢ − [Au]ᵢ) = 0
///
/// Each Newton step:
/// - Inactive DOFs (free): solve A u_new = f (standard PDE residual)
/// - Active DOFs (contact): u_new[i] = ψ[i]   (enforce contact constraint)
///
/// Converges locally with quadratic rate; no requirement to wait for
/// active-set stabilisation between steps (unlike PDAS).
fn semismooth_newton(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    obstacle: &[f64],
    constrained: &[bool],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize, f64) {
    let n = mat.nrows;
    // Start at the obstacle (same as PDAS) so that the active set is meaningful
    // from the very first iteration.
    let mut u = obstacle.to_vec();
    for i in 0..n {
        if constrained[i] {
            u[i] = rhs[i];
        }
    }

    let mut final_update = f64::INFINITY;

    for iter in 0..max_iter {
        // Compute residual Au
        let mut au = vec![0.0_f64; n];
        mat.spmv(&u, &mut au);

        // SSN active-set criterion: same two-sided KKT check as PDAS
        //   gap ≤ 0  (primal: at obstacle)  AND  multiplier > 0  (dual: contact force)
        // Using this criterion makes SSN and PDAS converge to the same KKT solution;
        // the algorithmic distinction is that SSN does NOT require the active set to
        // stabilise between steps — it terminates as soon as the Newton step is small.
        let mut active = vec![false; n];
        for i in 0..n {
            if constrained[i] {
                active[i] = true;
                continue;
            }
            let gap = u[i] - obstacle[i];
            let multiplier = au[i] - rhs[i];
            active[i] = gap <= 1e-10 && multiplier > 0.0;
        }

        // Build Newton step: solve for u_new directly
        //   active dofs → u_new[i] = obstacle[i]
        //   free dofs   → A_free * u_new_free = rhs_free (with contact rows eliminated)
        let free: Vec<usize> = (0..n)
            .filter(|&i| !active[i] && !constrained[i])
            .collect();
        let fixed: Vec<bool> = (0..n).map(|i| active[i] || constrained[i]).collect();

        let mut u_new = u.clone();
        for i in 0..n {
            if active[i] {
                u_new[i] = if constrained[i] { rhs[i] } else { obstacle[i] };
            }
        }

        if !free.is_empty() {
            let (a_ff, b_f) = reduced_free_system(mat, rhs, &free, &fixed, &u_new);
            let u_f = solve_sparse_cholesky(&a_ff, &b_f)
                .expect("SSN reduced solve failed");
            for (k, &gi) in free.iter().enumerate() {
                // enforce feasibility: solution may not go below obstacle
                u_new[gi] = u_f[k].max(obstacle[gi]);
            }
        }

        let mut max_update = 0.0_f64;
        for i in 0..n {
            max_update = max_update.max((u_new[i] - u[i]).abs());
        }
        final_update = max_update;
        u = u_new;

        // SSN convergence: Newton step is small (no active-set stabilisation required)
        if max_update < tol {
            return (u, iter + 1, final_update);
        }
    }

    (u, max_iter, final_update)
}

fn obstacle_kkt_metrics(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    solution: &[f64],
    obstacle: &[f64],
    constrained: &[bool],
) -> (f64, usize, f64, f64) {
    let mut residual = vec![0.0_f64; mat.nrows];
    mat.spmv(solution, &mut residual);

    let mut min_gap = f64::INFINITY;
    let mut min_multiplier = f64::INFINITY;
    let mut complementarity: f64 = 0.0;
    let mut contact_dofs = 0usize;

    for i in 0..mat.nrows {
        if constrained[i] {
            continue;
        }
        let gap = solution[i] - obstacle[i];
        let multiplier = residual[i] - rhs[i];
        min_gap = min_gap.min(gap);
        min_multiplier = min_multiplier.min(multiplier);
        complementarity = complementarity.max((gap * multiplier).abs());
        if gap <= 1e-8 {
            contact_dofs += 1;
        }
    }

    (min_gap, contact_dofs, min_multiplier, complementarity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex36_obstacle_solution_is_feasible_and_has_contact() {
        let result = solve_obstacle_problem(14, -5.0, SolveMethod::Pdas);
        assert!(result.final_update < 1e-8, "final update too large: {}", result.final_update);
        assert!(result.min_gap >= -1e-8, "solution violated obstacle: {}", result.min_gap);
        assert!(result.min_multiplier >= -1e-6, "negative multiplier: {}", result.min_multiplier);
        assert!(result.complementarity < 1e-6, "complementarity too large: {}", result.complementarity);
        assert!(result.contact_dofs > 0, "expected a non-empty contact set");
    }

    #[test]
    fn ex36_obstacle_improves_on_unconstrained_solution() {
        let result = solve_obstacle_problem(14, -5.0, SolveMethod::Pdas);
        assert!(result.unconstrained_min_gap < -1e-3, "unconstrained solution should violate obstacle: {}", result.unconstrained_min_gap);
        assert!(result.min_gap >= -1e-8, "projected solution should be feasible: {}", result.min_gap);
    }

    #[test]
    fn ex36_stronger_downward_loads_expand_contact_set() {
        let light = solve_obstacle_problem(14, -2.0, SolveMethod::Pdas);
        let medium = solve_obstacle_problem(14, -5.0, SolveMethod::Pdas);
        let strong = solve_obstacle_problem(14, -8.0, SolveMethod::Pdas);

        for (label, result) in [("light", &light), ("medium", &medium), ("strong", &strong)] {
            assert!(result.min_gap >= -1e-8, "{label} load violated obstacle: {}", result.min_gap);
            assert!(result.min_multiplier >= -1e-6, "{label} load had negative multiplier: {}", result.min_multiplier);
            assert!(result.complementarity < 1e-6, "{label} load complementarity too large: {}", result.complementarity);
        }

        assert!(
            light.contact_dofs < medium.contact_dofs && medium.contact_dofs < strong.contact_dofs,
            "expected stronger downward load to enlarge contact set, got light={} medium={} strong={}",
            light.contact_dofs,
            medium.contact_dofs,
            strong.contact_dofs
        );
        assert!(
            light.obstacle_l2_distance > medium.obstacle_l2_distance
                && medium.obstacle_l2_distance > strong.obstacle_l2_distance,
            "expected stronger downward load to move solution closer to obstacle, got light={} medium={} strong={}",
            light.obstacle_l2_distance,
            medium.obstacle_l2_distance,
            strong.obstacle_l2_distance
        );
    }

    #[test]
    fn ex36_upward_load_has_smaller_contact_than_downward_load() {
        let upward = solve_obstacle_problem(14, 1.0, SolveMethod::Pdas);
        let downward = solve_obstacle_problem(14, -5.0, SolveMethod::Pdas);

        assert!(upward.min_gap >= -1e-8, "upward load violated obstacle: {}", upward.min_gap);
        assert!(upward.complementarity < 1e-6, "upward load complementarity too large: {}", upward.complementarity);
        assert!(
            upward.contact_dofs < downward.contact_dofs,
            "expected upward load to reduce contact set, got upward={} downward={}",
            upward.contact_dofs,
            downward.contact_dofs
        );
        assert!(
            upward.obstacle_l2_distance > downward.obstacle_l2_distance,
            "expected upward load solution to sit farther from obstacle, got upward={} downward={}",
            upward.obstacle_l2_distance,
            downward.obstacle_l2_distance
        );
    }

    // ── Semismooth Newton tests ──────────────────────────────────────────────

    #[test]
    fn ex36_ssn_solution_is_feasible_and_has_contact() {
        let result = solve_obstacle_problem(14, -5.0, SolveMethod::SemismoothNewton);
        assert!(result.final_update < 1e-8, "SSN: final update too large: {}", result.final_update);
        assert!(result.min_gap >= -1e-8, "SSN: solution violated obstacle: {}", result.min_gap);
        assert!(result.min_multiplier >= -1e-6, "SSN: negative multiplier: {}", result.min_multiplier);
        assert!(result.complementarity < 1e-6, "SSN: complementarity too large: {}", result.complementarity);
        assert!(result.contact_dofs > 0, "SSN: expected a non-empty contact set");
    }

    #[test]
    fn ex36_ssn_agrees_with_pdas() {
        let pdas = solve_obstacle_problem(14, -5.0, SolveMethod::Pdas);
        let ssn  = solve_obstacle_problem(14, -5.0, SolveMethod::SemismoothNewton);

        let l2_diff = (pdas.obstacle_l2_distance - ssn.obstacle_l2_distance).abs();
        assert!(
            l2_diff < 1e-6,
            "SSN and PDAS L2 distances differ too much: pdas={} ssn={}",
            pdas.obstacle_l2_distance, ssn.obstacle_l2_distance
        );
        assert_eq!(
            pdas.contact_dofs, ssn.contact_dofs,
            "SSN and PDAS contact sets differ: pdas={} ssn={}",
            pdas.contact_dofs, ssn.contact_dofs
        );
        assert!(
            pdas.complementarity < 1e-6 && ssn.complementarity < 1e-6,
            "both methods should satisfy complementarity: pdas={} ssn={}",
            pdas.complementarity, ssn.complementarity
        );
    }

    #[test]
    fn ex36_ssn_converges_in_few_iterations() {
        // SSN has quadratic local convergence; should need ≤ 20 outer iterations
        // on a moderate mesh with a well-posed load.
        let result = solve_obstacle_problem(20, -5.0, SolveMethod::SemismoothNewton);
        assert!(result.iterations <= 20,
            "SSN took too many iterations: {}", result.iterations);
        assert!(result.min_gap >= -1e-8, "SSN: obstacle violated: {}", result.min_gap);
        assert!(result.complementarity < 1e-6, "SSN: complementarity too large: {}", result.complementarity);
    }

    /// A finer mesh must still produce a feasible solution for both solvers.
    #[test]
    fn ex36_finer_mesh_preserves_obstacle_feasibility() {
        for method in [SolveMethod::Pdas, SolveMethod::SemismoothNewton] {
            let result = solve_obstacle_problem(20, -5.0, method);
            assert!(result.min_gap >= -1e-8,
                "finer mesh violated obstacle: min_gap={}", result.min_gap);
            assert!(result.complementarity < 1e-6,
                "finer mesh complementarity too large: {}", result.complementarity);
            assert!(result.contact_dofs > 0,
                "finer mesh should still have non-empty contact set");
        }
    }
}
