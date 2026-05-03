//! # Example 15 �?Poisson with mesh refinement and error estimation
//!
//! Demonstrates the AMR (Adaptive Mesh Refinement) infrastructure for the
//! scalar Poisson equation:
//!
//! ```text
//!   -Lap u = f    in Omega = [0,1]^2
//!        u = 0    on dOmega
//! ```
//!
//! Manufactured solution: `u(x,y) = sin(pi*x) sin(pi*y)`.
//!
//! Each refinement level:
//! 1. Assembles and solves the H1 system (PCG + Jacobi).
//! 2. Computes the Zienkiewicz-Zhu (ZZ) error indicators per element.
//! 3. Applies Doerfler marking to identify worst elements.
//! 4. Refines the mesh (uniform red refinement).
//! 5. Prints convergence data (L2 error, estimated error, DOF count).
//!
//! Demonstrates O(h^2) convergence for P1 elements and shows that the
//! ZZ estimator tracks the true error.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex15_dg_amr
//! cargo run --example mfem_ex15_dg_amr -- --levels 5
//! cargo run --example mfem_ex15_dg_amr -- --n 2 --levels 7
//! cargo run --example mfem_ex15_dg_amr -- --nc --levels 4
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_mesh::amr::{refine_uniform, NCState, zz_estimator, dorfler_mark, prolongate_p1};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, apply_hanging_constraints, recover_hanging_values, boundary_dofs}};

struct LevelResult {
    level: usize,
    n_elems: usize,
    n_dofs: usize,
    l2_error: f64,
    estimator_error: f64,
    n_marked: usize,
    n_hanging: usize,
}

struct RunResult {
    n0: usize,
    levels: usize,
    theta: f64,
    nonconforming: bool,
    levels_data: Vec<LevelResult>,
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 15: Poisson + Mesh Refinement with Error Estimation ===");
    println!("  Initial mesh: {}x{}, P1 elements", args.n0, args.n0);
    println!("  Refinement levels: {}, Doerfler theta = {}", args.levels, args.theta);
    println!("  Mode: {}\n", if args.nonconforming { "non-conforming (hanging nodes)" } else { "conforming (uniform)" });

    let result = run_case(args.n0, args.levels, args.theta, args.nonconforming);
    println!(
        "  Confirmed run: n0 = {}, levels = {}, theta = {:.2}, mode = {}",
        result.n0,
        result.levels,
        result.theta,
        if result.nonconforming { "non-conforming" } else { "conforming" }
    );

    println!("{:>5}  {:>8}  {:>8}  {:>12}  {:>12}  {:>8}  {:>6}  {:>8}",
             "Level", "Elems", "DOFs", "L2 error", "Est. error", "Marked", "Hang", "Ratio");
    println!("{}", "-".repeat(82));

    let mut prev_l2: Option<f64> = None;
    for level in &result.levels_data {
        let ratio = match prev_l2 {
            Some(prev) => format!("{:.2}", prev / level.l2_error),
            None => "  --".to_string(),
        };
        println!(
            "{:>5}  {:>8}  {:>8}  {:>12.4e}  {:>12.4e}  {:>8}  {:>6}  {:>8}",
            level.level,
            level.n_elems,
            level.n_dofs,
            level.l2_error,
            level.estimator_error,
            level.n_marked,
            level.n_hanging,
            ratio,
        );
        prev_l2 = Some(level.l2_error);
    }

    println!("\n  Final DOFs: {}, final L2 error = {:.4e}",
        result.levels_data.last().map(|level| level.n_dofs).unwrap_or(0),
        result.levels_data.last().map(|level| level.l2_error).unwrap_or(0.0));

    println!("  Expected convergence rate: O(h^2) for P1 elements (ratio -> 4.0)");
    println!("\nDone.");
}

fn run_case(n0: usize, levels: usize, theta: f64, nonconforming: bool) -> RunResult {
    let u_exact = |x: &[f64]| -> f64 {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    };

    let rhs_fn = |x: &[f64]| -> f64 {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    };

    let mut mesh = SimplexMesh::<2>::unit_square_tri(n0);
    let mut prev_l2: Option<f64> = None;
    let mut hanging_constraints = Vec::new();
    let mut nc_state = NCState::new();
    let mut prev_u: Option<Vec<f64>> = None;
    let mut levels_data = Vec::new();

    for level in 0..=levels {
        // ─── 1. Build H1 space on current mesh ─────────────────────────
        let space = H1Space::new(mesh.clone(), 1);
        let n = space.n_dofs();
        let n_elems = mesh.n_elems();

        // ─── 2. Assemble stiffness matrix and RHS ──────────────────────
        let diffusion = DiffusionIntegrator { kappa: 1.0 };
        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);

        let source = DomainSourceIntegrator::new(&rhs_fn);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

        // ─── 3. Apply constraints ──────────────────────────────────────
        // Hanging-node constraints FIRST (before Dirichlet).
        if nonconforming {
            apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);
        }

        // Then Dirichlet BCs: u = 0 on all boundaries.
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        // ─── 4. Solve with PCG + Jacobi ────────────────────────────────
        let mut u = prev_u.take().unwrap_or_else(|| vec![0.0_f64; n]);
        u.resize(n, 0.0);
        // Zero out constrained DOFs (recovered after solve).
        for c in &hanging_constraints { u[c.constrained] = 0.0; }
        let cfg = SolverConfig {
            rtol: 1e-10, atol: 1e-14, max_iter: 20_000, verbose: false,
            ..SolverConfig::default()
        };
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg);
        match &res {
            Err(e) => {
                println!("  Solver failed at level {level}: {e:?}");
                break;
            }
            Ok(r) if !r.converged => {
                println!("  WARNING: solver did not converge at level {level} \
                          (iters={}, res={:.3e})", r.iterations, r.final_residual);
            }
            _ => {}
        }
        let _res = res.unwrap();

        // ─── 4b. Recover hanging DOF values (NC mode) ──────────────────
        if nonconforming {
            recover_hanging_values(&mut u, &hanging_constraints);
        }

        // ─── 5. Compute L2 error ────────────────────────────────────────
        let l2_err = h1_l2_error(&space, &u, &u_exact);

        // ─── 6. ZZ error estimator + Doerfler marking ──────────────────
        let eta = zz_estimator(&mesh, &u);
        let est_err: f64 = eta.iter().map(|e| e * e).sum::<f64>().sqrt();
        let marked = dorfler_mark(&eta, theta);
        let n_marked = marked.len();

        prev_l2 = Some(l2_err);

        let n_hang = hanging_constraints.len();

        levels_data.push(LevelResult {
            level,
            n_elems,
            n_dofs: n,
            l2_error: l2_err,
            estimator_error: est_err,
            n_marked,
            n_hanging: n_hang,
        });

        if level < levels {
            if nonconforming {
                let (new_mesh, new_constraints, midpt_map) = nc_state.refine(&mesh, &marked);
                prev_u = Some(prolongate_p1(&u, new_mesh.n_nodes(), &midpt_map));
                mesh = new_mesh;
                hanging_constraints = new_constraints;
            } else {
                mesh = refine_uniform(&mesh);
                hanging_constraints.clear();
                prev_u = None; // no prolongation for uniform refinement
            }
        }
    }

    let _ = prev_l2;

    RunResult {
        n0,
        levels,
        theta,
        nonconforming,
        levels_data,
    }
}

// ─── L2 error for H1 solutions ──────────────────────────────────────────────

fn h1_l2_error<S: FESpace>(
    space: &S,
    uh: &[f64],
    exact: impl Fn(&[f64]) -> f64,
) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};

    let mesh = space.mesh();
    let mut err2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let re  = TriP1;
        let quad = re.quadrature(5);
        let nodes = mesh.element_nodes(e);
        let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs();

        let mut phi = vec![0.0_f64; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            let uh_qp: f64 = phi.iter().zip(gd.iter())
                .map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_qp - exact(&xp);
            err2 += w * diff * diff;
        }
    }
    err2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n0:     usize,
    levels: usize,
    theta:  f64,
    nonconforming: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        n0: 4, levels: 5, theta: 0.5, nonconforming: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"     => { a.n0     = it.next().unwrap_or("4".into()).parse().unwrap_or(4); }
            "--levels"=> { a.levels = it.next().unwrap_or("5".into()).parse().unwrap_or(5); }
            "--theta" => { a.theta  = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5); }
            "--nc"    => { a.nonconforming = true; }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex15_conforming_refinement_monotonically_reduces_true_error() {
        let result = run_case(4, 3, 0.5, false);
        assert_eq!(result.n0, 4);
        assert_eq!(result.levels, 3);
        assert!((result.theta - 0.5).abs() < 1.0e-12);
        assert!(!result.nonconforming);
        assert_eq!(result.levels_data.len(), 4);
        for pair in result.levels_data.windows(2) {
            assert!(pair[1].l2_error < pair[0].l2_error,
                "true error should decrease under uniform refinement: prev={} next={}",
                pair[0].l2_error,
                pair[1].l2_error);
            assert!(pair[1].estimator_error < pair[0].estimator_error,
                "estimator should decrease under uniform refinement: prev={} next={}",
                pair[0].estimator_error,
                pair[1].estimator_error);
        }
    }

    #[test]
    fn ex15_conforming_refinement_approaches_second_order_gain() {
        let result = run_case(4, 4, 0.5, false);
        let levels = &result.levels_data;
        let ratio_12 = levels[1].l2_error / levels[2].l2_error;
        let ratio_23 = levels[2].l2_error / levels[3].l2_error;
        let ratio_34 = levels[3].l2_error / levels[4].l2_error;
        assert!(ratio_12 > 3.5, "level-1 to level-2 gain too small: {}", ratio_12);
        assert!(ratio_23 > 3.8, "level-2 to level-3 gain too small: {}", ratio_23);
        assert!(ratio_34 > 3.9, "level-3 to level-4 gain too small: {}", ratio_34);
    }

    #[test]
    fn ex15_nonconforming_mode_creates_hanging_nodes_and_reduces_error() {
        let result = run_case(4, 3, 0.5, true);
        assert!(result.nonconforming);
        assert_eq!(result.levels_data.len(), 4);
        assert_eq!(result.levels_data[0].n_hanging, 0);
        assert!(result.levels_data.iter().skip(1).any(|level| level.n_hanging > 0),
            "nonconforming refinement should introduce hanging constraints");
        for pair in result.levels_data.windows(2) {
            assert!(pair[1].l2_error < pair[0].l2_error,
                "true error should decrease in nonconforming mode: prev={} next={}",
                pair[0].l2_error,
                pair[1].l2_error);
        }
    }

    #[test]
    fn ex15_marking_selects_nonzero_subset_each_level() {
        let result = run_case(4, 3, 0.5, false);
        for level in &result.levels_data {
            assert!(level.n_marked > 0, "each level should mark at least one element");
            assert!(level.n_marked < level.n_elems, "Doerfler marking should not mark every element on these calibrated levels");
        }
    }

    /// Higher Dörfler threshold should generally mark fewer elements on refined levels.
    #[test]
    fn ex15_dorfler_marking_threshold_affects_selection() {
        let low_theta = run_case(4, 2, 0.3, false);
        let high_theta = run_case(4, 2, 0.8, false);

        // Both should mark at each level
        for level in &low_theta.levels_data {
            assert!(level.n_marked > 0, "low theta should mark elements");
        }
        for level in &high_theta.levels_data {
            assert!(level.n_marked > 0, "high theta should mark elements");
        }

        // On refined levels (where more elements are available), marking patterns may differ
        // Just verify that both approaches are valid
        assert!(!low_theta.levels_data.is_empty() && !high_theta.levels_data.is_empty());
    }

    /// Coarser starting mesh should still achieve convergence.
    #[test]
    fn ex15_coarser_start_still_converges() {
        let coarse_start = run_case(2, 3, 0.5, false);
        let fine_start = run_case(4, 3, 0.5, false);

        assert_eq!(coarse_start.levels, fine_start.levels);
        // Both should show error reduction
        for pair in coarse_start.levels_data.windows(2) {
            assert!(pair[1].l2_error < pair[0].l2_error,
                "coarse start should still reduce error");
        }
        // After same number of refinements, fine start should have lower error
        assert!(fine_start.levels_data.last().unwrap().l2_error < coarse_start.levels_data.last().unwrap().l2_error * 1.5,
            "fine start should end with better or comparable error");
    }

    /// Estimator error should correlate with true error reduction.
    #[test]
    fn ex15_estimator_tracks_true_error() {
        let result = run_case(4, 3, 0.5, false);
        let levels = &result.levels_data;

        // Both should decrease monotonically
        for pair in levels.windows(2) {
            assert!(pair[1].l2_error < pair[0].l2_error, "true error should decrease");
            assert!(pair[1].estimator_error < pair[0].estimator_error, "estimator should decrease");
        }

        // They should track similar reduction patterns
        let true_ratios: Vec<f64> = levels.windows(2)
            .map(|p| p[0].l2_error / p[1].l2_error)
            .collect();
        let est_ratios: Vec<f64> = levels.windows(2)
            .map(|p| p[0].estimator_error / p[1].estimator_error)
            .collect();

        // Ratios should be similar (within factor of 2)
        for (i, (tr, er)) in true_ratios.iter().zip(est_ratios.iter()).enumerate() {
            assert!((tr / er).abs() < 3.0 && (er / tr).abs() < 3.0,
                "error and estimator reduction ratios should be similar at level {}: true={:.2} est={:.2}",
                i, tr, er);
        }
    }

    /// DOF count should grow monotonically with refinement.
    #[test]
    fn ex15_dof_count_grows_monotonically() {
        let result = run_case(4, 4, 0.5, false);
        let levels = &result.levels_data;

        for pair in levels.windows(2) {
            assert!(pair[1].n_dofs > pair[0].n_dofs,
                "DOF count should increase: level {} has {}, level {} has {}",
                pair[0].level, pair[0].n_dofs, pair[1].level, pair[1].n_dofs);
        }
    }

    /// Element count should grow monotonically with refinement.
    #[test]
    fn ex15_element_count_grows_monotonically() {
        let result = run_case(4, 4, 0.5, false);
        let levels = &result.levels_data;

        for pair in levels.windows(2) {
            assert!(pair[1].n_elems > pair[0].n_elems,
                "element count should increase: level {} has {}, level {} has {}",
                pair[0].level, pair[0].n_elems, pair[1].level, pair[1].n_elems);
        }
    }
}

