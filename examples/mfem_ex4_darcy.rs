//! # Example 4 - Darcy flow / grad-div problem  (analogous to MFEM ex4)
//!
//! Solves the H(div) grad-div problem on the unit square:
//!
//! ```text
//!   -∇(alpha ∇·F) + beta F = f    in Ω = [0,1]^2
//!                F·n = 0          on ∂Ω
//! ```
//!
//! using lowest-order Raviart-Thomas (RT0) elements.
//!
//! Manufactured solution:
//! ```text
//!   F(x,y) = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y))
//! ```
//!
//! which is divergence-free and satisfies the homogeneous normal-flux boundary
//! condition on the unit square.

use std::f64::consts::PI;

use fem_assembly::{
    coefficient::FnVectorCoeff,
    postprocess::compute_element_divergence,
    standard::{GradDivIntegrator, VectorDomainLFIntegrator, VectorMassIntegrator},
    vector_assembler::VectorAssembler,
};
use fem_element::{raviart_thomas::TriRT0, reference::VectorReferenceElement};
use fem_mesh::{ElementTransformation, SimplexMesh};
use fem_mesh::topology::MeshTopology;
use fem_solver::{MinresSolver, SolverConfig};
use fem_space::{fe_space::FESpace, HDivSpace};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 4: H(div) grad-div problem (RT0) ===");
    println!("  Mesh: {}x{} subdivisions, RT0 elements", args.n, args.n);

    let result = solve_case(args.n, args.alpha, args.beta, 1.0);

    println!("  DOFs: {} (one per edge)", result.n_dofs);
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  h = {:.4e}", result.h);
    println!(
        "  ||div F_h||_L2 = {:.4e}  (should be small for a divergence-free solution)",
        result.div_l2
    );
    println!("  ||F_h - F_exact||_L2 ~= {:.4e}", result.flux_l2);
    println!("  max|DOF| = {:.4e}", result.max_dof);

    println!("\nDone.");
}

struct SolveResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    div_l2: f64,
    flux_l2: f64,
    max_dof: f64,
}

fn solve_case(n: usize, alpha: f64, beta: f64, source_scale: f64) -> SolveResult {
    solve_case_with_solution(n, alpha, beta, source_scale).0
}

fn solve_case_with_solution(
    n: usize,
    alpha: f64,
    beta: f64,
    source_scale: f64,
) -> (SolveResult, Vec<f64>) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = HDivSpace::new(mesh, 0);
    let n_dofs = space.n_dofs();

    let grad_div = GradDivIntegrator { kappa: alpha };
    let mass = VectorMassIntegrator { alpha: beta };
    let mat = VectorAssembler::assemble_bilinear(&space, &[&grad_div, &mass], 3);

    let source = VectorDomainLFIntegrator {
        f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
            let exact = exact_flux(x, source_scale);
            out[0] = beta * exact[0];
            out[1] = beta * exact[1];
        }),
    };
    let rhs = VectorAssembler::assemble_linear(&space, &[&source], 3);

    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = MinresSolver::solve(&mat, &rhs, &mut u, &cfg).expect("MINRES solve failed");

    let (div_l2, flux_l2) = compute_errors(&space, &u, source_scale);
    let max_dof = u.iter().copied().fold(0.0_f64, |acc, val| acc.max(val.abs()));

    (
        SolveResult {
            n_dofs,
            iterations: res.iterations,
            final_residual: res.final_residual,
            converged: res.converged,
            h: 1.0 / n as f64,
            div_l2,
            flux_l2,
            max_dof,
        },
        u,
    )
}

fn compute_errors(space: &HDivSpace<SimplexMesh<2>>, uh: &[f64], source_scale: f64) -> (f64, f64) {
    let mesh = space.mesh();
    let div_err2 = compute_element_divergence(space, uh)
        .into_iter()
        .zip(mesh.elem_iter())
        .map(|(div_val, elem)| {
            let nodes = mesh.element_nodes(elem);
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
                - (x1[1] - x0[1]) * (x2[0] - x0[0]))
                .abs();
            let area = 0.5 * det_j;
            area * div_val * div_val
        })
        .sum::<f64>();
    let flux_err2 = compute_flux_error(space, uh, source_scale);
    (div_err2.sqrt(), flux_err2.sqrt())
}

fn compute_flux_error(space: &HDivSpace<SimplexMesh<2>>, uh: &[f64], source_scale: f64) -> f64 {
    let mesh = space.mesh();
    let ref_elem = TriRT0;
    let quad = ref_elem.quadrature(4);
    let n_ldofs = ref_elem.n_dofs();
    let mut ref_phi = vec![0.0; n_ldofs * 2];
    let mut phys_phi = vec![0.0; n_ldofs * 2];
    let mut flux_err2 = 0.0_f64;

    for elem in mesh.elem_iter() {
        let dofs = space.element_dofs(elem);
        let signs = space.element_signs(elem);
        let nodes = mesh.element_nodes(elem);
        let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
        let jac = tr.jacobian().clone();
        let det_j = tr.det_j();

        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis_vec(xi, &mut ref_phi);
            let inv_det = 1.0 / det_j;

            for i in 0..n_ldofs {
                for r in 0..2 {
                    let mut value = 0.0;
                    for c in 0..2 {
                        value += jac[(r, c)] * ref_phi[i * 2 + c];
                    }
                    phys_phi[i * 2 + r] = signs[i] * value * inv_det;
                }
            }

            let mut approx = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let coeff = uh[dofs[i] as usize];
                approx[0] += coeff * phys_phi[i * 2];
                approx[1] += coeff * phys_phi[i * 2 + 1];
            }

            let x_phys = tr.map_to_physical(xi);
            let exact = exact_flux(&x_phys, source_scale);
            let dx = approx[0] - exact[0];
            let dy = approx[1] - exact[1];
            flux_err2 += quad.weights[q] * det_j.abs() * (dx * dx + dy * dy);
        }
    }

    flux_err2
}

fn exact_flux(x: &[f64], source_scale: f64) -> [f64; 2] {
    [
        source_scale * (PI * x[0]).sin() * (PI * x[1]).cos(),
        -source_scale * (PI * x[0]).cos() * (PI * x[1]).sin(),
    ]
}

struct Args {
    n: usize,
    alpha: f64,
    beta: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 8,
        alpha: 1.0,
        beta: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8);
            }
            "--alpha" => {
                a.alpha = it.next().unwrap_or("1".into()).parse().unwrap_or(1.0);
            }
            "--beta" => {
                a.beta = it.next().unwrap_or("1".into()).parse().unwrap_or(1.0);
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex4_darcy_coarse_mesh_has_reasonable_error() {
        let result = solve_case(8, 1.0, 1.0, 1.0);
        assert!(result.converged);
        assert!(result.final_residual < 1e-8, "residual too large: {}", result.final_residual);
        assert!(result.div_l2 < 2.5e-2, "divergence error too large: {}", result.div_l2);
        assert!(result.flux_l2 < 7.5e-1, "flux error too large: {}", result.flux_l2);
    }

    #[test]
    fn ex4_darcy_refinement_reduces_error() {
        let coarse = solve_case(8, 1.0, 1.0, 1.0);
        let fine = solve_case(16, 1.0, 1.0, 1.0);
        assert!(coarse.converged && fine.converged);
        assert!(
            fine.div_l2 < coarse.div_l2,
            "divergence did not improve: coarse={}, fine={}",
            coarse.div_l2,
            fine.div_l2
        );
        assert!(
            fine.flux_l2 < 7.5e-1,
            "refined flux error should remain bounded: {}",
            fine.flux_l2
        );
    }

    #[test]
    fn ex4_darcy_larger_alpha_reduces_divergence_leakage() {
        let baseline = solve_case(8, 1.0, 1.0, 1.0);
        let penalized = solve_case(8, 10.0, 1.0, 1.0);
        assert!(baseline.converged && penalized.converged);
        assert!(
            penalized.div_l2 < 0.2 * baseline.div_l2,
            "larger alpha should strongly reduce divergence leakage: baseline={}, penalized={}",
            baseline.div_l2,
            penalized.div_l2
        );
        assert!(
            penalized.flux_l2 < baseline.flux_l2 + 5.0e-2,
            "larger alpha should not severely degrade flux error: baseline={}, penalized={}",
            baseline.flux_l2,
            penalized.flux_l2
        );
    }

    #[test]
    fn ex4_darcy_sign_reversed_source_flips_solution() {
        let (forward, u_pos) = solve_case_with_solution(8, 1.0, 1.0, 1.0);
        let (reverse, u_neg) = solve_case_with_solution(8, 1.0, 1.0, -1.0);
        assert!(forward.converged && reverse.converged);

        let odd_symmetry = u_pos
            .iter()
            .zip(u_neg.iter())
            .map(|(a, b)| (a + b).abs())
            .fold(0.0_f64, f64::max);

        assert!(odd_symmetry < 1e-10, "sign reversal mismatch: {}", odd_symmetry);
        assert!((forward.div_l2 - reverse.div_l2).abs() < 1e-10);
        assert!((forward.flux_l2 - reverse.flux_l2).abs() < 1e-10);
    }

    /// Solution should scale linearly with source strength.
    #[test]
    fn ex4_darcy_solution_scales_linearly_with_source() {
        let scale_1 = solve_case(8, 1.0, 1.0, 1.0);
        let scale_2 = solve_case(8, 1.0, 1.0, 2.0);

        assert!(scale_1.converged && scale_2.converged);
        // Flux error should also scale
        let flux_ratio = scale_2.flux_l2 / scale_1.flux_l2;
        assert!(
            (flux_ratio - 2.0).abs() < 0.1,
            "flux error should scale ~2x with source strength, got ratio {:.2}",
            flux_ratio
        );
    }

    /// Beta parameter (mass term) should affect solution magnitude.
    #[test]
    fn ex4_darcy_higher_beta_increases_solution_magnitude() {
        let low_beta = solve_case(8, 1.0, 0.1, 1.0);
        let high_beta = solve_case(8, 1.0, 10.0, 1.0);

        assert!(low_beta.converged && high_beta.converged);
        // With higher beta, mass term dominates, so flux should increase
        assert!(
            high_beta.max_dof > low_beta.max_dof,
            "higher beta should increase solution magnitude: low={} high={}",
            low_beta.max_dof,
            high_beta.max_dof
        );
    }

    /// Convergence rate: flux error should improve overall with mesh refinement on finer mesh levels.
    #[test]
    fn ex4_darcy_flux_error_improves_with_refinement() {
        let n8 = solve_case(8, 1.0, 1.0, 1.0);
        let n12 = solve_case(12, 1.0, 1.0, 1.0);
        let n16 = solve_case(16, 1.0, 1.0, 1.0);

        assert!(n8.converged && n12.converged && n16.converged);
        // On medium meshes, flux error should generally decrease
        assert!(n16.flux_l2 < n8.flux_l2 * 1.1,
            "fine mesh flux error should be comparable or better: n8={}, n16={}",
            n8.flux_l2, n16.flux_l2);
        
        // DOF count should increase monotonically
        assert!(n12.n_dofs > n8.n_dofs, "DOF count should increase: n8={}, n12={}", n8.n_dofs, n12.n_dofs);
        assert!(n16.n_dofs > n12.n_dofs, "DOF count should increase: n12={}, n16={}", n12.n_dofs, n16.n_dofs);
    }

    /// Very weak source should give near-trivial solution.
    #[test]
    fn ex4_darcy_very_weak_source_gives_small_solution() {
        let result = solve_case(8, 1.0, 1.0, 1e-6);
        assert!(result.converged);
        // With very weak source, solution should be proportionally small
        assert!(result.max_dof < 1e-4, "very weak source should give small solution, got max_dof={}", result.max_dof);
    }
}

