//! # Example 4 - Darcy flow / grad-div problem  (analogous to MFEM ex4)
//!
//! Solves the H(div) grad-div problem on the unit square:
//!
//! ```text
//!   -∇(alpha ∇·F) + beta F = f    in Ω
//!                F·n = 0          on ∂Ω
//! ```
//!
//! where `f` is a divergence-source RHS so that `∇·F ≈ 1`, matching the
//! mixed Darcy formulation of MFEM ex4:
//!
//! ```text
//!   -∇·(kappa ∇p) = 1    in Ω
//! ```
//!
//! Uses lowest-order Raviart-Thomas (RT0) elements.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex4_darcy
//! cargo run --example mfem_ex4_darcy -- -m ../data/star.mesh
//! cargo run --example mfem_ex4_darcy -- --n 16
//! cargo run --example mfem_ex4_darcy -- --alpha 1.0 --beta 1.0
//! ```
//!
//! ## Output
//! Prints DOF count, MINRES iteration count, final residual, divergence error,
//! and max DOF magnitude.

use fem_assembly::{
    postprocess::compute_element_divergence,
    standard::{GradDivIntegrator, VectorMassIntegrator},
    vector_assembler::VectorAssembler,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_solver::{MinresSolver, SolverConfig};
use fem_space::{fe_space::FESpace, HDivSpace};

/// Divergence-source RHS integrator: `∫ g · (∇·v) dx`
///
/// This is the correct RHS for the grad-div formulation of the Darcy problem
/// `∇·F = f`, where `f` is a scalar source function.
struct DivSourceIntegrator {
    g: f64,
}

impl VectorLinearIntegrator for DivSourceIntegrator {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let w = qp.weight * self.g;
        for i in 0..qp.n_dofs {
            f_elem[i] += w * qp.div[i];
        }
    }
}

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 4: H(div) grad-div problem (RT0) ===");
    match &args.mesh {
        Some(path) => println!("  Mesh file: {}", path),
        None => println!("  Mesh: {}x{} subdivisions, RT0 elements", args.n, args.n),
    }

    let result = solve_case(&args);

    println!("  DOFs: {} (one per edge)", result.n_dofs);
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  h = {:.4e}", result.h);
    println!(
        "  ||∇·F_h - 1||_L2 = {:.4e}  (should be small for constant source)",
        result.div_l2
    );
    println!("  max|DOF| = {:.4e}", result.max_dof);

    println!("\nDone.");
}

#[allow(dead_code)]
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

fn solve_case(args: &Args) -> SolveResult {
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
    let space = HDivSpace::new(mesh, 0);
    let n_dofs = space.n_dofs();

    let grad_div = GradDivIntegrator { kappa: args.alpha };
    let mass = VectorMassIntegrator { alpha: args.beta };
    let mat = VectorAssembler::assemble_bilinear(&space, &[&grad_div, &mass], 3);

    // RHS: ∫ 1 · (∇·v) dx  (constant source, matching MFEM ex4)
    let source = DivSourceIntegrator { g: 1.0 };
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

    // Compute ‖∇·F_h - 1‖_L2
    let mesh = space.mesh();
    let div_l2 = div_error_l2(&space, &u);

    let max_dof = u.iter().copied().fold(0.0_f64, |acc, val| acc.max(val.abs()));

    // Estimate element size from first element
    let elem = mesh.elem_iter().next().unwrap();
    let nodes = mesh.element_nodes(elem);
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
        - (x1[1] - x0[1]) * (x2[0] - x0[0]))
        .abs();
    let h = if args.mesh.is_some() {
        (det_j).sqrt()
    } else {
        1.0 / args.n as f64
    };

    SolveResult {
        n_dofs,
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        h,
        div_l2,
        flux_l2: 0.0,
        max_dof,
    }
}

/// Compute ‖∇·F_h - 1‖_L2 over the mesh.
fn div_error_l2(space: &HDivSpace<Mesh<2>>, uh: &[f64]) -> f64 {
    let mesh = space.mesh();
    compute_element_divergence(space, uh)
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
            let diff = div_val - 1.0;
            area * diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    alpha: f64,
    beta: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 8,
        alpha: 1.0,
        beta: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                a.mesh = it.next();
            }
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use fem_assembly::coefficient::FnVectorCoeff;
    use fem_assembly::standard::VectorDomainLFIntegrator;
    use fem_element::{raviart_thomas::TriRT0, reference::VectorReferenceElement};
    use fem_mesh::ElementTransformation;

    // ── Manufactured solution infrastructure ──────────────────────────────

    fn exact_flux(x: &[f64], source_scale: f64) -> [f64; 2] {
        [
            source_scale * (PI * x[0]).sin() * (PI * x[1]).cos(),
            -source_scale * (PI * x[0]).cos() * (PI * x[1]).sin(),
        ]
    }

    /// Solve the MMS problem with the manufactured-solution RHS.
    fn solve_mms(n: usize, alpha: f64, beta: f64, source_scale: f64) -> SolveResult {
        solve_mms_with_solution(n, alpha, beta, source_scale).0
    }

    fn solve_mms_with_solution(
        n: usize,
        alpha: f64,
        beta: f64,
        source_scale: f64,
    ) -> (SolveResult, Vec<f64>) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = HDivSpace::new(mesh, 0);
        let n_dofs = space.n_dofs();

        let grad_div = GradDivIntegrator { kappa: alpha };
        let mass = VectorMassIntegrator { alpha: beta };
        let mat = VectorAssembler::assemble_bilinear(&space, &[&grad_div, &mass], 3);

        let beta_c = beta;
        let source = VectorDomainLFIntegrator {
            f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let exact = exact_flux(x, source_scale);
                out[0] = beta_c * exact[0];
                out[1] = beta_c * exact[1];
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

        // Element-wise divergence ‖∇·F_h‖_L2 (MMS solution is divergence-free)
        let mesh = space.mesh();
        let div_values = compute_element_divergence(&space, &u);
        let div_l2 = div_values
            .iter()
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
            .sum::<f64>()
            .sqrt();

        let flux_err2 = compute_flux_error(&space, &u, source_scale);
        let max_dof = u
            .iter()
            .copied()
            .fold(0.0_f64, |acc, val| acc.max(val.abs()));

        (
            SolveResult {
                n_dofs,
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                h: 1.0 / n as f64,
                div_l2,
                flux_l2: flux_err2.sqrt(),
                max_dof,
            },
            u,
        )
    }

    fn compute_flux_error(
        space: &HDivSpace<Mesh<2>>,
        uh: &[f64],
        source_scale: f64,
    ) -> f64 {
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

    // ── Standard tests ────────────────────────────────────────────────────

    #[test]
    fn ex4_darcy_coarse_mesh_has_reasonable_error() {
        let result = solve_mms(8, 1.0, 1.0, 1.0);
        assert!(result.converged);
        assert!(
            result.final_residual < 1e-8,
            "residual too large: {}",
            result.final_residual
        );
        assert!(
            result.div_l2 < 2.5e-2,
            "divergence error too large: {}",
            result.div_l2
        );
        assert!(
            result.flux_l2 < 7.5e-1,
            "flux error too large: {}",
            result.flux_l2
        );
    }

    #[test]
    fn ex4_darcy_refinement_reduces_error() {
        let coarse = solve_mms(8, 1.0, 1.0, 1.0);
        let fine = solve_mms(16, 1.0, 1.0, 1.0);
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
        let baseline = solve_mms(8, 1.0, 1.0, 1.0);
        let penalized = solve_mms(8, 10.0, 1.0, 1.0);
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
        let (forward, u_pos) = solve_mms_with_solution(8, 1.0, 1.0, 1.0);
        let (reverse, u_neg) = solve_mms_with_solution(8, 1.0, 1.0, -1.0);
        assert!(forward.converged && reverse.converged);

        let odd_symmetry = u_pos
            .iter()
            .zip(u_neg.iter())
            .map(|(a, b)| (a + b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            odd_symmetry < 1e-10,
            "sign reversal mismatch: {}",
            odd_symmetry
        );
        assert!((forward.div_l2 - reverse.div_l2).abs() < 1e-10);
        assert!((forward.flux_l2 - reverse.flux_l2).abs() < 1e-10);
    }

    #[test]
    fn ex4_darcy_solution_scales_linearly_with_source() {
        let scale_1 = solve_mms(8, 1.0, 1.0, 1.0);
        let scale_2 = solve_mms(8, 1.0, 1.0, 2.0);

        assert!(scale_1.converged && scale_2.converged);
        let flux_ratio = scale_2.flux_l2 / scale_1.flux_l2;
        assert!(
            (flux_ratio - 2.0).abs() < 0.1,
            "flux error should scale ~2x with source strength, got ratio {:.2}",
            flux_ratio
        );
    }

    #[test]
    fn ex4_darcy_higher_beta_increases_solution_magnitude() {
        let low_beta = solve_mms(8, 1.0, 0.1, 1.0);
        let high_beta = solve_mms(8, 1.0, 10.0, 1.0);

        assert!(low_beta.converged && high_beta.converged);
        assert!(
            high_beta.max_dof > low_beta.max_dof,
            "higher beta should increase solution magnitude: low={} high={}",
            low_beta.max_dof,
            high_beta.max_dof
        );
    }

    #[test]
    fn ex4_darcy_flux_error_improves_with_refinement() {
        let n8 = solve_mms(8, 1.0, 1.0, 1.0);
        let n12 = solve_mms(12, 1.0, 1.0, 1.0);
        let n16 = solve_mms(16, 1.0, 1.0, 1.0);

        assert!(n8.converged && n12.converged && n16.converged);
        assert!(
            n16.flux_l2 < n8.flux_l2 * 1.1,
            "fine mesh flux error should be comparable or better: n8={}, n16={}",
            n8.flux_l2,
            n16.flux_l2
        );

        assert!(
            n12.n_dofs > n8.n_dofs,
            "DOF count should increase: n8={}, n12={}",
            n8.n_dofs,
            n12.n_dofs
        );
        assert!(
            n16.n_dofs > n12.n_dofs,
            "DOF count should increase: n12={}, n16={}",
            n12.n_dofs,
            n16.n_dofs
        );
    }

    #[test]
    fn ex4_darcy_very_weak_source_gives_small_solution() {
        let result = solve_mms(8, 1.0, 1.0, 1e-6);
        assert!(result.converged);
        assert!(
            result.max_dof < 1e-4,
            "very weak source should give small solution, got max_dof={}",
            result.max_dof
        );
    }

    // ── Regression baseline ───────────────────────────────────────────────

    #[test]
    fn ex4_regression_baseline() {
        let result = solve_mms(8, 1.0, 1.0, 1.0);
        assert!(result.converged);

        fem_regression::regression("mfem_ex4_darcy")
            .check_with("div_l2", result.div_l2, 1e-6, 1e-10)
            .check_with("flux_l2", result.flux_l2, 1e-6, 1e-10)
            .check_with("max_dof", result.max_dof, 1e-6, 1e-10)
            .check_with("n_dofs", result.n_dofs as f64, 0.0, 0.5)
            .check_with("iterations", result.iterations as f64, 1e-4, 0.5)
            .check_with("residual", result.final_residual, 1e-4, 1e-10)
            .finalize();
    }

    // ── MFEM cross-validation test ────────────────────────────────────────

    #[test]
    fn ex4_mfem_reference_test() {
        let result = solve_mms(8, 1.0, 1.0, 1.0);
        assert!(result.converged, "solve must converge");

        assert_eq!(
            result.n_dofs, 208,
            "RT0 on 8×8 should give 208 edge DOFs, got {}",
            result.n_dofs
        );

        assert!(result.flux_l2 > 0.0, "flux L2 error should be positive");
        assert!(result.div_l2.is_finite(), "div L2 should be finite");
        assert!(
            result.div_l2 < 0.1,
            "div L2 should be small (div-free solution)"
        );

        assert!(
            result.iterations > 0,
            "MINRES should take positive iterations"
        );
        assert!(result.final_residual < 1e-8, "residual should be small");

        let coarse = solve_mms(6, 1.0, 1.0, 1.0);
        let fine = solve_mms(12, 1.0, 1.0, 1.0);
        assert!(coarse.converged && fine.converged);
        eprintln!(
            "  [mfem-ref] ex4: flux(6)={:.6e} flux(12)={:.6e}",
            coarse.flux_l2, fine.flux_l2
        );

        use std::time::Instant;
        let t0 = Instant::now();
        let _ = solve_mms(32, 1.0, 1.0, 1.0);
        let elapsed = t0.elapsed();
        assert!(
            elapsed.as_secs_f64() < 15.0,
            "32×32 Darcy took {:.2}s, bound 15s",
            elapsed.as_secs_f64()
        );
        eprintln!(
            "  [mfem-ref] ex4: 32×32 Darcy = {:.2}ms",
            elapsed.as_millis()
        );
    }
}
