//! # Example 8 -- Static Condensation baseline (toward MFEM ex8/hybr)
//!
//! This example demonstrates the algebraic static-condensation primitive used by
//! fem-rs non-conforming constraints:
//!
//! ```text
//!   u_h = 0.5 * (u_a + u_b)
//! ```
//!
//! by applying `apply_hanging_constraints` to a toy SPD system and comparing the
//! free-DOF solution against an explicit reduced system solve.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex8_hybridization
//! cargo run --example mfem_ex8_hybridization -- --rhs-scale 2.0
//! ```

use fem_linalg::CooMatrix;
use fem_mesh::amr::HangingNodeConstraint;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::constraints::{apply_hanging_constraints, recover_hanging_values};

fn main() {
    let args = parse_args();
    let result = run_case(args.rhs_scale);

    println!("=== fem-rs Example 8: static condensation baseline ===");
    println!("  rhs scale: {:.3}", args.rhs_scale);
    println!("  iterations: {}", result.iterations);
    println!("  final residual: {:.3e}", result.final_residual);
    println!("  converged: {}", result.converged);
    println!(
        "  constrained consistency |u_h - 0.5(u_a+u_b)| = {:.3e}",
        result.hanging_consistency
    );
    println!(
        "  reduced-system agreement max(|u_free - u_ref|) = {:.3e}",
        result.free_dof_mismatch
    );
    println!("  ||u||_2 = {:.3e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!();
    println!("Note: this is the algebraic baseline toward ex8/hybridization; mixed/hybrid FEM kernels are still pending.");
}

struct CaseResult {
    iterations: usize,
    final_residual: f64,
    converged: bool,
    hanging_consistency: f64,
    free_dof_mismatch: f64,
    solution_norm: f64,
    solution_checksum: f64,
}

fn run_case(rhs_scale: f64) -> CaseResult {
    // Dense 3x3 SPD toy matrix with one constrained (hanging) DOF at index 2.
    let a = [
        [4.0, -1.0, -1.0],
        [-1.0, 4.0, -1.0],
        [-1.0, -1.0, 4.0],
    ];
    let b = [1.0 * rhs_scale, 2.0 * rhs_scale, 0.5 * rhs_scale];

    let mut mat = dense3_to_csr(a);
    let mut rhs = b.to_vec();

    let constraints = vec![HangingNodeConstraint {
        constrained: 2,
        parent_a: 0,
        parent_b: 1,
    }];

    apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

    let mut x = vec![0.0_f64; 3];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    let solve =
        solve_pcg_jacobi(&mat, &rhs, &mut x, &cfg).expect("PCG solve failed in ex8 baseline");
    recover_hanging_values(&mut x, &constraints);

    let x_ref = solve_reduced_reference(a, b);
    let hanging_consistency = (x[2] - 0.5 * (x[0] + x[1])).abs();
    let free_dof_mismatch = (x[0] - x_ref[0]).abs().max((x[1] - x_ref[1]).abs());
    let solution_norm = x.iter().map(|value| value * value).sum::<f64>().sqrt();
    let solution_checksum = x
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum();

    CaseResult {
        iterations: solve.iterations,
        final_residual: solve.final_residual,
        converged: solve.converged,
        hanging_consistency,
        free_dof_mismatch,
        solution_norm,
        solution_checksum,
    }
}

fn dense3_to_csr(a: [[f64; 3]; 3]) -> fem_linalg::CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(3, 3);
    for (i, row) in a.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if val.abs() > 0.0 {
                coo.add(i, j, *val);
            }
        }
    }
    coo.into_csr()
}

fn solve_reduced_reference(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 2] {
    // P maps free dofs [u0,u1] to full dofs [u0,u1,0.5(u0+u1)].
    let p = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];

    // AP = A * P (3x2)
    let mut ap = [[0.0_f64; 2]; 3];
    for i in 0..3 {
        for j in 0..2 {
            ap[i][j] = a[i][0] * p[0][j] + a[i][1] * p[1][j] + a[i][2] * p[2][j];
        }
    }

    // Ared = P^T * A * P (2x2)
    let mut ared = [[0.0_f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            ared[i][j] = p[0][i] * ap[0][j] + p[1][i] * ap[1][j] + p[2][i] * ap[2][j];
        }
    }

    // bred = P^T * b (2)
    let bred = [
        p[0][0] * b[0] + p[1][0] * b[1] + p[2][0] * b[2],
        p[0][1] * b[0] + p[1][1] * b[1] + p[2][1] * b[2],
    ];

    solve_2x2(ared, bred)
}

fn solve_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> [f64; 2] {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    assert!(det.abs() > 1e-14, "reduced 2x2 system is singular");
    [
        (b[0] * a[1][1] - b[1] * a[0][1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ]
}

struct Args {
    rhs_scale: f64,
}

fn parse_args() -> Args {
    let mut args = Args { rhs_scale: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--rhs-scale" => {
                args.rhs_scale = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
            _ => {}
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex8_static_condensation_matches_reduced_reference() {
        let result = run_case(1.0);
        assert!(result.converged);
        assert!(result.final_residual < 1e-9, "residual too large: {}", result.final_residual);
        assert!(result.hanging_consistency < 1e-12, "hanging consistency error: {}", result.hanging_consistency);
        assert!(result.free_dof_mismatch < 1e-10, "free dof mismatch: {}", result.free_dof_mismatch);
    }

    #[test]
    fn ex8_static_condensation_scales_linearly_with_rhs() {
        let a = run_case(1.0);
        let b = run_case(3.0);
        assert!(a.converged && b.converged);
        assert!(a.free_dof_mismatch < 1e-10);
        assert!(b.free_dof_mismatch < 1e-10);
        assert!(
            (b.solution_norm / a.solution_norm - 3.0).abs() < 1.0e-12,
            "solution norm ratio mismatch: a={} b={}",
            a.solution_norm,
            b.solution_norm
        );
        assert!(
            (b.solution_checksum / a.solution_checksum - 3.0).abs() < 1.0e-12,
            "solution checksum ratio mismatch: a={} b={}",
            a.solution_checksum,
            b.solution_checksum
        );
    }

    #[test]
    fn ex8_static_condensation_zero_rhs_gives_trivial_solution() {
        let result = run_case(0.0);
        assert!(result.converged);
        assert!(result.hanging_consistency < 1.0e-12);
        assert!(result.free_dof_mismatch < 1.0e-12);
        assert!(result.solution_norm < 1.0e-14, "expected zero solution norm, got {}", result.solution_norm);
        assert!(
            result.solution_checksum.abs() < 1.0e-14,
            "expected zero checksum, got {}",
            result.solution_checksum
        );
    }

    #[test]
    fn ex8_static_condensation_sign_reversed_rhs_flips_solution() {
        let positive = run_case(1.0);
        let negative = run_case(-1.0);
        assert!(positive.converged && negative.converged);
        assert!(
            (positive.solution_norm - negative.solution_norm).abs() < 1.0e-12,
            "solution norm should be sign-invariant: positive={} negative={}",
            positive.solution_norm,
            negative.solution_norm
        );
        assert!(
            (positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-12,
            "checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum
        );
    }

    #[test]
    fn ex8_hanging_consistency_is_scale_invariant() {
        // The hanging-node constraint u_h = 0.5(u_a + u_b) should hold to
        // machine precision regardless of the RHS magnitude.
        for &scale in &[0.001f64, 1.0, 1000.0] {
            let r = run_case(scale);
            assert!(r.converged, "scale={}: did not converge", scale);
            assert!(r.hanging_consistency < 1.0e-12,
                "scale={}: hanging consistency {:.3e} exceeds tolerance", scale, r.hanging_consistency);
        }
    }

    #[test]
    fn ex8_free_dof_mismatch_is_scale_invariant() {
        // The agreement between the full-system solve and the reduced-system
        // reference should stay near machine precision at any scale.
        for &scale in &[0.1f64, 5.0, 50.0] {
            let r = run_case(scale);
            assert!(r.converged, "scale={}: did not converge", scale);
            assert!(r.free_dof_mismatch < 1.0e-10,
                "scale={}: free_dof_mismatch {:.3e} exceeds tolerance", scale, r.free_dof_mismatch);
        }
    }

    #[test]
    fn ex8_large_rhs_scale_converges_and_maintains_accuracy() {
        let large = run_case(100.0);
        assert!(large.converged);
        assert!(large.hanging_consistency < 1.0e-12,
            "large-scale hanging consistency: {:.3e}", large.hanging_consistency);
        assert!(large.free_dof_mismatch < 1.0e-10,
            "large-scale free dof mismatch: {:.3e}", large.free_dof_mismatch);
        // solution should be exactly 100× the unit case
        let unit = run_case(1.0);
        assert!((large.solution_norm / unit.solution_norm - 100.0).abs() < 1.0e-10,
            "solution_norm ratio mismatch: {:.6}", large.solution_norm / unit.solution_norm);
        assert!((large.solution_checksum / unit.solution_checksum - 100.0).abs() < 1.0e-10,
            "checksum ratio mismatch: {:.6}", large.solution_checksum / unit.solution_checksum);
    }

    #[test]
    fn ex8_fractional_scale_preserves_residual_and_consistency() {
        let frac = run_case(0.25);
        assert!(frac.converged);
        assert!(frac.final_residual < 1.0e-9,
            "fractional-scale residual too large: {:.3e}", frac.final_residual);
        assert!(frac.hanging_consistency < 1.0e-12,
            "fractional-scale hanging consistency: {:.3e}", frac.hanging_consistency);
        assert!(frac.free_dof_mismatch < 1.0e-10,
            "fractional-scale free dof mismatch: {:.3e}", frac.free_dof_mismatch);
        // solution should be exactly 1/4 of the unit case
        let unit = run_case(1.0);
        assert!((frac.solution_norm / unit.solution_norm - 0.25).abs() < 1.0e-12,
            "solution_norm ratio mismatch: {:.6}", frac.solution_norm / unit.solution_norm);
    }
}
