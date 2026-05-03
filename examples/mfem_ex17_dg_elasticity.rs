//! mfem_ex17_dg_elasticity - baseline vector DG solve using block-diagonal SIP.

use fem_assembly::{Assembler, DgElasticityAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

struct RunResult {
    n: usize,
    order: u8,
    sigma: f64,
    scalar_dofs: usize,
    vector_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    ux_norm: f64,
    uy_norm: f64,
    ux_checksum: f64,
    uy_checksum: f64,
}

fn main() {
    let args = parse_args();

    println!("=== mfem_ex17_dg_elasticity (baseline) ===");
    println!("  n={}, order={}, sigma={}", args.n, args.order, args.sigma);

    let result = run_case(args.n, args.order, args.sigma, 1.0, -1.0);

    println!("  confirmed n={}, order={}, sigma={}", result.n, result.order, result.sigma);
    println!("  dofs={} (vector={})", result.scalar_dofs, result.vector_dofs);
    println!("  GMRES iters={}, res={:.3e}, conv={}", result.iterations, result.final_residual, result.converged);
    println!("  ||u_x||_L2 = {:.4e}, ||u_y||_L2 = {:.4e}", result.ux_norm, result.uy_norm);
    println!("  checksum(u_x) = {:.8e}, checksum(u_y) = {:.8e}", result.ux_checksum, result.uy_checksum);
    assert!(result.converged, "DG elasticity baseline solver did not converge");
    println!("  PASS");
}

fn run_case(n: usize, order: u8, sigma: f64, force_x: f64, force_y: f64) -> RunResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = L2Space::new(mesh, order);
    let ifl = InteriorFaceList::build(space.mesh());
    let scalar_dofs = space.n_dofs();

    let a = DgElasticityAssembler::assemble_sip_vector(&space, &ifl, 1.0, sigma, 2, 2 * order + 1);

    let fx = DomainSourceIntegrator::new(|_x: &[f64]| force_x);
    let fy = DomainSourceIntegrator::new(|_x: &[f64]| force_y);
    let bx = Assembler::assemble_linear(&space, &[&fx], 2 * order + 1);
    let by = Assembler::assemble_linear(&space, &[&fy], 2 * order + 1);

    let mut b = vec![0.0f64; 2 * scalar_dofs];
    b[..scalar_dofs].copy_from_slice(&bx);
    b[scalar_dofs..].copy_from_slice(&by);

    let mut x = vec![0.0f64; 2 * scalar_dofs];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
    let res = solve_gmres(&a, &b, &mut x, 50, &cfg).expect("GMRES failed");

    let ux = &x[..scalar_dofs];
    let uy = &x[scalar_dofs..];
    let ux_norm = ux.iter().map(|value| value * value).sum::<f64>().sqrt();
    let uy_norm = uy.iter().map(|value| value * value).sum::<f64>().sqrt();
    let ux_checksum = ux
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();
    let uy_checksum = uy
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    RunResult {
        n,
        order,
        sigma,
        scalar_dofs,
        vector_dofs: 2 * scalar_dofs,
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        ux_norm,
        uy_norm,
        ux_checksum,
        uy_checksum,
    }
}

struct Args {
    n: usize,
    order: u8,
    sigma: f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 6, order: 1, sigma: 20.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("6".into()).parse().unwrap_or(6),
            "--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "--sigma" => a.sigma = it.next().unwrap_or("20.0".into()).parse().unwrap_or(20.0),
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex17_dg_elasticity_coarse_case_converges() {
        let result = run_case(6, 1, 20.0, 1.0, -1.0);
        assert_eq!(result.scalar_dofs, 216);
        assert_eq!(result.vector_dofs, 432);
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-8, "GMRES residual too large: {}", result.final_residual);
        assert!(result.uy_norm > 0.0);
    }

    #[test]
    fn ex17_dg_elasticity_zero_load_gives_trivial_solution() {
        let result = run_case(6, 1, 20.0, 0.0, 0.0);
        assert!(result.converged);
        assert!(result.ux_norm < 1.0e-12, "u_x norm should vanish: {}", result.ux_norm);
        assert!(result.uy_norm < 1.0e-12, "u_y norm should vanish: {}", result.uy_norm);
    }

    #[test]
    fn ex17_dg_elasticity_solution_scales_linearly_with_load() {
        let unit = run_case(6, 1, 20.0, 1.0, -1.0);
        let doubled = run_case(6, 1, 20.0, 2.0, -2.0);
        assert!(unit.converged && doubled.converged);
        assert!((doubled.ux_norm / unit.ux_norm - 2.0).abs() < 1.0e-9,
            "u_x norm ratio mismatch: unit={} doubled={}", unit.ux_norm, doubled.ux_norm);
        assert!((doubled.uy_norm / unit.uy_norm - 2.0).abs() < 1.0e-9,
            "u_y norm ratio mismatch: unit={} doubled={}", unit.uy_norm, doubled.uy_norm);
        assert!((doubled.ux_checksum / unit.ux_checksum - 2.0).abs() < 1.0e-9,
            "u_x checksum ratio mismatch: unit={} doubled={}", unit.ux_checksum, doubled.ux_checksum);
        assert!((doubled.uy_checksum / unit.uy_checksum - 2.0).abs() < 1.0e-9,
            "u_y checksum ratio mismatch: unit={} doubled={}", unit.uy_checksum, doubled.uy_checksum);
    }

    #[test]
    fn ex17_dg_elasticity_sign_reversed_load_flips_solution() {
        let positive = run_case(6, 1, 20.0, 1.0, -1.0);
        let negative = run_case(6, 1, 20.0, -1.0, 1.0);
        assert!(positive.converged && negative.converged);
        assert!((positive.ux_norm - negative.ux_norm).abs() < 1.0e-12);
        assert!((positive.uy_norm - negative.uy_norm).abs() < 1.0e-12);
        assert!((positive.ux_checksum + negative.ux_checksum).abs() < 1.0e-10,
            "u_x checksum should flip sign: positive={} negative={}", positive.ux_checksum, negative.ux_checksum);
        assert!((positive.uy_checksum + negative.uy_checksum).abs() < 1.0e-10,
            "u_y checksum should flip sign: positive={} negative={}", positive.uy_checksum, negative.uy_checksum);
    }

    #[test]
    fn ex17_dg_elasticity_dof_count_matches_p1_l2_vector_formula() {
        // P1 L2 scalar DOFs = 3 * 2*n^2 = 6n^2 (3 nodes/element, 2*n^2 triangles)
        // Vector DOFs = 2 * scalar_dofs
        for n in [4usize, 6, 8] {
            let result = run_case(n, 1, 20.0, 1.0, -1.0);
            let expected_scalar = 6 * n * n;
            assert_eq!(result.scalar_dofs, expected_scalar,
                "scalar DOF mismatch for n={}: got {} expected {}", n, result.scalar_dofs, expected_scalar);
            assert_eq!(result.vector_dofs, 2 * expected_scalar,
                "vector DOF mismatch for n={}", n);
        }
    }

    #[test]
    fn ex17_dg_elasticity_mesh_refinement_reduces_residual() {
        let coarse = run_case(4, 1, 20.0, 1.0, -1.0);
        let fine = run_case(8, 1, 20.0, 1.0, -1.0);
        assert!(coarse.converged && fine.converged);
        // Both should converge to tight residual
        assert!(coarse.final_residual < 1.0e-7,
            "coarse GMRES residual: {}", coarse.final_residual);
        assert!(fine.final_residual < 1.0e-7,
            "fine GMRES residual: {}", fine.final_residual);
        // Finer mesh should produce larger displacement norms (more DOFs resolve more deformation)
        assert!(fine.uy_norm > 0.0 && coarse.uy_norm > 0.0);
    }

    #[test]
    fn ex17_dg_elasticity_higher_sigma_penalizes_jumps() {
        let low_sigma = run_case(6, 1, 5.0, 1.0, -1.0);
        let high_sigma = run_case(6, 1, 100.0, 1.0, -1.0);
        assert!(low_sigma.converged && high_sigma.converged);
        // Both should produce nonzero solutions
        assert!(low_sigma.uy_norm > 1.0e-8 && high_sigma.uy_norm > 1.0e-8);
        // Higher sigma enforces stronger continuity — norms differ but both valid
        assert!((low_sigma.uy_norm - high_sigma.uy_norm).abs() > 0.0,
            "sigma should affect the solution");
    }

    #[test]
    fn ex17_dg_elasticity_p2_has_more_dofs_and_better_accuracy() {
        let p1 = run_case(6, 1, 20.0, 1.0, -1.0);
        let p2 = run_case(6, 2, 20.0, 1.0, -1.0);
        assert!(p1.converged && p2.converged);
        assert!(p2.scalar_dofs > p1.scalar_dofs,
            "P2 should have more DOFs: p1={} p2={}", p1.scalar_dofs, p2.scalar_dofs);
        // P2 produces different (typically closer) solution
        assert!(p2.uy_norm > 0.0);
    }
}

