//! # Example 9 �?DG Diffusion (SIP)  (analogous to MFEM ex9 / ex14)
//!
//! Solves the scalar Poisson equation using the Symmetric Interior Penalty (SIP)
//! Discontinuous Galerkin method:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω = [0,1]²
//!            u = 0    on ∂�?  (enforced weakly via penalty)
//! ```
//!
//! The SIP bilinear form is:
//! ```text
//!   a_h(u,v) = ∑_K ∫_K κ ∇u·∇v dx
//!              �?∑_F ∫_F {κ∇u}·[[v]] ds   (consistency)
//!              �?∑_F ∫_F {κ∇v}·[[u]] ds   (symmetry)
//!              + ∑_F σ/h_F ∫_F [[u]]·[[v]] ds  (penalty)
//! ```
//!
//! Manufactured solution: `u = sin(πx)sin(πy)`.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex9_dg_advection
//! cargo run --example mfem_ex9_dg_advection -- --n 16 --order 1 --sigma 20
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler, DgAssembler, InteriorFaceList,
    standard::DomainSourceIntegrator,
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

struct SolveResult {
    n: usize,
    order: u8,
    sigma: f64,
    n_nodes: usize,
    n_elements: usize,
    n_dofs: usize,
    n_interior_faces: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    l2_error: f64,
    solution_norm: f64,
    solution_checksum: f64,
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 9: SIP-DG Diffusion ===");
    println!("  Mesh: {}×{} subdivisions, P{} DG elements", args.n, args.n, args.order);
    println!("  Penalty σ = {}", args.sigma);

    let result = solve_case(args.n, args.order, args.sigma, 1.0);

    println!("  Nodes: {}, Elements: {}", result.n_nodes, result.n_elements);
    println!("  DOFs: {}  ({} per element)", result.n_dofs, result.n_dofs / result.n_elements);
    println!("  Interior faces: {}", result.n_interior_faces);
    println!("  Effective σ = {:.3}", result.sigma);
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );

    let h = 1.0 / result.n as f64;
    println!("  h = {h:.4e},  L²(DG) error = {:.4e}", result.l2_error);
    println!("  ||u_h||_L2 = {:.4e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!("  (Expected O(h^{}) for P{} DG)", result.order + 1, result.order);
    println!("\nDone.");
}

fn solve_case(n: usize, order: u8, sigma: f64, source_scale: f64) -> SolveResult {
    // ─── 1. Mesh and L² (DG) space ───────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(n);

    let space = L2Space::new(mesh, order);
    let n_dofs = space.n_dofs();

    // ─── 2. Pre-build interior face list ─────────────────────────────────────
    let ifl = InteriorFaceList::build(space.mesh());

    // ─── 3. Assemble SIP stiffness matrix ────────────────────────────────────
    let kappa = 1.0_f64;
    let mat   = DgAssembler::assemble_sip(&space, &ifl, kappa, sigma, order * 2 + 1);

    // ─── 4. Assemble RHS: f = 2π² sin(πx)sin(πy) ────────────────────────────
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        source_scale * 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let rhs = Assembler::assemble_linear(&space, &[&source], order * 2 + 1);

    // Note: Dirichlet BCs are enforced weakly (penalty) by DgAssembler.
    // No explicit row-zeroing needed.

    // ─── 5. Solve with GMRES (SIP matrix is symmetric but ill-conditioned) ───
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let res = solve_gmres(&mat, &rhs, &mut u, 50, &cfg)
        .expect("DG solve failed");

    // ─── 6. Element-level L² error ───────────────────────────────────────────
    let l2 = dg_l2_error(&space, &u, |x: &[f64]| {
        source_scale * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let solution_norm = u.iter().map(|value| value * value).sum::<f64>().sqrt();
    let solution_checksum = u
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    SolveResult {
        n,
        order,
        sigma,
        n_nodes: space.mesh().n_nodes(),
        n_elements: space.mesh().n_elems(),
        n_dofs,
        n_interior_faces: ifl.faces.len(),
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        l2_error: l2,
        solution_norm,
        solution_checksum,
    }
}

// ─── DG L² error ─────────────────────────────────────────────────────────────

fn dg_l2_error<S: fem_space::fe_space::FESpace>(
    space: &S,
    uh: &[f64],
    exact: impl Fn(&[f64]) -> f64,
) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};
    use fem_mesh::topology::MeshTopology;

    let mesh = space.mesh();
    let mut err2 = 0.0_f64;

    for e in 0..mesh.n_elements() as u32 {
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

struct Args { n: usize, order: u8, sigma: f64 }

fn parse_args() -> Args {
    let mut a = Args { n: 8, order: 1, sigma: 20.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"     => { a.n     = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--order" => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            "--sigma" => { a.sigma = it.next().unwrap_or("20".into()).parse().unwrap_or(20.0); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convergence_rate(coarse: &SolveResult, fine: &SolveResult) -> f64 {
        let h_coarse = 1.0 / coarse.n as f64;
        let h_fine = 1.0 / fine.n as f64;
        (fine.l2_error / coarse.l2_error).ln() / (h_fine / h_coarse).ln()
    }

    #[test]
    fn ex9_dg_coarse_case_converges_with_reasonable_error() {
        let result = solve_case(8, 1, 20.0, 1.0);
        assert!(result.converged);
        assert_eq!(result.n_nodes, 81);
        assert_eq!(result.n_elements, 128);
        assert_eq!(result.n_dofs, 384);
        assert_eq!(result.n_interior_faces, 176);
        assert!((result.sigma - 20.0).abs() < 1.0e-12);
        assert!(result.final_residual < 1.0e-9, "solver residual too large: {}", result.final_residual);
        assert!(result.l2_error < 2.0e-2, "coarse-mesh L2 error too large: {}", result.l2_error);
    }

    #[test]
    fn ex9_dg_refinement_recovers_second_order_convergence() {
        let coarse = solve_case(8, 1, 20.0, 1.0);
        let medium = solve_case(16, 1, 20.0, 1.0);
        let fine = solve_case(32, 1, 20.0, 1.0);
        assert!(medium.l2_error < coarse.l2_error);
        assert!(fine.l2_error < medium.l2_error);
        assert!(convergence_rate(&coarse, &medium) > 1.9);
        assert!(convergence_rate(&medium, &fine) > 1.9);
        assert!(fine.l2_error < 1.2e-3, "fine-mesh L2 error too large: {}", fine.l2_error);
    }

    #[test]
    fn ex9_dg_solution_scales_linearly_with_source() {
        let unit = solve_case(16, 1, 20.0, 1.0);
        let doubled = solve_case(16, 1, 20.0, 2.0);
        assert!(unit.converged && doubled.converged);
        assert!((doubled.solution_norm / unit.solution_norm - 2.0).abs() < 1.0e-9,
            "solution norm ratio mismatch: unit={} doubled={}", unit.solution_norm, doubled.solution_norm);
        assert!((doubled.solution_checksum / unit.solution_checksum - 2.0).abs() < 1.0e-9,
            "solution checksum ratio mismatch: unit={} doubled={}", unit.solution_checksum, doubled.solution_checksum);
        assert!((doubled.l2_error / unit.l2_error - 2.0).abs() < 1.0e-9,
            "L2 error ratio mismatch: unit={} doubled={}", unit.l2_error, doubled.l2_error);
    }

    #[test]
    fn ex9_dg_sign_reversed_source_flips_solution() {
        let positive = solve_case(16, 1, 20.0, 1.0);
        let negative = solve_case(16, 1, 20.0, -1.0);
        assert!(positive.converged && negative.converged);
        assert!((positive.solution_norm - negative.solution_norm).abs() < 1.0e-12);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "solution checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
        assert!((positive.l2_error - negative.l2_error).abs() < 1.0e-12);
    }

    #[test]
    fn ex9_dg_dof_count_matches_p1_l2_space_formula() {
        // P1 DG on n×n uniform triangular mesh:
        //   elements = 2*n^2, nodes = (n+1)^2
        //   DOFs/element = 3 (one per vertex for P1 DG)
        //   total DOFs = 2*n^2 * 3 = 6*n^2
        for n in [4usize, 8, 16] {
            let result = solve_case(n, 1, 20.0, 1.0);
            let expected_elem = 2 * n * n;
            let expected_dofs = 6 * n * n;
            assert_eq!(result.n_elements, expected_elem,
                "n={}: expected {} elements, got {}", n, expected_elem, result.n_elements);
            assert_eq!(result.n_dofs, expected_dofs,
                "n={}: expected {} DOFs, got {}", n, expected_dofs, result.n_dofs);
        }
    }

    #[test]
    fn ex9_dg_interior_face_count_is_positive_and_grows_with_mesh() {
        let coarse = solve_case(8, 1, 20.0, 1.0);
        let fine = solve_case(16, 1, 20.0, 1.0);
        assert!(coarse.n_interior_faces > 0,
            "coarse mesh should have interior faces, got 0");
        assert!(fine.n_interior_faces > coarse.n_interior_faces,
            "finer mesh should have more interior faces: coarse={} fine={}",
            coarse.n_interior_faces, fine.n_interior_faces);
        // For n×n uniform tri mesh: each element has 3 faces, minus 1 per shared edge
        // n=8: 128 elements × 3 = 384 half-faces, boundary ≈ 4*n=32 boundary faces
        // interior ≈ (384 - 32) / 2 = 176
        assert_eq!(coarse.n_interior_faces, 176,
            "n=8: expected 176 interior faces, got {}", coarse.n_interior_faces);
    }

    #[test]
    fn ex9_dg_higher_sigma_maintains_second_order_convergence() {
        // Larger penalty σ should not degrade convergence order
        let coarse = solve_case(8, 1, 40.0, 1.0);
        let medium = solve_case(16, 1, 40.0, 1.0);
        let fine = solve_case(32, 1, 40.0, 1.0);
        assert!(medium.l2_error < coarse.l2_error);
        assert!(fine.l2_error < medium.l2_error);
        assert!(convergence_rate(&coarse, &medium) > 1.9,
            "σ=40 coarse→medium rate: {:.2}", convergence_rate(&coarse, &medium));
        assert!(convergence_rate(&medium, &fine) > 1.9,
            "σ=40 medium→fine rate: {:.2}", convergence_rate(&medium, &fine));
    }

    #[test]
    fn ex9_dg_higher_order_achieves_better_accuracy() {
        // P2 DG should be more accurate than P1 DG on same mesh
        let p1 = solve_case(8, 1, 20.0, 1.0);
        let p2 = solve_case(8, 2, 20.0, 1.0);
        assert!(p2.converged && p1.converged);
        assert!(p2.l2_error < p1.l2_error,
            "P2 DG should outperform P1 DG: p1={:.3e} p2={:.3e}", p1.l2_error, p2.l2_error);
        // P2 DOFs = 6 dofs/element * 2n^2 = 12n^2 (for order=2, 6 dofs/tri)
        assert!(p2.n_dofs > p1.n_dofs,
            "P2 should have more DOFs than P1: p1={} p2={}", p1.n_dofs, p2.n_dofs);
    }
}

