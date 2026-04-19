//! # Example 7 Poisson with mixed Dirichlet/Neumann boundary conditions
//!
//! Solves the scalar Poisson equation with mixed boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω = [0,1]²
//!            u = 0    on Γ_D (bottom y=0 and left x=0 walls, tags 1,4)
//!    κ ∂u/∂n = g(x)   on Γ_N (top y=1 and right x=1 walls, tags 2,3)
//! ```
//!
//! Manufactured solution: `u(x,y) = x(1-x)y(1-y)`, which satisfies
//! homogeneous Dirichlet on all four walls.  However, we **deliberately**
//! apply Dirichlet only on two walls and Neumann on the other two to
//! demonstrate the assembly of the natural boundary term:
//!
//! ```text
//!   g(x,y) = κ (∂u/∂n)(x,y)   on Γ_N
//! ```
//!
//! The Neumann data is computed from the exact solution's normal derivative,
//! so the error against the exact solution should converge at O(h²).
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex7_neumann_mixed_bc
//! cargo run --example mfem_ex7_neumann_mixed_bc -- --n 32
//! cargo run --example mfem_ex7_neumann_mixed_bc -- --n 8 --n 16 --n 32  # convergence
//! ```

use fem_assembly::{
    assembler::face_dofs_p1,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, NeumannIntegrator},
    Assembler,
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
    H1Space,
};

// Exact solution: u(x,y) = scale * x(1-x) * y(1-y)
fn u_exact_scaled(x: &[f64], scale: f64) -> f64 {
    scale * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
}

// Source term: f = -Δu  for u = x(1-x)y(1-y)
//   Δu = ∂²u/∂x² + ∂²u/∂y²
//      = (-2)(y(1-y)) + (-2)(x(1-x))
//      = -2 [x(1-x) + y(1-y)]
// So f = -Δu = 2 [x(1-x) + y(1-y)]
fn f_rhs_scaled(x: &[f64], kappa: f64, scale: f64) -> f64 {
    2.0 * kappa * scale * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]))
}

struct SolveResult {
    n: usize,
    n_nodes: usize,
    n_dofs: usize,
    kappa: f64,
    solution_scale: f64,
    l2_error: f64,
    solution_norm: f64,
    solution_checksum: f64,
}

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 7: Poisson with mixed Dirichlet/Neumann BCs ===");
    println!("  BCs: Dirichlet on bottom+left (tags 1,4), Neumann on right+top (tags 2,3)");
    println!("  Exact solution: u = x(1-x)y(1-y)");
    println!();
    println!(
        "{:>5}  {:>8}  {:>8}  {:>12}  {:>8}",
        "n", "nodes", "dofs", "L² error", "rate"
    );
    println!("{}", "-".repeat(55));

    let mut prev_err = None::<f64>;
    let mut prev_h = None::<f64>;

    for &n in &args.n_list {
        let result = solve_case(n, args.kappa, 1.0);
        let h = 1.0 / n as f64;

        let rate = match (prev_err, prev_h) {
            (Some(e0), Some(h0)) => format!("{:.2}", (result.l2_error / e0).ln() / (h / h0).ln()),
            _ => "  --".to_string(),
        };

        println!(
            "  {:>3}  {:>8}  {:>8}  {:>12.4e}  {:>8}",
            result.n, result.n_nodes, result.n_dofs, result.l2_error, rate
        );
        if args.n_list.len() == 1 {
            println!(
                "       kappa = {:.3}, scale = {:.3}, ||u_h||_L2 = {:.4e}, checksum = {:.8e}",
                result.kappa,
                result.solution_scale,
                result.solution_norm,
                result.solution_checksum
            );
        }
        prev_err = Some(result.l2_error);
        prev_h = Some(h);
    }

    println!("\n  (Expected O(h²) convergence for P1 elements)");
    println!("Done.");
}

fn solve_case(n: usize, kappa: f64, solution_scale: f64) -> SolveResult {
    // ─── 1. Mesh and space ────────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();

    // ─── 2. Assemble stiffness K = κ ∇u·∇v dx ─────────────────────────────
    let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa }], 3);

    // ─── 3. Assemble volume RHS f v dx ─────────────────────────────────────
    let src = DomainSourceIntegrator::new(|x: &[f64]| f_rhs_scaled(x, kappa, solution_scale));
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

    // ─── 4. Assemble Neumann boundary term ∫_{Γ_N} g(x) v ds ────────────────
    //   Mesh tags (unit_square_tri convention):
    //     tag 1 = bottom (y=0), tag 2 = right (x=1),
    //     tag 3 = top (y=1),    tag 4 = left (x=0)
    //
    //   We apply Neumann on tags 2 (right) and 3 (top).
    //   On right wall (x=1): outward normal = (1,0), ∂u/∂n = ∂u/∂x = (1-2x)y(1-y)|_{x=1} = -y(1-y)
    //   On top wall  (y=1): outward normal = (0,1), ∂u/∂n = ∂u/∂y = x(1-x)(1-2y)|_{y=1} = -x(1-x)
    //
    //   The Neumann integrator computes g(x,n) v ds where g receives
    //   the physical coords x and the outward normal n at each face QP.

    let neumann_g = NeumannIntegrator::new(|x: &[f64], n: &[f64]| {
        // κ * ∂u/∂n at (x,y):
        //   On right (x): κ*(1-2x)*y*(1-y) the assembler calls this only for tagged faces
        //   On top   (y): κ*(1-2y)*x*(1-x)
        //   We return κ*(∂u/∂x * n_x + ∂u/∂y * n_y), but since the faces are
        //   axis-aligned and we know the normal from context, we can compute it
        //   directly. For generality we use the full gradient dotted with the
        //   actual normal passed by the integrator:
        let du_dx = solution_scale * (1.0 - 2.0 * x[0]) * x[1] * (1.0 - x[1]);
        let du_dy = solution_scale * x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]);
        kappa * (du_dx * n[0] + du_dy * n[1])
    });

    let face_dofs = face_dofs_p1(space.mesh());
    let neumann_rhs = Assembler::assemble_boundary_linear(
        ndofs,
        space.mesh(),
        &face_dofs,
        1,
        &[&neumann_g],
        &[2, 3], // right and top walls
        3,
    );
    for i in 0..ndofs {
        rhs[i] += neumann_rhs[i];
    }

    // ─── 5. Apply Dirichlet BCs u = 0 on bottom (tag 1) and left (tag 4) ─────
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // ─── 6. Solve ─────────────────────────────────────────────────────────────
    let mut u = vec![0.0_f64; ndofs];
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");

    // ─── 7. L² error ──────────────────────────────────────────────────────────
    let l2 = l2_error(&space, &u, |x| u_exact_scaled(x, solution_scale));
    let solution_norm = u.iter().map(|value| value * value).sum::<f64>().sqrt();
    let solution_checksum = u
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    SolveResult {
        n,
        n_nodes: space.mesh().n_nodes(),
        n_dofs: ndofs,
        kappa,
        solution_scale,
        l2_error: l2,
        solution_norm,
        solution_checksum,
    }
}

// ─── L² error helper ─────────────────────────────────────────────────────────

fn l2_error<S: FESpace>(space: &S, uh: &[f64], u_ex: impl Fn(&[f64]) -> f64) -> f64 {
    use fem_element::{lagrange::TriP1, ReferenceElement};
    use fem_mesh::topology::MeshTopology;

    let mesh = space.mesh();
    let mut err2 = 0.0_f64;

    for e in 0..mesh.n_elements() as u32 {
        let re = TriP1;
        let quad = re.quadrature(5);
        let nodes = mesh.element_nodes(e);
        let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();

        let mut phi = vec![0.0_f64; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            ];
            let uh_q: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_q - u_ex(&xp);
            err2 += w * diff * diff;
        }
    }
    err2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n_list: Vec<usize>,
    kappa: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n_list: vec![8, 16, 32],
        kappa: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                if let Some(v) = it.next() {
                    a.n_list = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                }
            }
            "--kappa" => {
                a.kappa = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
            _ => {}
        }
    }
    if a.n_list.is_empty() {
        a.n_list = vec![8, 16, 32];
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
    fn ex7_mixed_bc_coarse_mesh_has_reasonable_error() {
        let result = solve_case(8, 1.0, 1.0);
        assert_eq!(result.n_nodes, 81);
        assert_eq!(result.n_dofs, 81);
        assert!((result.kappa - 1.0).abs() < 1.0e-12);
        assert!((result.solution_scale - 1.0).abs() < 1.0e-12);
        assert!(result.l2_error < 1.5e-3, "coarse-mesh L2 error too large: {}", result.l2_error);
    }

    #[test]
    fn ex7_mixed_bc_refinement_recovers_second_order_convergence() {
        let coarse = solve_case(8, 1.0, 1.0);
        let medium = solve_case(16, 1.0, 1.0);
        let fine = solve_case(32, 1.0, 1.0);
        assert!(medium.l2_error < coarse.l2_error);
        assert!(fine.l2_error < medium.l2_error);
        assert!(convergence_rate(&coarse, &medium) > 1.9);
        assert!(convergence_rate(&medium, &fine) > 1.95);
        assert!(fine.l2_error < 1.0e-4, "fine-mesh L2 error too large: {}", fine.l2_error);
    }

    #[test]
    fn ex7_mixed_bc_manufactured_solution_remains_consistent_across_kappa() {
        let kappa1 = solve_case(16, 1.0, 1.0);
        let kappa2 = solve_case(16, 2.0, 1.0);
        assert!(kappa1.l2_error < 5.0e-4);
        assert!(kappa2.l2_error < 5.0e-4);
        let rel = (kappa2.l2_error - kappa1.l2_error).abs() / kappa1.l2_error.max(1.0e-14);
        assert!(rel < 0.05, "kappa sensitivity too large: k1={} k2={}", kappa1.l2_error, kappa2.l2_error);
    }

    #[test]
    fn ex7_mixed_bc_discrete_solution_is_kappa_invariant() {
        let low = solve_case(16, 0.25, 1.0);
        let unit = solve_case(16, 1.0, 1.0);
        let high = solve_case(16, 4.0, 1.0);

        assert!((low.solution_norm - unit.solution_norm).abs() < 1.0e-12,
            "solution norm should be kappa-invariant: low={} unit={}",
            low.solution_norm, unit.solution_norm);
        assert!((high.solution_norm - unit.solution_norm).abs() < 1.0e-12,
            "solution norm should be kappa-invariant: high={} unit={}",
            high.solution_norm, unit.solution_norm);
        assert!((low.solution_checksum - unit.solution_checksum).abs() < 1.0e-10,
            "solution checksum should be kappa-invariant: low={} unit={}",
            low.solution_checksum, unit.solution_checksum);
        assert!((high.solution_checksum - unit.solution_checksum).abs() < 1.0e-10,
            "solution checksum should be kappa-invariant: high={} unit={}",
            high.solution_checksum, unit.solution_checksum);
        assert!((low.l2_error - unit.l2_error).abs() < 1.0e-12,
            "absolute L2 error should be kappa-invariant: low={} unit={}",
            low.l2_error, unit.l2_error);
        assert!((high.l2_error - unit.l2_error).abs() < 1.0e-12,
            "absolute L2 error should be kappa-invariant: high={} unit={}",
            high.l2_error, unit.l2_error);
    }

    #[test]
    fn ex7_mixed_bc_solution_scales_linearly_with_manufactured_amplitude() {
        let half = solve_case(16, 1.0, 0.5);
        let unit = solve_case(16, 1.0, 1.0);
        let double = solve_case(16, 1.0, 2.0);

        assert!((unit.solution_norm / half.solution_norm - 2.0).abs() < 1.0e-12,
            "solution norm should scale linearly: half={} unit={}",
            half.solution_norm, unit.solution_norm);
        assert!((double.solution_norm / unit.solution_norm - 2.0).abs() < 1.0e-12,
            "solution norm should scale linearly: unit={} double={}",
            unit.solution_norm, double.solution_norm);
        assert!((unit.solution_checksum / half.solution_checksum - 2.0).abs() < 1.0e-12,
            "solution checksum should scale linearly: half={} unit={}",
            half.solution_checksum, unit.solution_checksum);
        assert!((double.solution_checksum / unit.solution_checksum - 2.0).abs() < 1.0e-12,
            "solution checksum should scale linearly: unit={} double={}",
            unit.solution_checksum, double.solution_checksum);
        assert!((unit.l2_error / half.l2_error - 2.0).abs() < 1.0e-12,
            "absolute L2 error should scale linearly: half={} unit={}",
            half.l2_error, unit.l2_error);
        assert!((double.l2_error / unit.l2_error - 2.0).abs() < 1.0e-12,
            "absolute L2 error should scale linearly: unit={} double={}",
            unit.l2_error, double.l2_error);
    }

    #[test]
    fn ex7_mixed_bc_sign_reversed_solution_flips_discrete_state() {
        let positive = solve_case(16, 1.0, 1.0);
        let negative = solve_case(16, 1.0, -1.0);
        assert!((positive.solution_norm - negative.solution_norm).abs() < 1.0e-12);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "solution checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
        assert!((positive.l2_error - negative.l2_error).abs() < 1.0e-12);
    }
}

