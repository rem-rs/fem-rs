//! # MFEM Example 7 — Screened Poisson on a sphere (adapted for 2D)
//!
//! Solves the screened Poisson equation with mixed boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) + u = f    in Ω
//!                u = g    on ∂Ω
//! ```
//!
//! using `f(x) = 7 x₁ x₂ / (x₁² + x₂²)` and exact solution `u(x) = x₁ x₂ / (x₁² + x₂²)`,
//! matching the spirit of MFEM Example 7 (which runs on a sphere surface in 3D).
//!
//! Reference: `mfem/ex7.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex7_neumann_mixed_bc
//! cargo run --example mfem_ex7_neumann_mixed_bc -- -m ../data/star.mesh
//! ```
//!
//! ## Structure
//! Main: MFEM-style problem with mesh loading.
//! Tests: Manufactured-solution verification with `u = x(1-x)y(1-y)`.

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

fn main() {
    let args = parse_args();

    println!("=== MFEM Example 7: Screened Poisson (2D adaptation) ===");

    // ─── 1. Mesh ─────────────────────────────────────────────────────────────
    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        println!("  Mesh file: {path}");
        read_mfem_file(path)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    } else {
        let n = args.n.unwrap_or(32);
        println!("  Unit-square tri mesh, n = {n}");
        SimplexMesh::<2>::unit_square_tri(n)
    };
    println!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ─── 2. H¹ space ─────────────────────────────────────────────────────────
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();
    println!("  H1Space: {ndofs} DOFs, order 1");

    // ─── 3. Assemble LHS:  diffusion + mass  (matching MFEM ex7) ──────────
    let mat = Assembler::assemble_bilinear(
        &space,
        &[
            &DiffusionIntegrator { kappa: 1.0 } as _,
            &MassIntegrator { rho: 1.0 } as _,
        ],
        3,
    );

    // ─── 4. Assemble RHS: ∫ f v dx   f(x) = 7 x₁ x₂ / (x₁² + x₂²) ─────
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        let r2 = x[0] * x[0] + x[1] * x[1];
        7.0 * x[0] * x[1] / r2.max(f64::MIN_POSITIVE)
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    // ─── 5. Dirichlet BC on all boundaries: u = x₁ x₂ / (x₁² + x₂²) ────
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals: Vec<f64> = bnd
        .iter()
        .map(|&dof| {
            let x = dm.dof_coord(dof);
            let r2 = x[0] * x[0] + x[1] * x[1];
            x[0] * x[1] / r2.max(f64::MIN_POSITIVE)
        })
        .collect();
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
    println!("  Dirichlet BC on {} DOFs", bnd.len());

    // ─── 6. Solve ─────────────────────────────────────────────────────────────
    let mut u = vec![0.0_f64; ndofs];
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");

    let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    let checksum = u
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64 + 1.0) * v)
        .sum::<f64>();

    println!("  Solve: {} PCG iterations, final residual = {:.3e}", res.iterations, res.final_residual);
    println!("  ||u||₂ = {:.6e}", u_norm);
    println!("  checksum = {:.8e}", checksum);
    println!("  Done.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: Option<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                a.mesh = it.next();
            }
            "--n" => {
                a.n = it.next().and_then(|s| s.parse().ok());
            }
            _ => {}
        }
    }
    a
}

// ─── Tests: manufactured-solution (MMS) verification ──────────────────────

#[cfg(test)]
mod tests {
    use fem_assembly::assembler::face_dofs_p1;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, NeumannIntegrator};
    use fem_assembly::Assembler;
    use fem_mesh::SimplexMesh;
    use fem_solver::{solve_pcg_jacobi, SolverConfig};
    use fem_space::constraints::{apply_dirichlet, boundary_dofs};
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    // Original MMS: u = x(1-x) y(1-y)
    fn exact_scaled(x: &[f64], scale: f64) -> f64 {
        scale * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    }

    fn rhs_scaled(x: &[f64], kappa: f64, scale: f64) -> f64 {
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

    fn solve_mms(n: usize, kappa: f64, solution_scale: f64) -> SolveResult {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let ndofs = space.n_dofs();

        // Stiffness κ ∇u·∇v
        let mut mat = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa }],
            3,
        );

        // Volume RHS f v
        let src = DomainSourceIntegrator::new(|x: &[f64]| rhs_scaled(x, kappa, solution_scale));
        let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

        // Neumann on tags 2 (right) and 3 (top)
        let neumann_g = NeumannIntegrator::new(|x: &[f64], n: &[f64]| {
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
            &[2, 3],
            3,
        );
        for i in 0..ndofs {
            rhs[i] += neumann_rhs[i];
        }

        // Dirichlet on tags 1 (bottom) and 4 (left)
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 4]);
        let bnd_vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        // Solve
        let mut u = vec![0.0_f64; ndofs];
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 10_000,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");

        // L² error
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
            let det_j =
                ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();
            let mut phi = vec![0.0_f64; re.n_dofs()];
            for (qi, xi) in quad.points.iter().enumerate() {
                re.eval_basis(xi, &mut phi);
                let w = quad.weights[qi] * det_j;
                let xp = [
                    x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                    x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
                ];
                let uh_q: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * u[di]).sum();
                let diff = uh_q - exact_scaled(&xp, solution_scale);
                err2 += w * diff * diff;
            }
        }

        let l2 = err2.sqrt();
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

    fn convergence_rate(coarse: &SolveResult, fine: &SolveResult) -> f64 {
        let h_coarse = 1.0 / coarse.n as f64;
        let h_fine = 1.0 / fine.n as f64;
        (fine.l2_error / coarse.l2_error).ln() / (h_fine / h_coarse).ln()
    }

    // ─── Tests ───────────────────────────────────────────────────────────────

    #[test]
    fn ex7_mms_coarse_mesh_has_reasonable_error() {
        let result = solve_mms(8, 1.0, 1.0);
        assert_eq!(result.n_nodes, 81);
        assert_eq!(result.n_dofs, 81);
        assert!(result.l2_error < 1.5e-3,
            "coarse-mesh L2 error too large: {}", result.l2_error);
    }

    #[test]
    fn ex7_mms_refinement_recovers_second_order_convergence() {
        let coarse = solve_mms(8, 1.0, 1.0);
        let medium = solve_mms(16, 1.0, 1.0);
        let fine = solve_mms(32, 1.0, 1.0);
        assert!(medium.l2_error < coarse.l2_error);
        assert!(fine.l2_error < medium.l2_error);
        assert!(convergence_rate(&coarse, &medium) > 1.9);
        assert!(convergence_rate(&medium, &fine) > 1.95);
        assert!(fine.l2_error < 1.0e-4, "fine-mesh L2 error too large: {}", fine.l2_error);
    }

    #[test]
    fn ex7_mms_kappa_invariance() {
        let k1 = solve_mms(16, 1.0, 1.0);
        let k2 = solve_mms(16, 2.0, 1.0);
        assert!(k1.l2_error < 5.0e-4);
        assert!(k2.l2_error < 5.0e-4);
        let rel = (k2.l2_error - k1.l2_error).abs() / k1.l2_error.max(1.0e-14);
        assert!(rel < 0.05, "kappa sensitivity too large: k1={} k2={}", k1.l2_error, k2.l2_error);
    }

    #[test]
    fn ex7_mms_solution_is_kappa_invariant() {
        let low = solve_mms(16, 0.25, 1.0);
        let unit = solve_mms(16, 1.0, 1.0);
        let high = solve_mms(16, 4.0, 1.0);
        assert!((low.solution_norm - unit.solution_norm).abs() < 1.0e-12);
        assert!((high.solution_norm - unit.solution_norm).abs() < 1.0e-12);
        assert!((low.solution_checksum - unit.solution_checksum).abs() < 1.0e-10);
        assert!((high.solution_checksum - unit.solution_checksum).abs() < 1.0e-10);
        assert!((low.l2_error - unit.l2_error).abs() < 1.0e-12);
        assert!((high.l2_error - unit.l2_error).abs() < 1.0e-12);
    }

    #[test]
    fn ex7_mms_linear_scaling_with_amplitude() {
        let half = solve_mms(16, 1.0, 0.5);
        let unit = solve_mms(16, 1.0, 1.0);
        let double_ = solve_mms(16, 1.0, 2.0);
        for (actual, expected) in [
            (unit.solution_norm / half.solution_norm, 2.0),
            (double_.solution_norm / unit.solution_norm, 2.0),
            (unit.solution_checksum / half.solution_checksum, 2.0),
            (double_.solution_checksum / unit.solution_checksum, 2.0),
            (unit.l2_error / half.l2_error, 2.0),
            (double_.l2_error / unit.l2_error, 2.0),
        ] {
            assert!((actual - expected).abs() < 1.0e-12,
                "scale ratio mismatch: {:.12} vs {expected}", actual);
        }
    }

    #[test]
    fn ex7_mms_sign_reversal_flips_state() {
        let pos = solve_mms(16, 1.0, 1.0);
        let neg = solve_mms(16, 1.0, -1.0);
        assert!((pos.solution_norm - neg.solution_norm).abs() < 1.0e-12);
        assert!((pos.solution_checksum + neg.solution_checksum).abs() < 1.0e-10);
        assert!((pos.l2_error - neg.l2_error).abs() < 1.0e-12);
    }

    #[test]
    fn ex7_mms_dof_count_matches_formula() {
        for n in [4usize, 8, 16] {
            let r = solve_mms(n, 1.0, 1.0);
            let expected = (n + 1) * (n + 1);
            assert_eq!(r.n_nodes, expected, "n={}: expected {expected} nodes, got {}", r.n_nodes);
            assert_eq!(r.n_dofs, expected, "n={}: expected {expected} DOFs, got {}", r.n_dofs);
        }
    }

    #[test]
    fn ex7_mms_fine_mesh_accuracy() {
        let result = solve_mms(32, 1.0, 1.0);
        assert!(result.l2_error < 1.0e-4,
            "fine mesh L2 error too large: {}", result.l2_error);
        let coarse = solve_mms(16, 1.0, 1.0);
        let rate = coarse.l2_error / result.l2_error;
        assert!(rate > 3.5, "expected ~4x error reduction, got {:.2}x", rate);
    }
}
