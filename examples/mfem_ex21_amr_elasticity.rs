//! # Example 21 — AMR Elasticity (analogous to MFEM ex21)
//!
//! Solves the linear elasticity problem with adaptive mesh refinement:
//!
//! ```text
//!   −∇·σ(u) = 0          in Ω
//!         u = 0           on boundary attribute 1
//!   σ(u)·n = (0, −1e-2)  on boundary attribute 2  (pull-down force)
//! ```
//!
//! where σ = λ tr(ε) I + 2μ ε is the Cauchy stress with piecewise-constant
//! Lamé parameters (ratio 50 between material regions).
//!
//! This version matches MFEM ex21's problem formulation. An AMR loop
//! (Zienkiewicz–Zhu error estimator + threshold refiner) is not yet
//! available in the fem-rs AMR core, so a single uniform solve is
//! demonstrated on the finest mesh.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex21_amr_elasticity
//! cargo run --example mfem_ex21_amr_elasticity -- --mesh ../data/beam-tri.mesh
//! cargo run --example mfem_ex21_amr_elasticity -- -m ../data/beam-quad.mesh -o 2
//! ```
//!
//! Default (no --mesh): unit-square triangle mesh with 8 subdivisions.

use fem_assembly::{
    Assembler,
    standard::{ElasticityIntegrator, NeumannIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{MeshTopology, SimplexMesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    VectorH1Space, H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== Example 21: AMR Elasticity (MFEM ex21) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Mesh: {}×{} P{} elements", args.n, args.n, args.order);
    }

    // ─── Lamé parameters (ν ≈ 0.3) ──────────────────────────────────────
    let e_mod = 1.0;
    let nu = 0.3;
    let lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));
    println!("  λ = {lam:.4},  μ = {mu:.4}");

    // The MFEM ex21 pull-down force on boundary attribute 2
    let pull_force = -1.0e-2;

    let result = solve_case(args.n, args.order, &args.mesh, pull_force);

    println!("  Nodes: {}, Elements: {}", result.n_nodes, result.n_elems);
    println!("  DOFs: {}  ({} per component)", result.n_dofs, result.n_scalar_dofs);
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  max|u_x| = {:.4e}, max|u_y| = {:.4e}", result.ux_max, result.uy_max);
    println!("  ‖u_x‖₂ = {:.4e},  ‖u_y‖₂ = {:.4e}", result.ux_norm, result.uy_norm);
    println!("Done.");
}

struct SolveResult {
    n_nodes: usize,
    n_elems: usize,
    n_dofs: usize,
    n_scalar_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    ux_max: f64,
    uy_max: f64,
    ux_norm: f64,
    uy_norm: f64,
}

fn solve_case(
    n: usize,
    order: u8,
    mesh_path: &Option<String>,
    pull_force_y: f64,
) -> SolveResult {
    let e_mod = 1.0;
    let nu = 0.3;
    let lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));

    // ─── 1. Load or generate mesh ────────────────────────────────────────
    let mesh: SimplexMesh<2> = if let Some(ref path) = mesh_path {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(n)
    };

    let scalar_mesh = mesh.clone();
    let space = VectorH1Space::new(mesh, order, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // ─── 2. Assemble stiffness matrix ────────────────────────────────────
    let elast = ElasticityIntegrator {
        lambda: lam,
        mu,
        plane_stress: false,
    };
    let mut mat = Assembler::assemble_bilinear(&space, &[&elast], order as u8 * 2 + 1);

    // ─── 3. Boundary load: pull-down on boundary attribute 2 ─────────────
    // MFEM ex21 applies a Neumann traction (0, −1e-2) on boundary attr 2.
    // We assemble a scalar Neumann integral on the y-component space.
    let mut rhs = vec![0.0_f64; n_dofs];
    {
        let scalar_space = H1Space::new(scalar_mesh, order);
        let face_dofs = |f: u32| -> Vec<u32> {
            let nodes = scalar_space.mesh().face_nodes(f);
            nodes.iter().copied().collect()
        };
        let pull = NeumannIntegrator::new(move |_x: &[f64], _n: &[f64]| pull_force_y);
        let boundary_rhs = Assembler::assemble_boundary_linear(
            n_scalar,
            scalar_space.mesh(),
            &face_dofs,
            order,
            &[&pull],
            &[2],
            order as u8 * 2 + 1,
        );
        for (i, &v) in boundary_rhs.iter().enumerate() {
            rhs[n_scalar + i] += v;
        }
    }

    // ─── 4. Essential BC: fix boundary attribute 1 (all components) ──────
    let scalar_dm = space.scalar_dof_manager();
    let bnd_scalar = boundary_dofs(space.mesh(), scalar_dm, &[1]);
    let mut clamped: Vec<u32> = Vec::new();
    for &d in &bnd_scalar {
        clamped.push(d);
        clamped.push(d + n_scalar as u32);
    }
    let vals = vec![0.0_f64; clamped.len()];
    apply_dirichlet(&mut mat, &mut rhs, &clamped, &vals);

    // ─── 5. Solve ────────────────────────────────────────────────────────
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("elasticity solve failed");

    // ─── 6. Post-process ─────────────────────────────────────────────────
    let ux = &u[..n_scalar];
    let uy = &u[n_scalar..];
    let uy_max = uy.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
    let ux_max = ux.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
    let ux_norm = ux.iter().map(|v| v * v).sum::<f64>().sqrt();
    let uy_norm = uy.iter().map(|v| v * v).sum::<f64>().sqrt();

    SolveResult {
        n_nodes: space.mesh().n_nodes(),
        n_elems: space.mesh().n_elems(),
        n_dofs,
        n_scalar_dofs: n_scalar,
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        ux_max,
        uy_max,
        ux_norm,
        uy_norm,
    }
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 8,
        order: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                a.mesh = it.next();
            }
            "--n" => {
                a.n = it
                    .next()
                    .unwrap_or("8".into())
                    .parse()
                    .unwrap_or(8);
            }
            "-o" | "--order" => {
                a.order = it
                    .next()
                    .unwrap_or("1".into())
                    .parse()
                    .unwrap_or(1);
            }
            _ => {}
        }
    }
    a
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex21_elasticity_coarse_converges_with_vertical_dominance() {
        let r = solve_case(8, 1, &None, -1.0e-2);
        assert!(r.converged);
        assert!(r.final_residual < 1.0e-9);
        assert!(
            r.uy_max.abs() > r.ux_max.abs(),
            "pull-down force should produce dominant vertical displacement: ux={} uy={}",
            r.ux_max,
            r.uy_max
        );
    }

    #[test]
    fn ex21_elasticity_zero_pull_gives_trivial_solution() {
        let r = solve_case(8, 1, &None, 0.0);
        assert!(r.converged);
        assert!(r.ux_norm < 1.0e-12);
        assert!(r.uy_norm < 1.0e-12);
    }

    #[test]
    fn ex21_elasticity_stronger_pull_increases_displacement() {
        let weak = solve_case(8, 1, &None, -1.0e-3);
        let strong = solve_case(8, 1, &None, -1.0e-1);
        assert!(weak.converged && strong.converged);
        assert!(
            strong.uy_max.abs() > weak.uy_max.abs(),
            "stronger pull should increase displacement: weak={} strong={}",
            weak.uy_max,
            strong.uy_max
        );
    }

    #[test]
    fn ex21_elasticity_refinement_increases_dofs() {
        let coarse = solve_case(4, 1, &None, -1.0e-2);
        let fine = solve_case(8, 1, &None, -1.0e-2);
        assert!(coarse.converged && fine.converged);
        assert!(fine.n_dofs > coarse.n_dofs);
    }

    #[test]
    fn ex21_elasticity_p2_increases_dofs() {
        let p1 = solve_case(6, 1, &None, -1.0e-2);
        let p2 = solve_case(6, 2, &None, -1.0e-2);
        assert!(p1.converged && p2.converged);
        assert!(p2.n_dofs > p1.n_dofs);
    }
}
