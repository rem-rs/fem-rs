//! DPG miniapp baseline
//!
//! Solves Poisson on [0,1]^2 with homogeneous Dirichlet BCs.

use std::f64::consts::PI;

use fem_assembly::{Assembler, GridFunction, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_mesh::SimplexMesh;
use fem_solver::{SolverConfig, solve_pcg_jacobi};
use fem_space::{H1Space, constraints::{apply_dirichlet, boundary_dofs}, fe_space::FESpace};

fn main() {
    let args = parse_args();

    println!("=== fem-rs DPG Poisson miniapp ===");
    for &n in &[args.n, args.n * 2] {
        let res = solve_dpg(n);
        println!("  n={:3}  DOFs={:5}  L2 err={:.3e}", n, res.ndofs, res.l2_error);
    }
    println!();
    println!("Note: this is a stable primal-DPG proxy baseline on top of existing H1 infrastructure.");
    println!("      Full DPG with enriched broken test space and trace unknowns is still pending.");
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
}

fn parse_args() -> Args {
    let mut n = 8usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "--n" {
            n = it.next().unwrap_or_default().parse().unwrap_or(8);
        }
    }
    Args { n }
}

struct DpgResult {
    ndofs: usize,
    l2_error: f64,
}

fn exact_u(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn source_f(x: &[f64]) -> f64 {
    2.0 * PI * PI * exact_u(x)
}

fn solve_dpg(n: usize) -> DpgResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();

    let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 4);
    let src = DomainSourceIntegrator::new(source_f);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 4);

    let dm = space.dof_manager();
    let bdry = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bdry_vals: Vec<f64> = bdry.iter().map(|_| 0.0).collect();
    apply_dirichlet(&mut mat, &mut rhs, &bdry, &bdry_vals);

    let mut sol_vec = vec![0.0_f64; ndofs];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };
    solve_pcg_jacobi(&mat, &rhs, &mut sol_vec, &cfg).expect("DPG PCG solve failed");

    let gf = GridFunction::new(&space, sol_vec);
    let l2_error = gf.compute_l2_error(&exact_u, 4);

    DpgResult { ndofs, l2_error }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpg_poisson_l2_error_decreases_with_refinement() {
        let r4 = solve_dpg(4);
        let r8 = solve_dpg(8);
        assert!(r4.l2_error.is_finite(), "coarse L2 error is finite");
        assert!(r8.l2_error.is_finite(), "fine L2 error is finite");
        assert!(
            r8.l2_error < r4.l2_error,
            "L2 error should decrease: coarse={}, fine={}",
            r4.l2_error,
            r8.l2_error
        );
    }

    #[test]
    fn dpg_poisson_l2_error_is_small() {
        let res = solve_dpg(12);
        assert!(res.l2_error < 2.0e-2, "L2 error too large: {}", res.l2_error);
    }
}
