//! plor_hex_solve — Low-Order Refined preconditioner for 3-D Poisson on tets.
//!
//! Demonstrates the LOR concept in 3D: solves a P2 system on a tetrahedral cube
//! mesh, using the P1 (vertex-only) matrix as an AMG preconditioner.
//!
//! Analogous to MFEM miniapp `plor-solvers` but extended to 3D.
//!
//! Usage:
//!   cargo run --example plor_hex_solve                          # default n=10
//!   cargo run --example plor_hex_solve -- --n 16                # 16×16×16 cube
//!   cargo run --example plor_hex_solve -- --n 8 --rtol 1e-10   # tighter tolerance

use std::f64::consts::PI;

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_assembly::Assembler;
use fem_amg::{AmgConfig, AmgSolver};
use fem_mesh::SimplexMesh;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

struct LorSolveResult {
    n_p1: usize,
    n_p2: usize,
    amg_p2_levels: usize,
    amg_p1_levels: usize,
    amg_iters: usize,
    jacobi_iters: usize,
    amg_converged: bool,
    l2_error: f64,
    final_residual: f64,
}

fn l2_error_p2(space: &H1Space<SimplexMesh<3>>, uh: &[f64]) -> f64 {
    use fem_element::{lagrange::TetP2, ReferenceElement};
    let mesh = space.mesh();
    let elem = TetP2;
    let qr = elem.quadrature(5);
    let mut err2 = 0.0_f64;
    let mut phi = vec![0.0_f64; elem.n_dofs()];

    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let x3 = mesh.node_coords(nodes[3]);
        let jac = [[x1[0]-x0[0], x2[0]-x0[0], x3[0]-x0[0]],
                   [x1[1]-x0[1], x2[1]-x0[1], x3[1]-x0[1]],
                   [x1[2]-x0[2], x2[2]-x0[2], x3[2]-x0[2]]];
        let det_j = (jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
                    -jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])
                    +jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])).abs();
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        for (qi, xi) in qr.points.iter().enumerate() {
            elem.eval_basis(xi, &mut phi);
            let w = qr.weights[qi] * det_j;
            let xp = [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1] + (x3[0]-x0[0])*xi[2],
                      x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1] + (x3[1]-x0[1])*xi[2],
                      x0[2] + (x1[2]-x0[2])*xi[0] + (x2[2]-x0[2])*xi[1] + (x3[2]-x0[2])*xi[2]];
            let uh_val: f64 = phi.iter().zip(dofs.iter()).map(|(&v, &d)| v * uh[d]).sum();
            let u_ex = (PI * xp[0]).sin() * (PI * xp[1]).sin() * (PI * xp[2]).sin();
            err2 += w * (uh_val - u_ex).powi(2);
        }
    }
    err2.sqrt()
}

fn solve_lor_3d(n: usize) -> LorSolveResult {
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);

    // Source: -∇²u = 3π² sin(πx) sin(πy) sin(πz)
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        3.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin()
    });

    // ── P2 system ─────────────────────────────────────────────────────────
    let space_p2 = H1Space::new(mesh.clone(), 2);
    let n_p2 = space_p2.n_dofs();
    let mut mat_p2 = Assembler::assemble_bilinear(&space_p2, &[&diffusion], 5);
    let mut rhs_p2 = Assembler::assemble_linear(&space_p2, &[&source], 5);
    let bnd_p2 = boundary_dofs(space_p2.mesh(), space_p2.dof_manager(), &[1, 2, 3, 4, 5, 6]);
    apply_dirichlet(&mut mat_p2, &mut rhs_p2, &bnd_p2, &vec![0.0; bnd_p2.len()]);

    // ── P1 (LOR) system ───────────────────────────────────────────────────
    let space_p1 = H1Space::new(mesh.clone(), 1);
    let n_p1 = space_p1.n_dofs();
    let mut mat_p1 = Assembler::assemble_bilinear(&space_p1, &[&diffusion], 3);
    let mut zero_p1 = vec![0.0_f64; n_p1];
    let bnd_p1 = boundary_dofs(space_p1.mesh(), space_p1.dof_manager(), &[1, 2, 3, 4, 5, 6]);
    apply_dirichlet(&mut mat_p1, &mut zero_p1, &bnd_p1, &vec![0.0; bnd_p1.len()]);

    // ── AMG hierarchies ───────────────────────────────────────────────────
    let amg_p2 = AmgSolver::setup(&mat_p2, AmgConfig::default());
    let amg_p2_lvl = amg_p2.n_levels();
    let amg_p1 = AmgSolver::setup(&mat_p1, AmgConfig::default());
    let amg_p1_lvl = amg_p1.n_levels();

    // ── Solve with AMG(P2) ────────────────────────────────────────────────
    let cfg = SolverConfig { rtol: 1e-7, atol: 0.0, max_iter: 800, verbose: false, ..Default::default() };
    let mut u_amg = vec![0.0_f64; n_p2];
    let res_amg = amg_p2.solve(&mat_p2, &rhs_p2, &mut u_amg, &cfg).expect("AMG P2 solve");

    // ── Solve with PCG-Jacobi ─────────────────────────────────────────────
    let mut u_jac = vec![0.0_f64; n_p2];
    let res_jac = solve_pcg_jacobi(&mat_p2, &rhs_p2, &mut u_jac, &cfg).expect("PCG-Jacobi solve");

    let l2 = l2_error_p2(&space_p2, &u_amg);

    LorSolveResult {
        n_p1, n_p2,
        amg_p2_levels: amg_p2_lvl,
        amg_p1_levels: amg_p1_lvl,
        amg_iters: res_amg.iterations,
        jacobi_iters: res_jac.iterations,
        amg_converged: res_amg.converged,
        l2_error: l2,
        final_residual: res_amg.final_residual,
    }
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("=== plor_hex_solve: 3D LOR preconditioner (P2 tet, unit cube) ===");
    println!("  Mesh: {}×{}×{} tets", n, n, n);

    let r = solve_lor_3d(n);

    println!("  P2 DOFs: {}, P1 DOFs: {}", r.n_p2, r.n_p1);
    println!("  AMG levels (P2): {}, AMG levels (P1/LOR): {}", r.amg_p2_levels, r.amg_p1_levels);
    println!("  AMG(P2) CG: {} iters, converged={}, residual={:.3e}", r.amg_iters, r.amg_converged, r.final_residual);
    println!("  PCG-Jacobi: {} iters", r.jacobi_iters);
    println!("  L² error: {:.6e}", r.l2_error);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plor_lor_p1_fewer_dofs_than_p2() {
        let r = solve_lor_3d(4);
        assert!(r.n_p1 < r.n_p2, "P1 DOFs {} >= P2 DOFs {}", r.n_p1, r.n_p2);
    }

    #[test]
    fn plor_amg_converges_3d() {
        let r = solve_lor_3d(6);
        assert!(r.amg_converged, "AMG did not converge");
        assert!(r.l2_error < 0.01, "L² error too large: {:.6e}", r.l2_error);
    }

    #[test]
    fn plor_amg_faster_than_jacobi() {
        let r = solve_lor_3d(6);
        assert!(r.amg_converged);
        assert!(r.amg_iters < r.jacobi_iters,
            "AMG({}) should beat Jacobi({})", r.amg_iters, r.jacobi_iters);
    }
}
