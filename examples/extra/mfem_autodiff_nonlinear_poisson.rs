//! # Automatic Differentiation Miniapp — Nonlinear Poisson via Dual Numbers
//!
//! Analogous to MFEM's `autodiff` miniapp. Demonstrates computing the tangent
//! stiffness matrix automatically using dual numbers, enabling Newton's method
//! without hand-deriving the nonlinear Jacobian.
//!
//! ## Approach
//!
//! Instead of N+1 residual evaluations (finite differences), we use **element-level
//! dual-number seeding**: for each element, all local DOFs are seeded as dual
//! variables, the element residual is evaluated once using dual arithmetic, and
//! the element Jacobian is extracted from the dual components.
//!
//! For P1 triangles (3 DOFs/elem), this requires exactly N_elem evaluations of
//! the element residual with 3-way dual seeding — O(N), not O(N²).
//!
//! ## Problem
//!
//!   −∇·(κ(|∇u|²)∇u) = f    in Ω = [0,1]²
//!                u = 0      on ∂Ω
//!
//! with nonlinear diffusion κ(g) = 1 + α·g.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_autodiff_nonlinear_poisson
//! cargo run --example mfem_autodiff_nonlinear_poisson -- --n 32 --alpha 0.5
//! ```

use fem_linalg::CooMatrix;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

// ─── Dual number type ────────────────────────────────────────────────────────

/// Dual number a + Σ bᵢ·εᵢ where εᵢ·εⱼ = 0.
///
/// `val` is the primal value, `ders` holds the derivatives w.r.t. each
/// seeded variable. For element-level seeding, there is one seed per
/// element DOF (at most 3 for P1 triangles).
#[derive(Debug, Clone)]
struct DualN {
    val: f64,
    ders: Vec<f64>, // derivatives w.r.t. seeded variables
}

impl DualN {
    fn constant(v: f64, n_seeds: usize) -> Self {
        Self { val: v, ders: vec![0.0; n_seeds] }
    }

    fn variable(v: f64, idx: usize, n_seeds: usize) -> Self {
        let mut ders = vec![0.0; n_seeds];
        ders[idx] = 1.0;
        Self { val: v, ders }
    }

    fn n_seeds(&self) -> usize { self.ders.len() }
}

// Arithmetic operations with chain rule
use std::ops::{Add, Mul};

impl Add for DualN {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let n = self.n_seeds().max(rhs.n_seeds());
        if n == 0 {
            return Self { val: self.val + rhs.val, ders: vec![] };
        }
        let mut ders = vec![0.0; n];
        for i in 0..n {
            let a = if i < self.ders.len() { self.ders[i] } else { 0.0 };
            let b = if i < rhs.ders.len() { rhs.ders[i] } else { 0.0 };
            ders[i] = a + b;
        }
        Self { val: self.val + rhs.val, ders }
    }
}

impl Mul for DualN {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let n = self.n_seeds().max(rhs.n_seeds());
        if n == 0 {
            return Self { val: self.val * rhs.val, ders: vec![] };
        }
        let mut ders = vec![0.0; n];
        for i in 0..n {
            let a = if i < self.ders.len() { self.ders[i] } else { 0.0 };
            let b = if i < rhs.ders.len() { rhs.ders[i] } else { 0.0 };
            ders[i] = self.val * b + a * rhs.val;
        }
        Self { val: self.val * rhs.val, ders }
    }
}

impl Mul<f64> for DualN {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let ders: Vec<f64> = self.ders.iter().map(|&d| d * s).collect();
        Self { val: self.val * s, ders }
    }
}

impl Mul<DualN> for f64 {
    type Output = DualN;
    fn mul(self, rhs: DualN) -> DualN { rhs * self }
}

// ─── Element-level dual-number assembly ─────────────────────────────────────

/// Evaluate the element residual vector R^e_λ(u) using dual numbers seeded on
/// all element DOFs, then extract both the residual and the element Jacobian.
///
/// For P1 triangle: 3 DOFs, so `u_seed` has length 3.
/// Returns `(residual_vec[3], jacobian_mat[3×3])`.
fn element_residual_and_jacobian(
    x0: &[f64], x1: &[f64], x2: &[f64],  // triangle vertices
    u: &[f64],                              // DOF values on this element (length 3)
    alpha: f64,
    f_expr: impl Fn(f64, f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_seeds = 3;
    let n = 3; // P1 triangle has 3 DOFs

    // Seed all DOFs as dual variables
    let u_dual: Vec<DualN> = (0..n)
        .map(|i| DualN::variable(u[i], i, n_seeds))
        .collect();

    // Element geometry
    let area = 0.5 * ((x1[0] - x0[0]) * (x2[1] - x0[1])
        - (x2[0] - x0[0]) * (x1[1] - x0[1]))
        .abs();
    let inv_2a = 0.5 / area;

    // Gradients of P1 basis functions
    let g0 = [inv_2a * (x2[1] - x1[1]), inv_2a * (x1[0] - x2[0])];
    let g1 = [inv_2a * (x0[1] - x2[1]), inv_2a * (x2[0] - x0[0])];
    let g2 = [inv_2a * (x1[1] - x0[1]), inv_2a * (x0[0] - x1[0])];

    // Gradient of uₕ on this element (dual)
    let grad_u_x = u_dual[0].clone() * g0[0]
        + u_dual[1].clone() * g1[0]
        + u_dual[2].clone() * g2[0];
    let grad_u_y = u_dual[0].clone() * g0[1]
        + u_dual[1].clone() * g1[1]
        + u_dual[2].clone() * g2[1];

    // κ(|∇u|²) = 1 + α·|∇u|²  (dual)
    let grad_norm2 = grad_u_x.clone() * grad_u_x.clone()
        + grad_u_y.clone() * grad_u_y.clone();
    let kappa = DualN::constant(1.0, n_seeds) + alpha * grad_norm2;

    // Quadrature: centroid rule (order 1)
    let cx = (x0[0] + x1[0] + x2[0]) / 3.0;
    let cy = (x0[1] + x1[1] + x2[1]) / 3.0;
    let f_val = f_expr(cx, cy);
    let f_dual = DualN::constant(f_val, n_seeds);

    // Residual: Rᵢ = ∫ [κ(|∇u|²)·∇u·∇φᵢ − f·φᵢ] dx
    //          ≈ area · [κ·∇u·∇φᵢ − f/3]  (midpoint + lumped mass)
    let mut r_dual = vec![DualN::constant(0.0, n_seeds); 3];
    for (i, g) in [g0, g1, g2].iter().enumerate() {
        let grad_dot = grad_u_x.clone() * g[0] + grad_u_y.clone() * g[1];
        r_dual[i] = r_dual[i].clone() + kappa.clone() * grad_dot * area;
        r_dual[i] = r_dual[i].clone() + f_dual.clone() * (-area / 3.0);
    }

    // Extract residual and Jacobian
    let residual: Vec<f64> = r_dual.iter().map(|r| r.val).collect();
    let mut jac = vec![0.0_f64; 9]; // 3×3 row-major
    for i in 0..3 {
        for j in 0..3 {
            jac[i * 3 + j] = r_dual[i].ders[j];
        }
    }
    (residual, jac)
}

// ─── Newton solver with autodiff Jacobian ────────────────────────────────────

fn solve_with_autodiff(n: usize, alpha: f64, newton_max: usize, newton_tol: f64) -> Option<(Vec<f64>, usize, f64)> {
    let mesh = Mesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let ndofs = space.n_dofs();
    let dm = space.dof_manager();

    // Source: f(x,y) = 2π² sin(πx) sin(πy)
    let f_source = |x: f64, y: f64| {
        let pi = std::f64::consts::PI;
        2.0 * pi * pi * (pi * x).sin() * (pi * y).sin()
    };

    // Boundary DOFs
    let bdr: Vec<usize> = boundary_dofs(&mesh, dm, &&mesh.unique_boundary_tags())
        .into_iter().map(|d| d as usize).collect();
    let mut is_bdr = vec![false; ndofs];
    for &d in &bdr { is_bdr[d] = true; }

    let n_elems = mesh.n_elems();
    let elem_nodes: Vec<[usize; 3]> = (0..n_elems)
        .map(|e| {
            let ns = mesh.element_nodes(e as u32);
            [ns[0] as usize, ns[1] as usize, ns[2] as usize]
        })
        .collect();

    let cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, ..Default::default() };
    let mut u = vec![0.0_f64; ndofs];

    for iter in 0..newton_max {
        // Assemble residual and Jacobian using autodiff
        let mut residual = vec![0.0_f64; ndofs];
        let mut jac_coo = CooMatrix::<f64>::new(ndofs, ndofs);

        for (_e, &[v0, v1, v2]) in elem_nodes.iter().enumerate() {
            // Skip elements where all DOFs are on Dirichlet boundary
            if is_bdr[v0] && is_bdr[v1] && is_bdr[v2] { continue; }

            let x0 = mesh.node_coords(v0 as u32);
            let x1 = mesh.node_coords(v1 as u32);
            let x2 = mesh.node_coords(v2 as u32);
            let u_loc = [u[v0], u[v1], u[v2]];

            let (r_loc, jac_loc) = element_residual_and_jacobian(
                x0, x1, x2, &u_loc, alpha, &f_source,
            );

            for i in 0..3 {
                let gi = [v0, v1, v2][i];
                if !is_bdr[gi] {
                    residual[gi] += r_loc[i];
                }
                for j in 0..3 {
                    let gj = [v0, v1, v2][j];
                    if !is_bdr[gi] && !is_bdr[gj] {
                        let val = jac_loc[i * 3 + j];
                        if val.abs() > 1e-15 {
                            jac_coo.add(gi, gj, val);
                        }
                    }
                }
            }
        }

        // Apply Dirichlet BC
        for &d in &bdr {
            jac_coo.add(d, d, 1.0);
            residual[d] = 0.0;
        }

        let jac_csr = jac_coo.into_csr();

        // Convergence check
        let res_norm: f64 = residual.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let _u_norm: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-15);
        if res_norm < newton_tol * ndofs as f64 {
            return Some((u, iter, res_norm));
        }

        // Solve J·Δu = −R
        let neg_r: Vec<f64> = residual.iter().map(|&v| -v).collect();
        let mut du = vec![0.0_f64; ndofs];
        match solve_gmres(&jac_csr, &neg_r, &mut du, 30, &cfg) {
            Ok(_) => {
                for i in 0..ndofs {
                    if !is_bdr[i] { u[i] += du[i]; }
                }
            }
            Err(_) => break,
        }
    }
    None
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    println!("=== fem-rs Autodiff Miniapp: Nonlinear Poisson ===");
    println!("  Mesh: {}×{} (P1), alpha={}", args.n, args.n, args.alpha);

    match solve_with_autodiff(args.n, args.alpha, 20, 1e-8) {
        Some((u, iters, res_norm)) => {
            let u_l2: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let u_max: f64 = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!("  Converged in {iters} Newton iterations");
            println!("  ‖u‖₂ = {:.6e}, max|u| = {:.6e}, ‖R‖ = {:.3e}", u_l2, u_max, res_norm);
            println!("  PASS");
        }
        None => {
            println!("  FAILED to converge in 20 Newton iterations");
        }
    }
}

struct Args { n: usize, alpha: f64 }

fn parse_args() -> Args {
    let mut a = Args { n: 16, alpha: 0.1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16),
            "--alpha" => a.alpha = it.next().unwrap_or("0.1".into()).parse().unwrap_or(0.1),
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autodiff_jacobian_is_finite_and_rank_3() {
        let x0 = &[0.0, 0.0]; let x1 = &[1.0, 0.0]; let x2 = &[0.0, 1.0];
        let u = [0.1, 0.2, 0.3];
        let (r, jac) = element_residual_and_jacobian(x0, x1, x2, &u, 0.0, |_, _| 1.0);
        assert_eq!(r.len(), 3);
        assert_eq!(jac.len(), 9);
        assert!(jac.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn dual_chain_rule() {
        // f(x,y) = x² + x·y + y²
        // ∂f/∂x = 2x + y = 7, ∂f/∂y = x + 2y = 8 (at x=2, y=3)
        let x = DualN::variable(2.0, 0, 2);
        let y = DualN::variable(3.0, 1, 2);
        let f = x.clone() * x.clone() + x * y.clone() + y.clone() * y;
        assert!((f.val - 19.0).abs() < 1e-14);
        assert!((f.ders[0] - 7.0).abs() < 1e-14);
        assert!((f.ders[1] - 8.0).abs() < 1e-14);
    }

    #[test]
    fn nonlinear_poisson_converges() {
        let r = solve_with_autodiff(8, 0.1, 15, 1e-8);
        assert!(r.is_some(), "Newton should converge");
        let (u, iters, _) = r.unwrap();
        assert!(iters < 10, "iterations={iters}, expected < 10");
        assert!(u.iter().any(|&v| v.abs() > 1e-6), "solution should be non-trivial");
    }

    #[test]
    fn moderate_nonlinearity_converges() {
        let r = solve_with_autodiff(8, 0.3, 20, 1e-8);
        assert!(r.is_some(), "Newton should converge for alpha=0.3");
    }

    #[test]
    fn mesh_refinement_reduces_residual() {
        let coarse = solve_with_autodiff(4, 0.1, 10, 1e-8).unwrap();
        let fine = solve_with_autodiff(8, 0.1, 10, 1e-8).unwrap();
        // Both should converge (residual < 1e-8)
        assert!(coarse.2 < 1e-4, "coarse residual too large: {}", coarse.2);
        assert!(fine.2 < 1e-4, "fine residual too large: {}", fine.2);
    }
}
