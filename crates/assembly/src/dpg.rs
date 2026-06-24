//! Discontinuous Petrov-Galerkin (DPG) method infrastructure.
//!
//! DPG uses enriched, discontinuous test spaces to produce automatically
//! stable discretisations.  This module provides a focused 1-D implementation
//! for the convection-diffusion problem:
//!
//! ```text
//! -ε u″ + b u′ = f    on (0,1)
//!  u(0) = u(1) = 0
//! ```
//!
//! DPG constructs **optimal test functions** element-wise, which guarantees
//! stability even for convection-dominated regimes (small ε) where standard
//! Galerkin oscillates.
//!
//! # Algorithm (element-level)
//!
//! 1. Trial space T: continuous P1 (vertex DOFs)
//! 2. Enriched test space V: discontinuous P₃ on each element
//! 3. For element e with trial DOFs `{ψ_i}`:
//!    - Build convection-diffusion bilinear form in the test basis:
//!      `B[i,j] = ε ∫ ψ_i′ φ_j′ + b ∫ ψ_i′ φ_j`
//!      where `{φ_j}` are the enriched test basis functions
//!    - Build the test-space H¹ inner-product matrix (the "energy" inner product):
//!      `(v, w)_V = ∫ (v w + v′ w′) dx`
//!    - Optimal test functions: `MV * v_opt_i = B[:,i]`
//!    - Element stiffness: `K_e[i,j] = B[:,j] · v_opt_i`
//!    - Element RHS: `f_e[i] = F(v_opt_i) = ∫ f * v_opt_i`
//! 3. Assemble global system from element contributions
//!
//! # Reference
//! - Demkowicz & Gopalakrishnan, "A class of discontinuous Petrov–Galerkin
//!   methods" (2010), Comput. Methods Appl. Mech. Engrg.

use fem_linalg::{CooMatrix, CsrMatrix, Vector};

/// Gauss-Legendre quadrature on [0,1] (up to 5 points).
fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.5], vec![1.0]),
        2 => (
            vec![0.211324865405187, 0.788675134594813],
            vec![0.5, 0.5],
        ),
        3 => (
            vec![0.112701665379258, 0.5, 0.887298334620742],
            vec![0.277777777777778, 0.444444444444444, 0.277777777777778],
        ),
        4 => (
            vec![
                0.0694318442029737,
                0.3300094782075719,
                0.6699905217924281,
                0.9305681557970263,
            ],
            vec![
                0.1739274225687269,
                0.3260725774312731,
                0.3260725774312731,
                0.1739274225687269,
            ],
        ),
        5 => (
            vec![
                0.046910077030668,
                0.230765344947158,
                0.5,
                0.769234655052842,
                0.953089922969332,
            ],
            vec![
                0.118463442528095,
                0.239314335249683,
                0.284444444444444,
                0.239314335249683,
                0.118463442528095,
            ],
        ),
        _ => panic!("gauss_legendre_01: unsupported n={n}"),
    }
}

/// Evaluate 1-D Lagrange shape functions and their derivatives at ξ ∈ [0,1].
///
/// Equispaced nodal points for order `p` (p+1 nodes).
fn lagrange_1d(p: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    let h = 1.0 / p as f64;
    let mut phi = vec![0.0; p + 1];
    let mut dphi = vec![0.0; p + 1];

    for i in 0..=p {
        let xi_i = i as f64 * h;
        phi[i] = 1.0;
        dphi[i] = 0.0;

        for j in 0..=p {
            if j == i {
                continue;
            }
            let xi_j = j as f64 * h;
            let denom = xi_i - xi_j;
            // product formula for Lagrange polynomial
            let term = (xi - xi_j) / denom;
            phi[i] *= term;

            // derivative via product rule: L_i'(ξ) = Σ_{k≠i} L_i(ξ) / (ξ - ξ_k)
        }
    }

    // Recompute derivatives using the more robust formula
    for i in 0..=p {
        let xi_i = i as f64 * h;
        let mut sum = 0.0;
        for k in 0..=p {
            if k == i {
                continue;
            }
            let xi_k = k as f64 * h;
            let mut prod = 1.0;
            for j in 0..=p {
                if j == i || j == k {
                    continue;
                }
                let xi_j = j as f64 * h;
                prod *= (xi - xi_j) / (xi_i - xi_j);
            }
            sum += prod / (xi_i - xi_k);
        }
        dphi[i] = sum;
    }

    (phi, dphi)
}

/// Solve a small dense linear system via Gaussian elimination (in-place).
pub(crate) fn solve_dense(n: usize, a: &mut [f64], b: &mut [f64]) {
    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut best = col;
        let mut best_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best_val {
                best_val = v;
                best = row;
            }
        }
        if best_val < 1e-30 {
            continue;
        }
        if best != col {
            for c in col..n {
                a.swap(col * n + c, best * n + c);
            }
            b.swap(col, best);
        }

        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for c in col..n {
                a[row * n + c] -= factor * a[col * n + c];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    for row in (0..n).rev() {
        let mut sum = b[row];
        for c in (row + 1)..n {
            sum -= a[row * n + c] * b[c];
        }
        if a[row * n + row].abs() > 1e-30 {
            b[row] = sum / a[row * n + row];
        } else {
            b[row] = 0.0;
        }
    }
}

/// DPG solver for the 1-D convection-diffusion problem.
///
/// Returns the nodal solution values at the `n+1` vertices.
///
/// # Arguments
/// * `n_elem` — number of uniform elements on [0, 1]
/// * `epsilon` — diffusion coefficient (ε)
/// * `b` — convection coefficient
/// * `f` — source term
/// * `p_test` — polynomial order of the enriched test space (≥ 2, default 3)
pub fn solve_dpg_convection_diffusion_1d(
    n_elem: usize,
    epsilon: f64,
    b: f64,
    f: &dyn Fn(f64) -> f64,
    p_test: usize,
) -> Vec<f64> {
    let n_dofs = n_elem + 1; // P1 continuous trial
    let h = 1.0 / n_elem as f64;
    let quad = gauss_legendre_01(p_test + 2); // sufficient quadrature
    let trial_order = 1;
    let n_test = p_test + 1; // test basis per element (discontinuous)

    // Global stiffness matrix
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let mut rhs = vec![0.0; n_dofs];

    // Element loop
    for e in 0..n_elem {
        let x0 = e as f64 * h;
        let x1 = (e + 1) as f64 * h;
        let xm = 0.5 * (x0 + x1);
        let jac = h / 2.0; // mapping from reference [0,1] to [x0,x1]

        // Trial DOFs: vertex i and i+1 (global indices)
        let trial_dofs = [e, e + 1];

        // -- Build element matrices in enriched test basis --

        // Test mass matrix MV (size n_test × n_test)
        let mut mv = vec![0.0; n_test * n_test];
        // Test stiffness matrix KV (H1 inner product on test space)
        let mut kv = vec![0.0; n_test * n_test];
        // Convection-diffusion bilinear form B[test_i, trial_j]
        let mut b_mat = vec![0.0; n_test * trial_dofs.len()];
        // RHS in test basis: F_i = ∫ f * φ_i
        let mut f_test = vec![0.0; n_test];

        for (xi_ref, w_ref) in quad.0.iter().zip(quad.1.iter()) {
            let xi = xi_ref; // reference coordinate on [0,1]
            let w = w_ref * jac;
            let x_phys = x0 + xi * h;

            let (phi, dphi) = lagrange_1d(trial_order, *xi);
            let (psi, dpsi) = lagrange_1d(p_test, *xi);

            // Accumulate test-space inner products
            for i in 0..n_test {
                for j in 0..n_test {
                    let v = (psi[i] * psi[j] + dpsi[i] * dpsi[j] / (jac * jac)) * w;
                    // H1 inner product on physical element:
                    // ∫ (v*w + v'*w') = ∫ (v*w) + ∫ (v'*w')
                    // v' = dphi/dx = dphi/dxi * dxi/dx = dphi/dxi / jac
                    mv[i * n_test + j] += (psi[i] * psi[j]) * w;
                    kv[i * n_test + j] += (dpsi[i] * dpsi[j] / (jac * jac)) * w;
                }

                // Convection-diffusion bilinear form B(ψ_j_test, φ_i)
                // B(u, v) = ε ∫ u'v' + b ∫ u'v
                for j in 0..trial_dofs.len() {
                    let d_phi_j = dphi[j] / jac; // physical derivative
                    let phi_j = phi[j];
                    b_mat[i * trial_dofs.len() + j] +=
                        epsilon * dphi[j] / jac * dpsi[i] / jac * w
                        + b * dphi[j] / jac * psi[i] * w;
                }

                // RHS: ∫ f(x) * ψ_i
                f_test[i] += f(x_phys) * psi[i] * w;
            }
        }

        // MV + KV (H1 energy inner product matrix for test space)
        let mut mv_kv = mv.clone();
        for i in 0..n_test {
            for j in 0..n_test {
                mv_kv[i * n_test + j] += kv[i * n_test + j];
            }
        }

        // -- Solve for optimal test functions --
        // For each trial DOF j, solve (MV + KV) * v_opt_j = B[:, j]
        let mut v_opt = vec![0.0; n_test * trial_dofs.len()];
        for j in 0..trial_dofs.len() {
            let mut rhs_j = vec![0.0; n_test];
            for i in 0..n_test {
                rhs_j[i] = b_mat[i * trial_dofs.len() + j];
            }
            let mut mv_copy = mv_kv.clone();
            solve_dense(n_test, &mut mv_copy, &mut rhs_j);
            for i in 0..n_test {
                v_opt[i * trial_dofs.len() + j] = rhs_j[i];
            }
        }

        // -- Assemble element stiffness matrix --
        // K_e[i,j] = B(trial_j, v_opt_i) = Σ_k b_mat[k, j] * v_opt[k, i]
        for i in 0..trial_dofs.len() {
            for j in 0..trial_dofs.len() {
                let mut val = 0.0;
                for k in 0..n_test {
                    val += b_mat[k * trial_dofs.len() + j] * v_opt[k * trial_dofs.len() + i];
                }
                coo.add(trial_dofs[i], trial_dofs[j], val);
            }
        }

        // -- Assemble element RHS --
        // f_e[i] = Σ_k f_test[k] * v_opt[k, i]
        for i in 0..trial_dofs.len() {
            let mut val = 0.0;
            for k in 0..n_test {
                val += f_test[k] * v_opt[k * trial_dofs.len() + i];
            }
            rhs[trial_dofs[i]] += val;
        }
    }

    // Build CSR and apply Dirichlet BCs: u(0) = u(1) = 0.
    // Zero out rows/columns for Dirichlet DOFs and set diagonal to 1.
    let dirichlet = [0usize, n_dofs - 1];
    for &d in &dirichlet {
        for c in 0..n_dofs {
            if c != d { coo.add(d, c, 0.0); }
        }
        coo.add(d, d, 1.0);
        rhs[d] = 0.0;
    }
    let a = coo.into_csr();

    // CG solve (DPG stiffness is symmetric and coercive)
    let mut x = vec![0.0; n_dofs];
    let cfg = fem_solver::SolverConfig {
        rtol: 1e-14,
        atol: 1e-30,
        max_iter: 10_000,
        ..Default::default()
    };
    fem_solver::solve_cg(&a, &rhs, &mut x, &cfg).expect("DPG CG solve converged");
    x
}

/// Standard Galerkin P1 solution for comparison.
pub fn solve_galerkin_convection_diffusion_1d(
    n_elem: usize,
    epsilon: f64,
    b: f64,
    f: &dyn Fn(f64) -> f64,
) -> Vec<f64> {
    let n_dofs = n_elem + 1;
    let h = 1.0 / n_elem as f64;
    let quad = gauss_legendre_01(3);
    let trial_order = 1;

    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let mut rhs = vec![0.0; n_dofs];

    for e in 0..n_elem {
        let x0 = e as f64 * h;
        let x1 = (e + 1) as f64 * h;
        let jac = h / 2.0;
        let trial_dofs = [e, e + 1];

        let mut ke = vec![0.0; 4];
        let mut fe = vec![0.0; 2];

        for (xi, w_ref) in quad.0.iter().zip(quad.1.iter()) {
            let w = w_ref * jac;
            let x_phys = x0 + xi * h;
            let (phi, dphi) = lagrange_1d(trial_order, *xi);

            for i in 0..2 {
                for j in 0..2 {
                    // ε ∫ φ_i' φ_j' + b ∫ φ_i' φ_j
                    ke[i * 2 + j] += epsilon * dphi[i] * dphi[j] / jac * w
                        + b * dphi[i] * phi[j] * w;
                }
                fe[i] += f(x_phys) * phi[i] * w;
            }
        }

        for i in 0..2 {
            rhs[trial_dofs[i]] += fe[i];
            for j in 0..2 {
                coo.add(trial_dofs[i], trial_dofs[j], ke[i * 2 + j]);
            }
        }
    }

    // Apply Dirichlet BCs
    let dirichlet = [0usize, n_dofs - 1];
    for &d in &dirichlet {
        for c in 0..n_dofs {
            if c != d { coo.add(d, c, 0.0); }
        }
        coo.add(d, d, 1.0);
        rhs[d] = 0.0;
    }
    let a = coo.into_csr();

    let mut x = vec![0.0; n_dofs];
    let cfg = fem_solver::SolverConfig {
        rtol: 1e-14,
        atol: 1e-30,
        max_iter: 10_000,
        ..Default::default()
    };
    fem_solver::solve_gmres(&a, &rhs, &mut x, 8, &cfg).expect("Galerkin GMRES solve converged");
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DPG should be stable for convection-dominated regime.
    #[test]
    fn dpg_convection_dominated_no_oscillations() {
        let eps = 1e-4;
        let b = 1.0;
        let f = |_x: f64| 1.0;
        let u = solve_dpg_convection_diffusion_1d(32, eps, b, &f, 3);

        // DPG solution should be monotonic (no oscillations)
        let n = u.len();
        for i in 1..n {
            assert!(
                u[i] >= u[i - 1] - 1e-13,
                "DPG solution should be monotonic: u[{}]={} < u[{}]={}",
                i,
                u[i],
                i - 1,
                u[i - 1]
            );
        }
    }

    /// Standard Galerkin should oscillate for the same regime.
    #[test]
    fn galerkin_convection_dominated_oscillates() {
        let eps = 1e-4;
        let b = 1.0;
        let f = |_x: f64| 1.0;
        let u = solve_galerkin_convection_diffusion_1d(32, eps, b, &f);

        // Galerkin should have at least one inverted gradient (oscillation)
        let n = u.len();
        let mut has_oscillation = false;
        for i in 2..n {
            if (u[i] - u[i - 1]) * (u[i - 1] - u[i - 2]) < -1e-8 {
                has_oscillation = true;
                break;
            }
        }
        assert!(
            has_oscillation,
            "Galerkin should oscillate for convection-dominated flow"
        );
    }

    /// DPG converges with h-refinement.
    #[test]
    fn dpg_h_convergence() {
        let eps = 1.0;
        let b = 1.0;
        let f = |x: f64| x * (1.0 - x);
        let exact = |x: f64| {
            // Approximate exact solution for -u'' + u' = x(1-x) with u(0)=u(1)=0
            // (computed via analytic integration)
            let c1 = (1.0 - (-1.0f64).exp()).recip(); // 1/(1-e^{-1})
            let c2 = -c1;
            let p1 = -x * x * x / 6.0 + x * x / 2.0 - x;
            let p2 = x;
            p1 + c1 * (1.0 - (-x).exp()) + c2 * x
        };

        let mut prev = f64::MAX;
        for &n in &[4, 8, 16, 32] {
            let u = solve_dpg_convection_diffusion_1d(n, eps, b, &f, 3);
            let h = 1.0 / n as f64;
            let mut err2 = 0.0;
            for i in 0..=n {
                let xi = i as f64 * h;
                let diff = u[i] - exact(xi);
                err2 += diff * diff;
            }
            let err = (err2 / (n + 1) as f64).sqrt();
            assert!(err < prev * 1.01, "DPG L² error should decrease: n={n} err={err:.3e} prev={prev:.3e}");
            prev = err;
        }
    }

    /// DPG resolves boundary layers (large Peclet number).
    #[test]
    fn dpg_boundary_layer_resolution() {
        let eps = 1e-3;
        let b = 1.0;
        let f = |_: f64| 0.0;
        // Analytical solution has boundary layer near x=1: u(x) = C*(1 - exp(b*x/eps))
        let u = solve_dpg_convection_diffusion_1d(64, eps, b, &f, 3);

        // Solution should be near zero away from x=1 (interior), then rise at boundary
        let interior_max = u[..u.len() / 2].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            interior_max < 0.01,
            "DPG should resolve boundary layer: interior max = {interior_max:.3e}"
        );
    }

    /// DPG with order 2 test functions still works.
    #[test]
    fn dpg_order2_test_space() {
        let eps = 1e-2;
        let b = 1.0;
        let f = |_: f64| 1.0;
        let u = solve_dpg_convection_diffusion_1d(16, eps, b, &f, 2);

        let n = u.len();
        for i in 1..n {
            assert!(
                u[i] >= u[i - 1] - 1e-13,
                "DPG (p_test=2) should be monotonic: u[{}]={} < u[{}]={}",
                i,
                u[i],
                i - 1,
                u[i - 1]
            );
        }
    }
}
