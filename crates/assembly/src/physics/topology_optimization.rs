//! Topology optimization helpers.
//!
//! Reusable utilities for density-based topology optimization (1:1 with
//! MFEM ex37-style algorithms).  This module provides the PDE filter,
//! sigmoid link function, and Bregman projection — the building blocks
//! that the full example assembles into an optimization loop.
//!
//! # Overview
//!
//! | Item | Purpose |
//! |------|---------|
//! | [`sigmoid`] / [`inv_sigmoid`] | Link function mapping `ℝ → (0,1)` |
//! | [`HelmholtzFilter`] | Assembles and solves `(ε²K+M)ρ̃ = Mρ` |
//! | [`bregman_volume_projection`] | Illinois-method root-find for volume constraint |
//! | [`solve_l2_projection`] | `M·g = r` (mass-matrix solve in control space) |

use fem_linalg::{CsrMatrix, CooMatrix, SolverConfig, PrintLevel};
use fem_mesh::topology::MeshTopology;
use fem_space::{
    constraints::{eliminate_dirichlet, expand_from_reduced},
    fe_space::FESpace,
};
use fem_solver::{solve_pcg_gssmoother};

use crate::Assembler;
use crate::standard::{DiffusionIntegrator, MassIntegrator};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Sigmoid helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Logistic sigmoid: `σ(x) = 1/(1+exp(-x))`.
/// Numerically stable for large |x|.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

/// Inverse sigmoid (logit): `log(ρ/(1-ρ))`.
/// Clamped to avoid infinities.
#[inline]
pub fn inv_sigmoid(rho: f64) -> f64 {
    let eps = 1e-12;
    let x = rho.clamp(eps, 1.0 - eps);
    (x / (1.0 - x)).ln()
}

/// Derivative of sigmoid: `σ'(x) = σ(x)·(1-σ(x))`.
#[inline]
pub fn der_sigmoid(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helmholtz PDE density filter
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Pre-assembled Helmholtz-type density filter.
///
/// Solves `(ε²·K + M)·ρ̃ = M·ρ` where `K` is the diffusion matrix
/// (stiffness) and `M` is the mass matrix of the filter FE space.
///
/// No essential boundary conditions are applied (natural Neumann).
pub struct HelmholtzFilter {
    /// System matrix `Af = ε²·K + M`.
    af: CsrMatrix<f64>,
    /// Mass matrix `M` (for RHS assembly).
    mass: CsrMatrix<f64>,
    /// Free DOF map.
    free_map: Vec<usize>,
    /// Constrained DOF map.
    constrained_map: Vec<usize>,
    /// Number of reduced (free) DOFs.
    n_sys: usize,
    /// Solver config.
    cfg: SolverConfig,
}

impl HelmholtzFilter {
    /// Build the filter system on a scalar FE space.
    ///
    /// - `eps` : filter length scale (the `ε` in `ε²K+M`)
    /// - `quad_order` : quadrature order for assembly (typically `2*order`)
    pub fn new_from_space<M: MeshTopology>(
        space: &impl FESpace<Mesh = M>,
        eps: f64,
        quad_order: u8,
    ) -> Self {
        let eps2 = eps * eps;

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let k = Assembler::assemble_bilinear(space, &[&diff], quad_order);

        let mass = MassIntegrator { rho: 1.0 };
        let m = Assembler::assemble_bilinear(space, &[&mass], quad_order);

        let n = k.nrows;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            for j in k.row_ptr[i]..k.row_ptr[i + 1] {
                let col = k.col_idx[j] as usize;
                coo.add(i, col, eps2 * k.values[j]);
            }
            for j in m.row_ptr[i]..m.row_ptr[i + 1] {
                let col = m.col_idx[j] as usize;
                coo.add(i, col, m.values[j]);
            }
        }
        let af = coo.into_csr();

        // No essential BCs for the filter
        let empty_clamped: Vec<u32> = Vec::new();
        let zero_rhs = vec![0.0_f64; n];
        let empty_vals: Vec<f64> = Vec::new();
        let (red_af, _r, free_map, constrained_map) =
            eliminate_dirichlet(&af, &zero_rhs, &empty_clamped, &empty_vals);
        let n_sys = red_af.nrows;

        HelmholtzFilter {
            af: red_af,
            mass: m,
            free_map,
            constrained_map,
            n_sys,
            cfg: SolverConfig {
                rtol: 1e-12,
                atol: 0.0,
                max_iter: 10000,
                verbose: false,
                print_level: PrintLevel::Silent,
            },
        }
    }

    /// Return the number of rows in the reduced system.
    pub fn n_sys(&self) -> usize {
        self.n_sys
    }

    /// Solve the forward filter: `Af · ρ̃ = M · ρ`.
    ///
    /// `rho_design` is the element-wise constant design density (length = n_elems).
    /// Returns the filtered density DOF vector.
    pub fn solve_forward<M: MeshTopology>(
        &self,
        rho_design: &[f64],
        space: &impl FESpace<Mesh = M>,
    ) -> Vec<f64> {
        let n = space.n_dofs();
        let mut rhs = vec![0.0_f64; n];

        // RHS = M · ρ_elem
        let nelems = space.mesh().n_elements();
        for e in 0..nelems as u32 {
            let dofs = space.element_dofs(e);
            let rho_e = rho_design[e as usize];
            for &d in dofs {
                let d_idx = d as usize;
                for j in self.mass.row_ptr[d_idx]..self.mass.row_ptr[d_idx + 1] {
                    if self.mass.col_idx[j] as usize == d_idx {
                        rhs[d_idx] += self.mass.values[j] * rho_e;
                        break;
                    }
                }
            }
        }

        self._solve(&rhs, n)
    }

    /// Solve the adjoint filter: `Af · w̃ = rhs_filter`.
    pub fn solve_adjoint(&self, rhs_filter: &[f64]) -> Vec<f64> {
        let n = rhs_filter.len();
        self._solve(rhs_filter, n)
    }

    fn _solve(&self, rhs_in: &[f64], n: usize) -> Vec<f64> {
        let rhs = if rhs_in.is_empty() {
            vec![0.0_f64; n]
        } else {
            rhs_in.to_vec()
        };
        let (_, red_rhs, _, _) = eliminate_dirichlet(&self.af, &rhs, &[], &[]);
        let mut x_red = vec![0.0_f64; self.n_sys];
        match solve_pcg_gssmoother(&self.af, &red_rhs, &mut x_red, &self.cfg) {
            Ok(_result) => {
                expand_from_reduced(&x_red, &self.free_map, &self.constrained_map, &[], n)
            }
            Err(e) => {
                eprintln!("HelmholtzFilter solve failed: {e}");
                vec![0.0_f64; n]
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// L² mass-matrix projection
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Solve `M·g = rhs` where `M` is the mass matrix of `space`.
///
/// Used to project a filter-space field into the control space
/// (the "G = M⁻¹·w̃" step in the entropic mirror descent).
pub fn solve_l2_projection<M: MeshTopology>(
    space: &impl FESpace<Mesh = M>,
    rhs: &[f64],
    quad_order: u8,
) -> Vec<f64> {
    let mass = MassIntegrator { rho: 1.0 };
    let m = Assembler::assemble_bilinear(space, &[&mass], quad_order);

    let n = space.n_dofs();
    let mut x = vec![0.0_f64; n];
    let _ = solve_pcg_gssmoother(
        &m,
        rhs,
        &mut x,
        &SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 5000,
            verbose: false,
            print_level: PrintLevel::Silent,
        },
    );
    x
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bregman volume projection (Illinois method)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Find `c` such that `∫_Ω sigmoid(ψ + c) dx = target_volume` via the
/// Illinois variant of regula falsi.
///
/// Updates `psi` in-place: `psi[i] += c`.
/// Returns the final volume `∫ sigmoid(psi) dx`.
///
/// Integration uses element-wise lumped quadrature via the FE space:
/// for each element, the average of `sigmoid(ψ_i + c)` over its DOFs is
/// taken as a proxy for the element integral.
pub fn bregman_volume_projection<M: MeshTopology>(
    psi: &mut [f64],
    space: &impl FESpace<Mesh = M>,
    target_volume: f64,
    tol: f64,
    max_its: usize,
) -> f64 {
    let nelems = space.mesh().n_elements() as u32;

    // Bracket and Illinois solve — closure lives only inside this block
    let c = {
        let f = |shift: f64| -> f64 {
            let mut sum = 0.0;
            for e in 0..nelems {
                let dofs = space.element_dofs(e);
                let n_dofs = dofs.len() as f64;
                let mut avg = 0.0;
                for &d in dofs {
                    avg += sigmoid(psi[d as usize] + shift);
                }
                sum += avg / n_dofs;
            }
            sum - target_volume
        };

        // Bracket
        let max_psi = psi.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let search = (max_psi + 10.0).max(5.0);
        let mut a = -search;
        let mut b = search;
        let mut f_a = f(a);
        let mut f_b = f(b);

        for _ in 0..20 {
            if f_a * f_b <= 0.0 {
                break;
            }
            a *= 2.0;
            b *= 2.0;
            f_a = f(a);
            f_b = f(b);
        }

        // Illinois
        let mut c = 0.0;
        let mut f_c = 0.0;
        let mut side = 0i8;
        let mut done = false;

        for _ in 0..max_its {
            if (b - a).abs() <= tol * (b + a).abs() {
                done = true;
                break;
            }
            c = (f_a * b - f_b * a) / (f_a - f_b);
            f_c = f(c);

            if f_c * f_b > 0.0 {
                b = c;
                f_b = f_c;
                if side == -1 {
                    f_a *= 0.5;
                }
                side = -1;
            } else if f_c * f_a > 0.0 {
                a = c;
                f_a = f_c;
                if side == 1 {
                    f_b *= 0.5;
                }
                side = 1;
            } else {
                done = true;
                break;
            }
        }

        if !done {
            eprintln!("Warning: Bregman projection did not converge within {max_its} iterations");
        }
        c
    };
    // Closure `f` dropped here — no borrow conflict with mutation

    for v in psi.iter_mut() {
        *v += c;
    }

    // Compute final volume ∫ sigmoid(psi) dx
    let mut final_vol = 0.0;
    for e in 0..nelems {
        let dofs = space.element_dofs(e);
        let n_dofs = dofs.len() as f64;
        let mut avg = 0.0;
        for &d in dofs {
            avg += sigmoid(psi[d as usize]);
        }
        final_vol += avg / n_dofs;
    }
    final_vol
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_identity() {
        let x: f64 = 0.0;
        assert!((sigmoid(x) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn sigmoid_inverse() {
        for rho in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let x = inv_sigmoid(rho);
            let rho_back = sigmoid(x);
            assert!((rho_back - rho).abs() < 1e-12, "rho={rho} -> {rho_back}");
        }
    }
}
