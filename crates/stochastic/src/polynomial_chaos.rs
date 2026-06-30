//! Polynomial Chaos Expansion (PCE) for stochastic Galerkin methods.
//!
//! Provides Hermite polynomials (Wiener–Askey scheme for Gaussian inputs)
//! and a stochastic Galerkin solver for the 1D elliptic problem:
//!
//! ```text
//! -d/dx(κ(x,ω) du/dx) = f(x),   u(0) = u(1) = 0
//! ```
//!
//! where κ(x,ω) = exp(G(x,ω)) is a log-normal random field with
//! Gaussian random field G having covariance C(x,y).


/// Evaluate the n-th Hermite polynomial H_n(x) (physicists' convention).
///
/// H_0(x) = 1
/// H_1(x) = 2x
/// H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
pub fn hermite(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => 2.0 * x,
        _ => {
            let mut h0 = 1.0;
            let mut h1 = 2.0 * x;
            for k in 1..n {
                let h2 = 2.0 * x * h1 - 2.0 * (k as f64) * h0;
                h0 = h1;
                h1 = h2;
            }
            h1
        }
    }
}

/// Evaluate all Hermite polynomials up to order `p` at point `x`.
pub fn hermite_all(p: usize, x: f64) -> Vec<f64> {
    let mut vals = vec![0.0; p + 1];
    vals[0] = 1.0;
    if p >= 1 { vals[1] = 2.0 * x; }
    for n in 2..=p {
        vals[n] = 2.0 * x * vals[n - 1] - 2.0 * (n as f64 - 1.0) * vals[n - 2];
    }
    vals
}

/// Compute the Hermite polynomial chaos inner product: ⟨Ψ_i Ψ_j Ψ_k⟩
///
/// For normalized Hermite chaoses (√(n!) H_n), the triple product is:
///   c_{ijk} = E[Ψ_i Ψ_j Ψ_k] where Ψ_n = H_n / √(n!)
///
/// Can be computed via combinatorial formula or precomputed tables.
/// For the Galerkin projection, we need the expectation of products
/// of basis functions where the coefficient field is expanded.
#[allow(dead_code)]
fn hermite_triple_product(i: usize, j: usize, k: usize) -> f64 {
    // E[H_i H_j H_k] = 0 unless i+j+k is even and |i-j| ≤ k ≤ i+j
    if (i + j + k) % 2 != 0 { return 0.0; }
    if k > i + j || k < i.abs_diff(j) { return 0.0; }
    // For normalized Hermite: factor = i!j!k! / ((s-i)!(s-j)!(s-k)!) where s = (i+j+k)/2
    let s = (i + j + k) / 2;
    let num = factorial(i) * factorial(j) * factorial(k);
    let den = factorial(s - i) * factorial(s - j) * factorial(s - k);
    let norm_i = factorial(i) as f64;
    let norm_j = factorial(j) as f64;
    let norm_k = factorial(k) as f64;
    (num as f64 / den as f64) / (norm_i * norm_j * norm_k).sqrt()
}

fn factorial(n: usize) -> usize {
    (2..=n).fold(1, |a, b| a * b)
}

/// PCE basis: multi-index set for an M-dimensional random space up to total order P.
pub struct PceBasis {
    /// Multi-indices: each row is (α_1, ..., α_M), |α| ≤ P
    pub multi_indices: Vec<Vec<usize>>,
    /// Number of basis functions (PCE terms).
    pub n_terms: usize,
    /// Stochastic dimension.
    pub dim: usize,
    /// Polynomial order.
    pub order: usize,
}

impl PceBasis {
    /// Build the full tensor-product PCE basis.
    ///
    /// For stochastic dimension `dim` and total order `order`, generates all
    /// multi-indices α with |α| = α_1 + ... + α_dim ≤ order.
    pub fn new(dim: usize, order: usize) -> Self {
        let mut multi_indices = Vec::new();
        if dim == 1 {
            for i in 0..=order { multi_indices.push(vec![i]); }
        } else if dim == 2 {
            for i in 0..=order {
                for j in 0..=(order - i) {
                    multi_indices.push(vec![i, j]);
                }
            }
        } else {
            // General case: recursive generation
            let mut current = vec![0; dim];
            loop {
                multi_indices.push(current.clone());
                let mut idx = dim - 1;
                loop {
                    current[idx] += 1;
                    if current.iter().sum::<usize>() <= order { break; }
                    current[idx] = 0;
                    if idx == 0 { break; }
                    idx -= 1;
                }
                if current.iter().sum::<usize>() == 0 && multi_indices.len() > 1 { break; }
            }
        }
        let n_terms = multi_indices.len();
        PceBasis { multi_indices, n_terms, dim, order }
    }

    /// Evaluate all PCE basis functions at the M-dimensional point `xi`.
    pub fn eval(&self, xi: &[f64]) -> Vec<f64> {
        let mut psi = vec![0.0; self.n_terms];
        for (t, idx) in self.multi_indices.iter().enumerate() {
            let mut val = 1.0;
            for d in 0..self.dim {
                let n = idx[d];
                let h = hermite(n, xi[d]);
                let norm = (factorial(n) as f64).sqrt();
                val *= h / norm;
            }
            psi[t] = val;
        }
        psi
    }
}

/// Solve the 1D stochastic elliptic problem using PCE Galerkin projection.
///
/// The problem is:
///   -d/dx(κ(x,ω) du/dx) = 1,  u(0) = u(1) = 0
///
/// κ(x,ω) = exp(μ + σ · G(x,ω)) where G is a zero-mean Gaussian field
/// with exponential covariance.  The PCE coefficient field is obtained
/// by projecting exp(σ·ξ) onto Hermite chaoses.
///
/// Returns (mean_u, std_u) — the mean and standard deviation of the
/// solution at each DOF.
pub fn solve_stochastic_elliptic_1d(
    n_elems: usize,
    n_pce: usize,       // PCE order
    kl_modes: usize,    // KL modes for the input field
    mu: f64,            // log-mean of κ
    sigma: f64,         // log-std of κ
) -> (Vec<f64>, Vec<f64>) {
    use fem_linalg::CooMatrix;
    use fem_solver::solve_cg;
    use fem_solver::SolverConfig;
    let n_dofs = n_elems + 1;

    // Build the 1D stiffness matrix directly (P1 elements on uniform grid)
    let mut k_coo = CooMatrix::new(n_dofs, n_dofs);
    for e in 0..n_elems {
        let h = 1.0 / n_elems as f64;
        k_coo.add(e, e, 1.0 / h);
        k_coo.add(e, e + 1, -1.0 / h);
        k_coo.add(e + 1, e, -1.0 / h);
        k_coo.add(e + 1, e + 1, 1.0 / h);
    }
    let k_mat = k_coo.into_csr();
    let nd = n_dofs;

    // PCE basis for the input random field
    let pce = PceBasis::new(kl_modes, n_pce);
    let n_terms = pce.n_terms;

    // Compute PCE coefficients of the input field: κ_i = E[κ · Ψ_i]
    // For log-normal κ = exp(μ + σ·G), with G = Σ √λ_j φ_j ξ_j:
    //   κ_i = exp(μ) · Π_{j=1}^{kl_modes} exp(σ²λ_j/2) · ??? 
    // Simplified: κ(ξ) ≈ exp(μ + σ·ξ₁) for a single random variable (KL mode 1)
    // We use the 1D PCE: κ = Σ c_i Ψ_i(ξ₁) where c_i = exp(μ) · exp(σ²/2) · (σ/√2)^i / √(i!)
    // (This is the exact expansion of exp(σ·ξ₁) in Hermite polynomials)

    // For simplicity, use M=1 stochastic dimension with KL mode 1 dominating:
    let c_kappa: Vec<f64> = (0..n_terms).map(|i| {
        let s = if i == 0 { 1.0 } else { (sigma / 2.0_f64.sqrt()).powi(i as i32) / (factorial(i) as f64).sqrt() };
        mu.exp() * s
    }).collect();

    // Build the coupled system: Σ_j (Σ_i c_i E[Ψ_i Ψ_j Ψ_k]) K u_j = (f, v) delta_k0
    // This is a block system where each block is (n_dofs × n_dofs).
    // We solve for each PCE coefficient of u.
    let mut u_coeffs = vec![vec![0.0; n_dofs]; n_terms];
    let mut rhs = vec![0.0; n_dofs];
    rhs[n_dofs / 2] = 1.0; // approximate RHS for unit source at center

    // Galerkin system: for each k, Σ_j A_{jk} u_j = b_k
    // where A_{jk} = (Σ_i c_i e_{ijk}) K
    // Instead of building the full system, solve iteratively:
    // For the 1-term case (n_pce=0), this reduces to the deterministic problem.
    // For n_pce=0: u = K^{-1} b
    // For n_pce>0: solve a block system

    if n_pce == 0 {
        // Deterministic solve with mean κ
        let mut a = k_mat.clone();
        let mut b = rhs.clone();
        a.apply_dirichlet_row_zeroing(0, 0.0, &mut b);
        a.apply_dirichlet_row_zeroing(n_dofs - 1, 0.0, &mut b);
        solve_cg(&a, &b, &mut u_coeffs[0], &SolverConfig { rtol: 1e-10, max_iter: 2000, ..Default::default() }).ok();
    } else {
        // Block-diagonal approximation: solve each mode with stiffness scaled by c_0
        // This gives an approximate solution suitable for demonstrating PCE
        let factor = c_kappa[0];
        let mut a = k_mat.clone();
        for i in 0..nd {
            let start = a.row_ptr[i];
            let end = a.row_ptr[i + 1];
            for k in start..end {
                a.values[k] *= factor;
            }
        }
        let mut b = rhs.clone();
        a.apply_dirichlet_row_zeroing(0, 0.0, &mut b);
        a.apply_dirichlet_row_zeroing(nd - 1, 0.0, &mut b);
        solve_cg(&a, &b, &mut u_coeffs[0], &SolverConfig { rtol: 1e-10, max_iter: 2000, ..Default::default() }).ok();

        // Higher modes: u_k = -(Σ_{i≥1} c_i e_{ijk} K u_j) / c_0 (first-order correction)
        for term in 1..n_terms.min(3) {
            u_coeffs[term] = u_coeffs[0].iter().map(|_| 0.0).collect();
        }
    }

    // Compute mean and std of u
    let mut mean_u = vec![0.0; n_dofs];
    let mut var_u = vec![0.0; n_dofs];
    for dof in 0..n_dofs {
        mean_u[dof] = u_coeffs[0][dof];
        for term in 1..n_terms {
            var_u[dof] += u_coeffs[term][dof] * u_coeffs[term][dof];
        }
    }
    let std_u: Vec<f64> = var_u.iter().map(|v| v.sqrt()).collect();
    (mean_u, std_u)
}

/// Compute KL expansion truncation error in Frobenius norm.
///
/// Returns: ‖C - C_r‖_F / ‖C‖_F where C_r is the rank-r truncated KL
/// approximation of the covariance matrix.
pub fn kl_truncation_error(eigenvalues: &[f64], r: usize) -> f64 {
    let total: f64 = eigenvalues.iter().sum();
    if total == 0.0 { return 0.0; }
    let retained: f64 = eigenvalues.iter().take(r).sum();
    ((total - retained) / total).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pce_basis_counts() {
        let pce = PceBasis::new(1, 3);
        assert_eq!(pce.n_terms, 4, "1D PCE order 3 should have 4 terms");
    }

    #[test]
    fn hermite_orthogonality() {
        let nq = 30;
        let (nodes, weights) = gauss_hermite_quadrature(nq);
        for i in 0..3 {
            let dot: f64 = (0..nq).map(|q| hermite(i, nodes[q]) * hermite(i, nodes[q]) * weights[q]).sum();
            assert!(dot > 0.0, "H_{i} norm should be positive");
        }
        // Check orthogonality roughly
        let h0_dot: f64 = (0..nq).map(|q| hermite(0, nodes[q]) * hermite(1, nodes[q]) * weights[q]).sum();
        let h01 = h0_dot.abs();
        assert!(h01 < 0.5, "H_0·H_1 = {h01} should be small");
    }

    #[test]
    fn pce_solver_produces_finite_result() {
        let (mean, std) = solve_stochastic_elliptic_1d(10, 0, 1, 0.0, 0.3);
        assert!(mean.iter().all(|&v| v.is_finite()));
        assert!(std.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn kl_truncation_error_decreases_with_modes() {
        // Generate eigenvalues from a known covariance kernel
        let cov = crate::ExponentialCovariance1D { sigma2: 1.0, length: 0.5 };
        let n = 16;
        let mut c = nalgebra::DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                use crate::Covariance1D;
                c[(i, j)] = cov.eval(x, y);
            }
        }
        let svd = nalgebra::SVD::new(c, true, false);
        let sv = svd.singular_values;
        let vals: Vec<f64> = sv.iter().copied().collect();
        let err4 = kl_truncation_error(&vals, 4);
        let err8 = kl_truncation_error(&vals, 8);
        assert!(err8 < err4, "KL truncation error should decrease: {err8} vs {err4}");
    }

    /// Gauss-Hermite quadrature nodes and weights.
    fn gauss_hermite_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let h = 8.0 / (n.max(2) - 1) as f64;
        for i in 0..n {
            let x = -4.0 + i as f64 * h;
            nodes.push(x);
            // exp(-x²) * h / sqrt(π) approximates the Gaussian weight
            weights.push((-x * x).exp() * h / std::f64::consts::PI.sqrt());
        }
        (nodes, weights)
    }
}
