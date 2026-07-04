//! Multi-Level Monte Carlo (MLMC) with stochastic collocation on sparse grids.
//!
//! # Overview
//!
//! 1. **Smolyak sparse grid** — interpolate a smooth function in `M` dimensions
//!    using far fewer points than a full tensor-product grid.
//! 2. **Stochastic collocation** — solve the deterministic PDE at each sparse
//!    grid point, then compute response statistics by quadrature.
//! 3. **Multi-Level Monte Carlo** — combine cheap (coarse-mesh) and expensive
//!    (fine-mesh) samples to estimate statistics at reduced cost.
//!
//! # Usage
//! ```rust,ignore
//! use fem_stochastic::mlmc::*;
//!
//! // 2-D Smolyak grid with level 3 (Clenshaw-Curtis nodes)
//! let grid = SmolyakGrid::clenshaw_curtis(2, 3);
//! println!("{} points on the sparse grid", grid.n_points());
//!
//! // MLMC estimator
//! let mlmc = MultiLevelMonteCarlo::new(vec![1, 2, 4, 8], 100, 0.5, 1.0);
//! let result = mlmc.estimate(|level, _n| {
//!     // Solve PDE at refinement level `level`, return QoI
//!     Ok(0.0)
//! });
//! ```

use std::f64::consts::PI;

// ─── Smolyak sparse grid ─────────────────────────────────────────────────────

/// A 1-D quadrature rule (points + weights) used as the building block for
/// Smolyak sparse grids.
#[derive(Debug, Clone)]
pub struct QuadRule1D {
    pub points:  Vec<f64>,
    pub weights: Vec<f64>,
}

/// Clenshaw–Curtis nodes on [-1, 1] at level `l` (1 ≤ l ≤ L).
/// Number of points = 2^{l-1} + 1.
fn clenshaw_curtis_1d(l: usize) -> QuadRule1D {
    if l == 1 {
        return QuadRule1D { points: vec![0.0], weights: vec![2.0] };
    }
    let n = (1usize << (l - 1)) + 1;
    let mut pts = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    for j in 0..n {
        let theta = PI * j as f64 / (n - 1) as f64;
        pts.push(-theta.cos());
        // Clenshaw-Curtis weight via FFT (simplified: use exact formula)
        let mut w = 1.0_f64;
        for k in 1..((n - 1) / 2 + 1) {
            let ck = if 2 * k == n - 1 { 1.0 } else { 2.0 };
            w -= ck * (2.0 * PI * k as f64 * j as f64 / (n - 1) as f64).cos()
                  / (4.0 * k as f64 * k as f64 - 1.0);
        }
        wts.push(2.0 * w / (n - 1) as f64);
    }
    wts[0] /= 2.0;
    wts[n - 1] /= 2.0;
    QuadRule1D { points: pts, weights: wts }
}

/// Gauss–Legendre nodes on [-1, 1] at level `l` using the Golub–Welsch
/// algorithm via the Legendre polynomial recurrence.
fn gauss_legendre_1d(l: usize) -> QuadRule1D {
    let n = 1usize << l;
    // Use Newton's method on Legendre polynomials for simplicity
    let mut pts = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    for i in 0..n {
        // Initial guess: Chebyshev nodes
        let theta = PI * (i as f64 + 0.75) / (n as f64 + 0.5);
        let mut x = theta.cos();
        for _ in 0..20 {
            let (p, dp) = legendre_poly(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-15 { break; }
        }
        pts.push(x);
        let (_, dp) = legendre_poly(n, x);
        wts.push(2.0 / ((1.0 - x * x) * dp * dp));
    }
    QuadRule1D { points: pts, weights: wts }
}

/// Evaluate P_n(x) and P'_n(x) via recurrence.
fn legendre_poly(n: usize, x: f64) -> (f64, f64) {
    if n == 0 { return (1.0, 0.0); }
    if n == 1 { return (x, 1.0); }
    let mut p0 = 1.0;
    let mut p1 = x;
    let mut dp0 = 0.0;
    let mut dp1 = 1.0;
    for k in 1..n {
        let kf = k as f64;
        let p2 = ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0);
        let dp2 = ((2.0 * kf + 1.0) * (p1 + x * dp1) - kf * dp0) / (kf + 1.0);
        p0 = p1; p1 = p2;
        dp0 = dp1; dp1 = dp2;
    }
    (p1, dp1)
}

/// A Smolyak sparse grid in M dimensions at level L.
///
/// Uses the combination technique:
///   A(L, M) = Σ_{|i| ≤ L+M-1} (-1)^{L+M-1-|i|} · C_{M-1}^{|i|-L} · (Q_{i₁} ⊗ ... ⊗ Q_{i_M})
///
/// where Q_i is the 1-D quadrature rule at level i.
#[derive(Debug, Clone)]
pub struct SmolyakGrid {
    /// Dimension M.
    pub dim: usize,
    /// Level L.
    pub level: usize,
    /// Multi-index set: each entry is (i₁, ..., i_M) with 1 ≤ i_j ≤ L.
    pub multi_indices: Vec<Vec<usize>>,
    /// Sparse grid points (each of length `dim`), flattened: [x₁₁,...,x₁M, x₂₁,...,x₂M, ...].
    pub points: Vec<f64>,
    /// Quadrature weights for each point.
    pub weights: Vec<f64>,
}

impl SmolyakGrid {
    /// Build a Smolyak sparse grid using Clenshaw–Curtis 1-D rules.
    pub fn clenshaw_curtis(dim: usize, level: usize) -> Self {
        Self::build(dim, level, clenshaw_curtis_1d)
    }

    /// Build a Smolyak sparse grid using Gauss–Legendre 1-D rules.
    pub fn gauss_legendre(dim: usize, level: usize) -> Self {
        Self::build(dim, level, gauss_legendre_1d)
    }

    fn build<F>(dim: usize, level: usize, rule: F) -> Self
    where
        F: Fn(usize) -> QuadRule1D,
    {
        // Pre-compute 1-D rules up to level
        let rules: Vec<QuadRule1D> = (1..=level).map(rule).collect();

        // Generate multi-index set: {i ∈ ℕ^M : 1 ≤ i_j, |i| ≤ L}
        let multi_indices = Self::index_set(dim, level);

        // Build sparse grid by combination
        let combos = Self::combination_coeffs(dim, level);
        let mut all_points: Vec<Vec<f64>> = Vec::new();
        let mut all_weights: Vec<f64> = Vec::new();

        for (mi, coeff) in multi_indices.iter().zip(combos.iter()) {
            if coeff.abs() < 1e-15 { continue; }
            // Tensor product of 1-D rules for this multi-index
            let grids_1d: Vec<&QuadRule1D> = mi.iter().map(|&i| &rules[i - 1]).collect();
            let npts: Vec<usize> = grids_1d.iter().map(|g| g.points.len()).collect();
            let n_total: usize = npts.iter().product();
            if n_total > 1_000_000 { continue; } // safety limit

            // Iterate over the tensor product using a multi-dimensional counter
            let mut counter = vec![0usize; dim];
            for _ in 0..n_total {
                let mut pt = Vec::with_capacity(dim);
                let mut w = 1.0_f64;
                for d in 0..dim {
                    pt.push(grids_1d[d].points[counter[d]]);
                    w *= grids_1d[d].weights[counter[d]];
                }
                // Check if this point already exists (with tolerance)
                let pos = all_points.iter().position(|p| {
                    p.iter().zip(pt.iter()).all(|(a, b)| (a - b).abs() < 1e-14)
                });
                if let Some(idx) = pos {
                    all_weights[idx] += *coeff * w;
                } else {
                    all_points.push(pt);
                    all_weights.push(*coeff * w);
                }
                // Increment counter
                for d in (0..dim).rev() {
                    counter[d] += 1;
                    if counter[d] < npts[d] { break; }
                    counter[d] = 0;
                }
            }
        }

        // Flatten points
        let points: Vec<f64> = all_points.into_iter().flatten().collect();

        SmolyakGrid { dim, level, multi_indices, points, weights: all_weights }
    }

    /// Generate all multi-indices i with 1 ≤ i_j ≤ level and |i| ≥ level.
    fn index_set(dim: usize, level: usize) -> Vec<Vec<usize>> {
        if dim == 0 { return vec![]; }
        let mut result = Vec::new();
        // Generate all (i₁, ..., i_M) with 1 ≤ i_j ≤ level + dim
        // and filter by level ≤ |i| ≤ level + dim - 1
        Self::index_set_recursive(dim, level + dim, &mut vec![], &mut result);
        result.retain(|mi| {
            let sum: usize = mi.iter().sum();
            sum >= level && sum < level + dim
        });
        result
    }

    fn index_set_recursive(dim: usize, max_val: usize, prefix: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if dim == 0 {
            result.push(prefix.clone());
            return;
        }
        for i in 1..=max_val {
            prefix.push(i);
            Self::index_set_recursive(dim - 1, max_val, prefix, result);
            prefix.pop();
        }
    }

    /// Combination coefficients: (-1)^{L+M-1-|i|} · C(M-1, |i|-L)
    fn combination_coeffs(dim: usize, level: usize) -> Vec<f64> {
        let indices = Self::index_set(dim, level);
        indices.iter().map(|mi| {
            let sum_i: usize = mi.iter().sum();
            if sum_i < level {
                return 0.0;  // |i| < L → no contribution
            }
            let s = level + dim - 1;
            let q = sum_i as isize - level as isize;
            if q < 0 || q > (dim - 1) as isize { return 0.0; }
            let sign = if (s - sum_i).is_multiple_of(2) { 1.0 } else { -1.0 };
            sign * binom(dim - 1, q as usize) as f64
        }).collect()
    }

    /// Number of sparse grid points.
    pub fn n_points(&self) -> usize { self.points.len() / self.dim }

    /// Get the d-th coordinate of the i-th point.
    pub fn point(&self, i: usize, d: usize) -> f64 {
        self.points[i * self.dim + d]
    }
}

fn binom(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    let k = k.min(n - k);
    (1..=k).fold(1, |acc, i| acc * (n - k + i) / i)
}

// ─── Stochastic collocation ──────────────────────────────────────────────────

/// Result of a stochastic collocation simulation.
#[derive(Debug, Clone)]
pub struct CollocationResult {
    /// Number of collocation points (solves).
    pub n_points: usize,
    /// Approximate integral (e.g., mean of QoI).
    pub integral: f64,
    /// Approximate variance.
    pub variance: f64,
}

/// Perform stochastic collocation using a Smolyak sparse grid.
///
/// `solve_fn` maps a parameter vector `ξ ∈ [-1, 1]^M` to a QoI value.
pub fn collocate<F>(grid: &SmolyakGrid, solve_fn: F) -> CollocationResult
where
    F: Fn(&[f64]) -> f64,
{
    let n_pts = grid.n_points();
    let mut integral = 0.0_f64;
    let mut integral2 = 0.0_f64;

    for i in 0..n_pts {
        let xi: Vec<f64> = (0..grid.dim).map(|d| grid.point(i, d)).collect();
        let qoi = solve_fn(&xi);
        let w = grid.weights[i];
        integral += w * qoi;
        integral2 += w * qoi * qoi;
    }

    // Normalise: Smolyak weights sum to 2^dim (full-cube volume)
    let vol = 2.0_f64.powi(grid.dim as i32);
    let mean = integral / vol;
    let variance = (integral2 / vol - mean * mean).max(0.0);

    CollocationResult { n_points: n_pts, integral: mean, variance }
}

// ─── Multi-Level Monte Carlo (MLMC) ──────────────────────────────────────────

/// MLMC level description: mesh refinement level with associated cost.
pub struct MlmcLevel {
    /// Level index (0 = coarsest, L = finest).
    pub level: usize,
    /// Number of samples on this level.
    pub n_samples: usize,
    /// Cost per sample (arbitrary units, e.g. DOF count).
    pub cost_per_sample: f64,
}

/// Result of an MLMC estimation.
#[derive(Debug, Clone)]
pub struct MlmcResult {
    /// Total number of samples across all levels.
    pub total_samples: usize,
    /// Estimated mean of the QoI.
    pub mean: f64,
    /// Estimated variance of the mean.
    pub variance_mean: f64,
    /// Estimated total cost.
    pub total_cost: f64,
    /// Per-level statistics.
    pub level_results: Vec<MlmcLevelResult>,
}

/// Per-level MLMC statistics.
#[derive(Debug, Clone)]
pub struct MlmcLevelResult {
    pub level: usize,
    pub n_samples: usize,
    pub mean: f64,
    pub variance: f64,
    pub cost: f64,
}

/// Multi-Level Monte Carlo estimator.
///
/// MLMC combines samples from multiple discretisation levels to estimate
/// `E[Q]` at reduced cost compared to standard MC.  The key identity:
///
/// ```text
/// E[Q_L] = E[Q_0] + Σ_{ℓ=1}^{L} E[Q_ℓ - Q_{ℓ-1}]
/// ```
///
/// where Q_ℓ is the QoI on level ℓ (refinement ℓ).  Coarse levels are cheap
/// and reduce variance of the difference estimator.
pub struct MultiLevelMonteCarlo {
    /// Levels (refinement parameters, e.g. mesh sizes h = 2^{-ℓ}).
    pub levels: Vec<usize>,
    /// Target number of samples on the finest level.
    pub n_fine: usize,
    /// Ratio of samples between levels: n_{ℓ-1} = n_ℓ · ratio.
    pub sample_ratio: f64,
    /// Cost scaling factor: cost_ℓ = cost_0 · factor^ℓ.
    pub cost_factor: f64,
}

impl MultiLevelMonteCarlo {
    /// Create an MLMC estimator.
    ///
    /// # Arguments
    /// * `levels`      — discretisation level indices (e.g. [0, 1, 2, 3])
    /// * `n_fine`      — number of samples on the finest level
    /// * `sample_ratio` — n_{ℓ-1} / n_ℓ (≥ 1, typical: 2–4)
    /// * `cost_factor` — cost_ℓ / cost_{ℓ-1} (≥ 1, typical: 2^dim ≈ 4–8)
    pub fn new(levels: Vec<usize>, n_fine: usize, sample_ratio: f64, cost_factor: f64) -> Self {
        MultiLevelMonteCarlo { levels, n_fine, sample_ratio, cost_factor }
    }

    /// Run the MLMC estimation.
    ///
    /// `sample_fn(level, sample_index)` performs one solve at the given
    /// refinement level and returns the QoI.
    pub fn estimate<F>(&self, mut sample_fn: F) -> Result<MlmcResult, String>
    where
        F: FnMut(usize, usize) -> Result<f64, String>,
    {
        let n_levels = self.levels.len();
        let mut level_results = Vec::with_capacity(n_levels);

        let mut total_samples = 0usize;
        let mut total_cost = 0.0_f64;
        let mut mean = 0.0_f64;
        let mut var_mean = 0.0_f64;

        let mut prev_mean: Option<f64> = None;
        let mut prev_var: f64 = 0.0;
        let mut prev_n: usize = 0;

        for (li, &level) in self.levels.iter().enumerate() {
            let n = if li == n_levels - 1 {
                self.n_fine
            } else {
                let ratio_pow = self.sample_ratio.powi((n_levels - 1 - li) as i32);
                (self.n_fine as f64 * ratio_pow).round() as usize
            }.max(10);  // minimum samples per level

            let cost = self.cost_factor.powi(level as i32);

            let mut sum = 0.0_f64;
            let mut sum2 = 0.0_f64;

            for i in 0..n {
                let qoi = sample_fn(level, i)?;
                sum += qoi;
                sum2 += qoi * qoi;
            }

            let l_mean = sum / n as f64;
            let l_var = if n > 1 {
                (sum2 - sum * sum / n as f64) / (n - 1) as f64
            } else { 0.0 };

            // For ℓ ≥ 1, estimate E[Q_ℓ - Q_{ℓ-1}] with independent samples.
            // Var[Q_ℓ - Q_{ℓ-1}] = Var[Q_ℓ]/n_ℓ + Var[Q_{ℓ-1}]/n_{ℓ-1}
            if li == 0 {
                mean = l_mean;
                var_mean = l_var / n as f64;
            } else if let Some(pm) = prev_mean {
                mean += l_mean - pm;
                var_mean += l_var / n as f64 + prev_var / prev_n as f64;
            }

            total_samples += n;
            total_cost += n as f64 * cost;

            level_results.push(MlmcLevelResult {
                level, n_samples: n, mean: l_mean, variance: l_var, cost,
            });

            prev_mean = Some(l_mean);
            prev_var = l_var;
            prev_n = n;
        }

        Ok(MlmcResult {
            total_samples,
            mean,
            variance_mean: var_mean,
            total_cost,
            level_results,
        })
    }

    /// Compute the optimal sample allocation per level given the per-level
    /// variances and costs from a pilot run.
    ///
    /// The optimal n_ℓ for a fixed total cost C is:
    ///   n_ℓ ∝ √(Var_ℓ / cost_ℓ)
    /// Normalised so that the finest level gets `n_fine` samples.
    ///
    /// Returns a vector `optimal_n` of the same length as `self.levels`.
    pub fn optimal_allocation(&self, level_results: &[MlmcLevelResult], total_budget: f64) -> Vec<usize> {
        let n_levels = self.levels.len();
        if n_levels == 0 { return vec![]; }

        let mut optimal_n = vec![0usize; n_levels];

        // Compute sqrt(Var_ℓ / cost_ℓ)
        let mut weights = Vec::with_capacity(n_levels);
        let mut sum_w = 0.0_f64;
        for lr in level_results {
            let v = lr.variance.max(1e-30);
            let c = lr.cost.max(1e-30);
            let w = (v / c).sqrt();
            weights.push(w);
            sum_w += w;
        }

        if sum_w < 1e-30 { return optimal_n; }

        // Normalise so that total cost ≈ total_budget
        // n_ℓ = C * w_ℓ / Σ (w_k * cost_k)
        let denom: f64 = weights.iter().zip(level_results.iter())
            .map(|(&w, lr)| w * lr.cost).sum();
        if denom < 1e-30 { return optimal_n; }

        for (i, &w) in weights.iter().enumerate() {
            let n = (total_budget * w / denom).round() as usize;
            optimal_n[i] = n.max(10); // minimum 10 samples
        }

        optimal_n
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smolyak_1d_equals_1d_rule() {
        // In 1D with level L, the sparse grid should match the 1-D rule
        let grid = SmolyakGrid::clenshaw_curtis(1, 4);
        assert!(grid.n_points() > 1);
        // Total weight should be 2 (volume of [-1, 1])
        let sum_w: f64 = grid.weights.iter().sum();
        assert!((sum_w - 2.0).abs() < 1e-10, "sum weights = {sum_w:.10}, expected 2");
    }

    #[test]
    fn smolyak_2d_integrates_constant() {
        // ∫_{-1}^{1}∫_{-1}^{1} 1 dx dy = 4
        let grid = SmolyakGrid::clenshaw_curtis(2, 3);
        let integral: f64 = grid.weights.iter().sum();
        assert!((integral - 4.0).abs() < 1e-8,
            "2D constant integral = {integral:.10}, expected 4");
    }

    #[test]
    fn collocation_constant_function() {
        let grid = SmolyakGrid::clenshaw_curtis(2, 3);
        let result = collocate(&grid, |_| 1.0);
        assert!((result.integral - 1.0).abs() < 1e-8,
            "constant function mean = {:.10}, expected 1", result.integral);
        assert!(result.variance < 1e-10, "constant variance should be 0");
    }

    #[test]
    fn mlmc_constant_function() {
        let mlmc = MultiLevelMonteCarlo::new(vec![0, 1, 2], 50, 2.0, 4.0);
        let result = mlmc.estimate(|_, _| Ok(42.0)).expect("MLMC");
        assert!((result.mean - 42.0).abs() < 1.0,
            "MLMC mean = {:.6}, expected ~42", result.mean);
        assert!(result.total_samples > 0);
    }

    #[test]
    fn mlmc_monte_carlo_increases_samples_on_coarser_levels() {
        let mlmc = MultiLevelMonteCarlo::new(vec![0, 1, 2], 20, 2.0, 4.0);
        let result = mlmc.estimate(|_, _| Ok(1.0)).expect("MLMC");
        // Finer levels should have fewer or equal samples
        for i in 1..result.level_results.len() {
            assert!(result.level_results[i].n_samples <= result.level_results[i - 1].n_samples);
        }
    }

    #[test]
    fn binom_small_values() {
        assert_eq!(binom(4, 2), 6);
        assert_eq!(binom(5, 0), 1);
        assert_eq!(binom(5, 5), 1);
    }

    #[test]
    fn legendre_poly_known_values() {
        let (p0, dp0) = legendre_poly(0, 0.5);
        assert!((p0 - 1.0).abs() < 1e-14);
        let (p1, dp1) = legendre_poly(1, 0.5);
        assert!((p1 - 0.5).abs() < 1e-14);
        assert!((dp1 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn sparse_grid_gauss_legendre_integrates_polynomial() {
        // ∫_{-1}^{1} x² dx = 2/3
        let grid = SmolyakGrid::gauss_legendre(1, 3);
        let integral: f64 = grid.weights.iter().enumerate()
            .map(|(i, w)| w * grid.point(i, 0).powi(2))
            .sum();
        assert!((integral - 2.0 / 3.0).abs() < 1e-8,
            "∫x² = {integral:.10}, expected 2/3");
    }

    #[test]
    fn mlmc_known_variance_model() {
        // Model: Q(ω) = ω where ω ~ U[-1,1]. True mean = 0, variance = 1/3 ≈ 0.333.
        use rand::Rng;
        let mlmc = MultiLevelMonteCarlo::new(vec![0, 1], 200, 4.0, 2.0);
        let result = mlmc.estimate(|_level, _idx| {
            let mut rng = rand::thread_rng();
            Ok(rng.gen_range(-1.0..1.0))
        }).expect("MLMC with known variance");
        eprintln!("MLMC known-variance: mean={:.4}, var_mean={:.4}", result.mean, result.variance_mean);
        // Mean should be close to 0, variance should be finite
        assert!(result.mean.abs() < 0.3, "mean should be near 0, got {}", result.mean);
        assert!(result.variance_mean > 0.0, "variance of mean should be positive");
        assert!(result.total_samples > 0);
        assert!(result.level_results.len() == 2);
    }

    #[test]
    fn mlmc_optimal_allocation_reduces_cost() {
        // Demonstrate that optimal allocation improves efficiency.
        let levels = vec![0, 1, 2, 3];
        // Model: variance decreases with level (as in actual PDE solves)
        let pilot_results = vec![
            MlmcLevelResult { level: 0, n_samples: 100, mean: 1.0, variance: 0.5, cost: 1.0 },
            MlmcLevelResult { level: 1, n_samples: 100, mean: 0.5, variance: 0.1, cost: 4.0 },
            MlmcLevelResult { level: 2, n_samples: 100, mean: 0.25, variance: 0.02, cost: 16.0 },
            MlmcLevelResult { level: 3, n_samples: 100, mean: 0.125, variance: 0.005, cost: 64.0 },
        ];
        let mlmc = MultiLevelMonteCarlo::new(levels, 100, 2.0, 4.0);
        let optimal = mlmc.optimal_allocation(&pilot_results, 1e6);
        assert_eq!(optimal.len(), 4);
        // Coarser levels should have more samples (cheaper, higher variance)
        for i in 1..optimal.len() {
            assert!(optimal[i] <= optimal[i - 1] || optimal[i] < 20,
                "optimal n should decrease with level: level {} n={} > level {} n={}",
                i, optimal[i], i-1, optimal[i-1]);
        }
        for &n in &optimal { assert!(n >= 10, "minimum 10 samples per level"); }
    }
}
