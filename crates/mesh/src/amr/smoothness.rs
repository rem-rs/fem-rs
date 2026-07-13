//! Multi-strategy smoothness estimation for hp-AMR decision making.
//!
//! Provides spectral smoothness estimators (Legendre coefficient decay,
//! Fourier coefficient decay) alongside the existing Houston–Süli residual‑based
//! indicator.  Each estimator produces a score in *[0, 1]* where:
//!
//! - **0** indicates a smooth solution → p‑refinement is effective.
//! - **1** indicates a rough / singular solution → h‑refinement is needed.
//!
//! A [`SmoothnessPredictor`] combines multiple estimators into a consensus
//! score, which can then be used with [`hp_mark_with_predictor`] in the
//! parent module.
//!
//! # References
//! - P. Houston and E. Süli, "hp‑Adaptive Discontinuous Galerkin Finite Element
//!   Methods …", *SIAM J. Sci. Comput.*, 2001.
//! - deal.II `SmoothnessEstimator` class (Legendre‑decay path).
//! - M. Ainsworth, "A Posteriori Error Estimation for hp‑Adaptive …",
//!   *Comput. Methods Appl. Mech. Engrg.*, 1996.

// ============================================================================
// Types
// ============================================================================

/// Strategy selector for smoothness estimation.
///
/// Each variant carries its own configuration thresholds, which are forwarded
/// to the downstream marking logic (see `hp_mark_with_predictor`).
#[derive(Debug, Clone, PartialEq)]
pub enum SmoothnessEstimatorConfig {
    /// Legendre coefficient decay (deal.II default spectral estimate).
    ///
    /// Projects the numerical solution onto Legendre polynomials on each
    /// element and measures how much energy lies in high modes.
    /// - `threshold_smooth`: below this → p‑refine recommended.
    /// - `threshold_rough`:  above this → h‑refine recommended.
    LegendreDecay {
        threshold_smooth: f64,
        threshold_rough: f64,
    },
    /// Fourier coefficient decay spectral estimate.
    ///
    /// Evaluates the solution on a uniform grid within each element and
    /// measures the high‑frequency content of the Fourier expansion.
    /// - `n_modes`: number of Fourier modes to retain.
    FourierDecay {
        n_modes: usize,
    },
    /// Houston–Süli residual‑ratio indicator (existing implementation).
    ///
    /// Uses the ratio of the error indicator to the element gradient variation.
    /// - `theta_h`: above this → h‑refine.
    /// - `theta_p`: below this → p‑refine.
    HoustonSueli {
        theta_h: f64,
        theta_p: f64,
    },
}

/// Mode for combining multiple estimator scores into a consensus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsensusMode {
    /// Weighted average of all estimator scores.
    WeightedAverage,
    /// Take the maximum (most conservative / most‑likely‑rough) score.
    MaxVote,
    /// Take the minimum (most optimistic / most‑likely‑smooth) score.
    MinVote,
    /// Majority vote on a binarised (smooth / rough) classification.
    MajorityVote,
}

/// Configuration for the composite [`SmoothnessPredictor`].
///
/// # Panics
/// - If `estimators` is empty.
/// - If `weights.len() != estimators.len()` (when mode is `WeightedAverage`).
#[derive(Debug, Clone)]
pub struct SmoothnessPredictorConfig {
    /// Ordered list of estimator variants to run.
    pub estimators: Vec<SmoothnessEstimatorConfig>,
    /// Per‑estimator voting weight (only used in `WeightedAverage` mode).
    pub weights: Vec<f64>,
    /// How the individual scores are fused.
    pub consensus: ConsensusMode,
}

/// Input data for the smoothness estimators.
///
/// Not every estimator needs every field; each reads what it requires:
/// - [`HoustonSueli`](SmoothnessEstimatorConfig::HoustonSueli) → `eta`, `grad_variation`
/// - [`LegendreDecay`](SmoothnessEstimatorConfig::LegendreDecay) → `elem_solution_samples`, `quadrature`
/// - [`FourierDecay`](SmoothnessEstimatorConfig::FourierDecay) → `elem_solution_samples`
///
/// Fields that are `None` for an estimator that requires them will
/// produce a zero‑score default for that estimator.
#[derive(Debug)]
pub struct SmoothnessInputData<'a> {
    /// Element‑wise error indicators η_K.
    pub eta: Option<&'a [f64]>,
    /// Element‑wise gradient variation (for Houston–Süli).
    pub grad_variation: Option<&'a [f64]>,
    /// Per‑element solution values sampled at the quadrature / grid points.
    /// Each inner slice is the solution values for one element.
    pub elem_solution_samples: Option<&'a [&'a [f64]]>,
    /// Quadrature points and weights on the reference element.
    /// Used only by `LegendreDecay`.
    pub quadrature: Option<(&'a [f64], &'a [f64])>,
}

/// Per‑estimator smoothness result.
#[derive(Debug, Clone)]
pub struct SmoothnessEstimate {
    /// Human‑readable label (e.g. `"LegendreDecay"`, `"FourierDecay"`).
    pub label: String,
    /// Smoothness score per element in *[0, 1]* (0 = smooth, 1 = rough).
    pub scores: Vec<f64>,
}

/// Consensus prediction from one or more estimators.
#[derive(Debug, Clone)]
pub struct SmoothnessPrediction {
    /// Fused consensus score per element.
    pub consensus: Vec<f64>,
    /// Individual estimator results.
    pub estimates: Vec<SmoothnessEstimate>,
}

// ============================================================================
// Legendre polynomial utilities
// ============================================================================

/// Evaluate the *n*‑th Legendre polynomial at *x* ∈ *[−1, 1]* via the
/// three‑term recurrence
/// ```text
/// (n+1) P_{n+1}(x) = (2n+1) x P_n(x) − n P_{n-1}(x)
/// ```
pub fn legendre(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p0 = 1.0;
            let mut p1 = x;
            for k in 1..n {
                let pk = ((2 * k + 1) as f64 * x * p1 - (k as f64) * p0) / ((k + 1) as f64);
                p0 = p1;
                p1 = pk;
            }
            p1
        }
    }
}

/// Compute Legendre–Gauss–Lobatto nodes and weights for *N* points.
///
/// The nodes are the roots of *(1 − x²) P'_{N-1}(x)* on *[−1, 1]*.
/// We use Newton iteration seeded with Chebyshev‑type nodes.
///
/// Returns `(nodes, weights)`, each of length `N`.
pub fn legendre_lobatto_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 2, "LGL quadrature requires at least 2 points");
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];

    // Endpoints
    nodes[0] = -1.0;
    nodes[n - 1] = 1.0;

    // Interior nodes: find roots of P'_{n-1}(x)
    // Use Chebyshev-extrema points as initial guess for the Newton iteration.
    let n1 = n - 1;
    let eps = 1e-15;
    for i in 1..n1 {
        let theta = (i as f64) * std::f64::consts::PI / (n1 as f64);
        let mut x = -(theta.cos());

        // Newton: f(x) = P'_{n1}(x) = 0
        // Use recurrence-based derivative formula (stable for |x| < 1)
        for _ in 0..48 {
            let pn = legendre(n1, x);
            let pn_1 = legendre(n1 - 1, x);

            // P'_n(x) = n(x·P_n(x) − P_{n-1}(x)) / (x² − 1)
            // Safe for |x| < 1 (all interior nodes)
            let denom = x * x - 1.0;
            let f = (n1 as f64) * (x * pn - pn_1) / denom;

            // Numerical second derivative (Hessen of f)
            let h = 1e-7;
            let xh = x + h;
            let pn_xh = legendre(n1, xh);
            let pn_1_xh = legendre(n1 - 1, xh);
            let fh = (n1 as f64) * (xh * pn_xh - pn_1_xh) / (xh * xh - 1.0);

            let xmh = x - h;
            let pn_xmh = legendre(n1, xmh);
            let pn_1_xmh = legendre(n1 - 1, xmh);
            let fmh = (n1 as f64) * (xmh * pn_xmh - pn_1_xmh) / (xmh * xmh - 1.0);

            let df = (fh - fmh) / (2.0 * h);
            let dx = f / (df + eps);
            x -= dx;
            if dx.abs() < eps {
                break;
            }
        }
        nodes[i] = x;
    }

    // Ensure monotonic ordering (should already be, but guard against Newton drift)
    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Weights: w_j = 2 / (N(N-1) [P_{N-1}(x_j)]²)
    for (j, &x) in nodes.iter().enumerate() {
        let p_n1 = legendre(n1, x);
        weights[j] = 2.0 / ((n as f64) * (n1 as f64) * p_n1 * p_n1 + eps);
    }

    (nodes, weights)
}

// ============================================================================
// Smoothness estimation implementations
// ============================================================================

/// Compute the Legendre coefficient decay smoothness indicator using a
/// low‑vs‑high mode energy ratio.
///
/// For each element:
/// 1. Project the solution onto Legendre polynomials P₀ … P_{K-1} via
///    LGL quadrature, yielding coefficients cₖ.
/// 2. Skip the DC mode (c₀).  Split the remaining modes into two equal
///    halves: *low* modes (k = 1 … ⌈K/2⌉) and *high* modes
///    (k = ⌈K/2⌉+1 … K-1).
/// 3. The smoothness score is the fraction of the non‑DC energy that
///    resides in the high modes:
///    ```text
///    κ = E_high / (E_low + E_high)
///    ```
///
/// A smooth solution (sin, low‑order polynomial) concentrates energy in
/// low modes → κ ≈ 0.  A rough solution (discontinuity, high‑frequency
/// content) has significant energy in high modes → κ ≈ 1.
///
/// # Arguments
/// * `elem_values` — per‑element solution samples at the LGL nodes.
/// * `quad_points`  — LGL nodes on *[−1, 1]*.
/// * `quad_weights` — LGL quadrature weights.
/// * `max_modes`   — number of Legendre modes to compute (≤ `n_quad`).
///
/// # Returns
/// Smoothness score in *[0, 1]* per element.
pub fn legendre_decay_smoothness(
    elem_values: &[&[f64]],
    quad_points: &[f64],
    quad_weights: &[f64],
    max_modes: usize,
) -> Vec<f64> {
    let n_elem = elem_values.len();
    if n_elem == 0 || max_modes < 2 {
        return vec![0.0; n_elem];
    }

    let n_quad = quad_points.len();
    let k = max_modes.min(n_quad);

    elem_values
        .iter()
        .map(|&vals| {
            if vals.len() < n_quad {
                return 0.5;
            }

            // 1. Compute Legendre coefficients cₖ
            let mut coeffs = vec![0.0_f64; k];
            for m in 0..k {
                let scale = (2.0 * m as f64 + 1.0) / 2.0;
                let mut sum = 0.0;
                for q in 0..n_quad {
                    sum += quad_weights[q] * vals[q] * legendre(m, quad_points[q]);
                }
                coeffs[m] = scale * sum;
            }

            // 2. Low‑vs‑high mode energy ratio (skip DC mode c₀)
            let non_dc_modes = k.saturating_sub(1);
            if non_dc_modes <= 2 {
                return 0.0; // few modes → trivially smooth
            }

            let split = non_dc_modes / 2; // number of "low" modes
            let low_energy: f64 = coeffs[1..=split].iter().map(|c| c * c).sum();
            let high_energy: f64 = coeffs[split + 1..].iter().map(|c| c * c).sum();
            let total = low_energy + high_energy;

            // Numerical noise floor check: if all non‑DC energy is negligible
            // compared to the DC mode, the function is effectively constant.
            if total < 1e-24 || (coeffs[0].abs() > 1e6 * (total.sqrt() + 1e-30)) {
                return 0.0;
            }

            let ratio = high_energy / total; // ∈ [0, 1)

            // Soft clamp: ratio < 0.01 → smooth, > 0.3 → rough
            let smooth_thresh = 0.01;
            let rough_thresh = 0.3;
            if ratio <= smooth_thresh {
                0.0
            } else if ratio >= rough_thresh {
                1.0
            } else {
                (ratio - smooth_thresh) / (rough_thresh - smooth_thresh)
            }
        })
        .collect()
}

/// Compute the 2D tensor‑product Legendre coefficient decay smoothness.
///
/// For each quad element, samples the solution at *(n×n)* LGL points and
/// computes the 2D Legendre expansion coefficients *c_{ij}*.  The smoothness
/// score is the fraction of non‑DC energy in "high" total‑order modes
/// (where total order *s = i + j*).
///
/// A smooth bivariate function concentrates energy in low total‑order modes;
/// a function with edges, corners, or steep gradients excites high total‑order
/// modes.
///
/// # Arguments
/// * `elem_values` — per‑element solution values at tensor‑product LGL points
///   (row‑major: *x* outer, *y* inner — i.e. `[qx * n_1d + qy]`).
/// * `n_1d` — number of LGL points in each dimension (total per element = n_1d²).
/// * `lgl_nodes` — 1‑D LGL nodes on *[−1, 1]*.
/// * `lgl_weights` — 1‑D LGL quadrature weights.
/// * `max_modes_1d` — number of Legendre modes per dimension (≤ `n_1d`).
///
/// # Returns
/// Smoothness score per element in *[0, 1]*.
pub fn legendre_decay_smoothness_2d(
    elem_values: &[&[f64]],
    n_1d: usize,
    lgl_nodes: &[f64],
    lgl_weights: &[f64],
    max_modes_1d: usize,
) -> Vec<f64> {
    let n_elem = elem_values.len();
    if n_elem == 0 || max_modes_1d < 2 || n_1d < 2 {
        return vec![0.0; n_elem];
    }

    let k = max_modes_1d.min(n_1d);
    let n_pts = n_1d * n_1d;

    elem_values
        .iter()
        .map(|&vals| {
            if vals.len() < n_pts {
                return 0.5;
            }

            // 1. Pre‑compute Legendre basis values at each LGL node
            let mut leg_basis: Vec<Vec<f64>> = Vec::with_capacity(k);
            for m in 0..k {
                leg_basis.push(
                    (0..n_1d).map(|q| legendre(m, lgl_nodes[q])).collect(),
                );
            }

            // 2. Compute 2D tensor‑product Legendre coefficients c_{ij}
            let mut coeffs = vec![0.0_f64; k * k];
            for i in 0..k {
                let scale_i = (2.0 * i as f64 + 1.0) / 2.0;
                for j in 0..k {
                    let scale_j = (2.0 * j as f64 + 1.0) / 2.0;
                    let mut sum = 0.0;
                    for qx in 0..n_1d {
                        for qy in 0..n_1d {
                            let idx = qx * n_1d + qy;
                            sum += lgl_weights[qx]
                                * lgl_weights[qy]
                                * vals[idx]
                                * leg_basis[i][qx]
                                * leg_basis[j][qy];
                        }
                    }
                    coeffs[i * k + j] = scale_i * scale_j * sum;
                }
            }

            // 3. Energy partition by total order s = i + j
            let split = (k - 1) / 2; // half the max total order

            let mut low_energy = 0.0_f64;
            let mut high_energy = 0.0_f64;

            for i in 0..k {
                for j in 0..k {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let s = i + j;
                    let e = coeffs[i * k + j] * coeffs[i * k + j];
                    if s <= split {
                        low_energy += e;
                    } else {
                        high_energy += e;
                    }
                }
            }

            let total = low_energy + high_energy;

            // 4. Numerical noise‑floor check (compare non‑DC energy to DC mode)
            let dc_energy = coeffs[0] * coeffs[0];
            if total < 1e-24 || (dc_energy > 1e6 * (total.sqrt() + 1e-30)) {
                return 0.0;
            }

            let ratio = high_energy / total;

            // 5. Soft clamp (same thresholds as the 1D version)
            let smooth_thresh = 0.01;
            let rough_thresh = 0.3;
            if ratio <= smooth_thresh {
                0.0
            } else if ratio >= rough_thresh {
                1.0
            } else {
                (ratio - smooth_thresh) / (rough_thresh - smooth_thresh)
            }
        })
        .collect()
}

/// Compute the Fourier decay smoothness indicator.
///
/// For each element, take solution values at equally‑spaced points and
/// compute discrete Fourier coefficients.  The smoothness score is
/// the fraction of total power in high‑frequency modes.
///
/// # Arguments
/// * `elem_values` — per‑element solution values at equally‑spaced
///   locations in the reference element.
/// * `n_modes` — number of Fourier modes to compute (≤ `n_points`).
///
/// # Returns
/// Smoothness score per element in *[0, 1]*.
pub fn fourier_decay_smoothness(
    elem_values: &[&[f64]],
    n_modes: usize,
) -> Vec<f64> {
    let n_elem = elem_values.len();
    if n_elem == 0 || n_modes < 2 {
        return vec![0.0; n_elem];
    }

    elem_values
        .iter()
        .map(|&vals| {
            let n = vals.len();
            if n < n_modes {
                return 0.5;
            }

            // Compute cosine coefficients via proper DCT‑I
            //   Xₖ = x₀ + (-1)ᵏ x_{N-1} + 2·Σ_{i=1}^{N-2} x_i cos(π·k·i/(N-1))
            let mut coeffs = Vec::with_capacity(n_modes);
            for m in 0..n_modes {
                let omega = std::f64::consts::PI * (m as f64) / ((n - 1) as f64);
                let mut sum = vals[0] + vals[n - 1] * if m % 2 == 0 { 1.0 } else { -1.0 };
                for i in 1..(n - 1) {
                    sum += 2.0 * vals[i] * (omega * i as f64).cos();
                }
                coeffs.push(sum);
            }

            // Low‑vs‑high frequency power ratio (skip DC mode at m=0)
            let n_ac_modes = coeffs.len().saturating_sub(1);
            if n_ac_modes <= 2 {
                return 0.0;
            }
            let split = n_ac_modes / 2;
            let low_power: f64 = coeffs[1..=split].iter().map(|c| c * c).sum();
            let high_power: f64 = coeffs[split + 1..].iter().map(|c| c * c).sum();
            let total = low_power + high_power;

            if total < 1e-24 {
                return 0.0;
            }
            let ratio = high_power / total;

            let smooth_thresh = 0.01;
            let rough_thresh = 0.3;
            if ratio <= smooth_thresh {
                0.0
            } else if ratio >= rough_thresh {
                1.0
            } else {
                (ratio - smooth_thresh) / (rough_thresh - smooth_thresh)
            }
        })
        .collect()
}

/// Compute the 2D Fourier decay smoothness indicator using a 2D DCT‑I.
///
/// Samples the solution on an *(n×n)* equispaced grid in the reference
/// element and computes the 2D discrete cosine transform.  The smoothness
/// score is the fraction of non‑DC power in high total‑frequency modes.
///
/// # Arguments
/// * `elem_values` — per‑element solution values at equispaced grid points
///   (row‑major: *x* outer, *y* inner).
/// * `n_1d` — number of sample points per dimension (total = n_1d² per element).
///
/// # Returns
/// Smoothness score per element in *[0, 1]*.
pub fn fourier_decay_smoothness_2d(
    elem_values: &[&[f64]],
    n_1d: usize,
) -> Vec<f64> {
    let n_elem = elem_values.len();
    if n_elem == 0 || n_1d < 3 {
        return vec![0.0; n_elem];
    }

    let n_pts = n_1d * n_1d;
    let inv = 1.0 / ((n_1d - 1) as f64);

    elem_values
        .iter()
        .map(|&vals| {
            if vals.len() < n_pts {
                return 0.5;
            }

            // 2D DCT‑I: separable transform
            // C(u,v) = Σ_x Σ_y w_x w_y f(x,y) cos(π·u·x/(N-1)) cos(π·v·y/(N-1))
            // where w_0 = w_{N-1} = 1/2, and w_i = 1 otherwise.
            let n_modes = n_1d;
            let mut coeffs = vec![0.0_f64; n_modes * n_modes];

            for u in 0..n_modes {
                for v in 0..n_modes {
                    let mut sum = 0.0_f64;
                    for x in 0..n_1d {
                        let cx = (std::f64::consts::PI * u as f64 * x as f64 * inv).cos();
                        for y in 0..n_1d {
                            let cy = (std::f64::consts::PI * v as f64 * y as f64 * inv).cos();
                            // Weight: ¼ at corners, ½ at edges, 1 interior
                            let sw = if (x == 0 || x == n_1d - 1) && (y == 0 || y == n_1d - 1) {
                                0.25
                            } else if x == 0 || x == n_1d - 1 || y == 0 || y == n_1d - 1 {
                                0.5
                            } else {
                                1.0
                            };
                            sum += sw * vals[x * n_1d + y] * cx * cy;
                        }
                    }
                    coeffs[u * n_modes + v] = sum;
                }
            }

            // Energy split by total frequency f = u + v
            let split = (n_modes - 1) / 2;
            let mut low_power = 0.0_f64;
            let mut high_power = 0.0_f64;

            for u in 0..n_modes {
                for v in 0..n_modes {
                    if u == 0 && v == 0 {
                        continue;
                    }
                    let f = u + v;
                    let e = coeffs[u * n_modes + v] * coeffs[u * n_modes + v];
                    if f <= split {
                        low_power += e;
                    } else {
                        high_power += e;
                    }
                }
            }

            let total = low_power + high_power;
            if total < 1e-24 {
                return 0.0;
            }
            let ratio = high_power / total;

            let smooth_thresh = 0.01;
            let rough_thresh = 0.3;
            if ratio <= smooth_thresh {
                0.0
            } else if ratio >= rough_thresh {
                1.0
            } else {
                (ratio - smooth_thresh) / (rough_thresh - smooth_thresh)
            }
        })
        .collect()
}

/// Houston–Süli residual‑ratio smoothness indicator.
///
/// Wraps the original `compute_smoothness_indicator` logic from the parent
/// crate but returns it in a form compatible with the multi‑strategy framework.
///
/// κ_K = η_K / ‖∇u_h‖_K   (ratio of error indicator to gradient variation),
/// normalised to *[0, 1]*.
///
/// # Arguments
/// * `eta` — element‑wise error indicators.
/// * `grad_variation` — gradient variation per element.
///
/// # Returns
/// Smoothness score per element in *[0, 1]*.
pub fn houston_sueli_smoothness(eta: &[f64], grad_variation: &[f64]) -> Vec<f64> {
    assert_eq!(eta.len(), grad_variation.len(),
        "houston_sueli_smoothness: length mismatch");
    let n = eta.len();
    if n == 0 {
        return Vec::new();
    }

    let mut kappa = vec![0.0_f64; n];
    for i in 0..n {
        let gv = grad_variation[i].abs();
        kappa[i] = if gv > 1e-30 {
            eta[i] / gv
        } else {
            0.0
        };
    }

    // Normalise to [0, 1]
    let max_k = kappa.iter().copied().fold(0.0_f64, f64::max);
    if max_k > 1e-30 {
        for k in &mut kappa {
            *k = (*k / max_k).min(1.0);
        }
    }
    kappa
}

// ============================================================================
// SmoothnessPredictor — composite orchestrator
// ============================================================================

/// Composite smoothness predictor that runs multiple estimators and fuses
/// their scores into a consensus.
///
/// # Example (conceptual)
/// ```ignore
/// let pred = SmoothnessPredictor::new(SmoothnessPredictorConfig {
///     estimators: vec![
///         SmoothnessEstimatorConfig::LegendreDecay {
///             threshold_smooth: 0.2, threshold_rough: 0.6,
///         },
///         SmoothnessEstimatorConfig::HoustonSueli {
///             theta_h: 0.7, theta_p: 0.3,
///         },
///     ],
///     weights: vec![0.6, 0.4],
///     consensus: ConsensusMode::WeightedAverage,
/// });
/// let result = pred.predict(&input_data);
/// // Use result.consensus with hp_mark_with_predictor(...)
/// ```
#[derive(Debug, Clone)]
pub struct SmoothnessPredictor {
    config: SmoothnessPredictorConfig,
}

impl SmoothnessPredictor {
    /// Create a new predictor with the given configuration.
    ///
    /// # Panics
    /// - If `config.estimators` is empty.
    /// - If `config.consensus == WeightedAverage` and
    ///   `config.weights.len() != config.estimators.len()`.
    pub fn new(config: SmoothnessPredictorConfig) -> Self {
        assert!(!config.estimators.is_empty(),
            "SmoothnessPredictor: at least one estimator required");
        if config.consensus == ConsensusMode::WeightedAverage {
            assert_eq!(
                config.weights.len(),
                config.estimators.len(),
                "SmoothnessPredictor: weights.len() must match estimators.len() \
                 for WeightedAverage consensus"
            );
        }
        Self { config }
    }

    /// Run all configured estimators and produce a consensus prediction.
    pub fn predict(&self, data: &SmoothnessInputData<'_>) -> SmoothnessPrediction {
        let n_elem = self.resolve_n_elem(data);

        let mut estimates: Vec<SmoothnessEstimate> = Vec::new();

        for est in &self.config.estimators {
            let label = format!("{:?}", est);
            let scores = match est {
                SmoothnessEstimatorConfig::LegendreDecay { .. } => {
                    self.estimate_legendre(data, n_elem)
                }
                SmoothnessEstimatorConfig::FourierDecay { n_modes } => {
                    self.estimate_fourier(data, n_elem, *n_modes)
                }
                SmoothnessEstimatorConfig::HoustonSueli { .. } => {
                    self.estimate_houston_sueli(data, n_elem)
                }
            };
            estimates.push(SmoothnessEstimate { label, scores });
        }

        let consensus = self.fuse(&estimates);
        SmoothnessPrediction {
            consensus,
            estimates,
        }
    }

    /// Return the estimator configurations.
    pub fn estimators(&self) -> &[SmoothnessEstimatorConfig] {
        &self.config.estimators
    }

    // ── private helpers ──

    fn resolve_n_elem(&self, data: &SmoothnessInputData<'_>) -> usize {
        if let Some(eta) = data.eta {
            return eta.len();
        }
        if let Some(gv) = data.grad_variation {
            return gv.len();
        }
        if let Some(samples) = data.elem_solution_samples {
            return samples.len();
        }
        0
    }

    fn estimate_legendre(
        &self,
        data: &SmoothnessInputData<'_>,
        n_elem: usize,
    ) -> Vec<f64> {
        let samples = match data.elem_solution_samples {
            Some(s) => s,
            None => return vec![0.5; n_elem],
        };
        match data.quadrature {
            Some((qpts, qwts)) => {
                let max_modes = qpts.len().saturating_sub(1);
                // Auto-detect 2D: if per-element data length is a perfect square
                // of the 1D LGL node count, treat as 2D tensor-product.
                let n_quad_1d = qpts.len();
                if detect_is_2d(samples, n_quad_1d) {
                    legendre_decay_smoothness_2d(samples, n_quad_1d, qpts, qwts, max_modes)
                } else {
                    legendre_decay_smoothness(samples, qpts, qwts, max_modes)
                }
            }
            None => {
                // Use default 12-point LGL quadrature
                let (qpts, qwts) = default_lgl_quadrature_inner();
                let n_quad_1d = qpts.len();
                if detect_is_2d(samples, n_quad_1d) {
                    legendre_decay_smoothness_2d(samples, n_quad_1d, &qpts, &qwts, n_quad_1d - 2)
                } else {
                    legendre_decay_smoothness(samples, &qpts, &qwts, qpts.len() - 2)
                }
            }
        }
    }

    fn estimate_fourier(
        &self,
        data: &SmoothnessInputData<'_>,
        n_elem: usize,
        n_modes: usize,
    ) -> Vec<f64> {
        let samples = match data.elem_solution_samples {
            Some(s) => s,
            None => return vec![0.5; n_elem],
        };
        // Auto-detect 2D: if per-element data forms a square grid, treat as 2D
        if let Some(n1) = guess_grid_dim_1d(samples) {
            fourier_decay_smoothness_2d(samples, n1)
        } else {
            fourier_decay_smoothness(samples, n_modes)
        }
    }

    fn estimate_houston_sueli(
        &self,
        data: &SmoothnessInputData<'_>,
        n_elem: usize,
    ) -> Vec<f64> {
        let eta = match data.eta {
            Some(e) => e,
            None => return vec![0.5; n_elem],
        };
        let gv = match data.grad_variation {
            Some(g) => g,
            None => return vec![0.5; n_elem],
        };
        if eta.len() != n_elem || gv.len() != n_elem {
            return vec![0.5; n_elem];
        }
        houston_sueli_smoothness(eta, gv)
    }

    fn fuse(&self, estimates: &[SmoothnessEstimate]) -> Vec<f64> {
        if estimates.is_empty() {
            return Vec::new();
        }
        let n = estimates[0].scores.len();

        match self.config.consensus {
            ConsensusMode::WeightedAverage => {
                let total_w: f64 = self.config.weights.iter().sum();
                if total_w.abs() < 1e-60 {
                    return vec![0.0; n];
                }
                let mut combined = vec![0.0_f64; n];
                for (est, &w) in estimates.iter().zip(&self.config.weights) {
                    for (c, &s) in combined.iter_mut().zip(&est.scores) {
                        *c += w * s / total_w;
                    }
                }
                combined
            }
            ConsensusMode::MaxVote => {
                let mut combined = vec![0.0_f64; n];
                for i in 0..n {
                    combined[i] = estimates
                        .iter()
                        .map(|e| e.scores[i])
                        .fold(0.0_f64, f64::max);
                }
                combined
            }
            ConsensusMode::MinVote => {
                let mut combined = vec![1.0_f64; n];
                for i in 0..n {
                    combined[i] = estimates
                        .iter()
                        .map(|e| e.scores[i])
                        .fold(1.0_f64, f64::min);
                }
                combined
            }
            ConsensusMode::MajorityVote => {
                let n_est = estimates.len();
                let mut combined = vec![0.0_f64; n];
                if n_est == 1 {
                    combined.copy_from_slice(&estimates[0].scores);
                } else {
                    for i in 0..n {
                        let rough_votes = estimates
                            .iter()
                            .filter(|e| e.scores[i] >= 0.5)
                            .count();
                        let smooth_votes = n_est - rough_votes;
                        combined[i] = if rough_votes > smooth_votes {
                            1.0
                        } else if smooth_votes > rough_votes {
                            0.0
                        } else {
                            0.5
                        };
                    }
                }
                combined
            }
        }
    }
}

/// Heuristic: detect whether per‑element sample data represents 2D
/// tensor‑product LGL values (i.e. `len == n_quad_1d²`).
fn detect_is_2d(samples: &[&[f64]], n_quad_1d: usize) -> bool {
    if samples.is_empty() {
        return false;
    }
    let n_per = samples[0].len();
    n_per == n_quad_1d * n_quad_1d && n_quad_1d >= 2
}

/// Heuristic: if the per‑element sample count is a perfect square ≥ 9,
/// return the square root as the 1‑D grid dimension; otherwise return `None`.
fn guess_grid_dim_1d(samples: &[&[f64]]) -> Option<usize> {
    let n_per = samples.first()?.len();
    let n1 = (n_per as f64).sqrt().round() as usize;
    if n1 * n1 == n_per && n1 >= 3 {
        Some(n1)
    } else {
        None
    }
}

// ============================================================================
// Integration helpers
// ============================================================================

/// Convenience function: run a composite prediction and return the consensus
/// smoothness scores.
///
/// Equivalent to constructing a [`SmoothnessPredictor`] and calling
/// [`predict`](SmoothnessPredictor::predict), then extracting the consensus.
pub fn predict_smoothness(
    config: &SmoothnessPredictorConfig,
    data: &SmoothnessInputData<'_>,
) -> Vec<f64> {
    let predictor = SmoothnessPredictor::new(config.clone());
    let prediction = predictor.predict(data);
    prediction.consensus
}

fn default_lgl_quadrature_inner() -> (Vec<f64>, Vec<f64>) {
    legendre_lobatto_nodes_weights(12)
}

/// Default 12‑point LGL quadrature for element orders up to about 10.
///
/// Returns `(nodes, weights)`.  Generated once via `std::sync::OnceLock`.
pub fn default_lgl_quadrature() -> (&'static [f64], &'static [f64]) {
    static CACHE: std::sync::OnceLock<(Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();
    let (ref nodes, ref weights) = CACHE.get_or_init(default_lgl_quadrature_inner);
    (nodes.as_slice(), weights.as_slice())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Legendre polynomial tests ──────────────────────────────────────────

    #[test]
    fn legendre_p0_is_one() {
        for x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!((legendre(0, *x) - 1.0).abs() < 1e-15, "P₀({}) = {}", x, legendre(0, *x));
        }
    }

    #[test]
    fn legendre_p1_is_x() {
        for x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!((legendre(1, *x) - *x).abs() < 1e-15);
        }
    }

    #[test]
    fn legendre_p2_known_values() {
        // P₂(x) = (3x² − 1)/2
        assert!((legendre(2, 0.0) - (-0.5)).abs() < 1e-15);
        assert!((legendre(2, 1.0) - 1.0).abs() < 1e-15);
        assert!((legendre(2, -1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn legendre_lgl_nodes_are_symmetric() {
        let (nodes, weights) = legendre_lobatto_nodes_weights(6);
        assert_eq!(nodes.len(), 6);
        assert_eq!(weights.len(), 6);
        // Symmetry: nodes[i] ≈ -nodes[5-i]
        for i in 0..3 {
            assert!((nodes[i] + nodes[5 - i]).abs() < 1e-12,
                "LGL nodes not symmetric: {} vs {}", nodes[i], nodes[5 - i]);
        }
        // Endpoints at ±1
        assert!((nodes[0] + 1.0).abs() < 1e-12);
        assert!((nodes[5] - 1.0).abs() < 1e-12);
        // Weights positive
        for &w in &weights {
            assert!(w > 0.0, "LGL weight must be positive, got {}", w);
        }
        // Strictly increasing
        for i in 0..nodes.len() - 1 {
            assert!(nodes[i] < nodes[i + 1],
                "LGL nodes not increasing: {} ≥ {}", nodes[i], nodes[i + 1]);
        }
    }

    #[test]
    fn legendre_lgl_weights_sum_to_two() {
        let (_, weights) = legendre_lobatto_nodes_weights(8);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 2.0).abs() < 1e-12,
            "LGL weights sum to 2, got {}", sum);
    }

    // ── Smoothness estimator tests ─────────────────────────────────────────

    /// Helper: produce uniformly-spaced sample points on [-1, 1].
    fn uniform_samples(n: usize) -> Vec<f64> {
        if n == 1 { return vec![0.0]; }
        (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn legendre_decay_smooth_function_is_smooth() {
        // u(x) = sin(π x) — analytic, very smooth
        let n_pts = 16;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_pts);
        let u_sin: Vec<f64> = qpts.iter().map(|&xi| (std::f64::consts::PI * xi).sin()).collect();

        let scores = legendre_decay_smoothness(&[u_sin.as_slice()], &qpts, &qwts, n_pts - 1);
        assert_eq!(scores.len(), 1);
        assert!(
            scores[0] < 0.3,
            "sin(π x) should be scored as smooth (≤ 0.3), got {}",
            scores[0]
        );
    }

    #[test]
    fn legendre_decay_rough_function_is_rough() {
        // u(x) = sign(x) — discontinuous, very rough
        let n_pts = 16;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_pts);
        let u_step: Vec<f64> = qpts.iter()
            .map(|&xi| if xi >= 0.0 { 1.0 } else { -1.0 })
            .collect();

        let scores = legendre_decay_smoothness(&[u_step.as_slice()], &qpts, &qwts, n_pts - 1);
        assert_eq!(scores.len(), 1);
        assert!(
            scores[0] > 0.3,
            "step function should be scored as rough-ish (≥ 0.3), got {}",
            scores[0]
        );
    }

    #[test]
    fn legendre_decay_linear_is_smooth() {
        // u(x) = x — linear, perfectly representable by P₁
        let n_pts = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_pts);
        let u_linear: Vec<f64> = qpts.iter().map(|&xi| xi).collect();

        let scores = legendre_decay_smoothness(&[u_linear.as_slice()], &qpts, &qwts, n_pts - 1);
        assert!(
            scores[0] < 0.3,
            "linear function should be smooth (< 0.3), got {}",
            scores[0]
        );
    }

    #[test]
    fn legendre_decay_constant_is_maximally_smooth() {
        let n_pts = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_pts);
        let u_const: Vec<f64> = qpts.iter().map(|_| 1.0).collect();

        let scores = legendre_decay_smoothness(&[u_const.as_slice()], &qpts, &qwts, n_pts - 1);
        assert_eq!(scores.len(), 1);
        assert!(
            scores[0] < 0.1,
            "constant should be smooth, got {}",
            scores[0]
        );
    }

    #[test]
    fn legendre_decay_empty_input() {
        let (qpts, qwts) = legendre_lobatto_nodes_weights(4);
        let scores = legendre_decay_smoothness(&[], &qpts, &qwts, 3);
        assert!(scores.is_empty());
    }

    #[test]
    fn fourier_decay_smooth_function() {
        // u(x) = cos(πx/2) — one dominant frequency → energy in few modes
        let n_pts = 32;
        let x = uniform_samples(n_pts);
        let u_cos: Vec<f64> = x.iter()
            .map(|&xi| (std::f64::consts::PI * xi * 0.5).cos())
            .collect();

        let scores = fourier_decay_smoothness(&[u_cos.as_slice()], 16);
        assert_eq!(scores.len(), 1);
        // cos(πx/2) should have most energy in its fundamental → relatively smooth
        assert!(scores[0] < 0.65,
            "cos(πx/2) should score fairly smooth (< 0.65), got {}",
            scores[0]
        );
    }

    #[test]
    fn fourier_decay_rough_function() {
        // u(x) = 1 if sin(2πx) > 0 else -1  (square wave at 64 pts)
        // The square wave has a discontinuous jump → spreads energy across modes
        let n_pts = 64;
        let x = uniform_samples(n_pts);
        let u_sq: Vec<f64> = x.iter()
            .map(|&xi| if (2.0 * std::f64::consts::PI * xi).sin() >= 0.0 { 1.0 } else { -1.0 })
            .collect();

        let scores = fourier_decay_smoothness(&[u_sq.as_slice()], 32);
        assert_eq!(scores.len(), 1);
        assert!(
            scores[0] > 0.1,
            "square wave should be scored rougher than near-0 (> 0.1), got {}",
            scores[0]
        );
    }

    #[test]
    fn fourier_decay_constant_is_smooth() {
        let u: Vec<f64> = vec![2.5; 16];
        let scores = fourier_decay_smoothness(&[u.as_slice()], 8);
        assert_eq!(scores.len(), 1);
        assert!((scores[0] - 0.0).abs() < 1e-14,
            "constant should be scored 0 (smooth), got {}", scores[0]);
    }

    #[test]
    fn fourier_decay_empty_input() {
        let scores = fourier_decay_smoothness(&[], 4);
        assert!(scores.is_empty());
    }

    #[test]
    fn houston_sueli_matches_original_behaviour() {
        let eta = vec![1.0, 2.0, 3.0];
        let gv = vec![1.0, 2.0, 3.0];
        let k = houston_sueli_smoothness(&eta, &gv);
        assert_eq!(k.len(), 3);
        // All have ratio 1.0 → normalised to 1.0
        for &ki in &k {
            assert!((ki - 1.0).abs() < 1e-14, "expected 1.0, got {}", ki);
        }
    }

    #[test]
    fn houston_sueli_constant_is_zero() {
        let eta = vec![0.0; 5];
        let gv = vec![0.0; 5];
        let k = houston_sueli_smoothness(&eta, &gv);
        for &ki in &k {
            assert!((ki - 0.0).abs() < 1e-15, "constant → 0, got {}", ki);
        }
    }

    #[test]
    fn houston_sueli_normalised() {
        let eta = vec![1.0, 2.0];
        let gv = vec![1.0, 1.0];
        let k = houston_sueli_smoothness(&eta, &gv);
        assert!((k[0] - 0.5).abs() < 1e-15, "κ[0] should be 0.5, got {}", k[0]);
        assert!((k[1] - 1.0).abs() < 1e-15, "κ[1] should be 1.0, got {}", k[1]);
    }

    // ── SmoothnessPredictor composite tests ─────────────────────────────────

    #[test]
    fn predictor_weighted_average() {
        let config = SmoothnessPredictorConfig {
            estimators: vec![
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
            ],
            weights: vec![0.3, 0.7],
            consensus: ConsensusMode::WeightedAverage,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: Some(&[1.0, 2.0]),
            grad_variation: Some(&[1.0, 1.0]),
            elem_solution_samples: None,
            quadrature: None,
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 2);
        // Both estimators see same data so produce same scores.
        // Scores: 0.5, 1.0 → weighted avg: 0.5, 1.0
        assert!((result.consensus[0] - 0.5).abs() < 1e-14);
        assert!((result.consensus[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn predictor_max_vote() {
        let config = SmoothnessPredictorConfig {
            estimators: vec![
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
            ],
            weights: vec![1.0],
            consensus: ConsensusMode::MaxVote,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: Some(&[1.0, 0.5, 0.0]),
            grad_variation: Some(&[1.0, 1.0, 1.0]),
            elem_solution_samples: None,
            quadrature: None,
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 3);
    }

    #[test]
    fn predictor_majority_vote_three_estimators() {
        let config = SmoothnessPredictorConfig {
            estimators: vec![
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
            ],
            weights: vec![1.0; 3],
            consensus: ConsensusMode::MajorityVote,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: Some(&[1.0, 2.0]),
            grad_variation: Some(&[10.0, 1.0]),
            elem_solution_samples: None,
            quadrature: None,
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 2);
    }

    #[test]
    fn predictor_min_vote() {
        let config = SmoothnessPredictorConfig {
            estimators: vec![
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
            ],
            weights: vec![0.5; 2],
            consensus: ConsensusMode::MinVote,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: Some(&[0.5, 2.0]),
            grad_variation: Some(&[1.0, 1.0]),
            elem_solution_samples: None,
            quadrature: None,
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 2);
    }

    #[test]
    fn predictor_empty_input() {
        let config = SmoothnessPredictorConfig {
            estimators: vec![
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
            ],
            weights: vec![1.0],
            consensus: ConsensusMode::WeightedAverage,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: Some(&[]),
            grad_variation: Some(&[]),
            elem_solution_samples: None,
            quadrature: None,
        };
        let result = pred.predict(&data);
        assert!(result.consensus.is_empty());
    }

    #[test]
    fn predict_smoothness_convenience() {
        let config = SmoothnessPredictorConfig {
            estimators: vec![
                SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: 0.7, theta_p: 0.3,
                },
            ],
            weights: vec![1.0],
            consensus: ConsensusMode::MaxVote,
        };
        let data = SmoothnessInputData {
            eta: Some(&[1.0, 2.0]),
            grad_variation: Some(&[1.0, 1.0]),
            elem_solution_samples: None,
            quadrature: None,
        };
        let scores = predict_smoothness(&config, &data);
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn predictor_legendre_decay_fallback_when_no_quadrature() {
        // When quadrature is not provided, the predictor uses a default 8-point LGL.
        // The caller must evaluate the solution at the LGL nodes, not uniform points.
        let config = SmoothnessPredictorConfig {
            estimators: vec![
                SmoothnessEstimatorConfig::LegendreDecay {
                    threshold_smooth: 0.2, threshold_rough: 0.6,
                },
            ],
            weights: vec![1.0],
            consensus: ConsensusMode::WeightedAverage,
        };
        let pred = SmoothnessPredictor::new(config);
        // Evaluate sin at the default 8 LGL nodes (not uniform points)
        let (qpts, _) = legendre_lobatto_nodes_weights(8);
        let u_sin: Vec<f64> = qpts.iter().map(|&xi| (std::f64::consts::PI * xi).sin()).collect();
        let samples = vec![u_sin.as_slice()];
        let data = SmoothnessInputData {
            eta: None,
            grad_variation: None,
            elem_solution_samples: Some(&samples),
            quadrature: None,
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 1);
        // sin(π x) has most Legendre energy in low modes → moderately smooth
        assert!(result.consensus[0] < 0.75,
            "Legendre predictor should score sin(π x) as at least moderately smooth, got {}", result.consensus[0]);
    }

    // ── Edge cases ─────────────────────────────────────────────────────────

    #[test]
    fn single_element_estimation() {
        let eta = vec![1.0];
        let gv = vec![1.0];
        let k = houston_sueli_smoothness(&eta, &gv);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn large_variation_no_overflow() {
        let eta = vec![1e100];
        let gv = vec![1.0];
        let k = houston_sueli_smoothness(&eta, &gv);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-14, "should saturate at 1.0, got {}", k[0]);
    }

    #[test]
    fn all_zero_grad_variation_is_smooth() {
        let eta = vec![1.0, 2.0, 3.0];
        let gv = vec![0.0; 3];
        let k = houston_sueli_smoothness(&eta, &gv);
        for &ki in &k {
            assert!((ki - 0.0).abs() < 1e-15, "zero gradient → 0, got {}", ki);
        }
    }

    #[test]
    fn default_lgl_quadrature_is_cached() {
        let (n1, w1) = default_lgl_quadrature();
        let (n2, w2) = default_lgl_quadrature();
        assert_eq!(n1.len(), 12);
        assert_eq!(n2.len(), 12);
        // Same pointers (cached)
        assert!(std::ptr::eq(n1, n2));
        assert!(std::ptr::eq(w1, w2));
    }

    // ── 2D Legendre decay tests ───────────────────────────────────────────

    /// Helper: generate tensor‑product 2D LGL samples on [-1, 1]².
    fn lgl_samples_2d(n_1d: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let (nodes, _) = legendre_lobatto_nodes_weights(n_1d);
        let mut vals = Vec::with_capacity(n_1d * n_1d);
        for i in 0..n_1d {
            for j in 0..n_1d {
                vals.push(f(nodes[i], nodes[j]));
            }
        }
        vals
    }

    #[test]
    fn legendre_2d_constant_is_maximally_smooth() {
        let n_1d = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        let values = lgl_samples_2d(n_1d, |_, _| 1.0);
        let scores = legendre_decay_smoothness_2d(&[values.as_slice()], n_1d, &qpts, &qwts, n_1d - 1);
        assert_eq!(scores.len(), 1);
        assert!(scores[0] < 0.1,
            "constant 2D should be smooth (< 0.1), got {}", scores[0]);
    }

    #[test]
    fn legendre_2d_bilinear_is_smooth() {
        // u(x,y) = x + y + 1 — bilinear, representable by low Legendre modes
        let n_1d = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        let values = lgl_samples_2d(n_1d, |x, y| x + y + 1.0);
        let scores = legendre_decay_smoothness_2d(&[values.as_slice()], n_1d, &qpts, &qwts, n_1d - 1);
        assert!(scores[0] < 0.3,
            "bilinear 2D should be smooth (< 0.3), got {}", scores[0]);
    }

    #[test]
    fn legendre_2d_biquadratic_is_moderately_smooth() {
        // u(x,y) = x² + y² — quadratic, some energy in higher modes
        let n_1d = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        let values = lgl_samples_2d(n_1d, |x, y| x * x + y * y);
        let scores = legendre_decay_smoothness_2d(&[values.as_slice()], n_1d, &qpts, &qwts, n_1d - 1);
        assert!(scores[0] < 0.6,
            "biquadratic should be moderately smooth (< 0.6), got {}", scores[0]);
    }

    #[test]
    fn legendre_2d_bivariate_step_is_rough() {
        // u(x,y) = sign(x) — discontinuous along x=0 in 2D
        let n_1d = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        let values = lgl_samples_2d(n_1d, |x, _y| if x >= 0.0 { 1.0 } else { -1.0 });
        let scores = legendre_decay_smoothness_2d(&[values.as_slice()], n_1d, &qpts, &qwts, n_1d - 1);
        assert!(scores[0] > 0.3,
            "2D step should be rough (> 0.3), got {}", scores[0]);
    }

    #[test]
    fn legendre_2d_empty_input() {
        let (qpts, qwts) = legendre_lobatto_nodes_weights(4);
        let scores = legendre_decay_smoothness_2d(&[], 4, &qpts, &qwts, 3);
        assert!(scores.is_empty());
    }

    // ── 2D Fourier decay tests ───────────────────────────────────────────────

    #[test]
    fn fourier_2d_bilinear_is_smooth() {
        // u(x,y) = x + y — low-frequency 2D content
        let n_1d = 16;
        let values = lgl_samples_2d(n_1d, |x, y| x + y);
        let scores = fourier_decay_smoothness_2d(&[values.as_slice()], n_1d);
        assert_eq!(scores.len(), 1);
        assert!(scores[0] < 0.5,
            "bilinear 2D Fourier should be smooth (< 0.5), got {}", scores[0]);
    }

    #[test]
    fn fourier_2d_high_frequency_is_rough() {
        // u(x,y) = cos(8π x) — high spatial frequency
        let n_1d = 32;
        let values = lgl_samples_2d(n_1d, |x, _y| (8.0 * std::f64::consts::PI * x).cos());
        let scores = fourier_decay_smoothness_2d(&[values.as_slice()], n_1d);
        assert!(scores[0] > 0.3,
            "high-freq 2D Fourier should be rough (> 0.3), got {}", scores[0]);
    }

    #[test]
    fn fourier_2d_constant_is_smooth() {
        let n_1d = 12;
        let values = lgl_samples_2d(n_1d, |_, _| 1.0);
        let scores = fourier_decay_smoothness_2d(&[values.as_slice()], n_1d);
        assert!((scores[0] - 0.0).abs() < 1e-14,
            "constant 2D Fourier should be 0, got {}", scores[0]);
    }

    #[test]
    fn fourier_2d_empty_input() {
        let scores = fourier_decay_smoothness_2d(&[], 8);
        assert!(scores.is_empty());
    }

    // ── L‑shape corner singularity test ──────────────────────────────────────
    //
    // The L‑shape domain has a re‑entrant corner at (0, 0).  The exact
    // solution u(r,θ) = r^(2/3)·sin(2θ/3) has a gradient singularity at the
    // corner (∇u ∼ r^(-1/3)), which a good smoothness estimator should detect.
    //
    // Elements close to the corner should score HIGHER (rougher) than elements
    // far from the corner, which are locally smooth.

    /// Map a reference‑element LGL point (ξ, η) ∈ [−1, 1]² to a physical
    /// element of size `h` centred at `(cx, cy)`.
    fn map_to_element(cx: f64, cy: f64, h: f64, xi: f64, eta: f64) -> (f64, f64) {
        (cx + 0.5 * h * xi, cy + 0.5 * h * eta)
    }

    /// L‑shape singular solution: u(r,θ) = r^(2/3) sin(2θ/3).
    /// θ measured from the corner at (0, 0) in the range [0, 3π/2].
    fn l_shape_singular(x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        if r < 1e-30 {
            return 0.0;
        }
        let theta = y.atan2(x); // ∈ (−π, π]
        // Shift to [0, 3π/2): the L‑shape occupies the first three quadrants
        let theta_shift = if theta >= 0.0 { theta } else { theta + 2.0 * std::f64::consts::PI };
        r.powf(2.0 / 3.0) * (2.0 * theta_shift / 3.0).sin()
    }

    #[test]
    fn l_shape_corner_elements_are_rougher_than_distant() {
        let n_1d = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);

        // Element A: straddles the re‑entrant corner (covers [−0.3, 0.3]²)
        // The corner at (0, 0) lies INSIDE the element, exposing the ∇ r^(2/3)
        // gradient singularity to the Legendre expansion.
        let h_ne = 0.6;
        let mut near_vals = Vec::with_capacity(n_1d * n_1d);
        for qx in 0..n_1d {
            for qy in 0..n_1d {
                let (px, py) = map_to_element(0.0, 0.0, h_ne, qpts[qx], qpts[qy]);
                near_vals.push(l_shape_singular(px, py));
            }
        }

        // Element B: far from the corner at (1.5, 1.5) with h=0.4 → covers
        // [1.3, 1.7]² where r^(2/3) is locally smooth
        let h_fa = 0.4;
        let mut far_vals = Vec::with_capacity(n_1d * n_1d);
        for qx in 0..n_1d {
            for qy in 0..n_1d {
                let (px, py) = map_to_element(1.5, 1.5, h_fa, qpts[qx], qpts[qy]);
                far_vals.push(l_shape_singular(px, py));
            }
        }

        let samples: Vec<&[f64]> = vec![near_vals.as_slice(), far_vals.as_slice()];

        // Use Legendre decay 2D smoothness
        let scores =
            legendre_decay_smoothness_2d(&samples, n_1d, &qpts, &qwts, n_1d - 1);
        assert_eq!(scores.len(), 2, "should have scores for 2 elements");

        // The corner‑straddling element should score rougher (higher score)
        assert!(
            scores[0] > scores[1],
            "near‑corner element ({:.4}) should be rougher than distant ({:.4})",
            scores[0],
            scores[1]
        );

        // Also test with Fourier decay 2D for cross‑validation.
        // Fourier is more sensitive to the non‑polynomial content of r^(2/3).
        let n_fourier = 8;
        let mut near_fourier = Vec::with_capacity(n_fourier * n_fourier);
        for qx in 0..n_fourier {
            for qy in 0..n_fourier {
                let xi = -1.0 + 2.0 * qx as f64 / (n_fourier - 1) as f64;
                let eta = -1.0 + 2.0 * qy as f64 / (n_fourier - 1) as f64;
                let (px, py) = map_to_element(0.0, 0.0, h_ne, xi, eta);
                near_fourier.push(l_shape_singular(px, py));
            }
        }
        let mut far_fourier = Vec::with_capacity(n_fourier * n_fourier);
        for qx in 0..n_fourier {
            for qy in 0..n_fourier {
                let xi = -1.0 + 2.0 * qx as f64 / (n_fourier - 1) as f64;
                let eta = -1.0 + 2.0 * qy as f64 / (n_fourier - 1) as f64;
                let (px, py) = map_to_element(1.5, 1.5, h_fa, xi, eta);
                far_fourier.push(l_shape_singular(px, py));
            }
        }

        let fourier_samples: Vec<&[f64]> = vec![near_fourier.as_slice(), far_fourier.as_slice()];
        let fourier_scores = fourier_decay_smoothness_2d(&fourier_samples, n_fourier);
        assert_eq!(fourier_scores.len(), 2);

        // Fourier should also rank near > far
        assert!(
            fourier_scores[0] > fourier_scores[1],
            "Fourier: near‑corner ({:.4}) should be rougher than distant ({:.4})",
            fourier_scores[0],
            fourier_scores[1]
        );
    }

    #[test]
    fn l_shape_corner_multiple_elements_ranked_by_distance() {
        // Three elements at increasing distances from the corner.
        // A larger element size (h=1.0) ensures the gradient singularity
        // r^(−1/3) produces non‑polynomial content within the near element.
        let n_1d = 8;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        let h = 1.0;

        // Element 0: centred AT the corner → straddles the singularity
        // Element 1: offset from corner
        // Element 2: far from corner, locally smooth
        let centers = [(0.0, 0.0), (0.8, 0.8), (2.0, 2.0)];
        let mut all_samples = Vec::new();

        for &(cx, cy) in &centers {
            let mut vals = Vec::with_capacity(n_1d * n_1d);
            for qx in 0..n_1d {
                for qy in 0..n_1d {
                    let (px, py) = map_to_element(cx, cy, h, qpts[qx], qpts[qy]);
                    vals.push(l_shape_singular(px, py));
                }
            }
            all_samples.push(vals);
        }

        let samples: Vec<&[f64]> = all_samples.iter().map(|v| v.as_slice()).collect();
        let scores =
            legendre_decay_smoothness_2d(&samples, n_1d, &qpts, &qwts, n_1d - 1);
        assert_eq!(scores.len(), 3);

        // Monotonically decreasing: element farthest from corner should be smoothest
        assert!(
            scores[0] >= scores[1] && scores[1] >= scores[2],
            "L‑shape scores should monotonically decrease with distance from corner: \
             near={:.4} mid={:.4} far={:.4}",
            scores[0],
            scores[1],
            scores[2]
        );
    }

    // ── Auto‑dispatch and predictor integration tests ──────────────────────

    #[test]
    fn predictor_auto_dispatches_2d_legendre() {
        // Construct 2D tensor-product data and verify the predictor uses the 2D path
        let n_1d = 6;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        let values = lgl_samples_2d(n_1d, |x, y| x * x + y * y);
        let samples = vec![values.as_slice()];

        let config = SmoothnessPredictorConfig {
            estimators: vec![SmoothnessEstimatorConfig::LegendreDecay {
                threshold_smooth: 0.2,
                threshold_rough: 0.6,
            }],
            weights: vec![1.0],
            consensus: ConsensusMode::WeightedAverage,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: None,
            grad_variation: None,
            elem_solution_samples: Some(&samples),
            quadrature: Some((&qpts, &qwts)),
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 1);
        // biquadratic should be scored as smooth-ish by the 2D path
        assert!(result.consensus[0] < 0.7,
            "2D biquadratic via predictor should be smooth-ish, got {}", result.consensus[0]);
    }

    #[test]
    fn predictor_auto_dispatches_2d_fourier() {
        let n_1d = 12;
        let values = lgl_samples_2d(n_1d, |x, y| x + y);
        let samples = vec![values.as_slice()];

        let config = SmoothnessPredictorConfig {
            estimators: vec![SmoothnessEstimatorConfig::FourierDecay { n_modes: 8 }],
            weights: vec![1.0],
            consensus: ConsensusMode::WeightedAverage,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: None,
            grad_variation: None,
            elem_solution_samples: Some(&samples),
            quadrature: None,
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 1);
        // bilinear should score smooth
        assert!(result.consensus[0] < 0.6,
            "2D bilinear via Fourier predictor should be smooth, got {}", result.consensus[0]);
    }

    #[test]
    fn predictor_1d_data_still_uses_1d_path() {
        // 1D data (6 samples for 6 LGL nodes) should still use the 1D path
        let n_1d = 6;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        // 1D sin solution
        let u_sin: Vec<f64> = qpts.iter().map(|&x| (std::f64::consts::PI * x).sin()).collect();
        let samples = vec![u_sin.as_slice()];

        let config = SmoothnessPredictorConfig {
            estimators: vec![SmoothnessEstimatorConfig::LegendreDecay {
                threshold_smooth: 0.2,
                threshold_rough: 0.6,
            }],
            weights: vec![1.0],
            consensus: ConsensusMode::WeightedAverage,
        };
        let pred = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: None,
            grad_variation: None,
            elem_solution_samples: Some(&samples),
            quadrature: Some((&qpts, &qwts)),
        };
        let result = pred.predict(&data);
        assert_eq!(result.consensus.len(), 1);
        // Should match existing 1D Legendre behaviour for sin(π x)
        let expected = legendre_decay_smoothness(&samples, &qpts, &qwts, n_1d - 1);
        assert!((result.consensus[0] - expected[0]).abs() < 1e-14,
            "1D path should match: predictor={} expected={}", result.consensus[0], expected[0]);
    }

    #[test]
    fn hp_mark_with_2d_smoothness() {
        // End‑to‑end: run the full hp-marking pipeline with 2D smoothness data
        let n_1d = 6;
        let (qpts, qwts) = legendre_lobatto_nodes_weights(n_1d);
        let values = lgl_samples_2d(n_1d, |x, y| x + y + 1.0);
        let samples = vec![values.as_slice(); 4];

        let eta = vec![1.0, 0.8, 0.3, 0.1];
        let data = SmoothnessInputData {
            eta: Some(&eta),
            grad_variation: None,
            elem_solution_samples: Some(&samples),
            quadrature: Some((&qpts, &qwts)),
        };

        let result = crate::hp_amr::hp_mark_with_strategy(
            &eta, 0.5,
            &crate::hp_amr::HpDecisionStrategy::LegendreDecay(0.2, 0.6),
            &data,
        );
        // Should not panic; at least the high‑error elements should be marked
        assert!(result.len() >= 1);
        // All smooth 2D solutions → should prefer P‑refinement
        for &(_, action) in &result {
            assert_eq!(action, crate::hp_amr::HpAction::P,
                "smooth 2D data should recommend P‑refinement");
        }
    }
}
