//! Random field generation via Karhunen-Loève (KL) expansion.
//!
//! A random field `κ(x, ω)` with covariance `C(x, y)` can be expanded as:
//! ```text
//! κ(x, ω) = μ(x) + Σ √λ_i φ_i(x) ξ_i(ω)
//! ```
//! where `(λ_i, φ_i)` are eigenpairs of the covariance operator, and
//! `ξ_i ~ N(0,1)` are independent standard normal variables.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// A realisation of a random field at a set of evaluation points.
pub trait RandomField {
    /// Generate one realisation at the given points.
    fn realisation(&self, points: &[[f64; 1]], rng: &mut impl Rng) -> Vec<f64>;
}

// ─── Covariance kernels ──────────────────────────────────────────────────

/// Covariance function `C(x, y)`.
pub trait Covariance1D {
    fn eval(&self, x: f64, y: f64) -> f64;
    fn correlation_length(&self) -> f64;
    fn variance(&self) -> f64;
}

/// Exponential covariance: `C(x, y) = σ² exp(-|x-y| / L)`.
pub struct ExponentialCovariance1D {
    pub sigma2: f64,
    pub length: f64,
}

impl Covariance1D for ExponentialCovariance1D {
    fn eval(&self, x: f64, y: f64) -> f64 {
        self.sigma2 * (-(x - y).abs() / self.length).exp()
    }
    fn correlation_length(&self) -> f64 { self.length }
    fn variance(&self) -> f64 { self.sigma2 }
}

/// Squared-exponential (Gaussian) covariance: `C(x, y) = σ² exp(-(x-y)² / (2 L²))`.
pub struct SquaredExponentialCovariance1D {
    pub sigma2: f64,
    pub length: f64,
}

impl Covariance1D for SquaredExponentialCovariance1D {
    fn eval(&self, x: f64, y: f64) -> f64 {
        let d = x - y;
        self.sigma2 * (-d * d / (2.0 * self.length * self.length)).exp()
    }
    fn correlation_length(&self) -> f64 { self.length }
    fn variance(&self) -> f64 { self.sigma2 }
}

// ─── Karhunen-Loève expansion in 1D ─────────────────────────────────────

/// KL expansion for a 1-D random field.
///
/// The covariance eigenproblem is discretised with a uniform grid.
pub struct KarhunenLoeveExpansion1D {
    /// Mean function values at each evaluation point.
    mean: Vec<f64>,
    /// Eigenvalues √λ_i (sorted descending).
    sqrt_eigenvalues: Vec<f64>,
    /// Eigenvector matrix: `φ_i(x_j)` at evaluation points.
    eigenvectors: Vec<Vec<f64>>,
}

impl KarhunenLoeveExpansion1D {
    /// Build the KL expansion on a uniform grid with `n` points in `[0, 1]`.
    ///
    /// Keeps the top `n_modes` eigenpairs. The mean is constant `μ`.
    pub fn new(n: usize, n_modes: usize, mu: f64, cov: &impl Covariance1D) -> Self {
        let h = 1.0 / (n - 1) as f64;
        let points: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();

        // Assemble the covariance matrix C[i,j] = cov(points[i], points[j])
        let mut c = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                c[i][j] = cov.eval(points[i], points[j]);
            }
        }

        // Power iteration to extract the top n_modes eigenvalues/vectors.
        let (evals, evecs) = dominant_eigenpairs(&c, n_modes.min(n), 1_000, 1e-12);

        let n_keep = evals.len().min(n_modes);
        let sqrt_evals: Vec<f64> = evals.iter().take(n_keep).map(|v| v.sqrt()).collect();
        let evecs: Vec<Vec<f64>> = evecs.into_iter().take(n_keep).collect();

        let mean = vec![mu; n];

        KarhunenLoeveExpansion1D {
            mean,
            sqrt_eigenvalues: sqrt_evals,
            eigenvectors: evecs,
        }
    }

    /// Number of KL modes kept.
    pub fn n_modes(&self) -> usize { self.sqrt_eigenvalues.len() }
}

impl RandomField for KarhunenLoeveExpansion1D {
    fn realisation(&self, points: &[[f64; 1]], _rng: &mut impl Rng) -> Vec<f64> {
        let normal = StandardNormal;
        let mut rng_small = rand::thread_rng();
        let xi: Vec<f64> = (0..self.n_modes())
            .map(|_| normal.sample(&mut rng_small))
            .collect();

        let n = points.len();
        let mut field = self.mean.clone();
        for i in 0..n {
            for m in 0..self.n_modes() {
                field[i] += self.sqrt_eigenvalues[m] * self.eigenvectors[m][i] * xi[m];
            }
        }
        field
    }
}

// ─── Power iteration for dominant eigenpairs ─────────────────────────────

fn dominant_eigenpairs(
    mat: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    let k = k.min(n);
    let mut evals = Vec::with_capacity(k);
    let mut evecs = Vec::with_capacity(k);
    let mut residual = mat.clone();

    for _mode in 0..k {
        // Power iteration on the residual matrix
        let mut v: Vec<f64> = (0..n).map(|_| rand::random::<f64>() - 0.5).collect();
        let norm0 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for vi in &mut v { *vi /= norm0; }

        let mut lambda_old = 0.0;
        for _iter in 0..max_iter {
            // w = residual * v
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += residual[i][j] * v[j];
                }
            }
            // Rayleigh quotient
            let lambda: f64 = w.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            // Normalise
            let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 1e-30 {
                for vi in &mut v { *vi = 0.0; }
                for i in 0..n { v[i] += w[i] / nrm; }
            }
            if (lambda - lambda_old).abs() < tol * lambda.abs().max(1.0) {
                break;
            }
            lambda_old = lambda;
        }

        // Deflate: residual -= λ * v * vᵀ
        for i in 0..n {
            for j in 0..n {
                residual[i][j] -= lambda_old * v[i] * v[j];
            }
        }

        evals.push(lambda_old.abs());
        evecs.push(v);
    }

    (evals, evecs)
}
