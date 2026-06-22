//! Monte Carlo simulation framework for stochastic FEM.
//!
//! Runs repeated solves with random field realisations and estimates
//! statistics of the QoI (Quantity of Interest).

use std::time::Instant;

/// Configuration for a Monte Carlo simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of samples.
    pub n_samples: usize,
    /// Batch size for progress reporting (0 = no reporting).
    pub report_every: usize,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        MonteCarloConfig { n_samples: 100, report_every: 0 }
    }
}

/// Statistics from a Monte Carlo simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Number of completed samples.
    pub n_samples: usize,
    /// Sample mean of the QoI.
    pub mean: f64,
    /// Sample variance of the QoI.
    pub variance: f64,
    /// Standard error (σ / √n).
    pub std_err: f64,
    /// Elapsed wall time.
    pub elapsed: std::time::Duration,
}

impl MonteCarloResult {
    /// Coefficient of variation (σ / μ).
    pub fn cv(&self) -> f64 {
        if self.mean.abs() > 1e-30 { self.variance.sqrt() / self.mean.abs() } else { 0.0 }
    }
}

/// Run a Monte Carlo simulation.
///
/// `sample_fn(i, rng)` generates the i-th QoI value.
pub fn run_monte_carlo<F>(config: &MonteCarloConfig, mut sample_fn: F) -> MonteCarloResult
where
    F: FnMut(usize, &mut rand::rngs::ThreadRng) -> f64,
{
    let t0 = Instant::now();
    let mut rng = rand::thread_rng();
    let n = config.n_samples;

    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;

    for i in 0..n {
        let qoi = sample_fn(i, &mut rng);
        sum += qoi;
        sum2 += qoi * qoi;

        if config.report_every > 0 && (i + 1) % config.report_every == 0 {
            let mean = sum / (i + 1) as f64;
            let var = if i > 0 { (sum2 - sum * sum / (i + 1) as f64) / i as f64 } else { 0.0 };
            println!("  MC sample {}/{}: QoI={:.6e}, mean={:.6e}, std={:.6e}",
                i + 1, n, qoi, mean, var.sqrt());
        }
    }

    let elapsed = t0.elapsed();
    let mean = sum / n as f64;
    let variance = if n > 1 {
        (sum2 - sum * sum / n as f64) / (n - 1) as f64
    } else {
        0.0
    };
    let std_err = variance.sqrt() / (n as f64).sqrt();

    MonteCarloResult { n_samples: n, mean, variance, std_err, elapsed }
}
