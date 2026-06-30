//! # fem-stochastic
//!
//! Random field generation and Monte Carlo simulation for stochastic finite
//! elements.
//!
//! ## Modules
//! - [`random_field`] — Karhunen-Loève (KL) expansion, exponential and
//!   squared-exponential covariance kernels.
//! - [`monte_carlo`] — Monte Carlo driver with batch reporting and
//!   online mean/variance computation.
//!
//! ## Quick start
//! ```no_run
//! use fem_stochastic::{
//!     Covariance1D, ExponentialCovariance1D,
//!     KarhunenLoeveExpansion1D, RandomField,
//!     MonteCarloConfig, run_monte_carlo,
//! };
//!
//! // Build KL expansion for a 1D random field
//! let cov = ExponentialCovariance1D { sigma2: 0.25, length: 0.5 };
//! let kl = KarhunenLoeveExpansion1D::new(32, 4, 0.0, &cov);
//!
//! // Run Monte Carlo
//! let result = run_monte_carlo(&MonteCarloConfig::default(), |_i, rng| {
//!     let pts: Vec<[f64; 1]> = (0..10).map(|i| [i as f64 / 9.0]).collect();
//!     let field = kl.realisation(&pts, rng);
//!     field.iter().sum::<f64>() // QoI
//! });
//! println!("Mean = {:.4e}, StdErr = {:.4e}", result.mean, result.std_err);
//! ```

pub mod mlmc;
pub mod random_field;
pub mod monte_carlo;
pub mod polynomial_chaos;

pub use random_field::{
    RandomField, KarhunenLoeveExpansion1D,
    ExponentialCovariance1D, SquaredExponentialCovariance1D, Covariance1D,
    RandomField2D, KarhunenLoeveExpansion2D,
    ExponentialCovariance2D, SquaredExponentialCovariance2D, Covariance2D,
};
pub use monte_carlo::{MonteCarloConfig, MonteCarloResult, run_monte_carlo};
