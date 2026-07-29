pub mod amr_refiner;
pub mod coefficient;
pub mod error_estimate;
pub mod flux_recovery;
pub mod grid_function;
pub mod postprocess;

// Re-export commonly used utility functions
pub use grid_function::{
    compute_l2_error_hcurl, compute_l2_error_hdiv, compute_l2_error_l2,
    vector_l2_norm,
};
