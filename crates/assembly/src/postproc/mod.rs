pub mod amr_refiner;
pub mod coefficient;
pub mod error_estimate;
pub mod flux_recovery;
pub mod grid_function;
pub mod postprocess;

// Re-export commonly used utility functions
pub use grid_function::vector_l2_norm;
