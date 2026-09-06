pub mod amr_refiner;
pub mod coefficient;
pub mod error_estimate;
pub mod flux_recovery;
pub mod grid_function;
pub mod grid_function_probe;
pub mod l2_zz_rt1;
pub mod postprocess;

// Re-export commonly used utility functions
pub use grid_function::{
    compute_l2_error_hcurl, compute_l2_error_hdiv, compute_l2_error_l2,
    vector_l2_norm,
};
pub use error_estimate::zz_estimator_l2_hdiv;
pub use grid_function_probe::{
    evaluate_vector_at_element, evaluate_curl_at_element, evaluate_div_at_element,
    get_value, get_vector_value, get_gradient, get_curl, get_divergence,
    get_element_bounds, get_nodal_values,
};
