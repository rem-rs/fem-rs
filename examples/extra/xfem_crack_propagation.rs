//! XFEM crack propagation: maximum hoop stress angle demo.
//!
//! Usage:
//!   cargo run --example xfem_crack_propagation

use fem_assembly::xfem_crack::{max_hoop_stress_angle, equivalent_k, CrackPropagationConfig, propagate_crack_front};
use fem_assembly::xfem::XfemLevelSet;

fn main() {
    println!("=== XFEM crack propagation ===");
    // Pure mode I (K_II=0): angle should be 0
    let theta_i = max_hoop_stress_angle(1.0, 0.0);
    println!("  Pure mode I: θ = {:.4}° (expected 0.0)", theta_i.to_degrees());
    // Pure mode II (K_I=0): angle ≈ -70.5°
    let theta_ii = max_hoop_stress_angle(0.0, 1.0);
    println!("  Pure mode II: θ = {:.4}° (expected ≈ -70.5°)", theta_ii.to_degrees());
    // Mixed mode
    let theta_m = max_hoop_stress_angle(1.0, 0.5);
    let k_eq = equivalent_k(1.0, 0.5, theta_m);
    println!("  Mixed mode (K_I=1, K_II=0.5): θ = {:.4}°, K_eq = {:.4}", theta_m.to_degrees(), k_eq);
    // Propagation
    let ls = XfemLevelSet::CrackLine { x1: [0.0, 0.5], x2: [0.5, 0.5] };
    let cfg = CrackPropagationConfig { delta_a: 0.1, ..Default::default() };
    let result = propagate_crack_front(&ls, &cfg, 1, 1.0, 0.2, 1.0, 0.3);
    println!("  Propagated crack front: θ_c = {:.4}°", result.theta_c.to_degrees());
    println!("  Done.");
}

#[cfg(test)]
mod tests {
    use fem_assembly::xfem_crack::max_hoop_stress_angle;
    #[test] fn pure_i() { assert!(max_hoop_stress_angle(1.0, 0.0).abs() < 1e-12); }
    #[test] fn pure_ii() { assert!(max_hoop_stress_angle(0.0, 1.0) < 0.0); }
}
