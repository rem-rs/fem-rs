//! DPG Poisson 2D on a unit-square mesh.
//!
//! Solves -Δu = f with DPG optimal test functions.
//! Uses the ultraweak formulation with optimal test space.
//!
//! Usage:
//!   cargo run --example dpg_poisson_2d

use std::time::Instant;

use fem_assembly::dpg_2d::solve_dpg_poisson_2d;
use fem_mesh::Mesh;

fn main() {
    println!("=== DPG Poisson 2D ===");
    let t0 = Instant::now();

    // RHS: f = 2π² sin(πx) sin(πy)
    let f = |x: f64, y: f64| {
        let p = std::f64::consts::PI;
        2.0 * p * p * (p * x).sin() * (p * y).sin()
    };

    let mesh = Mesh::<2>::unit_square_tri(8);
    let u = solve_dpg_poisson_2d(&mesh, &f);
    let n = u.len();
    let u_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  DOFs: {n}, ‖u‖ = {:.6e}", u_norm);
    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

#[cfg(test)]
mod tests {
    use fem_assembly::dpg_2d::solve_dpg_poisson_2d;
    use fem_mesh::Mesh;

    #[test]
    fn smoke() {
        let f = |_: f64, _: f64| 1.0;
        let mesh = Mesh::<2>::unit_square_tri(3);
        let u = solve_dpg_poisson_2d(&mesh, &f);
        assert!(u.iter().all(|v| v.is_finite()));
    }
}
