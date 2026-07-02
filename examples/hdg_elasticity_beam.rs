//! HDG elasticity 2D cantilever beam.
//!
//! Usage:
//!   cargo run --example hdg_elasticity_beam

use std::time::Instant;
use fem_assembly::hdg_elasticity::solve_hdg_elasticity;
use fem_mesh::SimplexMesh;

fn main() {
    println!("=== HDG elasticity: cantilever beam ===");
    let t0 = Instant::now();
    let mesh = SimplexMesh::<2>::unit_square_tri(6);
    let source = |_: &[f64]| vec![0.0, 0.0];
    let result = solve_hdg_elasticity(mesh, source, 1.0, 0.3);
    let u_norm: f64 = result.u.iter().map(|v| v*v).sum::<f64>().sqrt();
    let n = result.u.len();
    println!("  DOFs: u={n}, ‖u‖ = {u_norm:.6e}");
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::hdg_elasticity::solve_hdg_elasticity;
    use fem_mesh::SimplexMesh;
    #[test] fn smoke() {
        let r = solve_hdg_elasticity(SimplexMesh::<2>::unit_square_tri(4), |_: &[f64]| vec![0.0, 0.0], 1.0, 0.3);
        assert!(r.u.iter().all(|v| v.is_finite()));
    }
}
