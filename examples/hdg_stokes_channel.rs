//! HDG Stokes 2D lid-driven cavity.
//!
//! Usage:
//!   cargo run --example hdg_stokes_channel

use std::time::Instant;
use fem_assembly::hdg_stokes::solve_hdg_stokes;
use fem_mesh::SimplexMesh;

fn main() {
    println!("=== HDG Stokes: lid-driven cavity ===");
    let t0 = Instant::now();
    let mesh = SimplexMesh::<2>::unit_square_tri(6);
    let source = |_: &[f64]| vec![0.0, 0.0];
    let result = solve_hdg_stokes(mesh, source, 1.0);
    println!("  ‖u‖ = {:.6e}, ‖p‖ = {:.6e}",
             result.u.iter().map(|v| v*v).sum::<f64>().sqrt(),
             result.p.iter().map(|v| v*v).sum::<f64>().sqrt());
    let n_u = result.u.len();
    let n_p = result.p.len();
    println!("  DOFs: u={n_u}, p={n_p}, λ={}", result.lambda.len());
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::hdg_stokes::solve_hdg_stokes;
    use fem_mesh::SimplexMesh;
    #[test] fn smoke() {
        let r = solve_hdg_stokes(SimplexMesh::<2>::unit_square_tri(4), |_: &[f64]| vec![0.0, 0.0], 1.0);
        assert!(r.u.iter().all(|v| v.is_finite()));
    }
}
