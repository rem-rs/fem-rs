//! DPG Stokes 2D on a unit-square mesh.
//!
//! Usage:
//!   cargo run --example dpg_stokes_2d

use std::time::Instant;
use fem_assembly::dpg_stokes::solve_dpg_stokes_2d;
use fem_mesh::Mesh;

fn main() {
    println!("=== DPG Stokes 2D ===");
    let t0 = Instant::now();
    let mesh = Mesh::<2>::unit_square_tri(6);
    let f = |_x: f64, _y: f64| (0.0_f64, 0.0_f64);
    let (u, p, _lam) = solve_dpg_stokes_2d(&mesh, 1.0, &f);
    let u_norm: f64 = u.iter().map(|v| v*v).sum::<f64>().sqrt();
    let p_norm: f64 = p.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ‖u‖ = {u_norm:.6e}, ‖p‖ = {p_norm:.6e}");
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::dpg_stokes::solve_dpg_stokes_2d;
    use fem_mesh::Mesh;
    #[test] fn smoke() {
        let f = |_: f64, _: f64| (0.0_f64, 0.0_f64);
        let (u, p, _lam) = solve_dpg_stokes_2d(&Mesh::<2>::unit_square_tri(4), 1.0, &f);
        assert!(u.iter().chain(p.iter()).all(|v| v.is_finite()));
    }
}
