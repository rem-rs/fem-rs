//! Target-Matrix Optimisation (TMOP) for 2D mesh quality.
//!
//! Optimises node positions of a perturbed unit-square mesh using the
//! mean-ratio metric (ideal for maintaining element quality).
//!
//! Usage:
//!   cargo run --example mesh_quality_tmop

use std::time::Instant;

use fem_mesh::Mesh;
use fem_mesh::tmop::{tmop_optimise_2d, TmopMetric};

fn perturb_square(n: usize, amp: f64) -> Mesh<2> {
    let mut mesh = Mesh::<2>::unit_square_tri(n);
    let nn = mesh.n_nodes();
    for i in 0..nn {
        let x = mesh.coords[i * 2];
        let y = mesh.coords[i * 2 + 1];
        if x > 0.01 && x < 0.99 && y > 0.01 && y < 0.99 {
            mesh.coords[i * 2]     += amp * (3.0 * x * (1.0 - x) * y).sin();
            mesh.coords[i * 2 + 1] += amp * (3.0 * x * y * (1.0 - y)).cos();
        }
    }
    mesh
}

fn main() {
    println!("=== TMOP: 2D mesh quality optimisation ===");
    let t0 = Instant::now();

    let mesh = perturb_square(8, 0.1);
    let initial = mesh.coords.clone();
    let metric = TmopMetric::Shape;
    let optimised = tmop_optimise_2d(&mesh, &metric, 100, 0.01);

    let disp: f64 = initial.iter().zip(optimised.iter())
        .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    println!("  ‖displacement‖ = {:.6e}", disp);
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_mesh::Mesh;
    use fem_mesh::tmop::{tmop_optimise_2d, TmopMetric};
    #[test] fn smoke() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let r = tmop_optimise_2d(&mesh, &TmopMetric::Shape, 10, 0.01);
        assert!(r.iter().all(|v| v.is_finite()));
    }
}
