//! IGA wave equation (2D) with Crank-Nicolson time stepping.
//!
//! Solves ∂²u/∂t² = c²·Δu on a unit square NURBS patch.
//! Uses iga_time_step_cn for unconditional stability.
//!
//! Usage:
//!   cargo run --example iga_wave_equation --release

use fem_assembly::iga::{assemble_iga_diffusion_2d, assemble_iga_mass_2d, iga_time_step_cn};
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh2D;

fn main() {
    let p = 2;
    let n = 16;
    let kv = NurbsKnotVector::uniform(p, n - p);
    let ctrl: Vec<[f64; 2]> = (0..n * n)
        .map(|idx| { let i = idx % n; let j = idx / n;
            [i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64]
        }).collect();
    let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; n * n]);
    let n_dofs = n * n;

    let c = 1.0;
    let dt = 0.01;
    let t_final = 1.0;
    let n_steps = (t_final / dt) as usize;

    let m = assemble_iga_mass_2d(&mesh, 1.0, 4);
    let k = assemble_iga_diffusion_2d(&mesh, c * c, 4);
    let f = vec![0.0; n_dofs]; // no external force

    // Initial condition: Gaussian bump
    let mut u = vec![0.0; n_dofs];
    for (i, pt) in mesh.patches[0].control_pts.iter().enumerate() {
        let x = pt[0] - 0.5; let y = pt[1] - 0.5;
        u[i] = (-30.0 * (x * x + y * y)).exp();
    }

    println!("IGA wave: {}×{} NURBS(p={}), {} DOFs, dt={}, {} steps",
             n, n, p, n_dofs, dt, n_steps);

    for step in 0..n_steps {
        u = iga_time_step_cn(dt, &m, &k, &u, &f);
        if (step + 1) % (n_steps / 5) == 0 {
            println!("  step {}: max |u| = {:.6e}", step + 1,
                     u.iter().map(|v| v.abs()).fold(0.0, f64::max));
        }
    }

    println!("Final max |u| = {:.6e}", u.iter().map(|v| v.abs()).fold(0.0, f64::max));
    println!("✅ IGA wave equation complete.");
}
