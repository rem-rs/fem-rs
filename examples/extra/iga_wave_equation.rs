//! IGA wave equation (2D) with Newmark time integration.
//!
//! Solves ∂²u/∂t² = c²·Δu on a unit square NURBS patch using
//! the Newmark-β method (average acceleration, unconditionally stable).
//!
//! Usage:
//!   cargo run --example iga_wave_equation --release

use fem_assembly::iga::{assemble_iga_diffusion_2d, assemble_iga_mass_2d};
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh2D;
use fem_solver::ode::structural::{Newmark, NewmarkState};

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

    let c = 1.0; // wave speed
    let dt = 0.01;
    let t_final = 2.0;
    let n_steps = (t_final / dt) as usize;

    // Assemble IGA mass and stiffness (diffusion) matrices
    let m = assemble_iga_mass_2d(&mesh, 1.0, 4);
    let k = assemble_iga_diffusion_2d(&mesh, c * c, 4);

    // Initial condition: Gaussian bump at center
    let mut u = vec![0.0; n_dofs];
    for (i, pt) in mesh.patches[0].control_pts.iter().enumerate() {
        let x = pt[0] - 0.5; let y = pt[1] - 0.5;
        u[i] = (-30.0 * (x * x + y * y)).exp();
    }

    let mut state = NewmarkState::new(n_dofs);
    let nm = Newmark::default();

    println!("IGA wave: {}×{} NURBS(p={}), {} DOFs, dt={}, {} steps",
             n, n, p, n_dofs, dt, n_steps);

    let init_max = u.iter().map(|v| v.abs()).fold(0.0, f64::max);
    println!("  Initial max |u| = {:.6e}", init_max);

    for step in 0..n_steps {
        let f = vec![0.0; n_dofs]; // no external forcing
        nm.step(&m, &k, &f, dt, &mut u, &mut state, &[]);
        if (step + 1) % (n_steps / 5).max(1) == 0 {
            let ke: f64 = state.vel.iter().map(|v| v * v).sum::<f64>() * 0.5;
            println!("  step {}: max |u| = {:.6e}, KE = {:.6e}", step + 1,
                     u.iter().map(|v| v.abs()).fold(0.0, f64::max), ke);
        }
    }

    println!("Final max |u| = {:.6e}", u.iter().map(|v| v.abs()).fold(0.0, f64::max));
    println!("✅ IGA wave equation (Newmark) complete.");
}
