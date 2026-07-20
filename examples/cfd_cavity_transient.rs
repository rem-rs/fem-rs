//! CFD Example 2: Transient lid-driven cavity (BDF-2).
//!
//! ```text
//! cargo run --example cfd_cavity_transient --release
//! ```

use fem_assembly::physics::fluid_ns_transient::TransientNsDriver;
use fem_assembly::physics::fluid_cfd::NsConfig;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::constraints::boundary_dofs;

fn main() {
    println!("=== CFD: Transient Lid-Driven Cavity ===");
    let mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0); // coarse mesh for testing
    let re = 0.0_f64; // Stokes (no convection, just diffusion)

    let config = NsConfig { nu: if re > 0.0 { 1.0 / re } else { 1.0 }, rho: 1.0, quad_order: 3, ..NsConfig::default() };
    let mut driver = TransientNsDriver::new(mesh.clone(), 2, 1, config);
    driver.set_dt(0.05);

    // Pre-compute BC DOF lists (before entering the closure)
    let dm = driver.vel_space().scalar_dof_manager();
    let n_scalar = driver.vel_space().n_scalar_dofs();

    // No-slip wall DOFs (bottom + sides)
    // We need the actual tags. For make_cartesian_2d:
    // 1=bottom(y=0), 2=right(x=1), 3=top(y=1), 4=left(x=0), 5=??, 6=??
    // Let's use bottom + sides = no-slip, top = lid
    let no_slip_tags = [1i32, 2, 4];
    let lid_tags = [3i32];

    let no_slip_dofs: Vec<u32> = no_slip_tags.iter()
        .flat_map(|&t| boundary_dofs(&mesh, dm, &[t]))
        .collect();
    let lid_dofs: Vec<u32> = lid_tags.iter()
        .flat_map(|&t| boundary_dofs(&mesh, dm, &[t]))
        .collect();

    let nv = driver.n_vel();
    let np = driver.n_pres();
    let mut u = vec![0.0_f64; nv];
    let mut p = vec![0.0_f64; np];

    println!("  Re={:.0}, DOFs: vel={}, pres={}", re, nv, np);

    for step in 0..10 {
        let apply_bcs = |mat: &mut CsrMatrix<f64>, rhs: &mut [f64],
                         _b: &CsrMatrix<f64>, _extra: &[f64]| {
            // No-slip walls
            for &dof in &no_slip_dofs {
                let d = dof as usize;
                mat.apply_dirichlet_symmetric(d, 0.0, rhs);
                if d + n_scalar < rhs.len() {
                    mat.apply_dirichlet_symmetric(d + n_scalar, 0.0, rhs);
                }
            }
            // Top lid: ux = 1.0, uy = 0.0
            for &dof in &lid_dofs {
                let d = dof as usize;
                mat.apply_dirichlet_symmetric(d, 1.0, rhs);
                if d + n_scalar < rhs.len() {
                    mat.apply_dirichlet_symmetric(d + n_scalar, 0.0, rhs);
                }
            }
        };

        match driver.step(&mut u, &mut p, &apply_bcs) {
            Ok(()) => {
                let ke: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
                println!("  Step {:2}: ||u||={:.6e}", step, ke);
            }
            Err(e) => {
                eprintln!("  Step {} failed: {}", step, e);
                break;
            }
        }
    }
    println!("Done.");
}
