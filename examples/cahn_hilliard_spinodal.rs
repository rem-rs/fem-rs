//! Cahn-Hilliard spinodal decomposition on a 2D mesh.
//!
//! Solves the Cahn-Hilliard equation via a mixed FE formulation with
//! IMEX time stepping.  Starting from a near-homogeneous concentration
//! c ≈ 0.5, spinodal decomposition separates the domain into c-rich
//! and c-poor phases, driven by the double-well free energy.
//!
//! Usage:
//!   cargo run --example cahn_hilliard_spinodal

use std::fs;
use std::time::Instant;

use fem_assembly::cahn_allen::{solve_cahn_hilliard, CahnHilliardConfig};
use fem_io::vtk::{DataArray, VtkWriter};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

fn main() {
    println!("=== Cahn-Hilliard spinodal decomposition ===");
    let t0 = Instant::now();

    // Coarse mesh (4×4) — the IMEX time-stepper uses a consistent mass
    // matrix (fixed in this session) which improves stability but the
    // semi-implicit scheme still requires modest dt on fine meshes.
    let mesh = Mesh::<2>::unit_square_tri(4);

    let order: u8 = 1;
    let quad_order: u8 = 2;
    let n = fem_space::H1Space::new(mesh.clone(), order).n_dofs();
    let mut c0 = vec![0.5_f64; n];
    for i in 0..n {
        let x = mesh.node_coords(i as u32)[0];
        c0[i] += 0.1 * (std::f64::consts::PI * 2.0 * x).cos();
    }

    let cfg = CahnHilliardConfig {
        mobility: 1.0,
        epsilon: 0.5,
        dt: 1e-7,
        t_max: 1e-6,
        output_interval: 0,
        solver_cfg: fem_solver::SolverConfig {
            rtol: 1e-6,
            max_iter: 5000,
            ..fem_solver::SolverConfig::default()
        },
    };
    println!("  DOFs: {n}, ε = {:.4}, dt = {:.1e}, steps = {}",
             cfg.epsilon, cfg.dt, (cfg.t_max / cfg.dt).ceil());

    let result = solve_cahn_hilliard(&mesh, order, c0, quad_order, &cfg);
    println!("  Solved: {} time steps in {:.3}s",
             result.times.len(), t0.elapsed().as_secs_f64());
    println!("  c range: [{:.4}, {:.4}]",
             result.c.iter().fold(f64::INFINITY, |a, &v| a.min(v)),
             result.c.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)));

    fs::create_dir_all("output").ok();
    let mut writer = VtkWriter::new(&mesh);
    writer.add_point_data(DataArray::scalars("c", result.c.clone()));
    writer.write_file("output/cahn_hilliard_final.vtu").ok();
    println!("  VTK: output/cahn_hilliard_final.vtu");
    println!("  Done.");
}

#[cfg(test)]
mod tests {
    use fem_assembly::cahn_allen::{solve_cahn_hilliard, CahnHilliardConfig};
    use fem_mesh::Mesh;
    use fem_space::fe_space::FESpace;
    #[test] fn smoke() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = fem_space::H1Space::new(mesh.clone(), 1).n_dofs();
        let mut c0 = vec![0.5; n];
        for i in 0..n {
            c0[i] += 0.1 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
        }
        let result = solve_cahn_hilliard(&mesh, 1, c0, 2,
            &CahnHilliardConfig {
                epsilon: 0.5, dt: 1e-7, t_max: 3e-7,
                solver_cfg: fem_solver::SolverConfig { rtol: 1e-6, max_iter: 2000, ..Default::default() },
                ..CahnHilliardConfig::default()
            });
        assert!(result.c.iter().all(|v| v.is_finite()));
    }
}
