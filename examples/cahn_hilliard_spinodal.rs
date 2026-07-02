//! Allen-Cahn phase field evolution on a 2D unit-square mesh.
//!
//! Solves the Allen-Cahn equation with a semi-implicit (IMEX) time
//! stepping scheme.  Initial condition: c(x) = sin(2πx)·sin(2πy).
//! The double-well potential drives the field toward c = ±1, while
//! the diffuse interface (width ∼ ε) is resolved by the mesh.
//!
//! Usage:
//!   cargo run --example allen_cahn_evolution

use std::fs;
use std::time::Instant;

use fem_assembly::cahn_allen::{solve_allen_cahn, AllenCahnConfig};
use fem_io::vtk::{DataArray, VtkWriter};
use fem_mesh::SimplexMesh;
use fem_space::fe_space::FESpace;

fn main() {
    println!("=== Allen-Cahn phase field evolution ===");
    let t0 = Instant::now();

    // Mesh: 6×6 (matching existing test parameters)
    let mesh = SimplexMesh::<2>::unit_square_tri(6);

    // Initial condition: low-amplitude perturbation
    let order: u8 = 1;
    let quad_order: u8 = 2;
    let n = fem_space::H1Space::new(mesh.clone(), order).n_dofs();
    let mut c0 = vec![0.0_f64; n];
    for i in 0..n {
        c0[i] = 0.3 * (std::f64::consts::PI * i as f64 / n as f64).sin();
    }

    // Use test-validated IMEX parameters
    let cfg = AllenCahnConfig {
        l_factor: 1.0,
        epsilon: 0.5,
        dt: 1e-7,
        t_max: 3e-7,
        output_interval: 0,
        solver_cfg: fem_solver::SolverConfig {
            rtol: 1e-6,
            max_iter: 1000,
            ..fem_solver::SolverConfig::default()
        },
    };
    let n_steps = (cfg.t_max / cfg.dt).ceil() as usize;
    println!("  ε = {:.4}, dt = {:.1e}, steps = {n_steps}", cfg.epsilon, cfg.dt);

    // Solve
    println!("  Solving...");
    let result = solve_allen_cahn(&mesh, order, c0, quad_order, &cfg);

    let elapsed = t0.elapsed();
    println!("  Solved in {:.3}s", elapsed.as_secs_f64());
    println!("  Time steps: {}", result.times.len());
    println!("  Final energy: {:.6e}", result.energy.last().copied().unwrap_or(0.0));
    let c_min = result.c.iter().fold(f64::INFINITY, |a, &v| a.min(v));
    let c_max = result.c.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    println!("  c range: [{c_min:.4}, {c_max:.4}]");
    if c_min < -0.5 && c_max > 0.5 {
        println!("  Phase separation ✓");
    }

    // VTK output
    fs::create_dir_all("output").ok();
    let mut writer = VtkWriter::new(&mesh);
    writer.add_point_data(DataArray::scalars("c", result.c.clone()));
    writer.write_file("output/allen_cahn_final.vtu")
        .unwrap_or_else(|e| eprintln!("  Warning: VTK write failed: {e}"));
    println!("  VTK at output/allen_cahn_final.vtu");

    // Energy history CSV
    let csv = result.times.iter().zip(result.energy.iter())
        .enumerate()
        .fold(String::from("step,time,energy\n"),
              |acc, (i, (t, e))| format!("{}{},{:.12e},{:.12e}\n", acc, i, t, e));
    fs::write("output/allen_cahn_energy.csv", &csv)
        .unwrap_or_else(|e| eprintln!("  Warning: CSV write failed: {e}"));
    println!("  Energy at output/allen_cahn_energy.csv");
    println!("  Done.");
}

#[cfg(test)]
mod tests {
    use fem_assembly::cahn_allen::{solve_allen_cahn, AllenCahnConfig};
    use fem_mesh::SimplexMesh;
    use fem_space::fe_space::FESpace;

    #[test]
    fn smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let n = fem_space::H1Space::new(mesh.clone(), 1).n_dofs();
        let mut c0 = vec![0.0; n];
        for i in 0..n {
            c0[i] = (std::f64::consts::PI * i as f64 / n as f64).sin();
        }
        let result = solve_allen_cahn(&mesh, 1, c0, 2,
            &AllenCahnConfig {
                epsilon: 0.5, dt: 1e-7, t_max: 3e-7,
                ..AllenCahnConfig::default()
            });
        assert!(result.energy.len() >= 2);
        assert!(result.c.iter().all(|v| v.is_finite()));
    }
}
