//! Crystal plasticity FCC: single-element 3D test.
//!
//! Usage:
//!   cargo run --example crystal_plasticity_fcc

use std::time::Instant;
use fem_assembly::crystal_plasticity::{assemble_crystal_plasticity, CrystalConfig, CrystalState};
use fem_mesh::Mesh;
use fem_space::vector_h1::VectorH1Space;
use fem_space::fe_space::FESpace;

fn main() {
    println!("=== Crystal plasticity FCC (3D) ===");
    let t0 = Instant::now();
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let space = VectorH1Space::new(mesh, 1, 3);
    let cfg = CrystalConfig::aluminium();
    let n = space.n_dofs();
        let mut state = CrystalState::new(200);
    let u = vec![0.0; n]; // zero displacement
    let (f_int, k) = assemble_crystal_plasticity(space.mesh(), &space, &u, &cfg, &mut state, 1.0, 2);
    let f_norm: f64 = f_int.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  DOFs: {n}, ‖f_int‖ = {f_norm:.6e}");
    println!("  Matrix nnz: {}", k.values.len());
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::crystal_plasticity::{assemble_crystal_plasticity, CrystalConfig, CrystalState};
    use fem_mesh::Mesh;
    use fem_space::vector_h1::VectorH1Space;
    use fem_space::fe_space::FESpace;
    #[test] fn smoke() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let space = VectorH1Space::new(mesh, 1, 3);
        let cfg = CrystalConfig::aluminium();
    let mut state = CrystalState::new(500); // enough for the 3D mesh QPs
        let (f, _k) = assemble_crystal_plasticity(space.mesh(), &space, &vec![0.0; space.n_dofs()], &cfg, &mut state, 1.0, 2);
        assert!(f.iter().all(|v| v.is_finite()));
    }
}
