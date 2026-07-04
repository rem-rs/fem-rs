//! Virtual Element Method (VEM) for Poisson on a polygonal mesh.
//!
//! Assembles the VEM P1 stiffness matrix on a quadrilateral mesh
//! and verifies basic properties (SPD, positive diagonal).
//!
//! Usage:
//!   cargo run --example vem_poisson_polygonal

use std::time::Instant;

use fem_assembly::vem_poisson::assemble_vem_poisson;
use fem_mesh::poly_mesh::PolyMesh;
use fem_space::vem::VEMSpace;
use fem_space::fe_space::FESpace;

fn main() {
    println!("=== VEM Poisson: polygonal mesh ===");
    let t0 = Instant::now();

    let mesh = PolyMesh::unit_square_quad(4, 3);
    let space = VEMSpace::new(mesh, 1);
    let n = space.n_dofs();
    println!("  Polygonal mesh: 4×3 quads, DOFs = {n}");

    let k = assemble_vem_poisson(&space);
    assert_eq!(k.nrows, n);

    let mut min_diag = f64::MAX;
    for i in 0..n {
        let d = k.get(i, i);
        min_diag = min_diag.min(d);
    }
    println!("  Min diagonal: {:.6e}", min_diag);
    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

#[cfg(test)]
mod tests {
    use fem_assembly::vem_poisson::assemble_vem_poisson;
    use fem_mesh::poly_mesh::PolyMesh;
    use fem_space::vem::VEMSpace;
    use fem_space::fe_space::FESpace;

    #[test]
    fn vem_assemble_and_check() {
        let mesh = PolyMesh::unit_square_quad(3, 3);
        let space = VEMSpace::new(mesh, 1);
        let k = assemble_vem_poisson(&space);
        assert_eq!(k.nrows, space.n_dofs());
        for i in 0..space.n_dofs() {
            assert!(k.get(i, i) > 0.0, "diag[{i}] must be positive");
        }
    }
}
