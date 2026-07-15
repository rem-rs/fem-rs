//! Compare surface assembly against analytical formula for a single triangle on the sphere.
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_space::H1Space;
use fem_assembly::Assembler;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator};

/// Create a single-triangle surface mesh: 3 vertices on the unit sphere.
/// Triangle vertices at (1,0,0), (0,1,0), (0,0,1).
fn single_tri_sphere() -> Mesh<3> {
    let coords = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let conn = vec![0u32, 1, 2];
    let elem_tags = vec![1i32];
    // Boundary: 3 edges
    let face_conn = vec![0, 1, 1, 2, 2, 0];
    let face_tags = vec![1i32, 2, 3];
    Mesh::uniform(coords, conn, elem_tags, ElementType::Tri3, face_conn, face_tags, ElementType::Line2)
}

#[test]
fn single_tri_surface_assembly() {
    let mesh = single_tri_sphere();
    // Check surface area = π/4 ≈ 0.7854 (1/8 of unit sphere)
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();
    assert_eq!(ndofs, 3, "3 vertices = 3 DOFs for P1");
    
    // Mass matrix: M_ij = ∫ φ_i·φ_j dS
    // For a flat triangle with vertices on the sphere:
    // Area = 0.5 * |(v1-v0) × (v2-v0)|
    let v0 = [1.0, 0.0, 0.0]; let v1 = [0.0, 1.0, 0.0]; let v2 = [0.0, 0.0, 1.0];
    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
    let cx = e1[1]*e2[2] - e1[2]*e2[1];
    let cy = e1[2]*e2[0] - e1[0]*e2[2];
    let cz = e1[0]*e2[1] - e1[1]*e2[0];
    let area = 0.5 * (cx*cx + cy*cy + cz*cz).sqrt();
    eprintln!("  Triangle area: {:.6e} (expected π/8 ≈ 0.3927 for spherical triangle)", area);
    eprintln!("  Expected flat area = 0.5*|e1×e2| = {:.6e}", area);
    
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator{rho:1.0}], 2);
    let diff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator{kappa:1.0}], 2);
    
    eprintln!("  Mass matrix 3×3:");
    for i in 0..3 {
        eprintln!("    [{:.6e}, {:.6e}, {:.6e}]", mass.data[i*3], mass.data[i*3+1], mass.data[i*3+2]);
    }
    
    // The sum of all mass entries should give ∫ 1 dS = area
    let mass_sum: f64 = mass.data.iter().sum();
    eprintln!("  Sum(mass) = {:.6e} (should = area = {:.6e})", mass_sum, area);
    
    // For P1 on a flat triangle, mass matrix entries should be area/12 * [2,1,1; 1,2,1; 1,1,2]
    let expected = area / 12.0;
    eprintln!("  Expected mass[0][0] = area*2/12 = {:.6e}", expected*2.0);
    eprintln!("  Actual   mass[0][0] = {:.6e}", mass.data[0]);
    assert!((mass.data[0] - 2.0*expected).abs() / (2.0*expected) < 1e-10,
        "mass[0][0] mismatch: {:.10e} vs {:.10e}", mass.data[0], 2.0*expected);
    assert!((mass.data[1] - expected).abs() / expected < 1e-10,
        "mass[0][1] mismatch: {:.10e} vs {:.10e}", mass.data[1], expected);
    
    eprintln!("  ✅ Surface mass matrix matches analytical P1 flat triangle formula");
}
