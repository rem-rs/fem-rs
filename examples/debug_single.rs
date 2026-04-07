//! Minimal single-element diagnostic for Nedelec interpolation.

use std::f64::consts::PI;
use fem_element::{reference::VectorReferenceElement, nedelec::TriND1};

fn main() {
    // Single reference triangle: (0,0), (1,0), (0,1)
    // J = identity, det_j = 1, J^{-T} = identity
    // So Piola-transformed basis = reference basis itself.
    //
    // Exact solution: E = (sin(πy), sin(πx))
    //
    // Edge DOFs (tangential integrals, midpoint rule):
    //   e₀: v₀(0,0)→v₁(1,0), tangent = (1,0), mid = (0.5,0)
    //        DOF₀ = E(0.5,0)·(1,0) = (sin(0), sin(π/2))·(1,0) = 0
    //   e₁: v₁(1,0)→v₂(0,1), tangent = (-1,1), mid = (0.5,0.5)
    //        DOF₁ = E(0.5,0.5)·(-1,1) = (sin(π/2), sin(π/2))·(-1,1) = -1 + 1 = 0
    //   e₂: v₀(0,0)→v₂(0,1), tangent = (0,1), mid = (0,0.5)
    //        DOF₂ = E(0,0.5)·(0,1) = (sin(π/2), sin(0))·(0,1) = 0

    // All DOFs are zero because E·t = 0 on all edges of the reference triangle!
    // This is because E = (sin(πy), sin(πx)) happens to have zero tangential
    // component along the edges of the reference triangle [0,1]².
    //
    // Wait, the reference triangle is NOT the unit square. The edges are:
    //   e₀: y=0 line, E_x = sin(0) = 0 → DOF = 0
    //   e₁: x+y=1 line
    //   e₂: x=0 line, E_y = sin(0) = 0 → DOF = 0
    //
    // For e₁: mid = (0.5, 0.5), tangent = (-1, 1) (length √2)
    //   DOF₁ = E(0.5, 0.5)·(-1, 1) = sin(π/2)·(-1) + sin(π/2)·1 = -1 + 1 = 0
    //
    // ALL DOF values are 0! The function E = (sin(πy), sin(πx)) is
    // accidentally in the kernel of the DOF functionals on this triangle.
    //
    // This means E is "invisible" to the ND1 interpolation on the reference
    // triangle and on many elements in the unit-square mesh.

    println!("E = (sin(πy), sin(πx)) has zero tangential projection on");
    println!("all edges of the reference triangle (0,0),(1,0),(0,1).");
    println!();
    println!("This is NOT a good test function for Nedelec elements on");
    println!("structured triangular meshes of the unit square!");
    println!();

    // Better test: E = (y, -x) which is a constant-curl field.
    // e₀: mid=(0.5,0), tangent=(1,0), DOF = (0,-0.5)·(1,0) = 0
    // Hmm, still 0.
    //
    // Try E = (1, 0) — constant field.
    // e₀: DOF = (1,0)·(1,0) = 1, e₁: DOF = (1,0)·(-1,1) = -1, e₂: DOF = (1,0)·(0,1) = 0
    println!("Test: E = (1, 0) — constant vector field");
    let ref_elem = TriND1;
    let dof_coords = ref_elem.dof_coords();
    let tangents = [[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]]; // edge vectors (not unit)
    for (i, (mid, t)) in dof_coords.iter().zip(tangents.iter()).enumerate() {
        let e_at_mid = [1.0, 0.0]; // E = (1,0)
        let dof: f64 = e_at_mid[0] * t[0] + e_at_mid[1] * t[1];
        println!("  DOF[{i}] at ({:.1},{:.1}) = {dof:.4}", mid[0], mid[1]);
    }

    // Now reconstruct E_h at centroid (1/3, 1/3)
    let centroid = [1.0/3.0, 1.0/3.0];
    let mut phi = vec![0.0; 6];
    ref_elem.eval_basis_vec(&centroid, &mut phi);
    let dofs = [1.0, -1.0, 0.0]; // from above
    let mut eh = [0.0_f64; 2];
    for i in 0..3 {
        eh[0] += dofs[i] * phi[i*2];
        eh[1] += dofs[i] * phi[i*2+1];
    }
    println!("  E_h at centroid = ({:.6}, {:.6})", eh[0], eh[1]);
    println!("  E_exact         = (1.000000, 0.000000)");
}
