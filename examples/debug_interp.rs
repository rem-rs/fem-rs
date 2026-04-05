//! Quick diagnostic: interpolate exact solution and measure L² error.
//! This isolates the L² error computation from the assembly/solve chain.

use std::f64::consts::PI;
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_space::{HCurlSpace, fe_space::FESpace};
use fem_element::{reference::VectorReferenceElement, nedelec::TriND1};

fn main() {
    println!("=== Interpolation error diagnostic ===");
    for &n in &[4u32, 8, 16, 32] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n as usize);
        let space = HCurlSpace::new(mesh, 1);

        // Interpolate E = (sin(πy), sin(πx)) via DOF functionals
        let u = space.interpolate_vector(&|x| vec![(PI * x[1]).sin(), (PI * x[0]).sin()]);

        // Check DOF values
        let max_dof = u.as_slice().iter().cloned().fold(0.0_f64, f64::max);
        let min_dof = u.as_slice().iter().cloned().fold(0.0_f64, f64::min);
        let norm: f64 = u.as_slice().iter().map(|v| v*v).sum::<f64>().sqrt();

        // Compute L² error
        let l2 = l2_error_hcurl(&space, u.as_slice());
        let h = 1.0 / n as f64;
        println!("  h = {h:.4e},  DOF range [{min_dof:.4e}, {max_dof:.4e}], norm = {norm:.4e}");
    }
}

fn l2_error_hcurl(space: &HCurlSpace<SimplexMesh<2>>, uh: &[f64]) -> f64 {
    let mesh = space.mesh();
    let ref_elem = TriND1;
    let quad = ref_elem.quadrature(6);
    let n_ldofs = ref_elem.n_dofs();
    let mut err2 = 0.0_f64;
    let mut err2_nosign = 0.0_f64;
    let mut ref_phi = vec![0.0; n_ldofs * 2];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let j00 = x1[0] - x0[0]; let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1]; let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();
        let inv_det = 1.0 / (j00 * j11 - j01 * j10);
        let jit00 = j11 * inv_det; let jit01 = -j10 * inv_det;
        let jit10 = -j01 * inv_det; let jit11 = j00 * inv_det;

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j;
            let xp = [x0[0] + j00 * xi[0] + j01 * xi[1],
                       x0[1] + j10 * xi[0] + j11 * xi[1]];
            ref_elem.eval_basis_vec(xi, &mut ref_phi);
            let mut eh = [0.0_f64; 2];
            let mut eh_nosign = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let phi_x = jit00 * ref_phi[i * 2] + jit01 * ref_phi[i * 2 + 1];
                let phi_y = jit10 * ref_phi[i * 2] + jit11 * ref_phi[i * 2 + 1];
                eh[0] += s * uh[dofs[i]] * phi_x;
                eh[1] += s * uh[dofs[i]] * phi_y;
                eh_nosign[0] += uh[dofs[i]] * phi_x;
                eh_nosign[1] += uh[dofs[i]] * phi_y;
            }
            let ex = (PI * xp[1]).sin();
            let ey = (PI * xp[0]).sin();
            let dx = eh[0] - ex;
            let dy = eh[1] - ey;
            err2 += w * (dx * dx + dy * dy);
            let dx2 = eh_nosign[0] - ex;
            let dy2 = eh_nosign[1] - ey;
            err2_nosign += w * (dx2 * dx2 + dy2 * dy2);
        }
    }
    println!("  with signs: {:.6e},  without signs: {:.6e}", err2.sqrt(), err2_nosign.sqrt());
    err2.sqrt()
}
