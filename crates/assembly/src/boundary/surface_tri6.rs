//! Tri6 (P2) surface finite element integrators for 3-D embedded surfaces.
//!
//! Supports the same operations as [`super::SurfaceAssembler`] but for
//! 6-node quadratic triangles (P2) on a 2-D manifold in 3-D space.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use super::surface::get_coord3;
use super::surface::{surface_metric_at, pseudo_inverse, SurfaceTri6BilinearIntegrator, SurfaceTri6LinearIntegrator};

// ─── P2 Jacobian and metric helpers ────────────────────────────────────────

/// Evaluate P2 Jacobian at an arbitrary point (xi, eta) on the reference triangle.
fn p2_jacobian_at(x: &[[f64; 3]; 6], xi: f64, eta: f64) -> [[f64; 3]; 2] {
    let dxi = [
        4.0*xi + 4.0*eta - 3.0,    // dphi0/dxi
        4.0*xi - 1.0,               // dphi1/dxi
        0.0,                         // dphi2/dxi
        4.0 - 8.0*xi - 4.0*eta,    // dphi3/dxi
        4.0*eta,                     // dphi4/dxi
        -4.0*eta,                    // dphi5/dxi
    ];
    let det = [
        4.0*xi + 4.0*eta - 3.0,    // dphi0/deta
        0.0,                         // dphi1/deta
        4.0*eta - 1.0,              // dphi2/deta
        -4.0*xi,                     // dphi3/deta
        4.0*xi,                      // dphi4/deta
        4.0 - 4.0*xi - 8.0*eta,    // dphi5/deta
    ];
    let mut j0 = [0.0; 3]; let mut j1 = [0.0; 3];
    for i in 0..6 {
        for c in 0..3 { j0[c] += dxi[i] * x[i][c]; j1[c] += det[i] * x[i][c]; }
    }
    [j0, j1]
}

/// Surface Jacobian at centroid – public for use in error computation.
pub fn surface_jacobian_tri6(x: &[[f64; 3]; 6]) -> ([[f64; 3]; 2], f64, [f64; 3]) {
    let j = p2_jacobian_at(x, 1.0/3.0, 1.0/3.0);
    let (_det_g, sqrt_det_g) = surface_metric_at(&j);
    let nx: f64 = j[0][1]*j[1][2] - j[0][2]*j[1][1];
    let ny: f64 = j[0][2]*j[1][0] - j[0][0]*j[1][2];
    let nz: f64 = j[0][0]*j[1][1] - j[0][1]*j[1][0];
    let n_len: f64 = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-30);
    (j, sqrt_det_g, [nx/n_len, ny/n_len, nz/n_len])
}

// ─── P2 reference basis ───────────────────────────────────────────────────

/// Evaluate all 6 P2 basis functions at (xi, eta).
pub fn p2_basis_tri6(xi: f64, eta: f64) -> [f64; 6] {
    let s = 1.0 - xi - eta;
    [
        2.0 * s * (s - 0.5),
        2.0 * xi * (xi - 0.5),
        2.0 * eta * (eta - 0.5),
        4.0 * xi * s,
        4.0 * xi * eta,
        4.0 * eta * s,
    ]
}

/// Surface gradients of P2 basis at a given point.
fn p2_surface_gradients_at(pinv: &[[f64; 2]; 3], xi: f64, eta: f64) -> [[f64; 3]; 6] {
    let dxi = [
        4.0*xi + 4.0*eta - 3.0, 4.0*xi - 1.0, 0.0,
        4.0 - 8.0*xi - 4.0*eta, 4.0*eta, -4.0*eta,
    ];
    let det = [
        4.0*xi + 4.0*eta - 3.0, 0.0, 4.0*eta - 1.0,
        -4.0*xi, 4.0*xi, 4.0 - 4.0*xi - 8.0*eta,
    ];
    let mut sg = [[0.0; 3]; 6];
    for i in 0..6 {
        for c in 0..3 {
            sg[i][c] = dxi[i] * pinv[c][0] + det[i] * pinv[c][1];
        }
    }
    sg
}

// ─── P2 surface integrators ───────────────────────────────────────────────

/// Surface diffusion (Laplace-Beltrami) bilinear form for Tri6: `∫_Γ ∇_Γ u · ∇_Γ v dS`
pub struct SurfaceTri6DiffusionIntegrator;

impl SurfaceTri6DiffusionIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        // 3-point Gauss quadrature on reference triangle
        let qpts = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];
        let qwt = [1.0/6.0, 1.0/6.0, 1.0/6.0];

        for q in 0..3 {
            let (xi, eta) = (qpts[q][0], qpts[q][1]);
            let j = p2_jacobian_at(elem_nodes, xi, eta);
            let (det_g, sqrt_det_g) = surface_metric_at(&j);
            let pinv = pseudo_inverse(&j, &[[j[0][0]*j[0][0]+j[0][1]*j[0][1]+j[0][2]*j[0][2], j[0][0]*j[1][0]+j[0][1]*j[1][1]+j[0][2]*j[1][2]], [j[0][0]*j[1][0]+j[0][1]*j[1][1]+j[0][2]*j[1][2], j[1][0]*j[1][0]+j[1][1]*j[1][1]+j[1][2]*j[1][2]]], det_g);
            let sg = p2_surface_gradients_at(&pinv, xi, eta);

            let area_factor = sqrt_det_g * qwt[q];

            for i in 0..6 {
                for j in 0..6 {
                    let mut dot = 0.0;
                    for c in 0..3 { dot += sg[i][c] * sg[j][c]; }
                    k_elem[i * 6 + j] += dot * area_factor;
                }
            }
        }
    }
}

/// Surface mass bilinear form for Tri6: `∫_Γ u v dS`
pub struct SurfaceTri6MassIntegrator;

impl SurfaceTri6MassIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        let qpts = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];
        let qwt = [1.0/6.0, 1.0/6.0, 1.0/6.0];

        for q in 0..3 {
            let (xi, eta) = (qpts[q][0], qpts[q][1]);
            let j = p2_jacobian_at(elem_nodes, xi, eta);
            let (_det_g, sqrt_det_g) = surface_metric_at(&j);
            let phi = p2_basis_tri6(xi, eta);

            let area_factor = sqrt_det_g * qwt[q];

            for i in 0..6 {
                for j in 0..6 {
                    k_elem[i * 6 + j] += phi[i] * phi[j] * area_factor;
                }
            }
        }
    }
}

/// Surface domain source linear form for Tri6: `∫_Γ f(x) v(x) dS`
pub struct SurfaceTri6DomainSourceIntegrator<'a> {
    pub f: &'a dyn Fn(&[f64; 3]) -> f64,
}

impl SurfaceTri6DomainSourceIntegrator<'_> {
    pub fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 6], f_elem: &mut [f64; 6]) {
        let qpts = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];
        let qwt = [1.0/6.0, 1.0/6.0, 1.0/6.0];

        for q in 0..3 {
            let (xi, eta) = (qpts[q][0], qpts[q][1]);
            let j = p2_jacobian_at(elem_nodes, xi, eta);
            let (_det_g, sqrt_det_g) = surface_metric_at(&j);
            let phi = p2_basis_tri6(xi, eta);

            let area_factor = sqrt_det_g * qwt[q];

            // Physical point for source evaluation
            let mut x_phys = [0.0; 3];
            for i in 0..6 {
                for c in 0..3 {
                    x_phys[c] += phi[i] * elem_nodes[i][c];
                }
            }
            let f_val = (self.f)(&x_phys);

            for i in 0..6 {
                f_elem[i] += f_val * phi[i] * area_factor;
            }
        }
    }
}

// ─── Surface Assembler for Tri6 ───────────────────────────────────────────

/// Assemble a surface bilinear form using a Tri6 surface integrator.
pub struct SurfaceTri6Assembler;

impl SurfaceTri6Assembler {
    pub fn assemble_bilinear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceTri6BilinearIntegrator],
    ) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut coo = CooMatrix::new(n_dofs, n_dofs);

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 6 { continue; }
            let x: [[f64; 3]; 6] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
                get_coord3(mesh, nodes[3]),
                get_coord3(mesh, nodes[4]),
                get_coord3(mesh, nodes[5]),
            ];
            let mut ke = [0.0; 36];
            for integ in integrators {
                integ.add_to_element_matrix(&x, &mut ke);
            }
            for i in 0..6 {
                for j in 0..6 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, ke[i * 6 + j]);
                }
            }
        }
        coo.into_csr()
    }

    pub fn assemble_linear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceTri6LinearIntegrator],
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut rhs = vec![0.0; n_dofs];

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 6 { continue; }
            let x: [[f64; 3]; 6] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
                get_coord3(mesh, nodes[3]),
                get_coord3(mesh, nodes[4]),
                get_coord3(mesh, nodes[5]),
            ];
            let mut fe = [0.0; 6];
            for integ in integrators {
                integ.add_to_element_vector(&x, &mut fe);
            }
            for i in 0..6 {
                rhs[dofs[i] as usize] += fe[i];
            }
        }
        rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tri6_surface_diffusion_runs() {
        let x = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let integ = SurfaceTri6DiffusionIntegrator;
        let mut ke = [0.0; 36];
        integ.add_to_element_matrix(&x, &mut ke);
        let trace: f64 = (0..6).map(|i| ke[i * 6 + i]).sum();
        assert!(trace > 0.0, "diffusion matrix trace should be positive");
    }

    #[test]
    fn tri6_surface_mass_runs() {
        let x = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let integ = SurfaceTri6MassIntegrator;
        let mut ke = [0.0; 36];
        integ.add_to_element_matrix(&x, &mut ke);
        let trace: f64 = (0..6).map(|i| ke[i * 6 + i]).sum();
        assert!(trace > 0.0, "mass matrix trace should be positive");
    }
}



impl SurfaceTri6BilinearIntegrator for SurfaceTri6DiffusionIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

impl SurfaceTri6BilinearIntegrator for SurfaceTri6MassIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

impl SurfaceTri6LinearIntegrator for SurfaceTri6DomainSourceIntegrator<'_> {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 6], f_elem: &mut [f64; 6]) {
        self.add_to_element_vector(elem_nodes, f_elem);
    }
}
