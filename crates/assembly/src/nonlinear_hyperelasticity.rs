//! Compressible Neo-Hookean hyperelasticity.
//!
//! Implements [`NonlinearForm`] for finite-strain hyperelasticity using the
//! Newton–Raphson method.
//!
//! ## Strain energy density
//! ```text
//! ψ(F) = μ/2 (tr(C) - dim) - μ·ln(J) + λ/2·(ln(J))²
//! ```
//! where `F = I + ∇u`, `C = FᵀF`, `J = det(F)`.
//!
//! ## 1st Piola–Kirchhoff stress
//! ```text
//! P(F) = μ·F + (λ·ln(J) - μ)·F⁻ᵀ
//! ```
//!
//! ## Tangent modulus (consistent linearisation)
//! ```text
//! ∂P_{iI}/∂F_{jJ} = μ·δᵢⱼ·δᴵᴶ + λ·(F⁻¹)ᴵₖ·(F⁻¹)ᴶₖ·δᵢⱼ
//!                   - (λ·ln(J) - μ)·(F⁻¹)ᴶᵢ·(F⁻¹)ᴵⱼ
//! ```

#![allow(non_snake_case)]

use nalgebra::DMatrix;

use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::vector_h1::VectorH1Space;

use crate::nonlinear::NonlinearForm;

// ─── HyperelasticityForm ─────────────────────────────────────────────────────

/// Compressible Neo-Hookean hyperelasticity form.
///
/// Implements [`NonlinearForm`] for finite-strain elasticity using
/// `VectorH1Space` (interleaved DOFs: `[u0_x, u0_y, u1_x, u1_y, ...]`).
pub struct HyperelasticityForm<M: MeshTopology> {
    space: VectorH1Space<M>,
    /// First Lamé parameter.
    pub lambda: f64,
    /// Second Lamé parameter (shear modulus).
    pub mu: f64,
    /// Dirichlet BC: `(global_dof, value)` pairs.
    pub dirichlet: Vec<(usize, f64)>,
    /// Quadrature order for assembly.
    pub quad_order: u8,
}

impl<M: MeshTopology> HyperelasticityForm<M> {
    pub fn new(space: VectorH1Space<M>, lambda: f64, mu: f64,
               dirichlet: Vec<(usize, f64)>, quad_order: u8) -> Self {
        Self { space, lambda, mu, dirichlet, quad_order }
    }

    fn pk1_and_tangent(&self, f: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        let dim = f.nrows();
        let jac = f.determinant();
        let inv_f = f.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
        let inv_f_t = inv_f.transpose();
        let ln_j = jac.ln();

        let mut p = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            for I in 0..dim {
                p[(i, I)] = self.mu * f[(i, I)]
                    + (self.lambda * ln_j - self.mu) * inv_f_t[(i, I)];
            }
        }

        let n = dim * dim;
        let mut ct = DMatrix::zeros(n, n);
        let pre = self.lambda * ln_j - self.mu;
        for i in 0..dim {
            for I in 0..dim {
                let row = i * dim + I;
                for j in 0..dim {
                    for J in 0..dim {
                        let col = j * dim + J;
                        let mut val = 0.0;
                        if i == j && I == J { val += self.mu; }
                        if i == j {
                            let mut sum = 0.0;
                            for k in 0..dim { sum += inv_f[(I, k)] * inv_f[(J, k)]; }
                            val += self.lambda * sum;
                        }
                        val -= pre * inv_f[(J, i)] * inv_f[(I, j)];
                        ct[(row, col)] = val;
                    }
                }
            }
        }
        (p, ct)
    }
}

impl<M: MeshTopology> NonlinearForm for HyperelasticityForm<M> {
    fn n_dofs(&self) -> usize { self.space.n_dofs() }

    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();

        for i in 0..r.len() { r[i] = -rhs[i]; }

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);

            let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let jit = jac.try_inverse().expect("singular").transpose();

            let mut u_elem = vec![0.0_f64; n_vec];
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut f_elem = vec![0.0_f64; n_vec];
            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                let (p, _ct) = self.pk1_and_tangent(&f_mat);

                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        let mut s = 0.0;
                        for j in 0..dim { s += p[(i, j)] * gphys[k * dim + j]; }
                        f_elem[row] += w * s;
                    }
                }
            }
            for (k, &dof) in elem_dofs.iter().enumerate() { r[dof] += f_elem[k]; }
        }

        for &(dof, val) in &self.dirichlet { r[dof] = u[dof] - val; }
    }

    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();
        let n_dofs = self.space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);

            let elem_dofs: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let jit = jac.try_inverse().expect("singular").transpose();

            let mut u_elem = vec![0.0_f64; n_vec];
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut k_elem = vec![0.0_f64; n_vec * n_vec];
            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                let mut du = DMatrix::zeros(dim, dim);
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        for j in 0..dim {
                            du[(i, j)] += u_elem[k * dim + i] * gphys[k * dim + j];
                        }
                    }
                }
                let mut f_mat = DMatrix::identity(dim, dim);
                f_mat += &du;
                let (_p, ct) = self.pk1_and_tangent(&f_mat);

                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        for l in 0..n_ldofs {
                            for a in 0..dim {
                                let col = l * dim + a;
                                let mut val = 0.0;
                                for j in 0..dim {
                                    for b in 0..dim {
                                        val += ct[(i * dim + j, a * dim + b)]
                                            * gphys[k * dim + j]
                                            * gphys[l * dim + b];
                                    }
                                }
                                k_elem[row * n_vec + col] += w * val;
                            }
                        }
                    }
                }
            }
            coo.add_element_matrix(&elem_dofs, &k_elem);
        }

        let mut mat = coo.into_csr();
        for &(dof, _val) in &self.dirichlet {
            mat.apply_dirichlet_row_zeroing(dof, 0.0, &mut vec![0.0; n_dofs]);
        }
        mat
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("hyperelasticity ref_elem_vol: unsupported ({et:?}, {order})"),
    }
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col+1]);
        for row in 0..dim { j[(row,col)] = xc[row] - x0[row]; }
    }
    let det = j.determinant();
    (j, det)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::vector_h1::VectorH1Space;


    // F = I → P = 0 → residual = 0
    #[test]
    fn zero_displacement_zero_residual() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let form = HyperelasticityForm::new(space, 1.0, 0.3, vec![], 2);
        let mut r = vec![0.0_f64; n];
        form.residual(&vec![0.0; n], &vec![0.0; n], &mut r);
        let norm: f64 = r.iter().map(|x| x.abs()).sum();
        assert!(norm < 1e-12, "Zero displacement should give zero residual, got {norm}");
    }

    // Tangent matrix non-zero
    #[test]
    fn tangent_matrix_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let form = HyperelasticityForm::new(space, 1.0, 0.3, vec![], 2);
        let jac = form.jacobian(&vec![0.0; n]);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) {
                sum += jac.get(i, j).abs();
            }
        }
        assert!(sum > 0.0, "Tangent matrix should have non-zero entries");
    }


}
