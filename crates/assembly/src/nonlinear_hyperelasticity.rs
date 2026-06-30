//! Finite-strain hyperelasticity with multiple material models.
//!
//! Implements [`NonlinearForm`] for:
//! - **Neo-Hookean** (compressible)
//! - **Mooney–Rivlin** (incompressible + bulk penalty)
//! - **Ogden** (N=1,2,3)
//!
//! All models support 2D/3D via `VectorH1Space` and use the
//! Newton–Raphson solver with Armijo line-search from [`NewtonSolver`].
//!
//! ## Strain energy densities
//!
//! ### Neo-Hookean (compressible, implemented)
//! ```text
//! ψ = μ/2·(tr(C)-3) - μ·ln(J) + λ/2·(ln(J))²
//! ```
//!
//! ### Mooney–Rivlin
//! ```text
//! ψ = C10·(I₁-3) + C01·(I₂-3) + K/2·(J-1)²
//! ```
//!
//! ### Ogden (N-term)
//! ```text
//! ψ = Σ_{p=1}^N μ_p/α_p·(λ₁^{α_p}+λ₂^{α_p}+λ₃^{α_p}-3) + K/2·(J-1)²
//! ```
//!
//! where `C = FᵀF`, `I₁ = tr(C)`, `I₂ = ½((tr(C))² - tr(C²))`, `J = det(F)`.

#![allow(non_snake_case)]

use nalgebra::DMatrix;
use nalgebra::linalg::SVD;

use fem_element::{
    ReferenceElement,
    lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::vector_h1::VectorH1Space;

use crate::nonlinear::{NonlinearForm, NewtonSolver, NewtonConfig, NewtonResult};

/// Hyperelastic material model.
#[derive(Debug, Clone)]
pub enum HyperelasticModel {
    /// Compressible Neo-Hookean: `μ/2·(I₁-3) - μ·ln(J) + λ/2·(ln(J))²`.
    NeoHookean { mu: f64, lambda: f64 },
    /// Mooney–Rivlin: `C10·(I₁-3) + C01·(I₂-3) + K/2·(J-1)²`.
    MooneyRivlin { c10: f64, c01: f64, bulk_modulus: f64 },
    /// N-term Ogden: `Σ μ_p/α_p·(λ₁^α+λ₂^α+λ₃^α-3) + K/2·(J-1)²`.
    Ogden { params: Vec<(f64, f64)>, bulk_modulus: f64 },
}

impl HyperelasticModel {
    /// PK1 stress and consistent tangent for the model.
    fn pk1_and_tangent(&self, f: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        match self {
            HyperelasticModel::NeoHookean { mu, lambda } => {
                neo_hookean_pk1_tangent(f, *mu, *lambda)
            }
            HyperelasticModel::MooneyRivlin { c10, c01, bulk_modulus } => {
                mooney_rivlin_pk1_tangent(f, *c10, *c01, *bulk_modulus)
            }
            HyperelasticModel::Ogden { params, bulk_modulus } => {
                ogden_pk1_tangent(f, params, *bulk_modulus)
            }
        }
    }
}

// ─── Neo-Hookean (existing) ──────────────────────────────────────────────────

fn neo_hookean_pk1_tangent(f: &DMatrix<f64>, mu: f64, lambda: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = f.nrows();
    let jac = f.determinant();
    let inv_f = f.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
    let inv_f_t = inv_f.transpose();
    let ln_j = jac.ln();

    let mut p = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for I in 0..dim {
            p[(i, I)] = mu * f[(i, I)] + (lambda * ln_j - mu) * inv_f_t[(i, I)];
        }
    }

    let n = dim * dim;
    let mut ct = DMatrix::zeros(n, n);
    let pre = lambda * ln_j - mu;
    for i in 0..dim { for I in 0..dim {
        let row = i * dim + I;
        for j in 0..dim { for J in 0..dim {
            let col = j * dim + J;
            let mut val = 0.0;
            if i == j && I == J { val += mu; }
            if i == j {
                let mut sum = 0.0;
                for k in 0..dim { sum += inv_f[(I, k)] * inv_f[(J, k)]; }
                val += lambda * sum;
            }
            val -= pre * inv_f[(J, i)] * inv_f[(I, j)];
            ct[(row, col)] = val;
        }}
    }}
    (p, ct)
}

// ─── Mooney–Rivlin ───────────────────────────────────────────────────────────

fn mooney_rivlin_pk1_tangent(f: &DMatrix<f64>, c10: f64, c01: f64, K: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = f.nrows();
    let jac = f.determinant();
    let inv_f = f.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
    let inv_f_t = inv_f.transpose();
    let c = f.transpose() * f;
    let i1 = c.trace();
    // PK1 via push-forward of PK2
    let mut s_pk2 = DMatrix::zeros(dim, dim);
    let pk2_pre = 2.0 * (c10 + c01 * i1);
    let c_inv = inv_f * inv_f_t;
    for i in 0..dim { for j in 0..dim {
        let hat = if i == j { 1.0 } else { 0.0 };
        s_pk2[(i, j)] = pk2_pre * hat - 2.0 * c01 * c[(i, j)] + K * jac * (jac - 1.0) * c_inv[(i, j)];
    }}
    let p = f * &s_pk2;
    // Numerical tangent via central differences
    let ct = numerical_tangent(f, &|ft| {
        let j = ft.determinant();
        let i = ft.transpose();
        let ci = ft.transpose() * ft;
        let i1t = ci.trace();
        let ci_inv = ft.clone().try_inverse().map(|mi| { let mt = mi.transpose(); mi * mt }).unwrap_or_else(|| DMatrix::identity(dim, dim));
        let mut s2 = DMatrix::zeros(dim, dim);
        let pre = 2.0 * (c10 + c01 * i1t);
        for ii in 0..dim { for jj in 0..dim {
            let h = if ii == jj { 1.0 } else { 0.0 };
            s2[(ii, jj)] = pre * h - 2.0 * c01 * ci[(ii, jj)] + K * j * (j - 1.0) * ci_inv[(ii, jj)];
        }}
        ft * &s2
    });
    (p, ct)
}

// ─── Ogden ───────────────────────────────────────────────────────────────────

fn ogden_pk1_tangent(f: &DMatrix<f64>, params: &[(f64, f64)], K: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = f.nrows();
    let jac = f.determinant();
    let inv_f_t = f.clone().try_inverse().map(|m| m.transpose()).unwrap_or_else(|| DMatrix::identity(dim, dim));

    // Principal stretches via SVD of F
    let svd = SVD::new(f.clone(), true, true);
    let u_mat = svd.u.expect("SVD u failed");
    let v_t_mat = svd.v_t.expect("SVD v_t failed");
    let mut lam = vec![1.0_f64; dim];
    for i in 0..dim { lam[i] = svd.singular_values[i]; }

    // PK2 in spectral basis
    let mut s = DMatrix::zeros(dim, dim);
    let v_mat = v_t_mat.transpose(); // V

    for i in 0..dim {
        for I in 0..dim {
            let mut val = 0.0;
            for a in 0..dim {
                if lam[a].abs() < 1e-30 { continue; }
                let mut dW_dlam = 0.0;
                for (mu_p, alpha_p) in params {
                    dW_dlam += mu_p * lam[a].powf(alpha_p - 1.0);
                }
                dW_dlam += K * (jac - 1.0) * jac / lam[a];

                let uia = u_mat[(i, a)];
                let vIa = v_mat[(I, a)];
                val += dW_dlam / lam[a] * uia * vIa;
            }
            s[(i, I)] = val;
        }
    }

    // PK1: P = Σ (1/λ)·(dW/dλ)·n⊗N — computed via the spectral formula
    let p = s.clone();

    // Numerical tangent via central differences
    let ct = numerical_tangent(f, &|ft| {
        ogden_pk1_only(ft, params, K)
    });

    (p, ct)
}

/// Ogden PK1 stress only (no tangent) — for numerical differentiation.
fn ogden_pk1_only(f: &DMatrix<f64>, params: &[(f64, f64)], K: f64) -> DMatrix<f64> {
    let dim = f.nrows();
    let jac = f.determinant();
    let svd = SVD::new(f.clone(), true, true);
    let u_mat = svd.u.expect("SVD u failed");
    let v_t_mat = svd.v_t.expect("SVD v_t failed");
    let mut lam = vec![1.0_f64; dim];
    for i in 0..dim { lam[i] = svd.singular_values[i]; }
    let v_mat = v_t_mat.transpose();

    let mut p = DMatrix::zeros(dim, dim);
    for i in 0..dim { for I in 0..dim {
        let mut val = 0.0;
        for a in 0..dim {
            if lam[a].abs() < 1e-30 { continue; }
            let mut dW_dlam = 0.0;
            for (mu_p, alpha_p) in params {
                dW_dlam += mu_p * lam[a].powf(alpha_p - 1.0);
            }
            dW_dlam += K * (jac - 1.0) * jac / lam[a];
            val += dW_dlam / lam[a] * u_mat[(i, a)] * v_mat[(I, a)];
        }
        p[(i, I)] = val;
    }}
    p
}

// ─── Numerical tangent (fallback for non-analytical models) ──────────────────

fn numerical_tangent(f: &DMatrix<f64>, pk1_fn: &dyn Fn(&DMatrix<f64>) -> DMatrix<f64>) -> DMatrix<f64> {
    let dim = f.nrows();
    let n = dim * dim;
    let mut ct = DMatrix::zeros(n, n);
    let eps = 1e-8;
    let p0 = pk1_fn(f);

    for j in 0..dim {
        for J in 0..dim {
            let mut f_pert = f.clone();
            f_pert[(j, J)] += eps;
            let p_pert = pk1_fn(&f_pert);
            for i in 0..dim {
                for I in 0..dim {
                    let row = i * dim + I;
                    let col = j * dim + J;
                    ct[(row, col)] = (p_pert[(i, I)] - p0[(i, I)]) / eps;
                }
            }
        }
    }
    ct
}

// ─── HyperelasticityForm (refactored to use HyperelasticModel) ───────────────

/// Finite-strain hyperelasticity form with selectable material model.
pub struct HyperelasticityForm<M: MeshTopology> {
    space: VectorH1Space<M>,
    pub model: HyperelasticModel,
    pub dirichlet: Vec<(usize, f64)>,
    pub quad_order: u8,
}

impl<M: MeshTopology> HyperelasticityForm<M> {
    pub fn new(space: VectorH1Space<M>, model: HyperelasticModel,
               dirichlet: Vec<(usize, f64)>, quad_order: u8) -> Self {
        Self { space, model, dirichlet, quad_order }
    }

    /// Run Newton–Raphson with line search to solve `F(u) = 0`.
    pub fn solve(&self, rhs: &[f64], u: &mut [f64],
                 config: &NewtonConfig) -> Result<NewtonResult, NewtonResult> {
        NewtonSolver::new(config.clone()).solve(self, rhs, u)
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
                let (p, _ct) = self.model.pk1_and_tangent(&f_mat);

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
                let (_p, ct) = self.model.pk1_and_tangent(&f_mat);

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

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

    #[test]
    fn zero_displacement_zero_residual_neo() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let mut r = vec![0.0_f64; n];
        form.residual(&vec![0.0; n], &vec![0.0; n], &mut r);
        let norm: f64 = r.iter().map(|x| x.abs()).sum();
        assert!(norm < 1e-12, "Zero displacement should give zero residual, got {norm}");
    }

    #[test]
    fn tangent_matrix_nonzero_neo() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let jac = form.jacobian(&vec![0.0; n]);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) {
                sum += jac.get(i, j).abs();
            }
        }
        assert!(sum > 0.0, "Tangent matrix should have non-zero entries");
    }

    #[test]
    fn mooney_rivlin_tangent_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::MooneyRivlin { c10: 0.3, c01: 0.1, bulk_modulus: 1e3 };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let jac = form.jacobian(&vec![0.0; n]);
        let mut sum = 0.0;
        for i in 0..n.min(10) { for j in 0..n.min(10) { sum += jac.get(i, j).abs(); } }
        assert!(sum > 0.0, "MR tangent should be non-zero");
    }

    #[test]
    fn ogden_tangent_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let model = HyperelasticModel::Ogden {
            params: vec![(6.0e5, 1.3), (-1.0e5, 2.0)],
            bulk_modulus: 1e3,
        };
        let form = HyperelasticityForm::new(space, model, vec![], 2);
        let jac = form.jacobian(&vec![0.0; n]);
        let mut sum = 0.0;
        for i in 0..n.min(10) { for j in 0..n.min(10) { sum += jac.get(i, j).abs(); } }
        assert!(sum > 0.0, "Ogden tangent should be non-zero");
    }
}
