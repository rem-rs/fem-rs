//! Linear elasticity bilinear form integrator.
//!
//! Computes the element contribution to the symmetric bilinear form
//!
//! ```text
//! a(u, v) = ∫_Ω σ(u) : ε(v) dx
//!          = ∫_Ω [ λ (∇·u)(∇·v) + 2μ ε(u):ε(v) ] dx
//! ```
//!
//! where `ε(u) = ½(∇u + (∇u)ᵀ)` is the symmetric strain tensor and
//! `σ(u) = λ tr(ε)I + 2με` is the Cauchy stress (Lamé parameters λ, μ).
//!
//! # DOF convention
//! Element DOFs must be **interleaved** (node-major):
//! `[u_x(0), u_y(0), u_x(1), u_y(1), …]` as produced by [`VectorH1Space`].

use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the isotropic linear elasticity operator.
///
/// # Parameters
/// - `lambda`: first Lamé parameter
/// - `mu`:     shear modulus
/// - `plane_stress`: if true, modifies λ for 2D plane stress (σ_33 = 0).
///   For 3D meshes this flag has no effect.
pub struct ElasticityIntegrator<C1: ScalarCoeff = f64, C2: ScalarCoeff = f64> {
    pub lambda: C1,
    pub mu: C2,
    pub plane_stress: bool,
}

impl<C1: ScalarCoeff, C2: ScalarCoeff> ElasticityIntegrator<C1, C2> {
    pub fn new(lambda: C1, mu: C2) -> Self { Self { lambda, mu, plane_stress: false } }
    pub fn with_plane_stress(mut self, val: bool) -> Self { self.plane_stress = val; self }
}

impl<C1: ScalarCoeff, C2: ScalarCoeff> BilinearIntegrator for ElasticityIntegrator<C1, C2> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let dim   = qp.dim;
        let n     = qp.n_dofs;
        let n_nodes = n / dim;
        let w = qp.weight;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let lam = self.lambda.eval(&ctx);
        let mu  = self.mu.eval(&ctx);
        // Plane stress: λ* = 2λμ/(λ+2μ)
        let lam_eff = if self.plane_stress && dim == 2 { 2.0 * lam * mu / (lam + 2.0 * mu).max(1e-30) } else { lam };

        for k in 0..n_nodes {
            for a in 0..dim {
                let row = k * dim + a;
                let grada: Vec<f64> = (0..dim).map(|d| qp.grad_phys[k * dim + d]).collect();

                for l in 0..n_nodes {
                    for b in 0..dim {
                        let col = l * dim + b;
                        let gradb: Vec<f64> = (0..dim).map(|d| qp.grad_phys[l * dim + d]).collect();

                        let vol = lam_eff * grada[a] * gradb[b];
                        let mut shear = 0.0;
                        for i in 0..dim {
                            for j in 0..dim {
                                let eij_a = 0.5 * ((if j == a { grada[i] } else { 0.0 }) + (if i == a { grada[j] } else { 0.0 }));
                                let eij_b = 0.5 * ((if j == b { gradb[i] } else { 0.0 }) + (if i == b { gradb[j] } else { 0.0 }));
                                shear += eij_a * eij_b;
                            }
                        }
                        k_elem[row * n + col] += w * (vol + 2.0 * mu * shear);
                    }
                }
            }
        }
    }
}

// ─── Hyperelastic constitutive models ──────────────────────────────────────

/// Compute the second Piola-Kirchhoff stress S and tangent modulus C
/// for a given deformation gradient F and constitutive model.
pub trait HyperelasticModel: Send + Sync {
    /// Compute stress S (voigt, symmetric 3×3 → 6 components: xx,yy,zz,xy,xz,yz)
    /// and tangent modulus C (6×6 voigt).
    fn stress_and_modulus(&self, f: &[f64; 9], s: &mut [f64; 6], c: &mut [[f64; 6]; 6]);
}

/// St-Venant Kirchhoff model: S = λ tr(E) I + 2μ E, where E = ½(FᵀF - I).
pub struct StVenantKirchhoff { pub lambda: f64, pub mu: f64 }
impl StVenantKirchhoff {
    pub fn new(lambda: f64, mu: f64) -> Self { Self { lambda, mu } }
}
impl HyperelasticModel for StVenantKirchhoff {
    fn stress_and_modulus(&self, f: &[f64; 9], s: &mut [f64; 6], c: &mut [[f64; 6]; 6]) {
        let (f11,f12,f13,f21,f22,f23,f31,f32,f33) = (f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8]);
        // Green-Lagrange strain E = ½(FᵀF - I)
        let e11 = 0.5*(f11*f11+f21*f21+f31*f31-1.0);
        let e22 = 0.5*(f12*f12+f22*f22+f32*f32-1.0);
        let e33 = 0.5*(f13*f13+f23*f23+f33*f33-1.0);
        let e12 = 0.5*(f11*f12+f21*f22+f31*f32);
        let e13 = 0.5*(f11*f13+f21*f23+f31*f33);
        let e23 = 0.5*(f12*f13+f22*f23+f32*f33);
        let tr_e = e11+e22+e33;
        // S = λ tr(E) I + 2μ E
        s[0] = self.lambda * tr_e + 2.0 * self.mu * e11;
        s[1] = self.lambda * tr_e + 2.0 * self.mu * e22;
        s[2] = self.lambda * tr_e + 2.0 * self.mu * e33;
        s[3] = 2.0 * self.mu * e12;
        s[4] = 2.0 * self.mu * e13;
        s[5] = 2.0 * self.mu * e23;
        // Tangent modulus C = λ 1⊗1 + 2μ I (constant for StVK, in voigt)
        c.fill([0.0; 6]);
        let two_mu = 2.0 * self.mu;
        c[0][0] = self.lambda + two_mu; c[0][1] = self.lambda;       c[0][2] = self.lambda;
        c[1][0] = self.lambda;       c[1][1] = self.lambda + two_mu; c[1][2] = self.lambda;
        c[2][0] = self.lambda;       c[2][1] = self.lambda;       c[2][2] = self.lambda + two_mu;
        c[3][3] = two_mu;
        c[4][4] = two_mu;
        c[5][5] = two_mu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn stvk_identity_f_gives_zero_stress() {
        let model = StVenantKirchhoff::new(10.0, 5.0);
        let f = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0];
        let mut s = [0.0; 6]; let mut c = [[0.0; 6]; 6];
        model.stress_and_modulus(&f, &mut s, &mut c);
        for &v in &s { assert!(v.abs() < 1e-14); }
    }
    #[test] fn stvk_uniaxial_stretch() {
        let model = StVenantKirchhoff::new(1.0, 1.0);
        let f = [1.1,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0];
        let mut s = [0.0; 6]; let mut c = [[0.0; 6]; 6];
        model.stress_and_modulus(&f, &mut s, &mut c);
        assert!(s[0] > 0.0, "expected tensile stress, got {:.3e}", s[0]);
        assert!(s[3].abs() < 1e-14, "shear should be zero");
    }

    use crate::assembler::Assembler;
    use fem_mesh::Mesh;
    use fem_space::VectorH1Space;

    #[test] fn elasticity_matrix_symmetric() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let integ = ElasticityIntegrator::new(1.0, 1.0);
        let mat = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense(); let n = mat.nrows;
        for i in 0..n { for j in 0..n { assert!((dense[i*n+j]-dense[j*n+i]).abs() < 1e-11); } }
    }
    #[test] fn elasticity_row_sums_zero() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = VectorH1Space::new(mesh, 1, 2);
        let integ = ElasticityIntegrator::new(1.0, 0.5);
        let mat = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense(); let n = mat.nrows;
        for row in 0..n { let s: f64 = (0..n).map(|c| dense[row*n+c]).sum(); assert!(s.abs() < 1e-10); }
    }
    #[test] fn plane_stress_differs_from_plane_strain() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = VectorH1Space::new(mesh, 1, 2);
        let strain = ElasticityIntegrator::new(1.0, 0.5);
        let stress = ElasticityIntegrator::new(1.0, 0.5).with_plane_stress(true);
        let k_s = Assembler::assemble_bilinear(&space, &[&stress], 3);
        let k_e = Assembler::assemble_bilinear(&space, &[&strain], 3);
        let mut diff = 0.0;
        for i in 0..k_s.nrows { for j in 0..k_s.ncols { diff += (k_s.get(i,j)-k_e.get(i,j)).abs(); } }
        assert!(diff > 1e-10, "plane stress and strain should differ");
    }
}
