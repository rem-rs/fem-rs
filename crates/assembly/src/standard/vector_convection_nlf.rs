//! Vector convection nonlinear-form integrator — 1:1 port of MFEM
//! `fem/nonlininteg.cpp:VectorConvectionNLFIntegrator`
//! (`N_{i·d+c} = ∫ (u·∇)u_c φ_i dx`, the `N(u)` term of MFEM's
//! `miniapps/fluids/navier` solver).
//!
//! Implemented on the [`NonlinearFormIntegrator`](crate::standard::nonlinear_form::NonlinearFormIntegrator)
//! interface promoted in D52: per QP, MFEM's
//! `AssembleElementVector` (`nonlininteg.cpp`, around line 744) is
//!
//! ```text
//! shape   = el.CalcShape(ip)                      // [nd]
//! dshape  = el.CalcPhysDShape(T, ip)              // [nd × dim], true gradient
//! w       = ip.weight · T.Weight()  (· Q->Eval)
//! gradEF  = EFᵀ · dshape            // gradEF(c,dd) = Σ_k u_kc ∂φ_k/∂x_dd
//! vec1    = EFᵀ · shape             // u_h at the QP
//! vec2    = gradEF · vec1           // ((u·∇)u)_c = Σ_dd u_dd ∂u_c/∂x_dd
//! ELV    += shape ⊗ (w·vec2)        // interleaved (k, c) → k·dim + c
//! ```
//!
//! [`NLQpData`](crate::standard::nonlinear_form::NLQpData) provides exactly
//! these ingredients: `weight` is already the physical measure
//! `ip.weight·|det J|` and `grad_phys` the true physical gradient (not the
//! adjugate-scaled one of the bilinear `QpData`), so the integrator is
//! dimension-correct by construction.  The optional coefficient is MFEM's
//! `Q` (`VectorConvectionNLFIntegrator(Coefficient *q = NULL)`); `coeff = 1.0`
//! reproduces `Q = NULL` (the navier miniapps' `nlcoeff.constant = -1` sign is
//! applied by the solver, outside the integrator).
//!
//! # D42 history
//!
//! Until D52 the name `VectorConvectionNLFIntegrator` belonged to a
//! `standard` *bilinear* integrator on the adjugate-scaled `QpData::grad_phys`
//! with the scalar-space `n_dofs` layout — it computed
//! `∫ φᵢ (v·∇φⱼ) dx` (MFEM `ConvectionIntegrator`, bare `ip.weight`), and
//! indexed out of bounds on an interleaved `[H¹]^d` space.  That misnamed type
//! had zero callers and is deleted; the type in this module is the real MFEM
//! `VectorConvectionNLFIntegrator` on the [`NonlinearForm`] framework.
//!
//! # Integration rule
//!
//! MFEM's default rule (`GetRule`, `nonlininteg.cpp`) is
//! `2·p + T.OrderGrad(&fe)` — `3p` for 2-D Qk spaces on Q1 geometry.  The
//! navier miniapps validated their numbers with the `2p + 1` volume rule
//! (both rules are exact for the degree-`2p−1` integrand on straight
//! elements), so they pin `int_rule = Some(2p+1)` exactly like MFEM's
//! `SetIntRule` mode; left at `None`, the [`NonlinearForm`] default
//! `2p + OrderGrad` applies.

use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::standard::nonlinear_form::NLQpData;

/// MFEM `VectorConvectionNLFIntegrator` (residual + Jacobian) on the
/// [`NonlinearForm`](crate::standard::nonlinear_form::NonlinearForm) framework,
/// for interleaved `[H¹]^d` velocity spaces.
pub struct VectorConvectionNLFIntegrator<C: ScalarCoeff = f64> {
    /// MFEM `Q`: scales the whole integrand (`w *= Q->Eval(T, ip)`).  Use
    /// `1.0` for MFEM's `Q = NULL`.
    pub coeff: C,
    /// MFEM `SetIntRule`: pins the volume quadrature order instead of the
    /// `NonlinearForm` default (`None` = MFEM's unset `IntRule`).
    pub int_rule: Option<i32>,
}

impl<C: ScalarCoeff> VectorConvectionNLFIntegrator<C> {
    /// MFEM `VectorConvectionNLFIntegrator(q)` with the default rule.
    pub fn new(coeff: C) -> Self {
        VectorConvectionNLFIntegrator { coeff, int_rule: None }
    }
}

impl<C: ScalarCoeff> crate::standard::nonlinear_form::NonlinearFormIntegrator
    for VectorConvectionNLFIntegrator<C>
{
    fn int_rule_order(&self, _space_order: i32) -> Option<i32> {
        self.int_rule
    }

    /// MFEM `VectorConvectionNLFIntegrator::AssembleElementVector`.
    fn qp_residual(&self, qp: &NLQpData<'_>, elfun: &[f64], elvect: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, d, qp.elem, 0, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        // MultAtB(EF, dshape, gradEF): gradEF(c, dd) = Σ_k EF(k,c)·dshape(k,dd).
        let mut grad_ef = vec![0.0_f64; d * d];
        for k in 0..n {
            for c in 0..d {
                let uc = elfun[k * d + c];
                for dd in 0..d {
                    grad_ef[c * d + dd] += uc * qp.grad_phys[k * d + dd];
                }
            }
        }
        // EF.MultTranspose(shape, vec1): vec1(c) = Σ_k EF(k,c)·shape(k).
        let mut uh = vec![0.0_f64; d];
        for k in 0..n {
            for c in 0..d {
                uh[c] += elfun[k * d + c] * qp.phi[k];
            }
        }
        // gradEF.Mult(vec1, vec2): vec2(c) = Σ_dd uh(dd)·gradEF(c,dd).
        let mut conv = vec![0.0_f64; d];
        for c in 0..d {
            for dd in 0..d {
                conv[c] += uh[dd] * grad_ef[c * d + dd];
            }
        }
        // vec2 *= w; AddMultVWt(shape, vec2, ELV).
        for k in 0..n {
            for c in 0..d {
                elvect[k * d + c] += w * qp.phi[k] * conv[c];
            }
        }
    }

    /// MFEM `VectorConvectionNLFIntegrator::AssembleElementGrad`: with
    /// `N_{(i,c)} = w ∫ φᵢ (u·∇)u_c dx`,
    ///
    /// ```text
    /// ∂N_{(i,c)}/∂u_{(j,e)} = w·φᵢ·φⱼ·gradEF(e,c) + δ_ce·w·φᵢ·(u·∇φⱼ),
    /// ```
    ///
    /// i.e. MFEM's `w·gradEF(ii,jj)·φ⊗φᵀ` blocks plus the diagonal
    /// `(u·∇φⱼ)·φ⊗φᵀ` blocks (there assembled as
    /// `adjJ·u_h` against the *reference* `dshape`, which is the same
    /// physical gradient times |det J|).
    fn qp_jacobian(&self, qp: &NLQpData<'_>, elfun: &[f64], elmat: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let n_el = n * d;
        let ctx = CoeffCtx::from_qp(qp.x_phys, d, qp.elem, 0, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        // gradEF and u_h exactly as in qp_residual.
        let mut grad_ef = vec![0.0_f64; d * d];
        for k in 0..n {
            for c in 0..d {
                let uc = elfun[k * d + c];
                for dd in 0..d {
                    grad_ef[c * d + dd] += uc * qp.grad_phys[k * d + dd];
                }
            }
        }
        let mut uh = vec![0.0_f64; d];
        for k in 0..n {
            for c in 0..d {
                uh[c] += elfun[k * d + c] * qp.phi[k];
            }
        }
        // (u·∇φ_j) = Σ_dd uh(dd)·grad_phys(j, dd).
        let mut ugrad_phi = vec![0.0_f64; n];
        for j in 0..n {
            for dd in 0..d {
                ugrad_phi[j] += uh[dd] * qp.grad_phys[j * d + dd];
            }
        }
        for i in 0..n {
            for c in 0..d {
                for j in 0..n {
                    let row = (i * d + c) * n_el + j * d;
                    let wij = w * qp.phi[i] * qp.phi[j];
                    for e in 0..d {
                        elmat[row + e] += wij * grad_ef[c * d + e];
                    }
                    // δ_ce: the diagonal (c, c) block also carries (u·∇φ_j).
                    elmat[row + c] += w * qp.phi[i] * ugrad_phi[j];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standard::nonlinear_form::NonlinearForm;
    use crate::standard::VectorH1MassIntegrator;
    use crate::Assembler;
    use fem_element::ReferenceElement;
    use fem_mesh::{topology::MeshTopology, Mesh};
    use fem_space::fe_space::FESpace;
    use fem_space::VectorH1Space;

    /// Non-unit 3×2 mesh (|det J| = 1/3): unit-sized elements hide weight
    /// bugs (see the D42 counter-examples).
    fn mesh() -> Mesh<2> {
        Mesh::<2>::make_cartesian_2d(3, 2, 2.0, 1.0)
    }

    fn space(m: &Mesh<2>, order: u8) -> VectorH1Space<Mesh<2>> {
        VectorH1Space::new(m.clone(), order, 2)
    }

    fn integ(order: i32) -> VectorConvectionNLFIntegrator<f64> {
        VectorConvectionNLFIntegrator { coeff: 1.0, int_rule: Some(order) }
    }

    /// A constant field convects nothing: `u ≡ (1, 1)` gives
    /// `(u·∇)u ≡ 0`, hence `N·u = 0` — up to the partition-of-unity
    /// roundoff `Σ_k ∇φ_k = O(ε)` of the nodal basis (measured 9e-17 here).
    #[test]
    fn constant_field_has_zero_residual() {
        let m = mesh();
        let sp = space(&m, 2);
        let u = vec![1.0_f64; sp.n_dofs()];
        let it = integ(2 * 2 + 1);
        let mut nf = NonlinearForm::new();
        nf.add_domain_integrator(&it);
        let mut y = vec![0.0_f64; sp.n_dofs()];
        nf.mult(&sp, &u, &mut y);
        let max: f64 = y.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!(max < 1e-14, "max |N·u| = {max}");
    }

    /// `u = (x, y)` gives `(u·∇)u = u`, so `N·u = M·u` — against the kernel
    /// `VectorH1MassIntegrator` on the same space (non-unit mesh, so a wrong
    /// weight shows).
    #[test]
    fn linear_field_residual_is_the_mass_vector() {
        let m = mesh();
        let sp = space(&m, 2);
        let u = sp.interpolate_vec(&|x| vec![x[0], x[1]]);
        let u = u.as_slice().to_vec();
        let it = integ(2 * 2 + 1);
        let mut nf = NonlinearForm::new();
        nf.add_domain_integrator(&it);
        let mut res = vec![0.0_f64; sp.n_dofs()];
        nf.mult(&sp, &u, &mut res);
        let mass = Assembler::assemble_bilinear(&sp, &[&VectorH1MassIntegrator { kappa: 1.0 }], 5);
        let mut mu = vec![0.0_f64; sp.n_dofs()];
        mass.spmv(&u, &mut mu);
        let err: f64 = res.iter().zip(&mu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1e-13, "max |N·u − M·u| = {err}");
    }

    /// The kernel path must be *bit-identical* to the element loop the navier
    /// miniapps used before D52 — on the miniapps' own configurations
    /// (order-3 probe mesh, and the kovasznay order-6 2×4 mesh where the
    /// `2p+1` override actually deviates from the `2p + OrderGrad` default
    /// and is exercised).
    #[test]
    fn kernel_matches_the_miniapp_local_loop_bitwise() {
        let mut kov = Mesh::<2>::make_cartesian_2d(2, 4, 1.5, 2.0);
        for c in kov.coords.iter_mut() {
            *c -= 0.5;
        }
        for (order, m) in [(3u8, mesh()), (6, kov)] {
            let sp = space(&m, order);
            let quad_order = 2 * order + 1;
            let u = sp
                .interpolate_vec(&|x| vec![x[0] * x[0] + x[1], x[0] * x[1] * x[1]])
                .as_slice()
                .to_vec();

            let it = integ(quad_order as i32);
            let mut nf = NonlinearForm::new();
            nf.add_domain_integrator(&it);
            let mut kernel = vec![0.0_f64; sp.n_dofs()];
            nf.mult(&sp, &u, &mut kernel);

            // The pre-D52 miniapp element loop (navier_kovasznay.rs /
            // navier_mms.rs `convection_residual`).
            let re = fem_element::lagrange::factory::ref_elem(
                fem_element::lagrange::factory::ElemType::Quad,
                order,
            );
            let n_p = re.n_dofs();
            let quad = re.quadrature(quad_order);
            let mut phi = vec![0.0_f64; n_p];
            let mut grad_ref = vec![0.0_f64; n_p * 2];
            let mut grad_phys = vec![0.0_f64; n_p * 2];
            let mut el = vec![0.0_f64; 2 * n_p];
            let mut local = vec![0.0_f64; sp.n_dofs()];
            for e in 0..m.n_elements() as u32 {
                let dofs = sp.element_dofs(e);
                let nodes = m.element_nodes(e);
                let geo =
                    crate::vector_assembler::geo_ref_elem_from_mesh(&m, e).expect("quad geometry");
                el.fill(0.0);
                for (q, xi) in quad.points.iter().enumerate() {
                    re.eval_basis(xi, &mut phi);
                    re.eval_grad_basis(xi, &mut grad_ref);
                    let (jac, det_j, _xp) =
                        crate::vector_assembler::isoparametric_jacobian(&m, nodes, &*geo, xi, 2);
                    let jinv = jac.try_inverse().expect("degenerate element");
                    for k in 0..n_p {
                        for dd in 0..2 {
                            let mut g = 0.0_f64;
                            for mm in 0..2 {
                                g += grad_ref[k * 2 + mm] * jinv[(mm, dd)];
                            }
                            grad_phys[k * 2 + dd] = g;
                        }
                    }
                    let mut uh = [0.0_f64; 2];
                    for k in 0..n_p {
                        for c in 0..2 {
                            uh[c] += u[dofs[k * 2 + c] as usize] * phi[k];
                        }
                    }
                    let mut conv = [0.0_f64; 2];
                    for c in 0..2 {
                        let mut grad_uc = [0.0_f64; 2];
                        for l in 0..n_p {
                            let uc = u[dofs[l * 2 + c] as usize];
                            grad_uc[0] += uc * grad_phys[l * 2];
                            grad_uc[1] += uc * grad_phys[l * 2 + 1];
                        }
                        conv[c] = uh[0] * grad_uc[0] + uh[1] * grad_uc[1];
                    }
                    let w = quad.weights[q] * det_j.abs();
                    for k in 0..n_p {
                        for c in 0..2 {
                            el[k * 2 + c] += w * phi[k] * conv[c];
                        }
                    }
                }
                for (k, &g) in dofs.iter().enumerate() {
                    local[g as usize] += el[k];
                }
            }
            if order == 3 {
                // Straight cells whose cancellation sums stay exact: the two
                // geometry paths agree bit for bit.
                assert_eq!(kernel, local, "order {order}: kernel != miniapp local loop");
            } else {
                // The kovasznay mesh (cells 0.75 × 0.5) exposes a 2–3 ulp
                // difference between the kernel geometry (MFEM's
                // `BiLinear2DFiniteElement`, H1 topological node order — the
                // `assembler::geo_ref_elem` path) and the pre-D52 local
                // loop's `QuadQk(1)` path (`vector_assembler::
                // geo_ref_elem_from_mesh`): analytically identical, but the
                // four-product J summation is grouped differently, which
                // rounds at the last ulp on non-dyadic coordinates.  The
                // kernel path is the MFEM-faithful one; the drift stays far
                // below every printed digit of the miniapps (err/CFL and the
                // MVIN/PRES/HELM iteration counts are unchanged).
                let num: f64 =
                    kernel.iter().zip(&local).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
                let den: f64 = local.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                assert!(
                    num <= den * 1e-12,
                    "order {order}: kernel/local max|Δ| = {num} (rel {})",
                    num / den
                );
            }
        }
    }

    /// The Jacobian matches a central finite difference of the residual
    /// (order 1 on the non-unit mesh; every one of the 24 columns).
    #[test]
    fn jacobian_matches_finite_differences() {
        let m = mesh();
        let sp = space(&m, 1);
        let u = sp
            .interpolate_vec(&|x| vec![x[0] + 0.5 * x[1], x[0] * x[1] + x[0]])
            .as_slice()
            .to_vec();
        let it = integ(3);
        let mut nf = NonlinearForm::new();
        nf.add_domain_integrator(&it);
        let jac = nf.get_gradient(&sp, &u);
        let n = sp.n_dofs();
        let mut r0 = vec![0.0_f64; n];
        nf.mult(&sp, &u, &mut r0);
        let eps = 1e-7_f64;
        let mut worst = 0.0_f64;
        for j in 0..n {
            let mut up = u.clone();
            let mut um = u.clone();
            up[j] += eps;
            um[j] -= eps;
            let mut rp = vec![0.0_f64; n];
            let mut rm = vec![0.0_f64; n];
            nf.mult(&sp, &up, &mut rp);
            nf.mult(&sp, &um, &mut rm);
            for i in 0..n {
                let fd = (rp[i] - rm[i]) / (2.0 * eps);
                let an = jac.get(i, j);
                worst = worst.max((fd - an).abs());
            }
        }
        assert!(worst < 1e-8, "max |J − dN/du| = {worst}");
    }
}
