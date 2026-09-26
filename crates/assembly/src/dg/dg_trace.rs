//! MFEM's `DGTraceIntegrator` family — the face terms of the DG advection
//! (hyperbolic) form, transcribed 1:1 from `fem/bilininteg.cpp` of MFEM 4.10.
//!
//! | fem-rs | MFEM 4.10 |
//! |---|---|
//! | [`DgTraceIntegrator`] | `DGTraceIntegrator` (`fem/bilininteg.cpp:3480-3610`) |
//! | [`DgNonconservativeTraceIntegrator`] | `NonconservativeDGTraceIntegrator` (`fem/bilininteg.hpp:3627`) |
//!
//! `NonconservativeDGTraceIntegrator(ρ, u, a, b)` is
//! `TransposeIntegrator(new DGTraceIntegrator(ρ, u, −a, b))`; the one-argument
//! convenience constructor is `NonconservativeDGTraceIntegrator(u, a) =
//! TransposeIntegrator(new DGTraceIntegrator(u, −a, 0.5·a))`.  ex9/ex41 use
//! `α = −1` throughout (the `alpha` command-line option of `examples/ex9.cpp`).
//!
//! # Element matrix blocks (MFEM's `elmat`, per face quadrature point)
//!
//! `DGTraceIntegrator` with `alpha_in`, `beta_in`, the **unnormalised** face
//! normal `nor = CalcOrtho(Trans.Jacobian())` (so `|nor|` carries the face
//! measure), `un = vu·nor` evaluated at `Elem1`'s physical point, the bare face
//! quadrature weight `ipw` (`Σ ipw = 1` on a segment, `1/2` on a triangle), and
//! `shape` = `CalcPhysShape` (= `CalcShape` for the nodal L2/H1 bases here):
//!
//! ```text
//! a = 0.5·alpha_in·un ;  b = beta_in·|un|
//! w+ = ipw·(a+b) ; w- = ipw·(b−a)
//! A_11 += w+·φ₁ᵢφ₁ⱼ      A_21 −= w+·φ₂ᵢφ₁ⱼ
//! A_22 += w-·φ₂ᵢφ₂ⱼ      A_12 −= w-·φ₁ᵢφ₂ⱼ
//! ```
//!
//! and `NonconservativeDGTraceIntegrator` scatters the **transpose** of that
//! element matrix (MFEM `TransposeIntegrator::AssembleFaceMatrix`,
//! `fem/bilininteg.cpp:284`), i.e. with `w+`, `w-` as above:
//!
//! ```text
//! K_11 += w+·φ₁ᵢφ₁ⱼ      K_12 −= w+·φ₁ᵢφ₂ⱼ
//! K_22 += w-·φ₂ᵢφ₂ⱼ      K_21 −= w-·φ₂ᵢφ₁ⱼ
//! ```
//!
//! For `alpha = −1` (ex9/ex41) the two weights are one-sided: `w+ = ipw·un` when
//! `un < 0` and `0` otherwise, `w- = −ipw·un` when `un > 0` and `0` otherwise —
//! so exactly two of the four blocks are non-zero at an inflow/outflow point.
//! On a boundary face (`ndofs2 == 0`) only `w+` survives, with `ipw` *not*
//! halved: MFEM's interior branch uses `w = ip.weight/2` and its boundary
//! branch `w = ip.weight`.
//!
//! # Quadrature rule
//!
//! MFEM's default (`fem/bilininteg.cpp:3509-3523`) is
//! `min(OrderW1, OrderW2) + 2·max(order₁, order₂)`, `+1` for `Pk` spaces, on
//! the **face** geometry.  `assemble_dg_interior_faces` takes the rule order
//! from its caller, so pass that value to be 1:1 with MFEM — for order-`p` L2
//! elements on a straight (order-1) mesh that is `2p+1`; on the D805 curved
//! fixture (geometry order 3, `OrderW = min(5,5) = 5`, `p = 1`) it is `7`.
//!
//! # What this replaced
//!
//! D805-2: `DGAdvectionIntegrator`'s old face impl built the *conservative*
//! upwind form `−∫⟦v⟧F̂` — inflow contributions in blocks `(1,2)`/`(2,2)` and
//! outflow in `(1,1)`/`(2,1)`, a different (mirrored) sparsity from MFEM's and
//! therefore **not entry-by-entry equal to any MFEM integrator** (48/256 entries
//! wrong on the D805 fixture), while its comment claimed
//! `NonconservativeDGTraceIntegrator` equivalence.

use crate::postproc::coefficient::{CoeffCtx, VectorCoeff};

use super::dg_advection::{DgFaceIntegrator, DgFaceQpData};

/// MFEM `DGTraceIntegrator`: the raw (non-transposed) upwind trace form.
///
/// `un = velocity·nor` uses the **unnormalised** `nor`, and the contribution is
/// scaled by MFEM's bare `ip.weight`, exactly as in
/// `DGTraceIntegrator::AssembleFaceMatrix`.
pub struct DgTraceIntegrator<V: VectorCoeff> {
    /// Advection/convection velocity field `u`, evaluated on `Elem1`.
    pub velocity: V,
    /// MFEM `alpha` — scales the (signed) centred part `0.5·alpha·un`.
    pub alpha: f64,
    /// MFEM `beta` — scales the (positive) upwinding part `beta·|un|`.
    pub beta: f64,
}

/// MFEM `NonconservativeDGTraceIntegrator`: the **transpose** of
/// [`DgTraceIntegrator::new`]`(velocity, −a, b)`.
///
/// Build with [`DgNonconservativeTraceIntegrator::new`] to get MFEM's
/// one-argument constructor (`b = 0.5·a`).
pub struct DgNonconservativeTraceIntegrator<V: VectorCoeff> {
    /// Advection/convection velocity field `u`, evaluated on `Elem1`.
    pub velocity: V,
    /// MFEM's nonconservative coefficient `a` (ex9/ex41 pass `−1`).
    pub a: f64,
    /// MFEM's second coefficient `b`; `0.5·a` for the one-argument form.
    pub b: f64,
}

impl<V: VectorCoeff> DgTraceIntegrator<V> {
    /// MFEM `DGTraceIntegrator(u, alpha, beta)`.
    pub fn new(velocity: V, alpha: f64, beta: f64) -> Self {
        Self { velocity, alpha, beta }
    }
}

impl<V: VectorCoeff> DgNonconservativeTraceIntegrator<V> {
    /// MFEM `NonconservativeDGTraceIntegrator(u, a)` =
    /// `TransposeIntegrator(DGTraceIntegrator(u, −a, 0.5·a))`.
    pub fn new(velocity: V, a: f64) -> Self {
        Self { velocity, a, b: 0.5 * a }
    }
}

/// MFEM `DGTraceIntegrator`'s two per-point weights from the raw inputs:
/// `a = 0.5·alpha·un`, `b = beta·|un|`,
/// `(w_plus, w_minus) = (ipw·(a+b), ipw·(b−a))`, with `un = vu·nor` using the
/// **unnormalised** `nor` and `ipw` the **bare** face quadrature weight.
///
/// This is the single arithmetic source of the whole family: the interior four
/// blocks ([`add_dgtrace_blocks`]), the boundary one-element block and the
/// periodic seam all consume it, so none of them can drift from MFEM.
pub(crate) fn dgtrace_weights(ip_weight: f64, un: f64, alpha: f64, beta: f64) -> (f64, f64) {
    let a = 0.5 * alpha * un;
    let b = beta * un.abs();
    (ip_weight * (a + b), ip_weight * (b - a))
}

/// `un = vu·nor` at a face point, with the velocity evaluated on `Elem1`
/// (MFEM `u->Eval(vu, *Trans.Elem1, eip1)`) and the **unnormalised** `nor`.
pub(crate) fn face_un(
    velocity: &impl VectorCoeff,
    qp: &DgFaceQpData<'_>,
) -> f64 {
    let dim = qp.dim;
    let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_l, 0, None, None);
    let mut vu = [0.0_f64; 3];
    velocity.eval(&ctx, &mut vu[..dim]);
    (0..dim).map(|i| vu[i] * qp.unor[i]).sum()
}

/// The shared per-quadrature-point block accumulation of the family (see the
/// module docs for the two block layouts).
///
/// `transpose == false` fills MFEM's `DGTraceIntegrator` blocks; `true` fills
/// the transposed (`NonconservativeDGTraceIntegrator`) ones.
fn add_dgtrace_blocks(
    velocity: &impl VectorCoeff,
    alpha: f64,
    beta: f64,
    transpose: bool,
    qp: &DgFaceQpData<'_>,
    k_ll: &mut [f64],
    k_lr: &mut [f64],
    k_rl: &mut [f64],
    k_rr: &mut [f64],
) {
    let n_l = qp.n_dofs_l;
    let n_r = qp.n_dofs_r;

    let un = face_un(velocity, qp);
    let (w_plus, w_minus) = dgtrace_weights(qp.ip_weight, un, alpha, beta);

    if w_plus != 0.0 {
        for i in 0..n_l {
            for j in 0..n_l {
                k_ll[i * n_l + j] += w_plus * qp.phi_l[i] * qp.phi_l[j];
            }
        }
    }
    if w_minus != 0.0 {
        for i in 0..n_r {
            for j in 0..n_r {
                k_rr[i * n_r + j] += w_minus * qp.phi_r[i] * qp.phi_r[j];
            }
        }
    }
    if transpose {
        // K = Aᵀ block-wise: K_12 = A_21ᵀ, K_21 = A_12ᵀ.
        if w_plus != 0.0 {
            for i in 0..n_l {
                for j in 0..n_r {
                    k_lr[i * n_r + j] -= w_plus * qp.phi_l[i] * qp.phi_r[j];
                }
            }
        }
        if w_minus != 0.0 {
            for i in 0..n_r {
                for j in 0..n_l {
                    k_rl[i * n_l + j] -= w_minus * qp.phi_r[i] * qp.phi_l[j];
                }
            }
        }
    } else {
        if w_plus != 0.0 {
            for i in 0..n_r {
                for j in 0..n_l {
                    k_rl[i * n_l + j] -= w_plus * qp.phi_r[i] * qp.phi_l[j];
                }
            }
        }
        if w_minus != 0.0 {
            for i in 0..n_l {
                for j in 0..n_r {
                    k_lr[i * n_r + j] -= w_minus * qp.phi_l[i] * qp.phi_r[j];
                }
            }
        }
    }
}

impl<V: VectorCoeff> DgFaceIntegrator for DgTraceIntegrator<V> {
    fn add_to_face_matrix(
        &self,
        qp: &DgFaceQpData<'_>,
        k_ll: &mut [f64],
        k_lr: &mut [f64],
        k_rl: &mut [f64],
        k_rr: &mut [f64],
    ) {
        add_dgtrace_blocks(
            &self.velocity, self.alpha, self.beta, false, qp, k_ll, k_lr, k_rl, k_rr,
        );
    }
}

impl<V: VectorCoeff> DgFaceIntegrator for DgNonconservativeTraceIntegrator<V> {
    fn add_to_face_matrix(
        &self,
        qp: &DgFaceQpData<'_>,
        k_ll: &mut [f64],
        k_lr: &mut [f64],
        k_rl: &mut [f64],
        k_rr: &mut [f64],
    ) {
        add_nonconservative_blocks(&self.velocity, self.a, self.b, qp, k_ll, k_lr, k_rl, k_rr);
    }
}

/// MFEM `NonconservativeDGTraceIntegrator(u, a, b)` on an interior face: the
/// transpose of `DGTraceIntegrator(u, −a, b)`.
///
/// `pub(crate)` because `DGAdvectionIntegrator`'s own [`DgFaceIntegrator`] impl
/// (`dg_advection.rs`) routes through it, so both entry points share one
/// implementation of the block layout.
pub(crate) fn add_nonconservative_blocks(
    velocity: &impl VectorCoeff,
    a: f64,
    b: f64,
    qp: &DgFaceQpData<'_>,
    k_ll: &mut [f64],
    k_lr: &mut [f64],
    k_rl: &mut [f64],
    k_rr: &mut [f64],
) {
    add_dgtrace_blocks(velocity, -a, b, true, qp, k_ll, k_lr, k_rl, k_rr);
}

/// Scatter MFEM's one-element face block `w·φᵢφⱼ` (the `ndofs2 == 0` arm of
/// `DGTraceIntegrator::AssembleFaceMatrix`).
pub(crate) fn add_scalar_face_block(w: f64, phi: &[f64], k_ll: &mut [f64], n: usize) {
    if w == 0.0 {
        return;
    }
    for i in 0..n {
        for j in 0..n {
            k_ll[i * n + j] += w * phi[i] * phi[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_match_mfem_formulas() {
        // NonconservativeDGTraceIntegrator(u, a = -1) has the inner integrator
        // DGTraceIntegrator(u, alpha = -a = 1, beta = 0.5a = -0.5):
        //   a_in = 0.5*alpha*un = un/2 ; b_in = beta*|un| = -|un|/2
        // un > 0 -> w+ = ipw(un/2 - un/2) = 0, w- = ipw(-un/2 - un/2) = -ipw*un
        let (wp, wm) = dgtrace_weights(1.0, 3.0, 1.0, -0.5);
        assert_eq!(wp, 0.0);
        assert_eq!(wm, -3.0);
        // un < 0 -> w+ = ipw*un, w- = 0
        let (wp, wm) = dgtrace_weights(1.0, -3.0, 1.0, -0.5);
        assert_eq!(wp, -3.0);
        assert_eq!(wm, 0.0);
        // the raw DGTraceIntegrator(vel, -1, 0.5) of the D805 probe.
        let (wp, wm) = dgtrace_weights(1.0, -3.0, -1.0, 0.5);
        assert_eq!((wp, wm), (3.0, 0.0));
        let (wp, wm) = dgtrace_weights(1.0, 3.0, -1.0, 0.5);
        assert_eq!((wp, wm), (0.0, 3.0));
    }

    #[test]
    fn boundary_block_is_mfem_scatter() {
        let phi = [0.25, 0.75];
        let mut k = [0.0; 4];
        add_scalar_face_block(2.0, &phi, &mut k, 2);
        assert_eq!(
            k,
            [2.0 * 0.25 * 0.25, 2.0 * 0.25 * 0.75, 2.0 * 0.75 * 0.25, 2.0 * 0.75 * 0.75]
        );
        let mut k2 = [1.0; 4];
        add_scalar_face_block(0.0, &phi, &mut k2, 2);
        assert_eq!(k2, [1.0; 4]);
    }

    #[test]
    fn nonconservative_one_arg_matches_mfems_ctor() {
        let nc = DgNonconservativeTraceIntegrator::new(
            crate::postproc::coefficient::ConstantVectorCoeff(vec![1.0, 0.0]), -1.0);
        assert_eq!((nc.a, nc.b), (-1.0, -0.5));
        // inner DGTraceIntegrator(alpha, beta) = (-a, b)
        assert_eq!((-nc.a, nc.b), (1.0, -0.5));
    }
}
