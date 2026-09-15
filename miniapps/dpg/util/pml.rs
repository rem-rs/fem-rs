//! 1:1 port of MFEM's DPG PML utilities `miniapps/dpg/util/pml.{hpp,cpp}`
//! (MFEM 4.10): the [`CartesianPML`] stretched-coordinate region and the
//! `PmlCoefficient` / `PmlMatrixCoefficient` coefficient wrappers.
//!
//! Included in the `pmaxwell` miniapp with
//! `#[path = "util/pml.rs"] mod pml;` — a miniapp-layer util, exactly like the
//! C++ (this is *not* part of the fem-rs core crates).
//!
//! # Structure (C++ → Rust)
//!
//! | MFEM `pml.hpp`                                  | here                                    |
//! |-------------------------------------------------|-----------------------------------------|
//! | `CartesianPML` (+ `SetBoundaries`)               | [`CartesianPML::new`]                   |
//! | `CartesianPML::SetAttributes(mesh, …)`           | [`CartesianPML::mark_elements`]         |
//! | `SetOmega` / `SetEpsilonAndMu`                   | [`CartesianPML::set_omega`] / `…mu`     |
//! | `CartesianPML::StretchFunction`                  | [`CartesianPML::stretch_function`]      |
//! | `PmlCoefficient(f, pml)`                         | [`pml_scalar`]                          |
//! | `PmlMatrixCoefficient(dim, f, pml)`              | [`pml_matrix`]                          |
//! | `ConstantCoefficient` / `MatrixConstant…`        | [`const_scalar`] / [`const_matrix`]     |
//! | `ProductCoefficient(s, f)`                       | [`product`]                             |
//! | `ScalarMatrixProductCoefficient(s, M)`           | [`scalar_matrix_product`]               |
//! | `MatrixProductCoefficient(M, rot)`               | [`matrix_right_product_2d`]             |
//! | (transpose of the above, weak-divergence pairing) | [`matrix_right_product_2d_t`]          |
//! | `RestrictedCoefficient(c, attr)`                 | [`restricted_scalar`]                   |
//! | `MatrixRestrictedCoefficient(M, attr)`           | [`restricted_matrix`]                   |
//! | `detJ_r_function` / `detJ_i_function` / `abs_detJ_2_function`        | [`CartesianPML::det_j_r`] / `det_j_i` / `abs_det_j_2` |
//! | `detJ_Jt_J_inv_r_function` / `…_i_…` / `abs_detJ_Jt_J_inv_2_function` | [`CartesianPML::det_j_jt_j_inv_r`] / `…i` / `…2` |
//!
//! The acoustic `J^T J / |J|` stretching functions of `pml.cpp`
//! (`Jt_J_detJinv_*`) are *not* used by `pmaxwell` (they belong to the
//! acoustics PML) and are left unported until a PML acoustics miniapp needs
//! them.
//!
//! # Attribute restriction
//!
//! MFEM's `SetAttributes` stamps attribute 1/2 on the elements and the
//! `Restricted(Coefficient, attr)` wrappers gate by `T.Attribute`; fem-rs
//! carries the same information as the per-element in-PML flag vector
//! returned by [`CartesianPML::mark_elements`] (consumed through
//! [`fem_assembly::dpg::dpg_integrators::VolCtx::elem`] inside the gates).
//! `SetBoundaries` runs on the *global* mesh (the C++ constructor also runs
//! before the `ParMesh` split; its `MPI_Allreduce` box union is therefore
//! already baked into the serial result).

use std::sync::Arc;

use fem_assembly::dpg::dpg_integrators::{DpgSpatialMatrix, DpgSpatialScalar, VolCtx};
use fem_mesh::topology::MeshTopology;

/// Which attribute-marker array a [`restricted_*`] wrapper gates by — MFEM
/// `attr` (non-PML elements) vs `attrPML` (PML elements), both built by
/// `CartesianPML::SetAttributes`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PmlRegion {
    /// MFEM `attrPML` (`(*attrPML)[1] = 1`): gate *on* for PML elements.
    Pml,
    /// MFEM `attr` (`(*attr)[0] = 1`): gate *on* for non-PML elements.
    NonPml,
}

/// A simple Cartesian PML region — MFEM `CartesianPML`
/// (`miniapps/dpg/util/pml.cpp`), const-generic over the spatial dimension.
#[derive(Clone)]
pub struct CartesianPML<const D: usize> {
    /// `(dim, 2)` PML region length in each direction (C++ `length`).
    length: [[f64; 2]; D],
    /// `(dim, 2)` computational domain boundary (C++ `comp_dom_bdr`).
    comp_dom_bdr: [[f64; 2]; D],
    /// `(dim, 2)` domain boundary (C++ `dom_bdr`).
    dom_bdr: [[f64; 2]; D],
    /// C++ `omega` (default 0 until [`Self::set_omega`]).
    omega: f64,
    /// C++ `epsilon` (default Maxwell value 1).
    epsilon: f64,
    /// C++ `mu` (default Maxwell value 1).
    mu: f64,
}

impl<const D: usize> CartesianPML<D> {
    /// Constructor + `SetBoundaries` — the domain boundary box is the
    /// min/max over all *boundary-face* vertices (`GetBdrElementVertices`),
    /// and the computational domain shrinks it by the PML lengths.
    pub fn new(mesh: &fem_mesh::Mesh<D>, length: [[f64; 2]; D]) -> Self {
        let mut dom_bdr = [[0.0_f64; 2]; D];
        for b in dom_bdr.iter_mut() {
            b[0] = f64::INFINITY;
            b[1] = f64::NEG_INFINITY;
        }
        for f in 0..mesh.n_faces() as u32 {
            for &v in mesh.bface_nodes(f) {
                let p = mesh.coords_of(v);
                for (d, b) in dom_bdr.iter_mut().enumerate().take(D) {
                    b[0] = b[0].min(p[d]);
                    b[1] = b[1].max(p[d]);
                }
            }
        }
        let mut comp_dom_bdr = [[0.0_f64; 2]; D];
        for d in 0..D {
            comp_dom_bdr[d][0] = dom_bdr[d][0] + length[d][0];
            comp_dom_bdr[d][1] = dom_bdr[d][1] - length[d][1];
        }
        Self { length, comp_dom_bdr, dom_bdr, omega: 0.0, epsilon: 1.0, mu: 1.0 }
    }

    /// Spatial dimension (C++ `dim`).
    pub fn dim(&self) -> usize {
        D
    }

    /// Computational domain boundary (C++ `GetCompDomainBdr`).
    pub fn comp_domain_bdr(&self) -> &[[f64; 2]; D] {
        &self.comp_dom_bdr
    }

    /// Domain boundary (C++ `GetDomainBdr`).
    pub fn domain_bdr(&self) -> &[[f64; 2]; D] {
        &self.dom_bdr
    }

    /// `SetOmega(omega)`.
    pub fn set_omega(&mut self, omega: f64) {
        self.omega = omega;
    }

    /// `SetEpsilonAndMu(epsilon, mu)`.
    pub fn set_epsilon_and_mu(&mut self, epsilon: f64, mu: f64) {
        self.epsilon = epsilon;
        self.mu = mu;
    }

    /// `SetAttributes(mesh, …)` marker pass: for every element, `true` iff
    /// any vertex lies outside the computational-domain box (C++ marks those
    /// elements with attribute 2 and returns `elems[i] = 0`).
    pub fn mark_elements(&self, mesh: &fem_mesh::Mesh<D>) -> Vec<bool> {
        let mut in_pml = vec![false; mesh.n_elements()];
        for (e, flagged) in in_pml.iter_mut().enumerate() {
            for v in mesh.element_nodes(e as u32) {
                let p = mesh.node_coords(*v);
                for d in 0..D {
                    if p[d] > self.comp_dom_bdr[d][1] || p[d] < self.comp_dom_bdr[d][0] {
                        *flagged = true;
                        break;
                    }
                }
                if *flagged {
                    break;
                }
            }
        }
        in_pml
    }

    /// PML complex stretching function (C++ `StretchFunction`): writes the
    /// per-direction stretch `1 + i·σ_i(x)` into `dxs`.  Constants `n = 2`,
    /// `c = 5`; `k = ω·sqrt(ε·μ)`; σ activates outside the computational
    /// domain boundary with the `(d − δ)/δ` power profile of
    /// <https://doi.org/10.1006/jcph.1994.1159>.
    pub fn stretch_function(&self, x: &[f64], dxs: &mut [(f64, f64)]) {
        let n = 2.0_f64;
        let c = 5.0_f64;
        let k = self.omega * (self.epsilon * self.mu).sqrt();
        for (i, dx) in dxs.iter_mut().enumerate().take(D) {
            *dx = (1.0, 0.0);
            if x[i] >= self.comp_dom_bdr[i][1] {
                let coeff = n * c / k / self.length[i][1].powf(n);
                let s = (x[i] - self.comp_dom_bdr[i][1]).powf(n - 1.0).abs();
                *dx = (1.0, coeff * s);
            }
            if x[i] <= self.comp_dom_bdr[i][0] {
                let coeff = n * c / k / self.length[i][0].powf(n);
                let s = (x[i] - self.comp_dom_bdr[i][0]).powf(n - 1.0).abs();
                *dx = (1.0, coeff * s);
            }
        }
    }

    /// `detJ` of the stretching map — C++ `detJ_r_function` (real part).
    pub fn det_j_r(&self, x: &[f64]) -> f64 {
        self.stretch_det(x).0
    }

    /// Imaginary part of `detJ` — C++ `detJ_i_function`.
    pub fn det_j_i(&self, x: &[f64]) -> f64 {
        self.stretch_det(x).1
    }

    /// `|detJ|²` — C++ `abs_detJ_2_function`.
    pub fn abs_det_j_2(&self, x: &[f64]) -> f64 {
        let (re, im) = self.stretch_det(x);
        im * im + re * re
    }

    /// `detJ·(JᵀJ)⁻¹` real part (diagonal) — C++ `detJ_Jt_J_inv_r_function`;
    /// writes the row-major `dim × dim` matrix (zero off-diagonals).
    pub fn det_j_jt_j_inv_r(&self, x: &[f64], m: &mut [f64]) {
        let (det, dxs) = self.stretch_det_all(x);
        zero(m, D * D);
        for (i, dx) in dxs.iter().enumerate().take(D) {
            let a = c_div(det, c_pow2(*dx));
            m[i * D + i] = a.0;
        }
    }

    /// `detJ·(JᵀJ)⁻¹` imaginary part — C++ `detJ_Jt_J_inv_i_function`.
    pub fn det_j_jt_j_inv_i(&self, x: &[f64], m: &mut [f64]) {
        let (det, dxs) = self.stretch_det_all(x);
        zero(m, D * D);
        for (i, dx) in dxs.iter().enumerate().take(D) {
            let a = c_div(det, c_pow2(*dx));
            m[i * D + i] = a.1;
        }
    }

    /// `|detJ·(JᵀJ)⁻¹|²` (diagonal) — C++ `abs_detJ_Jt_J_inv_2_function`.
    pub fn abs_det_j_jt_j_inv_2(&self, x: &[f64], m: &mut [f64]) {
        let (det, dxs) = self.stretch_det_all(x);
        zero(m, D * D);
        for (i, dx) in dxs.iter().enumerate().take(D) {
            let a = c_div(det, c_pow2(*dx));
            m[i * D + i] = a.0 * a.0 + a.1 * a.1;
        }
    }

    /// `StretchFunction` + `∏ dxs[i]` (the C++ preamble shared by every
    /// coefficient function).
    fn stretch_det(&self, x: &[f64]) -> (f64, f64) {
        let mut dxs = [(1.0, 0.0); D];
        self.stretch_function(x, &mut dxs);
        let mut det = (1.0, 0.0);
        for dx in dxs.iter().take(D) {
            det = c_mul(det, *dx);
        }
        det
    }

    /// [`Self::stretch_det`] keeping the individual stretches.
    fn stretch_det_all(&self, x: &[f64]) -> ((f64, f64), [(f64, f64); D]) {
        let mut dxs = [(1.0, 0.0); D];
        self.stretch_function(x, &mut dxs);
        let mut det = (1.0, 0.0);
        for dx in dxs.iter().take(D) {
            det = c_mul(det, *dx);
        }
        (det, dxs)
    }
}

/// `(re·re − im·im, re·im + im·re)` — `std::complex` product.
fn c_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `z²` (C++ `pow(z, real_t(2))`).
fn c_pow2(z: (f64, f64)) -> (f64, f64) {
    c_mul(z, z)
}

/// `a / b` — `std::complex` division (`(a·conj(b))/|b|²`).
fn c_div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn zero(m: &mut [f64], n: usize) {
    for v in m.iter_mut().take(n) {
        *v = 0.0;
    }
}

// ── coefficient combinators (MFEM Coefficient wrappers of pml.hpp) ───────────

/// MFEM `PmlCoefficient(f, pml)` — a spatial scalar coefficient built from a
/// stretch function of `(pml, x)`.
pub fn pml_scalar<const D: usize, F>(f: F, pml: Arc<CartesianPML<D>>) -> DpgSpatialScalar
where
    F: Fn(&CartesianPML<D>, &[f64]) -> f64 + Send + Sync + 'static,
{
    Box::new(move |ctx: &VolCtx| f(&pml, &ctx.x))
}

/// MFEM `PmlMatrixCoefficient(dim, f, pml)` — a spatial matrix coefficient
/// built from a stretch function writing the row-major `dim × dim` matrix.
pub fn pml_matrix<const D: usize, F>(f: F, pml: Arc<CartesianPML<D>>) -> DpgSpatialMatrix
where
    F: Fn(&CartesianPML<D>, &[f64], &mut [f64]) + Send + Sync + 'static,
{
    Box::new(move |ctx: &VolCtx, out: &mut [f64]| f(&pml, &ctx.x, out))
}

/// MFEM `ConstantCoefficient(c)`.
pub fn const_scalar(c: f64) -> DpgSpatialScalar {
    Box::new(move |_| c)
}

/// MFEM `MatrixConstantCoefficient(m)` (row-major rows).
pub fn const_matrix(m: Vec<Vec<f64>>) -> DpgSpatialMatrix {
    let flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    Box::new(move |_, out| out[..flat.len()].copy_from_slice(&flat))
}

/// MFEM `ProductCoefficient(a, b)` with a constant leading factor — the
/// `ProductCoefficient(const_s, pml_f)` instantiations of `pmaxwell.cpp`.
pub fn product(s: f64, b: DpgSpatialScalar) -> DpgSpatialScalar {
    Box::new(move |ctx| s * b(ctx))
}

/// MFEM `ScalarMatrixProductCoefficient(s, M)` with a constant scalar.
pub fn scalar_matrix_product(s: f64, m: DpgSpatialMatrix) -> DpgSpatialMatrix {
    Box::new(move |ctx, out| {
        m(ctx, out);
        for v in out.iter_mut() {
            *v *= s;
        }
    })
}

/// MFEM `MatrixProductCoefficient(M, rot)` — right-multiply by the constant
/// 2-D rotation matrix `A = [0 1; −1 0]` (only used with `dim == 2`).
pub fn matrix_right_product_2d(m: DpgSpatialMatrix, r: [[f64; 2]; 2]) -> DpgSpatialMatrix {
    Box::new(move |ctx, out| {
        let mut a = [0.0_f64; 4];
        m(ctx, &mut a);
        for i in 0..2 {
            for j in 0..2 {
                out[i * 2 + j] = a[i * 2] * r[0][j] + a[i * 2 + 1] * r[1][j];
            }
        }
    })
}

/// Transpose of [`matrix_right_product_2d`]: `(M·rot)ᵀ`, the coefficient form
/// the [`DpgMixedVectorWeakDivergenceSpatialIntegrator`] pairing needs (MFEM
/// wraps the same matrix in a `TransposeIntegrator`).
pub fn matrix_right_product_2d_t(m: DpgSpatialMatrix, r: [[f64; 2]; 2]) -> DpgSpatialMatrix {
    Box::new(move |ctx, out| {
        let mut a = [0.0_f64; 4];
        m(ctx, &mut a);
        for i in 0..2 {
            for j in 0..2 {
                out[i * 2 + j] = a[j * 2] * r[0][i] + a[j * 2 + 1] * r[1][i];
            }
        }
    })
}

/// MFEM `RestrictedCoefficient(c, attr)` — gate by the per-element in-PML
/// flags (`VolCtx::elem`); zero outside the marked region.
pub fn restricted_scalar(
    c: DpgSpatialScalar,
    region: PmlRegion,
    in_pml: Arc<Vec<bool>>,
) -> DpgSpatialScalar {
    let want = region == PmlRegion::Pml;
    Box::new(move |ctx| {
        if in_pml[ctx.elem as usize] == want {
            c(ctx)
        } else {
            0.0
        }
    })
}

/// MFEM `MatrixRestrictedCoefficient(M, attr)`.
pub fn restricted_matrix(
    m: DpgSpatialMatrix,
    region: PmlRegion,
    in_pml: Arc<Vec<bool>>,
) -> DpgSpatialMatrix {
    let want = region == PmlRegion::Pml;
    Box::new(move |ctx, out| {
        if in_pml[ctx.elem as usize] == want {
            m(ctx, out);
        } else {
            zero(out, out.len());
        }
    })
}
