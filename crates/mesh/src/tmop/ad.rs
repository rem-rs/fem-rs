//! Native forward AD for the TMOP metrics whose MFEM 4.10 implementations are
//! AD-based (`fem/tmop.cpp`, section "AD related definitions": metrics 085,
//! 098, 342 and the A-metrics 011/014/036/050/051/107).
//!
//! Ported 1:1 from MFEM:
//! - `linalg/dual.hpp`: `future::dual<real_t, real_t>` ([`Ad1`]) and
//!   `future::dual<AD1Type, AD1Type>` ([`Ad2`]) with the same operator and
//!   `sqrt` / `pow(dual, real_t)` derivative formulas (operation order kept).
//! - `fem/tmop.cpp`: `ADGrad` (first derivatives, one seeded entry at a time),
//!   `ADHessian` (forward-forward second derivatives) and
//!   `TMOP_QualityMetric::DefaultAssembleH` (Hessian-tensor contraction with
//!   the basis derivative matrix).
//!
//! The scalar trait [`AdScalar`] is implemented for `f64` (plain metric
//! evaluation = MFEM's `EvalWMatrixForm`), [`Ad1`] (`ADGrad`) and [`Ad2`]
//! (`ADHessian`), mirroring the C++ `template <typename type>` metric
//! functions.

/// MFEM `future::dual<real_t, real_t>`: value + one derivative slot.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Ad1 {
    pub v: f64,
    pub g: f64,
}

/// MFEM `future::dual<AD1Type, AD1Type>`: nested dual for the forward-forward
/// Hessian.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Ad2 {
    pub v: Ad1,
    pub g: Ad1,
}

/// Scalar abstraction over `f64` / [`Ad1`] / [`Ad2`] mirroring the C++
/// `template <typename type>` parameter of the AD metric functions.
pub trait AdScalar: Copy + Default + PartialEq + std::fmt::Debug {
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    /// MFEM `dual / dual`:
    /// `{a.v/b.v, (a.g/b.v) - (a.v*b.g)/(b.v*b.v)}`.
    fn div(self, o: Self) -> Self;
    /// MFEM `real_t * dual`.
    fn scale(self, s: f64) -> Self;
    /// MFEM `dual + real_t`: `{v + r, g}`.
    fn add_real(self, r: f64) -> Self;
    /// MFEM `dual - real_t`: `{v - r, g}`.
    fn sub_real(self, r: f64) -> Self;
    /// MFEM `real_t / dual`: `{a/b.v, -(a/(b.v*b.v))*b.g}`.
    fn real_div(a: f64, self_: Self) -> Self;
    /// MFEM `dual / real_t`.
    fn div_real(self, r: f64) -> Self;
    fn neg(self) -> Self;
    /// MFEM `sqrt(dual)`: `{sqrt(v), g / (2*sqrt(v))}`.
    fn sqrt(self) -> Self;
    /// MFEM `pow(dual a, real_t b)`: `{pow(v, b), value*g*b/v}`.
    fn powf(self, b: f64) -> Self;
}

impl AdScalar for f64 {
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        self / o
    }
    #[inline]
    fn scale(self, s: f64) -> Self {
        s * self
    }
    #[inline]
    fn add_real(self, r: f64) -> Self {
        self + r
    }
    #[inline]
    fn sub_real(self, r: f64) -> Self {
        self - r
    }
    #[inline]
    fn real_div(a: f64, self_: Self) -> Self {
        a / self_
    }
    #[inline]
    fn div_real(self, r: f64) -> Self {
        self / r
    }
    #[inline]
    fn neg(self) -> Self {
        -self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn powf(self, b: f64) -> Self {
        f64::powf(self, b)
    }
}

impl AdScalar for Ad1 {
    #[inline]
    fn add(self, o: Self) -> Self {
        Ad1 {
            v: self.v + o.v,
            g: self.g + o.g,
        }
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Ad1 {
            v: self.v - o.v,
            g: self.g - o.g,
        }
    }
    /// MFEM `dual * dual`: `{a.v*b.v, b.v*a.g + a.v*b.g}` (same order).
    #[inline]
    fn mul(self, o: Self) -> Self {
        Ad1 {
            v: self.v * o.v,
            g: o.v * self.g + self.v * o.g,
        }
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        Ad1 {
            v: self.v / o.v,
            g: (self.g / o.v) - (self.v * o.g) / (o.v * o.v),
        }
    }
    #[inline]
    fn scale(self, s: f64) -> Self {
        Ad1 {
            v: self.v * s,
            g: self.g * s,
        }
    }
    #[inline]
    fn add_real(self, r: f64) -> Self {
        Ad1 {
            v: self.v + r,
            g: self.g,
        }
    }
    #[inline]
    fn sub_real(self, r: f64) -> Self {
        Ad1 {
            v: self.v - r,
            g: self.g,
        }
    }
    #[inline]
    fn real_div(a: f64, self_: Self) -> Self {
        Ad1 {
            v: a / self_.v,
            g: -(a / (self_.v * self_.v)) * self_.g,
        }
    }
    #[inline]
    fn div_real(self, r: f64) -> Self {
        Ad1 {
            v: self.v / r,
            g: self.g / r,
        }
    }
    #[inline]
    fn neg(self) -> Self {
        Ad1 {
            v: -self.v,
            g: -self.g,
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = f64::sqrt(self.v);
        Ad1 {
            v: s,
            g: self.g / (2.0 * s),
        }
    }
    #[inline]
    fn powf(self, b: f64) -> Self {
        // value = pow(v, b); gradient = value * g * b / v (exact MFEM order).
        let value = f64::powf(self.v, b);
        Ad1 {
            v: value,
            g: value * self.g * b / self.v,
        }
    }
}

impl AdScalar for Ad2 {
    #[inline]
    fn add(self, o: Self) -> Self {
        Ad2 {
            v: self.v.add(o.v),
            g: self.g.add(o.g),
        }
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Ad2 {
            v: self.v.sub(o.v),
            g: self.g.sub(o.g),
        }
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        Ad2 {
            v: self.v.mul(o.v),
            g: o.v.mul(self.g).add(self.v.mul(o.g)),
        }
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        Ad2 {
            v: self.v.div(o.v),
            g: self.g.div(o.v).sub(self.v.mul(o.g).div(o.v.mul(o.v))),
        }
    }
    #[inline]
    fn scale(self, s: f64) -> Self {
        Ad2 {
            v: self.v.scale(s),
            g: self.g.scale(s),
        }
    }
    #[inline]
    fn add_real(self, r: f64) -> Self {
        Ad2 {
            v: self.v.add_real(r),
            g: self.g,
        }
    }
    #[inline]
    fn sub_real(self, r: f64) -> Self {
        Ad2 {
            v: self.v.sub_real(r),
            g: self.g,
        }
    }
    #[inline]
    fn real_div(a: f64, self_: Self) -> Self {
        // MFEM `real_t a / dual b`:
        //   {a / b.value, -(a / (b.value * b.value)) * b.gradient}
        // with Ad1 value arithmetic (b.value * b.value is a dual product).
        let bb = self_.v.mul(self_.v);
        let inner = Ad1::real_div(a, bb);
        Ad2 {
            v: Ad1::real_div(a, self_.v),
            g: inner.neg().mul(self_.g),
        }
    }
    #[inline]
    fn div_real(self, r: f64) -> Self {
        Ad2 {
            v: self.v.div_real(r),
            g: self.g.div_real(r),
        }
    }
    #[inline]
    fn neg(self) -> Self {
        Ad2 {
            v: self.v.neg(),
            g: self.g.neg(),
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        // {sqrt(a.value), a.gradient / (2 * sqrt(a.value))} with Ad1 value
        // arithmetic (the recursive dual-of-dual composition).
        let value = self.v.sqrt();
        let two_sqrt = value.scale(2.0);
        Ad2 {
            v: value,
            g: self.g.div(two_sqrt),
        }
    }
    #[inline]
    fn powf(self, b: f64) -> Self {
        // value = pow(a.value, b) (an Ad1 via the recursive instantiation);
        // gradient = value * a.gradient * b / a.value.
        let value = self.v.powf(b);
        let g = value.mul(self.g).scale(b).div(self.v);
        Ad2 { v: value, g }
    }
}

/// MFEM `ADGrad` (dX = true): dmu/dX of a scalar `mu(x, y)` with `y` a fixed
/// parameter. Returns the flat derivative array (column-major, same indexing
/// as `x`).
pub fn ad_grad(
    mu: &dyn Fn(&[Ad1], &[Ad1]) -> Ad1,
    x: &[f64],
    y: Option<&[f64]>,
) -> Vec<f64> {
    let n = x.len();
    let mut adx: Vec<Ad1> = x.iter().map(|&v| Ad1 { v, g: 0.0 }).collect();
    let ady: Vec<Ad1> = match y {
        Some(yv) => yv.iter().map(|&v| Ad1 { v, g: 0.0 }).collect(),
        None => vec![Ad1 { v: 0.0, g: 0.0 }; n],
    };
    let mut out = vec![0.0; n];
    for i in 0..n {
        // adX[i] = AD1Type{x_i, 1.0}.
        adx[i] = Ad1 {
            v: x[i],
            g: 1.0,
        };
        let rez = mu(&adx, &ady);
        out[i] = rez.g;
        adx[i] = Ad1 {
            v: x[i],
            g: 0.0,
        };
    }
    out
}

/// MFEM `ADHessian`: forward-forward d²mu/dX². Returns the DenseTensor laid
/// out as `h[ii*n + jj] = d²mu/dx_ii dx_jj` — the `ii`-th slice of MFEM's
/// `DenseTensor d2mu_dX2(dim, dim, dim*dim)` with column-major inner entries,
/// i.e. slice `ii` is the matrix dP_ii/dJ of `DefaultAssembleH`.
pub fn ad_hessian(
    mu: &dyn Fn(&[Ad2], &[Ad2]) -> Ad2,
    x: &[f64],
    y: Option<&[f64]>,
) -> Vec<f64> {
    let n = x.len();
    let ady: Vec<Ad2> = match y {
        Some(yv) => yv.iter().map(|&v| val2(v)).collect(),
        None => vec![val2(0.0); n],
    };
    let mut aduu: Vec<Ad2> = x.iter().map(|&v| val2(v)).collect();

    let mut out = vec![0.0; n * n];
    for ii in 0..n {
        // aduu[ii].value = AD1Type{x_ii, 1.0}.
        aduu[ii].v.g = 1.0;
        for jj in 0..=ii {
            // aduu[jj].gradient = AD1Type{1.0, 0.0}.
            aduu[jj].g = Ad1 {
                v: 1.0,
                g: 0.0,
            };
            let rez = mu(&aduu, &ady);
            let gg = rez.g.g;
            out[ii * n + jj] = gg;
            out[jj * n + ii] = gg;
            // aduu[jj].gradient = AD1Type{0.0, 0.0}.
            aduu[jj].g = Ad1 {
                v: 0.0,
                g: 0.0,
            };
        }
        aduu[ii].v.g = 0.0;
    }
    out
}

fn val2(v: f64) -> Ad2 {
    Ad2 {
        v: Ad1 { v, g: 0.0 },
        g: Ad1 { v: 0.0, g: 0.0 },
    }
}

/// MFEM `TMOP_QualityMetric::DefaultAssembleH` (fem/tmop.cpp): contract the
/// Hessian tensor `H(r + c*dim)(rr, cc)` with the basis derivative matrix
/// `DS` (dof x dim, column-major) into the local gradient matrix `A`
/// ((dof*dim) x (dof*dim), column-major, block layout
/// `A(i + r*dof, j + rr*dof)` — the fem-rs `assemble_h` convention).
pub fn default_assemble_h(
    h: &[f64],
    ds: &[f64],
    dof: usize,
    dim: usize,
    weight: f64,
    a: &mut [f64],
) {
    // h layout: slice s = r + c*dim holds the matrix d(P_rc)/dJ with
    // column-major entries (rr, cc): h[s*dim*dim + rr + cc*dim].
    let ah = dof * dim;
    for r in 0..dim {
        for c in 0..dim {
            let slice = (r + c * dim) * dim * dim;
            for rr in 0..dim {
                for cc in 0..dim {
                    let entry_rr_cc = h[slice + rr + cc * dim];
                    for i in 0..dof {
                        for j in 0..dof {
                            // A(i + r*dof, j + rr*dof) += weight*DS(i,c)*DS(j,cc)*Hrc
                            let row = i + r * dof;
                            let col = j + rr * dof;
                            a[row + col * ah] +=
                                weight * ds[i + c * dof] * ds[j + cc * dof] * entry_rr_cc;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // mu(x) = x0*x1 + x2^2 + 3
    fn mu_test<S: AdScalar>(x: &[S], _y: &[S]) -> S {
        let a = x[0].mul(x[1]);
        let b = x[2].mul(x[2]);
        a.add(b).add_real(3.0)
    }

    #[test]
    fn ad_grad_poly() {
        let x = [2.0, 3.0, 1.5];
        let g = ad_grad(&mu_test::<Ad1>, &x, None);
        assert!((g[0] - 3.0).abs() < 1e-14);
        assert!((g[1] - 2.0).abs() < 1e-14);
        assert!((g[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn ad_hessian_poly() {
        let x = [2.0, 3.0, 1.5];
        let h = ad_hessian(&mu_test::<Ad2>, &x, None);
        // H = [[0,1,0],[1,0,0],[0,0,2]]
        assert!(h[0 * 3 + 0].abs() < 1e-14);
        assert!((h[0 * 3 + 1] - 1.0).abs() < 1e-14);
        assert!((h[2 * 3 + 2] - 2.0).abs() < 1e-14);
        assert!(h[0 * 3 + 2].abs() < 1e-14);
    }

    // mu(x, y) = 0.25/x0 * |x1 - y0|^2 + sqrt(x2)
    fn mu_test2<S: AdScalar>(x: &[S], y: &[S]) -> S {
        let d = x[1].sub(y[0]);
        let s = x[1].sub(y[0]);
        let f = d.mul(s);
        let t = AdScalar::real_div(0.25, x[0]).mul(f);
        t.add(x[2].sqrt())
    }

    #[test]
    fn ad_grad_with_y() {
        let x = [2.0, 3.0, 4.0];
        let y = [1.0];
        // dmu/dx0 = -0.25/x0^2 * |x1-y0|^2 = -0.25/4*4 = -0.25
        // dmu/dx1 = 0.5/x0 * (x1-y0) = 0.25*2 = 0.5
        // dmu/dx2 = 1/(2*sqrt(4)) = 0.25
        let g = ad_grad(&mu_test2::<Ad1>, &x, Some(&y));
        assert!((g[0] + 0.25).abs() < 1e-14);
        assert!((g[1] - 0.5).abs() < 1e-14);
        assert!((g[2] - 0.25).abs() < 1e-14);
        let h = ad_hessian(&mu_test2::<Ad2>, &x, Some(&y));
        // d2/dx0dx1 = -0.5/x0^2 * (x1-y0) = -0.5/4*2 = -0.25
        assert!((h[0 * 3 + 1] + 0.25).abs() < 1e-14);
        // d2/dx1dx1 = 0.5/x0 = 0.25
        assert!((h[1 * 3 + 1] - 0.25).abs() < 1e-14);
        // d2/dx2dx2 = -1/(4 x^{3/2}) = -1/32
        assert!((h[2 * 3 + 2] + 1.0 / 32.0).abs() < 1e-14);
        // d2/dx0dx0 = 0.5/x0^3 * |x1-y0|^2 = 0.5/8*4 = 0.25
        assert!((h[0] - 0.25).abs() < 1e-14);
    }

    #[test]
    fn ad1_sqrt_powf_chain() {
        // f(x) = sqrt(x)^3 = x^{3/2} -> f'(x) = (3/2) sqrt(x) = 3 at x = 4.
        let mu = |x: &[Ad1], _y: &[Ad1]| x[0].sqrt().powf(3.0);
        let g = ad_grad(&mu, &[4.0], None);
        assert!((g[0] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn default_assemble_h_layout() {
        // H = identity: dP_rc/dJ_rrcc = delta. A += weight * DS(i,c) DS(j,cc).
        let dof = 2;
        let dim = 2;
        let mut h = vec![0.0; dim * dim * dim * dim];
        for k in 0..dim * dim {
            h[k * dim * dim + k] = 1.0;
        }
        let ds = [0.5, -0.25, 1.0, 2.0]; // column-major 2x2
        let mut a = vec![0.0; dof * dim * dof * dim];
        default_assemble_h(&h, &ds, dof, dim, 2.0, &mut a);
        // With H = I the contraction reduces to
        // A(i + r*dof, j + r*dof) += weight * sum_c DS(i,c) DS(j,c).
        for i in 0..dof {
            for j in 0..dof {
                let mut s = 0.0;
                for c in 0..dim {
                    s += ds[i + c * dof] * ds[j + c * dof];
                }
                let expected = 2.0 * s;
                assert!((a[i + j * dof * dim] - expected).abs() < 1e-14);
            }
        }
    }
}

