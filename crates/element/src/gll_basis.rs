//! 1-D Gauss-Lobatto / integrated-GLL ("Gerritsma") bases.
//!
//! This mirrors MFEM's tensor-element convention
//! (`Poly_1D::ClosedPoints(p, BasisType::GaussLobatto)` for the *closed*
//! nodal basis and `BasisType::IntegratedGLL` for the *open* basis used by
//! `ND_HexahedronElement` / `RT_HexahedronElement`).
//!
//! # Reference frames (D721)
//!
//! MFEM's `Poly_1D` bases are `[0,1]`-parameterised.  The **quad** tensor
//! elements (`QuadRTk`/`QuadND`, `[0,1]²` since D364) reach those values
//! through [`ClosedBasis::new`] by evaluating the `[-1,1]` node set at
//! `2x−1`, which is the historical fem-rs frame kept here for that family
//! alone.  The **hex** elements moved to MFEM's `[0,1]³` in D721 and use
//! [`ClosedBasis::new_01`] (`gll_nodes_01` = the single centralized
//! `0.5·(x+1)` image of the `[-1,1]` table) so that the basis *derivatives*
//! are MFEM's `[0,1]` derivatives — the integrated (Gerritsma) open modes
//! `o_i = −Σ_{j≤i} c'_j` are then native and need no `1/2` chain factor.
//!
//! The open ("integrated") modes are
//!
//! ```text
//!     o_i(x) = -(c'_0(x) + c'_1(x) + ... + c'_i(x)),   i = 0..n-1
//! ```
//!
//! built from the derivatives of the degree-`n` GLL nodal basis `c_j`.  Each
//! `o_i` has *unit integral* over the unit interval:
//!
//! ```text
//!     ∫ o_i dx = -Σ_{j<=i} (c_j(1) - c_j(0)) = 1
//! ```
//!
//! because `c_j(1) = δ_{j,n-1}` and `c_j(0) = δ_{j,0}`.  This unit-integral
//! normalization is what makes the H(curl) line-integral dofs and the H(div)
//! flux dofs of the tensor elements exact duals of the basis, and
//! `(GaussLobatto, IntegratedGLL)` is the basis pair MFEM documents for LOR
//! discretizations.

const MAX_P: usize = 16;

/// `p + 1` Gauss-Lobatto-Legendre points on `[-1, 1]` (closed points).
pub(crate) fn gll_nodes(p: usize) -> &'static [f64] {
    use std::sync::OnceLock;
    static CACHE: [OnceLock<Vec<f64>>; MAX_P + 1] = {
        #[allow(clippy::declare_interior_mutable_const)]
        const NEW: OnceLock<Vec<f64>> = OnceLock::new();
        [NEW; MAX_P + 1]
    };
    assert!(p <= MAX_P, "gll_nodes: order {p} > {MAX_P} unsupported");
    CACHE[p].get_or_init(|| crate::quadrature::gauss_lobatto_arbitrary(p + 1).0)
}

/// `p` Gauss-Legendre points on `[-1, 1]` (open points, endpoints excluded).
pub(crate) fn gl_nodes(p: usize) -> &'static [f64] {
    use std::sync::OnceLock;
    static CACHE: [OnceLock<Vec<f64>>; MAX_P + 1] = {
        #[allow(clippy::declare_interior_mutable_const)]
        const NEW: OnceLock<Vec<f64>> = OnceLock::new();
        [NEW; MAX_P + 1]
    };
    assert!(p <= MAX_P, "gl_nodes: order {p} > {MAX_P} unsupported");
    CACHE[p].get_or_init(|| crate::quadrature::gauss_legendre_arbitrary(p).0)
}

/// `p + 1` Gauss-Lobatto-Legendre points on `[0, 1]` — MFEM's
/// `Poly_1D::ClosedPoints(p, GaussLobatto)` frame, reached as the single
/// centralized `0.5·(x+1)` image of the `[-1,1]` table (MFEM's
/// `QuadratureFunctions1D::GaussLobatto` stores exactly `z = (x+1)/2` for the
/// lower half and `1 − z` for the mirrored points).
pub(crate) fn gll_nodes_01(p: usize) -> Vec<f64> {
    gll_nodes(p).iter().map(|&x| 0.5 * (x + 1.0)).collect()
}

/// `p` Gauss-Legendre points on `[0, 1]` (open points) — MFEM's
/// `Poly_1D::OpenPoints(p-1, GaussLegendre)` frame.
///
/// D721: taken from the canonical `[0,1]` table ([`gauss_legendre_01`], the
/// hard-coded == MFEM values for `p ≤ 5`), **not** from the `[-1,1]` table via
/// `0.5·(x+1)`: MFEM iterates its `[0,1]` roots directly
/// (`QuadratureFunctions1D::GaussLegendre`'s `xi = ((1−z)+dz)/2`), and the
/// affine image of the `[-1,1]` roots differs by 1 ulp at the outer nodes for
/// every `p ≥ 2` (D339's finding, now removed from the element nodes too).
pub(crate) fn gl_nodes_01(p: usize) -> Vec<f64> {
    crate::quadrature::gauss_legendre_01(p).0
}

/// Values of the closed GLL nodal basis of degree `p` at one point.
pub(crate) struct ClosedVals {
    /// Nodal values `c_j(x)`.
    pub c: Vec<f64>,
    /// First derivatives `c'_j(x)`.
    pub dc: Vec<f64>,
    /// Second derivatives `c''_j(x)`.
    pub d2c: Vec<f64>,
}

/// Barycentric GLL nodal basis of degree `p` (`p + 1` modes) — see the module
/// docs for which reference frame each constructor serves.
#[derive(Clone)]
pub(crate) struct ClosedBasis {
    pub nodes: Vec<f64>,
    /// Barycentric weights `w_j = 1 / Π_{k!=j} (x_j - x_k)`.
    w: Vec<f64>,
}

impl ClosedBasis {
    /// Degree-`p` GLL nodal basis on the historical fem-rs `[-1,1]` frame —
    /// the quad tensor family's parameterization (`QuadRTk`/`QuadND` evaluate
    /// it at `2x−1`).
    pub fn new(p: usize) -> Self {
        Self::from_nodes(gll_nodes(p).to_vec())
    }

    /// Degree-`p` GLL nodal basis on MFEM's natural `[0,1]` frame (D721) —
    /// the **hex** tensor family's parameterization, so `dc`/`d2c` are
    /// derivatives w.r.t. the `[0,1]` reference coordinate.
    pub fn new_01(p: usize) -> Self {
        Self::from_nodes(gll_nodes_01(p))
    }

    fn from_nodes(nodes: Vec<f64>) -> Self {
        let n = nodes.len();
        // Barycentric weights in MFEM's `Poly_1D::Basis::Basis` accumulation
        // order (`for i { for j < i { xij = x(i)-x(j); w(i) *= xij;
        // w(j) *= -xij } }` then one reciprocal) — matching it keeps the
        // weights (and hence every value below) bit-identical; the historical
        // `w[j] /= Π(nodes[j]-nodes[k])` full loop differs by ~1 ulp for
        // `n ≥ 3` and is why the D680 RT1 gate needed this fix.
        let mut w = vec![1.0_f64; n];
        for i in 0..n {
            for j in 0..i {
                let xij = nodes[i] - nodes[j];
                w[i] *= xij;
                w[j] *= -xij;
            }
        }
        for v in w.iter_mut() {
            *v = 1.0 / *v;
        }
        ClosedBasis { nodes, w }
    }

    /// MFEM `Poly_1D::Basis::Eval(y, u)` **value-only** overload — the form
    /// `CalcShape`/`CalcVShape` use: stable-centre split with the *division*
    /// form `u(i) = l·w(i)/(y − x(i))`.  Bit-for-bit MFEM, and ~1 ulp away
    /// from the value returned by the derivative overload
    /// ([`ClosedBasis::eval_mfem`]), which `CalcDivShape`/`CalcCurlShape`
    /// consume.
    pub fn val_mfem(&self, y: f64) -> Vec<f64> {
        let n = self.nodes.len();
        if n == 1 {
            return vec![1.0];
        }
        if let Some(m) = self.nodes.iter().position(|&xn| xn == y) {
            let mut u = vec![0.0_f64; n];
            u[m] = 1.0;
            return u;
        }
        let (k, lk, l) = self.centre(y);
        let mut u = vec![0.0_f64; n];
        for i in 0..k {
            u[i] = l * self.w[i] / (y - self.nodes[i]);
        }
        u[k] = lk * self.w[k];
        for i in k + 1..n {
            u[i] = l * self.w[i] / (y - self.nodes[i]);
        }
        u
    }

    /// MFEM's stable-centre split of `∏(y − x_i)`: `(k, lk, l)` with
    /// `lk = ∏_{i≠k}(y−x_i)` and `l = lk·(y−x_k)` (`fem/fe/fe_base.cpp:1896`).
    fn centre(&self, y: f64) -> (usize, f64, f64) {
        let x = &self.nodes;
        let p = x.len() - 1;
        let mut k = 0usize;
        let mut lk = 1.0;
        while k < p {
            if y >= (x[k] + x[k + 1]) / 2.0 {
                lk *= y - x[k];
                k += 1;
            } else {
                for i in k + 1..=p {
                    lk *= y - x[i];
                }
                break;
            }
        }
        let l = lk * (y - x[k]);
        (k, lk, l)
    }

    /// MFEM `Poly_1D::Basis::Eval(y, u, d, d2)` — the derivative overload
    /// (`CalcDShape`/`CalcDivShape`/`CalcCurlShape`): values through the
    /// reciprocal-multiplication form `u(i) = l·si·w(i)` and MFEM's exact
    /// `lp`/`lp2`/`sk`/`sk2` derivative chains (`fem/fe/fe_base.cpp:2016`).
    pub fn eval_mfem(&self, y: f64) -> ClosedVals {
        let x = &self.nodes;
        let n = x.len();
        let mut c = vec![0.0_f64; n];
        let mut dc = vec![0.0_f64; n];
        let mut d2c = vec![0.0_f64; n];
        if n == 1 {
            return ClosedVals { c, dc, d2c };
        }
        if let Some(m) = x.iter().position(|&xn| xn == y) {
            // Exact evaluation at a node (the barycentric formula is singular).
            c[m] = 1.0;
            let a0: f64 = (0..n)
                .filter(|&k| k != m)
                .map(|k| 1.0 / (x[m] - x[k]))
                .sum();
            let b0: f64 = (0..n)
                .filter(|&k| k != m)
                .map(|k| {
                    let t = 1.0 / (x[m] - x[k]);
                    t * t
                })
                .sum();
            dc[m] = a0;
            d2c[m] = a0 * a0 - b0;
            let a0w: f64 = (0..n)
                .filter(|&k| k != m)
                .map(|k| self.w[k] / (x[m] - x[k]))
                .sum();
            for j in 0..n {
                if j != m {
                    let r = self.w[j] / self.w[m];
                    let d = x[m] - x[j];
                    dc[j] = r / d;
                    d2c[j] = (2.0 * r) * (-1.0 / (d * d) - a0w / (self.w[m] * d));
                }
            }
            return ClosedVals { c, dc, d2c };
        }
        let (k, lk, l) = self.centre(y);
        let (mut sk, mut sk2) = (0.0_f64, 0.0_f64);
        for i in 0..k {
            let si = 1.0 / (y - x[i]);
            sk += si;
            sk2 -= si * si;
            c[i] = l * si * self.w[i];
        }
        c[k] = lk * self.w[k];
        for i in k + 1..n {
            let si = 1.0 / (y - x[i]);
            sk += si;
            sk2 -= si * si;
            c[i] = l * si * self.w[i];
        }
        let lp = l * sk + lk;
        let lp2 = lp * sk + l * sk2 + sk * lk;
        for i in 0..k {
            dc[i] = (lp * self.w[i] - c[i]) / (y - x[i]);
            d2c[i] = (lp2 * self.w[i] - 2.0 * dc[i]) / (y - x[i]);
        }
        dc[k] = sk * c[k];
        d2c[k] = sk2 * c[k] + sk * dc[k];
        for i in k + 1..n {
            dc[i] = (lp * self.w[i] - c[i]) / (y - x[i]);
            d2c[i] = (lp2 * self.w[i] - 2.0 * dc[i]) / (y - x[i]);
        }
        ClosedVals { c, dc, d2c }
    }

    /// Evaluate all modes with first and second derivatives at `x`.
    pub fn eval(&self, x: f64) -> ClosedVals {
        let n = self.nodes.len();
        let mut c = vec![0.0_f64; n];
        let mut dc = vec![0.0_f64; n];
        let mut d2c = vec![0.0_f64; n];

        // Exact evaluation at a node (the barycentric formula is singular).
        for m in 0..n {
            if x == self.nodes[m] {
                c[m] = 1.0;
                // l'_m(x_m) = A0 = Σ_{k!=m} 1/(x_m - x_k)
                // l''_m(x_m) = A0^2 - Σ_{k!=m} 1/(x_m - x_k)^2
                let a0: f64 = (0..n)
                    .filter(|&k| k != m)
                    .map(|k| 1.0 / (self.nodes[m] - self.nodes[k]))
                    .sum();
                let b0: f64 = (0..n)
                    .filter(|&k| k != m)
                    .map(|k| {
                        let t = 1.0 / (self.nodes[m] - self.nodes[k]);
                        t * t
                    })
                    .sum();
                dc[m] = a0;
                d2c[m] = a0 * a0 - b0;
                // For j != m (Taylor expansion of the barycentric quotient):
                //   l'_j(x_m)  = (w_j / w_m) / (x_m - x_j)
                //   l''_j(x_m) = (2 w_j / w_m) [ -1/(x_m-x_j)^2
                //                               - (A0w/w_m) / (x_m-x_j) ]
                //   with A0w = Σ_{k!=m} w_k / (x_m - x_k)
                let a0w: f64 = (0..n)
                    .filter(|&k| k != m)
                    .map(|k| self.w[k] / (self.nodes[m] - self.nodes[k]))
                    .sum();
                for j in 0..n {
                    if j != m {
                        let r = self.w[j] / self.w[m];
                        let d = self.nodes[m] - self.nodes[j];
                        dc[j] = r / d;
                        d2c[j] = (2.0 * r) * (-1.0 / (d * d) - a0w / (self.w[m] * d));
                    }
                }
                return ClosedVals { c, dc, d2c };
            }
        }

        // General barycentric evaluation:
        //   l_j  = lam_j / S                      lam_j  = w_j / t_j
        //   l'_j = (lam'_j - l_j S') / S          lam'_j = -w_j / t_j^2
        //   l''_j = (lam''_j - 2 l'_j S' - l_j S'') / S
        //                                         lam''_j = 2 w_j / t_j^3
        let mut lam = vec![0.0_f64; n];
        let mut dlam = vec![0.0_f64; n];
        let mut d2lam = vec![0.0_f64; n];
        let (mut s, mut ds, mut d2s) = (0.0_f64, 0.0_f64, 0.0_f64);
        for j in 0..n {
            let t = x - self.nodes[j];
            lam[j] = self.w[j] / t;
            dlam[j] = -self.w[j] / (t * t);
            d2lam[j] = 2.0 * self.w[j] / (t * t * t);
            s += lam[j];
            ds += dlam[j];
            d2s += d2lam[j];
        }
        for j in 0..n {
            c[j] = lam[j] / s;
            dc[j] = (dlam[j] - c[j] * ds) / s;
            d2c[j] = (d2lam[j] - 2.0 * dc[j] * ds - c[j] * d2s) / s;
        }
        ClosedVals { c, dc, d2c }
    }

    /// Integrated (Gerritsma) open modes `o_i = -Σ_{j<=i} c'_j`
    /// (`n - 1` modes from `n` closed modes).
    pub fn integrated(&self, v: &ClosedVals) -> Vec<f64> {
        Self::partial_sums(&v.dc)
    }

    /// Derivatives of the integrated open modes `o'_i = -Σ_{j<=i} c''_j`.
    pub fn integrated_deriv(&self, v: &ClosedVals) -> Vec<f64> {
        Self::partial_sums(&v.d2c)
    }

    fn partial_sums(d: &[f64]) -> Vec<f64> {
        let n = d.len() - 1;
        let mut o = vec![0.0_f64; n];
        if n == 0 {
            return o;
        }
        o[0] = -d[0];
        for i in 1..n {
            o[i] = o[i - 1] - d[i];
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gll_nodes_endpoints_and_symmetry() {
        for p in 1..=8usize {
            let n = gll_nodes(p);
            assert_eq!(n.len(), p + 1);
            assert!((n[0] - (-1.0)).abs() < 1e-14);
            assert!((n[p] - 1.0).abs() < 1e-14);
            for i in 0..=p {
                assert!((n[i] + n[p - i]).abs() < 1e-13, "p={p} symmetry");
            }
        }
    }

    /// D721: the `[0,1]` images are the centralized `0.5·(x+1)` of the
    /// `[-1,1]` table, endpoints `0`/`1` included.
    #[test]
    fn gll_nodes_01_is_the_central_image() {
        for p in 1..=8usize {
            let n = gll_nodes_01(p);
            let n11 = gll_nodes(p);
            assert_eq!(n.len(), p + 1);
            assert_eq!(n[0], 0.0);
            assert_eq!(n[p], 1.0);
            for i in 0..=p {
                assert_eq!(n[i], 0.5 * (n11[i] + 1.0));
                assert!((n[i] + n[p - i] - 1.0).abs() < 1e-13, "p={p} symmetry");
            }
            for gl in gl_nodes_01(p) {
                assert!(gl > 0.0 && gl < 1.0, "open points inside (0,1)");
            }
        }
    }

    /// The integrated (Gerritsma) open modes of the `[0,1]` frame integrate
    /// to 1 over `[0,1]` — the unit-integral normalization the H(div)/H(curl)
    /// duals rely on, with no chain factor (D721).
    #[test]
    fn integrated_modes_01_have_unit_integral() {
        for p in 1..=8usize {
            let b = ClosedBasis::new_01(p);
            let (xs, ws) = crate::quadrature::gauss_legendre_01(p + 2);
            for i in 0..p {
                let mut acc = 0.0;
                for (q, &x) in xs.iter().enumerate() {
                    let v = b.eval(x);
                    acc += ws[q] * b.integrated(&v)[i];
                }
                assert!((acc - 1.0).abs() < 1e-13, "p={p} ∫o_{i} = {acc}");
            }
        }
    }

    #[test]
    fn closed_basis_partition_of_unity_and_interpolation() {
        for p in 1..=8usize {
            let b = ClosedBasis::new(p);
            for &x in &[-1.0, -0.97, -0.5, -0.13, 0.0, 0.31, 0.62, 0.88, 1.0] {
                let v = b.eval(x);
                let s: f64 = v.c.iter().sum();
                assert!((s - 1.0).abs() < 1e-13, "p={p} Σc at {x}: {s}");
                let ds: f64 = v.dc.iter().sum();
                assert!(ds.abs() < 1e-12, "p={p} Σc' at {x}: {ds}");
            }
            // Nodal interpolation property.
            for (m, &xm) in b.nodes.iter().enumerate() {
                let v = b.eval(xm);
                for (j, &cj) in v.c.iter().enumerate() {
                    let want = if j == m { 1.0 } else { 0.0 };
                    assert!((cj - want).abs() < 1e-13);
                }
            }
        }
    }

    /// First and second derivatives against high-order central differences.
    #[test]
    fn derivatives_match_finite_differences() {
        for p in 1..=6usize {
            let b = ClosedBasis::new(p);
            let h = 1e-4;
            for &x in &[-0.83, -0.5, -0.13, 0.0, 0.31, 0.62, 0.88] {
                let v = b.eval(x);
                let vp = b.eval(x + h);
                let vm = b.eval(x - h);
                for j in 0..=p {
                    let d1 = (vp.c[j] - vm.c[j]) / (2.0 * h);
                    let d2 = (vp.c[j] - 2.0 * v.c[j] + vm.c[j]) / (h * h);
                    assert!(
                        (v.dc[j] - d1).abs() < 1e-5 * (1.0 + v.dc[j].abs()),
                        "p={p} c'[{j}] at {x}"
                    );
                    assert!(
                        (v.d2c[j] - d2).abs() < 1e-4 * (1.0 + v.d2c[j].abs()),
                        "p={p} c''[{j}] at {x}"
                    );
                }
            }
        }
    }

    #[test]
    fn integrated_modes_have_unit_integral() {
        for p in 1..=8usize {
            let b = ClosedBasis::new(p);
            let (xs, ws) = crate::quadrature::gauss_legendre_arbitrary(p + 2);
            for i in 0..p {
                let mut acc = 0.0;
                for (q, &x) in xs.iter().enumerate() {
                    let v = b.eval(x);
                    acc += ws[q] * b.integrated(&v)[i];
                }
                assert!((acc - 1.0).abs() < 1e-13, "p={p} ∫o_{i} = {acc}");
            }
        }
    }

    /// o'_i must equal the partial sums of the closed-basis second
    /// derivatives (checked against finite differences of o_i).
    #[test]
    fn integrated_deriv_matches_finite_difference() {
        for p in 1..=6usize {
            let b = ClosedBasis::new(p);
            let h = 1e-5;
            for &x in &[-0.83, -0.5, 0.31, 0.88] {
                let o = b.integrated(&b.eval(x))[0];
                let op = b.integrated(&b.eval(x + h))[0];
                let om = b.integrated(&b.eval(x - h))[0];
                let fd = (op - om) / (2.0 * h);
                let an = b.integrated_deriv(&b.eval(x))[0];
                assert!((an - fd).abs() < 1e-5 * (1.0 + an.abs()), "p={p} at {x}");
                let _ = o;
            }
        }
    }
}

