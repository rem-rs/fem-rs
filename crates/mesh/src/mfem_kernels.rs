//! Bit-exact port of MFEM's `kernels::CalcSingularvalue<3>` (linalg/kernels.hpp).
//!
//! The tet refinement's octahedron-diagonal choice (`rt`) in
//! `UniformRefinement3D_base` selects the candidate with the smallest aspect
//! ratio σ₀/σ₂, where the singular values come from this exact algorithm.
//! Ties (e.g. for symmetric tets) are broken by the *floating-point noise* of
//! this specific algorithm, so a different SVD implementation (nalgebra)
//! selects different diagonals and produces a different refined mesh (and
//! hence a different FE solution).  This module replicates the C++ arithmetic
//! step-for-step so `tet_select_rt_debug` matches MFEM bit-for-bit.

/// `GetScalingFactor` (kernels.hpp): `mult = 2^d_exp` with
/// `d_max/mult ∈ [0.5, 1)`.  For normal finite `d_max > 0` this is a pure
/// power of two (exact scaling); subnormals/zero fall back to 1.
fn get_scaling_factor(d_max: f64) -> f64 {
    if d_max > 0.0 && d_max.is_normal() {
        // frexp(d_max) = (m, e) with d_max = m·2^e, m ∈ [0.5, 1);
        // mult = d_max/m = 2^e = 2^(unbiased_exp + 1).
        let bits = d_max.to_bits();
        let exp_field = (bits >> 52) & 0x7FF;
        let mult_bits = ((exp_field + 1) << 52) | (bits & 0x8000_0000_0000_0000);
        f64::from_bits(mult_bits)
    } else {
        1.0
    }
}

/// `internal::Eigenvalues2S` (kernels.hpp): one Jacobi rotation on the 2×2
/// symmetric block [[d1, d12], [d12, d2]].
fn eigenvalues2s(d12: &mut f64, d1: &mut f64, d2: &mut f64) {
    let sqrt_1_eps = (1.0 / f64::EPSILON).sqrt();
    if *d12 != 0.0 {
        let zeta = (*d2 - *d1) / (2.0 * *d12);
        let t = if zeta.abs() < sqrt_1_eps {
            *d12 * zeta.copysign(1.0 / (zeta.abs() + (1.0 + zeta * zeta).sqrt()))
        } else {
            *d12 * zeta.copysign(0.5 / zeta.abs())
        };
        *d1 -= t;
        *d2 += t;
    }
}

/// `internal::Vec_normalize3` / `Vec_normalize3_aux` (kernels.hpp).
fn vec_normalize3(x1: &mut f64, x2: &mut f64, x3: &mut f64) {
    let (n1, n2, n3);
    if x1.abs() >= x2.abs() {
        if x1.abs() >= x3.abs() {
            if *x1 != 0.0 {
                let m = x1.abs();
                let r = *x2 / m;
                let mut t = 1.0 + r * r;
                let r = *x3 / m;
                t = (1.0 / (t + r * r)).sqrt();
                n1 = t.copysign(*x1);
                let t = t / m;
                n2 = *x2 * t;
                n3 = *x3 * t;
            } else {
                n1 = 0.0;
                n2 = 0.0;
                n3 = 0.0;
            }
            *x1 = n1;
            *x2 = n2;
            *x3 = n3;
            return;
        }
    } else if x2.abs() >= x3.abs() {
        // Vec_normalize3_aux(x2, x1, x3, n2, n1, n3)
        let m = x2.abs();
        let r = *x1 / m;
        let mut t = 1.0 + r * r;
        let r = *x3 / m;
        t = (1.0 / (t + r * r)).sqrt();
        n2 = t.copysign(*x2);
        let t = t / m;
        n1 = *x1 * t;
        n3 = *x3 * t;
        *x1 = n1;
        *x2 = n2;
        *x3 = n3;
        return;
    }
    // Vec_normalize3_aux(x3, x1, x2, n3, n1, n2)
    let m = x3.abs();
    let r = *x1 / m;
    let mut t = 1.0 + r * r;
    let r = *x2 / m;
    t = (1.0 / (t + r * r)).sqrt();
    n3 = t.copysign(*x3);
    let t = t / m;
    n1 = *x1 * t;
    n2 = *x2 * t;
    *x1 = n1;
    *x2 = n2;
    *x3 = n3;
}

/// `internal::KernelVector2G` (kernels.hpp).  Returns `true` if the matrix is
/// zero (no kernel vector set).
fn kernel_vector2g(
    mode: usize,
    d1: &mut f64,
    d12: &mut f64,
    d21: &mut f64,
    d2: &mut f64,
) -> bool {
    let n1 = d1.abs() + d21.abs();
    let n2 = d2.abs() + d12.abs();
    let swap_columns = n2 > n1;
    let mut mu;

    if !swap_columns {
        if n1 == 0.0 {
            return true;
        }
        if mode == 0 {
            if d1.abs() > d21.abs() {
                std::mem::swap(d1, d21);
                std::mem::swap(d12, d2);
            }
        } else if d1.abs() < d21.abs() {
            std::mem::swap(d1, d21);
            std::mem::swap(d12, d2);
        }
    } else if mode == 0 {
        if d12.abs() > d2.abs() {
            std::mem::swap(d1, d2);
            std::mem::swap(d12, d21);
        } else {
            std::mem::swap(d1, d12);
            std::mem::swap(d21, d2);
        }
    } else if d12.abs() < d2.abs() {
        std::mem::swap(d1, d2);
        std::mem::swap(d12, d21);
    } else {
        std::mem::swap(d1, d12);
        std::mem::swap(d21, d2);
    }

    let mut n1 = (*d1).hypot(*d21);

    if *d21 != 0.0 {
        mu = n1.copysign(*d1);
        n1 = -*d21 * (*d21 / (*d1 + mu)); // = d1 - mu
        *d1 = mu;
        let (n1v, n2v);
        if n1.abs() <= d21.abs() {
            n1v = n1 / *d21;
            mu = (2.0 / (1.0 + n1v * n1v)) * (n1v * *d12 + *d2);
            *d2 -= mu;
            *d12 -= mu * n1v;
        } else {
            n2v = *d21 / n1;
            mu = (2.0 / (1.0 + n2v * n2v)) * (*d12 + n2v * *d2);
            *d2 -= mu * n2v;
            *d12 -= mu;
        }
    }

    // choose (z1,z2) to minimize |d1*z1 + d12*z2| + |d2*z2| with |z1|+|z2|=1
    mu = -*d12 / *d1;
    let n2v = 1.0 / (1.0 + mu.abs());
    if d1.abs() <= n2v * d2.abs() {
        *d2 = 0.0;
        *d1 = 1.0;
    } else {
        *d2 = n2v;
        *d1 = mu * n2v;
    }

    if swap_columns {
        std::mem::swap(d1, d2);
    }
    false
}

/// `internal::KernelVector3G_aux` (kernels.hpp).
#[allow(clippy::too_many_arguments)]
fn kernel_vector3g_aux(
    mode: usize,
    d1: &mut f64,
    d2: &mut f64,
    d3: &mut f64,
    c12: &mut f64,
    c13: &mut f64,
    c23: &mut f64,
    c21: &mut f64,
    c31: &mut f64,
    c32: &mut f64,
) -> usize {
    let kdim;
    let (mut mu, mut n1, n2, n3, mut s1, s2, s3);

    s1 = (*c21).hypot(*c31);
    n1 = (*d1).hypot(s1);

    if s1 != 0.0 {
        mu = n1.copysign(*d1);
        n1 = -s1 * (s1 / (*d1 + mu)); // = d1 - mu
        *d1 = mu;

        if n1.abs() >= c21.abs() {
            if n1.abs() >= c31.abs() {
                // n1 is max, (s1,s2,s3) <-- (1,c21/n1,c31/n1)
                s2 = *c21 / n1;
                s3 = *c31 / n1;
                mu = 2.0 / (1.0 + s2 * s2 + s3 * s3);
                n2 = mu * (*c12 + s2 * *d2 + s3 * *c32);
                n3 = mu * (*c13 + s2 * *c23 + s3 * *d3);
                *c12 -= n2;
                *d2 -= s2 * n2;
                *c32 -= s3 * n2;
                *c13 -= n3;
                *c23 -= s2 * n3;
                *d3 -= s3 * n3;
                // goto done_column_1
            } else {
                // c31 is max
                s1 = n1 / *c31;
                s2 = *c21 / *c31;
                mu = 2.0 / (1.0 + s1 * s1 + s2 * s2);
                n2 = mu * (s1 * *c12 + s2 * *d2 + *c32);
                n3 = mu * (s1 * *c13 + s2 * *c23 + *d3);
                *c12 -= s1 * n2;
                *d2 -= s2 * n2;
                *c32 -= n2;
                *c13 -= s1 * n3;
                *c23 -= s2 * n3;
                *d3 -= n3;
            }
        } else if c21.abs() >= c31.abs() {
            // c21 is max
            s1 = n1 / *c21;
            s3 = *c31 / *c21;
            mu = 2.0 / (1.0 + s1 * s1 + s3 * s3);
            n2 = mu * (s1 * *c12 + *d2 + s3 * *c32);
            n3 = mu * (s1 * *c13 + *c23 + s3 * *d3);
            *c12 -= s1 * n2;
            *d2 -= n2;
            *c32 -= s3 * n2;
            *c13 -= s1 * n3;
            *c23 -= n3;
            *d3 -= s3 * n3;
        } else {
            // c31 is max
            s1 = n1 / *c31;
            s2 = *c21 / *c31;
            mu = 2.0 / (1.0 + s1 * s1 + s2 * s2);
            n2 = mu * (s1 * *c12 + s2 * *d2 + *c32);
            n3 = mu * (s1 * *c13 + s2 * *c23 + *d3);
            *c12 -= s1 * n2;
            *d2 -= s2 * n2;
            *c32 -= n2;
            *c13 -= s1 * n3;
            *c23 -= s2 * n3;
            *d3 -= n3;
        }
    }

    // done_column_1:
    if kernel_vector2g(mode, d2, c23, c32, d3) {
        *d2 = *c12 / *d1;
        *d3 = *c13 / *d1;
        *d1 = 1.0;
        kdim = 2;
    } else {
        *d1 = -(*c12 * *d2 + *c13 * *d3) / *d1;
        kdim = 1;
    }

    vec_normalize3(d1, d2, d3);
    kdim
}

/// `internal::KernelVector3S` (kernels.hpp).
fn kernel_vector3s(
    mode: usize,
    d12: f64,
    d13: f64,
    d23: f64,
    d1: &mut f64,
    d2: &mut f64,
    d3: &mut f64,
) -> usize {
    let mut c12 = d12;
    let mut c13 = d13;
    let mut c23 = d23;
    let (mut c21, mut c31, mut c32, col, mut row);

    // l1-norms of the columns
    c32 = d1.abs() + c12.abs() + c13.abs();
    c31 = d2.abs() + c12.abs() + c23.abs();
    c21 = d3.abs() + c13.abs() + c23.abs();

    if c32 >= c21 {
        col = if c32 >= c31 { 1 } else { 2 };
    } else {
        col = if c31 >= c21 { 2 } else { 3 };
    }
    match col {
        1 => {
            if c32 == 0.0 {
                return 3;
            }
        }
        2 => {
            if c31 == 0.0 {
                return 3;
            }
            std::mem::swap(&mut c13, &mut c23);
            std::mem::swap(d1, d2);
        }
        _ => {
            if c21 == 0.0 {
                return 3;
            }
            std::mem::swap(&mut c12, &mut c23);
            std::mem::swap(d1, d3);
        }
    }

    // row pivoting depending on 'mode'
    if mode == 0 {
        if d1.abs() <= c13.abs() {
            row = if d1.abs() <= c12.abs() { 1 } else { 2 };
        } else {
            row = if c12.abs() <= c13.abs() { 2 } else { 3 };
        }
    } else if d1.abs() >= c13.abs() {
        row = if d1.abs() >= c12.abs() { 1 } else { 2 };
    } else {
        row = if c12.abs() >= c13.abs() { 2 } else { 3 };
    }
    match row {
        1 => {
            c21 = c12;
            c31 = c13;
            c32 = c23;
        }
        2 => {
            c21 = *d1;
            c31 = c13;
            c32 = c23;
            *d1 = c12;
            c12 = *d2;
            *d2 = *d1;
            c13 = c23;
            c23 = c31;
        }
        _ => {
            c21 = c12;
            c31 = *d1;
            c32 = c12;
            *d1 = c13;
            c12 = c23;
            c13 = *d3;
            *d3 = *d1;
        }
    }
    row = kernel_vector3g_aux(mode, d1, d2, d3, &mut c12, &mut c13, &mut c23, &mut c21, &mut c31, &mut c32);

    match col {
        2 => {
            std::mem::swap(d1, d2);
        }
        3 => {
            std::mem::swap(d1, d3);
        }
        _ => {}
    }
    row
}

/// `internal::Reduce3S` (kernels.hpp).  The matrix entries are modified in
/// place; the eigenvector (z1,z2,z3) is also modified.
#[allow(clippy::too_many_arguments)]
fn reduce3s(
    mode: usize,
    d1: &mut f64,
    d2: &mut f64,
    d3: &mut f64,
    d12: &mut f64,
    d13: &mut f64,
    d23: &mut f64,
    z1: &mut f64,
    z2: &mut f64,
    z3: &mut f64,
    v1: &mut f64,
    v2: &mut f64,
    v3: &mut f64,
    g: &mut f64,
) -> usize {
    let k;
    let (mut s, mut w1, mut w2, mut w3);

    if mode == 0 {
        if z1.abs() <= z3.abs() {
            k = if z1.abs() <= z2.abs() { 1 } else { 2 };
        } else {
            k = if z2.abs() <= z3.abs() { 2 } else { 3 };
        }
    } else if z1.abs() >= z3.abs() {
        k = if z1.abs() >= z2.abs() { 1 } else { 2 };
    } else {
        k = if z2.abs() >= z3.abs() { 2 } else { 3 };
    }
    match k {
        2 => {
            std::mem::swap(d13, d23);
            std::mem::swap(d1, d2);
            std::mem::swap(z1, z2);
        }
        3 => {
            std::mem::swap(d12, d23);
            std::mem::swap(d1, d3);
            std::mem::swap(z1, z3);
        }
        _ => {}
    }

    s = (*z2).hypot(*z3);

    if s == 0.0 {
        *v1 = 0.0;
        *v2 = 0.0;
        *v3 = 0.0;
        *g = 1.0;
    } else {
        *g = 1.0_f64.copysign(*z1);
        *v1 = -s * (s / (*z1 + *g)); // = z1 - g
        // normalize (v1,z2,z3) by its max-norm
        *g = v1.abs();
        if z2.abs() > *g {
            *g = z2.abs();
        }
        if z3.abs() > *g {
            *g = z3.abs();
        }
        *v1 /= *g;
        *v2 = *z2 / *g;
        *v3 = *z3 / *g;
        *g = 2.0 / (*v1 * *v1 + *v2 * *v2 + *v3 * *v3);

        // w = u - (g/2)(v^t u) v,  u = g A v
        w1 = *g * (*d1 * *v1 + *d12 * *v2 + *d13 * *v3);
        w2 = *g * (*d12 * *v1 + *d2 * *v2 + *d23 * *v3);
        w3 = *g * (*d13 * *v1 + *d23 * *v2 + *d3 * *v3);
        s = (*g / 2.0) * (*v1 * w1 + *v2 * w2 + *v3 * w3);
        w1 -= s * *v1;
        w2 -= s * *v2;
        w3 -= s * *v3;
        *d1 -= 2.0 * *v1 * w1;
        *d2 -= 2.0 * *v2 * w2;
        *d23 -= *v2 * w3 + *v3 * w2;
        *d3 -= 2.0 * *v3 * w3;
    }

    match k {
        2 => {
            std::mem::swap(z1, z2);
        }
        3 => {
            std::mem::swap(z1, z3);
        }
        _ => {}
    }
    k
}

/// MFEM `kernels::CalcSingularvalue<3>`: the `i`-th largest singular value of
/// the 3×3 matrix given in **row-major** order (`data`).  `i` ∈ {0, 1, 2}.
pub fn calc_singularvalue_3(data: &[f64; 9], i: usize) -> f64 {
    let mut d0 = data[0];
    let mut d3 = data[3];
    let mut d6 = data[6];
    let mut d1 = data[1];
    let mut d4 = data[4];
    let mut d7 = data[7];
    let mut d2 = data[2];
    let mut d5 = data[5];
    let mut d8 = data[8];

    let mut d_max = d0.abs();
    for &v in &[d1, d2, d3, d4, d5, d6, d7, d8] {
        if d_max < v.abs() {
            d_max = v.abs();
        }
    }
    let mult = get_scaling_factor(d_max);

    d0 /= mult;
    d1 /= mult;
    d2 /= mult;
    d3 /= mult;
    d4 /= mult;
    d5 /= mult;
    d6 /= mult;
    d7 /= mult;
    d8 /= mult;

    let b11 = d0 * d0 + d1 * d1 + d2 * d2;
    let b12 = d0 * d3 + d1 * d4 + d2 * d5;
    let b13 = d0 * d6 + d1 * d7 + d2 * d8;
    let b22 = d3 * d3 + d4 * d4 + d5 * d5;
    let b23 = d3 * d6 + d4 * d7 + d5 * d8;
    let b33 = d6 * d6 + d7 * d7 + d8 * d8;

    let mut aa = (b11 + b22 + b33) / 3.0; // tr(B)/3
    let mut c1;
    let mut c2;
    let mut c3;
    {
        let b11_b22 = (d0 - d3) * (d0 + d3) + (d1 - d4) * (d1 + d4) + (d2 - d5) * (d2 + d5);
        let b22_b33 = (d3 - d6) * (d3 + d6) + (d4 - d7) * (d4 + d7) + (d5 - d8) * (d5 + d8);
        let b33_b11 = (d6 - d0) * (d6 + d0) + (d7 - d1) * (d7 + d1) + (d8 - d2) * (d8 + d2);
        c1 = (b11_b22 - b33_b11) / 3.0;
        c2 = (b22_b33 - b11_b22) / 3.0;
        c3 = (b33_b11 - b22_b33) / 3.0;
    }
    let q = (2.0 * (b12 * b12 + b13 * b13 + b23 * b23) + c1 * c1 + c2 * c2 + c3 * c3) / 6.0;
    let r = (c1 * (b23 * b23 - c2 * c3) + b12 * (b12 * c3 - 2.0 * b13 * b23) + b13 * b13 * c2) / 2.0;

    let mut have_aa = q <= 0.0;
    let mut r_shift = 0.0;
    if !have_aa {
        let sqrt_q = q.sqrt();
        let sqrt_q3 = q * sqrt_q;
        let mut r_val = 0.0;
        if r.abs() >= sqrt_q3 {
            r_val = if r < 0.0 { 2.0 * sqrt_q } else { -2.0 * sqrt_q };
        } else {
            let rr = r / sqrt_q3;
            if rr.abs() <= 0.9 {
                if i == 2 {
                    aa -= 2.0 * sqrt_q * (rr.acos() / 3.0).cos(); // min
                } else if i == 0 {
                    aa -= 2.0 * sqrt_q * ((rr.acos() + 2.0 * std::f64::consts::PI) / 3.0).cos(); // max
                } else {
                    aa -= 2.0 * sqrt_q * ((rr.acos() - 2.0 * std::f64::consts::PI) / 3.0).cos(); // mid
                }
                have_aa = true;
            } else if rr < 0.0 {
                r_val = -2.0 * sqrt_q * ((rr.acos() + 2.0 * std::f64::consts::PI) / 3.0).cos(); // max
                if i == 0 {
                    aa += r_val;
                    have_aa = true;
                }
            } else {
                r_val = -2.0 * sqrt_q * (rr.acos() / 3.0).cos(); // min
                if i == 2 {
                    aa += r_val;
                    have_aa = true;
                }
            }
        }
        if !have_aa {
            r_shift = r_val;
            c1 -= r_val;
            c2 -= r_val;
            c3 -= r_val;
        }
    }

    if !have_aa {
        // mode = 1 (largest absolute value)
        let mode = 1usize;
        let mut b11 = b11;
        let mut b22 = b22;
        let mut b33 = b33;
        let mut b12 = b12;
        let mut b13 = b13;
        let mut b23 = b23;
        let kdim = kernel_vector3s(mode, b12, b13, b23, &mut c1, &mut c2, &mut c3);
        if kdim == 3 {
            aa += r_shift;
        } else {
            let mut v1 = 0.0;
            let mut v2 = 0.0;
            let mut v3 = 0.0;
            let mut g = 0.0;
            reduce3s(
                mode,
                &mut b11,
                &mut b22,
                &mut b33,
                &mut b12,
                &mut b13,
                &mut b23,
                &mut c1,
                &mut c2,
                &mut c3,
                &mut v1,
                &mut v2,
                &mut v3,
                &mut g,
            );
            eigenvalues2s(&mut b23, &mut b22, &mut b33);
            if i == 2 {
                aa = b11.min(b22).min(b33);
            } else if i == 1 {
                aa = if b11 <= b22 {
                    if b22 <= b33 {
                        b22
                    } else {
                        b11.max(b33)
                    }
                } else if b11 <= b33 {
                    b11
                } else {
                    b33.max(b22)
                };
            } else {
                aa = b11.max(b22).max(b33);
            }
        }
    }

    aa.abs().sqrt() * mult
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: identity → singular values 1,1,1.
    #[test]
    fn identity() {
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!((calc_singularvalue_3(&id, 0) - 1.0).abs() < 1e-14);
        assert!((calc_singularvalue_3(&id, 2) - 1.0).abs() < 1e-14);
    }

    /// Diagonal matrix diag(3, 2, 1) → σ0 = 3, σ2 = 1.
    #[test]
    fn diagonal() {
        let d = [3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        assert!((calc_singularvalue_3(&d, 0) - 3.0).abs() < 1e-13);
        assert!((calc_singularvalue_3(&d, 2) - 1.0).abs() < 1e-13);
    }

    /// MFEM probe ground truth for the marked tet0 of fichera-mixed:
    /// verts {(0,0,1),(1,0,0),(0,1,0),(1,1,1)} — the C++ CalcSingularvalue<3>
    /// gives σ0/σ2 per rt candidate: rt0=2, rt1=2.0000000000000004, rt2=2.
    #[test]
    fn marked_tet0_matches_mfem() {
        let v: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
        let mut j = [[0.0f64; 3]; 3];
        for t in 0..3 {
            for s in 0..3 {
                j[t][s] = v[s + 1][t] - v[0][t];
            }
        }
        let mut em = [[0.0f64; 6]; 3];
        for s in 0..3 {
            for t in 0..3 {
                em[t][s] = 0.5 * j[t][s];
            }
        }
        for t in 0..3 {
            em[t][3] = 0.5 * (j[t][0] + j[t][1]);
            em[t][4] = 0.5 * (j[t][0] + j[t][2]);
            em[t][5] = 0.5 * (j[t][1] + j[t][2]);
        }
        let cand: [[usize; 4]; 6] = [
            [0, 5, 1, 2], [0, 5, 2, 4],
            [1, 0, 4, 2], [1, 2, 4, 5],
            [2, 0, 1, 3], [2, 1, 5, 3],
        ];
        let perf_inv = [
            [1.0, -0.57735026918962584, -0.40824829046386302],
            [0.0, 1.1547005383792517, -0.40824829046386302],
            [0.0, 0.0, 1.2247448713915892],
        ];
        let mut kappas = [0.0f64; 3];
        for c in 0..3 {
            let mut kmax = 0.0f64;
            for k in 0..2 {
                let [b, a0, a1, a2] = cand[2 * c + k];
                let mut js = [[0.0f64; 3]; 3];
                for t in 0..3 {
                    js[t][0] = em[t][a0] - em[t][b];
                    js[t][1] = em[t][a1] - em[t][b];
                    js[t][2] = em[t][a2] - em[t][b];
                }
                let mut jp = [[0.0f64; 3]; 3];
                for t in 0..3 {
                    for cc in 0..3 {
                        jp[t][cc] = js[t][0] * perf_inv[0][cc]
                            + js[t][1] * perf_inv[1][cc]
                            + js[t][2] * perf_inv[2][cc];
                    }
                }
                let data: [f64; 9] = [
                    jp[0][0], jp[1][0], jp[2][0],
                    jp[0][1], jp[1][1], jp[2][1],
                    jp[0][2], jp[1][2], jp[2][2],
                ];
                let ar = calc_singularvalue_3(&data, 0) / calc_singularvalue_3(&data, 2);
                eprintln!("rt={c} k={k}: ar={ar:.17e}");
                kmax = kmax.max(ar);
            }
            kappas[c] = kmax;
            eprintln!("rt={c} kappa={kmax:.17e}");
        }
        // C++ ground truth: rt0=2, rt1=2.0000000000000004, rt2=2 → rt=0.
        assert!(kappas[1] > kappas[0], "rt1 kappa {:.17e} should exceed rt0 {:.17e}", kappas[1], kappas[0]);
        assert!((kappas[0] - 2.0).abs() < 1e-14);
        assert!((kappas[2] - 2.0).abs() < 1e-14);
    }
}
