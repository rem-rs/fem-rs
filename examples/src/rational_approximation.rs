//! AAA rational approximation of `f(z) = z^{-α}` (MFEM `examples/ex33.hpp`).
//!
//! Shared by `mfem_ex33`, `mfem_pex33` and the `spde/generate_random_field`
//! miniapp — mirroring how the C++ miniapp includes `ex33.hpp` textually.

use nalgebra::{DMatrix, Schur, SVD};

/// `RationalApproximation_AAA`: rational approximation of data `val` at points
/// `pt` in rational barycentric form (support points `z`, data `f`, weights `w`).
///
/// See pg. A1501 of Nakatsukasa et al. [1] and MFEM `ex33.hpp`.
pub fn rational_approximation_aaa(
    val: &[f64],
    pt: &[f64],
    tol: f64,
    max_order: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let size = val.len();
    assert_eq!(pt.len(), size, "size mismatch");

    // Initializations
    let mut j: Vec<usize> = (0..size).collect();
    let mut z: Vec<f64> = Vec::new();
    let mut f: Vec<f64> = Vec::new();
    let mut c_i: Vec<f64> = Vec::new(); // flattened Cauchy-matrix columns (col-major)

    // R(.) = mean of the value vector
    let mean_val = val.iter().sum::<f64>() / size as f64;
    let mut r = vec![mean_val; size];

    let mut w: Vec<f64> = Vec::new();
    for k in 0..max_order {
        // select next support point
        let mut idx = 0usize;
        let mut tmp_max = 0.0_f64;
        for (jj, &vj) in val.iter().enumerate() {
            let tmp = (vj - r[jj]).abs();
            if tmp > tmp_max {
                tmp_max = tmp;
                idx = jj;
            }
        }

        // Append support points and data values
        z.push(pt[idx]);
        f.push(val[idx]);

        // Update index vector (J.DeleteFirst(idx))
        if let Some(pos) = j.iter().position(|&x| x == idx) {
            j.remove(pos);
        }

        // next column in the Cauchy matrix
        for jj in 0..size {
            c_i.push(1.0 / (pt[jj] - pt[idx]));
        }
        let w_c = k + 1;

        // C = size×(k+1) column-major view of the accumulated Cauchy columns.
        let c = DMatrix::from_vec(size, w_c, c_i.clone());
        let mut ctemp = c.clone();

        // Ctemp.InvLeftScaling(val): Ctemp(i,j) /= val(i)
        for i in 0..size {
            let vi = val[i];
            for jj in 0..w_c {
                ctemp[(i, jj)] /= vi;
            }
        }
        // Ctemp.RightScaling(f): Ctemp(i,j) *= f(j)
        for jj in 0..w_c {
            let fj = f[jj];
            for i in 0..size {
                ctemp[(i, jj)] *= fj;
            }
        }

        // A = C - Ctemp, then A.LeftScaling(val): A(i,j) *= val(i)
        let mut a = c.clone() - ctemp;
        for i in 0..size {
            let vi = val[i];
            for jj in 0..w_c {
                a[(i, jj)] *= vi;
            }
        }

        // Am = A(J rows, all columns)
        let h_am = j.len();
        let mut am = DMatrix::zeros(h_am, w_c);
        for (i, &ii) in j.iter().enumerate() {
            for jj in 0..w_c {
                am[(i, jj)] = a[(ii, jj)];
            }
        }

        // SVD: w = Vt row k (= last column of V, the minimal singular vector).
        let svd = SVD::new(am, false, true);
        let vt = svd.v_t.expect("SVD with compute_v=true must return Vt");
        w = vt.row(k).iter().cloned().collect();

        // N = C·(w .* f), D = C·w
        let mut n = vec![0.0_f64; size];
        let mut d = vec![0.0_f64; size];
        for i in 0..size {
            for jj in 0..w_c {
                n[i] += c[(i, jj)] * w[jj] * f[jj];
                d[i] += c[(i, jj)] * w[jj];
            }
        }

        // R = val; R(ii) = N(ii)/D(ii) for ii in J
        r.copy_from_slice(val);
        for &ii in &j {
            r[ii] = n[ii] / d[ii];
        }

        // verr = val - R
        let mut verr_max = 0.0_f64;
        let mut val_norm_linf = 0.0_f64;
        for i in 0..size {
            verr_max = verr_max.max((val[i] - r[i]).abs());
            val_norm_linf = val_norm_linf.max(val[i].abs());
        }
        if verr_max <= tol * val_norm_linf {
            break;
        }
    }

    (z, f, w)
}

/// Roots of the polynomial `P(λ) = c0 + c1·λ + … + cn·λ^n` (coeffs ascending,
/// `cn ≠ 0`), keeping only the real parts (MFEM's dggev path discards the
/// imaginary parts of the finite generalized eigenvalues).
pub fn polynomial_roots_real(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len() - 1; // degree
    if n == 0 {
        return Vec::new();
    }
    let cn = coeffs[n];

    // Companion matrix (ones on the subdiagonal):
    //   [ -c(n-1)/cn  -c(n-2)/cn  …  -c0/cn ]
    //   [   1            0         …    0    ]
    //   [   0            1         …    0    ]
    //   …                                      (eigenvalues = roots of P)
    let mut m = DMatrix::<f64>::zeros(n, n);
    for jj in 0..n {
        m[(0, jj)] = -coeffs[n - 1 - jj] / cn;
    }
    for i in 1..n {
        m[(i, i - 1)] = 1.0;
    }

    let (_q, t) = Schur::new(m).unpack();
    let mut roots = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1, i)].abs() > 0.0 {
            // Real Schur 2×2 block: λ = tr/2 ± sqrt((tr/2)² - det)
            let (a, b, cc, dd) = (t[(i, i)], t[(i, i + 1)], t[(i + 1, i)], t[(i + 1, i + 1)]);
            let tr = (a + dd) / 2.0;
            let disc = ((a - dd) / 2.0).powi(2) + b * cc;
            if disc >= 0.0 {
                let s = disc.sqrt();
                roots.push(tr + s);
                roots.push(tr - s);
            } else {
                // complex pair: keep the real part only (MFEM EigenvaluesRealPart)
                roots.push(tr);
                roots.push(tr);
            }
            i += 2;
        } else {
            roots.push(t[(i, i)]);
            i += 1;
        }
    }
    roots
}

/// Coefficients (ascending powers) of `Σ_j weights[j]·∏_{k≠j}(λ - z[k])`
/// — the characteristic polynomial of the (E,B) pencil from MFEM's
/// `ComputePolesAndZeros` (its finite generalized eigenvalues).
pub fn weighted_poly_product(z: &[f64], weights: &[f64]) -> Vec<f64> {
    let m = z.len();
    assert_eq!(weights.len(), m, "weight/z size mismatch");
    let mut acc = vec![0.0_f64; m]; // degree m-1
    for jj in 0..m {
        let mut p = vec![1.0_f64];
        for (kk, &zk) in z.iter().enumerate() {
            if kk == jj {
                continue;
            }
            let mut q = vec![0.0_f64; p.len() + 1];
            for (i, &cc) in p.iter().enumerate() {
                q[i] += -zk * cc;
                q[i + 1] += cc;
            }
            p = q;
        }
        for (i, &cc) in p.iter().enumerate() {
            acc[i] += weights[jj] * cc;
        }
    }
    acc
}

/// `ComputePolesAndZeros` + `PartialFractionExpansion` (ex33.hpp): given the
/// barycentric form (z, f, w), return `(poles, coeffs)` of the partial-fraction
/// expansion `f(z) ≈ Σ_i c_i/(z - p_i)` for `f = z^{-α}`.
///
/// The poles/zeros are the finite real parts of the generalized eigenvalues of
/// the (E,B) pencil built in `ComputePolesAndZeros`; since B = diag(0,1,…,1)
/// the finite eigenvalues are exactly the roots of
/// `Σ_j w_j ∏_{k≠j}(λ - z_k) = 0` (poles) and `Σ_j w_j f_j ∏_{k≠j}(λ - z_k) = 0`
/// (zeros).  The exact zero root of the latter (z=0 is a support point with
/// f=0 ⇒ polynomial constant term is exactly 0) is removed, matching
/// `zeros.DeleteFirst(0.0)`.
pub fn poles_and_coeffs_from_barycentric(z: &[f64], f: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // scale = w·f / Σw
    let scale = w
        .iter()
        .zip(f.iter())
        .map(|(&wi, &fi)| wi * fi)
        .sum::<f64>()
        / w.iter().sum::<f64>();

    // poles: roots of Σ_j w_j ∏_{k≠j}(λ - z_k)
    let pole_poly = weighted_poly_product(z, w);
    let poles = polynomial_roots_real(&pole_poly);

    // zeros: roots of Σ_j (w_j f_j) ∏_{k≠j}(λ - z_k); drop the exact zero root
    let wf: Vec<f64> = w.iter().zip(f.iter()).map(|(&wi, &fi)| wi * fi).collect();
    let mut zero_poly = weighted_poly_product(z, &wf);
    if zero_poly[0] == 0.0 {
        // P(λ) = λ·Q(λ): divide by λ (zeros.DeleteFirst(0.0))
        zero_poly.remove(0);
    }
    let zeros = polynomial_roots_real(&zero_poly);

    // PartialFractionExpansion: c_i = scale·∏_j(p_i-z_j)/∏_{k≠i}(p_i-p_k)
    let psize = poles.len();
    let zsize = zeros.len();
    let mut coeffs = vec![scale; psize];
    for i in 0..psize {
        let mut tmp_numer = 1.0;
        for jj in 0..zsize {
            tmp_numer *= poles[i] - zeros[jj];
        }
        let mut tmp_denom = 1.0;
        for kk in 0..psize {
            if kk != i {
                tmp_denom *= poles[i] - poles[kk];
            }
        }
        coeffs[i] *= tmp_numer / tmp_denom;
    }

    (poles, coeffs)
}

/// `ComputePartialFractionApproximation` (ex33.hpp): rational approximation of
/// `f(z) = z^{-α}`, `0 < α < 1`, in partial-fraction form.  Defaults match the
/// MFEM call in ex33.cpp: lmax=1000, tol=1e-10, npoints=1000, max_order=100.
/// Returns `(coeffs, poles)` with `d_i = -poles[i] > 0`.
pub fn compute_partial_fraction_approximation(alpha: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(alpha < 1.0, "alpha must be less than 1");
    assert!(alpha > 0.0, "alpha must be greater than 0");

    let lmax = 1000.0_f64;
    let tol = 1e-10_f64;
    let npoints = 1000usize;
    let max_order = 100usize;
    assert!(npoints > 2, "npoints must be greater than 2");
    assert!(lmax > 0.0, "lmax must be greater than 0");
    assert!(tol > 0.0, "tol must be greater than 0");

    // Sample f(x) = x^{1-α} uniformly on [0, lmax].
    let dx = lmax / (npoints - 1) as f64;
    let x: Vec<f64> = (0..npoints).map(|i| dx * i as f64).collect();
    let val: Vec<f64> = x.iter().map(|&xi| xi.powf(1.0 - alpha)).collect();

    // Triple-A algorithm on f(x) = x^{1-a}.
    let (z, f, w) = rational_approximation_aaa(&val, &x, tol, max_order);

    // Poles, zeros and the partial-fraction expansion of f(z) = z^{-a}.
    let (poles, coeffs) = poles_and_coeffs_from_barycentric(&z, &f, &w);
    (coeffs, poles)
}
