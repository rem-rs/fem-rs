//! Round-102: FD verification of the 3D invariant Hessian-assembly kernels.
//!
//! Each `assemble_dd_*` kernel is checked against a central finite difference
//! of the corresponding first-derivative array (`get_di*`), contracted with a
//! deterministic basis-derivative matrix `DS` and weight, using MFEM's
//! contraction convention `A(i+nd*j, k+nd*l) = Σ_st w DS(i,s) ddI(j,s,l,t) DS(k,t)`
//! with `ddI(a,b,c,d) = ∂²I/∂J_ab∂J_cd`.
//!
//! These tests pinned the round-102 fix of `assemble_dd_i3b` (was a plain
//! rank-1 outer product; MFEM's kernel is the antisymmetrized "determinant"
//! pattern) and gate the ddI1b/ddI2/ddI2b kernels against regressions.

use fem_mesh::tmop::InvariantsEvaluator3D;

const ND: usize = 8;

/// Deterministic non-symmetric test Jacobian (det > 0).
fn test_jac() -> [f64; 9] {
    // column-major: j[r + 3c]
    let j = [
        1.1, 0.23, -0.13, //
        0.31, 0.87, 0.41, //
        -0.07, 0.19, 1.23,
    ];
    // Ensure det > 0 (it is: ~1.09).
    j
}

/// Deterministic DS matrix (nd x 3, column-major) and weight.
fn test_ds() -> (Vec<f64>, f64) {
    let mut ds = vec![0.0; ND * 3];
    for i in 0..ND {
        for c in 0..3 {
            ds[i + c * ND] = ((7 * i + 3 * c) % 11) as f64 / 11.0 - 0.5;
        }
    }
    (ds, 0.7)
}

/// Central FD Hessian of the first-derivative array `di` (length 9, col-major
/// entries of J): `dd[(a + 3*b) * 9 + (c + 3*d)] = ∂di[a + 3b]/∂J[c + 3d]`.
fn fd_hessian<F>(di: F) -> Vec<f64>
where
    F: Fn(&[f64; 9]) -> [f64; 9],
{
    let h = 1e-6;
    let j0 = test_jac();
    let mut dd = vec![0.0; 81];
    for c in 0..9 {
        let mut jp = j0;
        let mut jm = j0;
        jp[c] += h;
        jm[c] -= h;
        let dp = di(&jp);
        let dm = di(&jm);
        for a in 0..9 {
            dd[a * 9 + c] = (dp[a] - dm[a]) / (2.0 * h);
        }
    }
    dd
}

/// Contract `dd` (FD Hessian) with DS like MFEM's kernels do and compare with
/// the kernel output accumulated into a zero matrix.
fn check_kernel(name: &str, assemble: impl FnOnce(f64, &mut [f64]), dd: &[f64]) {
    let (ds, w) = test_ds();
    let mut a = vec![0.0; ND * 3 * ND * 3];
    assemble(w, &mut a);
    let ah = ND * 3;
    let mut afd = vec![0.0; ND * 3 * ND * 3];
    for j in 0..3 {
        for s in 0..3 {
            for l in 0..3 {
                for t in 0..3 {
                    let g = dd[(j + 3 * s) * 9 + (l + 3 * t)];
                    if g == 0.0 {
                        continue;
                    }
                    for i in 0..ND {
                        for k in 0..ND {
                            afd[i + ND * j + (k + ND * l) * ah] +=
                                w * ds[i + s * ND] * ds[k + t * ND] * g;
                        }
                    }
                }
            }
        }
    }
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for k in 0..a.len() {
        worst = worst.max((a[k] - afd[k]).abs());
        scale = scale.max(afd[k].abs());
    }
    let rel = if scale > 0.0 { worst / scale } else { worst };
    println!("{name}: worst abs={worst:.3e} scale={scale:.3e} rel={rel:.3e}");
    // FD noise floor for central differences of quadratic-form outputs.
    assert!(
        rel < 1e-6,
        "{name}: assembled Hessian disagrees with FD: rel={rel}"
    );
}

#[test]
fn d102_kernel_dd_i1_3d() {
    let jac = test_jac();
    let mut ie = InvariantsEvaluator3D::new(Some(&jac));
    ie.set_derivative_matrix(ND, &test_ds().0);
    let dd = fd_hessian(|j| {
        let mut e = InvariantsEvaluator3D::new(Some(j));
        *e.get_di1()
    });
    check_kernel("Assemble_ddI1", |w, a| ie.assemble_dd_i1(w, a), &dd);
}

#[test]
fn d102_kernel_dd_i2_3d() {
    let jac = test_jac();
    let mut ie = InvariantsEvaluator3D::new(Some(&jac));
    ie.set_derivative_matrix(ND, &test_ds().0);
    let dd = fd_hessian(|j| {
        let mut e = InvariantsEvaluator3D::new(Some(j));
        *e.get_di2()
    });
    check_kernel("Assemble_ddI2", |w, a| ie.assemble_dd_i2(w, a), &dd);
}

#[test]
fn d102_kernel_dd_i2b_3d() {
    let jac = test_jac();
    let mut ie = InvariantsEvaluator3D::new(Some(&jac));
    ie.set_derivative_matrix(ND, &test_ds().0);
    let dd = fd_hessian(|j| {
        let mut e = InvariantsEvaluator3D::new(Some(j));
        *e.get_di2b()
    });
    check_kernel("Assemble_ddI2b", |w, a| ie.assemble_dd_i2b(w, a), &dd);
}

#[test]
fn d102_kernel_dd_i1b_3d() {
    let jac = test_jac();
    let mut ie = InvariantsEvaluator3D::new(Some(&jac));
    ie.set_derivative_matrix(ND, &test_ds().0);
    let dd = fd_hessian(|j| {
        let mut e = InvariantsEvaluator3D::new(Some(j));
        *e.get_di1b()
    });
    check_kernel("Assemble_ddI1b", |w, a| ie.assemble_dd_i1b(w, a), &dd);
}

#[test]
fn d102_kernel_dd_i3b_3d() {
    let jac = test_jac();
    let mut ie = InvariantsEvaluator3D::new(Some(&jac));
    ie.set_derivative_matrix(ND, &test_ds().0);
    let dd = fd_hessian(|j| {
        let mut e = InvariantsEvaluator3D::new(Some(j));
        *e.get_di3b()
    });
    check_kernel("Assemble_ddI3b", |w, a| ie.assemble_dd_i3b(w, a), &dd);
}

#[test]
fn d102_kernel_dd_i3_3d() {
    let jac = test_jac();
    let mut ie = InvariantsEvaluator3D::new(Some(&jac));
    ie.set_derivative_matrix(ND, &test_ds().0);
    let dd = fd_hessian(|j| {
        let mut e = InvariantsEvaluator3D::new(Some(j));
        *e.get_di3()
    });
    check_kernel("Assemble_ddI3", |w, a| ie.assemble_dd_i3(w, a), &dd);
}

#[test]
fn d102_kernel_tprod_3d() {
    // TProd(X, Y) = w * D (X Y^t + Y X^t)/... against FD of the quadratic form.
    // d²/dJ² of (1/2)(x·dI)(y·dI)-like forms is awkward; instead verify TProd
    // contraction directly: A(i+nd*j,k+nd*l) = w Σ_st D_is (DXt_jt DYt_lt +
    // DYt_jt DXt_lt) D_kt with DXt = D X^t.
    let jac = test_jac();
    let (ds, w) = test_ds();
    let mut ie = InvariantsEvaluator3D::new(Some(&jac));
    ie.set_derivative_matrix(ND, &ds);
    let x = [0.3, -0.2, 0.11, 0.4, 0.15, -0.31, 0.22, 0.05, 0.4];
    let y = [0.21, 0.13, -0.4, 0.33, -0.12, 0.2, -0.05, 0.44, 0.1];
    let mut a = vec![0.0; ND * 3 * ND * 3];
    ie.assemble_tprod_xy(w, &x, &y, &mut a);
    // Reference: DXt = D X^t (nd x 3), DYt = D Y^t.
    let mut dxt = vec![0.0; ND * 3];
    let mut dyt = vec![0.0; ND * 3];
    for i in 0..ND {
        for t in 0..3 {
            // X col-major: X(r,c) = x[r + 3c]; (D X^t)(i,t) = Σ_s D(i,s) X(t,s)
            dxt[i + t * ND] = (0..3).map(|s| ds[i + s * ND] * x[t + 3 * s]).sum();
            dyt[i + t * ND] = (0..3).map(|s| ds[i + s * ND] * y[t + 3 * s]).sum();
        }
    }
    let ah = ND * 3;
    let mut aref = vec![0.0; ah * ah];
    // MFEM convention: A(i+nd*j,k+nd*l) = w [DXt(i,j) DYt(k,l) + DYt(i,j)
    // DXt(k,l)] with DZt(dof,comp) = Σ_s D(dof,s) Z(comp,s).
    for j in 0..3 {
        for l in 0..3 {
            for i in 0..ND {
                for k in 0..ND {
                    let g = dxt[i + ND * j] * dyt[k + ND * l]
                        + dyt[i + ND * j] * dxt[k + ND * l];
                    aref[i + ND * j + (k + ND * l) * ah] = w * g;
                }
            }
        }
    }
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for k in 0..a.len() {
        worst = worst.max((a[k] - aref[k]).abs());
        scale = scale.max(aref[k].abs());
    }
    let rel = if scale > 0.0 { worst / scale } else { worst };
    println!("TProd(X,Y): worst abs={worst:.3e} scale={scale:.3e} rel={rel:.3e}");
    assert!(rel < 1e-12, "TProd(X,Y) mismatch: rel={rel}");
}
