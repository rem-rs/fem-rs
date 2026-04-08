//! Integration tests for the fem-linalg crate.
//!
//! Tests CooMatrix → CsrMatrix conversion, sparse matrix arithmetic,
//! and dense Vector operations.

use fem_linalg::{CooMatrix, CsrMatrix, Vector, spadd};

// ─── CooMatrix / CsrMatrix ────────────────────────────────────────────────────

#[test]
fn coo_to_csr_spmv_correctness() {
    // Build the 3×3 tridiagonal matrix
    // [ 2 -1  0 ]
    // [-1  2 -1 ]
    // [ 0 -1  2 ]
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.add(0, 0, 2.0); coo.add(0, 1, -1.0);
    coo.add(1, 0, -1.0); coo.add(1, 1, 2.0); coo.add(1, 2, -1.0);
    coo.add(2, 1, -1.0); coo.add(2, 2, 2.0);
    let csr = coo.into_csr();

    // A * [1, 1, 1] = [1, 0, 1]
    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![0.0; 3];
    csr.spmv(&x, &mut y);
    assert!((y[0] - 1.0).abs() < 1e-14);
    assert!(y[1].abs() < 1e-14);
    assert!((y[2] - 1.0).abs() < 1e-14);
}

#[test]
fn coo_to_csr_dimensions() {
    let mut coo = CooMatrix::<f64>::new(5, 7);
    coo.add(0, 6, 1.0);
    coo.add(4, 0, 2.0);
    let csr = coo.into_csr();
    assert_eq!(csr.nrows, 5);
    assert_eq!(csr.ncols, 7);
    assert_eq!(csr.nnz(), 2);
}

#[test]
fn coo_duplicate_entries_are_summed() {
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.add(0, 0, 1.0);
    coo.add(0, 0, 2.0); // duplicate → should sum to 3
    coo.add(1, 1, 4.0);
    let csr = coo.into_csr();
    assert!((csr.get(0, 0) - 3.0).abs() < 1e-14);
    assert!((csr.get(1, 1) - 4.0).abs() < 1e-14);
}

#[test]
fn csr_add_method_equals_free_function() {
    let mut ca = CooMatrix::<f64>::new(3, 3);
    ca.add(0, 0, 1.0); ca.add(1, 1, 2.0); ca.add(2, 2, 3.0);
    let a = ca.into_csr();

    let mut cb = CooMatrix::<f64>::new(3, 3);
    cb.add(0, 1, 5.0); cb.add(1, 2, 6.0);
    let b = cb.into_csr();

    let c_method = a.add(&b);
    let c_free   = spadd(&a, &b);

    let dm = c_method.to_dense();
    let df = c_free.to_dense();
    assert_eq!(dm.len(), df.len());
    for (m, f) in dm.iter().zip(df.iter()) {
        assert!((m - f).abs() < 1e-14, "add method ≠ free function: {m} vs {f}");
    }
}

#[test]
fn csr_add_asymmetric_patterns() {
    // A has lower triangle, B has upper triangle → C has both
    let mut ca = CooMatrix::<f64>::new(3, 3);
    ca.add(1, 0, 10.0); ca.add(2, 0, 20.0); ca.add(2, 1, 30.0);
    let a = ca.into_csr();

    let mut cb = CooMatrix::<f64>::new(3, 3);
    cb.add(0, 1, 40.0); cb.add(0, 2, 50.0); cb.add(1, 2, 60.0);
    let b = cb.into_csr();

    let c = a.add(&b);
    // Lower triangle
    assert!((c.get(1, 0) - 10.0).abs() < 1e-14);
    assert!((c.get(2, 0) - 20.0).abs() < 1e-14);
    assert!((c.get(2, 1) - 30.0).abs() < 1e-14);
    // Upper triangle
    assert!((c.get(0, 1) - 40.0).abs() < 1e-14);
    assert!((c.get(0, 2) - 50.0).abs() < 1e-14);
    assert!((c.get(1, 2) - 60.0).abs() < 1e-14);
    // Diagonal should be zero
    for i in 0..3 {
        assert!(c.get(i, i).abs() < 1e-14, "c[{i},{i}] should be 0");
    }
}

#[test]
fn csr_axpby_linear_combination() {
    // axpby(2, B, 3) = 2*A + 3*B for non-overlapping patterns
    let mut ca = CooMatrix::<f64>::new(2, 2);
    ca.add(0, 0, 1.0); ca.add(1, 1, 2.0);
    let a = ca.into_csr();

    let mut cb = CooMatrix::<f64>::new(2, 2);
    cb.add(0, 1, 3.0); cb.add(1, 0, 4.0);
    let b = cb.into_csr();

    let c = a.axpby(2.0, &b, 3.0);
    assert!((c.get(0, 0) - 2.0).abs() < 1e-14); // 2*1
    assert!((c.get(1, 1) - 4.0).abs() < 1e-14); // 2*2
    assert!((c.get(0, 1) - 9.0).abs() < 1e-14); // 3*3
    assert!((c.get(1, 0) - 12.0).abs() < 1e-14); // 3*4
}

#[test]
fn csr_add_commutativity_on_symmetric() {
    // For symmetric A = B: a.add(&b) and b.add(&a) should be the same
    let mut ca = CooMatrix::<f64>::new(3, 3);
    ca.add(0, 0, 1.0); ca.add(0, 1, 2.0); ca.add(1, 0, 2.0); ca.add(1, 1, 3.0);
    let a = ca.into_csr();
    let b = a.clone();

    let ab = a.add(&b);
    let ba = b.add(&a);
    let dab = ab.to_dense();
    let dba = ba.to_dense();
    for (x, y) in dab.iter().zip(dba.iter()) {
        assert!((x - y).abs() < 1e-14);
    }
}

// ─── Vector ───────────────────────────────────────────────────────────────────

#[test]
fn vector_dot_product() {
    let a = Vector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    let d = a.dot(&b); // 1*4 + 2*5 + 3*6 = 32
    assert!((d - 32.0).abs() < 1e-14, "dot = {d}, expected 32");
}

#[test]
fn vector_norm() {
    let a = Vector::from_vec(vec![3.0_f64, 4.0]);
    let n = a.norm(); // sqrt(9+16) = 5
    assert!((n - 5.0).abs() < 1e-14, "norm = {n}, expected 5");
}

#[test]
fn vector_axpy() {
    // y += 2 * x: y = [1,2,3], x = [4,5,6] → y = [9,12,15]
    let x = Vector::from_vec(vec![4.0_f64, 5.0, 6.0]);
    let mut y = Vector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    y.axpy(2.0, &x);
    let s = y.as_slice();
    assert!((s[0] - 9.0).abs() < 1e-14);
    assert!((s[1] - 12.0).abs() < 1e-14);
    assert!((s[2] - 15.0).abs() < 1e-14);
}

#[test]
fn vector_zeros() {
    let v = Vector::<f64>::zeros(5);
    assert!(v.as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn vector_scale() {
    let mut v = Vector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    v.scale(3.0);
    let s = v.as_slice();
    assert!((s[0] - 3.0).abs() < 1e-14);
    assert!((s[1] - 6.0).abs() < 1e-14);
    assert!((s[2] - 9.0).abs() < 1e-14);
}
