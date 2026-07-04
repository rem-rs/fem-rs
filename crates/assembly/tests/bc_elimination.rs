use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::solve_sparse_cholesky;

fn lap1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i > 0     { coo.add(i, i - 1, -1.0); }
        if i < n - 1 { coo.add(i, i + 1, -1.0); }
    }
    coo.into_csr()
}

#[test]
fn eliminate_essential_bc_solves_correctly() {
    let n = 20;
    let mut a = lap1d(n);
    let mut rhs = vec![1.0; n];
    let ess = vec![0usize, n - 1];

    // Apply BC elimination (same algorithm as form.rs)
    let ess_set: std::collections::HashSet<usize> = ess.iter().copied().collect();
    for &d in &ess {
        // Row contributions
        for r in a.row_ptr[d]..a.row_ptr[d + 1] {
            let j = a.col_idx[r] as usize;
            if !ess_set.contains(&j) {
                rhs[j] -= a.values[r] * 0.0;
            }
        }
        // Zero column d
        for i in 0..n {
            if ess_set.contains(&i) { continue; }
            for r in a.row_ptr[i]..a.row_ptr[i + 1] {
                if a.col_idx[r] as usize == d {
                    a.values[r] = 0.0;
                    break;
                }
            }
        }
        // Zero row d
        for r in a.row_ptr[d]..a.row_ptr[d + 1] {
            let j = a.col_idx[r] as usize;
            if j != d { a.values[r] = 0.0; }
        }
        // Set diagonal
        for r in a.row_ptr[d]..a.row_ptr[d + 1] {
            if a.col_idx[r] as usize == d {
                a.values[r] = 1.0;
                break;
            }
        }
        rhs[d] = 0.0;
    }

    let x = solve_sparse_cholesky(&a, &rhs).unwrap();
    assert!((x[0]).abs() < 1e-14, "BC DOF 0 = {:.3e} (expected 0)", x[0]);
    assert!((x[n - 1]).abs() < 1e-14, "BC DOF {} = {:.3e} (expected 0)", n - 1, x[n - 1]);

    // Verify A·x ≈ rhs
    let mut ax = vec![0.0; n];
    a.spmv(&x, &mut ax);
    for i in 0..n {
        let diff = (ax[i] - rhs[i]).abs();
        assert!(diff < 1e-10, "node {i}: |Ax−b| = {:.3e}", diff);
    }
}
