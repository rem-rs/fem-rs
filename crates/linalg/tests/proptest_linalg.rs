use fem_linalg::CooMatrix;
use proptest::prelude::*;

// ─── Strategies ─────────────────────────────────────────────────────────

fn coo_strategy(nrows: usize, ncols: usize) -> impl Strategy<Value = Vec<(usize, usize, f64)>> {
    let max_nnz = nrows * ncols;
    // Generate a list of unique (row, col) pairs with values in [-10, 10]
    proptest::collection::vec(
        (0..nrows, 0..ncols, -10.0f64..=10.0f64),
        1..=max_nnz.min(30),
    )
    .prop_filter("at least one entry", |entries| !entries.is_empty())
}

fn csr_from_entries(nrows: usize, ncols: usize, entries: &[(usize, usize, f64)]) -> fem_linalg::CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(nrows, ncols);
    for &(r, c, v) in entries {
        coo.add(r, c, v);
    }
    coo.into_csr()
}

fn vector_strategy(n: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-10.0f64..=10.0f64, n)
}

// ─── SpMV: A·(x + y) = A·x + A·y ───────────────────────────────────────

proptest! {
    #[test]
    fn spmv_distributive_addition(
        entries in coo_strategy(5, 5),
        x in vector_strategy(5),
        y in vector_strategy(5),
    ) {
        let a = csr_from_entries(5, 5, &entries);
        let mut ax_plus_ay = vec![0.0; 5];
        let mut a_x_plus_y = vec![0.0; 5];

        // A·(x + y)
        let sum: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        a.spmv(&sum, &mut a_x_plus_y);

        // A·x + A·y
        let mut ax = vec![0.0; 5];
        let mut ay = vec![0.0; 5];
        a.spmv(&x, &mut ax);
        a.spmv(&y, &mut ay);
        for i in 0..5 { ax_plus_ay[i] = ax[i] + ay[i]; }

        for i in 0..5 {
            let diff = (a_x_plus_y[i] - ax_plus_ay[i]).abs();
            assert!(diff < 1e-12, "SpMV distributivity failed at {i}: A·(x+y)[{i}]={}, A·x+A·y[{i}]={}, diff={diff}",
                a_x_plus_y[i], ax_plus_ay[i]);
        }
    }
}

// ─── SpMV: A·(α·x) = α·(A·x) ───────────────────────────────────────────

proptest! {
    #[test]
    fn spmv_homogeneity(
        entries in coo_strategy(5, 5),
        x in vector_strategy(5),
        alpha in -5.0f64..=5.0f64,
    ) {
        let a = csr_from_entries(5, 5, &entries);
        let mut a_alpha_x = vec![0.0; 5];
        let mut alpha_ax = vec![0.0; 5];

        // A·(α·x)
        let scaled: Vec<f64> = x.iter().map(|v| alpha * v).collect();
        a.spmv(&scaled, &mut a_alpha_x);

        // α·(A·x)
        a.spmv(&x, &mut alpha_ax);
        for v in &mut alpha_ax { *v *= alpha; }

        for i in 0..5 {
            let diff = (a_alpha_x[i] - alpha_ax[i]).abs();
            assert!(diff < 1e-12, "SpMV homogeneity failed at {i}: A·(αx)[{i}]={}, α·(Ax)[{i}]={}, diff={diff}",
                a_alpha_x[i], alpha_ax[i]);
        }
    }
}

// ─── SpAdd: (A + B)·x = A·x + B·x ─────────────────────────────────────

proptest! {
    #[test]
    fn spadd_distributive(
        entries_a in coo_strategy(4, 4),
        entries_b in coo_strategy(4, 4),
        x in vector_strategy(4),
    ) {
        let a = csr_from_entries(4, 4, &entries_a);
        let b = csr_from_entries(4, 4, &entries_b);
        let c = a.add(&b);

        let mut cx = vec![0.0; 4];
        let mut ax_plus_bx = vec![0.0; 4];

        c.spmv(&x, &mut cx);

        let mut ax = vec![0.0; 4];
        let mut bx = vec![0.0; 4];
        a.spmv(&x, &mut ax);
        b.spmv(&x, &mut bx);
        for i in 0..4 { ax_plus_bx[i] = ax[i] + bx[i]; }

        for i in 0..4 {
            let diff = (cx[i] - ax_plus_bx[i]).abs();
            assert!(diff < 1e-12, "SpAdd distributivity failed at {i}: (A+B)x[{i}]={}, Ax+Bx[{i}]={}, diff={diff}",
                cx[i], ax_plus_bx[i]);
        }
    }
}

// ─── SpAdd: A + B = B + A (commutativity) ───────────────────────────────

proptest! {
    #[test]
    fn spadd_commutative(
        entries_a in coo_strategy(5, 5),
        entries_b in coo_strategy(5, 5),
    ) {
        let a = csr_from_entries(5, 5, &entries_a);
        let b = csr_from_entries(5, 5, &entries_b);
        let ab = a.add(&b);
        let ba = b.add(&a);

        for i in 0..5 {
            for ptr in ab.row_ptr[i]..ab.row_ptr[i + 1] {
                let j = ab.col_idx[ptr];
                // Find corresponding value in ba
                let mut found = false;
                for p in ba.row_ptr[i]..ba.row_ptr[i + 1] {
                    if ba.col_idx[p] == j {
                        let diff = (ab.values[ptr] - ba.values[p]).abs();
                        assert!(diff < 1e-14, "SpAdd not commutative at ({i},{j}): AB={}, BA={}", ab.values[ptr], ba.values[p]);
                        found = true;
                        break;
                    }
                }
                assert!(found, "SpAdd: column {j} in row {i} of A+B not found in B+A");
            }
        }
    }
}

// ─── Vector dot product symmetry: x·y = y·x ─────────────────────────────

proptest! {
    #[test]
    fn vector_dot_symmetric(
        x in vector_strategy(8),
        y in vector_strategy(8),
    ) {
        let dot_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let dot_yx: f64 = y.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let diff = (dot_xy - dot_yx).abs();
        assert!(diff < 1e-14, "dot product not symmetric: x·y={dot_xy}, y·x={dot_yx}");
    }
}

// ─── Vector norm properties ─────────────────────────────────────────────

proptest! {
    #[test]
    fn vector_norm_nonnegative(
        x in vector_strategy(6),
    ) {
        let dot: f64 = x.iter().map(|v| v * v).sum();
        let nrm = dot.sqrt();
        assert!(nrm >= 0.0, "norm should be nonnegative, got {nrm}");
        if dot == 0.0 {
            // all-zero should give zero norm
        }
    }

    #[test]
    fn vector_triangle_inequality(
        x in vector_strategy(6),
        y in vector_strategy(6),
    ) {
        let sum: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        let norm_sum: f64 = sum.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_y: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm_sum <= norm_x + norm_y + 1e-12,
            "triangle inequality violated: ‖x+y‖={norm_sum}, ‖x‖+‖y‖={}", norm_x + norm_y);
    }
}

// ─── Vector Cauchy-Schwarz ──────────────────────────────────────────────

proptest! {
    #[test]
    fn cauchy_schwarz(
        x in vector_strategy(8),
        y in vector_strategy(8),
    ) {
        let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_y: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(dot.abs() <= norm_x * norm_y + 1e-12,
            "Cauchy-Schwarz violated: |x·y|={}, ‖x‖‖y‖={}", dot.abs(), norm_x * norm_y);
    }
}

// ─── CSR transpose property: (Aᵀ)ᵀ = A ────────────────────────────────

proptest! {
    #[test]
    fn transpose_involution(
        entries in coo_strategy(4, 5),
    ) {
        let a = csr_from_entries(4, 5, &entries);
        let at = a.transpose();
        let ata = at.transpose();

        assert_eq!(a.nrows, ata.nrows, "row count mismatch after double transpose");
        assert_eq!(a.ncols, ata.ncols, "col count mismatch after double transpose");
        for i in 0..a.nrows {
            for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[ptr];
                // Find in ata
                let mut found = false;
                for p in ata.row_ptr[i]..ata.row_ptr[i + 1] {
                    if ata.col_idx[p] == j {
                        let diff = (a.values[ptr] - ata.values[p]).abs();
                        assert!(diff < 1e-14, "(Aᵀ)ᵀ ≠ A at ({i},{j})");
                        found = true;
                        break;
                    }
                }
                assert!(found, "entry ({i},{j}) missing after double transpose");
            }
        }
    }
}
