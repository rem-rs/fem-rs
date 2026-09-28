//! D842-1 pin: `solve_minres_dsmoother` must reproduce MFEM's
//! `MINRESSolver` + `DSmoother(1)` bit-for-bit.
//!
//! Oracle generated with MFEM 4.10 (`tmp/d91b/minres_probe.cpp`, serial
//! build `/home/quan/mfem410_gslib`, g++ -O3): an 8x8 SPD system with a
//! strongly varying diagonal, solved with
//!
//! ```text
//!   DSmoother prec(S);            // JACOBI, scale 1, 1 iteration
//!   MINRESSolver minres;          // iterative_mode = false
//!   minres.SetPrintLevel(-1); minres.SetRelTol(1e-8);
//!   minres.SetAbsTol(0.0);    minres.SetMaxIter(300);
//!   minres.SetPreconditioner(prec); minres.SetOperator(S);
//!   x = 0; minres.Mult(b, x);
//! ```
//!
//! The point of the port: ex10's Newton drift (0.0099624 vs 0.0099476 at
//! Newton iter 1) traced to the inner Jacobian solve using a split-scaled
//! `solve_minres_jacobi` instead of MFEM's left-preconditioned recurrence.

use fem_linalg::CsrMatrix;
use fem_solver::{solve_minres_dsmoother, PrintLevel, SolverConfig};

/// The probe matrix, laid out exactly like MFEM's CSR from the C++ probe:
/// `SparseMatrix::Add(i, j, val)` prepends each new node to the row's linked
/// list (`SearchRow`, sparsemat.hpp) and `Finalize` walks it via `->Prev`, so
/// a row fed with ascending columns is stored DESCENDING (with `isSorted`
/// blindly set to true — the C++ probe prints row 0 as `[(0,5), (0,0)]`).
/// Note the C++ matrix is lower-tridiagonal: `A(i+1, i) = A(i, i-1)` never
/// writes the super-diagonal.
fn probe_matrix() -> CsrMatrix<f64> {
    let n = 8usize;
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        a[i * n + i] = 10.0 + 100.0 * ((i * 7) % 5) as f64 + 0.5 * i as f64;
        if i > 0 {
            a[i * n + i - 1] = -2.0 - 0.1 * i as f64;
        }
    }
    a[0 * n + 5] = -0.75;
    a[5 * n + 0] = -0.75;
    a[2 * n + 7] = -0.3125;
    a[7 * n + 2] = -0.3125;

    // Descending-column rows, mirroring the MFEM linked-list layout.
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (0..n).rev() {
            if a[i * n + j] != 0.0 {
                col_idx.push(j as u32);
                values.push(a[i * n + j]);
            }
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values }
}

fn probe_rhs() -> Vec<f64> {
    (0..8)
        .map(|i| 1.0 / (1.0 + i as f64) - 0.15 * ((i % 3) as f64 - 1.0))
        .collect()
}

#[test]
fn minres_dsmoother_matches_mfem_oracle_bitwise() {
    let a = probe_matrix();
    let b = probe_rhs();
    let mut x = vec![0.0_f64; 8];
    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 0.0,
        max_iter: 300,
        verbose: false,
        print_level: PrintLevel::Silent,
        ..SolverConfig::default()
    };
    let res = solve_minres_dsmoother(&a, &b, &mut x, &cfg).expect("converged");

    // MFEM oracle: iterations = 9, final_norm = 1.70951125213290114e-09.
    assert_eq!(res.iterations, 9, "iteration count");
    assert!(res.converged, "converged flag");
    assert_eq!(res.final_residual.to_bits(), 1.70951125213290114e-09_f64.to_bits(),
        "final B-norm bitwise");

    // MFEM oracle solution vectors (printf %.17e).
    let oracle = [
        1.15630367197273570e-01_f64,
        3.52885396726617230e-03,
        4.65193196414163880e-04,
        3.59703986538434514e-03,
        6.68695171052031230e-04,
        8.40489404545360919e-03,
        1.47751112944382148e-03,
        3.12296647985188557e-04,
    ];
    for i in 0..8 {
        assert_eq!(
            x[i].to_bits(),
            oracle[i].to_bits(),
            "x[{i}] bitwise: got {:.17e}, want {:.17e}",
            x[i],
            oracle[i]
        );
    }
}
