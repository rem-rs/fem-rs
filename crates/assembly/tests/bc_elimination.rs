//! Essential-BC elimination semantics for `form.rs::BilinearForm::
//! eliminate_essential_bc` (D432).
//!
//! D432: every reaction must come from the **true column entry** `A[i,d]`
//! only — MFEM `SparseMatrix::EliminateRowCol(rc, sol, rhs, DIAG_ONE)`
//! (linalg/sparsemat.cpp:1914; the reaction `rhs(col) -= sol·A[k]` at :1959
//! reads row `col`'s own entry, exactly like `CsrMatrix::
//! apply_dirichlet_keep_diag` after D409).  The pre-D432 revision
//! *additionally* subtracted the pivot-row entries `A[d,j]` from `rhs[j]`,
//! which doubles every nonzero reaction on numerically symmetric matrices
//! (`A[d,j] == A[j,d]` there) and would flip the reaction sign on the
//! antisymmetric coupling blocks of saddle systems.
//!
//! `eliminate_essential_bc_nonzero_value_is_imposed_exactly` failed on the
//! pre-D432 code (doubled reactions ⇒ solved field ≉ imposed boundary data);
//! the red run is archived under `tmp/d410/d432_red.txt`.

use fem_assembly::form::BilinearForm;
use fem_assembly::standard::DiffusionIntegrator;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_solver::solve_sparse_cholesky;
use fem_space::H1Space;

fn lap1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i > 0     { coo.add(i, i - 1, -1.0); }
        if i < n - 1 { coo.add(i, i + 1, -1.0); }
    }
    coo.into_csr()
}

/// Matrix-surgery double mirroring the **fixed** `form.rs` algorithm (true
/// column reaction only, DIAG_ONE diagonal).  The homogeneous values (`0.0`)
/// exercised here pin the row/column clearing; the reaction *arithmetic* is
/// pinned with a nonzero essential value against the real API by
/// [`eliminate_essential_bc_nonzero_value_is_imposed_exactly`].
#[test]
fn eliminate_essential_bc_solves_correctly() {
    let n = 20;
    let mut a = lap1d(n);
    let mut rhs = vec![1.0; n];
    let ess = vec![0usize, n - 1];

    let ess_set: std::collections::HashSet<usize> = ess.iter().copied().collect();
    for &d in &ess {
        // Column d: reaction from the TRUE column entry `A[i,d]` only (D432),
        // then zero the entry.  The pre-D432 variant additionally subtracted
        // the pivot-row entries `A[d,j]` — removed.
        for i in 0..n {
            if ess_set.contains(&i) { continue; }
            for r in a.row_ptr[i]..a.row_ptr[i + 1] {
                if a.col_idx[r] as usize == d {
                    if i != d {
                        rhs[i] -= a.values[r] * 0.0; // homogeneous data here
                    }
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

/// D432 red/green: with a **nonzero** essential value the elimination must
/// impose it exactly.  Unit square (2 triangles, P1 ⇒ 4 dofs), pure Laplace
/// with `f = 0`: the assembled rows sum to zero (`K·1 = 0`), so `u ≡ g` is
/// the exact discrete solution of the eliminated system and the solved field
/// must equal the imposed data at **every** dof.
///
/// Pre-D432 the row+column double reaction produced `rhs[i] =
/// −(A[0,i]+A[i,0])·g = −2·A[i,0]·g` instead of `−A[i,0]·g`, and the solve
/// deviated from `u ≡ g` by O(1).
#[test]
fn eliminate_essential_bc_nonzero_value_is_imposed_exactly() {
    let mesh = Mesh::<2>::unit_square_tri(1);
    let space = H1Space::new(mesh, 1);
    let mut bf =
        BilinearForm::new(space).add_integrator(DiffusionIntegrator { kappa: 1.0 });
    bf.assemble(2);
    let n = bf.mat().unwrap().nrows;
    assert_eq!(n, 4, "unit square tri(1) P1: 4 dofs");

    let g = 2.0_f64;
    let mut rhs = vec![0.0_f64; n];
    bf.eliminate_essential_bc(&[0], &[g], &mut rhs);
    let x = solve_sparse_cholesky(bf.mat().unwrap(), &rhs).unwrap();

    let (worst_i, worst) = x
        .iter()
        .enumerate()
        .fold((0usize, 0.0_f64), |acc, (i, &v)| {
            if (v - g).abs() > acc.1 { (i, (v - g).abs()) } else { acc }
        });
    assert!(
        worst < 1e-10,
        "nonzero essential value not imposed: x[{worst_i}] deviates from {g} by {worst:.3e} \
         (reactions doubled? expected the exact discrete solution u ≡ {g})"
    );
}
