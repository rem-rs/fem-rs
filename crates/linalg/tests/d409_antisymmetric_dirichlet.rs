//! D409 regression: Dirichlet elimination must take the column reactions from
//! the **true column entries** `A[j,row]` — as MFEM does in
//! `SparseMatrix::EliminateRowCol(rc, sol, rhs, dpolicy)`
//! (linalg/sparsemat.cpp:1914; the reaction is `rhs(col) -= sol * A[k]`
//! at :1959, where `A[k]` is the entry of row `col` found by scanning
//! `J[k] == rc`) — not from the pivot-row entries `A[row,j]`.
//!
//! `A[j,row] == A[row,j]` holds only for *numerically* symmetric matrices.
//! Saddle systems `[A −Bᵀ; B 0]` are structurally symmetric but numerically
//! **antisymmetric** on the coupling blocks (`B[j,row] = −B[row,j]`), so a
//! row-driven elimination flips the sign of every nonzero essential reaction.
//! This is exactly the D401 isolation finding (pinning a divergence row to
//! −0.4167 produced rhs −0.4167 instead of +0.4167).

use fem_linalg::{CooMatrix, CsrMatrix};

/// 3×3 saddle mini-system `[D −Bᵀ; B ε]` with `D = 2I`, `B = [1 1]`:
/// ```text
///   row 0: [2  0  -1]     (−Bᵀ block)
///   row 1: [0  2  -1]
///   row 2: [1  1   ε]     ( B block)
/// ```
/// Numerically antisymmetric on the coupling pair (A[0,2] = −A[2,0]).
fn saddle3(eps: f64) -> CsrMatrix<f64> {
    let mut c = CooMatrix::<f64>::new(3, 3);
    c.add(0, 0, 2.0);
    c.add(0, 2, -1.0);
    c.add(1, 1, 2.0);
    c.add(1, 2, -1.0);
    c.add(2, 0, 1.0);
    c.add(2, 1, 1.0);
    c.add(2, 2, eps);
    c.into_csr()
}

/// `keep_diag` (MFEM DIAG_KEEP) on a numerically antisymmetric system:
/// the reaction on row 2 must use the true column value `A[2,0] = +1`,
/// giving `rhs[2] = 0.5 − 1·1.5 = −1.0`.  The row-driven variant would use
/// `A[0,2] = −1` and produce `+2.0` (sign flip).
#[test]
fn keep_diag_antisymmetric_reaction_sign() {
    let mut a = saddle3(1e-12);
    let mut rhs = [10.0_f64, 20.0, 0.5];
    a.apply_dirichlet_keep_diag(0, 1.5, &mut rhs);

    // Pivot row: diagonal kept, rhs[0] = A[0,0]·1.5 = 3.0.
    assert!((a.get(0, 0) - 2.0).abs() < 1e-14, "DIAG_KEEP must keep A[0,0]");
    assert!((rhs[0] - 3.0).abs() < 1e-14, "rhs[0] = {} want 3.0", rhs[0]);
    // Row and column off-diagonals of dof 0 eliminated.
    assert!(a.get(0, 2).abs() < 1e-14, "row off-diag not eliminated");
    assert!(a.get(2, 0).abs() < 1e-14, "column off-diag not eliminated");
    // Row 1 is not a neighbor of dof 0: untouched.
    assert!((rhs[1] - 20.0).abs() < 1e-14, "rhs[1] = {} want 20.0", rhs[1]);
    // True reaction from A[2,0] = +1: rhs[2] = 0.5 − 1.5 = −1.0
    // (row-driven would give 0.5 − (−1)·1.5 = +2.0).
    assert!(
        (rhs[2] - (-1.0)).abs() < 1e-14,
        "rhs[2] = {} want -1.0 (reaction sign flipped?)",
        rhs[2]
    );
}

/// Same check for the DIAG_ONE variant (`apply_dirichlet_symmetric`).
#[test]
fn symmetric_antisymmetric_reaction_sign() {
    let mut a = saddle3(1e-12);
    let mut rhs = [10.0_f64, 20.0, 0.5];
    a.apply_dirichlet_symmetric(0, 1.5, &mut rhs);

    assert!((a.get(0, 0) - 1.0).abs() < 1e-14, "DIAG_ONE must set A[0,0]=1");
    assert!((rhs[0] - 1.5).abs() < 1e-14, "rhs[0] = {} want 1.5", rhs[0]);
    assert!(a.get(0, 2).abs() < 1e-14);
    assert!(a.get(2, 0).abs() < 1e-14);
    assert!((rhs[1] - 20.0).abs() < 1e-14);
    assert!(
        (rhs[2] - (-1.0)).abs() < 1e-14,
        "rhs[2] = {} want -1.0 (reaction sign flipped?)",
        rhs[2]
    );
}

/// Structurally symmetric pattern where the pivot-row entry is numerically
/// zero but the mirrored column entry is **not**: the reaction must still be
/// applied (a row-value-driven skip leaves the column un-eliminated).
#[test]
fn keep_diag_zero_row_entry_nonzero_column_entry() {
    // A = [4 0; 3 5] — A[0,1] = 0 stored explicitly, mirror A[1,0] = 3.
    let mut a = CsrMatrix {
        nrows: 2,
        ncols: 2,
        row_ptr: vec![0, 2, 4],
        col_idx: vec![0, 1, 0, 1],
        values: vec![4.0, 0.0, 3.0, 5.0],
    };
    let mut rhs = [7.0_f64, 11.0];
    a.apply_dirichlet_keep_diag(0, 2.0, &mut rhs);

    assert!(
        (rhs[1] - (11.0 - 3.0 * 2.0)).abs() < 1e-14,
        "rhs[1] = {} want 5.0 (column reaction from A[1,0]=3)",
        rhs[1]
    );
    assert!(a.get(1, 0).abs() < 1e-14, "A[1,0] must be eliminated");
    assert!((a.get(0, 0) - 4.0).abs() < 1e-14, "DIAG_KEEP keeps A[0,0]=4");
    assert!((rhs[0] - 8.0).abs() < 1e-14, "rhs[0] = 4·2 = 8");
}

/// Structurally **asymmetric** pattern (pivot row has an entry whose mirror
/// does not exist): MFEM aborts here (`EliminateRowCol () #3`); the Rust
/// helper degrades gracefully — no reaction, no panic, row still eliminated.
#[test]
fn keep_diag_missing_mirror_is_graceful() {
    // A = [2 5; _ 3] — A[1,0] structurally absent.
    let mut a = CsrMatrix {
        nrows: 2,
        ncols: 2,
        row_ptr: vec![0, 2, 3],
        col_idx: vec![0, 1, 1],
        values: vec![2.0, 5.0, 3.0],
    };
    let mut rhs = [1.0_f64, 10.0];
    a.apply_dirichlet_keep_diag(0, 2.0, &mut rhs);

    assert!(
        (rhs[1] - 10.0).abs() < 1e-14,
        "rhs[1] = {} want 10.0 (no mirror ⇒ no reaction; row-driven would subtract A[0,1]·2 = 10)",
        rhs[1]
    );
    assert!(a.get(0, 1).abs() < 1e-14, "row off-diag eliminated");
    assert!((rhs[0] - 4.0).abs() < 1e-14, "rhs[0] = 2·2 = 4");
}

/// End-to-end sanity: after eliminating dof 0 in the saddle system, the
/// eliminated system must reproduce the essential value exactly for dof 0
/// and keep the constraint row consistent (B-block reaction present).
#[test]
fn keep_diag_saddle_solution_respects_essential_value() {
    let mut a = saddle3(1e-12);
    let mut rhs = [1.0_f64, 2.0, 3.0];
    a.apply_dirichlet_keep_diag(1, 0.25, &mut rhs);
    // Pivot row 1: rhs[1] = A[1,1]·0.25 = 0.5; reaction on row 2 from A[2,1]=1:
    // rhs[2] = 3.0 − 1·0.25 = 2.75.
    assert!((rhs[1] - 0.5).abs() < 1e-14);
    assert!(
        (rhs[2] - 2.75).abs() < 1e-14,
        "rhs[2] = {} want 2.75 (reaction from A[2,1]=+1, not A[1,2]=−1)",
        rhs[2]
    );
    assert!(a.get(2, 1).abs() < 1e-14);
    assert!(a.get(1, 2).abs() < 1e-14);
    assert!((a.get(1, 1) - 2.0).abs() < 1e-14);
}
