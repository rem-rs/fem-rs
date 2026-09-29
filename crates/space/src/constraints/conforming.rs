//! MFEM 4.10 hanging-dof true-dof compression (D843-3).
//!
//! 1:1 port of `FiniteElementSpace::BuildConformingInterpolation`
//! (`fem/fespace.cpp:1144`), `BilinearForm::ConformingAssemble`
//! (`fem/bilinearform.cpp:762`), and the elimination primitives
//! (`SparseMatrix::EliminateRowCol(int, SparseMatrix&, DiagonalPolicy)`,
//! `PartMult`, `AddMult`) that MFEM's non-conforming `FormLinearSystem`
//! path applies to reach the true-dof system solved by ex6's PCG.
//!
//! Bit-level conventions replicated from MFEM 4.10:
//! * constraint rows accumulate through an LIFO row builder with
//!   merge-on-insert (`SparseMatrix::_Add_` via `SetColPtr`);
//! * `Finalize` emits rows in LIFO order, then stable-sorts + merges
//!   duplicates (summing) when the row is not column-ascending;
//! * `cP` slave rows resolve through MFEM's multi-pass loop (ascending dof
//!   scans repeated until no row finalizes);
//! * `ConformingAssemble` computes `A_true = (R·A)·P` where the first
//!   product is a row selection (R has a single 1.0 per row) and the second
//!   uses MFEM's first-touch `Mult(A, B, C)` accumulation;
//! * `EliminateRowCol(DIAG_KEEP)` scans the row in stored order, moves
//!   off-diagonal row/column entries into `A_e` (prepended, so `A_e·x`
//!   accumulates in reverse-insertion order) and keeps the diagonal;
//! * RHS: `B = Pᵀb`, then `B -= A_e·X`, then `B[ess] = (A·X)[ess]`
//!   (`EliminateVDofsInRHS` = `AddMult(-1)` + `PartMult` assignment).

use fem_linalg::CsrMatrix;
use fem_mesh::amr::HangingNodeConstraint;

/// MFEM conforming interpolation pair (cP: constrained→true prolongation,
/// cR: true-dof restriction) on a fixed dof numbering.
pub struct ConformingInterpolation {
    n_dofs: usize,
    /// Finalized cP rows: `cp_rows[i]` = (column, value) pairs in stored
    /// (column-ascending) order; empty row = unconstrained dof.
    cp_rows: Vec<Vec<(u32, f64)>>,
    /// True-dof ids (vdofs) in ascending order (cR's row → column map).
    vdof_of_true: Vec<u32>,
}

impl ConformingInterpolation {
    /// Build cP/cR from the immediate hanging-node constraints of a
    /// non-conforming mesh (MFEM deps matrix, one row per slave dof).
    pub fn from_hanging_constraints(n_dofs: usize, cons: &[HangingNodeConstraint]) -> Self {
        // deps[slave] = (master, coef) sorted by master id — MFEM builds the
        // deps rows through plain `Add` calls (LIL, unmerged) and calls
        // `deps.Finalize()`, which sorts each row ascending and merges
        // duplicate columns by summing.
        let mut deps: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n_dofs];
        for c in cons {
            let slave = c.constrained;
            assert!(slave < n_dofs, "hanging dof {slave} out of range");
            assert!(
                deps[slave].is_empty(),
                "dof {slave} constrained twice (MFEM deps has one row per slave)"
            );
            let mut row: Vec<(u32, f64)> = vec![
                (c.parent_a as u32, c.coeff_a),
                (c.parent_b as u32, c.coeff_b),
            ];
            for &(p, w) in &c.extra {
                row.push((p as u32, w));
            }
            row.sort_by_key(|e| e.0);
            let mut merged: Vec<(u32, f64)> = Vec::with_capacity(row.len());
            for (col, val) in row {
                if let Some(last) = merged.last_mut().filter(|(c, _)| *c == col) {
                    last.1 += val;
                } else {
                    merged.push((col, val));
                }
            }
            deps[slave] = merged;
        }

        // LIFO row builder with merge-on-insert (`_Add_` via `SetColPtr`):
        // new entries are prepended; existing columns accumulate in place.
        fn lifo_add(row: &mut Vec<(u32, f64)>, col: u32, val: f64) {
            if let Some(e) = row.iter_mut().find(|(c, _)| *c == col) {
                e.1 += val;
            } else {
                row.push((col, val));
            }
        }
        // LIFO (reverse insertion) storage order → finalize: emit as stored,
        // stable-sort + merge duplicates when not column-ascending.
        fn finalize_row(lifo: &[(u32, f64)]) -> Vec<(u32, f64)> {
            let ascending = lifo.windows(2).all(|w| w[0].0 < w[1].0);
            if ascending {
                return lifo.to_vec();
            }
            let mut row: Vec<(u32, f64)> = lifo.to_vec();
            row.sort_by_key(|e| e.0);
            let mut merged: Vec<(u32, f64)> = Vec::with_capacity(row.len());
            for (col, val) in row {
                if let Some(last) = merged.last_mut().filter(|(c, _)| *c == col) {
                    last.1 += val;
                } else {
                    merged.push((col, val));
                }
            }
            merged
        }

        let mut cp_rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n_dofs];
        let mut vdof_of_true: Vec<u32> = Vec::new();
        // Multi-pass resolution of slave rows (MFEM's do/while loop). The
        // identity rows use the compressed true-dof index as the cP column
        // (MFEM `cP->Add(i, true_dof, 1.0)`), assigned in the same ascending
        // dof scan that fills cR_J.
        let mut finalized = vec![false; n_dofs];
        let mut true_dof = 0u32;
        for i in 0..n_dofs {
            if deps[i].is_empty() {
                vdof_of_true.push(i as u32);
                cp_rows[i] = vec![(true_dof, 1.0)];
                finalized[i] = true;
                true_dof += 1;
            }
        }
        loop {
            let mut finished = true;
            for dof in 0..n_dofs {
                if finalized[dof] || deps[dof].is_empty() {
                    continue;
                }
                if !deps[dof].iter().all(|(c, _)| finalized[*c as usize]) {
                    continue;
                }
                // cP row = Σ_j coef_j · cP_row(master_j) via AddRow: entries
                // arrive in the master row's *current* storage order (LIFO for
                // rows finalized in earlier passes), scaled, merged on insert.
                let mut lifo: Vec<(u32, f64)> = Vec::new();
                for (master, coef) in &deps[dof] {
                    for (col, val) in &cp_rows[*master as usize] {
                        lifo_add(&mut lifo, *col, val * coef);
                    }
                }
                cp_rows[dof] = finalize_row(&lifo);
                finalized[dof] = true;
                finished = false;
            }
            if finished {
                break;
            }
        }

        ConformingInterpolation { n_dofs, cp_rows, vdof_of_true }
    }

    /// `GetTrueVSize()`: number of unconstrained dofs.
    pub fn n_true(&self) -> usize {
        self.vdof_of_true.len()
    }

    /// True-dof ids (vdofs) in ascending order.
    pub fn true_dofs(&self) -> &[u32] {
        &self.vdof_of_true
    }

    /// Whether `dof` is a true (unconstrained) dof.
    pub fn is_true_dof(&self, dof: u32) -> bool {
        self.vdof_of_true.binary_search(&dof).is_ok()
    }

    /// True-dof **index** of a vdof (`GetEssentialTrueDofs` maps essential
    /// vdofs through cR — the elimination lists are true-dof indices, not
    /// vdof ids).
    pub fn true_index_of(&self, vdof: u32) -> Option<usize> {
        self.vdof_of_true.binary_search(&vdof).ok()
    }

    /// Number of constrained (hanging) dofs.
    pub fn n_constrained(&self) -> usize {
        self.n_dofs - self.vdof_of_true.len()
    }

    /// cP row access (stored order) — for pins against MFEM dumps.
    pub fn cp_row(&self, dof: usize) -> &[(u32, f64)] {
        &self.cp_rows[dof]
    }

    /// cR (restriction) applied to a full dof vector: `X[t] = x[vdof(t)]`.
    pub fn restrict(&self, x: &[f64]) -> Vec<f64> {
        self.vdof_of_true.iter().map(|&v| x[v as usize]).collect()
    }

    /// cP (prolongation) applied to a true-dof vector: `y = P·X`
    /// (`SparseMatrix::Mult`: per-row stored-order accumulation).
    pub fn prolongate(&self, x_true: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n_dofs];
        for (i, row) in self.cp_rows.iter().enumerate() {
            let mut b = 0.0;
            for (col, val) in row {
                b += val * x_true[*col as usize];
            }
            y[i] = b;
        }
        y
    }

    /// `Pᵀ·b` (`SparseMatrix::MultTranspose`: rows scanned ascending,
    /// entries scattered into the output).
    pub fn mult_transpose(&self, b: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.vdof_of_true.len()];
        for (i, row) in self.cp_rows.iter().enumerate() {
            let bi = b[i];
            for (col, val) in row {
                y[*col as usize] += val * bi;
            }
        }
        y
    }

    /// `BilinearForm::ConformingAssemble` without the in-place mutation:
    /// `A_true = (R·A)·P` where `R = Pᵀ`. Neither factor is a selection —
    /// the columns of P carry the hanging-dof coefficients, so `RA`'s row t
    /// is `A[vdof(t)] + Σ_h coef·A[h]` over the hanging dofs constrained to
    /// `t`. Both products use MFEM `Mult(A, B, C)` first-touch accumulation
    /// (rows in first-touch column order).
    pub fn compress(&self, a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
        assert_eq!(a.nrows, self.n_dofs, "matrix/space dof mismatch");

        // R = Transpose(P): row t of R = column t of P = (row i, value) pairs
        // in ascending i order (MFEM's Transpose scans P row-major).
        let mut r_rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); self.vdof_of_true.len()];
        for (i, row) in self.cp_rows.iter().enumerate() {
            for (col, val) in row {
                r_rows[*col as usize].push((i as u32, *val));
            }
        }

        // MFEM `Mult(A, B, C)` — first-touch accumulation, shared by both
        // products: per output row, iterate A's row entries in stored order,
        // scale B's rows, append new columns / accumulate existing ones.
        // `b_rows` holds (column, value) pairs per row (P's `cp_rows` or the
        // assembled RA rows wrapped back into pair form by the caller).
        fn mult_first_touch(
            a_cols: &[&[u32]],
            a_vals: &[&[f64]],
            b_cols: &[&[u32]],
            b_vals: &[&[f64]],
            ncols_b: usize,
        ) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
            // pass 1: count nonzeros per row (marker = output row id)
            let nrows = a_cols.len();
            let mut row_ptr = Vec::with_capacity(nrows + 1);
            row_ptr.push(0usize);
            let mut marker: Vec<Option<usize>> = vec![None; ncols_b];
            for t in 0..nrows {
                let mut nnz = row_ptr[t];
                for &ja in a_cols[t] {
                    for &jb in b_cols[ja as usize] {
                        let jbus = jb as usize;
                        if marker[jbus] != Some(t) {
                            marker[jbus] = Some(t);
                            nnz += 1;
                        }
                    }
                }
                row_ptr.push(nnz);
            }
            // pass 2: fill
            let nnz = row_ptr[nrows];
            let mut col_idx = Vec::with_capacity(nnz);
            let mut values = Vec::with_capacity(nnz);
            let mut row_marker: Vec<Option<usize>> = vec![None; ncols_b];
            for t in 0..nrows {
                let row_start = col_idx.len();
                for (ia, &ja) in a_cols[t].iter().enumerate() {
                    let a_entry = a_vals[t][ia];
                    for (ib, &jb) in b_cols[ja as usize].iter().enumerate() {
                        let b_entry = b_vals[ja as usize][ib];
                        let jbus = jb as usize;
                        match row_marker[jbus] {
                            Some(pos) if pos >= row_start => values[pos] += a_entry * b_entry,
                            _ => {
                                row_marker[jbus] = Some(col_idx.len());
                                col_idx.push(jb);
                                values.push(a_entry * b_entry);
                            }
                        }
                    }
                }
            }
            (row_ptr, col_idx, values)
        }

        let n_true = self.vdof_of_true.len();
        // RA = R·A
        let a_cols: Vec<&[u32]> = (0..self.n_dofs)
            .map(|i| &a.col_idx[a.row_ptr[i]..a.row_ptr[i + 1]])
            .collect();
        let a_vals: Vec<&[f64]> = (0..self.n_dofs)
            .map(|i| &a.values[a.row_ptr[i]..a.row_ptr[i + 1]])
            .collect();
        let ra_cols_owned: Vec<Vec<u32>> = r_rows
            .iter()
            .map(|r| r.iter().map(|(i, _)| *i).collect())
            .collect();
        let ra_vals_owned: Vec<Vec<f64>> = r_rows
            .iter()
            .map(|r| r.iter().map(|(_, v)| *v).collect())
            .collect();
        let ra_cols: Vec<&[u32]> = ra_cols_owned.iter().map(|v| v.as_slice()).collect();
        let ra_vals: Vec<&[f64]> = ra_vals_owned.iter().map(|v| v.as_slice()).collect();
        let (ra_ptr, ra_cidx, ra_data) =
            mult_first_touch(&ra_cols, &ra_vals, &a_cols, &a_vals, self.n_dofs);

        // A_true = RA·P (P's rows are `cp_rows`, already (col, value) pairs).
        let p_cols_owned: Vec<Vec<u32>> = self
            .cp_rows
            .iter()
            .map(|r| r.iter().map(|(c, _)| *c).collect())
            .collect();
        let p_vals_owned: Vec<Vec<f64>> = self
            .cp_rows
            .iter()
            .map(|r| r.iter().map(|(_, v)| *v).collect())
            .collect();
        let p_cols: Vec<&[u32]> = p_cols_owned.iter().map(|v| v.as_slice()).collect();
        let p_vals: Vec<&[f64]> = p_vals_owned.iter().map(|v| v.as_slice()).collect();
        let ra_cols2: Vec<&[u32]> = (0..n_true)
            .map(|t| &ra_cidx[ra_ptr[t]..ra_ptr[t + 1]])
            .collect();
        let ra_vals2: Vec<&[f64]> = (0..n_true)
            .map(|t| &ra_data[ra_ptr[t]..ra_ptr[t + 1]])
            .collect();
        let (row_ptr, col_idx, values) =
            mult_first_touch(&ra_cols2, &ra_vals2, &p_cols, &p_vals, n_true);

        CsrMatrix { nrows: n_true, ncols: n_true, row_ptr, col_idx, values }
    }
}

/// `SparseMatrix::EliminateRowCol(rc, A_e, DIAG_KEEP)`: move the off-diagonal
/// entries of the essential rows/columns into `A_e` (MFEM `Add` = prepend, so
/// the returned rows are in reverse-insertion order) and zero them in `mat`.
/// The diagonal is kept (DIAG_KEEP).
pub fn eliminate_rowcol_keep_diag(mat: &mut CsrMatrix<f64>, ess: &[u32]) -> Vec<Vec<(u32, f64)>> {
    let mut ae: Vec<Vec<(u32, f64)>> = vec![Vec::new(); mat.nrows];
    for &rc in ess {
        let rc = rc as usize;
        let row_end = mat.row_ptr[rc + 1];
        for j in mat.row_ptr[rc]..row_end {
            let col = mat.col_idx[j] as usize;
            if col == rc {
                continue; // DIAG_KEEP: diagonal untouched
            }
            // Ae.Add(rc, col, A[j]); A[j] = 0.0
            ae[rc].insert(0, (mat.col_idx[j], mat.values[j]));
            mat.values[j] = 0.0;
            // locate A[col][rc] by forward scan of row col (stored order)
            let (s, e) = (mat.row_ptr[col], mat.row_ptr[col + 1]);
            if let Some(k) = (s..e).find(|&k| mat.col_idx[k] == rc as u32) {
                ae[col].insert(0, (rc as u32, mat.values[k]));
                mat.values[k] = 0.0;
            }
        }
    }
    ae
}

/// `BilinearForm::EliminateVDofsInRHS(ess, X, B)`:
/// `B -= A_e·X` (LIFO-order rows, `AddMult` with a = −1), then
/// `B[ess] = (mat·X)[ess]` (`PartMult` assignment, stored-order rows).
pub fn eliminate_vdofs_in_rhs(
    ae: &[Vec<(u32, f64)>],
    mat: &CsrMatrix<f64>,
    ess: &[u32],
    x: &[f64],
    b: &mut [f64],
) {
    for (i, row) in ae.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let mut s = 0.0;
        for (col, val) in row {
            s += val * x[*col as usize];
        }
        b[i] += -1.0 * s;
    }
    for &r in ess {
        let r = r as usize;
        let (s, e) = (mat.row_ptr[r], mat.row_ptr[r + 1]);
        let mut acc = 0.0;
        for k in s..e {
            acc += mat.values[k] * x[mat.col_idx[k] as usize];
        }
        b[r] = acc;
    }
}
