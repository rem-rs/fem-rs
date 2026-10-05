//! D1230 (round 116): the MFEM → hypre PtAP handoff in-row storage order.
//!
//! C++ stores each matrix row of the assembled operator in an order that is a
//! deterministic function of the element-assembly history — neither ascending
//! nor arbitrary — and the RAP values hypre's AMS computes from A depend on
//! that storage order through the two-stage KT accumulation first-touch order
//! (`par_rap.c` `BuildCoarseOperatorKT`, stage-2).  The C++ chain, with
//! citations (mfem 4.10 / hypre 2.28):
//!
//! 1. `BilinearForm::Assemble` walks elements in ascending mesh order
//!    (`bilinearform.cpp:507`) and inserts each element matrix through
//!    `AddSubMatrix(vdofs, vdofs, elmat, skip_zeros)` (`bilinearform.cpp:559`)
//!    — rows outer, columns inner in ascending local order (`sparsemat.cpp:2804-2829`).
//! 2. A new (row, col) pair is PREPENDED to the row's linked list
//!    (`SparseMatrix::SearchRow`, `sparsemat.hpp:902-910`), so the list is in
//!    reverse insertion order; `Finalize` serializes it head-first via `Prev`
//!    (`sparsemat.cpp:1429-1433`) — each CSR row ends up in **reverse
//!    first-touch insertion order** (descending per-element runs).
//! 3. `ParBilinearForm::ParallelAssemble` wraps that CSR into hypre through
//!    the square block-diagonal constructor (`hypre.cpp:932-963`), which calls
//!    `hypre_CSRMatrixReorder` — "first entry in each row is the diagonal
//!    one" (`csr_matop.c:1543-1546`: the diagonal entry is SWAPPED with slot
//!    0, not removed-and-reinserted).  `MakePtAP` (`handle.cpp:124-178`) then
//!    runs `mfem::RAP`; with the conforming identity prolongation the product
//!    preserves row order and values exactly, so hypre's AMS RAP consumes A
//!    rows as **[diagonal] + reverse(first-touch insertion), with the
//!    diagonal swapped into slot 0**.
//!
//! fem-rs assembles bitwise-identical A values (d1115: 141448/141448 against
//! the C++ probe) but stores rows ascending; the RAP slot accumulation orders
//! then diverge at 1-2 ulp on multi-contribution slots (d115a: level-0 A_Pi
//! differed on 12511/29521 entries, max 4.657e-10), HMIS near-threshold
//! strength ties flip, and the hierarchy becomes structurally different —
//! the tesla `-bm` o2 7/6 vs C++ 8/7 coin toss.  d116a verified the inverse
//! direction: A in this handoff order (any Pi order — the KT transpose
//! normalizes R rows ascending, and each P row contributes at most one term
//! per output slot) reproduces hypre's level-0 `PiᵀAPi` bitwise 29521/29521
//! against the real MFEM 4.10 + hypre 2.28 run dump (`tmp/d116a/`).
//!
//! The reorder is a pure per-row permutation of stored entries: values move
//! with their columns and are never modified, so it is algebraically a no-op
//! on the operator.

use fem_linalg::CsrMatrix;

/// Rebuild a square CSR matrix in the C++ hypre-handoff form: canonical dof
/// numbering + per-row `[diagonal] + reverse first-touch insertion order`
/// (diagonal swapped into slot 0).
///
/// * `a` — the assembled operator in partition numbering (any in-row order;
///   its `(row, col, value)` multiset is preserved exactly).
/// * `to_canonical` — partition dof id → canonical (space) dof id, length
///   `a.nrows` (the D1095 serial-parity map, `DofPartition::unpermute_dof`);
///   the output is indexed by canonical ids in ascending canonical order.
/// * `element_dofs` — per element in ascending mesh order, the element's
///   local dof list in ascending local order, in canonical ids — the same
///   array MFEM's `GetElementVDofs` hands to `AddSubMatrix`.
///
/// Degenerate inputs keep their stored order instead of panicking: a row not
/// covered by the partition→canonical map passes through unmapped, and a
/// stored column no element can place (none exists for a conforming
/// operator — every stored coupling shares an element with its row) is
/// appended in original stored order after the reconstructed prefix.
pub fn mfem_ptap_handoff_matrix(
    a: &CsrMatrix<f64>,
    to_canonical: &[u32],
    element_dofs: &[Vec<u32>],
) -> CsrMatrix<f64> {
    debug_assert_eq!(a.nrows, a.ncols, "square FE operator expected");
    debug_assert_eq!(to_canonical.len(), a.nrows, "partition → canonical map");

    let n = a.nrows;
    let mut part_of_canon = vec![usize::MAX; n];
    for (p, &c) in to_canonical.iter().enumerate() {
        part_of_canon[c as usize] = p;
    }

    // Generation-stamped per-row markers: `present[cc] == r` tags the stored
    // columns of the current row, `inserted[cc] == r` the ones already placed
    // in the replay sequence.  Rows visit disjoint stamp values, so no
    // per-row clearing is needed.
    let mut present = vec![u32::MAX; n];
    let mut inserted = vec![u32::MAX; n];
    // Stored slot of column `cc` inside the current partition row (valid
    // only while `present[cc] == r`).
    let mut slot = vec![usize::MAX; n];

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(a.col_idx.len());
    let mut values = Vec::with_capacity(a.values.len());
    let mut order: Vec<u32> = Vec::new();
    row_ptr.push(0);

    for r in 0..n {
        order.clear();
        let p = part_of_canon[r];
        if p == usize::MAX {
            // Unmapped canonical row (impossible on the 1-rank handoff path,
            // where the map is a bijection): emit nothing to keep indices
            // total; the caller's numbering guarantees this never fires.
            row_ptr.push(col_idx.len());
            continue;
        }
        let cols = &a.col_idx[a.row_ptr[p]..a.row_ptr[p + 1]];
        let vals = &a.values[a.row_ptr[p]..a.row_ptr[p + 1]];

        for (k, &c) in cols.iter().enumerate() {
            let cc = to_canonical[c as usize] as usize;
            present[cc] = r as u32;
            slot[cc] = k;
        }

        // First-touch insertion replay: elements ascending (mesh order),
        // local dofs ascending within an element.  A stored column enters at
        // its first element that also holds the row — MFEM's prepended
        // linked list, serialized head-first, stores the REVERSE of this
        // sequence.
        for el in element_dofs {
            if !el.iter().any(|&d| d as usize == r) {
                continue;
            }
            for &c in el {
                let cc = c as usize;
                if present[cc] == r as u32 && inserted[cc] != r as u32 {
                    inserted[cc] = r as u32;
                    order.push(c);
                }
            }
        }
        order.reverse();

        // `hypre_CSRMatrixReorder` (csr_matop.c:1543-1546): swap the
        // diagonal with slot 0.
        if let Some(dp) = order.iter().position(|&c| c as usize == r) {
            if dp != 0 {
                order.swap(0, dp);
            }
        }

        for &c in &order {
            col_idx.push(c);
            values.push(vals[slot[c as usize]]);
        }
        // Stored columns the replay could not place (empty for a conforming
        // operator) keep their original stored order.
        for (k, &c) in cols.iter().enumerate() {
            let cc = to_canonical[c as usize] as usize;
            if inserted[cc] != r as u32 {
                col_idx.push(c);
                values.push(vals[k]);
            }
        }
        row_ptr.push(col_idx.len());
    }

    CsrMatrix {
        nrows: n,
        ncols: a.ncols,
        row_ptr,
        col_idx,
        values,
    }
}

/// Rectangular variant: the MFEM discrete-operator (`ParDiscreteLinearOperator`)
/// handoff storage order — the id_ND interpolation blocks hypre's
/// `HYPRE_AMSSetInterpolations` receives.
///
/// C++ chain: the local operator matrix is filled by `SetSubMatrix`
/// (`bilinearform.cpp:2502`) — the SAME prepend-on-new-column linked list as
/// `AddSubMatrix` (`sparsemat.hpp:902-910`), with values overwritten instead
/// of accumulated — and the per-component blocks go through the rectangular
/// block-diagonal constructor (`hypre.cpp:998-1060`), which (row starts ≠ col
/// starts) performs NO `hypre_CSRMatrixReorder`: the handoff order per row is
/// exactly **reverse of the first-inserting element's ascending trial-dof
/// list, filtered to the row's stored columns**.  A point-value ND
/// functional is written by every element containing the dof (value = LAST
/// writer, matching fem-rs `assemble_pi_blocks`' last host — d110a: Pi_x
/// rows bitwise), but columns are inserted only once — by the FIRST writer —
/// so the first element alone fixes the whole row order.
///
/// * `a` — the assembled block in partition numbering (any in-row order; the
///   `(row, col, value)` multiset is preserved exactly).
/// * `to_canonical_rows` / `to_canonical_cols` — partition → canonical dof
///   maps (the D1095 serial-parity maps); the output is indexed by canonical
///   ids (rows ascending, rectangular `a.nrows × a.ncols`).
/// * `host_of_row` — per partition row: the element whose `SetSubMatrix`
///   call FIRST inserted this row's columns (the first mesh element holding
///   the dof — later hosts overwrite values but insert nothing new, so the
///   first host alone fixes the insertion order).
/// * `element_cols` — per element, the element's trial (H¹) dof list in
///   ascending local order, in canonical ids.
pub fn mfem_discrete_op_handoff_matrix(
    a: &CsrMatrix<f64>,
    to_canonical_rows: &[u32],
    to_canonical_cols: &[u32],
    host_of_row: &[u32],
    element_cols: &[Vec<u32>],
) -> CsrMatrix<f64> {
    debug_assert_eq!(to_canonical_rows.len(), a.nrows, "row partition map");
    debug_assert_eq!(host_of_row.len(), a.nrows, "host map");

    let n = a.nrows;
    let mut part_of_canon = vec![usize::MAX; n];
    for (p, &c) in to_canonical_rows.iter().enumerate() {
        part_of_canon[c as usize] = p;
    }

    // Generation-stamped membership marker over canonical columns.
    let mut present = vec![u32::MAX; a.ncols.max(1)];
    let mut slot = vec![usize::MAX; a.ncols.max(1)];

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(a.col_idx.len());
    let mut values = Vec::with_capacity(a.values.len());
    let mut order: Vec<u32> = Vec::new();
    row_ptr.push(0);

    for r in 0..n {
        order.clear();
        let p = part_of_canon[r];
        if p == usize::MAX {
            row_ptr.push(col_idx.len());
            continue;
        }
        let cols = &a.col_idx[a.row_ptr[p]..a.row_ptr[p + 1]];
        let vals = &a.values[a.row_ptr[p]..a.row_ptr[p + 1]];
        for (k, &c) in cols.iter().enumerate() {
            let cc = to_canonical_cols[c as usize] as usize;
            present[cc] = r as u32;
            slot[cc] = k;
        }
        // The first-inserting host's ascending trial-dof list, filtered to
        // the stored set, reversed: the prepend + head-first traversal leaves
        // the CSR row in reverse insertion order (no diagonal reorder for a
        // rectangular block).  Emitted ids are canonical.
        if let Some(h) = element_cols.get(host_of_row[p] as usize) {
            for &c in h.iter().rev() {
                let cc = c as usize;
                if present[cc] == r as u32 {
                    order.push(c);
                }
            }
        }
        for &c in &order {
            col_idx.push(c);
            values.push(vals[slot[c as usize]]);
        }
        // Stored columns outside the host list (none for a point-value ND
        // functional) keep their original stored order, mapped to canonical.
        for &c in cols {
            let cc = to_canonical_cols[c as usize] as usize;
            if !order.iter().any(|&o| o as usize == cc) {
                col_idx.push(cc as u32);
                values.push(vals[slot[cc]]);
            }
        }
        row_ptr.push(col_idx.len());
    }

    CsrMatrix {
        nrows: n,
        ncols: a.ncols,
        row_ptr,
        col_idx,
        values,
    }
}

#[cfg(test)]
mod tests {
    //! Pin the handoff-order semantics on hand-traceable synthetic cases.

    use super::*;

    /// Three elements over four dofs: `[0,1,2]`, `[1,2,3]`, `[2,3,0]`.
    /// Row 0's stored structure is `{0,1,2,3}` (element 0 couples 0-1, 0-2;
    /// element 2 couples 0-3).  First-touch insertion for row 0:
    /// element 0 inserts 0,1,2 — element 2 inserts 3 — sequence `[0,1,2,3]`,
    /// reversed `[3,2,1,0]`, diagonal swapped to the front → `[0,2,1,3]`.
    #[test]
    fn reverse_first_touch_with_diag_swap() {
        let element_dofs = [vec![0u32, 1, 2], vec![1, 2, 3], vec![2, 3, 0]];
        // Stored ascending in partition numbering == canonical (identity map).
        let mut a = CsrMatrix::new_empty(4, 4);
        a.row_ptr = vec![0, 4, 8, 12, 16];
        a.col_idx = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        a.values = (0..16).map(|k| k as f64).collect();
        let to_canonical: Vec<u32> = (0..4).collect();

        let out = mfem_ptap_handoff_matrix(&a, &to_canonical, &element_dofs);

        // Row 0 follows the hand trace; the diagonal leads every row.
        assert_eq!(&out.col_idx[0..4], &[0, 2, 1, 3]);
        for r in 1..4usize {
            let row = &out.col_idx[out.row_ptr[r]..out.row_ptr[r + 1]];
            assert_eq!(row[0], r as u32, "diagonal must lead row {r}");
        }
        // Pure permutation: same (col, value) multiset, values moved intact.
        for r in 0..4usize {
            let src = (a.col_idx[a.row_ptr[r]..a.row_ptr[r + 1]]
                .iter()
                .zip(&a.values[a.row_ptr[r]..a.row_ptr[r + 1]]))
            .collect::<Vec<_>>();
            let dst = (out.col_idx[out.row_ptr[r]..out.row_ptr[r + 1]]
                .iter()
                .zip(&out.values[out.row_ptr[r]..out.row_ptr[r + 1]]))
            .collect::<Vec<_>>();
            let key = |m: &Vec<(&u32, &f64)>| {
                let mut v: Vec<(u32, u64)> = m
                    .iter()
                    .map(|&(c, val)| (*c, val.to_bits()))
                    .collect::<Vec<_>>();
                v.sort_unstable();
                v
            };
            assert_eq!(key(&src), key(&dst), "row {r} multiset must be intact");
        }
        // Spot-check the value movement on row 0: column 3 carries the value
        // stored at slot 3 (`3.0`), column 2 the slot-2 value (`2.0`).
        assert_eq!(&out.values[0..4], &[0.0, 2.0, 1.0, 3.0]);
    }

    /// A partition→canonical renumbering is applied together with the
    /// in-row reorder: output rows/cols live in canonical ids.
    #[test]
    fn canonical_renumber_and_order() {
        let element_dofs = [vec![0u32, 1, 2], vec![1, 2, 3], vec![2, 3, 0]];
        // Partition dof p lives at canonical id `3 - p`.
        let to_canonical: Vec<u32> = vec![3, 2, 1, 0];
        let mut a = CsrMatrix::new_empty(4, 4);
        a.row_ptr = vec![0, 4, 8, 12, 16];
        a.col_idx = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        a.values = (0..16).map(|k| k as f64).collect();

        let out = mfem_ptap_handoff_matrix(&a, &to_canonical, &element_dofs);

        // Canonical row 0 = partition row 3 (values 12,13,14,15 for
        // partition cols 0..3); the canonical col of partition col c is 3-c,
        // so the row stores canonical cols {3,2,1,0}.  First-touch replay:
        // element 0 = [0,1,2] contains 0 → insert 0,1,2; element 1 skips;
        // element 2 = [2,3,0] contains 0 → insert 3.  Sequence [0,1,2,3],
        // reversed [3,2,1,0], diagonal 0 at slot 3 swapped to the front →
        // [0,2,1,3].
        assert_eq!(&out.col_idx[0..4], &[0, 2, 1, 3]);
        // Canonical cols 0,2,1,3 = partition cols 3,1,2,0 → values
        // 15,13,14,12.
        assert_eq!(&out.values[0..4], &[15.0, 13.0, 14.0, 12.0]);
    }

    /// Rectangular handoff: row 1's host element lists trial dofs `[1, 3, 4]`
    /// and the stored row set is `{1, 3, 4, 5}` — column 5 was placed by a
    /// different element (defensive leftover) and must keep its stored
    /// position at the end, while the host columns come out in reverse
    /// (prepend) order `[4, 3, 1]`.
    #[test]
    fn rectangular_reverse_host_order() {
        let mut a = CsrMatrix::new_empty(2, 6);
        a.row_ptr = vec![0, 3, 7];
        a.col_idx = vec![1, 3, 4, 0, 1, 3, 4];
        a.values = vec![10.0, 30.0, 40.0, 0.0, 11.0, 13.0, 14.0];
        let to_canonical_rows = vec![0u32, 1u32];
        let to_canonical_cols: Vec<u32> = (0..6).collect();
        let host_of_row = vec![0u32, 1u32];
        let element_cols = [vec![0u32, 2, 5], vec![1, 3, 4]];

        let out = mfem_discrete_op_handoff_matrix(&a, &to_canonical_rows, &to_canonical_cols, &host_of_row, &element_cols);

        // Row 0's stored set {1,3,4} shares nothing with its host [0,2,5] →
        // all-defensive leftover in stored order [1,3,4].  Row 1: host list
        // [1,3,4] reversed → [4,3,1]; column 0 (not in the host list) is the
        // defensive leftover appended after the reconstructed prefix.
        assert_eq!(&out.col_idx[0..3], &[1, 3, 4]);
        assert_eq!(&out.values[0..3], &[10.0, 30.0, 40.0]);
        assert_eq!(&out.col_idx[3..7], &[4, 3, 1, 0]);
        assert_eq!(&out.values[3..7], &[14.0, 13.0, 11.0, 0.0]);
        // Pure permutation: multiset intact.
        let mut src: Vec<(u32, u64)> = a.col_idx.iter().zip(a.values.iter()).map(|(&c, &v)| (c, v.to_bits())).collect();
        let mut dst: Vec<(u32, u64)> = out.col_idx.iter().zip(out.values.iter()).map(|(&c, &v)| (c, v.to_bits())).collect();
        src.sort_unstable();
        dst.sort_unstable();
        assert_eq!(src, dst);
    }
}
