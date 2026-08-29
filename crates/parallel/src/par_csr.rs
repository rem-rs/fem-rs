//! Parallel distributed CSR matrix.
//!
//! [`ParCsrMatrix`] stores a distributed sparse matrix split into diagonal
//! (owned x owned) and off-diagonal (owned x ghost) blocks.  The parallel
//! SpMV uses ghost exchange to fetch remote vector entries before the local
//! matrix-vector products.

use std::sync::Arc;

use fem_core::Rank;
use fem_linalg::{CooMatrix, CsrMatrix};

use crate::comm::Comm;
use crate::dof_partition::DofPartition;
use crate::ghost::GhostExchange;
use crate::par_vector::ParVector;
use fem_mesh::amr::HangingNodeConstraint;

/// A distributed CSR matrix: `diag` block (owned columns) + `offd` block
/// (ghost columns).
///
/// Only rows for owned DOFs are stored; ghost rows are discarded during
/// construction (they are handled by the owning rank).
pub struct ParCsrMatrix {
    /// Diagonal block: `n_owned x n_owned`.
    pub(crate) diag: CsrMatrix<f64>,
    /// Off-diagonal block: `n_owned x n_ghost`.
    pub(crate) offd: CsrMatrix<f64>,
    /// Number of owned DOFs (= number of rows).
    pub(crate) n_owned: usize,
    /// Number of ghost DOFs.
    pub(crate) n_ghost: usize,
    /// Ghost exchange pattern for vector data.
    #[allow(dead_code)]
    dof_ghost_exchange: Arc<GhostExchange>,
    /// MPI communicator.
    #[allow(dead_code)]
    comm: Comm,
}

impl ParCsrMatrix {
    /// Build from pre-split diagonal and off-diagonal blocks.
    pub fn from_blocks(
        diag: CsrMatrix<f64>,
        offd: CsrMatrix<f64>,
        n_owned: usize,
        n_ghost: usize,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        ParCsrMatrix { diag, offd, n_owned, n_ghost, dof_ghost_exchange, comm }
    }

    /// Build from a local matrix (n_local x n_local where n_local = n_owned + n_ghost).
    ///
    /// Discards ghost rows (they are handled by the owning rank).  Splits
    /// columns into `diag` (col < n_owned) and `offd` (col >= n_owned,
    /// remapped to 0-based ghost index).
    ///
    /// `dof_part` is used to identify which ghost columns are actually cross-rank
    /// (owned by other ranks) vs local ghost-element DOFs owned by this rank.
    /// Only cross-rank ghost columns are placed in the offd block; local
    /// ghost-element columns are added to the diag block.
    pub fn from_local_matrix_with_partition(
        local: &CsrMatrix<f64>,
        n_owned: usize,
        dof_part: &DofPartition,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        let n_local = local.nrows;
        let n_ghost = n_local.saturating_sub(n_owned);

        // Build a set of cross-rank ghost local IDs for fast lookup.
        let cross_rank_ghosts: std::collections::HashSet<usize> = dof_part
            .ghost_dofs()
            .map(|(lid, _)| lid as usize)
            .collect();

        // Map from local ghost column index to offd column index.
        let ghost_col_map: std::collections::HashMap<usize, usize> = dof_part
            .ghost_dofs()
            .enumerate()
            .map(|(idx, (lid, _))| (lid as usize, idx))
            .collect();

        let n_cross_rank = cross_rank_ghosts.len();
        // Diag block: n_owned x (n_owned + local_ghost_cols)
        // But we use n_owned x n_owned for the owned-owned part,
        // and a separate structure for owned-local_ghost part.
        // For simplicity, use n_owned x n_local for diag (sparse).
        let mut diag_coo = CooMatrix::<f64>::new(n_owned, n_local);
        let mut offd_coo = CooMatrix::<f64>::new(n_owned, n_cross_rank);

        for row in 0..n_owned {
            for k in local.row_ptr[row]..local.row_ptr[row + 1] {
                let col = local.col_idx[k] as usize;
                let val = local.values[k];
                if val == 0.0 { continue; }
                if col < n_owned {
                    diag_coo.add(row, col, val);
                } else if cross_rank_ghosts.contains(&col) {
                    // Cross-rank ghost column -> offd block.
                    if let Some(&offd_col) = ghost_col_map.get(&col) {
                        offd_coo.add(row, offd_col, val);
                    }
                } else {
                    // Local ghost-element column (owned by this rank) -> diag block.
                    diag_coo.add(row, col, val);
                }
            }
        }

        let diag = diag_coo.into_csr();
        let offd = offd_coo.into_csr();

        ParCsrMatrix { diag, offd, n_owned, n_ghost: n_cross_rank, dof_ghost_exchange, comm }
    }

    /// Build from a local matrix (n_local x n_local where n_local = n_owned + n_ghost).
    ///
    /// Discards ghost rows (they are handled by the owning rank).  Splits
    /// columns into `diag` (col < n_owned) and `offd` (col >= n_owned,
    /// remapped to 0-based ghost index).
    pub fn from_local_matrix(
        local: &CsrMatrix<f64>,
        n_owned: usize,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        let n_local = local.nrows;
        let n_ghost = n_local.saturating_sub(n_owned);

        let mut diag_coo = CooMatrix::<f64>::new(n_owned, n_owned);

        for row in 0..n_owned {
            for k in local.row_ptr[row]..local.row_ptr[row + 1] {
                let col = local.col_idx[k] as usize;
                let val = local.values[k];
                if val == 0.0 { continue; }
                if col < n_owned {
                    diag_coo.add(row, col, val);
                }
            }
        }

        let diag = diag_coo.into_csr();
        let offd = if n_ghost > 0 {
            // Rebuild with exact column count to get correct ncols.
            let mut c = CooMatrix::<f64>::new(n_owned, n_ghost);
            for row in 0..n_owned {
                for k in local.row_ptr[row]..local.row_ptr[row + 1] {
                    let col = local.col_idx[k] as usize;
                    let val = local.values[k];
                    if val == 0.0 { continue; }
                    if col >= n_owned {
                        c.add(row, col - n_owned, val);
                    }
                }
            }
            c.into_csr()
        } else {
            CsrMatrix::new_empty(n_owned, 0)
        };

        ParCsrMatrix { diag, offd, n_owned, n_ghost, dof_ghost_exchange, comm }
    }

    /// Parallel SpMV: `y = A * x`.
    ///
    /// 1. Start the halo exchange for `x` and, on native MPI, overlap the
    ///    diagonal multiply `y[owned] = diag * x[owned]` while communication is
    ///    in flight ([`ParVector::update_ghosts_overlapping`]).
    /// 2. Complete the halo and apply `y[owned] += offd * x[ghost]`.
    ///
    /// On non-native backends the diagonal multiply runs strictly before blocking
    /// halo communication (same numerical result).
    pub fn spmv(&self, x: &mut ParVector, y: &mut ParVector) {
        let n = self.n_owned;
        let ng = self.n_ghost;
        let x_n = x.n_owned();

        // Post halo for x, overlap diagonal SpMV on native MPI (non-blocking P2P).
        x.update_ghosts_overlapping(|data| {
            // Use the vector's actual owned count for the diagonal multiply.
            let n_common = n.min(x_n).min(data.len());
            if n_common > 0 && n_common <= y.data.len() {
                self.diag.spmv(&data[..n_common], &mut y.data[..n_common]);
            }
        });

        if ng > 0 {
            // Use the vector's actual ghost count for indexing, not the matrix's.
            let x_ng = x.data.len().saturating_sub(x_n);
            let ng_common = ng.min(x_ng);
            if ng_common > 0 && x_n + ng_common <= x.data.len() && n <= y.data.len() {
                // Only multiply if dimensions match; otherwise skip (dimension mismatch
                // between matrix ghost columns and vector ghost section can occur
                // when the matrix and vector come from different parallel layouts).
                if self.offd.ncols == ng_common {
                    self.offd.spmv_add(
                        1.0,
                        &x.data[x_n..x_n + ng_common],
                        1.0,
                        &mut y.data[..n],
                    );
                }
            }
        }
    }

    /// Number of owned rows (= local portion of global matrix).
    pub fn n_owned(&self) -> usize { self.n_owned }

    /// Number of ghost columns (= columns from other ranks referenced by local rows).
    pub fn n_ghost(&self) -> usize { self.n_ghost }

    /// Diagonal block (owned × owned columns).
    pub fn diag_block(&self) -> &CsrMatrix<f64> { &self.diag }

    /// Off-diagonal block (owned × ghost columns).
    pub fn offd_block(&self) -> &CsrMatrix<f64> { &self.offd }

    /// Deep copy of `self` (clone diag, offd, ghost exchange, comm).
    pub fn clone_vec(&self) -> Self {
        ParCsrMatrix {
            diag: self.diag.clone(),
            offd: self.offd.clone(),
            n_owned: self.n_owned,
            n_ghost: self.n_ghost,
            dof_ghost_exchange: self.dof_ghost_exchange.clone(),
            comm: self.comm.clone(),
        }
    }

    /// Build a full local `CsrMatrix` (n_owned × n_total) from diag + offd.
    /// Columns `[0, n_owned)` come from `diag`; columns
    /// `[n_owned, n_total)` come from `offd` (shifted by `n_owned`).
    pub fn to_local_matrix(&self) -> CsrMatrix<f64> {
        let n_local = self.n_owned + self.n_ghost;
        let mut coo = CooMatrix::<f64>::new(self.n_owned, n_local);
        let d = &self.diag;
        for r in 0..d.nrows {
            for k in d.row_ptr[r]..d.row_ptr[r + 1] {
                coo.add(r, d.col_idx[k] as usize, d.values[k]);
            }
        }
        let o = &self.offd;
        for r in 0..o.nrows {
            for k in o.row_ptr[r]..o.row_ptr[r + 1] {
                coo.add(r, o.col_idx[k] as usize + self.n_owned, o.values[k]);
            }
        }
        coo.into_csr()
    }

    /// Mutable diagonal block.
    pub fn diag_block_mut(&mut self) -> &mut CsrMatrix<f64> { &mut self.diag }

    pub fn offd_block_mut(&mut self) -> &mut CsrMatrix<f64> { &mut self.offd }

    /// Apply hanging-node constraints to the assembled system:
    /// `K' = Pᵀ K P`, `f' = Pᵀ f` (constrained dofs eliminated, identity rows
    /// left for the constrained dofs so the solver can run on the full local
    /// vector).  Only **owned** constrained rows are processed; ghost rows are
    /// handled by their owning rank.  Constraint ids are **local dof ids**
    /// (P1: constrained = hanging edge-midpoint node, parents = coarse
    /// endpoints, coefficients 0.5/0.5).
    ///
    /// Mirrors the serial `apply_hanging_constraints` (PᵀKP via COO rebuild)
    /// but keeps the matrix `n_local × n_local` so it can be re-split into
    /// diag/offd blocks afterwards.
    pub fn apply_hanging_constraints(
        &mut self,
        constraints: &[HangingNodeConstraint],
        rhs: &mut ParVector,
        dof_part: &DofPartition,
    ) {
        if constraints.is_empty() {
            return;
        }
        let n_owned = self.n_owned;
        let n_local = n_owned + self.n_ghost;

        // Constraint ids are local **node ids** (H1 P1: dof id = node id);
        // the assembled matrix/rhs live in partition order — permute.
        let permute = |id: usize| dof_part.permute_dof(id as u32) as usize;
        let mut constraint_map: std::collections::HashMap<usize, Vec<(usize, f64)>> =
            std::collections::HashMap::new();
        for c in constraints {
            constraint_map.insert(
                permute(c.constrained),
                c.parents()
                    .map(|(p, w)| (permute(p), w))
                    .collect(),
            );
        }

        fn expand_dof(
            dof: usize,
            weight: f64,
            cmap: &std::collections::HashMap<usize, Vec<(usize, f64)>>,
            out: &mut Vec<(usize, f64)>,
            depth: usize,
        ) {
            if depth > 20 {
                return;
            }
            if let Some(parents) = cmap.get(&dof) {
                for &(p, coeff) in parents {
                    expand_dof(p, weight * coeff, cmap, out, depth + 1);
                }
            } else {
                out.push((dof, weight));
            }
        }

        // ── K' = Pᵀ K P on the full local matrix (n_local × n_local) ─────────
        let a = self.to_local_matrix();
        let mut coo = CooMatrix::<f64>::new(n_local, n_local);
        // cross-rank row sends (PᵀKP): rank → Vec<(global_row, Vec<(global_col, val)>)>
        let mut row_sends: std::collections::HashMap<Rank, Vec<(u32, Vec<(u32, f64)>)>> =
            std::collections::HashMap::new();
        let comm = self.comm.clone();
        for i in 0..n_local {
            if i >= n_owned {
                continue; // ghost rows: owned by another rank
            }
            let mut i_targets = Vec::new();
            expand_dof(i, 1.0, &constraint_map, &mut i_targets, 0);
            for k in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[k] as usize;
                let v = a.values[k];
                if v.abs() < 1e-30 {
                    continue;
                }
                let mut j_targets = Vec::new();
                expand_dof(j, 1.0, &constraint_map, &mut j_targets, 0);
                for &(ii, ai) in &i_targets {
                    if ii >= n_local {
                        continue;
                    }
                    for &(jj, aj) in &j_targets {
                        if jj >= n_local {
                            continue;
                        }
                        let entry = v * ai * aj;
                        if entry.abs() < 1e-30 {
                            continue;
                        }
                        if ii < n_owned {
                            coo.add(ii, jj, entry);
                        } else {
                            // row ii is a ghost dof: its owner must fold this
                            // entry into its copy of row ii.
                            let owner = dof_part.dof_owner(ii as u32);
                            row_sends
                                .entry(owner)
                                .or_default()
                                .push((dof_part.global_dof(ii as u32), vec![(dof_part.global_dof(jj as u32), entry)]));
                        }
                    }
                }
            }
        }
        // Exchange ghost rows: group by (global_row, global_col) at the owner.
        // NOTE: all ranks must enter the alltoallv (a rank with no sends
        // still participates with an empty payload), otherwise the collective
        // deadlocks — pex6 RT0 flux constraints can leave one rank with empty
        // row_sends while the other has entries.
        if comm.size() > 1 {
            // coalesce per-rank lists
            let payloads: Vec<(Rank, Vec<u8>)> = row_sends
                .iter()
                .map(|(&dst, list)| {
                    let mut buf = Vec::new();
                    for &(gr, ref row) in list {
                        buf.extend_from_slice(&gr.to_le_bytes());
                        buf.extend_from_slice(&(row.len() as u32).to_le_bytes());
                        for &(gc, v) in row {
                            buf.extend_from_slice(&gc.to_le_bytes());
                            buf.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    (dst, buf)
                })
                .collect();
            let incoming = comm.alltoallv_bytes(&payloads);
            let gid_to_local: std::collections::HashMap<u32, u32> = (0..n_local)
                .map(|d| (dof_part.global_dof(d as u32), d as u32))
                .collect();
            for (_src, bytes) in incoming {
                let mut off = 0usize;
                while off + 8 <= bytes.len() {
                    let gr = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                    let n_ent = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
                    off += 8;
                    let Some(&li) = gid_to_local.get(&gr) else {
                        off += n_ent * 12;
                        continue;
                    };
                    let li = li as usize;
                    if li >= n_owned {
                        off += n_ent * 12;
                        continue;
                    }
                    for _ in 0..n_ent {
                        let gc = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                        let val = f64::from_le_bytes(bytes[off + 4..off + 12].try_into().unwrap());
                        off += 12;
                        if let Some(&lc) = gid_to_local.get(&gc) {
                            coo.add(li, lc as usize, val);
                        }
                    }
                }
            }
        }
        // Identity rows for owned constrained dofs (solver keeps them = 0).
        for c in constraints {
            let ci = permute(c.constrained);
            if ci < n_owned {
                coo.add(ci, ci, 1.0);
            }
        }
        let new_a = coo.into_csr_sorted();

        // ── f' = Pᵀ f (owned + cross-rank ghost-parent contributions) ───────
        let mut new_rhs = vec![0.0_f64; n_owned];
        // cross-rank rhs sends: rank → Vec<(global_dof, contribution)>
        let mut rhs_sends: std::collections::HashMap<Rank, Vec<(u32, f64)>> =
            std::collections::HashMap::new();
        for i in 0..n_owned {
            let r = rhs.data[i];
            if r.abs() < 1e-30 {
                continue;
            }
            let mut targets = Vec::new();
            expand_dof(i, 1.0, &constraint_map, &mut targets, 0);
            for &(d, w) in &targets {
                if d < n_owned {
                    new_rhs[d] += w * r;
                } else if d < n_local {
                    let owner = dof_part.dof_owner(d as u32);
                    rhs_sends
                        .entry(owner)
                        .or_default()
                        .push((dof_part.global_dof(d as u32), w * r));
                }
            }
        }
        if comm.size() > 1 { // all ranks enter (empty sends are fine)
            let payloads: Vec<(Rank, Vec<u8>)> = rhs_sends
                .iter()
                .map(|(&dst, list)| {
                    let mut buf = Vec::with_capacity(list.len() * 12);
                    for &(g, c) in list {
                        buf.extend_from_slice(&g.to_le_bytes());
                        buf.extend_from_slice(&c.to_le_bytes());
                    }
                    (dst, buf)
                })
                .collect();
            let incoming = comm.alltoallv_bytes(&payloads);
            let gid_to_owned: std::collections::HashMap<u32, usize> = (0..n_owned)
                .map(|d| (dof_part.global_dof(d as u32), d))
                .collect();
            for (_src, bytes) in incoming {
                for chunk in bytes.chunks_exact(12) {
                    let g = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let c = f64::from_le_bytes(chunk[4..12].try_into().unwrap());
                    if let Some(&d) = gid_to_owned.get(&g) {
                        new_rhs[d] += c;
                    }
                }
            }
        }
        for c in constraints {
            let ci = permute(c.constrained);
            if ci < n_owned {
                new_rhs[ci] = 0.0;
            }
        }
        rhs.data[..n_owned].copy_from_slice(&new_rhs);

        // ── Re-split into diag/offd blocks ───────────────────────────────────
        // NB: use `from_local_matrix` (square diag), NOT the partition-aware
        // variant — a non-square diag breaks csr_spmm in the AMG solver
        // (pex33 regression lesson).
        let ge = self.dof_ghost_exchange.clone();
        let comm = self.comm.clone();
        *self = ParCsrMatrix::from_local_matrix(&new_a, n_owned, ge, comm);
    }

    /// Ghost exchange handle.
    pub fn ghost_exchange_handle(&self) -> Arc<GhostExchange> { self.dof_ghost_exchange.clone() }

    /// Extract the diagonal of the owned block.
    pub fn diagonal(&self) -> Vec<f64> {
        self.diag.diagonal()
    }

    /// Arc-wrapped ghost exchange (for sharing with other structures).
    pub fn ghost_exchange_arc(&self) -> Arc<GhostExchange> {
        Arc::clone(&self.dof_ghost_exchange)
    }

    /// The MPI communicator.
    pub fn comm(&self) -> &Comm { &self.comm }

    /// Apply Dirichlet BC at a local owned DOF: zero the row, set diagonal
    /// to 1, set `rhs[dof] = value`.
    pub fn apply_dirichlet_row(&mut self, local_dof: usize, value: f64, rhs: &mut [f64]) {
        assert!(local_dof < self.n_owned, "can only apply Dirichlet to owned DOFs");

        // Zero diag row and set diagonal to 1.
        self.diag.apply_dirichlet_row_zeroing(local_dof, value, rhs);

        // Zero offd row.
        if self.n_ghost > 0 {
            let start = self.offd.row_ptr[local_dof];
            let end = self.offd.row_ptr[local_dof + 1];
            for k in start..end {
                self.offd.values[k] = 0.0;
            }
        }
    }

    /// Apply Dirichlet BC at a local owned DOF using a `ParVector` as the RHS.
    pub fn apply_dirichlet_par(&mut self, local_dof: usize, value: f64, rhs: &mut ParVector) {
        self.apply_dirichlet_row(local_dof, value, rhs.as_slice_mut());
    }

    /// Apply a Dirichlet BC in MFEM's `DIAG_KEEP` style (`EliminateRowCol`,
    /// the `FormLinearSystem` default used by ex27p) at an owned DOF:
    /// - zero the off-diagonal entries of row and column `local_dof`
    ///   (symmetric elimination, keeps the matrix symmetric for CG);
    /// - **keep** the diagonal entry `A[local_dof, local_dof]` unchanged;
    /// - `rhs[local_dof] = A[local_dof, local_dof] · value`;
    /// - for every other owned row `j`: `rhs[j] -= A[j, local_dof] · value`.
    ///
    /// The off-diagonal (ghost-column) part of row `local_dof` is zeroed; the
    /// owned column `local_dof` never appears in the off-diagonal block (its
    /// columns are ghost DOFs only), so the symmetric elimination is confined
    /// to the diagonal block.
    pub fn apply_dirichlet_par_keep_diag(&mut self, local_dof: usize, value: f64, rhs: &mut ParVector) {
        assert!(local_dof < self.n_owned, "can only apply Dirichlet to owned DOFs");
        self.diag.apply_dirichlet_keep_diag(local_dof, value, rhs.as_slice_mut());
        // Zero the ghost-column part of this row.
        if self.n_ghost > 0 {
            let start = self.offd.row_ptr[local_dof];
            let end = self.offd.row_ptr[local_dof + 1];
            for k in start..end {
                self.offd.values[k] = 0.0;
            }
        }
    }

    /// Complete the MFEM `DIAG_KEEP` (FormLinearSystem) elimination for
    /// **non-homogeneous** essential DOFs whose couplings cross ranks.
    ///
    /// [`apply_dirichlet_par_keep_diag`] only eliminates the owned
    /// (diagonal-block) rows/columns of an essential DOF.  When the essential
    /// DOF is a **ghost column** here (owned by another rank), this rank's
    /// rows must still receive the `-A[j, d]·x_bc` contribution on the RHS and
    /// the column entries must be zeroed to keep the matrix symmetric for PCG.
    ///
    /// `ghost_ess` lists the local ghost slots (`0..n_ghost`) that are
    /// essential together with their Dirichlet values (callers obtain these
    /// from a global-id exchange of the locally-detected essential DOFs).
    pub fn apply_ghost_ess_columns(&mut self, ghost_ess: &[(usize, f64)], rhs: &mut ParVector) {
        if self.n_ghost == 0 || ghost_ess.is_empty() {
            return;
        }
        let offd = &mut self.offd;
        let rhs_data = rhs.as_slice_mut();
        for &(g, v) in ghost_ess {
            if v != 0.0 {
                for row in 0..self.n_owned {
                    let s = offd.row_ptr[row];
                    let e = offd.row_ptr[row + 1];
                    for k in s..e {
                        if offd.col_idx[k] as usize == g && offd.values[k] != 0.0 {
                            rhs_data[row] -= offd.values[k] * v;
                        }
                    }
                }
            }
        }
        // Zero the essential ghost columns (symmetry).
        for &(g, _) in ghost_ess {
            for row in 0..self.n_owned {
                let s = offd.row_ptr[row];
                let e = offd.row_ptr[row + 1];
                for k in s..e {
                    if offd.col_idx[k] as usize == g {
                        offd.values[k] = 0.0;
                    }
                }
            }
        }
    }

    /// MFEM‑style symmetric diagonal elimination: zero row AND column for
    /// `owned_dofs`, then set diagonal entry to `val` for each.
    ///
    /// This mimics `EliminateEssentialBCDiag` and is useful for eigenvalue
    /// problems where the RHS is not a fixed vector but the mass‑matrix
    /// product B x_k.  Symmetry is preserved (A_ij = A_ji for all i,j).
    pub fn eliminate_diag_symmetric(&mut self, owned_dofs: &[usize], val: f64) {
        self.eliminate_diag_symmetric_with_ghost(owned_dofs, &[], val)
    }

    /// Like [`eliminate_diag_symmetric`](Self::eliminate_diag_symmetric),
    /// but also zeroes the offd columns of `ghost_boundary_cols` (ghost‑side
    /// slots, i.e. offd column indices) so that boundary DOFs owned by other
    /// ranks have their *column* contributions removed on this rank too.
    ///
    /// Without this, a boundary DOF `d` owned on rank A appears in rank B's
    /// offd block as a ghost column whose values survive elimination (rank B
    /// only zeroes rows of its own boundary DOFs).  The resulting operator
    /// has A_ij ≠ A_ji across ranks for boundary‑adjacent pairs — asymmetric,
    /// which breaks PCG/CG on np > 1.  The caller obtains these indices by
    /// permuting the boundary DOFs to partition order and keeping slots
    /// ≥ `n_owned_dofs` (minus `n_owned_dofs`).
    pub fn eliminate_diag_symmetric_with_ghost(
        &mut self,
        owned_dofs: &[usize],
        ghost_boundary_cols: &[usize],
        val: f64,
    ) {
        let n_owned = self.n_owned;
        let diag = &mut self.diag;

        for &d in owned_dofs {
            if d >= n_owned { continue; }

            // ── Zero row d ──
            for p in diag.row_ptr[d]..diag.row_ptr[d + 1] {
                diag.values[p] = 0.0;
            }
            // ── Zero column d (all rows i != d that reference column d) ──
            for i in 0..n_owned {
                if i == d { continue; }
                for p in diag.row_ptr[i]..diag.row_ptr[i + 1] {
                    if diag.col_idx[p] as usize == d {
                        diag.values[p] = 0.0;
                        break;
                    }
                }
            }
            // ── Set diag[d] = val ──
            for p in diag.row_ptr[d]..diag.row_ptr[d + 1] {
                if diag.col_idx[p] as usize == d {
                    diag.values[p] = val;
                    break;
                }
            }

            // ── Zero offd row d ──
            if self.n_ghost > 0 {
                for p in self.offd.row_ptr[d]..self.offd.row_ptr[d + 1] {
                    self.offd.values[p] = 0.0;
                }
            }
        }

        // ── Zero offd columns of boundary DOFs owned by other ranks ──
        if self.n_ghost > 0 && !ghost_boundary_cols.is_empty() {
            let offd = &mut self.offd;
            let mut cols: std::collections::HashSet<usize> =
                ghost_boundary_cols.iter().copied().collect();
            for i in 0..n_owned {
                for p in offd.row_ptr[i]..offd.row_ptr[i + 1] {
                    let c = offd.col_idx[p] as usize;
                    if cols.contains(&c) {
                        offd.values[p] = 0.0;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_partition::partition_mesh;
    use crate::par_space::ParallelFESpace;
    use crate::ghost::GhostExchange;
    use crate::par_vector::ParVector;
    use fem_linalg::CooMatrix;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn par_csr_from_local_splits_correctly() {
        // 4x4 matrix with n_owned=2, n_ghost=2.
        // Row 0: (0,0)=1, (0,1)=2, (0,2)=3, (0,3)=4
        // Row 1: (1,0)=5, (1,1)=6, (1,2)=7, (1,3)=8
        // Row 2: (2,0)=9, ... (ghost row, should be discarded)
        // Row 3: (3,3)=10  (ghost row, should be discarded)
        let mut coo = CooMatrix::<f64>::new(4, 4);
        coo.add(0, 0, 1.0); coo.add(0, 1, 2.0); coo.add(0, 2, 3.0); coo.add(0, 3, 4.0);
        coo.add(1, 0, 5.0); coo.add(1, 1, 6.0); coo.add(1, 2, 7.0); coo.add(1, 3, 8.0);
        coo.add(2, 0, 9.0); coo.add(2, 1, 10.0); coo.add(2, 2, 11.0); coo.add(2, 3, 12.0);
        coo.add(3, 3, 10.0);
        let csr = coo.into_csr();

        // Use a trivial (serial) ghost exchange — no actual communication needed.
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let ghost_ex = Arc::new(GhostExchange::from_trivial());
            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, 2, ghost_ex, comm.clone(),
            );

            // diag should be 2x2: [(0,0)=1, (0,1)=2; (1,0)=5, (1,1)=6]
            assert_eq!(par_mat.diag.nrows, 2);
            assert_eq!(par_mat.diag.ncols, 2);
            assert!((par_mat.diag.get(0, 0) - 1.0).abs() < 1e-14);
            assert!((par_mat.diag.get(0, 1) - 2.0).abs() < 1e-14);
            assert!((par_mat.diag.get(1, 0) - 5.0).abs() < 1e-14);
            assert!((par_mat.diag.get(1, 1) - 6.0).abs() < 1e-14);

            // offd should be 2x2: [(0,0)=3, (0,1)=4; (1,0)=7, (1,1)=8]
            assert_eq!(par_mat.offd.nrows, 2);
            assert_eq!(par_mat.offd.ncols, 2);
            assert!((par_mat.offd.get(0, 0) - 3.0).abs() < 1e-14);
            assert!((par_mat.offd.get(0, 1) - 4.0).abs() < 1e-14);
            assert!((par_mat.offd.get(1, 0) - 7.0).abs() < 1e-14);
            assert!((par_mat.offd.get(1, 1) - 8.0).abs() < 1e-14);

            assert_eq!(par_mat.n_owned, 2);
            assert_eq!(par_mat.n_ghost, 2);
        });
    }

    #[test]
    fn par_csr_spmv_identity() {
        // Parallel SpMV with identity matrix on 2 ranks.
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let n_local = par_space.n_local_dofs();
            let n_owned = par_space.dof_partition().n_owned_dofs;

            // Build identity matrix over all local DOFs.
            let mut coo = CooMatrix::<f64>::new(n_local, n_local);
            for i in 0..n_local { coo.add(i, i, 1.0); }
            let csr = coo.into_csr();

            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, n_owned,
                par_space.dof_ghost_exchange_arc(),
                comm.clone(),
            );

            // x = [1, 2, 3, ...] for owned, ghosts will be filled by exchange.
            let mut x = ParVector::zeros(&par_space);
            for (i, v) in x.owned_slice_mut().iter_mut().enumerate() {
                *v = (i + 1) as f64;
            }

            let mut y = ParVector::zeros(&par_space);
            // `spmv` updates ghost DOFs in `x` before using the off-diagonal block.
            par_mat.spmv(&mut x, &mut y);

            // y[owned] should equal x[owned] (identity).
            for i in 0..n_owned {
                assert!(
                    (y.as_slice()[i] - x.as_slice()[i]).abs() < 1e-14,
                    "rank {}: spmv identity mismatch at owned DOF {i}: y={}, x={}",
                    comm.rank(), y.as_slice()[i], x.as_slice()[i]
                );
            }
        });
    }

    #[test]
    fn par_csr_from_local_serial() {
        // 3x3 tridiag on 1 rank: all columns are "owned".
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.add(0, 0, 2.0); coo.add(0, 1, -1.0);
        coo.add(1, 0, -1.0); coo.add(1, 1, 2.0); coo.add(1, 2, -1.0);
        coo.add(2, 1, -1.0); coo.add(2, 2, 2.0);
        let csr = coo.into_csr();

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let mesh = Mesh::<2>::unit_square_tri(2);
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, 3,
                par_space.dof_ghost_exchange_arc(),
                comm.clone(),
            );

            assert_eq!(par_mat.diag.nrows, 3);
            assert_eq!(par_mat.n_ghost, 0);
            // Diagonal values.
            let diag = par_mat.diagonal();
            assert!((diag[0] - 2.0).abs() < 1e-14);
            assert!((diag[1] - 2.0).abs() < 1e-14);
            assert!((diag[2] - 2.0).abs() < 1e-14);
        });
    }

    #[test]
    fn par_csr_spmv_serial() {
        // Verify serial SpMV gives correct result.
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let n = par_space.n_local_dofs();
            // Build identity matrix.
            let mut coo = CooMatrix::<f64>::new(n, n);
            for i in 0..n { coo.add(i, i, 1.0); }
            let csr = coo.into_csr();

            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, n,
                par_space.dof_ghost_exchange_arc(),
                comm.clone(),
            );

            // x = [1, 2, 3, ...]
            let mut x = ParVector::zeros(&par_space);
            for (i, v) in x.as_slice_mut().iter_mut().enumerate() {
                *v = (i + 1) as f64;
            }
            let mut y = ParVector::zeros(&par_space);
            par_mat.spmv(&mut x, &mut y);

            // y should equal x (identity).
            for i in 0..n {
                assert!(
                    (y.as_slice()[i] - x.as_slice()[i]).abs() < 1e-14,
                    "spmv mismatch at {i}"
                );
            }
        });
    }
}
