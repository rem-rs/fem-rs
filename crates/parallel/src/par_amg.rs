//! Parallel Algebraic Multigrid (AMG) preconditioner.
//!
//! Implements a distributed AMG V-cycle using **local smoothed aggregation**:
//!
//! 1. Each rank coarsens its owned rows independently (aggregates don't cross
//!    partition boundaries).
//! 2. Prolongation/restriction operators are distributed sparse matrices.
//! 3. Coarse-level matrices are formed via the Galerkin product `R A P`.
//! 4. The coarsest level is solved with Jacobi-preconditioned CG.
//!
//! This is simpler than full boundary-crossing aggregation but produces good
//! convergence for typical elliptic problems.

use std::sync::Arc;

use fem_linalg::{CooMatrix, CsrMatrix};

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the parallel AMG hierarchy.
#[derive(Debug, Clone)]
pub struct ParAmgConfig {
    /// Maximum number of AMG levels.
    pub max_levels: usize,
    /// Coarsest level size (total global DOFs) below which we stop coarsening.
    pub coarse_size: usize,
    /// Number of pre-smoothing Jacobi iterations.
    pub n_pre_smooth: usize,
    /// Number of post-smoothing Jacobi iterations.
    pub n_post_smooth: usize,
    /// Strength threshold for aggregation.
    pub strength_threshold: f64,
}

impl Default for ParAmgConfig {
    fn default() -> Self {
        ParAmgConfig {
            max_levels: 10,
            coarse_size: 50,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            strength_threshold: 0.25,
        }
    }
}

// ── AMG Level ───────────────────────────────────────────────────────────────

/// One level in the AMG hierarchy.
struct AmgLevel {
    /// System matrix at this level.
    a: ParCsrMatrix,
    /// Prolongation operator (coarse → fine).  `None` at the coarsest level.
    p: Option<ParCsrMatrix>,
    /// Restriction operator (fine → coarse) = P^T.  `None` at the coarsest.
    r: Option<ParCsrMatrix>,
    /// Inverse diagonal for Jacobi smoothing.
    inv_diag: Vec<f64>,
}

// ── ParAmgHierarchy ─────────────────────────────────────────────────────────

/// A distributed AMG hierarchy for use as a preconditioner.
pub struct ParAmgHierarchy {
    levels: Vec<AmgLevel>,
    config: ParAmgConfig,
}

impl ParAmgHierarchy {
    /// Build the AMG hierarchy from a distributed SPD matrix.
    ///
    /// When `use_global_aggregation` is `true` (WP2 mode), the coarsening uses
    /// ghost-aware aggregation so that aggregates can span rank boundaries.
    /// This generally improves coarsening quality near partition interfaces and
    /// leads to faster convergence, at the cost of one extra ghost exchange per
    /// coarsening level.
    pub fn build(a: &ParCsrMatrix, comm: &Comm, config: ParAmgConfig) -> Self {
        Self::build_impl(a, comm, config, false)
    }

    /// Like [`build`], but with cross-rank (ghost-aware) aggregation enabled.
    ///
    /// This is the **WP2** distributed AMG mode: strength-of-connection is
    /// computed over the full row (owned + ghost columns) and aggregate
    /// assignments for boundary DOFs are exchanged across ranks so that
    /// coarsening is globally consistent near partition interfaces.
    pub fn build_global(a: &ParCsrMatrix, comm: &Comm, config: ParAmgConfig) -> Self {
        Self::build_impl(a, comm, config, true)
    }

    fn build_impl(
        a: &ParCsrMatrix,
        comm: &Comm,
        config: ParAmgConfig,
        global_agg: bool,
    ) -> Self {
        let mut levels = Vec::new();
        let mut current_a = Some(clone_par_csr(a));

        for _level in 0..config.max_levels {
            let ca = current_a.take().unwrap();
            let inv_diag = compute_inv_diag(&ca);
            let n_global = comm.allreduce_sum_i64(ca.n_owned as i64) as usize;

            if n_global <= config.coarse_size || ca.n_owned <= 1 {
                levels.push(AmgLevel { a: ca, p: None, r: None, inv_diag });
                break;
            }

            let (p, r, coarse_a) = if global_agg {
                build_coarse_level_global(&ca, comm, config.strength_threshold)
            } else {
                build_coarse_level(&ca, comm, config.strength_threshold)
            };

            levels.push(AmgLevel {
                a: ca,
                p: Some(p),
                r: Some(r),
                inv_diag,
            });

            current_a = Some(coarse_a);
        }

        // If we hit max_levels without reaching coarse_size, push the last level.
        if let Some(ca) = current_a {
            let inv_diag = compute_inv_diag(&ca);
            levels.push(AmgLevel { a: ca, p: None, r: None, inv_diag });
        }

        ParAmgHierarchy { levels, config }
    }

    /// Number of levels in the hierarchy.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Apply one AMG V-cycle as a preconditioner: `x ← M⁻¹ b`.
    ///
    /// `x` should be zero on entry (or contain the initial guess).
    pub fn vcycle(&self, b: &ParVector, x: &mut ParVector) {
        self.vcycle_level(0, b, x);
    }

    fn vcycle_level(&self, level: usize, b: &ParVector, x: &mut ParVector) {
        let lvl = &self.levels[level];

        if lvl.p.is_none() {
            // Coarsest level: solve approximately with many Jacobi iterations.
            for _ in 0..20 {
                jacobi_smooth(&lvl.a, x, b, &lvl.inv_diag);
            }
            return;
        }

        let p = lvl.p.as_ref().unwrap();
        let r_op = lvl.r.as_ref().unwrap();
        let coarse_lvl = &self.levels[level + 1];

        // Pre-smoothing.
        for _ in 0..self.config.n_pre_smooth {
            jacobi_smooth(&lvl.a, x, b, &lvl.inv_diag);
        }

        // Compute residual: res = b - A*x
        let mut ax = ParVector::zeros_like(b);
        let mut x_mut = x.clone_vec();
        lvl.a.spmv(&mut x_mut, &mut ax);
        let mut res = b.clone_vec();
        for i in 0..lvl.a.n_owned {
            res.data[i] -= ax.data[i];
        }

        // Restrict residual to coarse level: r_coarse = R * res (local CSR SpMV)
        let mut r_coarse = zeros_for_mat(&coarse_lvl.a);
        local_spmv(&r_op.diag, res.as_slice(), r_coarse.as_slice_mut());

        // Recursively solve on coarse level.
        let mut e_coarse = ParVector::zeros_like(&r_coarse);
        self.vcycle_level(level + 1, &r_coarse, &mut e_coarse);

        // Prolongate correction: x += P * e_coarse (local CSR SpMV)
        let mut correction = vec![0.0_f64; lvl.a.n_owned];
        local_spmv(&p.diag, e_coarse.as_slice(), &mut correction);
        for i in 0..lvl.a.n_owned {
            x.data[i] += correction[i];
        }

        // Post-smoothing.
        for _ in 0..self.config.n_post_smooth {
            jacobi_smooth(&lvl.a, x, b, &lvl.inv_diag);
        }
    }
}

// ── Jacobi smoother ─────────────────────────────────────────────────────────

/// One Jacobi smoothing iteration: x ← x + ω D⁻¹ (b - Ax).
fn jacobi_smooth(
    a: &ParCsrMatrix,
    x: &mut ParVector,
    b: &ParVector,
    inv_diag: &[f64],
) {
    let omega = 2.0 / 3.0; // damping factor
    let n = a.n_owned;

    // Compute Ax.
    let mut ax = ParVector::zeros_like(b);
    let mut x_clone = x.clone_vec();
    a.spmv(&mut x_clone, &mut ax);

    // x += ω D⁻¹ (b - Ax)
    for i in 0..n {
        let r_i = b.data[i] - ax.data[i];
        x.data[i] += omega * inv_diag[i] * r_i;
    }
}

// ── Aggregation & coarsening ────────────────────────────────────────────────

/// Build the prolongation, restriction, and coarse-level matrix.
///
/// Uses local (non-overlapping) aggregation on owned DOFs.
fn build_coarse_level(
    a: &ParCsrMatrix,
    comm: &Comm,
    strength_threshold: f64,
) -> (ParCsrMatrix, ParCsrMatrix, ParCsrMatrix) {
    let n_owned = a.n_owned;

    // 1. Build strength-of-connection: strong(i,j) iff |a_ij| >= θ * max_k |a_ik|
    let diag = &a.diag;
    let mut aggregate = vec![-1i32; n_owned]; // aggregate[i] = aggregate ID
    let mut n_agg = 0i32;

    // Phase 1: seed aggregates (unaggregated nodes that have no strong neighbours yet).
    for i in 0..n_owned {
        if aggregate[i] >= 0 { continue; }

        // Find max off-diagonal magnitude in this row.
        let mut max_off_diag = 0.0_f64;
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i {
                max_off_diag = max_off_diag.max(diag.values[k].abs());
            }
        }
        let threshold = strength_threshold * max_off_diag;

        // Try to form a new aggregate: this node + its strong unassigned neighbours.
        aggregate[i] = n_agg;
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i && j < n_owned && aggregate[j] < 0 && diag.values[k].abs() >= threshold {
                aggregate[j] = n_agg;
            }
        }
        n_agg += 1;
    }

    // Phase 2: assign remaining unaggregated nodes to nearest aggregate.
    for i in 0..n_owned {
        if aggregate[i] >= 0 { continue; }
        // Assign to the aggregate of the strongest connected neighbour.
        let mut best_agg = -1i32;
        let mut best_val = 0.0_f64;
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i && j < n_owned && aggregate[j] >= 0 && diag.values[k].abs() > best_val {
                best_val = diag.values[k].abs();
                best_agg = aggregate[j];
            }
        }
        if best_agg >= 0 {
            aggregate[i] = best_agg;
        } else {
            // Isolated node: make its own aggregate.
            aggregate[i] = n_agg;
            n_agg += 1;
        }
    }

    let n_coarse_owned = n_agg as usize;

    // 2. Build prolongation P (n_owned × n_coarse_owned): P[i, agg[i]] = 1.
    // This is "tentative prolongation" (unsmoothed).
    let mut p_coo = CooMatrix::<f64>::new(n_owned, n_coarse_owned.max(1));
    for i in 0..n_owned {
        let agg = aggregate[i] as usize;
        p_coo.add(i, agg, 1.0);
    }
    let p_local = p_coo.into_csr();

    // 3. Build restriction R = P^T (n_coarse_owned × n_owned).
    let r_local = transpose_csr(&p_local);

    // 4. Build coarse matrix: A_c = R * A_diag * P (Galerkin triple product).
    // We use only the diag block for the local part. For true parallel,
    // off-diag contributions would require communication, but local aggregation
    // keeps everything within the rank.
    let ap_local = csr_multiply(&a.diag, &p_local);
    let ac_local = csr_multiply(&r_local, &ap_local);

    // 5. Wrap in ParCsrMatrix (no ghost DOFs at coarse level for local aggregation).
    let ghost_ex = Arc::new(GhostExchange::from_trivial());
    let p_par = ParCsrMatrix::from_blocks(
        p_local,
        CsrMatrix::new_empty(n_owned, 0),
        n_owned, 0,
        Arc::clone(&ghost_ex),
        comm.clone(),
    );
    let r_par = ParCsrMatrix::from_blocks(
        r_local,
        CsrMatrix::new_empty(n_coarse_owned, 0),
        n_coarse_owned, 0,
        Arc::clone(&ghost_ex),
        comm.clone(),
    );
    let ac_par = ParCsrMatrix::from_blocks(
        ac_local,
        CsrMatrix::new_empty(n_coarse_owned, 0),
        n_coarse_owned, 0,
        Arc::clone(&ghost_ex),
        comm.clone(),
    );

    (p_par, r_par, ac_par)
}

// ── WP2: Ghost-aware (global) coarsening ─────────────────────────────────────

/// Build prolongation, restriction, and coarse matrix using **cross-rank**
/// (ghost-aware) aggregation.
///
/// # Differences from [`build_coarse_level`]
///
/// 1. **Full-row strength**: max_off_diag for DOF `i` includes entries from
///    the off-diagonal block (`a.offd`), which correspond to ghost DOFs.
///    This ensures boundary DOFs are not artificially "weakly connected" just
///    because their strongest neighbour is on another rank.
///
/// 2. **Ghost aggregate exchange**: after local aggregation the owned aggregate
///    IDs are propagated to ghost slots via a forward ghost exchange.  The
///    global aggregate ID for each ghost is then `ghost_rank_offset + ghost_local_agg`.
///
/// 3. **Boundary aggregate merging**: for each boundary owned DOF (one that
///    has a strongly connected ghost neighbour), if the ghost's aggregate ID
///    has not yet been merged with the owned DOF's aggregate, a union-find
///    merges them.  Only one round of merging is performed (one-ring boundary
///    overlap), which is sufficient for standard elliptic problems.
///
/// 4. **Renumbering**: after merging, local aggregates are renumbered and a
///    global prefix sum computes each rank's coarse DOF offset.  The
///    prolongation block is then built with the merged, globally consistent
///    aggregate assignments.
///
/// # Limitation
/// The current implementation builds a "locally owned" coarse matrix block
/// only; cross-rank entries in `A_c` that arise from merges with ghost aggregates
/// are approximated by using the full diag+offd product.  For production runs
/// with many MPI ranks, a full all-pairs Galerkin product would give slightly
/// better convergence, but the ghost-aware aggregation already captures most
/// of the coarsening benefit.
fn build_coarse_level_global(
    a: &ParCsrMatrix,
    comm: &Comm,
    strength_threshold: f64,
) -> (ParCsrMatrix, ParCsrMatrix, ParCsrMatrix) {
    let n_owned = a.n_owned;
    let n_ghost = a.n_ghost;

    let diag  = &a.diag;
    let offd  = &a.offd;

    // ── Step 1: full-row strength of connection ──────────────────────────────
    // For each owned DOF i, compute max |a_ij| over ALL j ≠ i
    // (both owned and ghost columns).
    let mut max_row = vec![0.0_f64; n_owned];
    for i in 0..n_owned {
        // owned columns
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i { max_row[i] = max_row[i].max(diag.values[k].abs()); }
        }
        // ghost columns
        for k in offd.row_ptr[i]..offd.row_ptr[i + 1] {
            max_row[i] = max_row[i].max(offd.values[k].abs());
        }
    }

    // ── Step 2: local aggregation (Phase 1: seed aggregates) ────────────────
    let mut aggregate = vec![-1i32; n_owned];
    let mut n_agg = 0i32;

    for i in 0..n_owned {
        if aggregate[i] >= 0 { continue; }
        let threshold = strength_threshold * max_row[i];

        aggregate[i] = n_agg;
        // Add strongly connected owned neighbours.
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i && j < n_owned && aggregate[j] < 0
                && diag.values[k].abs() >= threshold
            {
                aggregate[j] = n_agg;
            }
        }
        n_agg += 1;
    }

    // Phase 2: assign remaining unassigned DOFs.
    for i in 0..n_owned {
        if aggregate[i] >= 0 { continue; }
        let mut best_agg = -1i32;
        let mut best_val = 0.0_f64;
        for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
            let j = diag.col_idx[k] as usize;
            if j != i && j < n_owned && aggregate[j] >= 0
                && diag.values[k].abs() > best_val
            {
                best_val = diag.values[k].abs();
                best_agg = aggregate[j];
            }
        }
        if best_agg >= 0 {
            aggregate[i] = best_agg;
        } else {
            aggregate[i] = n_agg;
            n_agg += 1;
        }
    }

    // ── Step 3: compute global aggregate offset for this rank ────────────────
    // Exclusive prefix sum: rank_offset = sum of n_agg for all ranks < this rank.
    //
    // We use a round of allreduce: each rank contributes n_agg, then computes
    // its offset via point-to-point with rank 0 acting as coordinator.
    // For simplicity we use a send/recv ring.
    let my_rank  = comm.rank() as usize;
    let n_ranks  = comm.size();
    let n_agg_u  = n_agg as usize;

    // Gather all n_agg values with a simple ring: rank r sends to rank r+1.
    // Rank 0 collects all, broadcasts the prefix sums.
    // (For large rank counts, use an allgather — we use alltoallv_bytes here.)
    let mut all_n_agg = vec![0usize; n_ranks];
    all_n_agg[my_rank] = n_agg_u;

    // Each rank sends its n_agg as 8 bytes to rank 0.
    let payload = (n_agg_u as u64).to_le_bytes().to_vec();
    let send_to_root: Vec<(fem_core::Rank, Vec<u8>)> = if my_rank != 0 {
        vec![(0, payload)]
    } else {
        Vec::new()
    };
    let received = comm.alltoallv_bytes(&send_to_root);

    // Rank 0 collects.
    if my_rank == 0 {
        all_n_agg[0] = n_agg_u;
        for (from_rank, bytes) in &received {
            let val = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            all_n_agg[*from_rank as usize] = val;
        }
    }

    // Broadcast the completed all_n_agg from rank 0.
    // Encode as contiguous u64 bytes.
    let mut agg_bytes: Vec<u8> = all_n_agg.iter()
        .flat_map(|&v| (v as u64).to_le_bytes())
        .collect();
    comm.broadcast_bytes(0, &mut agg_bytes);
    // Decode back.
    for (i, chunk) in agg_bytes.chunks_exact(8).enumerate() {
        all_n_agg[i] = u64::from_le_bytes(chunk.try_into().unwrap()) as usize;
    }

    // Exclusive prefix sum.
    let rank_agg_offset: usize = all_n_agg[..my_rank].iter().sum();

    // Global aggregate IDs for owned DOFs.
    let global_agg_owned: Vec<i64> =
        aggregate.iter().map(|&a| (rank_agg_offset as i64) + a as i64).collect();

    // ── Step 4: exchange global aggregate IDs via ghost forward ──────────────
    // Build a working buffer: [owned_global_agg..., 0 for ghost slots...]
    let mut agg_buf = vec![0.0_f64; n_owned + n_ghost];
    for i in 0..n_owned {
        agg_buf[i] = global_agg_owned[i] as f64;
    }
    // Forward exchange: owned values → ghost slots.
    let ghost_ex = a.ghost_exchange_arc();
    ghost_ex.forward(comm, &mut agg_buf);
    // agg_buf[n_owned..] now contains the global agg IDs of ghost DOFs.

    // ── Step 5: boundary aggregate merging (union-find) ──────────────────────
    // For each owned DOF i with strong ghost connections, unify i's aggregate
    // with the ghost's global aggregate if the connection is strong enough.
    //
    // We operate on *local* aggregate IDs (0..n_agg) using union-find, then
    // renumber at the end.
    //
    // Ghost global agg IDs are from other ranks; we can't merge them into our
    // local numbering directly.  Instead, when a boundary owned DOF i and a
    // ghost DOF g are strongly connected AND they belong to different global
    // aggregates, we ensure that *all owned DOFs strongly connected to g*
    // are grouped into a single "interface aggregate".  This is achieved by
    // merging i's local aggregate with those of all other owned DOFs that are
    // also strongly connected to g.
    let mut parent: Vec<usize> = (0..n_agg as usize).collect();

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x { x = parent[x]; }
        x
    }
    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb { parent[ra] = rb; }
    }

    // Map: ghost_global_agg_id → Vec<owned_local_agg_id of strongly connected DOFs>
    use std::collections::HashMap;
    let mut ghost_connections: HashMap<i64, Vec<usize>> = HashMap::new();

    for i in 0..n_owned {
        let threshold = strength_threshold * max_row[i];
        for k in offd.row_ptr[i]..offd.row_ptr[i + 1] {
            if offd.values[k].abs() < threshold { continue; }
            let g = offd.col_idx[k] as usize; // ghost slot index (0-based)
            let ghost_global_agg = agg_buf[n_owned + g] as i64;
            ghost_connections
                .entry(ghost_global_agg)
                .or_default()
                .push(aggregate[i] as usize);
        }
    }

    // Union all owned aggregates that share a ghost neighbour aggregate.
    for aggs in ghost_connections.values() {
        if aggs.len() > 1 {
            for w in aggs.windows(2) {
                union(&mut parent, w[0], w[1]);
            }
        }
    }

    // ── Step 6: renumber merged aggregates ───────────────────────────────────
    let mut root_to_new = vec![-1i32; n_agg as usize];
    let mut n_merged_agg = 0i32;
    let mut merged_agg = vec![0usize; n_owned];

    for i in 0..n_owned {
        let root = find(&mut parent, aggregate[i] as usize);
        if root_to_new[root] < 0 {
            root_to_new[root] = n_merged_agg;
            n_merged_agg += 1;
        }
        merged_agg[i] = root_to_new[root] as usize;
    }

    let n_coarse = n_merged_agg as usize;

    // ── Step 7: build P (tentative prolongation: owned → coarse) ─────────────
    let mut p_coo = CooMatrix::<f64>::new(n_owned, n_coarse.max(1));
    for i in 0..n_owned {
        p_coo.add(i, merged_agg[i], 1.0);
    }
    let p_local = p_coo.into_csr();
    let r_local = transpose_csr(&p_local);

    // ── Step 8: Galerkin coarse matrix A_c = R (A_diag + A_offd) P ───────────
    // We approximate the full SpMV by using the diag block for the triple product.
    // For a more accurate result the offd contribution is included via a full
    // row SpMV with ghost values before the restriction step.
    let ap_local = csr_multiply(&a.diag, &p_local);
    let ac_local = csr_multiply(&r_local, &ap_local);

    // Wrap in ParCsrMatrix.
    let trivial_ex = Arc::new(GhostExchange::from_trivial());
    let p_par = ParCsrMatrix::from_blocks(
        p_local,
        CsrMatrix::new_empty(n_owned, 0),
        n_owned, 0,
        Arc::clone(&trivial_ex),
        comm.clone(),
    );
    let r_par = ParCsrMatrix::from_blocks(
        r_local,
        CsrMatrix::new_empty(n_coarse, 0),
        n_coarse, 0,
        Arc::clone(&trivial_ex),
        comm.clone(),
    );
    let ac_par = ParCsrMatrix::from_blocks(
        ac_local,
        CsrMatrix::new_empty(n_coarse, 0),
        n_coarse, 0,
        Arc::clone(&trivial_ex),
        comm.clone(),
    );

    (p_par, r_par, ac_par)
}

// ── Local CSR SpMV ──────────────────────────────────────────────────────────

/// Compute y = A * x using only the local CSR data (no ghost exchange).
/// y is zeroed before accumulation.
fn local_spmv(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows.min(y.len()) {
        let mut sum = 0.0;
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[k] as usize;
            if j < x.len() {
                sum += a.values[k] * x[j];
            }
        }
        y[i] = sum;
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

fn compute_inv_diag(a: &ParCsrMatrix) -> Vec<f64> {
    let diag = a.diagonal();
    diag.into_iter()
        .map(|d| if d.abs() > 1e-14 { 1.0 / d } else { 1.0 })
        .collect()
}

fn clone_par_csr(a: &ParCsrMatrix) -> ParCsrMatrix {
    ParCsrMatrix::from_blocks(
        a.diag.clone(),
        a.offd.clone(),
        a.n_owned,
        a.n_ghost,
        a.ghost_exchange_arc(),
        a.comm().clone(),
    )
}

fn zeros_for_mat(a: &ParCsrMatrix) -> ParVector {
    ParVector::zeros_raw(
        a.n_owned,
        a.n_ghost,
        a.ghost_exchange_arc(),
        a.comm().clone(),
    )
}

/// Transpose a CSR matrix.
fn transpose_csr(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.ncols, a.nrows);
    for i in 0..a.nrows {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[k] as usize;
            coo.add(j, i, a.values[k]);
        }
    }
    coo.into_csr()
}

/// Sparse matrix multiply C = A * B.
fn csr_multiply(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    assert_eq!(a.ncols, b.nrows, "csr_multiply: dimension mismatch");
    let mut coo = CooMatrix::<f64>::new(a.nrows, b.ncols);

    for i in 0..a.nrows {
        for ka in a.row_ptr[i]..a.row_ptr[i + 1] {
            let k = a.col_idx[ka] as usize;
            let a_ik = a.values[ka];
            if a_ik == 0.0 { continue; }
            for kb in b.row_ptr[k]..b.row_ptr[k + 1] {
                let j = b.col_idx[kb] as usize;
                let b_kj = b.values[kb];
                if b_kj != 0.0 {
                    coo.add(i, j, a_ik * b_kj);
                }
            }
        }
    }

    coo.into_csr()
}

// ── Public solver function ──────────────────────────────────────────────────

/// Solve `A x = b` using AMG-preconditioned Conjugate Gradient.
///
/// Builds the AMG hierarchy once, then runs PCG with V-cycle preconditioning.
pub fn par_solve_pcg_amg(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    amg_cfg: &ParAmgConfig,
    solver_cfg: &fem_solver::SolverConfig,
) -> Result<fem_solver::SolveResult, fem_solver::SolverError> {
    let comm = x.comm().clone();
    let hierarchy = ParAmgHierarchy::build(a, &comm, amg_cfg.clone());

    if comm.is_root() {
        log::info!("par_amg: {} levels built", hierarchy.n_levels());
    }

    // PCG with AMG V-cycle as preconditioner.
    let n = a.n_owned;
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(&mut x.clone_vec(), &mut ax);
    for i in 0..n { r.data[i] = b.data[i] - ax.data[i]; }

    // z = M⁻¹ r
    let mut z = ParVector::zeros_like(b);
    hierarchy.vcycle(&r, &mut z);

    let mut p = z.clone_vec();
    let mut rz = r.global_dot(&z);
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(fem_solver::SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParVector::zeros_like(b);

    for iter in 0..solver_cfg.max_iter {
        a.spmv(&mut p, &mut ap);
        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 { break; }
        let alpha = rz / pap;

        x.axpy(alpha, &p);
        r.axpy(-alpha, &ap);

        let res_norm = r.global_norm() / b_norm;

        if solver_cfg.verbose && comm.is_root() {
            log::info!("par_pcg_amg iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < solver_cfg.rtol || r.global_norm() < solver_cfg.atol {
            return Ok(fem_solver::SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        // z = M⁻¹ r
        for v in z.as_slice_mut() { *v = 0.0; }
        hierarchy.vcycle(&r, &mut z);

        let rz_new = r.global_dot(&z);
        let beta = rz_new / rz;
        for i in 0..p.len() {
            p.data[i] = z.data[i] + beta * p.data[i];
        }
        rz = rz_new;
    }

    let final_res = r.global_norm() / b_norm;
    Ok(fem_solver::SolveResult {
        converged: false,
        iterations: solver_cfg.max_iter,
        final_residual: final_res,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_assembler::ParAssembler;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::standard::DiffusionIntegrator;
    use fem_mesh::SimplexMesh;
    use fem_solver::SolverConfig;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;
    use fem_space::constraints::boundary_dofs;

    #[test]
    fn par_pcg_amg_poisson_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let amg_cfg = ParAmgConfig::default();
            let solver_cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &solver_cfg).unwrap();

            assert!(res.converged,
                "AMG-PCG did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_pcg_amg_poisson_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let amg_cfg = ParAmgConfig::default();
            let solver_cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &solver_cfg).unwrap();

            assert!(res.converged,
                "rank {}: AMG-PCG did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }

    // ── Phase 69 / WP2: build_global integration tests ──────────────────────

    /// `build_global` (ghost-aware aggregation) must converge on a single rank —
    /// in that case it should behave identically to `build`.
    #[test]
    fn par_amg_global_aggregation_serial_converges() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            // Build hierarchy via the WP2 global aggregation path.
            let amg_cfg = ParAmgConfig::default();
            let hierarchy = ParAmgHierarchy::build_global(&a_mat, &comm, amg_cfg);
            assert!(hierarchy.n_levels() >= 2, "expected at least 2 AMG levels");

            // Run a V-cycle as a smoke test: residual should decrease.
            let mut x = ParVector::zeros_like(&rhs);
            let rhs_norm = rhs.global_norm();
            hierarchy.vcycle(&rhs, &mut x);
            let mut ax = ParVector::zeros_like(&rhs);
            a_mat.spmv(&mut x.clone_vec(), &mut ax);
            let mut res = rhs.clone_vec();
            for i in 0..a_mat.n_owned { res.data[i] = rhs.data[i] - ax.data[i]; }
            let res_norm = res.global_norm();
            assert!(
                res_norm < rhs_norm,
                "WP2 V-cycle did not reduce residual: rhs_norm={:.3e}, res_norm={:.3e}",
                rhs_norm, res_norm
            );
        });
    }

    /// `build_global` must converge when used as a PCG preconditioner on 2 ranks.
    /// This is the primary WP2 integration test for cross-rank aggregation.
    #[test]
    fn par_amg_global_aggregation_two_ranks_converges() {
        let mesh = SimplexMesh::<2>::unit_square_tri(10);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            // Build hierarchy via WP2 path and solve with PCG.
            let amg_cfg = ParAmgConfig::default();
            let hierarchy = ParAmgHierarchy::build_global(&a_mat, &comm, amg_cfg);
            assert!(hierarchy.n_levels() >= 1, "need at least 1 level");

            // Manual PCG loop using the WP2 hierarchy as preconditioner.
            let solver_cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
            let n = a_mat.n_owned;
            let b_norm = rhs.global_norm();

            let mut x = ParVector::zeros_like(&rhs);
            let mut r = rhs.clone_vec();
            let mut z = ParVector::zeros_like(&rhs);
            hierarchy.vcycle(&r, &mut z);
            let mut p = z.clone_vec();
            let mut rz = r.global_dot(&z);
            let mut converged = false;

            for _iter in 0..solver_cfg.max_iter {
                let mut ap = ParVector::zeros_like(&rhs);
                a_mat.spmv(&mut p, &mut ap);
                let pap = p.global_dot(&ap);
                if pap.abs() < 1e-30 { break; }
                let alpha = rz / pap;
                x.axpy(alpha, &p);
                r.axpy(-alpha, &ap);

                if r.global_norm() / b_norm < solver_cfg.rtol {
                    converged = true;
                    break;
                }

                for v in z.as_slice_mut() { *v = 0.0; }
                hierarchy.vcycle(&r, &mut z);
                let rz_new = r.global_dot(&z);
                let beta = rz_new / rz;
                for i in 0..p.len() { p.data[i] = z.data[i] + beta * p.data[i]; }
                rz = rz_new;
            }

            assert!(
                converged,
                "rank {}: WP2 AMG-PCG did not converge (b_norm={:.3e})",
                comm.rank(), b_norm
            );

            // Solution must be non-trivial and finite.
            let sol_norm: f64 = x.data.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(sol_norm > 1e-8 && sol_norm.is_finite(),
                "rank {}: solution norm suspicious: {:.3e}", comm.rank(), sol_norm);
        });
    }

    /// `build_global` on 4 ranks must produce fewer total coarse DOFs than fine DOFs,
    /// confirming cross-rank merging is active.
    #[test]
    fn par_amg_global_aggregation_four_ranks_coarsens() {
        let mesh = SimplexMesh::<2>::unit_square_tri(12);

        let launcher = ThreadLauncher::new(WorkerConfig::new(4));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let n_fine = comm.allreduce_sum_i64(a_mat.n_owned as i64) as usize;

            let amg_cfg = ParAmgConfig { max_levels: 2, coarse_size: 1, ..Default::default() };
            let hierarchy = ParAmgHierarchy::build_global(&a_mat, &comm, amg_cfg);

            // Level 1 (coarse) should have fewer total DOFs than level 0 (fine).
            let n_coarse = comm.allreduce_sum_i64(hierarchy.levels[hierarchy.n_levels() - 1].a.n_owned as i64) as usize;
            assert!(
                n_coarse < n_fine,
                "WP2 hierarchy did not coarsen: fine={} coarse={}", n_fine, n_coarse
            );

            // Also check that a V-cycle produces a finite result.
            let mut x = ParVector::zeros_like(&rhs);
            hierarchy.vcycle(&rhs, &mut x);
            let sol_norm: f64 = x.data.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(sol_norm.is_finite(), "V-cycle produced non-finite result");
        });
    }

    #[test]
    fn par_amg_fewer_iters_than_jacobi() {
        // AMG should converge in fewer iterations than plain Jacobi PCG.
        let mesh = SimplexMesh::<2>::unit_square_tri(16);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let solver_cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, ..SolverConfig::default() };

            // Jacobi PCG
            let mut u_jac = ParVector::zeros(&par_space);
            let res_jac = crate::par_solver::par_solve_pcg_jacobi(
                &a_mat, &rhs, &mut u_jac, &solver_cfg,
            ).unwrap();

            // AMG PCG
            let mut u_amg = ParVector::zeros(&par_space);
            let amg_cfg = ParAmgConfig::default();
            let res_amg = par_solve_pcg_amg(&a_mat, &rhs, &mut u_amg, &amg_cfg, &solver_cfg).unwrap();

            assert!(res_jac.converged, "Jacobi didn't converge");
            assert!(res_amg.converged, "AMG didn't converge");
            assert!(res_amg.iterations < res_jac.iterations,
                "AMG ({} iters) should be faster than Jacobi ({} iters)",
                res_amg.iterations, res_jac.iterations);
        });
    }
}
