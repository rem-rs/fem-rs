//! Parallel (distributed-memory) ultraweak DPG weak form.
//!
//! Port of MFEM's `ParDPGWeakForm` (`miniapps/dpg/util/pweakform.{hpp,cpp}`):
//! the serial [`DpgWeakForm`] is assembled on the rank-local sub-mesh and the
//! global system is formed as `Pᵀ A P`, where `P` is the block prolongation
//! from the rank-local (L-)vector layout of each trial block to the global
//! (true) DOF numbering.
//!
//! # Numbering
//!
//! * **Broken volume blocks** (`u`, `σ`): one DOF block per element, so a
//!   local DOF is owned by the rank owning its element and its global id is
//!   `global_elem * dofs_per_elem + k`.
//! * **Face-discontinuous trace blocks** (`σ̂`, `f̂`, …): a global face id
//!   comes from the sorted union of all ranks' face keys (the sorted tuple of
//!   *global* node ids); each face carries `dofs_per_face` consecutive DOFs.
//!   The face owner is the **lowest rank that holds the face**, which
//!   guarantees the owner can assemble the complete face row and knows the
//!   geometry of every one of its face DOFs.
//! * **Vertex-continuous H1-trace blocks** (`û`, `p̂`, 2-D only): the corner
//!   DOFs are the mesh vertex DOFs (global id = global node id, owner = node
//!   owner — the node-ghost layer guarantees that the owner holds every
//!   element incident to the node); the `p − 1` face-interior DOFs are
//!   numbered per face like a discontinuous block.
//!
//! # Why the identity node numbering
//!
//! The serial face tables of `SkeletonSpace` derive their *canonical* face
//! direction from the **local node ids**.  With the default compact per-rank
//! numbering the canonical direction of a shared face can differ between
//! ranks, which would flip the trace-DOF ordering and the element-side normal
//! sign and make the cross-rank ghost columns of `Pᵀ A P` inconsistent.
//! Parallel DPG therefore requires
//! [`partition_mesh_identity`](crate::par_partition::partition_mesh_identity)
//! (local node ids == global node ids), exactly like the RTk/NDk spaces.
//! Nodes present in the identity range but absent from the sub-mesh
//! ("phantom" nodes) have no support and are dropped from the numbering.
//!
//! # Static condensation
//!
//! `enable_static_condensation()` is forwarded to the serial weak form; the
//! parallel system is then the Schur complement over the exposed (trace)
//! blocks, with the same global trace numbering.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{DpgBilinear2, DpgLinear2, DpgTraceBilinear2};
use fem_assembly::dpg_weakform::DpgWeakForm;
use fem_core::Rank;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_csr::ParCsrMatrix;
use crate::par_dpg_numbering::{build_numbering, DpgBlockKind, SysBlock, INACTIVE};
use crate::par_vector::ParVector;
use crate::partition::MeshPartition;

/// A formed parallel DPG system.
pub struct ParDpgSystem {
    /// Parallel matrix `Pᵀ A P` over the global system DOFs.
    pub a: ParCsrMatrix,
    /// Parallel RHS `Pᵀ (b − A_e x_e)`.
    pub b: ParVector,
    /// Number of owned rows.
    pub n_owned: usize,
    /// Total number of local rows (owned + ghost).
    pub n_local: usize,
}

/// The parallel DPG weak form: a serial [`DpgWeakForm`] on the rank-local
/// sub-mesh plus the distributed true-DOF numbering.
pub struct ParDpgWeakForm<M: MeshTopology + Clone + 'static> {
    local: DpgWeakForm<M>,
    partition: MeshPartition,
    comm: Comm,
    rank: Rank,
    kinds: Vec<DpgBlockKind>,
    blocks: Vec<SysBlock>,
    /// `system-local dof -> absolute global dof` (`INACTIVE` if dropped).
    sys_global: Vec<u32>,
    /// `system-local dof -> compact id` (`INACTIVE` if dropped).
    perm: Vec<u32>,
    /// `compact id -> system-local dof`.
    inv_perm: Vec<u32>,
    /// `compact owned id -> global dof`.
    owned_global: Vec<u32>,
    /// `compact ghost id - n_owned -> global dof`.
    ghost_global: Vec<u32>,
    n_owned: usize,
    n_ghost: usize,
    /// Block offsets inside the owned compact segment (`len = nblocks + 1`).
    owned_block_offsets: Vec<usize>,
    ghost_exchange: Arc<GhostExchange>,
    n_global_dofs: usize,
    /// Global DOF count of **every** trial block (MFEM's `Σ GlobalTrueVSize`
    /// of the trial spaces, which is what the miniapps print even under static
    /// condensation).
    n_global_trial: Vec<usize>,
    condensed: bool,
    built: bool,
}

impl<M: MeshTopology + Clone + 'static> ParDpgWeakForm<M> {
    /// Create an empty parallel weak form on the rank-local `mesh`.
    ///
    /// `partition` must be the partition descriptor of `mesh` (for the
    /// `(local_mesh, partition)` pair obtained from
    /// [`ParallelMesh`](crate::par_mesh::ParallelMesh)) and must have been
    /// produced by
    /// [`partition_mesh_identity`](crate::par_partition::partition_mesh_identity).
    pub fn new(mesh: M, partition: MeshPartition, comm: Comm) -> Self {
        let rank = comm.rank();
        ParDpgWeakForm {
            local: DpgWeakForm::new(mesh),
            partition,
            comm,
            rank,
            kinds: Vec::new(),
            blocks: Vec::new(),
            sys_global: Vec::new(),
            perm: Vec::new(),
            inv_perm: Vec::new(),
            owned_global: Vec::new(),
            ghost_global: Vec::new(),
            n_owned: 0,
            n_ghost: 0,
            owned_block_offsets: vec![0],
            ghost_exchange: Arc::new(GhostExchange::from_trivial()),
            n_global_dofs: 0,
            n_global_trial: Vec::new(),
            condensed: false,
            built: false,
        }
    }

    // ── builder API (forwards to the serial weak form) ───────────────────────

    /// Volume quadrature order — see [`DpgWeakForm::set_quad_order`].
    pub fn set_quad_order(&mut self, order: u8) {
        self.local.set_quad_order(order);
    }

    /// Face quadrature order — see [`DpgWeakForm::set_face_quad_order`].
    pub fn set_face_quad_order(&mut self, order: u8) {
        self.local.set_face_quad_order(order);
    }

    /// Broken scalar L2 trial block (`u`).
    pub fn add_trial_scalar_space(&mut self, order: u8) -> usize {
        self.kinds.push(DpgBlockKind::Volume { vdim: 1 });
        self.local.add_trial_scalar_space(order)
    }

    /// Broken vector L2 trial block (`σ`).
    pub fn add_trial_vector_space(&mut self, order: u8, vdim: usize) -> usize {
        self.kinds.push(DpgBlockKind::Volume { vdim });
        self.local.add_trial_vector_space(order, vdim)
    }

    /// Face-discontinuous trace trial block (`σ̂`, `f̂`).
    pub fn add_trial_trace_space(&mut self, order: u8) -> usize {
        self.kinds.push(DpgBlockKind::FaceDiscontinuous);
        self.local.add_trial_trace_space(order)
    }

    /// Vertex-continuous H1-trace trial block (`û`, `p̂`) — 2-D.
    pub fn add_trial_trace_space_h1(&mut self, order: u8) -> usize {
        self.kinds.push(DpgBlockKind::FaceContinuous);
        self.local.add_trial_trace_space_h1(order)
    }

    /// 3-D ND trace trial block — the parallel numbering is not implemented.
    pub fn add_trial_trace_space_nd(&mut self, order: u8) -> usize {
        self.kinds.push(DpgBlockKind::FaceNd);
        self.local.add_trial_trace_space_nd(order)
    }

    /// Broken test block.
    pub fn add_test_space(&mut self, kind: VolKind, order: u8) -> usize {
        self.local.add_test_space(kind, order)
    }

    /// Trial integrator — see [`DpgWeakForm::add_trial_integrator`].
    pub fn add_trial_integrator(
        &mut self,
        integ: Box<dyn DpgBilinear2>,
        trial_block: usize,
        test_block: usize,
    ) {
        self.local.add_trial_integrator(integ, trial_block, test_block);
    }

    /// Test (Riesz) integrator — see [`DpgWeakForm::add_test_integrator`].
    pub fn add_test_integrator(
        &mut self,
        integ: Box<dyn DpgBilinear2>,
        row_block: usize,
        col_block: usize,
    ) {
        self.local.add_test_integrator(integ, row_block, col_block);
    }

    /// Trace-face trial integrator — see [`DpgWeakForm::add_trace_integrator`].
    pub fn add_trace_integrator(
        &mut self,
        integ: Box<dyn DpgTraceBilinear2>,
        trial_block: usize,
        test_block: usize,
    ) {
        self.local.add_trace_integrator(integ, trial_block, test_block);
    }

    /// Domain linear-form integrator.
    pub fn add_domain_lf_integrator(&mut self, integ: Box<dyn DpgLinear2>, test_block: usize) {
        self.local.add_domain_lf_integrator(integ, test_block);
    }

    /// `EnableStaticCondensation()`.
    pub fn enable_static_condensation(&mut self) {
        self.local.enable_static_condensation();
        self.condensed = true;
    }

    /// `StoreMatrices(true)` — required for [`Self::compute_residual`].
    pub fn store_matrices(&mut self, store: bool) {
        self.local.store_matrices(store);
    }

    // ── accessors ───────────────────────────────────────────────────────────

    /// The rank-local serial weak form (skeleton, block sizes, geometry).
    pub fn local(&self) -> &DpgWeakForm<M> {
        &self.local
    }

    /// Kind of trial block `b`.
    pub fn block_kind(&self, b: usize) -> DpgBlockKind {
        self.kinds[b]
    }

    /// Number of blocks in the *system* (all trial blocks, or only the
    /// exposed trace blocks under static condensation).
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Absolute global DOF of a **system**-index DOF, or `None` if the index
    /// carries no numbered DOF (static-condensation private DOFs, phantom
    /// nodes, non-traced ND trace DOFs).  Diagnostic helper.
    pub fn global_of_sys_local(&self, sys_local: usize) -> Option<u32> {
        match self.sys_global.get(sys_local) {
            Some(&g) if g != INACTIVE => Some(g),
            _ => None,
        }
    }

    /// Total number of global DOFs of the system.
    pub fn n_global_dofs(&self) -> usize {
        self.n_global_dofs
    }

    /// Global DOF count summed over **all** trial blocks — MFEM's
    /// `u_fes->GlobalTrueVSize() + σ_fes->… + û_fes->… + σ̂_fes->…`, i.e. the
    /// `Dofs` column of the miniapps.  Differs from
    /// [`Self::n_global_dofs`] only under static condensation (where the
    /// system itself is the trace-only Schur complement).
    pub fn n_global_trial_dofs(&self) -> usize {
        self.n_global_trial.iter().sum()
    }

    /// Number of owned rows on this rank.
    pub fn n_owned_dofs(&self) -> usize {
        self.n_owned
    }

    /// Number of ghost rows on this rank.
    pub fn n_ghost_dofs(&self) -> usize {
        self.n_ghost
    }

    /// Block offsets inside the owned compact segment (`len = nblocks + 1`).
    pub fn owned_block_offsets(&self) -> &[usize] {
        &self.owned_block_offsets
    }

    /// Ghost exchange handle covering the whole compact DOF vector.
    pub fn ghost_exchange_arc(&self) -> Arc<GhostExchange> {
        self.ghost_exchange.clone()
    }

    /// The communicator this weak form lives on.
    pub fn comm(&self) -> &Comm {
        &self.comm
    }

    /// The mesh partition descriptor of the rank-local sub-mesh.
    pub fn partition_ref(&self) -> &MeshPartition {
        &self.partition
    }

    // ── assemble / numbering ────────────────────────────────────────────────

    /// Assemble the rank-local matrix and build the distributed numbering.
    pub fn assemble(&mut self) {
        self.local.assemble();
        self.build_numbering();
    }

    fn build_numbering(&mut self) {
        self.rank = self.comm.rank();
        let n = build_numbering(&self.local, &self.kinds, &self.partition, &self.comm, self.condensed);
        self.blocks = n.blocks;
        self.sys_global = n.sys_global;
        self.perm = n.perm;
        self.inv_perm = n.inv_perm;
        self.owned_global = n.owned_global;
        self.ghost_global = n.ghost_global;
        self.n_owned = n.n_owned;
        self.n_ghost = n.n_ghost;
        self.owned_block_offsets = n.owned_block_offsets;
        self.ghost_exchange = Arc::new(n.ghost_exchange);
        self.n_global_dofs = n.n_global_dofs;
        self.n_global_trial = n.n_global_trial;
        self.built = true;
    }

    // ── linear system ───────────────────────────────────────────────────────

    /// `FormLinearSystem(ess_tdof_list, x, A, X, B)`.
    ///
    /// * `ess_global` — absolute global DOF ids of the essential (Dirichlet)
    ///   DOFs.
    /// * `x_local` — rank-local vector holding the prescribed values at the
    ///   local copies of those DOFs (length = serial local size).
    ///
    /// The elimination follows MFEM's `ParDPGWeakForm`: the essential columns
    /// of the rank-local matrix are eliminated *before* `Pᵀ A P`, and a unit
    /// diagonal is placed on the essential rows with the prescribed value on
    /// the RHS.
    pub fn form_linear_system(
        &mut self,
        ess_global: &[u32],
        x_local: &[f64],
    ) -> (ParDpgSystem, Vec<f64>, Vec<f64>) {
        assert!(self.built, "assemble() must run before form_linear_system()");
        let ess_set: BTreeSet<u32> = ess_global.iter().copied().collect();
        // `DpgWeakForm::form_linear_system` takes the essential DOFs and the
        // prescribed values in the **full** serial trial numbering, while the
        // returned system is indexed by the system index space.
        let mut ess_local: Vec<usize> = Vec::new();
        for blk in &self.blocks {
            for d in 0..blk.size {
                if blk.gof[d] == INACTIVE {
                    continue;
                }
                let g = blk.global_base as u32 + blk.gof[d];
                if ess_set.contains(&g) {
                    ess_local.push(blk.full_base + d);
                }
            }
        }
        ess_local.sort_unstable();
        let mut x_full = vec![0.0_f64; self.local.size()];
        let n = x_local.len().min(x_full.len());
        x_full[..n].copy_from_slice(&x_local[..n]);

        let (system, xs, b) = self.local.form_linear_system(&ess_local, &x_full, false);
        let mat = system.matrix();
        assert_eq!(
            mat.nrows,
            self.sys_global.len(),
            "parallel DPG system size mismatch"
        );

        let n_compact = self.n_owned + self.n_ghost;
        // Square `n_compact × n_compact` local layout with the ghost rows
        // left zero (only owned rows are filled below); `from_local_matrix`
        // discards the ghost rows and splits the `[owned | ghost]` columns
        // into diag/offd.  (Since the D167 fix it also accepts the
        // rectangular `n_owned × n_compact` form — the ghost-column count
        // derives from `local.ncols` — the square form is kept because it is
        // what the owned-row fill below produces naturally.)
        let mut coo = CooMatrix::<f64>::new(n_compact, n_compact);
        for r in 0..self.n_owned {
            let l = self.inv_perm[r] as usize;
            for k in mat.row_ptr[l]..mat.row_ptr[l + 1] {
                let pc = self.perm[mat.col_idx[k] as usize];
                if pc == INACTIVE {
                    continue;
                }
                let v = mat.values[k];
                if v != 0.0 {
                    coo.add(r, pc as usize, v);
                }
            }
        }
        let a_local = coo.into_csr();
        let a = ParCsrMatrix::from_local_matrix(
            &a_local,
            self.n_owned,
            self.ghost_exchange.clone(),
            self.comm.clone(),
        );

        // RHS: Pᵀ b (owned rows), ghost segment zeroed.
        let mut b_loc = vec![0.0_f64; n_compact];
        for r in 0..self.n_owned {
            b_loc[r] = b[self.inv_perm[r] as usize];
        }
        let b_par = ParVector::from_local_raw(
            b_loc,
            self.n_owned,
            self.ghost_exchange.clone(),
            self.comm.clone(),
        );

        // Initial guess X = Pᵀ xs (owned rows only).
        let mut x0 = vec![0.0_f64; n_compact];
        for r in 0..self.n_owned {
            x0[r] = xs[self.inv_perm[r] as usize];
        }
        let sys = ParDpgSystem {
            a,
            b: b_par,
            n_owned: self.n_owned,
            n_local: n_compact,
        };
        (sys, x0, x_full)
    }

    /// `RecoverFEMSolution(X, x)`: lift the global owned solution back to the
    /// rank-local (ghost-filled) trial vector in the layout the serial
    /// [`DpgWeakForm`] uses for post-processing.
    pub fn recover_fem_solution(&self, x_owned: &[f64]) -> Vec<f64> {
        assert!(self.built, "assemble() must run before recover_fem_solution()");
        let n_compact = self.n_owned + self.n_ghost;
        let mut data = vec![0.0_f64; n_compact];
        data[..self.n_owned].copy_from_slice(&x_owned[..self.n_owned.min(x_owned.len())]);
        if self.n_ghost > 0 {
            self.ghost_exchange.forward(&self.comm, &mut data);
        }
        let n_target = self.sys_global.len();
        let mut u = vec![0.0_f64; n_target];
        for s in 0..n_target {
            let p = self.perm[s];
            if p != INACTIVE {
                u[s] = data[p as usize];
            }
        }
        self.local.recover_fem_solution(&u)
    }

    /// Element-wise DPG residual `‖L⁻¹(B_e u_e − f_e)‖₂` on all local elements
    /// (owned + ghost), exactly like the serial method.  Requires
    /// `store_matrices(true)` and the **full** local trial vector produced by
    /// [`Self::recover_fem_solution`].
    pub fn compute_residual(&self, x_full_local: &[f64]) -> Vec<f64> {
        self.local.compute_residual(x_full_local)
    }

    /// Global `‖residual‖₂` over the **owned** elements (MFEM recomputes
    /// `sqrt(Σ_local res²)` across ranks).
    pub fn global_residual_norm(&self, x_full_local: &[f64]) -> f64 {
        let res = self.compute_residual(x_full_local);
        let rank = self.comm.rank();
        let mut acc = 0.0_f64;
        for (e, r) in res.iter().enumerate() {
            if self.partition.elem_owner[e] == rank {
                acc += r * r;
            }
        }
        self.comm.allreduce_sum_f64(acc).max(0.0).sqrt()
    }

    /// Local boundary-face DOFs of trial block `b` with their physical
    /// points: `(absolute global dof, point)`.
    ///
    /// A face is a *global* boundary face iff it has exactly one adjacent
    /// element, so every rank holding it sees it as a boundary face — the
    /// sets computed here are consistent across ranks without any exchange.
    /// The local DOF ids cover both owned and ghost copies, which is what the
    /// column elimination needs.
    pub fn trace_boundary_dofs(&self, b: usize) -> Vec<(u32, Vec<f64>)> {
        let bi = self
            .blocks
            .iter()
            .position(|blk| blk.trial == b)
            .expect("trace_boundary_dofs: block not in the system");
        let blk = &self.blocks[bi];
        let sk = self.local.skeleton(b);
        let mut out = Vec::new();
        for f in 0..sk.n_faces() {
            if !sk.is_boundary_face(f) {
                continue;
            }
            for (k, &d) in sk.face_dof_list(f).iter().enumerate() {
                if d >= blk.size || blk.gof[d] == INACTIVE {
                    continue;
                }
                out.push((
                    blk.global_base as u32 + blk.gof[d],
                    self.local.face_dof_point(sk, f, k),
                ));
            }
        }
        out
    }

    /// Write prescribed values into the rank-local vector for every essential
    /// DOF present locally, using `value(point)`.
    ///
    /// `pairs` must be the globally merged `(global dof, point)` list of
    /// [`Self::trace_boundary_dofs`].  `x_local` is in the **full** serial
    /// trial layout (the layout `DpgWeakForm::form_linear_system` expects for
    /// its `x` argument), so a statically condensed system must still be
    /// handed the full-length vector (`blk.full_base`, not the system index).
    pub fn fill_essential_values(
        &self,
        x_local: &mut [f64],
        pairs: &[(u32, Vec<f64>)],
        value: &dyn Fn(&[f64]) -> f64,
    ) {
        let mut by_id: HashMap<u32, f64> = HashMap::new();
        for (g, p) in pairs {
            by_id.entry(*g).or_insert_with(|| value(p));
        }
        for blk in &self.blocks {
            for d in 0..blk.size {
                if blk.gof[d] == INACTIVE {
                    continue;
                }
                let g = blk.global_base as u32 + blk.gof[d];
                if let Some(&v) = by_id.get(&g) {
                    let t = blk.full_base + d;
                    if t < x_local.len() {
                        x_local[t] = v;
                    }
                }
            }
        }
    }

    /// Merge `(global dof, point)` pairs across all ranks (lowest rank wins).
    pub fn merge_dof_points(&self, pairs: &[(u32, Vec<f64>)]) -> Vec<(u32, Vec<f64>)> {
        let mut map: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
        for (g, p) in pairs {
            map.entry(*g).or_insert_with(|| p.clone());
        }
        if self.comm.size() > 1 {
            let mut payload = Vec::with_capacity(pairs.len() * 12);
            for (g, p) in pairs {
                payload.extend_from_slice(&g.to_le_bytes());
                payload.extend_from_slice(&(p.len() as u32).to_le_bytes());
                for &v in p {
                    payload.extend_from_slice(&v.to_le_bytes());
                }
            }
            let sends: Vec<(Rank, Vec<u8>)> = (0..self.comm.size() as i32)
                .map(|r| (r, payload.clone()))
                .collect();
            let mut incoming = self.comm.alltoallv_bytes(&sends);
            incoming.sort_by_key(|(src, _)| *src);
            for (_src, bytes) in &incoming {
                let mut pos = 0usize;
                while pos + 8 <= bytes.len() {
                    let g = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                    let n = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
                    pos += 8;
                    if pos + 8 * n > bytes.len() {
                        break;
                    }
                    let mut p = Vec::with_capacity(n);
                    for i in 0..n {
                        p.push(f64::from_le_bytes(
                            bytes[pos + 8 * i..pos + 8 * i + 8].try_into().unwrap(),
                        ));
                    }
                    pos += 8 * n;
                    map.entry(g).or_insert(p);
                }
            }
        }
        map.into_iter().collect()
    }

    /// Absolute global ids of the owned compact rows.
    pub fn owned_global_ids(&self) -> &[u32] {
        &self.owned_global
    }

    /// Absolute global ids of the ghost compact rows (following the owned
    /// rows in the compact layout).
    pub fn ghost_global_ids(&self) -> &[u32] {
        &self.ghost_global
    }

    /// Owned×owned local sub-matrix of the whole parallel system, in the
    /// owned compact ordering (blocks in order, matching
    /// [`Self::owned_block_offsets`]) — the MFEM block-diagonal
    /// (`GSSmoother`-of-`BlockDiagonalPreconditioner`) preconditioner is built
    /// from this plus `owned_block_offsets`.
    pub fn owned_matrix(&self, a: &ParCsrMatrix) -> CsrMatrix<f64> {
        let n = self.n_owned;
        let diag = a.diag_block();
        let mut coo = CooMatrix::<f64>::new(n, n);
        for r in 0..n {
            for k in diag.row_ptr[r]..diag.row_ptr[r + 1] {
                let c = diag.col_idx[k] as usize;
                if c < n {
                    coo.add(r, c, diag.values[k]);
                }
            }
        }
        coo.into_csr()
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::Launcher;
    use crate::WorkerConfig;
    use fem_assembly::dpg::dpg_basis::{VolKind, VolVals};
    use fem_assembly::dpg::dpg_integrators::{
        DpgBilinear2, DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgMassIntegrator,
        DpgMixedScalarWeakGradientIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
        DpgTraceIntegrator, DpgVectorFEMassIntegrator, VolCtx,
    };
    use fem_mesh::Mesh;

    /// `-(σ, τ)` — MFEM `TransposeIntegrator(VectorFEMassIntegrator(-1))`.
    struct NegVectorMass;
    impl DpgBilinear2 for NegVectorMass {
        fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
            let d = ctx.dim;
            let (nt, nsc, nc) = (test.n_scalar, trial.n_scalar, trial.n_expanded);
            for k in 0..nt {
                for c in 0..d {
                    for j in 0..nsc {
                        m[k * nc + c * nsc + j] -= ctx.w * test.phi[k * d + c] * trial.phi[j];
                    }
                }
            }
        }
    }

    /// Builder surface shared by the serial and the parallel weak form, so a
    /// single function can wire up both (the Poisson ultraweak DPG block table
    /// of `miniapps/dpg/pdiffusion.rs`).
    trait DpgSpaceAdder {
        fn set_quad_order(&mut self, o: u8);
        fn set_face_quad_order(&mut self, o: u8);
        fn add_trial_scalar_space(&mut self, o: u8) -> usize;
        fn add_trial_vector_space(&mut self, o: u8, vdim: usize) -> usize;
        fn add_trial_trace_space_h1(&mut self, o: u8) -> usize;
        fn add_trial_trace_space(&mut self, o: u8) -> usize;
        fn add_test_space(&mut self, k: VolKind, o: u8) -> usize;
        fn add_trial_integrator(&mut self, i: Box<dyn DpgBilinear2>, t: usize, s: usize);
        fn add_test_integrator(&mut self, i: Box<dyn DpgBilinear2>, r: usize, c: usize);
        fn add_trace_integrator(&mut self, i: Box<dyn DpgTraceBilinear2>, t: usize, s: usize);
    }

    macro_rules! impl_adder {
        ($t:ty, $g:ident) => {
            impl<$g: MeshTopology + Clone + 'static> DpgSpaceAdder for $t {
                fn set_quad_order(&mut self, o: u8) {
                    self.set_quad_order(o)
                }
                fn set_face_quad_order(&mut self, o: u8) {
                    self.set_face_quad_order(o)
                }
                fn add_trial_scalar_space(&mut self, o: u8) -> usize {
                    self.add_trial_scalar_space(o)
                }
                fn add_trial_vector_space(&mut self, o: u8, v: usize) -> usize {
                    self.add_trial_vector_space(o, v)
                }
                fn add_trial_trace_space_h1(&mut self, o: u8) -> usize {
                    self.add_trial_trace_space_h1(o)
                }
                fn add_trial_trace_space(&mut self, o: u8) -> usize {
                    self.add_trial_trace_space(o)
                }
                fn add_test_space(&mut self, k: VolKind, o: u8) -> usize {
                    self.add_test_space(k, o)
                }
                fn add_trial_integrator(&mut self, i: Box<dyn DpgBilinear2>, t: usize, s: usize) {
                    self.add_trial_integrator(i, t, s)
                }
                fn add_test_integrator(&mut self, i: Box<dyn DpgBilinear2>, r: usize, c: usize) {
                    self.add_test_integrator(i, r, c)
                }
                fn add_trace_integrator(
                    &mut self,
                    i: Box<dyn DpgTraceBilinear2>,
                    t: usize,
                    s: usize,
                ) {
                    self.add_trace_integrator(i, t, s)
                }
            }
        };
    }
    impl_adder!(DpgWeakForm<M>, M);
    impl_adder!(ParDpgWeakForm<M>, M);

    fn add_diffusion_blocks<A: DpgSpaceAdder>(a: &mut A) -> (usize, usize, usize, usize) {
        let (p, to) = (1u8, 2u8);
        a.set_quad_order(2 * to);
        a.set_face_quad_order(to + p - 1);
        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space_h1(p);
        let hatsig = a.add_trial_trace_space(p - 1);
        let tau = a.add_test_space(VolKind::HDiv, to - 1);
        let v = a.add_test_space(VolKind::Scalar, to);
        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
            u,
            tau,
        );
        a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);
        a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
        a.add_test_integrator(Box::new(DpgVectorFEMassIntegrator { q: 1.0 }), tau, tau);
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
        a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
        (u, sig, hatu, hatsig)
    }

    /// Serial-full-mesh dof → parallel global dof.  The volume blocks and the
    /// `H1`-trace vertex dofs share the block-wise shift; the
    /// face-discontinuous trace block needs the sorted-global-face-key order.
    fn serial_to_global_map(ser: &DpgWeakForm<Mesh<2>>, hatsig: usize) -> Vec<u32> {
        let offs = ser.trial_offsets();
        let sk = ser.skeleton(hatsig);
        let nface = sk.n_faces();
        let keys: Vec<Vec<u32>> = (0..nface)
            .map(|f| {
                let mut k: Vec<u32> = sk.face_nodes(f).iter().copied().collect();
                k.sort_unstable();
                k
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        let fpar: Vec<u32> = keys
            .iter()
            .map(|k| sorted.binary_search(k).unwrap() as u32)
            .collect();
        let mut m = vec![u32::MAX; ser.size()];
        for b in 0..3 {
            for i in 0..ser.trial_block_sizes()[b] {
                m[offs[b] + i] = (offs[b] + i) as u32;
            }
        }
        for i in 0..nface {
            m[offs[hatsig] + i] = offs[hatsig] as u32 + fpar[i];
        }
        m
    }

    /// Regression test for the **cross-rank coupling** of `Pᵀ A P`: the
    /// 2-rank global matrix must equal the serial matrix assembled on the full
    /// mesh.
    ///
    /// This pins the (now-fixed) bug where
    /// `ParCsrMatrix::from_local_matrix` derived the ghost-column count from
    /// `local.nrows - n_owned`, so a non-square (`n_owned × n_compact`) input
    /// silently dropped the whole off-diagonal block and with it every
    /// cross-rank coupling (the count derives from `local.ncols` since the
    /// fix).  Symmetric PCG still
    /// "converged", to a wrong solution: `pdiffusion -sref 0` gave
    /// `L2 = 1.779e+00` instead of `1.021e+00`.
    #[test]
    fn two_rank_system_matches_serial_full_mesh() {
        let full = Arc::new(Mesh::<2>::unit_square_quad(2));
        let out = Arc::new(std::sync::Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let rank = comm.rank();
            let par_mesh = crate::par_partition::partition_mesh_identity(&ma, &comm);
            let local_mesh = par_mesh.local_mesh().clone();
            let part = par_mesh.partition().clone();
            let mut ap = ParDpgWeakForm::new(local_mesh, part, comm.clone());
            let (_u, _s, hatu, _hs) = add_diffusion_blocks(&mut ap);
            ap.assemble();
            let pairs = ap.trace_boundary_dofs(hatu);
            let merged = ap.merge_dof_points(&pairs);
            let ess: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();
            let mut xl = vec![0.0_f64; ap.local().size()];
            ap.fill_essential_values(&mut xl, &merged, &|_p: &[f64]| 0.0);
            let (sys, _x0, _b) = ap.form_linear_system(&ess, &xl);

            let mut ser = DpgWeakForm::new((*ma).clone());
            let (_u, _s, _hu, hss) = add_diffusion_blocks(&mut ser);
            ser.assemble();
            let s2g = serial_to_global_map(&ser, hss);
            let ess_set: BTreeSet<u32> = ess.iter().copied().collect();
            let mut ser_ess: Vec<usize> = (0..ser.size())
                .filter(|&l| s2g[l] != u32::MAX && ess_set.contains(&s2g[l]))
                .collect();
            ser_ess.sort_unstable();
            let ser_mat = match &ser
                .form_linear_system(&ser_ess, &vec![0.0; ser.size()], false)
                .0
            {
                fem_assembly::dpg_weakform::DpgSystem::Full { mat, .. } => mat.clone(),
                _ => unreachable!("uncondensed system expected"),
            };
            let mut glob2ser: HashMap<u32, usize> = HashMap::new();
            for (l, &g) in s2g.iter().enumerate() {
                if g != u32::MAX {
                    glob2ser.insert(g, l);
                }
            }
            let diag = sys.a.diag_block();
            let offd = sys.a.offd_block();
            assert!(
                offd.ncols > 0,
                "rank {rank}: 2-rank DPG matrix has an empty off-diagonal block"
            );
            let og = ap.owned_global_ids();
            let gg = ap.ghost_global_ids();
            let (mut maxabs, mut ncmp) = (0.0_f64, 0usize);
            for r in 0..sys.n_owned {
                let lr = glob2ser[&og[r]];
                let mut mine: HashMap<usize, f64> = HashMap::new();
                for k in diag.row_ptr[r]..diag.row_ptr[r + 1] {
                    *mine
                        .entry(glob2ser[&og[diag.col_idx[k] as usize]])
                        .or_insert(0.0) += diag.values[k];
                }
                for k in offd.row_ptr[r]..offd.row_ptr[r + 1] {
                    *mine
                        .entry(glob2ser[&gg[offd.col_idx[k] as usize]])
                        .or_insert(0.0) += offd.values[k];
                }
                let mut theirs: HashMap<usize, f64> = HashMap::new();
                for k in ser_mat.row_ptr[lr]..ser_mat.row_ptr[lr + 1] {
                    let g = s2g[ser_mat.col_idx[k] as usize];
                    if let Some(&lc) = glob2ser.get(&g) {
                        *theirs.entry(lc).or_insert(0.0) += ser_mat.values[k];
                    }
                }
                let mut cols: Vec<usize> = mine.keys().chain(theirs.keys()).cloned().collect();
                cols.sort_unstable();
                cols.dedup();
                for c in cols {
                    let d = (mine.get(&c).copied().unwrap_or(0.0)
                        - theirs.get(&c).copied().unwrap_or(0.0))
                    .abs();
                    ncmp += 1;
                    maxabs = maxabs.max(d);
                }
            }
            if rank == 0 {
                *out2.lock().unwrap() = Some(format!("entries={ncmp} maxabs={maxabs:.3e}"));
                assert!(ncmp > 0, "no matrix entries compared");
                assert!(
                    maxabs < 1e-11,
                    "2-rank P^T A P differs from the serial full-mesh matrix (maxabs={maxabs:.3e})"
                );
            }
        });
        let msg = out.lock().unwrap().clone();
        let msg = msg.expect("rank 0 did not publish the comparison");
        assert!(msg.contains("maxabs="), "comparison did not run: {msg}");
    }

    /// The owned DOF sets of the ranks must form a partition of
    /// `0..n_global_dofs`, and the block offsets must match the block count.
    #[test]
    fn numbering_is_a_global_partition() {        let full = Arc::new(Mesh::<2>::unit_square_quad(2));
        let out = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let par_mesh = crate::par_partition::partition_mesh_identity(&ma, &comm);
            let local_mesh = par_mesh.local_mesh().clone();
            let part = par_mesh.partition().clone();
            let mut ap = ParDpgWeakForm::new(local_mesh, part, comm.clone());
            let (_u, _s, _hatu, _hs) = add_diffusion_blocks(&mut ap);
            ap.assemble();
            let mut s = String::new();
            for g in ap.owned_global_ids() {
                s.push_str(&format!("o {g}\n"));
            }
            for g in ap.ghost_global_ids() {
                s.push_str(&format!("g {g}\n"));
            }
            s.push_str(&format!("n {}\n", ap.n_global_dofs()));
            s.push_str(&format!("b {}\n", ap.owned_block_offsets().len()));
            out2.lock().unwrap().push(s);
        });
        let mut owned: BTreeSet<u32> = BTreeSet::new();
        let mut n_global = 0usize;
        let mut n_ranks = 0usize;
        for s in out.lock().unwrap().iter() {
            let mut co: BTreeSet<u32> = BTreeSet::new();
            for l in s.lines() {
                let mut it = l.split_whitespace();
                match it.next().unwrap() {
                    "o" => {
                        let g: u32 = it.next().unwrap().parse().unwrap();
                        assert!(co.insert(g), "duplicate owned DOF on one rank");
                        assert!(owned.insert(g), "DOF owned by two ranks");
                    }
                    "g" => {
                        let g: u32 = it.next().unwrap().parse().unwrap();
                        assert!(!co.contains(&g), "DOF both owned and ghosted locally");
                    }
                    "n" => n_global = it.next().unwrap().parse().unwrap(),
                    "b" => assert_eq!(it.next().unwrap().parse::<usize>().unwrap(), 5),
                    _ => {}
                }
            }
            assert!(!co.is_empty(), "rank owns no DOFs");
            n_ranks += 1;
        }
        assert_eq!(n_ranks, 2);
        assert_eq!(owned.len(), n_global, "owned DOFs do not cover 0..n_global");
        assert_eq!(*owned.iter().next_back().unwrap(), (n_global - 1) as u32);
    }

    /// Static condensation: the essential rows of the reduced system must
    /// carry a unit diagonal and the prescribed value on the RHS.
    ///
    /// Pins `ParDpgWeakForm::fill_essential_values`, which must address the
    /// **full** serial trial layout (`SysBlock::full_base`) even though the
    /// system index space of a condensed system is the exposed (trace) compact
    /// range — writing to the system index instead corrupted the volume block's
    /// prescribed data and left the real essential slots at zero
    /// (`pdiffusion -sc` reported `L2 = 2.739e+00` instead of `1.021e+00`).
    #[test]
    fn condensed_essential_rows_are_eliminated() {
        let full = Arc::new(Mesh::<2>::unit_square_quad(2));
        let out = Arc::new(std::sync::Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let par_mesh = crate::par_partition::partition_mesh_identity(&ma, &comm);
            let local_mesh = par_mesh.local_mesh().clone();
            let part = par_mesh.partition().clone();
            let mut ap = ParDpgWeakForm::new(local_mesh, part, comm.clone());
            let (_u, _s, hatu, _hs) = add_diffusion_blocks(&mut ap);
            ap.enable_static_condensation();
            ap.assemble();
            let pairs = ap.trace_boundary_dofs(hatu);
            let merged = ap.merge_dof_points(&pairs);
            let ess: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();
            assert!(!ess.is_empty(), "no essential DOFs");
            // The full-length vector is still what `form_linear_system` wants.
            let mut xl = vec![0.0_f64; ap.local().size()];
            ap.fill_essential_values(&mut xl, &merged, &|_p: &[f64]| 1.0);
            let (sys, _x0, _xf) = ap.form_linear_system(&ess, &xl);
            let ess_set: BTreeSet<u32> = ess.iter().copied().collect();
            let mut checked = 0usize;
            for r in 0..sys.n_owned {
                let g = ap.owned_global_ids()[r];
                if !ess_set.contains(&g) {
                    continue;
                }
                checked += 1;
                let d = sys.a.diag_block();
                let row = d.row_ptr[r]..d.row_ptr[r + 1];
                assert_eq!(row.len(), 1, "essential row {r} is not a single diagonal entry");
                assert_eq!(d.col_idx[row.start] as usize, r, "essential row {r} not diagonal");
                assert!(
                    (d.values[row.start] - 1.0).abs() < 1e-14,
                    "essential row {r} diagonal = {}",
                    d.values[row.start]
                );
                assert!(
                    (sys.b.as_slice()[r] - 1.0).abs() < 1e-14,
                    "essential row {r} rhs = {}",
                    sys.b.as_slice()[r]
                );
            }
            assert!(checked > 0, "no owned essential rows checked");
            *out2.lock().unwrap() = Some(format!("checked={checked}"));
        });
        assert!(
            out.lock().unwrap().is_some(),
            "rank 0 did not report the essential-row check"
        );
    }
}
