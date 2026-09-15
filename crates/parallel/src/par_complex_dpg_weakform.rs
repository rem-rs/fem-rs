//! Parallel (distributed-memory) **complex** ultraweak DPG weak form.
//!
//! Port of MFEM's `ParComplexDPGWeakForm`
//! (`miniapps/dpg/util/pcomplexweakform.{hpp,cpp}`): the serial
//! [`ComplexDPGWeakForm`] is assembled on the rank-local sub-mesh and the
//! global system is formed as `Pᴴ A P` over the distributed true-DOF
//! numbering.  The numbering itself (broken volume blocks, face keys,
//! vertex-continuous 2-D H1 traces) is **scalar-agnostic** and shared with the
//! real [`ParDpgWeakForm`](crate::par_dpg_weakform::ParDpgWeakForm) via
//! [`crate::par_dpg_numbering`]; only the assembled values are complex.
//!
//! The formed system is a [`ParComplexCsrMatrix`] in the split re/im CSR
//! layout (`A = A_r + i·A_i`); MFEM's equivalent real 2×2 block operator
//! `[[A_r, −A_i], [A_i, A_r]]` is available for real-only solvers through
//! [`ComplexDpgSystem::to_real_block_csr`] on the serial side.
//!
//! # Status (D172)
//!
//! * Trace numbering: 2-D H1 + face-discontinuous, 3-D H1 (shared mesh edges)
//!   and 3-D ND (`ND_Trace_FECollection`, edge-shared H(curl) traces with the
//!   MFEM orientation semantics) — all implemented in
//!   [`crate::par_dpg_numbering`]; the complex `Pᴴ A P`/`Pᴴ b` split on top.
//! * Static condensation: both the uncondensed and the statically condensed
//!   parallel systems are wired ([`Self::form_linear_system`] /
//!   [`Self::recover_fem_solution`], mirrored on the real
//!   [`ParDpgWeakForm`](crate::par_dpg_weakform::ParDpgWeakForm)).
//! * `-pref`/`-pmg`/`Update()`: not implemented (same gaps as the real form).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{DpgBilinear2, DpgLinear2, DpgTraceBilinear2};
use fem_core::Rank;
use fem_linalg::complex_csr::ComplexCoo;
use fem_mesh::topology::MeshTopology;

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_complex_csr::ParComplexCsrMatrix;
use crate::par_dpg_numbering::{build_numbering, DpgBlockKind, SysBlock, INACTIVE};
use crate::par_vector::{ParComplexVector, ParVector};
use crate::partition::MeshPartition;

/// A formed parallel **complex** DPG system.
pub struct ParComplexDpgSystem {
    /// Parallel matrix `Pᴴ A P` over the global system DOFs
    /// (split re/im CSR).
    pub a: ParComplexCsrMatrix,
    /// Parallel RHS `Pᴴ (b − A_e x_e)`.
    pub b: ParComplexVector,
    /// Number of owned rows.
    pub n_owned: usize,
    /// Total number of local rows (owned + ghost).
    pub n_local: usize,
}

/// The parallel complex DPG weak form: a serial [`ComplexDPGWeakForm`] on the
/// rank-local sub-mesh plus the distributed true-DOF numbering (shared with
/// the real weak form).
pub struct ParComplexDPGWeakForm<M: MeshTopology + Clone + 'static> {
    local: ComplexDPGWeakForm<M>,
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
    /// Global DOF count of **every** trial block (MFEM's `Σ GlobalTrueVSize`,
    /// which is what the miniapps print).
    n_global_trial: Vec<usize>,
    condensed: bool,
    built: bool,
}

impl<M: MeshTopology + Clone + 'static> ParComplexDPGWeakForm<M> {
    /// Create an empty parallel complex weak form on the rank-local `mesh`.
    ///
    /// `partition` must have been produced by
    /// [`partition_mesh_identity`](crate::par_partition::partition_mesh_identity).
    pub fn new(mesh: M, partition: MeshPartition, comm: Comm) -> Self {
        let rank = comm.rank();
        ParComplexDPGWeakForm {
            local: ComplexDPGWeakForm::new(mesh),
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

    // ── builder API (forwards to the serial complex weak form) ──────────────

    /// Volume quadrature order — see [`ComplexDPGWeakForm::set_quad_order`].
    pub fn set_quad_order(&mut self, order: u8) {
        self.local.set_quad_order(order);
    }

    /// Per-trial-block volume quadrature order override — see
    /// [`ComplexDPGWeakForm::set_trial_quad_order`].
    pub fn set_trial_quad_order(&mut self, trial_block: usize, order: u8) {
        self.local.set_trial_quad_order(trial_block, order);
    }

    /// Face quadrature order — see [`ComplexDPGWeakForm::set_face_quad_order`].
    pub fn set_face_quad_order(&mut self, order: u8) {
        self.local.set_face_quad_order(order);
    }

    /// Broken scalar L2 trial block (`p`).
    pub fn add_trial_scalar_space(&mut self, order: u8) -> usize {
        self.kinds.push(DpgBlockKind::Volume { vdim: 1 });
        self.local.add_trial_scalar_space(order)
    }

    /// Broken vector L2 trial block (`u`).
    pub fn add_trial_vector_space(&mut self, order: u8, vdim: usize) -> usize {
        self.kinds.push(DpgBlockKind::Volume { vdim });
        self.local.add_trial_vector_space(order, vdim)
    }

    /// Face-discontinuous trace trial block (`û`).
    pub fn add_trial_trace_space(&mut self, order: u8) -> usize {
        self.kinds.push(DpgBlockKind::FaceDiscontinuous);
        self.local.add_trial_trace_space(order)
    }

    /// Vertex-continuous H1-trace trial block (`p̂`) — 2-D.
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

    /// Trial integrator (real/imag pair) — see
    /// [`ComplexDPGWeakForm::add_trial_integrator`].
    pub fn add_trial_integrator(
        &mut self,
        integ_r: Option<Box<dyn DpgBilinear2>>,
        integ_i: Option<Box<dyn DpgBilinear2>>,
        trial_block: usize,
        test_block: usize,
    ) {
        self.local.add_trial_integrator(integ_r, integ_i, trial_block, test_block);
    }

    /// Test (Riesz) integrator (real/imag pair).
    pub fn add_test_integrator(
        &mut self,
        integ_r: Option<Box<dyn DpgBilinear2>>,
        integ_i: Option<Box<dyn DpgBilinear2>>,
        row_block: usize,
        col_block: usize,
    ) {
        self.local.add_test_integrator(integ_r, integ_i, row_block, col_block);
    }

    /// Trace-face trial integrator (real/imag pair).
    pub fn add_trace_integrator(
        &mut self,
        integ_r: Option<Box<dyn DpgTraceBilinear2>>,
        integ_i: Option<Box<dyn DpgTraceBilinear2>>,
        trial_block: usize,
        test_block: usize,
    ) {
        self.local.add_trace_integrator(integ_r, integ_i, trial_block, test_block);
    }

    /// Domain linear-form integrator (real/imag pair).
    pub fn add_domain_lf_integrator(
        &mut self,
        integ_r: Option<Box<dyn DpgLinear2>>,
        integ_i: Option<Box<dyn DpgLinear2>>,
        test_block: usize,
    ) {
        self.local.add_domain_lf_integrator(integ_r, integ_i, test_block);
    }

    /// `EnableStaticCondensation()` — forwarded for assembly; the formed
    /// parallel system currently requires the uncondensed path.
    pub fn enable_static_condensation(&mut self) {
        self.local.enable_static_condensation();
        self.condensed = true;
    }

    /// `StoreMatrices(bool)`.
    pub fn store_matrices(&mut self, store: bool) {
        self.local.store_matrices(store);
    }

    // ── accessors ───────────────────────────────────────────────────────────

    /// The rank-local serial complex weak form (skeleton, block sizes).
    pub fn local(&self) -> &ComplexDPGWeakForm<M> {
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

    /// Absolute global DOF of a system-index DOF, or `None` if the index
    /// carries no numbered DOF.  Diagnostic helper.
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
    /// `Σ GlobalTrueVSize`, the `Dofs` column of the miniapps.
    pub fn n_global_trial_dofs(&self) -> usize {
        self.n_global_trial.iter().sum()
    }

    /// Per-trial-block global DOF counts (`len = n_trial_blocks`).
    pub fn n_global_trial(&self) -> &[usize] {
        &self.n_global_trial
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

    /// Assemble the rank-local matrix and build the distributed numbering
    /// (shared implementation with the real weak form).
    pub fn assemble(&mut self) {
        self.local.assemble();
        self.build_numbering();
    }

    fn build_numbering(&mut self) {
        self.rank = self.comm.rank();
        let n =
            build_numbering(&self.local, &self.kinds, &self.partition, &self.comm, self.condensed);
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

    /// `FormLinearSystem(ess_tdof_list, x, A, X, B)` — the uncondensed system
    /// (the `pacoustics` default) **and** the statically condensed one.
    ///
    /// * `ess_global` — absolute global DOF ids of the essential (Dirichlet)
    ///   DOFs.
    /// * `x_local_r` / `x_local_i` — rank-local vectors holding the prescribed
    ///   values at the local copies of those DOFs (full serial trial layout,
    ///   length = serial local size).
    ///
    /// The elimination follows MFEM's `ParComplexDPGWeakForm`: the essential
    /// columns of the rank-local matrix are eliminated *before* `Pᴴ A P`, a
    /// unit (real) diagonal is placed on the essential rows with the
    /// prescribed complex value on the RHS.  Under static condensation the
    /// rank-local form hands back its Schur complement over the exposed
    /// (trace) blocks, whose layout matches the condensed numbering built by
    /// [`Self::assemble`].
    ///
    /// Returns the formed system, the initial guess `X` (owned part of `Pᴴ x`,
    /// as a full local complex vector) and the full serial-layout prescribed
    /// vector (for post-processing), both in blocked `[re | im]` form.
    pub fn form_linear_system(
        &mut self,
        ess_global: &[u32],
        x_local_r: &[f64],
        x_local_i: &[f64],
    ) -> (ParComplexDpgSystem, Vec<f64>, Vec<f64>) {
        assert!(self.built, "assemble() must run before form_linear_system()");
        let ess_set: std::collections::BTreeSet<u32> = ess_global.iter().copied().collect();
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
        let mut x_full_r = vec![0.0_f64; self.local.size()];
        let mut x_full_i = vec![0.0_f64; self.local.size()];
        let n = x_full_r.len().min(x_local_r.len());
        x_full_r[..n].copy_from_slice(&x_local_r[..n]);
        let ni = x_full_i.len().min(x_local_i.len());
        x_full_i[..ni].copy_from_slice(&x_local_i[..ni]);

        // (serial elimination happens inside the complex weak form; the
        // returned vectors are blocked `[re | im]` stacks)
        let (system, xs_stack, b_stack) =
            self.local
                .form_linear_system(&ess_local, &x_full_r, &x_full_i);
        let half = system.n_complex();
        assert_eq!(
            half,
            self.sys_global.len(),
            "parallel complex DPG system size mismatch"
        );
        let mat_r = &system.mat_r;
        let mat_i = &system.mat_i;

        let n_compact = self.n_owned + self.n_ghost;
        // Permute the complex system into the compact [owned | ghost] column
        // layout; only owned rows are stored, so the local matrix handed to
        // the split is rectangular (n_owned × n_compact) — accepted since the
        // D167 fix of `ParCsrMatrix::from_local_matrix`'s real sibling logic.
        let mut diag_coo = ComplexCoo::new(self.n_owned, self.n_owned);
        let mut offd_coo = ComplexCoo::new(self.n_owned, self.n_ghost);
        for r in 0..self.n_owned {
            let l = self.inv_perm[r] as usize;
            for k in mat_r.row_ptr[l]..mat_r.row_ptr[l + 1] {
                let pc = self.perm[mat_r.col_idx[k] as usize];
                if pc != INACTIVE {
                    // Real part contributes (v, 0).
                    let v = mat_r.values[k];
                    if v != 0.0 {
                        self.add_complex_entry(&mut diag_coo, &mut offd_coo, r, pc as usize, v, 0.0);
                    }
                }
            }
            for k in mat_i.row_ptr[l]..mat_i.row_ptr[l + 1] {
                let pc = self.perm[mat_i.col_idx[k] as usize];
                if pc != INACTIVE {
                    // Imaginary part contributes (0, v).
                    let v = mat_i.values[k];
                    if v != 0.0 {
                        self.add_complex_entry(&mut diag_coo, &mut offd_coo, r, pc as usize, 0.0, v);
                    }
                }
            }
        }
        let a = ParComplexCsrMatrix::new(
            diag_coo.into_complex_csr(),
            offd_coo.into_complex_csr(),
            self.n_owned,
            self.n_ghost,
            self.ghost_exchange.clone(),
            self.comm.clone(),
        );

        // RHS: Pᴴ b (owned rows), ghost segment zeroed — blocked [re | im].
        let br = &b_stack[..half];
        let bi = &b_stack[half..];
        let mut b_loc_r = vec![0.0_f64; n_compact];
        let mut b_loc_i = vec![0.0_f64; n_compact];
        for r in 0..self.n_owned {
            b_loc_r[r] = br[self.inv_perm[r] as usize];
            b_loc_i[r] = bi[self.inv_perm[r] as usize];
        }
        let b = ParComplexVector {
            re: ParVector::from_local_raw(
                b_loc_r,
                self.n_owned,
                self.ghost_exchange.clone(),
                self.comm.clone(),
            ),
            im: ParVector::from_local_raw(
                b_loc_i,
                self.n_owned,
                self.ghost_exchange.clone(),
                self.comm.clone(),
            ),
        };

        // Initial guess X = Pᴴ xs (owned rows only).
        let xsr = &xs_stack[..half];
        let xsi = &xs_stack[half..];
        let mut x0_r = vec![0.0_f64; n_compact];
        let mut x0_i = vec![0.0_f64; n_compact];
        for r in 0..self.n_owned {
            x0_r[r] = xsr[self.inv_perm[r] as usize];
            x0_i[r] = xsi[self.inv_perm[r] as usize];
        }
        let mut x0 = x0_r;
        x0.extend_from_slice(&x0_i);
        let sys = ParComplexDpgSystem {
            a,
            b,
            n_owned: self.n_owned,
            n_local: n_compact,
        };
        (sys, x0, {
            let mut xf = x_full_r;
            xf.extend_from_slice(&x_full_i);
            xf
        })
    }

    /// Route one matrix entry into the diag/offd COO by compact column id.
    fn add_complex_entry(
        &self,
        diag: &mut ComplexCoo,
        offd: &mut ComplexCoo,
        row: usize,
        col: usize,
        re: f64,
        im: f64,
    ) {
        if col < self.n_owned {
            diag.add(row, col, re, im);
        } else {
            offd.add(row, col - self.n_owned, re, im);
        }
    }

    /// `RecoverFEMSolution(X, x)`: lift the global owned solution back to the
    /// rank-local (ghost-filled) trial vector in the blocked `[re | im]`
    /// layout the serial [`ComplexDPGWeakForm`] uses for post-processing.
    /// Under static condensation the serial form back-solves the private
    /// (volume-block) DOFs element-wise.
    pub fn recover_fem_solution(&self, x_owned: &[f64]) -> Vec<f64> {
        assert!(self.built, "assemble() must run before recover_fem_solution()");
        let n_compact = self.n_owned + self.n_ghost;
        let half = n_compact;
        let mut data = vec![0.0_f64; 2 * half];
        // `x_owned` is the stacked owned solution `[re(n_owned); im(n_owned)]`.
        let nr = self.n_owned.min(x_owned.len());
        data[..nr].copy_from_slice(&x_owned[..nr]);
        if x_owned.len() > self.n_owned {
            let ni = self
                .n_owned
                .min(x_owned.len() - self.n_owned);
            data[half..half + ni].copy_from_slice(&x_owned[self.n_owned..self.n_owned + ni]);
        }
        // The exchange must run on EVERY rank (not only those holding
        // ghosts): the channel sets are symmetric after `build_numbering`,
        // and a rank holding no ghosts may still have to *serve* another
        // rank's requests (the condensed 3-D system: the lowest rank owns
        // every shared trace dof).  Trivial on one rank (empty channels).
        self.ghost_exchange.forward(&self.comm, &mut data[..half]);
        self.ghost_exchange.forward(&self.comm, &mut data[half..]);
        let n_target = self.sys_global.len();
        let mut u = vec![0.0_f64; 2 * n_target];
        for s in 0..n_target {
            let p = self.perm[s];
            if p != INACTIVE {
                u[s] = data[p as usize];
                u[n_target + s] = data[half + p as usize];
            }
        }
        // The serial form unstacks `[re | im]` and (condensed case) back-solves
        // the volume blocks.
        let (out_r, out_i) = self.local.recover_fem_solution(&u);
        let mut out = out_r;
        out.extend_from_slice(&out_i);
        out
    }

    /// Local boundary-face DOFs of trial block `b` with their physical
    /// points: `(absolute global dof, point)` — mirror of
    /// [`crate::par_dpg_weakform::ParDpgWeakForm::trace_boundary_dofs`].
    ///
    /// A face is a *global* boundary face iff it has exactly one adjacent
    /// element, so every rank holding it sees it as a boundary face — the
    /// sets computed here are consistent across ranks without any exchange.
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
                    self.local.face_dof_point(&sk, f, k),
                ));
            }
        }
        out
    }

    /// Boundary-face DOFs of trial block `b` as
    /// `(absolute global dof, full serial trial index)` — covers the scalar
    /// trace families ([`SkeletonSpace`]) **and** the 3-D ND trace
    /// ([`TraceSpace`]).  The index form lets callers project vector /
    /// tangential boundary data themselves (MFEM
    /// `ProjectBdrCoefficientTangent` / `…Normal` need the face Jacobian at
    /// each DOF node, not just the point).
    pub fn trace_boundary_dofs_ix(&self, b: usize) -> Vec<(u32, usize)> {
        let bi = self
            .blocks
            .iter()
            .position(|blk| blk.trial == b)
            .expect("trace_boundary_dofs_ix: block not in the system");
        let blk = &self.blocks[bi];
        let mut out = Vec::new();
        if matches!(self.kinds[b], DpgBlockKind::FaceNd) {
            let tr = self.local.nd_trace(b);
            for f in 0..tr.n_faces() {
                if !tr.is_boundary_face(f) {
                    continue;
                }
                for &d in tr.face_dof_list(f) {
                    if d >= blk.size || blk.gof[d] == INACTIVE {
                        continue;
                    }
                    out.push((blk.global_base as u32 + blk.gof[d], blk.full_base + d));
                }
            }
        } else {
            let sk = self.local.skeleton(b);
            for f in 0..sk.n_faces() {
                if !sk.is_boundary_face(f) {
                    continue;
                }
                for &d in sk.face_dof_list(f) {
                    if d >= blk.size || blk.gof[d] == INACTIVE {
                        continue;
                    }
                    out.push((blk.global_base as u32 + blk.gof[d], blk.full_base + d));
                }
            }
        }
        out
    }

    /// Write prescribed values into the rank-local vector for every essential
    /// DOF present locally, using `value(point)` — mirror of
    /// [`crate::par_dpg_weakform::ParDpgWeakForm::fill_essential_values`].
    ///
    /// `pairs` must be the globally merged `(global dof, point)` list of
    /// [`Self::trace_boundary_dofs`].  `x_local` is in the **full** serial
    /// trial layout.
    pub fn fill_essential_values(
        &self,
        x_local_r: &mut [f64],
        x_local_i: &mut [f64],
        pairs: &[(u32, Vec<f64>)],
        value: &dyn Fn(&[f64]) -> (f64, f64),
    ) {
        let mut by_id: HashMap<u32, (f64, f64)> = HashMap::new();
        for (g, p) in pairs {
            by_id.entry(*g).or_insert_with(|| value(p));
        }
        for blk in &self.blocks {
            for d in 0..blk.size {
                if blk.gof[d] == INACTIVE {
                    continue;
                }
                let g = blk.global_base as u32 + blk.gof[d];
                if let Some(&(vr, vi)) = by_id.get(&g) {
                    let t = blk.full_base + d;
                    if t < x_local_r.len() {
                        x_local_r[t] = vr;
                        x_local_i[t] = vi;
                    }
                }
            }
        }
    }

    /// Merge `(global dof, point)` pairs across all ranks (lowest rank wins) —
    /// mirror of [`crate::par_dpg_weakform::ParDpgWeakForm::merge_dof_points`].
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

    /// Global `‖residual‖₂` over the **owned** elements (MFEM recomputes
    /// `sqrt(Σ_local res²)` across ranks, `pmaxwell.cpp:902-912`) — mirror of
    /// [`crate::par_dpg_weakform::ParDpgWeakForm::global_residual_norm`].
    ///
    /// Since the D195 fix the serial
    /// [`fem_assembly::ComplexDPGWeakForm::compute_residual`] gathers the
    /// element coefficients **unsigned** over the sign-folded stored columns,
    /// `(L⁻¹B̃D)·x = L⁻¹B̃·(D·x)`, which is exactly MFEM's
    /// `ComplexDPGWeakForm::ComputeResidual` value; the row-level square
    /// accumulation below (no per-element `sqrt` round-trip) is bit-identical
    /// to the round-37 `global_residual_norm_unfolded` workaround for every
    /// configuration.
    ///
    /// `x_full` is the blocked `[re | im]` local trial vector produced by
    /// [`Self::recover_fem_solution`].
    pub fn global_residual_norm(&self, x_full: &[f64]) -> f64 {
        let n = x_full.len() / 2;
        let (xr, xi) = (&x_full[..n], &x_full[n..]);
        let rank = self.comm.rank();
        let nblocks = self.local.n_trial_blocks();
        let mut acc = 0.0_f64;
        for e in 0..self.local.mesh().n_elements() {
            if self.partition.elem_owner[e] != rank {
                continue;
            }
            let (ybr, ybi, fr, fi, n_tr) = self.local.element_stored(e);
            // Element coefficients in the GLOBAL (unsigned) basis — the
            // stored blocks are sign-folded already (D195: the orientation
            // signs are applied exactly once, in the storage).
            let mut ur = vec![0.0_f64; n_tr];
            let mut ui = vec![0.0_f64; n_tr];
            let mut off = 0usize;
            for b in 0..nblocks {
                let vd = self.local.trial_element_vdofs(b, e as u32);
                for (li, &g) in vd.iter().enumerate() {
                    ur[off + li] = xr[g];
                    ui[off + li] = xi[g];
                }
                off += vd.len();
            }
            let rows = ybr.len() / n_tr;
            for k in 0..rows {
                let mut sr = -fr[k];
                let mut si = -fi[k];
                for j in 0..n_tr {
                    sr += ybr[k * n_tr + j] * ur[j] - ybi[k * n_tr + j] * ui[j];
                    si += ybr[k * n_tr + j] * ui[j] + ybi[k * n_tr + j] * ur[j];
                }
                acc += sr * sr + si * si;
            }
        }
        self.comm.allreduce_sum_f64(acc).max(0.0).sqrt()
    }

    /// Retired D195 workaround, kept as an alias of
    /// [`Self::global_residual_norm`] (which `miniapps/dpg/pmaxwell.rs`
    /// still names): the serial `compute_residual` now applies the trace
    /// orientation signs exactly once — through the sign-folded stored
    /// columns with an unsigned gather — so this method no longer needs its
    /// own unfolding math.  Bit-identical to the pre-simplification bypass.
    pub fn global_residual_norm_unfolded(&self, x_full: &[f64]) -> f64 {
        self.global_residual_norm(x_full)
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
}

impl<M: MeshTopology + Clone + 'static> crate::par_dpg_numbering::DpgNumberingLocal<M>
    for ComplexDPGWeakForm<M>
{
    fn n_trial_blocks(&self) -> usize {
        ComplexDPGWeakForm::n_trial_blocks(self)
    }
    fn trial_block_sizes(&self) -> Vec<usize> {
        ComplexDPGWeakForm::trial_block_sizes(self)
    }
    fn trial_offsets(&self) -> Vec<usize> {
        ComplexDPGWeakForm::trial_offsets(self)
    }
    fn exposed_block_offsets(&self) -> Vec<usize> {
        ComplexDPGWeakForm::exposed_block_offsets(self)
    }
    fn with_skeleton<R>(&self, tb: usize, f: impl FnOnce(&fem_assembly::dpg::dpg_basis::SkeletonSpace<M>) -> R) -> R {
        let sk = ComplexDPGWeakForm::skeleton(self, tb);
        f(&sk)
    }
    fn with_nd_trace<R>(
        &self,
        tb: usize,
        f: impl FnOnce(&fem_assembly::dpg::dpg_basis::TraceSpace<M>) -> R,
    ) -> R {
        let tr = ComplexDPGWeakForm::nd_trace(self, tb);
        f(&tr)
    }
    fn numbering_mesh(&self) -> &M {
        ComplexDPGWeakForm::mesh(self)
    }
    fn trial_element_vdofs(&self, tb: usize, e: u32) -> Vec<usize> {
        ComplexDPGWeakForm::trial_element_vdofs(self, tb, e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::Launcher;
    use crate::par_dpg_numbering::allreduce_sum_u64;
    use crate::par_partition::partition_mesh_identity;
    use crate::WorkerConfig;
    use fem_assembly::dpg::dpg_integrators::{
        DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgMassIntegrator,
        DpgMixedScalarWeakGradientIntegrator, DpgMixedVectorGradientIntegrator,
        DpgMixedVectorWeakDivergenceIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
        DpgTVectorFEMassIntegrator, DpgTraceIntegrator, DpgVectorFEDivergenceIntegrator,
        DpgVectorFEMassIntegrator,
    };
    use fem_mesh::Mesh;

    const OMEGA: f64 = std::f64::consts::PI * 2.0;

    /// Build the `pacoustics` (Helmholtz, prob 0) ultraweak setup on the
    /// rank-local mesh: trial p (L2, o−1), u (L2 vector, o−1), p̂ (H1 trace, o),
    /// û (RT trace, o−1); adjoint-graph test norm — C++ pacoustics.cpp block
    /// table (order 1, delta-order 1 ⇒ ω = 2π·rnum, rnum = 1).
    fn build_acoustics(a: &mut ParComplexDPGWeakForm<Mesh<2>>) -> [usize; 4] {
        let p = 1u8;
        let test_order = p + 1; // delta_order = 1
        a.set_quad_order((2 * test_order as usize).min(255) as u8);
        a.set_face_quad_order(test_order + p - 1);
        a.store_matrices(true);
        let ps = a.add_trial_scalar_space(p - 1);
        let us = a.add_trial_vector_space(p - 1, 2);
        let hatp = a.add_trial_trace_space_h1(p);
        let hatu = a.add_trial_trace_space(p - 1);
        let q = a.add_test_space(VolKind::Scalar, test_order);
        let v = a.add_test_space(VolKind::HDiv, test_order - 1);

        // iω (p, q) / −(u, ∇q) / −(p, ∇·v) / iω (u, v)
        a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: OMEGA })), ps, q);
        a.add_trial_integrator(Some(Box::new(DpgTGradientIntegrator { q: -1.0 })), None, us, q);
        a.add_trial_integrator(
            Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 })),
            None,
            ps,
            v,
        );
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: OMEGA })),
            us,
            v,
        );
        // <p̂, v·n> / <û, q>
        a.add_trace_integrator(Some(Box::new(DpgNormalTraceIntegrator)), None, hatp, v);
        a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hatu, q);

        // Adjoint graph norm (test integrators).
        a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, q, q);
        a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q, q);
        a.add_test_integrator(Some(Box::new(DpgDivDivIntegrator { q: 1.0 })), None, v, v);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, v, v);
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorGradientIntegrator {
                q: vec![vec![-OMEGA, 0.0], vec![0.0, -OMEGA]],
            })),
            v,
            q,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator {
                q: vec![vec![-OMEGA, 0.0], vec![0.0, -OMEGA]],
            })),
            q,
            v,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: OMEGA * OMEGA })),
            None,
            v,
            v,
        );
        a.add_test_integrator(
            Some(Box::new(DpgMassIntegrator { q: OMEGA * OMEGA })),
            None,
            q,
            q,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgVectorFEDivergenceIntegrator { q: -OMEGA })),
            q,
            v,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: -OMEGA })),
            v,
            q,
        );
        [ps, us, hatp, hatu]
    }

    /// C++ `mpirun -np {1,2} pacoustics -no-vis` (MFEM 4.10, defaults:
    /// inline-quad 4×4 mesh, order 1, rnum 1) prints `Dofs = 113` with
    /// per-space true sizes p̂⁻=16 (p), 32 (u), 25 (p̂), 40 (û).
    const CPP_DOFS: [usize; 4] = [16, 32, 25, 40];

    #[test]
    fn complex_acoustics_numbering_np2_matches_cpp_reference() {
        let full = Arc::new(Mesh::<2>::unit_square_quad(4));
        let out = Arc::new(std::sync::Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let rank = comm.rank();
            let par_mesh = partition_mesh_identity(&ma, &comm);
            let mut a = ParComplexDPGWeakForm::new(
                par_mesh.local_mesh().clone(),
                par_mesh.partition().clone(),
                comm.clone(),
            );
            let blocks = build_acoustics(&mut a);
            a.assemble();

            // Rank-invariant global counts — the C++ reference numbers.
            let n_global = a.n_global_trial();
            assert_eq!(
                n_global,
                &CPP_DOFS[..],
                "rank {rank}: per-block global trial dofs must match the C++ pacoustics"
            );
            assert_eq!(a.n_global_trial_dofs(), 113, "rank {rank}: total dofs");
            assert_eq!(a.n_global_dofs(), 113, "rank {rank}: system dofs (no sc)");

            // Owned dofs sum to the global count across ranks; every rank
            // holds ghosts of the cross-rank couplings.
            let owned_sum = allreduce_sum_u64(&comm, a.n_owned_dofs() as u64) as usize;
            assert_eq!(
                owned_sum, 113,
                "rank {rank}: owned dofs must sum to the global count"
            );
            if rank == 0 {
                *out2.lock().unwrap() = Some(format!(
                    "np2 ok: blocks {blocks:?} globals {n_global:?}, owned {} ghosts {}",
                    a.n_owned_dofs(),
                    a.n_ghost_dofs()
                ));
            }

            // The formed (uncondensed) system: RHS has the full compact
            // local layout, and the cross-rank ghost couplings survive the
            // split (non-empty offd block over all ranks).
            let ess: Vec<u32> = Vec::new();
            let nlocal = a.local().size();
            let zero_x = vec![0.0_f64; nlocal];
            let (sys, _x0, _xf) = a.form_linear_system(&ess, &zero_x, &zero_x);
            assert_eq!(
                sys.b.re.len(),
                a.n_owned_dofs() + a.n_ghost_dofs(),
                "rank {rank}: RHS must cover the compact local layout"
            );
            let offd_nnz: usize = {
                let o = sys.a.offd_block();
                o.re_vals.len()
            };
            let total_offd = allreduce_sum_u64(&comm, offd_nnz as u64) as usize;
            assert!(total_offd > 0, "cross-rank ghost couplings must survive");
        });
        let msg = out.lock().unwrap().clone();
        assert!(msg.is_some(), "rank 0 must report");
    }

    /// C++ `mpirun -np {1,2} pmaxwell -m inline-hex.mesh -no-vis -pref 0
    /// -prob 0` (MFEM 4.10, `inline-hex.mesh` = 4×4×4 hexes, order 1) prints
    /// `Dofs = 984` with per-space true sizes E=192, H=192, Ê=300, Ĥ=300
    /// (ND trace(1): 1 dof per each of the 300 mesh edges).
    #[test]
    fn complex_maxwell_nd_trace_numbering_np2_matches_cpp_reference() {
        // 2×2×2 hex grid (the `dpg_maxwell_3d` `hex2.mesh` reference): E=24,
        // H=24, Ê=Ĥ = 1 dof × 54 mesh edges, total 156 — the C++ Dofs column
        // of `pmaxwell -m hex2.mesh -o 1 -do 0`.
        use fem_assembly::dpg::dpg_integrators::{
            DpgCurl3dPairingIntegrator, DpgTVectorFEMassIntegrator, DpgTangentTraceIntegrator3D,
        };
        let full = Arc::new(Mesh::<3>::unit_cube_hex(2));
        let out = Arc::new(std::sync::Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let rank = comm.rank();
            let par_mesh = partition_mesh_identity(&ma, &comm);
            let mut a = ParComplexDPGWeakForm::new(
                par_mesh.local_mesh().clone(),
                par_mesh.partition().clone(),
                comm.clone(),
            );
            a.set_quad_order(4);
            a.store_matrices(true);
            let es = a.add_trial_vector_space(0, 3);
            let hs = a.add_trial_vector_space(0, 3);
            let hate = a.add_trial_trace_space_nd(1);
            let hath = a.add_trial_trace_space_nd(1);
            let g = a.add_test_space(VolKind::HCurl, 1);
            a.add_trial_integrator(
                None,
                Some(Box::new(DpgTVectorFEMassIntegrator { q: 1.0 })),
                es,
                g,
            );
            a.add_trial_integrator(
                Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })),
                None,
                hs,
                g,
            );
            a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);
            a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
            a.assemble();

            // Rank-invariant global counts — the C++ reference numbers.
            assert_eq!(
                a.n_global_trial(),
                &[24, 24, 54, 54],
                "rank {rank}: per-block global trial dofs must match the C++ pmaxwell"
            );
            assert_eq!(a.n_global_trial_dofs(), 156, "rank {rank}: total dofs");
            let owned_sum = allreduce_sum_u64(&comm, a.n_owned_dofs() as u64) as usize;
            assert_eq!(owned_sum, 156, "rank {rank}: owned dofs must sum to the total");

            // Essential BCs over the Ê ND trace (edge-shared dofs included):
            // the parallel boundary walk must reproduce the serial one's size
            // and the formed multi-rank system must keep cross-rank couplings.
            let pairs = a.trace_boundary_dofs_ix(hate);
            let ess_g: Vec<u32> = pairs.iter().map(|(g, _)| *g).collect();
            // Global count of the boundary dofs: owned ids are disjoint
            // across ranks, so summing the owned intersections counts each
            // global dof exactly once (48 = the boundary edges of the 2×2×2
            // grid = 54 total edges − 6 fully interior ones).
            let ess_set: std::collections::BTreeSet<u32> = ess_g.iter().copied().collect();
            let local_ess = a.owned_global_ids().iter().filter(|g| ess_set.contains(g)).count();
            let ess_sum = allreduce_sum_u64(&comm, local_ess as u64) as usize;
            assert_eq!(ess_sum, 48, "rank {rank}: global boundary ND dofs");
            let nlocal = a.local().size();
            let zero_x = vec![0.0_f64; nlocal];
            let (sys, _x0, _xf) = a.form_linear_system(&ess_g, &zero_x, &zero_x);
            let offd_nnz = sys.a.offd_block().re_vals.len();
            let total_offd = allreduce_sum_u64(&comm, offd_nnz as u64) as usize;
            assert!(total_offd > 0, "rank {rank}: cross-rank ghost couplings");
            if rank == 0 {
                *out2.lock().unwrap() = Some(format!(
                    "nd-trace np2 ok: globals {:?}, owned {} ghosts {}",
                    a.n_global_trial(),
                    a.n_owned_dofs(),
                    a.n_ghost_dofs()
                ));
            }
        });
        let msg = out.lock().unwrap().clone();
        assert!(msg.is_some(), "rank 0 must report: {msg:?}");
    }

    /// 3-D H1-trace parallel numbering (MFEM `H1_Trace_FECollection(p, 3)` on
    /// the 2×2×2 hex grid): `p = 1` → 27 vertex dofs; `p = 2` → 27 vertices +
    /// 1 dof × 54 edges + 1 face-interior dof × 36 faces = 117 — the MFEM
    /// `GlobalTrueVSize` layout (vertices, then edges, then face interiors).
    #[test]
    fn complex_h1_trace_3d_numbering_np2() {
        use fem_assembly::dpg::dpg_integrators::DpgMassIntegrator;
        let full = Arc::new(Mesh::<3>::unit_cube_hex(2));
        let out = Arc::new(std::sync::Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let rank = comm.rank();
            let par_mesh = partition_mesh_identity(&ma, &comm);
            let mut a = ParComplexDPGWeakForm::new(
                par_mesh.local_mesh().clone(),
                par_mesh.partition().clone(),
                comm.clone(),
            );
            a.set_quad_order(4);
            let ps = a.add_trial_scalar_space(0);
            let hatp1 = a.add_trial_trace_space_h1(1);
            let hatp2 = a.add_trial_trace_space_h1(2);
            let _ = hatp1;
            let q = a.add_test_space(VolKind::Scalar, 2);
            a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: 1.0 })), ps, q);
            a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q, q);
            a.assemble();
            assert_eq!(
                a.n_global_trial(),
                &[8, 27, 117],
                "rank {rank}: 3-D H1-trace global dofs (MFEM entity layout)"
            );
            let owned_sum = allreduce_sum_u64(&comm, a.n_owned_dofs() as u64) as usize;
            assert_eq!(owned_sum, 152, "rank {rank}: owned dofs must sum to the total");
            if rank == 0 {
                *out2.lock().unwrap() = Some(format!(
                    "h1-trace 3d np2 ok: globals {:?}, owned {} ghosts {}",
                    a.n_global_trial(),
                    a.n_owned_dofs(),
                    a.n_ghost_dofs()
                ));
            }
        });
        let msg = out.lock().unwrap().clone();
        assert!(msg.is_some(), "rank 0 must report: {msg:?}");
    }

    #[test]
    fn complex_acoustics_numbering_np1_matches_serial() {
        let full = Arc::new(Mesh::<2>::unit_square_quad(4));
        let out = Arc::new(std::sync::Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm| {
            let par_mesh = partition_mesh_identity(&ma, &comm);
            let mut a = ParComplexDPGWeakForm::new(
                par_mesh.local_mesh().clone(),
                par_mesh.partition().clone(),
                comm.clone(),
            );
            let _ = build_acoustics(&mut a);
            a.assemble();

            // At one rank the parallel numbering must reproduce the serial
            // complex weak form exactly.
            assert_eq!(a.n_owned_dofs(), a.local().size());
            assert_eq!(a.n_global_trial_dofs(), a.local().size());
            assert_eq!(a.n_global_trial(), &CPP_DOFS[..]);
            *out2.lock().unwrap() = Some(format!(
                "np1 ok: serial size {} == parallel owned {}",
                a.local().size(),
                a.n_owned_dofs()
            ));
        });
        let msg = out.lock().unwrap().clone();
        assert!(msg.is_some(), "rank 0 must report: {:?}", msg);
    }

    /// Regression (r35): the **formed parallel complex system** at one rank
    /// must equal the serial `ComplexDPGWeakForm` system entry-by-entry
    /// (complex `Pᴴ A P` degenerates to the identity permutation at `-np 1`),
    /// with essential elimination applied on both sides.  This mirrors the
    /// real form's `two_rank_system_matches_serial_full_mesh`, which caught
    /// the dropped off-diagonal block of `Pᵀ A P`.
    #[test]
    fn complex_np1_formed_system_matches_serial() {
        let full = Arc::new(Mesh::<2>::unit_square_quad(4));
        let out = Arc::new(std::sync::Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&full);
        ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm| {
            // Serial reference (full mesh, same block table as
            // `build_acoustics`, on the serial complex weak form).
            let mut ser = ComplexDPGWeakForm::new((*ma).clone());
            ser.set_quad_order((2 * 2 as usize).min(255) as u8);
            ser.set_face_quad_order(2 + 1 - 1);
            ser.store_matrices(true);
            let ps_ser = ser.add_trial_scalar_space(0);
            let us_ser = ser.add_trial_vector_space(0, 2);
            let hatp_ser = ser.add_trial_trace_space_h1(1);
            let hatu_ser = ser.add_trial_trace_space(0);
            let q_ser = ser.add_test_space(VolKind::Scalar, 2);
            let v_ser = ser.add_test_space(VolKind::HDiv, 2 - 1);
            ser.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: OMEGA })), ps_ser, q_ser);
            ser.add_trial_integrator(Some(Box::new(DpgTGradientIntegrator { q: -1.0 })), None, us_ser, q_ser);
            ser.add_trial_integrator(Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 })), None, ps_ser, v_ser);
            ser.add_trial_integrator(None, Some(Box::new(DpgTVectorFEMassIntegrator { q: OMEGA })), us_ser, v_ser);
            ser.add_trace_integrator(Some(Box::new(DpgNormalTraceIntegrator)), None, hatp_ser, v_ser);
            ser.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hatu_ser, q_ser);
            ser.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, q_ser, q_ser);
            ser.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q_ser, q_ser);
            ser.add_test_integrator(Some(Box::new(DpgDivDivIntegrator { q: 1.0 })), None, v_ser, v_ser);
            ser.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, v_ser, v_ser);
            ser.add_test_integrator(
                None,
                Some(Box::new(DpgMixedVectorGradientIntegrator {
                    q: vec![vec![-OMEGA, 0.0], vec![0.0, -OMEGA]],
                })),
                v_ser,
                q_ser,
            );
            ser.add_test_integrator(
                None,
                Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator {
                    q: vec![vec![-OMEGA, 0.0], vec![0.0, -OMEGA]],
                })),
                q_ser,
                v_ser,
            );
            ser.add_test_integrator(
                Some(Box::new(DpgVectorFEMassIntegrator { q: OMEGA * OMEGA })),
                None,
                v_ser,
                v_ser,
            );
            ser.add_test_integrator(
                None,
                Some(Box::new(DpgVectorFEDivergenceIntegrator { q: -OMEGA })),
                q_ser,
                v_ser,
            );
            ser.add_test_integrator(
                None,
                Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: -OMEGA })),
                v_ser,
                q_ser,
            );
            ser.add_test_integrator(
                Some(Box::new(DpgMassIntegrator { q: OMEGA * OMEGA })),
                None,
                q_ser,
                q_ser,
            );
            ser.assemble();
            let sk = ser.skeleton(hatp_ser);
            let base = ser.trial_offsets()[hatp_ser];
            let mut ess: Vec<usize> = Vec::new();
            for f in 0..sk.n_faces() {
                if !sk.is_boundary_face(f) {
                    continue;
                }
                for &d in sk.face_dof_list(f) {
                    ess.push(base + d);
                }
            }
            ess.sort_unstable();
            ess.dedup();
            let n_ser = ser.size();
            // Nonzero plane-wave prescribed values (ω = 2π, β = ω/√2).
            let beta = (2.0 * std::f64::consts::PI) / 2.0f64.sqrt();
            let mut x_ser_r = vec![0.0_f64; n_ser];
            let mut x_ser_i = vec![0.0_f64; n_ser];
            let coords = |n: u32| ser.mesh().node_coords(n).to_vec();
            for f in 0..sk.n_faces() {
                if !sk.is_boundary_face(f) {
                    continue;
                }
                let nodes = sk.face_nodes(f).to_vec();
                for (k, &d) in sk.face_dof_list(f).iter().enumerate() {
                    // linear 2-point interpolation, order-1 H1 trace
                    let s = k as f64 / 1.0;
                    let c0 = coords(nodes[0]);
                    let c1 = coords(nodes[nodes.len() - 1]);
                    let pt = [
                        (1.0 - s) * c0[0] + s * c1[0],
                        (1.0 - s) * c0[1] + s * c1[1],
                    ];
                    x_ser_r[base + d] = (beta * (pt[0] + pt[1])).cos();
                    x_ser_i[base + d] = (beta * (pt[0] + pt[1])).sin();
                }
            }
            let (ser_sys, ser_xs, ser_b) =
                ser.form_linear_system(&ess, &x_ser_r, &x_ser_i);
            let ser_r = &ser_sys.mat_r;
            let ser_i = &ser_sys.mat_i;
            let mut ser_lookup = std::collections::HashMap::<(usize, usize), (f64, f64)>::new();
            for r in 0..ser_r.nrows {
                for p in ser_r.row_ptr[r]..ser_r.row_ptr[r + 1] {
                    ser_lookup.insert((r, ser_r.col_idx[p] as usize), (ser_r.values[p], 0.0));
                }
                for p in ser_i.row_ptr[r]..ser_i.row_ptr[r + 1] {
                    ser_lookup
                        .entry((r, ser_i.col_idx[p] as usize))
                        .and_modify(|e| e.1 = ser_i.values[p])
                        .or_insert((0.0, ser_i.values[p]));
                }
            }

            // Parallel system at one rank.
            let par_mesh = partition_mesh_identity(&ma, &comm);
            let mut a = ParComplexDPGWeakForm::new(
                par_mesh.local_mesh().clone(),
                par_mesh.partition().clone(),
                comm.clone(),
            );
            let blocks_ix = build_acoustics(&mut a);
            a.assemble();
            let pairs = a.trace_boundary_dofs(blocks_ix[2]);
            let merged = a.merge_dof_points(&pairs);
            let ess_g: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();
            let n_local = a.local().size();
            let mut xr = vec![0.0_f64; n_local];
            let mut xi = vec![0.0_f64; n_local];
            a.fill_essential_values(&mut xr, &mut xi, &merged, &|pt: &[f64]| {
                let s = beta * (pt[0] + pt[1]);
                (s.cos(), s.sin())
            });
            let (sys, x0, _xf) = a.form_linear_system(&ess_g, &xr, &xi);

            // Compare: parallel compact (owned) rows vs serial rows, using the
            // absolute global ids to match rows/columns.
            let og: Vec<u32> = a.owned_global_ids().to_vec();
            let gg: Vec<u32> = a.ghost_global_ids().to_vec();
            // `global dof -> serial layout index` from the numbering blocks
            // (at one rank the local dof layout equals the serial layout).
            let mut g2s = std::collections::HashMap::<u32, usize>::new();
            for blk in &a.blocks {
                for d in 0..blk.size {
                    if blk.gof[d] != INACTIVE {
                        g2s.insert(blk.global_base as u32 + blk.gof[d], blk.full_base + d);
                    }
                }
            }
            let diag = sys.a.diag_block();
            let offd = sys.a.offd_block();
            let (mut ncmp, mut maxabs) = (0usize, 0.0_f64);
            for r in 0..sys.n_owned {
                let sr = g2s[&og[r]];
                let mut mine = std::collections::HashMap::<usize, (f64, f64)>::new();
                for p in diag.row_ptr[r]..diag.row_ptr[r + 1] {
                    let sc = g2s[&og[diag.col_idx[p] as usize]];
                    let e = mine.entry(sc).or_insert((0.0, 0.0));
                    e.0 += diag.re_vals[p];
                    e.1 += diag.im_vals[p];
                }
                for p in offd.row_ptr[r]..offd.row_ptr[r + 1] {
                    let sc = g2s[&gg[offd.col_idx[p] as usize]];
                    let e = mine.entry(sc).or_insert((0.0, 0.0));
                    e.0 += offd.re_vals[p];
                    e.1 += offd.im_vals[p];
                }
                let mut cols: Vec<usize> = mine.keys().copied().collect();
                for p in ser_r.row_ptr[sr]..ser_r.row_ptr[sr + 1] {
                    cols.push(ser_r.col_idx[p] as usize);
                }
                for p in ser_i.row_ptr[sr]..ser_i.row_ptr[sr + 1] {
                    cols.push(ser_i.col_idx[p] as usize);
                }
                cols.sort_unstable();
                cols.dedup();
                for sc in cols {
                    let (vr, vi) = ser_lookup.get(&(sr, sc)).copied().unwrap_or((0.0, 0.0));
                    let (mr, mi) = mine.get(&sc).copied().unwrap_or((0.0, 0.0));
                    let d = (vr - mr).abs().max((vi - mi).abs());
                    if d > maxabs {
                        maxabs = d;
                    }
                    ncmp += 1;
                }
            }
            // RHS comparison.
            let mut bmax = 0.0_f64;
            for r in 0..sys.n_owned {
                let sr = g2s[&og[r]];
                bmax = bmax
                    .max((sys.b.re.as_slice()[r] - ser_b[sr]).abs())
                    .max((sys.b.im.as_slice()[r] - ser_b[n_ser + sr]).abs());
            }
            // Initial-guess comparison.
            let mut x0max = 0.0_f64;
            let half0 = x0.len() / 2;
            for r in 0..sys.n_owned {
                let sr = g2s[&og[r]];
                x0max = x0max
                    .max((x0[r] - ser_xs[sr]).abs())
                    .max((x0[half0 + r] - ser_xs[n_ser + sr]).abs());
            }
            // Solve both systems and compare the solutions: serial via the
            // real doubled operator + plain CG (the `dpg_acoustics_2d` path),
            // parallel via the complex PCG.
            let big = ser_sys.to_real_block_csr();
            let mut xs_ser = ser_xs.clone();
            let cfg = fem_solver::SolverConfig {
                rtol: 1e-12,
                max_iter: 10000,
                ..fem_solver::SolverConfig::default()
            };
            let ser_res = fem_solver::solve_pcg_operator_precond(
                big.nrows,
                |x, y| big.spmv(x, y),
                &ser_b,
                &mut xs_ser,
                |_r, z| z.copy_from_slice(_r),
                &cfg,
            )
            .expect("serial CG");
            let exchange = a.ghost_exchange_arc();
            let half = x0.len() / 2;
            let mut xv = crate::par_vector::ParComplexVector {
                re: crate::par_vector::ParVector::from_local_raw(
                    x0[..half].to_vec(),
                    sys.n_owned,
                    exchange.clone(),
                    comm.clone(),
                ),
                im: crate::par_vector::ParVector::from_local_raw(
                    x0[half..].to_vec(),
                    sys.n_owned,
                    exchange,
                    comm.clone(),
                ),
            };
            let par_res = crate::par_complex_solver::par_solve_complex_pcg(
                &sys.a,
                &sys.b,
                &mut xv,
                &|r, ri, z, zi| {
                    z.copy_from_slice(r);
                    zi.copy_from_slice(ri);
                },
                &cfg,
            )
            .expect("parallel complex CG");
            let mut smax = 0.0_f64;
            for r in 0..sys.n_owned {
                let sr = g2s[&og[r]];
                smax = smax
                    .max((xv.re.as_slice()[r] - xs_ser[sr]).abs())
                    .max((xv.im.as_slice()[r] - xs_ser[n_ser + sr]).abs());
            }
            // Recover + DPG residual (C++ reference prints 1.374e+00 for this
            // configuration).
            let n_own = sys.n_owned;
            let mut x_owned = vec![0.0_f64; 2 * n_own];
            x_owned[..n_own].copy_from_slice(&xv.re.as_slice()[..n_own]);
            x_owned[n_own..].copy_from_slice(&xv.im.as_slice()[..n_own]);
            let x_full = a.recover_fem_solution(&x_owned);
            let residual = a.global_residual_norm(&x_full);
            // Recovered vector vs the serial solution (identity at np1).
            let mut rmax = 0.0_f64;
            for s in 0..n_ser {
                rmax = rmax
                    .max((x_full[s] - xs_ser[s]).abs())
                    .max((x_full[n_ser + s] - xs_ser[n_ser + s]).abs());
            }
            // Independent L2 error of the p block evaluated from the **serial**
            // solution (C++ reference prints 8.008e-01 for this configuration).
            let mesh_ser = ser.mesh();
            let base_p = ser.trial_offsets()[ps_ser];
            let et = mesh_ser.element_type(0);
            let (qpts, qwts) = fem_assembly::dpg::dpg_basis::vol_quadrature(et, 3);
            let simp = matches!(et, fem_mesh::ElementType::Tri3);
            let n_el = mesh_ser.n_elements() as u32;
            let mut l2p = 0.0_f64;
            for e in 0..n_el {
                let ph = xs_ser[base_p + e as usize];
                let pi_ = xs_ser[n_ser + base_p + e as usize];
                for (qi, xi) in qpts.iter().enumerate() {
                    let (det, xp) = if simp {
                        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(
                            mesh_ser,
                            mesh_ser.element_nodes(e),
                        );
                        (tr.det_j().abs(), tr.map_to_physical(xi))
                    } else {
                        let geo = fem_assembly::vector_assembler::geo_ref_elem_from_mesh(
                            mesh_ser, e,
                        )
                        .expect("geo");
                        let gnodes = mesh_ser.geometry_nodes(e).to_vec();
                        let (_j, det, xp) = fem_assembly::vector_assembler::isoparametric_jacobian(
                            mesh_ser, &gnodes, geo.as_ref(), xi, 2,
                        );
                        (det.abs(), xp)
                    };
                    let s = beta * (xp[0] + xp[1]);
                    l2p += qwts[qi]
                        * det
                        * ((ph - s.cos()).powi(2) + (pi_ - s.sin()).powi(2));
                }
            }
            *out2.lock().unwrap() = Some(format!(
                "entries={ncmp} maxabs={maxabs:.3e} bmax={bmax:.3e} x0max={x0max:.3e} \
                 smax={smax:.3e} (ser {} it, par {} it) residual={residual:.6e} l2p={:.6e} \
                 rmax={rmax:.3e}",
                ser_res.iterations,
                par_res.iterations,
                l2p.sqrt()
            ));
            assert!(ncmp > 0, "no entries compared");
            assert!(
                maxabs < 1e-11,
                "np1 P^H A P differs from the serial system (maxabs={maxabs:.3e})"
            );
            assert!(bmax < 1e-11, "np1 P^H b differs from the serial b (bmax={bmax:.3e})");
            assert!(x0max < 1e-14, "np1 initial guess differs (x0max={x0max:.3e})");
            assert!(
                smax < 1e-8,
                "np1 solution differs between serial CG and parallel complex CG (smax={smax:.3e})"
            );
            // Recovered (ghost-lifted) vector vs the serial solution — pins the
            // complex `recover_fem_solution`, which dropped the owned imaginary
            // segment before the r35 fix (prob-0 L2 came out 1.171 instead of
            // 8.008e-01).
            assert!(rmax < 1e-10, "np1 recovered solution differs (rmax={rmax:.3e})");
        });
        let msg = out.lock().unwrap().clone();
        if let Some(ref m) = msg {
            println!("complex_np1_formed_system_matches_serial: {m}");
        }
        assert!(msg.is_some(), "rank 0 must report: {msg:?}");
    }
}
