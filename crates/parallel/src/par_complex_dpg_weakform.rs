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
//! # Status (D172, partial)
//!
//! * 2-D trace numbering (H1 + face-discontinuous) and the complex
//!   `Pᴴ A P`/`Pᴴ b` split: implemented (the 3-D H1/ND trace numbering
//!   panics with a clear message, same as the real form).
//! * Static condensation: forwarded to the serial form for assembly, but
//!   [`Self::form_linear_system`] requires the **uncondensed** system (the
//!   C++ `pacoustics` default) — the condensed parallel path is not wired.
//! * `-pref`/`-pmg`/`Update()`: not implemented (same gaps as the real form).

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

    /// `FormLinearSystem(ess_tdof_list, x, A, X, B)` for the **uncondensed**
    /// system (the `pacoustics` default).
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
    /// prescribed complex value on the RHS.
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
        assert!(
            !self.condensed,
            "ParComplexDPGWeakForm::form_linear_system: the statically condensed parallel \
             system is not wired yet — run with static condensation disabled (the C++ \
             pacoustics default)"
        );
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
    pub fn recover_fem_solution(&self, x_owned: &[f64]) -> Vec<f64> {
        assert!(self.built, "assemble() must run before recover_fem_solution()");
        let n_compact = self.n_owned + self.n_ghost;
        let half = n_compact;
        let mut data = vec![0.0_f64; 2 * half];
        let nr = self.n_owned.min(x_owned.len());
        data[..nr].copy_from_slice(&x_owned[..nr]);
        if self.n_ghost > 0 {
            self.ghost_exchange.forward(&self.comm, &mut data[..half]);
            self.ghost_exchange.forward(&self.comm, &mut data[half..]);
        }
        let n_target = self.sys_global.len();
        let mut u = vec![0.0_f64; 2 * n_target];
        for s in 0..n_target {
            let p = self.perm[s];
            if p != INACTIVE {
                u[s] = data[p as usize];
                u[n_target + s] = data[half + p as usize];
            }
        }
        u
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
}
