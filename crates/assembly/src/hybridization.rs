//! Hybridization — 1:1 port of MFEM `fem/hybridization.hpp/.cpp`.
//!
//! Hybridization solves the assembled system `A x = b` through the Schur
//! system of the *trace* (constraint) space.  With `Â` the block-diagonal
//! matrix of element matrices and `C` the constraint matrix (whose kernel
//! is the image of the conforming assembler, i.e. `C` encodes the
//! inter-element continuity of the normal/tangential traces), the system
//! is solved as
//!
//! ```text
//!   H λ = C Â⁻¹ Rᵀ b          with  H = C Â⁻¹ Cᵀ      (ReduceRHS + solve)
//!   x   = R Â⁻¹ (Rᵀ b − Cᵀ λ)                          (ComputeSolution)
//! ```
//!
//! MFEM assembles `H` element-by-element from the Schur complements of the
//! free element blocks: for element `e`, the "free" hat dofs split into
//! *internal* (no `C` row) and *boundary* (carries a `C` row) dofs, and the
//! block contribution is `Hᵉ = Cᵦ Sᵦ⁻¹ Cᵦᵀ` with
//! `Sᵦ = Aᵦᵦ − Aᵦᵢ Aᵢᵢ⁻¹ Aᵢᵦ`.
//!
//! # C++ → Rust mapping
//!
//! | MFEM (`fem/hybridization.cpp`)          | Rust (this module) |
//! |-----------------------------------------|--------------------|
//! | `Hybridization(fespace, c_fespace)`     | [`Hybridization::new`] (trace space + integrator kinds play the role of `c_fespace` + `c_bfi`) |
//! | `Init(ess_tdof_list)`                   | [`Hybridization::init`] (hat offsets, `ConstructC`, hat-dof markers) |
//! | `ConstructC()`                          | [`construct_ct`] — interior-face assembly of the constraint matrix `Cᵀ` |
//! | `AssembleMatrix(el, A)`                 | [`Hybridization::assemble_matrix`] |
//! | `Finalize()` / `ComputeH()`             | [`Hybridization::finalize`] |
//! | `GetMatrix()`                           | [`Hybridization::get_matrix`] |
//! | `ReduceRHS(b, b_r)`                     | [`Hybridization::reduce_rhs`] |
//! | `ComputeSolution(b, sol_r, sol)`        | [`Hybridization::compute_solution`] (`MultAfInv` modes 0/1) |
//! | `Reset()`                               | [`Hybridization::reset`] |
//! | `AddBdrConstraintIntegrator` / `AssembleBdrMatrix` | **not ported** — ex4's hybridization route imposes boundary conditions through the essential-dof list instead |
//! | `EnableDeviceExecution` / `HybridizationExtension`, MPI (`pC`, `P_pc`, non-conforming meshes) | **not ported** (serial, host only) |
//!
//! The trace space is selected with [`TraceSpaceKind`] — MFEM's
//! `Hybridization` takes an arbitrary `c_fespace`; MFEM ex4 (Darcy) uses
//! `DG_Interface_FECollection(order−1)` with `NormalTraceJumpIntegrator`,
//! and an `H1_FECollection` `c_fes` is the other common choice (its
//! `GetFaceVDofs` on a 2-D mesh returns the two endpoint vertex dofs plus
//! the edge-interior dofs).  Both are provided, together with a 2-D
//! tangential-trace jump for H(curl) spaces (MFEM 4.10 ships no built-in
//! H(curl) jump integrator; see [`trace`] module docs).

mod trace;

pub use trace::{ConstraintIntegratorKind, TraceSpaceKind};

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use crate::vector_integrator::VectorBilinearIntegrator;
use crate::vector_assembler::accumulate_vector_bilinear_element;

/// Hybridization of a bilinear form (MFEM `Hybridization`).
///
/// Typical usage (ex4 Darcy route):
///
/// ```text
/// let mut hyb = Hybridization::new(TraceSpaceKind::FaceDG { order: rt_order },
///                                  ConstraintIntegratorKind::NormalTraceJump);
/// hyb.init(&space, &ess_dofs);
/// for e in 0..space.n_elems() { hyb.assemble_matrix(e, &elmat(e)); }
/// hyb.finalize();
/// let h = hyb.get_matrix().unwrap();      // SPD Schur system
/// let b_r = hyb.reduce_rhs(&b);           // b = BC-eliminated RHS
/// // ... solve h λ = b_r ...
/// hyb.compute_solution(&b, &λ, &mut x);   // x must hold the BC values
/// ```
pub struct Hybridization {
    trace: TraceSpaceKind,
    integrator: ConstraintIntegratorKind,

    // ── set by init() ──
    /// Number of elements.
    ne: usize,
    /// Per-element hat-dof offsets, `NE + 1` entries (MFEM `hat_offsets`).
    hat_offsets: Vec<usize>,
    /// Flattened element vdofs of the discontinuous ("hat") space.
    hat_vdofs: Vec<u32>,
    /// Per-hat-dof marker: `0` internal, `-1` boundary, `1` essential
    /// (MFEM `hat_dofs_marker`).
    hat_marker: Vec<i8>,
    /// The constraint matrix `Cᵀ` (`num_hat_dofs × n_trace`).
    ct: Option<CsrMatrix<f64>>,
    /// Size of the trace (constraint) space.
    n_trace: usize,

    // ── set by assemble_matrix() ──
    /// Per-element free block `[i…; b…] × [i…; b…]`, row-major
    /// (MFEM `Af_data` / `Af_offsets`).
    af_blocks: Vec<Vec<f64>>,

    // ── set by finalize() ──
    /// The hybridized Schur system `H` (MFEM `H`).
    h: Option<CsrMatrix<f64>>,
    /// Per-element block LU factors (built by `finalize`, consumed by the
    /// `MultAfInv` equivalents).
    factors: Vec<ElementFactor>,
}

/// Block LU factorization of one element's free block (the state MFEM
/// keeps inside `Af_data`/`Af_ipiv` after `ComputeH`'s `BlockFactor`).
struct ElementFactor {
    /// Number of internal ("i") dofs.
    i_len: usize,
    /// Number of boundary ("b") dofs.
    b_len: usize,
    /// LU of `A_ii` (`i_len × i_len`, row-major) and pivots.
    lu_ii: Vec<f64>,
    ipiv_ii: Vec<i32>,
    /// `A_ib` (`i_len × b_len`), needed for the mode-1 back solve.
    a_ib: Vec<f64>,
    /// `A_bi` (`b_len × i_len`).
    a_bi: Vec<f64>,
    /// LU of the Schur complement `S_b = A_bb − A_bi A_ii⁻¹ A_ib`
    /// (`b_len × b_len`, row-major) and pivots (MFEM `LU_bb`).
    lu_bb: Vec<f64>,
    ipiv_bb: Vec<i32>,
}

impl Hybridization {
    /// Create a hybridization for the given trace space and constraint
    /// integrator (MFEM constructor + `SetConstraintIntegrator`).
    pub fn new(trace: TraceSpaceKind, integrator: ConstraintIntegratorKind) -> Self {
        Hybridization {
            trace,
            integrator,
            ne: 0,
            hat_offsets: Vec::new(),
            hat_vdofs: Vec::new(),
            hat_marker: Vec::new(),
            ct: None,
            n_trace: 0,
            af_blocks: Vec::new(),
            h: None,
            factors: Vec::new(),
        }
    }

    /// Prepare the hybridization for assembly (MFEM `Init`).
    ///
    /// Counts the hat dofs of the discontinuous version of `fespace`,
    /// constructs the constraint matrix `Cᵀ` (`ConstructC`) and classifies
    /// the hat dofs into *essential* (depend only on essential true dofs),
    /// *internal* (free, no `C` row) and *boundary* (free, carries a `C`
    /// row).  Idempotent: a second call is a no-op, like MFEM.
    pub fn init<M, S>(&mut self, fespace: &S, ess_tdof_list: &[u32])
    where
        M: MeshTopology,
        S: FESpace<Mesh = M>,
    {
        if self.ct.is_some() {
            return;
        }

        // Hat space layout: concatenated element vdofs.
        let ne = fespace.mesh().n_elements();
        let mut hat_offsets = Vec::with_capacity(ne + 1);
        hat_offsets.push(0);
        let mut hat_vdofs = Vec::new();
        for e in 0..ne as u32 {
            let dofs = fespace.element_dofs(e);
            hat_vdofs.extend_from_slice(dofs);
            hat_offsets.push(hat_vdofs.len());
        }
        self.ne = ne;
        self.hat_offsets = hat_offsets;
        self.hat_vdofs = hat_vdofs;

        // Constraint matrix Cᵀ (ConstructC).
        let (ct, n_trace) =
            trace::construct_ct(fespace, &self.hat_offsets, self.trace, self.integrator);
        self.n_trace = n_trace;
        self.ct = Some(ct);

        // Essential hat dofs: those whose vdof is an essential true dof
        // (conforming spaces: vdof == tdof).
        let n_hat = self.hat_vdofs.len();
        let mut ess = vec![false; fespace.n_dofs()];
        for &d in ess_tdof_list {
            ess[d as usize] = true;
        }
        let mut marker = vec![0_i8; n_hat];
        for (j, &v) in self.hat_vdofs.iter().enumerate() {
            if ess[v as usize] {
                marker[j] = 1;
            }
        }

        // Free hat dofs with a non-empty Cᵀ row are "boundary" (-1).
        let ct = self.ct.as_ref().unwrap();
        for i in 0..n_hat {
            if marker[i] != 1 && ct.row_ptr[i + 1] > ct.row_ptr[i] {
                marker[i] = -1;
            }
        }
        self.hat_marker = marker;

        self.af_blocks = vec![Vec::new(); ne];
    }

    /// Assemble the element matrix `a` (`n_e × n_e`, row-major, ordering
    /// matches `fespace.element_dofs(el)`) into the hybridized system
    /// (MFEM `AssembleMatrix`).  Only the *free* sub-block (internal +
    /// boundary rows/columns) is stored.
    pub fn assemble_matrix(&mut self, el: usize, a: &[f64]) {
        assert!(
            self.ct.is_some(),
            "Hybridization::assemble_matrix: init() must be called first"
        );
        let base = self.hat_offsets[el];
        let n_e = self.hat_offsets[el + 1] - base;
        assert_eq!(
            a.len(),
            n_e * n_e,
            "Hybridization::assemble_matrix: element matrix size mismatch"
        );

        // GetIBDofs: local indices of the i-dofs and b-dofs.
        let mut i_dofs = Vec::new();
        let mut b_dofs = Vec::new();
        for j in 0..n_e {
            match self.hat_marker[base + j] {
                0 => i_dofs.push(j),
                -1 => b_dofs.push(j),
                _ => {}
            }
        }
        let i_len = i_dofs.len();
        let b_len = b_dofs.len();
        let f = i_len + b_len;

        let mut blk = vec![0.0_f64; f * f];
        for (rf, &r) in i_dofs.iter().chain(b_dofs.iter()).enumerate() {
            for (cf, &c) in i_dofs.iter().chain(b_dofs.iter()).enumerate() {
                blk[rf * f + cf] = a[r * n_e + c];
            }
        }
        self.af_blocks[el] = blk;
    }

    /// Finalize the construction of the hybridized matrix (MFEM `Finalize`
    /// + `ComputeH`): block-factors every element free block, computes the
    /// per-element Schur contributions `Hᵉ = Cᵦ Sᵦ⁻¹ Cᵦᵀ` and assembles
    /// them into `H`.  Idempotent.
    pub fn finalize(&mut self) {
        if self.h.is_some() {
            return;
        }
        let ct = self
            .ct
            .as_ref()
            .expect("Hybridization::finalize: init() must be called first");
        let ne = self.ne;

        let mut factors = Vec::with_capacity(ne);
        let mut h = CooMatrix::<f64>::new(self.n_trace, self.n_trace);

        // MFEM's c_dof_marker / c_mark_start generation trick, expressed
        // with a generation counter.
        let mut c_dof_mark = vec![-1_i64; self.n_trace];
        let mut generation: i64 = 0;

        for el in 0..ne {
            let base = self.hat_offsets[el];
            let n_e = self.hat_offsets[el + 1] - base;
            let n_free = (0..n_e)
                .filter(|&j| self.hat_marker[base + j] != 1)
                .count();
            if self.af_blocks[el].is_empty() {
                if n_free > 0 {
                    panic!(
                        "Hybridization::finalize: assemble_matrix() was not \
                         called for element {el}"
                    );
                }
                // Fully-essential element: no free block, no H contribution.
                factors.push(ElementFactor {
                    i_len: 0,
                    b_len: 0,
                    lu_ii: Vec::new(),
                    ipiv_ii: Vec::new(),
                    a_ib: Vec::new(),
                    a_bi: Vec::new(),
                    lu_bb: Vec::new(),
                    ipiv_bb: Vec::new(),
                });
                continue;
            }
            let blk = &self.af_blocks[el];

            // GetIBDofs on the local block — `i_loc`/`b_loc` hold the
            // original local dof indices (used to address the full element
            // numbering, e.g. Cᵀ rows at `base + loc`), while the free
            // block produced by `assemble_matrix` is indexed by the
            // compressed position (internal dofs first, then boundary
            // dofs, in marker-scan order).
            let mut i_loc = Vec::new();
            let mut b_loc = Vec::new();
            for j in 0..n_e {
                match self.hat_marker[base + j] {
                    0 => i_loc.push(j),
                    -1 => b_loc.push(j),
                    _ => {}
                }
            }
            let i_dofs: Vec<usize> = (0..i_loc.len()).collect();
            let b_dofs: Vec<usize> = (i_loc.len()..i_loc.len() + b_loc.len()).collect();
            let i_len = i_dofs.len();
            let b_len = b_dofs.len();
            let f = i_len + b_len;

            // Extract A_ii, A_ib, A_bi, A_bb from the free block.
            let mut lu_ii = vec![0.0_f64; i_len * i_len];
            for (r, &ri) in i_dofs.iter().enumerate() {
                for (c, &ci) in i_dofs.iter().enumerate() {
                    lu_ii[r * i_len + c] = blk[ri * f + ci];
                }
            }
            let mut a_ib = vec![0.0_f64; i_len * b_len];
            for (r, &ri) in i_dofs.iter().enumerate() {
                for (c, &bi) in b_dofs.iter().enumerate() {
                    a_ib[r * b_len + c] = blk[ri * f + bi];
                }
            }
            let mut a_bi = vec![0.0_f64; b_len * i_len];
            for (r, &br_) in b_dofs.iter().enumerate() {
                for (c, &ci) in i_dofs.iter().enumerate() {
                    a_bi[r * i_len + c] = blk[br_ * f + ci];
                }
            }
            let mut a_bb = vec![0.0_f64; b_len * b_len];
            for (r, &br_) in b_dofs.iter().enumerate() {
                for (c, &bc) in b_dofs.iter().enumerate() {
                    a_bb[r * b_len + c] = blk[br_ * f + bc];
                }
            }

            // LU_ii.Factor + BlockFactor: X = A_ii⁻¹ A_ib, S_b = A_bb − A_bi X.
            let mut ipiv_ii = vec![0_i32; i_len];
            lu_factor(&mut lu_ii, i_len, &mut ipiv_ii)
                .unwrap_or_else(|| panic!("Hybridization: singular A_ii in element {el}"));
            // The Schur complement uses the transformed block; `a_ib` stays
            // untransformed for the mode-1 back solve
            // (x_i = A_ii⁻¹ (el_i − A_ib x_b)).
            let mut x_ib = a_ib.clone();
            lu_solve(&lu_ii, &ipiv_ii, i_len, &mut x_ib, b_len); // x_ib ← A_ii⁻¹ A_ib
            let mut s_bb = a_bb;
            for r in 0..b_len {
                for c in 0..b_len {
                    let mut acc = 0.0;
                    for k in 0..i_len {
                        acc += a_bi[r * i_len + k] * x_ib[k * b_len + c];
                    }
                    s_bb[r * b_len + c] -= acc;
                }
            }
            let mut ipiv_bb = vec![0_i32; b_len];
            lu_factor(&mut s_bb, b_len, &mut ipiv_bb)
                .unwrap_or_else(|| panic!("Hybridization: singular Schur block in element {el}"));

            // Extract Cb_t (b_len × c_len) from the Cᵀ rows of the b-dofs.
            let mut c_dofs: Vec<usize> = Vec::new();
            generation += 1;
            for &bl in &b_loc {
                let row = base + bl; // global hat dof index
                for p in ct.row_ptr[row]..ct.row_ptr[row + 1] {
                    let cd = ct.col_idx[p] as usize;
                    if c_dof_mark[cd] < generation {
                        c_dof_mark[cd] = generation;
                        c_dofs.push(cd);
                    }
                }
            }
            let c_len = c_dofs.len();
            let mut col_of = vec![0_usize; self.n_trace];
            for (loc, &g) in c_dofs.iter().enumerate() {
                col_of[g] = loc;
            }
            let mut cb_t = vec![0.0_f64; b_len * c_len];
            for (r, &bl) in b_loc.iter().enumerate() {
                let row = base + bl;
                for p in ct.row_ptr[row]..ct.row_ptr[row + 1] {
                    let cd = ct.col_idx[p] as usize;
                    cb_t[r * c_len + col_of[cd]] = ct.values[p];
                }
            }

            // V = S_b⁻¹ Cb_t,  Hb = Cb_tᵀ V  (assembled into H).
            let cb_t_orig = cb_t.clone(); // MFEM: Sb_inv_Cb_t = Cb_t
            lu_solve(&s_bb, &ipiv_bb, b_len, &mut cb_t, c_len);
            for p in 0..c_len {
                for q in 0..c_len {
                    let mut acc = 0.0;
                    for k in 0..b_len {
                        acc += cb_t_orig[k * c_len + p] * cb_t[k * c_len + q];
                    }
                    // Hb = Cb_tᵀ · V: entry (p, q) = Σ_k Cb_t(k,p)·V(k,q)
                    if acc != 0.0 {
                        h.add(c_dofs[p], c_dofs[q], acc);
                    }
                }
            }

            factors.push(ElementFactor {
                i_len,
                b_len,
                lu_ii,
                ipiv_ii,
                a_ib,
                a_bi,
                lu_bb: s_bb,
                ipiv_bb,
            });
        }

        self.factors = factors;
        let mut h = h.into_csr();
        // MFEM `H->Finalize(skip_zeros, fix_empty_rows = true)`: trace dofs
        // touched by no element get an identity entry so the matrix has no
        // empty rows (relevant for the shared-vertex H1 trace space).
        for r in 0..h.nrows {
            if h.row_ptr[r + 1] == h.row_ptr[r] {
                let mut coo = CooMatrix::<f64>::new(h.nrows, h.ncols);
                for i in 0..h.nrows {
                    for p in h.row_ptr[i]..h.row_ptr[i + 1] {
                        coo.add(i, h.col_idx[p] as usize, h.values[p]);
                    }
                    if i == r {
                        coo.add(i, i, 1.0);
                    }
                }
                h = coo.into_csr();
                break;
            }
        }
        self.h = Some(h);
    }

    /// The hybridized Schur system `H` (MFEM `GetMatrix`), available after
    /// [`Self::finalize`].
    pub fn get_matrix(&self) -> Option<&CsrMatrix<f64>> {
        self.h.as_ref()
    }

    /// Number of trace-space (constraint) unknowns.
    pub fn n_trace_dofs(&self) -> usize {
        self.n_trace
    }

    /// Number of hat dofs of the discontinuous space.
    pub fn n_hat_dofs(&self) -> usize {
        self.hat_vdofs.len()
    }

    /// Perform the reduction of the right-hand side `b` to the
    /// hybridized system: `b_r = C (Âf)⁻¹ Rᵀ b` (MFEM `ReduceRHS`).
    ///
    /// `b` must be the *BC-eliminated* right-hand side of the original
    /// system (essential contributions already subtracted, as done by the
    /// standard `form_linear_system` step).
    pub fn reduce_rhs(&self, b: &[f64]) -> Vec<f64> {
        let ct = self.ct.as_ref().expect("reduce_rhs: init() first");
        assert!(
            !self.factors.is_empty(),
            "reduce_rhs: finalize() must be called first"
        );
        // bf = Af⁻¹ Rᵀ b  (mode 0).
        let bf = self.mult_af_inv(b, None);
        // b_r = Cᵀᵀ bf = C bf  (Cᵀ stored, so this is the transpose SpMV).
        let mut b_r = vec![0.0_f64; self.n_trace];
        for i in 0..ct.nrows {
            let bi = bf[i];
            if bi == 0.0 {
                continue;
            }
            for p in ct.row_ptr[i]..ct.row_ptr[i + 1] {
                b_r[ct.col_idx[p] as usize] += ct.values[p] * bi;
            }
        }
        b_r
    }

    /// Reconstruct the solution of the original system from the solution
    /// `sol_r` of the hybridized system (MFEM `ComputeSolution`).
    ///
    /// `sol` must already hold the correct essential boundary values; the
    /// non-essential dofs are overwritten with `x = R Âf⁻¹ (Rᵀ b − Cᵀ λ)`.
    pub fn compute_solution(&self, b: &[f64], sol_r: &[f64], sol: &mut [f64]) {
        let ct = self.ct.as_ref().expect("compute_solution: init() first");
        assert!(
            !self.factors.is_empty(),
            "compute_solution: finalize() must be called first"
        );
        // ct_l = Cᵀ λ  (Cᵀ is num_hat × n_trace, λ ∈ R^n_trace).
        let mut ct_l = vec![0.0_f64; self.hat_vdofs.len()];
        for i in 0..ct.nrows {
            let mut acc = 0.0;
            for p in ct.row_ptr[i]..ct.row_ptr[i + 1] {
                acc += ct.values[p] * sol_r[ct.col_idx[p] as usize];
            }
            ct_l[i] = acc;
        }
        // bf = Af⁻¹ (Rᵀ b − Cᵀ λ)  (mode 1).
        let bf = self.mult_af_inv(b, Some(&ct_l));

        // sol = R bf: scatter, skipping essential hat dofs.
        for (j, &v) in self.hat_vdofs.iter().enumerate() {
            if self.hat_marker[j] == 1 {
                continue;
            }
            sol[v as usize] = bf[j];
        }
    }

    /// Destroy the hybridized matrix while preserving the constraint
    /// matrix and the essential-dof classification (MFEM `Reset`).
    pub fn reset(&mut self) {
        self.h = None;
        self.factors.clear();
    }

    /// MFEM `MultAfInv`: per-element block solve with the factored free
    /// blocks.  `mode 0` (`ct_lambda = None`) computes `bf = Af⁻¹ Rᵀ b`
    /// with the internal part zero; `mode 1` computes
    /// `bf = Af⁻¹ (Rᵀ b − Cᵀ λ)` including the back solve.
    fn mult_af_inv(&self, b: &[f64], ct_lambda: Option<&[f64]>) -> Vec<f64> {
        let mode1 = ct_lambda.is_some();
        let n_hat = self.hat_vdofs.len();
        let mut bf = vec![0.0_f64; n_hat];
        // MFEM deduplicates shared vdofs when gathering the element loads:
        // the first element that touches a vdof reads b, later copies read
        // zero (`vdof_marker` in MultAfInv).
        let mut vdof_seen = vec![false; b.len()];

        for el in 0..self.ne {
            let base = self.hat_offsets[el];
            let n_e = self.hat_offsets[el + 1] - base;
            let fac = &self.factors[el];
            let i_len = fac.i_len;
            let b_len = fac.b_len;

            // el_vals = Rᵀ b at this element (shared vdofs deduplicated).
            let mut el_vals = vec![0.0_f64; n_e];
            for (k, j) in (base..base + n_e).enumerate() {
                let v = self.hat_vdofs[j] as usize;
                if !vdof_seen[v] {
                    vdof_seen[v] = true;
                    el_vals[k] = b[v];
                }
            }
            if mode1 {
                for (k, j) in (base..base + n_e).enumerate() {
                    el_vals[k] -= ct_lambda.unwrap()[j];
                }
            }

            // Local i/b index lists (must match finalize()).
            let mut i_dofs = Vec::with_capacity(i_len);
            let mut b_dofs = Vec::with_capacity(b_len);
            for j in 0..n_e {
                match self.hat_marker[base + j] {
                    0 => i_dofs.push(j),
                    -1 => b_dofs.push(j),
                    _ => {}
                }
            }

            let mut i_vals = vec![0.0_f64; i_len];
            for (k, &j) in i_dofs.iter().enumerate() {
                i_vals[k] = el_vals[j];
            }
            let mut b_vals = vec![0.0_f64; b_len];
            for (k, &j) in b_dofs.iter().enumerate() {
                b_vals[k] = el_vals[j];
            }

            // t = A_ii⁻¹ i_vals;  b_vals −= A_bi t;  b_vals = S_b⁻¹ b_vals.
            lu_solve(&fac.lu_ii, &fac.ipiv_ii, i_len, &mut i_vals, 1);
            for r in 0..b_len {
                let mut acc = 0.0;
                for k in 0..i_len {
                    acc += fac.a_bi[r * i_len + k] * i_vals[k];
                }
                b_vals[r] -= acc;
            }
            lu_solve(&fac.lu_bb, &fac.ipiv_bb, b_len, &mut b_vals, 1);

            if mode1 {
                // i_vals = A_ii⁻¹ (i_vals − A_ib b_vals)  (BlockBackSolve).
                for r in 0..i_len {
                    let mut acc = 0.0;
                    for k in 0..b_len {
                        acc += fac.a_ib[r * b_len + k] * b_vals[k];
                    }
                    i_vals[r] -= acc;
                }
                lu_solve(&fac.lu_ii, &fac.ipiv_ii, i_len, &mut i_vals, 1);
                for (k, &j) in i_dofs.iter().enumerate() {
                    bf[base + j] = i_vals[k];
                }
            }
            for (k, &j) in b_dofs.iter().enumerate() {
                bf[base + j] = b_vals[k];
            }
        }
        bf
    }
}

// ─── Dense LU (row-major, partial pivoting) ──────────────────────────────

/// In-place LU factorization with partial pivoting.  Returns `None` for a
/// (numerically) singular matrix.
fn lu_factor(a: &mut [f64], n: usize, ipiv: &mut [i32]) -> Option<()> {
    for k in 0..n {
        // Pivot search.
        let mut p = k;
        let mut maxv = a[k * n + k].abs();
        for i in k + 1..n {
            let v = a[i * n + k].abs();
            if v > maxv {
                maxv = v;
                p = i;
            }
        }
        if maxv < 1e-300 {
            return None;
        }
        ipiv[k] = p as i32;
        if p != k {
            for c in 0..n {
                a.swap(k * n + c, p * n + c);
            }
        }
        let piv = a[k * n + k];
        for i in k + 1..n {
            let m = a[i * n + k] / piv;
            a[i * n + k] = m;
            for c in k + 1..n {
                a[i * n + c] -= m * a[k * n + c];
            }
        }
    }
    Some(())
}

/// Solve `A X = B` in place for `B` (`n × nrhs`, row-major) given the LU
/// factors of `A`.
fn lu_solve(lu: &[f64], ipiv: &[i32], n: usize, b: &mut [f64], nrhs: usize) {
    // Apply row swaps.
    for k in 0..n {
        let p = ipiv[k] as usize;
        if p != k {
            for j in 0..nrhs {
                b.swap(k * nrhs + j, p * nrhs + j);
            }
        }
    }
    // Forward substitution (unit lower triangular).
    for col in 0..nrhs {
        for i in 1..n {
            let mut acc = b[i * nrhs + col];
            for k in 0..i {
                acc -= lu[i * n + k] * b[k * nrhs + col];
            }
            b[i * nrhs + col] = acc;
        }
    }
    // Back substitution.
    for col in 0..nrhs {
        for i in (0..n).rev() {
            let mut acc = b[i * nrhs + col];
            for k in i + 1..n {
                acc -= lu[i * n + k] * b[k * nrhs + col];
            }
            b[i * nrhs + col] = acc / lu[i * n + i];
        }
    }
}

// ─── Convenience: per-element matrix extraction ──────────────────────────

/// Compute the dense element matrix (`n × n`, row-major, ordering matches
/// `space.element_dofs(e)`) produced by `VectorAssembler`'s integrators.
///
/// This is the element-level input required by
/// [`Hybridization::assemble_matrix`]; MFEM's `BilinearForm` hands the same
/// local matrices to `Hybridization::AssembleMatrix` during `Assemble()`.
pub fn vector_element_matrix<S: FESpace>(
    space: &S,
    e: u32,
    integrators: &[&dyn VectorBilinearIntegrator],
    quad_order: u8,
) -> Vec<f64> {
    let mut coo = CooMatrix::<f64>::new(space.n_dofs(), space.n_dofs());
    accumulate_vector_bilinear_element(space, e, integrators, quad_order, &mut coo);
    let csr = coo.into_csr();
    let dofs = space.element_dofs(e);
    let n = dofs.len();
    let mut local: Vec<usize> = vec![usize::MAX; space.n_dofs()];
    for (i, &d) in dofs.iter().enumerate() {
        local[d as usize] = i;
    }
    let mut mat = vec![0.0_f64; n * n];
    for r in 0..n {
        let gr = dofs[r] as usize;
        for p in csr.row_ptr[gr]..csr.row_ptr[gr + 1] {
            let gc = csr.col_idx[p] as usize;
            let c = local[gc];
            if c != usize::MAX {
                mat[r * n + c] += csr.values[p];
            }
        }
    }
    mat
}

#[cfg(test)]
mod tests;
