//! High-level Form abstractions (MFEM-style BilinearForm / LinearForm).
//!
//! These wrapper types hold a space + integrator list and provide lazy
//! assembly, matrix/vector caching, and `Mult` / `MultTranspose` operations.
//!
//! # Example
//! ```rust,ignore
//! use fem_assembly::form::{BilinearForm, LinearForm};
//! use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
//!
//! let stiffness = BilinearForm::new(&space)
//!     .add_integrator(DiffusionIntegrator { kappa: 1.0 })
//!     .assemble(2);
//! let rhs = LinearForm::new(&space)
//!     .add_integrator(DomainSourceIntegrator::new(f))
//!     .assemble(3);
//! ```

use fem_core::types::DofId;
use fem_linalg::CsrMatrix;
use fem_space::fe_space::FESpace;

use crate::assembler::{Assembler, CondensedLinearSystem};
use crate::integrator::{BilinearIntegrator, LinearIntegrator};
use crate::static_cond::condense_rhs;
use crate::vector_assembler::VectorAssembler;
use crate::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator};

// ─── MFEM FormLinearSystem (D706 serial shape core) ───────────────────────────

/// MFEM `DiagonalPolicy` (the subset used by `FormLinearSystem` callers) —
/// the strategy parameter of [`eliminate_ess_tdofs`] and
/// [`BilinearForm::form_linear_system`].
///
/// Serial mirror of `fem_parallel::ElimPolicy` (the round-68 parallel D706
/// pair), so serial and parallel call sites name the same two elimination
/// shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElimPolicy {
    /// `DIAG_KEEP`: symmetric row/col elimination, **diagonal preserved** —
    /// keeps the operator's scaling (CG-friendly).  MFEM 4.10's
    /// `BilinearForm` default (`bilinearform.hpp:153`, `diag_policy =
    /// DIAG_KEEP`).
    DiagKeep,
    /// `DIAG_ONE`: row and column zeroed, diagonal set to 1.
    DiagOne,
}

/// **MFEM `BilinearForm::FormLinearSystem(ess_tdof_list, x, b, A, X, B)`**
/// shape core on a bare assembled matrix (D706 serial analog of the round-68
/// `ParVectorAssembler::form_linear_system` /
/// `ParCsrMatrix::eliminate_ess_tdofs`).
///
/// One-stop essential-BC elimination with the **projected solution values**:
/// `ess_tdof_list` (true-dof order, as produced by
/// `fem_space::constraints::boundary_dofs` and friends) selects the essential
/// true dofs; their values are read from `x` — the projection of the
/// solution / initial guess (MFEM's `x.ProjectCoefficient(…)` restricted to
/// the true dofs) — so the values can no longer be decoupled from the
/// indices (the D697 accident surface: hardcoded 0.0 silently homogenized a
/// non-homogeneous PEC boundary).  On return
///   - `a` is the eliminated operator (policy [`ElimPolicy`]),
///   - `b` is the eliminated RHS `B`: interior rows carry the
///     `b − Ae·x` reactions (`EliminateVDofsInRHS`, bilinearform.cpp:1239 —
///     `b -= mat_e·x` over the pre-elimination column entries), ess rows
///     `B(r) = A(r,r)·x(r)` under [`ElimPolicy::DiagKeep`] (via
///     `PartMult`: only the diagonal survives the row) and `B(r) = x(r)`
///     under [`ElimPolicy::DiagOne`],
///   - the returned vector is the true-dof solution vector `X`: a **bitwise
///     copy** of `x` (conforming space: `R = I`, i.e. MFEM's
///     `copy_interior = 1` default), ready as the iterative initial guess.
///
/// The per-dof kernels are the pinned 1:1 ports already used by the
/// hand-rolled call sites — `CsrMatrix::apply_dirichlet_keep_diag` (MFEM
/// `SparseMatrix::EliminateRowCol(rc, sol, rhs, DIAG_KEEP)`,
/// linalg/sparsemat.cpp:1914, true-column reactions per D409/D432) and
/// `CsrMatrix::apply_dirichlet_symmetric` (the `DIAG_ONE` twin) — applied in
/// ess-list order, which is exactly the former two-step hand pattern
/// (`values` pulled from `x`, then a per-dof kernel loop); the eliminated
/// system is therefore reproduced **bitwise** by construction (pinned by
/// `tests/d706_form_linear_system.rs`).
pub fn eliminate_ess_tdofs(
    a: &mut CsrMatrix<f64>,
    ess_tdof_list: &[DofId],
    x: &[f64],
    b: &mut [f64],
    policy: ElimPolicy,
) -> Vec<f64> {
    let vals: Vec<f64> = ess_tdof_list.iter().map(|&d| x[d as usize]).collect();
    match policy {
        ElimPolicy::DiagKeep => {
            fem_space::constraints::apply_dirichlet(a, b, ess_tdof_list, &vals)
        }
        ElimPolicy::DiagOne => {
            fem_space::constraints::apply_dirichlet_diag_one(a, b, ess_tdof_list, &vals)
        }
    }
    // X = R·x; conforming space: R = I → X = x (copy_interior = 1).
    x.to_vec()
}

// ─── BilinearForm (scalar-valued) ─────────────────────────────────────────────

pub struct BilinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn BilinearIntegrator>>,
    cached: Option<CsrMatrix<f64>>,
    /// MFEM `BilinearForm::static_cond` — set by
    /// [`BilinearForm::enable_static_condensation`].
    static_cond: bool,
    /// Quadrature order of the last `assemble()` — the SC path re-enters the
    /// D839-2 condensed core, which takes it explicitly.
    assembled_quad: Option<u8>,
    /// Condensed-system record of the last SC `form_linear_system` (MFEM's
    /// `static_cond` object state: reduced operator plus the per-element Schur
    /// factors consumed by [`BilinearForm::recover_fem_solution`]).
    cond_sys: Option<CondensedLinearSystem>,
}

impl<S: FESpace> BilinearForm<S> {
    pub fn new(space: S) -> Self {
        BilinearForm {
            space,
            integrators: Vec::new(),
            cached: None,
            static_cond: false,
            assembled_quad: None,
            cond_sys: None,
        }
    }
    pub fn add_integrator(mut self, integ: impl BilinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &CsrMatrix<f64> {
        let refs: Vec<&dyn BilinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let mat = Assembler::assemble_bilinear(&self.space, &refs, quad_order);
        self.cached = Some(mat);
        self.assembled_quad = Some(quad_order);
        self.cached.as_ref().unwrap()
    }

    /// **MFEM `BilinearForm::EnableStaticCondensation()`**
    /// (fem/bilinearform.cpp:144, LEGACY full-matrix assembly): after this,
    /// [`BilinearForm::form_linear_system`] reduces the system to the trace
    /// dofs — MFEM `static_cond->ReduceSystem(x, b, X, B, copy_interior)`
    /// (bilinearform.cpp:885-889) — and
    /// [`BilinearForm::recover_fem_solution`] back-substitutes the eliminated
    /// private dofs (`static_cond->ComputeSolution`,
    /// bilinearform.cpp:1000-1004).  The whole MFEM transparency contract:
    /// `-sc` becomes a caller-side switch on the same `FormLinearSystem` /
    /// `RecoverFEMSolution` pair instead of a specialized entry.
    ///
    /// Like MFEM's `ReducesTrueVSize()` guard the switch is a no-op when the
    /// space has no interior ("bubble") dofs: the reduced system then equals
    /// the full system element-for-element.
    pub fn enable_static_condensation(&mut self) {
        self.static_cond = true;
    }

    /// MFEM `BilinearForm::StaticCondensationEnabled()`.
    pub fn static_condensation_enabled(&self) -> bool {
        self.static_cond
    }

    pub fn mat(&self) -> Option<&CsrMatrix<f64>> { self.cached.as_ref() }
    pub fn space(&self) -> &S { &self.space }

    /// Compute `y += A * x` (MFEM's `BilinearForm::AddMult`).
    ///
    /// # Panics
    /// Panics if `assemble()` has not been called first.
    pub fn add_mult(&self, x: &[f64], y: &mut [f64]) {
        let a = self.cached.as_ref().expect("assemble() must be called first");
        a.spmv_add(1.0, x, 1.0, y);
    }

    /// **MFEM `BilinearForm::FormLinearSystem(ess_tdof_list, x, b, A, X, B)`**
    /// (fem/bilinearform.cpp:826, conforming + full-matrix branch) — the
    /// one-stop shape entry on the assembled form: `FormSystemMatrix`
    /// eliminates the ess rows/cols of the cached operator under `policy`,
    /// `EliminateVDofsInRHS` gives the interior rows the `b − Ae·x` reactions
    /// and the ess rows `A(r,r)·x(r)` (DiagKeep) / `x(r)` (DiagOne), and `X`
    /// (returned) is the bitwise copy of the projected `x` — see
    /// [`eliminate_ess_tdofs`] for the full contract.
    ///
    /// # Static condensation (D839-2)
    ///
    /// With [`BilinearForm::enable_static_condensation`] set, the entry takes
    /// MFEM's SC branch instead (bilinearform.cpp:885-889): the element-private
    /// dofs are Schur-eliminated through the round-91 condensed core
    /// ([`Assembler::form_linear_system_condensed`]), the caller-supplied `b`
    /// is reduced via [`condense_rhs`] (`ReduceRHS`), `X` becomes the kept-dof
    /// copy of `x` (`ReduceSolution`, staticcond.cpp:389) and the essential
    /// dofs — which stay in the reduced system exactly as in
    /// `SetEssentialTrueDofs` + `EliminateReducedTrueDofs(policy)`
    /// (bilinearform.cpp:959-962) — are eliminated there under `policy`.
    /// On return
    ///   - `mat()` holds the reduced, ess-eliminated operator (MFEM hands the
    ///     caller the reduced `A` through the `OperatorHandle`),
    ///   - `b[..X.len()]` holds the reduced, ess-eliminated RHS `B` (MFEM's
    ///     out-vector `B`; the trailing entries of `b` are stale),
    ///   - the returned vector is `X`, of length `mat().nrows()`.
    /// Solve the reduced system, then call
    /// [`BilinearForm::recover_fem_solution`].
    ///
    /// # Panics
    /// Panics if `assemble()` has not been called first.
    pub fn form_linear_system(
        &mut self,
        ess_tdof_list: &[DofId],
        x: &[f64],
        b: &mut [f64],
        policy: ElimPolicy,
    ) -> Vec<f64> {
        if self.static_cond {
            let quad = self.assembled_quad.expect("assemble() must be called first");
            let refs: Vec<&dyn BilinearIntegrator> =
                self.integrators.iter().map(|bx| bx.as_ref()).collect();
            // A-side through the round-91 condensed core: per-element Schur
            // elimination of the element-private dofs (ex29 -sc anchor).  The
            // core is entered with an empty linear list — the caller's `b` is
            // reduced separately below, from the same factors.
            let a_full = self.cached.take().expect("assemble() must be called first");
            let mut sys =
                Assembler::form_linear_system_condensed(&self.space, &refs, &[], quad, ess_tdof_list);
            // MFEM `ReduceRHS` (staticcond.cpp:309) on the caller's `b`; also
            // installs the caller's interior loads into the recovery records
            // so `recover_fem_solution` back-substitutes with the same RHS.
            let mut g = condense_rhs(&mut sys, &a_full, b);
            // MFEM `ReduceSolution` (staticcond.cpp:389): X = x at the kept
            // dofs, ascending compaction (conforming space, copy_interior = 1).
            let n_red = sys.reduced.nrows;
            let mut x_red = vec![0.0_f64; n_red];
            for (d, &rid) in sys.reduced_id.iter().enumerate() {
                if rid != u32::MAX {
                    x_red[rid as usize] = x[d];
                }
            }
            // Essential dofs stay in the reduced system and are eliminated
            // there under the caller's policy — MFEM
            // `SetEssentialTrueDofs` + `EliminateReducedTrueDofs(diag_policy)`
            // (bilinearform.cpp:959-962, staticcond.cpp:286).
            let ess_red: Vec<DofId> = ess_tdof_list
                .iter()
                .map(|&d| sys.reduced_id[d as usize])
                .collect();
            let xr = eliminate_ess_tdofs(&mut sys.reduced, &ess_red, &x_red, &mut g, policy);
            b[..n_red].copy_from_slice(&g);
            self.cached = Some(sys.reduced.clone());
            self.cond_sys = Some(sys);
            return xr;
        }
        let a = self.cached.as_mut().expect("assemble() must be called first");
        eliminate_ess_tdofs(a, ess_tdof_list, x, b, policy)
    }

    /// **MFEM `BilinearForm::RecoverFEMSolution(X, b, x)`**
    /// (fem/bilinearform.cpp:995).  With static condensation on, back-substitutes
    /// the eliminated element-private dofs — `static_cond->ComputeSolution`
    /// (bilinearform.cpp:1000-1004), via the D839-2
    /// [`crate::assembler::recover_condensed_interior`].  Without, `X` is `x`
    /// (`x.SyncMemory(X)`): the bitwise copy.
    ///
    /// The argument is the solved system vector (reduced when SC is on); the
    /// return value is the full finite-element grid-function vector.
    pub fn recover_fem_solution(&self, x: &[f64]) -> Vec<f64> {
        match self.cond_sys.as_ref() {
            Some(sys) => crate::assembler::recover_condensed_interior(sys, x),
            None => x.to_vec(),
        }
    }

    /// The condensed-system record of the last SC
    /// [`BilinearForm::form_linear_system`] — full-dof → reduced-row
    /// compaction (`reduced_id`) plus the per-element recovery factors;
    /// `None` when the switch is off or has not run yet.  Read side of
    /// MFEM's `BilinearForm::static_cond` state.
    pub fn condensed_system(&self) -> Option<&CondensedLinearSystem> {
        self.cond_sys.as_ref()
    }

    /// Eliminate essential (Dirichlet) BCs symmetrically.
    ///
    /// Modifies the cached matrix `A` and `rhs` in-place so that
    /// the solution satisfies `u[d] = bc_vals[i]` for each `d = ess_dofs[i]`.
    ///
    /// Reaction semantics (D432): each reaction is taken from the **true
    /// column entry** `A[i,d]` only — MFEM
    /// `SparseMatrix::EliminateRowCol(rc, sol, rhs, DIAG_ONE)`
    /// (linalg/sparsemat.cpp:1914; the reaction `rhs(col) -= sol·A[k]` at
    /// :1959 reads row `col`'s own entry), matching
    /// `CsrMatrix::apply_dirichlet_keep_diag` (D409).  An earlier revision
    /// *also* subtracted the pivot-row entries `A[d,j]`, doubling every
    /// nonzero reaction on numerically symmetric matrices (`A[d,j] ==
    /// A[j,d]` there) and flipping signs on antisymmetric coupling blocks.
    pub fn eliminate_essential_bc(&mut self, ess_dofs: &[usize], bc_vals: &[f64], rhs: &mut [f64]) {
        let a = self.cached.as_mut().expect("assemble() must be called first");
        let n = a.nrows;
        let ess: std::collections::HashSet<usize> = ess_dofs.iter().copied().collect();
        for (pos, &d) in ess_dofs.iter().enumerate() {
            let val = bc_vals[pos];
            // Column contributions only (D432): `rhs[i] -= A[i,d] * val`
            // from the true column entry, then zero it.  The symmetric row
            // counterpart `A[d,j]` must NOT also be subtracted.
            for i in 0..n {
                if ess.contains(&i) { continue; }
                for r in a.row_ptr[i]..a.row_ptr[i + 1] {
                    if a.col_idx[r] as usize == d {
                        if i != d {
                            rhs[i] -= a.values[r] * val;
                        }
                        // Zero this entry (row i, column d)
                        a.values[r] = 0.0;
                        break;
                    }
                }
            }
            // Zero row d and set diagonal
            for r in a.row_ptr[d]..a.row_ptr[d + 1] {
                a.values[r] = 0.0;
            }
            // Find diagonal entry in row d and set to 1
            for r in a.row_ptr[d]..a.row_ptr[d + 1] {
                if a.col_idx[r] as usize == d {
                    a.values[r] = 1.0;
                    break;
                }
            }
            rhs[d] = val;
        }
    }

    /// Diagonal-only BC elimination (D450).
    ///
    /// For each essential DOF `d = ess_dofs[i]` this does **only** two things:
    /// sets the stored diagonal entry `A[d,d] = 1.0` and `rhs[d] = bc_vals[i]`.
    ///
    /// It does **not** zero the off-diagonal entries of row `d`/column `d`,
    /// and it computes **no reactions** (no `rhs[i] -= A[i,d]·val`).  This is
    /// the eigenvalue-style `EliminateEssentialBCDiag` semantics (MFEM
    /// `DiagonalPolicy::DIAG_ONE`), appropriate when reactions are not needed
    /// (e.g. generalized eigenproblems) or when the essential rows/cols are
    /// already decoupled by the caller.  For full symmetric elimination with
    /// reaction recovery use [`Self::eliminate_essential_bc`].
    pub fn eliminate_essential_bc_from_diag(
        &mut self,
        ess_dofs: &[usize],
        bc_vals: &[f64],
        rhs: &mut [f64],
    ) {
        let a = self.cached.as_mut().expect("assemble() must be called first");
        for (pos, &d) in ess_dofs.iter().enumerate() {
            for r in a.row_ptr[d]..a.row_ptr[d + 1] {
                if a.col_idx[r] as usize == d {
                    a.values[r] = 1.0;
                    break;
                }
            }
            rhs[d] = bc_vals[pos];
        }
    }
}

// ─── Free helpers: MFEM SparseMatrix::EliminateCols ───────────────────────────

/// Eliminate essential columns from a (possibly rectangular) matrix —
/// 1:1 port of MFEM `SparseMatrix::EliminateCols(cols, &x, &b)`.
///
/// For every row `i` and stored entry `(i, c)` with `c` an essential column:
/// `b[i] -= A[i,c] * x[c]`, then `A[i,c] = 0`.  Used e.g. for
/// `MixedBilinearForm::EliminateTrialEssentialBC` (ex36 obstacle problem).
pub fn eliminate_cols(
    a: &mut CsrMatrix<f64>,
    ess_cols: &[usize],
    x: &[f64],
    rhs: &mut [f64],
) {
    let mut marker = vec![false; a.ncols];
    for &c in ess_cols {
        if c < a.ncols {
            marker[c] = true;
        }
    }
    for i in 0..a.nrows {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[p] as usize;
            if marker[c] {
                rhs[i] -= a.values[p] * x[c];
                a.values[p] = 0.0;
            }
        }
    }
}

// ─── LinearForm (scalar-valued) ────────────────────────────────────────────────

pub struct LinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn LinearIntegrator>>,
    cached: Option<Vec<f64>>,
}

impl<S: FESpace> LinearForm<S> {
    pub fn new(space: S) -> Self {
        LinearForm { space, integrators: Vec::new(), cached: None }
    }
    pub fn add_integrator(mut self, integ: impl LinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &[f64] {
        let refs: Vec<&dyn LinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let rhs = Assembler::assemble_linear(&self.space, &refs, quad_order);
        self.cached = Some(rhs);
        self.cached.as_ref().unwrap()
    }
    pub fn vec(&self) -> Option<&[f64]> { self.cached.as_deref() }
}

// ─── VectorBilinearForm (H(curl) / H(div) valued) ──────────────────────────────

pub struct VectorBilinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn VectorBilinearIntegrator>>,
    cached: Option<CsrMatrix<f64>>,
}

impl<S: FESpace> VectorBilinearForm<S> {
    pub fn new(space: S) -> Self {
        VectorBilinearForm { space, integrators: Vec::new(), cached: None }
    }
    pub fn add_integrator(mut self, integ: impl VectorBilinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &CsrMatrix<f64> {
        let refs: Vec<&dyn VectorBilinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let mat = VectorAssembler::assemble_bilinear(&self.space, &refs, quad_order);
        self.cached = Some(mat);
        self.cached.as_ref().unwrap()
    }
    /// Expose the cached matrix (MFEM's `BilinearForm::mat()`).
    pub fn mat(&self) -> Option<&CsrMatrix<f64>> { self.cached.as_ref() }
    /// Reference to the FE space.
    pub fn space(&self) -> &S { &self.space }
}

// ─── VectorLinearForm ──────────────────────────────────────────────────────────

pub struct VectorLinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn VectorLinearIntegrator>>,
    cached: Option<Vec<f64>>,
}

impl<S: FESpace> VectorLinearForm<S> {
    pub fn new(space: S) -> Self {
        VectorLinearForm { space, integrators: Vec::new(), cached: None }
    }
    pub fn add_integrator(mut self, integ: impl VectorLinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &[f64] {
        let refs: Vec<&dyn VectorLinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let rhs = VectorAssembler::assemble_linear(&self.space, &refs, quad_order);
        self.cached = Some(rhs);
        self.cached.as_ref().unwrap()
    }
    /// Expose the cached vector (MFEM's `LinearForm`).
    pub fn vec(&self) -> Option<&[f64]> { self.cached.as_deref() }
    /// Reference to the FE space.
    pub fn space(&self) -> &S { &self.space }
}

/// Pin a pressure DOF in a Stokes saddle-point system.
///
/// Zeroes row `dof` of `B` (n_p × n_u) and column `dof` of `B^T` (n_u × n_p)
/// to remove the constant nullspace of the discrete pressure.
/// Equivalent to MFEM's `EliminateEssentialBC` on a single pressure DOF.
pub fn pin_pressure_dof(
    b: &mut CsrMatrix<f64>,
    bt: &mut CsrMatrix<f64>,
    dof: usize,
) {
    for ptr in b.row_ptr[dof]..b.row_ptr[dof + 1] {
        b.values[ptr] = 0.0;
    }
    for i in 0..bt.nrows {
        for ptr in bt.row_ptr[i]..bt.row_ptr[i + 1] {
            if bt.col_idx[ptr] as usize == dof {
                bt.values[ptr] = 0.0;
            }
        }
    }
}

/// Form the linear system by applying essential BCs (MFEM's `FormLinearSystem`).
///
/// Eliminates essential DOFs from the system, returning the reduced matrix and RHS.
pub fn form_linear_system(
    a: &CsrMatrix<f64>,
    rhs: &[f64],
    ess_dofs: &[u32],
    bc_vals: &[f64],
) -> (CsrMatrix<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
    fem_space::constraints::dirichlet::eliminate_dirichlet(a, rhs, ess_dofs, bc_vals)
}

/// Recover the full FEM solution (MFEM's `RecoverFEMSolution`).
pub fn recover_fem_solution(
    x_red: &[f64],
    free: &[usize],
    constrained: &[usize],
    bc_vals: &[f64],
    n_full: usize,
) -> Vec<f64> {
    fem_space::constraints::dirichlet::expand_from_reduced(x_red, free, constrained, bc_vals, n_full)
}
