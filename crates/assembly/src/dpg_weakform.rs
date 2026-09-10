//! DPG (Discontinuous Petrov-Galerkin) weak formulation.
//!
//! 1:1 port of MFEM's `miniapps/dpg/util/weakform.hpp` / `weakform.cpp`
//! (class `DPGWeakForm`), with the `BlockStaticCondensation` from
//! `blockstaticcond.hpp` folded in as [`BlockStaticCondensation`].
//!
//! Given the variational formulation `a(u, v) = b(v)` (i.e. `A u = b`), the
//! DPG weak form assembles the minimum-residual (normal) system
//!
//! ```text
//!     Aᵀ G⁻¹ A u = Aᵀ G⁻¹ b
//! ```
//!
//! where `G` is the Riesz (Gram) operator on the broken (discontinuous) test
//! spaces.  `G` is defined and inverted element-wise (Cholesky), so the global
//! assembly has the same structure as a standard FEM assembly:
//!
//! ```text
//! for each element e:
//!     B_e  (test × trial)   from trial integrators (volume + trace faces)
//!     G_e  (test × test)    from test integrators
//!     f_e  (test)           from linear-form integrators
//!     G_e = L Lᵀ (Cholesky);  Y = L⁻¹ B_e;  y = L⁻¹ f_e
//!     A_e = Yᵀ Y;  b_e = Yᵀ y        (element normal system)
//!     scatter A_e / b_e with the trial-space element (or face) vdofs
//! ```
//!
//! C++ → Rust feature map:
//! * `Assemble` → [`DpgWeakForm::assemble`]
//! * `FormLinearSystem` → [`DpgWeakForm::form_linear_system`]
//! * `EliminateVDofs` / `EliminateVDofsInRHS` → applied inside
//!   [`DpgWeakForm::form_linear_system`] (MFEM `DIAG_ONE` policy)
//! * `EnableStaticCondensation` → [`DpgWeakForm::enable_static_condensation`]
//! * `ComputeResidual` → [`DpgWeakForm::compute_residual`] (requires
//!   [`DpgWeakForm::store_matrices`])
//! * `RecoverFEMSolution` → [`DpgWeakForm::recover_fem_solution`]
//! * `Update` (AMR) → [`DpgWeakForm::update`] — resets the assembly only;
//!   see the method docs for the non-conforming-AMR kernel gap.
//! * `ComplexDPGWeakForm` → [`crate::complex_dpg_weakform`].

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};

use crate::dpg::dpg_basis::{
    eval_face_lagrange, eval_vol_space, face_param_to_elem_ref, local_face_table, ref_node_coords,
    scalar_ref_elem, vol_quadrature, VolKind, VolVals,
};
use crate::dpg::dpg_basis::SkeletonSpace;
use crate::dpg::dpg_integrators::{
    DpgBilinear2, DpgLinear2, DpgTraceBilinear2, FaceCtx, FaceVals, VolCtx,
};
use crate::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};

// ─── Space descriptors ───────────────────────────────────────────────────────

/// A trial space of the DPG weak form.
enum TrialSpace<M: MeshTopology> {
    /// Broken volume space: consecutive per-element DOFs.
    Volume {
        /// Space kind (scalar or `byNODES`-expanded vector).
        kind: VolKind,
        /// Polynomial order.
        order: u8,
        /// DOFs per element (incl. vdim expansion).
        dofs_per_elem: usize,
    },
    /// Trace (skeleton) space: face-based DOFs shared between neighbors.
    Trace {
        /// Skeleton space (Lagrange face bases).
        skeleton: SkeletonSpace<M>,
    },
}

/// A broken (element-local) test space.
#[derive(Clone)]
struct TestSpace {
    /// Space kind.
    kind: VolKind,
    /// Polynomial order.
    order: u8,
}

/// Per-element static-condensation working data.
struct CondensationData {
    /// Trial blocks kept in the reduced (Schur) system (trace blocks).
    exposed_blocks: Vec<usize>,
    /// Trial blocks eliminated per element (volume blocks).
    private_blocks: Vec<usize>,
    /// Per element: (LU of `A_pp`, pivot permutation).
    lu: Vec<(Vec<f64>, Vec<usize>)>,
    /// Per element: `A_pe` (private-row × exposed-col coupling).
    pe: Vec<Vec<f64>>,
    /// Per element: `A_pp⁻¹ b_p`.
    bpe: Vec<Vec<f64>>,
    /// Per element: (`n_private`, `n_exposed`).
    sizes: Vec<(usize, usize)>,
}

/// The DPG weak form over a mesh.
///
/// Generic over the mesh type; supports multiple trial spaces (broken volume
/// L2 families and skeleton traces) and multiple broken test spaces, mirroring
/// MFEM's `DPGWeakForm(trial_fes, test_fec)`.
pub struct DpgWeakForm<M: MeshTopology + Clone + 'static> {
    mesh: M,
    dim: usize,
    trial_spaces: Vec<TrialSpace<M>>,
    test_spaces: Vec<TestSpace>,
    /// Trial integrators per test block: `(trial_block, integrator)`.
    trial_integs: Vec<Vec<(usize, Box<dyn DpgBilinear2>)>>,
    /// Test integrators: `(row_block, col_block, integrator)` → `G[row, col]`.
    test_integs: Vec<(usize, usize, Box<dyn DpgBilinear2>)>,
    /// Linear-form integrators per test block.
    lf_integs: Vec<(usize, Box<dyn DpgLinear2>)>,
    /// Trace integrators: `(trial_block, test_block, integrator)`.
    trace_integs: Vec<(usize, usize, Box<dyn DpgTraceBilinear2>)>,
    /// Quadrature order for volume integrals.
    quad_order: u8,
    /// Quadrature order for face integrals.
    face_quad_order: u8,

    // Assembled state
    mat: Option<CsrMatrix<f64>>,
    y: Option<Vec<f64>>,
    dof_offsets: Vec<usize>,
    assembled: bool,
    /// Static-condensation configuration (set by `enable_static_condensation`).
    cond: Option<(Vec<usize>, Vec<usize>)>, // (exposed, private) block ids
    /// Stored per-element condensation data after `assemble`.
    cond_data: Option<CondensationData>,
    reduced_mat: Option<CsrMatrix<f64>>,
    reduced_y: Option<Vec<f64>>,
    store_matrices: bool,
    /// Per-element stored (`Y = L⁻¹B`, `y = L⁻¹f`, `n_trial_e`) for
    /// [`Self::compute_residual`].
    stored: Vec<(Vec<f64>, Vec<f64>, usize)>,
}

/// Static condensation descriptor (port of MFEM `BlockStaticCondensation`
/// specialized to the DPG structure: exposed dofs = trace (skeleton) blocks,
/// eliminated ("private") dofs = broken volume blocks).
pub struct BlockStaticCondensation {
    /// Trial block indices that remain in the reduced (Schur) system.
    pub exposed_blocks: Vec<usize>,
    /// Trial block indices eliminated on each element.
    pub private_blocks: Vec<usize>,
}

/// A formed DPG linear system (MFEM returns an `OperatorHandle` holding
/// either the full `BlockMatrix` or the Schur matrix).
pub enum DpgSystem {
    /// Full system over the trial blocks.
    Full {
        /// Global system matrix.
        mat: CsrMatrix<f64>,
        /// Trial block offsets.
        offsets: Vec<usize>,
    },
    /// Statically condensed (Schur-complement) system over the trace blocks.
    Condensed {
        /// Schur matrix over the exposed (trace) blocks (global dof numbering
        /// of the trace dofs is preserved).
        mat: CsrMatrix<f64>,
        /// Exposed block offsets within the condensed system.
        offsets: Vec<usize>,
    },
}

impl DpgSystem {
    /// Total size of the (possibly reduced) system.
    pub fn size(&self) -> usize {
        match self {
            DpgSystem::Full { mat, .. } | DpgSystem::Condensed { mat, .. } => mat.nrows,
        }
    }

    /// System matrix reference.
    pub fn matrix(&self) -> &CsrMatrix<f64> {
        match self {
            DpgSystem::Full { mat, .. } | DpgSystem::Condensed { mat, .. } => mat,
        }
    }

    /// Apply the system operator.
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        self.matrix().spmv(x, y);
    }

    /// Extract diagonal block `b` (by system block offsets) — used by
    /// block preconditioners (MFEM `BlockDiagonalPreconditioner`).
    pub fn diag_block(&self, b: usize, offsets: &[usize]) -> CsrMatrix<f64> {
        let mat = self.matrix();
        let (r0, r1) = (offsets[b], offsets[b + 1]);
        let mut coo = CooMatrix::<f64>::new(r1 - r0, r1 - r0);
        for i in r0..r1 {
            for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                let c = mat.col_idx[p] as usize;
                if (r0..r1).contains(&c) {
                    coo.add(i - r0, c - r0, mat.values[p]);
                }
            }
        }
        coo.into_csr()
    }
}

// ─── Dense linear algebra helpers ────────────────────────────────────────────

/// Cholesky factorization `A = L Lᵀ` (in place, row-major `n × n`).
/// Returns `false` if not positive definite.
fn cholesky_factor(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        d = d.sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    true
}

/// Forward substitution `L X = B` in place (`b`: row-major `n × k`).
fn cholesky_lsolve(l: &[f64], n: usize, b: &mut [f64], k: usize) {
    for i in 0..n {
        for kk in 0..k {
            let mut s = b[i * k + kk];
            for j in 0..i {
                s -= l[i * n + j] * b[j * k + kk];
            }
            b[i * k + kk] = s / l[i * n + i];
        }
    }
}

/// LU factorization with partial pivoting (in place); returns the permutation.
pub(crate) fn lu_factor(a: &mut [f64], n: usize) -> Vec<usize> {
    let mut piv: Vec<usize> = (0..n).collect();
    for c in 0..n {
        let mut best = c;
        let mut bv = a[c * n + c].abs();
        for r in (c + 1)..n {
            let v = a[r * n + c].abs();
            if v > bv {
                bv = v;
                best = r;
            }
        }
        if best != c {
            for k in 0..n {
                a.swap(c * n + k, best * n + k);
            }
            piv.swap(c, best);
        }
        let p = a[c * n + c];
        if p.abs() < 1e-300 {
            continue;
        }
        for r in (c + 1)..n {
            let f = a[r * n + c] / p;
            a[r * n + c] = f;
            for k in (c + 1)..n {
                a[r * n + k] -= f * a[c * n + k];
            }
        }
    }
    piv
}

/// Solve `A x = b` given an [`lu_factor`] factorization (in place on `b`).
pub(crate) fn lu_solve(lu: &[f64], n: usize, piv: &[usize], b: &mut [f64]) {
    let y: Vec<f64> = piv.iter().map(|&p| b[p]).collect();
    for i in 0..n {
        let mut s = y[i];
        for j in 0..i {
            s -= lu[i * n + j] * y[j];
        }
        b[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= lu[i * n + j] * b[j];
        }
        b[i] = if lu[i * n + i].abs() > 1e-300 { s / lu[i * n + i] } else { 0.0 };
    }
}

// ─── Weak form ───────────────────────────────────────────────────────────────

impl<M: MeshTopology + Clone + 'static> DpgWeakForm<M> {
    /// Create a DPG weak form over `mesh`.
    pub fn new(mesh: M) -> Self {
        let dim = mesh.dim() as usize;
        Self {
            mesh,
            dim,
            trial_spaces: Vec::new(),
            test_spaces: Vec::new(),
            trial_integs: Vec::new(),
            test_integs: Vec::new(),
            lf_integs: Vec::new(),
            trace_integs: Vec::new(),
            quad_order: 6,
            face_quad_order: 4,
            mat: None,
            y: None,
            dof_offsets: Vec::new(),
            assembled: false,
            cond: None,
            cond_data: None,
            reduced_mat: None,
            reduced_y: None,
            store_matrices: false,
            stored: Vec::new(),
        }
    }

    /// Set the volume-integration quadrature order (default 6).
    pub fn set_quad_order(&mut self, order: u8) {
        self.quad_order = order;
    }

    /// Add a broken scalar-L2 trial space of given `order` — MFEM
    /// `FiniteElementSpace(mesh, L2_FECollection(order, dim))`.  Returns the
    /// trial block index.
    pub fn add_trial_scalar_space(&mut self, order: u8) -> usize {
        let n = scalar_ref_elem(self.element_type(), order).n_dofs();
        self.trial_spaces.push(TrialSpace::Volume {
            kind: VolKind::Scalar,
            order,
            dofs_per_elem: n,
        });
        self.trial_integs.push(Vec::new());
        self.trial_spaces.len() - 1
    }

    /// Add a broken vector-L2 trial space (`vdim` components, `byNODES`
    /// layout) — MFEM `FiniteElementSpace(mesh, L2_FECollection(order), vdim)`.
    pub fn add_trial_vector_space(&mut self, order: u8, vdim: usize) -> usize {
        let n = scalar_ref_elem(self.element_type(), order).n_dofs() * vdim;
        self.trial_spaces.push(TrialSpace::Volume {
            kind: VolKind::Vector { vdim },
            order,
            dofs_per_elem: n,
        });
        self.trial_integs.push(Vec::new());
        self.trial_spaces.len() - 1
    }

    /// Add a trace (skeleton) trial space of face order `order` — MFEM
    /// `H1_Trace_FECollection(order, dim)` or `RT_Trace_FECollection(order,
    /// dim)` (both traces are the nodal `P_order` space on each face; the RT
    /// orientation sign lives in the trace integrators' ±1 scale).
    pub fn add_trial_trace_space(&mut self, order: u8) -> usize {
        let sk = SkeletonSpace::new(self.mesh.clone(), order);
        self.trial_spaces.push(TrialSpace::Trace { skeleton: sk });
        self.trial_integs.push(Vec::new());
        self.trial_spaces.len() - 1
    }

    /// Add a broken test space — MFEM `test_fec.Append(fec)`; returns the
    /// test block index.
    pub fn add_test_space(&mut self, kind: VolKind, order: u8) -> usize {
        self.test_spaces.push(TestSpace { kind, order });
        self.test_spaces.len() - 1
    }

    /// Add a trial integrator — MFEM `AddTrialIntegrator(bfi, n, m)` with
    /// `n = trial_block`, `m = test_block`.
    pub fn add_trial_integrator(
        &mut self,
        integ: Box<dyn DpgBilinear2>,
        trial_block: usize,
        test_block: usize,
    ) {
        assert!(trial_block < self.trial_spaces.len());
        assert!(test_block < self.test_spaces.len());
        self.trial_integs[test_block].push((trial_block, integ));
    }

    /// Add a test (Riesz/Gram) integrator into `G[row, col]` — MFEM
    /// `AddTestIntegrator(bfi, n, m)`.
    pub fn add_test_integrator(
        &mut self,
        integ: Box<dyn DpgBilinear2>,
        row_block: usize,
        col_block: usize,
    ) {
        assert!(row_block < self.test_spaces.len());
        assert!(col_block < self.test_spaces.len());
        self.test_integs.push((row_block, col_block, integ));
    }

    /// Add a trace-face trial integrator — MFEM
    /// `AddTrialIntegrator(TraceIntegrator-family, n, m)`.
    pub fn add_trace_integrator(
        &mut self,
        integ: Box<dyn DpgTraceBilinear2>,
        trial_block: usize,
        test_block: usize,
    ) {
        assert!(trial_block < self.trial_spaces.len());
        assert!(test_block < self.test_spaces.len());
        self.trace_integs.push((trial_block, test_block, integ));
    }

    /// Add a linear-form integrator on test block `b` — MFEM
    /// `AddDomainLFIntegrator(lfi, n)`.
    pub fn add_domain_lf_integrator(&mut self, integ: Box<dyn DpgLinear2>, test_block: usize) {
        assert!(test_block < self.test_spaces.len());
        self.lf_integs.push((test_block, integ));
    }

    /// `EnableStaticCondensation()`: the global system keeps only the trace
    /// (skeleton) blocks; volume blocks are eliminated element-wise.
    pub fn enable_static_condensation(&mut self) {
        let exposed: Vec<usize> = (0..self.trial_spaces.len())
            .filter(|&i| self.is_trace_block(i))
            .collect();
        let private: Vec<usize> = (0..self.trial_spaces.len())
            .filter(|&i| !self.is_trace_block(i))
            .collect();
        assert!(!exposed.is_empty(), "static condensation requires a trace block");
        self.cond = Some((exposed, private));
    }

    /// `StoreMatrices(bool)`: store per-element `L⁻¹B`/`L⁻¹f` for
    /// [`Self::compute_residual`].
    pub fn store_matrices(&mut self, store: bool) {
        self.store_matrices = store;
    }

    /// Number of trial blocks.
    pub fn n_trial_blocks(&self) -> usize {
        self.trial_spaces.len()
    }

    /// Number of test blocks.
    pub fn n_test_blocks(&self) -> usize {
        self.test_spaces.len()
    }

    /// Trial block sizes (global DOF counts).
    pub fn trial_block_sizes(&self) -> Vec<usize> {
        self.trial_spaces
            .iter()
            .map(|s| match s {
                TrialSpace::Volume { dofs_per_elem, .. } => dofs_per_elem * self.mesh.n_elements(),
                TrialSpace::Trace { skeleton } => skeleton.n_dofs(),
            })
            .collect()
    }

    /// Cumulative trial block offsets (`len = nblocks + 1`).
    pub fn trial_offsets(&self) -> Vec<usize> {
        let mut off = vec![0usize];
        for s in self.trial_block_sizes() {
            off.push(off.last().unwrap() + s);
        }
        off
    }

    /// Global DOF count of the uncondensed system.
    pub fn size(&self) -> usize {
        self.trial_block_sizes().iter().sum()
    }

    /// Whether `assemble()` has been run.
    pub fn is_assembled(&self) -> bool {
        self.assembled
    }

    /// Access the skeleton space of trace block `b`.
    pub fn skeleton(&self, b: usize) -> &SkeletonSpace<M> {
        match &self.trial_spaces[b] {
            TrialSpace::Trace { skeleton } => skeleton,
            _ => panic!("block {b} is not a trace space"),
        }
    }

    /// Mesh reference.
    pub fn mesh(&self) -> &M {
        &self.mesh
    }

    /// Element vdofs of trial block `b` on element `e` (trace blocks:
    /// concatenated face dofs in local face order; volume blocks: consecutive
    /// per-element range) — MFEM `GetElementVDofs` / `GetFaceVDofs`.
    pub fn trial_element_vdofs(&self, b: usize, e: u32) -> Vec<usize> {
        let base = self.trial_offsets()[b];
        match &self.trial_spaces[b] {
            TrialSpace::Volume { dofs_per_elem, .. } => {
                let s = *dofs_per_elem;
                (base + e as usize * s..base + (e as usize + 1) * s).collect()
            }
            TrialSpace::Trace { skeleton } => {
                skeleton.element_dofs(e).iter().map(|&d| base + d).collect()
            }
        }
    }

    /// Whether trial block `b` is a trace space.
    pub fn is_trace_block(&self, b: usize) -> bool {
        matches!(self.trial_spaces[b], TrialSpace::Trace { .. })
    }

    /// Physical point of face DOF `k` of skeleton face `f` (for essential BC
    /// projection).
    pub fn face_dof_point(&self, sk: &SkeletonSpace<M>, f: usize, k: usize) -> Vec<f64> {
        let dim = self.dim;
        let p = sk.order() as usize;
        let nodes = sk.face_nodes(f);
        let coords: Vec<Vec<f64>> =
            nodes.iter().map(|&n| self.mesh.node_coords(n).to_vec()).collect();
        let params = face_dof_params(dim, sk.is_quad_face(f), p, k);
        let mut x = vec![0.0; dim];
        if dim == 2 {
            let s = params[0];
            for d in 0..dim {
                x[d] = (1.0 - s) * coords[0][d] + s * coords[1][d];
            }
        } else if sk.is_quad_face(f) {
            let (s, t) = (params[0], params[1]);
            for d in 0..dim {
                x[d] = (1.0 - s) * (1.0 - t) * coords[0][d]
                    + s * (1.0 - t) * coords[1][d]
                    + s * t * coords[2][d]
                    + (1.0 - s) * t * coords[3][d];
            }
        } else {
            let (s, t) = (params[0], params[1]);
            let r = 1.0 - s - t;
            for d in 0..dim {
                x[d] = r * coords[0][d] + s * coords[1][d] + t * coords[2][d];
            }
        }
        x
    }

    // ─── Assembly ────────────────────────────────────────────────────────────

    /// `Assemble()`: element-wise DPG assembly of the normal equations.
    pub fn assemble(&mut self) {
        let mesh = self.mesh.clone();
        let dim = self.dim;
        let n_elem = mesh.n_elements();
        let et = self.element_type();
        let is_simplex = matches!(et, ElementType::Tri3 | ElementType::Tet4);
        let is_quad_face_geom = dim == 3 && matches!(et, ElementType::Hex8);
        let geo_elem = if is_simplex { None } else { geo_ref_elem_from_mesh(&mesh, 0) };
        let trial_offsets = self.trial_offsets();
        let nblocks = self.trial_spaces.len();
        let n_total = trial_offsets[nblocks];

        let test_sizes: Vec<usize> = self
            .test_spaces
            .iter()
            .map(|t| t.kind.n_dofs_per_elem(et, t.order))
            .collect();
        let mut test_offsets = vec![0usize];
        for &s in &test_sizes {
            test_offsets.push(test_offsets.last().unwrap() + s);
        }
        let n_te = test_offsets[test_offsets.len() - 1];

        let (qpts, qwts) = vol_quadrature(et, self.quad_order);
        let n_qp = qpts.len();

        let face_rule_tri = if dim == 3 {
            Some(crate::dpg::dpg_basis::face_quadrature(3, false, self.face_quad_order))
        } else {
            None
        };
        let face_rule_quad = if dim == 3 {
            Some(crate::dpg::dpg_basis::face_quadrature(3, true, self.face_quad_order))
        } else {
            Some(crate::dpg::dpg_basis::face_quadrature(2, false, self.face_quad_order))
        };
        let _ = ref_node_coords(et); // reference tables kept for curved-face extension

        let cond_info = self.cond.clone(); // Option<(exposed, private)>
        let mut cond_lu: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
        let mut cond_pe: Vec<Vec<f64>> = Vec::new();
        let mut cond_bpe: Vec<Vec<f64>> = Vec::new();
        let mut cond_sizes: Vec<(usize, usize)> = Vec::new();
        let red_n = cond_info.as_ref().map(|(exp, _)| {
            let sizes = self.trial_block_sizes();
            exp.iter().map(|&b| sizes[b]).sum::<usize>()
        });
        let mut red_coo = red_n.map(|n| CooMatrix::<f64>::new(n, n));
        let mut red_y = red_n.map(|n| vec![0.0_f64; n]);

        let mut mat_coo = CooMatrix::<f64>::new(n_total, n_total);
        let mut y_global = vec![0.0_f64; n_total];
        self.stored.clear();

        for e in 0..n_elem as u32 {
            let nodes = mesh.element_nodes(e);
            let simplex = if is_simplex {
                Some(fem_mesh::ElementTransformation::from_simplex_nodes(&mesh, nodes))
            } else {
                None
            };
            let geo_nodes = if is_simplex { Vec::new() } else { mesh.geometry_nodes(e).to_vec() };
            let geo = geo_elem.as_deref();

            // Per-quadrature-point test/trial values
            let mut qp_test: Vec<Vec<VolVals>> = Vec::with_capacity(self.test_spaces.len());
            for ts in &self.test_spaces {
                qp_test.push(
                    qpts
                        .iter()
                        .map(|xi| {
                            let (jac, det, _) =
                                element_geo_at(&mesh, simplex.as_ref(), geo, &geo_nodes, xi, dim);
                            let jit = inv_transpose(&jac, dim);
                            let mut v = VolVals::default();
                            eval_vol_space(ts.kind, ts.order, et, dim, &jac, det, &jit, xi, None, &mut v);
                            v
                        })
                        .collect(),
                );
            }
            let mut qp_trial: Vec<Option<Vec<VolVals>>> = Vec::with_capacity(nblocks);
            for ts in &self.trial_spaces {
                match ts {
                    TrialSpace::Volume { kind, order, .. } => qp_trial.push(Some(
                        qpts
                            .iter()
                            .map(|xi| {
                                let (jac, det, _) = element_geo_at(
                                    &mesh, simplex.as_ref(), geo, &geo_nodes, xi, dim,
                                );
                                let jit = inv_transpose(&jac, dim);
                                let mut v = VolVals::default();
                                eval_vol_space(*kind, *order, et, dim, &jac, det, &jit, xi, None, &mut v);
                                v
                            })
                            .collect(),
                    )),
                    TrialSpace::Trace { .. } => qp_trial.push(None),
                }
            }

            // Element trial offsets
            let mut tr_offs = vec![0usize];
            for b in 0..nblocks {
                let n = match &self.trial_spaces[b] {
                    TrialSpace::Volume { dofs_per_elem, .. } => *dofs_per_elem,
                    TrialSpace::Trace { skeleton } => skeleton.element_dofs(e).len(),
                };
                tr_offs.push(tr_offs.last().unwrap() + n);
            }
            let n_tr = tr_offs[nblocks];

            let mut g = vec![0.0_f64; n_te * n_te];
            let mut bmat = vec![0.0_f64; n_te * n_tr];
            let mut fvec = vec![0.0_f64; n_te];

            // Linear forms
            for (tb, integ) in &self.lf_integs {
                for q in 0..n_qp {
                    let (_jac, det, xp) = element_geo_at(
                        &mesh,
                        simplex.as_ref(),
                        geo,
                        &geo_nodes,
                        &qpts[q],
                        dim,
                    );
                    let ctx = VolCtx { w: qwts[q] * det.abs(), x: xp, dim, elem: e };
                    let n = test_sizes[*tb];
                    let mut fv = vec![0.0_f64; n];
                    integ.assemble_linear(&ctx, &qp_test[*tb][q], &mut fv);
                    for (i, &v) in fv.iter().enumerate() {
                        fvec[test_offsets[*tb] + i] += v;
                    }
                }
            }

            // Test integrators (G blocks)
            for (row, col, integ) in &self.test_integs {
                let nr = test_sizes[*row];
                let nc = test_sizes[*col];
                let mut ge = vec![0.0_f64; nr * nc];
                for q in 0..n_qp {
                    let (_jac, det, xp) = element_geo_at(
                        &mesh,
                        simplex.as_ref(),
                        geo,
                        &geo_nodes,
                        &qpts[q],
                        dim,
                    );
                    let ctx = VolCtx { w: qwts[q] * det.abs(), x: xp, dim, elem: e };
                    integ.assemble2(&ctx, &qp_test[*col][q], &qp_test[*row][q], &mut ge);
                }
                let (r0, c0) = (test_offsets[*row], test_offsets[*col]);
                for i in 0..nr {
                    for j in 0..nc {
                        g[(r0 + i) * n_te + c0 + j] += ge[i * nc + j];
                    }
                }
            }

            // Volume trial integrators → B[test, trial]
            for tb in 0..self.test_spaces.len() {
                for (tbb, integ) in &self.trial_integs[tb] {
                    let nr = test_sizes[tb];
                    let nc = tr_offs[*tbb + 1] - tr_offs[*tbb];
                    let mut be = vec![0.0_f64; nr * nc];
                    for q in 0..n_qp {
                        let (_jac, det, xp) = element_geo_at(
                            &mesh,
                            simplex.as_ref(),
                            geo,
                            &geo_nodes,
                            &qpts[q],
                            dim,
                        );
                        let ctx = VolCtx { w: qwts[q] * det.abs(), x: xp, dim, elem: e };
                        let tv = qp_trial[*tbb].as_ref().unwrap()[q].clone();
                        integ.assemble2(&ctx, &tv, &qp_test[tb][q], &mut be);
                    }
                    let (r0, c0) = (test_offsets[tb], tr_offs[*tbb]);
                    for i in 0..nr {
                        for j in 0..nc {
                            bmat[(r0 + i) * n_tr + c0 + j] += be[i * nc + j];
                        }
                    }
                }
            }

            // Trace integrators: per local face (MFEM AssembleTraceFaceMatrix)
            let lfs = local_face_table(&nodes, dim);
            for (tbb, tb, integ) in &self.trace_integs {
                let sk = match &self.trial_spaces[*tbb] {
                    TrialSpace::Trace { skeleton } => skeleton,
                    _ => panic!("trace integrator on non-trace block"),
                };
                let nr = test_sizes[*tb];
                let col_base = tr_offs[*tbb];
                for (li, lf) in lfs.iter().enumerate() {
                    let fid = sk.elem_face_id(e, li);
                    let nfd = sk.dofs_per_face(fid);
                    let is_qf = if dim == 3 { sk.is_quad_face(fid) } else { false };
                    let (fpts, fwts): (&Vec<Vec<f64>>, &Vec<f64>) = if dim == 3 {
                        if is_qf {
                            (&face_rule_quad.as_ref().unwrap().0, &face_rule_quad.as_ref().unwrap().1)
                        } else {
                            (&face_rule_tri.as_ref().unwrap().0, &face_rule_tri.as_ref().unwrap().1)
                        }
                    } else {
                        (&face_rule_quad.as_ref().unwrap().0, &face_rule_quad.as_ref().unwrap().1)
                    };
                    // Sign of the element-side face contribution: the element's
                    // local face cycle against the canonical (global face
                    // storage) direction.  This is MFEM's element-face
                    // orientation (`Elem2` faces carry the −1), and it makes
                    // the element's outward normal consistent with the
                    // canonical face normal returned by `face_geo_at` for
                    // BOTH adjacent elements.  (The previous "first-seen
                    // element ⇒ +1" rule was equivalent only while faces were
                    // stored in the generating element's local direction.)
                    let scale = sk.elem_face_orientation(e, li) as f64;
                    let mut be = vec![0.0_f64; nr * nfd];
                    for (q, fparam) in fpts.iter().enumerate() {
                        let (xp, normal, measure) =
                            face_geo_at(&mesh, sk, fid, fparam, dim);
                        // Initial guess from the element's local face
                        // parametrization; Newton-invert to the exact
                        // reference coordinates of the physical face point.
                        let xi0 = face_param_to_elem_ref(
                            et,
                            lf,
                            is_quad_face_geom && is_qf,
                            fparam,
                        );
                        let xiref = invert_element_map(
                            &mesh,
                            simplex.as_ref(),
                            geo,
                            &geo_nodes,
                            &xp,
                            dim,
                            &xi0,
                        );
                        let (jac, det, _) = element_geo_at(
                            &mesh,
                            simplex.as_ref(),
                            geo,
                            &geo_nodes,
                            &xiref,
                            dim,
                        );
                        let jit = inv_transpose(&jac, dim);
                        let mut tv = VolVals::default();
                        let ts = &self.test_spaces[*tb];
                        eval_vol_space(
                            ts.kind, ts.order, et, dim, &jac, det, &jit, &xiref, None, &mut tv,
                        );
                        let mut fphi = vec![0.0_f64; nfd];
                        eval_face_lagrange(dim, is_qf, sk.order() as usize, fparam, &mut fphi);
                        let ctx = FaceCtx {
                            ip_weight: fwts[q],
                            measure,
                            normal,
                            scale,
                            dim,
                            x: xp,
                        };
                        let fv = FaceVals::scalar(fphi);
                        integ.assemble_trace2(&ctx, &fv, &tv, &mut be);
                    }
                    let r0 = test_offsets[*tb];
                    let coff = col_base + self.trace_face_col_offset(e, *tbb, li);
                    for i in 0..nr {
                        for j in 0..nfd {
                            bmat[(r0 + i) * n_tr + coff + j] += be[i * nfd + j];
                        }
                    }
                }
            }

            // Normal equations: G = L Lᵀ; Y = L⁻¹B; y = L⁻¹f
            let mut l = g;
            if !cholesky_factor(&mut l, n_te) {
                panic!(
                    "DpgWeakForm::assemble: test-space Gram matrix not SPD on element {e} \
                     (check the test integrators define a complete norm)"
                );
            }
            let mut yb = bmat;
            cholesky_lsolve(&l, n_te, &mut yb, n_tr);
            cholesky_lsolve(&l, n_te, &mut fvec, 1);
            if self.store_matrices {
                self.stored.push((yb.clone(), fvec.clone(), n_tr));
            }

            // A_e = Yᵀ Y ; b_e = Yᵀ y
            let mut a_e = vec![0.0_f64; n_tr * n_tr];
            for i in 0..n_tr {
                for j in 0..n_tr {
                    let mut s = 0.0;
                    for k in 0..n_te {
                        s += yb[k * n_tr + i] * yb[k * n_tr + j];
                    }
                    a_e[i * n_tr + j] = s;
                }
            }
            let mut b_e = vec![0.0_f64; n_tr];
            for i in 0..n_tr {
                let mut s = 0.0;
                for k in 0..n_te {
                    s += yb[k * n_tr + i] * fvec[k];
                }
                b_e[i] = s;
            }

            match &cond_info {
                None => {
                    for bi in 0..nblocks {
                        let vdofs_i = self.trial_element_vdofs(bi, e);
                        for (li, &gd) in vdofs_i.iter().enumerate() {
                            y_global[gd] += b_e[tr_offs[bi] + li];
                        }
                        for bj in 0..nblocks {
                            let vdofs_j = self.trial_element_vdofs(bj, e);
                            for (li, &gr) in vdofs_i.iter().enumerate() {
                                for (cj, &gc) in vdofs_j.iter().enumerate() {
                                    let v = a_e[(tr_offs[bi] + li) * n_tr + tr_offs[bj] + cj];
                                    if v != 0.0 {
                                        mat_coo.add(gr, gc, v);
                                    }
                                }
                            }
                        }
                    }
                }
                Some((exp_blocks, priv_blocks)) => {
                    let priv_ranges: Vec<(usize, usize)> =
                        priv_blocks.iter().map(|&b| (tr_offs[b], tr_offs[b + 1])).collect();
                    let exp_ranges: Vec<(usize, usize)> =
                        exp_blocks.iter().map(|&b| (tr_offs[b], tr_offs[b + 1])).collect();
                    let n_priv: usize = priv_ranges.iter().map(|(a, b)| b - a).sum();
                    let n_exp: usize = exp_ranges.iter().map(|(a, b)| b - a).sum();

                    let mut a_pp = vec![0.0_f64; n_priv * n_priv];
                    let mut a_pe = vec![0.0_f64; n_priv * n_exp];
                    let mut a_ep = vec![0.0_f64; n_exp * n_priv];
                    let mut a_ee = vec![0.0_f64; n_exp * n_exp];
                    let mut b_p = vec![0.0_f64; n_priv];
                    let mut b_e2 = vec![0.0_f64; n_exp];
                    let mut ip = 0usize;
                    for &(r0, r1) in &priv_ranges {
                        let mut jp = 0usize;
                        for &(c0, c1) in &priv_ranges {
                            for i in r0..r1 {
                                for j in c0..c1 {
                                    a_pp[(ip + i - r0) * n_priv + jp + j - c0] = a_e[i * n_tr + j];
                                }
                            }
                            jp += c1 - c0;
                        }
                        let mut je = 0usize;
                        for &(c0, c1) in &exp_ranges {
                            for i in r0..r1 {
                                for j in c0..c1 {
                                    a_pe[(ip + i - r0) * n_exp + je + j - c0] = a_e[i * n_tr + j];
                                }
                            }
                            je += c1 - c0;
                        }
                        for i in r0..r1 {
                            b_p[ip + i - r0] = b_e[i];
                        }
                        ip += r1 - r0;
                    }
                    let mut ie = 0usize;
                    for &(r0, r1) in &exp_ranges {
                        let mut jp = 0usize;
                        for &(c0, c1) in &priv_ranges {
                            for i in r0..r1 {
                                for j in c0..c1 {
                                    a_ep[(ie + i - r0) * n_priv + jp + j - c0] = a_e[i * n_tr + j];
                                }
                            }
                            jp += c1 - c0;
                        }
                        let mut je2 = 0usize;
                        for &(c0, c1) in &exp_ranges {
                            for i in r0..r1 {
                                for j in c0..c1 {
                                    a_ee[(ie + i - r0) * n_exp + je2 + j - c0] = a_e[i * n_tr + j];
                                }
                            }
                            je2 += c1 - c0;
                        }
                        for i in r0..r1 {
                            b_e2[ie + i - r0] = b_e[i];
                        }
                        ie += r1 - r0;
                    }

                    // S = A_ee - A_ep A_pp⁻¹ A_pe ; s = b_e - A_ep A_pp⁻¹ b_p
                    let mut lu = a_pp;
                    let piv = lu_factor(&mut lu, n_priv);
                    let mut ape_inv = a_pe.clone();
                    for col in 0..n_exp {
                        let mut colv: Vec<f64> =
                            (0..n_priv).map(|r| ape_inv[r * n_exp + col]).collect();
                        lu_solve(&lu, n_priv, &piv, &mut colv);
                        for r in 0..n_priv {
                            ape_inv[r * n_exp + col] = colv[r];
                        }
                    }
                    let mut xpp = b_p;
                    lu_solve(&lu, n_priv, &piv, &mut xpp);

                    let mut se = a_ee;
                    for i in 0..n_exp {
                        for j in 0..n_exp {
                            let mut s = 0.0;
                            for k in 0..n_priv {
                                s += a_ep[i * n_priv + k] * ape_inv[k * n_exp + j];
                            }
                            se[i * n_exp + j] -= s;
                        }
                    }
                    let mut s_rhs = b_e2;
                    for i in 0..n_exp {
                        let mut s = 0.0;
                        for k in 0..n_priv {
                            s += a_ep[i * n_priv + k] * xpp[k];
                        }
                        s_rhs[i] -= s;
                    }

                    cond_lu.push((lu, piv));
                    cond_pe.push(a_pe);
                    cond_bpe.push(xpp);
                    cond_sizes.push((n_priv, n_exp));

                    // Scatter into the reduced system (compact exposed-dof
                    // numbering; `eoff` maps exposed block order to Schur
                    // indices).
                    let mut eoff = vec![0usize];
                    for &b in exp_blocks {
                        eoff.push(eoff.last().unwrap() + (tr_offs[b + 1] - tr_offs[b]));
                    }
                    for (bi, &b) in exp_blocks.iter().enumerate() {
                        let vdofs_i = self.trial_element_vdofs(b, e);
                        let r0 = eoff[bi];
                        for (li, _gd) in vdofs_i.iter().enumerate() {
                            red_y.as_mut().unwrap()[r0 + li] += s_rhs[r0 + li];
                        }
                        for (bj, &b2) in exp_blocks.iter().enumerate() {
                            let vdofs_j = self.trial_element_vdofs(b2, e);
                            let c0 = eoff[bj];
                            for li in 0..vdofs_i.len() {
                                for cj in 0..vdofs_j.len() {
                                    let v = se[(r0 + li) * n_exp + c0 + cj];
                                    if v != 0.0 {
                                        red_coo.as_mut().unwrap().add(r0 + li, c0 + cj, v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.mat = Some(mat_coo.into_csr());
        self.y = Some(y_global);
        self.dof_offsets = trial_offsets;
        if let Some((exp, prv)) = cond_info {
            self.cond_data = Some(CondensationData {
                exposed_blocks: exp,
                private_blocks: prv,
                lu: cond_lu,
                pe: cond_pe,
                bpe: cond_bpe,
                sizes: cond_sizes,
            });
            self.reduced_mat = red_coo.map(|c| c.into_csr());
            self.reduced_y = red_y;
        }
        self.assembled = true;
    }

    /// `FormLinearSystem(ess_tdof_list, x, ...)`: essential-BC elimination
    /// (`DIAG_ONE`) on the (possibly condensed) system.  Returns
    /// `(system, X initial guess, B rhs)`; `B = y − M_e x`.
    pub fn form_linear_system(
        &mut self,
        ess_dofs: &[usize],
        x: &[f64],
        copy_interior: bool,
    ) -> (DpgSystem, Vec<f64>, Vec<f64>) {
        assert!(self.assembled, "assemble() must run before form_linear_system()");
        let n_full = self.size();
        let mut xg = vec![0.0_f64; n_full];
        if copy_interior {
            xg.copy_from_slice(&x[..n_full.min(x.len())]);
        }
        for &d in ess_dofs {
            xg[d] = x[d];
        }
        if self.cond.is_some() {
            let mat = self.reduced_mat.clone().unwrap();
            let mut b = self.reduced_y.clone().unwrap();
            // Translate global ess dofs to compact exposed-dof indices.
            let eoffs = self.exposed_block_offsets();
            let cd = self.cond_data.as_ref().expect("condensation data missing");
            let mut ess_compact = Vec::with_capacity(ess_dofs.len());
            for &d in ess_dofs {
                let mut c = None;
                for (bi, &blk) in cd.exposed_blocks.iter().enumerate() {
                    let base = self.dof_offsets[blk];
                    let sz = self.trial_block_sizes()[blk];
                    if d >= base && d < base + sz {
                        c = Some(eoffs[bi] + (d - base));
                        break;
                    }
                }
                if let Some(cc) = c {
                    ess_compact.push(cc);
                }
            }
            let (mat_e, mat2) = eliminate_row_cols(&mat, &ess_compact);
            // RHS correction with the compact ess data.
            let mut xc = vec![0.0_f64; mat.nrows];
            for &d in ess_dofs {
                let mut c = None;
                for (bi, &blk) in cd.exposed_blocks.iter().enumerate() {
                    let base = self.dof_offsets[blk];
                    let sz = self.trial_block_sizes()[blk];
                    if d >= base && d < base + sz {
                        c = Some(eoffs[bi] + (d - base));
                        break;
                    }
                }
                if let Some(cc) = c {
                    xc[cc] = xg[d];
                }
            }
            let mut corr = vec![0.0_f64; mat.nrows];
            mat_e.spmv(&xc, &mut corr);
            for i in 0..b.len() {
                b[i] -= corr[i];
            }
            self.reduced_mat = Some(mat2.clone());
            let offsets = eoffs.clone();
            (DpgSystem::Condensed { mat: mat2, offsets }, xc, b)
        } else {
            let mat = self.mat.clone().unwrap();
            let mut b = self.y.clone().unwrap();
            let (mat_e, mat2) = eliminate_row_cols(&mat, ess_dofs);
            let mut corr = vec![0.0_f64; mat.nrows];
            mat_e.spmv(&xg, &mut corr);
            for i in 0..b.len() {
                b[i] -= corr[i];
            }
            let mut xs = vec![0.0_f64; mat.nrows];
            if copy_interior {
                xs.copy_from_slice(&xg);
            } else {
                for &d in ess_dofs {
                    xs[d] = xg[d];
                }
            }
            self.mat = Some(mat2.clone());
            let offsets = self.dof_offsets.clone();
            (DpgSystem::Full { mat: mat2, offsets }, xs, b)
        }
    }

    /// `RecoverFEMSolution(X, x)`: map the solved vector to the full trial
    /// vector; with static condensation the private (volume) dofs are
    /// recovered element-wise: `x_p = A_pp⁻¹ (b_p − A_pe x_e)`.
    pub fn recover_fem_solution(&self, xs: &[f64]) -> Vec<f64> {
        let n = self.size();
        let mut x = vec![0.0_f64; n];
        match &self.cond_data {
            Some(cd) => {
                // `xs` is in compact exposed-dof ordering; scatter to global.
                let eoffs = self.exposed_block_offsets();
                for (bi, &b) in cd.exposed_blocks.iter().enumerate() {
                    let base = self.dof_offsets[b];
                    let sz = self.trial_block_sizes()[b];
                    for k in 0..sz {
                        x[base + k] = xs[eoffs[bi] + k];
                    }
                }
                for e in 0..self.mesh.n_elements() as u32 {
                    let (lu, piv) = &cd.lu[e as usize];
                    let (n_priv, n_exp) = cd.sizes[e as usize];
                    // gather x_e in the exposed-block order of this element
                    let mut x_e = vec![0.0_f64; n_exp];
                    for (bj, &b) in cd.exposed_blocks.iter().enumerate() {
                        let vd = self.trial_element_vdofs(b, e);
                        let base = self.dof_offsets[b];
                        for (li, &g) in vd.iter().enumerate() {
                            x_e[eoffs[bj] + li] = xs[eoffs[bj] + (g - base)];
                        }
                    }
                    // rhs = xpp - A_pe x_e, then solve
                    let a_pe = &cd.pe[e as usize];
                    let mut rhs = cd.bpe[e as usize].clone();
                    for i in 0..n_priv {
                        let mut s = 0.0;
                        for j in 0..n_exp {
                            s += a_pe[i * n_exp + j] * x_e[j];
                        }
                        rhs[i] -= s;
                    }
                    lu_solve(lu, n_priv, piv, &mut rhs);
                    let mut off = 0usize;
                    for &b in &cd.private_blocks {
                        let vd = self.trial_element_vdofs(b, e);
                        for (li, &g) in vd.iter().enumerate() {
                            x[g] = rhs[off + li];
                        }
                        off += vd.len();
                    }
                }
            }
            None => {
                x.copy_from_slice(&xs[..n]);
            }
        }
        x
    }

    /// `ComputeResidual(x)`: element-wise DPG residual
    /// `res[e] = ‖L⁻¹(B_e u_e − f_e)‖₂` — the test-norm residual (identical
    /// to MFEM, which stores `L⁻¹B`, `L⁻¹f` and takes the Euclidean norm of
    /// their difference).
    pub fn compute_residual(&self, x: &[f64]) -> Vec<f64> {
        assert!(self.store_matrices, "call store_matrices(true) before assemble()");
        let mut out = Vec::with_capacity(self.stored.len());
        for (e, (yb, yv, n_tr)) in self.stored.iter().enumerate() {
            let mut u = vec![0.0_f64; *n_tr];
            let mut off = 0usize;
            for b in 0..self.trial_spaces.len() {
                let vd = self.trial_element_vdofs(b, e as u32);
                for (li, &g) in vd.iter().enumerate() {
                    u[off + li] = x[g];
                }
                off += vd.len();
            }
            let rows = yb.len() / n_tr;
            let mut r2 = 0.0;
            for k in 0..rows {
                let mut s = -yv[k];
                for j in 0..*n_tr {
                    s += yb[k * n_tr + j] * u[j];
                }
                r2 += s * s;
            }
            out.push(r2.sqrt());
        }
        out
    }

    /// `Update()` after mesh modification.
    ///
    /// **TODO (kernel gap):** the C++ method supports AMR through
    /// `ConformingAssemble` / `BuildProlongation` (block prolongation
    /// `P` / restriction `R` between non-conforming and conforming spaces).
    /// fem-rs does not yet expose hanging-node constraints for arbitrary
    /// trace/volume space combinations, so this port resets the assembled
    /// state only; uniform refinement rebuilds the weak form on the refined
    /// mesh instead.
    pub fn update(&mut self) {
        self.mat = None;
        self.y = None;
        self.assembled = false;
        self.cond_data = None;
        self.reduced_mat = None;
        self.reduced_y = None;
        self.stored.clear();
    }

    /// Offsets of the exposed (trace) blocks within the condensed system.
    pub fn exposed_block_offsets(&self) -> Vec<usize> {
        match &self.cond_data {
            Some(cd) => {
                let mut off = vec![0usize];
                for &b in &cd.exposed_blocks {
                    off.push(off.last().unwrap() + self.trial_block_sizes()[b]);
                }
                off
            }
            None => Vec::new(),
        }
    }

    fn element_type(&self) -> ElementType {
        self.mesh.element_type(0)
    }

    fn trace_face_col_offset(&self, e: u32, trace_block: usize, local_face: usize) -> usize {
        let sk = match &self.trial_spaces[trace_block] {
            TrialSpace::Trace { skeleton } => skeleton,
            _ => unreachable!(),
        };
        (0..local_face).map(|li| sk.dofs_per_face(sk.elem_face_id(e, li))).sum()
    }
}

// ─── Geometry helpers ────────────────────────────────────────────────────────

pub(crate) fn element_geo_at<M: MeshTopology>(
    mesh: &M,
    simplex: Option<&fem_mesh::ElementTransformation>,
    geo_elem: Option<&dyn fem_element::ReferenceElement>,
    geo_nodes: &[u32],
    xi: &[f64],
    dim: usize,
) -> (nalgebra::DMatrix<f64>, f64, Vec<f64>) {
    if let Some(tr) = simplex {
        let xp = tr.map_to_physical(xi);
        (tr.jacobian().clone(), tr.det_j(), xp)
    } else {
        let ge = geo_elem.expect("isoparametric geometry element missing");
        isoparametric_jacobian(mesh, geo_nodes, ge, xi, dim)
    }
}

pub(crate) fn inv_transpose(jac: &nalgebra::DMatrix<f64>, dim: usize) -> nalgebra::DMatrix<f64> {
    let inv = jac.clone().try_inverse().expect("singular element Jacobian");
    let mut jit = nalgebra::DMatrix::<f64>::zeros(dim, dim);
    for r in 0..dim {
        for c in 0..dim {
            jit[(r, c)] = inv[(c, r)];
        }
    }
    jit
}

/// Invert the element geometry map: find reference coordinates `xi` with
/// `x(xi) = target` (Newton, used to evaluate element bases at face
/// quadrature points; the face's global parametrization may be oriented
/// differently from the element's local face, so a direct parametrization
/// map is not sufficient).
#[allow(clippy::too_many_arguments)]
pub(crate) fn invert_element_map<M: MeshTopology>(
    mesh: &M,
    simplex: Option<&fem_mesh::ElementTransformation>,
    geo_elem: Option<&dyn fem_element::ReferenceElement>,
    geo_nodes: &[u32],
    target: &[f64],
    dim: usize,
    xi0: &[f64],
) -> Vec<f64> {
    let mut xi = xi0.to_vec();
    for _ in 0..16 {
        let (jac, _det, xp) = element_geo_at(mesh, simplex, geo_elem, geo_nodes, &xi, dim);
        let mut res2 = 0.0;
        let mut res = vec![0.0_f64; dim];
        for d in 0..dim {
            res[d] = target[d] - xp[d];
            res2 += res[d] * res[d];
        }
        if res2 < 1e-28 {
            break;
        }
        // delta = J⁻¹ res
        let inv = jac.try_inverse().expect("singular Jacobian in face map inversion");
        let mut delta = vec![0.0_f64; dim];
        for r in 0..dim {
            let mut s = 0.0;
            for c in 0..dim {
                s += inv[(r, c)] * res[c];
            }
            delta[r] = s;
        }
        for d in 0..dim {
            xi[d] += delta[d];
        }
    }
    xi
}

/// Face geometry at a reference-face parameter point: physical coordinates,
/// unscaled normal (MFEM `CalcOrtho(J_face)`, Euclidean length = face
/// measure), and the face measure.  Straight (affine) faces only.
pub fn face_geo_at<M: MeshTopology + Clone>(
    mesh: &M,
    sk: &SkeletonSpace<M>,
    f: usize,
    param: &[f64],
    dim: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let fnodes = sk.face_nodes(f);
    let c: Vec<Vec<f64>> = fnodes.iter().map(|&n| mesh.node_coords(n).to_vec()).collect();
    if dim == 2 {
        let s = param[0];
        let x = vec![
            (1.0 - s) * c[0][0] + s * c[1][0],
            (1.0 - s) * c[0][1] + s * c[1][1],
        ];
        let dx = c[1][0] - c[0][0];
        let dy = c[1][1] - c[0][1];
        // CalcOrtho for a segment: (dy, -dx); |n| = measure
        let normal = vec![dy, -dx];
        let measure = (dx * dx + dy * dy).sqrt();
        (x, normal, measure)
    } else if sk.is_quad_face(f) {
        let (s, t) = (param[0], param[1]);
        let x = vec![
            (1.0 - s) * (1.0 - t) * c[0][0]
                + s * (1.0 - t) * c[1][0]
                + s * t * c[2][0]
                + (1.0 - s) * t * c[3][0],
            (1.0 - s) * (1.0 - t) * c[0][1]
                + s * (1.0 - t) * c[1][1]
                + s * t * c[2][1]
                + (1.0 - s) * t * c[3][1],
            (1.0 - s) * (1.0 - t) * c[0][2]
                + s * (1.0 - t) * c[1][2]
                + s * t * c[2][2]
                + (1.0 - s) * t * c[3][2],
        ];
        // ∂x/∂s = (c1-c0)(1-t) + (c2-c3)t ; ∂x/∂t = (c3-c0)(1-s) + (c2-c1)s
        let mut dxs = [0.0f64; 3];
        let mut dxt = [0.0f64; 3];
        for d in 0..3 {
            dxs[d] = (c[1][d] - c[0][d]) * (1.0 - t) + (c[2][d] - c[3][d]) * t;
            dxt[d] = (c[3][d] - c[0][d]) * (1.0 - s) + (c[2][d] - c[1][d]) * s;
        }
        let normal = cross3(&dxs, &dxt);
        let measure = norm3(&normal);
        (x, normal, measure)
    } else {
        let (s, t) = (param[0], param[1]);
        let r = 1.0 - s - t;
        let x = vec![
            r * c[0][0] + s * c[1][0] + t * c[2][0],
            r * c[0][1] + s * c[1][1] + t * c[2][1],
            r * c[0][2] + s * c[1][2] + t * c[2][2],
        ];
        let mut d1 = [0.0f64; 3];
        let mut d2 = [0.0f64; 3];
        for d in 0..3 {
            d1[d] = c[1][d] - c[0][d];
            d2[d] = c[2][d] - c[0][d];
        }
        let normal = cross3(&d1, &d2);
        let measure = norm3(&normal);
        (x, normal, measure)
    }
}

/// MFEM `CalcOrtho` for a 3-D face: the plain cross product `J_s × J_t` of the
/// canonical face Jacobian columns (see `dpg_basis::face_normal_3d`), **without**
/// the 1/2 reference-simplex factor.  Its Euclidean norm equals
/// `Trans.Weight()` (the face surface measure) for both triangular and
/// quadrilateral faces (`|a×b|² = |a|²|b|² − (a·b)²`), which is exactly what
/// the trace integrators multiply their reference weights by.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> Vec<f64> {
    vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(v: &[f64]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Face-DOF parameter coordinates matching `eval_face_lagrange`'s node order.
///
/// * 2-D edge: node `k` at `s = k/p`.
/// * 3-D quad: tensor node `(s,t) = (k%(p+1)/p, k/(p+1)/p)`.
/// * 3-D triangle: the enumeration of `eval_face_lagrange` is row-major over
///   `row = a + b` (`row = 0..=p`, `a = 0..=row`, `b = row − a`), so index `k`
///   satisfies `k = row(row+1)/2 + a` — *not* MFEM's `H1_TriangleElement`
///   ordering `i*(p+1) − i(i−1)/2 + j` used by [`tri_face_dof_index`].
pub fn face_dof_params(dim: usize, is_quad: bool, p: usize, k: usize) -> Vec<f64> {
    if dim == 2 {
        vec![k as f64 / p as f64]
    } else if is_quad {
        let s = k % (p + 1);
        let t = k / (p + 1);
        vec![s as f64 / p as f64, t as f64 / p as f64]
    } else {
        let mut row = 0usize;
        let mut acc = 0usize;
        while acc + (row + 1) <= k {
            acc += row + 1;
            row += 1;
        }
        let a = k - acc;
        let b = row - a;
        let pq = p as f64;
        vec![a as f64 / pq, b as f64 / pq]
    }
}

/// Eliminate rows/cols `dofs` (MFEM `SparseMatrix::EliminateRowCols` with
/// `DIAG_ONE`); returns `(eliminated entries matrix, modified matrix)`.
pub(crate) fn eliminate_row_cols(mat: &CsrMatrix<f64>, dofs: &[usize]) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let n = mat.nrows;
    let ess: std::collections::HashSet<usize> = dofs.iter().copied().collect();
    let mut a_coo = CooMatrix::<f64>::new(n, n);
    let mut e_coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if ess.contains(&i) || ess.contains(&j) {
                e_coo.add(i, j, v);
            } else {
                a_coo.add(i, j, v);
            }
        }
    }
    for &d in dofs {
        a_coo.add(d, d, 1.0);
    }
    (e_coo.into_csr(), a_coo.into_csr())
}

// ─── Block preconditioner ────────────────────────────────────────────────────

/// Block-diagonal preconditioner with one Gauss–Seidel sweep per block —
/// MFEM `BlockDiagonalPreconditioner` of per-block `GSSmoother`s.  Essential
/// dofs carry unit diagonal rows after [`DpgWeakForm::form_linear_system`]
/// (MFEM `DIAG_ONE`), so they need no special treatment here.
pub struct DpgBlockGs {
    /// Diagonal blocks.
    pub blocks: Vec<CsrMatrix<f64>>,
    /// Block offsets.
    pub offsets: Vec<usize>,
}

impl DpgBlockGs {
    /// Build from a formed system and its block offsets.
    pub fn new(sys: &DpgSystem, offsets: &[usize]) -> Self {
        Self::from_matrix(sys.matrix(), offsets)
    }

    /// Build from a plain matrix and block offsets (e.g. the doubled real
    /// operator of a complex DPG system).
    pub fn from_matrix(mat: &CsrMatrix<f64>, offsets: &[usize]) -> Self {
        let nb = offsets.len() - 1;
        let blocks = (0..nb).map(|b| {
            let (r0, r1) = (offsets[b], offsets[b + 1]);
            let mut coo = CooMatrix::<f64>::new(r1 - r0, r1 - r0);
            for i in r0..r1 {
                for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                    let c = mat.col_idx[p] as usize;
                    if (r0..r1).contains(&c) {
                        coo.add(i - r0, c - r0, mat.values[p]);
                    }
                }
            }
            coo.into_csr()
        }).collect();
        Self { blocks, offsets: offsets.to_vec() }
    }

    /// Apply: `z ← M⁻¹ r` with `M = (D+L) D⁻¹ (D+U)` per block (symmetric
    /// Gauss–Seidel — symmetric, hence PCG-compatible).
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        z.fill(0.0);
        for b in 0..self.blocks.len() {
            let (r0, r1) = (self.offsets[b], self.offsets[b + 1]);
            let n = r1 - r0;
            let blk = &self.blocks[b];
            let mut zb = vec![0.0_f64; n];
            // forward: (D + L) t = r
            for i in 0..n {
                let mut s = r[r0 + i];
                let mut diag = 1.0;
                for p in blk.row_ptr[i]..blk.row_ptr[i + 1] {
                    let c = blk.col_idx[p] as usize;
                    let v = blk.values[p];
                    if c < i {
                        s -= v * zb[c];
                    } else if c == i {
                        diag = v;
                    }
                }
                zb[i] = if diag.abs() > 1e-300 { s / diag } else { 0.0 };
            }
            // t ← D t
            for i in 0..n {
                let mut diag = 1.0;
                for p in blk.row_ptr[i]..blk.row_ptr[i + 1] {
                    if blk.col_idx[p] as usize == i {
                        diag = blk.values[p];
                        break;
                    }
                }
                zb[i] *= diag;
            }
            // backward: (D + U) z = t
            for i in (0..n).rev() {
                let mut s = zb[i];
                let mut diag = 1.0;
                for p in blk.row_ptr[i]..blk.row_ptr[i + 1] {
                    let c = blk.col_idx[p] as usize;
                    let v = blk.values[p];
                    if c > i {
                        s -= v * zb[c];
                    } else if c == i {
                        diag = v;
                    }
                }
                zb[i] = if diag.abs() > 1e-300 { s / diag } else { 0.0 };
            }
            z[r0..r1].copy_from_slice(&zb);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpg::dpg_integrators::{
        DpgDiffusionIntegrator, DpgDomainLFIntegrator, DpgMassIntegrator,
        DpgMixedScalarWeakGradientIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
        DpgTraceIntegrator,
    };
    use fem_mesh::Mesh;

    /// Adapter: `-(σ, τ)` for vector-L2 σ and H(div) τ — MFEM
    /// `TransposeIntegrator(VectorFEMassIntegrator(-1))`.
    struct NegVectorMass;
    impl DpgBilinear2 for NegVectorMass {
        fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
            let d = ctx.dim;
            let nt = test.n_scalar;
            let nsc = trial.n_scalar;
            let nc = trial.n_expanded;
            for k in 0..nt {
                for c in 0..d {
                    for j in 0..nsc {
                        m[k * nc + c * nsc + j] -= ctx.w * test.phi[k * d + c] * trial.phi[j];
                    }
                }
            }
        }
    }

    /// Ultraweak DPG for `-Δu = 1` on the unit square (C++ `diffusion.cpp`,
    /// prob 0): trial u (L2 p−1), σ (vector L2 p−1), û (trace p), σ̂ (trace
    /// p−1); test τ (RT p−1), v (H1 p); "space-induced" graph norm.
    #[test]
    fn poisson_dpg_2d_quad_p1_assembles() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let p = 1u8;
        let mut a: DpgWeakForm<Mesh<2>> = DpgWeakForm::new(mesh);
        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space(p);
        let hatsig = a.add_trial_trace_space(p - 1);
        let tau = a.add_test_space(VolKind::HDiv, p - 1 + 1); // RT_{p+δ-1}
        let v = a.add_test_space(VolKind::Scalar, p + 1); // H1_{p+δ}

        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
            u,
            tau,
        );
        a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);

        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgDivDivIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgVectorFEMassIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
        a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
        a.add_domain_lf_integrator(Box::new(DpgDomainLFIntegrator { f: |_| 1.0 }), v);

        a.store_matrices(true);
        a.assemble();
        assert_eq!(a.n_trial_blocks(), 4);

        // essential BCs: all boundary dofs of û
        let hat_base = a.trial_offsets()[hatu];
        let sk = a.skeleton(hatu);
        let mut ess = Vec::new();
        for f in 0..sk.n_faces() {
            if sk.is_boundary_face(f) {
                for &d in sk.face_dof_list(f) {
                    ess.push(hat_base + d);
                }
            }
        }
        assert!(!ess.is_empty());
        let x0 = vec![0.0_f64; a.size()];
        let (sys, _xs, b) = a.form_linear_system(&ess, &x0, false);
        assert_eq!(sys.size(), a.size());
        assert!(b.iter().any(|&v| v != 0.0), "rhs must be nonzero");

        // The normal-equation matrix must be symmetric.
        {
            let m = sys.matrix();
            let mut asym = 0.0_f64;
            for i in 0..m.nrows {
                for p in m.row_ptr[i]..m.row_ptr[i + 1] {
                    let j = m.col_idx[p] as usize;
                    let vij = m.values[p];
                    let mut vji = 0.0_f64;
                    for q in m.row_ptr[j]..m.row_ptr[j + 1] {
                        if m.col_idx[q] as usize == i {
                            vji = m.values[q];
                            break;
                        }
                    }
                    asym = asym.max((vij - vji).abs());
                }
            }
            eprintln!("system asymmetry: {asym:.3e}");
            assert!(asym < 1e-11, "system matrix must be symmetric, got {asym:.3e}");
        }

        // residuals on the zero solution must be finite/positive
        let res = a.compute_residual(&x0);
        assert_eq!(res.len(), a.mesh().n_elements());
        assert!(res.iter().all(|&r| r.is_finite() && r >= 0.0));

        // static condensation path
        let mut a2: DpgWeakForm<Mesh<2>> = build_same(Mesh::<2>::unit_square_quad(2));
        a2.enable_static_condensation();
        a2.assemble();
        let hat_base2 = a2.trial_offsets()[hatu];
        let sk2 = a2.skeleton(hatu);
        let mut ess2 = Vec::new();
        for f in 0..sk2.n_faces() {
            if sk2.is_boundary_face(f) {
                for &d in sk2.face_dof_list(f) {
                    ess2.push(hat_base2 + d);
                }
            }
        }
        let (sys2, _xs2, _b2) = a2.form_linear_system(&ess2, &x0, false);
        let full = a2.size();
        assert!(sys2.size() < full, "Schur system must be smaller");
    }

    /// Exact-solution consistency: for the Poisson ultraweak form, the exact
    /// solution (u, σ = ∇u, û = u|ₛ, σ̂ = −σ·n) must satisfy the assembled
    /// normal equations `A x = b` element-wise (before BC elimination).
    #[test]
    fn poisson_dpg_exact_solution_satisfies_system() {
        use fem_element::{ReferenceElement, VectorReferenceElement};
        use fem_mesh::MeshTopology as MT;

        let _pi = std::f64::consts::PI;
        let mesh = Mesh::<2>::unit_square_quad(2);
        let p = 1u8;
        let mut a: DpgWeakForm<Mesh<2>> = DpgWeakForm::new(mesh.clone());
        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space(p);
        let hatsig = a.add_trial_trace_space(p - 1);
        let tau = a.add_test_space(VolKind::HDiv, p);
        let v = a.add_test_space(VolKind::Scalar, p + 1);
        a.add_trial_integrator(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }), u, tau);
        a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);
        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgDivDivIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgVectorFEMassIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
        a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
        a.add_domain_lf_integrator(Box::new(DpgDomainLFIntegrator { f: |_x: &[f64]| 0.0 }), v);
        a.assemble();

        let et = mesh.element_type(0);
        let n_scalar = scalar_ref_elem(et, p - 1).n_dofs();
        let dof_par = scalar_ref_elem(et, p - 1).dof_coords();
        // Constant solution u ≡ 1: (u, σ = 0, û = 1, σ̂ = 0, f = 0) satisfies
        // the first-order system and every face identity exactly, so the
        // assembled normal equations must hold to machine precision.
        let exact_u = |_x: &[f64]| 1.0_f64;

        let mut x = vec![0.0_f64; a.size()];
        let offs = a.trial_offsets();

        // u and σ blocks: nodal interpolation at the L2 dof points.
        for e in 0..mesh.n_elements() as u32 {
            for (k, xi) in dof_par.iter().enumerate() {
                let (_jac, _det, xp) = {
                    let geo = crate::vector_assembler::geo_ref_elem_from_mesh(&mesh, e).unwrap();
                    let gnodes = mesh.geometry_nodes(e).to_vec();
                    crate::vector_assembler::isoparametric_jacobian(
                        &mesh,
                        &gnodes,
                        geo.as_ref(),
                        xi,
                        2,
                    )
                };
                x[offs[u] + e as usize * n_scalar + k] = exact_u(&xp);
            }
        }

        // û block: u on every skeleton face; σ̂ block: −σ·n̂ = 0.
        // `face_dof_list` (not the `face_dofs` range) is the accessor valid in
        // the continuous H1-trace mode: there the corner dofs are mesh-vertex
        // ids, so a face's global dofs are not a consecutive range.
        let sk = a.skeleton(hatu);
        let hatu_base = offs[hatu];
        for f in 0..sk.n_faces() {
            let dofs = sk.face_dof_list(f).to_vec();
            for (k, &d) in dofs.iter().enumerate() {
                let pt = a.face_dof_point(&sk, f, k);
                x[hatu_base + d] = exact_u(&pt);
            }
        }

        // A x − b must vanish.
        let mat = a.mat.as_ref().unwrap();
        let b = a.y.as_ref().unwrap();
        let mut ax = vec![0.0_f64; mat.nrows];
        mat.spmv(&x, &mut ax);
        let mut worst = 0.0_f64;
        for i in 0..ax.len() {
            worst = worst.max((ax[i] - b[i]).abs());
        }
        eprintln!("poisson DPG exact-solution residual: {worst:.3e}");
        assert!(worst < 1e-9, "exact solution must satisfy A x = b, got {worst:.3e}");
        let _ = (u, sig, hatu, hatsig, tau, v, &dof_par);
    }

    /// Per-element probe: recompute element `e`'s `A_e x_e − b_e` blocks per
    /// trial block pair to localize assembly inconsistencies.
    #[test]
    fn poisson_dpg_elementwise_debug() {
        use fem_mesh::MeshTopology as MT;

        let pi = std::f64::consts::PI;
        let mesh = Mesh::<2>::unit_square_quad(2);
        let p = 1u8;
        let mut a: DpgWeakForm<Mesh<2>> = DpgWeakForm::new(mesh.clone());
        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space(p);
        let hatsig = a.add_trial_trace_space(p - 1);
        let tau = a.add_test_space(VolKind::HDiv, p);
        let v = a.add_test_space(VolKind::Scalar, p + 1);
        a.add_trial_integrator(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }), u, tau);
        a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);
        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgDivDivIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgVectorFEMassIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
        a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
        a.add_domain_lf_integrator(Box::new(DpgDomainLFIntegrator { f: |_x: &[f64]| 0.0 }), v);
        a.store_matrices(true);
        a.assemble();

        // Element residual r_e = A_e x_e − b_e via the scattered global
        // system restricted to element-private rows (u, σ blocks are
        // element-private!): rows of the u/σ blocks give exactly the
        // element's contribution.
        let et = mesh.element_type(0);
        let n_scalar = scalar_ref_elem(et, p - 1).n_dofs();
        let dof_par = scalar_ref_elem(et, p - 1).dof_coords();
        // u == 1, sigma == 0, f == 0: the residual then tests exactly the
        // divergence-theorem identity -(1, div tau) + <1, tau·n> = 0.
        let mut x = vec![0.0_f64; a.size()];
        let offs = a.trial_offsets();
        for e in 0..mesh.n_elements() as u32 {
            for (k, _xi) in dof_par.iter().enumerate() {
                x[offs[u] + e as usize * n_scalar + k] = 1.0;
            }
        }
        let sk = a.skeleton(hatu);
        for f in 0..sk.n_faces() {
            for &d in sk.face_dof_list(f) {
                x[offs[hatu] + d] = 1.0;
            }
        }

        let mat = a.mat.as_ref().unwrap();
        let b = a.y.as_ref().unwrap();
        let mut ax = vec![0.0_f64; mat.nrows];
        mat.spmv(&x, &mut ax);

        // Print the raw global residual on the element-private σ rows.
        for e in 0..mesh.n_elements() as u32 {
            for c in 0..2 {
                for k in 0..n_scalar {
                    let g = offs[sig] + e as usize * n_scalar * 2 + c * n_scalar + k;
                    let w = ax[g] - b[g];
                    if w.abs() > 1e-12 {
                        eprintln!("  sigma resid elem {e} comp {c} dof {k}: {w:.6e}");
                    }
                }
            }
        }

        // Weak form's own per-element residual: z = L⁻¹(B x_e − f_e) from the
        // stored L⁻¹B / L⁻¹f.
        for (e, (yb, yv, n_tr)) in a.stored.iter().enumerate() {
            let mut u_e = vec![0.0_f64; *n_tr];
            let mut off = 0usize;
            for blk in 0..a.trial_spaces.len() {
                let vd = a.trial_element_vdofs(blk, e as u32);
                for (li, &g) in vd.iter().enumerate() {
                    u_e[off + li] = x[g];
                }
                off += vd.len();
            }
            let rows = yb.len() / n_tr;
            let mut worst = 0.0f64;
            let mut wi = 0usize;
            for k in 0..rows {
                let mut s = -yv[k];
                for j in 0..*n_tr {
                    s += yb[k * n_tr + j] * u_e[j];
                }
                if s.abs() > worst {
                    worst = s.abs();
                    wi = k;
                }
            }
            let n_tau_dofs = crate::dpg::dpg_basis::vector_ref_elem(et, p).n_dofs();
            eprintln!(
                "elem {e}: |L⁻¹(Bx−f)| worst {worst:.3e} at row {wi} ({} = tau block, {} = v block)",
                wi < n_tau_dofs,
                wi >= n_tau_dofs
            );
        }

        // Direct diagnostic: r_tau[e] = -(1, div tau)_T + sum_faces scale *
        // int_face tau·n ds, computed independently of the weak form.
        let tau_order = p; // HDiv test order
        let (qpts, qwts) = vol_quadrature(et, 8);
        let n_tau = crate::dpg::dpg_basis::vector_ref_elem(et, tau_order).n_dofs();
        let skh = a.skeleton(hatu);
        for e in 0..mesh.n_elements() as u32 {
            let nodes = mesh.element_nodes(e);
            let tr = fem_mesh::ElementTransformation::from_simplex_nodes(&mesh, nodes);
            let mut r_tau = vec![0.0_f64; n_tau];
            // volume term
            let mut tv = VolVals::default();
            for (q, xi) in qpts.iter().enumerate() {
                let (jac, det, _xp) = if matches!(et, ElementType::Tri3) {
                    let xp = tr.map_to_physical(xi);
                    (tr.jacobian().clone(), tr.det_j(), xp)
                } else {
                    let geo = crate::vector_assembler::geo_ref_elem_from_mesh(&mesh, e).unwrap();
                    let gnodes = mesh.geometry_nodes(e).to_vec();
                    crate::vector_assembler::isoparametric_jacobian(
                        &mesh, &gnodes, geo.as_ref(), xi, 2,
                    )
                };
                let jit = inv_transpose(&jac, 2);
                eval_vol_space(VolKind::HDiv, tau_order, et, 2, &jac, det, &jit, xi, None, &mut tv);
                for i in 0..n_tau {
                    r_tau[i] -= qwts[q] * det.abs() * tv.div[i];
                }
            }
            // face terms
            let lfs = local_face_table(&nodes, 2);
            for (li, lf) in lfs.iter().enumerate() {
                let fid = skh.elem_face_id(e, li);
                // Same canonical-orientation convention as the assembly.
                let scale = skh.elem_face_orientation(e, li) as f64;
                let (fpts, fwts) = crate::dpg::dpg_basis::face_quadrature(2, false, 6);
                let mut tv2 = VolVals::default();
                for (q, fparam) in fpts.iter().enumerate() {
                    let (xp, normal, _measure) = face_geo_at(&mesh, &skh, fid, fparam, 2);
                    let xi0 = crate::dpg::dpg_basis::face_param_to_elem_ref(et, lf, false, fparam);
                    let geo_dbg = crate::vector_assembler::geo_ref_elem_from_mesh(&mesh, e).unwrap();
                    let gnodes_dbg = mesh.geometry_nodes(e).to_vec();
                    let xiref = invert_element_map(
                        &mesh,
                        None::<&fem_mesh::ElementTransformation>,
                        Some(geo_dbg.as_ref()),
                        &gnodes_dbg,
                        &xp,
                        2,
                        &xi0,
                    );
                    let (jac, det, _x2) = if matches!(et, ElementType::Tri3) {
                        let tr2 = fem_mesh::ElementTransformation::from_simplex_nodes(&mesh, nodes);
                        let xp2 = tr2.map_to_physical(&xiref);
                        (tr2.jacobian().clone(), tr2.det_j(), xp2)
                    } else {
                        let geo = crate::vector_assembler::geo_ref_elem_from_mesh(&mesh, e).unwrap();
                        let gnodes = mesh.geometry_nodes(e).to_vec();
                        crate::vector_assembler::isoparametric_jacobian(
                            &mesh, &gnodes, geo.as_ref(), &xiref, 2,
                        )
                    };
                    let jit = inv_transpose(&jac, 2);
                    eval_vol_space(VolKind::HDiv, tau_order, et, 2, &jac, det, &jit, &xiref, None, &mut tv2);
                    let flux = tv2.phi[0] * normal[0] + tv2.phi[1] * normal[1];
                    let _ = xp;
                    for i in 0..n_tau {
                        let fi = tv2.phi[i * 2] * normal[0] + tv2.phi[i * 2 + 1] * normal[1];
                        r_tau[i] += fwts[q] * scale * fi;
                        let _ = flux;
                    }
                }
            }
            let worst = r_tau.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            if worst > 1e-12 {
                eprintln!("elem {e}: r_tau worst {worst:.3e}");
                for (li, _) in lfs.iter().enumerate() {
                    let fid = skh.elem_face_id(e, li);
                    eprintln!(
                        "   face {fid} boundary={} nodes {:?}",
                        skh.is_boundary_face(fid),
                        skh.face_nodes(fid)
                    );
                }
            }
        }
        let _ = (&ax, &b, dof_par, n_scalar, &offs, u);
        // sigma-block rows (byNODES)
        for e in 0..mesh.n_elements() as u32 {
            let mut w = 0.0_f64;
            for c in 0..2 {
                for k in 0..n_scalar {
                    let g = offs[sig] + e as usize * n_scalar * 2 + c * n_scalar + k;
                    w = w.max((ax[g] - b[g]).abs());
                }
            }
            eprintln!("elem {e}: sigma-block residual {w:.3e}");
        }
    }

    /// 3-D trace orientation convention: for every element and every local
    /// face, `scale · n_canonical` (exactly what the trace assemblers use:
    /// `ctx.normal` times `ctx.scale`) must be the element's **outward**
    /// normal.  MFEM guarantees this by storing each face in its generating
    /// element's local face cycle with the outward winding
    /// (`Geometry::Constants<ElemType>::FaceVert`), so any local face table
    /// whose cycles are not outward-consistent silently flips the sign of the
    /// trace contributions on the affected faces.
    #[test]
    fn dpg_3d_canonical_face_normal_is_outward() {
        for (mesh, name) in [
            (Mesh::<3>::unit_cube_hex(1), "hex"),
            (Mesh::<3>::unit_cube_tet(1), "tet"),
        ] {
            let sk = SkeletonSpace::new(mesh.clone(), 1);
            for e in 0..mesh.n_elements() as u32 {
                let nodes = mesh.element_nodes(e);
                let center: Vec<f64> = (0..3)
                    .map(|d| {
                        nodes.iter().map(|&n| mesh.node_coords(n)[d]).sum::<f64>()
                            / nodes.len() as f64
                    })
                    .collect();
                let lfs = local_face_table(&nodes, 3);
                for li in 0..lfs.len() {
                    let fid = sk.elem_face_id(e, li);
                    let scale = sk.elem_face_orientation(e, li) as f64;
                    let is_qf = sk.is_quad_face(fid);
                    let param = if is_qf { vec![0.5, 0.5] } else { vec![1.0 / 3.0, 1.0 / 3.0] };
                    let (xp, normal, _m) = face_geo_at(&mesh, &sk, fid, &param, 3);
                    let dot: f64 = (0..3).map(|d| normal[d] * (xp[d] - center[d])).sum();
                    assert!(
                        scale * dot > 0.0,
                        "{name} elem {e} local face {li} (global {fid}): scale {scale} \
                         flips the canonical normal away from the outward direction \
                         (n·(x_face − x_elem) = {dot:.3e})"
                    );
                }
            }
        }
    }

    /// `face_geo_at` 3-D normal/measure must equal MFEM `CalcOrtho(J_face)`
    /// (= `J_s × J_t`, no 1/2 factor) and the surface measure
    /// `|J_s × J_t| = Trans.Weight()`: the divergence theorem
    /// `∫_Ω ∇·w dV = ∫_∂Ω w·n dS` is the analytic identity that pins both.
    #[test]
    fn dpg_3d_face_measure_matches_divergence_theorem() {
        // w(x) = x  →  ∇·w = 3,  ∫_∂Ω (x·n) dS = 3·|Ω| = 3 for the unit cube.
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let sk = SkeletonSpace::new(mesh.clone(), 1);
        let (fpts, fwts) = crate::dpg::dpg_basis::face_quadrature(3, true, 3);
        let mut flux = 0.0_f64;
        for f in 0..sk.n_faces() {
            if !sk.is_boundary_face(f) {
                continue;
            }
            for (qi, p) in fpts.iter().enumerate() {
                let (xp, normal, measure) = face_geo_at(&mesh, &sk, f, p, 3);
                assert!(
                    (norm3(&normal) - measure).abs() < 1e-13,
                    "measure must equal |CalcOrtho(J_face)|"
                );
                flux += fwts[qi] * (xp[0] * normal[0] + xp[1] * normal[1] + xp[2] * normal[2]);
            }
        }
        assert!(
            (flux - 3.0).abs() < 1e-12,
            "divergence theorem: ∫_∂Ω x·n dS = {flux} (expected 3 = 3·|Ω|); \
             the 1/2 factor in `cross_half` halves this"
        );
    }

    /// Face param ↔ element-ref ↔ physical consistency on all geometries.
    #[test]
    fn face_param_element_ref_consistency() {
        {
            let mesh = Mesh::<2>::unit_square_quad(2);
            let et = mesh.element_type(0);
            let sk = SkeletonSpace::new(mesh.clone(), 1);
            for e in 0..mesh.n_elements() as u32 {
                let nodes = mesh.element_nodes(e);
                let lfs = local_face_table(&nodes, 2);
                for (li, _lf) in lfs.iter().enumerate() {
                    let fid = sk.elem_face_id(e, li);
                    let param = [0.37f64];
                    let (xp_face, _, _) = face_geo_at(&mesh, &sk, fid, &param, 2);
                    let xi0 = crate::dpg::dpg_basis::face_param_to_elem_ref(
                        et,
                        &lfs[li],
                        false,
                        &param,
                    );
                    let geo = crate::vector_assembler::geo_ref_elem_from_mesh(&mesh, e).unwrap();
                    let gnodes = mesh.geometry_nodes(e).to_vec();
                    let xiref = invert_element_map(
                        &mesh,
                        None::<&fem_mesh::ElementTransformation>,
                        Some(geo.as_ref()),
                        &gnodes,
                        &xp_face,
                        2,
                        &xi0,
                    );
                    let (_jac, _det, xp_el) = crate::vector_assembler::isoparametric_jacobian(
                        &mesh, &gnodes, geo.as_ref(), &xiref, 2,
                    );
                    let d: f64 = xp_face.iter().zip(xp_el.iter()).map(|(a, b)| (a - b).abs()).sum();
                    assert!(
                        d < 1e-12,
                        "quad elem {e} local {li}: face point {xp_face:?} vs element-mapped {xp_el:?}"
                    );
                }
            }
        }
        {
            let mesh = Mesh::<2>::unit_square_tri(2);
            let et = mesh.element_type(0);
            let sk = SkeletonSpace::new(mesh.clone(), 1);
            for e in 0..mesh.n_elements() as u32 {
                let nodes = mesh.element_nodes(e);
                let lfs = local_face_table(&nodes, 2);
                for (li, _lf) in lfs.iter().enumerate() {
                    let fid = sk.elem_face_id(e, li);
                    let param = [0.37f64];
                    let (xp_face, _, _) = face_geo_at(&mesh, &sk, fid, &param, 2);
                    let xi0 =
                        crate::dpg::dpg_basis::face_param_to_elem_ref(et, &lfs[li], false, &param);
                    let tr = fem_mesh::ElementTransformation::from_simplex_nodes(&mesh, nodes);
                    let xiref = invert_element_map(&mesh, Some(&tr), None, &[], &xp_face, 2, &xi0);
                    let xp_el = tr.map_to_physical(&xiref);
                    let d: f64 = xp_face.iter().zip(xp_el.iter()).map(|(a, b)| (a - b).abs()).sum();
                    assert!(
                        d < 1e-12,
                        "tri elem {e} local {li}: face point {xp_face:?} vs element-mapped {xp_el:?}"
                    );
                }
            }
        }
    }

    fn build_same(mesh: Mesh<2>) -> DpgWeakForm<Mesh<2>> {
        let p = 1u8;
        let mut a: DpgWeakForm<Mesh<2>> = DpgWeakForm::new(mesh);
        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space(p);
        let hatsig = a.add_trial_trace_space(p - 1);
        let tau = a.add_test_space(VolKind::HDiv, p);
        let v = a.add_test_space(VolKind::Scalar, p + 1);
        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
            u,
            tau,
        );
        a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);
        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgDivDivIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(
            Box::new(crate::dpg::dpg_integrators::DpgVectorFEMassIntegrator { q: 1.0 }),
            tau,
            tau,
        );
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
        a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
        a.add_domain_lf_integrator(Box::new(DpgDomainLFIntegrator { f: |_| 1.0 }), v);
        a
    }
}

/// Public re-exports of the dense LU helpers for miniapps.
pub fn lu_factor_public(a: &mut [f64]) -> Vec<usize> {
    let n = (a.len() as f64).sqrt().round() as usize;
    lu_factor(a, n)
}

/// Solve `A x = b` with an [`lu_factor_public`] factorization (in place).
pub fn lu_solve_public(lu: &[f64], piv: &[usize], b: &mut [f64]) {
    let n = (lu.len() as f64).sqrt().round() as usize;
    lu_solve(lu, n, piv, b);
}
