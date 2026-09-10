//! H(div) saddle-point linear solver — serial cut of MFEM
//! `miniapps/hdiv-linear-solver/` shared sources
//! (`hdiv_linear_solver.{hpp,cpp}` + `discrete_divergence.{hpp,cpp}` +
//! `change_basis.{hpp,cpp}`).
//!
//! Solves the H(div) saddle-point system (MFEM `HdivSaddlePointSolver`)
//!
//! ```text
//!     [  L     D  ] [ p ]   [ f ]
//!     [ Dᵀ    -R  ] [ q ] = [ g ]
//! ```
//!
//! where `L` is the L² mass matrix with coefficient `L_coeff`, `R` is the
//! RT (H(div)) mass matrix with coefficient `R_coeff`, and `D` is the
//! discrete divergence (`∫ q div(u) dx`, L² rows × RT columns).  `Mode::Darcy`
//! leaves `D` unweighted (mixed Poisson/Darcy); `Mode::GradDiv` corresponds to
//! the grad-div problem `α u − grad(β div u) = f` (in this serial cut the
//! β-scaling of the divergence blocks is implicit in β = 1, the value used by
//! the `grad_div` miniapp).
//!
//! MINRES is preconditioned by the block-diagonal factor
//! `diag(AMG(S), Jacobi(R))` with the approximate Schur complement
//! `S = D·diag(R)⁻¹·Dᵀ + diag(L)⁻¹` (the (1,1) term is dropped when the L²
//! block is identically zero, i.e. the Darcy α = 0 case).
//!
//! # Deviations from the C++ original (fem-rs serial, matrix-based)
//!
//! * **Change of basis is the identity** (`change_basis.{hpp,cpp}` is
//!   subsumed).  The C++ solver assembles its internal operators in
//!   IntegratedGLL / INTEGRAL-map-type spaces and applies
//!   `ChangeOfBasis_L2` / `ChangeOfBasis_RT` (per-element tensor kernels) to
//!   move between the user's basis and the solver-internal basis; the
//!   divergence matrix even becomes a pure ±1 incidence matrix there.  None of
//!   that machinery exists in fem-rs (single-basis `L2Space`/`HDivSpace`, no
//!   INTEGRAL map type), and it is *exact bookkeeping*, not an algorithmic
//!   step: this port assembles `L`, `R`, `D` once, in the fem-rs basis, and
//!   solves the system directly.  Consequently the `L_inv` congruence
//!   transformation of the C++ `Mult` is not applied either (it only
//!   renormalizes the (1,1) block); the MINRES iterant differs, the discrete
//!   solution does not.
//! * **No `detJ²`/`detJ` QF rescaling** (C++ `convert_map_type` path): those
//!   factors exist solely to convert between VALUE- and INTEGRAL-map-type L²
//!   spaces.  fem-rs L² spaces are VALUE-mapped only, so the projected
//!   coefficient is used as-is.
//! * **QuadratureFunction usage** (the #9 unlock this port exercises):
//!   - MFEM `L_coeff.Project(W_coeff_qf)` → [`QuadratureFunction`] filled by
//!     evaluating the coefficient at every quadrature point of the
//!     [`QuadratureSpace`] built on the mass integration rule
//!     (`QuadratureSpace::new` + `map_quadrature_points`).
//!   - MFEM `W_mix_coeff_qf = W_coeff_qf` / `= 1.0` → `QuadratureFunction::
//!     assign` / `QuadratureFunction::fill`.
//!   - MFEM `MassIntegrator(W_coeff)` (QF coefficient) → element-local
//!     assembly that consumes the QF values per element via
//!     `QuadratureFunction::get_values` on the space's own integration rules
//!     ([`assemble_l2_mass_qf`]).  The C++ `DGMassInverse` and `RAPOperator`
//!     congruence pieces exist only for that (subsumed) transformation and
//!     have no serial analogue here.
//!   - MFEM `det_J_qf = geom->detJ` → the geometric factors are carried by
//!     `QuadratureSpace::get_weights` (rule weight × |det J| per QP).
//! * **Schur (1,1) term**: C++ adds `diag(L)·d⁻²`-scaled diagonals (a basis
//!   artifact); this port adds `diag(L)⁻¹`, the standard mass-Schur
//!   approximation — the two coincide for the constant β = 1 used by the
//!   miniapps.
//! * hypre `BoomerAMG` on `S` → `fem-amg` smoothed-aggregation AMG.
//! * 2-D only (C++ also supports 3-D): the serial harness and validation runs
//!   use `star.mesh`; fem-rs `HDivSpace`/`L2Space` do cover 3-D but the
//!   boundary-flux assembly helpers validated here are 2-D.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fem_amg::{AmgConfig, AmgSolver};
use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::qfunction::QuadratureFunction;
use fem_assembly::qspace::{QuadratureSpace, QuadratureSpaceBase};
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_element::lagrange::factory::{QuadL2GL, TriPk};
use fem_element::reference::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::darcy_solvers::{mfem_minres, IterSolveParameters};
use fem_space::fe_space::FESpace;
use fem_space::hdiv::HDivSpace;
use fem_space::l2::L2Space;

/// Which type of saddle-point problem is being solved (MFEM
/// `HdivSaddlePointSolver::Mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Grad-div problem (`α u − grad(β div u) = f`).
    GradDiv,
    /// Darcy/mixed-Poisson problem (`α p − div(β grad p) = f`; α may be 0).
    Darcy,
}

/// Solver parameters (MFEM `HdivSaddlePointSolver::Setup` MINRES settings:
/// `rel_tol = 1e-12`, `abs_tol = 0`, `max_iter = 500`).
fn minres_params() -> IterSolveParameters {
    IterSolveParameters {
        print_level: 0,
        max_iter: 500,
        abs_tol: 0.0,
        rel_tol: 1e-12,
    }
}

/// Serial port of MFEM `HdivSaddlePointSolver` (see the module docs for the
/// serial cut deviations).
pub struct HdivSaddlePointSolver {
    n_l2: usize,
    n_rt: usize,
    /// Whether the L² block is identically zero (Darcy α = 0).
    zero_l2_block: bool,
    /// L² mass with the QF-projected coefficient `W` (absent when zero block).
    l: Option<CsrMatrix<f64>>,
    /// Discrete divergence with the essential-RT columns eliminated.
    d: CsrMatrix<f64>,
    /// Its transpose (kept to avoid re-transposing inside `apply_op`).
    dt: CsrMatrix<f64>,
    /// Discrete divergence without BC elimination (MFEM `D_e`).
    d_e: CsrMatrix<f64>,
    /// RT mass with essential rows/columns eliminated to the identity.
    r: CsrMatrix<f64>,
    /// RT mass without BC elimination (MFEM `R_e`).
    r_e: CsrMatrix<f64>,
    /// AMG hierarchy on the approximate Schur complement
    /// `S = D·diag(R)⁻¹·Dᵀ (+ L)` (MFEM `S_inv`, hypre BoomerAMG).
    s_amg: AmgSolver<f64>,
    /// `diag(R)⁻¹` with entries 1 at essential dofs (MFEM `R_inv` Jacobi).
    r_inv_diag: Vec<f64>,
    ess_rt_dofs: Vec<usize>,
    /// Prescribed essential values (MFEM `x_bc`, set via [`Self::set_bc`]).
    x_bc: Vec<f64>,
    param: IterSolveParameters,
    last_iters: AtomicUsize,
    last_converged: AtomicBool,
}

impl HdivSaddlePointSolver {
    /// Build the solver (MFEM constructor + `Setup()`).
    ///
    /// `l2` / `rt` must live on `mesh`; `ess_rt_dofs` lists the essential
    /// (Dirichlet-normal) RT dofs — pass the boundary dofs before calling
    /// [`Self::mult`], and call [`Self::set_bc`] with the prescribed values.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh: &Mesh<2>,
        l2: &L2Space<Mesh<2>>,
        rt: &HDivSpace<Mesh<2>>,
        l_coeff: f64,
        r_coeff: f64,
        ess_rt_dofs: &[usize],
        mode: Mode,
    ) -> Self {
        let n_l2 = l2.n_dofs();
        let n_rt = rt.n_dofs();
        let zero_l2_block = mode == Mode::Darcy && l_coeff == 0.0;
        if mode == Mode::GradDiv {
            assert!(
                !zero_l2_block,
                "Mode::GRAD_DIV incompatible with zero coefficient."
            );
        }

        // MFEM `qs(mesh, GetMassIntRule(fes_l2))`: the quadrature space lives
        // on the L2 mass-integration rule; we take fem-rs' per-geometry rule
        // exact for 2·k+1 degrees (k = L2 order).
        let fe_order = l2.order() as usize;
        let qo = (2 * fe_order + 1).max(2) as i32;
        let qs = QuadratureSpace::new(mesh, qo);

        // MFEM `QuadratureFunction W_coeff_qf(qs)` + `L_coeff.Project(qf)`.
        // The coefficient is the constant `l_coeff` (the miniapps pass
        // `ConstantCoefficient`s), so the projected values are constant; the
        // per-element point mapping is kept to mirror `Coefficient::Project`.
        let mut w_qf = QuadratureFunction::new(&qs);
        if !zero_l2_block {
            let sdim = 2_usize;
            let mut xs: Vec<f64> = Vec::new();
            let mut det: Vec<f64> = Vec::new();
            for e in 0..qs.get_ne() {
                let nqp = qs.get_int_rule(e).n_points();
                xs.resize(nqp * sdim, 0.0);
                det.resize(nqp, 0.0);
                qs.map_quadrature_points(e, &mut xs, &mut det);
                let vals = w_qf.get_values_mut(e);
                vals.fill(l_coeff);
            }
        }

        // MFEM `W_mix_coeff_qf`: scaled by the coefficient in grad-div mode,
        // unweighted in Darcy mode.
        let mut w_mix_qf = QuadratureFunction::new(&qs);
        match mode {
            Mode::GradDiv => w_mix_qf.assign(w_qf.values()),
            Mode::Darcy => w_mix_qf.fill(1.0),
        }

        // L² mass with the QF coefficient (MFEM `MassIntegrator(W_coeff)`).
        let l = if zero_l2_block {
            None
        } else {
            Some(assemble_l2_mass_qf(l2, &qs, &w_qf))
        };

        // RT mass with the R coefficient (MFEM `VectorFEMassIntegrator`).
        let r_full = VectorAssembler::assemble_bilinear(
            rt,
            &[&VectorMassIntegrator { alpha: r_coeff }],
            (2 * rt.order() as usize + 1).max(2) as u8,
        );

        // Discrete divergence (MFEM `FormDiscreteDivergenceMatrix`).
        let d_e = assemble_hdiv_l2_mixed(
            l2,
            rt,
            &[&HDivL2DivIntegrator],
            (2 * fe_order + 1).max(2) as u8,
        );
        let d = eliminate_columns(&d_e, ess_rt_dofs);
        let dt = d.transpose();

        // MFEM `mass_rt.FormSystemMatrix(ess, R)` / `R_e`.
        let r = eliminate_rows_cols(&r_full, ess_rt_dofs);
        let r_e = r_full.clone();

        // MFEM `R_diag` (ess entries set to 1) + `Reciprocal`.
        let mut r_inv_diag: Vec<f64> = (0..n_rt)
            .map(|j| {
                let dj = r_full.get(j, j);
                if dj.abs() < 1e-300 { 1.0 } else { 1.0 / dj }
            })
            .collect();
        for &i in ess_rt_dofs {
            r_inv_diag[i] = 1.0;
        }

        // Approximate Schur complement `S = D·diag(R)⁻¹·Dᵀ (+ L)` — the
        // mass-Schur term is the full L² block when it is nonzero (the serial
        // analogue of the MFEM transformed-system preconditioner, whose
        // `diag(L)/d²` term equals `diag(L)⁻¹`-scaled `L` for the constant
        // β = 1 used by the miniapps).
        let mut s = {
            let mut ds = d.clone();
            for i in 0..ds.nrows {
                for p in ds.row_ptr[i]..ds.row_ptr[i + 1] {
                    let j = ds.col_idx[p] as usize;
                    ds.values[p] *= r_inv_diag[j];
                }
            }
            ds.multiply(&dt)
        };
        if let Some(lm) = &l {
            let mut coo = CooMatrix::<f64>::new(n_l2, n_l2);
            for i in 0..n_l2 {
                for p in s.row_ptr[i]..s.row_ptr[i + 1] {
                    coo.add(i, s.col_idx[p] as usize, s.values[p]);
                }
                let dii = lm.get(i, i);
                if dii.abs() > 1e-300 {
                    coo.add(i, i, dii);
                } else {
                    coo.add(i, i, 1.0);
                }
            }
            s = coo.into_csr();
        }
        let s_amg = AmgSolver::setup(&guard_zero_diagonal(&s), AmgConfig::default());

        Self {
            n_l2,
            n_rt,
            zero_l2_block,
            l,
            dt,
            d,
            d_e,
            r,
            r_e,
            s_amg,
            r_inv_diag,
            ess_rt_dofs: ess_rt_dofs.to_vec(),
            x_bc: Vec::new(),
            param: minres_params(),
            last_iters: AtomicUsize::new(0),
            last_converged: AtomicBool::new(false),
        }
    }

    /// MFEM `SetBC`: prescribed essential-RT values (full RT vector layout).
    pub fn set_bc(&mut self, x_rt: &[f64]) {
        assert_eq!(x_rt.len(), self.n_rt, "SetBC: size mismatch");
        self.x_bc = x_rt.to_vec();
    }

    /// MFEM `EliminateBC`: fold the essential values into the RHS.
    fn eliminate_bc(&self, b: &mut [f64]) {
        let n_ess = self.ess_rt_dofs.len();
        if n_ess == 0 {
            return;
        }
        let (n_l2, n_rt) = (self.n_l2, self.n_rt);
        assert_eq!(
            self.x_bc.len(),
            n_rt,
            "SetBC must be called before Mult when essential dofs exist"
        );
        // z: BC values at essential dofs, zero elsewhere.
        let mut z = vec![0.0_f64; n_rt];
        for &i in &self.ess_rt_dofs {
            z[i] = self.x_bc[i];
        }
        // bE -= D_e·z
        let mut dze = vec![0.0_f64; n_l2];
        self.d_e.spmv(&z, &mut dze);
        for (bi, di) in b[..n_l2].iter_mut().zip(&dze) {
            *bi -= di;
        }
        // bF += R_e·z
        let mut rze = vec![0.0_f64; n_rt];
        self.r_e.spmv(&z, &mut rze);
        for (bi, ri) in b[n_l2..].iter_mut().zip(&rze) {
            *bi += ri;
        }
        // Insert the RT BCs at the essential dofs (the R block appears with
        // multiplier −1 in the operator).
        for &i in &self.ess_rt_dofs {
            b[n_l2 + i] = -z[i];
        }
    }

    /// The 2×2 block operator (MFEM `A_block`).
    fn apply_op(&self, v: &[f64], w: &mut [f64]) {
        let (n_l2, n_rt) = (self.n_l2, self.n_rt);
        let (ve, vf) = (&v[..n_l2], &v[n_l2..]);
        match &self.l {
            Some(lm) => lm.spmv(ve, &mut w[..n_l2]),
            None => w[..n_l2].fill(0.0),
        }
        let mut dv = vec![0.0_f64; n_l2];
        self.d.spmv(vf, &mut dv);
        for (wi, di) in w[..n_l2].iter_mut().zip(&dv) {
            *wi += di;
        }
        self.dt.spmv(ve, &mut w[n_l2..]);
        let mut rv = vec![0.0_f64; n_rt];
        self.r.spmv(vf, &mut rv);
        for (wi, ri) in w[n_l2..].iter_mut().zip(&rv) {
            *wi -= ri;
        }
    }

    /// The block-diagonal preconditioner `diag(AMG(S), Jacobi(R))`.
    fn apply_prec(&self, v: &[f64], w: &mut [f64]) {
        let n_l2 = self.n_l2;
        let se = self.s_amg.precond_apply(&v[..n_l2]);
        w[..n_l2].copy_from_slice(&se);
        for (wi, (&vi, &di)) in w[n_l2..]
            .iter_mut()
            .zip(v[n_l2..].iter().zip(&self.r_inv_diag))
        {
            *wi = vi * di;
        }
    }

    /// MFEM `Mult`: solve for the L² (block 0) and RT (block 1) unknowns.
    pub fn mult(&self, b: &[f64], x: &mut [f64]) {
        let n = self.n_l2 + self.n_rt;
        assert_eq!(b.len(), n, "HdivSaddlePointSolver::mult: rhs size");
        assert_eq!(x.len(), n, "HdivSaddlePointSolver::mult: solution size");
        let mut b_prime = b.to_vec();
        self.eliminate_bc(&mut b_prime);
        let (iters, converged, _) = mfem_minres(
            n,
            &|v: &[f64], w: &mut [f64]| self.apply_op(v, w),
            Some(&|v: &[f64], w: &mut [f64]| self.apply_prec(v, w)),
            &b_prime,
            x,
            &self.param,
        );
        self.last_iters.store(iters, Ordering::Relaxed);
        self.last_converged.store(converged, Ordering::Relaxed);
    }

    /// MFEM `GetNumIterations`.
    pub fn num_iterations(&self) -> usize {
        self.last_iters.load(Ordering::Relaxed)
    }

    /// Whether the last [`Self::mult`] converged.
    pub fn converged(&self) -> bool {
        self.last_converged.load(Ordering::Relaxed)
    }

    /// Block offsets `[0, n_l2, n_l2 + n_rt]` (MFEM `GetOffsets`).
    pub fn offsets(&self) -> [usize; 3] {
        [0, self.n_l2, self.n_l2 + self.n_rt]
    }

    /// Whether the L² block was identically zero (Darcy α = 0).
    pub fn zero_l2_block(&self) -> bool {
        self.zero_l2_block
    }

    /// The uneliminated discrete divergence `D_e` (MFEM `D_e`), L² rows ×
    /// RT columns.  Exposed so the miniapps can assemble boundary-flux RHS
    /// liftings consistent with the assembled blocks (see `darcy.rs`).
    pub fn d_e_matrix(&self) -> &CsrMatrix<f64> {
        &self.d_e
    }

    /// The uneliminated RT mass matrix `R_e` (MFEM `R_e`).
    pub fn r_e_matrix(&self) -> &CsrMatrix<f64> {
        &self.r_e
    }
}

/// L² mass matrix with a `QuadratureFunction` coefficient (MFEM
/// `MassIntegrator(QuadratureFunctionCoefficient)` at the quadrature space's
/// integration rules): element-local `K[i][j] += (w_q·|detJ_q|)·W(x_q)·φᵢφⱼ`.
fn assemble_l2_mass_qf(
    l2: &L2Space<Mesh<2>>,
    qs: &QuadratureSpace<'_>,
    w_qf: &QuadratureFunction<'_>,
) -> CsrMatrix<f64> {
    let mesh = l2.mesh();
    let weights = qs.get_weights();
    let mut coo = CooMatrix::<f64>::new(l2.n_dofs(), l2.n_dofs());
    for e in 0..mesh.n_elements() as u32 {
        let fe = l2_ref_elem(mesh, e, l2.order());
        let n = fe.n_dofs();
        let rule = qs.get_int_rule(e as usize);
        let base = qs.offset(e as usize);
        let w_vals = w_qf.get_values(e as usize);
        let mut phi = vec![0.0_f64; n];
        let mut ke = vec![0.0_f64; n * n];
        for (q, xi) in rule.points.iter().enumerate() {
            fe.eval_basis(xi, &mut phi);
            let w = weights[base + q] * w_vals[q];
            for i in 0..n {
                let wi = w * phi[i];
                for j in 0..n {
                    ke[i * n + j] += wi * phi[j];
                }
            }
        }
        let dofs = l2.element_dofs(e);
        debug_assert_eq!(dofs.len(), n, "L2 QF mass: local dof count mismatch");
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                coo.add(gi as usize, gj as usize, ke[i * n + j]);
            }
        }
    }
    coo.into_csr()
}

/// Reference L² element matching fem-rs' `L2Space` basis (MFEM
/// `L2_FECollection` default Gauss-Legendre / discontinuous simplex nodes).
fn l2_ref_elem(mesh: &Mesh<2>, e: u32, order: u8) -> Box<dyn ReferenceElement> {
    match mesh.element_type(e) {
        fem_mesh::ElementType::Tri3 if order == 0 => Box::new(L2P0),
        fem_mesh::ElementType::Tri3 => Box::new(TriPk::new(order as usize)),
        fem_mesh::ElementType::Quad4 if order == 0 => Box::new(L2P0),
        fem_mesh::ElementType::Quad4 => Box::new(QuadL2GL::new(order as usize)),
        other => panic!("l2_ref_elem: unsupported element type {other:?}"),
    }
}

/// Order-0 (piecewise constant) L² reference element, value ≡ 1.
struct L2P0;
impl ReferenceElement for L2P0 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], values: &mut [f64]) { values[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        grads[0] = 0.0;
        grads[1] = 0.0;
    }
    fn quadrature(&self, _order: u8) -> fem_element::reference::QuadratureRule {
        fem_element::reference::QuadratureRule {
            points: vec![vec![0.5, 0.5]],
            weights: vec![1.0],
        }
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.5, 0.5]] }
}

/// Zero the entries in the given columns (MFEM miniapp
/// `discrete_divergence.cpp:EliminateColumns`).
fn eliminate_columns(a: &CsrMatrix<f64>, ess_cols: &[usize]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            if !ess_cols.contains(&j) {
                coo.add(i, j, a.values[p]);
            }
        }
    }
    coo.into_csr()
}

/// Symmetric elimination to the identity at the given dofs (MFEM
/// `BilinearForm::FormSystemMatrix(ess)`).
fn eliminate_rows_cols(a: &CsrMatrix<f64>, ess: &[usize]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        if ess.contains(&i) {
            coo.add(i, i, 1.0);
            continue;
        }
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            if !ess.contains(&j) {
                coo.add(i, j, a.values[p]);
            }
        }
    }
    coo.into_csr()
}

/// Set `A[i][i] = 1` for numerically empty rows so the AMG hierarchy is SPD
/// (local copy of `fem_solver::darcy_solvers::guard_zero_diagonal`).
fn guard_zero_diagonal(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        let mut has_diag = false;
        let mut row_max = 0.0_f64;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            coo.add(i, a.col_idx[p] as usize, a.values[p]);
            if a.col_idx[p] as usize == i {
                has_diag = true;
            }
            row_max = row_max.max(a.values[p].abs());
        }
        if !has_diag && row_max < 1e-30 {
            coo.add(i, i, 1.0);
        }
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::refine_uniform;
    use fem_space::l2::L2Space;

    /// The QF-projected mass of a constant coefficient equals the plain mass
    /// matrix assembled by the standard integrator (validates the
    /// QuadratureFunction-driven element assembly end to end).
    #[test]
    fn qf_mass_matches_constant_mass() {
        let mesh = refine_uniform(&refine_uniform(&Mesh::<2>::unit_square_tri(2)));
        let l2 = L2Space::new(mesh.clone(), 1);
        let qs = QuadratureSpace::new(&mesh, 3);
        let mut w_qf = QuadratureFunction::new(&qs);
        w_qf.fill(2.5); // `ConstantCoefficient(2.5).Project(qf)`
        let m_qf = assemble_l2_mass_qf(&l2, &qs, &w_qf);
        let m_std = fem_assembly::Assembler::assemble_bilinear(
            &l2,
            &[&fem_assembly::standard::MassIntegrator { rho: 2.5 }],
            3,
        );
        assert_eq!(m_qf.nrows, m_std.nrows);
        let mut dmax = 0.0_f64;
        for i in 0..m_qf.nrows {
            for j in 0..m_qf.ncols {
                dmax = dmax.max((m_qf.get(i, j) - m_std.get(i, j)).abs());
            }
        }
        assert!(dmax < 1e-12, "QF mass mismatch: {dmax:.3e}");
    }

    /// Row-0 consistency on the quad unit square: the L² projection `q_src`
    /// of the exact flux satisfies `D·q_src = (f, ψ)` to discretization
    /// accuracy (validates the divergence block against the volume load).
    #[test]
    fn darcy_row0_consistency_quads() {
        use fem_assembly::standard::{DomainSourceIntegrator, VectorMassIntegrator};
        use fem_assembly::vector_assembler::VectorAssembler;
        use fem_assembly::postproc::coefficient::FnVectorCoeff;
        use fem_assembly::Assembler;

        // p = x(1−x) + y(1−y), q = −∇p = (2x−1, 2y−1), f = −Δp = 4.
        let p_ex = |x: &[f64]| x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]);
        let q_ex = |x: &[f64], out: &mut [f64]| {
            out[0] = 2.0 * x[0] - 1.0;
            out[1] = 2.0 * x[1] - 1.0;
        };

        let mut mesh = Mesh::<2>::unit_square_quad(1);
        for _ in 0..3 {
            mesh = refine_uniform(&mesh);
        }
        let rt = HDivSpace::new(mesh.clone(), 1);
        let l2 = L2Space::new(mesh.clone(), 1);
        let n_rt = rt.n_dofs();
        let n_l2 = l2.n_dofs();
        let qo = 5_u8;

        let b_l2 = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(|_| 4.0)], qo);
        let r_mass = VectorAssembler::assemble_bilinear(
            &rt,
            &[&fem_assembly::standard::VectorMassIntegrator { alpha: 1.0 }],
            qo,
        );
        let rhs_q = VectorAssembler::assemble_linear(
            &rt,
            &[&fem_assembly::standard::VectorDomainLFIntegrator {
                f: FnVectorCoeff(q_ex),
            }],
            qo,
        );
        // L² projection of q (AMG-CG on the RT mass).
        let mut q_proj = vec![0.0_f64; n_rt];
        fem_amg::solve_amg_cg(
            &r_mass,
            &rhs_q,
            &mut q_proj,
            &fem_amg::AmgConfig::default(),
            &fem_linalg::SolverConfig {
                rtol: 1e-13,
                atol: 1e-14,
                max_iter: 500,
                verbose: false,
                print_level: fem_linalg::PrintLevel::Silent,
            },
        )
        .expect("projection solve");

        let solver = HdivSaddlePointSolver::new(&mesh, &l2, &rt, 0.0, 1.0, &[], Mode::Darcy);
        let mut dq = vec![0.0_f64; n_l2];
        solver.d.spmv(&q_proj, &mut dq);
        let r0: f64 = dq
            .iter()
            .zip(&b_l2)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let _ = p_ex;
        assert!(r0 < 5e-3, "row-0 residual {r0:.3e}");
    }

    /// End-to-end Darcy α = 0 solve on the quad unit square with the lifted
    /// boundary RHS (nonzero boundary pressure): the MINRES solution must
    /// match the exact solution to discretization accuracy.
    #[test]
    fn darcy_alpha0_lifted_full_solve() {
        use fem_assembly::standard::{DomainSourceIntegrator, VectorMassIntegrator};
        use fem_assembly::postproc::grid_function::GridFunction;
        use fem_assembly::postproc::coefficient::FnVectorCoeff;
        use fem_assembly::vector_assembler::VectorAssembler;
        use fem_assembly::Assembler;

        // p = x(1−x) + y(1−y) (NONZERO on ∂Ω), q = −∇p, f = 4.
        let p_ex = |x: &[f64]| x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]);
        let q_ex = |x: &[f64], out: &mut [f64]| {
            out[0] = 2.0 * x[0] - 1.0;
            out[1] = 2.0 * x[1] - 1.0;
        };

        let mut mesh = Mesh::<2>::unit_square_quad(1);
        for _ in 0..3 {
            mesh = refine_uniform(&mesh);
        }
        let rt = HDivSpace::new(mesh.clone(), 1);
        let l2 = L2Space::new(mesh.clone(), 1);
        let n_rt = rt.n_dofs();
        let n_l2 = l2.n_dofs();
        let qo = 5_u8;

        let solver = HdivSaddlePointSolver::new(&mesh, &l2, &rt, 0.0, 1.0, &[], Mode::Darcy);
        let b_l2 = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(|_| 4.0)], qo);
        // Lifted boundary RHS from the projected exact solution.
        let l_mass =
            Assembler::assemble_bilinear(&l2, &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }], qo);
        let rhs_p = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(p_ex)], qo);
        let rhs_q = VectorAssembler::assemble_linear(
            &rt,
            &[&fem_assembly::standard::VectorDomainLFIntegrator {
                f: FnVectorCoeff(q_ex),
            }],
            qo,
        );
        let mut p_src = vec![0.0_f64; n_l2];
        fem_amg::solve_amg_cg(
            &l_mass,
            &rhs_p,
            &mut p_src,
            &fem_amg::AmgConfig::default(),
            &fem_linalg::SolverConfig {
                rtol: 1e-13,
                atol: 1e-14,
                max_iter: 500,
                verbose: false,
                print_level: fem_linalg::PrintLevel::Silent,
            },
        )
        .expect("p projection");
        let mut q_src = vec![0.0_f64; n_rt];
        fem_amg::solve_amg_cg(
            solver.r_e_matrix(),
            &rhs_q,
            &mut q_src,
            &fem_amg::AmgConfig::default(),
            &fem_linalg::SolverConfig {
                rtol: 1e-13,
                atol: 1e-14,
                max_iter: 500,
                verbose: false,
                print_level: fem_linalg::PrintLevel::Silent,
            },
        )
        .expect("q projection");
        let mut b_rt = vec![0.0_f64; n_rt];
        solver.d_e_matrix().transpose().spmv(&p_src, &mut b_rt);
        let mut rq = vec![0.0_f64; n_rt];
        solver.r_e_matrix().spmv(&q_src, &mut rq);
        for (b, r) in b_rt.iter_mut().zip(&rq) {
            *b -= r;
        }

        let mut b = vec![0.0_f64; n_l2 + n_rt];
        b[..n_l2].copy_from_slice(&b_l2);
        b[n_l2..].copy_from_slice(&b_rt);
        let mut x = vec![0.0_f64; n_l2 + n_rt];
        solver.mult(&b, &mut x);
        assert!(solver.converged(), "MINRES did not converge");
        let p = GridFunction::new(&l2, x[..n_l2].to_vec());
        let err = p.compute_l2_error(&p_ex, 6);
        assert!(err < 0.02, "full-solve error {err:.3e}");
    }
}
