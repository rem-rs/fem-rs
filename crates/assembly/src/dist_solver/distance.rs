//! Distance solvers — 1:1 serial port of MFEM `miniapps/common/dist_solver.{hpp,cpp}`:
//! [`DistanceSolver`], [`HeatDistanceSolver`] (Crane et al., "Geodesics in Heat"),
//! [`NormalizationDistanceSolver`] and [`PLapDistanceSolver`] (Belyaev et al.),
//! plus the `NormalizedGradCoefficient` / `PProductCoefficient` helpers, the
//! `DiffuseField` utility and the `DomainLFGradIntegrator` linear form.
//!
//! MFEM verifies at runtime that the solvers receive scalar H1 spaces
//! (`dynamic_cast<const H1_FECollection*>`); the Rust signatures encode that
//! requirement in the type system instead.

use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, L2Space, VectorH1Space};

use crate::assembler::{ref_elem_vol_for_space, Assembler};
use crate::postproc::coefficient::GridFunctionCoeff;
use crate::postproc::grid_function::GridFunction;
use crate::standard::source::DomainSourceIntegratorCoeff;
use crate::standard::{DiffusionIntegrator, MassIntegrator};
use crate::FixedOrder;

use super::{
    elem_dof_points, elem_geometry, project_disc_average, solve_singular_ortho_cg, spd_solve,
    transform_grads, DistScalarCoeff, DistVectorCoeff, LevelSetFn,
};

/// Base trait of the distance solvers (MFEM `DistanceSolver`).
///
/// Computes the (optionally signed) distance to the zero level set of a
/// coefficient in the space of the given DOF vector.
pub trait DistanceSolver<M: MeshTopology + Clone + 'static>: Send + Sync {
    /// Compute a scalar distance field into `distance` (DOF vector on `space`).
    /// Some solvers produce a *signed* distance (different signs on both sides
    /// of the zero level set).
    fn compute_scalar_distance(
        &self,
        zero_level_set: &LevelSetFn,
        distance: &mut [f64],
        space: &H1Space<M>,
    );

    /// Compute a vector distance field (magnitude + starting direction of the
    /// shortest path) into `distance` (DOF vector on `space_v`, `byNODES`
    /// component layout).
    ///
    /// Default implementation (MFEM `DistanceSolver::ComputeVectorDistance`):
    /// solve the scalar problem in the matching scalar H1 space, then call
    /// [`scalar_dist_to_vector`].
    fn compute_vector_distance(
        &self,
        zero_level_set: &LevelSetFn,
        distance: &mut [f64],
        space_v: &VectorH1Space<M>,
    ) {
        assert_eq!(
            space_v.n_dofs(),
            space_v.n_scalar_dofs() * space_v.mesh().dim() as usize,
            "This function expects a vector space of the mesh dimension!"
        );
        let pfes_s = H1Space::new(space_v.mesh().clone(), space_v.order());
        let dist_s = self.compute_scalar_distance_owned(zero_level_set, &pfes_s);
        let v = scalar_dist_to_vector(&pfes_s, &dist_s);
        distance.copy_from_slice(&v);
    }

    /// Scalar solve returning a freshly allocated DOF vector.
    fn compute_scalar_distance_owned(
        &self,
        zero_level_set: &LevelSetFn,
        space: &H1Space<M>,
    ) -> Vec<f64> {
        let mut d = vec![0.0_f64; space.n_dofs()];
        self.compute_scalar_distance(zero_level_set, &mut d, space);
        d
    }
}

/// Convert a scalar distance field to a vector distance field
/// (MFEM `DistanceSolver::ScalarDistToVector`).
///
/// The result layout is `byNODES`: `dist_v[i + d * n]` is the `d`-component at
/// DOF `i`.  The vector points towards the zero level set and has magnitude
/// `|dist_s|` (approximated by the local gradient).
pub fn scalar_dist_to_vector<M: MeshTopology, S: FESpace<Mesh = M>>(
    space: &S,
    dist_s: &[f64],
) -> Vec<f64> {
    let mesh = space.mesh();
    let dim = mesh.topological_dim() as usize;
    let n = space.n_dofs();
    let mut dist_v = vec![0.0_f64; dim * n];
    let mut magn = vec![0.0_f64; n];
    let mut der_d = vec![0.0_f64; n];
    for d in 0..dim {
        // der_d = discrete derivative of dist_s along axis d, sampled at the
        // DOF nodes (MFEM `dist_s.GetDerivative(1, d, der)`).
        for v in der_d.iter_mut() {
            *v = 0.0;
        }
        for e in mesh.elem_iter() {
            let order = space.element_order(e);
            let re = ref_elem_vol_for_space(space, mesh.element_type(e), order);
            let nd = re.n_dofs();
            let mut grad_ref = vec![0.0_f64; nd * dim];
            let mut grad_phys = vec![0.0_f64; nd * dim];
            let dofs = space.element_dofs(e);
            for (xi, _x) in elem_dof_points(space, e) {
                re.eval_grad_basis(&xi, &mut grad_ref);
                let (jit, _det, _xp) = elem_geometry(mesh, e, &xi, dim);
                transform_grads(&jit, &grad_ref, &mut grad_phys, nd, dim);
                // ∂u/∂x_d at this DOF node: Σ_j u_j ∂φ_j/∂x_d.
                for i in 0..nd {
                    let mut dud = 0.0_f64;
                    for j in 0..nd {
                        dud += dist_s[dofs[j] as usize] * grad_phys[j * dim + d];
                    }
                    der_d[dofs[i] as usize] += dud;
                }
            }
        }
        for i in 0..n {
            magn[i] += der_d[i] * der_d[i];
            // The vector must point towards the level zero set.
            dist_v[i + d * n] = if dist_s[i] > 0.0 { -der_d[i] } else { der_d[i] };
        }
    }
    for i in 0..n {
        let vec_magn = (magn[i] + 1e-12).sqrt();
        for d in 0..dim {
            dist_v[i + d * n] *= dist_s[i].abs() / vec_magn;
        }
    }
    dist_v
}

// ─── Heat method ─────────────────────────────────────────────────────────────

/// K. Crane et al: "Geodesics in Heat: A New Approach to Computing Distance
/// Based on Heat Flow", DOI:10.1145/2516971.2516977.
///
/// The computed distance is *not* signed.  With `transform = false` the solver
/// can also be applied to point sources.
pub struct HeatDistanceSolver {
    /// Diffusion coefficient `t` of the heat step.
    pub parameter_t: f64,
    /// Optional initial smoothing steps of the level set (`smooth_steps`).
    pub smooth_steps: u32,
    /// Number of diffusion iterations (`diffuse_iter`).
    pub diffuse_iter: u32,
    /// Transform the level set into a source-type bump (assumes range [-1, 1]).
    pub transform: bool,
}

impl HeatDistanceSolver {
    pub fn new(diff_coeff: f64) -> Self {
        HeatDistanceSolver {
            parameter_t: diff_coeff,
            smooth_steps: 0,
            diffuse_iter: 1,
            transform: true,
        }
    }
}

impl<M: MeshTopology + Clone + 'static> DistanceSolver<M> for HeatDistanceSolver {
    fn compute_scalar_distance(
        &self,
        zero_level_set: &LevelSetFn,
        distance: &mut [f64],
        space: &H1Space<M>,
    ) {
        let mesh = space.mesh();
        let p = space.order() as usize;

        // Step 0 - transform the input level set into a source-type bump.
        let mut source: Vec<f64> = space.interpolate(zero_level_set).as_slice().to_vec();
        // Optional smoothing of the initial level set.
        if self.smooth_steps > 0 {
            diffuse_field(space, &mut source, self.smooth_steps);
        }
        // Transform so that the peak is at 0.  Assumes range [-1, 1].
        if self.transform {
            for x in source.iter_mut() {
                *x = if *x < -1.0 || *x > 1.0 {
                    0.0
                } else {
                    (1.0 - *x) * (1.0 + *x)
                };
            }
        }

        // Shared operator: mass + t·diffusion.  MFEM assembles `a_d` (with BC)
        // and `a_n` (without BC) with identical integrators — the same matrix;
        // only the essential-DOF treatment differs.  Quadrature orders mirror
        // MFEM defaults: mass `2p`, diffusion `p + OrderGrad = 2p − 1`.
        let a0 = Assembler::assemble_bilinear(
            space,
            &[
                &MassIntegrator { rho: 1.0 },
                &FixedOrder::new(
                    DiffusionIntegrator { kappa: self.parameter_t },
                    (2 * p - 1) as u8,
                ),
            ],
            (2 * p) as u8,
        );

        // Essential (Dirichlet) DOFs: the whole true boundary.
        let tags = super::boundary_tags(mesh);
        let ess: Vec<usize> =
            fem_space::constraints::dirichlet::boundary_dofs(mesh, space.dof_manager(), &tags)
                .iter()
                .map(|&d| d as usize)
                .collect();

        let mut diffused_source = vec![0.0_f64; space.n_dofs()];
        for _ in 0..self.diffuse_iter {
            // RHS: ∫ source · φ (MFEM `DomainLFIntegrator`, order `p`).
            let b = Assembler::assemble_linear(
                space,
                &[&DomainSourceIntegratorCoeff::new(GridFunctionCoeff::new(source.clone()))],
                p as u8,
            );

            // Solve with Dirichlet BC (u = 0 on the boundary).
            let mut a_dir = a0.clone();
            let mut b_dir = b.clone();
            for &d in &ess {
                a_dir.apply_dirichlet_symmetric(d, 0.0, &mut b_dir);
            }
            let mut u_dirichlet = vec![0.0_f64; space.n_dofs()];
            spd_solve(&a_dir, &b_dir, &mut u_dirichlet, 1e-12, 0.0, 500);

            // Solve with Neumann BC (no essential DOFs).
            let mut u_neumann = vec![0.0_f64; space.n_dofs()];
            spd_solve(&a0, &b, &mut u_neumann, 1e-12, 0.0, 500);

            for i in 0..diffused_source.len() {
                diffused_source[i] = 0.5 * (u_neumann[i] + u_dirichlet[i]);
            }
            source.copy_from_slice(&diffused_source);
        }

        // Step 2 - solve for the distance using the normalized gradient.
        let diffused_gf = GridFunction::new(space, diffused_source);
        let grad_u = NormalizedGradCoefficient { u: &diffused_gf };
        let b2 = assemble_lf_grad(space, &grad_u, (2 * p - 1) as u8);

        let k2 = Assembler::assemble_bilinear(
            space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            (2 * p - 1) as u8,
        );
        // No BC — singular system solved with orthogonalizing CG (OrthoSolver).
        solve_singular_ortho_cg(&k2, &b2, distance);

        // Shift the distance values to have minimum at zero.
        let d_min = distance.iter().cloned().fold(f64::INFINITY, f64::min);
        for x in distance.iter_mut() {
            *x -= d_min;
        }
    }
}

/// Smooth a field by Jacobi relaxation of the homogeneous Laplace equation
/// (MFEM `DiffuseField`: `HypreSmoother(A, type=0, smooth_steps)` applied to a
/// zero RHS with `iterative_mode = true`).
pub fn diffuse_field<S: FESpace>(space: &S, field: &mut [f64], smooth_steps: u32) {
    let p = space.order() as usize;
    let k = Assembler::assemble_bilinear(
        space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        (2 * p - 1) as u8,
    );
    let diag = k.diagonal();
    let mut kf = vec![0.0_f64; field.len()];
    for _ in 0..smooth_steps {
        k.spmv(field, &mut kf);
        for i in 0..field.len() {
            field[i] -= kf[i] / diag[i];
        }
    }
}

// ─── Normalization solver ────────────────────────────────────────────────────

/// A. Belyaev et al: "On Variational and PDE-based Distance Function
/// Approximations", Section 6, DOI:10.1111/cgf.12611.
///
/// Computationally cheap, accurate for distance approximations only near the
/// zero level set.
pub struct NormalizationDistanceSolver;

impl<M: MeshTopology + Clone + 'static> DistanceSolver<M> for NormalizationDistanceSolver {
    fn compute_scalar_distance(
        &self,
        u_coeff: &LevelSetFn,
        dist: &mut [f64],
        space: &H1Space<M>,
    ) {
        // u_gf = ProjectCoefficient(u_coeff) — nodal interpolation.
        let u_dofs: Vec<f64> = space.interpolate(u_coeff).as_slice().to_vec();
        let u_gf = GridFunction::new(space, u_dofs);
        // dist = ProjectDiscCoefficient(NormalizationCoeff(u_gf), ARITHMETIC).
        let d = project_disc_average(space, |e, xi, _x| {
            let u_value = u_gf.evaluate_at_element(e, xi);
            let u_grad = u_gf.evaluate_gradient_at_element(e, xi);
            let g2: f64 = u_grad.iter().map(|g| g * g).sum();
            u_value / (u_value * u_value + g2 + 1e-12).sqrt()
        });
        dist.copy_from_slice(&d);
    }
}

// ─── p-Laplace solver ────────────────────────────────────────────────────────

/// A. Belyaev et al: "On Variational and PDE-based Distance Function
/// Approximations", Section 7, DOI:10.1111/cgf.12611.
///
/// The computed distance is *signed*.  Uses power continuation
/// `pp = 2, 3, …, maxp−1` with a Newton solve per power
/// (MFEM: `NewtonSolver` + GMRES + `HypreBoomerAMG`).
pub struct PLapDistanceSolver {
    /// Maximum value of the power `p`.
    pub maxp: i32,
    /// Maximum Newton iterations per power.
    pub newton_iter: i32,
    pub newton_rel_tol: f64,
    pub newton_abs_tol: f64,
}

impl PLapDistanceSolver {
    pub fn new(maxp: i32, newton_iter: i32, rtol: f64, atol: f64) -> Self {
        PLapDistanceSolver {
            maxp,
            newton_iter,
            newton_rel_tol: rtol,
            newton_abs_tol: atol,
        }
    }

    pub fn set_max_power(&mut self, new_pp: i32) {
        self.maxp = new_pp;
    }
}

impl Default for PLapDistanceSolver {
    fn default() -> Self {
        PLapDistanceSolver::new(30, 10, 1e-7, 1e-12)
    }
}

impl<M: MeshTopology + Clone + 'static> DistanceSolver<M> for PLapDistanceSolver {
    fn compute_scalar_distance(
        &self,
        func: &LevelSetFn,
        fdist: &mut [f64],
        fesd: &H1Space<M>,
    ) {
        let d = plap_distance(func, fesd, &self);
        fdist.copy_from_slice(&d);
    }
}

impl PLapDistanceSolver {
    /// p-Laplace distance with an L² (DG) output space (MFEM also accepts L2
    /// `fdist` spaces: `check_h1 || check_l2`).
    pub fn compute_scalar_distance_l2<M: MeshTopology + Clone + 'static>(
        &self,
        func: &LevelSetFn,
        fdist: &mut [f64],
        fesd: &L2Space<M>,
    ) {
        let d = plap_distance(func, fesd, &self);
        fdist.copy_from_slice(&d);
    }
}

/// Shared p-Laplace pipeline (works for any scalar output space; the internal
/// continuation space is H1 of the same order, as in MFEM).
fn plap_distance<M, S>(
    func: &LevelSetFn,
    fesd: &S,
    opts: &PLapDistanceSolver,
) -> Vec<f64>
where
    M: MeshTopology + Clone + 'static,
    S: FESpace<Mesh = M>,
{
    let mesh = fesd.mesh();
    let order = fesd.order();

    // Internal H1 space for the p-harmonic function (MFEM: `fesp`, byVDIM —
    // ordering is irrelevant in serial).
    let fesp = H1Space::new(mesh.clone(), order);

    // wf = ProjectCoefficient(func); gf = ∇wf (GradientGridFunctionCoefficient).
    let wf_dofs: Vec<f64> = fesp.interpolate(func).as_slice().to_vec();
    let wf = GridFunction::new(&fesp, wf_dofs);
    let gf = GradientGridFunctionCoeff { u: &wf };

    // xf true dofs initialized to 1 (MFEM: `*sv = 1.0`).
    let mut sv = vec![1.0_f64; fesp.n_dofs()];

    let mut nf = super::filter::NonlinearForm::new();
    let pint = super::filter::PUMPLaplacian::new(Box::new(super::FnDist(func)), Box::new(gf));
    nf.add_domain_integrator(&pint);

    // Power continuation with a Newton solve per power: pp = 2 … maxp−1.
    for pp in 2..opts.maxp {
        newton_solve(
            &nf,
            &fesp,
            &mut sv,
            pp as f64,
            opts.newton_iter,
            opts.newton_rel_tol,
            opts.newton_abs_tol,
        );
    }

    // fdist = ProjectCoefficient(|func| · xf)  (PProductCoefficient).
    let xf = GridFunction::new(&fesp, sv);
    let tsol = PProductCoefficient { basef: func, corrf: GfScalarCoeff { gf: &xf } };
    project_disc_average(fesd, |e, xi, x| tsol.eval(e, &xi, x))
}

/// Newton solve of the zero-RHS nonlinear system
/// (MFEM `NewtonSolver` with `iterative_mode = true`, GMRES + AMG inner solve:
/// `J c = r`, `x ← x − c`).
fn newton_solve<M: MeshTopology>(
    nf: &super::filter::NonlinearForm<'_>,
    space: &H1Space<M>,
    x: &mut [f64],
    pp: f64,
    max_iter: i32,
    rel_tol: f64,
    abs_tol: f64,
) {
    nf.set_power_hint(pp);
    let mut r = vec![0.0_f64; x.len()];
    nf.mult(space, x, &mut r);
    let r0norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r0norm == 0.0 {
        return;
    }
    for _ in 0..max_iter {
        let norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= abs_tol + rel_tol * r0norm {
            break;
        }
        let j = nf.get_gradient(space, x);
        // Solve J c = r, then x ← x − c.
        let mut c = vec![0.0_f64; x.len()];
        super::nonsym_solve(&j, &r, &mut c, rel_tol / 10.0, abs_tol / 10.0, 500);
        for (xi, ci) in x.iter_mut().zip(c.iter()) {
            *xi -= ci;
        }
        nf.mult(space, x, &mut r);
    }
}

// ─── Coefficients ────────────────────────────────────────────────────────────

/// MFEM `NormalizedGradCoefficient`: `V = -∇u / |∇u|` — points towards the zero
/// level set of `u`.
pub struct NormalizedGradCoefficient<'c, 'x, S: FESpace> {
    pub(crate) u: &'c GridFunction<'x, S>,
}

impl<S: FESpace> DistVectorCoeff for NormalizedGradCoefficient<'_, '_, S> {
    fn eval(&self, elem: u32, xi: &[f64], _x: &[f64], out: &mut [f64]) {
        let grad = self.u.evaluate_gradient_at_element(elem, xi);
        let norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt() + 1e-12;
        for (o, g) in out.iter_mut().zip(grad.iter()) {
            *o = -g / norm;
        }
    }
}

/// Gradient of a grid function (MFEM `GradientGridFunctionCoefficient`).
pub(crate) struct GradientGridFunctionCoeff<'c, 'x, S: FESpace> {
    pub(crate) u: &'c GridFunction<'x, S>,
}

impl<S: FESpace> DistVectorCoeff for GradientGridFunctionCoeff<'_, '_, S> {
    fn eval(&self, elem: u32, xi: &[f64], _x: &[f64], out: &mut [f64]) {
        let grad = self.u.evaluate_gradient_at_element(elem, xi);
        out.copy_from_slice(&grad);
    }
}

/// Scalar evaluation of a grid function at `(elem, xi)`
/// (MFEM `GridFunctionCoefficient`).
pub struct GfScalarCoeff<'c, 'x, S: FESpace> {
    pub(crate) gf: &'c GridFunction<'x, S>,
}

impl<S: FESpace> DistScalarCoeff for GfScalarCoeff<'_, '_, S> {
    fn eval(&self, elem: u32, xi: &[f64], _x: &[f64]) -> f64 {
        self.gf.evaluate_at_element(elem, xi)
    }
}

/// MFEM `PProductCoefficient`: `|basef| · corrf`.
pub struct PProductCoefficient<'c, 'x, S: FESpace> {
    pub(crate) basef: &'c LevelSetFn,
    pub(crate) corrf: GfScalarCoeff<'c, 'x, S>,
}

impl<S: FESpace> DistScalarCoeff for PProductCoefficient<'_, '_, S> {
    fn eval(&self, elem: u32, xi: &[f64], x: &[f64]) -> f64 {
        let mut u = (self.basef)(x);
        let c = self.corrf.eval(elem, xi, x);
        if u < 0.0 {
            u = -u;
        }
        u * c
    }
}

// ─── Linear-form assembly (element-aware coefficients) ──────────────────────

/// Assemble `b_i = ∫ (v · ∇φ_i) dx` (MFEM `DomainLFGradIntegrator`).
pub(crate) fn assemble_lf_grad<S: FESpace>(
    space: &S,
    v: &dyn DistVectorCoeff,
    quad_order: u8,
) -> Vec<f64> {
    let mesh = space.mesh();
    let dim = mesh.topological_dim() as usize;
    let mut b = vec![0.0_f64; space.n_dofs()];
    let mut vv = vec![0.0_f64; dim];
    for e in mesh.elem_iter() {
        let order = space.element_order(e);
        let re = ref_elem_vol_for_space(space, mesh.element_type(e), order);
        let nd = re.n_dofs();
        let dofs = space.element_dofs(e);
        let rule = re.quadrature(quad_order);
        let mut grad_ref = vec![0.0_f64; nd * dim];
        let mut grad_phys = vec![0.0_f64; nd * dim];
        let mut fe = vec![0.0_f64; nd];
        for (q, xi) in rule.points.iter().enumerate() {
            let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
            let w = rule.weights[q] * det.abs();
            re.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&jit, &grad_ref, &mut grad_phys, nd, dim);
            v.eval(e, xi, &xp, &mut vv);
            for i in 0..nd {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += vv[d] * grad_phys[i * dim + d];
                }
                fe[i] += w * dot;
            }
        }
        for (i, &d) in dofs.iter().enumerate() {
            b[d as usize] += fe[i];
        }
    }
    b
}

/// Assemble `b_i = ∫ f φ_i dx` for an element-aware scalar coefficient
/// (MFEM `DomainLFIntegrator` with a `Coefficient`).
pub(crate) fn assemble_lf_scalar<S: FESpace>(
    space: &S,
    f: &dyn Fn(u32, &[f64], &[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let mesh = space.mesh();
    let dim = mesh.topological_dim() as usize;
    let mut b = vec![0.0_f64; space.n_dofs()];
    for e in mesh.elem_iter() {
        let order = space.element_order(e);
        let re = ref_elem_vol_for_space(space, mesh.element_type(e), order);
        let nd = re.n_dofs();
        let dofs = space.element_dofs(e);
        let rule = re.quadrature(quad_order);
        let mut phi = vec![0.0_f64; nd];
        let mut fe = vec![0.0_f64; nd];
        for (q, xi) in rule.points.iter().enumerate() {
            let (_jit, det, xp) = elem_geometry(mesh, e, xi, dim);
            let w = rule.weights[q] * det.abs();
            re.eval_basis(xi, &mut phi);
            let fv = f(e, xi, &xp);
            for i in 0..nd {
                fe[i] += w * fv * phi[i];
            }
        }
        for (i, &d) in dofs.iter().enumerate() {
            b[d as usize] += fe[i];
        }
    }
    b
}
