//! Minimal serial [`NonlinearForm`] framework — MFEM `fem/nonlinearform.{hpp,cpp}`
//! (`Mult` / `GetGradient` / `GetEnergy` driving `NonlinearFormIntegrator`s).
//!
//! Promoted in D52 from the `dist_solver::filter` port of MFEM
//! `miniapps/common/dist_solver.{hpp,cpp}`, where the framework lived next to
//! its first users ([`ScreenedPoisson`](crate::dist_solver::filter::ScreenedPoisson),
//! [`PUMPLaplacian`](crate::dist_solver::filter::PUMPLaplacian),
//! [`PDEFilter`](crate::dist_solver::filter::PDEFilter) — still in
//! `dist_solver::filter`, which re-exports these types so the historical
//! `dist_solver::filter::` paths stay valid).
//!
//! # Not to be confused with `physics::nonlinear::NonlinearForm`
//!
//! [`crate::physics::nonlinear::NonlinearForm`] is a *solver-side* form: it
//! wraps an abstract residual `Operator` + Jacobian provider and drives
//! Newton/Anderson/trust-region solvers.  This module is the *assembly-side*
//! counterpart of MFEM `NonlinearForm`: it owns the element/quadrature loops
//! and asks [`NonlinearFormIntegrator`]s for per-quadrature-point
//! energy/residual/Jacobian contributions.  The navier miniapps combine the
//! two: this form assembles `N(u)` (MFEM `NonlinearForm::Mult`), the physics
//! module integrates it in time.
//!
//! # Element-DOF layout
//!
//! The form loops the volume elements of a caller-provided [`FESpace`].  The
//! element residual/Jacobian vectors are indexed by the space's *element DOF
//! list* (`element_dofs`), which for vector (`[H¹]^d`) spaces is the
//! interleaved layout `(basis k, component c) → k·d + c` — exactly MFEM's
//! `EF.UseExternalData(elfun, nd, dim)` view.  [`NLQpData::n_dofs`] counts the
//! scalar basis functions `nd`; the element vectors have `nd·d` entries.
//!
//! # Integration rules
//!
//! By default the form drives all its integrators with MFEM's generic
//! nonlinear-form order `2p + OrderGrad`, with `OrderGrad = p − 1` for Pk/Qk
//! spaces on P1 geometry (the rule the `dist_solver` miniapps validated
//! against C++).  An integrator can pin the shared rule per MFEM
//! `NonlinearFormIntegrator::SetIntRule` by returning `Some(order)` from
//! [`NonlinearFormIntegrator::int_rule_order`] (fem-rs selects rules by
//! order, `IntRules.Get(geom, order)`; the first override wins).
//!
//! Like the C++ `NonlinearForm::AddDomainIntegrator(integrator*)`, the form
//! stores *references* to externally owned integrators, so the caller can keep
//! mutating them (power continuation via `PUMPLaplacian::set_power`).

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use crate::assembler::ref_elem_vol_for_space;

// The per-QP geometry (`ElementTransformation` at the quadrature point) is the
// dist_solver port's helper — `pub(super)`, i.e. crate-internal.  The
// framework keeps using it so the navier miniapps and the filter miniapp see
// bit-identical geometry through one code path.
use crate::dist_solver::{elem_geometry, transform_grads};

// ─── Quadrature-point context (MFEM: ElementTransformation at the QP) ────────

/// Per-quadrature-point data handed to a [`NonlinearFormIntegrator`].
pub struct NLQpData<'a> {
    /// Number of local scalar-basis DOFs on this element (`nd`; the element
    /// residual has `nd·dim` entries in the interleaved `(k, c) → k·dim + c`
    /// layout).
    pub n_dofs: usize,
    /// Spatial dimension.
    pub dim: usize,
    /// Effective integration weight: quadrature weight × |det J| (MFEM
    /// `ip.weight · T.Weight()`).
    pub weight: f64,
    /// Basis values at this QP; length `n_dofs`.
    pub phi: &'a [f64],
    /// Physical gradients at this QP (MFEM `CalcPhysDShape`), row-major
    /// `[n_dofs × dim]`.
    pub grad_phys: &'a [f64],
    /// Physical coordinates of this QP; length `dim`.
    pub x_phys: &'a [f64],
    /// Reference coordinates of this QP (MFEM `IntegrationPoint` on the element
    /// transformation); length `dim`.
    pub xi: &'a [f64],
    /// Element index.
    pub elem: u32,
}

/// MFEM `NonlinearFormIntegrator` (element energy / residual / Jacobian), split
/// per quadrature point: the [`NonlinearForm`] driver loops the QPs and calls
/// the accumulators.
///
/// Like the C++ integrators ("these are not thread-safe!"), implementors are
/// only required to be [`Send`]: the serial [`NonlinearForm`] never shares them
/// across threads.  Interior-mutable state (`set_power`, `set_input`) is legal.
pub trait NonlinearFormIntegrator: Send {
    /// Extra quadrature-order offset for the vector/Jacobian assemblies
    /// (MFEM `PUMPLaplacian` uses `+1` for its vector and gradient forms).
    fn order_bonus(&self) -> i32 {
        0
    }

    /// Integration-rule override (MFEM `NonlinearFormIntegrator::SetIntRule`):
    /// `Some(order)` pins the shared volume quadrature of the form to
    /// `IntRules.Get(geom, order)`; `None` keeps the form default
    /// `2p + OrderGrad (+ order_bonus)`.
    fn int_rule_order(&self, _space_order: i32) -> Option<i32> {
        None
    }

    /// Energy contribution at this QP (MFEM `GetElementEnergy` inner loop).
    fn qp_energy(&self, _qp: &NLQpData<'_>, _elfun: &[f64]) -> f64 {
        0.0
    }

    /// Residual accumulation at this QP (MFEM `AssembleElementVector` inner loop).
    fn qp_residual(&self, qp: &NLQpData<'_>, elfun: &[f64], elvect: &mut [f64]);

    /// Jacobian accumulation at this QP (MFEM `AssembleElementGrad` inner loop).
    fn qp_jacobian(&self, qp: &NLQpData<'_>, elfun: &[f64], elmat: &mut [f64]);

    /// Live power update for power-continuation solvers (MFEM
    /// `pint->SetPower(pp)`; interior-mutable, no-op for most integrators).
    fn set_power(&self, _pp: f64) {}
}

// ─── NonlinearForm ───────────────────────────────────────────────────────────

/// Minimal serial `NonlinearForm`: references domain integrators and assembles
/// the residual (`mult`) and the gradient (`get_gradient`) over the elements of
/// a caller-provided space.
pub struct NonlinearForm<'a> {
    integs: Vec<&'a dyn NonlinearFormIntegrator>,
}

impl Default for NonlinearForm<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> NonlinearForm<'a> {
    pub fn new() -> Self {
        NonlinearForm { integs: Vec::new() }
    }

    /// Add a domain integrator (MFEM `NonlinearForm::AddDomainIntegrator`).
    pub fn add_domain_integrator(&mut self, integ: &'a dyn NonlinearFormIntegrator) {
        self.integs.push(integ);
    }

    /// Forward the continuation power to the integrators
    /// (MFEM `pint->SetPower(pp)` before each Newton solve).
    pub fn set_power_hint(&self, pp: f64) {
        for integ in &self.integs {
            integ.set_power(pp);
        }
    }

    /// Residual `y = F(x)` (MFEM `NonlinearForm::Mult` with zero RHS).
    pub fn mult<S: FESpace>(&self, space: &S, x: &[f64], y: &mut [f64]) {
        let quad_order = self.volume_quad_order(space, 0);
        for v in y.iter_mut() {
            *v = 0.0;
        }
        self.assemble_residual(space, x, y, quad_order);
    }

    /// Jacobian `J = ∂F/∂x` at `x` (MFEM `NonlinearForm::GetGradient`).
    pub fn get_gradient<S: FESpace>(&self, space: &S, x: &[f64]) -> CsrMatrix<f64> {
        let n = space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n, n);
        let quad_order = self.volume_quad_order(space, 1);
        self.assemble_jacobian(space, x, &mut coo, quad_order);
        coo.into_csr_sorted()
    }

    /// Total energy `E(x)` (MFEM `NonlinearForm::GetEnergy`).
    pub fn energy<S: FESpace>(&self, space: &S, x: &[f64]) -> f64 {
        let quad_order = self.volume_quad_order(space, 0);
        let mut total = 0.0_f64;
        self.assemble_energy(space, x, &mut total, quad_order);
        total
    }

    /// MFEM quadrature order: `2·p + OrderGrad (+ bonus + integrator extras)`
    /// with `OrderGrad = p − 1` for Pk/Qk spaces on P1 geometry, overridden
    /// per integrator by [`NonlinearFormIntegrator::int_rule_order`]
    /// (MFEM `SetIntRule`; the first override wins).
    fn volume_quad_order<S: FESpace>(&self, space: &S, bonus: i32) -> u8 {
        let p = space.order() as i32;
        if let Some(o) = self.integs.iter().find_map(|i| i.int_rule_order(p)) {
            return o.clamp(1, 32) as u8;
        }
        let extra: i32 = self.integs.iter().map(|i| i.order_bonus()).sum();
        (2 * p + (p - 1) + bonus + extra).clamp(1, 32) as u8
    }

    /// Reference element and the element's DOF list on `space` — the list has
    /// `nd·d` entries for a vector space (interleaved `(k, c) → k·d + c`) and
    /// `nd` entries for a scalar space.
    fn element_ref_elem<S: FESpace>(
        space: &S,
        e: u32,
    ) -> (Box<dyn fem_element::ReferenceElement>, Vec<usize>) {
        let mesh = space.mesh();
        let order = space.element_order(e);
        let re = ref_elem_vol_for_space(space, mesh.element_type(e), order);
        let dofs = space.element_dofs(e);
        let global_dofs: Vec<usize> = dofs.iter().map(|&d| d as usize).collect();
        (re, global_dofs)
    }

    fn assemble_residual<S: FESpace>(
        &self,
        space: &S,
        x: &[f64],
        y: &mut [f64],
        quad_order: u8,
    ) {
        let mesh = space.mesh();
        let dim = mesh.topological_dim() as usize;
        for e in mesh.elem_iter() {
            let (re, global_dofs) = Self::element_ref_elem(space, e);
            let n = re.n_dofs();
            let n_el = global_dofs.len();
            let mut elfun = vec![0.0_f64; n_el];
            for (i, &g) in global_dofs.iter().enumerate() {
                elfun[i] = x[g];
            }
            let mut elvect = vec![0.0_f64; n_el];
            let rule = re.quadrature(quad_order);
            let mut phi = vec![0.0_f64; n];
            let mut grad_ref = vec![0.0_f64; n * dim];
            let mut grad_phys = vec![0.0_f64; n * dim];
            for (q, xi) in rule.points.iter().enumerate() {
                let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
                let w = rule.weights[q] * det.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, n, dim);
                let qp = NLQpData {
                    n_dofs: n,
                    dim,
                    weight: w,
                    phi: &phi,
                    grad_phys: &grad_phys,
                    x_phys: &xp,
                    xi,
                    elem: e,
                };
                for integ in &self.integs {
                    integ.qp_residual(&qp, &elfun, &mut elvect);
                }
            }
            for (i, &g) in global_dofs.iter().enumerate() {
                y[g] += elvect[i];
            }
        }
    }

    fn assemble_jacobian<S: FESpace>(
        &self,
        space: &S,
        x: &[f64],
        coo: &mut CooMatrix<f64>,
        quad_order: u8,
    ) {
        let mesh = space.mesh();
        let dim = mesh.topological_dim() as usize;
        for e in mesh.elem_iter() {
            let (re, global_dofs) = Self::element_ref_elem(space, e);
            let n = re.n_dofs();
            let n_el = global_dofs.len();
            let mut elfun = vec![0.0_f64; n_el];
            for (i, &g) in global_dofs.iter().enumerate() {
                elfun[i] = x[g];
            }
            let mut elmat = vec![0.0_f64; n_el * n_el];
            let rule = re.quadrature(quad_order);
            let mut phi = vec![0.0_f64; n];
            let mut grad_ref = vec![0.0_f64; n * dim];
            let mut grad_phys = vec![0.0_f64; n * dim];
            for (q, xi) in rule.points.iter().enumerate() {
                let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
                let w = rule.weights[q] * det.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, n, dim);
                let qp = NLQpData {
                    n_dofs: n,
                    dim,
                    weight: w,
                    phi: &phi,
                    grad_phys: &grad_phys,
                    x_phys: &xp,
                    xi,
                    elem: e,
                };
                for integ in &self.integs {
                    integ.qp_jacobian(&qp, &elfun, &mut elmat);
                }
            }
            coo.add_element_matrix(&global_dofs, &elmat);
        }
    }

    fn assemble_energy<S: FESpace>(&self, space: &S, x: &[f64], total: &mut f64, quad_order: u8) {
        let mesh = space.mesh();
        let dim = mesh.topological_dim() as usize;
        for e in mesh.elem_iter() {
            let (re, global_dofs) = Self::element_ref_elem(space, e);
            let n = re.n_dofs();
            let n_el = global_dofs.len();
            let mut elfun = vec![0.0_f64; n_el];
            for (i, &g) in global_dofs.iter().enumerate() {
                elfun[i] = x[g];
            }
            let rule = re.quadrature(quad_order);
            let mut phi = vec![0.0_f64; n];
            let mut grad_ref = vec![0.0_f64; n * dim];
            let mut grad_phys = vec![0.0_f64; n * dim];
            for (q, xi) in rule.points.iter().enumerate() {
                let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
                let w = rule.weights[q] * det.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, n, dim);
                let qp = NLQpData {
                    n_dofs: n,
                    dim,
                    weight: w,
                    phi: &phi,
                    grad_phys: &grad_phys,
                    x_phys: &xp,
                    xi,
                    elem: e,
                };
                for integ in &self.integs {
                    *total += integ.qp_energy(&qp, &elfun);
                }
            }
        }
    }
}
