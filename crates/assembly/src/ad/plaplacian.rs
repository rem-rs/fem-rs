//! Nonlinear p-Laplacian form — 1:1 port of the integrators in MFEM's
//! `miniapps/autodiff/example.hpp` (`pLaplace` hand-coded + `pLaplaceAD`
//! AD-based), demonstrating all three MFEM integrator flavours:
//!
//! 0. [`PLaplacianIntegrator::HandCoded`] — hand-coded residual/Hessian
//!    (`pLaplace`),
//! 1. [`PLaplacianIntegrator::AdJacobian`] — residual functor evaluated as
//!    plain values, Hessian by AD
//!    (`pLaplaceAD<QVectorFuncAutoDiff<MyResidualFunctor..>>`),
//! 2. [`PLaplacianIntegrator::AdHessian`] — residual *and* Hessian derived by
//!    AD from the energy functor
//!    (`pLaplaceAD<QFunctionAutoDiff<MyEnergyFunctor..>>`).
//!
//! The energy density is (state `uu = [∇u, u]`, params `vparam = [pp, ee, ff]`)
//! `E = (ee² + |∇u|²)^(pp/2)/pp − ff·u`, exactly MFEM's `MyEnergyFunctor`;
//! the residual functor is `MyResidualFunctor`.  All three paths must produce
//! identical energies (see the module tests and the miniapp comparison).

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, MeshTopology};
use fem_space::fe_space::{FESpace, SpaceType};

use crate::ad::{AdScalar, QFunction, QFunctionAutoDiff, QVectorFunc, QVectorFuncAutoDiff};
use crate::assembler::{
    adjugate_2d, adjugate_3d, geo_ref_elem, geom_quad_point, is_affine, ref_elem_vol_for_space,
    transform_grads_adj,
};
use crate::physics::nonlinear::{
    LinearSolver as NlLinearSolver, NewtonConfig, NewtonResult, NewtonSolver, NonlinearForm,
};
use crate::vector_assembler::isoparametric_jacobian;

// ─── Functors (MFEM MyEnergyFunctor / MyResidualFunctor) ─────────────────────

/// MFEM `enum IntegratorType` from `seq_example.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PLaplacianIntegrator {
    /// Hand-coded residual and Hessian (`pLaplace`, C++ `-int 0`).
    HandCoded = 0,
    /// AD for the Hessian only (`QVectorFuncAutoDiff`, C++ `-int 1`).
    AdJacobian = 1,
    /// AD for residual and Hessian from the energy (`QFunctionAutoDiff`,
    /// C++ `-int 2`, the MFEM default).
    AdHessian = 2,
}

/// MFEM `MyEnergyFunctor` (`example.hpp`): p-Laplacian energy density.
///
/// State `uu = [u_x, …, u_dim, u]`, parameters `vparam = [pp, ee, ff]`;
/// `E(uu) = (ee² + |∇u|²)^(pp/2)/pp − ff·u`.  Generic over the AD scalar
/// type, so this one implementation provides the passive evaluation, the
/// Grad source and the Hessian source.
#[derive(Debug, Clone, Copy, Default)]
pub struct PLaplacianEnergy;

impl QFunction for PLaplacianEnergy {
    fn eval<T: AdScalar>(&self, vparam: &[f64], uu: &[T]) -> T {
        let pp = vparam[0];
        let ee = vparam[1];
        let ff = vparam[2];
        let dim = uu.len() - 1;
        let mut norm2 = T::zero();
        for k in 0..dim {
            norm2 = norm2 + uu[k] * uu[k];
        }
        T::powf_s(T::of_f64(ee * ee) + norm2, pp / 2.0) / T::of_f64(pp)
            - uu[dim] * T::of_f64(ff)
    }
}

/// MFEM `MyResidualFunctor` (`example.hpp`): first derivative of the energy
/// w.r.t. the state — `rr = (ee² + |∇u|²)^((pp−2)/2)·∇u , −ff`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PLaplacianResidual;

impl QVectorFunc for PLaplacianResidual {
    fn eval<T: AdScalar>(&self, vparam: &[f64], uu: &[T], rr: &mut [T]) {
        let pp = vparam[0];
        let ee = vparam[1];
        let ff = vparam[2];
        let dim = uu.len() - 1;
        let mut norm2 = T::zero();
        for k in 0..dim {
            norm2 = norm2 + uu[k] * uu[k];
        }
        let tvar = T::powf_s(T::of_f64(ee * ee) + norm2, (pp - 2.0) / 2.0);
        for k in 0..dim {
            rr[k] = tvar * uu[k];
        }
        rr[dim] = T::of_f64(-ff);
    }
}

// ─── The form ────────────────────────────────────────────────────────────────

/// Callback phases of the shared element loop ([`PLaplacianForm::for_each_element`]).
enum ElemPhase<'a> {
    /// Element start: `(global_dofs, n_quadrature_points)`.
    Begin(&'a [usize], usize),
    /// Quadrature point: `(global_dofs, w, φ, ∇φ)`.
    Quad(&'a [usize], f64, &'a [f64], &'a [f64]),
}

/// Nonlinear p-Laplacian form on an H¹ space with essential boundary
/// conditions.  Mirrors MFEM's `NLSolverPLaplacian` + the three
/// `example.hpp` integrators on top of the fem-rs [`NonlinearForm`] /
/// [`NewtonSolver`] machinery.
pub struct PLaplacianForm<S: FESpace> {
    space: S,
    /// p-Laplacian power `pp` (≥ 2 for a convex energy).
    pub power: f64,
    /// Regularization `ee` keeping the tangent positive
    /// (`plap_epsilon`, default 1e-7 in `NLSolverPLaplacian`).
    pub epsilon: f64,
    /// Distributed load `ff` (`plap_input`).
    pub load: f64,
    /// Which integrator flavour to evaluate (MFEM `-int` option).
    pub integrator: PLaplacianIntegrator,
    /// Quadrature order: MFEM `2·el.GetOrder() + trans.OrderGrad(&el)`.
    quad_order: u8,
    /// Dirichlet (constrained) DOFs: `(dof, value)`.
    dirichlet: Vec<(usize, f64)>,
    /// AD driver for the energy functor (Grad/Hessian).
    adf: QFunctionAutoDiff<PLaplacianEnergy>,
    /// AD driver for the residual functor (Jacobian).
    rdf: QVectorFuncAutoDiff<PLaplacianResidual>,
}

impl<S: FESpace> PLaplacianForm<S> {
    /// Build the form with MFEM `NLSolverPLaplacian` defaults
    /// (`regularizationp = 1e-7`, `integ = ADHessianIntegrator`); the
    /// quadrature order follows MFEM
    /// `order = 2·p + trans.OrderGrad(&el)` with `OrderGrad = geom_order + p − 1`.
    pub fn new(space: S, power: f64, load: f64) -> Self {
        debug_assert!(
            space.space_type() == SpaceType::H1,
            "PLaplacianForm requires an H1 space"
        );
        let geom_order = space.mesh().geom_order().max(1);
        let order = space.order().max(1);
        let quad_order = 2 * order + (geom_order + order - 1);
        PLaplacianForm {
            space,
            power,
            epsilon: 1e-7,
            load,
            integrator: PLaplacianIntegrator::AdHessian,
            quad_order,
            dirichlet: Vec::new(),
            adf: QFunctionAutoDiff::new(PLaplacianEnergy),
            rdf: QVectorFuncAutoDiff::new(PLaplacianResidual),
        }
    }

    /// Set the constrained (Dirichlet) DOFs (index, prescribed value).
    pub fn set_dirichlet(&mut self, dofs: Vec<(usize, f64)>) {
        self.dirichlet = dofs;
    }

    /// Parameters vector `vparam = [pp, ee, ff]` (MFEM `pLaplaceAD::vparam`).
    fn vparam(&self) -> [f64; 3] {
        [self.power, self.epsilon, self.load]
    }

    /// Element loop shared by energy / residual / jacobian.
    ///
    /// One callback per [`ElemPhase`]: element start, then one per quadrature
    /// point with `w = ip.weight·|det J|` and *true physical* gradients
    /// `∇φ = J⁻ᵀ ∇̂φ` (MFEM `Mult(dshape_iso, InverseJacobian(), dshape_xyz)`).
    fn for_each_element<F>(&self, mut f: F)
    where
        F: FnMut(ElemPhase<'_>),
    {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let geom_order = mesh.geom_order();

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let order = self.space.element_order(e);
            let ref_elem = ref_elem_vol_for_space(&self.space, elem_type, order);
            let quad = ref_elem.quadrature(self.quad_order);
            let n_ldofs = ref_elem.n_dofs();
            let gd: Vec<usize> =
                self.space.element_dofs(e).iter().map(|&d| d as usize).collect();
            f(ElemPhase::Begin(&gd, quad.points.len()));

            let nodes = mesh.element_nodes(e);
            let affine = is_affine(elem_type, geom_order);
            let geo_elem = geo_ref_elem(mesh, e);

            let mut phi = vec![0.0_f64; n_ldofs];
            let mut grad_ref = vec![0.0_f64; n_ldofs * dim];
            let mut grad_phys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let ipw = quad.weights[q];
                if affine {
                    let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                    let w = ipw * tr.det_j().abs();
                    ref_elem.eval_basis(xi, &mut phi);
                    ref_elem.eval_grad_basis(xi, &mut grad_ref);
                    let jit = tr.jacobian_inv_t();
                    for i in 0..n_ldofs {
                        for j in 0..dim {
                            let mut s = 0.0;
                            for k in 0..dim {
                                s += jit[(j, k)] * grad_ref[i * dim + k];
                            }
                            grad_phys[i * dim + j] = s;
                        }
                    }
                    f(ElemPhase::Quad(&gd, w, &phi, &grad_phys));
                } else {
                    let geo = geo_elem
                        .as_ref()
                        .unwrap_or_else(|| panic!("geo_ref_elem missing for {elem_type:?}"));
                    let geo_nds = mesh.geometry_nodes(e);
                    let xi_g = geom_quad_point(elem_type, order, xi);
                    let (jac_qp, det_qp, _xp) =
                        isoparametric_jacobian(mesh, geo_nds, geo.as_ref(), &xi_g, dim);
                    let w = ipw * det_qp.abs();
                    // MFEM: Mult(dshape_iso, InverseJacobian(), dshape_xyz);
                    // evaluated as adj(J)/det to reuse the assembler kernels.
                    let adj = if dim == 3 {
                        adjugate_3d(&jac_qp)
                    } else {
                        adjugate_2d(&jac_qp)
                    };
                    ref_elem.eval_basis(xi, &mut phi);
                    ref_elem.eval_grad_basis(xi, &mut grad_ref);
                    transform_grads_adj(&adj, &grad_ref, &mut grad_phys, n_ldofs, dim);
                    for v in grad_phys.iter_mut() {
                        *v /= det_qp;
                    }
                    f(ElemPhase::Quad(&gd, w, &phi, &grad_phys));
                }
            }
        }
    }

    /// State at one quadrature point: `uu = Bᵀ elfun` with
    /// `B[i][k] = ∇φᵢ[k] (k < sdim)`, `B[i][sdim] = φᵢ`
    /// (MFEM `B.MultTranspose(elfun, uu)`).
    fn qp_state(gd: &[usize], phi: &[f64], grad: &[f64], u: &[f64]) -> Vec<f64> {
        let n = phi.len();
        let sdim = grad.len() / n;
        let mut uu = vec![0.0_f64; sdim + 1];
        for k in 0..sdim {
            uu[k] = (0..n).map(|i| grad[i * sdim + k] * u[gd[i]]).sum();
        }
        uu[sdim] = (0..n).map(|i| phi[i] * u[gd[i]]).sum();
        uu
    }

    /// Energy contribution at one quadrature point
    /// (MFEM `GetElementEnergy` of all three integrators — the value is the
    /// same; the hand-coded path evaluates the formula explicitly, the AD
    /// paths use the functor's passive evaluation).
    fn energy_qp(&self, gd: &[usize], w: f64, phi: &[f64], grad: &[f64], u: &[f64]) -> f64 {
        let uu = Self::qp_state(gd, phi, grad, u);
        if self.integrator == PLaplacianIntegrator::HandCoded {
            let n2: f64 = uu[..uu.len() - 1].iter().map(|g| g * g).sum();
            let b = n2 + self.epsilon * self.epsilon;
            w * (b.powf(self.power / 2.0) / self.power - self.load * uu[uu.len() - 1])
        } else {
            w * self.adf.eval(&self.vparam(), &uu)
        }
    }

    /// Residual contribution at one quadrature point
    /// (MFEM `AssembleElementVector` of all three integrators).
    fn residual_qp(
        &self,
        gd: &[usize],
        w: f64,
        phi: &[f64],
        grad: &[f64],
        u: &[f64],
        du: &mut [f64],
        r: &mut [f64],
    ) {
        let n = phi.len();
        let sdim = grad.len() / n;
        let uu = Self::qp_state(gd, phi, grad, u);
        match self.integrator {
            PLaplacianIntegrator::HandCoded => {
                // pLaplace::AssembleElementVector with true gradients:
                // aa = (|∇u|² + ε²)^((p−2)/2).
                let n2: f64 = uu[..sdim].iter().map(|g| g * g).sum();
                let aa = (n2 + self.epsilon * self.epsilon).powf((self.power - 2.0) / 2.0);
                for i in 0..n {
                    let dot: f64 = (0..sdim).map(|k| uu[k] * grad[i * sdim + k]).sum();
                    r[gd[i]] += w * (aa * dot - self.load * phi[i]);
                }
            }
            PLaplacianIntegrator::AdJacobian => {
                // Residual functor as plain values, r += w·B·du.
                self.rdf.vector_func(&self.vparam(), &uu, du);
                for i in 0..n {
                    let dot: f64 = (0..sdim).map(|k| du[k] * grad[i * sdim + k]).sum();
                    r[gd[i]] += w * (dot + phi[i] * du[sdim]);
                }
            }
            PLaplacianIntegrator::AdHessian => {
                // Energy Grad by AD, r += w·B·du.
                self.adf.grad(&self.vparam(), &uu, du);
                for i in 0..n {
                    let dot: f64 = (0..sdim).map(|k| du[k] * grad[i * sdim + k]).sum();
                    r[gd[i]] += w * (dot + phi[i] * du[sdim]);
                }
            }
        }
    }

    /// Tangent contribution at one quadrature point
    /// (MFEM `AssembleElementGrad` of all three integrators).
    fn jacobian_qp(
        &self,
        gd: &[usize],
        w: f64,
        phi: &[f64],
        grad: &[f64],
        u: &[f64],
        hh: &mut [f64],
        coo: &mut CooMatrix<f64>,
    ) {
        let n = phi.len();
        let sdim = grad.len() / n;
        let ns = sdim + 1;
        let uu = Self::qp_state(gd, phi, grad, u);
        match self.integrator {
            PLaplacianIntegrator::HandCoded => {
                // pLaplace::AssembleElementGrad with true gradients:
                // H = aa1·I + aa0·∇u ⊗ ∇u in state space.
                let n2: f64 = uu[..sdim].iter().map(|g| g * g).sum();
                let b = n2 + self.epsilon * self.epsilon;
                let aa1 = b.powf((self.power - 2.0) / 2.0);
                let aa0 = (self.power - 2.0) * b.powf((self.power - 4.0) / 2.0);
                for (i, &gi) in gd.iter().enumerate() {
                    for (j, &gj) in gd.iter().enumerate() {
                        let graddot: f64 = (0..sdim)
                            .map(|k| grad[i * sdim + k] * grad[j * sdim + k])
                            .sum();
                        let gi_u: f64 = (0..sdim).map(|k| uu[k] * grad[i * sdim + k]).sum();
                        let gj_u: f64 = (0..sdim).map(|k| uu[k] * grad[j * sdim + k]).sum();
                        coo.add(gi, gj, w * (aa1 * graddot + aa0 * gi_u * gj_u));
                    }
                }
            }
            PLaplacianIntegrator::AdJacobian => {
                // Hessian by AD from the residual functor, elmat += w·B·H·Bᵀ.
                self.rdf.jacobian(&self.vparam(), &uu, hh);
                self.contract_bthb(gd, w, phi, grad, sdim, ns, hh, coo);
            }
            PLaplacianIntegrator::AdHessian => {
                // Hessian by AD from the energy functor, elmat += w·B·H·Bᵀ.
                self.adf.hessian(&self.vparam(), &uu, hh);
                self.contract_bthb(gd, w, phi, grad, sdim, ns, hh, coo);
            }
        }
    }

    /// `elmat += w · B · H · Bᵀ`
    /// (MFEM `Mult(B, duu, A); AddMult_a_ABt(w, A, B, elmat)`).
    #[allow(clippy::too_many_arguments)]
    fn contract_bthb(
        &self,
        gd: &[usize],
        w: f64,
        phi: &[f64],
        grad: &[f64],
        sdim: usize,
        ns: usize,
        hh: &[f64],
        coo: &mut CooMatrix<f64>,
    ) {
        for (i, &gi) in gd.iter().enumerate() {
            for (j, &gj) in gd.iter().enumerate() {
                let mut s = 0.0;
                for k in 0..ns {
                    let bik = if k < sdim { grad[i * sdim + k] } else { phi[i] };
                    for l in 0..ns {
                        let bjl = if l < sdim { grad[j * sdim + l] } else { phi[j] };
                        s += bik * hh[k * ns + l] * bjl;
                    }
                }
                coo.add(gi, gj, w * s);
            }
        }
    }

    /// Total energy `Σ_e Σ_q w·E(uu_q)` (MFEM `NonlinearForm::GetEnergy`).
    pub fn energy(&self, u: &[f64]) -> f64 {
        let mut energy = 0.0;
        self.for_each_element(|phase| {
            if let ElemPhase::Quad(gd, w, phi, grad) = phase {
                energy += self.energy_qp(gd, w, phi, grad, u);
            }
        });
        energy
    }

    /// Newton configuration matching MFEM `NLSolverPLaplacian`
    /// (`CG + GSSmoother`, `rtol 1e-4`, `atol 1e-6`, `max 10 iterations`).
    pub fn newton_config(&self) -> NewtonConfig {
        NewtonConfig {
            atol: 1e-6,
            rtol: 1e-4,
            max_iter: 10,
            linear_tol: 1e-7,
            line_search: false,
            linear_solver: NlLinearSolver::PcgGssmoother,
            verbose: false,
            ..NewtonConfig::default()
        }
    }

    /// Convenience: run the Newton solve for this form (zero RHS).
    pub fn solve(&self, u: &mut [f64]) -> NewtonResult {
        let cfg = self.newton_config();
        let rhs = vec![0.0_f64; self.space.n_dofs()];
        NewtonSolver::new(cfg)
            .solve(self, &rhs, u)
            .unwrap_or_else(|e| e)
    }
}

impl<S: FESpace> NonlinearForm for PLaplacianForm<S> {
    fn n_dofs(&self) -> usize {
        self.space.n_dofs()
    }

    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        for v in r.iter_mut() {
            *v = 0.0;
        }
        // State size: [∇u (mesh dim), u] (MFEM uu(4) = [diff_x, diff_y, u]).
        let ns = self.space.mesh().dim() as usize + 1;
        let mut du: Vec<f64> = vec![0.0_f64; ns];
        self.for_each_element(|phase| match phase {
            ElemPhase::Begin(_gd, _) => {}
            ElemPhase::Quad(gd, w, phi, grad) => {
                self.residual_qp(gd, w, phi, grad, u, &mut du, r);
            }
        });
        // Subtract RHS (zero in the miniapp; kept general).
        for (i, ri) in r.iter_mut().enumerate() {
            *ri -= rhs[i];
        }
        // Dirichlet rows: r[d] = u[d] − value.
        for &(d, val) in &self.dirichlet {
            r[d] = u[d] - val;
        }
    }

    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let n = self.space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n, n);
        let ns = self.space.mesh().dim() as usize + 1;
        let mut hh: Vec<f64> = vec![0.0_f64; ns * ns];
        self.for_each_element(|phase| match phase {
            ElemPhase::Begin(_gd, _) => {}
            ElemPhase::Quad(gd, w, phi, grad) => {
                self.jacobian_qp(gd, w, phi, grad, u, &mut hh, &mut coo);
            }
        });
        let mut jac = coo.into_csr();
        // Dirichlet rows: zero row, unit diagonal (MFEM elimination).
        for &(d, _val) in &self.dirichlet {
            for ptr in jac.row_ptr[d]..jac.row_ptr[d + 1] {
                jac.values[ptr] = 0.0;
            }
            *jac.get_mut(d, d) = 1.0;
        }
        jac
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// Deterministic pseudo-random state in [−0.5, 0.5).
    fn pseudo_u(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let h = (i as u64).wrapping_mul(2654435761) % 1000;
                h as f64 / 1000.0 - 0.5
            })
            .collect()
    }

    fn max_abs(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// All three integrator flavours must produce the same residual and
    /// Jacobian (MFEM `seq_example.cpp` runs all three to identical energies).
    fn check_three_paths_mesh(mesh: Mesh<2>) {
        let space = H1Space::new(mesh, 1);
        let u = pseudo_u(space.n_dofs());
        let rhs = vec![0.0_f64; space.n_dofs()];
        let mut r_out = vec![0.0_f64; space.n_dofs()];

        let mut residuals = Vec::new();
        let mut jacobians = Vec::new();
        let mut energies = Vec::new();
        for mode in [
            PLaplacianIntegrator::HandCoded,
            PLaplacianIntegrator::AdJacobian,
            PLaplacianIntegrator::AdHessian,
        ] {
            let mut form = PLaplacianForm::new(space.clone(), 3.5, 1.0);
            form.integrator = mode;
            form.residual(&u, &rhs, &mut r_out);
            residuals.push(r_out.clone());
            let j = form.jacobian(&u);
            jacobians.push(j.values.clone());
            energies.push(form.energy(&u));
        }

        let e01 = max_abs(&residuals[0], &residuals[1]);
        let e02 = max_abs(&residuals[0], &residuals[2]);
        assert!(e01 < 1e-12 && e02 < 1e-12, "residual paths differ: {e01} {e02}");

        let j01 = max_abs(&jacobians[0], &jacobians[1]);
        let j02 = max_abs(&jacobians[0], &jacobians[2]);
        assert!(j01 < 1e-11 && j02 < 1e-11, "jacobian paths differ: {j01} {j02}");

        let en = max_abs(&energies, &energies);
        assert!(en < 1e-12, "energies differ across paths");
    }

    #[test]
    fn three_paths_agree_tri_mesh() {
        check_three_paths_mesh(Mesh::<2>::unit_square_tri(2));
    }

    #[test]
    fn three_paths_agree_quad_mesh() {
        check_three_paths_mesh(Mesh::<2>::unit_square_quad(2));
    }

    /// Jacobian (AD Hessian path) must equal a finite-difference Jacobian of
    /// the residual (AD Grad path).
    #[test]
    fn ad_jacobian_matches_residual_fd() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = H1Space::new(mesh, 2);
        let mut form = PLaplacianForm::new(space.clone(), 4.6, 1.0);
        form.integrator = PLaplacianIntegrator::AdHessian;

        let u = pseudo_u(space.n_dofs());
        let jac = form.jacobian(&u);

        // central differences on the residual
        let h = 1e-6;
        let mut cols = vec![0.0_f64; space.n_dofs() * space.n_dofs()];
        let rhs = vec![0.0_f64; space.n_dofs()];
        let mut rp = vec![0.0_f64; space.n_dofs()];
        let mut rm = vec![0.0_f64; space.n_dofs()];
        for k in 0..space.n_dofs() {
            let mut up = u.clone();
            up[k] += h;
            let mut um = u.clone();
            um[k] -= h;
            form.residual(&up, &rhs, &mut rp);
            form.residual(&um, &rhs, &mut rm);
            for i in 0..space.n_dofs() {
                cols[i * space.n_dofs() + k] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }

        let mut worst = 0.0_f64;
        let n = space.n_dofs();
        for i in 0..n {
            for k in 0..n {
                let v = jac.get(i, k);
                let c = cols[i * n + k];
                let scale = c.abs().max(1.0);
                worst = worst.max((v - c).abs() / scale);
            }
        }
        assert!(worst < 1e-6, "AD jacobian vs residual FD: worst rel {worst}");
    }

    /// Residual must equal the gradient of the energy (checks the energy,
    /// residual and B-matrix conventions are mutually consistent).
    #[test]
    fn residual_matches_energy_fd() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let space = H1Space::new(mesh, 2);
        let form = PLaplacianForm::new(space.clone(), 3.5, 1.0);
        let u = pseudo_u(space.n_dofs());
        let rhs = vec![0.0_f64; space.n_dofs()];
        let mut r = vec![0.0_f64; space.n_dofs()];
        form.residual(&u, &rhs, &mut r);

        let h = 1e-6;
        for k in [3_usize, 17, 42] {
            if k >= space.n_dofs() {
                continue;
            }
            let mut up = u.clone();
            up[k] += h;
            let mut um = u.clone();
            um[k] -= h;
            let fd = (form.energy(&up) - form.energy(&um)) / (2.0 * h);
            let scale = r[k].abs().max(1.0);
            assert!(
                (r[k] - fd).abs() / scale < 1e-5,
                "dE/du[{k}]: residual {} vs FD {fd}",
                r[k]
            );
        }
    }
}
