//! Nonlinear-form framework and PDE filter — port of the `ScreenedPoisson`,
//! `PUMPLaplacian` integrators and the `PDEFilter` low-pass filter from MFEM
//! `miniapps/common/dist_solver.{hpp,cpp}`.
//!
//! MFEM drives these through `NonlinearForm` (`GetElementEnergy` /
//! `AssembleElementVector` / `AssembleElementGrad` per integrator); this module
//! provides the matching minimal [`NonlinearForm`] with element/quadrature
//! loops identical to the C++ (including MFEM's per-integrator quadrature
//! orders `2p + OrderGrad` and the `+1` bonus of `PUMPLaplacian`).
//!
//! Like the C++ `NonlinearForm::AddDomainIntegrator(integrator*)`, the form
//! stores *references* to externally owned integrators, so the caller can keep
//! mutating them (power continuation via `PUMPLaplacian::set_power`).

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

use crate::assembler::ref_elem_vol_for_space;

use super::{elem_geometry, nonsym_solve, transform_grads, DistScalarCoeff, DistVectorCoeff};

// ─── Quadrature-point context (MFEM: ElementTransformation at the QP) ────────

/// Per-quadrature-point data handed to a [`NonlinearFormIntegrator`].
pub struct NLQpData<'a> {
    /// Number of local DOFs on this element.
    pub n_dofs: usize,
    /// Spatial dimension.
    pub dim: usize,
    /// Effective integration weight: quadrature weight × |det J|.
    pub weight: f64,
    /// Basis values at this QP; length `n_dofs`.
    pub phi: &'a [f64],
    /// Physical gradients at this QP, row-major `[n_dofs × dim]`.
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
    /// with `OrderGrad = p − 1` for Pk/Qk spaces on P1 geometry.
    fn volume_quad_order<S: FESpace>(&self, space: &S, bonus: i32) -> u8 {
        let p = space.order() as i32;
        let extra: i32 = self.integs.iter().map(|i| i.order_bonus()).sum();
        (2 * p + (p - 1) + bonus + extra).clamp(1, 32) as u8
    }

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
            let nd = re.n_dofs();
            let mut elfun = vec![0.0_f64; nd];
            for (i, &g) in global_dofs.iter().enumerate() {
                elfun[i] = x[g];
            }
            let mut elvect = vec![0.0_f64; nd];
            let rule = re.quadrature(quad_order);
            let mut phi = vec![0.0_f64; nd];
            let mut grad_ref = vec![0.0_f64; nd * dim];
            let mut grad_phys = vec![0.0_f64; nd * dim];
            for (q, xi) in rule.points.iter().enumerate() {
                let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
                let w = rule.weights[q] * det.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, nd, dim);
                let qp = NLQpData {
                    n_dofs: nd,
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
            let nd = re.n_dofs();
            let mut elfun = vec![0.0_f64; nd];
            for (i, &g) in global_dofs.iter().enumerate() {
                elfun[i] = x[g];
            }
            let mut elmat = vec![0.0_f64; nd * nd];
            let rule = re.quadrature(quad_order);
            let mut phi = vec![0.0_f64; nd];
            let mut grad_ref = vec![0.0_f64; nd * dim];
            let mut grad_phys = vec![0.0_f64; nd * dim];
            for (q, xi) in rule.points.iter().enumerate() {
                let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
                let w = rule.weights[q] * det.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, nd, dim);
                let qp = NLQpData {
                    n_dofs: nd,
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
            let nd = re.n_dofs();
            let mut elfun = vec![0.0_f64; nd];
            for (i, &g) in global_dofs.iter().enumerate() {
                elfun[i] = x[g];
            }
            let rule = re.quadrature(quad_order);
            let mut phi = vec![0.0_f64; nd];
            let mut grad_ref = vec![0.0_f64; nd * dim];
            let mut grad_phys = vec![0.0_f64; nd * dim];
            for (q, xi) in rule.points.iter().enumerate() {
                let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
                let w = rule.weights[q] * det.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, nd, dim);
                let qp = NLQpData {
                    n_dofs: nd,
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

// ─── ScreenedPoisson ─────────────────────────────────────────────────────────

/// Formulation for the Screened-Poisson equation (MFEM `ScreenedPoisson`).
/// The positive part of the input coefficient supplies unit volumetric loading,
/// the negative part negative unit volumetric loading.  The parameter `rh` is
/// the radius of a linear cone filter with a similar smoothing effect; it
/// determines the length scale of the smoothing.
///
/// This integrator is *linear* in `u` (the Jacobian is independent of `elfun`),
/// which [`PDEFilter`] exploits to cache the operator.
pub struct ScreenedPoisson<'a> {
    diffcoef: f64,
    func: std::cell::RefCell<Box<dyn DistScalarCoeff + 'a>>,
}

impl<'a> ScreenedPoisson<'a> {
    /// `diffcoef = (rh / (2·√3))²` (MFEM constructor).
    pub fn new(nfunc: Box<dyn DistScalarCoeff + 'a>, rh: f64) -> Self {
        let rd = rh / (2.0 * (3.0_f64).sqrt());
        ScreenedPoisson {
            diffcoef: rd * rd,
            func: std::cell::RefCell::new(nfunc),
        }
    }

    /// Replace the input coefficient (MFEM `SetInput`).
    pub fn set_input(&self, nfunc: Box<dyn DistScalarCoeff + 'a>) {
        *self.func.borrow_mut() = nfunc;
    }
}

impl NonlinearFormIntegrator for ScreenedPoisson<'_> {
    fn qp_energy(&self, qp: &NLQpData<'_>, elfun: &[f64]) -> f64 {
        let n = qp.n_dofs;
        let d = qp.dim;
        let fval = self.func.borrow().eval(qp.elem, qp.xi, qp.x_phys);
        // qval = Bᵀ·elfun; ngrad2 = |qval|².
        let mut ngrad2 = 0.0_f64;
        for jj in 0..d {
            let mut qj = 0.0_f64;
            for i in 0..n {
                qj += qp.grad_phys[i * d + jj] * elfun[i];
            }
            ngrad2 += qj * qj;
        }
        let mut energy = qp.weight * ngrad2 * self.diffcoef * 0.5;
        let pval: f64 = (0..n).map(|i| qp.phi[i] * elfun[i]).sum();
        energy += qp.weight * pval * pval * 0.5;
        // External load: -1 if fval > 0, +1 if fval < 0.
        if fval > 0.0 {
            energy -= qp.weight * pval;
        } else if fval < 0.0 {
            energy += qp.weight * pval;
        }
        energy
    }

    fn qp_residual(&self, qp: &NLQpData<'_>, elfun: &[f64], elvect: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let fval = self.func.borrow().eval(qp.elem, qp.xi, qp.x_phys);
        // lvec = B·(Bᵀ·elfun): diffusion part of the residual.
        let mut qval = vec![0.0_f64; d];
        for jj in 0..d {
            for i in 0..n {
                qval[jj] += qp.grad_phys[i * d + jj] * elfun[i];
            }
        }
        for i in 0..n {
            let mut lvec_i = 0.0_f64;
            for jj in 0..d {
                lvec_i += qp.grad_phys[i * d + jj] * qval[jj];
            }
            elvect[i] += qp.weight * self.diffcoef * lvec_i;
        }
        // Mass part: w·(φ·u)·φ.
        let pval: f64 = (0..n).map(|i| qp.phi[i] * elfun[i]).sum();
        for i in 0..n {
            elvect[i] += qp.weight * pval * qp.phi[i];
        }
        // Load: -1 if fval > 0, +1 if fval < 0.
        if fval > 0.0 {
            for i in 0..n {
                elvect[i] -= qp.weight * qp.phi[i];
            }
        } else if fval < 0.0 {
            for i in 0..n {
                elvect[i] += qp.weight * qp.phi[i];
            }
        }
    }

    fn qp_jacobian(&self, qp: &NLQpData<'_>, _elfun: &[f64], elmat: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let w = qp.weight;
        // AddMult_a_VVt(w, shape): mass block.
        for i in 0..n {
            let avi = w * qp.phi[i];
            for j in 0..n {
                elmat[i * n + j] += avi * qp.phi[j];
            }
        }
        // AddMult_a_AAt(w·diffcoef, B): diffusion block (B·Bᵀ).
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0_f64;
                for dd in 0..d {
                    dot += qp.grad_phys[i * d + dd] * qp.grad_phys[j * d + dd];
                }
                elmat[i * n + j] += w * self.diffcoef * dot;
            }
        }
    }
}

// ─── PUMPLaplacian ───────────────────────────────────────────────────────────

/// p-Laplace integrator of the PUM formulation (MFEM `PUMPLaplacian`):
/// `∫ ((|f|·∇u + u·∇|f|)² + ee²)^(p/2) / p ∓ (f·u) dx`.
pub struct PUMPLaplacian<'a> {
    func: Box<dyn DistScalarCoeff + 'a>,
    fgrad: Box<dyn DistVectorCoeff + 'a>,
    pp: std::cell::Cell<f64>,
    ee: std::cell::Cell<f64>,
}

impl<'a> PUMPLaplacian<'a> {
    pub fn new(nfunc: Box<dyn DistScalarCoeff + 'a>, nfgrad: Box<dyn DistVectorCoeff + 'a>) -> Self {
        PUMPLaplacian {
            func: nfunc,
            fgrad: nfgrad,
            pp: std::cell::Cell::new(2.0),
            ee: std::cell::Cell::new(1e-7),
        }
    }

    /// Set the regularization `ee` (MFEM `SetReg`).
    pub fn set_reg(&self, ee: f64) {
        self.ee.set(ee);
    }
}

impl NonlinearFormIntegrator for PUMPLaplacian<'_> {
    fn order_bonus(&self) -> i32 {
        1 // MFEM: the vector and gradient orders carry `+1`
    }

    fn set_power(&self, pp: f64) {
        self.pp.set(pp);
    }

    fn qp_energy(&self, qp: &NLQpData<'_>, elfun: &[f64]) -> f64 {
        let n = qp.n_dofs;
        let d = qp.dim;
        let pp = self.pp.get();
        let ee = self.ee.get();
        let mut fval = self.func.eval(qp.elem, qp.xi, qp.x_phys);
        let mut vgrad = vec![0.0_f64; d];
        self.fgrad.eval(qp.elem, qp.xi, qp.x_phys, &mut vgrad);
        let tval = fval;
        if fval < 0.0 {
            fval = -fval;
            for v in vgrad.iter_mut() {
                *v = -*v;
            }
        }
        // B columns: fval·dshape_col + vgrad[jj]·shape; qval = Bᵀ·elfun.
        let mut ngrad2 = 0.0_f64;
        for jj in 0..d {
            let mut qj = 0.0_f64;
            for i in 0..n {
                qj += (fval * qp.grad_phys[i * d + jj] + vgrad[jj] * qp.phi[i]) * elfun[i];
            }
            ngrad2 += qj * qj;
        }
        let mut energy = qp.weight * (ngrad2 + ee * ee).powf(pp / 2.0) / pp;
        let pval: f64 = (0..n).map(|i| qp.phi[i] * elfun[i]).sum();
        if tval > 0.0 {
            energy -= qp.weight * pval * tval;
        } else if tval < 0.0 {
            energy += qp.weight * pval * tval;
        }
        energy
    }

    fn qp_residual(&self, qp: &NLQpData<'_>, elfun: &[f64], elvect: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let pp = self.pp.get();
        let ee = self.ee.get();
        let mut fval = self.func.eval(qp.elem, qp.xi, qp.x_phys);
        let mut vgrad = vec![0.0_f64; d];
        self.fgrad.eval(qp.elem, qp.xi, qp.x_phys, &mut vgrad);
        let tval = fval;
        if fval < 0.0 {
            fval = -fval;
            for v in vgrad.iter_mut() {
                *v = -*v;
            }
        }
        // qval = Bᵀ·elfun.
        let mut qval = vec![0.0_f64; d];
        for jj in 0..d {
            for i in 0..n {
                qval[jj] += (fval * qp.grad_phys[i * d + jj] + vgrad[jj] * qp.phi[i]) * elfun[i];
            }
        }
        let mut ngrad2 = 0.0_f64;
        for q in &qval {
            ngrad2 += q * q;
        }
        // lvec = B·qval; elvect += w·aa·lvec with aa = (ngrad²+ee²)^((pp−2)/2).
        // B(i,jj) = fval·grad_phys(i,jj) + vgrad(jj)·phi(i), so
        // lvec_i = fval·(∇φ_i·qval) + (vgrad·qval)·φ_i.
        let aa = (ngrad2 + ee * ee).powf((pp - 2.0) / 2.0);
        let vq: f64 = (0..d).map(|jj| vgrad[jj] * qval[jj]).sum();
        for i in 0..n {
            let mut lvec_i = 0.0_f64;
            for jj in 0..d {
                lvec_i += qp.grad_phys[i * d + jj] * qval[jj];
            }
            lvec_i = fval * lvec_i + vq * qp.phi[i];
            elvect[i] += qp.weight * aa * lvec_i;
        }
        // Load: -w·fval·φ if tval > 0, +w·fval·φ if tval < 0.
        if tval > 0.0 {
            for i in 0..n {
                elvect[i] -= qp.weight * fval * qp.phi[i];
            }
        } else if tval < 0.0 {
            for i in 0..n {
                elvect[i] += qp.weight * fval * qp.phi[i];
            }
        }
    }

    fn qp_jacobian(&self, qp: &NLQpData<'_>, elfun: &[f64], elmat: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let pp = self.pp.get();
        let ee = self.ee.get();
        let mut fval = self.func.eval(qp.elem, qp.xi, qp.x_phys);
        let mut vgrad = vec![0.0_f64; d];
        self.fgrad.eval(qp.elem, qp.xi, qp.x_phys, &mut vgrad);
        if fval < 0.0 {
            fval = -fval;
            for v in vgrad.iter_mut() {
                *v = -*v;
            }
        }
        // qval = Bᵀ·elfun.
        let mut qval = vec![0.0_f64; d];
        for jj in 0..d {
            for i in 0..n {
                qval[jj] += (fval * qp.grad_phys[i * d + jj] + vgrad[jj] * qp.phi[i]) * elfun[i];
            }
        }
        let mut ngrad2 = 0.0_f64;
        for q in &qval {
            ngrad2 += q * q;
        }
        let aa = ngrad2 + ee * ee;
        let aa1 = aa.powf((pp - 2.0) / 2.0);
        let aa0 = (pp - 2.0) * aa.powf((pp - 4.0) / 2.0);
        // AddMult_a_VVt(w·aa0, lvec) + AddMult_a_AAt(w·aa1, B), with
        // lvec = B·qval = fval·(∇φᵢ·qval) + (vgrad·qval)·φ_i and the full
        // B rows B(i,dd) = fval·∇φ_i(dd) + φ_i·vgrad(dd).
        let vq: f64 = (0..d).map(|jj| vgrad[jj] * qval[jj]).sum();
        for i in 0..n {
            let mut lvec_i = 0.0_f64;
            for jj in 0..d {
                lvec_i += qp.grad_phys[i * d + jj] * qval[jj];
            }
            lvec_i = fval * lvec_i + vq * qp.phi[i];
            for j in 0..n {
                let mut lvec_j = 0.0_f64;
                for jj in 0..d {
                    lvec_j += qp.grad_phys[j * d + jj] * qval[jj];
                }
                lvec_j = fval * lvec_j + vq * qp.phi[j];
                elmat[i * n + j] += qp.weight * aa0 * lvec_i * lvec_j;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0_f64;
                for dd in 0..d {
                    // B(i,dd)·B(j,dd) with B = fval·∇φ + φ·vgrad.
                    let bi = fval * qp.grad_phys[i * d + dd] + vgrad[dd] * qp.phi[i];
                    let bj = fval * qp.grad_phys[j * d + dd] + vgrad[dd] * qp.phi[j];
                    dot += bi * bj;
                }
                elmat[i * n + j] += qp.weight * aa1 * dot;
            }
        }
    }
}

// ─── PDEFilter ───────────────────────────────────────────────────────────────

/// Low-pass filter based on the Screened Poisson equation
/// (MFEM `PDEFilter`; B. S. Lazarov, O. Sigmund: "Filters in topology
/// optimization based on Helmholtz-type differential equations",
/// DOI:10.1002/nme.3072).
///
/// Owns its internal H1 space of order `order` (default 2) built on a clone of
/// the mesh — like the C++, which holds `ParFiniteElementSpace fesp` by value.
/// The lifetime parameter `'a` is the lifetime of the accepted input
/// coefficients (inferred at first use; `'static` for closures).
pub struct PDEFilter<'a, M: MeshTopology + Clone + 'static> {
    rr: f64,
    maxiter: usize,
    rtol: f64,
    atol: f64,
    /// Internal space (owned — the C++ holds it by value too).
    fesp: H1Space<M>,
    /// ScreenedPoisson integrator (created eagerly with a placeholder input;
    /// MFEM creates it lazily on the first `Filter` — numerically identical
    /// because the operator is independent of the input coefficient).
    sint: ScreenedPoisson<'a>,
    /// Filtered field DOFs on `fesp` (MFEM `gf`).
    gf: Vec<f64>,
    /// Solution of the last filtering (MFEM `sv`).
    sv: Vec<f64>,
    /// Cached operator `A = diffcoef·K + M` (MFEM caches `nf->GetGradient(*sv)`
    /// on the first call — the ScreenedPoisson Jacobian is independent of both
    /// `x` and the input coefficient).
    a: Option<CsrMatrix<f64>>,
}

impl<'a, M: MeshTopology + Clone + 'static> PDEFilter<'a, M> {
    pub fn new(mesh: M, rh: f64) -> Self {
        PDEFilter::with_options(mesh, rh, 2, 100, 1e-12, 1e-15, 0)
    }

    pub fn with_options(
        mesh: M,
        rh: f64,
        order: u8,
        maxiter: usize,
        rtol: f64,
        atol: f64,
        _print_lv: u8,
    ) -> Self {
        let fesp = H1Space::new(mesh, order);
        let n = fesp.n_dofs();
        PDEFilter {
            rr: rh,
            maxiter,
            rtol,
            atol,
            sint: ScreenedPoisson::new(Box::new(super::ConstDistCoeff(0.0)), rh),
            gf: vec![0.0; n],
            sv: vec![0.0; n],
            fesp,
            a: None,
        }
    }

    /// Number of DOFs of the internal filter space.
    pub fn n_dofs(&self) -> usize {
        self.fesp.n_dofs()
    }

    /// Filter a coefficient field into `ffield` (DOFs on the internal space).
    ///
    /// MFEM `PDEFilter::Filter(Coefficient&, ParGridFunction&)`: on the first
    /// call the ScreenedPoisson operator is assembled and cached; later calls
    /// only swap the input coefficient (`SetInput`).
    pub fn filter_coeff(&mut self, func: Box<dyn DistScalarCoeff + 'a>, ffield: &mut [f64]) {
        assert_eq!(
            ffield.len(),
            self.fesp.n_dofs(),
            "PDEFilter: ffield must live on the internal filter space"
        );
        let n = self.fesp.n_dofs();
        let zeros = vec![0.0_f64; n];

        // sint->SetInput(func) (also covers the lazy first-call construction).
        self.sint.set_input(func);

        let mut nf = NonlinearForm::new();
        nf.add_domain_integrator(&self.sint);

        if self.a.is_none() {
            // Operator: gradient at x = 0 (independent of x and of the input).
            self.a = Some(nf.get_gradient(&self.fesp, &zeros));
        }

        // RHS: residual at x = 0, i.e. R(0) = −b.
        let mut rhs = vec![0.0_f64; n];
        nf.mult(&self.fesp, &zeros, &mut rhs);

        // Solve A·sv = R(0); the filtered field is the negated solution.
        let a = self.a.as_ref().expect("PDEFilter operator");
        let mut sv = vec![0.0_f64; n];
        nonsym_solve(a, &rhs, &mut sv, self.rtol, self.atol, self.maxiter);
        for (g, s) in self.gf.iter_mut().zip(sv.iter()) {
            *g = -s;
        }
        self.sv.copy_from_slice(&sv);
        ffield.copy_from_slice(&self.gf);
    }

    /// Filter a grid function given by its DOF vector on the internal space
    /// (MFEM `Filter(ParGridFunction&, ParGridFunction&)` via
    /// `GridFunctionCoefficient`).  The DOFs are captured into a self-owned
    /// coefficient, so this works for any lifetime of the filter.
    pub fn filter_gf(&mut self, func_dofs: &[f64], ffield: &mut [f64]) {
        assert_eq!(func_dofs.len(), self.fesp.n_dofs());
        // Capture everything the evaluation needs by value ('static).
        let mesh = self.fesp.mesh();
        let order = self.fesp.order();
        let n_e = mesh.n_elements();
        let mut elem_types = Vec::with_capacity(n_e);
        let mut elem_dofs = Vec::with_capacity(n_e);
        for e in mesh.elem_iter() {
            elem_types.push(mesh.element_type(e));
            elem_dofs.push(self.fesp.element_dofs(e).iter().map(|&d| d as usize).collect());
        }
        self.filter_coeff(
            Box::new(OwnedGfCoeff {
                order,
                elem_types,
                elem_dofs,
                dofs: func_dofs.to_vec(),
            }),
            ffield,
        );
    }

    /// DOFs of the last filtered field on the internal space.
    pub fn filtered_dofs(&self) -> &[f64] {
        &self.gf
    }
}

/// Owned-DOF grid function coefficient: evaluates `sum_j u_j phi_j(xi)` with
/// the H1 reference basis of the internal filter space (a self-contained
/// `GridFunctionCoefficient`).
struct OwnedGfCoeff {
    order: u8,
    elem_types: Vec<fem_mesh::element_type::ElementType>,
    elem_dofs: Vec<Vec<usize>>,
    dofs: Vec<f64>,
}

impl DistScalarCoeff for OwnedGfCoeff {
    fn eval(&self, elem: u32, xi: &[f64], _x: &[f64]) -> f64 {
        let re = crate::assembler::ref_elem_vol_h1(self.elem_types[elem as usize], self.order);
        let nd = re.n_dofs();
        let mut phi = vec![0.0_f64; nd];
        re.eval_basis(xi, &mut phi);
        let dofs = &self.elem_dofs[elem as usize];
        (0..nd).map(|i| self.dofs[dofs[i]] * phi[i]).sum()
    }
}
