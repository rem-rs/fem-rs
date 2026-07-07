//! Scalar isotropic continuum damage mechanics (CDM).
//!
//! Implements coupled damage-elasticity with:
//! - **Kachanov-type** damage evolution: `ḋ = (Y/S₀)ⁿ`
//! - **Exponential (Mazars-type)** damage: `d = 1 - exp(-(κ-κ₀)/κ_c)`
//! - Staggered (alternating) solver: solve elasticity → compute damage → repeat
//!
//! ## Constitutive law
//! ```text
//! σ = (1 - d)·C : ε
//! ```
//!
//! where `d ∈ [0,1)` is the scalar damage variable, `C` is the elastic
//! stiffness, and `ε` is the small-strain tensor.
//!
//! ## Damage evolution (Kachanov)
//! ```text
//! Y = ½·ε:C:ε   (energy release rate)
//! ḋ = (Y / S₀)ⁿ  for Y > Y₀
//! ```
//!
//! ## Usage
//! ```rust,ignore
//! use fem_assembly::damage::*;
//!
//! let cfg = DamageConfig::kachanov(2e5, 0.3, 1e-3, 1.0, 2.0);
//! let mut state = DamageState::new(n_qp);
//! let solver = StaggeredDamageSolver::new(space, cfg, quad_order);
//! let u = solver.solve(&stiffness, &rhs, &mut state, 50, 1e-8);
//! ```

use nalgebra::DMatrix;
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

/// Damage evolution law type.
#[derive(Debug, Clone)]
pub enum DamageLaw {
    /// Kachanov-type: ḋ = (Y/S₀)ⁿ, Y = ½·ε:C:ε
    Kachanov { S0: f64, n: f64, Y0: f64 },
    /// Exponential (Mazars): d = 1 - exp(-(κ - κ₀)/κ_c)
    Exponential { kappa_0: f64, kappa_c: f64, d_max: f64 },
}

/// Damage material configuration.
#[derive(Debug, Clone)]
pub struct DamageConfig {
    pub E: f64,
    pub nu: f64,
    pub law: DamageLaw,
    /// Residual stiffness (keeps system well-posed as d→1).
    pub kappa_eps: f64,
}

impl DamageConfig {
    pub fn kachanov(E: f64, nu: f64, S0: f64, n: f64, Y0: f64) -> Self {
        Self { E, nu, law: DamageLaw::Kachanov { S0, n, Y0 }, kappa_eps: 1e-8 }
    }
    pub fn exponential(E: f64, nu: f64, kappa_0: f64, kappa_c: f64) -> Self {
        Self { E, nu, law: DamageLaw::Exponential { kappa_0, kappa_c, d_max: 0.99 }, kappa_eps: 1e-8 }
    }
    fn mu(&self) -> f64 { self.E / (2.0 * (1.0 + self.nu)) }
    fn lambda(&self) -> f64 { self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu)) }
    fn elastic_stiffness(&self, dim: usize) -> DMatrix<f64> {
        let mu = self.mu();
        let lam = self.lambda();
        if dim == 2 {
            DMatrix::from_row_slice(3, 3, &[
                lam + 2.0*mu, lam,           0.0,
                lam,           lam + 2.0*mu,  0.0,
                0.0,          0.0,           mu,
            ])
        } else {
            let mut c = DMatrix::zeros(6, 6);
            for i in 0..3 { for j in 0..3 { c[(i,j)] = lam; } c[(i,i)] = lam + 2.0*mu; }
            c[(3,3)] = mu; c[(4,4)] = mu; c[(5,5)] = mu;
            c
        }
    }
}

/// Per-quadrature-point damage state.
pub struct DamageState {
    /// Damage variable d at each QP.
    pub d: Vec<f64>,
    /// History variable (maximum equivalent strain / energy).
    pub history: Vec<f64>,
}

impl DamageState {
    pub fn new(n_qp: usize) -> Self {
        Self { d: vec![0.0; n_qp], history: vec![0.0; n_qp] }
    }
}

// ─── Damage update kernel ─────────────────────────────────────────────────────

/// Update damage variable at one QP based on the current strain.
/// Returns the new damage value d (and the history variable is updated in-place).
fn update_damage(eps: &[f64], cfg: &DamageConfig, h: &mut f64, d_old: f64) -> f64 {
    // Equivalent strain (Mazars definition)
    let eps_eq = eps.iter().map(|v| v * v).sum::<f64>().max(0.0_f64).sqrt();
    *h = h.max(eps_eq);
    match &cfg.law {
        DamageLaw::Kachanov { S0, n, Y0 } => {
            // Energy release rate
            let c = cfg.elastic_stiffness(if eps.len() <= 3 { 2 } else { 3 });
            let mut Y = 0.0;
            for i in 0..eps.len() {
                for j in 0..eps.len() { Y += 0.5 * eps[i] * c[(i,j)] * eps[j]; }
            }
            if Y <= *Y0 { return d_old; }
            let d_new = (Y / S0).powf(*n);
            d_new.min(0.99).max(d_old)
        }
        DamageLaw::Exponential { kappa_0, kappa_c, d_max } => {
            if *h <= *kappa_0 { return d_old; }
            let d_new = 1.0 - (-(*h - *kappa_0) / kappa_c).exp();
            d_new.min(*d_max).max(d_old)
        }
    }
}

// ─── Strain energy computation ────────────────────────────────────────────────

fn strain_at_qp(u_elem: &[f64], gphys: &[f64], dim: usize, n_ldofs: usize) -> Vec<f64> {
    let n_comp = if dim == 2 { 3 } else { 6 };
    let mut eps = vec![0.0; n_comp];
    for k in 0..n_ldofs {
        for i in 0..dim { eps[i] += u_elem[k * dim + i] * gphys[k * dim + i]; }
        if dim == 2 {
            eps[2] += u_elem[k * dim] * gphys[k * dim + 1]
                    + u_elem[k * dim + 1] * gphys[k * dim];
        } else {
            eps[3] += u_elem[k*dim] * gphys[k*dim+1] + u_elem[k*dim+1] * gphys[k*dim];
            eps[4] += u_elem[k*dim+1] * gphys[k*dim+2] + u_elem[k*dim+2] * gphys[k*dim+1];
            eps[5] += u_elem[k*dim] * gphys[k*dim+2] + u_elem[k*dim+2] * gphys[k*dim];
        }
    }
    eps
}

// ─── Assembly ─────────────────────────────────────────────────────────────────

/// Assemble the damaged stiffness matrix and internal force vector.
///
/// `d_at_qp` — damage values at each QP (from previous staggered iteration).
pub fn assemble_damaged_elasticity<M: MeshTopology>(
    mesh: &M,
    space: &dyn FESpace<Mesh = M>,
    u: &[f64],
    d_at_qp: &[f64],
    cfg: &DamageConfig,
    quad_order: u8,
) -> (Vec<f64>, CsrMatrix<f64>) {
    let dim = mesh.dim() as usize;
    let n_dofs = space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let mut f_int = vec![0.0; n_dofs];
    let c_e = cfg.elastic_stiffness(dim);
    let kap = cfg.kappa_eps;

    // Build element DOF cache
    let mut elem_dofs_cache = Vec::new();
    let mut n_vec = 0usize;
    for e in mesh.elem_iter() {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        n_vec = dofs.len(); // n_ldofs * dim for vector space
        elem_dofs_cache.push(dofs);
    }
    let n_vec = elem_dofs_cache[0].len();
    let n_ldofs = ref_elem_vol(mesh.element_type(0), space.order()).n_dofs();

    let mut qp_idx = 0usize;
    for (el, e) in mesh.elem_iter().enumerate() {
        let et = mesh.element_type(e);
        let re = ref_elem_vol(et, space.order());
        let quad = re.quadrature(quad_order);
        let n_qp_e = quad.weights.len();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac_2(mesh, nodes, dim);
        let jit = jac.try_inverse().map(|m| m.transpose()).unwrap_or_else(|| DMatrix::identity(dim, dim));

        let mut u_elem = vec![0.0_f64; n_vec];
        for (k, &dof) in elem_dofs_cache[el].iter().enumerate() { u_elem[k] = u[dof]; }

        let mut k_elem = vec![0.0_f64; n_vec * n_vec];
        let mut f_elem = vec![0.0_f64; n_vec];
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut gref = vec![0.0_f64; n_ldofs * dim];
        let mut gphys = vec![0.0_f64; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

            let eps = strain_at_qp(&u_elem, &gphys, dim, n_ldofs);
            let d_val = d_at_qp[qp_idx + q].min(0.99);
            let degrade = (1.0 - d_val).powi(2) + kap;

            // Stress: σ = (1-d)² · C : ε
            let mut sigma = vec![0.0; if dim == 2 { 3 } else { 6 }];
            for i in 0..sigma.len() {
                for j in 0..sigma.len() { sigma[i] += c_e[(i,j)] * eps[j]; }
                sigma[i] *= 1.0 - d_val;
            }

            // Internal force: f_int += Bᵀ·σ·w
            for k in 0..n_ldofs {
                for i in 0..dim {
                    let row = k * dim + i;
                    let mut s = 0.0;
                    for j in 0..dim {
                        let sig_idx = if i == j { i } else if dim == 2 { 2 } else { 3 + (i+j-1)%3 };
                        s += sigma[sig_idx] * gphys[k * dim + j];
                    }
                    f_elem[row] += w * s;
                }
            }

            // Tangent: K += Bᵀ·(1-d)²·C·B·w
            for k in 0..n_ldofs {
                for i in 0..dim {
                    let row = k * dim + i;
                    for l in 0..n_ldofs {
                        for a in 0..dim {
                            let col = l * dim + a;
                            let mut val = 0.0;
                            for j in 0..dim {
                                for b in 0..dim {
                                    let cij = if i==j { i } else if dim==2 { 2 } else { 3+(i+j-1)%3 };
                                    let cab = if a==b { a } else if dim==2 { 2 } else { 3+(a+b-1)%3 };
                                    val += c_e[(cij, cab)] * gphys[k*dim+j] * gphys[l*dim+b];
                                }
                            }
                            k_elem[row * n_vec + col] += w * degrade * val;
                        }
                    }
                }
            }
        }

        let dofs = &elem_dofs_cache[el];
        coo.add_element_matrix(dofs, &k_elem);
        for (k, &dof) in dofs.iter().enumerate() { f_int[dof] += f_elem[k]; }
        qp_idx += n_qp_e;
    }

    (f_int, coo.into_csr())
}

/// Update the damage field based on current displacement solution.
/// Returns the new per-QP damage values.
pub fn update_damage_field<M: MeshTopology>(
    mesh: &M,
    space: &dyn FESpace<Mesh = M>,
    u: &[f64],
    state: &mut DamageState,
    cfg: &DamageConfig,
    quad_order: u8,
) {
    let dim = mesh.dim() as usize;
    let mut qp_idx = 0usize;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re = ref_elem_vol(et, space.order());
        let quad = re.quadrature(quad_order);
        let n_qp_e = quad.weights.len();
        let nodes = mesh.element_nodes(e);
        let (jac, _det_j) = simplex_jac_2(mesh, nodes, dim);
        let jit = jac.try_inverse().map(|m| m.transpose()).unwrap_or_else(|| DMatrix::identity(dim, dim));
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_ldofs = dofs.len();
        let n_vec = n_ldofs * dim;

        let mut u_elem = vec![0.0_f64; n_vec];
        for (k, &dof) in dofs.iter().enumerate() { u_elem[k] = u[dof]; }
        let mut gref = vec![0.0_f64; n_ldofs * dim];
        let mut gphys = vec![0.0_f64; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let _w = quad.weights[q];
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);
            let eps = strain_at_qp(&u_elem, &gphys, dim, n_ldofs);
            let idx = qp_idx + q;
            let d_old = state.d[idx];
            state.d[idx] = update_damage(&eps, cfg, &mut state.history[idx], d_old);
        }
        qp_idx += n_qp_e;
    }
}

// ─── Staggered solver ─────────────────────────────────────────────────────────

/// Staggered (alternating) solver for coupled damage-elasticity.
///
/// Each iteration:
/// 1. Solve (1-d)²·K·u = f for displacements
/// 2. Update damage d from strain energy
/// 3. Repeat until convergence
pub struct StaggeredDamageSolver;

impl StaggeredDamageSolver {
    #[allow(clippy::too_many_arguments)]
    pub fn solve<M: MeshTopology>(
        mesh: &M,
        space: &dyn FESpace<Mesh = M>,
        _stiffness: &CsrMatrix<f64>,   // undamaged elastic stiffness
        rhs: &[f64],
        cfg: &DamageConfig,
        state: &mut DamageState,
        max_iter: usize,
        tol: f64,
        quad_order: u8,
    ) -> Vec<f64> {
        let n = space.n_dofs();
        let mut u = vec![0.0; n];

        for iter in 0..max_iter {
            // Solve damaged elasticity
            let (f_int, k_damaged) = assemble_damaged_elasticity(mesh, space, &u, &state.d, cfg, quad_order);
            let mut res = vec![0.0; n];
            for i in 0..n { res[i] = f_int[i] - rhs[i]; }
            let rn: f64 = res.iter().map(|v| v*v).sum::<f64>().sqrt();
            let bn: f64 = rhs.iter().map(|v| v*v).sum::<f64>().sqrt().max(1e-30);
            if rn < tol * bn.max(1.0) { break; }

            let mut du = vec![0.0; n];
            let neg_r: Vec<f64> = res.iter().map(|v| -v).collect();
            let _ = fem_solver::solve_cg(&k_damaged, &neg_r, &mut du,
                &fem_solver::SolverConfig { rtol: 1e-10, max_iter: 500, ..Default::default() });

            let mut alpha = 1.0;
            for _ in 0..10 {
                let mut u_new = u.clone();
                for i in 0..n { u_new[i] += alpha * du[i]; }
                let (f_new, _) = assemble_damaged_elasticity(mesh, space, &u_new, &state.d, cfg, quad_order);
                let mut r_new = vec![0.0; n];
                for i in 0..n { r_new[i] = f_new[i] - rhs[i]; }
                let rn_new: f64 = r_new.iter().map(|v| v*v).sum::<f64>().sqrt();
                if rn_new < rn || alpha < 1e-8 { u = u_new; break; }
                alpha *= 0.5;
            }

            // Update damage field
            update_damage_field(mesh, space, &u, state, cfg, quad_order);

            if iter > 0 && rn < tol * bn.max(1.0) { break; }
        }
        u
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("damage: unsupported element {:?} order {}", et, order),
    }
}

fn simplex_jac_2<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for c in 0..dim {
        let xc = mesh.node_coords(nodes[c+1]);
        for r in 0..dim { j[(r,c)] = xc[r] - x0[r]; }
    }
    (j.clone(), j.determinant())
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn damage_state_initialises() {
        let s = DamageState::new(10);
        assert_eq!(s.d.len(), 10);
        assert!(s.d.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn kachanov_damage_evolves() {
        let cfg = DamageConfig::kachanov(2e5, 0.3, 1e-3, 2.0, 1e-6);
        let mut h = 0.0;
        // Strain large enough to drive damage
        let eps = vec![0.05, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = update_damage(&eps, &cfg, &mut h, 0.0);
        assert!(d > 0.0, "expected damage growth, got {d}");
    }

    #[test]
    fn exponential_damage_saturates() {
        let cfg = DamageConfig::exponential(2e5, 0.3, 1e-3, 0.01);
        let mut h = 0.0;
        let eps = vec![0.1, 0.0, 0.0];
        let d = update_damage(&eps, &cfg, &mut h, 0.0);
        assert!(d > 0.0 && d <= 0.99, "expected damage in (0,0.99], got {d}");
    }

    #[test]
    fn damaged_stiffness_assembles() {
        use fem_space::VectorH1Space;
        use fem_space::fe_space::FESpace;
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let mesh_ref = space.mesh();
        let u = vec![0.0; space.n_dofs()];
        let n_qp: usize = mesh_ref.elem_iter().map(|e| {
            ref_elem_vol(mesh_ref.element_type(e), 1).quadrature(2).weights.len()
        }).sum();
        let d_at_qp = vec![0.0; n_qp.max(1)];
        let cfg = DamageConfig::exponential(2e5, 0.3, 0.001, 0.01);
        let (f, k) = assemble_damaged_elasticity(mesh_ref, &space, &u, &d_at_qp, &cfg, 2);
        assert_eq!(f.len(), space.n_dofs());
        assert_eq!(k.nrows, space.n_dofs());
    }
}
