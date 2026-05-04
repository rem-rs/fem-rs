//! Partial assembly (matrix-free) operators.
//!
//! **Partial assembly** avoids forming the full global sparse matrix and instead
//! computes the matrix-vector product `y = K x` by iterating over elements and
//! accumulating element-level contributions.  This reduces memory from O(n_dofs²)
//! to O(n_dofs) and avoids the global assembly gather/scatter entirely.
//!
//! # Provided operators
//!
//! | Operator | Form | Description |
//! |----------|------|-------------|
//! | [`PAMassOperator`] | `∫ ρ u v dx` | Lumped or consistent mass |
//! | [`PADiffusionOperator`] | `∫ κ ∇u·∇v dx` | Scalar diffusion |
//!
//! # Design
//!
//! Each operator implements the [`MatFreeOperator`] trait:
//! ```rust,ignore
//! trait MatFreeOperator {
//!     fn apply(&self, x: &[f64], y: &mut [f64]);
//!     fn n_dofs(&self) -> usize;
//! }
//! ```
//!
//! Operators can be used directly with iterative solvers via a thin wrapper.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::partial::{PADiffusionOperator, MatFreeOperator};
//!
//! let op = PADiffusionOperator::new(&space, 1.0, 3);
//! let mut y = vec![0.0; space.n_dofs()];
//! op.apply(&x, &mut y);  // y += K x  (matrix-free)
//! ```

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use nalgebra::DMatrix;
use fem_element::{ReferenceElement, lagrange::{TetP1, TetP2, TetP3, TriP1, TriP2, TriP3}};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use fem_mesh::ElementTransformation;
use fem_space::fe_space::SpaceType;
use crate::vector_assembler::{
    apply_signs, geo_ref_elem, isoparametric_jacobian,
    piola_hcurl_basis, piola_hcurl_curl, vec_ref_elem,
};

// ─── MatFreeOperator trait ───────────────────────────────────────────────────

/// A matrix-free linear operator `y = A x`.
pub trait MatFreeOperator: Send + Sync {
    /// Apply: `y += A x`.  **Does not zero `y` first.**
    fn apply(&self, x: &[f64], y: &mut [f64]);

    /// Apply: `y = A x` (zeroes y first).
    fn apply_zero(&self, x: &[f64], y: &mut [f64]) {
        y.iter_mut().for_each(|v| *v = 0.0);
        self.apply(x, y);
    }

    /// Number of DOFs (rows = cols).
    fn n_dofs(&self) -> usize;
}

// ─── PAMassOperator ───────────────────────────────────────────────────────────

/// Matrix-free consistent mass operator `M x` where `M[i,j] = ∫ ρ φᵢ φⱼ dx`.
///
/// Each call to `apply` sweeps all elements, evaluates the mass at each
/// quadrature point, and scatters into `y`.  No matrix is stored.
pub struct PAMassOperator<S: FESpace, C: ScalarCoeff = f64> {
    space:      S,
    rho:        C,
    quad_order: u8,
}

impl<S: FESpace, C: ScalarCoeff> PAMassOperator<S, C> {
    /// Construct the operator.
    ///
    /// - `space`      — FE space (ownership taken).
    /// - `rho`        — density coefficient (can be constant `f64` or spatially varying).
    /// - `quad_order` — quadrature order for integration.
    pub fn new(space: S, rho: C, quad_order: u8) -> Self {
        PAMassOperator { space, rho, quad_order }
    }
}

impl<S: FESpace, C: ScalarCoeff> MatFreeOperator for PAMassOperator<S, C> {
    fn n_dofs(&self) -> usize { self.space.n_dofs() }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let mesh  = self.space.mesh();
        let dim   = mesh.dim() as usize;
        let order = self.space.order();

        let mut phi = Vec::<f64>::new();

        for e in mesh.elem_iter() {
            let et   = mesh.element_type(e);
            let re   = ref_elem(et, order);
            let n    = re.n_dofs();
            let quad = re.quadrature(self.quad_order);
            let gd: Vec<usize> = self.space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let elem_tag = mesh.element_tag(e);
            let x0 = mesh.node_coords(nodes[0]);

            phi.resize(n, 0.0);

            // Element-level x values
            let x_elem: Vec<f64> = gd.iter().map(|&di| x[di]).collect();
            let mut y_elem = vec![0.0_f64; n];

            for (qi, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[qi] * det_j.abs();
                re.eval_basis(xi, &mut phi);

                let xp: Vec<f64> = (0..dim)
                    .map(|i| x0[i] + (0..dim).map(|k| jac[(i,k)] * xi[k]).sum::<f64>())
                    .collect();
                let ctx = CoeffCtx::from_qp(&xp, dim, e, elem_tag, None, None);
                let rho_qp = self.rho.eval(&ctx);

                // y_elem[i] += w * ρ * φᵢ * Σⱼ φⱼ xⱼ
                let ux: f64 = phi.iter().zip(x_elem.iter()).map(|(ph, xe)| ph * xe).sum();
                for i in 0..n {
                    y_elem[i] += w * rho_qp * phi[i] * ux;
                }
            }

            for (i, &gi) in gd.iter().enumerate() {
                y[gi] += y_elem[i];
            }
        }
    }
}

// ─── PADiffusionOperator ──────────────────────────────────────────────────────

/// Matrix-free diffusion operator `K x` where `K[i,j] = ∫ κ ∇φᵢ·∇φⱼ dx`.
///
/// Supports spatially varying `kappa` via any [`ScalarCoeff`] implementation.
pub struct PADiffusionOperator<S: FESpace, K: ScalarCoeff = f64> {
    space:      S,
    kappa:      K,
    quad_order: u8,
}

impl<S: FESpace, K: ScalarCoeff> PADiffusionOperator<S, K> {
    /// Construct with any coefficient (constant `f64`, `FnCoeff`, `PWConstCoeff`, etc.).
    pub fn new(space: S, kappa: K, quad_order: u8) -> Self {
        PADiffusionOperator { space, kappa, quad_order }
    }
}

impl<S: FESpace, K: ScalarCoeff> MatFreeOperator for PADiffusionOperator<S, K> {
    fn n_dofs(&self) -> usize { self.space.n_dofs() }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let mesh  = self.space.mesh();
        let dim   = mesh.dim() as usize;
        let order = self.space.order();

        let mut grad_ref  = Vec::<f64>::new();
        let mut grad_phys = Vec::<f64>::new();

        for e in mesh.elem_iter() {
            let et   = mesh.element_type(e);
            let re   = ref_elem(et, order);
            let n    = re.n_dofs();
            let quad = re.quadrature(self.quad_order);
            let gd: Vec<usize> = self.space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let jit = jac.clone().try_inverse().unwrap().transpose();
            let x0  = mesh.node_coords(nodes[0]);
            let elem_tag = mesh.element_tag(e);

            grad_ref.resize(n * dim, 0.0);
            grad_phys.resize(n * dim, 0.0);

            let x_elem: Vec<f64> = gd.iter().map(|&di| x[di]).collect();
            let mut y_elem = vec![0.0_f64; n];

            for (qi, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[qi] * det_j.abs();
                re.eval_grad_basis(xi, &mut grad_ref);
                xform_grads(&jit, &grad_ref, &mut grad_phys, n, dim);

                // Physical coords at quadrature point
                let xp: Vec<f64> = (0..dim)
                    .map(|i| x0[i] + (0..dim).map(|k| jac[(i,k)] * xi[k]).sum::<f64>())
                    .collect();
                let ctx = CoeffCtx::from_qp(&xp, dim, e, elem_tag, None, None);
                let kappa_qp = self.kappa.eval(&ctx);

                // ∇u at this qp = Σⱼ xⱼ ∇φⱼ
                let grad_u: Vec<f64> = (0..dim).map(|d| {
                    x_elem.iter().zip(grad_phys.chunks(dim)).map(|(&xj, gj)| xj * gj[d]).sum::<f64>()
                }).collect();

                // y_elem[i] += w κ ∇u·∇φᵢ
                for i in 0..n {
                    let dot: f64 = (0..dim).map(|d| grad_u[d] * grad_phys[i*dim+d]).sum();
                    y_elem[i] += w * kappa_qp * dot;
                }
            }

            for (i, &gi) in gd.iter().enumerate() {
                y[gi] += y_elem[i];
            }
        }
    }
}

// ─── Lumped mass (diagonal) ───────────────────────────────────────────────────

/// Lumped (row-sum) mass operator: diagonal scaling `M_lumped x = diag(m) x`.
///
/// This is equivalent to integrating each row of the mass matrix and using it
/// as a diagonal mass.  Useful for explicit time-stepping.
pub struct LumpedMassOperator<S: FESpace> {
    /// Diagonal mass entries (one per DOF).
    pub diag: Vec<f64>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: FESpace> LumpedMassOperator<S> {
    /// Assemble the lumped mass by integrating row sums.
    pub fn assemble(space: &S, rho: f64, quad_order: u8) -> Self {
        let mesh  = space.mesh();
        let dim   = mesh.dim() as usize;
        let order = space.order();
        let n     = space.n_dofs();
        let mut diag = vec![0.0_f64; n];

        let mut phi = Vec::<f64>::new();

        for e in mesh.elem_iter() {
            let et   = mesh.element_type(e);
            let re   = ref_elem(et, order);
            let nl   = re.n_dofs();
            let quad = re.quadrature(quad_order);
            let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (_, det_j) = simplex_jac(mesh, nodes, dim);
            phi.resize(nl, 0.0);

            for (qi, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[qi] * det_j.abs();
                re.eval_basis(xi, &mut phi);
                let phi_sum: f64 = phi.iter().sum();
                for i in 0..nl {
                    // Lumped: M_ii = ∫ ρ φᵢ (Σⱼ φⱼ) dx  [row-sum lumping]
                    diag[gd[i]] += w * rho * phi[i] * phi_sum;
                }
            }
        }

        LumpedMassOperator { diag, _phantom: std::marker::PhantomData }
    }

    /// Apply: `y[i] += diag[i] * x[i]`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.diag.len() { y[i] += self.diag[i] * x[i]; }
    }

    /// Multiply: `y[i] = diag[i] * x[i]`.
    pub fn apply_zero(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.diag.len() { y[i] = self.diag[i] * x[i]; }
    }

    /// Invert: `y[i] = x[i] / diag[i]`.  Used for explicit time-stepping.
    pub fn apply_inverse(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.diag.len() {
            y[i] = if self.diag[i].abs() > 1e-14 { x[i] / self.diag[i] } else { 0.0 };
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn ref_elem(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("partial ref_elem: unsupported ({et:?}, {order})"),
    }
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
    }
    let det = j.determinant();
    (j, det)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j, k)] * gr[i * dim + k]; }
            gp[i * dim + j] = s;
        }
    }
}

// ─── HcurlMatrixFreeOperator ─────────────────────────────────────────────────

/// Matrix-free operator for H(curl): applies `(mu_inv · K_curl + alpha · M_e) x`.
///
/// Element matrices are precomputed at construction time (once) and stored.
/// Each call to `apply` iterates over elements, gathers DOFs with orientation
/// signs, multiplies by the precomputed element matrix, and scatters back —
/// **without forming the global sparse matrix**.
///
/// # Physical problem
///
/// Represents the weak form of `∇×(μ⁻¹ ∇×E) + α E = f`:
/// ```text
/// a(E, F) = μ⁻¹ ∫ (∇×E)·(∇×F) dx + α ∫ E·F dx
/// ```
///
/// # Memory savings
///
/// Global CSR stores `O(n_edges × avg_nnz_per_row)` values; partial assembly
/// stores `O(n_elem × n_ldofs²)`.  For TriND1 both are ~9 per element, but
/// partial assembly also avoids the global gather/scatter at assembly time and
/// enables streaming apply patterns.
///
/// # Supported element types
///
/// | Element | Order | DOFs/elem | Matrix size |
/// |---------|-------|-----------|-------------|
/// | `Tri3` (TriND1) | 1 | 3 | 3×3 |
/// | `Tri3` (TriND2) | 2 | 8 | 8×8 |
/// | `Tet4` (TetND1) | 1 | 6 | 6×6 |
/// | `Tet4` (TetND2) | 2 | 20 | 20×20 |
/// | `Quad4` (QuadND1/2) | 1/2 | 4/12 | 4×4 / 12×12 |
/// | `Hex8` (HexND1/2) | 1/2 | 12/54 | 12×12 / 54×54 |
pub struct HcurlMatrixFreeOperator {
    n_dofs:        usize,
    dofs_per_elem: usize,
    n_elem:        usize,
    /// Flattened global DOF indices: `elem_dofs[e * n + i]` = global DOF i of element e.
    elem_dofs:  Vec<u32>,
    /// Flattened element matrices with orientation signs incorporated:
    /// `elem_mats[e * n² + i * n + j]` = sign[i] * sign[j] * K_e_unsigned[i, j].
    elem_mats:  Vec<f64>,
}

impl HcurlMatrixFreeOperator {
    /// Precompute element matrices for `(mu_inv · K_curl + alpha · M_e)`.
    ///
    /// - `space`      — H(curl) FE space.
    /// - `mu_inv`     — scalar inverse permeability (1/μ).
    /// - `alpha`      — scalar mass coefficient (e.g. ω² ε for frequency-domain).
    /// - `quad_order` — quadrature order; use `order + 2` for accuracy.
    pub fn new<S: FESpace>(space: &S, mu_inv: f64, alpha: f64, quad_order: u8) -> Self {
        let mesh       = space.mesh();
        let dim        = mesh.dim() as usize;
        let n_dofs_g   = space.n_dofs();
        let n_elem     = mesh.n_elements();
        let stype      = space.space_type();
        let elem_type0 = mesh.element_type(0);

        assert_eq!(stype, SpaceType::HCurl,
            "HcurlMatrixFreeOperator requires an H(curl) space");

        let ref_elem       = vec_ref_elem(stype, elem_type0, dim, space.order());
        let n              = ref_elem.n_dofs();   // DOFs per element
        let quad           = ref_elem.quadrature(quad_order);
        let curl_dim       = if dim == 2 { 1 } else { 3 };

        let mut all_dofs  = Vec::with_capacity(n_elem * n);
        let mut all_mats  = Vec::with_capacity(n_elem * n * n);

        // Work buffers
        let mut ref_phi  = vec![0.0_f64; n * dim];
        let mut ref_curl = vec![0.0_f64; n * curl_dim];
        let mut ref_div  = vec![0.0_f64; n];
        let mut phi      = vec![0.0_f64; n * dim];
        let mut curl     = vec![0.0_f64; n * curl_dim];
        let mut div_buf  = vec![0.0_f64; n];

        for e in mesh.elem_iter() {
            let global_dofs = space.element_dofs(e);
            let signs_opt   = space.element_signs(e);
            let nodes       = mesh.element_nodes(e);
            let elem_type   = mesh.element_type(e);

            // Store DOFs and signs
            all_dofs.extend_from_slice(global_dofs);
            let elem_sign_slice: Vec<f64> = if let Some(s) = signs_opt {
                s.to_vec()
            } else {
                vec![1.0_f64; n]
            };

            let use_iso = matches!(elem_type, ElementType::Quad4 | ElementType::Hex8);
            let geo_elem = geo_ref_elem(elem_type);
            let affine_tr = if use_iso {
                None
            } else {
                Some(ElementTransformation::from_simplex_nodes(mesh, nodes))
            };

            let mut k_e = vec![0.0_f64; n * n];

            for (q, xi) in quad.points.iter().enumerate() {
                let (jac, det_j, _xp) = if use_iso {
                    let ge = geo_elem.as_ref().expect("geo_ref_elem for iso");
                    isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, dim)
                } else {
                    let tr = affine_tr.as_ref().unwrap();
                    (tr.jacobian().clone(), tr.det_j(), tr.map_to_physical(xi))
                };

                let j_inv_t = jac.clone().try_inverse()
                    .expect("degenerate element in HcurlMatrixFreeOperator")
                    .transpose();
                let w = quad.weights[q] * det_j.abs();

                ref_elem.eval_basis_vec(xi, &mut ref_phi);
                ref_elem.eval_curl(xi, &mut ref_curl);
                ref_elem.eval_div(xi, &mut ref_div);

                piola_hcurl_basis(&j_inv_t, &ref_phi, &mut phi, n, dim);
                piola_hcurl_curl(&jac, det_j, &ref_curl, &mut curl, n, dim);
                div_buf[..ref_div.len()].copy_from_slice(&ref_div);

                apply_signs(&elem_sign_slice, &mut phi, &mut curl, &mut div_buf, n, dim, curl_dim);

                for i in 0..n {
                    for j in 0..n {
                        // Curl-curl contribution: mu_inv * ∫ curl(φ_i)·curl(φ_j) dx
                        let cc: f64 = (0..curl_dim).map(|c| curl[i*curl_dim+c] * curl[j*curl_dim+c]).sum();
                        // Vector mass contribution: alpha * ∫ φ_i·φ_j dx
                        let mm: f64 = (0..dim).map(|c| phi[i*dim+c] * phi[j*dim+c]).sum();
                        k_e[i * n + j] += w * (mu_inv * cc + alpha * mm);
                    }
                }
            }

            all_mats.extend_from_slice(&k_e);
        }

        HcurlMatrixFreeOperator {
            n_dofs: n_dofs_g,
            dofs_per_elem: n,
            n_elem,
            elem_dofs:  all_dofs,
            elem_mats:  all_mats,
        }
    }
}

impl MatFreeOperator for HcurlMatrixFreeOperator {
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n  = self.dofs_per_elem;
        let n2 = n * n;

        for e in 0..self.n_elem {
            let dofs = &self.elem_dofs [e * n .. (e + 1) * n];
            let mat  = &self.elem_mats [e * n2 .. (e + 1) * n2];

            // Gather: x_e[i] = x[dofs[i]]
            // (Signs are already incorporated into the precomputed element matrix.)
            let x_e: Vec<f64> = (0..n).map(|i| x[dofs[i] as usize]).collect();

            // z_e[i] = Σ_j K_e_signed[i,j] * x_e[j]
            let mut z_e = vec![0.0_f64; n];
            for i in 0..n {
                let row = &mat[i * n .. (i + 1) * n];
                z_e[i] = row.iter().zip(x_e.iter()).map(|(m, xe)| m * xe).sum();
            }

            // Scatter: y[dofs[i]] += z_e[i]
            for i in 0..n {
                y[dofs[i] as usize] += z_e[i];
            }
        }
    }
}

/// Solve `(mu_inv · K_curl + alpha · M_e) x = b` using matrix-free CG.
///
/// Uses the `HcurlMatrixFreeOperator` to apply the operator without forming
/// the global sparse matrix.  Suitable for large meshes where CSR assembly
/// would exhaust memory.
///
/// Returns `(x, n_iters, residual_norm)`.
pub fn solve_hcurl_matrix_free<S: FESpace>(
    space: &S,
    rhs: &[f64],
    mu_inv: f64,
    alpha: f64,
    quad_order: u8,
    rtol: f64,
    max_iter: usize,
) -> (Vec<f64>, usize, f64) {
    let op = HcurlMatrixFreeOperator::new(space, mu_inv, alpha, quad_order);
    let n = op.n_dofs();

    let mut x = vec![0.0_f64; n];
    let mut r = rhs.to_vec();
    let mut p = r.clone();
    let r0_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();

    if r0_norm < 1e-300 {
        return (x, 0, 0.0);
    }

    let mut rr = r.iter().map(|v| v * v).sum::<f64>();
    let mut iters = 0;

    for _ in 0..max_iter {
        // Ap = op * p
        let mut ap = vec![0.0_f64; n];
        op.apply_zero(&p, &mut ap);

        let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
        if pap.abs() < 1e-300 { break; }
        let alpha_cg = rr / pap;

        for i in 0..n {
            x[i] += alpha_cg * p[i];
            r[i] -= alpha_cg * ap[i];
        }

        let rr_new: f64 = r.iter().map(|v| v * v).sum();
        let res_norm = rr_new.sqrt();
        iters += 1;

        if res_norm < rtol * r0_norm { break; }

        let beta = rr_new / rr;
        rr = rr_new;
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
    }

    let res_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt() / r0_norm;
    (x, iters, res_norm)
}

// ─── Maxwell eigenproblem with AMG-preconditioned LOBPCG ─────────────────────

/// Solve the Maxwell H(curl) generalised eigenproblem
/// `K x = λ M x` using LOBPCG preconditioned by an AMG V-cycle.
///
/// The discrete gradient matrix `G` (H1 → H(curl)) is used to project out
/// the discrete nullspace of the curl-curl operator (gradient modes).
///
/// # Arguments
/// - `k_curl` — assembled curl-curl stiffness: `∫ μ⁻¹ curl u · curl v`.
/// - `m_mass` — assembled vector mass: `∫ α u · v`.
/// - `grad`   — discrete gradient matrix `G: H1 → H(curl)`.
/// - `n_eigen` — number of eigenpairs to compute.
/// - `cfg`     — LOBPCG configuration.
///
/// # Returns
/// `EigenResult` with the first `n_eigen` non-zero eigenvalues (curl modes).
pub fn solve_hcurl_eigen_preconditioned_amg(
    k_curl:  &fem_linalg::CsrMatrix<f64>,
    m_mass:  &fem_linalg::CsrMatrix<f64>,
    grad:    &fem_linalg::CsrMatrix<f64>,
    n_eigen: usize,
    cfg:     &fem_solver::LobpcgConfig,
) -> Result<fem_solver::EigenResult, String> {
    use fem_solver::{SolverConfig};
    use fem_amg::solve_amg_cg;
    use nalgebra::DMatrix;

    // Convert the discrete gradient columns to a dense constraint matrix so
    // that LOBPCG projects out grad-modes (the curl-nullspace).
    let n = k_curl.nrows;
    let n_grad = grad.ncols;
    let mut grad_dense = DMatrix::<f64>::zeros(n, n_grad);
    for r in 0..n {
        let start = grad.row_ptr[r];
        let end   = grad.row_ptr[r + 1];
        for k in start..end {
            let c = grad.col_idx[k] as usize;
            grad_dense[(r, c)] = grad.values[k];
        }
    }

    // Build a regularised matrix K_reg = K_curl + reg * M_mass for AMG.
    // (Pure K_curl is singular on the gradient modes, which makes AMG struggle.)
    let reg = 0.1_f64;
    let k_reg = k_curl.axpby(1.0, m_mass, reg);

    // AMG preconditioner: one V-cycle per LOBPCG residual block column.
    let amg_cfg = fem_amg::AmgConfig::default();
    let amg_prec = move |r: &DMatrix<f64>| -> DMatrix<f64> {
        let nrows = r.nrows();
        let ncols = r.ncols();
        let mut z = DMatrix::<f64>::zeros(nrows, ncols);
        let sc = SolverConfig { max_iter: 80, rtol: 1e-3, ..SolverConfig::default() };
        for c in 0..ncols {
            let rhs: Vec<f64> = (0..nrows).map(|i| r[(i, c)]).collect();
            let mut x = vec![0.0_f64; nrows];
            let _ = solve_amg_cg(&k_reg, &rhs, &mut x, &amg_cfg, &sc);
            for i in 0..nrows { z[(i, c)] = x[i]; }
        }
        z
    };

    fem_solver::lobpcg_constrained_preconditioned(
        k_curl,
        Some(m_mass),
        n_eigen,
        &grad_dense,
        amg_prec,
        cfg,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use crate::coefficient::FnCoeff;
    use crate::assembler::Assembler;
    use crate::standard::{DiffusionIntegrator, MassIntegrator};

    /// M_matfree x == M_assembled x for a constant vector x = 1.
    #[test]
    fn pa_mass_matches_assembled() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();

        // Assembled mass matrix.
        let m_assembled = Assembler::assemble_bilinear(
            &space, &[&MassIntegrator { rho: 1.0 }], 3);

        // Matrix-free mass operator.
        let op = PAMassOperator::new(space, 1.0, 3);

        let x = vec![1.0_f64; n];
        let mut y_mf  = vec![0.0_f64; n];
        let mut y_asm = vec![0.0_f64; n];
        op.apply(&x, &mut y_mf);
        m_assembled.spmv(&x, &mut y_asm);

        let err: f64 = y_mf.iter().zip(y_asm.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-11, "PA mass vs assembled: ‖diff‖ = {err:.3e}");
    }

    /// K_matfree x == K_assembled x for a constant vector x = 1.
    #[test]
    fn pa_diffusion_matches_assembled() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();

        let k_assembled = Assembler::assemble_bilinear(
            &space, &[&DiffusionIntegrator { kappa: 2.0 }], 3);

        let op = PADiffusionOperator::new(space, FnCoeff(|_: &[f64]| 2.0), 3);

        let x = vec![1.0_f64; n];
        let mut y_mf  = vec![0.0_f64; n];
        let mut y_asm = vec![0.0_f64; n];
        op.apply(&x, &mut y_mf);
        k_assembled.spmv(&x, &mut y_asm);

        let err: f64 = y_mf.iter().zip(y_asm.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-11, "PA diffusion vs assembled: ‖diff‖ = {err:.3e}");
    }

    /// Lumped mass: sum of diagonal = total mass = ρ * ∫ 1 dx = 1 for ρ=1.
    #[test]
    fn lumped_mass_sum_equals_area() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let lm = LumpedMassOperator::assemble(&space, 1.0, 3);
        let total_mass: f64 = lm.diag.iter().sum();
        // Area of unit square = 1, so total mass = 1.
        assert!((total_mass - 1.0).abs() < 1e-11,
            "lumped mass sum = {total_mass:.6e}, expected 1.0");
    }

    /// PA diffusion: K 1 = 0 for a 1-by-1 system since ∇1 = 0.
    #[test]
    fn pa_diffusion_of_constant_is_zero() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let op = PADiffusionOperator::new(space, FnCoeff(|_: &[f64]| 1.0), 3);
        let x = vec![1.0_f64; n]; // u = 1 (constant)
        let mut y = vec![0.0_f64; n];
        op.apply(&x, &mut y);
        let max_y = y.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_y < 1e-12, "K * 1 should be 0, got max_y={max_y:.3e}");
    }

    /// Repeated apply: apply twice should equal 2x the single apply.
    #[test]
    fn pa_mass_linearity() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(3);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let op = PAMassOperator::new(space, 1.0, 3);

        let x: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let mut y1 = vec![0.0_f64; n];
        let mut y2 = vec![0.0_f64; n];
        op.apply(&x, &mut y1);
        op.apply(&x, &mut y2);
        let twice: Vec<f64> = y1.iter().map(|&v| 2.0 * v).collect();
        let mut y_sum = y1.clone();
        y_sum.iter_mut().zip(y2.iter()).for_each(|(a, b)| *a += b);
        let err: f64 = y_sum.iter().zip(twice.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-13, "linearity check: {err:.3e}");
    }

    // ── H(curl) partial assembly ──────────────────────────────────────────────

    /// `HcurlMatrixFreeOperator` applied to x=1 matches the assembled sparse matrix.
    #[test]
    fn hcurl_mf_matches_assembled_tri_nd1() {
        use fem_space::HCurlSpace;
        use crate::VectorAssembler;
        use crate::standard::{CurlCurlIntegrator, VectorMassIntegrator};

        let mesh  = fem_mesh::SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n     = space.n_dofs();
        let mu_inv = 2.0_f64;
        let alpha  = 0.5_f64;

        // Assembled sparse operator.
        let k_asm = VectorAssembler::assemble_bilinear(
            &space,
            &[&CurlCurlIntegrator { mu: mu_inv },
              &VectorMassIntegrator { alpha }],
            3,
        );

        // Matrix-free operator.
        let op = HcurlMatrixFreeOperator::new(&space, mu_inv, alpha, 3);

        let x: Vec<f64> = (0..n).map(|i| ((i + 1) as f64) * 0.1).collect();
        let mut y_asm = vec![0.0_f64; n];
        let mut y_mf  = vec![0.0_f64; n];

        k_asm.spmv(&x, &mut y_asm);
        op.apply_zero(&x, &mut y_mf);

        let err: f64 = y_mf.iter().zip(y_asm.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let ref_norm: f64 = y_asm.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err / ref_norm.max(1e-15) < 1e-11,
            "H(curl) mf vs assembled: rel_err = {:.3e}", err / ref_norm.max(1e-15));
    }

    /// `HcurlMatrixFreeOperator` for 3D TetND1 matches the assembled matrix.
    #[test]
    fn hcurl_mf_matches_assembled_tet_nd1() {
        use fem_space::HCurlSpace;
        use crate::VectorAssembler;
        use crate::standard::{CurlCurlIntegrator, VectorMassIntegrator};

        let mesh  = fem_mesh::SimplexMesh::<3>::unit_cube_tet(2);
        let space = HCurlSpace::new(mesh, 1);
        let n     = space.n_dofs();
        let mu_inv = 1.0_f64;
        let alpha  = 1.0_f64;

        let k_asm = VectorAssembler::assemble_bilinear(
            &space,
            &[&CurlCurlIntegrator { mu: mu_inv },
              &VectorMassIntegrator { alpha }],
            3,
        );

        let op = HcurlMatrixFreeOperator::new(&space, mu_inv, alpha, 3);

        let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.01).collect();
        let mut y_asm = vec![0.0_f64; n];
        let mut y_mf  = vec![0.0_f64; n];

        k_asm.spmv(&x, &mut y_asm);
        op.apply_zero(&x, &mut y_mf);

        let err: f64 = y_mf.iter().zip(y_asm.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let ref_norm: f64 = y_asm.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel = err / ref_norm.max(1e-15);
        assert!(rel < 1e-11,
            "3D H(curl) mf vs assembled: rel_err = {:.3e}", rel);
    }

    /// `solve_hcurl_matrix_free` converges on a pure-mass problem (K_curl=0, alpha=1).
    #[test]
    fn solve_hcurl_matrix_free_pure_mass_converges() {
        use fem_space::HCurlSpace;

        // alpha=1, mu_inv=0: pure vector mass problem → M x = b, trivial.
        let mesh  = fem_mesh::SimplexMesh::<2>::unit_square_tri(3);
        let space = HCurlSpace::new(mesh, 1);
        let n     = space.n_dofs();

        // Assemble M for RHS
        use crate::VectorAssembler;
        use crate::standard::VectorMassIntegrator;
        let m = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: 1.0 }], 3);
        let x_exact: Vec<f64> = (0..n).map(|i| ((i + 1) as f64) * 0.05).collect();
        let mut rhs = vec![0.0_f64; n];
        m.spmv(&x_exact, &mut rhs);

        let (x_sol, iters, res) = solve_hcurl_matrix_free(&space, &rhs, 0.0, 1.0, 3, 1e-10, 1000);

        assert!(res < 1e-8, "residual {res:.3e} not converged after {iters} iters");
        let err: f64 = x_sol.iter().zip(x_exact.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let xnorm: f64 = x_exact.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err / xnorm < 1e-8, "solution error {:.3e}", err / xnorm);
    }

    // ── AMG-preconditioned Maxwell eigen ──────────────────────────────────────

    /// P3 smoke gate: AMG-preconditioned LOBPCG returns eigenvalues for a
    /// small 2-D Maxwell cavity on the unit square.
    #[test]
    fn hcurl_eigen_amg_preconditioned_lobpcg_smoke() {
        use fem_space::{HCurlSpace, H1Space};
        use crate::VectorAssembler;
        use crate::standard::{CurlCurlIntegrator, VectorMassIntegrator};
        use crate::discrete_op::DiscreteLinearOperator;
        use fem_solver::LobpcgConfig;

        let mesh   = fem_mesh::SimplexMesh::<2>::unit_square_tri(8);
        let hcurl  = HCurlSpace::new(mesh.clone(), 1);
        let h1     = H1Space::new(mesh, 1);

        // Curl-curl stiffness + vector mass
        let k_curl = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&CurlCurlIntegrator { mu: 1.0 }],
            3,
        );
        let m_mass = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&VectorMassIntegrator { alpha: 1.0 }],
            3,
        );

        // Discrete gradient G: H1 → H(curl)
        let grad = DiscreteLinearOperator::gradient(&h1, &hcurl)
            .expect("gradient matrix build failed");

        let n = hcurl.n_dofs();
        let n_grad = h1.n_dofs();
        assert!(n > 100, "expected > 100 H(curl) DOFs for smoke gate, got {n}");
        assert!(n_grad > 50, "expected > 50 H1 DOFs, got {n_grad}");

        let cfg = LobpcgConfig { max_iter: 400, tol: 1e-4, verbose: false };
        let result = solve_hcurl_eigen_preconditioned_amg(&k_curl, &m_mass, &grad, 4, &cfg)
            .expect("LOBPCG failed");

        assert_eq!(result.eigenvalues.len(), 4, "expected 4 eigenvalues");
        // All eigenvalues should be finite and non-negative
        for (i, &ev) in result.eigenvalues.iter().enumerate() {
            assert!(ev.is_finite(), "eigenvalue[{i}] = {ev} is not finite");
            assert!(ev >= -1e-6, "eigenvalue[{i}] = {ev:.6e} is spuriously negative");
        }
    }

    /// Large-scale AMG-LOBPCG smoke gate (n > 200 free DOFs, k=3 modes).
    #[test]
    fn hcurl_eigen_amg_large_scale_smoke() {
        use fem_space::{HCurlSpace, H1Space};
        use crate::VectorAssembler;
        use crate::standard::{CurlCurlIntegrator, VectorMassIntegrator};
        use crate::discrete_op::DiscreteLinearOperator;
        use fem_solver::LobpcgConfig;

        // n=7 → ≈168 Tri3 elements → 3D: unit_square_tri(7) gives ~98+49*2 edges
        let mesh  = fem_mesh::SimplexMesh::<2>::unit_square_tri(8);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let h1    = H1Space::new(mesh, 1);

        let k_curl = VectorAssembler::assemble_bilinear(
            &hcurl, &[&CurlCurlIntegrator { mu: 1.0 }], 3);
        let m_mass = VectorAssembler::assemble_bilinear(
            &hcurl, &[&VectorMassIntegrator { alpha: 1.0 }], 3);
        let grad = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        let n = hcurl.n_dofs();
        assert!(n > 200, "need > 200 DOFs for large-scale smoke, got {n}");

        let cfg = LobpcgConfig { max_iter: 500, tol: 1e-4, verbose: false };
        let result = solve_hcurl_eigen_preconditioned_amg(&k_curl, &m_mass, &grad, 3, &cfg)
            .expect("large-scale AMG-LOBPCG failed");

        assert_eq!(result.eigenvalues.len(), 3);
        for (i, &ev) in result.eigenvalues.iter().enumerate() {
            assert!(ev.is_finite(), "eigenvalue[{i}] = {ev} is not finite");
            assert!(ev >= -1e-6, "eigenvalue[{i}] = {ev:.6e} is spuriously negative");
        }
    }
}

