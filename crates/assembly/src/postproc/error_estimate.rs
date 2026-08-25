//! Error estimation for adaptive mesh refinement (AMR).
//! ZZ and Kelly estimators using GridFunction for arbitrary order + 2D/3D.

use nalgebra::DMatrix;

use fem_element::lagrange::{QuadQ1, QuadQ2, TetP1, TetP2, TetP3, TriP1, TriP2, TriP3};
use fem_element::ReferenceElement;
use fem_mesh::amr::HangingNodeConstraint;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::constraints::{apply_hanging_constraints, recover_hanging_values};
use fem_space::FESpace;
use fem_solver::{solve_pcg_gssmoother, SolverConfig};
use crate::postproc::grid_function::GridFunction;
use crate::standard::MassIntegrator;
use crate::Assembler;
// ─── Reference element helper (same as grid_function.rs) ──────────────────────

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

/// True for simplex element types (Tri3, Tri6, Tet4, …).
fn is_simplex(elem_type: ElementType) -> bool {
    matches!(elem_type, ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10)
}

/// Geometric-mapping Jacobian at reference point `xi` on element `e`.
///
/// **Simplex** (Tri3, Tri6, Tet4): P1 mapping → constant Jacobian from nodes 0..dim.
/// **Quad** (Quad4): Q1 bilinear mapping → correct bilinear Jacobian at (ξ,η).
///
/// Returns `(J, det J)` where J is the `dim × dim` Jacobian matrix.
fn geom_jacobian<M: MeshTopology>(mesh: &M, nodes: &[u32], xi: &[f64], dim: usize, elem_type: ElementType) -> (DMatrix<f64>, f64) {
    if is_simplex(elem_type) {
        // Simplex: P1 mapping, Jacobian = [x1-x0, x2-x0, …] (constant)
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim {
                j[(row, col)] = xc[row] - x0[row];
            }
        }
        let det = j.determinant();
        (j, det)
    } else if dim == 2 && nodes.len() >= 4 {
        // Quad: Q1 bilinear mapping at (ξ, η)
        let (e, n) = (xi[0], xi[1]);
        let c = |i: usize| mesh.node_coords(nodes[i]);
        let j00 = 0.25 * (-(1.0 - n) * c(0)[0] + (1.0 - n) * c(1)[0] + (1.0 + n) * c(2)[0] - (1.0 + n) * c(3)[0]);
        let j01 = 0.25 * (-(1.0 - e) * c(0)[0] - (1.0 + e) * c(1)[0] + (1.0 + e) * c(2)[0] + (1.0 - e) * c(3)[0]);
        let j10 = 0.25 * (-(1.0 - n) * c(0)[1] + (1.0 - n) * c(1)[1] + (1.0 + n) * c(2)[1] - (1.0 + n) * c(3)[1]);
        let j11 = 0.25 * (-(1.0 - e) * c(0)[1] - (1.0 + e) * c(1)[1] + (1.0 + e) * c(2)[1] + (1.0 - e) * c(3)[1]);
        let det = j00 * j11 - j01 * j10;
        let jac = DMatrix::from_row_slice(2, 2, &[j00, j01, j10, j11]);
        (jac, det)
    } else {
        // Fallback: simplex-like (nodes 0..dim)
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim.min(nodes.len().saturating_sub(1)) {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim {
                j[(row, col)] = xc[row] - x0[row];
            }
        }
        (j.clone(), j.determinant())
    }
}

/// Transform reference-coordinate gradients to physical gradients.
fn transform_grads(j_inv_t: &DMatrix<f64>, grad_ref: &[f64], grad_phys: &mut [f64], n_ldofs: usize, dim: usize) {
    for i in 0..n_ldofs {
        for d in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += j_inv_t[(d, k)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + d] = s;
        }
    }
}

/// Evaluate the physical gradient ∇u_h at reference point `xi` on element `e`,
/// using the correct geometric Jacobian and the full basis (including edge and
/// interior DOFs for higher-order spaces).
fn eval_grad_at<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    space: &impl FESpace<Mesh = M>,
    dofs: &[f64],
    xi: &[f64],
    elem_type: ElementType,
) -> Vec<f64> {
    let dim = mesh.dim() as usize;
    let order = space.order();
    let ref_elem = ref_elem_vol(elem_type, order);
    let n_ldofs = ref_elem.n_dofs();
    let elem_dofs = space.element_dofs(elem);
    let nodes = mesh.element_nodes(elem);

    let (jac, _det) = geom_jacobian(mesh, nodes, xi, dim, elem_type);
    let j_inv_t = jac.try_inverse().unwrap_or_default().transpose();

    let mut grad_ref = vec![0.0; n_ldofs * dim];
    ref_elem.eval_grad_basis(xi, &mut grad_ref);

    let mut grad_phys = vec![0.0; n_ldofs * dim];
    transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

    let mut grad = vec![0.0; dim];
    for i in 0..n_ldofs {
        let c = dofs[elem_dofs[i] as usize];
        for d in 0..dim {
            grad[d] += c * grad_phys[i * dim + d];
        }
    }
    grad
}

// ─── ElementIndicators ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ElementIndicators {
    pub eta: Vec<f64>,
    pub total_error: f64,
    pub estimator_name: &'static str,
    /// Per-element anisotropic refinement flags (bit k set ⇒ direction k is
    /// dominant in the flux error, MFEM `ZZErrorEstimator` aniso_flags:
    /// threshold 0.15·3/dim on d_xyz[k]/Σd_xyz).  `None` when the estimator
    /// did not compute directional energies (isotropic path).
    pub aniso_flags: Option<Vec<u8>>,
}

impl ElementIndicators {
    pub fn new(eta: Vec<f64>, name: &'static str) -> Self {
        let total_error = eta.iter().map(|v| v * v).sum::<f64>().sqrt();
        ElementIndicators { eta, total_error, estimator_name: name, aniso_flags: None }
    }

    pub fn dorfler_mark(&self, theta: f64) -> Vec<u32> {
        let target = theta.clamp(0.0, 1.0) * self.total_error;
        let mut idx: Vec<u32> = (0..self.eta.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| self.eta[b as usize].partial_cmp(&self.eta[a as usize]).unwrap());
        let mut acc = 0.0;
        let mut marked = Vec::new();
        for e in idx {
            acc += self.eta[e as usize];
            marked.push(e);
            if acc >= target { break; }
        }
        marked
    }

    /// Mark elements whose error exceeds a local absolute threshold.
    ///
    /// Returns indices of elements with `η > max_err`.
    /// Equivalent to MFEM's `ThresholdRefiner::SetLocalErrorGoal(max_err)`.
    pub fn threshold_mark(&self, max_err: f64) -> Vec<u32> {
        self.eta
            .iter()
            .enumerate()
            .filter(|(_, &e)| e > max_err)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Mark elements whose error is below a derefinement threshold.
    ///
    /// Returns indices of elements with `η < threshold`.
    /// Equivalent to MFEM's `ThresholdDerefiner::SetThreshold(threshold)`.
    pub fn derefine_mark(&self, threshold: f64) -> Vec<u32> {
        self.eta
            .iter()
            .enumerate()
            .filter(|(_, &e)| e < threshold)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// Mark elements whose error exceeds a local absolute threshold.
///
/// Returns indices of elements with `η > max_err`.
/// Equivalent to MFEM's `ThresholdRefiner::SetLocalErrorGoal(max_err)`.
pub fn threshold_mark(eta: &[f64], max_err: f64) -> Vec<u32> {
    eta.iter()
        .enumerate()
        .filter(|(_, &e)| e > max_err)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Mark elements whose error is below a derefinement threshold.
///
/// Returns indices of elements with `η < threshold`.
/// Equivalent to MFEM's `ThresholdDerefiner::SetThreshold(threshold)`.
pub fn derefine_mark(eta: &[f64], threshold: f64) -> Vec<u32> {
    eta.iter()
        .enumerate()
        .filter(|(_, &e)| e < threshold)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Element volume/area for a mesh element (used internally).
fn elem_vol(m: &dyn MeshTopology, e: u32) -> f64 {
    let n = m.element_nodes(e);
    let npe = n.len();
    if m.dim() == 2 {
        if npe == 4 {
            // Quadrilateral: shoelace formula
            let (x0, x1, x2, x3) = (m.node_coords(n[0]), m.node_coords(n[1]), m.node_coords(n[2]), m.node_coords(n[3]));
            0.5 * (x0[0]*x1[1] + x1[0]*x2[1] + x2[0]*x3[1] + x3[0]*x0[1]
                  - x1[0]*x0[1] - x2[0]*x1[1] - x3[0]*x2[1] - x0[0]*x3[1]).abs()
        } else if npe >= 3 {
            // Triangle: cross product
            let (x0, x1, x2) = (m.node_coords(n[0]), m.node_coords(n[1]), m.node_coords(n[2]));
            0.5 * ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs()
        } else { 1.0 }
    } else if npe >= 4 {
        // 3D: tetrahedron volume
        let (x0, x1, x2, x3) = (m.node_coords(n[0]), m.node_coords(n[1]), m.node_coords(n[2]), m.node_coords(n[3]));
        let a = [x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]];
        let b = [x2[0]-x0[0], x2[1]-x0[1], x2[2]-x0[2]];
        let c = [x3[0]-x0[0], x3[1]-x0[1], x3[2]-x0[2]];
        let cr = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
        (cr[0]*c[0] + cr[1]*c[1] + cr[2]*c[2]).abs() / 6.0
    } else { 1.0 }
}

/// ZZ gradient-recovery error estimator using GridFunction.
pub fn zz_estimator<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where M: MeshTopology, S: FESpace<Mesh = M> {
    let m: &M = gf.space().mesh();
    let ne = m.n_elements(); let d = m.dim() as usize;
    let xi = if d == 2 { vec![1.0/3.0, 1.0/3.0] } else { vec![0.25, 0.25, 0.25] };

    let eg: Vec<Vec<f64>> = (0..ne as u32).map(|e| gf.evaluate_gradient_at_element(e, &xi)).collect();
    let nn = m.n_nodes();
    let mut ns: Vec<Vec<f64>> = (0..nn).map(|_| vec![0.0; d]).collect();
    let mut nc = vec![0u32; nn];
    for e in 0..ne as u32 { for &n in m.element_nodes(e) { for di in 0..d { ns[n as usize][di] += eg[e as usize][di]; } nc[n as usize] += 1; } }
    for n in 0..nn { if nc[n] > 0 { for di in 0..d { ns[n][di] /= nc[n] as f64; } } }

    let mut eta = vec![0.0; ne];
    for e in 0..ne as u32 {
        let nlist = m.element_nodes(e); let npe = nlist.len();
        let mut rec = vec![0.0; d];
        for &n in nlist { for di in 0..d { rec[di] += ns[n as usize][di] / npe as f64; } }
        eta[e as usize] = ((0..d).map(|di| (eg[e as usize][di] - rec[di]).powi(2)).sum::<f64>() * elem_vol(m, e)).sqrt();
    }
    ElementIndicators::new(eta, "ZZ")
}

/// Zienkiewicz-Zhu stress-recovery estimator for linear elasticity
/// (1:1 with MFEM ex21's `ZienkiewiczZhuEstimator` + `ElasticityIntegrator`).
///
/// Per element the **stress** `σ(u) = λ tr(ε)I + 2μ ε` (symmetric tensor,
/// `dim(dim+1)/2` components) is evaluated at the centroid; a nodal average
/// gives the recovered `σ*`; then `η_K = ‖σ_h|_K − σ*|_K‖_{L²(K)}`.
///
/// `lam`/`mu` map an element attribute to the Lamé constants (the C++ side
/// evaluates `PWConstCoefficient` per element).
pub fn zz_estimator_stress<M, S>(gf: &GridFunction<'_, S>, lam: &dyn Fn(i32) -> f64, mu: &dyn Fn(i32) -> f64) -> ElementIndicators
where M: MeshTopology, S: FESpace<Mesh = M> {
    use fem_element::ReferenceElement;
    let m: &M = gf.space().mesh();
    let ne = m.n_elements(); let d = m.dim() as usize;
    let tdim = d * (d + 1) / 2; // symmetric tensor components
    let order = gf.space().order();
    let dofs = gf.dofs();

    // σ_h per element per flux-node (P1 flux element → mesh vertices).
    let mut eg: Vec<Vec<Vec<f64>>> = vec![Vec::new(); ne];
    for e in 0..ne as u32 {
        let elem_type = m.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let elem_dofs = gf.space().element_dofs(e);
        let nodes = m.element_nodes(e);
        let npe = nodes.len();
        let (l, mu_c) = (lam(m.element_tag(e)), mu(m.element_tag(e)));

        let mut sigmas = Vec::with_capacity(npe);
        for k in 0..npe {
            // Reference coordinates of flux node k (a vertex).
            let xi = ref_vertex_coords(d, npe, k);
            let (jac, _det_j) = geom_jacobian(m, nodes, &xi, d, elem_type);
            let j_inv_t = jac.try_inverse().unwrap_or_default().transpose();

            let mut grad_ref = vec![0.0; n_ldofs * d];
            ref_elem.eval_grad_basis(&xi, &mut grad_ref);
            let mut grad_phys = vec![0.0; n_ldofs * d];
            transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, d);

            let mut grad_u = vec![vec![0.0; d]; d];
            for c in 0..d {
                for l in 0..n_ldofs {
                    let val = dofs[elem_dofs[l * d + c] as usize]; // node-major vector dof table
                    for dir in 0..d {
                        grad_u[c][dir] += val * grad_phys[l * d + dir];
                    }
                }
            }

            if d == 2 {
                let exx = grad_u[0][0];
                let eyy = grad_u[1][1];
                let exy = 0.5 * (grad_u[0][1] + grad_u[1][0]);
                let tr = exx + eyy;
                sigmas.push(vec![l * tr + 2.0 * mu_c * exx,
                                 l * tr + 2.0 * mu_c * eyy,
                                 2.0 * mu_c * exy]);
            } else {
                let exx = grad_u[0][0]; let eyy = grad_u[1][1]; let ezz = grad_u[2][2];
                let exy = 0.5 * (grad_u[0][1] + grad_u[1][0]);
                let exz = 0.5 * (grad_u[0][2] + grad_u[2][0]);
                let eyz = 0.5 * (grad_u[1][2] + grad_u[2][1]);
                let tr = exx + eyy + ezz;
                sigmas.push(vec![l * tr + 2.0 * mu_c * exx,
                                 l * tr + 2.0 * mu_c * eyy,
                                 l * tr + 2.0 * mu_c * ezz,
                                 2.0 * mu_c * exy,
                                 2.0 * mu_c * exz,
                                 2.0 * mu_c * eyz]);
            }
        }
        eg[e as usize] = sigmas;
    }

    // Nodal average of the stress (recovered σ*).
    let nn = m.n_nodes();
    let mut ns: Vec<Vec<f64>> = (0..nn).map(|_| vec![0.0; tdim]).collect();
    let mut nc = vec![0u32; nn];
    for e in 0..ne as u32 {
        let nodes = m.element_nodes(e);
        for (k, &n) in nodes.iter().enumerate() {
            for di in 0..tdim { ns[n as usize][di] += eg[e as usize][k][di]; }
            nc[n as usize] += 1;
        }
    }
    for n in 0..nn { if nc[n] > 0 { for di in 0..tdim { ns[n][di] /= nc[n] as f64; } } }

    // Element error: strain energy of s = σ_h − σ* at the centroid (the flux
    // difference is linear on P1; a 1-point rule matches MFEM's integration
    // order for P1 flux elements).
    let mut eta = vec![0.0; ne];
    for e in 0..ne as u32 {
        let nodes = m.element_nodes(e);
        let npe = nodes.len();
        let (l, mu_c) = (lam(m.element_tag(e)), mu(m.element_tag(e)));
        let mut s = vec![0.0; tdim];
        for k in 0..npe {
            for di in 0..tdim {
                s[di] += (eg[e as usize][k][di] - ns[nodes[k] as usize][di]) / npe as f64;
            }
        }
        let pt_e = if d == 2 {
            let tr_e = (s[0] + s[1]) / (2.0 * (mu_c + l));
            let l_tr = l * tr_e;
            0.25 / mu_c * (s[0] * (s[0] - l_tr) + s[1] * (s[1] - l_tr) + 2.0 * s[2] * s[2])
        } else {
            let tr_e = (s[0] + s[1] + s[2]) / (2.0 * (3.0 * l + 2.0 * mu_c));
            let l_tr = l * tr_e;
            0.25 / mu_c * (s[0] * (s[0] - l_tr) + s[1] * (s[1] - l_tr) + s[2] * (s[2] - l_tr)
                           + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
        };
        eta[e as usize] = (pt_e * elem_vol(m, e)).max(0.0).sqrt();
    }
    ElementIndicators::new(eta, "ZZ-stress")
}

/// Reference (natural) coordinates of the k-th vertex of a simplex of
/// dimension `d` with `npe` vertices (used to evaluate the flux at the
/// P1 flux element's nodes).
fn ref_vertex_coords(d: usize, npe: usize, k: usize) -> Vec<f64> {
    let mut xi = vec![0.0; d];
    match (d, npe, k) {
        (2, 3, 0) => {}
        (2, 3, 1) => { xi[0] = 1.0; }
        (2, 3, 2) => { xi[1] = 1.0; }
        (2, 4, 0) => { xi[0] = -1.0; xi[1] = -1.0; }
        (2, 4, 1) => { xi[0] = 1.0; xi[1] = -1.0; }
        (2, 4, 2) => { xi[0] = 1.0; xi[1] = 1.0; }
        (2, 4, 3) => { xi[0] = -1.0; xi[1] = 1.0; }
        (3, 4, 0) => {}
        (3, 4, 1) => { xi[0] = 1.0; }
        (3, 4, 2) => { xi[1] = 1.0; }
        (3, 4, 3) => { xi[2] = 1.0; }
        (3, 8, kk) => {
            xi[0] = if kk & 1 != 0 { 1.0 } else { -1.0 };
            xi[1] = if kk & 2 != 0 { 1.0 } else { -1.0 };
            xi[2] = if kk & 4 != 0 { 1.0 } else { -1.0 };
        }
        _ => {}
    }
    xi
}
///
/// Recovers a smoothed gradient per component by solving the global L²
/// projection on the **scalar** solution space:
/// `(G_c, v) = (∂u_c/∂x_d, v)`  for each component `c` and direction `d`,
/// then `η_K = ‖∇u_h|_K − G|_K‖_{L²(K)}` summed over components.
///
/// This is the vector analogue of [`zz_estimator_l2`]; MFEM ex21 uses
/// `L2ZienkiewiczZhuEstimator` with `H1FluxReproducer`, whose recovery space
/// is the same-order H¹ space (flux = stress here is approximated by the
/// per-component gradient recovery).
pub fn zz_estimator_l2_vector<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where
    M: MeshTopology + Clone,
    S: FESpace<Mesh = M>,
{
    use fem_space::constraints::form_linear_system;
    let mref: &M = gf.space().mesh();
    let ne = mref.n_elements();
    let d = mref.dim() as usize;
    let order = gf.space().order();

    let space_ref: &S = gf.space();
    let nd = space_ref.n_dofs();
    // Vector space: component-major layout, n_scalar scalar DOFs per component.
    let n_scalar = nd / d;
    if n_scalar * d != nd {
        return ElementIndicators::new(vec![0.0; ne], "ZZ-vec");
    }

    // Recovery space = scalar H¹ of the same order (mass matrix and RHS live
    // on the scalar space; the vector solution indexes dof c·n_scalar + s).
    let scalar_space = fem_space::H1Space::new(mref.clone(), order);
    let quad_order = (order as u8) * 2 + 2;
    let mass = MassIntegrator { rho: 1.0 };
    let m_mat = Assembler::assemble_bilinear(&scalar_space, &[&mass], quad_order);

    // RHS per component: rhs[c][s] = ∫_Ω ∂u_c/∂x_d · φ_s (vector of length d per scalar dof).
    let dofs = gf.dofs();
    let mut rhs = vec![vec![vec![0.0; n_scalar]; d]; d];

    for e in 0..ne as u32 {
        let elem_type = mref.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mref.element_nodes(e);
        let elem_dofs = scalar_space.element_dofs(e);
        let quad = ref_elem.quadrature(quad_order);

        let mut phi = vec![0.0; n_ldofs];
        let mut grad_ref = vec![0.0; n_ldofs * d];
        let mut grad_phys = vec![0.0; n_ldofs * d];

        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = geom_jacobian(mref, nodes, xi, d, elem_type);
            let w_abs_det = quad.weights[q] * det_j.abs();
            let j_inv_t = jac.try_inverse().unwrap_or_default().transpose();

            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, d);

            // grad_u[c][dir] = Σ_i u[c*n_scalar + dof_i] · ∂φ_i/∂x_dir
            let mut grad_u = vec![vec![0.0; d]; d];
            for i in 0..n_ldofs {
                let s = elem_dofs[i] as usize;
                for c in 0..d {
                    let val = dofs[c * n_scalar + s];
                    for di in 0..d {
                        grad_u[c][di] += val * grad_phys[i * d + di];
                    }
                }
            }

            ref_elem.eval_basis(xi, &mut phi);

            for (i, &s) in elem_dofs.iter().enumerate() {
                let s = s as usize;
                for c in 0..d {
                    for di in 0..d {
                        rhs[c][di][s] += w_abs_det * grad_u[c][di] * phi[i];
                    }
                }
            }
        }
    }

    // Solve M g_{c,dir} = rhs[c][dir] for the recovered gradient.
    let mut recovered: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; n_scalar]; d]; d];
    for c in 0..d {
        for di in 0..d {
            let mut b = rhs[c][di].clone();
            let mut g = vec![0.0; n_scalar];
            let empty: Vec<u32> = Vec::new();
            let empty_vals: Vec<f64> = Vec::new();
            let mut m_clone = m_mat.clone();
            form_linear_system(&mut m_clone, &mut b, &mut g, &empty, &empty_vals);
            if let Ok(res) = fem_solver::solve_pcg_gssmoother(
                &m_clone,
                &b,
                &mut g,
                &fem_solver::SolverConfig {
                    rtol: 1e-10,
                    atol: 0.0,
                    max_iter: 2000,
                    verbose: false,
                    ..fem_solver::SolverConfig::default()
                },
            ) {
                let _ = res;
            }
            recovered[c][di] = g;
        }
    }

    // Element error: η_K = sqrt( Σ_c Σ_di ∫_K (∂u_c/∂x_d − G_{c,di})² )
    let mut eta = vec![0.0; ne];
    for e in 0..ne as u32 {
        let elem_type = mref.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mref.element_nodes(e);
        let elem_dofs = scalar_space.element_dofs(e);
        let quad = ref_elem.quadrature(quad_order);

        let mut phi = vec![0.0; n_ldofs];
        let mut grad_ref = vec![0.0; n_ldofs * d];
        let mut grad_phys = vec![0.0; n_ldofs * d];
        let mut e2 = 0.0;

        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = geom_jacobian(mref, nodes, xi, d, elem_type);
            let w_abs_det = quad.weights[q] * det_j.abs();
            let j_inv_t = jac.try_inverse().unwrap_or_default().transpose();

            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, d);

            let mut grad_u = vec![vec![0.0; d]; d];
            let mut rec = vec![vec![0.0; d]; d];
            for i in 0..n_ldofs {
                let s = elem_dofs[i] as usize;
                ref_elem.eval_basis(xi, &mut phi);
                for c in 0..d {
                    let val = dofs[c * n_scalar + s];
                    for di in 0..d {
                        grad_u[c][di] += val * grad_phys[i * d + di];
                        rec[c][di] += recovered[c][di][s] * phi[i];
                    }
                }
            }
            for c in 0..d {
                for di in 0..d {
                    let diff = grad_u[c][di] - rec[c][di];
                    e2 += diff * diff * w_abs_det;
                }
            }
        }
        eta[e as usize] = e2.sqrt();
    }
    ElementIndicators::new(eta, "ZZ-vec")
}

/// ZZ gradient-recovery error estimator using **L² projection** (MFEM-compatible).
///
/// This is a convenience wrapper for conforming (non-NC) meshes.
/// For non-conforming meshes with hanging nodes, use [`zz_estimator_l2_nc`].
pub fn zz_estimator_l2<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where
    M: MeshTopology + Clone,
    S: FESpace<Mesh = M>,
{
    zz_estimator_l2_nc(gf, &[])
}

/// ZZ gradient-recovery error estimator using **L² projection** (MFEM-compatible),
/// with hanging-node constraint support for non-conforming meshes.
///
/// Recovers a smoothed gradient `G(u)` by solving the global L² projection:
/// ```text
/// (G, v) = (∇u_h, v)   ∀ v ∈ V_h
/// ```
/// where `V_h` is a scalar H¹ FE space of the **same order** as the solution.
/// This yields `M·g = f`, where M is the mass matrix and
/// `f_d[i] = ∫_Ω ∂u_h/∂x_d · φ_i dΩ`.
///
/// Key features:
/// - **Same-order recovery**: the recovered gradient space has the same polynomial
///   order as the solution (matching MFEM's `ZienkiewiczZhuEstimator`).
/// - **Correct geometric Jacobian**: uses bilinear Q1 Jacobian for Quad4 elements
///   (simplex Jacobian for Tri3/Tet4), evaluated at each quadrature point.
/// - **Full quadrature** for both mass matrix and RHS assembly.
/// - **Hanging-node constraints**: `constraints` are applied to the mass matrix
///   and RHS before solving, and `recover_hanging_values` is called after.
///
/// The per-element error indicator is:
/// ```text
/// η_K = ‖∇u_h|_K − G|_K‖_{L²(K)}
/// ```
pub fn zz_estimator_l2_nc<M, S>(gf: &GridFunction<'_, S>, constraints: &[HangingNodeConstraint]) -> ElementIndicators
where
    M: MeshTopology + Clone,
    S: FESpace<Mesh = M>,
{
    let mref: &M = gf.space().mesh();
    let ne = mref.n_elements();
    let d = mref.dim() as usize;
    let order = gf.space().order();

    // ── 1. Use the solution space as the recovery space ─────────────────────
    let space_ref: &S = gf.space();
    let nd = space_ref.n_dofs();

    // ── 2. Assemble mass matrix M on the solution space ─────────────────────
    let quad_order = (order as u8) * 2 + 2;
    let mass = MassIntegrator { rho: 1.0 };
    let mut m_mat = Assembler::assemble_bilinear(space_ref, &[&mass], quad_order);

    // ── 3. Assemble RHS F_d for each component ─────────────────────────────
    let dofs = gf.dofs();
    let mut rhs = vec![vec![0.0; nd]; d];

    for e in 0..ne as u32 {
        let elem_type = mref.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mref.element_nodes(e);
        let elem_dofs = space_ref.element_dofs(e);
        let quad = ref_elem.quadrature(quad_order);

        let mut phi = vec![0.0; n_ldofs];
        let mut grad_ref = vec![0.0; n_ldofs * d];
        let mut grad_phys = vec![0.0; n_ldofs * d];

        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = geom_jacobian(mref, nodes, xi, d, elem_type);
            let w_abs_det = quad.weights[q] * det_j.abs();
            let j_inv_t = jac.try_inverse().unwrap_or_default().transpose();

            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, d);

            let mut grad_u = vec![0.0; d];
            for i in 0..n_ldofs {
                let c = dofs[elem_dofs[i] as usize];
                for di in 0..d {
                    grad_u[di] += c * grad_phys[i * d + di];
                }
            }

            ref_elem.eval_basis(xi, &mut phi);

            for (i, &dof) in elem_dofs.iter().enumerate() {
                for di in 0..d {
                    rhs[di][dof as usize] += w_abs_det * grad_u[di] * phi[i];
                }
            }
        }
    }

    // ── 4. Apply hanging-node constraints to M and each RHS component ──────
    if !constraints.is_empty() {
        for di in 0..d {
            apply_hanging_constraints(&mut m_mat, &mut rhs[di], constraints);
        }
    }

    // ── 5. Solve M·g_d = rhs_d for each component ──────────────────────────
    let cfg = SolverConfig {
        rtol: 1e-14,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };
    let mut g = vec![vec![0.0; nd]; d];
    for di in 0..d {
        // Use the constraint-modified matrix (m_mat may have been modified by
        // apply_hanging_constraints for all components, but only the first
        // call actually changes it since subsequent calls with the same
        // constraints produce the same matrix).
        if let Err(e) = solve_pcg_gssmoother(&m_mat, &rhs[di], &mut g[di], &cfg) {
            eprintln!("  Warning: CG mass matrix solve (comp {di}) failed: {e}");
        }
    }

    // ── 6. Recover hanging-node DOFs for each component ────────────────────
    if !constraints.is_empty() {
        for di in 0..d {
            recover_hanging_values(&mut g[di], constraints);
        }
    }

    // ── 7. Compute element error indicators ────────────────────────────────
    let mut eta = vec![0.0; ne as usize];
    let mut phi = Vec::new();
    let mut grad_ref = Vec::new();
    let mut grad_phys = Vec::new();

    for e in 0..ne as u32 {
        let elem_type = mref.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mref.element_nodes(e);
        let elem_dofs = space_ref.element_dofs(e);
        let quad = ref_elem.quadrature(quad_order);

        phi.resize(n_ldofs, 0.0);
        grad_ref.resize(n_ldofs * d, 0.0);
        grad_phys.resize(n_ldofs * d, 0.0);

        let mut err_sq = 0.0;

        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = geom_jacobian(mref, nodes, xi, d, elem_type);
            let w_abs_det = quad.weights[q] * det_j.abs();
            let j_inv_t = jac.try_inverse().unwrap_or_default().transpose();

            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, d);

            let mut grad_u = vec![0.0; d];
            for i in 0..n_ldofs {
                let c = dofs[elem_dofs[i] as usize];
                for di in 0..d {
                    grad_u[di] += c * grad_phys[i * d + di];
                }
            }

            ref_elem.eval_basis(xi, &mut phi);
            let mut grad_g = vec![0.0; d];
            for i in 0..n_ldofs {
                let dof = elem_dofs[i] as usize;
                for di in 0..d {
                    grad_g[di] += g[di][dof] * phi[i];
                }
            }

            let diff_sq: f64 = (0..d)
                .map(|di| (grad_u[di] - grad_g[di]).powi(2))
                .sum();
            err_sq += w_abs_det * diff_sq;
        }

        eta[e as usize] = err_sq.sqrt();
    }

    ElementIndicators::new(eta, "ZZ(L²)")
}

/// ZZ error estimator using **DOF-level averaging** (MFEM-compatible, serial version).
///
/// This matches MFEM's `ZienkiewiczZhuEstimator` algorithm:
/// 1. For each element, compute ∇u_h at the **flux space's DOF locations**
///    (all DOF nodes of the element, not just vertex nodes): for Q2 this includes
///    edge midpoints and interior nodes.
/// 2. **DOF averaging** (equivalent to `ComputeFlux` → `SumFluxAndCount`):
///    for each global DOF, average ∇u_h from all adjacent elements.
/// 3. For each element, integrate ‖∇u_h − G‖² using the flux space's shape
///    functions and the integrator's `ComputeFluxEnergy` integration rule
///    (full quadrature at `2 × order`).
///
/// The per-element error is:
/// ```text
/// η_K² = ∫_K ‖∇u_h − G‖² dΩ  ≈  f^T · M_K · f
/// ```
/// where `f = flux_coeff − smoothed_coeff` are the DOF coefficients of the
/// flux difference and `M_K` is the element mass matrix.
///
/// `hanging` is accepted for API compatibility but is NOT used by the
/// recovery: matching MFEM's `SumFluxAndCount` (gridfunc.cpp), hanging DOFs
/// participate in the plain DOF average through the fine neighbor elements
/// that use them as (positive) corner DOFs — there is no parent
/// interpolation and no skipping.  (The 0.5/0.5 parent interpolation is only
/// applied to the solution GridFunction via `recover_hanging_values`.)
pub fn zz_estimator_nodal<M, S>(
    gf: &GridFunction<'_, S>,
    _hanging: &[HangingNodeConstraint],
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m: &M = gf.space().mesh();
    let ne = m.n_elements();
    let nd = gf.space().n_dofs();
    let d = m.dim() as usize;
    let order = gf.space().order();

    // ── 1. Compute element gradients at ALL DOF locations ───────────────────
    // Like MFEM's SumFluxAndCount (gridfunc.cpp): for each element, compute
    // ∇u_h at each DOF of the element (vertex, edge, interior) using the
    // correct geometric Jacobian.  Accumulate at global DOFs and count.
    //
    // Hanging DOFs are NOT skipped: they appear as (positive) corner DOFs of
    // the fine neighbor elements, which contribute to the average — the same
    // set of elements that defines their constrained value.  MFEM's H1
    // Q1 element dofs are just its 4 corner nodes (Mesh::GetElementVertices),
    // so a hanging node is never a dof of the large parent element; it is
    // recovered purely by averaging the fine-element fluxes (no parent
    // interpolation, unlike the solution GridFunction).
    let mut dof_grad = vec![vec![0.0; d]; nd];
    let mut dof_count = vec![0usize; nd];

    for e in 0..ne as u32 {
        let elem_type = m.element_type(e);
        let elem_dofs = gf.space().element_dofs(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = m.element_nodes(e);

        // Get DOF reference coordinates for this element type
        let dof_coords = ref_elem.dof_coords();

        for (i, &dof) in elem_dofs.iter().enumerate() {
            let idx = dof as usize;
            let xi = &dof_coords[i];
            let g = eval_grad_at(m, e, gf.space(), gf.dofs(), xi, elem_type);
            for di in 0..d {
                dof_grad[idx][di] += g[di];
            }
            dof_count[idx] += 1;
        }
    }

    // Average: flux(dof) = sum(adjacent element fluxes) / count
    for i in 0..nd {
        let c = dof_count[i] as f64;
        if c > 0.0 {
            for di in 0..d {
                dof_grad[i][di] /= c;
            }
        }
    }

    // ── 2. Per-element error via element mass matrix ────────────────────────
    // Like MFEM: for each element, compute flux_coeff at DOFs (element flux),
    // subtract dof_grad (smoothed flux), and integrate ‖diff‖² via
    // ComputeFluxEnergy (i.e., f^T · M_elem · f).
    //
    // M_elem is the element mass matrix with integration rule 2×order.
    let quad_order = (order as u8) * 2;
    let mut eta = vec![0.0; ne];
    let mut aniso = vec![0u8; ne];

    for e in 0..ne as u32 {
        let elem_type = m.element_type(e);
        let elem_dofs = gf.space().element_dofs(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = m.element_nodes(e);
        let quad = ref_elem.quadrature(quad_order);

        // Compute flux difference DOF vector f = (element flux − smoothed flux)
        // at the flux-space DOF coordinates (MFEM ComputeElementFlux evaluates
        // the gradient at fluxelem.GetNodes()).
        let dof_coords = ref_elem.dof_coords();
        let mut f = vec![0.0; n_ldofs * d];
        for (i, &dof) in elem_dofs.iter().enumerate() {
            let idx = dof as usize;
            // element flux at DOF i (from solution gradient)
            let xi = &dof_coords[i];
            let eg = eval_grad_at(m, e, gf.space(), gf.dofs(), xi, elem_type);
            for di in 0..d {
                f[i * d + di] = eg[di] - dof_grad[idx][di];
            }
        }

        // Energy: ∫‖f‖² (physical components) with per-direction d_xyz
        // mirroring MFEM `DiffusionIntegrator::ComputeFluxEnergy`'s d_energy
        // (fem/bilininteg.cpp): at each quadrature point expand f via the
        // (physical) shape functions, then decompose the energy into
        // REFERENCE-domain components through vec = Jᵀ·pointflux:
        //     eng    += w·|detJ|·(pointflux·pointflux)
        //     d_xyz[k] += w·|detJ|·(Jᵀ·pointflux)ₖ²
        // (total energy is the physical L2 norm; the directional split is
        //  done in the reference frame — this is what MFEM's aniso_flags
        //  threshold 0.15·3/dim is applied to).
        let mut d_xyz = vec![0.0; d];
        let mut eng: f64 = 0.0;
        let mut phi = vec![0.0; n_ldofs];
        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = geom_jacobian(m, nodes, xi, d, elem_type);
            let w_det = quad.weights[q] * det_j.abs();
            ref_elem.eval_basis(xi, &mut phi);
            // pointflux(k) = Σ_j f[j,k]·φ_j(xi)
            let mut pointflux = vec![0.0; d];
            for k in 0..d {
                for j in 0..n_ldofs {
                    pointflux[k] += f[j * d + k] * phi[j];
                }
            }
            eng += w_det * (pointflux.iter().map(|v| v * v).sum::<f64>());
            // ref-domain components: vec = Jᵀ·pointflux
            let mut vec = vec![0.0; d];
            for r in 0..d {
                for c in 0..d {
                    vec[r] += jac[(c, r)] * pointflux[c];
                }
            }
            for k in 0..d {
                d_xyz[k] += w_det * vec[k] * vec[k];
            }
        }

        eta[e as usize] = eng.sqrt();
        // MFEM aniso_flags (gridfunc.cpp ZZErrorEstimator): the directional
        // split uses the REFERENCE-domain energy d_xyz, so the ratio is
        // d_xyz[k] / Σ_k d_xyz[k] — NOT eng (physical norm).
        let sum_e: f64 = d_xyz.iter().sum();
        if sum_e > 0.0 {
            let thresh = 0.15 * 3.0 / d as f64;
            let mut flag: u8 = 0;
            for k in 0..d {
                if d_xyz[k] / sum_e > thresh {
                    flag |= 1 << k;
                }
            }
            aniso[e as usize] = flag;
        }
    }

    let mut indicators = ElementIndicators::new(eta, "ZZ(nodal)");
    indicators.aniso_flags = Some(aniso);
    indicators
}

/// Kelly face-jump error estimator using GridFunction.
pub fn kelly_estimator<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where M: MeshTopology, S: FESpace<Mesh = M> {
    let m: &M = gf.space().mesh();
    let ne = m.n_elements(); let d = m.dim() as usize;
    let xi = if d == 2 { vec![1.0/3.0, 1.0/3.0] } else { vec![0.25, 0.25, 0.25] };
    let eg: Vec<Vec<f64>> = (0..ne as u32).map(|e| gf.evaluate_gradient_at_element(e, &xi)).collect();

    let mut fm = std::collections::HashMap::<Vec<u32>, Vec<u32>>::new();
    for e in 0..ne as u32 {
        let nd = m.element_nodes(e);
        let faces: Vec<Vec<u32>> = if nd.len() >= 3 {
            let (n0,n1,n2) = (nd[0], nd[1], nd[2]);
            // 2-D triangles (3 nodes) → 3 edges; 2-D quads (4 nodes) → 4 edges.
            // The bare `d == 2` check is insufficient — quads also satisfy d == 2
            // but need the full 4-edge list, not the triangle edge list.
            if nd.len() == 3 { vec![vec![n0,n1], vec![n1,n2], vec![n0,n2]] }
            else if nd.len() >= 4 { let n3 = nd[3]; vec![vec![n0,n1], vec![n1,n2], vec![n2,n3], vec![n3,n0]] }
            else { continue; }
        } else if nd.len() >= 4 && d == 3 {
            let (n0,n1,n2,n3) = (nd[0], nd[1], nd[2], nd[3]);
            vec![vec![n1,n2,n3], vec![n0,n2,n3], vec![n0,n1,n3], vec![n0,n1,n2]]
        } else { vec![] };
        for f in &faces { let mut k = f.clone(); k.sort_unstable(); fm.entry(k).or_default().push(e); }
    }

    let mut eta = vec![0.0; ne];
    for (key, el) in &fm {
        if el.len() != 2 { continue; }
        let (e0, e1) = (el[0] as usize, el[1] as usize);
        let (g0, g1) = (&eg[e0], &eg[e1]);
        if d == 2 && key.len() == 2 {
            let (xa, xb) = (m.node_coords(key[0]), m.node_coords(key[1]));
            let h = ((xb[0]-xa[0]).powi(2)+(xb[1]-xa[1]).powi(2)).sqrt();
            if h < 1e-30 { continue; }
            let j = (g0[0]-g1[0])*(xb[1]-xa[1])/h + (g0[1]-g1[1])*(-xb[0]+xa[0])/h;
            eta[e0] += h*j*j; eta[e1] += h*j*j;
        } else if d == 3 && key.len() == 3 {
            let (xa,xb,xc) = (m.node_coords(key[0]), m.node_coords(key[1]), m.node_coords(key[2]));
            let v1 = [xb[0]-xa[0], xb[1]-xa[1], xb[2]-xa[2]];
            let v2 = [xc[0]-xa[0], xc[1]-xa[1], xc[2]-xa[2]];
            let cr = [v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]];
            let area = 0.5 * (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt();
            if area < 1e-30 { continue; }
            let nrm = (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt();
            let j = (g0[0]-g1[0])*cr[0]/nrm + (g0[1]-g1[1])*cr[1]/nrm + (g0[2]-g1[2])*cr[2]/nrm;
            eta[e0] += area*j*j; eta[e1] += area*j*j;
        }
    }
    for e in 0..ne { eta[e] = eta[e].sqrt(); }
    ElementIndicators::new(eta, "Kelly")
}

// ─── Residual-based a posteriori error estimator ──────────────────────────

/// Residual-based a posteriori error estimator for Poisson-type problems.
///
/// For each element `e`:
/// ```text
/// η_e² = h_e² ∫_e (f + Δu_h)² dx  +  ½ Σ_{f ∈ ∂e} h_f ∫_f [[∂u_h/∂n]]² ds
/// ```
/// where:
/// - `r_e = f + Δu_h` is the element interior residual (Δu_h = 0 for P1)
/// - `j_f = [[∂u_h/∂n]]` is the jump of normal derivative across interior face `f`
/// - `h_e` is the element diameter, `h_f` the face diameter
///
/// For P1 Lagrange elements, `Δu_h = 0` so the interior residual reduces to `f`.
///
/// # Arguments
/// * `gf` - GridFunction containing the finite element solution
/// * `f` - Source function `f(x, y, z)` returning the right-hand side value
pub fn residual_estimator<M, S>(
    gf: &GridFunction<'_, S>,
    f: &dyn Fn(&[f64]) -> f64,
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m: &M = gf.space().mesh();
    let ne = m.n_elements();
    let d = m.dim() as usize;

    // Barycentric coordinates for element centroid evaluation
    let xi: Vec<f64> = if d == 2 { vec![1.0 / 3.0; 2] } else { vec![0.25; 4] };

    // ─── Per-element data ──────────────────────────────────────────────────
    let mut elem_grad: Vec<Vec<f64>> = Vec::with_capacity(ne);
    let mut elem_diam: Vec<f64> = Vec::with_capacity(ne);
    let mut elem_vols: Vec<f64> = Vec::with_capacity(ne);
    let mut elem_centroid: Vec<Vec<f64>> = Vec::with_capacity(ne);

    for e in 0..ne as u32 {
        let grad = gf.evaluate_gradient_at_element(e, &xi);
        elem_grad.push(grad);
        let vol = elem_vol(m, e);
        elem_vols.push(vol);
        let nodes = m.element_nodes(e);
        // Centroid
        let c: Vec<f64> = (0..d)
            .map(|k| nodes.iter().map(|&n| m.node_coords(n)[k]).sum::<f64>() / nodes.len() as f64)
            .collect();
        elem_centroid.push(c);
        // Diameter
        let mut max_d = 0.0;
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                let xi = m.node_coords(nodes[i]);
                let xj = m.node_coords(nodes[j]);
                let dist = (0..d).map(|k| (xi[k] - xj[k]).powi(2)).sum::<f64>().sqrt();
                if dist > max_d { max_d = dist; }
            }
        }
        elem_diam.push(max_d.max(1e-14));
    }

    // ─── Interior residual: h_e² ∫_e f² dx ────────────────────────────────
    let mut eta_sq = vec![0.0; ne];
    for e in 0..ne as u32 {
        let f_val = f(&elem_centroid[e as usize]);
        eta_sq[e as usize] = elem_diam[e as usize].powi(2) * f_val * f_val * elem_vols[e as usize];
    }

    // ─── Face jump term ───────────────────────────────────────────────────
    // Build face map: sorted node set -> [elem0, elem1]
    let mut face_map: std::collections::HashMap<Vec<u32>, Vec<u32>> = std::collections::HashMap::new();
    for e in 0..ne as u32 {
        let nd = m.element_nodes(e);
        let faces: Vec<Vec<u32>> = if d == 2 {
            // Edge faces for 2D: each consecutive pair of nodes
            let npe = nd.len();
            (0..npe).map(|i| vec![nd[i], nd[(i + 1) % npe]]).collect()
        } else {
            // Face faces for 3D
            match nd.len() {
                4 | 10 => {
                    // Tet: 4 triangular faces
                    let (n0, n1, n2, n3) = (nd[0], nd[1], nd[2], nd[3]);
                    vec![vec![n1, n2, n3], vec![n0, n2, n3], vec![n0, n1, n3], vec![n0, n1, n2]]
                }
                8 | 20 => {
                    // Hex: 6 quad faces
                    let (n0, n1, n2, n3, n4, n5, n6, n7) =
                        (nd[0], nd[1], nd[2], nd[3], nd[4], nd[5], nd[6], nd[7]);
                    vec![
                        vec![n0, n1, n2, n3], vec![n4, n5, n6, n7],
                        vec![n0, n1, n5, n4], vec![n2, n3, n7, n6],
                        vec![n0, n3, n7, n4], vec![n1, n2, n6, n5],
                    ]
                }
                _ => vec![],
            }
        };
        for mut f in faces {
            f.sort_unstable();
            face_map.entry(f).or_default().push(e);
        }
    }

    for (_key, el) in &face_map {
        if el.len() != 2 { continue; } // boundary face, skip
        let (e0, e1) = (el[0] as usize, el[1] as usize);

        // Face diameter (from sorted-node face key)
        let fnodes = _key.as_slice();
        let nf_nodes = fnodes.len();
        let mut h_f = 0.0;
        for i in 0..nf_nodes {
            for j in i + 1..nf_nodes {
                let xi = m.node_coords(fnodes[i]);
                let xj = m.node_coords(fnodes[j]);
                let dist = (0..d).map(|k| (xi[k] - xj[k]).powi(2)).sum::<f64>().sqrt();
                if dist > h_f { h_f = dist; }
            }
        }
        h_f = h_f.max(1e-14);

        // Face outward normal (from e0's perspective)
        let normal: Vec<f64> = if d == 2 {
            let (a, b) = (m.node_coords(fnodes[0]), m.node_coords(fnodes[1]));
            let tx = b[0] - a[0]; let ty = b[1] - a[1];
            let len = (tx * tx + ty * ty).sqrt().max(1e-14);
            vec![-ty / len, tx / len]
        } else {
            let (a, b, c) = (m.node_coords(fnodes[0]), m.node_coords(fnodes[1]), m.node_coords(fnodes[2]));
            let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cr = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]];
            let len = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt().max(1e-14);
            vec![cr[0] / len, cr[1] / len, cr[2] / len]
        };

        // Jump: [[∇u_h · n]]
        let jump: f64 = (0..d).map(|k| (elem_grad[e0][k] - elem_grad[e1][k]) * normal[k]).sum();
        let jump_sq = jump * jump;

        // Face area
        let face_area = if d == 2 {
            let (a, b) = (m.node_coords(fnodes[0]), m.node_coords(fnodes[1]));
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
        } else {
            let (a, b, c) = (m.node_coords(fnodes[0]), m.node_coords(fnodes[1]), m.node_coords(fnodes[2]));
            let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cr = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]];
            0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt()
        };

        let face_contrib = h_f * jump_sq * face_area;
        eta_sq[e0] += 0.5 * face_contrib;
        eta_sq[e1] += 0.5 * face_contrib;
    }

    let eta: Vec<f64> = eta_sq.iter().map(|&v| v.sqrt()).collect();
    ElementIndicators::new(eta, "Residual")
}

/// DWR (Dual-Weighted Residual) goal-oriented error estimator.
///
/// Estimates the error in a quantity of interest `J(u)`:
/// ```text
/// |J(u) - J(u_h)| ≈ Σ_K η_K
/// η_K = |∫_K f · ω_K dx| + ½ Σ_{f ⊂ ∂K} ∫_f [[∇u_h · n]] · ω_f ds
/// ```
///
/// The dual fluctuation `ω_K` on element K approximates `z_h - ẑ_h` via
/// the element-wise deviation from the mean: `ω_K² = h_K²/12 · |∇z_h|²`.
/// This yields a non-zero indicator even for P1 x P1 spaces, equivalent to
/// the standard heuristic DWR estimator used in deal.II and MFEM.
///
/// # Arguments
/// * `u_gf` - Primal solution (GridFunction)
/// * `z_dofs` - Dual solution DOF vector (same space as primal)
/// * `f` - Source function `f(x, y, z)` returning the right-hand side value
///
/// # Returns
/// Element-wise error indicators `η_K` in `ElementIndicators`.
pub fn dwr_estimator<M, S>(
    u_gf: &GridFunction<'_, S>,
    z_dofs: &[f64],
    f: &dyn Fn(&[f64]) -> f64,
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m: &M = u_gf.space().mesh();
    let ne = m.n_elements();
    let d = m.dim() as usize;

    let xi: Vec<f64> = if d == 2 { vec![1.0 / 3.0; 2] } else { vec![0.25; 4] };
    // ─── Per-element data ──────────────────────────────────────────────────
    let mut elem_grad: Vec<Vec<f64>> = Vec::with_capacity(ne);
    let mut elem_vols: Vec<f64> = Vec::with_capacity(ne);
    let mut elem_centroid: Vec<Vec<f64>> = Vec::with_capacity(ne);
    // Dual gradient (for computing ω_K ≈ mean deviation)
    let mut dual_grad: Vec<Vec<f64>> = Vec::with_capacity(ne);
    let mut elem_diam: Vec<f64> = Vec::with_capacity(ne);

    // Build element gradients for primal and dual
    for e in 0..ne as u32 {
        // Primal gradient
        let grad = u_gf.evaluate_gradient_at_element(e, &xi);
        elem_grad.push(grad);
        let vol = elem_vol(m, e);
        elem_vols.push(vol);
        let nodes = m.element_nodes(e);
        let c: Vec<f64> = (0..d)
            .map(|k| nodes.iter().map(|&n| m.node_coords(n)[k]).sum::<f64>() / nodes.len() as f64)
            .collect();
        elem_centroid.push(c);
        // Diameter
        let mut max_d = 0.0;
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                let xi = m.node_coords(nodes[i]);
                let xj = m.node_coords(nodes[j]);
                let dist = (0..d).map(|k| (xi[k] - xj[k]).powi(2)).sum::<f64>().sqrt();
                if dist > max_d { max_d = dist; }
            }
        }
        elem_diam.push(max_d.max(1e-14));

        // Dual gradient (for ω_K computation)
        // Use Stokes' formula on P1 or directly compute from DOFs
        if d == 2 && nodes.len() == 3 {
            let (n0, n1, n2) = (nodes[0], nodes[1], nodes[2]);
            let z0 = z_dofs[n0 as usize]; let z1 = z_dofs[n1 as usize]; let z2 = z_dofs[n2 as usize];
            let [x0, y0] = [m.node_coords(n0)[0], m.node_coords(n0)[1]];
            let [x1, y1] = [m.node_coords(n1)[0], m.node_coords(n1)[1]];
            let [x2, y2] = [m.node_coords(n2)[0], m.node_coords(n2)[1]];
            let j00 = x1 - x0; let j01 = x2 - x0;
            let j10 = y1 - y0; let j11 = y2 - y0;
            let det = j00 * j11 - j01 * j10;
            let inv_det = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
            // (z1 - z0) = ∇z · (x1-x0, y1-y0); (z2 - z0) = ∇z · (x2-x0, y2-y0)
            let dzx = inv_det * ( j11 * (z1 - z0) - j10 * (z2 - z0));
            let dzy = inv_det * (-j01 * (z1 - z0) + j00 * (z2 - z0));
            dual_grad.push(vec![dzx, dzy]);
        } else if d == 3 && nodes.len() == 4 {
            let (n0, n1, n2, n3) = (nodes[0], nodes[1], nodes[2], nodes[3]);
            let z = [z_dofs[n0 as usize], z_dofs[n1 as usize], z_dofs[n2 as usize], z_dofs[n3 as usize]];
            let x0 = m.node_coords(n0); let x1 = m.node_coords(n1);
            let x2 = m.node_coords(n2); let x3 = m.node_coords(n3);
            let mut j = nalgebra::Matrix3::<f64>::zeros();
            for c in 0..3 { j[(c, 0)] = x1[c] - x0[c]; j[(c, 1)] = x2[c] - x0[c]; j[(c, 2)] = x3[c] - x0[c]; }
            let inv_j = j.try_inverse().unwrap_or_else(nalgebra::Matrix3::zeros);
            let dz = nalgebra::Vector3::new(z[1] - z[0], z[2] - z[0], z[3] - z[0]);
            let g = inv_j.transpose() * dz;
            dual_grad.push(vec![g[0], g[1], g[2]]);
        } else {
            dual_grad.push(vec![0.0; d]);
        }
    }

    let mut eta = vec![0.0_f64; ne];

    // ─── Interior contribution: ∫_K f · ω_K dx ────────────────────────────
    // ω_K² = h_K² · |∇z_h|² (scaled to approximate L2 deviation from mean)
    for e in 0..ne {
        let f_val = f(&elem_centroid[e]);
        let grad_z_sq: f64 = dual_grad[e].iter().map(|&g| g * g).sum();
        let omega = elem_diam[e] * grad_z_sq.sqrt();
        eta[e] += f_val.abs() * omega * elem_vols[e];
    }

    // ─── Face jump contribution: ½ ∫_f [[∇u_h · n]] · ω_f ds ─────────────
    // Build face map: sorted node set -> [elem0, elem1]
    let mut face_map: std::collections::HashMap<Vec<u32>, Vec<u32>> = std::collections::HashMap::new();
    for e in 0..ne as u32 {
        let nd = m.element_nodes(e);
        let faces: Vec<Vec<u32>> = if d == 2 {
            let npe = nd.len();
            (0..npe).map(|i| vec![nd[i], nd[(i + 1) % npe]]).collect()
        } else {
            match nd.len() {
                4 | 10 => {
                    let (n0, n1, n2, n3) = (nd[0], nd[1], nd[2], nd[3]);
                    vec![vec![n1, n2, n3], vec![n0, n2, n3], vec![n0, n1, n3], vec![n0, n1, n2]]
                }
                8 | 20 => {
                    let (n0, n1, n2, n3, n4, n5, n6, n7) =
                        (nd[0], nd[1], nd[2], nd[3], nd[4], nd[5], nd[6], nd[7]);
                    vec![
                        vec![n0, n1, n2, n3], vec![n4, n5, n6, n7],
                        vec![n0, n1, n5, n4], vec![n2, n3, n7, n6],
                        vec![n0, n3, n7, n4], vec![n1, n2, n6, n5],
                    ]
                }
                _ => vec![],
            }
        };
        for mut f in faces {
            f.sort_unstable();
            face_map.entry(f).or_default().push(e);
        }
    }

    for (_key, el) in &face_map {
        if el.len() != 2 { continue; }
        let (e0, e1) = (el[0] as usize, el[1] as usize);

        let normal: Vec<f64> = if d == 2 {
            let (a, b) = (m.node_coords(_key[0]), m.node_coords(_key[1]));
            let tx = b[0] - a[0]; let ty = b[1] - a[1];
            let len = (tx * tx + ty * ty).sqrt().max(1e-14);
            vec![-ty / len, tx / len]
        } else {
            let (a, b, c) = (m.node_coords(_key[0]), m.node_coords(_key[1]), m.node_coords(_key[2]));
            let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cr = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]];
            let len = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt().max(1e-14);
            vec![cr[0] / len, cr[1] / len, cr[2] / len]
        };

        // Jump: [[∇u_h · n]]
        let jump: f64 = (0..d).map(|k| (elem_grad[e0][k] - elem_grad[e1][k]) * normal[k]).sum();

        let face_area = if d == 2 {
            let (a, b) = (m.node_coords(_key[0]), m.node_coords(_key[1]));
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
        } else {
            let (a, b, c) = (m.node_coords(_key[0]), m.node_coords(_key[1]), m.node_coords(_key[2]));
            let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cr = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]];
            0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt()
        };

        // Face diameter for dual weight scaling
        let mut h_f = 0.0;
        for i in 0.._key.len() {
            for j in i + 1.._key.len() {
                let xi = m.node_coords(_key[i]);
                let xj = m.node_coords(_key[j]);
                let dist = (0..d).map(|k| (xi[k] - xj[k]).powi(2)).sum::<f64>().sqrt();
                if dist > h_f { h_f = dist; }
            }
        }
        h_f = h_f.max(1e-14);

        // Dual weight at face: avg of element dual gradients scaled by h
        let grad_z0_sq: f64 = dual_grad[e0].iter().map(|&g| g * g).sum();
        let grad_z1_sq: f64 = dual_grad[e1].iter().map(|&g| g * g).sum();
        let omega_face = h_f * 0.5 * (grad_z0_sq.sqrt() + grad_z1_sq.sqrt());

        let face_contrib = 0.5 * jump.abs() * omega_face * face_area;
        eta[e0] += face_contrib;
        eta[e1] += face_contrib;
    }

    ElementIndicators::new(eta, "DWR")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test] fn zz_linear_exact() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        for &e in &zz_estimator(&gf).eta { assert!(e < 1e-12); }
    }

    #[test] fn zz_l2_linear_exact() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        for &e in &zz_estimator_l2(&gf).eta { assert!(e < 1e-10, "L² estimator should be exact for linear functions, got e={e}"); }
    }

    #[test] fn zz_l2_quadratic_nonzero() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[0] + x[1]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let eta = zz_estimator_l2(&gf).eta;
        assert!(eta.iter().sum::<f64>() > 0.0, "L² estimator should be > 0 for quadratic");
        // L² projection should give more accurate recovery → smaller total L² error
        // Compare against DOF-level averaging (both using full L² quadrature)
        let eta_nodal = zz_estimator_nodal(&gf, &[]).eta;
        let total_l2: f64 = eta.iter().sum();
        let total_nodal: f64 = eta_nodal.iter().sum();
        assert!(total_l2 < total_nodal,
            "L² projection ({:.6e}) should beat DOF-level nodal averaging ({:.6e}) for quadratic",
            total_l2, total_nodal);
    }

    #[test] fn zz_quadratic_nonzero() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[0] + x[1]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        assert!(zz_estimator(&gf).eta.iter().sum::<f64>() > 0.0);
    }

    #[test] fn kelly_linear_exact() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        for &e in &kelly_estimator(&gf).eta { assert!(e < 1e-12); }
    }

    #[test] fn kelly_quadratic_nonzero() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        assert!(kelly_estimator(&gf).eta.iter().sum::<f64>() > 0.0);
    }

    #[test] fn zz_3d_linear() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]+x[1]+x[2]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        for &e in &zz_estimator(&gf).eta { assert!(e < 1e-12); }
    }

    #[test] fn zz_3d_nonzero() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[1] + x[2]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        assert!(zz_estimator(&gf).eta.iter().sum::<f64>() > 0.0);
    }

    #[test] fn dorfler_marks() {
        let ind = ElementIndicators::new(vec![10.0, 5.0, 2.0], "t");
        assert!(!ind.dorfler_mark(0.5).is_empty());
    }

    #[test]
    fn dwr_linear_u_linear_z() {
        // u = x + y, f = 0, z = x + y (dual = primal)
        // For a linear solution where dual = primal, ω_K should be zero
        // (since z_h is linear and nodal recovery doesn't change it)
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let u_dofs = s.interpolate(&|x| x[0] + x[1]);
        let z_dofs = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, u_dofs.as_slice().to_vec());
        let ind = dwr_estimator(&gf, z_dofs.as_slice(), &|_| 0.0);
        for &e in &ind.eta {
            assert!(e < 1e-12, "DWR should be near zero when dual=primal");
        }
    }

    #[test]
    fn dwr_quadratic_u_nonlinear_z() {
        // u = x^2 + y^2, f = -4, z = sin(πx)sin(πy) (nonlinear dual, poorly resolved by P1)
        // DWR should be > 0 since ω_K ≠ 0 for the dual
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let u_dofs = s.interpolate(&|x| x[0] * x[0] + x[1] * x[1]);
        let z_dofs = s.interpolate(&|x| (std::f64::consts::PI * x[0]).sin()
            * (std::f64::consts::PI * x[1]).sin());
        let gf = GridFunction::new(&s, u_dofs.as_slice().to_vec());
        let ind = dwr_estimator(&gf, z_dofs.as_slice(), &|_| -4.0);
        assert!(ind.total_error > 0.0, "DWR should be > 0 for sin dual");
    }

    #[test]
    fn dwr_refinement_reduces_indicator() {
        // Primal with quadratic solution + source, dual solved for a different RHS
        // u = x^2 + y^2, f = -4 (constant source)
        // z = sin(πx) * y (dual differs from primal, has more structure)
        let f_u = &|_: &[f64]| -4.0;
        let z_fn = &|x: &[f64]| (std::f64::consts::PI * x[0]).sin() * x[1];
        let m_coarse = Mesh::<2>::unit_square_tri(2);
        let m_fine = Mesh::<2>::unit_square_tri(8);
        let s_coarse = H1Space::new(m_coarse, 1);
        let s_fine = H1Space::new(m_fine, 1);
        let u_fn = &|x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let u_coarse = s_coarse.interpolate(u_fn);
        let u_fine = s_fine.interpolate(u_fn);
        let z_coarse = s_coarse.interpolate(z_fn);
        let z_fine = s_fine.interpolate(z_fn);
        let gf_coarse = GridFunction::new(&s_coarse, u_coarse.as_slice().to_vec());
        let gf_fine = GridFunction::new(&s_fine, u_fine.as_slice().to_vec());
        let ind_coarse = dwr_estimator(&gf_coarse, z_coarse.as_slice(), f_u);
        let ind_fine = dwr_estimator(&gf_fine, z_fine.as_slice(), f_u);
        assert!(ind_coarse.total_error > 0.0, "coarse DWR should be > 0");
        assert!(
            ind_fine.total_error < ind_coarse.total_error,
            "refinement should reduce DWR: fine={} coarse={}",
            ind_fine.total_error, ind_coarse.total_error
        );
    }

    #[test]
    fn residual_linear_solution() {
        // u = x + y  =>  -Δu = 0, f = 0
        // For linear functions, the residual estimator should be near zero
        // since ∇u is constant, face jumps are zero.
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let ind = residual_estimator(&gf, &|_| 0.0);
        for &e in &ind.eta {
            assert!(e < 1e-12, "residual indicator should be near zero for linear f=0");
        }
    }

    #[test]
    fn residual_quadratic_nonzero() {
        // u = x^2 + y^2  =>  -Δu = -4, f = -4
        // P1 approx of quadratic has non-zero face jumps
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] * x[0] + x[1] * x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let ind = residual_estimator(&gf, &|_| -4.0);
        assert!(ind.eta.iter().sum::<f64>() > 0.0, "indicators should be > 0 for quadratic");
    }

    #[test]
    fn residual_refinement_reduces_error() {
        // u = sin(πx)sin(πy) => f = 2π² sin(πx)sin(πy)
        let f = &|x: &[f64]| 2.0 * std::f64::consts::PI * std::f64::consts::PI
            * (std::f64::consts::PI * x[0]).sin()
            * (std::f64::consts::PI * x[1]).sin();
        let m_coarse = Mesh::<2>::unit_square_tri(2);
        let m_fine = Mesh::<2>::unit_square_tri(8);
        let s_coarse = H1Space::new(m_coarse, 1);
        let s_fine = H1Space::new(m_fine, 1);
        let d_coarse = s_coarse.interpolate(&|x| (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin());
        let d_fine = s_fine.interpolate(&|x| (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin());
        let gf_coarse = GridFunction::new(&s_coarse, d_coarse.as_slice().to_vec());
        let gf_fine = GridFunction::new(&s_fine, d_fine.as_slice().to_vec());
        let ind_coarse = residual_estimator(&gf_coarse, f);
        let ind_fine = residual_estimator(&gf_fine, f);
        assert!(
            ind_fine.total_error < ind_coarse.total_error,
            "refinement should reduce residual estimator: fine={} coarse={}",
            ind_fine.total_error,
            ind_coarse.total_error
        );
    }
}
