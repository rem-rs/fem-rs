//! Error estimation for adaptive mesh refinement (AMR).
//! ZZ and Kelly estimators using GridFunction for arbitrary order + 2D/3D.

use nalgebra::DMatrix;

use fem_element::lagrange::{QuadQ1, QuadQ2, TetP1, TetP2, TriP1};
use fem_element::lagrange::factory::{TriPk, TetPk};
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
// âââ Reference element helper (same as grid_function.rs) ââââââââââââââââââââââ

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriPk::new(3)),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetPk::new(3)),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

/// True for simplex element types (Tri3, Tri6, Tet4, â¦).
fn is_simplex(elem_type: ElementType) -> bool {
    matches!(elem_type, ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10)
}

/// Geometric-mapping Jacobian at reference point `xi` on element `e`.
///
/// **Simplex** (Tri3, Tri6, Tet4): P1 mapping â constant Jacobian from nodes 0..dim.
/// **Quad** (Quad4): Q1 bilinear mapping â correct bilinear Jacobian at (Î¾,Î·).
///
/// Returns `(J, det J)` where J is the `dim Ã dim` Jacobian matrix.
fn geom_jacobian<M: MeshTopology>(mesh: &M, nodes: &[u32], xi: &[f64], dim: usize, elem_type: ElementType) -> (DMatrix<f64>, f64) {
    if is_simplex(elem_type) {
        // Simplex: P1 mapping, Jacobian = [x1-x0, x2-x0, â¦] (constant)
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
        // Quad: Q1 bilinear mapping at (Î¾, Î·)
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

/// Evaluate the physical gradient âu_h at reference point `xi` on element `e`,
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

// âââ ElementIndicators âââââââââââââââââââââââââââââââââââââââââââââââââââââââ

#[derive(Debug, Clone)]
pub struct ElementIndicators {
    pub eta: Vec<f64>,
    pub total_error: f64,
    pub estimator_name: &'static str,
    /// Per-element anisotropic refinement flags (bit k set â direction k is
    /// dominant in the flux error, MFEM `ZZErrorEstimator` aniso_flags:
    /// threshold 0.15Â·3/dim on d_xyz[k]/Î£d_xyz).  `None` when the estimator
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
    /// Returns indices of elements with `Î· > max_err`.
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
    /// Returns indices of elements with `Î· < threshold`.
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
/// Returns indices of elements with `Î· > max_err`.
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
/// Returns indices of elements with `Î· < threshold`.
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
/// Per element the **stress** `Ï(u) = Î» tr(Îµ)I + 2Î¼ Îµ` (symmetric tensor,
/// `dim(dim+1)/2` components) is evaluated at the centroid; a nodal average
/// gives the recovered `Ï*`; then `Î·_K = âÏ_h|_K â Ï*|_Kâ_{LÂ²(K)}`.
///
/// `lam`/`mu` map an element attribute to the LamÃ© constants (the C++ side
/// evaluates `PWConstCoefficient` per element).
pub fn zz_estimator_stress<M, S>(gf: &GridFunction<'_, S>, lam: &dyn Fn(i32) -> f64, mu: &dyn Fn(i32) -> f64) -> ElementIndicators
where M: MeshTopology, S: FESpace<Mesh = M> {
    let m: &M = gf.space().mesh();
    let ne = m.n_elements(); let d = m.dim() as usize;
    let tdim = d * (d + 1) / 2; // symmetric tensor components
    let order = gf.space().order();
    let dofs = gf.dofs();

    // Ï_h per element per flux-node (P1 flux element â mesh vertices).
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

    // Nodal average of the stress (recovered Ï*).
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

    // Element error: strain energy of s = Ï_h â Ï* at the centroid (the flux
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
/// Recovers a smoothed gradient per component by solving the global LÂ²
/// projection on the **scalar** solution space:
/// `(G_c, v) = (âu_c/âx_d, v)`  for each component `c` and direction `d`,
/// then `Î·_K = ââu_h|_K â G|_Kâ_{LÂ²(K)}` summed over components.
///
/// This is the vector analogue of [`zz_estimator_l2`]; MFEM ex21 uses
/// `L2ZienkiewiczZhuEstimator` with `H1FluxReproducer`, whose recovery space
/// is the same-order HÂ¹ space (flux = stress here is approximated by the
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

    // Recovery space = scalar HÂ¹ of the same order (mass matrix and RHS live
    // on the scalar space; the vector solution indexes dof cÂ·n_scalar + s).
    let scalar_space = fem_space::H1Space::new(mref.clone(), order);
    let quad_order = (order as u8) * 2 + 2;
    let mass = MassIntegrator { rho: 1.0 };
    let m_mat = Assembler::assemble_bilinear(&scalar_space, &[&mass], quad_order);

    // RHS per component: rhs[c][s] = â«_Î© âu_c/âx_d Â· Ï_s (vector of length d per scalar dof).
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

            // grad_u[c][dir] = Î£_i u[c*n_scalar + dof_i] Â· âÏ_i/âx_dir
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

    // Element error: Î·_K = sqrt( Î£_c Î£_di â«_K (âu_c/âx_d â G_{c,di})Â² )
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

/// ZZ gradient-recovery error estimator using **LÂ² projection** (MFEM-compatible).
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

/// ZZ gradient-recovery error estimator using **LÂ² projection** (MFEM-compatible),
/// with hanging-node constraint support for non-conforming meshes.
///
/// Recovers a smoothed gradient `G(u)` by solving the global LÂ² projection:
/// ```text
/// (G, v) = (âu_h, v)   â v â V_h
/// ```
/// where `V_h` is a scalar HÂ¹ FE space of the **same order** as the solution.
/// This yields `MÂ·g = f`, where M is the mass matrix and
/// `f_d[i] = â«_Î© âu_h/âx_d Â· Ï_i dÎ©`.
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
/// Î·_K = ââu_h|_K â G|_Kâ_{LÂ²(K)}
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

    // ââ 1. Use the solution space as the recovery space âââââââââââââââââââââ
    let space_ref: &S = gf.space();
    let nd = space_ref.n_dofs();

    // ââ 2. Assemble mass matrix M on the solution space âââââââââââââââââââââ
    let quad_order = (order as u8) * 2 + 2;
    let mass = MassIntegrator { rho: 1.0 };
    let mut m_mat = Assembler::assemble_bilinear(space_ref, &[&mass], quad_order);

    // ââ 3. Assemble RHS F_d for each component âââââââââââââââââââââââââââââ
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

    // ââ 4. Apply hanging-node constraints to M and each RHS component ââââââ
    if !constraints.is_empty() {
        for di in 0..d {
            apply_hanging_constraints(&mut m_mat, &mut rhs[di], constraints);
        }
    }

    // ââ 5. Solve MÂ·g_d = rhs_d for each component ââââââââââââââââââââââââââ
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

    // ââ 6. Recover hanging-node DOFs for each component ââââââââââââââââââââ
    if !constraints.is_empty() {
        for di in 0..d {
            recover_hanging_values(&mut g[di], constraints);
        }
    }

    // ââ 7. Compute element error indicators ââââââââââââââââââââââââââââââââ
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

    ElementIndicators::new(eta, "ZZ(LÂ²)")
}

/// ZZ error estimator using **DOF-level averaging** (MFEM-compatible, serial version).
///
/// This matches MFEM's `ZienkiewiczZhuEstimator` algorithm:
/// 1. For each element, compute âu_h at the **flux space's DOF locations**
///    (all DOF nodes of the element, not just vertex nodes): for Q2 this includes
///    edge midpoints and interior nodes.
/// 2. **DOF averaging** (equivalent to `ComputeFlux` â `SumFluxAndCount`):
///    for each global DOF, average âu_h from all adjacent elements.
/// 3. For each element, integrate ââu_h â GâÂ² using the flux space's shape
///    functions and the integrator's `ComputeFluxEnergy` integration rule
///    (full quadrature at `2 Ã order`).
///
/// The per-element error is:
/// ```text
/// Î·_KÂ² = â«_K ââu_h â GâÂ² dÎ©  â  f^T Â· M_K Â· f
/// ```
/// where `f = flux_coeff â smoothed_coeff` are the DOF coefficients of the
/// flux difference and `M_K` is the element mass matrix.
///
/// `hanging` is accepted for API compatibility but is NOT used by the
/// recovery: matching MFEM's `SumFluxAndCount` (gridfunc.cpp), hanging DOFs
/// participate in the plain DOF average through the fine neighbor elements
/// that use them as (positive) corner DOFs â there is no parent
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

    // ââ 1. Compute element gradients at ALL DOF locations âââââââââââââââââââ
    // Like MFEM's SumFluxAndCount (gridfunc.cpp): for each element, compute
    // âu_h at each DOF of the element (vertex, edge, interior) using the
    // correct geometric Jacobian.  Accumulate at global DOFs and count.
    //
    // Hanging DOFs are NOT skipped: they appear as (positive) corner DOFs of
    // the fine neighbor elements, which contribute to the average â the same
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
        let _ = ref_elem.n_dofs();
        

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

    // ââ 2. Per-element error via element mass matrix ââââââââââââââââââââââââ
    // Like MFEM: for each element, compute flux_coeff at DOFs (element flux),
    // subtract dof_grad (smoothed flux), and integrate âdiffâÂ² via
    // ComputeFluxEnergy (i.e., f^T Â· M_elem Â· f).
    //
    // M_elem is the element mass matrix with integration rule 2Ãorder.
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

        // Compute flux difference DOF vector f = (element flux â smoothed flux)
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

        // Energy: â«âfâÂ² (physical components) with per-direction d_xyz
        // mirroring MFEM `DiffusionIntegrator::ComputeFluxEnergy`'s d_energy
        // (fem/bilininteg.cpp): at each quadrature point expand f via the
        // (physical) shape functions, then decompose the energy into
        // REFERENCE-domain components through vec = JáµÂ·pointflux:
        //     eng    += wÂ·|detJ|Â·(pointfluxÂ·pointflux)
        //     d_xyz[k] += wÂ·|detJ|Â·(JáµÂ·pointflux)âÂ²
        // (total energy is the physical L2 norm; the directional split is
        //  done in the reference frame â this is what MFEM's aniso_flags
        //  threshold 0.15Â·3/dim is applied to).
        let mut d_xyz = vec![0.0; d];
        let mut eng: f64 = 0.0;
        let mut phi = vec![0.0; n_ldofs];
        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = geom_jacobian(m, nodes, xi, d, elem_type);
            let w_det = quad.weights[q] * det_j.abs();
            ref_elem.eval_basis(xi, &mut phi);
            // pointflux(k) = Î£_j f[j,k]Â·Ï_j(xi)
            let mut pointflux = vec![0.0; d];
            for k in 0..d {
                for j in 0..n_ldofs {
                    pointflux[k] += f[j * d + k] * phi[j];
                }
            }
            eng += w_det * (pointflux.iter().map(|v| v * v).sum::<f64>());
            // ref-domain components: vec = JáµÂ·pointflux
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
        // d_xyz[k] / Î£_k d_xyz[k] â NOT eng (physical norm).
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
            // 2-D triangles (3 nodes) â 3 edges; 2-D quads (4 nodes) â 4 edges.
            // The bare `d == 2` check is insufficient â quads also satisfy d == 2
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

// âââ Residual-based a posteriori error estimator ââââââââââââââââââââââââââ

/// Residual-based a posteriori error estimator for Poisson-type problems.
///
/// For each element `e`:
/// ```text
/// Î·_eÂ² = h_eÂ² â«_e (f + Îu_h)Â² dx  +  Â½ Î£_{f â âe} h_f â«_f [[âu_h/ân]]Â² ds
/// ```
/// where:
/// - `r_e = f + Îu_h` is the element interior residual (Îu_h = 0 for P1)
/// - `j_f = [[âu_h/ân]]` is the jump of normal derivative across interior face `f`
/// - `h_e` is the element diameter, `h_f` the face diameter
///
/// For P1 Lagrange elements, `Îu_h = 0` so the interior residual reduces to `f`.
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

    // âââ Per-element data ââââââââââââââââââââââââââââââââââââââââââââââââââ
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

    // âââ Interior residual: h_eÂ² â«_e fÂ² dx ââââââââââââââââââââââââââââââââ
    let mut eta_sq = vec![0.0; ne];
    for e in 0..ne as u32 {
        let f_val = f(&elem_centroid[e as usize]);
        eta_sq[e as usize] = elem_diam[e as usize].powi(2) * f_val * f_val * elem_vols[e as usize];
    }

    // âââ Face jump term âââââââââââââââââââââââââââââââââââââââââââââââââââ
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

        // Jump: [[âu_h Â· n]]
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
/// |J(u) - J(u_h)| â Î£_K Î·_K
/// Î·_K = |â«_K f Â· Ï_K dx| + Â½ Î£_{f â âK} â«_f [[âu_h Â· n]] Â· Ï_f ds
/// ```
///
/// The dual fluctuation `Ï_K` on element K approximates `z_h - zÌ_h` via
/// the element-wise deviation from the mean: `Ï_KÂ² = h_KÂ²/12 Â· |âz_h|Â²`.
/// This yields a non-zero indicator even for P1 x P1 spaces, equivalent to
/// the standard heuristic DWR estimator used in deal.II and MFEM.
///
/// # Arguments
/// * `u_gf` - Primal solution (GridFunction)
/// * `z_dofs` - Dual solution DOF vector (same space as primal)
/// * `f` - Source function `f(x, y, z)` returning the right-hand side value
///
/// # Returns
/// Element-wise error indicators `Î·_K` in `ElementIndicators`.
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
    // âââ Per-element data ââââââââââââââââââââââââââââââââââââââââââââââââââ
    let mut elem_grad: Vec<Vec<f64>> = Vec::with_capacity(ne);
    let mut elem_vols: Vec<f64> = Vec::with_capacity(ne);
    let mut elem_centroid: Vec<Vec<f64>> = Vec::with_capacity(ne);
    // Dual gradient (for computing Ï_K â mean deviation)
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

        // Dual gradient (for Ï_K computation)
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
            // (z1 - z0) = âz Â· (x1-x0, y1-y0); (z2 - z0) = âz Â· (x2-x0, y2-y0)
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

    // âââ Interior contribution: â«_K f Â· Ï_K dx ââââââââââââââââââââââââââââ
    // Ï_KÂ² = h_KÂ² Â· |âz_h|Â² (scaled to approximate L2 deviation from mean)
    for e in 0..ne {
        let f_val = f(&elem_centroid[e]);
        let grad_z_sq: f64 = dual_grad[e].iter().map(|&g| g * g).sum();
        let omega = elem_diam[e] * grad_z_sq.sqrt();
        eta[e] += f_val.abs() * omega * elem_vols[e];
    }

    // âââ Face jump contribution: Â½ â«_f [[âu_h Â· n]] Â· Ï_f ds âââââââââââââ
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

        // Jump: [[âu_h Â· n]]
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

// âââ AnisotropicErrorEstimator / LpErrorEstimator / LS-ZZ (fem/estimators.*) ââ

/// MFEM `AnisotropicErrorEstimator`: an error estimator that additionally
/// provides one anisotropic flag per element (bit `k` set â the flux error is
/// dominant in direction `k`, i.e. the element should be split in direction
/// `k`).
pub trait AnisotropicErrorEstimator {
    /// MFEM `GetLocalErrors`.
    fn get_local_errors(&self) -> &[f64];
    /// MFEM `GetTotalError`.
    fn get_total_error(&self) -> f64;
    /// MFEM `GetAnisotropicFlags`: empty when anisotropic estimates are not
    /// available or not enabled.
    fn get_anisotropic_flags(&self) -> Vec<i32>;
}

impl AnisotropicErrorEstimator for ElementIndicators {
    fn get_local_errors(&self) -> &[f64] {
        &self.eta
    }
    fn get_total_error(&self) -> f64 {
        self.total_error
    }
    fn get_anisotropic_flags(&self) -> Vec<i32> {
        match &self.aniso_flags {
            Some(v) => v.iter().map(|&b| b as i32).collect(),
            None => Vec::new(),
        }
    }
}

/// P1/Q1 basis values of the element vertices at a reference point `xi`
/// (natural domains: simplex [0,1]^d barycentric, quad/hex [-1,1]^d).
fn vertex_shapes(_elem_type: ElementType, xi: &[f64], npe: usize) -> Vec<f64> {
    let mut s = vec![0.0_f64; npe];
    match npe {
        3 => {
            s[0] = 1.0 - xi[0] - xi[1];
            s[1] = xi[0];
            s[2] = xi[1];
        }
        4 if xi.len() == 2 => {
            let (e, n) = (xi[0], xi[1]);
            s[0] = 0.25 * (1.0 - e) * (1.0 - n);
            s[1] = 0.25 * (1.0 + e) * (1.0 - n);
            s[2] = 0.25 * (1.0 + e) * (1.0 + n);
            s[3] = 0.25 * (1.0 - e) * (1.0 + n);
        }
        4 => {
            s[0] = 1.0 - xi[0] - xi[1] - xi[2];
            s[1] = xi[0];
            s[2] = xi[1];
            s[3] = xi[2];
        }
        8 => {
            for (k, v) in s.iter_mut().enumerate() {
                let sx = if k & 1 != 0 { xi[0] } else { -xi[0] };
                let sy = if k & 2 != 0 { xi[1] } else { -xi[1] };
                let sz = if k & 4 != 0 { xi[2] } else { -xi[2] };
                *v = 0.125 * (1.0 + sx) * (1.0 + sy) * (1.0 + sz);
            }
        }
        _ => {}
    }
    s
}

/// Physical coordinates of a reference point under the linear element
/// geometry (the geometric model of all estimators in this file).
fn phys_point<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    xi: &[f64],
    dim: usize,
    elem_type: ElementType,
) -> Vec<f64> {
    let shapes = vertex_shapes(elem_type, xi, nodes.len());
    let mut x = vec![0.0_f64; dim];
    for (i, &n) in nodes.iter().enumerate() {
        let xc = mesh.node_coords(n);
        for d in 0..dim {
            x[d] += shapes[i] * xc[d];
        }
    }
    x
}

/// Gauss-Legendre rule of total polynomial degree `order` on the natural
/// reference domain of the element geometry.
fn geom_rule(elem_type: ElementType, order: u8) -> (Vec<Vec<f64>>, Vec<f64>) {
    use fem_element::quadrature as q;
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => {
            let r = q::tri_rule(order);
            (r.points, r.weights)
        }
        ElementType::Quad4 => {
            let r = q::quad_rule(order);
            (r.points, r.weights)
        }
        ElementType::Tet4 | ElementType::Tet10 => {
            let r = q::tet_rule(order);
            (r.points, r.weights)
        }
        ElementType::Hex8 => {
            let r = q::hex_rule(order);
            (r.points, r.weights)
        }
        other => panic!("geom_rule: unsupported element type {other:?}"),
    }
}

/// MFEM `LpErrorEstimator` + `GridFunction::ComputeElementLpErrors`: per
/// element,
///
/// ```text
/// eta_e = ( â«_e |u_h - u_ex|^p dx )^(1/p)          (finite p)
/// eta_e = max_q |u_h(x_q) - u_ex(x_q)|             (p = infinity)
/// ```
///
/// with the integration rule of order `2*order + 3`. `total_error` mirrors
/// the C++ `pow(error_estimates.Sum(), 1/p)` for finite `p` (note: this is
/// *not* the global Lp norm; it is kept verbatim for 1:1 behavior). For
/// `p = INFINITY` the C++ total is degenerate (`pow(sum, 0)`), so the max
/// element error is reported instead.
pub fn lp_error_estimator<M, S>(
    gf: &GridFunction<'_, S>,
    p: f64,
    exact: &dyn Fn(&[f64]) -> f64,
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m: &M = gf.space().mesh();
    let ne = m.n_elements() as usize;
    let d = m.dim() as usize;
    let order = gf.space().order();
    let _dofs = gf.dofs();
    let inf = f64::INFINITY;

    let mut eta = vec![0.0_f64; ne];
    for e in 0..ne as u32 {
        let elem_type = m.element_type(e);
        let (points, weights) = geom_rule(elem_type, 2 * order + 3);
        let nodes = m.element_nodes(e);
        for (q, xi) in points.iter().enumerate() {
            let (_, det) = geom_jacobian(m, nodes, xi, d, elem_type);
            let x = phys_point(m, nodes, xi, d, elem_type);
            let uh = gf.evaluate_at_element(e, xi);
            let mut diff = (uh - exact(&x)).abs();
            if p < inf {
                diff = diff.powf(p);
                eta[e as usize] += weights[q] * det * diff;
            } else {
                eta[e as usize] = eta[e as usize].max(diff);
            }
        }
        if p < inf {
            // negative quadrature weights may cause the error to be negative
            eta[e as usize] = eta[e as usize].abs().powf(1.0 / p);
        }
    }    let total = if p < inf {
        let sum: f64 = eta.iter().sum();
        sum.powf(1.0 / p)
    } else {
        eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    };
    ElementIndicators {
        eta,
        total_error: total,
        estimator_name: "Lp",
        aniso_flags: None,
    }
}

/// Shifted Legendre polynomials P~_0..P~_order at `x` (MFEM
/// `Linear1D::CalcLegendre`: P~_n(x) = P_n(2x-1)).
fn calc_legendre_shifted(order: u8, x: f64, poly: &mut [f64]) {
    poly[0] = 1.0;
    if order >= 1 {
        poly[1] = 2.0 * x - 1.0;
    }
    for n in 2..=order as usize {
        poly[n] = ((2 * n - 1) as f64 * (2.0 * x - 1.0) * poly[n - 1]
            - (n - 1) as f64 * poly[n - 2])
            / n as f64;
    }
}

/// MFEM `TensorProductLegendre`: tensor-product shifted Legendre basis of
/// degree `order` per dimension, mapped from the (rotated, 2D) bounding box
/// `[xmin, xmax]` to [0,1]^dim.
fn tensor_product_legendre(
    dim: usize,
    order: u8,
    x: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    angle: f64,
    midpoint: &[f64],
) -> Vec<f64> {
    let rotate = dim == 2 && (angle != 0.0 || midpoint.iter().any(|&v| v != 0.0));
    let xr = |c: usize| -> f64 {
        if rotate {
            let tx = x[0] - midpoint[0];
            let ty = x[1] - midpoint[1];
            match c {
                0 => tx * (-angle).cos() - ty * (-angle).sin(),
                _ => tx * (-angle).sin() + ty * (-angle).cos(),
            }
        } else {
            x[c]
        }
    };
    let np = order as usize + 1;
    let mut px = vec![0.0_f64; np];
    let mut py = vec![0.0_f64; np];
    let mut pz = vec![0.0_f64; np];
    calc_legendre_shifted(order, (xr(0) - xmin[0]) / (xmax[0] - xmin[0]), &mut px);
    if dim > 1 {
        calc_legendre_shifted(order, (xr(1) - xmin[1]) / (xmax[1] - xmin[1]), &mut py);
    }
    if dim == 3 {
        calc_legendre_shifted(order, (xr(2) - xmin[2]) / (xmax[2] - xmin[2]), &mut pz);
    }
    let nb = (np as u32).pow(dim as u32) as usize;
    let mut poly = vec![0.0_f64; nb];
    match dim {
        1 => poly[..np].copy_from_slice(&px),
        2 => {
            for j in 0..np {
                for i in 0..np {
                    poly[i + np * j] = px[i] * py[j];
                }
            }
        }
        3 => {
            for k in 0..np {
                for j in 0..np {
                    for i in 0..np {
                        poly[i + np * j + np * np * k] = px[i] * py[j] * pz[k];
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    poly
}

/// MFEM `GridFunction::ComputeElementGradError`: the L2 error of the gradient
/// against `exgrad` on element `e`, rule of order `2*order + 3`.
fn compute_element_grad_error<M, S>(
    gf: &GridFunction<'_, S>,
    e: u32,
    exgrad: &dyn Fn(&[f64]) -> Vec<f64>,
    flux_order: u8,
    (points, weights): (&[Vec<f64>], &[f64]),
) -> f64
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m: &M = gf.space().mesh();
    let d = m.dim() as usize;
    let elem_type = m.element_type(e);
    let nodes = m.element_nodes(e);
    let _ = flux_order;
    let mut err = 0.0_f64;
    for (q, xi) in points.iter().enumerate() {
        let (_, det) = geom_jacobian(m, nodes, xi, d, elem_type);
        let x = phys_point(m, nodes, xi, d, elem_type);
        let grad = gf.evaluate_gradient_at_element(e, xi);
        let ex = exgrad(&x);
        let mut e2 = 0.0_f64;
        for c in 0..d {
            e2 += (ex[c] - grad[c]) * (ex[c] - grad[c]);
        }
        err += weights[q] * det * e2;
    }
    err.abs().sqrt()
}

/// MFEM `LSZZErrorEstimator` for the diffusion flux of a scalar H1 field: the
/// flux `âu_h` of the two elements of every interior face patch (with equal
/// attributes) is fitted by a global tensor-product shifted-Legendre
/// polynomial of degree `patch_order` (least squares over the `2p+1`-order
/// Gauss points of the patch, rotated bounding box in 2D); the element error
/// is the L2 norm of `poly â âu_h` per face contribution, calibrated by
/// `eta /= counters/2`.
///
/// Deviation vs C++: the flux is always the physical gradient (the C++
/// delegates to `BilinearFormIntegrator::ComputeElementFlux`); boundary-face
/// patches and attribute-crossing patches are skipped exactly like the C++
/// (`counters == 0` â `eta = INFINITY`).
pub fn ls_zz_estimator<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m: &M = gf.space().mesh();
    let d = m.dim() as usize;
    let ne = m.n_elements() as usize;
    let order = gf.space().order();

    let mut error_estimates = vec![0.0_f64; ne];
    let mut counters = vec![0_u32; ne];
    let mut total_error = 0.0_f64;

    // Interior face patches (shared sorted node key â the two side elements).
    let mut fm = std::collections::HashMap::<Vec<u32>, Vec<u32>>::new();
    for e in 0..ne as u32 {
        let nd = m.element_nodes(e);
        let faces: Vec<Vec<u32>> = if nd.len() == 3 {
            vec![vec![nd[0], nd[1]], vec![nd[1], nd[2]], vec![nd[0], nd[2]]]
        } else if nd.len() == 4 && d == 2 {
            vec![
                vec![nd[0], nd[1]],
                vec![nd[1], nd[2]],
                vec![nd[2], nd[3]],
                vec![nd[3], nd[0]],
            ]
        } else if d == 3 {
            vec![
                vec![nd[1], nd[2], nd[3]],
                vec![nd[0], nd[2], nd[3]],
                vec![nd[0], nd[1], nd[3]],
                vec![nd[0], nd[1], nd[2]],
            ]
        } else {
            continue;
        };
        for f in &faces {
            let mut k = f.clone();
            k.sort_unstable();
            fm.entry(k).or_default().push(e);
        }
    }

    for (key, patch) in &fm {
        if patch.len() != 2 {
            continue; // boundary face
        }
        let (el1, el2) = (patch[0], patch[1]);
        // Skip patches crossing an attribute interface
        // (subdomain_reconstruction = true, the C++ default).
        if m.element_tag(el1) != m.element_tag(el2) {
            continue;
        }

        let patch_order = order.max(order);
        let np = patch_order as usize + 1;
        let nb = (np as u32).pow(d as u32) as usize;
        let flux_order = 2 * patch_order + 1;

        // 2.B. Rotated bounding box of the patch (2D: aligned with the face).
        let mut xmin = vec![f64::INFINITY; d];
        let mut xmax = vec![f64::NEG_INFINITY; d];
        let mut angle = 0.0_f64;
        let mut midpoint = vec![0.0_f64; d];
        if d == 2 {
            let (a, b) = (m.node_coords(key[0]), m.node_coords(key[1]));
            for dd in 0..2 {
                midpoint[dd] = 0.5 * (a[dd] + b[dd]);
            }
            angle = (b[1] - a[1]).atan2(b[0] - a[0]);
        }
        for &ielem in patch {
            let et = m.element_type(ielem);
            let (points, _w) = geom_rule(et, flux_order);
            let nodes = m.element_nodes(ielem);
            for xi in &points {
                let x = phys_point(m, nodes, xi, d, et);
                let xr: Vec<f64> = if d == 2 {
                    let tx = x[0] - midpoint[0];
                    let ty = x[1] - midpoint[1];
                    vec![
                        tx * (-angle).cos() - ty * (-angle).sin(),
                        tx * (-angle).sin() + ty * (-angle).cos(),
                    ]
                } else {
                    x.clone()
                };
                for dd in 0..d {
                    xmax[dd] = xmax[dd].max(xr[dd]);
                    xmin[dd] = xmin[dd].min(xr[dd]);
                }
            }
        }

        // 2.C. Normal equations over the patch quadrature points.
        let mut a_mat = DMatrix::<f64>::zeros(nb, nb);
        let mut b_mat = DMatrix::<f64>::zeros(nb, d);
        for &ielem in patch {
            let et = m.element_type(ielem);
            let (points, _w) = geom_rule(et, flux_order);
            let nodes = m.element_nodes(ielem);
            for xi in &points {
                let (_, det) = geom_jacobian(m, nodes, xi, d, et);
                let x = phys_point(m, nodes, xi, d, et);
                let fl = gf.evaluate_gradient_at_element(ielem, xi);
                let pvec = tensor_product_legendre(d, patch_order, &x, &xmin, &xmax, angle, &midpoint);
                for l in 0..nb {
                    for lp in 0..nb {
                        a_mat[(l, lp)] += pvec[l] * pvec[lp];
                    }
                    for n in 0..d {
                        b_mat[(l, n)] += pvec[l] * fl[n];
                    }
                }
                let _ = det;
            }
        }

        // 2.D/2.E. Tichonov regularization and solve.
        for i in 0..nb {
            a_mat[(i, i)] += 0.0; // tichonov_coeff = 0 (C++ default)
        }
        // Note: the C++ checks factorization success and retries with a
        // 1e-8 diagonal shift when singular; nalgebra's LU solve handles the
        // (square, generically nonsingular) system directly.
        let lu = a_mat.clone().lu();
        let coeffs = lu.solve(&b_mat).expect("LSZZErrorEstimator: singular normal equations");

        // 2.F/3. Gradient error of the global polynomial per element.
        let exgrad = |x: &[f64]| -> Vec<f64> {
            let pvec = tensor_product_legendre(d, patch_order, x, &xmin, &xmax, angle, &midpoint);
            let mut f = vec![0.0_f64; d];
            for i in 0..nb {
                for j in 0..d {
                    f[j] += coeffs[(i, j)] * pvec[i];
                }
            }
            f
        };
        let mut patch_error = 0.0_f64;
        for &ielem in patch {
            let et = m.element_type(ielem);
            let (points, weights) = geom_rule(et, 2 * order + 3);
            let element_error = compute_element_grad_error(
                gf,
                ielem,
                &exgrad,
                flux_order,
                (&points, &weights),
            );
            let element_error = element_error * element_error;
            patch_error += element_error;
            error_estimates[ielem as usize] += element_error;
            counters[ielem as usize] += 1;
        }
        total_error += patch_error;
    }

    // 4. Calibration.
    for (eta_e, cnt) in error_estimates.iter_mut().zip(counters.iter()) {
        if *cnt == 0 {
            *eta_e = f64::INFINITY;
        } else {
            *eta_e /= *cnt as f64 / 2.0;
            *eta_e = eta_e.sqrt();
        }
    }
    let total = (total_error / d as f64).sqrt();
    ElementIndicators {
        eta: error_estimates,
        total_error: total,
        estimator_name: "LS-ZZ",
        aniso_flags: None,
    }
}

/// MFEM `ZienkiewiczZhuEstimator` with `SetAnisotropic(true)`: the ZZ gradient
/// recovery error with per-element anisotropic flags from the directional
/// flux energies (MFEM `DiffusionIntegrator::ComputeFluxEnergy` with
/// `d_energy`): `d_xyz[k] = â« (J^T Îflux)_kÂ² dx` over the rule of order
/// `2*(order+1)`; bit `k` of the flag is set when
/// `d_xyz[k]/Î£d_xyz > 0.15Â·3/dim`.
///
/// The recovered (smooth) flux is the nodal-averaged gradient of
/// [`zz_estimator`], evaluated on the element by vertex interpolation.
pub fn zz_estimator_aniso<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m: &M = gf.space().mesh();
    let ne = m.n_elements() as usize;
    let d = m.dim() as usize;
    let order = gf.space().order();

    // Element gradients at the vertices + nodal recovery (as zz_estimator).
    let eg: Vec<Vec<Vec<f64>>> = (0..ne as u32)
        .map(|e| {
            let nodes = m.element_nodes(e);
            let npe = nodes.len();
            (0..npe)
                .map(|k| {
                    let xi = ref_vertex_coords(d, npe, k);
                    gf.evaluate_gradient_at_element(e, &xi)
                })
                .collect()
        })
        .collect();
    let nn = m.n_nodes();
    let mut ns: Vec<Vec<f64>> = (0..nn).map(|_| vec![0.0; d]).collect();
    let mut nc = vec![0_u32; nn];
    for e in 0..ne as u32 {
        for (k, &n) in m.element_nodes(e).iter().enumerate() {
            for di in 0..d {
                ns[n as usize][di] += eg[e as usize][k][di];
            }
            nc[n as usize] += 1;
        }
    }
    for n in 0..nn {
        if nc[n] > 0 {
            for di in 0..d {
                ns[n as usize][di] /= nc[n as usize] as f64;
            }
        }
    }

    let thresh = 0.15 * 3.0 / d as f64;
    let mut eta = vec![0.0_f64; ne];
    let mut flags = vec![0_u8; ne];
    for e in 0..ne as u32 {
        let et = m.element_type(e);
        let nodes = m.element_nodes(e);
        let npe = nodes.len();
        let (points, weights) = geom_rule(et, 2 * (order + 1));
        let mut energy = 0.0_f64;
        let mut d_xyz = vec![0.0_f64; d];
        for (q, xi) in points.iter().enumerate() {
            let (jac, det) = geom_jacobian(m, nodes, xi, d, et);
            let flux = gf.evaluate_gradient_at_element(e, xi);
            // Recovered flux interpolated from the nodal values.
            let shapes = vertex_shapes(et, xi, npe);
            let mut rec = vec![0.0_f64; d];
            for (k, &n) in nodes.iter().enumerate() {
                for di in 0..d {
                    rec[di] += shapes[k] * ns[n as usize][di];
                }
            }
            let w = weights[q] * det;
            for c in 0..d {
                let dfc = flux[c] - rec[c];
                energy += w * dfc * dfc;
                // (J^T Δflux)_k with Δflux = flux − rec at this point.
                let mut jtf = 0.0_f64;
                for dd in 0..d {
                    jtf += jac[(dd, c)] * (flux[dd] - rec[dd]);
                }
                d_xyz[c] += w * jtf * jtf;
            }
        }
        eta[e as usize] = energy.max(0.0).sqrt();
        let sum: f64 = d_xyz.iter().sum();
        if sum > 0.0 {
            let mut flag = 0_u8;
            for k in 0..d {
                if d_xyz[k] / sum > thresh {
                    flag |= 1 << k;
                }
            }
            flags[e as usize] = flag;
        }
    }
    let total_error = eta.iter().map(|v| v * v).sum::<f64>().sqrt();
    ElementIndicators {
        eta,
        total_error,
        estimator_name: "ZZ-aniso",
        aniso_flags: Some(flags),
    }
}

// âââ Tests âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// Lp estimator: the exact field reproduced by the FE space gives ~0.
    #[test]
    fn lp_error_exact_field() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 2);
        let d = s.interpolate(&|x| x[0] * x[0] + x[1] * x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let ind = lp_error_estimator(&gf, 2.0, &|x| x[0] * x[0] + x[1] * x[1]);
        for &e in &ind.eta {
            assert!(e < 1e-10, "L2 error should vanish, got {e}");
        }
        // C++-verbatim total: pow(sum(eta), 1/p) { tiny etas still amplify.
        let want = ind.eta.iter().sum::<f64>().powf(0.5);
        assert!((ind.total_error - want).abs() < 1e-14);
        assert!(ind.total_error < 1e-5);
    }

    /// Lp estimator on a constant offset: every element error is the offset
    /// (analytically known error field).
    #[test]
    fn lp_error_known_offset() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        // Constant error field |u_h - (u_h + 0.5)| = 0.5: the Lp element error
        // is 0.5 * area^(1/p) (triangles of the 4x4 mesh have area 1/32).
        let area = 1.0_f64 / 32.0;
        let ind1 = lp_error_estimator(&gf, 1.0, &|x| x[0] + x[1] + 0.5);
        for &e in &ind1.eta {
            assert!((e - 0.5 * area).abs() < 1e-12, "L1 eta = {e}");
        }
        let ind2 = lp_error_estimator(&gf, 2.0, &|x| x[0] + x[1] + 0.5);
        for &e in &ind2.eta {
            assert!((e - 0.5 * area.sqrt()).abs() < 1e-12, "L2 eta = {e}");
        }
        let inf = f64::INFINITY;
        let indinf = lp_error_estimator(&gf, inf, &|x| x[0] + x[1] + 0.5);
        for &e in &indinf.eta {
            assert!((e - 0.5).abs() < 1e-12, "Linf eta = {e}");
        }
        // C++-verbatim total: pow(sum(eta), 1/p).
        let want = (0.5 * area * ind1.eta.len() as f64).powf(1.0);
        assert!((ind1.total_error - want).abs() < 1e-12);
    }

    /// Lp estimator with a linear error ramp: eta = (â« (cÂ·h)Â²)^(1/2) is
    /// element-size dependent; check the value on one element analytically.
    #[test]
    fn lp_error_ramp_l2_value() {
        // Single unit square, Q1, u_h = 0; exact = 2x â error density (2x)Â².
        // â«âÂ¹â«âÂ¹ 4xÂ² dx dy = 4/3 â eta = sqrt(4/3).
        let m = Mesh::<2>::make_cartesian_2d(1, 1, 1.0, 1.0);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|_| 0.0);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let ind = lp_error_estimator(&gf, 2.0, &|x| 2.0 * x[0]);
        assert!((ind.eta[0] - (4.0_f64 / 3.0).sqrt()).abs() < 1e-12, "eta = {}", ind.eta[0]);
    }

    /// LS-ZZ: exact for fields whose gradient lies in the patch-polynomial
    /// space (linear u â constant gradient).
    #[test]
    fn ls_zz_linear_exact() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let ind = ls_zz_estimator(&gf);
        for &e in &ind.eta {
            assert!(e < 1e-10, "LS-ZZ should be exact for linear u, got {e}");
        }
    }

    /// LS-ZZ for a quadratic field on P1 elements: the fitted flux is the P1
    /// gradient (piecewise constant, element-dependent), which the degree-1
    /// patch polynomial cannot represent exactly - nonzero error, and smaller
    /// than for a trigonometric field.
    #[test]
    fn ls_zz_quadratic_exact_trig_nonzero() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] * x[0] + x[1] * x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let ind = ls_zz_estimator(&gf);
        let quad_total: f64 = ind.eta.iter().sum();
        assert!(quad_total > 1e-6, "quadratic LS-ZZ should be nonzero: {quad_total}");

        let d2 = s.interpolate(&|x| (std::f64::consts::PI * x[0]).sin());
        let gf2 = GridFunction::new(&s, d2.as_slice().to_vec());
        let ind2 = ls_zz_estimator(&gf2);
        let trig_total: f64 = ind2.eta.iter().sum();
        assert!(
            trig_total > 1e-4,
            "LS-ZZ should be nonzero for a trigonometric field"
        );
        assert!(
            quad_total < trig_total,
            "quadratic flux error {quad_total} should beat trig {trig_total}"
        );
    }

    /// Anisotropic ZZ: for u = xÂ² the flux error is x-dominant (flag bit 0),
    /// for u = yÂ² it is y-dominant (flag bit 1).
    #[test]
    fn zz_aniso_flags_direction() {
        let m = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        let s = H1Space::new(m, 2);

        let d0 = s.interpolate(&|x| x[0] * x[0]);
        let gf0 = GridFunction::new(&s, d0.as_slice().to_vec());
        let ind0 = zz_estimator_aniso(&gf0);
        let flags0 = ind0.get_anisotropic_flags();
        assert_eq!(flags0.len(), ind0.eta.len());
        for (e, &f) in flags0.iter().enumerate() {
            assert!(
                f & 1 != 0 || f & 2 != 0,
                "element {e}: expected a split flag, got {f}"
            );
            assert!(f & 1 != 0, "element {e}: u=x^2 must flag direction 0 (f={f})");
        }

        let d1 = s.interpolate(&|x| x[1] * x[1]);
        let gf1 = GridFunction::new(&s, d1.as_slice().to_vec());
        let ind1 = zz_estimator_aniso(&gf1);
        for (e, &f) in ind1.get_anisotropic_flags().iter().enumerate() {
            assert!(f & 2 != 0, "element {e}: u=y^2 must flag direction 1 (f={f})");
        }

        // The trait object path (MFEM AnisotropicErrorEstimator).
        let ani: &dyn AnisotropicErrorEstimator = &ind0;
        assert_eq!(ani.get_local_errors().len(), ind0.eta.len());
        assert!((ani.get_total_error() - ind0.total_error).abs() < 1e-14);
    }

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
        for &e in &zz_estimator_l2(&gf).eta { assert!(e < 1e-10, "LÂ² estimator should be exact for linear functions, got e={e}"); }
    }

    #[test] fn zz_l2_quadratic_nonzero() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[0] + x[1]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let eta = zz_estimator_l2(&gf).eta;
        assert!(eta.iter().sum::<f64>() > 0.0, "LÂ² estimator should be > 0 for quadratic");
        // LÂ² projection should give more accurate recovery â smaller total LÂ² error
        // Compare against DOF-level averaging (both using full LÂ² quadrature)
        let eta_nodal = zz_estimator_nodal(&gf, &[]).eta;
        let total_l2: f64 = eta.iter().sum();
        let total_nodal: f64 = eta_nodal.iter().sum();
        assert!(total_l2 < total_nodal,
            "LÂ² projection ({:.6e}) should beat DOF-level nodal averaging ({:.6e}) for quadratic",
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
        // For a linear solution where dual = primal, Ï_K should be zero
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
        // u = x^2 + y^2, f = -4, z = sin(Ïx)sin(Ïy) (nonlinear dual, poorly resolved by P1)
        // DWR should be > 0 since Ï_K â  0 for the dual
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
        // z = sin(Ïx) * y (dual differs from primal, has more structure)
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
        // u = x + y  =>  -Îu = 0, f = 0
        // For linear functions, the residual estimator should be near zero
        // since âu is constant, face jumps are zero.
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
        // u = x^2 + y^2  =>  -Îu = -4, f = -4
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
        // u = sin(Ïx)sin(Ïy) => f = 2ÏÂ² sin(Ïx)sin(Ïy)
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

// âââ H(div) RT0 ZZ estimator (wrapper for l2_zz::l2_zz_estimator) ââââââââââ
// This provides the MFEM L2ZZEstimator (H(div) RT0 projection) interface
// that returns ElementIndicators, matching the other estimators in this module.
// The raw l2_zz_estimator is Quad4-only and returns Vec<f64>; this wrapper
// converts to ElementIndicators for API consistency.

pub fn zz_estimator_l2_hdiv(mesh: &fem_mesh::Mesh<2>, u: &[f64]) -> ElementIndicators {
    let eta = crate::postproc::l2_zz_rt1::l2_zz_estimator(mesh, u);
    ElementIndicators::new(eta, "ZZ(L2-Hdiv)")
}
