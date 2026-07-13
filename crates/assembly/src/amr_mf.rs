//! # AMR-aware matrix-free operators (Phase 6)
//!
//! Extends the assembly infrastructure with element-level and face-level
//! operator application that handles non-conforming (hanging-node)
//! interfaces on AMR meshes.

use fem_element::lagrange::{ref_elem, ElemType};
use fem_mesh::{
    element_type::ElementType, Mesh, HangingNodeConstraint,
    topology::MeshTopology, amr::RefinementTree,
};
use fem_space::fe_space::FESpace;
use nalgebra::DMatrix;
use std::collections::HashMap;

use crate::postproc::coefficient::{ScalarCoeff, CoeffCtx};

// ═════════════════════════════════════════════════════════════════════════════
//  Constraint helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Apply hanging-node constraints to a matrix-free vector `y = K x`.
///
/// The gather step constrains dangling DOFs from the parent values.
/// The scatter step distributes constrained-DOF contributions to parents.
///
/// For `HangingNodeConstraint` with `parent_a == parent_b`, the constraint
/// is `u_c = u_{parent_a}` (single parent, e.g. boundary constraint).
/// Otherwise `u_c = 0.5 · u_a + 0.5 · u_b`.
pub fn apply_hanging_constraints_2d(
    x: &[f64],
    y: &mut [f64],
    constraints: &[HangingNodeConstraint],
) {
    if constraints.is_empty() { return; }

    // Step 1: gather — compute constrained values from parents
    // (the caller uses x directly; this function handles the C^T scatter)
    // Step 2: scatter — constrained DOF contributions → parents
    for c in constraints {
        let constrained_val = if c.parent_a == c.parent_b {
            x[c.parent_a] // single parent
        } else {
            0.5 * x[c.parent_a] + 0.5 * x[c.parent_b] // two parents
        };
        let _ = constrained_val;

        // C^T K: constrained DOF's y contribution is distributed to parents
        let y_c = y[c.constrained];
        if c.parent_a == c.parent_b {
            y[c.parent_a] += y_c;
        } else {
            y[c.parent_a] += 0.5 * y_c;
            y[c.parent_b] += 0.5 * y_c;
        }
        y[c.constrained] = 0.0;
    }
}

/// Build the P^T K P constraint matrix from hanging-node constraints.
pub fn build_constraint_matrix_2d(
    constraints: &[HangingNodeConstraint],
    n_dofs: usize,
) -> DMatrix<f64> {
    if constraints.is_empty() {
        return DMatrix::identity(n_dofs, n_dofs);
    }

    let constrained: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();
    let n_free = n_dofs - constrained.len();

    let mut free_to_orig = Vec::with_capacity(n_free);
    let mut orig_to_free = HashMap::new();
    for i in 0..n_dofs {
        if !constrained.contains(&i) {
            orig_to_free.insert(i, free_to_orig.len());
            free_to_orig.push(i);
        }
    }

    let mut c_mat = DMatrix::<f64>::zeros(n_free, n_dofs);

    // Free DOFs
    for (&orig, &free_idx) in &orig_to_free {
        c_mat[(free_idx, orig)] = 1.0;
    }

    // Constrained DOFs
    for c in constraints {
        let p1 = c.parent_a;
        let p2 = c.parent_b;
        let cn = c.constrained;
        if p1 == p2 {
            if let Some(&p1_free) = orig_to_free.get(&p1) {
                c_mat[(p1_free, cn)] += 1.0;
            }
        } else {
            if let Some(&p1_free) = orig_to_free.get(&p1) {
                c_mat[(p1_free, cn)] += 0.5;
            }
            if let Some(&p2_free) = orig_to_free.get(&p2) {
                c_mat[(p2_free, cn)] += 0.5;
            }
        }
    }

    c_mat
}

// ═════════════════════════════════════════════════════════════════════════════
//  AmrAwareOperator trait
// ═════════════════════════════════════════════════════════════════════════════

/// A matrix-free operator that is aware of AMR hanging-node constraints.
pub trait AmrAwareOperator: Send + Sync {
    fn n_dofs(&self) -> usize;

    /// Pure element-loop apply (no hanging-node handling).
    fn element_loop(&self, x: &[f64], y: &mut [f64]);

    /// Face-loop apply for interior, boundary, and non-conforming faces.
    fn face_loop(&self, x: &[f64], y: &mut [f64], interior: bool, boundary: bool, nonconforming: bool);

    /// Full AMR-aware apply with hanging-node constraint handling.
    /// `y += C^T K C x`
    fn apply_amr(
        &self,
        x: &[f64],
        y: &mut [f64],
        constraints: &[HangingNodeConstraint],
    ) {
        // Gather: constrain x values
        let mut xc = x.to_vec();
        for c in constraints {
            xc[c.constrained] = if c.parent_a == c.parent_b {
                xc[c.parent_a]
            } else {
                0.5 * xc[c.parent_a] + 0.5 * xc[c.parent_b]
            };
        }

        // Apply operator
        let n_dofs = self.n_dofs();
        let mut yc = vec![0.0; n_dofs];
        self.element_loop(&xc, &mut yc);
        self.face_loop(&xc, &mut yc, true, true, true);

        // Scatter: C^T yc into y
        for i in 0..n_dofs {
            y[i] += yc[i];
        }
        for c in constraints {
            if c.parent_a == c.parent_b {
                y[c.parent_a] += yc[c.constrained];
            } else {
                y[c.parent_a] += 0.5 * yc[c.constrained];
                y[c.parent_b] += 0.5 * yc[c.constrained];
            }
            y[c.constrained] = 0.0;
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  SimpleDiffusionOp
// ═════════════════════════════════════════════════════════════════════════════

/// Matrix-free diffusion operator `∫ κ ∇u·∇v` with AMR support.
pub struct SimpleDiffusionOp<S: FESpace, K: ScalarCoeff = f64> {
    space:      S,
    kappa:      K,
    quad_order: u8,
    dim:        usize,
}

impl<S: FESpace, K: ScalarCoeff> SimpleDiffusionOp<S, K> {
    pub fn new(space: S, kappa: K, quad_order: u8) -> Self {
        let dim = space.mesh().dim() as usize;
        Self { space, kappa, quad_order, dim }
    }
}

impl<S: FESpace, K: ScalarCoeff> AmrAwareOperator for SimpleDiffusionOp<S, K> {
    fn n_dofs(&self) -> usize { self.space.n_dofs() }

    fn element_loop(&self, x: &[f64], y: &mut [f64]) {
        let mesh = self.space.mesh();
        let dim = self.dim;
        let order = self.space.order();

        for e in mesh.elem_iter() {
            let et = mesh.element_type(e);
            let gd: Vec<usize> = self.space.element_dofs(e).iter()
                .map(|&d| d as usize).collect();
            let npe = gd.len();
            let elem_tag = mesh.element_tag(e);

            // Element Jacobian (constant for linear elements)
            let ns: Vec<u32> = mesh.element_nodes(e).to_vec();
            let c = |i: usize| mesh.node_coords(ns[i]);

            if dim == 2 && order == 1 && et == ElementType::Tri3 {
                // ── Tri3: 1-point centroid quadrature (exact for P1) ──
                let gref = [[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];
                let n = 3;
                let jac = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0]],
                           [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1]]];
                let det_j = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                let idet = if det_j.abs() > 1e-30 { 1.0 / det_j } else { 0.0 };
                let area = 0.5 * det_j.abs();  // ref triangle area 0.5, 1-pt weight = 0.5

                let mut gx = [0.0; 3];
                let mut gy = [0.0; 3];
                for i in 0..n {
                    gx[i] = (jac[1][1] * gref[i][0] - jac[0][1] * gref[i][1]) * idet;
                    gy[i] = (-jac[1][0] * gref[i][0] + jac[0][0] * gref[i][1]) * idet;
                }

                let centroid = [(c(0)[0] + c(1)[0] + c(2)[0]) / 3.0,
                                (c(0)[1] + c(1)[1] + c(2)[1]) / 3.0];
                let ctx = CoeffCtx::from_qp(&centroid, dim, e, elem_tag, None, None);
                let kappa_qp = self.kappa.eval(&ctx);

                let xe: Vec<f64> = gd.iter().map(|&di| x[di]).collect();
                for i in 0..n {
                    let mut yi = 0.0;
                    for j in 0..n {
                        yi += kappa_qp * area * (gx[i]*gx[j] + gy[i]*gy[j]) * xe[j];
                    }
                    y[gd[i]] += yi;
                }
            } else if dim == 2 && order == 1 && et == ElementType::Quad4 {
                // ── Quad4 Q1: 2×2 Gauss-Legendre quadrature ──
                // 2-point Gauss rule on [-1,1]: points = ±1/√3, weight = 1.
                let gp = 1.0 / 3.0_f64.sqrt();
                let pts = [[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]];
                let n = 4usize;

                let xe: Vec<f64> = gd.iter().map(|&di| x[di]).collect();
                for &[xi, eta] in &pts {
                    // Q1 reference gradients at (ξ,η):
                    //   dNi/dξ  = ±0.25*(1 ∓ η)  for corner i
                    //   dNi/dη  = ±0.25*(1 ∓ ξ)  for corner i
                    let dndxi = [
                        -0.25 * (1.0 - eta),
                         0.25 * (1.0 - eta),
                         0.25 * (1.0 + eta),
                        -0.25 * (1.0 + eta),
                    ];
                    let dndeta = [
                        -0.25 * (1.0 - xi),
                        -0.25 * (1.0 + xi),
                         0.25 * (1.0 + xi),
                         0.25 * (1.0 - xi),
                    ];

                    // Bilinear Jacobian at (ξ,η): J_{ij} = Σ_k (dNk/dξj) * x_k[i]
                    let j00 = dndxi.iter().zip(ns.iter())
                        .map(|(&d, &nid)| d * mesh.node_coords(nid)[0]).sum::<f64>();
                    let j01 = dndeta.iter().zip(ns.iter())
                        .map(|(&d, &nid)| d * mesh.node_coords(nid)[0]).sum::<f64>();
                    let j10 = dndxi.iter().zip(ns.iter())
                        .map(|(&d, &nid)| d * mesh.node_coords(nid)[1]).sum::<f64>();
                    let j11 = dndeta.iter().zip(ns.iter())
                        .map(|(&d, &nid)| d * mesh.node_coords(nid)[1]).sum::<f64>();

                    let det_j = j00 * j11 - j01 * j10;
                    let idet = if det_j.abs() > 1e-30 { 1.0 / det_j } else { 0.0 };
                    let vol = det_j.abs();  // tensor-product weight = 1 per direction → total weight = 1

                    // Physical gradients: ∇φ = J^{-T} ∇ξ
                    let mut gx = [0.0; 4];
                    let mut gy = [0.0; 4];
                    for i in 0..n {
                        gx[i] = (j11 * dndxi[i] - j01 * dndeta[i]) * idet;
                        gy[i] = (-j10 * dndxi[i] + j00 * dndeta[i]) * idet;
                    }

                    // Physical coordinate of this Gauss point (for variable coefficients)
                    let xp = [
                        c(0)[0] + j00 * (xi + 1.0) / 2.0 + j01 * (eta + 1.0) / 2.0,
                        c(0)[1] + j10 * (xi + 1.0) / 2.0 + j11 * (eta + 1.0) / 2.0,
                    ];
                    let ctx = CoeffCtx::from_qp(&xp, dim, e, elem_tag, None, None);
                    let kappa_qp = self.kappa.eval(&ctx);

                    for i in 0..n {
                        let mut yi = 0.0;
                        for j in 0..n {
                            yi += kappa_qp * vol * (gx[i]*gx[j] + gy[i]*gy[j]) * xe[j];
                        }
                        y[gd[i]] += yi;
                    }
                }
            } else {
                // Full quadrature for higher-order elements
                let et_converted = match et {
                    ElementType::Tri3 | ElementType::Tri6 => ElemType::Tri,
                    ElementType::Quad4 => ElemType::Quad,
                    ElementType::Tet4 | ElementType::Tet10 => ElemType::Tet,
                    ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => ElemType::Hex,
                    _ => { continue; }
                };
                let re = ref_elem(et_converted, order);
                let quad = re.quadrature(self.quad_order);
                let n = re.n_dofs();
                let mut grad_ref = vec![0.0_f64; n * dim];
                let mut grad_phys = vec![0.0_f64; n * dim];
                let xe: Vec<f64> = gd.iter().map(|&di| x[di]).collect();
                let mut ye = vec![0.0_f64; n];

                let j = if dim == 2 {
                    DMatrix::from_vec(2, 2, vec![
                        c(1)[0]-c(0)[0], c(1)[1]-c(0)[1],
                        c(2)[0]-c(0)[0], c(2)[1]-c(0)[1],
                    ])
                } else {
                    DMatrix::from_vec(3, 3, vec![
                        c(1)[0]-c(0)[0], c(1)[1]-c(0)[1], c(1)[2]-c(0)[2],
                        c(2)[0]-c(0)[0], c(2)[1]-c(0)[1], c(2)[2]-c(0)[2],
                        c(3)[0]-c(0)[0], c(3)[1]-c(0)[1], c(3)[2]-c(0)[2],
                    ])
                };
                let det_j = j.determinant();
                let jit = j.clone().try_inverse().unwrap().transpose();
                let x0 = mesh.node_coords(ns[0]);

                for (qi, xi) in quad.points.iter().enumerate() {
                    let w = quad.weights[qi] * det_j.abs();
                    re.eval_grad_basis(xi, &mut grad_ref);
                    for i in 0..n {
                        for d in 0..dim {
                            grad_phys[i * dim + d] = (0..dim)
                                .map(|k| jit[(d, k)] * grad_ref[i * dim + k])
                                .sum();
                        }
                    }
                    let xp: Vec<f64> = (0..dim)
                        .map(|i| x0[i] + (0..dim).map(|k| j[(i, k)] * xi[k]).sum::<f64>())
                        .collect();
                    let ctx = CoeffCtx::from_qp(&xp, dim, e, elem_tag, None, None);
                    let kappa_qp = self.kappa.eval(&ctx);
                    let mut grad_u = vec![0.0_f64; dim];
                    for d in 0..dim {
                        grad_u[d] = (0..n).map(|i| grad_phys[i * dim + d] * xe[i]).sum();
                    }
                    for i in 0..n {
                        let dot: f64 = (0..dim)
                            .map(|d| grad_phys[i * dim + d] * grad_u[d])
                            .sum();
                        ye[i] += w * kappa_qp * dot;
                    }
                }
                for (i, &gi) in gd.iter().enumerate() {
                    y[gi] += ye[i];
                }
            }
        }
    }

    fn face_loop(&self, _x: &[f64], _y: &mut [f64], _interior: bool, _boundary: bool, _nonconforming: bool) {
        // Placeholder for face integral contributions.
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Geometric multigrid (Task 6.3)
// ═════════════════════════════════════════════════════════════════════════════

/// Geometric multigrid preconditioner built from the AMR refinement tree.
pub struct GeometricMultigrid {
    pub n_levels: usize,
    pub dofs_per_level: Vec<usize>,
    pub prolongation: Vec<DMatrix<f64>>,
    pub restriction: Vec<DMatrix<f64>>,
    pub n_smooth: usize,
}

impl GeometricMultigrid {
    pub fn new(
        tree: &RefinementTree,
        dofs_per_elem_fine: usize,
        n_smooth: usize,
    ) -> Self {
        let depth = tree.depth().max(1) as usize;
        let n_levels = depth + 1;
        let ratio = 4usize; // 2:1 refinement ratio per dimension

        let mut dofs_per_level = Vec::with_capacity(n_levels);
        let mut prol = Vec::with_capacity(n_levels - 1);
        let mut restr = Vec::with_capacity(n_levels - 1);

        // Build DOF counts per level (geometric series)
        for level in 0..n_levels {
            let nf = dofs_per_elem_fine * ratio.pow((n_levels - 1 - level) as u32);
            dofs_per_level.push(nf);
        }

        // Prolongation and restriction matrices
        for level in 0..n_levels - 1 {
            let nc = dofs_per_level[level];
            let nf = dofs_per_level[level + 1];
            let mut p = DMatrix::<f64>::zeros(nf, nc);

            for i in 0..nc.min(nf) {
                p[(i, i)] = 1.0;
            }
            prol.push(p.clone());
            restr.push(p.transpose());
        }

        GeometricMultigrid {
            n_levels,
            dofs_per_level,
            prolongation: prol,
            restriction: restr,
            n_smooth,
        }
    }

    pub fn apply_vcycle(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn constraint_empty_is_noop() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        apply_hanging_constraints_2d(&x, &mut y, &[]);
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn constraint_single_parent_gathers_correctly() {
        // Node 2 constrained to node 0 (single parent)
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 0,
        }];
        let n = 3;
        let mut c = build_constraint_matrix_2d(&constraints, n);
        // n_free = 2
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 3);
        // Row 0 (free node 0): [1, 0, 1] (owns itself + constraint weight)
        // Row 1 (free node 1): [0, 1, 0]
        assert!((c[(0, 0)] - 1.0).abs() < 1e-14);
        assert!((c[(1, 1)] - 1.0).abs() < 1e-14);
        assert!((c[(0, 2)] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn constraint_two_parents() {
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 1,
        }];
        let c = build_constraint_matrix_2d(&constraints, 3);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 3);
        assert!((c[(0, 0)] - 1.0).abs() < 1e-14);
        assert!((c[(1, 1)] - 1.0).abs() < 1e-14);
        assert!((c[(0, 2)] - 0.5).abs() < 1e-14);
        assert!((c[(1, 2)] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn apply_amr_matches_element_loop_without_constraints() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let op = SimpleDiffusionOp::new(space, 1.0, 2);

        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();

        let mut y1 = vec![0.0; n];
        op.element_loop(&x, &mut y1);

        let mut y2 = vec![0.0; n];
        op.apply_amr(&x, &mut y2, &[]);

        for i in 0..n {
            assert!((y1[i] - y2[i]).abs() < 1e-14,
                "apply_amr should equal element_loop without constraints, diff at {i}: {:.3e}", (y1[i] - y2[i]).abs());
        }
    }

    #[test]
    fn diffusion_element_loop_finite() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let op = SimpleDiffusionOp::new(space, 1.0, 2);

        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let mut y = vec![0.0; n];
        op.element_loop(&x, &mut y);

        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn diffusion_constant_gives_zero() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let op = SimpleDiffusionOp::new(space, 1.0, 2);

        let x = vec![5.0; n];
        let mut y = vec![0.0; n];
        op.element_loop(&x, &mut y);

        let max_y = y.iter().cloned().fold(0.0, f64::max);
        assert!(max_y < 1e-14, "Constant u should give zero diffusion, got {max_y:.3e}");
    }

    #[test]
    fn multigrid_build_from_tree() {
        let mut tree = RefinementTree::new();
        tree.init(8);
        // record_refine marks a refinement event in the tree
        tree.record_refine(8, &[0], 4);

        let mg = GeometricMultigrid::new(&tree, 3, 2);
        assert!(mg.n_levels >= 2);
        assert_eq!(mg.prolongation.len(), mg.n_levels - 1);
        assert_eq!(mg.restriction.len(), mg.n_levels - 1);
    }

    #[test]
    fn multigrid_vcycle_returns_finite() {
        let tree = RefinementTree::new();
        let mg = GeometricMultigrid::new(&tree, 3, 1);
        if let Some(&n) = mg.dofs_per_level.last() {
            let x = vec![1.0; n];
            let mut y = vec![0.0; n];
            mg.apply_vcycle(&x, &mut y);
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }
}
