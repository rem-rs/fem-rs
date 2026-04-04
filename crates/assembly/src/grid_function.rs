//! Grid function: a DOF coefficient vector paired with its finite element space.
//!
//! [`GridFunction`] wraps a DOF vector and provides field evaluation, error
//! norms (L², H¹ semi, full H¹), and per-element gradient computation.

use nalgebra::DMatrix;

use fem_element::lagrange::{TetP1, TriP1, TriP2};
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

// ─── Reference element factory (mirrors assembler.rs) ──────────────────────

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

// ─── Jacobian helpers (same as assembler.rs) ───────────────────────────────

fn simplex_jacobian<M: MeshTopology>(
    mesh: &M,
    geo_nodes: &[u32],
    dim: usize,
) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(geo_nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(geo_nodes[col + 1]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    let det = j.determinant();
    (j, det)
}

fn phys_coords(x0: &[f64], j: &DMatrix<f64>, xi: &[f64], dim: usize) -> Vec<f64> {
    let mut xp = x0.to_vec();
    for i in 0..dim {
        for k in 0..dim {
            xp[i] += j[(i, k)] * xi[k];
        }
    }
    xp
}

fn transform_grads(
    j_inv_t: &DMatrix<f64>,
    grad_ref: &[f64],
    grad_phys: &mut [f64],
    n_ldofs: usize,
    dim: usize,
) {
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

// ─── GridFunction ──────────────────────────────────────────────────────────

/// A finite element grid function: a DOF coefficient vector paired with its space.
///
/// Provides field evaluation and post-processing (error norms, gradient recovery).
pub struct GridFunction<'a, S: FESpace> {
    space: &'a S,
    dofs: Vec<f64>,
}

impl<'a, S: FESpace> GridFunction<'a, S> {
    /// Create a new grid function from a space reference and DOF coefficients.
    ///
    /// # Panics
    /// Panics if `dofs.len() != space.n_dofs()`.
    pub fn new(space: &'a S, dofs: Vec<f64>) -> Self {
        assert_eq!(
            dofs.len(),
            space.n_dofs(),
            "GridFunction::new: dofs length {} != space n_dofs {}",
            dofs.len(),
            space.n_dofs(),
        );
        GridFunction { space, dofs }
    }

    /// Read-only access to the DOF coefficient vector.
    pub fn dofs(&self) -> &[f64] {
        &self.dofs
    }

    /// Mutable access to the DOF coefficient vector.
    pub fn dofs_mut(&mut self) -> &mut [f64] {
        &mut self.dofs
    }

    /// Reference to the underlying finite element space.
    pub fn space(&self) -> &S {
        self.space
    }

    /// Evaluate the grid function at reference point `xi` on element `elem`.
    ///
    /// Computes `u_h(xi) = Σ_i c_i φ_i(xi)` where `c_i` are the local DOF
    /// coefficients and `φ_i` are the reference basis functions.
    pub fn evaluate_at_element(&self, elem: u32, xi: &[f64]) -> f64 {
        let mesh = self.space.mesh();
        let order = self.space.order();
        let elem_type = mesh.element_type(elem);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = self.space.element_dofs(elem);

        let mut phi = vec![0.0; n_ldofs];
        ref_elem.eval_basis(xi, &mut phi);

        let mut val = 0.0;
        for i in 0..n_ldofs {
            val += self.dofs[elem_dofs[i] as usize] * phi[i];
        }
        val
    }

    /// Evaluate the physical gradient ∇u_h at reference point `xi` on element `elem`.
    ///
    /// Returns a vector of length `dim` containing `[∂u/∂x, ∂u/∂y, ...]`.
    pub fn evaluate_gradient_at_element(&self, elem: u32, xi: &[f64]) -> Vec<f64> {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();
        let elem_type = mesh.element_type(elem);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = self.space.element_dofs(elem);
        let nodes = mesh.element_nodes(elem);

        // Jacobian and its inverse-transpose.
        let (jac, _det_j) = simplex_jacobian(mesh, nodes, dim);
        let j_inv_t = jac.try_inverse().expect("degenerate element").transpose();

        // Reference gradients.
        let mut grad_ref = vec![0.0; n_ldofs * dim];
        ref_elem.eval_grad_basis(xi, &mut grad_ref);

        // Physical gradients.
        let mut grad_phys = vec![0.0; n_ldofs * dim];
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

        // Sum contributions: ∇u_h = Σ_i c_i ∇φ_i
        let mut grad = vec![0.0; dim];
        for i in 0..n_ldofs {
            let c = self.dofs[elem_dofs[i] as usize];
            for d in 0..dim {
                grad[d] += c * grad_phys[i * dim + d];
            }
        }
        grad
    }

    /// Compute the L² error norm: `‖u_h − u_exact‖_{L²}`.
    ///
    /// # Arguments
    /// * `exact` — the exact solution as a function of physical coordinates.
    /// * `quad_order` — polynomial order that the quadrature rule integrates exactly.
    pub fn compute_l2_error(
        &self,
        exact: &dyn Fn(&[f64]) -> f64,
        quad_order: u8,
    ) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();

        let mut err2 = 0.0;

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let quad = ref_elem.quadrature(quad_order);

            let elem_dofs = self.space.element_dofs(e);
            let nodes = mesh.element_nodes(e);

            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let x0 = mesh.node_coords(nodes[0]);

            let mut phi = vec![0.0; n_ldofs];

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();

                ref_elem.eval_basis(xi, &mut phi);

                // u_h at this quadrature point.
                let mut uh = 0.0;
                for i in 0..n_ldofs {
                    uh += self.dofs[elem_dofs[i] as usize] * phi[i];
                }

                let xp = phys_coords(x0, &jac, xi, dim);
                let ue = exact(&xp);

                err2 += w * (uh - ue) * (uh - ue);
            }
        }

        err2.sqrt()
    }

    /// Compute the H¹ semi-norm error: `|u_h − u_exact|_{H¹} = ‖∇u_h − ∇u_exact‖_{L²}`.
    ///
    /// # Arguments
    /// * `exact_grad` — the exact gradient as a function of physical coordinates,
    ///   returning a vector of length `dim`.
    /// * `quad_order` — polynomial order that the quadrature rule integrates exactly.
    pub fn compute_h1_error(
        &self,
        exact_grad: &dyn Fn(&[f64]) -> Vec<f64>,
        quad_order: u8,
    ) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();

        let mut err2 = 0.0;

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let quad = ref_elem.quadrature(quad_order);

            let elem_dofs = self.space.element_dofs(e);
            let nodes = mesh.element_nodes(e);

            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let j_inv_t = jac.clone().try_inverse().unwrap().transpose();
            let x0 = mesh.node_coords(nodes[0]);

            let mut grad_ref = vec![0.0; n_ldofs * dim];
            let mut grad_phys = vec![0.0; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();

                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

                // ∇u_h at this quadrature point.
                let mut grad_uh = vec![0.0; dim];
                for i in 0..n_ldofs {
                    let c = self.dofs[elem_dofs[i] as usize];
                    for d in 0..dim {
                        grad_uh[d] += c * grad_phys[i * dim + d];
                    }
                }

                let xp = phys_coords(x0, &jac, xi, dim);
                let ge = exact_grad(&xp);

                let mut diff2 = 0.0;
                for d in 0..dim {
                    let diff = grad_uh[d] - ge[d];
                    diff2 += diff * diff;
                }
                err2 += w * diff2;
            }
        }

        err2.sqrt()
    }

    /// Compute the full H¹ norm error: `‖u_h − u_exact‖_{H¹}`.
    ///
    /// This is `sqrt(‖u_h − u‖²_{L²} + |u_h − u|²_{H¹})`.
    pub fn compute_h1_full_error(
        &self,
        exact: &dyn Fn(&[f64]) -> f64,
        exact_grad: &dyn Fn(&[f64]) -> Vec<f64>,
        quad_order: u8,
    ) -> f64 {
        let l2 = self.compute_l2_error(exact, quad_order);
        let h1_semi = self.compute_h1_error(exact_grad, quad_order);
        (l2 * l2 + h1_semi * h1_semi).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, fe_space::FESpace};

    /// Build a P1 space on a unit-square mesh and interpolate `f`.
    fn make_p1(n: usize, f: &dyn Fn(&[f64]) -> f64) -> (H1Space<SimplexMesh<2>>, Vec<f64>) {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(f);
        let dofs = v.as_slice().to_vec();
        (space, dofs)
    }

    #[test]
    fn evaluate_at_element_p1_linear() {
        // For P1, a linear function u(x,y) = 2x + 3y should be exactly reproduced.
        let f = |x: &[f64]| 2.0 * x[0] + 3.0 * x[1];
        let (space, dofs) = make_p1(4, &f);
        let gf = GridFunction::new(&space, dofs);

        // Evaluate at the centroid of element 0 in reference coordinates.
        let xi = vec![1.0 / 3.0, 1.0 / 3.0];
        let mesh = space.mesh();
        let nodes = mesh.element_nodes(0);
        let dim = 2;

        // Compute the physical coordinate of this ref point.
        let (jac, _) = simplex_jacobian(mesh, nodes, dim);
        let x0 = mesh.node_coords(nodes[0]);
        let xp = phys_coords(x0, &jac, &xi, dim);

        let uh = gf.evaluate_at_element(0, &xi);
        let exact = f(&xp);
        assert!(
            (uh - exact).abs() < 1e-12,
            "P1 should exactly reproduce linear functions: uh={uh}, exact={exact}"
        );
    }

    #[test]
    fn evaluate_gradient_p1_linear() {
        // ∇(2x + 3y) = [2, 3], should be exact for P1.
        let f = |x: &[f64]| 2.0 * x[0] + 3.0 * x[1];
        let (space, dofs) = make_p1(4, &f);
        let gf = GridFunction::new(&space, dofs);

        let xi = vec![0.25, 0.25];
        let grad = gf.evaluate_gradient_at_element(0, &xi);
        assert!((grad[0] - 2.0).abs() < 1e-12, "∂u/∂x should be 2.0, got {}", grad[0]);
        assert!((grad[1] - 3.0).abs() < 1e-12, "∂u/∂y should be 3.0, got {}", grad[1]);
    }

    #[test]
    fn compute_l2_error_exact_interpolation() {
        // For a linear function on P1, the interpolation is exact → L² error ≈ 0.
        let f = |x: &[f64]| 1.0 + x[0] - 0.5 * x[1];
        let (space, dofs) = make_p1(8, &f);
        let gf = GridFunction::new(&space, dofs);

        let err = gf.compute_l2_error(&f, 4);
        assert!(err < 1e-12, "L² error for exact P1 interpolation should be ~0, got {err}");
    }

    #[test]
    fn compute_h1_error_linear_function() {
        // For a linear function on P1, the gradient is exactly represented.
        // H¹ semi-norm error should be ~0.
        let f = |x: &[f64]| 2.0 * x[0] + 3.0 * x[1];
        let exact_grad = |_x: &[f64]| vec![2.0, 3.0];
        let (space, dofs) = make_p1(8, &f);
        let gf = GridFunction::new(&space, dofs);

        let err = gf.compute_h1_error(&exact_grad, 4);
        assert!(err < 1e-12, "H¹ semi-norm error for linear P1 should be ~0, got {err}");
    }

    #[test]
    fn compute_h1_full_error_linear() {
        let f = |x: &[f64]| x[0] + x[1];
        let exact_grad = |_x: &[f64]| vec![1.0, 1.0];
        let (space, dofs) = make_p1(8, &f);
        let gf = GridFunction::new(&space, dofs);

        let err = gf.compute_h1_full_error(&f, &exact_grad, 4);
        assert!(err < 1e-12, "Full H¹ error for linear P1 should be ~0, got {err}");
    }

    #[test]
    fn compute_l2_error_convergence_quadratic() {
        // u(x,y) = x² + y² is NOT in P1 → nonzero error that should decrease
        // with mesh refinement at rate h² in L².
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];

        let mut prev_err = f64::MAX;
        for &n in &[4usize, 8, 16] {
            let (space, dofs) = make_p1(n, &f);
            let gf = GridFunction::new(&space, dofs);
            let err = gf.compute_l2_error(&f, 4);
            assert!(err < prev_err, "L² error should decrease with refinement");
            prev_err = err;
        }
    }
}
