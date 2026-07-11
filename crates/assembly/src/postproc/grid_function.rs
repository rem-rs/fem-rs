//! Grid function: a DOF coefficient vector paired with its finite element space.
//!
//! [`GridFunction`] wraps a DOF vector and provides field evaluation, error
//! norms (L², H¹ semi, full H¹), per-element gradient computation, and L²
//! projection from a coefficient.

use nalgebra::DMatrix;

use fem_element::lagrange::{QuadQ1, QuadQ2, TetP1, TetP2, TetP3, TriP1, TriP2, TriP3};
use fem_element::ReferenceElement;
use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::fe_space::FESpace;

use crate::assembler::Assembler;
use crate::standard::{DomainSourceIntegrator, MassIntegrator};

// ─── Reference element factory (mirrors assembler.rs) ──────────────────────

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

/// Compute the L² projection of a scalar coefficient onto the FE space.
///
/// Solves `M c = b` where `M` is the mass matrix and
/// `b_i = ∫_{Ω} φ_i(x) f(x) dx`.
///
/// # Returns
/// The DOF coefficient vector `c` of length `space.n_dofs()`.
pub fn project_coefficient<S: FESpace>(
    space: &S,
    coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    quad_order: u8,
) -> Vec<f64> {
    let m: CsrMatrix<f64> = Assembler::assemble_bilinear(space, &[&MassIntegrator { rho: 1.0 }], quad_order);
    let b = Assembler::assemble_linear(space, &[&DomainSourceIntegrator::new(coeff)], quad_order);
    let mut x = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig {
        rtol: 1e-14,
        atol: 1e-30,
        max_iter: 10_000,
        ..Default::default()
    };
    solve_cg(&m, &b, &mut x, &cfg).expect("L² projection CG solve converged");
    x
}

impl<'a, S: FESpace> GridFunction<'a, S> {
    /// Create a GridFunction by L²-projecting a coefficient onto the space.
    ///
    /// This solves `M c = b` where `M` is the mass matrix and
    /// `b_i = ∫ φ_i f dx`. Unlike `interpolate` (which evaluates `f` at
    /// nodal points), this produces the optimal L² approximation.
    pub fn from_projection(space: &'a S, coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync), quad_order: u8) -> Self {
        let dofs = project_coefficient(space, coeff, quad_order);
        GridFunction::new(space, dofs)
    }

    /// In-place L² projection: replace DOFs with projected values.
    pub fn project_coefficient(&mut self, coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync), quad_order: u8) {
        self.dofs = project_coefficient(self.space, coeff, quad_order);
    }

    /// Project a scalar coefficient onto the boundary DOFs flagged by
    /// `bdr_attr` using the given `DofManager`.
    ///
    /// Equivalent to MFEM's `GridFunction::ProjectBdrCoefficient` for H1
    /// spaces.  The `dof_manager` is typically obtained from `space.dof_manager()`
    /// when `S = H1Space<M>`.
    pub fn project_bdr_coefficient(
        &mut self,
        coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
        bdr_attr: &[i32],
        dm: &fem_space::DofManager,
    ) {
        use fem_space::constraints::dirichlet::boundary_dofs;
        use fem_mesh::topology::MeshTopology;
        let mesh = self.space.mesh();
        let dofs = boundary_dofs(mesh, dm, bdr_attr);
        for &d in &dofs {
            let x = dm.dof_coord(d);
            self.dofs[d as usize] = coeff(x);
        }
    }
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

    /// Compute the L¹ error norm: `‖u_h − u_exact‖_{L¹}`.
    pub fn compute_l1_error(
        &self,
        exact: &dyn Fn(&[f64]) -> f64,
        quad_order: u8,
    ) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();

        let mut err = 0.0;
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
                let mut uh = 0.0;
                for i in 0..n_ldofs {
                    uh += self.dofs[elem_dofs[i] as usize] * phi[i];
                }
                let xp = phys_coords(x0, &jac, xi, dim);
                let ue = exact(&xp);
                err += w * (uh - ue).abs();
            }
        }
        err
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

    /// Compute the L¹ error norm.
    /// ...existing code...
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

    /// Compute the W¹,¹ semi-norm error: `|u_h − u_exact|_{W¹,¹} = ∫ |∇u_h − ∇u_exact| dΩ`.
    pub fn compute_w1_error(
        &self,
        exact_grad: &dyn Fn(&[f64]) -> Vec<f64>,
        quad_order: u8,
    ) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();

        let mut err = 0.0;
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

                let mut grad_uh = vec![0.0; dim];
                for i in 0..n_ldofs {
                    let c = self.dofs[elem_dofs[i] as usize];
                    for d in 0..dim { grad_uh[d] += c * grad_phys[i * dim + d]; }
                }
                let xp = phys_coords(x0, &jac, xi, dim);
                let ge = exact_grad(&xp);

                let mut diff_norm = 0.0;
                for d in 0..dim {
                    let diff = grad_uh[d] - ge[d];
                    diff_norm += diff.abs();
                }
                err += w * diff_norm;
            }
        }
        err
    }
}
#[cfg(test)]
mod tests {
    use fem_mesh::Mesh;
    use fem_mesh::topology::MeshTopology;
    use fem_space::{H1Space, fe_space::FESpace};

    #[test]
    fn interpolate_linear_exact_p1() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let f = |x: &[f64]| 2.0 * x[0] + 3.0 * x[1];
        let dofs_vec = space.interpolate(&f);
        let dofs = dofs_vec.as_slice();
        let coords = dof_coords_2d(&space);
        for (i, xi) in coords.iter().enumerate() {
            let fexact = f(xi);
            assert!((dofs[i] - fexact).abs() < 1e-14,
                "DOF {i}: interpolated {:.10e} ≠ exact {:.10e}", dofs[i], fexact);
        }
    }

    fn dof_coords_2d(space: &H1Space<Mesh<2>>) -> Vec<[f64; 2]> {
        let mesh = space.mesh();
        (0..mesh.n_nodes() as u32).map(|n| {
            let c = mesh.node_coords(n);
            [c[0], c[1]]
        }).collect()
    }

    #[test]
    fn interpolate_at_dof_coords_matches_function() {
        let f = |x: &[f64]| (x[0] * std::f64::consts::PI).sin();
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let dofs_vec = space.interpolate(&f);
        let dofs = dofs_vec.as_slice();
        let coords = dof_coords_2d(&space);
        let err: f64 = coords.iter().zip(dofs.iter())
            .map(|(xi, &v)| (v - f(xi)).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-14, "interpolation at nodes should match function: err={err:.6e}");
    }
}
