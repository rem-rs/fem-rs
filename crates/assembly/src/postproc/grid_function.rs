//! Grid function: a DOF coefficient vector paired with its finite element space.
//!
//! [`GridFunction`] wraps a DOF vector and provides field evaluation, error
//! norms (L², H¹ semi, full H¹), per-element gradient computation, and L²
//! projection from a coefficient.

use nalgebra::DMatrix;

use fem_element::lagrange::{TetP1, TetP2, TriP1};
use fem_element::lagrange::factory::{TriPk, TetPk};
use fem_element::quadrature::quad_rule_01;
use fem_element::{vec_ref_elem, VecFamily, ReferenceElement, QuadratureRule, VectorReferenceElement};
use fem_linalg::CsrMatrix;
use fem_mesh::element_jacobian_at;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::{EdgeKey, HCurlSpace, HDivSpace, L2Space};
use fem_mesh::Mesh;

use crate::assembler::Assembler;
use crate::standard::{DomainSourceIntegrator, MassIntegrator};
use crate::vector_assembler::{piola_hcurl_basis, piola_hcurl_curl, piola_hdiv_basis, piola_hdiv_div};

// ─── Reference element factory (mirrors assembler.rs) ──────────────────────

pub(crate) fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriPk::new(3)),
        // Quad4: order 0 (L² P0) is the constant element; orders 1+ use
        // QuadQk (Gauss-Lobatto nodes on [0,1]^2) — must match the
        // assembler's reference element (assembler.rs::ref_elem_vol).  The
        // legacy QuadQ1/Q2/Q3 live on [-1,1]^2 while quadrature rules
        // (quad_rule_01) live on [0,1]^2 — mixing them gave wrong L2 errors.
        (ElementType::Quad4, 0) => Box::new(P0),
        (ElementType::Quad4, o) => Box::new(fem_element::lagrange::QuadQk::new(o as usize)),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetPk::new(3)),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

/// Constant (P0) reference element on `[0,1]²` — 1 DOF, basis ≡ 1.
struct P0;

impl ReferenceElement for P0 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        g[0] = 0.0;
        g[1] = 0.0;
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule_01(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.5, 0.5]] }
}

// ─── Jacobian helpers (same as assembler.rs) ───────────────────────────────

pub(crate) fn simplex_jacobian(mesh: &dyn MeshTopology,
    geo_nodes: &[u32],
    dim: usize,
) -> (DMatrix<f64>, f64) {
    let edim = mesh.dim() as usize;
    if edim != dim {
        // Surface mesh: compute 3×2 Jacobian using all coordinates,
        // then use the norm of the cross product for the measure.
        // J is a (edim × dim) matrix, but we only need |J₁ × J₂|.
        let x0 = mesh.node_coords(geo_nodes[0]);
        let x1 = mesh.node_coords(geo_nodes[1]);
        // Quad4 reference axes (0,0),(1,0),(0,1) map to nodes 1 and 3 (CCW);
        // simplices use node 2 for the second axis.
        let x2 = mesh.node_coords(geo_nodes[if geo_nodes.len() >= 4 { 3 } else { 2 }]);
        // Edge vectors in 3D
        let mut e1 = [0.0f64; 3]; for i in 0..edim { e1[i] = x1[i] - x0[i]; }
        let mut e2 = [0.0f64; 3]; for i in 0..edim { e2[i] = x2[i] - x0[i]; }
        // Cross product
        let cx = e1[1]*e2[2] - e1[2]*e2[1];
        let cy = e1[2]*e2[0] - e1[0]*e2[2];
        let cz = e1[0]*e2[1] - e1[1]*e2[0];
        let area_2d = (cx*cx + cy*cy + cz*cz).sqrt();
        // Return a 2×2 "dummy" Jacobian with the determinant = area_2d
        // This preserves the J^{-T} computation for gradient transformation.
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        j[(0,0)] = 1.0; j[(1,1)] = 1.0;
        (j, area_2d)
    } else {
        // Standard flat mesh: dim×dim Jacobian.
        // Reference axes map to physical nodes: Quad4 is CCW
        // (0,0),(1,0),(1,1),(0,1) so the (0,1) axis is node 3 (not node 2);
        // Hex8 +x/+y/+z corners are nodes 1/3/4; simplices use nodes 1..dim.
        let x0 = mesh.node_coords(geo_nodes[0]);
        let axis_nodes: &[usize] = match (dim, geo_nodes.len()) {
            (2, 4) => &[1, 3],      // Quad4
            (3, 8) => &[1, 3, 4],   // Hex8
            _ => &[1, 2, 3],        // Tri3/Tri6 (dim=2), Tet4/Tet10 (dim=3)
        };
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim {
            let xc = mesh.node_coords(geo_nodes[axis_nodes[col]]);
            for row in 0..dim {
                j[(row, col)] = xc[row] - x0[row];
            }
        }
        let det = j.determinant();
        (j, det)
    }
}

pub(crate) fn phys_coords(x0: &[f64], j: &DMatrix<f64>, xi: &[f64], dim: usize) -> Vec<f64> {    let mut xp = x0.to_vec();
    for i in 0..dim {
        for k in 0..dim {
            xp[i] += j[(i, k)] * xi[k];
        }
    }
    xp
}

/// High-order geometry Jacobian for volume elements (curved bodies).
///
/// `J_{ij}(ξ) = Σ_k x_k[i] · ∂φ_k/∂ξ_j` where `φ_k` are the **geometry**
/// basis functions (e.g. `QuadQk` on `[0,1]^d`) and `x_k` the high-order
/// geometry node coordinates from [`MeshTopology::geom_coords_of`].
///
/// Returns `(J_col_major, det J, x_phys)` with `J` stored column-major
/// (`j[i + d·dim] = ∂x_i/∂ξ_d`), matching the scalar assembler.
fn iso_jacobian_geom<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    geo_elem: &dyn ReferenceElement,
    xi: &[f64],
    dim: usize,
) -> (Vec<f64>, f64, Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0_f64; n_geo * dim];
    let mut phi_geo = vec![0.0_f64; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);

    let mut j = vec![0.0_f64; dim * dim];
    let mut xp = vec![0.0_f64; dim];
    for k in 0..n_geo {
        let xk = mesh.geom_coords_of(nodes[k]);
        for i in 0..dim {
            xp[i] += phi_geo[k] * xk[i];
            for d in 0..dim {
                j[i + d * dim] += xk[i] * grad_geo[k * dim + d];
            }
        }
    }
    let det = match dim {
        2 => j[0] * j[3] - j[1] * j[2],
        3 => j[0] * (j[4] * j[8] - j[5] * j[7])
            - j[1] * (j[3] * j[8] - j[5] * j[6])
            + j[2] * (j[3] * j[7] - j[4] * j[6]),
        _ => panic!("iso_jacobian_geom: unsupported dim {dim}"),
    };
    (j, det, xp)
}

/// Element Jacobian: uses affine (simplex) Jacobian for triangles/tets,
/// and isoparametric Jacobian for quads/hexes (evaluated at `xi`).
///
/// Returns `(jacobian_matrix, determinant, physical_coords)`.
/// For surface meshes, the determinant is the area element.
fn element_jacobian<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    nodes: &[u32],
    xi: &[f64],
    dim: usize,
) -> (DMatrix<f64>, f64, Vec<f64>) {
    let elem_type = mesh.element_type(elem);
    let needs_iso = matches!(elem_type,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20
        | ElementType::Prism6 | ElementType::Prism15
        | ElementType::Pyramid5);

    if needs_iso {
        // Isoparametric mapping via geometry reference element.
        // Quad geometry lives on [0,1]^2 (QuadQk, matching MFEM's [0,1]^2
        // reference square); the solution basis also lives on [0,1]^2 for
        // all orders (QuadQk), so the Jacobian is sampled at matching points.
        let geo: Box<dyn ReferenceElement> = match elem_type {
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
                use fem_element::lagrange::factory::QuadQk;
                Box::new(QuadQk::new(mesh.geom_order().max(1) as usize))
            }
            _ => ref_elem_vol(ElementType::Quad4, 1) as Box<dyn ReferenceElement>,
        };
        let n_geo = geo.n_dofs();
        let mut grad_geo = vec![0.0_f64; n_geo * dim];
        let mut phi_geo = vec![0.0_f64; n_geo];
        geo.eval_grad_basis(xi, &mut grad_geo);
        geo.eval_basis(xi, &mut phi_geo);

        let mut j = DMatrix::<f64>::zeros(dim, dim);
        let mut xp = vec![0.0_f64; dim];
        for k in 0..n_geo {
            let xk = mesh.node_coords(nodes[k]);
            for i in 0..dim {
                xp[i] += phi_geo[k] * xk[i];
                for d in 0..dim {
                    j[(i, d)] += xk[i] * grad_geo[k * dim + d];
                }
            }
        }
        let det = j.determinant();
        (j, det, xp)
    } else {
        let (jac, det) = simplex_jacobian(mesh, nodes, dim);
        // For simplex elements, physical coords = x0 + J * xi
        let x0 = mesh.node_coords(nodes[0]);
        let xp = phys_coords(x0, &jac, xi, dim);
        (jac, det, xp)
    }
}

/// Physical coordinates for a surface triangle in 3D.
/// Maps reference coords (ξ₁, ξ₂) to 3D using edge vectors e₁, e₂.
fn surface_phys_coords(x0: &[f64], e1: &[f64], e2: &[f64], xi: &[f64]) -> Vec<f64> {
    let n = x0.len();
    let mut xp = x0.to_vec();
    for i in 0..n {
        xp[i] += e1[i] * xi[0] + e2[i] * xi[1];
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

/// L²-project a GridFunction from one space to another.
///
/// Solves `M_tgt c_tgt = b` where `M_tgt` is the mass matrix of the target
/// space and `b_i = ∫ φ_i^{tgt} u_src dx`.  The source GridFunction is
/// evaluated at the target space's quadrature points.
///
/// This is useful for:
/// - Transferring solutions between meshes (after remeshing)
/// - Projecting from a fine space to a coarse space (coarsening)
/// - Initializing a solution on a new space
///
/// # Arguments
/// * `src` — source GridFunction
/// * `tgt_space` — target finite element space
/// * `quad_order` — quadrature order (should be ≥ order_src + order_tgt)
///
/// # Returns
/// A new GridFunction on the target space.
pub fn project_grid_function<'a, S1: FESpace, S2: FESpace>(
    src: &GridFunction<'a, S1>,
    tgt_space: &'a S2,
    quad_order: u8,
) -> GridFunction<'a, S2> {
    let src_mesh = src.space().mesh();
    let tgt_mesh = tgt_space.mesh();
    let tgt_order = tgt_space.order();
    let src_order = src.space().order();

    // Build the mass matrix of the target space
    let m_tgt = Assembler::assemble_bilinear(tgt_space, &[&MassIntegrator { rho: 1.0 }], quad_order);

    // Build the RHS: b_i = ∫ φ_i^{tgt} u_src dx
    let n_tgt = tgt_space.n_dofs();
    let mut b = vec![0.0_f64; n_tgt];

    // For each element in the target space, integrate φ_i^{tgt} * u_src
    let n_elems = tgt_mesh.n_elements() as u32;
    for e in 0..n_elems {
        let elem_type = tgt_mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, tgt_order);
        let n_ldofs = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(quad_order);
        let elem_dofs = tgt_space.element_dofs(e);
        let nodes = tgt_mesh.element_nodes(e);
        let dim = tgt_mesh.topological_dim() as usize;
        let (jac, det_j) = simplex_jacobian(tgt_mesh, nodes, dim);
        let x0 = tgt_mesh.node_coords(nodes[0]);
        let mut phi = vec![0.0; n_ldofs];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            ref_elem.eval_basis(xi, &mut phi);

            // Evaluate u_src at the physical point
            let xp = phys_coords(x0, &jac, xi, dim);
            // Find which source element contains xp and evaluate
            // For simplicity, evaluate using the source space's evaluate_at_element
            // (this is a simplified version - full version needs point location)
            let u_src = evaluate_at_point(src, &xp, src_mesh, src_order);

            for i in 0..n_ldofs {
                b[elem_dofs[i] as usize] += w * phi[i] * u_src;
            }
        }
    }

    // Solve M_tgt c_tgt = b
    let mut x = vec![0.0; n_tgt];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 1e-30,
        max_iter: 10_000,
        ..Default::default()
    };
    solve_cg(&m_tgt, &b, &mut x, &cfg).expect("project_grid_function: CG solve failed");

    GridFunction::new(tgt_space, x)
}

/// Evaluate a GridFunction at a physical point (simplified).
///
/// Finds the element containing the point and evaluates the basis functions.
/// For points not in any element, returns 0.0.
fn evaluate_at_point<S: FESpace>(
    gf: &GridFunction<'_, S>,
    xp: &[f64],
    mesh: &dyn MeshTopology,
    order: u8,
) -> f64 {
    let n_elems = mesh.n_elements() as u32;
    let dim = mesh.topological_dim() as usize;

    // Brute-force search for the element containing the point
    // (full version would use a point locator like FindPointsGSLIB)
    for e in 0..n_elems {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let elem_dofs = gf.space().element_dofs(e);
        let nodes = mesh.element_nodes(e);
        let (jac, _det_j) = simplex_jacobian(mesh, nodes, dim);
        let x0 = mesh.node_coords(nodes[0]);

        // Transform to reference coordinates (simplified: assume affine)
        // For a proper implementation, use Newton iteration
        let xi = if dim == 2 {
            // Simple inverse for affine triangles
            let dx = xp[0] - x0[0];
            let dy = xp[1] - x0[1];
            let j_inv = match jac.try_inverse() {
                Some(inv) => inv,
                None => continue,
            };
            let r = j_inv[(0, 0)] * dx + j_inv[(0, 1)] * dy;
            let s = j_inv[(1, 0)] * dx + j_inv[(1, 1)] * dy;
            vec![r, s]
        } else {
            continue;
        };

        // Check if the point is inside the reference element
        let eps = 1e-10;
        let sum: f64 = xi.iter().sum();
        if xi.iter().all(|&x| x >= -eps) && sum <= 1.0 + eps {
            let mut phi = vec![0.0; n_ldofs];
            ref_elem.eval_basis(&xi, &mut phi);
            let mut val = 0.0;
            for i in 0..n_ldofs {
                val += gf.dofs()[elem_dofs[i] as usize] * phi[i];
            }
            return val;
        }
    }
    0.0 // Point not found
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

    /// Estimate the minimum and maximum values of the grid function.
    ///
    /// Uses piecewise linear bounds (evaluation at vertices) plus recursive
    /// subdivision for higher-order elements, matching MFEM 4.10's
    /// `GridFunction::GetBounds()`.
    ///
    /// For linear elements, this returns the exact min/max (at vertices).
    /// For higher-order elements, the bounds may be conservative.
    ///
    /// # Returns
    /// `(u_min, u_max)` — the estimated lower and upper bounds.
    ///
    /// # Example
    /// ```rust,ignore
    /// let gf = GridFunction::from_projection(&space, &coeff, 3);
    /// let (u_min, u_max) = gf.get_bounds();
    /// assert!(u_min <= u_max);
    /// ```
    pub fn get_bounds(&self) -> (f64, f64) {
        let mesh = self.space.mesh();
        let order = self.space.order();
        let n_elems = mesh.n_elements();

        if n_elems == 0 {
            return (f64::INFINITY, f64::NEG_INFINITY);
        }

        let mut global_min = f64::INFINITY;
        let mut global_max = f64::NEG_INFINITY;

        // For order 1 (linear), the extrema are at vertices — nodal values are exact.
        // For higher-order, we subdivide each element and evaluate at subdivision points.
        let subdivisions = if order <= 1 { 1 } else { order as usize };

        for e in 0..n_elems as u32 {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let elem_dofs = self.space.element_dofs(e);

            // Collect local DOF values for this element
            let local_dofs: Vec<f64> = elem_dofs.iter().map(|&d| self.dofs[d as usize]).collect();

            // Evaluate at vertices first (exact for linear, good bounds for higher-order)
            let vertices: Vec<Vec<f64>> = match elem_type {
                ElementType::Tri3 | ElementType::Tri6 => {
                    vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]
                }
                ElementType::Quad4 | ElementType::Quad9 => {
                    vec![vec![-1.0, -1.0], vec![1.0, -1.0], vec![1.0, 1.0], vec![-1.0, 1.0]]
                }
                _ => vec![],
            };

            for vi in &vertices {
                let mut phi = vec![0.0; n_ldofs];
                ref_elem.eval_basis(vi, &mut phi);
                let mut val = 0.0;
                for i in 0..n_ldofs {
                    val += local_dofs[i] * phi[i];
                }
                if val < global_min { global_min = val; }
                if val > global_max { global_max = val; }
            }

            // Evaluate at subdivision points within the element
            for si in 0..subdivisions {
                for sj in 0..subdivisions {
                    let xi = match elem_type {
                        ElementType::Tri3 | ElementType::Tri6 => {
                            // Barycentric subdivision for triangles
                            let a = si as f64 / subdivisions as f64;
                            let b = sj as f64 / subdivisions as f64;
                            let c = 1.0 - a - b;
                            if c < 0.0 { continue; }
                            // Convert barycentric to reference coords
                            vec![b, c]
                        }
                        ElementType::Quad4 | ElementType::Quad9 => {
                            // Tensor-product subdivision for quads
                            let x = -1.0 + 2.0 * (si as f64 + 0.5) / subdivisions as f64;
                            let y = -1.0 + 2.0 * (sj as f64 + 0.5) / subdivisions as f64;
                            vec![x, y]
                        }
                        _ => {
                            // Default: evaluate at centroid
                            vec![1.0 / 3.0, 1.0 / 3.0]
                        }
                    };

                    // Evaluate basis at subdivision point
                    let mut phi = vec![0.0; n_ldofs];
                    ref_elem.eval_basis(&xi, &mut phi);

                    // Compute u_h(xi)
                    let mut val = 0.0;
                    for i in 0..n_ldofs {
                        val += local_dofs[i] * phi[i];
                    }

                    if val < global_min { global_min = val; }
                    if val > global_max { global_max = val; }
                }
            }
        }

        (global_min, global_max)
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
        let dim = mesh.topological_dim() as usize;
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

    /// Evaluate the vector field u_h(xi) on element `elem` for vector-valued
    /// spaces (H(curl), H(div), VectorH1).
    pub fn evaluate_vector_at_element(&self, elem: u32, xi: &[f64]) -> Vec<f64> {
        let mesh = self.space.mesh();
        let edim = mesh.dim() as usize;
        let order = self.space.order();
        let stype = self.space.space_type();
        let elem_type = mesh.element_type(elem);
        let vre = crate::vector_assembler::vec_ref_elem(stype, elem_type, edim, order);
        let n_ldofs = vre.n_dofs();
        let elem_dofs = self.space.element_dofs(elem);
        let signs = self.space.element_signs(elem);
        let nodes = mesh.element_nodes(elem);
        let (jac, det_j) = simplex_jacobian(mesh, nodes, edim);
        let mut ref_vals = vec![0.0; n_ldofs * edim];
        vre.eval_basis_vec(xi, &mut ref_vals);
        if let Some(sgns) = signs {
            for i in 0..n_ldofs { for d in 0..edim { ref_vals[i * edim + d] *= sgns[i]; } }
        }
        let mut phys_vals = vec![0.0; n_ldofs * edim];
        match stype {
            fem_space::fe_space::SpaceType::HCurl => {
                piola_hcurl_basis(&jac.transpose(), &ref_vals, &mut phys_vals, n_ldofs, edim);
            }
            fem_space::fe_space::SpaceType::HDiv => {
                piola_hdiv_basis(&jac, det_j, &ref_vals, &mut phys_vals, n_ldofs, edim);
            }
            _ => { piola_hdiv_basis(&jac, det_j, &ref_vals, &mut phys_vals, n_ldofs, edim); }
        }
        let mut val = vec![0.0; edim];
        for i in 0..n_ldofs {
            let c = self.dofs[elem_dofs[i] as usize];
            for d in 0..edim { val[d] += c * phys_vals[i * edim + d]; }
        }
        val
    }
    /// Evaluate the curl of an H(curl) field at reference point `xi` on element `elem`.
    pub fn evaluate_curl_at_element(&self, elem: u32, xi: &[f64]) -> Vec<f64> {
        let mesh = self.space.mesh();
        let edim = mesh.dim() as usize;
        let order = self.space.order();
        let stype = self.space.space_type();
        let elem_type = mesh.element_type(elem);
        let vre = crate::vector_assembler::vec_ref_elem(stype, elem_type, edim, order);
        let n_ldofs = vre.n_dofs();
        let elem_dofs = self.space.element_dofs(elem);
        let signs = self.space.element_signs(elem);
        let nodes = mesh.element_nodes(elem);
        let (jac, det_j) = simplex_jacobian(mesh, nodes, edim);
        let is_surface = mesh.topological_dim() as usize != edim;
        let curl_dim = if edim == 2 || is_surface { 1 } else { 3 };
        let mut ref_curl = vec![0.0; n_ldofs * curl_dim];
        vre.eval_curl(xi, &mut ref_curl);
        if let Some(sgns) = signs {
            for i in 0..n_ldofs { for d in 0..curl_dim { ref_curl[i * curl_dim + d] *= sgns[i]; } }
        }
        let mut phys_curl = vec![0.0; n_ldofs * curl_dim];
        piola_hcurl_curl(&jac, det_j, &ref_curl, &mut phys_curl, n_ldofs, edim);
        let mut val = vec![0.0; curl_dim];
        for i in 0..n_ldofs {
            let c = self.dofs[elem_dofs[i] as usize];
            for d in 0..curl_dim { val[d] += c * phys_curl[i * curl_dim + d]; }
        }
        val
    }
    /// Evaluate the divergence of an H(div) field at reference point `xi` on element `elem`.
    pub fn evaluate_div_at_element(&self, elem: u32, xi: &[f64]) -> f64 {
        let mesh = self.space.mesh();
        let edim = mesh.dim() as usize;
        let order = self.space.order();
        let stype = self.space.space_type();
        let elem_type = mesh.element_type(elem);
        let vre = crate::vector_assembler::vec_ref_elem(stype, elem_type, edim, order);
        let n_ldofs = vre.n_dofs();
        let elem_dofs = self.space.element_dofs(elem);
        let signs = self.space.element_signs(elem);
        let nodes = mesh.element_nodes(elem);
        let (jac, det_j) = simplex_jacobian(mesh, nodes, edim);
        let mut ref_div = vec![0.0; n_ldofs];
        vre.eval_div(xi, &mut ref_div);
        if let Some(sgns) = signs { for i in 0..n_ldofs { ref_div[i] *= sgns[i]; } }
        let mut phys_div = vec![0.0; n_ldofs];
        piola_hdiv_div(det_j, &ref_div, &mut phys_div, n_ldofs);
        let mut val = 0.0;
        for i in 0..n_ldofs { val += self.dofs[elem_dofs[i] as usize] * phys_div[i]; }
        val
    }
    /// Compute per-element min/max bounds.
    pub fn get_element_bounds(&self) -> Vec<(f64, f64)> {
        let mesh = self.space.mesh();
        let n_elems = mesh.n_elements();
        let mut bounds = Vec::with_capacity(n_elems as usize);
        for e in 0..n_elems {
            let order = self.space.element_order(e as u32);
            let subdivisions = if order <= 1 { 1 } else { order as usize };
            let mut local_min = f64::INFINITY;
            let mut local_max = f64::NEG_INFINITY;
            for si in 0..subdivisions {
                for sj in 0..subdivisions {
                    let xi = vec![si as f64 / subdivisions as f64,
                                   sj as f64 / subdivisions as f64];
                    let val = self.evaluate_at_element(e as u32, &xi);
                    if val < local_min { local_min = val; }
                    if val > local_max { local_max = val; }
                }
            }
            bounds.push((local_min, local_max));
        }
        bounds
    }
    /// Extract nodal values.
    pub fn compute_l1_error(
        &self,
        exact: &dyn Fn(&[f64]) -> f64,
        quad_order: u8,
    ) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.topological_dim() as usize;
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
        let n_elems = mesh.n_elements() as u32;
        self.compute_l2_error_owned(exact, quad_order, n_elems)
    }

    /// Like [`Self::compute_l2_error`] but restricted to the first
    /// `n_owned_elems` elements of the (local) mesh.  In a partitioned
    /// setting the local mesh is `[owned | ghost]`, so passing the rank's
    /// `n_owned_elems` integrates over owned elements only — ghost elements
    /// would otherwise be counted once per rank holding them (pex36).
    pub fn compute_l2_error_owned(
        &self,
        exact: &dyn Fn(&[f64]) -> f64,
        quad_order: u8,
        n_owned_elems: u32,
    ) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.topological_dim() as usize;
        let order = self.space.order();

        let mut err2 = 0.0;

        for e in 0..n_owned_elems {
            let e = e as u32;
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let quad = ref_elem.quadrature(quad_order);

            let elem_dofs = self.space.element_dofs(e);
            let nodes = mesh.element_nodes(e);

            // Check for high-order geometry (curved surface).
            let g_order = mesh.geom_order();
            let use_ho_geo = g_order > 1;

            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let x0 = mesh.node_coords(nodes[0]);

            // Surface mesh: compute edge vectors for correct 3D coordinate mapping.
            let edim = mesh.dim() as usize;
            let is_surface = edim != dim;
            let (e1_3d, e2_3d) = if is_surface {
                let x1 = mesh.node_coords(nodes[1]);
                // Quad4 CCW: the (0,1) reference axis maps to node 3 (see
                // simplex_jacobian); simplices use node 2.
                let x2 = mesh.node_coords(nodes[if nodes.len() >= 4 { 3 } else { 2 }]);
                let mut e1 = vec![0.0; edim]; for i in 0..edim { e1[i] = x1[i] - x0[i]; }
                let mut e2 = vec![0.0; edim]; for i in 0..edim { e2[i] = x2[i] - x0[i]; }
                (e1, e2)
            } else {
                (vec![], vec![])
            };

            // High-order geometry element for curved surface integration.
            let geo_elem = if use_ho_geo {
                Some(ref_elem_vol(elem_type, g_order))
            } else {
                None
            };
            let geo_nodes = if use_ho_geo {
                mesh.geometry_nodes(e)
            } else {
                nodes
            };

            let mut phi = vec![0.0; n_ldofs];

            for (q, xi) in quad.points.iter().enumerate() {
                let (w, xp) = if use_ho_geo && is_surface {                    // Curved surface: use geometry element for metric-based area + coords
                    let ge = geo_elem.as_ref().unwrap();
                    let tdim = dim;
                    let mut grad_geo = vec![0.0; ge.n_dofs() * tdim];
                    let mut phi_geo = vec![0.0; ge.n_dofs()];
                    ge.eval_grad_basis(xi, &mut grad_geo);
                    ge.eval_basis(xi, &mut phi_geo);

                    // 3×2 Jacobian: J[i][d] = Σ_k x_k[i] · ∂φ_k/∂ξ_d
                    let mut j = vec![0.0; edim * tdim];
                    let mut xp_curved = vec![0.0; edim];
                    for k in 0..ge.n_dofs() {
                        let xk = mesh.geom_coords_of(geo_nodes[k]);
                        for i in 0..edim {
                            xp_curved[i] += phi_geo[k] * xk[i];
                            for d in 0..tdim {
                                j[i + d * edim] += xk[i] * grad_geo[k * tdim + d];
                            }
                        }
                    }
                    // Metric G = J^T·J → det(G)
                    let g00 = j[0]*j[0] + j[1]*j[1] + j[2]*j[2];
                    let g01 = j[0]*j[3] + j[1]*j[4] + j[2]*j[5];
                    let g11 = j[3]*j[3] + j[4]*j[4] + j[5]*j[5];
                    let det_g = (g00 * g11 - g01 * g01).abs();
                    let measure = det_g.sqrt();
                    (quad.weights[q] * measure, xp_curved)
                } else if use_ho_geo {
                    // Volume element with high-order geometry (curved 2D/3D
                    // body): evaluate through the geometry element, which
                    // shares the [0,1]^d reference domain with the QuadQk
                    // solution basis.
                    let ge = geo_elem.as_ref().unwrap();
                    let (jac, det_j, xp) =
                        iso_jacobian_geom(mesh, geo_nodes, ge.as_ref(), xi, dim);
                    (quad.weights[q] * det_j.abs(), xp)
                } else if is_surface {
                    let w = quad.weights[q] * det_j.abs();
                    let xp = surface_phys_coords(x0, &e1_3d, &e2_3d, xi);
                    (w, xp)
                } else {
                    let w = quad.weights[q] * det_j.abs();
                    let xp = phys_coords(x0, &jac, xi, dim);
                    (w, xp)
                };

                ref_elem.eval_basis(xi, &mut phi);

                let mut uh = 0.0;
                for i in 0..n_ldofs {
                    uh += self.dofs[elem_dofs[i] as usize] * phi[i];
                }

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
        let n_elems = mesh.n_elements() as u32;
        self.compute_h1_error_owned(exact_grad, quad_order, n_elems)
    }

    /// Like [`Self::compute_h1_error`] but restricted to the first
    /// `n_owned_elems` elements (owned-only in a partitioned setting; see
    /// [`Self::compute_l2_error_owned`]).
    pub fn compute_h1_error_owned(
        &self,
        exact_grad: &dyn Fn(&[f64]) -> Vec<f64>,
        quad_order: u8,
        n_owned_elems: u32,
    ) -> f64 {
        let mesh = self.space.mesh();
        let dim = mesh.topological_dim() as usize;
        let order = self.space.order();

        let mut err2 = 0.0;

        for e in 0..n_owned_elems {
            let e = e as u32;
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let quad = ref_elem.quadrature(quad_order);

            let elem_dofs = self.space.element_dofs(e);
            let nodes = mesh.element_nodes(e);

            // High-order geometry path (curved bodies and surfaces).
            let g_order = mesh.geom_order();
            let use_ho_geo = g_order > 1;
            let geo_elem = if use_ho_geo {
                Some(ref_elem_vol(elem_type, g_order))
            } else {
                None
            };
            let geo_nodes = if use_ho_geo {
                mesh.geometry_nodes(e)
            } else {
                nodes
            };

            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let j_inv_t = jac.clone().try_inverse().unwrap().transpose();
            let x0 = mesh.node_coords(nodes[0]);

            let mut grad_ref = vec![0.0; n_ldofs * dim];
            let mut grad_phys = vec![0.0; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let (w, j_inv_t, xp) = if use_ho_geo {
                    // Curved volume/surface: Jacobian from the geometry
                    // element (same [0,1]^d reference domain as QuadQk).
                    let ge = geo_elem.as_ref().unwrap();
                    let (jac_g, det_g, xp) =
                        iso_jacobian_geom(mesh, geo_nodes, ge.as_ref(), xi, dim);
                    let jm = DMatrix::from_fn(dim, dim, |i, d| jac_g[i + d * dim]);
                    let jit = jm.try_inverse().expect("invertible geometry Jacobian").transpose();
                    (quad.weights[q] * det_g.abs(), jit, xp)
                } else {
                    let w = quad.weights[q] * det_j.abs();
                    let xp = phys_coords(x0, &jac, xi, dim);
                    (w, j_inv_t.clone(), xp)
                };

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
        let dim = mesh.topological_dim() as usize;
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

    /// Compute element-wise L² errors between this grid function and `exact`.
    ///
    /// Returns a vector of length `n_elements()` where entry `e` is
    /// `‖u_h − u_exact‖_{L²(K_e)}`.
    ///
    /// Equivalent to MFEM's `GridFunction::ComputeElementL2Errors`.
    pub fn compute_element_l2_errors(
        &self,
        exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
        quad_order: u8,
    ) -> Vec<f64> {
        let mesh = self.space.mesh();
        let dim = mesh.topological_dim() as usize;
        let order = self.space.order();
        let ne = mesh.n_elements() as usize;
        let mut errors = vec![0.0; ne];

        for e in mesh.elem_iter() {
            let eidx = e as usize;
            let elem_type = mesh.element_type(e);
            // L2 spaces use Gauss-Legendre node Lagrange bases (MFEM
            // L2_FECollection default); QuadL2GL matches the L2Space DOF
            // ordering.  Other spaces use the H1-style reference elements.
            let ref_elem: Box<dyn ReferenceElement> =
                if self.space.space_type() == fem_space::fe_space::SpaceType::L2
                    && matches!(elem_type, ElementType::Quad4) {
                    Box::new(fem_element::lagrange::QuadL2GL::new(order as usize))
                } else {
                    ref_elem_vol(elem_type, order)
                };
            let n_ldofs = ref_elem.n_dofs();
            let quad = ref_elem.quadrature(quad_order);
            let elem_dofs = self.space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let mut phi = vec![0.0; n_ldofs];
            let mut err2 = 0.0;
            // For L2 spaces on quads, reproduce MFEM's bit-identical
            // (dof·l_x)·l_y tensor summation (left-associative) — a
            // pre-multiplied phi = lx·ly differs by 1 ulp and flips
            // threshold-edge markings in CoefficientRefiner.
            let l2gl: Option<fem_element::lagrange::QuadL2GL> =
                if self.space.space_type() == fem_space::fe_space::SpaceType::L2
                    && matches!(elem_type, ElementType::Quad4) {
                    Some(fem_element::lagrange::QuadL2GL::new(order as usize))
                } else { None };
            for (q, xi) in quad.points.iter().enumerate() {
                let (_jac, det_j, xp) = element_jacobian(mesh, e, nodes, xi, dim);
                let w = quad.weights[q] * det_j.abs();
                let uh = if let Some(ref gl) = l2gl {
                    let p = order as usize;
                    let (lx, _) = gl.eval_1d(xi[0]);
                    let (ly, _) = gl.eval_1d(xi[1]);
                    let mut s = 0.0;
                    for ix in 0..=p {
                        for iy in 0..=p {
                            let k = gl.dof_index(ix, iy);
                            s += self.dofs[elem_dofs[k] as usize] * lx[ix] * ly[iy];
                        }
                    }
                    s
                } else {
                    ref_elem.eval_basis(xi, &mut phi);
                    let mut s = 0.0;
                    for i in 0..n_ldofs {
                        s += self.dofs[elem_dofs[i] as usize] * phi[i];
                    }
                    s
                };
                let ue = exact(&xp);
                err2 += w * (uh - ue) * (uh - ue);
            }
            errors[eidx] = err2.sqrt();
        }
        errors
    }
}
/// Compute the L² norm of a coefficient function over the mesh.
///
/// Equivalent to MFEM's `ComputeLpNorm(2.0, coeff, mesh, irs)`.
pub fn compute_coeff_l2_norm(
    mesh: &impl MeshTopology,
    coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    quad_order: u8,
) -> f64 {
    let dim = mesh.topological_dim() as usize;
    let mut norm2 = 0.0;
    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        // Quad integration lives on [0,1]^2 with weight sum 1 (MFEM IntRules
        // convention); QuadQ1's quadrature is on [-1,1]^2 (weight sum 4) and
        // would give 4x the MFEM norm.  Use quad_rule_01 for quads.
        let quad = if matches!(elem_type, ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9) {
            fem_element::quadrature::quad_rule_01(quad_order)
        } else {
            ref_elem_vol(elem_type, 1).quadrature(quad_order)
        };
        let nodes = mesh.element_nodes(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (_jac, det_j, xp) = element_jacobian(mesh, e, nodes, xi, dim);
            let w = quad.weights[q] * det_j.abs();
            let fv = coeff(&xp);
            norm2 += w * fv * fv;
        }
    }
    norm2.sqrt()
}

/// Same as [`compute_coeff_l2_norm`] but restricted to the first `n_elems`
/// elements of the local mesh (`0..n_elems`).  Parallel use: a partitioned
/// mesh lists owned elements first and ghosts after, so summing this over the
/// owned range per rank and allreducing the squares gives the true global
/// norm without double-counting ghost elements.
pub fn compute_coeff_l2_norm_first_n(
    mesh: &impl MeshTopology,
    coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    quad_order: u8,
    n_elems: usize,
) -> f64 {
    let dim = mesh.topological_dim() as usize;
    let mut norm2 = 0.0;
    for e in 0..n_elems as u32 {
        let elem_type = mesh.element_type(e);
        // Quad integration lives on [0,1]^2 with weight sum 1 (MFEM IntRules
        // convention); QuadQ1's quadrature is on [-1,1]^2 (weight sum 4) and
        // would give 4x the MFEM norm.  Use quad_rule_01 for quads.
        let quad = if matches!(elem_type, ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9) {
            fem_element::quadrature::quad_rule_01(quad_order)
        } else {
            ref_elem_vol(elem_type, 1).quadrature(quad_order)
        };
        let nodes = mesh.element_nodes(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (_jac, det_j, xp) = element_jacobian(mesh, e, nodes, xi, dim);
            let w = quad.weights[q] * det_j.abs();
            let fv = coeff(&xp);
            norm2 += w * fv * fv;
        }
    }
    norm2.sqrt()
}

/// Project a vector function onto the tangential component of HCurl boundary DOFs.
///
/// For each boundary edge on a face with attribute in `bdr_attr`, evaluates
/// `coeff(x_mid).tangential` at the edge midpoint and sets the HCurl DOF value.
/// Equivalent to MFEM's `GridFunction::ProjectBdrCoefficientTangent` for ND spaces.
pub fn project_bdr_coefficient_tangent(
    nd_dofs: &mut [f64],
    nd_space: &HCurlSpace<fem_mesh::Mesh<3>>,
    coeff: &dyn Fn(&[f64], &mut [f64]),
    bdr_attr: &[i32],
) {
    use std::collections::HashSet;
    use fem_space::EdgeKey;
    let mesh = nd_space.mesh();

    // Collect boundary edges
    let mut edges: HashSet<EdgeKey> = HashSet::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        if bdr_attr.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            for i in 0..nodes.len() {
                let a = nodes[i];
                let b = nodes[(i + 1) % nodes.len()];
                edges.insert(EdgeKey::new(a, b));
            }
        }
    }
    // Project onto each edge DOF
    for ek in &edges {
        if let Some(dofs) = nd_space.edge_dofs(*ek) {
            let pa = mesh.node_coords(ek.0);
            let pb = mesh.node_coords(ek.1);
            let mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5];
            let mut fval = [0.0_f64; 3];
            coeff(&mid, &mut fval);
            // Tangential component: f · t  where t = (b-a)/|b-a|
            let tx = pb[0] - pa[0];
            let ty = pb[1] - pa[1];
            let tz = pb[2] - pa[2];
            let len = (tx*tx + ty*ty + tz*tz).sqrt();
            if len > 0.0 {
                let ft = (fval[0]*tx + fval[1]*ty + fval[2]*tz) / len;
                for &d in &dofs { nd_dofs[d as usize] = ft; }
            }
        }
    }
}

/// 2D version: project boundary edge tangent components for H(curl).
///
/// For each boundary edge on a face with attribute in `bdr_attr`, evaluates
/// `coeff(x_mid).tangential` at the edge midpoint and sets the HCurl DOF value.
/// Equivalent to MFEM's `GridFunction::ProjectBdrCoefficientTangent` for 2D ND spaces.
///
/// **Lowest-order only** (`order == 1`): the single-point (midpoint) evaluation
/// times the edge length matches MFEM's ND1 dof functional `u·(J t)`; for
/// `order > 1` each edge has multiple DOFs which would need distinct moments.
pub fn project_bdr_coefficient_tangent_2d(
    nd_dofs: &mut [f64],
    nd_space: &HCurlSpace<fem_mesh::Mesh<2>>,
    coeff: &dyn Fn(&[f64], &mut [f64]),
    bdr_attr: &[i32],
) {
    assert_eq!(nd_space.order(), 1,
        "project_bdr_coefficient_tangent_2d: ND order > 1 not supported (edge DOFs need moments)");
    use std::collections::HashSet;
    use fem_space::EdgeKey;
    let mesh = nd_space.mesh();

    // Collect boundary edges
    let mut edges: HashSet<EdgeKey> = HashSet::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        if bdr_attr.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            for i in 0..nodes.len() {
                let a = nodes[i];
                let b = nodes[(i + 1) % nodes.len()];
                edges.insert(EdgeKey::new(a, b));
            }
        }
    }
    // Project onto each edge DOF.
    // MFEM ND1 dof functional: ∫_e u·t̂ ds ≈ u(mid)·(J t) where J t is the
    // *physical* tangent vector (length = edge length), i.e. the tangential
    // component scaled by the edge length (cf. ND_R2D LocalInterpolation
    // "I_k = vshape_k · J t_k").  Do NOT normalise the tangent.
    for ek in &edges {
        if let Some(dofs) = nd_space.edge_dofs(*ek) {
            let pa = mesh.node_coords(ek.0);
            let pb = mesh.node_coords(ek.1);
            let mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];
            let mut fval = [0.0_f64; 2];
            coeff(&mid, &mut fval);
            // Tangential component scaled by edge length: f · (b-a)
            let tx = pb[0] - pa[0];
            let ty = pb[1] - pa[1];
            let ft = fval[0] * tx + fval[1] * ty;
            for &d in &dofs { nd_dofs[d as usize] = ft; }
        }
    }
}

// ─── HCurl L² projection ───────────────────────────────────────────────────

/// Project a vector function onto H(curl) via the mass-matrix solve
/// `M · u = b`, where `b_i = ∫ f(x) · φ_i(x) dx`.
///
/// Equivalent to MFEM's `GridFunction::ProjectCoefficient` for ND spaces.
/// The coefficient closure `f` receives `(x_phys, out)` and fills `out[0..dim]`.
pub fn project_hcurl_coefficient(
    nd_space: &HCurlSpace<fem_mesh::Mesh<3>>,
    coeff: &(dyn Fn(&[f64], &mut [f64]) + Send + Sync),
    quad_order: u8,
) -> Vec<f64> {
    use crate::vector_assembler::VectorAssembler;
    use crate::standard::{VectorMassIntegrator, VectorDomainLFIntegrator};
    use crate::coefficient::FnVectorCoeff;
    use fem_solver::{solve_cg, SolverConfig};

    let mass = VectorMassIntegrator { alpha: 1.0 };
    let m = VectorAssembler::assemble_bilinear(nd_space, &[&mass], quad_order);
    let src = VectorDomainLFIntegrator { f: FnVectorCoeff(coeff) };
    let rhs = VectorAssembler::assemble_linear(nd_space, &[&src], quad_order);
    let mut u = vec![0.0; nd_space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 5000, verbose: false, ..Default::default() };
    solve_cg(&m, &rhs, &mut u, &cfg).expect("HCurl L² projection CG solve");
    u
}

/// 2-D variant: project a vector function onto H(curl) for a 2-D mesh.
pub fn project_hcurl_coefficient_2d(
    nd_space: &HCurlSpace<fem_mesh::Mesh<2>>,
    coeff: &(dyn Fn(&[f64], &mut [f64]) + Send + Sync),
    quad_order: u8,
) -> Vec<f64> {
    use crate::vector_assembler::VectorAssembler;
    use crate::standard::{VectorMassIntegrator, VectorDomainLFIntegrator};
    use crate::coefficient::FnVectorCoeff;
    use fem_solver::{solve_cg, SolverConfig};

    let mass = VectorMassIntegrator { alpha: 1.0 };
    let m = VectorAssembler::assemble_bilinear(nd_space, &[&mass], quad_order);
    let src = VectorDomainLFIntegrator { f: FnVectorCoeff(coeff) };
    let rhs = VectorAssembler::assemble_linear(nd_space, &[&src], quad_order);
    let mut u = vec![0.0; nd_space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 5000, verbose: false, ..Default::default() };
    solve_cg(&m, &rhs, &mut u, &cfg).expect("HCurl 2-D L² projection CG solve");
    u
}

/// Project a vector function onto H(div) for a 2-D mesh.
///
/// Computes the L² projection by solving `M · u = b` where `M` is the
/// H(div) mass matrix and `b_i = ∫ f(x) · φ_i(x) dx`.
///
/// Equivalent to MFEM's `GridFunction::ProjectCoefficient` for RT spaces.
pub fn project_hdiv_coefficient_2d(
    rt_space: &HDivSpace<fem_mesh::Mesh<2>>,
    coeff: &(dyn Fn(&[f64], &mut [f64]) + Send + Sync),
    quad_order: u8,
) -> Vec<f64> {
    use crate::vector_assembler::VectorAssembler;
    use crate::standard::{VectorMassIntegrator, VectorDomainLFIntegrator};
    use crate::coefficient::FnVectorCoeff;
    use fem_solver::{solve_cg, SolverConfig};

    let mass = VectorMassIntegrator { alpha: 1.0 };
    let m = VectorAssembler::assemble_bilinear(rt_space, &[&mass], quad_order);
    let src = VectorDomainLFIntegrator { f: FnVectorCoeff(coeff) };
    let rhs = VectorAssembler::assemble_linear(rt_space, &[&src], quad_order);
    let mut u = vec![0.0; rt_space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 5000, verbose: false, ..Default::default() };
    solve_cg(&m, &rhs, &mut u, &cfg).expect("HDiv 2-D L² projection CG solve");
    u
}

/// Project a vector function onto H(div) for a 3-D mesh.
pub fn project_hdiv_coefficient_3d(
    rt_space: &HDivSpace<fem_mesh::Mesh<3>>,
    coeff: &(dyn Fn(&[f64], &mut [f64]) + Send + Sync),
    quad_order: u8,
) -> Vec<f64> {
    use crate::vector_assembler::VectorAssembler;
    use crate::standard::{VectorMassIntegrator, VectorDomainLFIntegrator};
    use crate::coefficient::FnVectorCoeff;
    use fem_solver::{solve_cg, SolverConfig};

    let mass = VectorMassIntegrator { alpha: 1.0 };
    let m = VectorAssembler::assemble_bilinear(rt_space, &[&mass], quad_order);
    let src = VectorDomainLFIntegrator { f: FnVectorCoeff(coeff) };
    let rhs = VectorAssembler::assemble_linear(rt_space, &[&src], quad_order);
    let mut u = vec![0.0; rt_space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 5000, verbose: false, ..Default::default() };
    solve_cg(&m, &rhs, &mut u, &cfg).expect("HDiv 3-D L² projection CG solve");
    u
}

/// Compute the L² norm of a vector field given its DOF values and the mass matrix.
///
/// ‖u‖_{L²} = sqrt(u^T M u)
///
/// where `M` is the mass matrix and `u` is the vector of DOF coefficients.
///
/// # Example
/// ```ignore
/// let mass = Assembler::assemble_bilinear(&space, &[&VectorH1MassIntegrator { kappa: 1.0 }], quad_order);
/// let l2_norm = vector_l2_norm(&mass, &dofs);
/// ```
pub fn vector_l2_norm(mass: &CsrMatrix<f64>, dofs: &[f64]) -> f64 {
    let mut m_u = vec![0.0; dofs.len()];
    mass.spmv(dofs, &mut m_u);
    let dot: f64 = dofs.iter().zip(m_u.iter()).map(|(a, b)| a * b).sum();
    dot.max(0.0).sqrt()
}

// ─── H(curl) L² error ────────────────────────────────────────────────────

/// Compute the L² error of an H(curl) field against an exact vector field.
///
/// ‖u_h − u_exact‖_{L²} = (∫_Ω |u_h(x) − u_exact(x)|² dx)^{1/2}
///
/// Supports 2D and 3D meshes with affine and isoparametric geometry.
///
/// Elements for which `exclude_elems[e]` is `true` are skipped (e.g. PML
/// region elements). Pass `None` to include all elements.
pub fn compute_l2_error_hcurl<M: MeshTopology>(
    dofs: &[f64],
    nd_space: &HCurlSpace<M>,
    exact: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    quad_order: u8,
    exclude_elems: Option<&[bool]>,
) -> f64 {
    let mesh = nd_space.mesh();
    let dim = mesh.topological_dim() as usize;
    let mut err2 = 0.0;

    for e in mesh.elem_iter() {
        // Skip excluded elements (e.g. PML region)
        if let Some(mask) = exclude_elems {
            if e as usize >= mask.len() || mask[e as usize] { continue; }
        }
        let et = mesh.element_type(e);
        let vre = crate::vector_assembler::vec_ref_elem(
            fem_space::fe_space::SpaceType::HCurl, et, dim, nd_space.order());
        let n_ldofs = vre.n_dofs();
        let quad = vre.quadrature(quad_order);
        let elem_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = nd_space.element_signs(e);
        let mut ref_bv = vec![0.0; n_ldofs * dim];
        let nodes = mesh.element_nodes(e);
        let use_iso = matches!(et,
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
            | ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27
            | ElementType::Prism6 | ElementType::Prism15
            | ElementType::Pyramid5);
        let geo_elem = if use_iso { crate::geo_ref_elem_from_mesh(mesh, e) } else { None };

        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, jac, xp) = if use_iso {
                let ge = geo_elem.as_ref().unwrap();
                let (jac, det, xp_vec) = crate::isoparametric_jacobian(mesh, &nodes, ge.as_ref(), xi, dim);
                (quad.weights[qi] * det.abs(), jac, xp_vec)
            } else {
                let (jac, xp_vec) = element_jacobian_at(mesh, e, xi, dim);
                (quad.weights[qi] * jac.determinant().abs(), jac, xp_vec)
            };

            let jac_inv_t = jac.try_inverse().unwrap_or_else(|| DMatrix::<f64>::identity(dim, dim)).transpose();

            vre.eval_basis_vec(xi, &mut ref_bv);

            // HCurl covariant Piola: φ_phys = J^{-T} · φ̂_ref
            let mut uh = vec![0.0; dim];
            for i in 0..n_ldofs {
                let s = signs[i] as f64;
                let coeff = dofs[elem_dofs[i]];
                for c in 0..dim {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += jac_inv_t[(c, k)] * ref_bv[i * dim + k];
                    }
                    uh[c] += s * coeff * sum;
                }
            }

            let ex = exact(&xp);
            for c in 0..dim {
                let d = uh[c] - ex[c];
                err2 += w * d * d;
            }
        }
    }
    err2.max(0.0).sqrt()
}

// ─── H(div) L² error ─────────────────────────────────────────────────────

/// Compute the L² error of an H(div) field against an exact vector field.
///
/// ‖w_h − w_exact‖_{L²} = (∫_Ω |w_h(x) − w_exact(x)|² dx)^{1/2}
///
/// Uses the contravariant Piola transform: ψ_phys = (1/det(J)) · J · ψ̂_ref
pub fn compute_l2_error_hdiv<M: MeshTopology>(
    dofs: &[f64],
    rt_space: &HDivSpace<M>,
    exact: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    quad_order: u8,
    exclude_elems: Option<&[bool]>,
) -> f64 {
    let mesh = rt_space.mesh();
    let dim = mesh.topological_dim() as usize;
    let mut err2 = 0.0;

    for e in mesh.elem_iter() {
        // Skip excluded elements (e.g. ghost elements in a parallel run).
        if let Some(mask) = exclude_elems {
            if e as usize >= mask.len() || mask[e as usize] { continue; }
        }
        let et = mesh.element_type(e);
        let vre = crate::vector_assembler::vec_ref_elem(
            fem_space::fe_space::SpaceType::HDiv, et, dim, rt_space.order());
        let n_ldofs = vre.n_dofs();
        let quad = vre.quadrature(quad_order);
        let elem_dofs: Vec<usize> = rt_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = rt_space.element_signs(e);
        let mut ref_bv = vec![0.0; n_ldofs * dim];
        let nodes = mesh.element_nodes(e);
        let use_iso = matches!(et,
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
            | ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27
            | ElementType::Prism6 | ElementType::Prism15
            | ElementType::Pyramid5);
        let geo_elem = if use_iso { crate::geo_ref_elem_from_mesh(mesh, e) } else { None };

        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, jac, xp) = if use_iso {
                let ge = geo_elem.as_ref().unwrap();
                let (jac, det, xp_vec) = crate::isoparametric_jacobian(mesh, &nodes, ge.as_ref(), xi, dim);
                (quad.weights[qi] * det.abs(), jac, xp_vec)
            } else {
                let (jac, xp_vec) = element_jacobian_at(mesh, e, xi, dim);
                (quad.weights[qi] * jac.determinant().abs(), jac, xp_vec)
            };

            vre.eval_basis_vec(xi, &mut ref_bv);

            // H(div) contravariant Piola: ψ_phys = (1/det(J)) · J · ψ̂_ref
            let det_j = jac.determinant();
            let inv_det = if det_j.abs() > 1e-80 { 1.0 / det_j } else { 0.0 };

            let mut uh = vec![0.0; dim];
            for i in 0..n_ldofs {
                let s = signs[i] as f64;
                let coeff = dofs[elem_dofs[i]];
                for c in 0..dim {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += jac[(c, k)] * ref_bv[i * dim + k];
                    }
                    uh[c] += s * coeff * inv_det * sum;
                }
            }

            let ex = exact(&xp);
            for c in 0..dim {
                let d = uh[c] - ex[c];
                err2 += w * d * d;
            }
        }
    }
    err2.max(0.0).sqrt()
}

// ─── L2 scalar L² error (P0-aware) ───────────────────────────────────────

/// Compute the L² error of a scalar L² field against an exact scalar function.
///
/// Supports P0 (constant per element) and higher-order L2 spaces.
/// For P0, uses the HDiv reference element's quadrature rule.
pub fn compute_l2_error_l2<M: MeshTopology>(
    dofs: &[f64],
    l2_space: &L2Space<M>,
    exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    quad_order: u8,
    exclude_elems: Option<&[bool]>,
) -> f64 {
    let mesh = l2_space.mesh();
    let dim = mesh.topological_dim() as usize;
    let order = l2_space.order();
    let mut err2 = 0.0;

    for e in mesh.elem_iter() {
        // Skip excluded elements (e.g. ghost elements in a parallel run).
        if let Some(mask) = exclude_elems {
            if e as usize >= mask.len() || mask[e as usize] { continue; }
        }
        let et = mesh.element_type(e);
        let nodes = mesh.element_nodes(e);
        let use_iso = matches!(et,
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
            | ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27
            | ElementType::Prism6 | ElementType::Prism15
            | ElementType::Pyramid5);
        let geo_elem = if use_iso { crate::geo_ref_elem_from_mesh(mesh, e) } else { None };

        let (quad, n_ldofs, use_lagrange) = if order == 0 {
            // P0: use HDiv reference element's quadrature
            let vre = vec_ref_elem(VecFamily::RaviartThomas, et.to_elem_type(), 0);
            (vre.quadrature(quad_order), 1usize, false)
        } else {
            let re = et.ref_elem(order);
            (re.quadrature(quad_order), re.n_dofs(), true)
        };

        let elem_dofs: Vec<usize> = l2_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let mut phi_buf = if use_lagrange { vec![0.0; n_ldofs] } else { vec![] };

        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, xp) = if use_iso {
                let ge = geo_elem.as_ref().unwrap();
                let (jac, det, xp_vec) = crate::isoparametric_jacobian(mesh, &nodes, ge.as_ref(), xi, dim);
                (quad.weights[qi] * det.abs(), xp_vec)
            } else {
                let (jac, xp_vec) = element_jacobian_at(mesh, e, xi, dim);
                (quad.weights[qi] * jac.determinant().abs(), xp_vec)
            };

            let uh = if order == 0 {
                dofs[elem_dofs[0]]  // P0: single constant per element
            } else {
                let re = et.ref_elem(order);
                re.eval_basis(xi, &mut phi_buf);
                let mut val = 0.0;
                for i in 0..n_ldofs {
                    val += dofs[elem_dofs[i]] * phi_buf[i];
                }
                val
            };

            let ue = exact(&xp);
            err2 += w * (uh - ue) * (uh - ue);
        }
    }
    err2.max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::GridFunction;
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

    /// `compute_l2_error_hcurl` must reconstruct the FE field with the SAME
    /// reference basis as the assembly, otherwise ‖E_h‖_L² disagrees with the
    /// mass-matrix norm uᵀMu. Regression test for ex25: the error routine
    /// previously used `fem_element::vec_ref_elem` (TriNDk Vandermonde basis)
    /// instead of the assembly's `vector_assembler::vec_ref_elem` (TriND1
    /// Whitney basis), inflating the reported L² error (0.817 vs C++ 0.144).
    #[test]
    fn hcurl_l2_norm_matches_mass_matrix() {
        use crate::standard::VectorMassIntegrator;
        use crate::vector_assembler::VectorAssembler;
        use super::compute_l2_error_hcurl;
        use fem_space::HCurlSpace;

        let mesh = Mesh::<2>::unit_square_tri(6);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();
        let q = 3u8;

        // Mass matrix via the assembly path (same reference basis as the
        // system matrix that defines the solution DOFs).
        let m = VectorAssembler::assemble_bilinear(
            &space, &[&VectorMassIntegrator { alpha: 1.0 }], q);

        // Deterministic pseudo-random DOF vector with nonzero curl content.
        let u: Vec<f64> = (0..n).map(|i| {
            let s = (i as f64 + 0.5) * 1.7;
            s.sin() + 0.3 * (i as f64).cos()
        }).collect();

        // ‖E_h‖_L² via the error routine (exact field = 0).
        let zero = |_xp: &[f64]| -> Vec<f64> { vec![0.0; 2] };
        let norm_err = compute_l2_error_hcurl(&u, &space, &zero, q, None);

        // ‖E_h‖_L² via the mass matrix: sqrt(uᵀ M u).
        let mut mu = vec![0.0; n];
        m.spmv(&u, &mut mu);
        let norm_mass: f64 = u.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>().sqrt();

        assert!((norm_err - norm_mass).abs() < 1e-9 * norm_mass.max(1e-12),
            "hcurl L2 norm mismatch: error-routine {norm_err:.10e} vs mass-matrix {norm_mass:.10e}");
    }

    /// Regression for ex33 (1:1 MFEM comparison): `compute_l2_error` on a
    /// Quad4 mesh must use the same [0,1]² reference element (QuadQk) as the
    /// assembler and map the (0,1) reference axis through node 3 (CCW
    /// ordering).  The legacy QuadQ1/Q2 ([-1,1]²) mixed with the [0,1]²
    /// quadrature rules, and the node-2 axis mapping, produced an L2 error of
    /// ~0.0566 for a solution that is pointwise correct to 1e-6.
    #[test]
    fn quad4_l2_error_is_consistent() {
        use crate::GridFunction;
        use std::f64::consts::PI;

        // 8×8 quad mesh on [0,1]² (same discretization as ex33's
        // inline-quad verification configuration).
        let mut mesh = Mesh::<2>::unit_square_quad(8);
        for _ in 0..2 {
            mesh = fem_mesh::refine_uniform(&mesh);
        }
        let space = H1Space::new(mesh, 2);
        let sol = |x: &[f64]| -> f64 {
            let mut v = 1.0;
            for &xi in x {
                v *= (PI * xi).sin();
            }
            v
        };
        let dofs = space.interpolate(&sol);
        let gf = GridFunction::new(&space, dofs.as_slice().to_vec());

        // ‖sol‖_L² = 1/2 on the unit square (independent of interpolation;
        // quadrature(7) = 4×4 Gauss integrates the sine to ~1e-7).
        let ex_norm = gf.compute_l2_error(&|_| 0.0, 7);
        assert!((ex_norm - 0.5).abs() < 1e-5,
            "‖sin(πx)sin(πy)‖_L² = {ex_norm:.10e}, expected 0.5");

        // Q2 interpolation error of the smooth function on h=1/8 is O(h³),
        // far below the 0.05+ error the old [-1,1]²/node-2 paths produced.
        let l2 = gf.compute_l2_error(&sol, 7);
        assert!(l2 < 1e-3, "Quad4 L2 error too large: {l2:.6e}");
    }

    #[test]
    fn get_bounds_linear_element() {
        // For linear elements, get_bounds should return exact min/max (at vertices).
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        // f(x,y) = x + y on [0,1]²: min = 0 (at (0,0)), max = 2 (at (1,1))
        let f = |x: &[f64]| x[0] + x[1];
        let dofs_vec = space.interpolate(&f);
        let gf = GridFunction::new(&space, dofs_vec.into_vec());
        let (u_min, u_max) = gf.get_bounds();
        assert!((u_min - 0.0).abs() < 1e-10, "min = {u_min}, expected 0.0");
        assert!((u_max - 2.0).abs() < 1e-10, "max = {u_max}, expected 2.0");
    }

    #[test]
    fn get_bounds_quadratic_element() {
        // For quadratic elements, get_bounds returns approximate bounds.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 2);
        // f(x,y) = x² + y² on [0,1]²: min = 0, max = 2
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let gf = GridFunction::from_projection(&space, &f, 4);
        let (u_min, u_max) = gf.get_bounds();
        // Bounds should contain the true range [0, 2]
        assert!(u_min <= 0.01, "min = {u_min}, should be near 0");
        assert!(u_max >= 1.99, "max = {u_max}, should be near 2");
        assert!(u_min <= u_max, "min should be <= max");
    }

    use super::project_grid_function;

    #[test]
    fn project_grid_function_identity() {
        // Projecting a P1 function onto the same space should give back the same function
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh.clone(), 1);
        let f = |x: &[f64]| x[0] + x[1];
        let gf = GridFunction::from_projection(&space, &f, 3);

        // Project onto the same space
        let gf_proj = project_grid_function(&gf, &space, 3);

        // Should be close to the original
        let l2_err = gf_proj.compute_l2_error(&f, 3);
        assert!(l2_err < 1e-10, "projection error = {l2_err}, expected < 1e-10");
    }
}
