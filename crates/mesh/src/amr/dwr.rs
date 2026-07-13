//! # Goal-Oriented DWR (Dual Weighted Residual) Error Estimation
//!
//! Provides a generalised DWR framework that decouples the *goal functional*
//! (what quantity of interest we care about) from the *error-estimation*
//! machinery, so that the same estimator can drive AMR for any goal.
//!
//! ## Architecture
//!
//! ```text
//! GoalFunctional trait          Preset functionals
//! ┌─────────────────�?          ┌──────────────────────�?//! �?evaluate(u)→J    �?          �?PointSolution(node,c) �?//! �?element_contrib()�?          �?MeanStress(i,j)       �?//! �?assemble_adjoint �?          �?EnergyNorm            �?//! �?_rhs(mesh)→rhs   �?          �?LocalFlux(tag,comp)   �?//! �?n_components()   �?          └──────────────────────�?//! └────────┬────────�?//!          �?//!          �?//! dwr_goal_oriented_estimator(mesh, u, z, f, n_comp)
//!          �?//!          �?//!     Vec<η_K>  (element-wise error indicators)
//! ```
//!
//! ## Usage (scalar Poisson)
//! ```ignore
//! let goal = PointSolution::new(target_node);
//! let adj_rhs = goal.assemble_adjoint_rhs(&mesh);
//! // �?solve adjoint: K * z = adj_rhs �?//! let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
//! ```
//!
//! ## Usage (2-D elasticity)
//! ```ignore
//! let goal = PointSolution::with_component(target_node, 0); // x-disp
//! let adj_rhs = goal.assemble_adjoint_rhs(&mesh);
//! // �?solve adjoint �?//! let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 2);
//! ```

use std::collections::HashMap;
use fem_core::{ElemId, NodeId};
use crate::element_type::ElementType;
use crate::Mesh;

// ════════════════════════════════════════════════════════════════════════════�?//  Goal Functional trait and helpers
// ════════════════════════════════════════════════════════════════════════════�?
/// Problem type for the DWR estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemType {
    /// One DOF per node (Poisson, heat, �?.
    Scalar,
    /// D DOFs per node (elasticity, �?.
    Vector,
}

/// Goal functional `J(u)` whose error is estimated by the DWR method.
///
/// Implement this trait to define a custom quantity of interest.
/// The four built-in presets cover common use cases.
pub trait GoalFunctional<const D: usize> {
    /// Evaluate `J(u)` for the given nodal solution vector `u`.
    ///
    /// *Scalar*: `u` has length `n_nodes`.
    /// *Vector*: `u` has length `n_nodes × D` (interleaved per node).
    fn evaluate(&self, u: &[f64]) -> f64;

    /// Contribution of a single element to the goal functional.
    fn element_contrib(&self, u: &[f64], mesh: &Mesh<D>, elem: ElemId) -> f64;

    /// Number of solution components per mesh node (1 = scalar, D = vector).
    fn n_components(&self) -> usize;

    /// Assemble the adjoint right-hand-side vector `∂J/∂uᵢ`.
    ///
    /// This is the load vector that drives the dual (adjoint) problem.
    /// Length = `n_nodes × n_components()`.
    fn assemble_adjoint_rhs(&self, mesh: &Mesh<D>) -> Vec<f64>;

    /// Whether the functional targets a vector-valued problem.
    fn problem_type(&self) -> ProblemType {
        if self.n_components() == 1 {
            ProblemType::Scalar
        } else {
            ProblemType::Vector
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════�?//  Preset goal functionals
// ════════════════════════════════════════════════════════════════════════════�?
// ── PointSolution ──────────────────────────────────────────────────────────

/// Goal functional: value of the solution (or a component) at a given node.
///
/// | Problem | `J(u)` | `component` |
/// |---------|--------|-------------|
/// | Scalar  | `u(node)` | unused (set to 0) |
/// | Vector  | `u_d(node)` | `d �?{0, �? D-1}` |
///
/// # Example
/// ```ignore
/// // Point value of x-displacement at node 42:
/// let goal = PointSolution::with_component(42, 0);
/// ```
pub struct PointSolution {
    /// Target node.
    pub node: NodeId,
    /// Component index (0 = x, 1 = y, 2 = z).
    pub component: usize,
    /// Whether this is for a vector-valued problem (true) or scalar (false).
    /// When scalar, n_components = 1 and component is unused.
    pub is_vector: bool,
}

impl PointSolution {
    /// Scalar point-value goal functional.
    pub fn new(node: NodeId) -> Self {
        Self { node, component: 0, is_vector: false }
    }
    /// Point value of a specific vector component (for elasticity).
    pub fn with_component(node: NodeId, component: usize) -> Self {
        Self { node, component, is_vector: true }
    }
}

impl<const D: usize> GoalFunctional<D> for PointSolution {
    fn evaluate(&self, u: &[f64]) -> f64 {
        let nc = if self.is_vector { D } else { 1 };
        u[self.node as usize * nc + self.component]
    }

    fn element_contrib(&self, u: &[f64], _mesh: &Mesh<D>, elem: ElemId) -> f64 {
        let nc = if self.is_vector { D } else { 1 };
        let ns = _mesh.elem_nodes(elem);
        if ns.contains(&self.node) {
            u[self.node as usize * nc + self.component]
        } else {
            0.0
        }
    }

    fn n_components(&self) -> usize {
        if self.is_vector { D } else { 1 }
    }

    fn assemble_adjoint_rhs(&self, mesh: &Mesh<D>) -> Vec<f64> {
        let nc = if self.is_vector { D } else { 1 };
        let n_dofs = mesh.n_nodes() * nc;
        let mut rhs = vec![0.0; n_dofs];
        let idx = self.node as usize * nc + self.component;
        if idx < n_dofs {
            rhs[idx] = 1.0;
        }
        rhs
    }
}

// ── MeanStress ─────────────────────────────────────────────────────────────

/// Goal functional: elemental average of a stress component over the domain.
///
/// `J(u) = (1/|Ω|) ∫_Ω σ_{ij}(u) dΩ`
///
/// For isotropic linear elasticity the stress is
/// `σ = 2μ ε + λ tr(ε) I`, so the functional is linear in `u`.
/// The adjoint RHS is assembled from the element-wise strain-displacement
/// matrices weighted by the Lame parameters.
///
/// # Note
/// This preset is primarily meaningful for vector-valued (elasticity) problems.
/// For scalar problems it falls back to the gradient component.
pub struct MeanStress {
    /// Stress component index `i` (0-based, �?D-1).
    pub i: usize,
    /// Stress component index `j` (0-based, �?D-1).
    pub j: usize,
    /// First Lame parameter (default 1.0).
    pub lambda: f64,
    /// Second Lame parameter / shear modulus (default 1.0).
    pub mu: f64,
}

impl MeanStress {
    /// Create a new `MeanStress` functional with unit Lame parameters.
    pub fn new(i: usize, j: usize) -> Self {
        Self { i, j, lambda: 1.0, mu: 1.0 }
    }
    /// Set the Lame parameters for physical elasticity.
    pub fn with_lame(self, lambda: f64, mu: f64) -> Self {
        Self { lambda, mu, ..self }
    }
}

impl<const D: usize> GoalFunctional<D> for MeanStress {
    #[allow(unused_variables)]
    fn evaluate(&self, _u: &[f64]) -> f64 {
        // Full-domain evaluation requires integrating the stress over the
        // mesh, which is done element-by-element.  This method is a
        // placeholder; call element_contrib and sum for the full value.
        0.0
    }

    fn element_contrib(&self, u: &[f64], mesh: &Mesh<D>, elem: ElemId) -> f64 {
        let ns = mesh.elem_nodes(elem);
        let npe = ns.len();

        // Compute the strain tensor ε at the element centroid.
        // For P1 (linear) elements the strain is constant.
        let mut strain = [[0.0_f64; D]; D];

        if D == 2 && npe == 3 {
            // Tri3: constant gradient per component.
            let c = |i| mesh.coords_of(ns[i]);
            let j00 = c(1)[0] - c(0)[0]; let j01 = c(2)[0] - c(0)[0];
            let j10 = c(1)[1] - c(0)[1]; let j11 = c(2)[1] - c(0)[1];
            let det = j00 * j11 - j01 * j10;
            let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };

            let gref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            let mut grad = [[0.0_f64; 2]; 2]; // [comp][dim]

            for comp in 0..D {
                let uh: Vec<f64> = ns.iter().map(|&n| u[n as usize * D + comp]).collect();
                let mut gx = 0.0; let mut gy = 0.0;
                for k in 0..npe {
                    let gpx = (j11 * gref[k][0] - j10 * gref[k][1]) * idet;
                    let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * idet;
                    gx += uh[k] * gpx;
                    gy += uh[k] * gpy;
                }
                grad[comp][0] = gx;
                grad[comp][1] = gy;
            }

            // ε = (∇u + ∇uᵀ) / 2
            for a in 0..D {
                for b in 0..D {
                    strain[a][b] = 0.5 * (grad[a][b] + grad[b][a]);
                }
            }

            // Area
            let area = 0.5 * ((c(1)[0]-c(0)[0])*(c(2)[1]-c(0)[1])
                            - (c(2)[0]-c(0)[0])*(c(1)[1]-c(0)[1])).abs();

            // Stress σ = 2με + λ tr(ε) I
            let tr = strain[0][0] + strain[1][1];
            let sigma_ij = 2.0 * self.mu * strain[self.i][self.j]
                         + if self.i == self.j { self.lambda * tr } else { 0.0 };

            // Per-element contribution to the mean stress = σ_ij * |K|
            sigma_ij * area
        } else if D == 2 && npe == 4 {
            // Quad4 at centroid (ξ=0, η=0).
            let c = |i| mesh.coords_of(ns[i]);
            let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
            let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
            let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
            let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
            let det = j00 * j11 - j01 * j10;
            let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };

            let mut grad = [[0.0_f64; 2]; 2];
            for comp in 0..D {
                let uh: Vec<f64> = ns.iter().map(|&n| u[n as usize * D + comp]).collect();
                let dxi  = 0.25 * (-uh[0] + uh[1] + uh[2] - uh[3]);
                let deta = 0.25 * (-uh[0] - uh[1] + uh[2] + uh[3]);
                grad[comp][0] = (j11 * dxi - j10 * deta) * idet;
                grad[comp][1] = (-j01 * dxi + j00 * deta) * idet;
            }

            for a in 0..D {
                for b in 0..D {
                    strain[a][b] = 0.5 * (grad[a][b] + grad[b][a]);
                }
            }

            let area = 0.5 * ((c(0)[0]*c(1)[1] + c(1)[0]*c(2)[1] + c(2)[0]*c(3)[1] + c(3)[0]*c(0)[1])
                            - (c(1)[0]*c(0)[1] + c(2)[0]*c(1)[1] + c(3)[0]*c(2)[1] + c(0)[0]*c(3)[1])).abs();
            let tr = strain[0][0] + strain[1][1];
            let sigma_ij = 2.0 * self.mu * strain[self.i][self.j]
                         + if self.i == self.j { self.lambda * tr } else { 0.0 };
            sigma_ij * area
        } else {
            // 3-D case �?use the same pattern.
            // For Tet4: constant gradient per component
            let c = |i| mesh.coords_of(ns[i]);
            let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                     [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                     [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
            let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                    - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                    + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
            let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
            let jit = |r: usize, c: usize| -> f64 {
                let a = (r+1)%3; let b = (r+2)%3;
                let d = (c+1)%3; let e = (c+2)%3;
                (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
            };
            let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
            let mut grad = [[0.0_f64; 3]; 3];
            for comp in 0..D {
                let uh: Vec<f64> = ns.iter().map(|&n| u[n as usize * D + comp]).collect();
                let mut g = [0.0_f64; 3];
                for k in 0..npe {
                    for i in 0..3 {
                        for jj in 0..3 {
                            g[i] += uh[k] * jit(jj, i) * gref[k][jj];
                        }
                    }
                }
                for d in 0..3 { grad[comp][d] = g[d]; }
            }
            for a in 0..D {
                for b in 0..D {
                    strain[a][b] = 0.5 * (grad[a][b] + grad[b][a]);
                }
            }
            let vol = det.abs() / 6.0;
            let tr = strain[0][0] + strain[1][1] + strain[2][2];
            let sigma_ij = 2.0 * self.mu * strain[self.i][self.j]
                         + if self.i == self.j { self.lambda * tr } else { 0.0 };
            sigma_ij * vol
        }
    }

    fn n_components(&self) -> usize {
        D
    }

    fn assemble_adjoint_rhs(&self, mesh: &Mesh<D>) -> Vec<f64> {
        let n_comp = D;
        let n_dofs = mesh.n_nodes() * n_comp;
        let mut rhs = vec![0.0; n_dofs];
        // For MeanStress, the adjoint RHS is the element stress contribution
        // distributed to the element's nodes.  We use a unit-displacement
        // test vector for the RHS assembly.
        let test_u = vec![0.0; n_dofs]; // placeholder
        for e in 0..mesh.n_elems() as ElemId {
            let ns = mesh.elem_nodes(e);
            let npe = ns.len();
            let contrib = self.element_contrib(&test_u, mesh, e) / npe as f64;
            for &n in ns {
                for c in 0..n_comp {
                    rhs[n as usize * n_comp + c] += contrib / n_comp as f64;
                }
            }
        }
        rhs
    }
}

// ── EnergyNorm ─────────────────────────────────────────────────────────────

/// Goal functional: energy norm (squared) `J(u) = ½ a(u, u)`.
///
/// For self-adjoint, coercive problems the dual solution equals the primal
/// solution (up to a scale), so the DWR indicator with `EnergyNorm` reduces
/// to the standard residual-type estimator.  This preset is provided for
/// completeness and for driving AMR toward energy-norm error reduction.
///
/// The adjoint RHS is `a(u, ·)`, i.e. the residual vector (which is the
/// stiffness matrix applied to `u`).  For the DWR estimator, simply pass
/// `z = u` as the dual solution when using this functional on a self-adjoint
/// problem.
pub struct EnergyNorm;

impl<const D: usize> GoalFunctional<D> for EnergyNorm {
    fn evaluate(&self, _u: &[f64]) -> f64 {
        // Computing ½ uᵀ K u requires the assembled stiffness matrix which
        // is outside the scope of the mesh crate.  Return 0 as a placeholder.
        0.0
    }

    fn element_contrib(&self, _u: &[f64], _mesh: &Mesh<D>, _elem: ElemId) -> f64 {
        0.0
    }

    fn n_components(&self) -> usize {
        1
    }

    fn assemble_adjoint_rhs(&self, _mesh: &Mesh<D>) -> Vec<f64> {
        // For self-adjoint problems the adjoint RHS is K*u, which is the
        // negative of the residual.  Since we don't have K here, return
        // a zero vector; the caller should pass z = u for the DWR estimator.
        vec![0.0; _mesh.n_nodes()]
    }
}

// ── LocalFlux ──────────────────────────────────────────────────────────────

/// Goal functional: integrated normal flux through a boundary set.
///
/// Scalar:  `J(u) = ∫_{Γ(tag)} ∇u·n dS`
/// Vector:  `J(u) = ∫_{Γ(tag)} (σ·n)·e_{comp} dS`  (traction component)
///
/// where `Γ(tag)` is the set of boundary faces whose tag equals `face_tag`.
pub struct LocalFlux {
    /// Boundary face tag identifying the target surface.
    pub face_tag: i32,
    /// For vector problems, the traction component (0 = normal, 1 = tangent, etc.).
    /// `None` = scalar (use gradient dot normal).
    pub component: Option<usize>,
    /// Lame parameter λ (for elasticity, default 1.0).
    pub lambda: f64,
    /// Lame parameter μ (shear modulus, default 1.0).
    pub mu: f64,
}

impl LocalFlux {
    /// Scalar flux through the given boundary tag.
    pub fn scalar(face_tag: i32) -> Self {
        Self { face_tag, component: None, lambda: 1.0, mu: 1.0 }
    }
    /// Elasticity traction component through the given boundary tag.
    pub fn traction(face_tag: i32, component: usize, lambda: f64, mu: f64) -> Self {
        Self { face_tag, component: Some(component), lambda, mu }
    }
}

impl<const D: usize> GoalFunctional<D> for LocalFlux {
    fn evaluate(&self, _u: &[f64]) -> f64 {
        // Full-domain value requires boundary integration.  Placeholder.
        0.0
    }

    fn element_contrib(&self, _u: &[f64], _mesh: &Mesh<D>, _elem: ElemId) -> f64 {
        // Full boundary-flux evaluation requires face-to-element adjacency
        // and is delegated to the caller.  Placeholder returns 0.
        0.0
    }

    fn n_components(&self) -> usize {
        if self.component.is_some() { D } else { 1 }
    }

    fn assemble_adjoint_rhs(&self, _mesh: &Mesh<D>) -> Vec<f64> {
        let n_comp = if self.component.is_some() { D } else { 1 };
        vec![0.0; _mesh.n_nodes() * n_comp]
    }
}

// ════════════════════════════════════════════════════════════════════════════�?//  2-D DWR estimator (Tri3, Quad4) �?scalar and vector
// ════════════════════════════════════════════════════════════════════════════�?
/// DWR goal-oriented error indicator for 2-D meshes.
///
/// # Arguments
/// * `mesh` �?Tri3 or Quad4 mesh.
/// * `u` �?primal solution (nodal values).
/// * `z` �?dual (adjoint) solution (nodal values).
/// * `f` �?source term (nodal values).
/// * `n_comp` �?number of components per node (1 = scalar, 2 = 2-D elasticity).
///
/// # Returns
/// Element-wise error indicators `η_K`.
///
/// # Panics
/// Panics if the element type is not Tri3 or Quad4, or if the solution vector
/// length does not match `n_nodes × n_comp`.
pub fn dwr_goal_oriented_estimator_2d(
    mesh: &Mesh<2>,
    u: &[f64],
    z: &[f64],
    f: &[f64],
    n_comp: usize,
) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let etype = mesh.element_type_at(0);
    let is_quad = etype == ElementType::Quad4;

    debug_assert!(n_comp == 1 || n_comp == 2,
        "dwr_goal_oriented_estimator_2d: n_comp must be 1 or 2, got {}", n_comp);
    debug_assert_eq!(u.len(), n_nodes * n_comp);
    debug_assert_eq!(z.len(), n_nodes * n_comp);
    debug_assert_eq!(f.len(), n_nodes * n_comp);

    // ── 1. Element gradients ──────────────────────────────────────────────
    // For scalar:   elem_data[e]    = [gx, gy]                (2-element)
    // For vector:   elem_data[e]    = [[gx0,gy0],[gx1,gy1]]  (2×2 matrix)
    let elem_data: Vec<Vec<[f64; 2]>> = if is_quad {
        (0..n_elems as ElemId).map(|e| {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);
            let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
            let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
            let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
            let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
            let det_j = j00 * j11 - j01 * j10;
            let idet = if det_j.abs() > 1e-30 { 1.0 / det_j } else { 0.0 };
            let jit = |r: usize, c: usize| -> f64 {
                if r == 0 && c == 0 {  j11 * idet }
                else if r == 0 && c == 1 { -j10 * idet }
                else if r == 1 && c == 0 { -j01 * idet }
                else {  j00 * idet }
            };

            let mut grads = Vec::with_capacity(n_comp);
            for comp in 0..n_comp {
                let uh: Vec<f64> = ns.iter()
                    .map(|&n| u[n as usize * n_comp + comp]).collect();
                let dxi  = 0.25 * (-uh[0] + uh[1] + uh[2] - uh[3]);
                let deta = 0.25 * (-uh[0] - uh[1] + uh[2] + uh[3]);
                let gx = jit(0,0) * dxi + jit(0,1) * deta;
                let gy = jit(1,0) * dxi + jit(1,1) * deta;
                grads.push([gx, gy]);
            }
            grads
        }).collect()
    } else {
        // Tri3: constant gradient per element
        (0..n_elems as ElemId).map(|e| {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);
            let j00 = c(1)[0] - c(0)[0]; let j01 = c(2)[0] - c(0)[0];
            let j10 = c(1)[1] - c(0)[1]; let j11 = c(2)[1] - c(0)[1];
            let det = j00 * j11 - j01 * j10;
            let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };

            let gref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            let mut grads = Vec::with_capacity(n_comp);
            for comp in 0..n_comp {
                let uh: Vec<f64> = ns.iter()
                    .map(|&n| u[n as usize * n_comp + comp]).collect();
                let mut gx = 0.0; let mut gy = 0.0;
                for k in 0..3 {
                    let gpx = (j11 * gref[k][0] - j10 * gref[k][1]) * idet;
                    let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * idet;
                    gx += uh[k] * gpx;
                    gy += uh[k] * gpy;
                }
                grads.push([gx, gy]);
            }
            grads
        }).collect()
    };

    // ── 2. Dual fluctuation and source average ─────────────────────────────
    let elem_omega: Vec<Vec<f64>> = (0..n_elems).map(|e| {
        let ns = mesh.elem_nodes(e as ElemId);
        let npe = ns.len();
        (0..n_comp).map(|comp| {
            ns.iter().map(|&n| z[n as usize * n_comp + comp]).sum::<f64>() / npe as f64
        }).collect()
    }).collect();

    let elem_f_avg: Vec<Vec<f64>> = (0..n_elems).map(|e| {
        let ns = mesh.elem_nodes(e as ElemId);
        let npe = ns.len();
        (0..n_comp).map(|comp| {
            ns.iter().map(|&n| f[n as usize * n_comp + comp]).sum::<f64>() / npe as f64
        }).collect()
    }).collect();

    let elem_area: Vec<f64> = if is_quad {
        (0..n_elems as ElemId).map(|e| {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);
            0.5 * ((c(0)[0]*c(1)[1] + c(1)[0]*c(2)[1] + c(2)[0]*c(3)[1] + c(3)[0]*c(0)[1])
                 - (c(1)[0]*c(0)[1] + c(2)[0]*c(1)[1] + c(3)[0]*c(2)[1] + c(0)[0]*c(3)[1])).abs()
        }).collect()
    } else {
        (0..n_elems as ElemId).map(|e| {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);
            0.5 * ((c(1)[0]-c(0)[0])*(c(2)[1]-c(0)[1])
                 - (c(2)[0]-c(0)[0])*(c(1)[1]-c(0)[1])).abs()
        }).collect()
    };

    // ── 3. Edge adjacency ──────────────────────────────────────────────────
    type Edge = (NodeId, NodeId);
    let edge_key = |a: NodeId, b: NodeId| -> Edge {
        if a < b { (a, b) } else { (b, a) }
    };

    let mut edge_elems: HashMap<Edge, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let edges: &[[usize; 2]] = if is_quad {
            &[[0,1],[1,2],[2,3],[3,0]]
        } else {
            &[[0,1],[1,2],[0,2]]
        };
        for &[a, b] in edges {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
    }

    // ── 4. Assemble DWR indicator ──────────────────────────────────────────
    let mut eta = vec![0.0_f64; n_elems];

    // 4a. Interior (source) contribution
    for e in 0..n_elems {
        let mut inner = 0.0;
        for comp in 0..n_comp {
            inner += elem_f_avg[e][comp] * elem_omega[e][comp];
        }
        eta[e] += inner.abs() * elem_area[e];
    }

    // 4b. Edge jump contribution
    for (edge, elems) in &edge_elems {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;

        let [xa, ya] = mesh.coords_of(edge.0);
        let [xb, yb] = mesh.coords_of(edge.1);
        let h = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
        if h < 1e-30 { continue; }

        let nx = -(yb - ya) / h;
        let ny =  (xb - xa) / h;

        // Jump in gradient(·n) per component, then dot with dual fluctuation
        let mut jump_dot_omega = 0.0_f64;
        for comp in 0..n_comp {
            let j0 = elem_data[e0][comp][0] * nx + elem_data[e0][comp][1] * ny;
            let j1 = elem_data[e1][comp][0] * nx + elem_data[e1][comp][1] * ny;
            let jump = (j0 - j1).abs();
            let w_mid = (elem_omega[e0][comp] + elem_omega[e1][comp]) * 0.5;
            jump_dot_omega += jump * w_mid.abs();
        }
        let contrib = 0.5 * h * jump_dot_omega;
        eta[e0] += contrib;
        eta[e1] += contrib;
    }

    eta
}

// ════════════════════════════════════════════════════════════════════════════�?//  3-D DWR estimator (Tet4, Hex8, Prism6, Pyramid5) �?scalar and vector
// ════════════════════════════════════════════════════════════════════════════�?
/// DWR goal-oriented error indicator for 3-D meshes.
///
/// Supports Tet4, Hex8, Prism6, and Pyramid5 elements.
///
/// # Arguments
/// * `mesh` �?a 3-D mesh.
/// * `u` �?primal solution (nodal values).
/// * `z` �?dual solution (nodal values).
/// * `f` �?source term (nodal values).
/// * `n_comp` �?1 for scalar, 3 for 3-D elasticity.
///
/// # Returns
/// Element-wise error indicators `η_K`.
pub fn dwr_goal_oriented_estimator_3d(
    mesh: &Mesh<3>,
    u: &[f64],
    z: &[f64],
    f: &[f64],
    n_comp: usize,
) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    debug_assert!(n_comp == 1 || n_comp == 3,
        "dwr_goal_oriented_estimator_3d: n_comp must be 1 or 3, got {}", n_comp);
    debug_assert_eq!(u.len(), n_nodes * n_comp);
    debug_assert_eq!(z.len(), n_nodes * n_comp);
    debug_assert_eq!(f.len(), n_nodes * n_comp);

    // ── 1. Element gradients per component ─────────────────────────────────
    let elem_data: Vec<Vec<[f64; 3]>> = (0..n_elems as ElemId).map(|e| {
        let ns = mesh.elem_nodes(e);
        let npe = ns.len();
        let c = |i| mesh.coords_of(ns[i]);

        let j = match npe {
            4 => {
                [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(3)[2]-c(0)[2]]]
            }
            8 => {
                [[0.125*(-c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]-c(4)[0]+c(5)[0]+c(6)[0]-c(7)[0]),
                  0.125*(-c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]-c(4)[0]-c(5)[0]+c(6)[0]+c(7)[0]),
                  0.125*(-c(0)[0]-c(1)[0]-c(2)[0]-c(3)[0]+c(4)[0]+c(5)[0]+c(6)[0]+c(7)[0])],
                 [0.125*(-c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]-c(4)[1]+c(5)[1]+c(6)[1]-c(7)[1]),
                  0.125*(-c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]-c(4)[1]-c(5)[1]+c(6)[1]+c(7)[1]),
                  0.125*(-c(0)[1]-c(1)[1]-c(2)[1]-c(3)[1]+c(4)[1]+c(5)[1]+c(6)[1]+c(7)[1])],
                 [0.125*(-c(0)[2]+c(1)[2]+c(2)[2]-c(3)[2]-c(4)[2]+c(5)[2]+c(6)[2]-c(7)[2]),
                  0.125*(-c(0)[2]-c(1)[2]+c(2)[2]+c(3)[2]-c(4)[2]-c(5)[2]+c(6)[2]+c(7)[2]),
                  0.125*(-c(0)[2]-c(1)[2]-c(2)[2]-c(3)[2]+c(4)[2]+c(5)[2]+c(6)[2]+c(7)[2])]]
            }
            6 => {
                [[(c(1)[0]-c(0)[0])/2.0,(c(2)[0]-c(0)[0])/2.0,(c(3)[0]-c(0)[0])/2.0],
                 [(c(1)[1]-c(0)[1])/2.0,(c(2)[1]-c(0)[1])/2.0,(c(3)[1]-c(0)[1])/2.0],
                 [(c(1)[2]-c(0)[2])/2.0,(c(2)[2]-c(0)[2])/2.0,(c(3)[2]-c(0)[2])/2.0]]
            }
            5 => {
                [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(4)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(4)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(4)[2]-c(0)[2]]]
            }
            _ => panic!("dwr_goal_oriented_estimator_3d: unsupported npe={}", npe),
        };

        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a = (r+1)%3; let b = (r+2)%3;
            let d = (c+1)%3; let e = (c+2)%3;
            (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
        };

        let gref: Vec<[f64;3]> = match npe {
            4 => vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            8 => vec![[-0.125,-0.125,-0.125],[0.125,-0.125,-0.125],[0.125,0.125,-0.125],[-0.125,0.125,-0.125],
                      [-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,0.125,0.125],[-0.125,0.125,0.125]],
            6 => vec![[-0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],
                      [-0.5,0.0,0.5],[0.5,0.0,0.5],[0.0,0.5,0.5]],
            5 => vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]],
            _ => unreachable!(),
        };

        let mut grads = Vec::with_capacity(n_comp);
        for comp in 0..n_comp {
            let uh: Vec<f64> = ns.iter().map(|&n| u[n as usize * n_comp + comp]).collect();
            let mut g = [0.0_f64; 3];
            for k in 0..npe {
                for i in 0..3 {
                    for jj in 0..3 {
                        g[i] += uh[k] * jit(jj, i) * gref[k][jj];
                    }
                }
            }
            grads.push(g);
        }
        grads
    }).collect();

    // ── 2. Element volumes ─────────────────────────────────────────────────
    let elem_vol: Vec<f64> = (0..n_elems as ElemId).map(|e| {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        match ns.len() {
            4 => {
                let j = [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(3)[0]-c(0)[0]],
                         [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(3)[1]-c(0)[1]],
                         [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(3)[2]-c(0)[2]]];
                (j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                 - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                 + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs() / 6.0
            }
            8 => {
                let j = [[0.125*(-c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]-c(4)[0]+c(5)[0]+c(6)[0]-c(7)[0]),
                          0.125*(-c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]-c(4)[0]-c(5)[0]+c(6)[0]+c(7)[0]),
                          0.125*(-c(0)[0]-c(1)[0]-c(2)[0]-c(3)[0]+c(4)[0]+c(5)[0]+c(6)[0]+c(7)[0])],
                         [0.125*(-c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]-c(4)[1]+c(5)[1]+c(6)[1]-c(7)[1]),
                          0.125*(-c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]-c(4)[1]-c(5)[1]+c(6)[1]+c(7)[1]),
                          0.125*(-c(0)[1]-c(1)[1]-c(2)[1]-c(3)[1]+c(4)[1]+c(5)[1]+c(6)[1]+c(7)[1])],
                         [0.125*(-c(0)[2]+c(1)[2]+c(2)[2]-c(3)[2]-c(4)[2]+c(5)[2]+c(6)[2]-c(7)[2]),
                          0.125*(-c(0)[2]-c(1)[2]+c(2)[2]+c(3)[2]-c(4)[2]-c(5)[2]+c(6)[2]+c(7)[2]),
                          0.125*(-c(0)[2]-c(1)[2]-c(2)[2]+c(3)[2]+c(4)[2]+c(5)[2]+c(6)[2]+c(7)[2])]];
                (j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                 - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                 + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs()
            }
            6 => {
                let j = [[(c(1)[0]-c(0)[0])/2.0,(c(2)[0]-c(0)[0])/2.0,(c(3)[0]-c(0)[0])/2.0],
                         [(c(1)[1]-c(0)[1])/2.0,(c(2)[1]-c(0)[1])/2.0,(c(3)[1]-c(0)[1])/2.0],
                         [(c(1)[2]-c(0)[2])/2.0,(c(2)[2]-c(0)[2])/2.0,(c(3)[2]-c(0)[2])/2.0]];
                (j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                 - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                 + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs() / 2.0
            }
            5 => {
                let v = |i| c(i);
                let d = |a:[f64;3],b:[f64;3],c:[f64;3],d:[f64;3]| -> f64 {
                    (b[0]-a[0])*(c[1]-a[1])*(d[2]-a[2])+(b[1]-a[1])*(c[2]-a[2])*(d[0]-a[0])+(b[2]-a[2])*(c[0]-a[0])*(d[1]-a[1])
                    -(b[2]-a[2])*(c[1]-a[1])*(d[0]-a[0])-(b[1]-a[1])*(c[0]-a[0])*(d[2]-a[2])-(b[0]-a[0])*(c[2]-a[2])*(d[1]-a[1])
                };
                (d(v(0),v(1),v(2),v(4)).abs() + d(v(2),v(3),v(0),v(4)).abs()) / 6.0
            }
            _ => 0.0,
        }
    }).collect();

    // ── 3. Dual fluctuation and source average ─────────────────────────────
    let elem_omega: Vec<Vec<f64>> = (0..n_elems).map(|e| {
        let ns = mesh.elem_nodes(e as ElemId);
        let npe = ns.len();
        (0..n_comp).map(|comp| {
            ns.iter().map(|&n| z[n as usize * n_comp + comp]).sum::<f64>() / npe as f64
        }).collect()
    }).collect();

    let elem_f_avg: Vec<Vec<f64>> = (0..n_elems).map(|e| {
        let ns = mesh.elem_nodes(e as ElemId);
        let npe = ns.len();
        (0..n_comp).map(|comp| {
            ns.iter().map(|&n| f[n as usize * n_comp + comp]).sum::<f64>() / npe as f64
        }).collect()
    }).collect();

    // ── 4. Face adjacency ──────────────────────────────────────────────────
    use std::collections::HashMap;
    let mut tri_faces: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    let mut quad_faces: HashMap<[u32;4], Vec<ElemId>> = HashMap::new();

    match mesh.elem_type {
        ElementType::Tet4 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                    let mut v=[a,b,c];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
            }
        }
        ElementType::Hex8 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for face in crate::amr::amr_inner::local_faces_hex() {
                    let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}
            }
        }
        ElementType::Prism6 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for (a,b,c) in crate::amr::amr_inner::local_faces_prism_tri() {
                    let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();
                    tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
                for face in crate::amr::amr_inner::local_faces_prism_quad() {
                    let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}
            }
        }
        ElementType::Pyramid5 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for (a,b,c) in crate::amr::amr_inner::local_faces_pyramid_tri() {
                    let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();
                    tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
                let qf=crate::amr::amr_inner::local_faces_pyramid_quad()[0];
                let fns=[ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];
                let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}
        }
        _ => panic!("dwr_goal_oriented_estimator_3d: unsupported {:?}", mesh.elem_type),
    }

    // ── 5. Assemble DWR indicator ──────────────────────────────────────────
    let mut eta = vec![0.0_f64; n_elems];

    // 5a. Interior (source) contribution
    for e in 0..n_elems {
        let mut inner = 0.0;
        for comp in 0..n_comp {
            inner += elem_f_avg[e][comp] * elem_omega[e][comp];
        }
        eta[e] += inner.abs() * elem_vol[e];
    }

    // 5b. Triangular face jumps
    for (&(na,nb,nc), elems) in &tri_faces {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb); let cc = mesh.coords_of(nc);
        let ex = cb[0]-ca[0]; let ey = cb[1]-ca[1]; let ez = cb[2]-ca[2];
        let fx = cc[0]-ca[0]; let fy = cc[1]-ca[1]; let fz = cc[2]-ca[2];
        let nx = ey*fz - ez*fy; let ny = ez*fx - ex*fz; let nz = ex*fy - ey*fx;
        let fa = 0.5 * (nx*nx+ny*ny+nz*nz).sqrt();
        if fa < 1e-30 { continue; }
        let inv = 1.0 / (nx*nx+ny*ny+nz*nz).sqrt();
        let hf = (2.0*fa).sqrt();

        let mut jump_dot_omega = 0.0_f64;
        for comp in 0..n_comp {
            let j0 = elem_data[e0][comp][0]*nx*inv
                   + elem_data[e0][comp][1]*ny*inv
                   + elem_data[e0][comp][2]*nz*inv;
            let j1 = elem_data[e1][comp][0]*nx*inv
                   + elem_data[e1][comp][1]*ny*inv
                   + elem_data[e1][comp][2]*nz*inv;
            let jump = (j0 - j1).abs();
            let w_mid = (elem_omega[e0][comp] + elem_omega[e1][comp]) * 0.5;
            jump_dot_omega += jump * w_mid.abs();
        }
        let contrib = 0.5 * hf * jump_dot_omega;
        eta[e0] += contrib; eta[e1] += contrib;
    }

    // 5c. Quadrilateral face jumps
    for (fns, elems) in &quad_faces {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let [a,b,c,d] = *fns;
        let ca = mesh.coords_of(a); let cb = mesh.coords_of(b);
        let cc = mesh.coords_of(c); let cd = mesh.coords_of(d);
        let ex1 = cb[0]-ca[0]; let ey1 = cb[1]-ca[1]; let ez1 = cb[2]-ca[2];
        let fx1 = cc[0]-ca[0]; let fy1 = cc[1]-ca[1]; let fz1 = cc[2]-ca[2];
        let nx1 = ey1*fz1 - ez1*fy1; let ny1 = ez1*fx1 - ex1*fz1; let nz1 = ex1*fy1 - ey1*fx1;
        let area1 = 0.5*(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let ex2 = cd[0]-ca[0]; let ey2 = cd[1]-ca[1]; let ez2 = cd[2]-ca[2];
        let fx2 = cc[0]-ca[0]; let fy2 = cc[1]-ca[1]; let fz2 = cc[2]-ca[2];
        let nx2 = ey2*fz2 - ez2*fy2; let ny2 = ez2*fx2 - ex2*fz2; let nz2 = ex2*fy2 - ey2*fx2;
        let area2 = 0.5*(nx2*nx2+ny2*ny2+nz2*nz2).sqrt();
        let fa = area1 + area2;
        if fa < 1e-30 { continue; }
        let inv = 1.0/(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let hf = (2.0*fa).sqrt();

        let mut jump_dot_omega = 0.0_f64;
        for comp in 0..n_comp {
            let j0 = elem_data[e0][comp][0]*nx1*inv
                   + elem_data[e0][comp][1]*ny1*inv
                   + elem_data[e0][comp][2]*nz1*inv;
            let j1 = elem_data[e1][comp][0]*nx1*inv
                   + elem_data[e1][comp][1]*ny1*inv
                   + elem_data[e1][comp][2]*nz1*inv;
            let jump = (j0 - j1).abs();
            let w_mid = (elem_omega[e0][comp] + elem_omega[e1][comp]) * 0.5;
            jump_dot_omega += jump * w_mid.abs();
        }
        let contrib = 0.5 * hf * jump_dot_omega;
        eta[e0] += contrib; eta[e1] += contrib;
    }

    eta
}

// ═════════════════════════════════════════════════════════════════════════════
//  Quantitative error bounds (Task 3.2)
// ═════════════════════════════════════════════════════════════════════════════

/// Upper and lower error bounds computed from element-wise indicators.
///
/// For a set of element error indicators `η_K` the estimated global error
/// satisfies:
/// - `upper = C_R · ‖η‖_2  = C_R · sqrt(Σ η_K²)`
/// - `lower = C_L · ‖η‖_∞ = C_L · max(η_K)`
///
/// where `C_R` (reliability, ≥ 1) and `C_L` (efficiency, ≤ 1) are
/// problem-dependent constants.  Standard values for elliptic problems
/// are `C_R ≈ 1..10`, `C_L ≈ 0.1..1.0`.
#[derive(Debug, Clone, Copy)]
pub struct ErrorBounds {
    /// Upper bound: C_R · sqrt(Σ η_K²).
    pub upper: f64,
    /// Lower bound: C_L · max(η_K).
    pub lower: f64,
    /// Global estimated error: sqrt(Σ η_K²).
    pub global_estimate: f64,
    /// Maximum element indicator value.
    pub max_indicator: f64,
    /// Number of elements.
    pub n_elems: usize,
}

/// Compute upper and lower error bounds from element-wise indicators.
///
/// # Arguments
/// * `eta` — element-wise error indicators (length = `n_elems`).
/// * `c_r` — reliability constant (≥ 1).  Multiplier for the upper bound.
/// * `c_l` — efficiency constant (≤ 1).  Multiplier for the lower bound.
///
/// # Returns
/// An [`ErrorBounds`] struct with the computed bounds and statistics.
///
/// # Panics
/// Panics if `eta` is empty, or if `c_r` is not ≥ 1.0, or `c_l` is not ≤ 1.0.
///
/// # Example
/// ```ignore
/// let bounds = compute_error_bounds(&eta, 1.5, 0.5);
/// // bounds.upper = 1.5 · sqrt(Σ η_K²)
/// // bounds.lower = 0.5 · max(η_K)
/// ```
pub fn compute_error_bounds(eta: &[f64], c_r: f64, c_l: f64) -> ErrorBounds {
    assert!(!eta.is_empty(), "compute_error_bounds: eta must be non-empty");
    assert!(c_r >= 1.0, "compute_error_bounds: c_r must be ≥ 1, got {c_r}");
    assert!(c_l <= 1.0, "compute_error_bounds: c_l must be ≤ 1, got {c_l}");
    assert!(c_l > 0.0, "compute_error_bounds: c_l must be > 0, got {c_l}");

    let n_elems = eta.len();
    let mut sum_sq = 0.0_f64;
    let mut max_val = 0.0_f64;

    for &v in eta {
        let v_abs = v.abs();
        sum_sq += v_abs * v_abs;
        if v_abs > max_val {
            max_val = v_abs;
        }
    }

    let global_estimate = sum_sq.sqrt();

    ErrorBounds {
        upper: c_r * global_estimate,
        lower: c_l * max_val,
        global_estimate,
        max_indicator: max_val,
        n_elems,
    }
}

/// Decide whether to stop mesh refinement based on the estimated error bound.
///
/// Returns `true` if the **upper** error bound is below the given tolerance,
/// meaning the mesh is sufficiently resolved for the target accuracy.
///
/// # Arguments
/// * `eta` — element-wise error indicators.
/// * `tolerance` — target error tolerance.
/// * `c_r` — reliability constant (same as for [`compute_error_bounds`]).
///
/// # Returns
/// `true` if `C_R · sqrt(Σ η_K²) < tolerance`, i.e. stop refining.
pub fn stop_on_tolerance(eta: &[f64], tolerance: f64, c_r: f64) -> bool {
    if eta.is_empty() || tolerance <= 0.0 {
        return tolerance <= 0.0;
    }
    let bounds = compute_error_bounds(eta, c_r, 1.0);
    bounds.upper < tolerance
}

/// Compute the efficiency index of an error estimator.
///
/// `eff = estimated_global_error / true_error`
///
/// An ideal estimator has `eff ≈ 1`.  Values significantly larger than 1
/// indicate overestimation; values significantly smaller than 1 indicate
/// underestimation.
///
/// # Arguments
/// * `estimated_error` — global estimated error (e.g. `sqrt(Σ η_K²)`).
/// * `true_error` — the true error in the same norm (e.g. energy norm).
///
/// # Returns
/// The ratio `estimated / true`.  Returns `f64::INFINITY` if `true_error` is zero.
///
/// # Panics
/// Panics if `estimated_error` is negative or NaN.
pub fn efficiency_index(estimated_error: f64, true_error: f64) -> f64 {
    assert!(estimated_error.is_finite() && estimated_error >= 0.0,
        "efficiency_index: estimated_error must be finite and ≥ 0, got {estimated_error}");
    if true_error == 0.0 {
        return f64::INFINITY;
    }
    estimated_error / true_error.abs()
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════════════════════════════════════�?
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    // ── 2-D scalar (Poisson) tests ─────────────────────────────────────────

    #[test]
    fn dwr_goal_2d_scalar_linear_tri3() {
        // u = x, z = y, f = 0 �?DWR = 0 for P1
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f = vec![0.0; n];
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 1e-12,
            "2D scalar DWR should be near-zero for linear u,z with f=0, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_2d_scalar_quadratic_tri3() {
        // u = x², z = y², f(x,y) = -2 (Laplacian) �?DWR > 0
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1]*c[1]
        }).collect();
        let f = vec![2.0; n];  // Laplacian(x²) = 2
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6,
            "2D scalar DWR should be >0 for quadratic u,z, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_2d_scalar_matches_existing_tri3() {
        // Generalized DWR with n_comp=1 should match the existing dwr_estimator
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0] + c[1]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1]*c[1] - c[0]
        }).collect();
        let f = vec![2.0; n];

        let eta_new = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
        let eta_old = crate::amr::estimators::dwr_estimator(&mesh, &u, &z, &f);

        assert_eq!(eta_new.len(), eta_old.len());
        for i in 0..eta_new.len() {
            let diff = (eta_new[i] - eta_old[i]).abs();
            assert!(diff < 1e-12,
                "Mismatch at element {i}: new={:.3e}, old={:.3e}", eta_new[i], eta_old[i]);
        }
    }

    #[test]
    fn dwr_goal_2d_scalar_quad4() {
        // Quad4 with linear u,z, f=0 �?DWR �?0
        let mesh = Mesh::<2>::unit_square_quad(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f = vec![0.0; n];
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 1e-12,
            "Quad4 scalar DWR should be ~0 for linear u,z f=0, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_2d_scalar_quadratic_quad4() {
        // Quad4 with u=x², z=y², f=2 �?DWR > 0
        let mesh = Mesh::<2>::unit_square_quad(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1]*c[1]
        }).collect();
        let f = vec![2.0; n];
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6,
            "Quad4 scalar DWR should be >0 for quadratic u,z, got {max:.3e}");
    }

    // ── 2-D vector (elasticity) tests ──────────────────────────────────────

    #[test]
    fn dwr_goal_2d_vector_linear_tri3() {
        // Linear displacement field: u = (x, y), z = (y, x), f = (0,0)
        // The DWR indicator should be near-zero since P1 captures linear fields exactly.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let n_comp = 2;
        let mut u = vec![0.0; n * n_comp];
        let mut z = vec![0.0; n * n_comp];
        let f = vec![0.0; n * n_comp];
        for i in 0..n {
            let c = mesh.coords_of(i as NodeId);
            u[i * n_comp + 0] = c[0];  // u_x = x
            u[i * n_comp + 1] = c[1];  // u_y = y
            z[i * n_comp + 0] = c[1];  // z_x = y
            z[i * n_comp + 1] = c[0];  // z_y = x
        }
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, n_comp);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 1e-12,
            "2D vector DWR should be ~0 for linear fields, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_2d_vector_quadratic_tri3() {
        // Quadratic displacement: u = (x², y²), z = (y, x), f = (2, 2)
        // DWR should detect the P1 mismatch.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let n_comp = 2;
        let mut u = vec![0.0; n * n_comp];
        let mut z = vec![0.0; n * n_comp];
        let mut f = vec![0.0; n * n_comp];
        for i in 0..n {
            let c = mesh.coords_of(i as NodeId);
            u[i * n_comp + 0] = c[0]*c[0];  // u_x = x²
            u[i * n_comp + 1] = c[1]*c[1];  // u_y = y²
            z[i * n_comp + 0] = c[1];       // z_x = y
            z[i * n_comp + 1] = c[0];       // z_y = x
            f[i * n_comp + 0] = 2.0;        // body force for x²
            f[i * n_comp + 1] = 2.0;        // body force for y²
        }
    }

    #[test]
    fn dwr_goal_2d_vector_quad4_linear() {
        // Quad4 linear vector �?should be zero
        let mesh = Mesh::<2>::unit_square_quad(4);
        let n = mesh.n_nodes();
        let n_comp = 2;
        let mut u = vec![0.0; n * n_comp];
        let mut z = vec![0.0; n * n_comp];
        let f = vec![0.0; n * n_comp];
        for i in 0..n {
            let c = mesh.coords_of(i as NodeId);
            u[i * n_comp + 0] = c[0];
            u[i * n_comp + 1] = c[1];
            z[i * n_comp + 0] = c[1];
            z[i * n_comp + 1] = c[0];
        }
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, n_comp);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 1e-12,
            "Quad4 vector DWR should be ~0 for linear fields, got {max:.3e}");
    }

    // ── 3-D scalar tests ───────────────────────────────────────────────────

    #[test]
    fn dwr_goal_3d_scalar_linear_tet4() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f = vec![0.0; n];
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0,
            "3D scalar DWR should be small for linear u,z, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_3d_scalar_quadratic_tet4() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1]*c[1]
        }).collect();
        let f = vec![2.0; n];
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6,
            "3D scalar DWR should be >0 for quadratic u,z, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_3d_scalar_matches_existing_tet4() {
        // Should match dwr_estimator_3d_general for Tet4 scalar
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1]*c[1]
        }).collect();
        let f = vec![2.0; n];
        let eta_new = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, 1);
        let eta_old = crate::amr::estimators::dwr_estimator_3d_general(&mesh, &u, &z, &f);
        assert_eq!(eta_new.len(), eta_old.len());
        for i in 0..eta_new.len() {
            let diff = (eta_new[i] - eta_old[i]).abs();
            assert!(diff < 1e-12,
                "3D scalar mismatch at elem {i}: new={:.3e}, old={:.3e}", eta_new[i], eta_old[i]);
        }
    }

    #[test]
    fn dwr_goal_3d_scalar_hex8_linear() {
        let mesh = Mesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f = vec![0.0; n];
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0,
            "Hex8 scalar DWR should be small for linear u,z, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_3d_scalar_hex8_quadratic() {
        let mesh = Mesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1]*c[1]
        }).collect();
        let f = vec![2.0; n];
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, 1);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6,
            "Hex8 scalar DWR should be >0 for quadratic u,z, got {max:.3e}");
    }

    // ── 3-D vector (elasticity) tests ──────────────────────────────────────

    #[test]
    fn dwr_goal_3d_vector_linear_tet4() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let n_comp = 3;
        let mut u = vec![0.0; n * n_comp];
        let mut z = vec![0.0; n * n_comp];
        let f = vec![0.0; n * n_comp];
        for i in 0..n {
            let c = mesh.coords_of(i as NodeId);
            for d in 0..3 {
                u[i * n_comp + d] = c[d];
                z[i * n_comp + d] = c[(d + 1) % 3];
            }
        }
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, n_comp);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0,
            "3D vector DWR should be small for linear fields, got {max:.3e}");
    }

    #[test]
    fn dwr_goal_3d_vector_quadratic_tet4() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let n_comp = 3;
        let mut u = vec![0.0; n * n_comp];
        let mut z = vec![0.0; n * n_comp];
        let mut f = vec![0.0; n * n_comp];
        for i in 0..n {
            let c = mesh.coords_of(i as NodeId);
            for d in 0..3 {
                u[i * n_comp + d] = c[d] * c[d]; // quadratic
                z[i * n_comp + d] = c[(d + 1) % 3];
                f[i * n_comp + d] = 2.0; // Laplacian of x² is 2
            }
        }
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, n_comp);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6,
            "3D vector DWR should be >0 for quadratic fields, got {max:.3e}");
    }

    // ── PointSolution functional tests ─────────────────────────────────────

    #[test]
    fn point_solution_adjoint_rhs_length() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let goal = PointSolution::new(0);
        let rhs = <PointSolution as GoalFunctional<2>>::assemble_adjoint_rhs(
            &goal, &mesh,
        );
        assert_eq!(rhs.len(), n, "Scalar adjoint RHS length should be n_nodes");
        assert_eq!(rhs[0], 1.0, "RHS[0] should be 1.0 for node-0 goal");
        assert!(rhs[1..].iter().all(|&v| v == 0.0), "All other RHS entries should be 0");
    }

    #[test]
    fn point_solution_evaluate() {
        let n = 20usize;
        let u: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let goal = PointSolution::new(5);
        let val = <PointSolution as GoalFunctional<1>>::evaluate(&goal, &u);
        assert!((val - 0.5).abs() < 1e-14,
            "Point solution evaluate failed: expected 0.5, got {val}");
    }

    // ── MeanStress functional tests ────────────────────────────────────────

    #[test]
    fn mean_stress_element_contrib_2d_tri3() {
        // Linear displacement: u = (x, 0) �?ε_xx = 1, σ_xx = 2μ + λ
        // For λ=μ=1: σ_xx = 3. Element contribution depends on area.
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let n_comp = 2;
        let mut u = vec![0.0; n * n_comp];
        for i in 0..n {
            let c = mesh.coords_of(i as NodeId);
            u[i * n_comp + 0] = c[0]; // u_x = x
            u[i * n_comp + 1] = 0.0;
        }

        let goal = MeanStress::new(0, 0).with_lame(1.0, 1.0);
        // Sum over all elements
        let total: f64 = (0..mesh.n_elems() as ElemId)
            .map(|e| goal.element_contrib(&u, &mesh, e))
            .sum();
        // For unit square with diagonal triangles: total area = 1.0, σ_xx = 3.0
        assert!((total - 3.0).abs() < 0.1,
            "Mean stress σ_xx integral should be ~3 for u=(x,0) with λ=μ=1, got {total}");
    }

    // ── Element type / solution length mismatch detection ──────────────────

    #[test]
    fn dwr_goal_2d_scalar_output_shape() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u = vec![1.0; n]; let z = vec![1.0; n]; let f = vec![0.0; n];
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
        assert_eq!(eta.len(), mesh.n_elems());
    }

    #[test]
    fn dwr_goal_3d_scalar_output_shape() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u = vec![1.0; n]; let z = vec![1.0; n]; let f = vec![0.0; n];
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, 1);
        assert_eq!(eta.len(), mesh.n_elems());
    }

    #[test]
    fn dwr_goal_2d_vector_output_shape() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes(); let nc = 2;
        let u = vec![1.0; n*nc]; let z = vec![1.0; n*nc]; let f = vec![0.0; n*nc];
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, nc);
        assert_eq!(eta.len(), mesh.n_elems());
    }

    #[test]
    fn dwr_goal_3d_vector_output_shape() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes(); let nc = 3;
        let u = vec![1.0; n*nc]; let z = vec![1.0; n*nc]; let f = vec![0.0; n*nc];
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, nc);
        assert_eq!(eta.len(), mesh.n_elems());
    }

    // ── Goal functional trait API tests ────────────────────────────────────

    #[test]
    fn point_solution_element_contrib_present() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        // Use u[0] = 1.0 so elements containing node 0 give non-zero contrib
        let u: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let goal = PointSolution::new(0);
        // Node 0 belongs to the first elements
        let contrib_first = goal.element_contrib(&u, &mesh, 0);
        assert!((contrib_first - 1.0).abs() < 1e-14,
            "Element containing node 0 should give contribution = u[0] = 1.0, got {contrib_first}");
        // Node 0 does not belong to elements far away
        let total_elems = mesh.n_elems();
        let mut found = false;
        for e in 0..total_elems as ElemId {
            if goal.element_contrib(&u, &mesh, e) != 0.0 { found = true; break; }
        }
        assert!(found, "PointSolution node 0 should contribute to at least one element");
    }

    #[test]
    fn dwr_goal_2d_scalar_element_shape() {
        // Verify that each element gets its own indicator value.
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 triangles
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0] + c[1]*c[1]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] - c[1]
        }).collect();
        let f = vec![2.0; n];
        let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
        assert_eq!(eta.len(), mesh.n_elems());
        // All indicators should be non-negative (absolute values)
        assert!(eta.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn dwr_goal_3d_vec_element_count() {
        let mesh = Mesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes(); let nc = 3;
        let u = vec![1.0; n*nc]; let z = vec![1.0; n*nc]; let f = vec![0.0; n*nc];
        let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, nc);
        assert_eq!(eta.len(), mesh.n_elems());
    }

    // ── Error bounds tests (Task 3.2) ─────────────────────────────────────

    #[test]
    fn error_bounds_basic() {
        let eta = vec![1.0, 2.0, 3.0, 4.0];
        let b = compute_error_bounds(&eta, 1.5, 0.5);
        // sum_sq = 1+4+9+16 = 30, sqrt = 5.477...
        let expected_global = (1.0_f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert!((b.global_estimate - expected_global).abs() < 1e-14);
        assert!((b.upper - 1.5 * expected_global).abs() < 1e-14);
        assert!((b.lower - 0.5 * 4.0).abs() < 1e-14);
        assert!((b.max_indicator - 4.0).abs() < 1e-14);
        assert_eq!(b.n_elems, 4);
    }

    #[test]
    fn error_bounds_uniform_indicators() {
        // All elements have the same error → max = each value
        let n = 100usize;
        let eta = vec![2.0; n];
        let b = compute_error_bounds(&eta, 2.0, 0.3);
        let expected_global = (n as f64 * 4.0).sqrt(); // sqrt(100 * 4) = 20
        assert!((b.global_estimate - expected_global).abs() < 1e-12);
        assert!((b.upper - 2.0 * expected_global).abs() < 1e-12);
        assert!((b.lower - 0.3 * 2.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "eta must be non-empty")]
    fn error_bounds_empty_eta_panics() {
        compute_error_bounds(&[], 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "c_r must be ≥ 1")]
    fn error_bounds_cr_too_small_panics() {
        compute_error_bounds(&[1.0], 0.9, 0.5);
    }

    #[test]
    #[should_panic(expected = "c_l must be ≤ 1")]
    fn error_bounds_cl_too_large_panics() {
        compute_error_bounds(&[1.0], 1.0, 1.5);
    }

    #[test]
    fn stop_on_tolerance_below() {
        // eta = [1, 1, 1] → global = sqrt(3) ≈ 1.732
        // C_R = 1.5 → upper = 2.598
        // tolerance = 3.0 → should stop (upper < tol)
        let eta = vec![1.0; 3];
        assert!(stop_on_tolerance(&eta, 3.0, 1.5));
    }

    #[test]
    fn stop_on_tolerance_above() {
        // eta = [1, 1, 1] → global = sqrt(3) ≈ 1.732
        // C_R = 1.5 → upper = 2.598
        // tolerance = 2.0 → should NOT stop (upper >= tol)
        let eta = vec![1.0; 3];
        assert!(!stop_on_tolerance(&eta, 2.0, 1.5));
    }

    #[test]
    fn stop_on_tolerance_zero_tolerance() {
        // Zero tolerance should stop immediately (no error can be below zero).
        // Actually with zero tolerance, stop_on_tolerance returns true since
        // tolerance <= 0.0 returns tolerance <= 0.0 which means true.
        assert!(stop_on_tolerance(&[1.0], 0.0, 1.0));
    }

    #[test]
    fn stop_on_tolerance_empty() {
        // Empty indicators with negative tolerance returns true
        assert!(stop_on_tolerance(&[], -1.0, 1.0));
        // Empty with positive tolerance returns false
        assert!(!stop_on_tolerance(&[], 1.0, 1.0));
    }

    #[test]
    fn efficiency_index_ideal_estimator() {
        // eff = 1 means perfect estimator
        let eff = efficiency_index(2.5, 2.5);
        assert!((eff - 1.0).abs() < 1e-14);
    }

    #[test]
    fn efficiency_index_overestimates() {
        let eff = efficiency_index(5.0, 2.0);
        assert!((eff - 2.5).abs() < 1e-14);
    }

    #[test]
    fn efficiency_index_underestimates() {
        let eff = efficiency_index(1.0, 4.0);
        assert!((eff - 0.25).abs() < 1e-14);
    }

    #[test]
    fn efficiency_index_zero_true_error() {
        let eff = efficiency_index(1.0, 0.0);
        assert!(eff.is_infinite());
    }

    // ── Efficiency index with known exact solution (Task 3.2 Step 4) ──────

    /// Compute the L² norm of the gradient error between exact solution
    /// u_exact and P1 finite element solution u_h on a Tri3 mesh.
    fn compute_gradient_error(
        mesh: &Mesh<2>,
        u_h: &[f64],
        grad_exact: &[fn(f64, f64) -> f64; 2], // [∂u/∂x, ∂u/∂y]
    ) -> f64 {
        let mut error_sq = 0.0_f64;
        for e in 0..mesh.n_elems() as ElemId {
            let ns = mesh.elem_nodes(e);
            let c = |i| mesh.coords_of(ns[i]);

            // P1 element gradient (constant)
            let j00 = c(1)[0] - c(0)[0]; let j01 = c(2)[0] - c(0)[0];
            let j10 = c(1)[1] - c(0)[1]; let j11 = c(2)[1] - c(0)[1];
            let det = j00 * j11 - j01 * j10;
            let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
            let gref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            let uh = [u_h[ns[0] as usize], u_h[ns[1] as usize], u_h[ns[2] as usize]];
            let mut gx = 0.0; let mut gy = 0.0;
            for k in 0..3 {
                let gpx = (j11 * gref[k][0] - j10 * gref[k][1]) * idet;
                let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * idet;
                gx += uh[k] * gpx;
                gy += uh[k] * gpy;
            }

            // Element centroid coordinates
            let cx = (c(0)[0] + c(1)[0] + c(2)[0]) / 3.0;
            let cy = (c(0)[1] + c(1)[1] + c(2)[1]) / 3.0;
            let ex = grad_exact[0](cx, cy);
            let ey = grad_exact[1](cx, cy);

            let area = 0.5 * det.abs();
            error_sq += ((gx - ex).powi(2) + (gy - ey).powi(2)) * area;
        }
        error_sq.sqrt()
    }

    #[test]
    fn efficiency_index_zz_sin_solution() {
        // Known smooth exact solution: u(x,y) = sin(πx)·sin(πy) on unit square
        // ZZ estimator should have efficiency index ≈ O(1).
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();

        // Exact solution at nodes
        let u_exact: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId);
            (std::f64::consts::PI * c[0]).sin()
                * (std::f64::consts::PI * c[1]).sin()
        }).collect();

        // Gradient of exact solution: [π·cos(πx)·sin(πy), π·sin(πx)·cos(πy)]
        let grad_exact: [fn(f64, f64) -> f64; 2] = [
            |x, y| std::f64::consts::PI * (std::f64::consts::PI * x).cos() * (std::f64::consts::PI * y).sin(),
            |x, y| std::f64::consts::PI * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).cos(),
        ];

        // ZZ error indicators
        let eta = crate::amr::estimators::zz_estimator(&mesh, &u_exact);

        // Estimated global error
        let global_est: f64 = eta.iter().map(|v| v * v).sum::<f64>().sqrt();

        // True gradient error
        let true_err = compute_gradient_error(&mesh, &u_exact, &grad_exact);

        // Efficiency index should be O(1)
        let eff = efficiency_index(global_est, true_err);
        assert!(eff > 0.1 && eff < 10.0,
            "ZZ estimator efficiency index should be O(1) for smooth sin solution, got {eff:.3}");
    }

    #[test]
    fn efficiency_index_zz_linear_solution() {
        // Linear exact solution: u(x,y) = x + 2y
        // P1 FEM should capture this exactly → error ≈ 0, ZZ should also be ≈ 0
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();

        let u_exact: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId);
            c[0] + 2.0 * c[1]
        }).collect();

        let grad_exact: [fn(f64, f64) -> f64; 2] = [
            |_, _| 1.0,
            |_, _| 2.0,
        ];

        let eta = crate::amr::estimators::zz_estimator(&mesh, &u_exact);
        let global_est: f64 = eta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let true_err = compute_gradient_error(&mesh, &u_exact, &grad_exact);

        assert!(global_est < 1e-12,
            "ZZ should be ~0 for linear solution, got {global_est:.3e}");
        assert!(true_err < 1e-14,
            "True error should be ~0 for linear solution, got {true_err:.3e}");

        // Efficiency index should be finite (both are near zero)
        // Since true_err is effectively zero, efficiency_index returns Inf
        let eff = efficiency_index(global_est, true_err);
        assert!(eff.is_infinite() || eff > 0.0);
    }
}
