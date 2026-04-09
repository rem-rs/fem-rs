//! Post-processing utilities: element-wise gradient, curl, divergence, and
//! nodal gradient recovery (Zienkiewicz-Zhu style).

use nalgebra::DMatrix;

use fem_element::lagrange::{TetP1, TetP2, TetP3, TriP1, TriP2, TriP3};
use fem_element::nedelec::{TetND1, TriND1, TetND2, TriND2};
use fem_element::raviart_thomas::{TetRT0, TriRT0, TetRT1, TriRT1};
use fem_element::reference::VectorReferenceElement;
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementTransformation;
use fem_space::fe_space::{FESpace, SpaceType};

// ─── Reference element factories ───────────────────────────────────────────

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

fn vec_ref_elem(space_type: SpaceType, dim: usize, order: u8) -> Box<dyn VectorReferenceElement> {
    match (space_type, dim, order) {
        (SpaceType::HCurl, 2, 1) => Box::new(TriND1),
        (SpaceType::HCurl, 2, 2) => Box::new(TriND2),
        (SpaceType::HCurl, 3, 1) => Box::new(TetND1),
        (SpaceType::HCurl, 3, 2) => Box::new(TetND2),
        (SpaceType::HDiv, 2, 0) => Box::new(TriRT0),
        (SpaceType::HDiv, 2, 1) => Box::new(TriRT1),
        (SpaceType::HDiv, 3, 0) => Box::new(TetRT0),
        (SpaceType::HDiv, 3, 1) => Box::new(TetRT1),
        _ => panic!("vec_ref_elem: unsupported (space_type={space_type:?}, dim={dim}, order={order})"),
    }
}

// ─── Jacobian helpers ──────────────────────────────────────────────────────

fn simplex_jacobian<M: MeshTopology>(
    mesh: &M,
    geo_nodes: &[u32],
) -> (DMatrix<f64>, f64) {
    let tr = ElementTransformation::from_simplex_nodes(mesh, geo_nodes);
    (tr.jacobian().clone(), tr.det_j())
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

// ─── Element-wise gradient (scalar H1/L2 spaces) ──────────────────────────

/// Compute element-wise gradient of a scalar FE solution.
///
/// Evaluates the gradient at the element centroid.
/// Returns a `Vec` of length `n_elements`, each entry is `[du/dx, du/dy, ...]`.
pub fn compute_element_gradients<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let order = space.order();

    let mut result = Vec::with_capacity(mesh.n_elements());

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        let (jac, _det_j) = simplex_jacobian(mesh, nodes);
        let j_inv_t = jac.try_inverse().expect("degenerate element").transpose();

        // Evaluate at centroid: (1/3, 1/3) for tri, (1/4, 1/4, 1/4) for tet.
        let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];

        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut grad_phys = vec![0.0; n_ldofs * dim];
        ref_elem.eval_grad_basis(&xi, &mut grad_ref);
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

        let mut grad = vec![0.0; dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..dim {
                grad[d] += c * grad_phys[i * dim + d];
            }
        }
        result.push(grad);
    }

    result
}

// ─── H1 seminorm error ────────────────────────────────────────────────────

/// Compute the H¹ seminorm error `‖∇u_exact − ∇u_h‖_{L²(Ω)}`.
///
/// Integrates `|∇u_exact − ∇u_h|²` over every element using Gaussian
/// quadrature (not just the centroid), and returns the square root.
///
/// # Arguments
/// * `space`      — finite element space (H¹, scalar)
/// * `dofs`       — FE solution coefficient vector (length = `space.n_dofs()`)
/// * `grad_exact` — closure mapping physical coordinates `x` to the exact
///                  gradient vector (length = mesh dimension)
/// * `quad_order` — quadrature accuracy order.  Use `order * 2 + 2` or higher
///                  for P1/P2 solutions.
///
/// # Returns
/// `sqrt( ∫_Ω |∇u_exact - ∇u_h|² dΩ )`
pub fn compute_h1_error<S: FESpace>(
    space: &S,
    dofs: &[f64],
    grad_exact: impl Fn(&[f64]) -> Vec<f64>,
    quad_order: u8,
) -> f64 {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let order = space.order();

    let mut err2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        // ── Jacobian for this element ────────────────────────────────────
        let (jac, det_j) = simplex_jacobian(mesh, nodes);
        let j_inv_t = jac
            .try_inverse()
            .expect("degenerate element in compute_h1_error")
            .transpose();

        // Cache vertex coordinates for physical-point mapping.
        // For a 2-D simplex: x = x0 + J * xi (J built from x1-x0, x2-x0).
        let x0: Vec<f64> = mesh.node_coords(nodes[0]).to_vec();
        let jac_cols: Vec<Vec<f64>> = (1..=dim)
            .map(|k| {
                let xk = mesh.node_coords(nodes[k]);
                (0..dim).map(|d| xk[d] - x0[d]).collect()
            })
            .collect();

        // ── Quadrature ───────────────────────────────────────────────────
        let quad = ref_elem.quadrature(quad_order);
        let mut grad_ref = vec![0.0_f64; n_ldofs * dim];
        let mut grad_phys = vec![0.0_f64; n_ldofs * dim];

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j.abs();

            // Reference → physical coordinates.
            let mut x_phys: Vec<f64> = x0.clone();
            for k in 0..dim {
                for d in 0..dim {
                    x_phys[d] += jac_cols[k][d] * xi[k];
                }
            }

            // Basis gradients in reference space, then transform to physical.
            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

            // FE gradient: ∇u_h = Σ_i c_i ∇φ_i(x)
            let mut grad_h = vec![0.0_f64; dim];
            for i in 0..n_ldofs {
                let c = dofs[elem_dofs[i] as usize];
                for d in 0..dim {
                    grad_h[d] += c * grad_phys[i * dim + d];
                }
            }

            // Exact gradient at this physical point.
            let grad_ex = grad_exact(&x_phys);

            // Accumulate ‖∇u_exact − ∇u_h‖² weighted by quadrature weight.
            let diff2: f64 = (0..dim).map(|d| (grad_ex[d] - grad_h[d]).powi(2)).sum();
            err2 += w * diff2;
        }
    }

    err2.sqrt()
}

// ─── Kelly error indicators ───────────────────────────────────────────────────

/// Compute element-wise Kelly error indicators for a scalar H¹ solution.
///
/// The Kelly indicator for element `K` is:
/// ```text
///   η_K = sqrt( Σ_{E ∈ ∂K ∩ Ω_int} h_E * |[∇u_h · n_E]|² )
/// ```
/// where the sum is over interior edges, `h_E` is the edge length, and
/// `[∇u_h · n_E]` is the normal gradient jump across the edge.
///
/// This is a simplified (facet-jump) variant without volume residuals.
/// Suitable as a refinement indicator for AMR marking (Dörfler / maximum strategy).
///
/// # Arguments
/// * `space`          — finite element space (H¹, scalar)
/// * `dofs`           — FE solution coefficient vector
/// * `interior_faces` — pre-computed interior face list (from [`InteriorFaceList::build`])
///
/// # Returns
/// A `Vec<f64>` of length `n_elements` where `result[e]` is `η_e²`
/// (take square root for the actual indicator).
pub fn compute_kelly_indicators<S: FESpace>(
    space: &S,
    dofs: &[f64],
    interior_faces: &crate::InteriorFaceList,
) -> Vec<f64> {
    let mesh = space.mesh();
    let n_elems = mesh.n_elements();

    // Pre-compute element-wise gradients (centroid evaluation).
    let grads = compute_element_gradients(space, dofs);

    let mut indicators = vec![0.0_f64; n_elems];

    for face in &interior_faces.faces {
        let na = face.face_nodes[0];
        let nb = face.face_nodes[1];

        let (h_e, nx, ny) = edge_length_and_normal_2d(mesh, na, nb);

        let gl = &grads[face.elem_left as usize];
        let gr = &grads[face.elem_right as usize];

        // Normal gradient jump: [∇u_h · n]
        let jump = (gl[0] - gr[0]) * nx + (gl[1] - gr[1]) * ny;
        let contrib = h_e * jump * jump;

        indicators[face.elem_left  as usize] += contrib;
        indicators[face.elem_right as usize] += contrib;
    }

    indicators
}

/// Compute the 2-D edge length and outward unit normal for the edge `(na, nb)`.
/// The normal is perpendicular to the edge direction, rotated 90° counterclockwise.
fn edge_length_and_normal_2d<M: MeshTopology>(
    mesh: &M,
    na: u32,
    nb: u32,
) -> (f64, f64, f64) {
    let xa = mesh.node_coords(na);
    let xb = mesh.node_coords(nb);
    let dx = xb[0] - xa[0];
    let dy = xb[1] - xa[1];
    let h = (dx * dx + dy * dy).sqrt();
    // Unit normal: rotate edge vector 90° CCW, then normalise.
    // (The sign convention doesn't matter for jump² magnitude.)
    (h, -dy / h, dx / h)
}

// ─── Element-wise curl (H(curl) spaces) ────────────────────────────────────

/// Compute element-wise curl of an H(curl) FE solution.
///
/// Evaluates at the element centroid.
/// For 2D: each entry is a `Vec<f64>` of length 1 (scalar curl).
/// For 3D: each entry is a `Vec<f64>` of length 3 (vector curl).
pub fn compute_element_curl<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();

    let ref_elem = vec_ref_elem(stype, dim, space.order());
    let n_ldofs = ref_elem.n_dofs();
    let curl_dim = if dim == 2 { 1 } else { 3 };

    let mut result = Vec::with_capacity(mesh.n_elements());

    // Centroid in reference coordinates.
    let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];

    let mut ref_curl = vec![0.0; n_ldofs * curl_dim];
    let mut phys_curl = vec![0.0; n_ldofs * curl_dim];

    for e in mesh.elem_iter() {
        let elem_dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let nodes = mesh.element_nodes(e);

        let (jac, det_j) = simplex_jacobian(mesh, nodes);

        ref_elem.eval_curl(&xi, &mut ref_curl);

        // H(curl) curl transform: curl_phys = J * curl_ref / det_j (3D)
        //                          curl_phys = curl_ref / det_j (2D scalar)
        let inv_det = 1.0 / det_j;
        if dim == 2 {
            for i in 0..n_ldofs {
                phys_curl[i] = ref_curl[i] * inv_det;
            }
        } else {
            for i in 0..n_ldofs {
                for r in 0..3 {
                    let mut s = 0.0;
                    for c in 0..3 {
                        s += jac[(r, c)] * ref_curl[i * 3 + c];
                    }
                    phys_curl[i * 3 + r] = s * inv_det;
                }
            }
        }

        // Apply orientation signs.
        if let Some(s) = signs {
            for i in 0..n_ldofs {
                for c in 0..curl_dim {
                    phys_curl[i * curl_dim + c] *= s[i];
                }
            }
        }

        // Sum contributions: curl(u_h) = Σ_i c_i curl(φ_i)
        let mut curl_val = vec![0.0; curl_dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..curl_dim {
                curl_val[d] += c * phys_curl[i * curl_dim + d];
            }
        }
        result.push(curl_val);
    }

    result
}

// ─── Element-wise divergence (H(div) spaces) ──────────────────────────────

/// Compute element-wise divergence of an H(div) FE solution.
///
/// Evaluates at the element centroid.
/// Returns a `Vec<f64>` of length `n_elements`.
pub fn compute_element_divergence<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<f64> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();

    let ref_elem = vec_ref_elem(stype, dim, space.order());
    let n_ldofs = ref_elem.n_dofs();

    let mut result = Vec::with_capacity(mesh.n_elements());

    let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];

    let mut ref_div = vec![0.0; n_ldofs];
    let mut phys_div = vec![0.0; n_ldofs];

    for e in mesh.elem_iter() {
        let elem_dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let nodes = mesh.element_nodes(e);

        let (_jac, det_j) = simplex_jacobian(mesh, nodes);

        ref_elem.eval_div(&xi, &mut ref_div);

        // H(div) divergence transform: div_phys = div_ref / det_j
        let inv_det = 1.0 / det_j;
        for i in 0..n_ldofs {
            phys_div[i] = ref_div[i] * inv_det;
        }

        // Apply orientation signs.
        if let Some(s) = signs {
            for i in 0..n_ldofs {
                phys_div[i] *= s[i];
            }
        }

        // Sum: div(u_h) = Σ_i c_i div(φ_i)
        let mut div_val = 0.0;
        for i in 0..n_ldofs {
            div_val += dofs[elem_dofs[i] as usize] * phys_div[i];
        }
        result.push(div_val);
    }

    result
}

// ─── Nodal gradient recovery (Zienkiewicz-Zhu) ────────────────────────────

/// Nodal gradient recovery via area-weighted averaging (Zienkiewicz-Zhu style).
///
/// For a scalar H1 solution, computes the smoothed gradient at each mesh node
/// by averaging the element-wise constant gradients weighted by element area.
///
/// Returns a `Vec` of `dim` vectors, each of length `n_nodes`:
/// `result[d][node]` = d-th component of the recovered gradient at `node`.
pub fn recover_gradient_nodal<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let n_nodes = mesh.n_nodes();
    let order = space.order();

    // Accumulate area-weighted gradient at each node.
    let mut grad_accum = vec![vec![0.0; n_nodes]; dim];
    let mut area_accum = vec![0.0; n_nodes];

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        let (jac, det_j) = simplex_jacobian(mesh, nodes);
        let j_inv_t = jac.try_inverse().expect("degenerate element").transpose();

        // Element area/volume (for a simplex: |det_j| / d!)
        let elem_area = det_j.abs() / match dim {
            2 => 2.0,
            3 => 6.0,
            _ => 1.0,
        };

        // Gradient at centroid.
        let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];
        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut grad_phys = vec![0.0; n_ldofs * dim];
        ref_elem.eval_grad_basis(&xi, &mut grad_ref);
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

        let mut grad = vec![0.0; dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..dim {
                grad[d] += c * grad_phys[i * dim + d];
            }
        }

        // Distribute to element vertices (first dim+1 nodes are vertices).
        let n_verts = dim + 1;
        for v in 0..n_verts {
            let node = nodes[v] as usize;
            area_accum[node] += elem_area;
            for d in 0..dim {
                grad_accum[d][node] += elem_area * grad[d];
            }
        }
    }

    // Divide by total accumulated weight.
    for node in 0..n_nodes {
        if area_accum[node] > 0.0 {
            for d in 0..dim {
                grad_accum[d][node] /= area_accum[node];
            }
        }
    }

    grad_accum
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, HCurlSpace, HDivSpace, fe_space::FESpace};

    #[test]
    fn element_gradients_linear_function() {
        // u(x,y) = 3x - 2y → ∇u = [3, -2] everywhere (constant per element).
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let grads = compute_element_gradients(&space, dofs);
        assert_eq!(grads.len(), space.mesh().n_elements());

        for (e, g) in grads.iter().enumerate() {
            assert!(
                (g[0] - 3.0).abs() < 1e-10,
                "elem {e}: ∂u/∂x = {}, expected 3.0",
                g[0]
            );
            assert!(
                (g[1] + 2.0).abs() < 1e-10,
                "elem {e}: ∂u/∂y = {}, expected -2.0",
                g[1]
            );
        }
    }

    #[test]
    fn recover_gradient_nodal_linear() {
        // For u(x,y) = 3x - 2y, the recovered gradient at every node should
        // be [3, -2] exactly (since all contributing elements have the same gradient).
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let grad = recover_gradient_nodal(&space, dofs);
        assert_eq!(grad.len(), 2); // dim = 2
        assert_eq!(grad[0].len(), space.mesh().n_nodes());

        for node in 0..space.mesh().n_nodes() {
            assert!(
                (grad[0][node] - 3.0).abs() < 1e-10,
                "node {node}: ∂u/∂x = {}, expected 3.0",
                grad[0][node]
            );
            assert!(
                (grad[1][node] + 2.0).abs() < 1e-10,
                "node {node}: ∂u/∂y = {}, expected -2.0",
                grad[1][node]
            );
        }
    }

    #[test]
    fn element_curl_hcurl_basic() {
        // Sanity check: compute element curls on a simple mesh, verify the
        // output has the right length.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();
        let dofs = vec![0.0; n]; // zero field → curl should be zero

        let curls = compute_element_curl(&space, &dofs);
        assert_eq!(curls.len(), space.mesh().n_elements());

        for (e, c) in curls.iter().enumerate() {
            assert_eq!(c.len(), 1, "2D curl should be scalar");
            assert!(
                c[0].abs() < 1e-12,
                "elem {e}: curl of zero field should be 0, got {}",
                c[0]
            );
        }
    }

    #[test]
    fn element_divergence_hdiv_basic() {
        // Sanity check: compute element divergences for zero field.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let n = space.n_dofs();
        let dofs = vec![0.0; n];

        let divs = compute_element_divergence(&space, &dofs);
        assert_eq!(divs.len(), space.mesh().n_elements());

        for (e, &d) in divs.iter().enumerate() {
            assert!(
                d.abs() < 1e-12,
                "elem {e}: div of zero field should be 0, got {d}"
            );
        }
    }

    // ── H1 error tests ────────────────────────────────────────────────────────

    #[test]
    fn h1_error_linear_function() {
        // u = 3x - 2y → ∇u = [3, -2] exactly for P1.
        // So H1 seminorm error should be essentially machine-zero.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let err = compute_h1_error(&space, dofs, |_| vec![3.0, -2.0], 5);
        assert!(
            err < 1e-10,
            "H1 error for linear u should be ~0, got {err:.3e}"
        );
    }

    #[test]
    fn h1_error_poisson_p1_convergence() {
        // P1 Poisson: u = sin(πx)sin(πy), ∇u_exact = [π cos(πx)sin(πy), π sin(πx)cos(πy)]
        // H1 seminorm error should converge at O(h¹) for P1 (rate ≈ 1).
        use std::f64::consts::PI;
        use crate::assembler::Assembler;
        use crate::standard::{DiffusionIntegrator, DomainSourceIntegrator};
        use fem_space::constraints::{apply_dirichlet, boundary_dofs};

        let ns = [8usize, 16, 32];
        let mut prev: Option<(f64, f64)> = None;

        for &n in &ns {
            let mesh = SimplexMesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 1);
            let ndofs = space.n_dofs();

            let mut mat = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa: 1.0 }],
                3,
            );
            let src = DomainSourceIntegrator::new(|x: &[f64]| {
                2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
            });
            let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

            let dm = space.dof_manager();
            let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
            apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

            let mut u = vec![0.0_f64; ndofs];
            fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u, &fem_solver::SolverConfig {
                rtol: 1e-12, max_iter: 10_000, verbose: false, ..fem_solver::SolverConfig::default()
            }).unwrap();

            let err = compute_h1_error(&space, &u, |x| {
                vec![
                    PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
                    PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                ]
            }, 5);

            let h = 1.0 / n as f64;
            if let Some((e0, h0)) = prev {
                let rate = (err / e0).ln() / (h / h0).ln();
                assert!(
                    rate > 0.85,
                    "P1 H1 convergence rate = {rate:.3}, expected ≥ 0.85 (n={n})"
                );
            }
            prev = Some((err, h));
        }
    }

    #[test]
    fn h1_error_poisson_p2_convergence() {
        // P2 Poisson: H1 seminorm error should converge at O(h²) (rate ≈ 2).
        use std::f64::consts::PI;
        use crate::assembler::Assembler;
        use crate::standard::{DiffusionIntegrator, DomainSourceIntegrator};
        use fem_space::constraints::{apply_dirichlet, boundary_dofs};

        let ns = [4usize, 8, 16];
        let mut prev: Option<(f64, f64)> = None;

        for &n in &ns {
            let mesh = SimplexMesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 2);
            let ndofs = space.n_dofs();

            let mut mat = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa: 1.0 }],
                5,
            );
            let src = DomainSourceIntegrator::new(|x: &[f64]| {
                2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
            });
            let mut rhs = Assembler::assemble_linear(&space, &[&src], 5);

            let dm = space.dof_manager();
            let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
            apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

            let mut u = vec![0.0_f64; ndofs];
            fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u, &fem_solver::SolverConfig {
                rtol: 1e-12, max_iter: 10_000, verbose: false, ..fem_solver::SolverConfig::default()
            }).unwrap();

            let err = compute_h1_error(&space, &u, |x| {
                vec![
                    PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
                    PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                ]
            }, 7);

            let h = 1.0 / n as f64;
            if let Some((e0, h0)) = prev {
                let rate = (err / e0).ln() / (h / h0).ln();
                assert!(
                    rate > 1.8,
                    "P2 H1 convergence rate = {rate:.3}, expected ≥ 1.8 (n={n})"
                );
            }
            prev = Some((err, h));
        }
    }

    // ── Kelly indicator tests ─────────────────────────────────────────────────

    #[test]
    fn kelly_zero_for_linear_solution() {
        // u = 3x - 2y: P1 reproduces this exactly → constant gradient per element
        // → no gradient jump across any interior edge → all indicators = 0.
        use crate::InteriorFaceList;
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let ifl = InteriorFaceList::build(space.mesh());
        let indicators = compute_kelly_indicators(&space, dofs, &ifl);

        assert_eq!(indicators.len(), space.mesh().n_elements());
        for (e, &ind) in indicators.iter().enumerate() {
            assert!(
                ind.abs() < 1e-20,
                "elem {e}: Kelly indicator should be 0 for linear u, got {ind:.3e}"
            );
        }
    }

    #[test]
    fn kelly_nonzero_for_nonlinear_solution() {
        // For a Poisson solution on a coarse mesh, gradient jumps between adjacent
        // elements are non-zero → total indicators > 0.
        use crate::{Assembler, InteriorFaceList, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
        use fem_space::constraints::{apply_dirichlet, boundary_dofs};
        use std::f64::consts::PI;

        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh.clone(), 1);
        let ndofs = space.n_dofs();

        let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let src = DomainSourceIntegrator::new(|x: &[f64]| {
            2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);

        let mut u = vec![0.0_f64; ndofs];
        fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u, &fem_solver::SolverConfig {
            rtol: 1e-12, max_iter: 10_000, verbose: false,
            ..fem_solver::SolverConfig::default()
        }).unwrap();

        let ifl = InteriorFaceList::build(space.mesh());
        let indicators = compute_kelly_indicators(&space, &u, &ifl);

        assert_eq!(indicators.len(), space.mesh().n_elements());
        let total: f64 = indicators.iter().sum();
        assert!(total > 0.0, "total Kelly indicator should be > 0, got {total}");
    }
}
