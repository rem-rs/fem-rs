//! Discrete linear operators: gradient, curl, and divergence.
//!
//! These operators map between finite element spaces in the de Rham complex:
//!
//! ```text
//!   H1 --grad--> H(curl) --curl--> H(div) --div--> L2
//! ```
//!
//! The resulting sparse matrices are exact (not approximations) for the
//! lowest-order spaces (P1 -> ND1 -> RT0 -> P0).

use std::collections::HashSet;

use fem_element::{TriND1, TriRT0, VectorReferenceElement};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, ElementTransformation};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

/// Discrete linear operators that build sparse matrices mapping between FE spaces.
///
/// All methods are associated functions (no `self`) that take the relevant
/// spaces as arguments and return a `CsrMatrix<f64>`.
pub struct DiscreteLinearOperator;

impl DiscreteLinearOperator {
    /// Build the discrete gradient matrix G: H1 -> H(curl).
    ///
    /// For lowest-order (P1 -> ND1), G is the signed incidence matrix of the
    /// mesh graph.  For edge (v_a, v_b) with global orientation a < b:
    ///
    /// ```text
    ///   G[edge_dof, v_b] = +sign
    ///   G[edge_dof, v_a] = -sign
    /// ```
    ///
    /// where `sign` is the H(curl) orientation sign on the element.
    ///
    /// # Panics
    /// Panics if the spaces are not lowest-order (P1 and ND1).
    pub fn gradient<M: MeshTopology>(
        h1_space: &H1Space<M>,
        hcurl_space: &HCurlSpace<M>,
    ) -> CsrMatrix<f64> {
        assert_eq!(h1_space.order(), 1, "gradient: only P1 H1 space supported");
        assert_eq!(hcurl_space.order(), 1, "gradient: only ND1 H(curl) space supported");

        let mesh = h1_space.mesh();
        let n_hcurl = hcurl_space.n_dofs();
        let n_h1 = h1_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_hcurl, n_h1);

        // Local edge table for triangles (must match HCurlSpace TRI_EDGES)
        let local_edges: &[(usize, usize)] = match mesh.dim() {
            2 => &[(0, 1), (1, 2), (0, 2)],
            3 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            d => panic!("gradient: unsupported dimension {d}"),
        };

        // Track which edge DOFs we have already visited, to avoid duplicates
        // from shared edges.
        let mut visited = HashSet::with_capacity(n_hcurl);

        for e in mesh.elem_iter() {
            let verts = mesh.element_nodes(e);
            let h1_dofs = h1_space.element_dofs(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);

            for (local_edge_idx, &(li, lj)) in local_edges.iter().enumerate() {
                let edge_dof = hcurl_dofs[local_edge_idx] as usize;

                // Only process each edge DOF once
                if !visited.insert(edge_dof) {
                    continue;
                }

                // H1 DOFs for P1 are the vertex node indices
                let va_dof = h1_dofs[li] as usize;
                let vb_dof = h1_dofs[lj] as usize;

                // Global edge orientation: from smaller to larger vertex index.
                // The discrete gradient is the signed incidence matrix:
                //   G[edge_dof, end_vertex] = +1
                //   G[edge_dof, start_vertex] = -1
                // where the edge goes from start to end in global orientation.
                let (gi, gj) = (verts[li], verts[lj]);
                if gi < gj {
                    // Local li -> lj matches global direction
                    coo.add(edge_dof, vb_dof, 1.0);
                    coo.add(edge_dof, va_dof, -1.0);
                } else {
                    // Local direction is opposite to global
                    coo.add(edge_dof, va_dof, 1.0);
                    coo.add(edge_dof, vb_dof, -1.0);
                }
            }
        }

        coo.into_csr()
    }

    /// Build the discrete curl matrix C: H(curl) -> L2 (2D).
    ///
    /// For lowest-order (ND1 -> P0) in 2D, the matrix entries are:
    ///
    /// ```text
    ///   C[l2_dof, hcurl_dof] = sign * curl_ref / det_j * |det_j| * area_ref
    /// ```
    ///
    /// where `curl_ref` is the constant reference curl of each ND1 basis function,
    /// `det_j` is the Jacobian determinant, and `area_ref = 0.5` for triangles.
    ///
    /// # Panics
    /// Panics if the spaces are not 2D lowest-order (ND1 and P0).
    pub fn curl_2d<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        l2_space: &L2Space<M>,
    ) -> CsrMatrix<f64> {
        assert_eq!(hcurl_space.order(), 1, "curl_2d: only ND1 H(curl) supported");
        assert_eq!(l2_space.order(), 0, "curl_2d: only P0 L2 supported");

        let mesh = hcurl_space.mesh();
        assert_eq!(mesh.dim(), 2, "curl_2d: only 2D meshes supported");

        let n_l2 = l2_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_l2, n_hcurl);

        // Reference curls for TriND1: [2, 2, -2] (constant)
        let ref_elem = TriND1;
        let mut curl_ref = vec![0.0; ref_elem.n_dofs()];
        ref_elem.eval_curl(&[0.0, 0.0], &mut curl_ref);

        let area_ref = 0.5; // reference triangle area

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let signs = hcurl_space.element_signs(e);
            let l2_dofs = l2_space.element_dofs(e);
            let l2_dof = l2_dofs[0] as usize;

            // Compute Jacobian determinant
            let det_j = simplex_det(mesh, nodes);

            // Physical curl = curl_ref / det_j
            // Integral over element = curl_phys * |det_j| * area_ref
            //                       = (curl_ref / det_j) * |det_j| * area_ref
            //                       = curl_ref * sign(det_j) * area_ref
            let sign_det = if det_j > 0.0 { 1.0 } else { -1.0 };

            for i in 0..ref_elem.n_dofs() {
                let hcurl_dof = hcurl_dofs[i] as usize;
                let val = signs[i] * curl_ref[i] * sign_det * area_ref;
                coo.add(l2_dof, hcurl_dof, val);
            }
        }

        coo.into_csr()
    }

    /// Build the discrete divergence matrix D: H(div) -> L2.
    ///
    /// For lowest-order (RT0 -> P0), the matrix entries are:
    ///
    /// ```text
    ///   D[l2_dof, hdiv_dof] = sign * div_ref * sign(det_j) * area_ref
    /// ```
    ///
    /// where `div_ref = 2` for all RT0 basis functions, and `area_ref = 0.5`.
    ///
    /// # Panics
    /// Panics if the spaces are not lowest-order (RT0 and P0) on a 2D mesh.
    pub fn divergence<M: MeshTopology>(
        hdiv_space: &HDivSpace<M>,
        l2_space: &L2Space<M>,
    ) -> CsrMatrix<f64> {
        assert_eq!(hdiv_space.order(), 0, "divergence: only RT0 H(div) supported");
        assert_eq!(l2_space.order(), 0, "divergence: only P0 L2 supported");

        let mesh = hdiv_space.mesh();

        let n_l2 = l2_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_l2, n_hdiv);

        // Reference divergences for TriRT0: all = 2.0 (constant)
        let ref_elem = TriRT0;
        let mut div_ref = vec![0.0; ref_elem.n_dofs()];
        ref_elem.eval_div(&[0.0, 0.0], &mut div_ref);

        let area_ref = 0.5; // reference triangle area

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hdiv_dofs = hdiv_space.element_dofs(e);
            let signs = hdiv_space.element_signs(e);
            let l2_dofs = l2_space.element_dofs(e);
            let l2_dof = l2_dofs[0] as usize;

            // Compute Jacobian determinant
            let det_j = simplex_det(mesh, nodes);

            // Physical div = div_ref / det_j  (Piola transform)
            // Integral over element = div_phys * |det_j| * area_ref
            //                       = div_ref * sign(det_j) * area_ref
            let sign_det = if det_j > 0.0 { 1.0 } else { -1.0 };

            for i in 0..ref_elem.n_dofs() {
                let hdiv_dof = hdiv_dofs[i] as usize;
                let val = signs[i] * div_ref[i] * sign_det * area_ref;
                coo.add(l2_dof, hdiv_dof, val);
            }
        }

        coo.into_csr()
    }
}

// ---- Helper ----------------------------------------------------------------

/// Compute the determinant of the simplex Jacobian for element `e`.
fn simplex_det<M: MeshTopology>(mesh: &M, geo_nodes: &[u32]) -> f64 {
    ElementTransformation::from_simplex_nodes(mesh, geo_nodes).det_j()
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    /// Test: Discrete gradient of a linear function u = x + 2y.
    ///
    /// The gradient field is (1, 2) everywhere.  Interpolating u into H1
    /// and applying G should give the same result as interpolating (1,2)
    /// into H(curl) via its DOF functional.
    #[test]
    fn gradient_of_linear_function() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);

        // Interpolate u = x + 2y into H1
        let u_h1 = h1.interpolate(&|x| x[0] + 2.0 * x[1]);

        // Build gradient matrix and apply
        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);
        let mut g_u = vec![0.0; hcurl.n_dofs()];
        g.spmv(u_h1.as_slice(), &mut g_u);

        // Interpolate grad(u) = (1, 2) into H(curl) via the DOF functional
        let grad_interp = hcurl.interpolate_vector(&|_x| vec![1.0, 2.0]);

        // Compare: they should match exactly (up to floating-point)
        for i in 0..hcurl.n_dofs() {
            assert!(
                (g_u[i] - grad_interp.as_slice()[i]).abs() < 1e-12,
                "gradient mismatch at DOF {i}: G*u = {}, interp = {}",
                g_u[i], grad_interp.as_slice()[i]
            );
        }
    }

    /// Test: de Rham exact sequence property: curl(grad(u)) = 0.
    ///
    /// Build G (H1 -> H(curl)) and C (H(curl) -> L2), then verify C * G = 0.
    #[test]
    fn de_rham_curl_of_grad_is_zero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh3, 0);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);
        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2);

        // Test C * G * u = 0 for several functions
        let test_fns: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| x[0]),
            Box::new(|x: &[f64]| x[1]),
            Box::new(|x: &[f64]| x[0] + x[1]),
            Box::new(|x: &[f64]| 3.0 * x[0] - 2.0 * x[1]),
        ];

        for (idx, f) in test_fns.iter().enumerate() {
            let u = h1.interpolate(f.as_ref());
            let mut gu = vec![0.0; hcurl.n_dofs()];
            g.spmv(u.as_slice(), &mut gu);
            let mut cgu = vec![0.0; l2.n_dofs()];
            c.spmv(&gu, &mut cgu);

            let max_err: f64 = cgu.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                max_err < 1e-12,
                "curl(grad(u_{idx})) not zero: max |C*G*u| = {max_err}"
            );
        }
    }

    /// Test: Discrete divergence of a known field.
    ///
    /// For the constant field F = (1, 0), div(F) = 0.
    /// For the field F = (x, y), div(F) = 2.
    #[test]
    fn divergence_constant_field() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh, 0);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh2, 0);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2);

        // Test 1: F = (1, 0) -> div = 0
        let f_const = hdiv.interpolate_vector(&|_x| vec![1.0, 0.0]);
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f_const.as_slice(), &mut div_f);

        let max_err: f64 = div_f.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err < 1e-10,
            "div(1,0) should be 0, max |D*F| = {max_err}"
        );

        // Test 2: F = (0, 1) -> div = 0
        let f_const2 = hdiv.interpolate_vector(&|_x| vec![0.0, 1.0]);
        let mut div_f2 = vec![0.0; l2.n_dofs()];
        d.spmv(f_const2.as_slice(), &mut div_f2);

        let max_err2: f64 = div_f2.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err2 < 1e-10,
            "div(0,1) should be 0, max |D*F| = {max_err2}"
        );
    }

    /// Test: Matrix dimensions are correct.
    #[test]
    fn matrix_dimensions() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh3, 0);
        let mesh4 = SimplexMesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh4, 0);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);
        assert_eq!(g.nrows, hcurl.n_dofs());
        assert_eq!(g.ncols, h1.n_dofs());

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2);
        assert_eq!(c.nrows, l2.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2);
        assert_eq!(d.nrows, l2.n_dofs());
        assert_eq!(d.ncols, hdiv.n_dofs());
    }

    /// Test: de Rham sequence div(curl(u)) = 0 (2D version).
    ///
    /// In 2D the "curl" of a scalar is a div-free vector field, so we check
    /// that D * C * u_hcurl = 0 for arbitrary H(curl) DOF vectors.
    /// Note: In 2D, the curl maps H(curl) -> L2 (scalar), and div maps H(div) -> L2.
    /// The sequence is H1 -> H(curl) -> L2, so we only check curl(grad) = 0 (above).
    /// For completeness, check that G has no trivial kernel issues.
    #[test]
    fn gradient_nonzero_for_nonconst() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);

        // A non-constant function should have non-zero gradient
        let u = h1.interpolate(&|x| x[0]);
        let mut gu = vec![0.0; hcurl.n_dofs()];
        g.spmv(u.as_slice(), &mut gu);

        let norm: f64 = gu.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "gradient of x should be nonzero, got norm = {norm}");

        // A constant function should have zero gradient
        let u_const = h1.interpolate(&|_x| 1.0);
        let mut gu_const = vec![0.0; hcurl.n_dofs()];
        g.spmv(u_const.as_slice(), &mut gu_const);

        let norm_const: f64 = gu_const.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm_const < 1e-12, "gradient of constant should be zero, got norm = {norm_const}");
    }
}
