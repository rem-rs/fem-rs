//! Element geometry transformation utilities.
//!
//! Provides:
//! - [`ElementTransformation`] — affine simplex transformation (MFEM `ElementTransformation`)
//! - [`geometry_jacobian`] — compute Jacobian at a reference point for any element type
//! - [`xform_grads`] — transform reference gradients to physical space

use fem_core::{ElemId, NodeId};
use nalgebra::DMatrix;

use crate::topology::MeshTopology;

/// Affine element transformation for simplex geometries.
///
/// For a simplex with vertex coordinates `x0, x1, ..., x_dim`,
/// `J[:,k] = x_{k+1} - x_0` and `x(ξ) = x0 + J ξ`.
#[derive(Debug, Clone)]
pub struct ElementTransformation {
    dim: usize,
    x0: Vec<f64>,
    jacobian: DMatrix<f64>,
    det_j: f64,
    jacobian_inv_t: DMatrix<f64>,
}

impl ElementTransformation {
    /// Build a simplex transformation from mesh element id.
    pub fn from_simplex<M: MeshTopology>(mesh: &M, elem: ElemId) -> Self {
        let nodes = mesh.element_nodes(elem);
        Self::from_simplex_nodes(mesh, nodes)
    }

    /// Build a simplex transformation from a node slice.
    ///
    /// Uses the first `dim + 1` nodes as simplex vertices.  Coordinates come
    /// from the vertex table (`node_coords`); callers that need per-element
    /// geometry (curved / geometrically periodic meshes) should resolve the
    /// geometry node ids themselves via [`MeshTopology::geometry_nodes`] /
    /// [`MeshTopology::geom_coords_of`] (see [`element_jacobian_at`], and the
    /// isoparametric assembly paths, which do exactly that).
    pub fn from_simplex_nodes<M: MeshTopology>(mesh: &M, geo_nodes: &[u32]) -> Self {
        let dim = mesh.dim() as usize;
        assert!(
            geo_nodes.len() > dim,
            "ElementTransformation::from_simplex_nodes: need at least dim+1 nodes"
        );

        let x0 = mesh.node_coords(geo_nodes[0]).to_vec();
        let mut jac = DMatrix::<f64>::zeros(dim, dim);
        // Column order must match the reference-element axes of the SOLUTION
        // basis.  For Tet4/Hex8 the node order is the axis order, but for
        // Prism6 the PrismPk reference is (ξ0 = layer xi, ξ1 = tri eta,
        // ξ2 = tri zeta) with vertices [0,1,2,3,4,5] =
        // (0,0,0),(0,1,0),(0,0,1),(1,0,0),(1,1,0),(1,0,1) — so ∂x/∂ξ0 comes
        // from vertex 3, ∂x/∂ξ1 from vertex 1, ∂x/∂ξ2 from vertex 2.
        let col_of: Vec<usize> = match geo_nodes.len() {
            6 => vec![3, 1, 2], // Prism6: (ξ0, ξ1, ξ2) = (layer, tri-eta, tri-zeta)
            _ => (0..dim).map(|i| i + 1).collect(),
        };
        for col in 0..dim {
            let xc = mesh.node_coords(geo_nodes[col_of[col]]);
            for row in 0..dim {
                jac[(row, col)] = xc[row] - x0[row];
            }
        }

        let det_j = jac.determinant();
        let jacobian_inv_t = jac
            .clone()
            .try_inverse()
            .expect("ElementTransformation: degenerate simplex element")
            .transpose();

        Self {
            dim,
            x0,
            jacobian: jac,
            det_j,
            jacobian_inv_t,
        }
    }

    /// Spatial dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Jacobian matrix `J`.
    pub fn jacobian(&self) -> &DMatrix<f64> {
        &self.jacobian
    }

    /// Jacobian determinant `det(J)`.
    pub fn det_j(&self) -> f64 {
        self.det_j
    }

    /// Inverse-transpose Jacobian `J^{-T}`.
    pub fn jacobian_inv_t(&self) -> &DMatrix<f64> {
        &self.jacobian_inv_t
    }

    /// Reference-to-physical map for affine simplex elements.
    pub fn map_to_physical(&self, xi: &[f64]) -> Vec<f64> {
        assert_eq!(
            xi.len(),
            self.dim,
            "ElementTransformation::map_to_physical: xi dimension mismatch"
        );
        let mut xp = self.x0.clone();
        for i in 0..self.dim {
            for k in 0..self.dim {
                xp[i] += self.jacobian[(i, k)] * xi[k];
            }
        }
        xp
    }
}

/// Compute the geometry Jacobian determinant and inverse-transpose at a
/// reference point for a mesh element of any type (MFEM: `ElementTransformation`).
///
/// Returns `(detJ, J^{-T})` where `J_{ij} = ∂x_i/∂ξ_j` is the Jacobian of
/// the reference-to-physical mapping, computed from the element's **geometry**
/// nodal coordinates ([`MeshTopology::geometry_nodes`] / [`MeshTopology::geom_coords_of`]
/// — per-element geometry when the mesh carries one, e.g. curved or
/// geometrically periodic meshes; else the vertex table) and the linear (P1)
/// reference-element gradient basis.
///
/// Supports all element types: Tri3, Quad4, Tet4, Hex8, Prism6, etc.
///
/// # Panics
/// Panics if the element's geometry Jacobian is singular.
pub fn geometry_jacobian(
    mesh: &dyn MeshTopology,
    elem: u32,
    xi: &[f64],
    dim: usize,
) -> (f64, DMatrix<f64>) {
    let et = mesh.element_type(elem);
    let n_pe = mesh.element_nodes(elem).len();
    // Per-element geometry only when it is a P1-sized table (geometrically
    // periodic meshes); high-order curved geometry keeps the previous
    // vertex-table behavior here (the isoparametric paths handle curvature).
    let gnodes = mesh.geometry_nodes(elem);
    let nodes: &[NodeId] = if gnodes.len() == n_pe { gnodes } else { mesh.element_nodes(elem) };
    let n_ldofs = nodes.len();
    let re_geom = et.ref_elem(1);
    let mut grad = vec![0.0_f64; n_ldofs * dim];
    re_geom.eval_grad_basis(xi, &mut grad);
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    for k in 0..n_ldofs {
        let x = mesh.geom_coords_of(nodes[k]);
        for i in 0..dim {
            for j in 0..dim {
                jac[(i, j)] += x[i] * grad[k * dim + j];
            }
        }
    }
    let det = jac.determinant();
    let inv = jac.try_inverse().expect("singular Jacobian in geometry_jacobian");
    (det, inv.transpose())
}

/// Transform reference-element gradients to physical space:
/// `∇_phys = J^{-T} ∇_ref` (MFEM: `ElementTransformation` gradient transform).
pub fn xform_grads(ji: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for a in 0..n {
        for j in 0..dim {
            gp[a * dim + j] = (0..dim).map(|k| ji[(j, k)] * gr[a * dim + k]).sum();
        }
    }
}

/// Compute the full Jacobian matrix `J` and the physical point `x_phys`
/// at a reference point for a given element.
///
/// Returns `(J, x_phys)` where `J_{ij} = ∂x_i/∂ξ_j` and `x_phys = x(ξ)`.
/// Uses the linear (P1) reference element for the geometry mapping and the
/// element's **geometry** coordinates ([`MeshTopology::geometry_nodes`] /
/// [`MeshTopology::geom_coords_of`] — per-element geometry when present, e.g.
/// curved or geometrically periodic meshes; else the vertex table).
///
/// Supported element types: Tri3, Quad4, Tet4, Hex8, Prism6.
///
/// MFEM: `ElementTransformation::Jacobian()` + `ElementTransformation::Transform()`
pub fn element_jacobian_at<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    xi: &[f64],
    dim: usize,
) -> (DMatrix<f64>, Vec<f64>) {
    let et = mesh.element_type(elem);
    let re = et.ref_elem(1);
    let npe = re.n_dofs();
    let mut grad = vec![0.0_f64; npe * dim];
    let mut phi = vec![0.0_f64; npe];
    re.eval_basis(xi, &mut phi);
    re.eval_grad_basis(xi, &mut grad);
    // Per-element geometry only when it is a P1-sized table (geometrically
    // periodic meshes); high-order curved geometry keeps the previous
    // vertex-table behavior here (the isoparametric paths handle curvature).
    let gnodes = mesh.geometry_nodes(elem);
    let nodes: &[NodeId] = if gnodes.len() == npe { gnodes } else { mesh.element_nodes(elem) };
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0_f64; dim];
    for k in 0..npe {
        let c = mesh.geom_coords_of(nodes[k]);
        for i in 0..dim {
            xp[i] += c[i] * phi[k];
            for j in 0..dim {
                jac[(i, j)] += c[i] * grad[k * dim + j];
            }
        }
    }
    (jac, xp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    #[test]
    fn tri2d_det_and_map() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let tr = ElementTransformation::from_simplex(&mesh, 0);
        assert_eq!(tr.dim(), 2);
        assert!(tr.det_j().abs() > 1e-14);

        // Reference centroid for triangle.
        let x = tr.map_to_physical(&[1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(x.len(), 2);
    }
}


/// Locate points in the mesh (serial brute-force `Mesh::FindPoints`).
///
/// For each of `npts` points (row-major `points[i*dim + j]`) finds the
/// containing element and the reference coordinates by Newton inversion of
/// the isoparametric (P1 geometry) mapping. Points outside the mesh get
/// element id `-1` and zero reference coordinates.
///
/// Returns `(elem_ids, ref_coords)` with `elem_ids.len() == npts`.
///
/// Note: MFEM uses a BVH-accelerated search with the same semantics for
/// straight meshes; the found elements and reference coordinates agree.
pub fn find_points<M: MeshTopology + ?Sized>(
    mesh: &M,
    points: &[f64],
    npts: usize,
) -> (Vec<i64>, Vec<Vec<f64>>) {
    use crate::element_type::ElementType;

    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements();
    let eps = 1e-8;

    // Per-element vertex bounding boxes.
    let mut bboxes: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as u32 {
        let nds = mesh.element_nodes(e);
        let mut lo = vec![f64::INFINITY; dim];
        let mut hi = vec![f64::NEG_INFINITY; dim];
        for &n in nds {
            let c = mesh.node_coords(n);
            for d in 0..dim {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        bboxes.push((lo, hi));
    }

    // Reference-domain containment per element type.
    fn contained(et: ElementType, xi: &[f64], eps: f64) -> bool {
        match et {
            ElementType::Tri3 | ElementType::Tri6 => {
                xi[0] >= -eps && xi[1] >= -eps && xi[0] + xi[1] <= 1.0 + eps
            }
            ElementType::Tet4 | ElementType::Tet10 => {
                xi.iter().all(|&t| t >= -eps) && xi.iter().sum::<f64>() <= 1.0 + eps
            }
            ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
                xi[0] >= -eps
                    && xi[1] >= -eps
                    && xi[2] >= -eps
                    && xi[1] + xi[2] <= 1.0 + eps
                    && xi[0] <= 1.0 + eps
            }
            _ => xi.iter().all(|&t| t >= -eps && t <= 1.0 + eps),
        }
    }

    // Newton inversion for one point in one element, using MFEM-canonical
    // vertex-ordering geometry interpolation (linear/bilinear/trilinear).
    // Returns reference coordinates if the converged point lies in-domain.
    fn locate<M: MeshTopology + ?Sized>(
        mesh: &M,
        e: u32,
        target: &[f64],
        eps: f64,
    ) -> Option<Vec<f64>> {
        use crate::element_type::ElementType;
        let dim = mesh.dim() as usize;
        let et = mesh.element_type(e);
        let nds = mesh.element_nodes(e);

        // Matches MFEM Geometry vertex orderings (same as the mesh files).
        let x_of = |k: usize| mesh.node_coords(nds[k]);
        let map = |xi: &[f64], x: &mut [f64]| {
            for v in x.iter_mut() {
                *v = 0.0;
            }
            match et {
                ElementType::Tri3 | ElementType::Tri6 => {
                    // (1-s-t) v0 + s v1 + t v2
                    let (s, t) = (xi[0], xi[1]);
                    let w = [1.0 - s - t, s, t];
                    for k in 0..3 {
                        let c = x_of(k);
                        for d in 0..2 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                ElementType::Quad4 => {
                    let (s, t) = (xi[0], xi[1]);
                    let w = [
                        (1.0 - s) * (1.0 - t),
                        s * (1.0 - t),
                        s * t,
                        (1.0 - s) * t,
                    ];
                    for k in 0..4 {
                        let c = x_of(k);
                        for d in 0..2 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                ElementType::Tet4 | ElementType::Tet10 => {
                    let w = [1.0 - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]];
                    for k in 0..4 {
                        let c = x_of(k);
                        for d in 0..3 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                ElementType::Hex8 => {
                    let (s, t, u) = (xi[0], xi[1], xi[2]);
                    let w = [
                        (1.0 - s) * (1.0 - t) * (1.0 - u),
                        s * (1.0 - t) * (1.0 - u),
                        s * t * (1.0 - u),
                        (1.0 - s) * t * (1.0 - u),
                        (1.0 - s) * (1.0 - t) * u,
                        s * (1.0 - t) * u,
                        s * t * u,
                        (1.0 - s) * t * u,
                    ];
                    for k in 0..8 {
                        let c = x_of(k);
                        for d in 0..3 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                _ => panic!(
                    "find_points: unsupported element type {et:?} (straight Quad4/Tri3/Tet4/Hex8 only)"
                ),
            }
        };
        let jacobian_at = |xi: &[f64], jac: &mut [f64]| {
            let h = 1e-7;
            let n = xi.len();
            let mut xp = vec![0.0; n];
            let mut xm = vec![0.0; n];
            let mut xi_p = xi.to_vec();
            let mut xi_m = xi.to_vec();
            for col in 0..n {
                xi_p.copy_from_slice(xi);
                xi_m.copy_from_slice(xi);
                xi_p[col] += h;
                xi_m[col] -= h;
                map(&xi_p, &mut xp);
                map(&xi_m, &mut xm);
                for row in 0..n {
                    jac[row * n + col] = (xp[row] - xm[row]) / (2.0 * h);
                }
            }
        };

        // Start at the reference-domain centroid.
        let mut xi: Vec<f64> = match et {
            ElementType::Tri3 | ElementType::Tri6 => vec![1.0 / 3.0; 2],
            _ => vec![0.5; dim],
        };

        let mut x = vec![0.0_f64; dim];
        let mut jac = vec![0.0_f64; dim * dim];
        for _iter in 0..30 {
            map(&xi, &mut x);
            jacobian_at(&xi, &mut jac);
            let mut rhs: Vec<f64> = target.to_vec();
            for d in 0..dim {
                rhs[d] -= x[d];
            }
            let mut a = jac.clone();
            let Some(delta) = gauss_solve(&mut a, &mut rhs, dim) else {
                return None;
            };
            let mut norm = 0.0;
            for d in 0..dim {
                xi[d] += delta[d];
                norm += delta[d] * delta[d];
            }
            if norm < 1e-28 {
                break;
            }
        }

        // Residual check + containment.
        map(&xi, &mut x);
        let mut res2 = 0.0;
        for d in 0..dim {
            res2 += (x[d] - target[d]).powi(2);
        }
        if res2 < 1e-20 && contained(et, &xi, eps) {
            Some(xi)
        } else {
            None
        }
    }

    let mut elem_ids: Vec<i64> = Vec::with_capacity(npts);
    let mut ref_coords: Vec<Vec<f64>> = Vec::with_capacity(npts);

    for i in 0..npts {
        let target = &points[i * dim..i * dim + dim];
        let mut found: Option<(u32, Vec<f64>)> = None;
        // Pass 1: bbox-filtered candidates; pass 2: all elements (points on
        // shared boundaries may sit outside every padded bbox).
        for pass in 0..2 {
            for e in 0..n_elems as u32 {
                if pass == 0 {
                    let (lo, hi) = &bboxes[e as usize];
                    let inside_bbox = (0..dim).all(|d| {
                        target[d] >= lo[d] - 1e-9 && target[d] <= hi[d] + 1e-9
                    });
                    if !inside_bbox {
                        continue;
                    }
                }
                if let Some(xi) = locate(mesh, e, target, eps) {
                    found = Some((e, xi));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        match found {
            Some((e, xi)) => {
                elem_ids.push(e as i64);
                ref_coords.push(xi);
            }
            None => {
                elem_ids.push(-1);
                ref_coords.push(vec![0.0; dim]);
            }
        }
    }
    (elem_ids, ref_coords)
}

/// Small dense solver (Gaussian elimination with partial pivoting).
/// Solves in place; `a` is row-major `n x n`, `b` length `n`.
fn gauss_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in col + 1..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let inv = 1.0 / a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] * inv;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in r + 1..n {
            s -= a[r * n + c] * x[c];
        }
        x[r] = s / a[r * n + r];
    }
    Some(x)
}

#[cfg(test)]
mod find_points_tests {
    use crate::simplex::Mesh;
    use crate::transformation::find_points;
    use crate::element_type::ElementType;

    #[test]
    fn find_points_quad_2elem() {
        let m = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        // points: inside elem0, inside elem3, on shared edge, outside
        let pts = vec![0.25, 0.25, 0.75, 0.75, 0.5, 0.25, 5.0, 5.0];
        let (ids, xis) = find_points(&m, &pts, 4);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 3);
        assert!(ids[2] >= 0, "boundary point should be found");
        assert_eq!(ids[3], -1);
        assert!((xis[0][0] - 0.5).abs() < 1e-10 && (xis[0][1] - 0.5).abs() < 1e-10);
        assert!((xis[1][0] - 0.5).abs() < 1e-10 && (xis[1][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn find_points_tri_mesh() {
        // unit square split into two triangles (diagonal BL-TR)
        let m = Mesh::<2>::unit_square_tri(1);
        let pts = vec![0.1, 0.1, 0.9, 0.9, 0.7, 0.2];
        let (ids, _xis) = find_points(&m, &pts, 3);
        assert!(ids.iter().all(|&i| i >= 0), "all points inside the square");
    }

    #[test]
    fn find_points_hex_unit() {
        let m = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, true);
        let pts = vec![0.3, 0.2, 0.9, 0.5, 0.5, 0.5, 1.1, 0.5, 0.5];
        let (ids, xis) = find_points(&m, &pts, 3);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 0);
        assert_eq!(ids[2], -1);
        assert!((xis[0][0] - 0.3).abs() < 1e-10);
        assert!((xis[1][2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn find_points_tet_unit() {
        let m = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, true);
        let pts = vec![0.1, 0.1, 0.1, 0.5, 0.5, 0.5];
        let (ids, _xis) = find_points(&m, &pts, 2);
        assert_eq!(ids[0], 0, "point near origin is in the corner tet");
        let _ = ids[1]; // may be in either tet of the diagonal split
    }
}
