//! Common DG utilities — shared across all DG assembly modules.
//!
//! Extracted to eliminate duplicated copies of reference-element dispatch,
//! Jacobian computation, gradient transforms, and face-geometry helpers.
//!
//! Each function is individually documented so callers can find what they need
//! without reading the implementation.

use std::collections::HashMap;
use nalgebra::DMatrix;

use fem_element::{
    ReferenceElement,
    lagrange::{
        SegP1, SegP2, SegP3,
        TriP1,
        TetP1, TetP2,
        QuadQ1, QuadQ2, QuadQk,
        factory::QuadL2GL,
    },
    lagrange::factory::{TriPk, TetPk},
};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};

// ═══════════════════════════════════════════════════════════════════════════════
// Reference-element dispatch
// ═══════════════════════════════════════════════════════════════════════════════

/// Return the volume reference element for a given element type and polynomial
/// order.  Supports triangles (Tri3), quadrilaterals (Quad4), and tetrahedra
/// (Tet4) up to order 3, with dynamic-order QuadQk for Quad4 at order > 2.
// MFEM: FECollection::FiniteElementForGeometry
pub fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, 3) => Box::new(TriPk::new(3)),
        // Quad4 in the DG/L2 path uses the Gauss-Legendre tensor basis
        // (MFEM L2_FECollection / DG_FECollection default), on [0,1]².
        (ElementType::Quad4, 1) => Box::new(QuadL2GL::new(1)),
        (ElementType::Quad4, 2) => Box::new(QuadL2GL::new(2)),
        (ElementType::Quad4, 3) => Box::new(QuadL2GL::new(3)),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetPk::new(3)),
        _ => panic!("ref_elem_vol: unsupported ({et:?}, order={order})"),
    }
}

/// Dynamic-order volume reference element — wraps `ref_elem_vol` but falls
/// back to [`QuadQk`] for Quad4 at arbitrary order > 2.
// MFEM: FECollection::FiniteElementForGeometry
pub fn ref_elem_vol_dynamic(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Quad4, order) if order > 2 => Box::new(QuadQk::new_lex(order as usize)),
        _ => ref_elem_vol(et, order),
    }
}

/// Face reference element (Line2 for 2-D, Tri3 for 3-D).
// MFEM: FECollection::FiniteElementForGeometry (face)
pub fn ref_elem_face(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        (ElementType::Line2, 3) => Box::new(SegP3),
        (ElementType::Tri3, 1)  => Box::new(TriP1),
        _ => panic!("ref_elem_face: unsupported ({et:?}, order={order})"),
    }
}

/// Return a Crouzeix-Raviart reference element by type and order.
// MFEM: FECollection (Crouzeix-Raviart)
pub fn ref_elem_cr(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(fem_element::CrTri1),
        (ElementType::Tri3, 2) => Box::new(fem_element::CrTri2),
        (ElementType::Tet4, 1) => Box::new(fem_element::CrTet1),
        (ElementType::Tet4, 2) => Box::new(fem_element::CrTet2),
        _ => panic!("ref_elem_cr: unsupported ({et:?}, order={order})"),
    }
}

/// Return a Q1_rot (Rannacher–Turek) reference element for Quad4.
// MFEM: FECollection (Rannacher-Turek Q1_rot)
pub fn ref_elem_q1rot(et: ElementType) -> Box<dyn ReferenceElement> {
    match et {
        ElementType::Quad4 => Box::new(fem_element::Q1RotRef),
        _ => panic!("ref_elem_q1rot: unsupported {et:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Jacobian helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Affine Jacobian of a simplex (Tri/Tet) or centroid Jacobian for a bilinear
/// quad, returned as a `DMatrix` together with its determinant.
///
/// For elements with >3 nodes (quadrilaterals in 2-D) the centroid Jacobian of
/// the bilinear mapping on `[-1,1]²` is computed, scaled by 0.5 to match the
/// `[0,1]` reference-domain convention used by `phys_to_ref`.
// MFEM: CalcJacobian
pub fn simplex_jac<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    dim: usize,
) -> (DMatrix<f64>, f64) {
    if nodes.len() > 3 {
        // Quadrilateral — centroid Jacobian of bilinear mapping on [-1,1]²,
        // then scaled by 0.5 for [0,1]-based reference coordinates.
        let x: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(3)])[0]).collect();
        let y: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(3)])[1]).collect();
        let dxi  = [-0.5,  0.5,  0.5, -0.5];
        let deta = [-0.5, -0.5,  0.5,  0.5];
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for k in 0..4 {
            j[(0, 0)] += dxi[k]  * x[k];
            j[(0, 1)] += deta[k] * x[k];
            j[(1, 0)] += dxi[k]  * y[k];
            j[(1, 1)] += deta[k] * y[k];
        }
        let det = j.determinant();
        return (j, det);
    }
    // Simplex: affine mapping from [0,1]^dim
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
}

/// Bilinear quad Jacobian at a reference point `(xi, eta)` in `[-1,1]²`.
// MFEM: CalcJacobian (quad)
pub fn quad_jac_at(x: &[f64], y: &[f64], xi: f64, eta: f64) -> (DMatrix<f64>, f64) {
    let dxi  = [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)];
    let deta = [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)];
    let mut j = DMatrix::<f64>::zeros(2, 2);
    for k in 0..4 {
        j[(0, 0)] += dxi[k]  * x[k];
        j[(0, 1)] += deta[k] * x[k];
        j[(1, 0)] += dxi[k]  * y[k];
        j[(1, 1)] += deta[k] * y[k];
    }
    j *= 0.25;
    (j.clone(), j.determinant())
}

/// Per-point Jacobian of the bilinear quad map on the reference square `[0,1]²`
/// with the **topological** (CCW) node order `(0,0),(1,0),(1,1),(0,1)` —
/// the order MFEM's `ElementTransformation` uses for element nodes.  (The L2
/// solution basis is lexicographic; geometry is always topological.)
pub fn quad_jac_at_01(x: &[f64], y: &[f64], xi: f64, eta: f64) -> (DMatrix<f64>, f64) {
    // N1=(1-x)(1-y)@(0,0), N2=x(1-y)@(1,0), N3=xy@(1,1), N4=(1-x)y@(0,1)
    let dxi  = [-(1.0 - eta),  (1.0 - eta),  eta, -eta];
    let deta = [-(1.0 - xi),  -xi,   xi,  (1.0 - xi)];
    let mut j = DMatrix::<f64>::zeros(2, 2);
    for k in 0..4 {
        j[(0, 0)] += dxi[k]  * x[k];
        j[(0, 1)] += deta[k] * x[k];
        j[(1, 0)] += dxi[k]  * y[k];
        j[(1, 1)] += deta[k] * y[k];
    }
    (j.clone(), j.determinant())
}

/// Exact inverse of the bilinear quad map on `[0,1]²` via Newton iteration
/// (the affine `phys_to_ref` is only an approximation for bilinear maps; MFEM
/// uses `ElementTransformation::TransformBack` which converges to machine
/// precision).  Node order matches [`quad_jac_at_01`].
pub fn phys_to_ref_quad_01(
    x: &[f64],
    y: &[f64],
    xp: &[f64],
    xi0: &[f64],
) -> Vec<f64> {
    let mut xi = vec![xi0[0], xi0[1]];
    for _ in 0..12 {
        // Bilinear map value at (xi, eta): X = Σ N_k(ξ,η) x_k, topological
        // node order N1=(1-x)(1-y)@0, N2=x(1-y)@1, N3=xy@2, N4=(1-x)y@3
        // (matching quad_jac_at_01).
        let (nx0, nx1) = (1.0 - xi[0], xi[0]);
        let (ny0, ny1) = (1.0 - xi[1], xi[1]);
        let xv = nx0 * ny0 * x[0] + nx1 * ny0 * x[1] + nx1 * ny1 * x[2] + nx0 * ny1 * x[3];
        let yv = nx0 * ny0 * y[0] + nx1 * ny0 * y[1] + nx1 * ny1 * y[2] + nx0 * ny1 * y[3];
        let (j, _d) = quad_jac_at_01(x, y, xi[0], xi[1]);
        let det = j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)];
        if det.abs() < 1e-16 {
            break;
        }
        let fx = xp[0] - xv;
        let fy = xp[1] - yv;
        if fx * fx + fy * fy < 1e-26 {
            break;
        }
        let dxi = (j[(1, 1)] * fx - j[(0, 1)] * fy) / det;
        let deta = (-j[(1, 0)] * fx + j[(0, 0)] * fy) / det;
        xi[0] += dxi;
        xi[1] += deta;
    }
    xi
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gradient transforms
// ═══════════════════════════════════════════════════════════════════════════════

/// Transform reference-element gradients to physical gradients:
/// `∇_phys = J^{-T} ∇_ref`.
// MFEM: TransformGrad
pub fn xform_grads(
    jit: &DMatrix<f64>,
    gr: &[f64],
    gp: &mut [f64],
    n: usize,
    dim: usize,
) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += jit[(j, k)] * gr[i * dim + k];
            }
            gp[i * dim + j] = s;
        }
    }
}

/// Map a physical point back to reference coordinates:
/// `ξ = J^{-1}(x_phys − x_0)`.
// MFEM: PointToRef
pub fn phys_to_ref(
    jac: &DMatrix<f64>,
    x0: &[f64],
    xp: &[f64],
    dim: usize,
) -> Vec<f64> {
    let j_inv = match jac.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            eprintln!("warning: degenerate element in phys_to_ref, using identity");
            DMatrix::identity(dim, dim)
        }
    };
    let dx: Vec<f64> = (0..dim).map(|i| xp[i] - x0[i]).collect();
    let mut xi = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim {
            xi[i] += j_inv[(i, k)] * dx[k];
        }
    }
    xi
}

// ═══════════════════════════════════════════════════════════════════════════════
// Face geometry
// ═══════════════════════════════════════════════════════════════════════════════

/// 2-D face geometry: returns `(edge_length, unit_normal)`.
///
/// The normal is the 90° CCW rotation of the edge direction.
/// Use `orient_normal_outward` to guarantee outward orientation from the
/// adjacent element.
// MFEM: CalcOrtho (2D face Jacobian)
pub fn face_geom_2d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> (f64, Vec<f64>) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx * dx + dy * dy).sqrt();
    (len, vec![-dy / len, dx / len])
}

/// 3-D face geometry: returns `(face_area, unit_normal)` using the cross
/// product of two edge vectors.
// MFEM: CalcOrtho (3D face Jacobian)
pub fn face_geom_3d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> (f64, Vec<f64>) {
    let a = mesh.node_coords(nodes[0]);
    let b = mesh.node_coords(nodes[1]);
    let c = mesh.node_coords(nodes[2]);
    let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cr = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let area = 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2])
        .sqrt()
        .max(1e-30);
    let nrm = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2])
        .sqrt()
        .max(1e-30);
    (area, vec![cr[0] / nrm, cr[1] / nrm, cr[2] / nrm])
}

/// Ensure `normal` points outward from `elem` by checking against the element
/// centroid.  If `dot(normal, face_midpoint − centroid) < 0`, the normal
/// points inward → flip it.
// MFEM: no direct MFEM equivalent; computed from element geometry
pub fn orient_normal_outward<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    face_nodes: &[u32],
    normal: &mut [f64],
) {
    let dim = mesh.dim() as usize;
    let enodes = mesh.element_nodes(elem);
    let npe = enodes.len();
    // Element centroid
    let mut centroid = vec![0.0_f64; dim];
    for &n in enodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            centroid[d] += c[d];
        }
    }
    for d in 0..dim {
        centroid[d] /= npe as f64;
    }
    // Face midpoint
    let mut midpoint = vec![0.0_f64; dim];
    for &n in face_nodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            midpoint[d] += c[d];
        }
    }
    for d in 0..dim {
        midpoint[d] /= face_nodes.len() as f64;
    }
    // Check orientation
    let dot: f64 = (0..dim)
        .map(|d| normal[d] * (midpoint[d] - centroid[d]))
        .sum();
    if dot < 0.0 {
        for d in 0..dim {
            normal[d] = -normal[d];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mesh helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a map from face ID → owning element (one element per face).
///
/// The strategy: for each volume element, iterate its local faces (vertex
/// tuples), sort each tuple, and register the element in a global map keyed
/// by the sorted tuple.  Then for each `mesh.face` look up its sorted node
/// key to recover the owning element.
///
/// Returns a `HashMap<face_id, elem_id>` for boundary-face iteration.
// MFEM: Mesh::FaceToElementTable
pub fn build_face_elem_map<M: MeshTopology>(
    mesh: &M,
    dim: usize,
) -> HashMap<u32, u32> {
    let local_faces = |npe: usize| -> Vec<Vec<usize>> {
        match (npe, dim) {
            (3, 2) => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
            (4, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 3]],
            (4, 3) => vec![
                vec![1, 2, 3],
                vec![0, 2, 3],
                vec![0, 1, 3],
                vec![0, 1, 2],
            ],
            _ => vec![],
        }
    };

    let mut vol_face_map: HashMap<Vec<u32>, u32> = HashMap::new();
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let npe = nodes.len();
        for lf in local_faces(npe) {
            let mut key: Vec<u32> = lf.iter().map(|&k| nodes[k]).collect();
            key.sort_unstable();
            vol_face_map.entry(key).or_insert(e);
        }
    }

    let mut result = HashMap::new();
    for f in mesh.face_iter() {
        let fnodes = mesh.face_nodes(f);
        let mut key: Vec<u32> = fnodes.to_vec();
        key.sort_unstable();
        if let Some(&elem) = vol_face_map.get(&key) {
            result.insert(f, elem);
        }
    }
    result
}

/// Find an element that owns a face by scanning all elements.
/// A simpler but O(n_elem) alternative to `build_face_elem_map`.
///
/// Useful when only a single face lookup is needed.
// MFEM: Mesh::FaceToElement
pub fn find_face_elem<M: MeshTopology>(
    mesh: &M,
    _face_id: u32,
    face_nodes: &[u32],
) -> u32 {
    let mut fkey: Vec<u32> = face_nodes.to_vec();
    fkey.sort_unstable();
    for e in mesh.elem_iter() {
        let enodes = mesh.element_nodes(e);
        if enodes.len() < 3 {
            continue;
        }
        let count = fkey.iter().filter(|&n| enodes.contains(n)).count();
        if count >= 2 {
            return e;
        }
    }
    0
}
