//! Moment-fitting cut integration rules — 1:1 port of MFEM `MomentFittingIntRules`
//! (fem/intrules_cut.cpp, Mueller–Kummer–Oberlack 2013).
//!
//! Constructs quadrature rules for integration over implicit interfaces
//! (zero level set) and subdomains (level-set > 0) on cut elements, using
//! moment-fitting with SVD least squares.
//!
//! Reference domain convention: `[0,1]^d` (matching MFEM `Geometry::SEGMENT /
//! SQUARE / CUBE`).

use std::collections::HashMap;

use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::{FESpace, H1Space};

use super::div_free_3d_data::div_free_3d;

// MFEM `CutIntegrationRules` tolerances (double precision).
const TOL_1: f64 = 1e-12;
const TOL_2: f64 = 1e-15;

// ─── Basic helpers ────────────────────────────────────────────────────────────

/// Binomial coefficient `C(n, k)` (MFEM `Poly_1D::Binom(n)` semantics).
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut c = 1.0_f64;
    for i in 0..k {
        c *= (n - i) as f64 / (i + 1) as f64;
    }
    c
}

/// Gauss-Legendre rule on [0,1] with `n` points (MFEM
/// `QuadratureFunctions1D::GaussLegendre`). Points ascending.
fn gl_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    fem_element::quadrature::gauss_legendre_01(n)
}

/// Tensor-product rule on [0,1]^2, MFEM `IntegrationRule(irx, iry)` point
/// order: `j*nx+i` (x varies fastest).
fn tensor_rule_2d(nx: usize, ny: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (xs, wsx) = gl_01(nx);
    let (ys, wsy) = gl_01(ny);
    let mut pts = Vec::with_capacity(nx * ny);
    let mut wts = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            pts.push(vec![xs[i], ys[j]]);
            wts.push(wsx[i] * wsy[j]);
        }
    }
    (pts, wts)
}

/// Tensor-product rule on [0,1]^3, MFEM order `iz*nx*ny + iy*nx + ix`.
fn tensor_rule_3d(nx: usize, ny: usize, nz: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (xs, wsx) = gl_01(nx);
    let (ys, wsy) = gl_01(ny);
    let (zs, wsz) = gl_01(nz);
    let mut pts = Vec::with_capacity(nx * ny * nz);
    let mut wts = Vec::with_capacity(nx * ny * nz);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                pts.push(vec![xs[ix], ys[iy], zs[iz]]);
                wts.push(wsx[ix] * wsy[iy] * wsz[iz]);
            }
        }
    }
    (pts, wts)
}

/// Number of points of the MFEM `IntegrationRules.Get(geom, order)` rule.
pub fn rule_npts(geom_dim: usize, order: usize) -> usize {
    let n = order / 2 + 1;
    match geom_dim {
        1 => n,
        2 => n * n,
        3 => n * n * n,
        _ => unreachable!(),
    }
}

/// Exact degree of a Gauss rule with n points.
fn gl_exact_order(n: usize) -> usize {
    2 * n - 1
}

/// A quadrature rule: reference points + weights (MFEM `IntegrationRule`).
#[derive(Debug, Clone)]
pub struct CutRule {
    pub points: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    /// Exact polynomial order of the rule (MFEM `GetOrder`).
    pub order: usize,
}

impl CutRule {
    pub fn n_points(&self) -> usize {
        self.weights.len()
    }
}

// ─── Mesh geometry view ───────────────────────────────────────────────────────

// MFEM `Constants<Geometry::CUBE>::FaceVert[6][4]`.
const HEX_FACE_VERTS: [[usize; 4]; 6] = [
    [3, 2, 1, 0], // z=0
    [0, 1, 5, 4], // y=0
    [1, 2, 6, 5], // y=1
    [2, 3, 7, 6], // x=1
    [3, 0, 4, 7], // x=0
    [4, 5, 6, 7], // z=1
];

// MFEM `Constants<Geometry::SQUARE>::Edges[4][2]`.
const QUAD_EDGES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];

/// Minimal mesh geometry for the moment-fitting routines (Q1 elements).
///
/// `dim` = reference dimension (2 or 3), `sdim` = embedding dimension (2 for
/// the main 2-D mesh, 3 for 2-D faces of a 3-D mesh).
#[derive(Debug, Clone)]
pub struct CutGeom {
    pub dim: usize,
    pub sdim: usize,
    /// Element → vertex node ids.
    pub elem_verts: Vec<Vec<u32>>,
    /// Node coordinates (physical).
    pub coords: Vec<Vec<f64>>,
    // 3-D only:
    /// Total number of faces (interior + boundary), MFEM numbering.
    pub n_faces: usize,
    /// Element → global face ids (local face 0..5 → global).
    pub elem_faces: Vec<Vec<u32>>,
    /// Global face → vertex node ids (MFEM `FaceVert` order).
    pub face_verts: Vec<Vec<u32>>,
}

impl CutGeom {
    /// A 1-D mesh: `coords` = node positions (length n+1 for n segments), one
    /// segment per element (MFEM `inline-segment.mesh` style).
    pub fn from_1d(coords: &[f64]) -> Self {
        let mut elem_verts = Vec::with_capacity(coords.len() - 1);
        for i in 0..coords.len() - 1 {
            elem_verts.push(vec![i as u32, i as u32 + 1]);
        }
        let coords: Vec<Vec<f64>> = coords.iter().map(|&x| vec![x]).collect();
        CutGeom {
            dim: 1,
            sdim: 1,
            elem_verts,
            coords,
            n_faces: 0,
            elem_faces: vec![],
            face_verts: vec![],
        }
    }

    pub fn from_mesh2(mesh: &Mesh<2>) -> Self {
        let ne = mesh.n_elems();
        let mut elem_verts = Vec::with_capacity(ne);
        for e in 0..ne as u32 {
            elem_verts.push(mesh.element_nodes(e).to_vec());
        }
        let nv = mesh.n_nodes();
        let mut coords = Vec::with_capacity(nv);
        for n in 0..nv as u32 {
            coords.push(mesh.node_coords(n).to_vec());
        }
        CutGeom {
            dim: 2,
            sdim: 2,
            elem_verts,
            coords,
            n_faces: 0,
            elem_faces: vec![],
            face_verts: vec![],
        }
    }

    pub fn from_mesh3(mesh: &Mesh<3>) -> Self {
        let ne = mesh.n_elems();
        let mut elem_verts = Vec::with_capacity(ne);
        for e in 0..ne as u32 {
            elem_verts.push(mesh.element_nodes(e).to_vec());
        }
        let nv = mesh.n_nodes();
        let mut coords = Vec::with_capacity(nv);
        for n in 0..nv as u32 {
            coords.push(mesh.node_coords(n).to_vec());
        }
        // Global face list (interior + boundary), first-seen order.
        let mut face_map: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut elem_faces = Vec::with_capacity(ne);
        let mut face_verts: Vec<Vec<u32>> = vec![];
        for e in 0..ne as u32 {
            let verts = &elem_verts[e as usize];
            let mut faces = Vec::with_capacity(6);
            for lf in 0..6 {
                let fv: Vec<u32> = HEX_FACE_VERTS[lf].iter().map(|&i| verts[i]).collect();
                let mut key = fv.clone();
                key.sort_unstable();
                let id = match face_map.get(&key) {
                    Some(&id) => id,
                    None => {
                        let id = face_verts.len() as u32;
                        face_map.insert(key, id);
                        face_verts.push(fv);
                        id
                    }
                };
                faces.push(id);
            }
            elem_faces.push(faces);
        }
        CutGeom {
            dim: 3,
            sdim: 3,
            elem_verts,
            coords,
            n_faces: face_verts.len(),
            elem_faces,
            face_verts,
        }
    }

    /// A single 2-D quad element (the local face mesh of `ComputeFaceWeights`),
    /// embedded in 3-D. `coords` = the 4 face vertices (MFEM face order).
    pub fn from_face(coords: &[Vec<f64>]) -> Self {
        CutGeom {
            dim: 2,
            sdim: 3,
            elem_verts: vec![vec![0, 1, 2, 3]],
            coords: coords.to_vec(),
            n_faces: 0,
            elem_faces: vec![],
            face_verts: vec![],
        }
    }

    #[inline]
    pub fn vert_coord(&self, e: u32, vi: usize) -> &[f64] {
        &self.coords[self.elem_verts[e as usize][vi] as usize]
    }

    /// Q1 map: reference xi (len `dim`) → physical point (len `sdim`).
    /// Reference vertices follow the MFEM quad/hex ordering.
    pub fn map_phys(&self, e: u32, xi: &[f64]) -> Vec<f64> {
        let mut p = vec![0.0; self.sdim];
        match self.dim {
            1 => {
                let c0 = self.vert_coord(e, 0);
                let c1 = self.vert_coord(e, 1);
                let t = xi[0];
                p[0] = (1.0 - t) * c0[0] + t * c1[0];
            }
            2 => {
                let (x, y) = (xi[0], xi[1]);
                // MFEM quad vertices: 0:(0,0) 1:(1,0) 2:(1,1) 3:(0,1)
                let ref_verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
                for (k, &(rx, ry)) in ref_verts.iter().enumerate() {
                    let f = if rx == 0.0 { 1.0 - x } else { x }
                        * if ry == 0.0 { 1.0 - y } else { y };
                    let c = self.vert_coord(e, k);
                    for d in 0..self.sdim {
                        p[d] += f * c[d];
                    }
                }
            }
            3 => {
                let (x, y, z) = (xi[0], xi[1], xi[2]);
                // MFEM hex vertices (x-fastest within y rows, y rows reversed).
                let ref_verts = [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (0.0, 1.0, 1.0),
                ];
                for (k, &(rx, ry, rz)) in ref_verts.iter().enumerate() {
                    let f = if rx == 0.0 { 1.0 - x } else { x }
                        * if ry == 0.0 { 1.0 - y } else { y }
                        * if rz == 0.0 { 1.0 - z } else { z };
                    let c = self.vert_coord(e, k);
                    for d in 0..self.sdim {
                        p[d] += f * c[d];
                    }
                }
            }
            _ => unreachable!(),
        }
        p
    }

    /// Element Jacobian (constant for axis-aligned Q1 elements): sdim × dim.
    /// Gradients of the Q1 basis evaluated at the reference point (0,0,…):
    /// `∂φ_i/∂ξⱼ = ±1` for the vertex active in direction j, 0 otherwise
    /// (valid at any reference point for parallelepiped elements).
    pub fn jacobian(&self, e: u32) -> nalgebra::DMatrix<f64> {
        let mut j = nalgebra::DMatrix::zeros(self.sdim, self.dim);
        match self.dim {
            1 => {
                let c0 = self.vert_coord(e, 0);
                let c1 = self.vert_coord(e, 1);
                j[(0, 0)] = c1[0] - c0[0];
            }
            2 => {
                // MFEM quad vertices: 0:(0,0) 1:(1,0) 2:(1,1) 3:(0,1)
                // Gradient of φ_k at the reference point (0,0): ±1 in the
                // active directions, 0 otherwise.
                let grad = [
                    (-1.0, -1.0), // φ=(1-x)(1-y)
                    (1.0, 0.0),   // φ=x(1-y)
                    (0.0, 0.0),   // φ=xy
                    (0.0, 1.0),   // φ=(1-x)y
                ];
                for (k, &(dxi, deta)) in grad.iter().enumerate() {
                    let c = self.vert_coord(e, k);
                    for d in 0..self.sdim {
                        j[(d, 0)] += dxi * c[d];
                        j[(d, 1)] += deta * c[d];
                    }
                }
            }
            3 => {
                // MFEM hex vertices (see map_phys). Gradient of the trilinear
                // basis at the reference point (0,0,0).
                let grad = [
                    (-1.0, -1.0, -1.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                ];
                for (k, &(dxi, deta, dnu)) in grad.iter().enumerate() {
                    let c = self.vert_coord(e, k);
                    for d in 0..self.sdim {
                        j[(d, 0)] += dxi * c[d];
                        j[(d, 1)] += deta * c[d];
                        j[(d, 2)] += dnu * c[d];
                    }
                }
            }
            _ => unreachable!(),
        }
        j
    }

    /// |det J| (dim == sdim) or `sqrt(|det(JᵀJ)|)` (dim < sdim).
    pub fn det_j(&self, e: u32) -> f64 {
        let j = self.jacobian(e);
        if self.dim == self.sdim {
            j.determinant().abs()
        } else {
            let jt = j.transpose();
            let m = &jt * &j;
            m.determinant().abs().sqrt()
        }
    }

    /// Inverse Newton map (MFEM `InverseElementTransformation::NewtonSolve`,
    /// tolerances, initial guess = element center). dim == sdim: `J⁻¹`;
    /// dim < sdim: least-squares `(JᵀJ)⁻¹Jᵀ`.
    pub fn transform_back(&self, e: u32, pt: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut xip = vec![0.5; dim];
        let max_iter = 16;
        let ref_tol = 1e-15;
        let phys_rtol = 4e-15;
        let phys_tol = phys_rtol * pt.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        for _ in 0..max_iter {
            let y = self.map_phys(e, &xip);
            let mut err: f64 = 0.0;
            for d in 0..self.sdim {
                err = err.max((pt[d] - y[d]).abs());
            }
            if err < phys_tol {
                return xip;
            }
            let j = self.jacobian(e);
            let jt = j.transpose();
            let step_mat = if self.dim == self.sdim {
                j.try_inverse()
            } else {
                let m = &jt * &j;
                m.try_inverse().map(|minv| minv * jt)
            };
            let step_mat = match step_mat {
                Some(s) => s,
                None => return xip,
            };
            let mut dx = vec![0.0; dim];
            for r in 0..dim {
                for c in 0..self.sdim {
                    dx[r] += step_mat[(r, c)] * (pt[c] - y[c]);
                }
            }
            for d in 0..dim {
                xip[d] += dx[d];
            }
            let dx_norm = dx.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            if dx_norm < ref_tol {
                return xip;
            }
        }
        xip
    }

    /// Inverse map for a 1-D element (linear, exact).
    pub fn transform_back_1d(&self, e: u32, pt: &[f64]) -> f64 {
        let c0 = self.vert_coord(e, 0)[0];
        let c1 = self.vert_coord(e, 1)[0];
        (pt[0] - c0) / (c1 - c0)
    }

    /// Face transform: reference [0,1]^2 (face vertices) → physical 3-D.
    /// MFEM quad face vertices: 0:(0,0) 1:(1,0) 2:(1,1) 3:(0,1).
    pub fn face_map(&self, face: u32, xi: &[f64]) -> Vec<f64> {
        let (x, y) = (xi[0], xi[1]);
        let verts = &self.face_verts[face as usize];
        let mut p = vec![0.0; 3];
        for (k, &v) in verts.iter().enumerate() {
            let f = match k {
                0 => (1.0 - x) * (1.0 - y),
                1 => x * (1.0 - y),
                2 => x * y,
                3 => (1.0 - x) * y,
                _ => unreachable!(),
            };
            let c = &self.coords[v as usize];
            for d in 0..3 {
                p[d] += f * c[d];
            }
        }
        p
    }
}

// ─── Thin SVD (MFEM `DenseMatrixSVD` path) ───────────────────────────────────

struct SvdData {
    u: nalgebra::DMatrix<f64>,
    v_t: nalgebra::DMatrix<f64>,
    s: Vec<f64>,
}

fn svd_decompose(mat: &nalgebra::DMatrix<f64>) -> SvdData {
    let svd = nalgebra::linalg::SVD::new(mat.clone(), true, true);
    let u = svd.u.expect("SVD u");
    let v_t = svd.v_t.expect("SVD v_t");
    let s = svd.singular_values.iter().copied().collect();
    SvdData { u, v_t, s }
}

/// Underdetermined/overdetermined least-squares `Mat x ≈ rhs` with minimum
/// norm via thin SVD: `x = V Σ⁻¹ Uᵀ b`, dropping singular values ≤ TOL_1.
/// `k = min(m, n)` singular values are used (mathematically identical to
/// MFEM's full `DenseMatrixSVD` — extra zero singular values contribute 0).
fn svd_solve_lsq(svd: &SvdData, rhs: &[f64], n_out: usize) -> Vec<f64> {
    let m = rhs.len();
    let k = svd.u.ncols();
    assert!(svd.v_t.ncols() >= n_out, "v_t cols {} < n_out {}", svd.v_t.ncols(), n_out);
    let mut temp = vec![0.0; k];
    for i in 0..k {
        let mut acc = 0.0;
        for j in 0..m {
            acc += svd.u[(j, i)] * rhs[j];
        }
        temp[i] = acc;
    }
    let mut temp2 = vec![0.0; k];
    for i in 0..k {
        if svd.s[i] > TOL_1 {
            temp2[i] = temp[i] / svd.s[i];
        }
    }
    let mut x = vec![0.0; n_out];
    for ip in 0..n_out {
        let mut acc = 0.0;
        for i in 0..k {
            acc += svd.v_t[(i, ip)] * temp2[i];
        }
        x[ip] = acc;
    }
    x
}

// ─── Level-set H1(lsOrder) projection data ───────────────────────────────────

/// Nodal H1(lsOrder) projection of the level set + element dof map.
struct LsData {
    /// Element → global dof ids (main mesh) or local 0..n-1 (face).
    elem_dofs: Vec<Vec<u32>>,
    /// Nodal coefficient values (global dof index → value).
    coeffs: Vec<f64>,
    /// Reference element for gradient evaluation (QuadQk / HexQk).
    ref_elem: Box<dyn fem_element::ReferenceElement>,
    /// Reference dimension (2 or 3).
    dim: usize,
}

impl LsData {
    fn build_main_2d(mesh: &Mesh<2>, ls_order: usize, level_set: &dyn Fn(&[f64]) -> f64) -> Self {
        let space = H1Space::new(mesh.clone(), ls_order as u8);
        let coeffs = space.interpolate(&|x: &[f64]| level_set(x)).as_slice().to_vec();
        let ne = mesh.n_elems();
        let mut elem_dofs = Vec::with_capacity(ne);
        for e in 0..ne as u32 {
            elem_dofs.push(space.element_dofs(e).to_vec());
        }
        let ref_elem: Box<dyn fem_element::ReferenceElement> =
            Box::new(fem_element::lagrange::QuadQk::new(ls_order));
        LsData { elem_dofs, coeffs, ref_elem, dim: 2 }
    }

    fn build_main_3d(mesh: &Mesh<3>, ls_order: usize, level_set: &dyn Fn(&[f64]) -> f64) -> Self {
        let space = H1Space::new(mesh.clone(), ls_order as u8);
        let coeffs = space.interpolate(&|x: &[f64]| level_set(x)).as_slice().to_vec();
        let ne = mesh.n_elems();
        let mut elem_dofs = Vec::with_capacity(ne);
        for e in 0..ne as u32 {
            elem_dofs.push(space.element_dofs(e).to_vec());
        }
        let ref_elem: Box<dyn fem_element::ReferenceElement> =
            Box::new(fem_element::lagrange::HexQk::new(ls_order));
        LsData { elem_dofs, coeffs, ref_elem, dim: 3 }
    }

    /// Level-set projection on a 2-D face embedded in 3-D: GLL-node
    /// interpolation (MFEM's local face mesh `ProjectCoefficient`).
    fn build_face(
        geom: &CutGeom,
        ls_order: usize,
        level_set: &dyn Fn(&[f64]) -> f64,
    ) -> Self {
        let ref_elem: Box<dyn fem_element::ReferenceElement> =
            Box::new(fem_element::lagrange::QuadQk::new(ls_order));
        let nodes = ref_elem.dof_coords();
        // Bilinear map from the face reference [0,1]^2 to physical 3-D, using
        // the face vertices in MFEM face order (geom.coords[0..4]).
        let ref_verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut coeffs = Vec::with_capacity(nodes.len());
        for n in &nodes {
            let mut p = [0.0; 3];
            for (k, &(rx, ry)) in ref_verts.iter().enumerate() {
                let f = if rx == 0.0 { 1.0 - n[0] } else { n[0] }
                    * if ry == 0.0 { 1.0 - n[1] } else { n[1] };
                let c = &geom.coords[k];
                for d in 0..3 {
                    p[d] += f * c[d];
                }
            }
            coeffs.push(level_set(&p));
        }
        let elem_dofs = vec![(0..nodes.len() as u32).collect()];
        LsData { elem_dofs, coeffs, ref_elem, dim: 2 }
    }

    /// Reference gradient of the projected level set at reference point xi.
    fn grad_ref(&self, e: u32, xi: &[f64]) -> Vec<f64> {
        let n = self.ref_elem.n_dofs();
        let mut grads = vec![0.0; n * self.dim];
        // QuadQk uses the [0,1]^2 reference domain (matching MFEM's H1 quad);
        // HexQk uses [-1,1]^3, so map the [0,1]^3 integration point there
        // (and rescale the gradient back to the [0,1]^3 convention, where
        // d/dξ = 2·d/dxi_hex).
        let scale = if self.dim == 3 { 2.0 } else { 1.0 };
        let xi_e = if self.dim == 3 {
            vec![2.0 * xi[0] - 1.0, 2.0 * xi[1] - 1.0, 2.0 * xi[2] - 1.0]
        } else {
            xi.to_vec()
        };
        self.ref_elem.eval_grad_basis(&xi_e, &mut grads);
        let dofs = &self.elem_dofs[e as usize];
        let mut g = vec![0.0; self.dim];
        for (i, &d) in dofs.iter().enumerate() {
            let c = self.coeffs[d as usize];
            for j in 0..self.dim {
                g[j] += grads[i * self.dim + j] * c;
            }
        }
        for j in 0..self.dim {
            g[j] *= scale;
        }
        g
    }

    /// Physical gradient: J⁻ᵀ grad_ref (MFEM `GridFunction::GetGradient`).
    fn grad_phys(&self, e: u32, xi: &[f64], geom: &CutGeom) -> Vec<f64> {
        let gr = self.grad_ref(e, xi);
        let j = geom.jacobian(e);
        let jinv_t = j.try_inverse().expect("J invertible").transpose();
        let mut gp = vec![0.0; self.dim];
        for i in 0..self.dim {
            for k in 0..self.dim {
                gp[i] += jinv_t[(i, k)] * gr[k];
            }
        }
        gp
    }
}

// ─── Moment-fitting rules ─────────────────────────────────────────────────────

/// Moment-fitting integration rules (1:1 port of MFEM `MomentFittingIntRules`).
///
/// The candidate integration points (`ir`) are fixed after Init*; weights are
/// recomputed per element by moment-fitting.
pub struct MomentFitting<'a> {
    order: usize,
    ls_order: usize,
    level_set: &'a dyn Fn(&[f64]) -> f64,
    /// Candidate integration points (reference), fixed after Init*.
    ir_points: Vec<Vec<f64>>,
    /// Candidate weights (updated per element by Compute*).
    ir_weights: Vec<f64>,
    /// Exact order of the candidate rule.
    ir_order: usize,
    /// Number of surface basis functions (2-D/3-D).
    n_basis: usize,
    /// Number of volume basis functions (2-D/3-D).
    n_basis_volume: usize,
    /// Precomputed SVD of the volume moment matrix (InitVolume).
    volume_svd: Option<SvdData>,
    // 3-D face cache:
    face_ip: Vec<Vec<f64>>,
    face_weights: Vec<Vec<f64>>,
    face_weights_comp: Vec<f64>,
    /// Mesh geometry.
    geom: CutGeom,
    /// Level-set H1(lsOrder) projection (None in 1-D — not needed).
    ls: Option<LsData>,
}

impl<'a> MomentFitting<'a> {
    /// Moment fitting on a 1-D mesh (segments). `coords` = node positions.
    pub fn new_1d(
        coords: &[f64],
        order: usize,
        ls_order: usize,
        level_set: &'a dyn Fn(&[f64]) -> f64,
    ) -> Self {
        let geom = CutGeom::from_1d(coords);
        Self::mk(order, ls_order, level_set, geom, None)
    }

    /// Moment fitting on a 2-D mesh.
    pub fn new_2d(
        mesh: &Mesh<2>,
        order: usize,
        ls_order: usize,
        level_set: &'a dyn Fn(&[f64]) -> f64,
    ) -> Self {
        let geom = CutGeom::from_mesh2(mesh);
        let ls = LsData::build_main_2d(mesh, ls_order, level_set);
        Self::mk(order, ls_order, level_set, geom, Some(ls))
    }

    /// Moment fitting on a 3-D mesh.
    pub fn new_3d(
        mesh: &Mesh<3>,
        order: usize,
        ls_order: usize,
        level_set: &'a dyn Fn(&[f64]) -> f64,
    ) -> Self {
        let geom = CutGeom::from_mesh3(mesh);
        let ls = LsData::build_main_3d(mesh, ls_order, level_set);
        Self::mk(order, ls_order, level_set, geom, Some(ls))
    }

    /// Moment fitting on a single 2-D face embedded in 3-D (recursion of
    /// `ComputeFaceWeights`). `coords` = 4 face vertices (MFEM face order).
    fn new_face(
        coords: &[Vec<f64>],
        order: usize,
        ls_order: usize,
        level_set: &'a dyn Fn(&[f64]) -> f64,
    ) -> Self {
        let geom = CutGeom::from_face(coords);
        let ls = LsData::build_face(&geom, ls_order, level_set);
        Self::mk(order, ls_order, level_set, geom, Some(ls))
    }

    fn mk(
        order: usize,
        ls_order: usize,
        level_set: &'a dyn Fn(&[f64]) -> f64,
        geom: CutGeom,
        ls: Option<LsData>,
    ) -> Self {
        MomentFitting {
            order,
            ls_order,
            level_set,
            ir_points: vec![],
            ir_weights: vec![],
            ir_order: 0,
            n_basis: 0,
            n_basis_volume: 0,
            volume_svd: None,
            face_ip: vec![],
            face_weights: vec![],
            face_weights_comp: vec![],
            geom,
            ls,
        }
    }

    fn dim(&self) -> usize {
        self.geom.dim
    }

    fn eval_ls(&self, e: u32, xi: &[f64]) -> f64 {
        let p = self.geom.map_phys(e, xi);
        (self.level_set)(&p)
    }

    // ── Basis functions ──────────────────────────────────────────────────────

    /// Divergence-free basis on [-1,1]^2 (MFEM `DivFreeBasis2D`).
    fn div_free_basis_2d(&self, ip: &[f64]) -> Vec<[f64; 2]> {
        let o = self.order;
        let nb = self.n_basis;
        let x = -1.0 + 2.0 * ip[0];
        let y = -1.0 + 2.0 * ip[1];
        let mut shape = vec![[0.0; 2]; nb];
        for c in 0..=o {
            shape[2 * c][1] = x.powi(c as i32);
            shape[2 * c + 1][0] = y.powi(c as i32);
        }
        let mut count = 2 * o + 2;
        for c in 1..=o {
            for expo in (1..=c).rev() {
                let ce = binom(c, expo);
                let ce1 = binom(c, expo - 1);
                shape[count][0] = ce * x.powi(expo as i32) * y.powi((c - expo) as i32);
                shape[count][1] = -ce1 * x.powi((expo - 1) as i32) * y.powi((c - expo + 1) as i32);
                count += 1;
            }
        }
        shape
    }

    /// Orthogonalized divergence-free basis (MFEM `OrthoBasis2D`): modified
    /// Gram-Schmidt w.r.t. the `2*Order+1` quadrature rule. Returns the basis
    /// evaluated at `xi` (each call rebuilds `shapeMFN` like MFEM).
    fn ortho_basis_2d(&self, xi: &[f64]) -> Vec<[f64; 2]> {
        let nb = self.n_basis;
        let mgs_n = self.order + 1; // (2*Order+1)/2+1 points per dim
        let (mgs_pts, mgs_wts) = tensor_rule_2d(mgs_n, mgs_n);
        let npts = mgs_pts.len();
        // shapeMFN[p][row] = DivFreeBasis2D at quad point p
        let mut shape_mfn: Vec<Vec<[f64; 2]>> = Vec::with_capacity(npts);
        for p in 0..npts {
            shape_mfn.push(self.div_free_basis_2d(&mgs_pts[p]));
        }
        let mut shape = self.div_free_basis_2d(xi);
        for step in 1..nb {
            let mut den = 0.0;
            let mut num = 0.0;
            for p in 0..npts {
                let u = shape_mfn[p][step];
                let v = shape_mfn[p][step - 1];
                den += (v[0] * v[0] + v[1] * v[1]) * mgs_wts[p];
                num += (u[0] * v[0] + u[1] * v[1]) * mgs_wts[p];
            }
            let coeff = if den != 0.0 { num / den } else { 0.0 };
            for count in step..nb {
                shape[count][0] += coeff * shape[step - 1][0];
                shape[count][1] += coeff * shape[step - 1][1];
            }
            for p in 0..npts {
                for count in step..nb {
                    shape_mfn[p][count][0] += coeff * shape_mfn[p][step - 1][0];
                    shape_mfn[p][count][1] += coeff * shape_mfn[p][step - 1][1];
                }
            }
        }
        shape
    }

    /// Orthonormalized divergence-free basis on [-1,1]^3 (MFEM
    /// `DivFreeBasis::GetDivFree3DBasis`).
    fn ortho_basis_3d(&self, xi: &[f64]) -> Vec<[f64; 3]> {
        let x = -1.0 + 2.0 * xi[0];
        let y = -1.0 + 2.0 * xi[1];
        let z = -1.0 + 2.0 * xi[2];
        div_free_3d(x, y, z, self.order)
    }

    /// Monomial basis on [-1,1]^2 (MFEM `Basis2D`).
    fn basis_2d(&self, xi: &[f64]) -> Vec<f64> {
        let o = self.order;
        let x = -1.0 + 2.0 * xi[0];
        let y = -1.0 + 2.0 * xi[1];
        let mut shape = vec![0.0; self.n_basis_volume];
        let mut count = 0;
        for c in 0..=o {
            for expo in 0..=c {
                shape[count] = x.powi(expo as i32) * y.powi((c - expo) as i32);
                count += 1;
            }
        }
        shape
    }

    /// Antiderivatives of the monomial basis on [-1,1]^2 (MFEM `BasisAD2D`).
    fn basis_ad_2d(&self, xi: &[f64]) -> Vec<[f64; 2]> {
        let o = self.order;
        let x = -1.0 + 2.0 * xi[0];
        let y = -1.0 + 2.0 * xi[1];
        let mut shape = vec![[0.0; 2]; self.n_basis_volume];
        let mut count = 0;
        for c in 0..=o {
            for expo in 0..=c {
                shape[count][0] = 0.25 * x.powi((expo + 1) as i32) * y.powi((c - expo) as i32)
                    / (expo + 1) as f64;
                shape[count][1] = 0.25 * x.powi(expo as i32) * y.powi((c - expo + 1) as i32)
                    / (c - expo + 1) as f64;
                count += 1;
            }
        }
        shape
    }

    /// Monomial basis on [-1,1]^3 (MFEM `Basis3D`).
    fn basis_3d(&self, xi: &[f64]) -> Vec<f64> {
        let o = self.order;
        let x = -1.0 + 2.0 * xi[0];
        let y = -1.0 + 2.0 * xi[1];
        let z = -1.0 + 2.0 * xi[2];
        let mut shape = vec![0.0; self.n_basis_volume];
        let mut count = 0;
        for c in 0..=o {
            for expo in 0..=c {
                for expo2 in 0..=(c - expo) {
                    shape[count] = x.powi(expo as i32) * y.powi(expo2 as i32)
                        * z.powi((c - expo - expo2) as i32);
                    count += 1;
                }
            }
        }
        shape
    }

    /// Antiderivatives of the monomial basis on [-1,1]^3 (MFEM `BasisAD3D`).
    fn basis_ad_3d(&self, xi: &[f64]) -> Vec<[f64; 3]> {
        let o = self.order;
        let x = -1.0 + 2.0 * xi[0];
        let y = -1.0 + 2.0 * xi[1];
        let z = -1.0 + 2.0 * xi[2];
        let mut shape = vec![[0.0; 3]; self.n_basis_volume];
        let mut count = 0;
        for c in 0..=o {
            for expo in 0..=c {
                for expo2 in 0..=(c - expo) {
                    shape[count][0] = x.powi((expo + 1) as i32) * y.powi(expo2 as i32)
                        * z.powi((c - expo - expo2) as i32)
                        / (6.0 * (expo + 1) as f64);
                    shape[count][1] = x.powi(expo as i32) * y.powi((expo2 + 1) as i32)
                        * z.powi((c - expo - expo2) as i32)
                        / (6.0 * (expo2 + 1) as f64);
                    shape[count][2] = x.powi(expo as i32) * y.powi(expo2 as i32)
                        * z.powi((c - expo - expo2 + 1) as i32)
                        / (6.0 * (c - expo + expo2 + 1) as f64); // MFEM BasisAD3D z-denom: c-expo+expo2+1
                    count += 1;
                }
            }
        }
        shape
    }

    // ── Init (MFEM `InitSurface` / `InitVolume` / `Clear`) ──────────────────

    /// MFEM `InitSurface`: candidate points for the given order.
    fn init_surface(&mut self, order: usize) {
        self.order = order;
        let d = self.dim();
        if d == 1 {
            self.n_basis = 0;
            let (pts, wts) = gl_01(1);
            self.ir_points = pts.iter().map(|&x| vec![x]).collect();
            self.ir_weights = wts;
            self.ir_order = 1;
        } else {
            self.n_basis = match d {
                2 => 2 * (order + 1) + order * (order + 1) / 2,
                _ => match order {
                    0 => 3,
                    1 => 11,
                    2 => 26,
                    3 => 50,
                    4 => 85,
                    5 => 133,
                    6 => 196,
                    _ => 276,
                },
            };
            let mut qorder = 0;
            while rule_npts(d, qorder) <= self.n_basis {
                qorder += 1;
            }
            let n = qorder / 2 + 1;
            let (pts, wts) = if d == 2 {
                tensor_rule_2d(n, n)
            } else {
                tensor_rule_3d(n, n, n)
            };
            self.ir_points = pts;
            self.ir_weights = wts;
            self.ir_order = gl_exact_order(n);
        }
    }

    /// MFEM `InitVolume`: candidate points (order+1 surface basis) + SVD.
    fn init_volume(&mut self, order: usize) {
        self.init_surface(order + 1);
        self.order = order;
        let d = self.dim();
        if d == 1 {
            self.n_basis_volume = 0;
            let n = order / 2 + 1;
            let (pts, wts) = gl_01(n);
            self.ir_points = pts.iter().map(|&x| vec![x]).collect();
            self.ir_weights = wts;
            self.ir_order = gl_exact_order(n);
        } else {
            self.n_basis_volume = if d == 2 {
                (order + 1) * (order + 2) / 2
            } else {
                let mut s = 0;
                for p in 0..=order {
                    s += (p + 1) * (p + 2) / 2;
                }
                s
            };
            let npts = self.ir_points.len();
            let mut mat = nalgebra::DMatrix::zeros(self.n_basis_volume, npts);
            for ip in 0..npts {
                let shape = if d == 2 {
                    self.basis_2d(&self.ir_points[ip])
                } else {
                    self.basis_3d(&self.ir_points[ip])
                };
                for (r, &v) in shape.iter().enumerate() {
                    mat[(r, ip)] = v;
                }
            }
            self.volume_svd = Some(svd_decompose(&mat));
        }
    }

    /// Clear the per-element state (MFEM `Clear`).
    fn clear(&mut self) {
        self.n_basis = 0;
        self.n_basis_volume = 0;
        self.volume_svd = None;
        self.face_ip = vec![];
        self.face_weights = vec![];
        self.face_weights_comp = vec![];
    }
}

// ─── Per-element weight computations (1-D) ───────────────────────────────────

impl<'a> MomentFitting<'a> {
    /// MFEM `ComputeSurfaceWeights1D`.
    fn compute_surface_weights_1d(&mut self, e: u32) {
        let ip0 = [0.0];
        let ip1 = [1.0];
        let v0 = self.eval_ls(e, &ip0);
        let v1 = self.eval_ls(e, &ip1);
        let (xi_val, w_val) = if v0 * v1 < 0.0 {
            let mut a = ip0;
            let mut b = ip1;
            let mut ip2 = [0.5];
            loop {
                let v = self.eval_ls(e, &ip2);
                if !(v > TOL_1 || v < -TOL_1) {
                    break;
                }
                if self.eval_ls(e, &a) * v < 0.0 {
                    b = ip2;
                } else {
                    a = ip2;
                }
                ip2 = [(a[0] + b[0]) / 2.0];
            }
            (ip2[0], 1.0 / self.geom.det_j(e))
        } else if v0 > 0.0 && v1 <= TOL_1 {
            (1.0, 1.0 / self.geom.det_j(e))
        } else if v1 > 0.0 && v0 <= TOL_1 {
            (0.0, 1.0 / self.geom.det_j(e))
        } else {
            (0.5, 0.0)
        };
        self.ir_points[0][0] = xi_val;
        self.ir_weights[0] = w_val;
    }

    /// 1-D bisection root of the level set (MFEM free function `bisect`).
    fn bisect_1d(&self, e: u32) -> f64 {
        let ip0 = [0.0];
        let ip1 = [1.0];
        let v0 = self.eval_ls(e, &ip0);
        let v1 = self.eval_ls(e, &ip1);
        let mut x = 0.5;
        if v0 * v1 < 0.0 {
            let mut a = ip0;
            let mut b = ip1;
            let mut ip2 = [0.5];
            loop {
                let v = self.eval_ls(e, &ip2);
                if !(v > 1e-12 || v < -1e-12) {
                    break;
                }
                if self.eval_ls(e, &a) * v < 0.0 {
                    b = ip2;
                } else {
                    a = ip2;
                }
                ip2 = [(a[0] + b[0]) / 2.0];
            }
            x = ip2[0];
        }
        x
    }

    /// MFEM `ComputeVolumeWeights1D`.
    fn compute_volume_weights_1d(&mut self, e: u32) {
        let n = self.ir_points.len();
        let (gpts, gwts) = gl_01(n);
        let ip0 = [0.0];
        let ip1 = [1.0];
        let v0 = self.eval_ls(e, &ip0);
        let v1 = self.eval_ls(e, &ip1);
        if v0 * v1 < 0.0 {
            let (start, length) = if v0 > 0.0 {
                (0.0, self.bisect_1d(e))
            } else {
                (self.bisect_1d(e), 1.0 - self.bisect_1d(e))
            };
            for ip in 0..n {
                self.ir_points[ip][0] = start + gpts[ip] * length;
                self.ir_weights[ip] = gwts[ip] * length;
            }
        } else if v0 <= -TOL_1 || v1 <= -TOL_1 {
            for ip in 0..n {
                self.ir_points[ip][0] = gpts[ip];
                self.ir_weights[ip] = 0.0;
            }
        } else {
            for ip in 0..n {
                self.ir_points[ip][0] = gpts[ip];
                self.ir_weights[ip] = gwts[ip];
            }
        }
    }
}
// ─── Per-element weight computations (2-D) ───────────────────────────────────

/// Layout of an edge/face relative to the level set (MFEM `Layout`).
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    Inside,
    Intersected,
    Outside,
}

impl<'a> MomentFitting<'a> {
    /// Walk the 4 edges of the quad element: compute the edge classification,
    /// the stored endpoints (bisected for intersected edges), and the
    /// `edge_int` flags. Shared by `ComputeSurfaceWeights2D` and
    /// `ComputeVolumeWeights2D` (MFEM code is duplicated there; we factor the
    /// identical walk).
    #[allow(clippy::type_complexity)]
    fn edge_walk_2d(
        &self,
        e: u32,
    ) -> (
        [[f64; 3]; 4],
        [[f64; 3]; 4],
        [f64; 4],
        [bool; 4],
        bool,
        bool,
    ) {
        let mut point_a = [[0.0; 3]; 4];
        let mut point_b = [[0.0; 3]; 4];
        let mut edgelength = [0.0; 4];
        let mut edge_int = [false; 4];
        let mut interior = true;
        let mut element_int = false;
        for edge in 0..4 {
            let (ia, ib) = (QUAD_EDGES[edge][0], QUAD_EDGES[edge][1]);
            let pa = self.geom.vert_coord(e, ia).to_vec();
            let pb = self.geom.vert_coord(e, ib).to_vec();
            let mut edgevec = vec![0.0; self.geom.sdim];
            for d in 0..self.geom.sdim {
                edgevec[d] = pa[d] - pb[d];
            }
            edgelength[edge] = edgevec.iter().map(|v| v * v).sum::<f64>().sqrt();

            let ipa = self.geom.transform_back(e, &pa);
            let ipb = self.geom.transform_back(e, &pb);
            let va = self.eval_ls(e, &ipa);
            let vb = self.eval_ls(e, &ipb);

            if va < -TOL_1 || vb < -TOL_1 {
                interior = false;
            }

            let mut p0 = pa.clone();
            let mut p1 = pb.clone();
            let layout = if va > -TOL_1 && vb > -TOL_1 {
                Layout::Inside
            } else if va > TOL_2 && vb <= 0.0 {
                Layout::Intersected
            } else if va <= 0.0 && vb > TOL_2 {
                std::mem::swap(&mut p0, &mut p1);
                Layout::Intersected
            } else {
                Layout::Outside
            };

            if layout == Layout::Intersected {
                // Bisect to the level-set zero on the edge (physical space).
                let mut pc = p0.clone();
                let mut mid = pc.clone();
                for d in 0..self.geom.sdim {
                    mid[d] = (pc[d] + p1[d]) / 2.0;
                }
                let mut ip = self.geom.transform_back(e, &mid);
                loop {
                    let v = self.eval_ls(e, &ip);
                    if !(v > TOL_1 || v < -TOL_1) {
                        break;
                    }
                    if v > TOL_1 {
                        pc = mid.clone();
                    } else {
                        p1 = mid.clone();
                    }
                    for d in 0..self.geom.sdim {
                        mid[d] = (pc[d] + p1[d]) / 2.0;
                    }
                    ip = self.geom.transform_back(e, &mid);
                }
                p1 = mid;
            }

            for d in 0..3 {
                point_a[edge][d] = if d < self.geom.sdim { p0[d] } else { 0.0 };
                point_b[edge][d] = if d < self.geom.sdim { p1[d] } else { 0.0 };
            }
            edge_int[edge] = layout == Layout::Inside || layout == Layout::Intersected;
            if edge_int[edge] {
                element_int = true;
            }
        }
        (point_a, point_b, edgelength, edge_int, interior, element_int)
    }

    /// 2-D edge unit normal in the reference plane (MFEM `ComputeSurfaceWeights2D`).
    fn edge_normal_2d(edge: usize) -> [f64; 2] {
        let mut normal = [0.0; 2];
        if edge == 0 || edge == 2 {
            normal[1] = 1.0;
        }
        if edge == 1 || edge == 3 {
            normal[0] = 1.0;
        }
        if edge == 0 || edge == 3 {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
        }
        normal
    }

    /// MFEM `ComputeSurfaceWeights2D`.
    fn compute_surface_weights_2d(&mut self, e: u32) {
        let nb = self.n_basis;
        let npts = self.ir_points.len();
        let (point_a, point_b, edgelength, edge_int, interior, _) = self.edge_walk_2d(e);
        let mut element_int = false;
        for &ei in &edge_int {
            if ei {
                element_int = true;
            }
        }

        let mut rhs = vec![0.0; nb];
        let mut mat = nalgebra::DMatrix::zeros(nb, npts);

        // Integrate over the 1-D edges.
        for edge in 0..4 {
            if edge_int[edge] && !interior {
                element_int = true;
                let normal = Self::edge_normal_2d(edge);
                let p0 = point_a[edge];
                let p1 = point_b[edge];
                let (seg_pts, seg_wts) = gl_01(2 * self.order + 1);
                for (ip, &w) in seg_pts.iter().zip(seg_wts.iter()) {
                    let mut dist = [0.0; 3];
                    for d in 0..3 {
                        dist[d] = p1[d] - p0[d];
                    }
                    let mut point = [0.0; 3];
                    for d in 0..3 {
                        point[d] = p0[d] + dist[d] * ip;
                    }
                    let intpoint = self.geom.transform_back(e, &point);
                    let shapes = self.ortho_basis_2d(&intpoint);
                    let dist_len =
                        (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]).sqrt();
                    for dof in 0..nb {
                        let grad = shapes[dof];
                        rhs[dof] -= (grad[0] * normal[0] + grad[1] * normal[1]) * w
                            * dist_len
                            / edgelength[edge];
                    }
                }
            }
        }

        // Integrate over the area for the interface term, form the matrix.
        if element_int && !interior {
            for ip in 0..npts {
                let xi = &self.ir_points[ip];
                let grad_ref = self.ls.as_ref().expect("ls").grad_ref(e, xi);
                let mut normal = [0.0; 2];
                let nrm = (grad_ref[0] * grad_ref[0] + grad_ref[1] * grad_ref[1]).sqrt();
                if nrm != 0.0 {
                    normal[0] = -grad_ref[0] / nrm;
                    normal[1] = -grad_ref[1] / nrm;
                }
                let shapes = self.ortho_basis_2d(xi);
                for dof in 0..nb {
                    mat[(dof, ip)] = shapes[dof][0] * normal[0] + shapes[dof][1] * normal[1];
                }
            }
            // Solve the underdetermined linear system.
            let svd = svd_decompose(&mat);
            let weights = svd_solve_lsq(&svd, &rhs, npts);
            self.ir_weights = weights;
        } else {
            for ip in 0..npts {
                self.ir_weights[ip] = 0.0;
            }
        }
    }

    /// MFEM `ComputeVolumeWeights2D` (needs the surface rule `sir`).
    fn compute_volume_weights_2d(&mut self, e: u32, sir: Option<&CutRule>) {
        let nb = self.n_basis_volume;
        let npts = self.ir_points.len();
        let (point_a, point_b, edgelength, edge_int, interior, _) = self.edge_walk_2d(e);
        let mut element_int = false;
        for &ei in &edge_int {
            if ei {
                element_int = true;
            }
        }

        let mut rhs = vec![0.0; nb];

        // Integrate over the 1-D edges (divergence theorem).
        for edge in 0..4 {
            if edge_int[edge] && !interior {
                element_int = true;
                let normal = Self::edge_normal_2d(edge);
                let p0 = point_a[edge];
                let p1 = point_b[edge];
                let (seg_pts, seg_wts) = gl_01(2 * self.order + 1);
                for (ip, &w) in seg_pts.iter().zip(seg_wts.iter()) {
                    let mut dist = [0.0; 3];
                    for d in 0..3 {
                        dist[d] = p1[d] - p0[d];
                    }
                    let mut point = [0.0; 3];
                    for d in 0..3 {
                        point[d] = p0[d] + dist[d] * ip;
                    }
                    let intpoint = self.geom.transform_back(e, &point);
                    let shapes = self.basis_ad_2d(&intpoint);
                    let dist_len =
                        (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]).sqrt();
                    for dof in 0..nb {
                        let adiv = shapes[dof];
                        rhs[dof] += (adiv[0] * normal[0] + adiv[1] * normal[1]) * w
                            * dist_len
                            / edgelength[edge];
                    }
                }
            }
        }

        // Integrate over the interface using the surface rule.
        if element_int && !interior {
            let sir = sir.expect("surface rule for volume weights");
            let np = sir.n_points();
            for ip in 0..np {
                let xi = &sir.points[ip];
                let grad_ref = self.ls.as_ref().expect("ls").grad_ref(e, xi);
                let mut normal = [0.0; 2];
                let nrm = (grad_ref[0] * grad_ref[0] + grad_ref[1] * grad_ref[1]).sqrt();
                if nrm != 0.0 {
                    normal[0] = -grad_ref[0] / nrm;
                    normal[1] = -grad_ref[1] / nrm;
                }
                let shapes = self.basis_ad_2d(xi);
                for dof in 0..nb {
                    let adiv = shapes[dof];
                    rhs[dof] += (adiv[0] * normal[0] + adiv[1] * normal[1]) * sir.weights[ip];
                }
            }

            // Solve with the precomputed volume SVD.
            let svd = self.volume_svd.as_ref().expect("volume SVD");
            let weights = svd_solve_lsq(svd, &rhs, npts);
            self.ir_weights = weights;
        } else {
            for ip in 0..npts {
                self.ir_weights[ip] = 0.0;
            }
        }

        // Fully inside the subdomain → standard Gauss rule with at least the
        // current number of points (MFEM: `ir2.GetNPoints() < ir.GetNPoints()`
        // loop over `IntegrationRules.Get`).
        if interior {
            let cur = self.ir_points.len();
            let mut qorder = 0;
            while rule_npts(2, qorder) < cur {
                qorder += 1;
            }
            let n = qorder / 2 + 1;
            let (pts, wts) = tensor_rule_2d(n, n);
            self.ir_points = pts;
            self.ir_weights = wts;
        }
    }
}
// ─── Per-element weight computations (3-D) ───────────────────────────────────

impl<'a> MomentFitting<'a> {
    /// 3-D face unit normal in the reference frame (MFEM ex38.cpp convention
    /// for local faces 0..5 of the hex).
    fn face_normal_3d(lf: usize) -> [f64; 3] {
        let mut normal = [0.0; 3];
        if lf == 0 || lf == 5 {
            normal[2] = 1.0;
        }
        if lf == 1 || lf == 3 {
            normal[1] = 1.0;
        }
        if lf == 2 || lf == 4 {
            normal[0] = 1.0;
        }
        if lf == 0 || lf == 1 || lf == 4 {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }
        normal
    }

    /// MFEM `ComputeFaceWeights`: 3-D face integration rules (2-D volume rules
    /// on each face, cached per global face).
    fn compute_face_weights(&mut self, e: u32) {
        let n_faces = self.geom.n_faces;
        if self.face_ip.is_empty() {
            self.face_weights_comp = vec![0.0; n_faces];
        }
        let faces = self.geom.elem_faces[e as usize].clone();
        for &face in &faces {
            if self.face_weights_comp[face as usize] == 0.0 {
                self.face_weights_comp[face as usize] = 1.0;
                let fv = self.geom.face_verts[face as usize].clone();
                let coords: Vec<Vec<f64>> = fv
                    .iter()
                    .map(|&v| self.geom.coords[v as usize].clone())
                    .collect();
                let mut face_rules = Self::new_face(&coords, self.order, self.ls_order, self.level_set);
                let rule = face_rules.get_volume_integration_rule(0, None);
                if self.face_ip.len() != rule.n_points() {
                    self.face_ip = rule.points.clone();
                    self.face_weights = vec![vec![0.0; n_faces]; rule.n_points()];
                    self.face_weights_comp = vec![0.0; n_faces];
                    self.face_weights_comp[face as usize] = 1.0;
                }
                for (ip, &w) in rule.weights.iter().enumerate() {
                    self.face_weights[ip][face as usize] = w;
                }
            }
        }
    }

    /// Classify the 6 faces of element `e`; returns `(interior, element_int)`
    /// (MFEM `ComputeSurfaceWeights3D` / `ComputeVolumeWeights3D`).
    fn face_classify_3d(&self, e: u32) -> (bool, bool) {
        let verts = &self.geom.elem_verts[e as usize];
        let mut interior = true;
        let mut element_int = false;
        for lf in 0..6 {
            let fv = HEX_FACE_VERTS[lf];
            let mut any_neg = false;
            let mut any_pos = false;
            for &vi in &fv {
                let v = verts[vi];
                let xi = self.geom.transform_back(e, &self.geom.coords[v as usize]);
                let ls = self.eval_ls(e, &xi);
                if ls < -TOL_1 {
                    any_neg = true;
                }
                if ls > -TOL_1 {
                    any_pos = true;
                }
            }
            if any_neg {
                interior = false;
            }
            if any_pos {
                element_int = true;
            }
        }
        (interior, element_int)
    }

    /// MFEM `ComputeSurfaceWeights3D`.
    fn compute_surface_weights_3d(&mut self, e: u32) {
        self.compute_face_weights(e);
        let nb = self.n_basis;
        let npts = self.ir_points.len();
        let mut rhs = vec![0.0; nb];
        let mut mat = nalgebra::DMatrix::zeros(nb, npts);

        let (interior, element_int) = self.face_classify_3d(e);
        let faces = self.geom.elem_faces[e as usize].clone();
        let n_face_ip = self.face_ip.len();

        for lf in 0..6 {
            let normal = Self::face_normal_3d(lf);
            let face = faces[lf];
            for ip in 0..n_face_ip {
                let point = self.geom.face_map(face, &self.face_ip[ip]);
                let ipoint = self.geom.transform_back(e, &point);
                let shape = self.ortho_basis_3d(&ipoint);
                let fw = self.face_weights[ip][face as usize];
                for dof in 0..nb {
                    let grad = shape[dof];
                    rhs[dof] -= (grad[0] * normal[0] + grad[1] * normal[1] + grad[2] * normal[2])
                        * fw;
                }
            }
        }

        if element_int && !interior {
            for ip in 0..npts {
                let xi = &self.ir_points[ip];
                let grad_ref = self.ls.as_ref().expect("ls").grad_ref(e, xi);
                let nrm = (grad_ref[0] * grad_ref[0]
                    + grad_ref[1] * grad_ref[1]
                    + grad_ref[2] * grad_ref[2])
                    .sqrt();
                let mut normal = [0.0; 3];
                if nrm != 0.0 {
                    normal[0] = -grad_ref[0] / nrm;
                    normal[1] = -grad_ref[1] / nrm;
                    normal[2] = -grad_ref[2] / nrm;
                }
                let shapes = self.ortho_basis_3d(xi);
                for dof in 0..nb {
                    let grad = shapes[dof];
                    mat[(dof, ip)] =
                        grad[0] * normal[0] + grad[1] * normal[1] + grad[2] * normal[2];
                }
            }
            let svd = svd_decompose(&mat);
            let weights = svd_solve_lsq(&svd, &rhs, npts);
            self.ir_weights = weights;
        } else {
            for ip in 0..npts {
                self.ir_weights[ip] = 0.0;
            }
        }
    }

    /// MFEM `ComputeVolumeWeights3D` (needs the surface rule `sir`).
    fn compute_volume_weights_3d(&mut self, e: u32, sir: Option<&CutRule>) {
        self.order += 1;
        self.compute_face_weights(e);
        self.order -= 1;
        let nb = self.n_basis_volume;
        let npts = self.ir_points.len();
        let mut rhs = vec![0.0; nb];

        let (interior, element_int) = self.face_classify_3d(e);
        let faces = self.geom.elem_faces[e as usize].clone();
        let n_face_ip = self.face_ip.len();

        for lf in 0..6 {
            let normal = Self::face_normal_3d(lf);
            let face = faces[lf];
            for ip in 0..n_face_ip {
                let point = self.geom.face_map(face, &self.face_ip[ip]);
                let ipoint = self.geom.transform_back(e, &point);
                let shape = self.basis_ad_3d(&ipoint);
                let fw = self.face_weights[ip][face as usize];
                for dof in 0..nb {
                    let adiv = shape[dof];
                    rhs[dof] += (adiv[0] * normal[0] + adiv[1] * normal[1] + adiv[2] * normal[2])
                        * fw;
                }
            }
        }

        if element_int && !interior {
            let sir = sir.expect("surface rule for 3-D volume weights");
            let np = sir.n_points();
            for ip in 0..np {
                let xi = &sir.points[ip];
                let grad_ref = self.ls.as_ref().expect("ls").grad_ref(e, xi);
                let nrm = (grad_ref[0] * grad_ref[0]
                    + grad_ref[1] * grad_ref[1]
                    + grad_ref[2] * grad_ref[2])
                    .sqrt();
                let mut normal = [0.0; 3];
                if nrm != 0.0 {
                    normal[0] = -grad_ref[0] / nrm;
                    normal[1] = -grad_ref[1] / nrm;
                    normal[2] = -grad_ref[2] / nrm;
                }
                let shapes = self.basis_ad_3d(xi);
                for dof in 0..nb {
                    let adiv = shapes[dof];
                    rhs[dof] += (adiv[0] * normal[0] + adiv[1] * normal[1] + adiv[2] * normal[2])
                        * sir.weights[ip];
                }
            }
            let svd = self.volume_svd.as_ref().expect("volume SVD");
            let weights = svd_solve_lsq(svd, &rhs, npts);
            self.ir_weights = weights;
        } else {
            for ip in 0..npts {
                self.ir_weights[ip] = 0.0;
            }
        }

        // Fully inside the subdomain → standard Gauss rule with at least the
        // current number of points (MFEM: `ir2.GetNPoints() < ir.GetNPoints()`
        // loop over `IntegrationRules.Get`).
        if interior {
            let cur = self.ir_points.len();
            let mut qorder = 0;
            while rule_npts(3, qorder) < cur {
                qorder += 1;
            }
            let n = qorder / 2 + 1;
            let (pts, wts) = tensor_rule_3d(n, n, n);
            self.ir_points = pts;
            self.ir_weights = wts;
        }
    }

    fn current_rule(&self) -> CutRule {
        CutRule {
            points: self.ir_points.clone(),
            weights: self.ir_weights.clone(),
            order: self.ir_order,
        }
    }
}

// ─── Public API (MFEM `GetSurfaceIntegrationRule` / `GetVolumeIntegrationRule`
// ─── / `GetSurfaceWeights`) ──────────────────────────────────────────────────

impl<'a> MomentFitting<'a> {
    /// Cut-surface integration rule for element `e` (MFEM
    /// `GetSurfaceIntegrationRule`). The returned rule contains the candidate
    /// points and the moment-fitted weights.
    pub fn get_surface_integration_rule(&mut self, e: u32) -> CutRule {
        if self.n_basis == 0 {
            self.init_surface(self.order);
        }
        if self.dim() == 3 {
            self.face_ip = vec![];
            self.face_weights = vec![];
            self.face_weights_comp = vec![];
        }
        match self.dim() {
            1 => self.compute_surface_weights_1d(e),
            2 => self.compute_surface_weights_2d(e),
            _ => self.compute_surface_weights_3d(e),
        }
        self.current_rule()
    }

    /// Transformation weights for surface integration (MFEM `GetSurfaceWeights`):
    /// `|∇ϕ_phys| / |∇ϕ_ref|`, only where the surface rule has non-zero weights.
    pub fn get_surface_weights(&self, e: u32, sir: &CutRule) -> Vec<f64> {
        let n = sir.n_points();
        let mut weights = vec![0.0; n];
        let mut computeweights = false;
        for &w in &sir.weights {
            if w != 0.0 {
                computeweights = true;
            }
        }
        if self.dim() > 1 && computeweights {
            for ip in 0..n {
                let xi = &sir.points[ip];
                let gp = self.ls.as_ref().expect("ls").grad_phys(e, xi, &self.geom);
                let gr = self.ls.as_ref().expect("ls").grad_ref(e, xi);
                let normphys = gp.iter().map(|v| v * v).sum::<f64>().sqrt();
                let normref = gr.iter().map(|v| v * v).sum::<f64>().sqrt();
                if normref != 0.0 {
                    weights[ip] = normphys / normref;
                }
            }
        }
        weights
    }

    /// Cut-volume integration rule for element `e` (MFEM
    /// `GetVolumeIntegrationRule`). `sir` is the optional corresponding
    /// surface rule (reused when its order matches).
    pub fn get_volume_integration_rule(&mut self, e: u32, sir: Option<&CutRule>) -> CutRule {
        if self.n_basis == 0 || self.n_basis_volume == 0 {
            self.clear();
            self.init_volume(self.order);
        }
        if self.dim() == 3 {
            self.face_ip = vec![];
            self.face_weights = vec![];
            self.face_weights_comp = vec![];
        }

        if self.dim() == 1 {
            self.clear();
            self.init_volume(self.order);
            self.compute_volume_weights_1d(e);
            return self.current_rule();
        }

        // Obtain the surface rule (order+1).
        let need_surf = match sir {
            Some(s) => s.order.wrapping_sub(1) != self.ir_order,
            None => true,
        };
        let s_rule: CutRule = if need_surf {
            let saved = self.order;
            self.order += 1;
            match self.dim() {
                2 => self.compute_surface_weights_2d(e),
                _ => self.compute_surface_weights_3d(e),
            }
            let r = self.current_rule();
            self.order = saved;
            r
        } else {
            sir.expect("sir").clone()
        };

        match self.dim() {
            2 => self.compute_volume_weights_2d(e, Some(&s_rule)),
            _ => self.compute_volume_weights_3d(e, Some(&s_rule)),
        }
        self.current_rule()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_lvl(x: &[f64]) -> f64 {
        1.0 - (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }

    /// 2-D moment-fitting surface integral of 3x²-y² over the unit circle in
    /// [-1.6,1.6]² (MFEM ex38 surface2d r3 → 6.2831308145e0).
    #[test]
    fn surface2d_r3_matches_mfem() {
        let mut mesh: Mesh<2> = Mesh::make_cartesian_2d(1, 1, 3.2, 3.2);
        mesh.translate([-1.6, -1.6]);
        for _ in 0..3 {
            mesh = fem_mesh::refine_uniform(&mesh);
        }
        let lvl = |x: &[f64]| 1.0 - (x[0] * x[0] + x[1] * x[1]);
        let mut mf = MomentFitting::new_2d(&mesh, 2, 2, &lvl);
        let geom = CutGeom::from_mesh2(&mesh);
        let mut sum = 0.0;
        for e in 0..geom.elem_verts.len() as u32 {
            let sir = mf.get_surface_integration_rule(e);
            let sw = mf.get_surface_weights(e, &sir);
            for ip in 0..sir.n_points() {
                let p = geom.map_phys(e, &sir.points[ip]);
                sum += sir.weights[ip] * sw[ip] * geom.det_j(e)
                    * (3.0 * p[0] * p[0] - p[1] * p[1]);
            }
        }
        assert!((sum - 6.2831308145e0).abs() < 1e-8, "surface2d r3 = {sum}");
    }

    /// 3-D moment-fitting surface integral of 4-3x²+2y²-z² over the unit sphere
    /// in [-1.6,1.6]³ (MFEM ex38 surface3d r1 → 3.9434187022e1).
    #[test]
    fn surface3d_r1_matches_mfem() {
        let coords = vec![
            -1.6, -1.6, -1.6, 1.6, -1.6, -1.6, 1.6, 1.6, -1.6, -1.6, 1.6, -1.6, //
            -1.6, -1.6, 1.6, 1.6, -1.6, 1.6, 1.6, 1.6, 1.6, -1.6, 1.6, 1.6,
        ];
        let mut mesh: Mesh<3> = Mesh::uniform(
            coords,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![1],
            fem_mesh::ElementType::Hex8,
            vec![],
            vec![],
            fem_mesh::ElementType::Quad4,
        );
        mesh = fem_mesh::refine_uniform_3d(&mesh);
        let lvl = |x: &[f64]| sphere_lvl(x);
        let mut mf = MomentFitting::new_3d(&mesh, 2, 2, &lvl);
        let geom = CutGeom::from_mesh3(&mesh);
        let mut sum = 0.0;
        for e in 0..geom.elem_verts.len() as u32 {
            let sir = mf.get_surface_integration_rule(e);
            let sw = mf.get_surface_weights(e, &sir);
            for ip in 0..sir.n_points() {
                let p = geom.map_phys(e, &sir.points[ip]);
                sum += sir.weights[ip] * sw[ip] * geom.det_j(e)
                    * (4.0 - 3.0 * p[0] * p[0] + 2.0 * p[1] * p[1] - p[2] * p[2]);
            }
        }
        assert!((sum - 3.9434187022e1).abs() < 1e-6, "surface3d r1 = {sum}");
    }

    /// 3-D div-free basis matches MFEM GetDivFree3DBasis at a sample point.
    #[test]
    fn div_free_3d_matches_mfem() {
        let s = div_free_3d(0.3, -0.2, 0.1, 2);
        assert!((s[0][0] - 0.3535533905932738).abs() < 1e-15);
        assert!((s[5][0] - 0.6123724356957945 * (-0.2)).abs() < 1e-14);
        let s3 = div_free_3d(0.3, -0.2, 0.1, 3);
        // -0.6846531968814576*eta + 2.053959590644373*eta*nu^2 at (0.3,-0.2,0.1)
        let expect = -0.6846531968814576 * (-0.2) + 2.053959590644373 * (-0.2) * 0.1f64.powi(2);
        assert!((s3[28][0] - expect).abs() < 1e-14);
    }
}
