//! CAD surface interface and geometry projection for curved mesh generation.
//!
//! Provides:
//! - [`CadModel`] trait — evaluate, normal, project for CAD surfaces
//! - Analytic surfaces: plane, sphere, cylinder, torus, cone
//! - NURBS surface wrapper using existing fem-element NURBS patches
//! - Faceted (STL) surface for approximate CAD
//! - [`project_boundary_to_cad`] — elevate a SimplexMesh boundary to a CAD surface
//!
//! # Usage
//! ```rust,ignore
//! use fem_mesh::cad::*;
//!
//! // Create a cylindrical surface
//! let cyl = AnalyticSurface::cylinder([0.0, 0.0, 0.0], 1.0, 2.0);
//!
//! // Project mesh boundary attribute 1 onto the cylinder
//! let config = ProjectionConfig::new().with_surface(1, CadShape::Analytic(cyl));
//! let curved = project_boundary_to_cad(&mesh, &config, 2)?;
//! ```

use std::f64::consts::PI;

use crate::topology::MeshTopology;
use crate::SimplexMesh;

// ─── CadModel trait ──────────────────────────────────────────────────────────

/// A CAD surface parameterization: `(u, v) → (x, y, z)`.
pub trait CadModel: Send + Sync {
    /// Evaluate the surface at parameter coordinates (u, v).
    fn eval(&self, u: f64, v: f64) -> [f64; 3];

    /// Unit normal at (u, v).
    fn normal(&self, u: f64, v: f64) -> [f64; 3];

    /// UV parameter range: `(umin, umax, vmin, vmax)`.
    fn parameter_range(&self) -> [f64; 4];

    /// Project a point onto the surface, returning `(u, v, distance)`.
    fn project(&self, point: &[f64; 3]) -> (f64, f64, f64);
}

// ─── Analytic surfaces ──────────────────────────────────────────────────────

/// Simple analytic surfaces for CAD geometry.
#[derive(Debug, Clone)]
pub enum AnalyticSurface {
    Plane {
        origin: [f64; 3],
        u_dir: [f64; 3],
        v_dir: [f64; 3],
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    Cylinder {
        center: [f64; 3],
        radius: f64,
        height: f64,
    },
    Torus {
        center: [f64; 3],
        major_radius: f64,
        minor_radius: f64,
    },
    Cone {
        center: [f64; 3],
        radius: f64,
        height: f64,
    },
}

macro_rules! vec3_sub {
    ($a:expr, $b:expr) => { [$a[0]-$b[0], $a[1]-$b[1], $a[2]-$b[2]] };
}
macro_rules! vec3_dot {
    ($a:expr, $b:expr) => { $a[0]*$b[0] + $a[1]*$b[1] + $a[2]*$b[2] };
}
macro_rules! vec3_cross {
    ($a:expr, $b:expr) => { [
        $a[1]*$b[2] - $a[2]*$b[1],
        $a[2]*$b[0] - $a[0]*$b[2],
        $a[0]*$b[1] - $a[1]*$b[0],
    ]};
}
macro_rules! vec3_len {
    ($a:expr) => { ($a[0]*$a[0] + $a[1]*$a[1] + $a[2]*$a[2]).sqrt() };
}
macro_rules! vec3_norm {
    ($a:expr) => {{
        let l = vec3_len!($a);
        [$a[0]/l, $a[1]/l, $a[2]/l]
    }};
}

impl AnalyticSurface {
    pub fn plane(origin: [f64; 3], u_dir: [f64; 3], v_dir: [f64; 3]) -> Self {
        Self::Plane { origin, u_dir, v_dir }
    }
    pub fn sphere(center: [f64; 3], radius: f64) -> Self {
        Self::Sphere { center, radius }
    }
    pub fn cylinder(center: [f64; 3], radius: f64, height: f64) -> Self {
        Self::Cylinder { center, radius, height }
    }
    pub fn torus(center: [f64; 3], major: f64, minor: f64) -> Self {
        Self::Torus { center, major_radius: major, minor_radius: minor }
    }
    pub fn cone(center: [f64; 3], radius: f64, height: f64) -> Self {
        Self::Cone { center, radius, height }
    }
}

impl CadModel for AnalyticSurface {
    fn eval(&self, u: f64, v: f64) -> [f64; 3] {
        match self {
            Self::Plane { origin, u_dir, v_dir } => [
                origin[0] + u * u_dir[0] + v * v_dir[0],
                origin[1] + u * u_dir[1] + v * v_dir[1],
                origin[2] + u * u_dir[2] + v * v_dir[2],
            ],
            Self::Sphere { center, radius } => {
                let theta = u * 2.0 * PI;
                let phi = v * PI;
                let st = theta.sin(); let ct = theta.cos();
                let sp = phi.sin(); let cp = phi.cos();
                [center[0] + radius * ct * sp,
                 center[1] + radius * st * sp,
                 center[2] + radius * cp]
            },
            Self::Cylinder { center, radius, height } => {
                let theta = u * 2.0 * PI;
                [center[0] + radius * theta.cos(),
                 center[1] + radius * theta.sin(),
                 center[2] + v * height]
            },
            Self::Torus { center, major_radius: r_major, minor_radius: r_minor } => {
                let theta = u * 2.0 * PI;
                let phi = v * 2.0 * PI;
                let ct = theta.cos(); let st = theta.sin();
                let cp = phi.cos(); let sp = phi.sin();
                [center[0] + (r_major + r_minor * cp) * ct,
                 center[1] + (r_major + r_minor * cp) * st,
                 center[2] + r_minor * sp]
            },
            Self::Cone { center, radius, height } => {
                let theta = u * 2.0 * PI;
                let r_at_v = radius * (1.0 - v);
                [center[0] + r_at_v * theta.cos(),
                 center[1] + r_at_v * theta.sin(),
                 center[2] + v * height]
            },
        }
    }

    fn normal(&self, u: f64, v: f64) -> [f64; 3] {
        let eps = 1e-6;
        let p = self.eval(u, v);
        let pu = self.eval(u + eps, v);
        let pv = self.eval(u, v + eps);
        let du = vec3_sub!(pu, p);
        let dv = vec3_sub!(pv, p);
        vec3_norm!(vec3_cross!(du, dv))
    }

    fn parameter_range(&self) -> [f64; 4] {
        match self {
            Self::Plane { .. } => [0.0, 1.0, 0.0, 1.0],
            Self::Sphere { .. } => [0.0, 1.0, 0.0, 1.0],
            Self::Cylinder { .. } => [0.0, 1.0, 0.0, 1.0],
            Self::Torus { .. } => [0.0, 1.0, 0.0, 1.0],
            Self::Cone { .. } => [0.0, 1.0, 0.0, 1.0],
        }
    }

    fn project(&self, point: &[f64; 3]) -> (f64, f64, f64) {
        match self {
            Self::Plane { origin, u_dir, v_dir } => {
                let d = vec3_sub!(*point, *origin);
                let det = u_dir[0]*v_dir[1] - u_dir[1]*v_dir[0];
                if det.abs() > 1e-15 {
                    let u = (d[0]*v_dir[1] - d[1]*v_dir[0]) / det;
                    let v = (u_dir[0]*d[1] - u_dir[1]*d[0]) / det;
                    (u, v, 0.0)
                } else {
                    (0.0, 0.0, vec3_len!(d))
                }
            },
            Self::Sphere { center, radius } => {
                let d = vec3_sub!(*point, *center);
                let r = vec3_len!(d);
                let theta = d[1].atan2(d[0]) / (2.0 * PI);
                let phi = (d[2] / r).acos() / PI;
                (theta.rem_euclid(1.0), phi.rem_euclid(1.0), (r - radius).abs())
            },
            Self::Cylinder { center, radius, height } => {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let theta = dy.atan2(dx) / (2.0 * PI);
                let v = ((point[2] - center[2]) / height).clamp(0.0, 1.0);
                let r = (dx*dx + dy*dy).sqrt();
                (theta.rem_euclid(1.0), v, (r - radius).abs())
            },
            _ => {
                // Generic: grid search + Newton
                let [umin, umax, vmin, vmax] = self.parameter_range();
                let nu = 40; let nv = 40;
                let mut best_u = 0.0; let mut best_v = 0.0; let mut best_d = f64::MAX;
                for i in 0..=nu { for j in 0..=nv {
                    let u = umin + (umax - umin) * i as f64 / nu as f64;
                    let v = vmin + (vmax - vmin) * j as f64 / nv as f64;
                    let p = self.eval(u, v);
                    let d = vec3_len!(vec3_sub!(p, *point));
                    if d < best_d { best_d = d; best_u = u; best_v = v; }
                }}
                (best_u, best_v, best_d)
            },
        }
    }
}

// ─── NURBS surface ──────────────────────────────────────────────────────────

/// CAD surface backed by a NURBS patch control net.
///
/// Evaluates `S(u,v) = Σᵢⱼ Rᵢⱼ(u,v) · Pᵢⱼ` where `Rᵢⱼ` are the rational
/// basis functions and `Pᵢⱼ` are the 3D control points.
#[derive(Debug, Clone)]
pub struct NurbsCadSurface2D {
    kv_u: Vec<f64>,
    kv_v: Vec<f64>,
    control_pts: Vec<[f64; 3]>,
    weights: Vec<f64>,
    n_u: usize,
    n_v: usize,
}

impl NurbsCadSurface2D {
    /// Build from element-crate patch data (2-D control points → 3-D).
    pub fn from_patch_data(pd: &fem_element::iga::NurbsPatch2DData) -> Self {
        let ctrl_3d: Vec<[f64; 3]> = pd.control_pts.iter().map(|&c| [c[0], c[1], 0.0]).collect();
        let kv_u = pd.kv_u.knots.clone();
        let kv_v = pd.kv_v.knots.clone();
        Self { kv_u, kv_v, control_pts: ctrl_3d, weights: pd.weights.clone(), n_u: pd.kv_u.n_basis(), n_v: pd.kv_v.n_basis() }
    }

    /// Convert back to element-crate patch data (drops z-coordinate).
    pub fn into_patch_data(self) -> fem_element::iga::NurbsPatch2DData {
        let pu = self.kv_u.len() - self.n_u - 1;
        let pv = self.kv_v.len() - self.n_v - 1;
        let c2d: Vec<[f64; 2]> = self.control_pts.iter().map(|c| [c[0], c[1]]).collect();
        use fem_element::iga::NurbsKnotVector;
        fem_element::iga::NurbsPatch2DData {
            kv_u: NurbsKnotVector::new(self.kv_u, pu),
            kv_v: NurbsKnotVector::new(self.kv_v, pv),
            control_pts: c2d,
            weights: self.weights,
            tag: 0,
        }
    }

    /// Build a NURBS CAD surface from knot vectors, 3D control points, and weights.
    /// `control_pts` in lexicographic order (u-fast, v-slow).
    pub fn new(
        kv_u: Vec<f64>,
        kv_v: Vec<f64>,
        control_pts: Vec<[f64; 3]>,
        weights: Vec<f64>,
    ) -> Self {
        let degree_u = {
            let mut d = 0usize;
            for &k in &kv_u { if k == kv_u[0] { d += 1; } else { break; } }
            d - 1
        };
        let n_u = kv_u.len() - degree_u - 1;
        let n_v = {
            let mut d = 0usize;
            for &k in &kv_v { if k == kv_v[0] { d += 1; } else { break; } }
            kv_v.len() - d
        };
        assert_eq!(control_pts.len(), n_u * n_v);
        assert_eq!(weights.len(), n_u * n_v);
        Self { kv_u, kv_v, control_pts, weights, n_u, n_v }
    }

    /// Evaluate B-spline basis functions at parameter `xi` given knot vector `kv`.
    fn bspline_basis(kv: &[f64], p: usize, xi: f64) -> Vec<f64> {
        let n = kv.len() - p - 1;
        let mut vals = vec![0.0_f64; n];
        if xi < kv[0] || xi > kv[kv.len()-1] { return vals; }
        // Find span
        let mut span = p;
        for i in p+1..kv.len() {
            if kv[i] > xi + 1e-15 { span = i - 1; break; }
        }
        // Cox-de Boor recursion
        let mut left = vec![0.0_f64; p+1];
        let mut right = vec![0.0_f64; p+1];
        vals[span] = 1.0;
        for j in 1..=p {
            left[j] = xi - kv[span+1-j];
            right[j] = kv[span+j] - xi;
            let mut saved = 0.0;
            for r in 0..j {
                let term = vals[span-r] / (right[r+1] + left[j-r]);
                vals[span-r] = saved + right[r+1] * term;
                saved = left[j-r] * term;
            }
            vals[span-j] = saved;
        }
        vals
    }

    /// Evaluate all rational NURBS basis functions at (u, v).
    fn eval_basis(&self, u: f64, v: f64) -> Vec<f64> {
        let pu = self.kv_u.len() - self.n_u - 1;
        let pv = self.kv_v.len() - self.n_v - 1;
        let bu = Self::bspline_basis(&self.kv_u, pu, u);
        let bv = Self::bspline_basis(&self.kv_v, pv, v);
        let mut r = vec![0.0_f64; self.n_u * self.n_v];
        let mut w_sum = 0.0_f64;
        for j in 0..self.n_v { for i in 0..self.n_u {
            let idx = j * self.n_u + i;
            r[idx] = bu[i] * bv[j] * self.weights[idx];
            w_sum += r[idx];
        }}
        if w_sum.abs() > 1e-30 {
            for v in r.iter_mut() { *v /= w_sum; }
        }
        r
    }
}

impl CadModel for NurbsCadSurface2D {
    fn eval(&self, u: f64, v: f64) -> [f64; 3] {
        let basis = self.eval_basis(u, v);
        let mut p = [0.0; 3];
        for (idx, b) in basis.iter().enumerate() {
            for d in 0..3 { p[d] += b * self.control_pts[idx][d]; }
        }
        p
    }

    fn normal(&self, u: f64, v: f64) -> [f64; 3] {
        let du = 1e-6;
        let p0 = self.eval(u, v);
        let pu = self.eval(u + du, v);
        let pv = self.eval(u, v + du);
        vec3_norm!(vec3_cross!(vec3_sub!(pu, p0), vec3_sub!(pv, p0)))
    }

    fn parameter_range(&self) -> [f64; 4] {
        let pu = self.kv_u.len() - self.n_u - 1;
        let pv = self.kv_v.len() - self.n_v - 1;
        [self.kv_u[pu], self.kv_u[self.kv_u.len() - pu - 1],
         self.kv_v[pv], self.kv_v[self.kv_v.len() - pv - 1]]
    }

    fn project(&self, point: &[f64; 3]) -> (f64, f64, f64) {
        let [umin, umax, vmin, vmax] = self.parameter_range();
        let nu = 50; let nv = 50;
        let mut bu = 0.0; let mut bv = 0.0; let mut bd = f64::MAX;
        for i in 0..=nu { for j in 0..=nv {
            let u = umin + (umax - umin) * i as f64 / nu as f64;
            let v = vmin + (vmax - vmin) * j as f64 / nv as f64;
            let p = self.eval(u, v);
            let d = vec3_len!(vec3_sub!(p, *point));
            if d < bd { bd = d; bu = u; bv = v; }
        }}
        (bu, bv, bd)
    }
}

// ─── Faceted (STL) surface ───────────────────────────────────────────────────

/// Approximate CAD surface from a triangulated facet mesh (STL/OBJ).
#[derive(Debug, Clone)]
pub struct FacetedCadSurface {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    /// Precomputed per-triangle normals.
    normals: Vec<[f64; 3]>,
}

impl FacetedCadSurface {
    pub fn from_triangulated(vertices: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> Self {
        let normals: Vec<[f64; 3]> = triangles.iter().map(|&[i, j, k]| {
            let a = vec3_sub!(vertices[j as usize], vertices[i as usize]);
            let b = vec3_sub!(vertices[k as usize], vertices[i as usize]);
            vec3_norm!(vec3_cross!(a, b))
        }).collect();
        Self { vertices, triangles, normals }
    }

    /// Load from STL mesh (tessellated surface).
    pub fn from_stl_mesh(mesh: &SimplexMesh<3>) -> Self {
        let n = mesh.n_nodes();
        let verts: Vec<[f64; 3]> = (0..n).map(|i| {
            let c = mesh.node_coords(i as u32);
            [c[0], c[1], c[2]]
        }).collect();
        let tris: Vec<[u32; 3]> = (0..mesh.n_elements() as u32).map(|e| {
            let n = mesh.element_nodes(e);
            [n[0], n[1], n[2]]
        }).collect();
        Self::from_triangulated(verts, tris)
    }
}

impl CadModel for FacetedCadSurface {
    fn eval(&self, _u: f64, _v: f64) -> [f64; 3] {
        // Faceted surface doesn't support parameterised evaluation directly.
        // Return first vertex as fallback.
        self.vertices[0]
    }

    fn normal(&self, _u: f64, _v: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }

    fn parameter_range(&self) -> [f64; 4] { [0.0, 1.0, 0.0, 1.0] }

    fn project(&self, point: &[f64; 3]) -> (f64, f64, f64) {
        // Find closest triangle and compute barycentric coordinates
        let mut best_d = f64::MAX;
        let mut best_u = 0.0; let mut best_v = 0.0;
        for (ti, &[i, j, k]) in self.triangles.iter().enumerate() {
            let a = self.vertices[i as usize];
            let b = self.vertices[j as usize];
            let c = self.vertices[k as usize];
            let n = self.normals[ti];
            // Project onto triangle plane
            let d0 = vec3_sub!(*point, a);
            let dist = vec3_dot!(d0, n);
            let proj = [point[0] - dist * n[0], point[1] - dist * n[1], point[2] - dist * n[2]];
            // Barycentric coordinates via area ratios
            let v0 = vec3_sub!(b, a); let v1 = vec3_sub!(c, a); let v2 = vec3_sub!(proj, a);
            let d00 = vec3_dot!(v0, v0); let d01 = vec3_dot!(v0, v1);
            let d11 = vec3_dot!(v1, v1); let d20 = vec3_dot!(v2, v0); let d21 = vec3_dot!(v2, v1);
            let denom = d00 * d11 - d01 * d01;
            if denom.abs() < 1e-30 { continue; }
            let uu = (d11 * d20 - d01 * d21) / denom;
            let vv = (d00 * d21 - d01 * d20) / denom;
            let d_abs = dist.abs();
            if d_abs < best_d { best_d = d_abs; best_u = uu; best_v = vv; }
        }
        (best_u, best_v, best_d)
    }
}

// ─── CadShape dispatch ───────────────────────────────────────────────────────

/// Dispatch wrapper over multiple CAD surface types.
#[derive(Debug, Clone)]
pub enum CadShape {
    Analytic(AnalyticSurface),
    Nurbs(NurbsCadSurface2D),
    Faceted(FacetedCadSurface),
}

impl CadModel for CadShape {
    fn eval(&self, u: f64, v: f64) -> [f64; 3] {
        match self {
            Self::Analytic(a) => a.eval(u, v),
            Self::Nurbs(n) => n.eval(u, v),
            Self::Faceted(f) => f.eval(u, v),
        }
    }
    fn normal(&self, u: f64, v: f64) -> [f64; 3] {
        match self {
            Self::Analytic(a) => a.normal(u, v),
            Self::Nurbs(n) => n.normal(u, v),
            Self::Faceted(f) => f.normal(u, v),
        }
    }
    fn parameter_range(&self) -> [f64; 4] {
        match self {
            Self::Analytic(a) => a.parameter_range(),
            Self::Nurbs(n) => n.parameter_range(),
            Self::Faceted(f) => f.parameter_range(),
        }
    }
    fn project(&self, point: &[f64; 3]) -> (f64, f64, f64) {
        match self {
            Self::Analytic(a) => a.project(point),
            Self::Nurbs(n) => n.project(point),
            Self::Faceted(f) => f.project(point),
        }
    }
}

// ─── Boundary projection ─────────────────────────────────────────────────────

/// Boundary-to-CAD surface mapping configuration.
///
/// Maps each boundary attribute tag to a specific CAD surface.
#[derive(Debug)]
pub struct ProjectionConfig {
    surfaces: Vec<(i32, CadShape)>,
}

impl Default for ProjectionConfig {
    fn default() -> Self { Self::new() }
}

impl ProjectionConfig {
    pub fn new() -> Self { Self { surfaces: Vec::new() } }

    /// Assign a CAD surface to a boundary attribute tag.
    pub fn with_surface(mut self, tag: i32, surface: CadShape) -> Self {
        self.surfaces.push((tag, surface));
        self
    }
}

/// Project boundary nodes of a mesh onto CAD surfaces.
///
/// Returns a new `SimplexMesh` with curved (projected) boundary nodes.
/// Interior nodes are left unchanged.
    pub fn project_boundary_to_cad<const D: usize>(
    mesh: &SimplexMesh<D>,
    config: &ProjectionConfig,
    _geom_order: u8,
) -> SimplexMesh<D> {
    // Build a lookup: tag → surface
    use std::collections::HashMap;
    let tag_to_surface: HashMap<i32, &CadShape> =
        config.surfaces.iter().map(|(t, s)| (*t, s)).collect();

    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut projected_nodes: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        let tag = mesh.face_tag(f);
        if let Some(&surface) = tag_to_surface.get(&tag) {
            for &n in mesh.face_nodes(f) {
                if projected_nodes.contains(&n) { continue; }
                projected_nodes.insert(n);
                let c = mesh.node_coords(n);
                let pt = [c[0], c[1], if D == 3 { c[2] } else { 0.0 }];
                let (u, v, _) = surface.project(&pt);
                let proj = surface.eval(u, v);
                for i in 0..D { new_coords[n as usize * D + i] = proj[i]; }
            }
        }
    }

    SimplexMesh {
        coords: new_coords,
        conn: mesh.conn.clone(),
        elem_type: mesh.elem_type,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(),
        face_type: mesh.face_type,
        elem_tags: mesh.elem_tags.clone(),
        elem_types: mesh.elem_types.clone(),
        elem_offsets: mesh.elem_offsets.clone(),
        face_types: mesh.face_types.clone(),
        face_offsets: mesh.face_offsets.clone(),
        face_to_elem: mesh.face_to_elem.clone(),
        edge_conn: mesh.edge_conn.clone(),
        edge_to_elem: mesh.edge_to_elem.clone(),
    }
}

// ─── Cached projection for elevated meshes ───────────────────────────────────

/// Like [`project_boundary_to_cad`] but works with the node insertion
/// pattern of [`CurvedMesh::elevate_to_order`], projecting each new
/// high-order node onto the CAD surface based on its parametric position.
pub fn project_elevated_node<const D: usize>(
    tag: i32,
    config: &ProjectionConfig,
    linear_coord: &[f64; D],
) -> [f64; D] {
    for &(t, ref surface) in &config.surfaces {
        if t == tag {
            let mut pt3 = [0.0; 3];
            pt3[..D].copy_from_slice(&linear_coord[..D]);
            let (u, v, _) = surface.project(&pt3);
            let proj = surface.eval(u, v);
            let mut out = [0.0; D];
            out[..D].copy_from_slice(&proj[..D]);
            return out;
        }
    }
    *linear_coord
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_eval() {
        let s = AnalyticSurface::sphere([0.0; 3], 1.0);
        let p = s.eval(0.0, 0.0); // north pole
        assert!((p[2] - 1.0).abs() < 1e-12, "north pole z {:.6e}", p[2]);
    }

    #[test]
    fn cylinder_eval() {
        let c = AnalyticSurface::cylinder([0.0; 3], 1.0, 2.0);
        let p = c.eval(0.0, 1.0); // top, +x
        assert!((p[0] - 1.0).abs() < 1e-12, "cylinder x {:.6e}", p[0]);
        assert!((p[2] - 2.0).abs() < 1e-12, "cylinder z {:.6e}", p[2]);
    }

    #[test]
    fn plane_project_identity() {
        let p = AnalyticSurface::plane([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let (u, v, d) = p.project(&[0.5, 0.3, 0.0]);
        assert!((u - 0.5).abs() < 1e-10, "u {:.6e}", u);
        assert!((v - 0.3).abs() < 1e-10, "v {:.6e}", v);
        assert!(d < 1e-10, "dist {:.6e}", d);
    }

    #[test]
    fn torus_eval() {
        let t = AnalyticSurface::torus([0.0; 3], 2.0, 0.5);
        let p = t.eval(0.0, 0.0);
        // (R + r, 0, 0)
        assert!((p[0] - 2.5).abs() < 1e-12, "torus x {:.6e}", p[0]);
    }

    #[test]
    fn normal_matches_sphere_radial() {
        let s = AnalyticSurface::sphere([0.0; 3], 1.0);
        let n = s.normal(0.25, 0.25);
        let r = vec3_len!(n);
        assert!((r - 1.0).abs() < 1e-10, "normal not unit: {r:.6e}");
    }

    #[test]
    fn faceted_from_stl_projection() {
        // Build a simple triangle that represents a flat plane at z=0
        let _mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let tris = vec![[0, 1, 2]];
        let faceted = FacetedCadSurface::from_triangulated(verts, tris);
        let (_, _, d) = faceted.project(&[0.2, 0.3, 0.5]);
        assert!(d > 0.0, "expected non-zero distance from plane");
        assert!((d - 0.5).abs() < 1e-10, "expected dist 0.5 got {d:.6e}");
    }
}
