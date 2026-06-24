//! 2-D Euler equations via DG with explicit RK time stepping.
//!
//! Conservative state: U = (ρ, ρu, ρv, ρE)
//! Primitive: (ρ, u, v, p),  γ = 1.4

use fem_mesh::topology::MeshTopology;
use fem_element::lagrange::{TriP1, SegP1};
use fem_element::ReferenceElement;

/// 2-D Euler physics and numerical fluxes.
#[derive(Clone)]
pub struct Euler2D {
    pub gamma: f64,
}

impl Default for Euler2D { fn default() -> Self { Self { gamma: 1.4 } } }

impl Euler2D {
    pub fn prim_to_cons(&self, rho: f64, u: f64, v: f64, p: f64) -> [f64; 4] {
        let e = p / (self.gamma - 1.0) + 0.5 * rho * (u * u + v * v);
        [rho, rho * u, rho * v, e]
    }
    pub fn cons_to_prim(&self, q: &[f64; 4]) -> (f64, f64, f64, f64) {
        let rho = q[0].max(1e-14);
        let u = q[1] / rho; let v = q[2] / rho;
        let ke = 0.5 * rho * (u * u + v * v);
        let p = ((self.gamma - 1.0) * (q[3] - ke)).max(1e-14);
        (rho, u, v, p)
    }
    /// Physical flux in x-direction F(U)
    pub fn flux_x(&self, q: &[f64; 4]) -> [f64; 4] {
        let (r, u, v, p) = self.cons_to_prim(q);
        [r*u, r*u*u + p, r*u*v, u*(q[3] + p)]
    }
    /// Physical flux in y-direction G(U)
    pub fn flux_y(&self, q: &[f64; 4]) -> [f64; 4] {
        let (r, u, v, p) = self.cons_to_prim(q);
        [r*v, r*u*v, r*v*v + p, v*(q[3] + p)]
    }
    /// Rotated flux in direction n = (nx, ny): F_n = nx*F + ny*G
    pub fn flux_n(&self, q: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        let fx = self.flux_x(q); let fy = self.flux_y(q);
        [fx[0]*n[0]+fy[0]*n[1], fx[1]*n[0]+fy[1]*n[1], fx[2]*n[0]+fy[2]*n[1], fx[3]*n[0]+fy[3]*n[1]]
    }
    pub fn max_speed(&self, q: &[f64; 4]) -> f64 {
        let (r, u, v, p) = self.cons_to_prim(q);
        let a = (self.gamma * p / r).sqrt();
        (u*u + v*v).sqrt() + a
    }

    /// Roe-Pike numerical flux in direction n (delegates to LF for now)
    pub fn roe_flux(&self, ql: &[f64; 4], qr: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        self.lax_friedrichs_flux(ql, qr, n)
    }

    /// Local Lax-Friedrichs (Rusanov) flux
    pub fn lax_friedrichs_flux(&self, ql: &[f64; 4], qr: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        let fl = self.flux_n(ql, n); let fr = self.flux_n(qr, n);
        let a = self.max_speed(ql).max(self.max_speed(qr));
        [0.5*(fl[0]+fr[0])-0.5*a*(qr[0]-ql[0]), 0.5*(fl[1]+fr[1])-0.5*a*(qr[1]-ql[1]),
         0.5*(fl[2]+fr[2])-0.5*a*(qr[2]-ql[2]), 0.5*(fl[3]+fr[3])-0.5*a*(qr[3]-ql[3])]
    }
}

/// DG(P1) solver for 2-D Euler equations on Tri3 meshes.
pub struct DgEuler2D {
    mesh: Box<dyn MeshTopology + Send + Sync>,
    euler: Euler2D,
    n_elems: usize,
    n_dofs: usize,       // total DOFs = n_elems * 3 * 4 (per-elem P1 × 4 components)
    dofs_per_elem: usize,
}

impl DgEuler2D {
    pub fn new(mesh: impl MeshTopology + Send + Sync + 'static) -> Self {
        let n_elems = mesh.n_elements();
        let dofs_per_elem = 3; // P1 triangle
        let n_dofs = n_elems * dofs_per_elem * 4; // 4 Euler components
        Self { mesh: Box::new(mesh), euler: Euler2D::default(), n_elems, n_dofs, dofs_per_elem }
    }

    /// Index into the flat array: (elem, comp, local_dof)
    fn idx(&self, e: u32, c: usize, ld: usize) -> usize { (e as usize * self.dofs_per_elem + ld) * 4 + c }

    /// Project initial condition (rho, u, v, p) -> conservative U
    pub fn project_initial(&self, init: &dyn Fn(f64, f64) -> (f64, f64, f64, f64)) -> Vec<f64> {
        let euler = &self.euler;
        let mut u = vec![0.0; self.n_dofs];
        let ref_elem = TriP1;
        let qr = ref_elem.quadrature(2);
        let mut phi = vec![0.0; 3];
        for e in 0..self.n_elems as u32 {
            let enodes = self.mesh.element_nodes(e);
            let mut mass = [[0.0; 3]; 3];
            let mut rhs = [[0.0; 4]; 3];
            for q in 0..qr.n_points() {
                let xi = &qr.points[q]; let w = qr.weights[q];
                ref_elem.eval_basis(xi, &mut phi);
                let (_jac, det) = affine_jacobian_det(&*self.mesh, &enodes);
                let vol = (w * det).abs();
                let (cx, cy) = affine_map(&*self.mesh, &enodes, xi);
                let (r, uvel, vvel, p) = init(cx, cy);
                let cons = euler.prim_to_cons(r, uvel, vvel, p);
                for i in 0..3 {
                    for j in 0..3 { mass[i][j] += vol * phi[i] * phi[j]; }
                    for c in 0..4 { rhs[i][c] += vol * phi[i] * cons[c]; }
                }
            }
            let minv = inv3(mass);
            for i in 0..3 {
                for c in 0..4 {
                    let mut s = 0.0;
                    for j in 0..3 { s += minv[i][j] * rhs[j][c]; }
                    u[self.idx(e, c, i)] = s;
                }
            }
        }
        u
    }

    /// Compute dU/dt = -div(F) using DG(P1) with LF flux
    pub fn rhs(&self, u: &[f64]) -> Vec<f64> {
        let euler = &self.euler;
        let mut du = vec![0.0; self.n_dofs];
        let tri = TriP1; let seg = SegP1;
        let qr_vol = tri.quadrature(2); let qr_face = seg.quadrature(2);
        let mut phi = vec![0.0; 3]; let mut grad = vec![0.0; 6];

        // Volume integral: ∫ ∇φ·F(U) dΩ
        for e in 0..self.n_elems as u32 {
            let enodes = self.mesh.element_nodes(e);
            let (jac, det) = affine_jacobian_det(&*self.mesh, &enodes);
            let _ = jac;
            let inv_j = affine_inv_jac(jac);
            for q in 0..qr_vol.n_points() {
                let xi = &qr_vol.points[q]; let w = qr_vol.weights[q];
                tri.eval_basis(xi, &mut phi); tri.eval_grad_basis(xi, &mut grad);
                let vol = (w * det).abs();
                // Compute U at this qp
                let mut uqp = [0.0; 4];
                for i in 0..3 { for c in 0..4 { uqp[c] += phi[i] * u[self.idx(e, c, i)]; } }
                let fx = euler.flux_x(&uqp); let fy = euler.flux_y(&uqp);
                // Physical gradient: grad_phys = J^{-T} grad_ref
                for i in 0..3 {
                    let gx = inv_j[0][0]*grad[i*2] + inv_j[0][1]*grad[i*2+1];
                    let gy = inv_j[1][0]*grad[i*2] + inv_j[1][1]*grad[i*2+1];
                    for c in 0..4 {
                        du[self.idx(e, c, i)] += vol * (fx[c] * gx + fy[c] * gy);
                    }
                }
            }
        }

        // Face integral: ∫ φ·F_num(U^-, U^+, n̂) dΓ (with sign for left element)
        let interior_faces = build_interior_faces(&*self.mesh);
        for face in &interior_faces {
            let (el, er) = (face.0, face.1);
            let mut fnodes: Vec<u32> = face.2.clone(); fnodes.sort_unstable();
            let fn_l = face.2.clone();
            let en_l = self.mesh.element_nodes(el);
            let en_r = self.mesh.element_nodes(er);
            // Face normal (outward from left) and Jacobian
            let (nx, ny) = face_normal(&*self.mesh, &en_l, el, &fn_l);
            let face_jac = face_size(&*self.mesh, &fn_l);
            // Map face qp to reference element for left and right
            for q in 0..qr_face.n_points() {
                let t = qr_face.points[q][0]; let w = qr_face.weights[q] * face_jac;
                let xi_l = map_to_elem(el, &fn_l, t, &en_l);
                let xi_r = map_to_elem(er, &fn_l, t, &en_r);
                tri.eval_basis(&xi_l, &mut phi);
                let mut ul = [0.0; 4];
                for i in 0..3 { for c in 0..4 { ul[c] += phi[i] * u[self.idx(el, c, i)]; } }
                tri.eval_basis(&xi_r, &mut phi);
                let mut ur = [0.0; 4];
                for i in 0..3 { for c in 0..4 { ur[c] += phi[i] * u[self.idx(er, c, i)]; } }
                let fstar = euler.lax_friedrichs_flux(&ul, &ur, &[nx, ny]);
                // Left element: flux contributes to both sides
                tri.eval_basis(&xi_l, &mut phi);
                for i in 0..3 { for c in 0..4 { du[self.idx(el, c, i)] -= w * phi[i] * fstar[c]; } }
                tri.eval_basis(&xi_r, &mut phi);
                for i in 0..3 { for c in 0..4 { du[self.idx(er, c, i)] += w * phi[i] * fstar[c]; } }
            }
        }
        // Boundary: reflecting/slip wall (zero normal flux)
        for e in 0..self.n_elems as u32 {
            let enodes = self.mesh.element_nodes(e);
            for lf in 0..3 {
                let fnodes = tri_face_nodes(lf, &enodes);
                let mut key: Vec<u32> = fnodes.iter().copied().collect();
                key.sort_unstable();
                let is_interior = interior_faces.iter().any(|f| { let mut k = f.2.clone(); k.sort_unstable(); k == key });
                if !is_interior {
                    let (nx, ny) = face_normal(&*self.mesh, &enodes, e, &fnodes);
                    let face_jac = face_size(&*self.mesh, &fnodes);
                    for q in 0..qr_face.n_points() {
                        let t = qr_face.points[q][0]; let w = qr_face.weights[q] * face_jac;
                        let xi = map_to_elem(e, &fnodes, t, &enodes);
                        tri.eval_basis(&xi, &mut phi);
                        let mut uqp = [0.0; 4];
                        for i in 0..3 { for c in 0..4 { uqp[c] += phi[i] * u[self.idx(e, c, i)]; } }
                        // Reflecting BC: mirror velocity -> set normal velocity to 0
                        let (r, uv, vv, p) = euler.cons_to_prim(&uqp);
                        let un = uv*nx + vv*ny;
                        let q_ref = euler.prim_to_cons(r, uv-2.*un*nx, vv-2.*un*ny, p);
                        let fstar = euler.lax_friedrichs_flux(&uqp, &q_ref, &[nx, ny]);
                        for i in 0..3 { for c in 0..4 { du[self.idx(e, c, i)] -= w * phi[i] * fstar[c]; } }
                    }
                }
            }
        }
        du
    }

    /// SSP-RK3 step
    pub fn step_rk3(&self, u: &mut [f64], dt: f64) {
        let k1 = self.rhs(u);
        let mut u1: Vec<f64> = u.iter().zip(k1.iter()).map(|(a,b)| a + dt*b).collect();
        let k2 = self.rhs(&u1);
        for i in 0..self.n_dofs {
            u1[i] = 0.75*u[i] + 0.25*(u1[i] + dt*k2[i]);
        }
        let k3 = self.rhs(&u1);
        for i in 0..self.n_dofs {
            u[i] = (1./3.)*u[i] + (2./3.)*(u1[i] + dt*k3[i]);
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn affine_jacobian_det(mesh: &dyn MeshTopology, nodes: &[u32]) -> ([[f64;2];2], f64) {
    let p0 = mesh.node_coords(nodes[0]); let p1 = mesh.node_coords(nodes[1]); let p2 = mesh.node_coords(nodes[2]);
    let j = [[p1[0]-p0[0], p2[0]-p0[0]], [p1[1]-p0[1], p2[1]-p0[1]]];
    (j, j[0][0]*j[1][1] - j[0][1]*j[1][0])
}
fn affine_inv_jac(j: [[f64;2];2]) -> [[f64;2];2] {
    let d = j[0][0]*j[1][1] - j[0][1]*j[1][0]; let id = 1.0/d;
    [[j[1][1]*id, -j[0][1]*id], [-j[1][0]*id, j[0][0]*id]]
}
fn affine_map(mesh: &dyn MeshTopology, nodes: &[u32], xi: &[f64]) -> (f64, f64) {
    let p0 = mesh.node_coords(nodes[0]); let p1 = mesh.node_coords(nodes[1]); let p2 = mesh.node_coords(nodes[2]);
    let x = p0[0] + xi[0]*(p1[0]-p0[0]) + xi[1]*(p2[0]-p0[0]);
    let y = p0[1] + xi[0]*(p1[1]-p0[1]) + xi[1]*(p2[1]-p0[1]);
    (x, y)
}

fn inv3(m: [[f64;3];3]) -> [[f64;3];3] {
    let d = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    let id = 1.0/d;
    [[(m[1][1]*m[2][2]-m[1][2]*m[2][1])*id, (m[0][2]*m[2][1]-m[0][1]*m[2][2])*id, (m[0][1]*m[1][2]-m[0][2]*m[1][1])*id],
     [(m[1][2]*m[2][0]-m[1][0]*m[2][2])*id, (m[0][0]*m[2][2]-m[0][2]*m[2][0])*id, (m[0][2]*m[1][0]-m[0][0]*m[1][2])*id],
     [(m[1][0]*m[2][1]-m[1][1]*m[2][0])*id, (m[0][1]*m[2][0]-m[0][0]*m[2][1])*id, (m[0][0]*m[1][1]-m[0][1]*m[1][0])*id]]
}

fn tri_face_nodes(lf: usize, enodes: &[u32]) -> Vec<u32> {
    match lf { 0 => vec![enodes[0], enodes[1]], 1 => vec![enodes[1], enodes[2]], 2 => vec![enodes[0], enodes[2]], _ => unreachable!() }
}

fn face_normal(mesh: &dyn MeshTopology, enodes: &[u32], _elem: u32, fnodes: &[u32]) -> (f64, f64) {
    let pa = mesh.node_coords(fnodes[0]); let pb = mesh.node_coords(fnodes[1]);
    let dx = pb[0]-pa[0]; let dy = pb[1]-pa[1];
    let len = (dx*dx + dy*dy).sqrt();
    let nx = dy/len; let ny = -dx/len; // outward perpendicular (right-hand rule)
    // Ensure outward: check against element centroid
    let cent = [(mesh.node_coords(enodes[0])[0] + mesh.node_coords(enodes[1])[0] + mesh.node_coords(enodes[2])[0])/3.0,
                (mesh.node_coords(enodes[0])[1] + mesh.node_coords(enodes[1])[1] + mesh.node_coords(enodes[2])[1])/3.0];
    let fmx = (pa[0] + pb[0])/2.0; let fmy = (pa[1] + pb[1])/2.0;
    let to_cent = [cent[0]-fmx, cent[1]-fmy];
    if nx*to_cent[0] + ny*to_cent[1] > 0.0 { (-nx, -ny) } else { (nx, ny) }
}

fn face_size(mesh: &dyn MeshTopology, fnodes: &[u32]) -> f64 {
    let pa = mesh.node_coords(fnodes[0]); let pb = mesh.node_coords(fnodes[1]);
    let dx = pb[0]-pa[0]; let dy = pb[1]-pa[1];
    (dx*dx + dy*dy).sqrt()
}

fn map_to_elem(_elem: u32, fnodes: &[u32], t: f64, enodes: &[u32]) -> Vec<f64> {
    // Find which local face fnodes correspond to
    for lf in 0..3 {
        let tf = tri_face_nodes(lf, enodes);
        let mut k1 = tf.clone(); k1.sort_unstable();
        let mut k2 = fnodes.to_vec(); k2.sort_unstable();
        if k1 == k2 {
            return match lf {
                0 => vec![t, 0.0],
                1 => vec![1.0 - t, t],
                2 => vec![0.0, 1.0 - t],
                _ => unreachable!()
            };
        }
    }
    vec![0.0, 0.0]
}

fn build_interior_faces(mesh: &dyn MeshTopology) -> Vec<(u32, u32, Vec<u32>)> {
    use std::collections::HashMap;
    let mut map: HashMap<Vec<u32>, (u32, Vec<u32>)> = HashMap::new();
    let mut faces = Vec::new();
    for e in mesh.elem_iter() {
        let enodes = mesh.element_nodes(e);
        for lf in 0..3 {
            let fno = tri_face_nodes(lf, &enodes);
            let mut key = fno.clone(); key.sort_unstable();
            match map.remove(&key) {
                None => { map.insert(key, (e, fno.clone())); }
                Some((prev, _fnodes)) => { faces.push((prev, e, fno.clone())); }
            }
        }
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn euler_flux_consistency() {
        let e = Euler2D::default();
        let q = e.prim_to_cons(1.0, 0.5, 0.3, 1.0);
        let f = e.lax_friedrichs_flux(&q, &q, &[1.0, 0.0]);
        let fx = e.flux_x(&q);
        for i in 0..4 { assert!((f[i] - fx[i]).abs() < 1e-12); }
    }

    #[test]
    fn dg_euler_2d_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dg = DgEuler2D::new(mesh);
        let u = dg.project_initial(&|_x, _y| (1.0, 0.5, 0.0, 1.0));
        let du = dg.rhs(&u);
        for v in &du { assert!(v.is_finite(), "non-finite RHS"); }
    }
}
