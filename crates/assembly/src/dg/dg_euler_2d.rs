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

    /// Roe-Pike numerical flux in direction n = (nx, ny).
    ///
    /// Standard Roe flux with **Harten-Hyman entropy fix**: if a wave speed
    /// straddles zero, replace |λ| with the smoother value
    /// `(λ_L² + λ_R²) / (λ_R − λ_L)` on the transonic branch. This
    /// eliminates the well-known Roe entropy-glitch across sonic points.
    ///
    /// Reference: LeVeque §15.3, Toro §11.
    pub fn roe_flux(&self, ql: &[f64; 4], qr: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        let g = self.gamma;
        let (nx, ny) = (n[0], n[1]);

        // Primitive states.
        let (rl, ul, vl, pl) = self.cons_to_prim(ql);
        let (rr, ur, vr, pr) = self.cons_to_prim(qr);
        let hl = (ql[3] + pl) / rl;
        let hr = (qr[3] + pr) / rr;

        // Roe averages.
        let srl = rl.sqrt();
        let srr = rr.sqrt();
        let inv_sum = 1.0 / (srl + srr);
        let u_h = (srl * ul + srr * ur) * inv_sum;
        let v_h = (srl * vl + srr * vr) * inv_sum;
        let h_h = (srl * hl + srr * hr) * inv_sum;
        let q2  = u_h * u_h + v_h * v_h;
        let a2  = ((g - 1.0) * (h_h - 0.5 * q2)).max(1e-14);
        let a_h = a2.sqrt();

        // Normal/tangential Roe-averaged velocities.
        let vn_h = u_h * nx + v_h * ny;
        // Wave speeds.
        let lam1 = vn_h - a_h;
        let lam2 = vn_h;
        let lam3 = vn_h + a_h;

        // Left/right normal velocities for entropy fix.
        let vnl = ul * nx + vl * ny;
        let vnr = ur * nx + vr * ny;
        let al = (g * pl / rl).sqrt();
        let ar = (g * pr / rr).sqrt();

        // Harten-Hyman entropy fix. delta = max(0, λ_R − λ_L) per acoustic wave.
        let fix = |lam: f64, ll: f64, lr: f64| -> f64 {
            let delta = (lr - ll).max(0.0);
            if lam.abs() < 0.5 * delta && delta > 1e-14 {
                0.5 * (lam * lam / delta + delta)
            } else {
                lam.abs()
            }
        };
        let abs_l1 = fix(lam1, vnl - al, vnr - ar);
        let abs_l2 = lam2.abs();
        let abs_l3 = fix(lam3, vnl + al, vnr + ar);

        // Jumps.
        let dr = rr - rl;
        let du = ur - ul;
        let dv = vr - vl;
        let dp = pr - pl;
        let dvn = du * nx + dv * ny;

        // Wave strengths (α_k coefficients) — use Roe-averaged density.
        let r_h = srl * srr;
        let alpha1 = 0.5 * (dp - r_h * a_h * dvn) / a2;
        let alpha2 = dr - dp / a2;
        let alpha3 = 0.5 * (dp + r_h * a_h * dvn) / a2;

        // Right eigenvectors K_k in conservative variables.
        // K_1 = (1, u - a·nx, v - a·ny, H - a·vn)
        // K_2 (density-only) = (1, u, v, ½q²) — combined with shear (contact) waves
        // K_3 = (1, u + a·nx, v + a·ny, H + a·vn)
        // Additionally, shear jumps carry a tangential-velocity wave with speed λ_2.
        let tx = -ny;
        let ty =  nx;
        let dvt = du * tx + dv * ty;
        // Shear (contact) wave strength for tangential velocity.
        let alpha_shear = r_h * dvt;

        let k1 = [1.0, u_h - a_h * nx, v_h - a_h * ny, h_h - a_h * vn_h];
        let k2_dens = [1.0, u_h, v_h, 0.5 * q2];
        let k2_shear = [0.0, tx, ty, u_h * tx + v_h * ty];
        let k3 = [1.0, u_h + a_h * nx, v_h + a_h * ny, h_h + a_h * vn_h];

        // Central flux average.
        let fl = self.flux_n(ql, n);
        let fr = self.flux_n(qr, n);

        let mut f = [0.0_f64; 4];
        for i in 0..4 {
            let diss = abs_l1 * alpha1 * k1[i]
                     + abs_l2 * (alpha2 * k2_dens[i] + alpha_shear * k2_shear[i])
                     + abs_l3 * alpha3 * k3[i];
            f[i] = 0.5 * (fl[i] + fr[i]) - 0.5 * diss;
        }
        f
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
    n_dofs: usize,
    dofs_per_elem: usize,
    pub use_limiter: bool,
}

impl DgEuler2D {
    pub fn new(mesh: impl MeshTopology + 'static) -> Self {
        let n_elems = mesh.n_elements();
        let dofs_per_elem = 3; // P1 triangle
        let n_dofs = n_elems * dofs_per_elem * 4; // 4 Euler components
        Self { mesh: Box::new(mesh), euler: Euler2D::default(), n_elems, n_dofs, dofs_per_elem, use_limiter: false }
    }
    pub fn with_limiter(mut self, on: bool) -> Self { self.use_limiter = on; self }

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
                let (_jac, det) = affine_jacobian_det(&*self.mesh, enodes);
                let vol = (w * det).abs();
                let (cx, cy) = affine_map(&*self.mesh, enodes, xi);
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
            let (jac, det) = affine_jacobian_det(&*self.mesh, enodes);
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
            let (nx, ny) = face_normal(&*self.mesh, en_l, el, &fn_l);
            let face_jac = face_size(&*self.mesh, &fn_l);
            // Map face qp to reference element for left and right
            for q in 0..qr_face.n_points() {
                let t = qr_face.points[q][0]; let w = qr_face.weights[q] * face_jac;
                let xi_l = map_to_elem(el, &fn_l, t, en_l);
                let xi_r = map_to_elem(er, &fn_l, t, en_r);
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
                let fnodes = tri_face_nodes(lf, enodes);
                let mut key: Vec<u32> = fnodes.to_vec();
                key.sort_unstable();
                let is_interior = interior_faces.iter().any(|f| { let mut k = f.2.clone(); k.sort_unstable(); k == key });
                if !is_interior {
                    let (nx, ny) = face_normal(&*self.mesh, enodes, e, &fnodes);
                    let face_jac = face_size(&*self.mesh, &fnodes);
                    for q in 0..qr_face.n_points() {
                        let t = qr_face.points[q][0]; let w = qr_face.weights[q] * face_jac;
                        let xi = map_to_elem(e, &fnodes, t, enodes);
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
        let face_elems = if self.use_limiter { Some(self.build_face_elems_2d()) } else { None };
        let k1 = self.rhs(u);
        let mut u1: Vec<f64> = u.iter().zip(k1.iter()).map(|(a,b)| a + dt*b).collect();
        if let Some(ref fe) = face_elems { self.apply_limiter_2d(&mut u1, fe); }
        let k2 = self.rhs(&u1);
        for i in 0..self.n_dofs { u1[i] = 0.75*u[i] + 0.25*(u1[i] + dt*k2[i]); }
        if let Some(ref fe) = face_elems { self.apply_limiter_2d(&mut u1, fe); }
        let k3 = self.rhs(&u1);
        for i in 0..self.n_dofs { u[i] = (1./3.)*u[i] + (2./3.)*(u1[i] + dt*k3[i]); }
        if let Some(ref fe) = face_elems { self.apply_limiter_2d(u, fe); }
    }

    fn build_face_elems_2d(&self) -> Vec<(u32, Option<u32>)> {
        use std::collections::HashMap;
        let mut fm: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut faces = Vec::new();
        for e in 0..self.n_elems as u32 {
            let en = self.mesh.element_nodes(e);
            for lf in 0..3 {
                let (a, b) = match lf { 0 => (en[0], en[1]), 1 => (en[1], en[2]), _ => (en[2], en[0]) };
                let mut key = vec![a, b]; key.sort_unstable();
                match fm.remove(&key) { None => { fm.insert(key, e); } Some(prev) => { faces.push((prev, Some(e))); } }
            }
        }
        for (_, l) in fm { faces.push((l, None)); }
        faces
    }

    fn apply_limiter_2d(&self, u: &mut [f64], face_elems: &[(u32, Option<u32>)]) {
        let dofs_per_elem = self.dofs_per_elem; // 3 for TriP1
        let mut comp_buf = vec![0.0_f64; self.n_elems * dofs_per_elem];
        for c in 0..4 {
            for e in 0..self.n_elems {
                for i in 0..dofs_per_elem { comp_buf[e * dofs_per_elem + i] = u[self.idx(e as u32, c, i)]; }
            }
            crate::physics::hyperbolic::limiter_barth_jespersen(
                &mut comp_buf, self.n_elems, dofs_per_elem, &[], face_elems,
            );
            for e in 0..self.n_elems {
                for i in 0..dofs_per_elem { u[self.idx(e as u32, c, i)] = comp_buf[e * dofs_per_elem + i]; }
            }
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
            let fno = tri_face_nodes(lf, enodes);
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
    use fem_mesh::Mesh;

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
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dg = DgEuler2D::new(mesh);
        let u = dg.project_initial(&|_x, _y| (1.0, 0.5, 0.0, 1.0));
        let du = dg.rhs(&u);
        for v in &du { assert!(v.is_finite(), "non-finite RHS"); }
    }

    // ── Roe flux regression tests (Phase 0.2 fix) ─────────────────────────

    /// Consistency: F(U, U, n) = F_phys(U, n).
    #[test]
    fn roe_flux_consistent_on_uniform_state() {
        let euler = Euler2D::default();
        let q = euler.prim_to_cons(1.0, 0.5, -0.3, 1.2);
        let n = [1.0_f64 / 2.0_f64.sqrt(), 1.0_f64 / 2.0_f64.sqrt()];
        let f_roe = euler.roe_flux(&q, &q, &n);
        let f_phys = euler.flux_n(&q, &n);
        for i in 0..4 {
            assert!((f_roe[i] - f_phys[i]).abs() < 1e-12,
                "Roe(U,U) must equal F(U): comp {i}: roe={} phys={}", f_roe[i], f_phys[i]);
        }
    }

    /// Roe MUST differ from Lax-Friedrichs on a genuine Riemann problem —
    /// otherwise the earlier stub (roe → LF) is still active.
    #[test]
    fn roe_flux_differs_from_lax_friedrichs() {
        let euler = Euler2D::default();
        // 1-D Sod-like shock along x direction.
        let ql = euler.prim_to_cons(1.0, 0.0, 0.0, 1.0);
        let qr = euler.prim_to_cons(0.125, 0.0, 0.0, 0.1);
        let n = [1.0_f64, 0.0];
        let f_roe = euler.roe_flux(&ql, &qr, &n);
        let f_lf  = euler.lax_friedrichs_flux(&ql, &qr, &n);
        let diff: f64 = (0..4).map(|i| (f_roe[i] - f_lf[i]).abs()).sum();
        assert!(diff > 1e-6,
            "Roe and Lax-Friedrichs must produce different fluxes on a shock; \
             stub regression? ‖diff‖ = {diff:.3e}");
    }

    /// Roe with entropy fix must remain finite through a sonic point.
    #[test]
    fn roe_flux_transonic_entropy_fix_finite() {
        let euler = Euler2D::default();
        // Transonic rarefaction (left state supersonic, right subsonic).
        let ql = euler.prim_to_cons(3.0, 0.9,  0.0, 3.0);
        let qr = euler.prim_to_cons(1.0, 0.1,  0.0, 1.0);
        let n = [1.0_f64, 0.0];
        let f = euler.roe_flux(&ql, &qr, &n);
        for &v in &f {
            assert!(v.is_finite(), "Roe flux must be finite through sonic point: {v}");
        }
    }
}
