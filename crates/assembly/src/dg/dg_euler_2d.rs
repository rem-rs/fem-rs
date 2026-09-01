//! 2-D Euler equations via DG with explicit RK time stepping.
//! Supports arbitrary polynomial order (P1, P2, P3) on Tri3 meshes.

use std::collections::HashMap;
use nalgebra::DMatrix;
use fem_mesh::topology::MeshTopology;
use fem_element::lagrange::factory::{TriPk, TetPk};
use fem_element::lagrange::SegP1;
use fem_element::ReferenceElement;

/// 2-D Euler physics and numerical fluxes.
#[derive(Clone)]
pub struct Euler2D { pub gamma: f64 }
impl Default for Euler2D { fn default() -> Self { Self { gamma: 1.4 } } }

impl Euler2D {
    pub fn prim_to_cons(&self, rho: f64, u: f64, v: f64, p: f64) -> [f64; 4] {
        let e = p/(self.gamma-1.0) + 0.5*rho*(u*u+v*v);
        [rho, rho*u, rho*v, e]
    }
    pub fn cons_to_prim(&self, q: &[f64; 4]) -> (f64, f64, f64, f64) {
        let rho = q[0].max(1e-14); let u = q[1]/rho; let v = q[2]/rho;
        let ke = 0.5*rho*(u*u+v*v); let p = ((self.gamma-1.0)*(q[3]-ke)).max(1e-14);
        (rho, u, v, p)
    }
    pub fn flux_x(&self, q: &[f64; 4]) -> [f64; 4] {
        let (r,u,v,p)=self.cons_to_prim(q); [r*u, r*u*u+p, r*u*v, u*(q[3]+p)]
    }
    pub fn flux_y(&self, q: &[f64; 4]) -> [f64; 4] {
        let (r,u,v,p)=self.cons_to_prim(q); [r*v, r*u*v, r*v*v+p, v*(q[3]+p)]
    }
    pub fn flux_n(&self, q: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        let fx=self.flux_x(q); let fy=self.flux_y(q);
        [fx[0]*n[0]+fy[0]*n[1], fx[1]*n[0]+fy[1]*n[1], fx[2]*n[0]+fy[2]*n[1], fx[3]*n[0]+fy[3]*n[1]]
    }
    pub fn max_speed(&self, q: &[f64; 4]) -> f64 {
        let (r,u,v,p)=self.cons_to_prim(q); let a=(self.gamma*p/r).sqrt(); (u*u+v*v).sqrt()+a
    }
    pub fn lax_friedrichs_flux(&self, ql: &[f64; 4], qr: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        let fl=self.flux_n(ql,n); let fr=self.flux_n(qr,n);
        let a=self.max_speed(ql).max(self.max_speed(qr));
        let mut f=[0.0;4]; for i in 0..4 { f[i]=0.5*(fl[i]+fr[i])-0.5*a*(qr[i]-ql[i]); } f
    }
    pub fn roe_flux(&self, ql: &[f64; 4], qr: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        let g = self.gamma;
        let (nx, ny) = (n[0], n[1]);
        let (rl, ul, vl, pl) = self.cons_to_prim(ql);
        let (rr, ur, vr, pr) = self.cons_to_prim(qr);
        let hl = (ql[3] + pl) / rl;
        let hr = (qr[3] + pr) / rr;
        let srl = rl.sqrt(); let srr = rr.sqrt(); let isum = 1.0 / (srl + srr);
        let u_h = (srl*ul + srr*ur) * isum;
        let v_h = (srl*vl + srr*vr) * isum;
        let h_h = (srl*hl + srr*hr) * isum;
        let q2 = u_h*u_h + v_h*v_h;
        let a2 = ((g - 1.0) * (h_h - 0.5*q2)).max(1e-14);
        let a_h = a2.sqrt();
        let vn_h = u_h*nx + v_h*ny;
        let lam1 = vn_h - a_h; let lam2 = vn_h; let lam3 = vn_h + a_h;
        let eps = 0.1 * a_h;
        let fix = |l: f64| if l.abs() < eps { (l*l + eps*eps) / (2.0*eps) } else { l.abs() };
        let (a1, a2, a3) = (fix(lam1), fix(lam2), fix(lam3));
        let dr  = qr[0] - ql[0]; let dru = qr[1] - ql[1];
        let drv = qr[2] - ql[2]; let dre = qr[3] - ql[3];
        let dvs = dr - (qr[3] - qr[1]*ur - qr[2]*vr + pr/(g-1.0)) / (a_h*a_h) + (ql[3] - ql[1]*ul - ql[2]*vl + pl/(g-1.0)) / (a_h*a_h);
        let dp = (qr[3] - qr[1]*ur - qr[2]*vr + pr/(g-1.0)) - (ql[3] - ql[1]*ul - ql[2]*vl + pl/(g-1.0));
        let alpha1 = 0.5 * (dvs + dp / a_h) / a_h;
        let alpha2 = dr - dp / (a_h*a_h);
        let alpha3 = 0.5 * (dvs - dp / a_h) / a_h;
        let r1 = [1.0, u_h - a_h*nx, v_h - a_h*ny, h_h - a_h*vn_h];
        let r2 = [1.0, u_h, v_h, 0.5*q2 + h_h];
        let r3 = [1.0, u_h + a_h*nx, v_h + a_h*ny, h_h + a_h*vn_h];
        let fl = self.flux_n(ql, n); let fr = self.flux_n(qr, n);
        let mut f = [0.0; 4];
        for i in 0..4 {
            f[i] = 0.5*(fl[i]+fr[i]) - 0.5*(a1*alpha1*r1[i] + a2*alpha2*r2[i] + a3*alpha3*r3[i]);
        }
        f
    }
    pub fn hllc_flux(&self, ql: &[f64; 4], qr: &[f64; 4], n: &[f64; 2]) -> [f64; 4] {
        let g = self.gamma;
        let (nx, ny) = (n[0], n[1]);
        let (rl, ul, vl, pl) = self.cons_to_prim(ql);
        let (rr, ur, vr, pr) = self.cons_to_prim(qr);
        let al = (g*pl/rl).sqrt(); let ar = (g*pr/rr).sqrt();
        let unl = ul*nx + vl*ny; let unr = ur*nx + vr*ny;
        let rl_sqrt = rl.sqrt(); let rr_sqrt = rr.sqrt();
        let u_tilde = (rl_sqrt*unl + rr_sqrt*unr) / (rl_sqrt + rr_sqrt);
        let a_tilde = ((rl_sqrt*al + rr_sqrt*ar) / (rl_sqrt + rr_sqrt)
                     + 0.5*(unl - unr).abs()).max(1e-14);
        let s_l = (u_tilde - a_tilde).min(unl - al);
        let s_r = (u_tilde + a_tilde).max(unr + ar);
        let s_m = (pr - pl + rl*unl*(s_l - unl) - rr*unr*(s_r - unr))
                / (rl*(s_l - unl) - rr*(s_r - unr)).max(1e-14);
        let fl = self.flux_n(ql, n);
        let fr = self.flux_n(qr, n);
        if s_l >= 0.0 { return fl; }
        if s_r <= 0.0 { return fr; }
        if s_m >= 0.0 {
            let fact = rl*(s_l - unl) / (s_l - s_m);
            let mut qs = *ql;
            qs[0] = fact;
            qs[1] = fact * s_m * nx + ql[1] - rl*unl*nx;
            qs[2] = fact * s_m * ny + ql[2] - rl*unl*ny;
            qs[3] = fact * (ql[3]/rl + (s_m - unl)*(s_m + pl/(rl*(s_l - unl))));
            let mut f = fl;
            for i in 0..4 { f[i] += s_l * (qs[i] - ql[i]); }
            f
        } else {
            let fact = rr*(s_r - unr) / (s_r - s_m);
            let mut qs = *qr;
            qs[0] = fact;
            qs[1] = fact * s_m * nx + qr[1] - rr*unr*nx;
            qs[2] = fact * s_m * ny + qr[2] - rr*unr*ny;
            qs[3] = fact * (qr[3]/rr + (s_m - unr)*(s_m + pr/(rr*(s_r - unr))));
            let mut f = fr;
            for i in 0..4 { f[i] += s_r * (qs[i] - qr[i]); }
            f
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EulerFluxKind {
    #[default]
    LaxFriedrichs,
    Roe,
    Hllc,
}

pub struct DgEuler2D {
    mesh: Box<dyn MeshTopology + Send + Sync>,
    euler: Euler2D,
    n_elems: usize,
    n_dofs: usize,
    order: u8,
    dofs_per_elem: usize,
    pub use_limiter: bool,
    pub periodic: bool,
    pub flux_kind: EulerFluxKind,
}

impl DgEuler2D {
    pub fn new(mesh: impl MeshTopology + 'static) -> Self { Self::with_order(mesh, 1) }
    pub fn with_order(mesh: impl MeshTopology + 'static, order: u8) -> Self {
        let n_elems = mesh.n_elements();
        let et = mesh.element_type(0);
        let re = ref_elem_vol(et, order);
        let dofs_per_elem = re.n_dofs();
        let n_dofs = n_elems * dofs_per_elem * 4;
        Self { mesh: Box::new(mesh), euler: Euler2D::default(), n_elems, n_dofs, order, dofs_per_elem, use_limiter: false, periodic: false, flux_kind: EulerFluxKind::default() }
    }
    pub fn n_dofs(&self) -> usize { self.n_dofs }
    pub fn dofs_per_elem(&self) -> usize { self.dofs_per_elem }
    pub fn order(&self) -> u8 { self.order }

    pub fn h_min(&self) -> f64 {
        let mut h = f64::MAX;
        for e in 0..self.n_elems as u32 {
            let en = self.mesh.element_nodes(e);
            let p0=self.mesh.node_coords(en[0]); let p1=self.mesh.node_coords(en[1]); let p2=self.mesh.node_coords(en[2]);
            let l01=((p1[0]-p0[0]).powi(2)+(p1[1]-p0[1]).powi(2)).sqrt();
            let l02=((p2[0]-p0[0]).powi(2)+(p2[1]-p0[1]).powi(2)).sqrt();
            let l12=((p2[0]-p1[0]).powi(2)+(p2[1]-p1[1]).powi(2)).sqrt();
            h = h.min(l01.min(l02.min(l12)));
        }
        h
    }
    pub fn compute_error(&self, u: &[f64], u_ref: &[f64]) -> f64 {
        (0..self.n_dofs).map(|i| (u[i]-u_ref[i]).powi(2)).sum::<f64>().sqrt()
    }

    fn idx(&self, e: u32, c: usize, ld: usize) -> usize { (e as usize * self.dofs_per_elem + ld) * 4 + c }

    pub fn project_initial(&self, init: &dyn Fn(f64, f64) -> (f64, f64, f64, f64)) -> Vec<f64> {
        let euler = &self.euler; let dp = self.dofs_per_elem;
        let mut u = vec![0.0; self.n_dofs];
        let et = self.mesh.element_type(0);
        let re = ref_elem_vol(et, self.order); let q_ord = (2*self.order+1).max(1);
        let qr = re.quadrature(q_ord);
        for e in 0..self.n_elems as u32 {
            let enodes = self.mesh.element_nodes(e);
            let mut mass = DMatrix::<f64>::zeros(dp, dp);
            let mut rhs = DMatrix::<f64>::zeros(dp, 4);
            let mut phi = vec![0.0; dp];
            for (qi, xi) in qr.points.iter().enumerate() {
                let w = qr.weights[qi];
                re.eval_basis(xi, &mut phi);
                let (_jac, det) = affine_jacobian_det(&*self.mesh, enodes);
                let vol = (w * det).abs();
                let (cx, cy) = affine_map(&*self.mesh, enodes, xi);
                let (r, uvel, vvel, p) = init(cx, cy);
                let cons = euler.prim_to_cons(r, uvel, vvel, p);
                for i in 0..dp { for j in 0..dp { mass[(i,j)] += vol * phi[i] * phi[j]; } }
                for i in 0..dp { for c in 0..4 { rhs[(i,c)] += vol * phi[i] * cons[c]; } }
            }
            let minv = mass.try_inverse().expect("singular mass matrix");
            for i in 0..dp { for c in 0..4 {
                let mut s = 0.0; for j in 0..dp { s += minv[(i,j)] * rhs[(j,c)]; }
                u[self.idx(e, c, i)] = s;
            }}
        }
        u
    }

    pub fn rhs(&self, u: &[f64]) -> Vec<f64> {
        let euler = &self.euler; let dp = self.dofs_per_elem; let dim = 2;
        let mut du = vec![0.0; self.n_dofs];
        let et = self.mesh.element_type(0);
        let re = ref_elem_vol(et, self.order);
        let q_ord = (2*self.order+1).max(1);
        let qr_vol = re.quadrature(q_ord);
        let qr_face = SegP1.quadrature(2);
        let interior_faces = build_interior_faces(&*self.mesh);

        // Face flux interior
        for face in &interior_faces {
            let (el, er, fn_l) = (face.0, face.1, face.2.clone());
            let en_l = self.mesh.element_nodes(el); let en_r = self.mesh.element_nodes(er);
            let (nx, ny) = face_normal(&*self.mesh, en_l, el, &fn_l);
            let fjac = face_size(&*self.mesh, &fn_l);
            let mut phi = vec![0.0; dp];
            for q in 0..qr_face.n_points() {
                let t = qr_face.points[q][0]; let w = qr_face.weights[q] * fjac;
                let xi_l = map_to_elem(el, &fn_l, t, en_l);
                let xi_r = map_to_elem(er, &fn_l, t, en_r);
                re.eval_basis(&xi_l, &mut phi); let mut ul = [0.0; 4];
                for i in 0..dp { for c in 0..4 { ul[c] += phi[i] * u[self.idx(el, c, i)]; }}
                re.eval_basis(&xi_r, &mut phi); let mut ur = [0.0; 4];
                for i in 0..dp { for c in 0..4 { ur[c] += phi[i] * u[self.idx(er, c, i)]; }}
                let fstar = match self.flux_kind {
                    EulerFluxKind::LaxFriedrichs => euler.lax_friedrichs_flux(&ul, &ur, &[nx, ny]),
                    EulerFluxKind::Roe => euler.roe_flux(&ul, &ur, &[nx, ny]),
                    EulerFluxKind::Hllc => euler.hllc_flux(&ul, &ur, &[nx, ny]),
                };
                re.eval_basis(&xi_l, &mut phi);
                for i in 0..dp { for c in 0..4 { du[self.idx(el, c, i)] -= w * phi[i] * fstar[c]; }}
                re.eval_basis(&xi_r, &mut phi);
                for i in 0..dp { for c in 0..4 { du[self.idx(er, c, i)] += w * phi[i] * fstar[c]; }}
            }
        }

        // Boundary (reflecting)
        if !self.periodic {
            for e in 0..self.n_elems as u32 {
                let enodes = self.mesh.element_nodes(e);
                for lf in 0..3 {
                    let fnodes = tri_face_nodes(lf, enodes);
                    let mut k: Vec<u32> = fnodes.to_vec(); k.sort_unstable();
                    let is_int = interior_faces.iter().any(|f| { let mut k2=f.2.clone(); k2.sort_unstable(); k2==k });
                    if is_int { continue; }
                    let (nx, ny) = face_normal(&*self.mesh, enodes, e, &fnodes);
                    let fjac = face_size(&*self.mesh, &fnodes);
                    let mut phi = vec![0.0; dp];
                    for q in 0..qr_face.n_points() {
                        let t = qr_face.points[q][0]; let w = qr_face.weights[q] * fjac;
                        let xi = map_to_elem(e, &fnodes, t, enodes);
                        re.eval_basis(&xi, &mut phi); let mut uqp = [0.0; 4];
                        for i in 0..dp { for c in 0..4 { uqp[c] += phi[i] * u[self.idx(e, c, i)]; }}
                        let (r, uv, vv, p) = euler.cons_to_prim(&uqp);
                        let un = uv*nx+vv*ny;
                        let qref = euler.prim_to_cons(r, uv-2.*un*nx, vv-2.*un*ny, p);
                        let fstar = match self.flux_kind {
                            EulerFluxKind::LaxFriedrichs => euler.lax_friedrichs_flux(&uqp, &qref, &[nx, ny]),
                            EulerFluxKind::Roe => euler.roe_flux(&uqp, &qref, &[nx, ny]),
                            EulerFluxKind::Hllc => euler.hllc_flux(&uqp, &qref, &[nx, ny]),
                        };
                        for i in 0..dp { for c in 0..4 { du[self.idx(e, c, i)] -= w * phi[i] * fstar[c]; }}
                    }
                }
            }
        }

        // Volume + inverse mass
        let mut grad = vec![0.0; dp * dim];
        for e in 0..self.n_elems as u32 {
            let enodes = self.mesh.element_nodes(e);
            let (_jac, det) = affine_jacobian_det(&*self.mesh, enodes);
            let inv_j = affine_inv_jac(_jac);

            // Read face contributions from du
            let mut elem_fac = DMatrix::<f64>::zeros(dp, 4);
            for i in 0..dp { for c in 0..4 { elem_fac[(i,c)] = du[self.idx(e, c, i)]; }}

            // Volume integral + mass matrix
            let mut elem_vol = DMatrix::<f64>::zeros(dp, 4);
            let mut mass = DMatrix::<f64>::zeros(dp, dp);
            let mut phi = vec![0.0; dp];
            for q in 0..qr_vol.n_points() {
                let xi = &qr_vol.points[q]; let w = qr_vol.weights[q];
                re.eval_basis(xi, &mut phi); re.eval_grad_basis(xi, &mut grad);
                let vol = (w * det).abs();
                for i in 0..dp { for j in 0..dp { mass[(i,j)] += vol * phi[i] * phi[j]; }}
                let mut uqp = [0.0; 4];
                for i in 0..dp { for c in 0..4 { uqp[c] += phi[i] * u[self.idx(e, c, i)]; }}
                let fx = euler.flux_x(&uqp); let fy = euler.flux_y(&uqp);
                for i in 0..dp {
                    let gx = inv_j[0][0]*grad[i*dim] + inv_j[1][0]*grad[i*dim+1];
                    let gy = inv_j[0][1]*grad[i*dim] + inv_j[1][1]*grad[i*dim+1];
                    for c in 0..4 { elem_vol[(i,c)] += vol * (fx[c]*gx + fy[c]*gy); }
                }
            }
            // M⁻¹ * (vol + face)
            let minv = mass.try_inverse().expect("singular mass matrix");
            for i in 0..dp { for c in 0..4 {
                let mut s = 0.0; for j in 0..dp { s += minv[(i,j)] * (elem_vol[(j,c)] + elem_fac[(j,c)]); }
                du[self.idx(e, c, i)] = s;
            }}
        }
        du
    }

    pub fn step_rk3(&self, u: &mut [f64], dt: f64) {
        let fe = if self.use_limiter { Some(self.build_neighbors()) } else { None };
        let k1 = self.rhs(u); let mut u1: Vec<f64> = (0..self.n_dofs).map(|i| u[i]+dt*k1[i]).collect();
        if let Some(ref f) = fe { self.limit(&mut u1, f); }
        let k2 = self.rhs(&u1);
        for i in 0..self.n_dofs { u1[i] = 0.75*u[i] + 0.25*(u1[i] + dt*k2[i]); }
        if let Some(ref f) = fe { self.limit(&mut u1, f); }
        let k3 = self.rhs(&u1);
        for i in 0..self.n_dofs { u[i] = (1.0/3.0)*u[i] + (2.0/3.0)*(u1[i] + dt*k3[i]); }
        if let Some(ref f) = fe { self.limit(u, f); }
    }

    fn build_neighbors(&self) -> Vec<(u32, Option<u32>)> {
        let mut fm: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut faces = Vec::new();
        for e in 0..self.n_elems as u32 { let en = self.mesh.element_nodes(e);
            for lf in 0..3 { let (a,b)=match lf{0=>(en[0],en[1]),1=>(en[1],en[2]),_=>(en[2],en[0])};
                let mut k=vec![a,b]; k.sort();
                match fm.remove(&k) { None=>{fm.insert(k,e);} Some(p)=>{faces.push((p,Some(e)));} }
            }
        }
        for (_, l) in fm { faces.push((l, None)); }
        faces
    }

    fn limit(&self, u: &mut [f64], f: &[(u32, Option<u32>)]) {
        let dp = self.dofs_per_elem; let mut buf = vec![0.0; self.n_elems * dp];
        for c in 0..4 {
            for e in 0..self.n_elems { for i in 0..dp { buf[e*dp+i] = u[self.idx(e as u32,c,i)]; }}
            crate::physics::hyperbolic::limiter_barth_jespersen(&mut buf, self.n_elems, dp, &[], f);
            for e in 0..self.n_elems { for i in 0..dp { u[self.idx(e as u32,c,i)] = buf[e*dp+i]; }}
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn affine_jacobian_det(mesh: &dyn MeshTopology, nodes: &[u32]) -> ([[f64;2];2], f64) {
    let p0=mesh.node_coords(nodes[0]);let p1=mesh.node_coords(nodes[1]);let p2=mesh.node_coords(nodes[2]);
    let j=[[p1[0]-p0[0],p2[0]-p0[0]],[p1[1]-p0[1],p2[1]-p0[1]]]; (j, j[0][0]*j[1][1]-j[0][1]*j[1][0])
}
fn affine_inv_jac(j:[[f64;2];2])->[[f64;2];2]{
    let d=j[0][0]*j[1][1]-j[0][1]*j[1][0];let id=1.0/d;
    [[j[1][1]*id,-j[0][1]*id],[-j[1][0]*id,j[0][0]*id]]
}
fn affine_map(mesh:&dyn MeshTopology,nodes:&[u32],xi:&[f64])->(f64,f64){
    let p0=mesh.node_coords(nodes[0]);let p1=mesh.node_coords(nodes[1]);let p2=mesh.node_coords(nodes[2]);
    (p0[0]+xi[0]*(p1[0]-p0[0])+xi[1]*(p2[0]-p0[0]),p0[1]+xi[0]*(p1[1]-p0[1])+xi[1]*(p2[1]-p0[1]))
}

fn ref_elem_vol(et: fem_mesh::element_type::ElementType, o: u8) -> Box<dyn ReferenceElement> {
    use fem_mesh::element_type::ElementType::*;
    use fem_element::lagrange::factory::{TriPk, TetPk};
use fem_element::lagrange::*;
    match (et, o) {
        (Tri3,1)=>Box::new(TriP1),(Tri3,2)=>Box::new(TriPk::new(2)),(Tri3,3)=>Box::new(TriPk::new(3)),
        (Tet4,1)=>Box::new(TetP1),(Tet4,2)=>Box::new(TetP2),(Tet4,3)=>Box::new(TetPk::new(3)),
        _=>panic!("ref_elem_vol: unsupported ({et:?}, o={o})"),
    }
}

fn tri_face_nodes(lf:usize,enodes:&[u32])->Vec<u32>{
    match lf{0=>vec![enodes[0],enodes[1]],1=>vec![enodes[1],enodes[2]],_=>vec![enodes[0],enodes[2]]}
}

fn face_normal(mesh:&dyn MeshTopology,enodes:&[u32],_elem:u32,fnodes:&[u32])->(f64,f64){
    let pa=mesh.node_coords(fnodes[0]);let pb=mesh.node_coords(fnodes[1]);
    let dx=pb[0]-pa[0];let dy=pb[1]-pa[1];let len=(dx*dx+dy*dy).sqrt();
    let (nx,ny)=(dy/len,-dx/len);
    let cent=[(mesh.node_coords(enodes[0])[0]+mesh.node_coords(enodes[1])[0]+mesh.node_coords(enodes[2])[0])/3.0,
              (mesh.node_coords(enodes[0])[1]+mesh.node_coords(enodes[1])[1]+mesh.node_coords(enodes[2])[1])/3.0];
    let fmx=(pa[0]+pb[0])/2.0;let fmy=(pa[1]+pb[1])/2.0;
    if nx*(cent[0]-fmx)+ny*(cent[1]-fmy)>0.0{(-nx,-ny)}else{(nx,ny)}
}

fn face_size(mesh:&dyn MeshTopology,fnodes:&[u32])->f64{
    let pa=mesh.node_coords(fnodes[0]);let pb=mesh.node_coords(fnodes[1]);
    ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt()
}

fn map_to_elem(_elem:u32,fnodes:&[u32],t:f64,enodes:&[u32])->Vec<f64>{
    for lf in 0..3{let tf=tri_face_nodes(lf,enodes);
        let mut k1=tf.clone();k1.sort_unstable();
        let mut k2=fnodes.to_vec();k2.sort_unstable();
        if k1==k2{return match lf{0=>vec![t,0.0],1=>vec![1.0-t,t],_=>vec![0.0,1.0-t]}}
    }
    vec![0.0,0.0]
}

fn build_interior_faces(mesh:&dyn MeshTopology)->Vec<(u32,u32,Vec<u32>)>{
    let mut map:HashMap<Vec<u32>,(u32,Vec<u32>)>=HashMap::new();
    let mut faces=Vec::new();
    for e in mesh.elem_iter(){let enodes=mesh.element_nodes(e);
        for lf in 0..3{let fno=tri_face_nodes(lf,enodes);
            let mut key=fno.clone();key.sort_unstable();
            match map.remove(&key){None=>{map.insert(key,(e,fno.clone()));}Some((prev,_))=>{faces.push((prev,e,fno.clone()));}}
        }
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_element::lagrange::factory::{TriPk, TetPk};
use fem_element::lagrange::*;

    fn make_dg(n: usize, order: u8) -> DgEuler2D {
        let mesh = Mesh::<2>::unit_square_tri(n);
        DgEuler2D::with_order(mesh, order)
    }

    #[test]
    fn euler_flux_consistency() {
        let e=Euler2D::default(); let q=e.prim_to_cons(1.0,0.5,0.3,1.0);
        let f=e.lax_friedrichs_flux(&q,&q,&[1.0,0.0]); let fx=e.flux_x(&q);
        for i in 0..4{assert!((f[i]-fx[i]).abs()<1e-12);}
    }

    #[test]
    fn dg_euler_2d_p1_finite() {
        let dg = make_dg(4, 1);
        let u = dg.project_initial(&|_,_| (1.0, 0.5, 0.0, 1.0));
        let du = dg.rhs(&u);
        for v in &du { assert!(v.is_finite()); }
    }

    #[test]
    fn dg_euler_2d_p2_finite() {
        let dg = make_dg(4, 2);
        let u = dg.project_initial(&|_,_| (1.0, 0.5, 0.0, 1.0));
        let du = dg.rhs(&u);
        for v in &du { assert!(v.is_finite()); }
    }

    #[test]
    fn dg_euler_2d_p3_finite() {
        let dg = make_dg(4, 3);
        let u = dg.project_initial(&|_,_| (1.0, 0.5, 0.0, 1.0));
        let du = dg.rhs(&u);
        for v in &du { assert!(v.is_finite()); }
    }

    #[test]
    fn dg_euler_2d_p123_finite_all_orders() {
        for order in 1..=3 {
            let dg = make_dg(4, order);
            let u = dg.project_initial(&|x,y| (1.0+0.1*(x+y).sin(), 0.7, 0.3, 1.0));
            let du = dg.rhs(&u);
            for v in &du { assert!(v.is_finite(), "order={}: non-finite RHS", order); }
        }
    }

    #[test]
    fn dg_euler_2d_p123_multi_element_order() {
        for (order, expected_dofs) in &[(1u8, 1536usize), (2, 3072), (3, 5120)] {
            let dg = make_dg(8, *order);
            assert_eq!(dg.n_dofs(), *expected_dofs);
            let u = dg.project_initial(&|x,y| (1.0+0.1*(x+y).sin(), 0.7, 0.3, 1.0));
            let du = dg.rhs(&u);
            for v in &du { assert!(v.is_finite(), "order={}: non-finite RHS", order); }
        }
    }

    #[test]
    fn dg_euler_2d_p123_step_rk3_finite() {
        for order in 1..=3 {
            let dg = make_dg(4, order);
            let mut u = dg.project_initial(&|x,y| (1.0+0.05*(x+y).sin(), 0.7, 0.3, 1.0));
            dg.step_rk3(&mut u, 0.001);
            for v in &u { assert!(v.is_finite(), "order={}: non-finite u", order); }
        }
    }

    #[test]
    fn p1_vs_p2_dof_count() {
        let dg1 = make_dg(6, 1);
        let dg2 = make_dg(6, 2);
        assert!(dg2.n_dofs() > dg1.n_dofs());
    }

    #[test]
    fn p1_uniform_flow_rhs_trace() {
        let dg = make_dg(2, 1); // 2×2 = 8 triangles
        let u0 = dg.project_initial(&|_,_| (1.0, 0.0, 0.0, 1.0));
        let du = dg.rhs(&u0);
        let norm: f64 = du.iter().map(|v| v*v).sum::<f64>().sqrt();
        eprintln!("  2×2: RHS norm = {:.6e}", norm);
        // Print non-zero entries
        for (i, &v) in du.iter().enumerate() {
            if v.abs() > 1e-12 {
                let dp = 3;
                let e = i / (dp * 4);
                let rem = i % (dp * 4);
                let ld = rem / 4;
                let c = rem % 4;
                eprintln!("    du[{}] = {:.6e} (elem={}, ldof={}, comp={})", i, v, e, ld, c);
            }
        }
        // Expect RHS to be ~0
        assert!(norm < 1e-12, "RHS norm for uniform flow should be ~0, got {:.6e}", norm);
    }

    #[test]
    fn p1_uniform_flow_preserved_over_many_steps() {
        let dg = make_dg(4, 1);
        let u0 = dg.project_initial(&|_,_| (1.0, 0.0, 0.0, 1.0));
        
        // Check if RHS is zero for uniform flow
        let du = dg.rhs(&u0);
        let norm: f64 = du.iter().map(|v| v*v).sum::<f64>().sqrt();
        eprintln!("  RHS norm for uniform flow = {:.6e}", norm);
        
        if norm > 1e-14 {
            // Find the max entry
            let mut max_v = 0.0f64;
            let mut max_i = 0usize;
            for (i, &v) in du.iter().enumerate() {
                if v.abs() > max_v.abs() {
                    max_v = v;
                    max_i = i;
                }
            }
            let dp = dg.dofs_per_elem();
            let e = max_i / (dp * 4);
            let ld = (max_i % (dp * 4)) / 4;
            let c = max_i % 4;
            eprintln!("  Max RHS entry: du[{}] = {:.6e} (elem={}, ldof={}, comp={})",
                     max_i, max_v, e, ld, c);
        }
        
        let mut u = u0.clone();
        let h = dg.h_min();
        let dt = 0.1 * h / 3.0;
        eprintln!("  h={:.6e}, dt={:.6e}", h, dt);
        
        for step in 0..100 {
            dg.step_rk3(&mut u, dt);
            if step % 20 == 0 {
                let du = dg.rhs(&u);
                let norm: f64 = du.iter().map(|v| v*v).sum::<f64>().sqrt();
                eprintln!("  step={}: RHS norm = {:.6e}, u[0] = {:.6e}", step, norm, u[0]);
            }
        }
        let err: f64 = (0..dg.n_dofs()).map(|i| (u[i] - u0[i]).powi(2)).sum::<f64>().sqrt();
        eprintln!("  Final error = {:.6e}", err);
        assert!(err < 1e-10, "uniform flow at rest should be steady, err={:.6e}", err);
    }
}
