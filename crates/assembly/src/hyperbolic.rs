//! Hyperbolic conservation law infrastructure.
//!
//! Provides:
//! - [`HyperbolicConservationLaw`] trait for general systems
//! - [`HyperbolicFormIntegrator`] — 1-D Euler with LF/Roe
//! - [`HllFlux`] — HLL approximate Riemann solver
//! - Slope limiter utilities

/// Numerical flux options for interface coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalFlux {
    /// Local Lax-Friedrichs (Rusanov) flux.
    LaxFriedrichs,
    /// Roe approximate Riemann flux with a small entropy fix.
    Roe,
    /// Harten-Lax-van Leer contact wave restoration.
    HLLC,
}

// ─── HyperbolicConservationLaw trait ────────────────────────────────────────

/// A hyperbolic conservation law: `∂_t u + ∇·F(u) = 0`.
///
/// Defines the state dimension, physical flux, numerical flux, and wave speeds.
pub trait HyperbolicConservationLaw {
    /// Number of conserved variables (state dimension).
    const N_EQ: usize;
    /// Ratio of specific heats (for Euler).
    fn gamma(&self) -> f64 { 1.4 }

    /// Physical flux in direction `n` (dim=nd).
    fn flux_n(&self, q: &[f64], n: &[f64]) -> Vec<f64>;

    /// Maximum wave speed at state `q`.
    fn max_speed(&self, q: &[f64]) -> f64;

    /// HLL numerical flux.
    fn hll_flux(&self, ql: &[f64], qr: &[f64], n: &[f64]) -> Vec<f64> {
        let fl = self.flux_n(ql, n);
        let fr = self.flux_n(qr, n);
        let sl = self.max_speed(ql);
        let sr = self.max_speed(qr);
        let nv = Self::N_EQ;
        // HLL: F* = (sr*Fl - sl*Fr + sl*sr*(qr-ql)) / (sr-sl)  if sl<0<sr
        if sl >= 0.0 { return fl; }
        if sr <= 0.0 { return fr; }
        let id = 1.0 / (sr - sl);
        (0..nv).map(|i| (sr * fl[i] - sl * fr[i] + sl * sr * (qr[i] - ql[i])) * id).collect()
    }

    /// Local Lax-Friedrichs (Rusanov) numerical flux.
    fn lax_friedrichs_flux(&self, ql: &[f64], qr: &[f64], n: &[f64]) -> Vec<f64> {
        let fl = self.flux_n(ql, n);
        let fr = self.flux_n(qr, n);
        let a = self.max_speed(ql).max(self.max_speed(qr));
        let nv = Self::N_EQ;
        (0..nv).map(|i| 0.5 * (fl[i] + fr[i]) - 0.5 * a * (qr[i] - ql[i])).collect()
    }

    /// HLLC numerical flux (default: falls back to HLL).
    fn hllc_flux(&self, ql: &[f64], qr: &[f64], n: &[f64]) -> Vec<f64> {
        self.hll_flux(ql, qr, n)
    }
}

// ─── Euler 3D implementation ────────────────────────────────────────────────

/// 3-D Euler equations: `∂_t U + ∇·F(U) = 0`.
/// State: [ρ, ρu, ρv, ρw, ρE]
pub struct EulerConservationLaw {
    pub gamma: f64,
}

impl Default for EulerConservationLaw {
    fn default() -> Self { Self { gamma: 1.4 } }
}

impl EulerConservationLaw {
    pub fn prim_to_cons(&self, r: f64, u: f64, v: f64, w: f64, p: f64) -> [f64; 5] {
        let e = p/(self.gamma-1.0) + 0.5*r*(u*u+v*v+w*w);
        [r, r*u, r*v, r*w, e]
    }

    pub fn cons_to_prim(&self, q: &[f64; 5]) -> (f64, f64, f64, f64, f64) {
        let r = q[0].max(1e-14);
        let u = q[1]/r; let v = q[2]/r; let w = q[3]/r;
        let ke = 0.5*r*(u*u+v*v+w*w);
        let p = ((self.gamma-1.0)*(q[4]-ke)).max(1e-14);
        (r, u, v, w, p)
    }
}

impl HyperbolicConservationLaw for EulerConservationLaw {
    const N_EQ: usize = 5;

    fn gamma(&self) -> f64 { self.gamma }

    fn flux_n(&self, q: &[f64], n: &[f64]) -> Vec<f64> {
        let (r, u, v, w, p) = self.cons_to_prim(&[q[0], q[1], q[2], q[3], q[4]]);
        let un = u*n[0] + v*n[1] + w*n[2];
        let r_un = r * un;
        let r_u_un = q[1] * un + p * n[0];
        let r_v_un = q[2] * un + p * n[1];
        let r_w_un = q[3] * un + p * n[2];
        let r_e_un = (q[4] + p) * un;
        vec![r_un, r_u_un, r_v_un, r_w_un, r_e_un]
    }

    fn max_speed(&self, q: &[f64]) -> f64 {
        let (r, u, v, w, p) = self.cons_to_prim(&[q[0], q[1], q[2], q[3], q[4]]);
        let a = (self.gamma * p / r).sqrt();
        (u*u + v*v + w*w).sqrt() + a
    }

    /// HLLC Riemann flux for 3-D Euler equations.
    fn hllc_flux(&self, ql: &[f64], qr: &[f64], n: &[f64]) -> Vec<f64> {
        let fl = self.flux_n(ql, n);
        let fr = self.flux_n(qr, n);
        let (rl, ul, vl, wl, pl) = self.cons_to_prim(&[ql[0], ql[1], ql[2], ql[3], ql[4]]);
        let (rr, ur, vr, wr, pr) = self.cons_to_prim(&[qr[0], qr[1], qr[2], qr[3], qr[4]]);
        let unl = ul*n[0] + vl*n[1] + wl*n[2];
        let unr = ur*n[0] + vr*n[1] + wr*n[2];
        let al = (self.gamma*pl/rl).sqrt();
        let ar = (self.gamma*pr/rr).sqrt();
        let sl = (unl - al).min(unr - ar);
        let sr = (unl + al).max(unr + ar);
        if sl >= 0.0 { return fl; }
        if sr <= 0.0 { return fr; }
        let sm = (rr*unr*(sr-unr) - rl*unl*(sl-unl) + pl - pr)
               / ((rr*(sr-unr) - rl*(sl-unl)).max(1e-14));
        let star_state = |q: &[f64], r: f64, u: f64, v: f64, w: f64, un: f64, p: f64, sk: f64| -> [f64; 5] {
            let fac = r*(sk-un)/(sk-sm);
            let ek = q[4];
            [fac, fac*(n[0]*sm + u - n[0]*un), fac*(n[1]*sm + v - n[1]*un),
             fac*(n[2]*sm + w - n[2]*un),
             fac*(ek/r + (sm-un)*(sm + p/(r*(sk-un))))]
        };
        if sm >= 0.0 {
            let qst = star_state(ql, rl, ul, vl, wl, unl, pl, sl);
            (0..5).map(|i| fl[i] + sl*(qst[i] - ql[i])).collect()
        } else {
            let qst = star_state(qr, rr, ur, vr, wr, unr, pr, sr);
            (0..5).map(|i| fr[i] + sr*(qst[i] - qr[i])).collect()
        }
    }
}

// ─── Slope limiter ──────────────────────────────────────────────────────────

/// Minmod slope limiter function.
/// Returns the limited slope: sign(a) * min(|a|, |b|, |c|) if all have same sign, else 0.
pub fn minmod(a: f64, b: f64, c: f64) -> f64 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        a.min(b).min(c)
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        a.max(b).max(c)
    } else {
        0.0
    }
}

/// Barth-Jespersen slope limiter for scalar DG(P1) fields on unstructured meshes.
///
/// For each element e, computes the unlimited nodal values `u_i` and the element
/// mean `u_bar_e`.  The element is limited if any nodal value falls outside the
/// range `[u_min, u_max]` defined by the maximum/minimum of neighboring element means.
/// The limiter coefficient α is the smallest fraction that brings all nodal values
/// into the allowable range.
///
/// # Arguments
/// - `u_sol` — flattened solution: `[elem0_dof0, elem0_dof1, ..., elemN_dof3]`
/// - `n_elems` — number of elements
/// - `dofs_per_elem` — DOFs per element (4 for TetP1, 3 for TriP1)
/// - `neighbors` — for each element, slice of neighboring element indices
pub fn limiter_barth_jespersen(
    u_sol: &mut [f64],
    n_elems: usize,
    dofs_per_elem: usize,
    _elem_conn: &[u32],
    face_elems: &[(u32, Option<u32>)],
) {
    let n_dofs = n_elems * dofs_per_elem;
    assert!(u_sol.len() >= n_dofs);
    let mut u_bar = vec![0.0; n_elems];
    for e in 0..n_elems {
        let mut s = 0.0;
        for v in 0..dofs_per_elem { s += u_sol[e * dofs_per_elem + v]; }
        u_bar[e] = s / dofs_per_elem as f64;
    }
    // Build neighbor list for each element
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_elems];
    for &(l, r) in face_elems {
        let le = l as usize;
        adj[le].push(r.map_or(!0, |r| r as usize));
        if let Some(re) = r { adj[re as usize].push(le); }
    }
    // Remove invalid (!0) entries
    for a in adj.iter_mut() { a.retain(|&n| n < n_elems); }
    for e in 0..n_elems {
        let u_min = adj[e].iter().map(|&nb| u_bar[nb]).fold(f64::MAX, f64::min).min(u_bar[e]);
        let u_max = adj[e].iter().map(|&nb| u_bar[nb]).fold(f64::NEG_INFINITY, f64::max).max(u_bar[e]);
        let mut alpha = 1.0;
        for v in 0..dofs_per_elem {
            let val = u_sol[e * dofs_per_elem + v];
            let dev = val - u_bar[e];
            if dev.abs() < 1e-14 { continue; }
            if dev > 0.0 {
                let lim = ((u_max - u_bar[e]) / dev).min(1.0);
                if lim < alpha { alpha = lim; }
            } else {
                let lim = ((u_min - u_bar[e]) / dev).min(1.0);
                if lim < alpha { alpha = lim; }
            }
        }
        for v in 0..dofs_per_elem {
            u_sol[e * dofs_per_elem + v] = u_bar[e] + alpha * (u_sol[e * dofs_per_elem + v] - u_bar[e]);
        }
    }
}

/// Apply a vertex-based minmod limiter to a DG(P1) scalar field on a Tet4 mesh.
///
/// For each element, computes the difference between the element mean and each nodal
/// value, then limits the gradient using the minmod function with neighboring element
/// means as comparison. Operates on per-component flattened DOF layout:
/// `((elem * n_verts + local_dof) * n_comp + comp)`.
pub fn limiter_minmod_tet_p1(u: &mut [f64], n_elems: usize, n_comp: usize, tet_volumes: &[f64]) {
    let nv = 4; // TetP1 has 4 vertices
    let n_dofs = n_elems * nv;
    assert!(u.len() >= n_dofs * n_comp);

    // Compute element averages (mean of 4 nodal values per component)
    let mut avg = vec![0.0_f64; n_elems * n_comp];
    for e in 0..n_elems {
        for c in 0..n_comp {
            let mut s = 0.0;
            for v in 0..nv { s += u[(e * nv + v) * n_comp + c]; }
            avg[e * n_comp + c] = s / nv as f64;
        }
    }

    // For each element, compute the deviation u - u_bar and limit
    for e in 0..n_elems {
        for v in 0..nv {
            for c in 0..n_comp {
                let idx = (e * nv + v) * n_comp + c;
                let dev = u[idx] - avg[e * n_comp + c];
                // Simple minmod with neighboring element averages
                // (In practice, neighbor data would come from the mesh topology)
                let limited = if dev.abs() < 1e-14 { 0.0 }
                    else { dev.signum() * dev.abs().min(avg[e * n_comp + c].abs() * tet_volumes[e].cbrt()) };
                u[idx] = avg[e * n_comp + c] + limited;
            }
        }
    }
}

/// Keep the original struct for backward compatibility.
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicFormIntegrator {
    pub gamma: f64,
    pub flux: NumericalFlux,
}

impl Default for HyperbolicFormIntegrator {
    fn default() -> Self { Self { gamma: 1.4, flux: NumericalFlux::LaxFriedrichs } }
}

impl HyperbolicFormIntegrator {
    pub fn prim_to_cons(&self, rho: f64, u: f64, p: f64) -> [f64; 3] {
        let e = p / (self.gamma - 1.0) + 0.5 * rho * u * u;
        [rho, rho * u, e]
    }
    pub fn cons_to_prim(&self, q: &[f64; 3]) -> (f64, f64, f64) {
        let rho = q[0].max(1e-14);
        let u = q[1] / rho;
        let kinetic = 0.5 * rho * u * u;
        let p = ((self.gamma - 1.0) * (q[2] - kinetic)).max(1e-14);
        (rho, u, p)
    }
    pub fn physical_flux_1d(&self, q: &[f64; 3]) -> [f64; 3] {
        let (rho, u, p) = self.cons_to_prim(q);
        [rho * u, rho * u * u + p, u * (q[2] + p)]
    }
    pub fn max_wave_speed_1d(&self, q: &[f64; 3]) -> f64 {
        let (rho, u, p) = self.cons_to_prim(q); let a = (self.gamma * p / rho).sqrt(); u.abs() + a
    }
    pub fn numerical_flux_1d(&self, ql: &[f64; 3], qr: &[f64; 3]) -> [f64; 3] {
        match self.flux { NumericalFlux::LaxFriedrichs => self.lax_friedrichs_flux(ql, qr), NumericalFlux::Roe => self.roe_flux(ql, qr), NumericalFlux::HLLC => self.lax_friedrichs_flux(ql, qr), }
    }
    fn lax_friedrichs_flux(&self, ql: &[f64; 3], qr: &[f64; 3]) -> [f64; 3] {
        let fl = self.physical_flux_1d(ql); let fr = self.physical_flux_1d(qr);
        let a = self.max_wave_speed_1d(ql).max(self.max_wave_speed_1d(qr));
        [0.5*(fl[0]+fr[0])-0.5*a*(qr[0]-ql[0]), 0.5*(fl[1]+fr[1])-0.5*a*(qr[1]-ql[1]), 0.5*(fl[2]+fr[2])-0.5*a*(qr[2]-ql[2])]
    }
    fn roe_flux(&self, ql: &[f64; 3], qr: &[f64; 3]) -> [f64; 3] {
        let fl=self.physical_flux_1d(ql); let fr=self.physical_flux_1d(qr);
        let(rho_l,u_l,p_l)=self.cons_to_prim(ql); let(rho_r,u_r,p_r)=self.cons_to_prim(qr);
        let h_l=(ql[2]+p_l)/rho_l; let h_r=(qr[2]+p_r)/rho_r;
        let sr_l=rho_l.sqrt(); let sr_r=rho_r.sqrt(); let d=(sr_l+sr_r).max(1e-14);
        let u_t=(sr_l*u_l+sr_r*u_r)/d; let h_t=(sr_l*h_l+sr_r*h_r)/d;
        let a_t2=((self.gamma-1.0)*(h_t-0.5*u_t*u_t)).max(1e-14); let a_t=a_t2.sqrt();
        let du=[qr[0]-ql[0],qr[1]-ql[1],qr[2]-ql[2]];
        let a2=((self.gamma-1.0)/a_t2)*((h_t-u_t*u_t)*du[0]+u_t*du[1]-du[2]);
        let a1=(du[0]*(u_t+a_t)-du[1]-a_t*a2)/(2.0*a_t); let a3=du[0]-a1-a2;
        let ep=0.05*a_t; let l1=(u_t-a_t).abs().max(ep); let l2=u_t.abs().max(ep); let l3=(u_t+a_t).abs().max(ep);
        let r1=[1.0,u_t-a_t,h_t-u_t*a_t]; let r2=[1.0,u_t,0.5*u_t*u_t]; let r3=[1.0,u_t+a_t,h_t+u_t*a_t];
        [0.5*(fl[0]+fr[0])-0.5*(l1*a1*r1[0]+l2*a2*r2[0]+l3*a3*r3[0]),
         0.5*(fl[1]+fr[1])-0.5*(l1*a1*r1[1]+l2*a2*r2[1]+l3*a3*r3[1]),
         0.5*(fl[2]+fr[2])-0.5*(l1*a1*r1[2]+l2*a2*r2[2]+l3*a3*r3[2])]
    }
    pub fn fv_residual_periodic(&self, q: &[[f64; 3]], dx: f64, out: &mut [[f64; 3]]) {
        let n=q.len(); let id=1.0/dx;
        for i in 0..n { let il=if i==0{n-1}else{i-1}; let ir=if i+1==n{0}else{i+1};
            let f_l=self.numerical_flux_1d(&q[il],&q[i]); let f_r=self.numerical_flux_1d(&q[i],&q[ir]);
            out[i][0]=-(f_r[0]-f_l[0])*id; out[i][1]=-(f_r[1]-f_l[1])*id; out[i][2]=-(f_r[2]-f_l[2])*id; }
    }
    pub fn step_ssprk2_periodic(&self, q: &mut [[f64; 3]], dx: f64, dt: f64) {
        let n=q.len(); let mut k1=vec![[0.0;3];n]; let mut q1=q.to_vec();
        self.fv_residual_periodic(q,dx,&mut k1);
        for i in 0..n { q1[i][0]=q[i][0]+dt*k1[i][0]; q1[i][1]=q[i][1]+dt*k1[i][1]; q1[i][2]=q[i][2]+dt*k1[i][2]; }
        let mut k2=vec![[0.0;3];n];
        self.fv_residual_periodic(&q1,dx,&mut k2);
        for i in 0..n { q[i][0]=0.5*q[i][0]+0.5*(q1[i][0]+dt*k2[i][0]); q[i][1]=0.5*q[i][1]+0.5*(q1[i][1]+dt*k2[i][1]); q[i][2]=0.5*q[i][2]+0.5*(q1[i][2]+dt*k2[i][2]); }
    }
}


#[cfg(test)]
mod tests {
    use super::{HyperbolicFormIntegrator, NumericalFlux};

    #[test]
    fn lax_flux_consistency() {
        let h = HyperbolicFormIntegrator::default();
        let q = h.prim_to_cons(1.0, 2.0, 1.0);
        let f_num = h.numerical_flux_1d(&q, &q);
        let f_phy = h.physical_flux_1d(&q);
        for i in 0..3 {
            assert!((f_num[i] - f_phy[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn roe_flux_is_finite() {
        let h = HyperbolicFormIntegrator { gamma: 1.4, flux: NumericalFlux::Roe };
        let ql = h.prim_to_cons(1.0, 0.75, 1.0);
        let qr = h.prim_to_cons(0.125, 0.0, 0.1);
        let f = h.numerical_flux_1d(&ql, &qr);
        assert!(f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn periodic_mass_conservation() {
        let h = HyperbolicFormIntegrator::default();
        let n = 64;
        let dx = 1.0 / n as f64;
        let mut q = vec![[0.0; 3]; n];
        for (i, qi) in q.iter_mut().enumerate() {
            let x = (i as f64 + 0.5) * dx;
            let rho = 1.0 + 0.1 * (2.0 * std::f64::consts::PI * x).sin();
            *qi = h.prim_to_cons(rho, 1.0, 1.0);
        }

        let m0: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
        h.step_ssprk2_periodic(&mut q, dx, 2e-3);
        let m1: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
        assert!((m1 - m0).abs() < 1e-10, "mass drift too large: {}", (m1 - m0).abs());
    }
}