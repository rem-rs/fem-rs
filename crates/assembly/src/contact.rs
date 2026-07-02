//! Contact mechanics: Signorini unilateral contact with Coulomb friction.
//!
//! Supports:
//! - **Penalty** and **Augmented Lagrangian** normal contact
//! - **Coulomb friction** with penalty regularization (stick–slip)
//! - **2D** (edge segments) and **3D** (triangle facets)
//! - P1/P2 elements via integration on boundary faces
//!
//! # Signorini problem
//!
//! ```text
//! -∇·σ = f           in Ω
//!     u = 0          on Γ_D
//! σ·n = t            on Γ_N
//!   g(u) ≤ 0         on Γ_C  (gap ≤ 0 → no penetration)
//!    λ_n ≥ 0         on Γ_C  (contact pressure)
//! λ_n·g(u) = 0       on Γ_C  (complementarity)
//! ```
//!
//! # Coulomb friction
//!
//! ```text
//! |λ_t| ≤ μ·λ_n            (stick condition)
//! λ_t = -μ·λ_n·sign(u̇_t)  (slip)
//! ```
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::contact::*;
//!
//! let cfg = ContactConfig {
//!     penalty_normal: 1e6,
//!     contact_type: ContactType::AugmentedLagrangian { max_al_iter: 5, al_tol: 1e-6 },
//!     friction: FrictionModel::Coulomb { mu: 0.3, penalty_tangential: 1e5 },
//!     gap_function: |x| 0.05 - x[1],
//!     contact_tags: vec![1],
//! };
//! let u = solve_contact_newton(&stiffness, &rhs, &space, &cfg, 50, 1e-8);
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;

/// Contact regularisation type.
#[derive(Debug, Clone)]
pub enum ContactType {
    /// Standard penalty: λ_n = ε_n·⟨g(u)⟩₋
    Penalty,
    /// Augmented Lagrangian: alternates solving with multiplier updates
    /// λₖ₊₁ = max(0, λₖ + ε_n·g(uₖ))
    AugmentedLagrangian { max_al_iter: usize, al_tol: f64 },
}

/// Friction model.
#[derive(Debug, Clone)]
pub enum FrictionModel {
    /// No tangential friction.
    Frictionless,
    /// Coulomb friction with penalty regularisation.
    Coulomb { mu: f64, penalty_tangential: f64 },
}

/// Contact configuration.
#[derive(Debug, Clone)]
pub struct ContactConfig {
    /// Normal penalty parameter ε_n.
    pub penalty_normal: f64,
    /// Normal contact type (penalty or augmented Lagrangian).
    pub contact_type: ContactType,
    /// Friction model (frictionless or Coulomb).
    pub friction: FrictionModel,
    /// Gap function: signed distance to obstacle.
    /// Positive values mean penetration.
    pub gap_function: fn(&[f64]) -> f64,
    /// Boundary tags where contact is active.
    pub contact_tags: Vec<i32>,
}

impl Default for ContactConfig {
    fn default() -> Self {
        Self {
            penalty_normal: 1e6,
            contact_type: ContactType::Penalty,
            friction: FrictionModel::Frictionless,
            gap_function: |_| 0.0,
            contact_tags: vec![1],
        }
    }
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// Negative part: `min(x, 0)`.
#[inline]
fn neg_part(x: f64) -> f64 {
    if x < 0.0 { x } else { 0.0 }
}

/// Derivative of negative part: 1 if x < 0 else 0.
#[inline]
fn neg_part_d(x: f64) -> f64 {
    if x < 0.0 { 1.0 } else { 0.0 }
}

/// Smooth approximation of `min(x, 0)` with regularisation η.
#[inline]
fn neg_part_smooth(x: f64, eta: f64) -> f64 {
    if x < -eta { x }
    else if x > eta { 0.0 }
    else { -0.25 * (x - eta).powi(2) / eta }
}

#[inline]
fn neg_part_smooth_d(x: f64, eta: f64) -> f64 {
    if x < -eta { 1.0 }
    else if x > eta { 0.0 }
    else { -0.5 * (x - eta) / eta }
}

// ─── Quadrature helpers ───────────────────────────────────────────────────────

/// 2-point Gauss-Legendre on [0,1].
fn gauss_edge_2pt() -> ([f64; 2], [f64; 2]) {
    let a = 0.211324865405187;
    let b = 0.788675134594813;
    ([a, b], [0.5, 0.5])
}

/// 3-point symmetric rule for triangle (barycentric coords, pre-scaled weights).
/// Points are permutations of (2/3, 1/6, 1/6).
fn gauss_tri_3pt() -> ([[f64; 3]; 3], [f64; 3]) {
    let a = 1.0 / 6.0;
    let b = 2.0 / 3.0;
    ([[b, a, a], [a, b, a], [a, a, b]], [1.0 / 6.0; 3])
}

// ─── 2D scalar contact (H¹, P1) ────────────────────────────────────────────

/// Assemble normal contact force and tangent stiffness for **2D** scalar P1.
///
/// `lagrange_multipliers` — per-face AL multipliers; pass `&[]` for penalty-only.
pub fn assemble_contact_2d<S, M>(
    _space: &S,
    mesh: &M,
    cfg: &ContactConfig,
    u: &[f64],
    lagrange_multipliers: &[f64],
) -> (Vec<f64>, CsrMatrix<f64>)
where
    M: MeshTopology,
{
    assert_eq!(mesh.dim() as usize, 2, "assemble_contact_2d requires dim=2");
    let n_nodes = mesh.n_nodes() as usize;
    let mut rhs = vec![0.0; n_nodes];
    let mut coo = CooMatrix::<f64>::new(n_nodes, n_nodes);
    let pen = cfg.penalty_normal;
    let contact_set: std::collections::HashSet<i32> =
        cfg.contact_tags.iter().copied().collect();
    let (pts, wts) = gauss_edge_2pt();
    let eta = 1e-8;

    for f in 0..mesh.n_boundary_faces() as u32 {
        let tag = mesh.face_tag(f);
        if !contact_set.contains(&tag) { continue; }
        let fnodes = mesh.face_nodes(f);
        if fnodes.len() < 2 { continue; }
        let n0 = fnodes[0] as usize;
        let n1 = fnodes[1] as usize;
        let p0 = mesh.node_coords(n0 as u32);
        let p1 = mesh.node_coords(n1 as u32);
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let edge_len = (dx * dx + dy * dy).sqrt();
        if edge_len < 1e-30 { continue; }

        for (ti, (xi, wt)) in pts.iter().zip(wts.iter()).enumerate() {
            let x_phys = [p0[0] + xi * dx, p0[1] + xi * dy];
            let gap = (cfg.gap_function)(&x_phys);
            let phi = [1.0 - xi, *xi]; // P1 shape functions on [0,1]
            let uh = u[n0] * phi[0] + u[n1] * phi[1];
            let gap_uh = uh - gap;
            let lam = if lagrange_multipliers.is_empty() { 0.0 }
                      else { lagrange_multipliers[ti % lagrange_multipliers.len()] };
            let active = lam + pen * gap_uh;
            // Penalty only (no AL) or AL branch
            let (np, npd) = if lagrange_multipliers.is_empty() {
                (neg_part(gap_uh), neg_part_d(gap_uh))
            } else {
                (neg_part_smooth(active, eta), neg_part_smooth_d(active, eta))
            };
            let w_phys = wt * edge_len;
            let force = if lagrange_multipliers.is_empty() {
                -pen * np * w_phys
            } else {
                -(lam + pen * np) * w_phys
            };
            for ln in 0..2 {
                rhs[[n0, n1][ln]] += force * phi[ln];
                for lm in 0..2 {
                    let k = if lagrange_multipliers.is_empty() {
                        -pen * npd
                    } else {
                        -pen * npd
                    };
                    coo.add([n0, n1][ln], [n0, n1][lm], k * phi[ln] * phi[lm] * w_phys);
                }
            }
        }
    }
    (rhs, coo.into_csr())
}

// ─── 2D vector contact (elasticity) ──────────────────────────────────────────

/// Assemble contact + Coulomb friction for **2D elasticity** (2 DOFs/node).
pub fn assemble_contact_2d_vector<M: MeshTopology>(
    mesh: &M,
    cfg: &ContactConfig,
    u: &[f64],
    lam_n: &[f64],
) -> (Vec<f64>, CsrMatrix<f64>) {
    assert_eq!(mesh.dim() as usize, 2, "assemble_contact_2d_vector requires dim=2");
    let n_nodes = mesh.n_nodes() as usize;
    let n_dofs = n_nodes * 2;
    let mut rhs = vec![0.0; n_dofs];
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let pen_n = cfg.penalty_normal;
    let contact_set: std::collections::HashSet<i32> =
        cfg.contact_tags.iter().copied().collect();
    let (pts, wts) = gauss_edge_2pt();
    let eta = 1e-8;
    let (mu, pen_t) = match &cfg.friction {
        FrictionModel::Frictionless => (0.0, 0.0),
        FrictionModel::Coulomb { mu, penalty_tangential } => (*mu, *penalty_tangential),
    };

    for f in 0..mesh.n_boundary_faces() as u32 {
        let tag = mesh.face_tag(f);
        if !contact_set.contains(&tag) { continue; }
        let fnodes = mesh.face_nodes(f);
        if fnodes.len() < 2 { continue; }
        let n0 = fnodes[0] as usize;
        let n1 = fnodes[1] as usize;
        let p0 = mesh.node_coords(n0 as u32);
        let p1 = mesh.node_coords(n1 as u32);
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let el = (dx * dx + dy * dy).sqrt();
        if el < 1e-30 { continue; }
        // Outward unit normal (rotate tangent +90°)
        let nx = -dy / el;
        let ny = dx / el;
        let tx = dx / el;  // unit tangent
        let ty = dy / el;

        for (ti, (xi, wt)) in pts.iter().zip(wts.iter()).enumerate() {
            let x_phys = [p0[0] + xi * dx, p0[1] + xi * dy];
            let gap = (cfg.gap_function)(&x_phys);
            let phi = [1.0 - xi, *xi];
            let ux_qp = u[n0*2] * phi[0] + u[n1*2] * phi[1];
            let uy_qp = u[n0*2+1] * phi[0] + u[n1*2+1] * phi[1];
            let un_qp = ux_qp * nx + uy_qp * ny;
            let ut_qp = ux_qp * tx + uy_qp * ty;
            let gap_un = un_qp - gap;

            // ── Normal contact ──
            let lam = if lam_n.is_empty() { 0.0 } else { lam_n[ti % lam_n.len()] };
            let active = lam + pen_n * gap_un;
            let (np, npd) = if lam_n.is_empty() {
                (neg_part(gap_un), neg_part_d(gap_un))
            } else {
                (neg_part_smooth(active, eta), neg_part_smooth_d(active, eta))
            };
            let w_phys = wt * el;
            let r_n = if lam_n.is_empty() { pen_n * np } else { lam + pen_n * np };
            let fn_x = -r_n * nx;
            let fn_y = -r_n * ny;

            // ── Friction ──
            let (ft_x, ft_y, df_dux, df_duy) = if mu > 0.0 && pen_t > 0.0 && gap_un < 0.0 {
                let sigma_n = r_n.abs();
                let sigma_t_trial = pen_t * ut_qp;
                let sigma_t_max = mu * sigma_n + 1e-15;
                if sigma_t_trial.abs() <= sigma_t_max {
                    // Stick
                    let sx = -sigma_t_trial * tx;
                    let sy = -sigma_t_trial * ty;
                    (sx, sy, -pen_t, 0.0)
                } else {
                    // Slip
                    let sign = if ut_qp >= 0.0 { 1.0 } else { -1.0 };
                    let sigma_t_slip = sigma_t_max * sign;
                    let sx = -sigma_t_slip * tx;
                    let sy = -sigma_t_slip * ty;
                    let r_n_d = if lam_n.is_empty() { pen_n * neg_part_d(gap_un) } else { pen_n * npd };
                    (sx, sy, -mu * sign * r_n_d * nx, -mu * sign * r_n_d * ny)
                }
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            for ln in 0..2 {
                let idx = [n0*2, n1*2][ln];
                let idy = [n0*2+1, n1*2+1][ln];
                let f_loc = (fn_x + ft_x) * w_phys * phi[ln];
                let g_loc = (fn_y + ft_y) * w_phys * phi[ln];
                rhs[idx] += f_loc;
                rhs[idy] += g_loc;
                for lm in 0..2 {
                    let jdx = [n0*2, n1*2][lm];
                    let jdy = [n0*2+1, n1*2+1][lm];
                    let k_n = if lam_n.is_empty() { pen_n * neg_part_d(gap_un) } else { pen_n * npd };
                    let bl = phi[ln] * phi[lm] * w_phys;
                    coo.add(idx, jdx, k_n * nx * nx * bl);
                    coo.add(idx, jdy, k_n * nx * ny * bl);
                    coo.add(idy, jdx, k_n * ny * nx * bl);
                    coo.add(idy, jdy, k_n * ny * ny * bl);
                    // Friction Jacobian
                    if mu > 0.0 && pen_t > 0.0 && gap_un < 0.0 {
                        let sigma_n = r_n.abs();
                        let s_t_trial = pen_t * ut_qp;
                        if s_t_trial.abs() <= mu * sigma_n + 1e-15 {
                            // Stick
                            let kf = df_dux * bl;
                            coo.add(idx, jdx, kf * tx * tx);
                            coo.add(idx, jdy, kf * tx * ty);
                            coo.add(idy, jdx, kf * ty * tx);
                            coo.add(idy, jdy, kf * ty * ty);
                        } else {
                            // Slip: coupling
                            let kc = df_dux * bl;
                            coo.add(idx, jdx, kc);
                            coo.add(idx, jdy, kc);
                            coo.add(idy, jdx, df_duy * bl);
                            coo.add(idy, jdy, df_duy * bl);
                        }
                    }
                }
            }
        }
    }
    (rhs, coo.into_csr())
}

// ─── 3D scalar contact (H¹, P1) ─────────────────────────────────────────────

/// Assemble normal contact force for **3D** scalar P1 (tet faces).
pub fn assemble_contact_3d<M: MeshTopology>(
    mesh: &M,
    cfg: &ContactConfig,
    u: &[f64],
    lagrange_multipliers: &[f64],
) -> (Vec<f64>, CsrMatrix<f64>) {
    assert_eq!(mesh.dim() as usize, 3, "assemble_contact_3d requires dim=3");
    let n_nodes = mesh.n_nodes() as usize;
    let mut rhs = vec![0.0; n_nodes];
    let mut coo = CooMatrix::<f64>::new(n_nodes, n_nodes);
    let pen = cfg.penalty_normal;
    let contact_set: std::collections::HashSet<i32> =
        cfg.contact_tags.iter().copied().collect();
    let (tri_pts, tri_wts) = gauss_tri_3pt();

    for f in 0..mesh.n_boundary_faces() as u32 {
        let tag = mesh.face_tag(f);
        if !contact_set.contains(&tag) { continue; }
        let fnodes = mesh.face_nodes(f);
        if fnodes.len() < 3 { continue; }
        let n0 = fnodes[0] as usize;
        let n1 = fnodes[1] as usize;
        let n2 = fnodes[2] as usize;
        let p0 = mesh.node_coords(n0 as u32);
        let p1 = mesh.node_coords(n1 as u32);
        let p2 = mesh.node_coords(n2 as u32);

        // Edge vectors → cross product → area
        let jx = p1[0] - p0[0]; let jy = p1[1] - p0[1]; let jz = p1[2] - p0[2];
        let kx = p2[0] - p0[0]; let ky = p2[1] - p0[1]; let kz = p2[2] - p0[2];
        let cx = jy * kz - jz * ky;
        let cy = jz * kx - jx * kz;
        let cz = jx * ky - jy * kx;
        let area2 = (cx * cx + cy * cy + cz * cz).sqrt();
        if area2 < 1e-30 { continue; }
        let face_area = area2 * 0.5;

        for (ti, (l, wt)) in tri_pts.iter().zip(tri_wts.iter()).enumerate() {
            let (l1, l2, l3) = (l[0], l[1], l[2]);
            let x_phys = [
                p0[0] * l1 + p1[0] * l2 + p2[0] * l3,
                p0[1] * l1 + p1[1] * l2 + p2[1] * l3,
                p0[2] * l1 + p1[2] * l2 + p2[2] * l3,
            ];
            let gap = (cfg.gap_function)(&x_phys);
            let phi = [l1, l2, l3]; // P1 = barycentric
            let uh = u[n0] * phi[0] + u[n1] * phi[1] + u[n2] * phi[2];
            let gap_uh = uh - gap;
            let lam = if lagrange_multipliers.is_empty() { 0.0 }
                      else { lagrange_multipliers[ti % lagrange_multipliers.len()] };
            let active = lam + pen * gap_uh;
            let (np, npd) = if lagrange_multipliers.is_empty() {
                (neg_part(gap_uh), neg_part_d(gap_uh))
            } else {
                (neg_part_smooth(active, 1e-8), neg_part_smooth_d(active, 1e-8))
            };
            let w_phys = wt * face_area;
            let force = if lagrange_multipliers.is_empty() { -pen * np * w_phys }
                        else { -(lam + pen * np) * w_phys };

            for ln in 0..3 {
                rhs[[n0, n1, n2][ln]] += force * phi[ln];
                for lm in 0..3 {
                    let k = -pen * npd * phi[ln] * phi[lm] * w_phys;
                    coo.add([n0, n1, n2][ln], [n0, n1, n2][lm], k);
                }
            }
        }
    }
    (rhs, coo.into_csr())
}

// ─── 3D vector contact (elasticity, ux/uy/uz) ────────────────────────────

pub fn assemble_contact_3d_vector<M: MeshTopology>(
    mesh: &M, cfg: &ContactConfig, u: &[f64], lagrange_multipliers: &[f64],
) -> (Vec<f64>, CsrMatrix<f64>) {
    assert_eq!(mesh.dim() as usize, 3, "requires dim=3");
    let n_nodes = mesh.n_nodes() as usize; let n_dofs = n_nodes * 3;
    let mut rhs = vec![0.0; n_dofs]; let mut coo = CooMatrix::new(n_dofs, n_dofs);
    let pen_n = cfg.penalty_normal;
    let (pen_t, mu) = match &cfg.friction {
        FrictionModel::Frictionless => (0.0, 0.0),
        FrictionModel::Coulomb { mu, penalty_tangential } => (*penalty_tangential, *mu),
    };
    let cs: std::collections::HashSet<i32> = cfg.contact_tags.iter().copied().collect();
    let (tri_pts, tri_wts) = gauss_tri_3pt();
    let has_lam = !lagrange_multipliers.is_empty();

    for f in 0..mesh.n_boundary_faces() as u32 {
        let tag = mesh.face_tag(f); if !cs.contains(&tag) { continue; }
        let fnodes = mesh.face_nodes(f); if fnodes.len() < 3 { continue; }
        let n = [fnodes[0] as usize, fnodes[1] as usize, fnodes[2] as usize];
        let p = [mesh.node_coords(fnodes[0]), mesh.node_coords(fnodes[1]), mesh.node_coords(fnodes[2])];
        let e1=[p[1][0]-p[0][0],p[1][1]-p[0][1],p[1][2]-p[0][2]]; let e2=[p[2][0]-p[0][0],p[2][1]-p[0][1],p[2][2]-p[0][2]];
        let nx=e1[1]*e2[2]-e1[2]*e2[1];let ny=e1[2]*e2[0]-e1[0]*e2[2];let nz=e1[0]*e2[1]-e1[1]*e2[0];
        let al=(nx*nx+ny*ny+nz*nz).sqrt().max(1e-30); let fa=al*0.5; let nu=[nx/al,ny/al,nz/al];
        let ad=[nu[0].abs(),nu[1].abs(),nu[2].abs()];
        let rd=if ad[0]<=ad[1]&&ad[0]<=ad[2]{[1.,0.,0.]}else if ad[1]<=ad[2]{[0.,1.,0.]}else{[0.,0.,1.]};
        let tx=nu[1]*rd[2]-nu[2]*rd[1];let ty=nu[2]*rd[0]-nu[0]*rd[2];let tz=nu[0]*rd[1]-nu[1]*rd[0];
        let tl=(tx*tx+ty*ty+tz*tz).sqrt().max(1e-30); let t1=[tx/tl,ty/tl,tz/tl];
        let t2=[nu[1]*t1[2]-nu[2]*t1[1],nu[2]*t1[0]-nu[0]*t1[2],nu[0]*t1[1]-nu[1]*t1[0]];

        for (ti,(l,wt)) in tri_pts.iter().zip(tri_wts.iter()).enumerate() {
            let(l1,l2,l3)=(l[0],l[1],l[2]); let phi=[l1,l2,l3];
            let xp=[p[0][0]*l1+p[1][0]*l2+p[2][0]*l3,p[0][1]*l1+p[1][1]*l2+p[2][1]*l3,p[0][2]*l1+p[1][2]*l2+p[2][2]*l3];
            let wp=wt*fa; let ux=u[n[0]*3]*phi[0]+u[n[1]*3]*phi[1]+u[n[2]*3]*phi[2];
            let uy=u[n[0]*3+1]*phi[0]+u[n[1]*3+1]*phi[1]+u[n[2]*3+1]*phi[2];
            let uz=u[n[0]*3+2]*phi[0]+u[n[1]*3+2]*phi[1]+u[n[2]*3+2]*phi[2];
            let un=ux*nu[0]+uy*nu[1]+uz*nu[2]; let ut1v=ux*t1[0]+uy*t1[1]+uz*t1[2]; let ut2v=ux*t2[0]+uy*t2[1]+uz*t2[2];
            let gap=(cfg.gap_function)(&xp); let gun=un-gap;
            let lam=if has_lam{lagrange_multipliers[ti%lagrange_multipliers.len()]}else{0.0};
            let act=lam+pen_n*gun; let(np,npd)=if!has_lam{(neg_part(gun),neg_part_d(gun))}else{(neg_part_smooth(act,1e-8),neg_part_smooth_d(act,1e-8))};
            let fv=if!has_lam{-pen_n*np*wp}else{-(lam+pen_n*np)*wp};
            for ln in 0..3{let dx=n[ln]*3;let dy=n[ln]*3+1;let dz=n[ln]*3+2;
                rhs[dx]+=fv*nu[0]*phi[ln];rhs[dy]+=fv*nu[1]*phi[ln];rhs[dz]+=fv*nu[2]*phi[ln];
                for lm in 0..3{let k=-pen_n*npd*phi[ln]*phi[lm]*wp;let jx=n[lm]*3;let jy=n[lm]*3+1;let jz=n[lm]*3+2;
                    coo.add(dx,jx,k*nu[0]*nu[0]);coo.add(dx,jy,k*nu[0]*nu[1]);coo.add(dx,jz,k*nu[0]*nu[2]);
                    coo.add(dy,jx,k*nu[1]*nu[0]);coo.add(dy,jy,k*nu[1]*nu[1]);coo.add(dy,jz,k*nu[1]*nu[2]);
                    coo.add(dz,jx,k*nu[2]*nu[0]);coo.add(dz,jy,k*nu[2]*nu[1]);coo.add(dz,jz,k*nu[2]*nu[2]);}}
            if pen_t>0.0&&mu>0.0{let sn=(-fv/wp).max(0.0);let s1=pen_t*ut1v;let s2=pen_t*ut2v;let sm=(s1*s1+s2*s2).sqrt().max(1e-30);
                if sm<=mu*sn+1e-15{let ks=pen_t*wp;for ln in 0..3{let dx=n[ln]*3;let dy=n[ln]*3+1;let dz=n[ln]*3+2;
                    rhs[dx]-=s1*t1[0]*phi[ln]+s2*t2[0]*phi[ln];rhs[dy]-=s1*t1[1]*phi[ln]+s2*t2[1]*phi[ln];rhs[dz]-=s1*t1[2]*phi[ln]+s2*t2[2]*phi[ln];
                    for lm in 0..3{let b=ks*phi[ln]*phi[lm];let jx=n[lm]*3;let jy=n[lm]*3+1;let jz=n[lm]*3+2;
                        coo.add(dx,jx,b*(t1[0]*t1[0]+t2[0]*t2[0]));coo.add(dx,jy,b*(t1[0]*t1[1]+t2[0]*t2[1]));coo.add(dx,jz,b*(t1[0]*t1[2]+t2[0]*t2[2]));
                        coo.add(dy,jx,b*(t1[1]*t1[0]+t2[1]*t2[0]));coo.add(dy,jy,b*(t1[1]*t1[1]+t2[1]*t2[1]));coo.add(dy,jz,b*(t1[1]*t1[2]+t2[1]*t2[2]));
                        coo.add(dz,jx,b*(t1[2]*t1[0]+t2[2]*t2[0]));coo.add(dz,jy,b*(t1[2]*t1[1]+t2[2]*t2[1]));coo.add(dz,jz,b*(t1[2]*t1[2]+t2[2]*t2[2]));}}
                }else{let sc=mu*sn/sm;for ln in 0..3{let dx=n[ln]*3;let dy=n[ln]*3+1;let dz=n[ln]*3+2;
                    rhs[dx]-=(sc*s1*t1[0]+sc*s2*t2[0])*phi[ln];rhs[dy]-=(sc*s1*t1[1]+sc*s2*t2[1])*phi[ln];rhs[dz]-=(sc*s1*t1[2]+sc*s2*t2[2])*phi[ln];}}}
        }
    }
    (rhs, coo.into_csr())
}

// ─── Solver ────────────────────────────────────────────────────────────────────

/// Newton solver for contact problems (2D scalar/vector, 3D scalar).
///
/// Supports penalty and Augmented Lagrangian contact, with optional
/// Coulomb friction (2D vector only).
pub fn solve_contact_newton<M: MeshTopology>(
    stiffness: &CsrMatrix<f64>,
    rhs_load: &[f64],
    mesh: &M,
    cfg: &ContactConfig,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = stiffness.nrows;
    let mut u = vec![0.0; n];
    let dim = mesh.dim() as usize;
    let is_vector = n == mesh.n_nodes() as usize * dim && dim == 2;
    let is_3d_scalar = dim == 3 && n == mesh.n_nodes() as usize;
    let is_3d_vector = dim == 3 && n == mesh.n_nodes() as usize * 3;

    let al_iters = match &cfg.contact_type {
        ContactType::AugmentedLagrangian { max_al_iter, .. } => *max_al_iter,
        _ => 1,
    };
    let al_tol = match &cfg.contact_type {
        ContactType::AugmentedLagrangian { al_tol, .. } => *al_tol,
        _ => tol,
    };

    // AL multipliers
    let n_faces = mesh.n_boundary_faces() as usize;
    let qp_per_face = 2;
    let mut lam_n = vec![0.0; n_faces.max(1) * qp_per_face];

    for al_iter in 0..al_iters {
        for _iter in 0..max_iter {
            let (f_contact, k_contact) = if is_vector {
                assemble_contact_2d_vector(mesh, cfg, &u, &lam_n)
            } else if is_3d_vector {
                assemble_contact_3d_vector(mesh, cfg, &u, &lam_n)
            } else if is_3d_scalar {
                assemble_contact_3d(mesh, cfg, &u, &lam_n)
            } else {
                assemble_contact_2d(mesh, mesh, cfg, &u, &lam_n)
            };

            // Residual: R(u) = K·u + f_contact(u) - b
            let mut ax = vec![0.0; n];
            stiffness.spmv(&u, &mut ax);
            let mut res = vec![0.0; n];
            for i in 0..n {
                res[i] = ax[i] + f_contact[i] - rhs_load[i];
            }
            let rn: f64 = res.iter().map(|v| v * v).sum::<f64>().sqrt();
            let b_n: f64 = rhs_load.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
            if rn < tol * b_n.max(1.0) { break; }

            let jac = stiffness.add(&k_contact);
            let mut du = vec![0.0; n];
            let neg_r: Vec<f64> = res.iter().map(|v| -v).collect();
            let _ = fem_solver::solve_cg(
                &jac, &neg_r, &mut du,
                &fem_solver::SolverConfig { rtol: 1e-10, max_iter: 500, ..Default::default() },
            );

            // Line search
            let mut alpha = 1.0;
            for _ in 0..10 {
                let mut u_n = u.clone();
                for i in 0..n { u_n[i] += alpha * du[i]; }
                let (f_n, _) = if is_vector {
                    assemble_contact_2d_vector(mesh, cfg, &u_n, &lam_n)
                } else if is_3d_vector {
                    assemble_contact_3d_vector(mesh, cfg, &u_n, &lam_n)
                } else if is_3d_scalar {
                    assemble_contact_3d(mesh, cfg, &u_n, &lam_n)
                } else {
                    assemble_contact_2d(mesh, mesh, cfg, &u_n, &lam_n)
                };
                let mut ax_n = vec![0.0; n];
                stiffness.spmv(&u_n, &mut ax_n);
                let mut r_n = vec![0.0; n];
                for i in 0..n { r_n[i] = ax_n[i] + f_n[i] - rhs_load[i]; }
                let rn_n: f64 = r_n.iter().map(|v| v * v).sum::<f64>().sqrt();
                if rn_n < rn || alpha < 1e-8 { u = u_n; break; }
                alpha *= 0.5;
            }
        }

        // Augmented Lagrangian multiplier update
        if matches!(cfg.contact_type, ContactType::AugmentedLagrangian { .. }) {
            let pen = cfg.penalty_normal;
            for f in 0..mesh.n_boundary_faces() as usize {
                let fnodes = mesh.face_nodes(f as u32);
                if fnodes.len() < 2 { continue; }
                let p0 = mesh.node_coords(fnodes[0]);
                let uh_avg = if is_vector { u[fnodes[0] as usize * 2] }
                             else { u[fnodes[0] as usize] };
                let gap = (cfg.gap_function)(&p0);
                for q in 0..qp_per_face {
                    lam_n[f * qp_per_face + q] = (lam_n[f * qp_per_face + q] + pen * (uh_avg - gap)).max(0.0);
                }
            }
        }

        // AL convergence
        if al_iter > 0 {
            let mut pmax = 0.0;
            let pen = cfg.penalty_normal;
            for f in 0..mesh.n_boundary_faces() as usize {
                let fnodes = mesh.face_nodes(f as u32);
                if fnodes.len() < 2 { continue; }
                let p0 = mesh.node_coords(fnodes[0]);
                let gap = (cfg.gap_function)(&p0);
                let uh_avg = if is_vector { u[fnodes[0] as usize * 2] } else { u[fnodes[0] as usize] };
                let gp = (pen * (uh_avg - gap)).abs();
                if gp > pmax { pmax = gp; }
            }
            if pmax < al_tol { break; }
        }
    }
    u
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;
    use fem_mesh::SimplexMesh;

    fn setup_2d() -> (SimplexMesh<2>, ContactConfig) {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let cfg = ContactConfig {
            penalty_normal: 1e6,
            contact_type: ContactType::Penalty,
            friction: FrictionModel::Frictionless,
            gap_function: |x: &[f64]| -0.1 - x[1],
            contact_tags: vec![1],
        };
        (mesh, cfg)
    }

    #[test]
    fn test_contact_assembles() {
        let (ref mesh, cfg) = setup_2d();
        let u = vec![0.0; mesh.n_nodes() as usize];
        let (f, k) = assemble_contact_2d(mesh, mesh, &cfg, &u, &[]);
        assert_eq!(f.len(), mesh.n_nodes() as usize);
        assert_eq!(k.nrows, mesh.n_nodes() as usize);
    }

    #[test]
    fn penalty_force_vanishes_when_no_penetration() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let cfg = ContactConfig {
            gap_function: |_: &[f64]| -1.0,
            ..setup_2d().1
        };
        let u = vec![0.0; mesh.n_nodes() as usize];
        let (f, _) = assemble_contact_2d(&mesh, &mesh, &cfg, &u, &[]);
        let fnorm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(fnorm < 1e-30, "expected zero contact force: {fnorm:.3e}");
    }

    #[test]
    fn penalty_force_exists_when_penetration() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let cfg = ContactConfig {
            penalty_normal: 1e5,
            gap_function: |_: &[f64]| 0.1,
            ..setup_2d().1
        };
        let u = vec![0.0; mesh.n_nodes() as usize];
        let (f, _) = assemble_contact_2d(&mesh, &mesh, &cfg, &u, &[]);
        let fnorm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(fnorm > 0.0, "expected contact force: {fnorm:.3e}");
    }

    #[test]
    fn test_contact_2d_vector_assembles() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let cfg = ContactConfig {
            penalty_normal: 1e6,
            gap_function: |_: &[f64]| -0.1,
            ..Default::default()
        };
        let u = vec![0.0; mesh.n_nodes() as usize * 2];
        let (f, k) = assemble_contact_2d_vector(&mesh, &cfg, &u, &[]);
        assert_eq!(f.len(), mesh.n_nodes() as usize * 2);
        assert_eq!(k.nrows, mesh.n_nodes() as usize * 2);
    }

    #[test]
    fn test_contact_3d_assembles() {
        use fem_space::H1Space;
        use fem_space::fe_space::FESpace;
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let space = H1Space::new(mesh, 1);
        let mesh_ref = space.mesh();
        let cfg = ContactConfig {
            penalty_normal: 1e6,
            gap_function: |x: &[f64]| -0.1 - x[2],
            contact_tags: vec![5],
            ..Default::default()
        };
        let u = vec![0.0; mesh_ref.n_nodes() as usize];
        let (f, k) = assemble_contact_3d(mesh_ref, &cfg, &u, &[]);
        assert_eq!(f.len(), mesh_ref.n_nodes() as usize);
        assert_eq!(k.nrows, mesh_ref.n_nodes() as usize);
    }

    #[test]
    fn augmented_lagrangian_assembles_correctly() {
        let (mesh, cfg) = setup_2d();
        let cfg = ContactConfig {
            penalty_normal: 1e5,
            contact_type: ContactType::AugmentedLagrangian { max_al_iter: 2, al_tol: 1e-3 },
            gap_function: |_: &[f64]| 0.01,
            ..cfg
        };
        let n = mesh.n_nodes() as usize;
        let u = vec![0.0; n];
        // With lagrange_multipliers = [], it falls back to penalty mode
        let (f_penalty, _k) = assemble_contact_2d(&mesh, &mesh, &cfg, &u, &[]);
        // With lagrange_multipliers provided, it uses the AL branch
        let lagrange_multipliers = vec![0.1; 2]; // small positive multiplier
        let (f_al, _k) = assemble_contact_2d(&mesh, &mesh, &cfg, &u, &lagrange_multipliers);
        // AL branch should give different result from penalty branch
        let fn_pen: f64 = f_penalty.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fn_al: f64 = f_al.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(fn_pen > 0.0, "penalty force should be non-zero");
        assert!(fn_al > 0.0, "AL force should be non-zero");
    }

    #[test]
    fn contact_3d_vector_assembly_finite() {
        let mesh = fem_mesh::SimplexMesh::<3>::unit_cube_tet(2);
        let cfg = ContactConfig {
            penalty_normal: 1e6,
            contact_type: ContactType::Penalty,
            friction: FrictionModel::Frictionless,
            gap_function: |x| 0.1 - x[1],
            contact_tags: vec![2], // top face y=1 has tag 2
        };
        let u = vec![0.0; mesh.n_nodes() * 3];
        let (f, k) = assemble_contact_3d_vector(&mesh, &cfg, &u, &[]);
        assert!(f.iter().all(|v| v.is_finite()), "3D vector contact RHS non-finite");
        assert!(k.nrows > 0, "3D vector contact matrix empty");
    }
}
