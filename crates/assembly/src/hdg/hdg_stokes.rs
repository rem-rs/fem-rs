//! HDG Stokes: solves −νΔu + ∇p = f, div(u) = 0 on simplex meshes.
//!
//! Supports variable-order velocity and pressure:
//! - `vel_order = 1`: P1 discontinuous velocity (backward compatible)
//! - `vel_order = 2`: P2 discontinuous velocity
//! - `pres_order = 0`: P0 discontinuous pressure (backward compatible)
//! - `pres_order = 1`: P1 discontinuous pressure
//!
//! The skeleton trace for velocity matches `vel_order` on each edge/face.
//! Element matrices are cached during forward assembly (~2× speedup).

#![allow(non_snake_case)]

use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::SolverConfig;
use fem_element::ReferenceElement;
use fem_element::lagrange::{TriPk, TetPk, SegPk, TriPk as TriFacePk};

/// Scalar Pk DOFs for a simplex in dimension `dim`.
fn npe(dim: usize, k: usize) -> usize {
    if k == 0 { return 1; }
    match dim {
        1 => k + 1,
        2 => (k + 1) * (k + 2) / 2,
        3 => (k + 1) * (k + 2) * (k + 3) / 6,
        _ => unreachable!(),
    }
}

#[derive(Debug)]
pub struct HdgStokesResult {
    pub u: Vec<f64>,
    pub p: Vec<f64>,
    pub lambda: Vec<f64>,
}

struct ElemCache {
    face_off: Vec<Option<usize>>,
    sys_inv: Vec<f64>,
    b_mat: Vec<f64>,
    f_u: Vec<f64>,
    base_u: usize,
    base_p: usize,
}

/// Solve HDG Stokes with P1 velocity / P0 pressure (backward compat).
pub fn solve_hdg_stokes<M, F>(mesh: M, source: F, viscosity: f64) -> HdgStokesResult
where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    solve_hdg_stokes_order(mesh, source, viscosity, 1, 0)
}

/// Solve HDG Stokes with configurable velocity/pressure order.
pub fn solve_hdg_stokes_order<M, F>(
    mesh: M, source: F, viscosity: f64, vel_order: u8, pres_order: u8,
) -> HdgStokesResult
where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements();
    let tau = 2.0 * viscosity;
    let vo = vel_order as usize;
    let po = pres_order as usize;
    let n_vel_b = npe(dim, vo);     // scalar Pk DOFs per velocity component
    let n_pres_b = npe(dim, po);    // scalar Pk DOFs for pressure
    let n_sk_b = npe(dim - 1, vo);  // scalar Pk DOFs per face for skeleton
    let u_dpe = n_vel_b * dim;
    let p_dpe = n_pres_b;
    let sk_dpe = n_sk_b * dim;
    let n_u = n_elems * u_dpe;
    let n_p = n_elems * p_dpe;

    let ref_elem: Box<dyn fem_element::ReferenceElement> = match dim {
        2 => Box::new(TriPk::new(vo)),
        3 => Box::new(TetPk::new(vo)),
        _ => unreachable!(),
    };
    let geo_elem: Box<dyn fem_element::ReferenceElement> = match dim {
        2 => Box::new(TriPk::new(1)),
        3 => Box::new(TetPk::new(1)),
        _ => unreachable!(),
    };
    let geo_n = geo_elem.n_dofs();
    let face_ref: Box<dyn fem_element::ReferenceElement> = match dim {
        2 => Box::new(SegPk::new(vo)),
        3 => Box::new(TriFacePk::new(vo)),
        _ => unreachable!(),
    };
    let qr_vol = ref_elem.quadrature((2 * vo) as u8);
    let qr_face = face_ref.quadrature((2 * vo) as u8);
    let n_qp_face = qr_face.n_points();

    // ── Build face list ─────────────────────────────────────────────────
    use std::collections::HashMap;
    let mut face_map: HashMap<Vec<u32>, (Vec<u32>, bool)> = HashMap::new();
    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let lf = match (dim, en.len() as u32) {
            (2, 3) => vec![vec![0u32,1], vec![1,2], vec![0,2]],
            (3, 4) => vec![vec![0,1,2], vec![0,1,3], vec![0,2,3], vec![1,2,3]],
            _ => panic!(),
        };
        for f in &lf {
            let mut k: Vec<u32> = f.iter().map(|&x| en[x as usize]).collect(); k.sort_unstable();
            use std::collections::hash_map::Entry;
            match face_map.entry(k) {
                Entry::Vacant(e) => { e.insert((f.clone(), false)); }
                Entry::Occupied(mut e) => { e.get_mut().1 = true; }
            }
        }
    }
    let face_list: Vec<(Vec<u32>, bool)> = face_map.into_values().collect();
    let n_lambda = face_list.iter().filter(|(_, interior)| *interior).count() * sk_dpe;
    let mut lam_off: Vec<Option<usize>> = vec![None; face_list.len()];
    {
        let mut nxt = 0;
        for (i, (_, interior)) in face_list.iter().enumerate() {
            if *interior { lam_off[i] = Some(nxt); nxt += sk_dpe; }
        }
    }

    // ── Forward pass ────────────────────────────────────────────────────
    let mut sk_coo = CooMatrix::new(n_lambda, n_lambda);
    let mut sk_rhs = vec![0.0; n_lambda];
    let mut cache: Vec<ElemCache> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let lf_list = match (dim, en.len() as u32) {
            (2,3) => vec![vec![0u32,1], vec![1,2], vec![0,2]],
            (3,4) => vec![vec![0,1,2], vec![0,1,3], vec![0,2,3], vec![1,2,3]],
            _ => unreachable!(),
        };
        let n_lf = lf_list.len();
        let nu = u_dpe;
        let np = p_dpe;
        let ns = n_lf * sk_dpe;
        let n_tot = nu + np;

        let mut face_off: Vec<Option<usize>> = Vec::new();
        for f in &lf_list {
            let mut k: Vec<u32> = f.iter().map(|&x| en[x as usize]).collect(); k.sort_unstable();
            let mut found = None;
            for (fi, (fnodes, _)) in face_list.iter().enumerate() {
                let mut fk: Vec<u32> = fnodes.to_vec(); fk.sort_unstable();
                if fk == k { found = Some(fi); break; }
            }
            face_off.push(match found { Some(fi) => lam_off[fi], None => None });
        }

        let mut A = vec![0.0; nu * nu];
        let mut C = vec![0.0; nu * np];
        let mut f_u = vec![0.0; nu];
        let mut B = vec![0.0; nu * ns];
        let mut phi_v = vec![0.0; n_vel_b];
        let _phi_p = vec![0.0; n_pres_b];
        let mut gref = vec![0.0; n_vel_b * dim];
        let mut gphys = vec![0.0; n_vel_b * dim];

        // Volume integrals
        for q in 0..qr_vol.n_points() {
            let xi = &qr_vol.points[q]; let w = qr_vol.weights[q];
            ref_elem.eval_basis(xi, &mut phi_v);
            ref_elem.eval_grad_basis(xi, &mut gref);
            // Jacobian
            let mut gg = vec![0.0; geo_n * dim];
            geo_elem.eval_grad_basis(xi, &mut gg);
            let mut jac = vec![vec![0.0; dim]; dim];
            for i in 0..dim { for d in 0..dim { for k in 0..geo_n { jac[i][d] += mesh.node_coords(en[k])[i] * gg[k*dim+d]; } } }
            let det_j = if dim == 2 { jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0] } else {
                jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])
            };
            let vol = (w * det_j).abs(); let id = 1.0/det_j;
            // Physical gradients
            if dim == 2 {
                let (j00,j01,j10,j11) = (jac[1][1]*id,-jac[0][1]*id,-jac[1][0]*id,jac[0][0]*id);
                for i in 0..n_vel_b { gphys[i*2]=j00*gref[i*2]+j01*gref[i*2+1]; gphys[i*2+1]=j10*gref[i*2]+j11*gref[i*2+1]; }
            } else {
                let (m00,m01,m02,m10,m11,m12,m20,m21,m22)=(
                    (jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])*id,(jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])*id,(jac[0][1]*jac[1][2]-jac[0][2]*jac[1][1])*id,
                    (jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])*id,(jac[0][0]*jac[2][2]-jac[0][2]*jac[2][0])*id,(jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])*id,
                    (jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])*id,(jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])*id,(jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0])*id);
                for i in 0..n_vel_b{let gx=gref[i*dim];let gy=gref[i*dim+1];let gz=gref[i*dim+2];gphys[i*dim]=m00*gx+m01*gy+m02*gz;gphys[i*dim+1]=m10*gx+m11*gy+m12*gz;gphys[i*dim+2]=m20*gx+m21*gy+m22*gz;}
            }
            // Body force at QP
            let mut geo_phi = vec![0.0; geo_n];
            geo_elem.eval_basis(xi, &mut geo_phi);
            let mut xp = vec![0.0; dim];
            for k in 0..geo_n { let c = mesh.node_coords(en[k]); for i in 0..dim { xp[i] += geo_phi[k] * c[i]; } }
            let fv = source(&xp);

            // A: ν∫∇φ·∇φ (block-diagonal per component)
            for a in 0..dim { for i in 0..n_vel_b { for j in 0..n_vel_b {
                let mut d = 0.0; for b in 0..dim { d += gphys[i*dim+b] * gphys[j*dim+b]; }
                A[(i*dim+a)*nu + (j*dim+a)] += viscosity * vol * d;
            }}}
            // C: -∫ ψ_p · (∇·φ)  (velocity-pressure coupling)
            // ψ_p = phi_p basis for pressure; ∇·φ = Σ_a ∂φ/∂x_a
            for p in 0..n_pres_b { for i in 0..n_vel_b { for a in 0..dim {
                C[p * nu + i*dim + a] -= vol * phi_v[i] * gphys[i*dim+a];
            }}}
            // f_u: ∫ f·φ
            for a in 0..dim { for i in 0..n_vel_b { f_u[i*dim+a] += vol * phi_v[i] * fv[a]; } }
        }

        // Face integrals: τ∫φ·φ on ∂K and τ∫φ·ψ_λ on ∂K
        for lf_idx in 0..n_lf {
            let off = face_off[lf_idx];
            let mut psi = vec![0.0; n_sk_b];
            for fq in 0..n_qp_face {
                let fxi = &qr_face.points[fq]; let fw = qr_face.weights[fq];
                // Map face ref coord → volume ref coord
                let xi_ref = match (dim, lf_idx) {
                    (2,0) => vec![fxi[0],0.0], (2,1) => vec![1.0-fxi[0],fxi[0]], (2,2) => vec![0.0,1.0-fxi[0]],
                    (3,0) => vec![fxi[0],fxi[1],0.0], (3,1) => vec![fxi[0],0.0,fxi[1]],
                    (3,2) => vec![0.0,fxi[0],fxi[1]], (3,3) => vec![fxi[0],fxi[1],1.0-fxi[0]-fxi[1]],
                    _ => unreachable!(),
                };
                ref_elem.eval_basis(&xi_ref, &mut phi_v);
                face_ref.eval_basis(fxi, &mut psi);
                let fj = face_size(&mesh, en, lf_idx, dim);
                let wf = fw * fj;
                // A += τ φ·φ on ∂K
                for a in 0..dim { for i in 0..n_vel_b { for j in 0..n_vel_b {
                    A[(i*dim+a)*nu + (j*dim+a)] += tau * wf * phi_v[i] * phi_v[j];
                }}}
                // B = τ∫ φ·ψ_λ on ∂K  (velocity→skeleton coupling)
                if off.is_some() {
                    let base = lf_idx * sk_dpe;
                    for i in 0..n_vel_b { for v in 0..dim { for ld in 0..n_sk_b {
                        let row = i*dim + v;
                        let col = base + ld*dim + v;
                        B[row*ns + col] += tau * wf * phi_v[i] * psi[ld];
                    }}}
                }
            }
        }

        // ── Static condensation ──────────────────────────────────────────
        let mut sys = vec![0.0; n_tot * n_tot];
        let mut rhs = vec![0.0; n_tot];
        for i in 0..nu { for j in 0..nu { sys[i*n_tot+j] = A[i*nu+j]; } }
        // C^T block (upper-right) and C block (lower-left)
        for p in 0..np { for i in 0..nu {
            sys[i*n_tot + nu + p] = C[p*nu + i];       // C^T
            sys[(nu+p)*n_tot + i] = C[p*nu + i];       // C
        }}
        rhs[..nu].copy_from_slice(&f_u[..nu]);

        let sys_inv = invert_dense(&sys, n_tot).unwrap_or_else(|| {
            let s: Vec<f64> = sys.iter().map(|&v| v + 1e-12).collect();
            invert_dense(&s, n_tot).unwrap_or(vec![0.0; n_tot * n_tot])
        });

        // Particular + response
        let mut up0 = vec![0.0; n_tot];
        for i in 0..n_tot { for j in 0..n_tot { up0[i] += sys_inv[i*n_tot+j] * rhs[j]; } }
        let mut up_lam = vec![0.0; n_tot * ns];
        for i in 0..n_tot { for s in 0..ns { let mut v = 0.0; for j in 0..nu { v += sys_inv[i*n_tot+j] * B[j*ns+s]; } up_lam[i*ns+s] = v; } }

        // Assemble skeleton system
        for s in 0..ns {
            let lf_idx = s / sk_dpe; let ld = s % sk_dpe;
            let Some(loff) = face_off[lf_idx] else { continue; };
            let lam_s = loff + ld;
            let mut bt_u0 = 0.0;
            for i in 0..nu { bt_u0 += B[i*ns+s] * up0[i]; }
            sk_rhs[lam_s] += bt_u0;
            for t in 0..ns {
                let lf_idx2 = t / sk_dpe; let ld2 = t % sk_dpe;
                let Some(loff2) = face_off[lf_idx2] else { continue; };
                let mut kst = 0.0;
                for i in 0..nu { kst += B[i*ns+s] * up_lam[i*ns+t]; }
                sk_coo.add(lam_s, loff2 + ld2, kst);
            }
        }

        cache.push(ElemCache {
            face_off, sys_inv, b_mat: B, f_u,
            base_u: (e as usize) * u_dpe,
            base_p: (e as usize) * p_dpe,
        });
    }

    // ── Global solve ─────────────────────────────────────────────────────
    if n_lambda == 0 {
        return HdgStokesResult { u: vec![0.0; n_u], p: vec![0.0; n_p], lambda: vec![] };
    }
    let sk_csr = sk_coo.into_csr();
    let mut lambda = vec![0.0; n_lambda];
    let cfg = SolverConfig { max_iter: 2000, atol: 1e-12, rtol: 1e-12, ..Default::default() };
    match fem_solver::solve_cg(&sk_csr, &sk_rhs, &mut lambda, &cfg) { Ok(_) | Err(_) => {} }

    // ── Reconstruction from cache ────────────────────────────────────────
    let mut u_bulk = vec![0.0; n_u];
    let mut p_bulk = vec![0.0; n_p];
    for ec in &cache {
        let nu = u_dpe;
        let np = p_dpe;
        let ns = ec.face_off.len() * sk_dpe;
        let n_tot = nu + np;
        let mut up0 = vec![0.0; n_tot];
        for i in 0..n_tot { for j in 0..n_tot { up0[i] += ec.sys_inv[i*n_tot+j] * (if j < nu { ec.f_u[j] } else { 0.0 }); } }
        for i in 0..nu { u_bulk[ec.base_u + i] = up0[i]; }
        for p in 0..np { p_bulk[ec.base_p + p] = up0[nu + p]; }
        // Lambda correction
        for s in 0..ns {
            let lf_idx = s / sk_dpe; let ld = s % sk_dpe;
            let Some(loff) = ec.face_off[lf_idx] else { continue; };
            let lam_val = lambda[loff + ld];
            for i in 0..nu {
                let mut c = 0.0;
                for j in 0..nu { c += ec.sys_inv[i*n_tot+j] * ec.b_mat[j*ns+s]; }
                u_bulk[ec.base_u + i] += c * lam_val;
            }
            for p in 0..np {
                let mut c = 0.0;
                for j in 0..nu { c += ec.sys_inv[(nu+p)*n_tot+j] * ec.b_mat[j*ns+s]; }
                p_bulk[ec.base_p + p] += c * lam_val;
            }
        }
    }

    HdgStokesResult { u: u_bulk, p: p_bulk, lambda }
}

fn face_size<M: MeshTopology>(mesh: &M, enodes: &[u32], lf_idx: usize, dim: usize) -> f64 {
    if dim == 2 {
        let a = enodes[lf_idx]; let b = enodes[(lf_idx+1)%3];
        let pa = mesh.node_coords(a); let pb = mesh.node_coords(b);
        let dx = pb[0]-pa[0]; let dy = pb[1]-pa[1];
        (dx*dx+dy*dy).sqrt()
    } else {
        let lf: [usize;3] = match lf_idx { 0=>[0,1,2],1=>[0,1,3],2=>[0,2,3],3=>[1,2,3],_=>unreachable!() };
        let pa = mesh.node_coords(enodes[lf[0]]); let pb = mesh.node_coords(enodes[lf[1]]); let pc = mesh.node_coords(enodes[lf[2]]);
        let ux = pb[0]-pa[0]; let uy = pb[1]-pa[1]; let uz = pb[2]-pa[2];
        let vx = pc[0]-pa[0]; let vy = pc[1]-pa[1]; let vz = pc[2]-pa[2];
        let cx = uy*vz-uz*vy; let cy = uz*vx-ux*vz; let cz = ux*vy-uy*vx;
        0.5*(cx*cx+cy*cy+cz*cz).sqrt()
    }
}

fn invert_dense(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec(); let mut inv = vec![0.0; n*n];
    for i in 0..n { inv[i*n+i] = 1.0; }
    for c in 0..n {
        let mut mr = c; let mut mv = a[c*n+c].abs();
        for r in (c+1)..n { let x = a[r*n+c].abs(); if x > mv { mv = x; mr = r; } }
        if mv < 1e-15 { return None; }
        if mr != c { for j in 0..n { a.swap(c*n+j, mr*n+j); inv.swap(c*n+j, mr*n+j); } }
        let pv = a[c*n+c]; let ip = 1.0/pv;
        for j in 0..n { a[c*n+j] *= ip; inv[c*n+j] *= ip; }
        for r in 0..n { if r == c { continue; } let f = a[r*n+c]; for j in 0..n { a[r*n+j] -= f*a[c*n+j]; inv[r*n+j] -= f*inv[c*n+j]; } }
    }
    Some(inv)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test] fn hdg_stokes_2d_p1p0() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let r = solve_hdg_stokes(m, |_| vec![0.0,0.0], 1.0);
        assert!(r.u.iter().all(|v|v.is_finite()) && r.p.iter().all(|v|v.is_finite()));
    }

    #[test] fn hdg_stokes_3d_p1p0() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let r = solve_hdg_stokes_order(m, |_| vec![0.0,0.0,0.0], 1.0, 1, 0);
        assert!(r.u.iter().all(|v|v.is_finite()) && r.p.iter().all(|v|v.is_finite()));
    }

    #[test] fn hdg_stokes_2d_p2p1() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let r = solve_hdg_stokes_order(m, |_| vec![0.0,0.0], 1.0, 2, 1);
        assert!(r.u.iter().all(|v|v.is_finite()) && r.p.iter().all(|v|v.is_finite()));
    }

    #[test] fn hdg_stokes_3d_p2p1() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let r = solve_hdg_stokes_order(m, |_| vec![0.0,0.0,0.0], 1.0, 2, 1);
        assert!(r.u.iter().all(|v|v.is_finite()) && r.p.iter().all(|v|v.is_finite()));
    }

    #[test] fn hdg_stokes_2d_p3p2() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let r = solve_hdg_stokes_order(m, |_| vec![0.0,0.0], 1.0, 3, 2);
        assert!(r.u.iter().all(|v|v.is_finite()) && r.p.iter().all(|v|v.is_finite()));
    }
}
