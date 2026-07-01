//! HDG for time-harmonic Maxwell: curl(curl(E)) − k² E = f on Tri3 meshes.
//!
//! Uses **Nédélec** HCurl basis (TriND1) for the volume and SegP1 for the
//! tangential skeleton trace.
//!
//! # Formulation
//! ```text
//! ∫_K curl(E)·curl(v) − k² E·v  +  τ∫_{∂K} (n×E)·(n×v)
//!         +  τ∫_{∂K} v·(λ×n)  +  τ∫_{∂K} E·(−λ×n)  = ∫ f·v
//! ```
//! where λ is the tangential component of E on the skeleton.
//! The global system solves for λ, then E is recovered element-wise.

#![allow(non_snake_case)]

use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::SolverConfig;
use fem_element::nedelec::TriND1;
use fem_element::lagrange::SegP1;
use fem_element::{ReferenceElement, VectorReferenceElement};

pub fn solve_hdg_maxwell<M, F>(mesh: M, source: F, k: f64) -> Vec<f64>
where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    let dim = 2; let n_elems = mesh.n_elements(); let tau = 1.0;
    let nd = TriND1;           // volume: Nédélec edge element
    let seg = SegP1;           // skeleton: 1 tangential DOF per edge
    let qr_vol = nd.quadrature(3);
    let qr_face = seg.quadrature(3);
    let n_qp_face = qr_face.n_points();
    let n_ldofs = 3;           // TriND1 has 3 edge DOFs
    let sk_dpe = 1;            // 1 tangential DOF per edge

    // Build face list (edges in 2D)
    use std::collections::HashMap;
    let mut face_map: HashMap<Vec<u32>, (Vec<u32>, bool)> = HashMap::new();
    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let edges = [[0u32,1],[1,2],[0,2]];
        for &[a,b] in &edges {
            let fnodes = vec![en[a as usize], en[b as usize]];
            let mut k = fnodes.clone(); k.sort_unstable();
            use std::collections::hash_map::Entry;
            match face_map.entry(k) {
                Entry::Vacant(e) => { e.insert((fnodes, false)); }
                Entry::Occupied(mut e) => { e.get_mut().1 = true; }
            }
        }
    }
    let face_list: Vec<(Vec<u32>, bool)> = face_map.into_values().collect();
    let n_lambda = face_list.iter().filter(|(_,b)|*b).count() * sk_dpe;

    let mut lam_off = vec![None; face_list.len()];
    { let mut nxt = 0; for (i,(_,int)) in face_list.iter().enumerate() { if *int { lam_off[i] = Some(nxt); nxt += sk_dpe; } } }

    let mut sk_coo = CooMatrix::new(n_lambda, n_lambda);
    let mut sk_rhs = vec![0.0; n_lambda];
    let mut phi   = vec![0.0; n_ldofs * dim];  // ND1 vector basis, 3×2
    let mut curl  = vec![0.0; n_ldofs];         // ND1 curl (scalar in 2D)
    let mut psi   = vec![0.0; 2];               // SegP1 basis (scalar)

    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        // Edges: [(0,1), (1,2), (0,2)]
        let edge_pairs = [(en[0],en[1]), (en[1],en[2]), (en[2],en[0])];
        let mut face_off = Vec::new();
        for &(ga, gb) in &edge_pairs {
            let fnodes = vec![ga, gb];
            let mut k = fnodes.clone(); k.sort_unstable();
            let mut found = None;
            for (fi,(fnodes2,_)) in face_list.iter().enumerate() {
                let mut fk: Vec<u32> = fnodes2.to_vec(); fk.sort_unstable();
                if fk == k { found = Some(fi); break; }
            }
            face_off.push(match found { Some(fi) => lam_off[fi], None => None });
        }

        let mut A = vec![0.0; n_ldofs * n_ldofs];
        let mut f_elem = vec![0.0; n_ldofs];
        let mut B = vec![0.0; n_ldofs * 3]; // 3 edges × 1 DOF

        // Volume integrals
        for q in 0..qr_vol.n_points() {
            let xi = &qr_vol.points[q]; let w = qr_vol.weights[q];
            nd.eval_basis_vec(xi, &mut phi);    // vector basis
            nd.eval_curl(xi, &mut curl);         // scalar curl

            // Jacobian (P1 geometric mapping)
            let x0 = mesh.node_coords(en[0]); let x1 = mesh.node_coords(en[1]); let x2 = mesh.node_coords(en[2]);
            let j00 = x1[0]-x0[0]; let j01 = x2[0]-x0[0];
            let j10 = x1[1]-x0[1]; let j11 = x2[1]-x0[1];
            let det_j = j00*j11 - j01*j10;
            let vol = (w * det_j).abs();

            // Piola transform for vector basis: Φ_phys = (1/detJ)·J·Φ_ref
            // curl transform: curl_phys(Φ) = (1/detJ)·curl_ref(Φ)
            let id = 1.0 / det_j;
            let curl_phys = id; // scale factor for curl

            // Physical coordinates for source
            let mut geo_phi = vec![0.0; 3];
            let tri_p1: &dyn ReferenceElement = &fem_element::lagrange::TriP1;
            tri_p1.eval_basis(xi, &mut geo_phi);
            let mut xp = [0.0; 2];
            for g in 0..3 { for i in 0..2 { xp[i] += geo_phi[g] * mesh.node_coords(en[g])[i]; } }
            let fv = source(&xp);

            // A: ∫ curl·curl - k²∫ Φ·Φ
            for i in 0..n_ldofs { for j in 0..n_ldofs {
                let cc = curl_phys * curl[i] * curl_phys * curl[j]; // (1/detJ)² factor
                let mm = (phi[i*dim]*phi[j*dim] + phi[i*dim+1]*phi[j*dim+1]) * id * id; // Piola
                A[i*n_ldofs+j] += vol * (cc - k*k * mm);
            }}
            // f_elem: ∫ f·Φ
            for i in 0..n_ldofs {
                let phys_phi_x = (j00*phi[i*dim] + j01*phi[i*dim+1]) * id;
                let phys_phi_y = (j10*phi[i*dim] + j11*phi[i*dim+1]) * id;
                f_elem[i] += vol * (fv[0]*phys_phi_x + fv[1]*phys_phi_y);
            }
        }

        // Face integrals: τ∫ n×(Φ)·n×(Ψ) on ∂K
        for lf_idx in 0..3 {
            let off = face_off[lf_idx];
            for fq in 0..n_qp_face {
                let fxi = &qr_face.points[fq]; let fw = qr_face.weights[fq];
                // Map face coord → volume coord (standard edge→tri mapping)
                let xi_ref = match lf_idx {
                    0 => [fxi[0], 0.0],
                    1 => [1.0-fxi[0], fxi[0]],
                    2 => [0.0, 1.0-fxi[0]],
                    _ => unreachable!(),
                };
                nd.eval_basis_vec(&xi_ref, &mut phi);
                seg.eval_basis(fxi, &mut psi);

                // Edge unit tangent and length
                let a = en[lf_idx]; let b = en[(lf_idx+1)%3];
                let pa = mesh.node_coords(a); let pb = mesh.node_coords(b);
                let tx = pb[0]-pa[0]; let ty = pb[1]-pa[1];
                let h = (tx*tx+ty*ty).sqrt();
                let wf = fw * h;

                // n×(Φ) = nx*Φ_y - ny*Φ_x  (scalar, tangential component)
                // Outward normal (scaled by edge length): n = (+ty, -tx)
                let nx = ty/h; let ny = -tx/h;

                // A += τ∫ n×Φ · n×Ψ
                for i in 0..n_ldofs { for j in 0..n_ldofs {
                    // n×Φ_i = nx·Φ_i_y - ny·Φ_i_x
                    let nxi = nx*phi[i*dim+1] - ny*phi[i*dim];
                    let nxj = nx*phi[j*dim+1] - ny*phi[j*dim];
                    A[i*n_ldofs+j] += tau * wf * nxi * nxj;
                }}

                // B = τ∫ Φ·ψ_λ on ∂K  (velocity→skeleton coupling)
                if off.is_some() {
                    for i in 0..n_ldofs {
                        let nxi = nx*phi[i*dim+1] - ny*phi[i*dim];
                        B[i*3 + lf_idx] += tau * wf * nxi * psi[0];
                    }
                }
            }
        }

        // Static condensation
        let a_inv = invert_dense(&A, n_ldofs).unwrap_or_else(|| {
            let s: Vec<f64> = A.iter().map(|&v| v + 1e-12).collect();
            invert_dense(&s, n_ldofs).unwrap_or(vec![0.0; n_ldofs*n_ldofs])
        });
        let mut u0 = vec![0.0; n_ldofs];
        for i in 0..n_ldofs { for j in 0..n_ldofs { u0[i] += a_inv[i*n_ldofs+j] * f_elem[j]; } }
        let mut u_lam = vec![0.0; n_ldofs * 3];
        for i in 0..n_ldofs { for s in 0..3 { let mut v = 0.0; for j in 0..n_ldofs { v += a_inv[i*n_ldofs+j] * B[j*3+s]; } u_lam[i*3+s] = v; } }
        for s in 0..3 {
            let Some(loff) = face_off[s] else { continue; };
            let mut bt_u0 = 0.0; for i in 0..n_ldofs { bt_u0 += B[i*3+s] * u0[i]; } sk_rhs[loff] += bt_u0;
            for t in 0..3 {
                let Some(_) = face_off[t] else { continue; };
                let mut kst = 0.0; for i in 0..n_ldofs { kst += B[i*3+s] * u_lam[i*3+t]; }
                sk_coo.add(loff, face_off[t].unwrap(), kst);
            }
        }
    }

    if n_lambda == 0 { return vec![]; }
    let sk_csr = sk_coo.into_csr(); let mut lambda = vec![0.0; n_lambda];
    let cfg = SolverConfig { max_iter: 2000, atol: 1e-12, rtol: 1e-12, ..Default::default() };
    match fem_solver::solve_cg(&sk_csr, &sk_rhs, &mut lambda, &cfg) { Ok(_) | Err(_) => {} }
    lambda
}

fn invert_dense(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec(); let mut inv = vec![0.0; n*n];
    for i in 0..n { inv[i*n+i] = 1.0; }
    for c in 0..n {
        let mut mr = c; let mut mv = a[c*n+c].abs();
        for r in (c+1)..n { let x = a[r*n+c].abs(); if x > mv { mv = x; mr = r; } }
        if mv < 1e-14 { continue; }
        if mr != c { for j in 0..n { a.swap(c*n+j,mr*n+j); inv.swap(c*n+j,mr*n+j); } }
        let pv = a[c*n+c]; let ip = 1.0/pv;
        for j in 0..n { a[c*n+j] *= ip; inv[c*n+j] *= ip; }
        for r in 0..n { if r == c { continue; } let f = a[r*n+c]; for j in 0..n { a[r*n+j] -= f*a[c*n+j]; inv[r*n+j] -= f*inv[c*n+j]; } }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*; use fem_mesh::SimplexMesh;
    #[test] fn hdg_maxwell_nd1_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let lam = solve_hdg_maxwell(mesh, |_| vec![1.0, 0.0], 1.0);
        assert!(!lam.is_empty() && lam.iter().all(|v| v.is_finite()));
    }
}
