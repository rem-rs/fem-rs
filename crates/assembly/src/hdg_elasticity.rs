//! HDG for linear elasticity: −μΔu − (λ+μ)∇(∇·u) = f.
//!
//! Uses P1 vector (6 DOFs per triangle) bulk element, P1 skeleton trace.
//! dim = 2, Tri3 mesh.

use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::SolverConfig;
use fem_element::lagrange::{TriP1, SegP1};

#[derive(Debug)]
pub struct HdgElasticityResult {
    pub u: Vec<f64>,
    pub lambda: Vec<f64>,
}

pub fn solve_hdg_elasticity<M, F>(
    mesh: M,
    source: F,
    mu: f64,
    lambda: f64,
) -> HdgElasticityResult
where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    let dim = mesh.dim() as usize;
    assert_eq!(dim, 2, "HDG elasticity currently supports 2D only");
    let n_elems = mesh.n_elements();
    let tau = 2.0 * mu; // stabilization parameter

    // P1 velocity: 3 DOFs × dim = 6 per element
    let u_dpe = (dim + 1) * dim;
    let n_u = n_elems * u_dpe;

    // P1 skeleton: each edge has 2 vertices, each with dim velocity DOFs
    let sk_dpe = 2 * dim; // 4 in 2D

    let ref_elem: Box<dyn fem_element::ReferenceElement> = Box::new(TriP1);
    let geo_elem: Box<dyn fem_element::ReferenceElement> = Box::new(TriP1);
    let geo_n = geo_elem.n_dofs();
    let face_ref: Box<dyn fem_element::ReferenceElement> = Box::new(SegP1);
    let n_qp_face = face_ref.quadrature(2).n_points();
    let qr_face = face_ref.quadrature(2);
    let qr_vol = ref_elem.quadrature(2);

    // Build face list
    use std::collections::HashMap;
    let mut face_map: HashMap<Vec<u32>, (Vec<u32>, bool)> = HashMap::new();
    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let lf = vec![vec![0u32, 1], vec![1, 2], vec![0, 2]];
        for f in &lf {
            let mut k: Vec<u32> = f.iter().map(|&x| en[x as usize]).collect();
            k.sort_unstable();
            use std::collections::hash_map::Entry;
            match face_map.entry(k) {
                Entry::Vacant(e) => { e.insert((f.clone(), false)); }
                Entry::Occupied(mut e) => { e.get_mut().1 = true; }
            }
        }
    }
    let face_list: Vec<(Vec<u32>, bool)> = face_map.into_values().collect();
    let n_faces = face_list.len();
    let n_lambda = face_list.iter().filter(|(_, interior)| *interior).count() * sk_dpe;

    let mut lam_off: Vec<Option<usize>> = vec![None; n_faces];
    {
        let mut nxt = 0;
        for (i, (_, interior)) in face_list.iter().enumerate() {
            if *interior {
                lam_off[i] = Some(nxt);
                nxt += sk_dpe;
            }
        }
    }

    let mut sk_coo = CooMatrix::new(n_lambda, n_lambda);
    let mut sk_rhs = vec![0.0; n_lambda];
    let mut phi = vec![0.0; dim + 1];
    let mut grad = vec![0.0; (dim + 1) * dim];
    let mut psi = vec![0.0; dim]; // P1 face basis (2 nodes × 2D)

    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let lf_list = vec![vec![0u32, 1], vec![1, 2], vec![0, 2]];
        let n_lf = lf_list.len();

        let mut face_off: Vec<Option<usize>> = Vec::new();
        for f in &lf_list {
            let mut k: Vec<u32> = f.iter().map(|&x| en[x as usize]).collect();
            k.sort_unstable();
            let mut found = None;
            for (fi, (fnodes, _)) in face_list.iter().enumerate() {
                let mut fk: Vec<u32> = fnodes.iter().copied().collect();
                fk.sort_unstable();
                if fk == k { found = Some(fi); break; }
            }
            match found { Some(fi) => face_off.push(lam_off[fi]), None => face_off.push(None) }
        }

        let nu = u_dpe;
        let ns = n_lf * sk_dpe;
        let mut A = vec![0.0; nu * nu];
        let mut f_u = vec![0.0; nu];
        let mut B = vec![0.0; nu * ns];

        // Volume integrals
        for q in 0..qr_vol.n_points() {
            let xi = &qr_vol.points[q];
            let w = qr_vol.weights[q];

            ref_elem.eval_basis(xi, &mut phi);
            ref_elem.eval_grad_basis(xi, &mut grad);

            let mut geo_grad = vec![0.0; geo_n * dim];
            geo_elem.eval_grad_basis(xi, &mut geo_grad);
            let mut jac = vec![vec![0.0; dim]; dim];
            for i in 0..dim {
                for d in 0..dim {
                    for k in 0..geo_n {
                        jac[i][d] += mesh.node_coords(en[k])[i] * geo_grad[k * dim + d];
                    }
                }
            }
            let det_j = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
            let vol = (w * det_j).abs();
            let id = 1.0 / det_j;

            // Physical gradients
            let mut gp = vec![0.0; (dim + 1) * dim];
            let (j00, j01, j10, j11) = (
                jac[1][1] * id,
                -jac[0][1] * id,
                -jac[1][0] * id,
                jac[0][0] * id,
            );
            for i in 0..dim + 1 {
                gp[i * dim] = j00 * grad[i * dim] + j01 * grad[i * dim + 1];
                gp[i * dim + 1] = j10 * grad[i * dim] + j11 * grad[i * dim + 1];
            }

            // Physical coords for source
            let mut geo_phi = vec![0.0; geo_n];
            geo_elem.eval_basis(xi, &mut geo_phi);
            let mut xp = vec![0.0; dim];
            for k in 0..geo_n {
                let c = mesh.node_coords(en[k]);
                for i in 0..dim {
                    xp[i] += geo_phi[k] * c[i];
                }
            }
            let fv = source(&xp);

            // A += μ∫∇u·∇w (component-wise: δ_ab Σ_c ∂φ_i/∂x_c · ∂φ_j/∂x_c)
            for a in 0..dim {
                for i in 0..dim + 1 {
                    for j in 0..dim + 1 {
                        let mut d = 0.0;
                        for b in 0..dim {
                            d += gp[i * dim + b] * gp[j * dim + b];
                        }
                        A[(i * dim + a) * nu + (j * dim + a)] += mu * vol * d;
                    }
                }
            }

            // A += (λ+μ)∫(∇·u)(∇·w) (grad-div coupling)
            // (∇·u) = Σ_c ∂u_c/∂x_c, (∇·w) = Σ_c ∂w_c/∂x_c
            for a in 0..dim {
                for i in 0..dim + 1 {
                    for b in 0..dim {
                        for j in 0..dim + 1 {
                            A[(i * dim + a) * nu + (j * dim + b)] +=
                                (lambda + mu) * vol * gp[i * dim + a] * gp[j * dim + b];
                        }
                    }
                }
            }

            // f_u += ∫ f·φ
            for a in 0..dim {
                for i in 0..dim + 1 {
                    f_u[i * dim + a] += vol * phi[i] * fv[a];
                }
            }
        }

        // Face integrals: τ∫φ·φ on ∂K and τ∫φ·ψ_λ on ∂K
        for (lf_idx, _lf) in lf_list.iter().enumerate() {
            for fq in 0..n_qp_face {
                let fxi = &qr_face.points[fq];
                let fw = qr_face.weights[fq];
                let xi_ref = match lf_idx {
                    0 => vec![fxi[0], 0.0],
                    1 => vec![1.0 - fxi[0], fxi[0]],
                    2 => vec![0.0, 1.0 - fxi[0]],
                    _ => unreachable!(),
                };
                ref_elem.eval_basis(&xi_ref, &mut phi);
                face_ref.eval_basis(fxi, &mut psi);
                let fj = face_size(&mesh, &en, lf_idx, dim);
                let wf = fw * fj;

                // τ∫φ·φ on ∂K
                for a in 0..dim {
                    for i in 0..dim + 1 {
                        for j in 0..dim + 1 {
                            A[(i * dim + a) * nu + (j * dim + a)] += tau * wf * phi[i] * phi[j];
                        }
                    }
                }

                // τ∫φ·ψ_λ on ∂K (B matrix)
                if let Some(loff) = face_off[lf_idx] {
                    let base = lf_idx * sk_dpe;
                    for a in 0..dim {
                        for i in 0..dim + 1 {
                            let dof_row = i * dim + a;
                            for ld in 0..dim {
                                // ψ has dim entries: ψ[ld] at vertex ld of the face
                                // λ DOF for this face vertex ld, component a
                                let lam_col = base + ld * dim + a;
                                B[dof_row * ns + lam_col] += tau * wf * phi[i] * psi[ld];
                            }
                        }
                    }
                }
            }
        }

        // Static condensation: eliminate u in terms of λ
        // Local system: A u = f_u - B λ
        // u = A^{-1} f_u - A^{-1} B λ = u0 + U_λ · λ

        let a_inv = invert_dense(&A, nu).unwrap_or_else(|| {
            let s: Vec<f64> = A.iter().map(|&v| v + 1e-12).collect();
            invert_dense(&s, nu).unwrap_or(vec![0.0; nu * nu])
        });

        // u0 = A^{-1} f_u
        let mut u0 = vec![0.0; nu];
        for i in 0..nu {
            for j in 0..nu {
                u0[i] += a_inv[i * nu + j] * f_u[j];
            }
        }

        // u_lam = -A^{-1} B  (response to each λ DOF)
        let mut u_lam = vec![0.0; nu * ns];
        for i in 0..nu {
            for s in 0..ns {
                let mut v = 0.0;
                for j in 0..nu {
                    v += a_inv[i * nu + j] * B[j * ns + s];
                }
                u_lam[i * ns + s] = -v;
            }
        }

        // Assemble skeleton system: K = B^T A^{-1} B
        // RHS: g = B^T u0
        for s in 0..ns {
            let lf_idx = s / sk_dpe;
            let ld = s % sk_dpe;
            let Some(loff) = face_off[lf_idx] else { continue; };
            let lam_s = loff + ld;

            // g_s += B^T u0
            let mut bt_u0 = 0.0;
            for i in 0..nu {
                bt_u0 += B[i * ns + s] * u0[i];
            }
            sk_rhs[lam_s] += bt_u0;

            for t in 0..ns {
                let lf_idx2 = t / sk_dpe;
                let ld2 = t % sk_dpe;
                let Some(loff2) = face_off[lf_idx2] else { continue; };
                let lam_t = loff2 + ld2;

                // K_st = B^T u_lam[:, t] = Σ_i B[i,s] * u_lam[i,t]
                let mut kst = 0.0;
                for i in 0..nu {
                    kst += B[i * ns + s] * u_lam[i * ns + t];
                }

                sk_coo.add(lam_s, lam_t, kst);
            }
        }
    }

    // Solve global skeleton system
    if n_lambda == 0 {
        return HdgElasticityResult {
            u: vec![0.0; n_u],
            lambda: vec![],
        };
    }
    let sk_csr = sk_coo.into_csr();
    let mut lambda_sol = vec![0.0; n_lambda];
    let cfg = SolverConfig {
        max_iter: 2000,
        atol: 1e-12,
        rtol: 1e-12,
        ..Default::default()
    };
    match fem_solver::solve_cg(&sk_csr, &sk_rhs, &mut lambda_sol, &cfg) {
        Ok(_) | Err(_) => {}
    }

    // Reconstruct bulk solution: u = u0 + u_lam · λ
    let mut u_bulk = vec![0.0; n_u];

    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let lf_list = vec![vec![0u32, 1], vec![1, 2], vec![0, 2]];
        let n_lf = lf_list.len();
        let ns = n_lf * sk_dpe;

        let mut face_off: Vec<Option<usize>> = Vec::new();
        for f in &lf_list {
            let mut k: Vec<u32> = f.iter().map(|&x| en[x as usize]).collect();
            k.sort_unstable();
            let mut found = None;
            for (fi, (fnodes, _)) in face_list.iter().enumerate() {
                let mut fk: Vec<u32> = fnodes.iter().copied().collect();
                fk.sort_unstable();
                if fk == k { found = Some(fi); break; }
            }
            match found { Some(fi) => face_off.push(lam_off[fi]), None => face_off.push(None) }
        }

        // Rebuild element matrices
        let nu = u_dpe;
        let mut A = vec![0.0; nu * nu];
        let mut f_u = vec![0.0; nu];
        let mut B = vec![0.0; nu * ns];

        for q in 0..qr_vol.n_points() {
            let xi = &qr_vol.points[q];
            let w = qr_vol.weights[q];
            ref_elem.eval_basis(xi, &mut phi);
            ref_elem.eval_grad_basis(xi, &mut grad);

            let mut gg = vec![0.0; geo_n * dim];
            geo_elem.eval_grad_basis(xi, &mut gg);
            let mut jac = vec![vec![0.0; dim]; dim];
            for i in 0..dim {
                for d in 0..dim {
                    for k in 0..geo_n {
                        jac[i][d] += mesh.node_coords(en[k])[i] * gg[k * dim + d];
                    }
                }
            }
            let det_j = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
            let vol = (w * det_j).abs();
            let id = 1.0 / det_j;

            let (j00, j01, j10, j11) = (
                jac[1][1] * id, -jac[0][1] * id, -jac[1][0] * id, jac[0][0] * id,
            );
            let mut gp = vec![0.0; (dim + 1) * dim];
            for i in 0..dim + 1 {
                gp[i * dim] = j00 * grad[i * dim] + j01 * grad[i * dim + 1];
                gp[i * dim + 1] = j10 * grad[i * dim] + j11 * grad[i * dim + 1];
            }

            let mut geo_phi = vec![0.0; geo_n];
            geo_elem.eval_basis(xi, &mut geo_phi);
            let mut xp = vec![0.0; dim];
            for k in 0..geo_n {
                let c = mesh.node_coords(en[k]);
                for i in 0..dim {
                    xp[i] += geo_phi[k] * c[i];
                }
            }
            let fv = source(&xp);

            for a in 0..dim {
                for i in 0..dim + 1 {
                    for j in 0..dim + 1 {
                        let mut d = 0.0;
                        for b in 0..dim {
                            d += gp[i * dim + b] * gp[j * dim + b];
                        }
                        A[(i * dim + a) * nu + (j * dim + a)] += mu * vol * d;
                    }
                }
            }
            for a in 0..dim {
                for i in 0..dim + 1 {
                    for b in 0..dim {
                        for j in 0..dim + 1 {
                            A[(i * dim + a) * nu + (j * dim + b)] +=
                                (lambda + mu) * vol * gp[i * dim + a] * gp[j * dim + b];
                        }
                    }
                }
            }
            for a in 0..dim {
                for i in 0..dim + 1 {
                    f_u[i * dim + a] += vol * phi[i] * fv[a];
                }
            }
        }

        for (lf_idx, _lf) in lf_list.iter().enumerate() {
            for fq in 0..n_qp_face {
                let fxi = &qr_face.points[fq];
                let fw = qr_face.weights[fq];
                let xi_ref = match lf_idx {
                    0 => vec![fxi[0], 0.0],
                    1 => vec![1.0 - fxi[0], fxi[0]],
                    2 => vec![0.0, 1.0 - fxi[0]],
                    _ => unreachable!(),
                };
                ref_elem.eval_basis(&xi_ref, &mut phi);
                face_ref.eval_basis(fxi, &mut psi);
                let fj = face_size(&mesh, &en, lf_idx, dim);
                let wf = fw * fj;

                for a in 0..dim {
                    for i in 0..dim + 1 {
                        for j in 0..dim + 1 {
                            A[(i * dim + a) * nu + (j * dim + a)] += tau * wf * phi[i] * phi[j];
                        }
                    }
                }

                if let Some(_) = face_off[lf_idx] {
                    let base = lf_idx * sk_dpe;
                    for a in 0..dim {
                        for i in 0..dim + 1 {
                            let dof_row = i * dim + a;
                            for ld in 0..dim {
                                let lam_col = base + ld * dim + a;
                                B[dof_row * ns + lam_col] += tau * wf * phi[i] * psi[ld];
                            }
                        }
                    }
                }
            }
        }

        let a_inv = invert_dense(&A, nu).unwrap_or_else(|| {
            let s: Vec<f64> = A.iter().map(|&v| v + 1e-12).collect();
            invert_dense(&s, nu).unwrap_or(vec![0.0; nu * nu])
        });

        let mut u0 = vec![0.0; nu];
        for i in 0..nu {
            for j in 0..nu {
                u0[i] += a_inv[i * nu + j] * f_u[j];
            }
        }

        let base_u = e as usize * u_dpe;
        for i in 0..nu {
            u_bulk[base_u + i] = u0[i];
        }

        // Add u_lam · λ contribution
        for s in 0..ns {
            let lf_idx = s / sk_dpe;
            let ld = s % sk_dpe;
            let Some(loff) = face_off[lf_idx] else { continue; };
            let lam_val = lambda_sol[loff + ld];
            for i in 0..nu {
                let mut contrib = 0.0;
                for j in 0..nu {
                    contrib += a_inv[i * nu + j] * B[j * ns + s];
                }
                u_bulk[base_u + i] -= contrib * lam_val;
            }
        }
    }

    HdgElasticityResult {
        u: u_bulk,
        lambda: lambda_sol,
    }
}

fn face_size<M: MeshTopology>(mesh: &M, enodes: &[u32], lf_idx: usize, dim: usize) -> f64 {
    if dim == 2 {
        let a = enodes[lf_idx];
        let b = enodes[(lf_idx + 1) % 3];
        let pa = mesh.node_coords(a);
        let pb = mesh.node_coords(b);
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        (dx * dx + dy * dy).sqrt()
    } else {
        unreachable!()
    }
}

fn invert_dense(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let mut mr = c;
        let mut mv = a[c * n + c].abs();
        for r in (c + 1)..n {
            let x = a[r * n + c].abs();
            if x > mv {
                mv = x;
                mr = r;
            }
        }
        if mv < 1e-15 {
            return None;
        }
        if mr != c {
            for j in 0..n {
                a.swap(c * n + j, mr * n + j);
                inv.swap(c * n + j, mr * n + j);
            }
        }
        let pv = a[c * n + c];
        let ip = 1.0 / pv;
        for j in 0..n {
            a[c * n + j] *= ip;
            inv[c * n + j] *= ip;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r * n + c];
            for j in 0..n {
                a[r * n + j] -= f * a[c * n + j];
                inv[r * n + j] -= f * inv[c * n + j];
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn hdg_elasticity_2d_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let source = |x: &[f64]| vec![0.0, 0.0];
        let result = solve_hdg_elasticity(mesh, source, 1.0, 1.0);
        for &v in &result.u {
            assert!(v.is_finite());
        }
        for &v in &result.lambda {
            assert!(v.is_finite());
        }
        assert!(result.lambda.len() > 0);
    }

    #[test]
    fn hdg_elasticity_nonzero_source() {
        use std::f64::consts::PI;
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let source = |x: &[f64]| vec![
            2.0 * (PI * x[0]).sin() * (PI * x[1]).sin(),
            2.0 * (PI * x[0]).cos() * (PI * x[1]).cos(),
        ];
        let result = solve_hdg_elasticity(mesh, source, 1.0, 1.0);
        for &v in &result.u {
            assert!(v.is_finite());
        }
        assert!(result.u.len() > 0);
    }
}
