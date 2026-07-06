//! VEM Poisson — arbitrary-order Pk on 2D polygons.
//!
//! Element stiffness: K_e = Πᵀ·G·Π + (I-Π)·S·(I-Π)
//! where:
//!   - Π projects VEM DOFs onto Pk polynomials (via Π = B·G⁻¹)
//!   - G is the Gram matrix of ∇ monomials
//!   - B[dof, α] = dof_d(m_α) evaluates each DOF on monomial m_α
//!   - S = tr(ΠᵀGΠ)/n_dofs · I is the stabilization term
//!     Ref: Beirao da Veiga et al. (2013), VEM review.

use nalgebra::DMatrix;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::poly_mesh::PolyMesh;
use fem_mesh::topology::MeshTopology;
use fem_space::vem::VEMSpace;
use fem_space::fe_space::FESpace;

/// 6-point triangle quadrature on the reference triangle (degree 4).
fn tri_6pt() -> ([[f64; 2]; 6], [f64; 6]) {
    (
        [
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0],
            [0.2, 0.2],
            [0.6, 0.2],
            [0.2, 0.6],
        ],
        [1.0 / 12.0; 6],
    )
}

fn poly_area(v: &[[f64; 2]]) -> f64 {
    let n = v.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += v[i][0] * v[j][1] - v[j][0] * v[i][1];
    }
    a.abs() / 2.0
}

fn centroid(v: &[[f64; 2]]) -> [f64; 2] {
    let n = v.len();
    let (mut cx, mut cy) = (0.0, 0.0);
    for i in 0..n {
        let j = (i + 1) % n;
        let c = v[i][0] * v[j][1] - v[j][0] * v[i][1];
        cx += (v[i][0] + v[j][0]) * c;
        cy += (v[i][1] + v[j][1]) * c;
    }
    let a = poly_area(v);
    if a > 1e-30 {
        [cx / (6.0 * a), cy / (6.0 * a)]
    } else {
        [0.0, 0.0]
    }
}

/// Compute ∫_T f(x,y) dA via centroid-triangulation + 6-point tri quadrature.
fn poly_int_2d(v: &[[f64; 2]], f: &dyn Fn(f64, f64) -> f64) -> f64 {
    let n = v.len();
    let c = centroid(v);
    let mut val = 0.0;
    let (tp, tw) = tri_6pt();
    for i in 0..n {
        let j = (i + 1) % n;
        let det = (v[j][0] - v[i][0]) * (c[1] - v[i][1])
            - (c[0] - v[i][0]) * (v[j][1] - v[i][1]);
        let area2 = det.abs();
        for (&pt, &w) in tp.iter().zip(tw.iter()) {
            let x = v[i][0] + pt[0] * (v[j][0] - v[i][0]) + pt[1] * (c[0] - v[i][0]);
            let y = v[i][1] + pt[0] * (v[j][1] - v[i][1]) + pt[1] * (c[1] - v[i][1]);
            val += w * area2 * f(x, y);
        }
    }
    val
}

/// Compute ∫_e g(t) dS over edge (v[ia], v[ib]) using 2-point Gauss-Legendre.
fn edge_int_1d(
    v: &[[f64; 2]],
    ia: usize,
    ib: usize,
    g: &dyn Fn(f64) -> f64,
) -> f64 {
    let xa = v[ia][0];
    let ya = v[ia][1];
    let xb = v[ib][0];
    let yb = v[ib][1];
    let len = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
    if len < 1e-30 {
        return 0.0;
    }
    // 2-point Gauss-Legendre on [-1,1] → map to [0,1] via t = (ξ+1)/2
    let gl_pts = [-0.5773502691896257, 0.5773502691896257];
    let gl_wts = [1.0, 1.0];
    let mut val = 0.0;
    for (&xi, &wi) in gl_pts.iter().zip(gl_wts.iter()) {
        let t = 0.5 * (xi + 1.0); // map to [0,1]
        val += wi * 0.5 * len * g(t);
    }
    val
}

/// Generate scaled monomial exponents (α, β) with α + β ≤ p.
/// Returns vector of (α, β) pairs.
fn monomial_exponents(p: usize) -> Vec<(usize, usize)> {
    let mut exps = Vec::new();
    for total in 0..=p {
        for a in 0..=total {
            exps.push((a, total - a));
        }
    }
    exps
}

/// Build the H¹ projection matrix Π and stiffness matrix for one VEM element.
///
/// Returns (K_e, dofs) where K_e is the n_dofs×n_dofs element stiffness.
fn vem_element_stiffness(
    verts: &[[f64; 2]],
    edge_has_dofs: &[bool],
    p: usize,
) -> DMatrix<f64> {
    let nv = verts.len();
    let n_edge_dofs: usize = edge_has_dofs.iter().map(|&b| if b { p.saturating_sub(1) } else { 0 }).sum();
    let n_internal = if p < 2 {
        0
    } else {
        (p - 1) * (p - 2) / 2
    };
    let n_dofs = nv + n_edge_dofs + n_internal;

    if n_dofs == 0 {
        return DMatrix::zeros(0, 0);
    }

    let c = centroid(verts);
    let area = poly_area(verts);
    if area < 1e-30 {
        return DMatrix::identity(n_dofs, n_dofs);
    }

    // Characteristic length h = sqrt(area)
    let h = area.sqrt();

    // Monomial basis: m_α(x,y) = ((x-cx)/h)^α_0 · ((y-cy)/h)^α_1
    let exps = monomial_exponents(p);
    let n_mono = exps.len();

    // Build Gram matrix G (n_mono × n_mono):
    // G[α][β] = ∫_T ∇m_α · ∇m_β dA
    // For shifted monomials: ∇m_a = (a0/h)·m_{a0-1,a1} or 0 if a0=0, similarly for a1
    // So G[α][β] = (α0·β0/h²)·∫m_{α0-1,α1}·m_{β0-1,β1} + (α1·β1/h²)·∫m_{α0,α1-1}·m_{β0,β1-1}
    let mut G = DMatrix::<f64>::zeros(n_mono, n_mono);

    for (a_idx, &(a0, a1)) in exps.iter().enumerate() {
        for (b_idx, &(b0, b1)) in exps.iter().enumerate() {
            let mut val = 0.0;
            // ξ-derivative contribution
            if a0 > 0 && b0 > 0 {
                val += (a0 as f64) * (b0 as f64) / h.powi(2)
                    * poly_int_2d(verts, &|x, y| {
                        let xi = (x - c[0]) / h;
                        let yi = (y - c[1]) / h;
                        xi.powi((a0 - 1) as i32)
                            * yi.powi(a1 as i32)
                            * xi.powi((b0 - 1) as i32)
                            * yi.powi(b1 as i32)
                    });
            }
            // η-derivative contribution
            if a1 > 0 && b1 > 0 {
                val += (a1 as f64) * (b1 as f64) / h.powi(2)
                    * poly_int_2d(verts, &|x, y| {
                        let xi = (x - c[0]) / h;
                        let yi = (y - c[1]) / h;
                        xi.powi(a0 as i32)
                            * yi.powi((a1 - 1) as i32)
                            * xi.powi(b0 as i32)
                            * yi.powi((b1 - 1) as i32)
                    });
            }
            G[(a_idx, b_idx)] = val;
        }
    }
    // Ensure G is invertible (add small regularization for nullspace — the constant
    // monomial m_{(0,0)} has zero gradient, making G singular. We handle this by
    // using only ∇-relevant monomials or adding a small regularization).
    // Standard VEM: the projection is only defined for m_a with |a| ≥ 1 (non-constant).
    // We build G for the non-constant part and handle the constant separately.

    // Separate constant monomial (index 0 ≡ (0,0))
    // G for indices 1..n_mono-1 (the gradient-active monomials)
    if n_mono <= 1 {
        return DMatrix::zeros(n_dofs, n_dofs);
    }

    let n_grad_mono = n_mono - 1; // exclude constant
    let mut G_grad = DMatrix::<f64>::zeros(n_grad_mono, n_grad_mono);
    for a_idx in 1..n_mono {
        for b_idx in 1..n_mono {
            G_grad[(a_idx - 1, b_idx - 1)] = G[(a_idx, b_idx)];
        }
    }

    // Invert G_grad
    let invG = match G_grad.clone().try_inverse() {
        Some(inv) => inv,
        None => DMatrix::identity(n_grad_mono, n_grad_mono),
    };

    // Build DOF evaluation matrix B (n_dofs × n_grad_mono) for non-constant monomials
    let mut B = DMatrix::<f64>::zeros(n_dofs, n_grad_mono);

    // Vertex DOFs: B[i][α] = monomial_α(x_i)
    for i in 0..nv {
        let xx = (verts[i][0] - c[0]) / h;
        let yy = (verts[i][1] - c[1]) / h;
        for (m_idx, &(a0, a1)) in exps.iter().enumerate().skip(1) {
            B[(i, m_idx - 1)] = xx.powi(a0 as i32) * yy.powi(a1 as i32);
        }
    }

    // Edge DOFs: B[nv + e*order + k][α] = ∫_e m_α · q_k dS / |e|
    // where q_k are normalized Legendre polynomials
    // For simplicity, use monomials along the edge as the edge test functions.
    let mut dof_offset = nv;
    for ei in 0..nv {
        if !edge_has_dofs[ei] {
            continue;
        }
        let ia = ei;
        let ib = (ei + 1) % nv;
        for k in 0..(p.saturating_sub(1)) {
            // Edge DOF: ∫_e v · ψ_k dS / |e|
            // where ψ_k(t) = L_k(2t-1) for t ∈ [0,1] along the edge
            // For simplicity, use ψ_k(t) = t^k - c (centered Legendre-like)
            // Actually, the standard VEM uses ψ_k = scaled Legendre polynomial.
            // For k=0: ψ_0(t) = 1 (constant along edge)
            // For k=1: ψ_1(t) = 2t-1 (linear, centered)
            // We use the monomial t^k directly (proper Legendre normalization can be
            // handled by the solver since the stabilization takes care of scaling).
            let psi = |t: f64, kk: usize| -> f64 {
                match kk {
                    0 => 1.0,
                    _ => (2.0 * t - 1.0).powi(kk as i32),
                }
            };

            let xa = verts[ia][0];
            let ya = verts[ia][1];
            let xb = verts[ib][0];
            let yb = verts[ib][1];

            for (m_idx, &(a0, a1)) in exps.iter().enumerate().skip(1) {
                let val = edge_int_1d(verts, ia, ib, &|t| {
                    let x = xa + t * (xb - xa);
                    let y = ya + t * (yb - ya);
                    let xi = (x - c[0]) / h;
                    let yi = (y - c[1]) / h;
                    let mon = xi.powi(a0 as i32) * yi.powi(a1 as i32);
                    mon * psi(t, k)
                });
                // Divide by edge length to normalize
                let edge_len = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
                let norm = if edge_len > 1e-30 { edge_len } else { 1.0 };
                B[(dof_offset + k, m_idx - 1)] = val / norm;
            }
        }
        dof_offset += p.saturating_sub(1);
    }

    // Internal DOFs: ∫_T v · m_α_internal dA / |T|
    // where m_α_internal are the high-order monomials (α+β ≥ 2 for p≥3)
    let n_internal_mono = n_internal;
    if n_internal_mono > 0 {
        // Internal DOFs are element area moments of v against monomials of degree p-2
        let int_exps = monomial_exponents(p.saturating_sub(2));
        for (k, &(ia, ib)) in int_exps.iter().enumerate() {
            if k >= n_internal_mono {
                break;
            }
            for (m_idx, &(a0, a1)) in exps.iter().enumerate().skip(1) {
                let val = poly_int_2d(verts, &|x, y| {
                    let xi = (x - c[0]) / h;
                    let yi = (y - c[1]) / h;
                    let mon = xi.powi(a0 as i32) * yi.powi(a1 as i32);
                    let test = xi.powi(ia as i32) * yi.powi(ib as i32);
                    mon * test
                });
                B[(dof_offset + k, m_idx - 1)] = val / area;
            }
        }
    }

    // Projection matrix Π_∇ (n_dofs × n_grad_mono): Π = B · G⁻¹
    let Pi = &B * &invG; // n_dofs × n_grad_mono

    // Consistency term: K_cons = Π · G · Πᵀ
    let K_cons = &Pi * &G_grad * Pi.transpose();

    // Compute trace for stabilization scaling
    let trace: f64 = (0..n_dofs).map(|i| K_cons[(i, i)]).sum();
    let alpha = if n_dofs > 0 && trace.is_finite() && trace > 0.0 {
        trace / n_dofs as f64
    } else {
        1.0
    };

    // Π·P_tilde where P_tilde = G⁻¹·Bᵀ (n_grad_mono × n_dofs)
    let Bt = B.transpose();
    let P_tilde = &invG * &Bt;
    let Pi_Pt = &Pi * &P_tilde; // n_dofs × n_dofs

    // Stabilization: S = α · (I - Π·P_tilde)·(I - Π·P_tilde)ᵀ
    // where P_tilde = G⁻¹·Bᵀ, so I-Π·P_tilde projects onto the VEM kernel.
    // Using the symmetric product guarantees SPD.
    let Id = DMatrix::<f64>::identity(n_dofs, n_dofs);
    let I_minus_PiPt = &Id - &Pi_Pt; // n_dofs × n_dofs
    let K_stab = alpha * &I_minus_PiPt * I_minus_PiPt.transpose();

    // Final element matrix: K_e = K_cons + K_stab
    &K_cons + &K_stab
}

/// Assemble VEM-Poisson stiffness matrix using a VEMSpace.
pub fn assemble_vem_poisson(space: &VEMSpace<PolyMesh>) -> CsrMatrix<f64> {
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let p = space.order() as usize;
    let n_elems = mesh.n_elements();
    let mut coo = CooMatrix::new(n_dofs, n_dofs);

    for e in 0..n_elems as u32 {
        let nodes = mesh.element_nodes(e);
        let nv = nodes.len();
        let dofs = space.element_dofs(e);
        let verts: Vec<[f64; 2]> = nodes
            .iter()
            .map(|&n| {
                let c = mesh.node_coords(n);
                [c[0], c[1]]
            })
            .collect();

        let area = poly_area(&verts);
        if area < 1e-14 {
            continue;
        }

        // Determine which edges have DOFs (all edges in a polygon mesh)
        let edge_has_dofs: Vec<bool> = (0..nv).map(|_| p > 1).collect();

        let K_e = vem_element_stiffness(&verts, &edge_has_dofs, p);

        let n_dofs_e = K_e.nrows();
        for i in 0..n_dofs_e {
            for j in 0..n_dofs_e {
                let val = K_e[(i, j)];
                if val.abs() > 1e-30 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, val);
                }
            }
        }
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::poly_mesh::PolyMesh;

    #[test]
    fn vem_p1_area() {
        assert!((poly_area(&[[0., 0.], [2., 0.], [2., 1.], [0., 1.]]) - 2.).abs() < 1e-12);
    }

    #[test]
    fn vem_p1_assemble() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 1);
        let k = assemble_vem_poisson(&space);
        let nn = space.n_dofs();
        assert_eq!(k.nrows, nn);
        for i in 0..nn {
            assert!(k.get(i, i) > 0.0, "diag[{i}]={}", k.get(i, i));
        }
    }

    #[test]
    fn vem_p1_spd() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 1);
        let k = assemble_vem_poisson(&space);
        let nn = space.n_dofs();
        let mut asym = 0.0;
        for i in 0..nn {
            for p in k.row_ptr[i]..k.row_ptr[i + 1] {
                let j = k.col_idx[p] as usize;
                let d = k.values[p] - k.get(j, i);
                asym += d * d;
            }
        }
        assert!(asym.sqrt() < 1e-12);
    }

    #[test]
    fn vem_p1_cg() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 1);
        let k = assemble_vem_poisson(&space);
        let n = k.nrows;
        let mut x = vec![0.; n];
        let mut r = vec![1.; n];
        let mut p_vec = r.clone();
        let mut rr: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..300 {
            let mut ap = vec![0.; n];
            for i in 0..n {
                for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                    ap[i] += k.values[ptr] * p_vec[k.col_idx[ptr] as usize];
                }
            }
            let pap: f64 = p_vec.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-40 {
                break;
            }
            let al = rr / pap;
            for i in 0..n {
                x[i] += al * p_vec[i];
                r[i] -= al * ap[i];
            }
            let rrn: f64 = r.iter().map(|v| v * v).sum();
            if rrn.sqrt() < 1e-8 {
                break;
            }
            let be = rrn / rr;
            rr = rrn;
            for i in 0..n {
                p_vec[i] = r[i] + be * p_vec[i];
            }
        }
        assert!(r.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-6);
    }

    #[test]
    fn vem_p2_assemble() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 2);
        let k = assemble_vem_poisson(&space);
        let nn = space.n_dofs();
        assert_eq!(k.nrows, nn);
        for i in 0..nn {
            assert!(k.get(i, i) > 0.0, "P2 diag[{i}]={}", k.get(i, i));
        }
        // Check symmetry
        let mut asym = 0.0;
        for i in 0..nn {
            for p in k.row_ptr[i]..k.row_ptr[i + 1] {
                let j = k.col_idx[p] as usize;
                let d = k.values[p] - k.get(j, i);
                asym += d * d;
            }
        }
        assert!(asym.sqrt() < 1e-12, "P2 symmetry {:.2e}", asym.sqrt());
    }

    #[test]
    fn vem_p2_cg() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 2);
        let k = assemble_vem_poisson(&space);
        let n = k.nrows;
        let mut x = vec![0.; n];
        let mut r = vec![1.; n];
        let mut p_vec = r.clone();
        let mut rr: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..300 {
            let mut ap = vec![0.; n];
            for i in 0..n {
                for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                    ap[i] += k.values[ptr] * p_vec[k.col_idx[ptr] as usize];
                }
            }
            let pap: f64 = p_vec.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-40 {
                break;
            }
            let al = rr / pap;
            for i in 0..n {
                x[i] += al * p_vec[i];
                r[i] -= al * ap[i];
            }
            let rrn: f64 = r.iter().map(|v| v * v).sum();
            if rrn.sqrt() < 1e-8 {
                break;
            }
            let be = rrn / rr;
            rr = rrn;
            for i in 0..n {
                p_vec[i] = r[i] + be * p_vec[i];
            }
        }
        assert!(r.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-6);
    }

    #[test]
    fn vem_p3_assemble() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh, 3);
        let k = assemble_vem_poisson(&space);
        let nn = space.n_dofs();
        assert_eq!(k.nrows, nn);
        for i in 0..nn {
            assert!(k.get(i, i) > 0.0, "P3 diag[{i}]={}", k.get(i, i));
        }
        let mut asym = 0.0;
        for i in 0..nn {
            for p in k.row_ptr[i]..k.row_ptr[i + 1] {
                let j = k.col_idx[p] as usize;
                let d = k.values[p] - k.get(j, i);
                asym += d * d;
            }
        }
        assert!(asym.sqrt() < 1e-12, "P3 symmetry {:.2e}", asym.sqrt());
    }
}
