//! True ultraweak DPG solver for the 1-D Poisson/Helmholtz problem
//! `-u'' - k² u = f` on (0,1), u(0)=u(1)=0 (k=0 for Poisson).
//!
//! 1-D translation of MFEM's `miniapps/dpg/diffusion.cpp` (the MFEM DPG weak
//! form machinery covers all dims; fem-rs's shared DPG kernel in
//! `fem_assembly::dpg_weakform` covers 2-D/3-D geometries, so this example
//! writes the element loop out directly — the *method* is the same true
//! ultraweak DPG).  First-order system:  σ − u' = 0,  −σ' − k² u = f, with
//! traces û = u|Γ, σ̂ = σ|Γ.
//!
//! Trial spaces (order 1, cf. diffusion.cpp): u ∈ P0 broken, σ ∈ P0 broken,
//! û ∈ P1 skeleton, σ̂ ∈ P0 skeleton.  Test spaces (order+1 enrichment):
//! τ ∈ P1 broken (1-D `RT_{test_order-1}`), v ∈ P2 broken (`H¹_{test_order}`).
//! Element rows (n = (−1, +1) the outward normals):
//! ```text
//!     −(u, τ') − (σ, τ) + <û, τ n>              = 0        ∀ τ ∈ P1
//!     (σ, v') − <σ̂, v n> − k² (u, v)            = (f, v)   ∀ v ∈ P2
//! ```
//! Test (space-induced) norm: `(τ,τ) + (τ',τ') + (v,v) + (v',v')`.
//!
//! Manufactured solution u = sin(πx) (f = (π² − k²) sin(πx)).

const PI: f64 = std::f64::consts::PI;

/// 3-point Gauss-Legendre on [0,1] (exact through degree 5 — P2×P2 products).
const QP: [f64; 3] = [0.5 * (1.0 - 0.7745966692414834), 0.5, 0.5 * (1.0 + 0.7745966692414834)];
const QW: [f64; 3] = [5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

/// P2 basis on [0,1] (nodes 0, 1/2, 1) and its ξ-derivative.
fn p2(xi: f64) -> [f64; 3] {
    [(1.0 - xi) * (1.0 - 2.0 * xi), 4.0 * xi * (1.0 - xi), xi * (2.0 * xi - 1.0)]
}
fn p2_d(xi: f64) -> [f64; 3] {
    [-3.0 + 4.0 * xi, 4.0 - 8.0 * xi, -1.0 + 4.0 * xi]
}

fn main() {
    let n_elem: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let k: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let h = 1.0 / n_elem as f64;

    // Layout: u: n (P0 broken), sigma: n (P0), û: n+1, σ̂: n+1.
    let n_u = n_elem;
    let off_u = 0;
    let off_s = off_u + n_u;
    let off_hu = off_s + n_u;
    let off_hs = off_hu + n_elem + 1;
    let n_total = off_hs + n_elem + 1;

    let mut a_dense = vec![0.0_f64; n_total * n_total];
    let mut rhs = vec![0.0_f64; n_total];

    for e in 0..n_elem {
        // Tests: tau ∈ P1 (2 rows), v ∈ P2 (3 rows).
        let n_te = 5usize;
        let mut g = vec![0.0_f64; n_te * n_te];
        // cols: u_e, σ_e, û_e, û_{e+1}, σ̂_e, σ̂_{e+1}
        let n_co = 6usize;
        let mut bmat = vec![0.0_f64; n_te * n_co];
        let mut fvec = vec![0.0_f64; n_te];

        // P1 (τ) values / derivatives: τ = (1-ξ, ξ), dτ/dξ = (-1, 1);
        // ∫ τ' dx = dτ (the h factors cancel in the ∫·dx terms).
        let dtau = [-1.0_f64, 1.0_f64];

        for q in 0..3 {
            let xi = QP[q];
            let w = QW[q] * h; // physical weight
            let tau = [1.0 - xi, xi];
            let psi = p2(xi);
            let dpsi = p2_d(xi); // dψ/dξ; dψ/dx = dψ/ξ / h

            // Test norm blocks: (τ,τ) + (τ',τ') rows 0..2,
            // (v,v) + (v',v') rows 2..5.
            for i in 0..2 {
                for j in 0..2 {
                    g[i * n_te + j] += w * tau[i] * tau[j]
                        + QW[q] / h * dtau[i] * dtau[j];
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    g[(2 + i) * n_te + (2 + j)] += w * psi[i] * psi[j]
                        + QW[q] / h * dpsi[i] * dpsi[j];
                }
            }

            // Row τ: −(u, τ') − (σ, τ)   (traces added below).
            //   −(u, τ_i') = −u_e ∫ τ_i' dx = −u_e dτ_i, per point −QW dτ_i
            //   −(σ, τ_i)  = −σ_e ∫ τ_i dx, per point −w τ_i
            for i in 0..2 {
                bmat[i * n_co] += -QW[q] * dtau[i];
                bmat[i * n_co + 1] += -w * tau[i];
            }
            // Row v: (σ, v') − k² (u, v)   (traces added below).
            //   (σ, v_j') = σ_e ∫ v_j' dx, per point QW dψ_j
            //   −k² (u, v_j) = −k² u_e ∫ v_j dx, per point −k² w ψ_j
            for j in 0..3 {
                bmat[(2 + j) * n_co + 1] += QW[q] * dpsi[j];
                bmat[(2 + j) * n_co] += -k * k * w * psi[j];
                // RHS (f, v_j), f = (π² − k²) sin(πx).
                let x_phys = (e as f64 + xi) * h;
                fvec[2 + j] += w * (PI * PI - k * k) * (PI * x_phys).sin() * psi[j];
            }
        }

        // Trace terms (point evaluations, outward normals n = (−1, +1)):
        //   row τ: +<û, τ n>  →  B[τ_i, û_L] = −τ_i(0), B[τ_i, û_R] = +τ_i(1)
        //   row v: −<σ̂, v n>  →  B[v_j, σ̂_L] = +v_j(0), B[v_j, σ̂_R] = −v_j(1)
        for i in 0..2 {
            let t0 = if i == 0 { 1.0 } else { 0.0 };
            let t1 = if i == 1 { 1.0 } else { 0.0 };
            bmat[i * n_co + 2] += -t0;
            bmat[i * n_co + 3] += t1;
        }
        for j in 0..3 {
            let v0 = p2(0.0)[j];
            let v1 = p2(1.0)[j];
            bmat[(2 + j) * n_co + 4] += v0;
            bmat[(2 + j) * n_co + 5] += -v1;
        }

        // Cholesky of G, Y = L⁻¹B, y = L⁻¹f, A_e = YᵀY, b_e = Yᵀy.
        let mut l = g.clone();
        for j in 0..n_te {
            let mut d = l[j * n_te + j];
            for kk in 0..j {
                d -= l[j * n_te + kk] * l[j * n_te + kk];
            }
            d = d.sqrt();
            l[j * n_te + j] = d;
            for i in (j + 1)..n_te {
                let mut s = l[i * n_te + j];
                for kk in 0..j {
                    s -= l[i * n_te + kk] * l[j * n_te + kk];
                }
                l[i * n_te + j] = s / d;
            }
        }
        let mut yb = bmat.clone();
        for i in 0..n_te {
            for c in 0..n_co {
                let mut s = yb[i * n_co + c];
                for j in 0..i {
                    s -= l[i * n_te + j] * yb[j * n_co + c];
                }
                yb[i * n_co + c] = s / l[i * n_te + i];
            }
        }
        for i in 0..n_te {
            let mut s = fvec[i];
            for j in 0..i {
                s -= l[i * n_te + j] * fvec[j];
            }
            fvec[i] = s / l[i * n_te + i];
        }
        let mut a_e = vec![0.0_f64; n_co * n_co];
        let mut b_e = vec![0.0_f64; n_co];
        for i in 0..n_co {
            for j in 0..n_co {
                let mut s = 0.0;
                for kk in 0..n_te {
                    s += yb[kk * n_co + i] * yb[kk * n_co + j];
                }
                a_e[i * n_co + j] = s;
            }
            let mut s = 0.0;
            for kk in 0..n_te {
                s += yb[kk * n_co + i] * fvec[kk];
            }
            b_e[i] = s;
        }

        // Scatter.
        let vdofs = [
            off_u + e,
            off_s + e,
            off_hu + e,
            off_hu + e + 1,
            off_hs + e,
            off_hs + e + 1,
        ];
        for (li, &gd) in vdofs.iter().enumerate() {
            rhs[gd] += b_e[li];
            for (lj, &gd2) in vdofs.iter().enumerate() {
                a_dense[gd * n_total + gd2] += a_e[li * n_co + lj];
            }
        }
    }

    // Essential BCs: û = 0 at x=0,1 (u(0)=u(1)=0); σ̂ left free (natural).
    let ess = [off_hu, off_hu + n_elem];
    for &d in &ess {
        for j in 0..n_total {
            a_dense[d * n_total + j] = 0.0;
            a_dense[j * n_total + d] = 0.0;
        }
        a_dense[d * n_total + d] = 1.0;
        rhs[d] = 0.0;
    }

    // Dense LU solve.
    for c in 0..n_total {
        let mut best = c;
        for r in c..n_total {
            if a_dense[r * n_total + c].abs() > a_dense[best * n_total + c].abs() {
                best = r;
            }
        }
        if best != c {
            for kk in 0..n_total {
                a_dense.swap(c * n_total + kk, best * n_total + kk);
            }
            rhs.swap(c, best);
        }
        let p = a_dense[c * n_total + c];
        for r in (c + 1)..n_total {
            let f = a_dense[r * n_total + c] / p;
            a_dense[r * n_total + c] = f;
            for kk in (c + 1)..n_total {
                a_dense[r * n_total + kk] -= f * a_dense[c * n_total + kk];
            }
        }
    }
    for i in 0..n_total {
        for r in (i + 1)..n_total {
            rhs[r] -= a_dense[r * n_total + i] * rhs[i];
        }
    }
    for i in (0..n_total).rev() {
        for j in (i + 1)..n_total {
            rhs[i] -= a_dense[i * n_total + j] * rhs[j];
        }
        rhs[i] /= a_dense[i * n_total + i];
    }

    // L2 errors of u and σ (both P0 cellwise constants).
    let mut err2u = 0.0;
    let mut err2s = 0.0;
    for e in 0..n_elem {
        for q in 0..3 {
            let x = (e as f64 + QP[q]) * h;
            let uh = rhs[off_u + e];
            let ue = (PI * x).sin();
            err2u += QW[q] * h * (uh - ue) * (uh - ue);
            let sh = rhs[off_s + e];
            let se = PI * (PI * x).cos();
            err2s += QW[q] * h * (sh - se) * (sh - se);
        }
    }
    println!("True 1D ultraweak DPG for -u'' - k^2 u = f (u = sin(pi x))");
    println!(
        "  elements = {n_elem}, k = {k}, L2 error u = {:.3e}, sigma = {:.3e}",
        err2u.sqrt(),
        err2s.sqrt()
    );
}
