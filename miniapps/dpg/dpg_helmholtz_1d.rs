//! True ultraweak DPG solver for `-u'' = f` on (0,1), u(0)=u(1)=0.
//!
//! 1D companion of MFEM's `miniapps/dpg/diffusion.cpp` (the MFEM DPG weak
//! form applies to 1D as well; fem-rs's shared DPG kernel in
//! `fem_assembly::dpg_weakform` currently covers 2D/3D geometries, so this
//! example writes the element loop out directly — the *method* is the same
//! true ultraweak DPG: broken L2 trial fields, skeleton traces, enriched
//! broken tests, element-wise `G = LLᵀ` inversion, `A = BᵀG⁻¹B` assembly).
//!
//! First-order system:  σ − u' = 0,  −σ' = f, with traces û = u|Γ,
//! σ̂ = σ|Γ.  Trial: u ∈ P1 (nodal), σ ∈ P0 broken, û ∈ P1 skeleton,
//! σ̂ ∈ P0 skeleton.  Tests: τ ∈ P1, v ∈ P1 (broken), rows
//! ```text
//!     (σ, τ)  + (u, τ') − <û, τ n> = 0        ∀ τ
//!     −(σ, v') − <σ̂, v>            = −(f, v)  ∀ v
//! ```
//! Manufactured solution u = sin(πx) (f = π² sin(πx)).

const QP: [f64; 2] = [0.2113248654051871, 0.7886751345948129];
const QW: [f64; 2] = [0.5, 0.5];
const PI: f64 = std::f64::consts::PI;

fn main() {
    let n_elem: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let h = 1.0 / n_elem as f64;

    // Layout: u: n_elem+1 nodal; sigma: n_elem (P0); û: n_elem+1; σ̂: n_elem+1.
    let n_u = n_elem + 1;
    let n_sig = n_elem;
    let n_hatu = n_elem + 1;
    let n_hatsig = n_elem + 1;
    let off_u = 0;
    let off_s = n_u;
    let off_hu = off_s + n_sig;
    let off_hs = off_hu + n_hatu;
    let n_total = off_hs + n_hatsig;

    let mut a_dense = vec![0.0_f64; n_total * n_total];
    let mut rhs = vec![0.0_f64; n_total];

    // P1 basis on [0,1]: (1-x), x; derivative (-1, 1).
    for e in 0..n_elem {
        let n_te = 4usize; // tau(2) + v(2)
        let mut g = vec![0.0_f64; n_te * n_te];
        let mut bmat = vec![0.0_f64; n_te * 6]; // cols: sigma, u0, u1, huL, huR, hs
        let mut fvec = vec![0.0_f64; n_te];

        for q in 0..2 {
            let xq = QP[q];
            let w = QW[q] * h;
            let xm = xq; // local coordinate
            // tau = v = P1 here; phi = (1-x, x), dphi = (-1, 1)
            let phi = [1.0 - xm, xm];
            let dphi = [-1.0, 1.0];
            // tau rows: (sigma, tau): sigma P0 value on element = const (col 0),
            // but sigma is a P0 unknown: B[tau_i, sigma_e] = int tau_i dx
            for i in 0..2 {
                g[i * n_te + i] += w; // (tau_i, tau_i) graph-norm mass on diag
                for j in 0..2 {
                    // tau graph norm: (tau', tau') too
                    g[i * n_te + j] += w * dphi[i] * dphi[j];
                    // v rows are also P1: v graph norm (v',v')+(v,v) sits in
                    // rows 2..4, cols 2..4
                    g[(2 + i) * n_te + (2 + j)] += w * (dphi[i] * dphi[j] + phi[i] * phi[j]);
                }
                // B[tau_i, sigma] = int tau_i dx
                bmat[i * 6 + 0] += w * phi[i];
                // B[tau_i, u_j] = int u_j tau_i' dx = dphi_i * int phi_j dx
                bmat[i * 6 + 1] += dphi[i] * h * 0.5;
                bmat[i * 6 + 2] += dphi[i] * h * 0.5;
                // B[v_i, sigma] = -int v_i' dx = -(v_i(1)-v_i(0))
            }
            // v rows: -(sigma, v') -> -(dphi_i) * sigma const: B[v_i, sigma] = -int v_i'
            for i in 0..2 {
                let row = 2 + i;
                bmat[row * 6 + 0] += -w * dphi[i];
                // RHS -(f, v) assembled with f = pi^2 sin(pi x)
                let x_phys = e as f64 * h + xm * h;
                let fv = PI * PI * (PI * x_phys).sin();
                fvec[row] += w * fv * phi[i];
            }
        }

        // Trace terms (point evaluations at the two faces):
        // B[tau_i, huL] = +tau_i(0), B[tau_i, huR] = -tau_i(1)
        // B[v_i, hsL]  = -v_i(0),    B[v_i, hsR]  = +v_i(1)
        let tau0 = [1.0, 0.0];
        let tau1 = [0.0, 1.0];
        for i in 0..2 {
            bmat[i * 6 + 3] += tau0[i];
            bmat[i * 6 + 4] += -tau1[i];
            bmat[(2 + i) * 6 + 4] += phi_right(i);
            bmat[(2 + i) * 6 + 5] += -phi_right(i);
        }

        // Cholesky of G, Y = L⁻¹B, y = L⁻¹f, A_e = YᵀY, b_e = Yᵀy.
        let mut l = g.clone();
        for j in 0..n_te {
            let mut d = l[j * n_te + j];
            for k in 0..j {
                d -= l[j * n_te + k] * l[j * n_te + k];
            }
            d = d.sqrt();
            l[j * n_te + j] = d;
            for i in (j + 1)..n_te {
                let mut s = l[i * n_te + j];
                for k in 0..j {
                    s -= l[i * n_te + k] * l[j * n_te + k];
                }
                l[i * n_te + j] = s / d;
            }
        }
        let mut yb = bmat.clone();
        for i in 0..n_te {
            for kk in 0..6 {
                let mut s = yb[i * 6 + kk];
                for j in 0..i {
                    s -= l[i * n_te + j] * yb[j * 6 + kk];
                }
                yb[i * 6 + kk] = s / l[i * n_te + i];
            }
        }
        for i in 0..n_te {
            let mut s = fvec[i];
            for j in 0..i {
                s -= l[i * n_te + j] * fvec[j];
            }
            fvec[i] = s / l[i * n_te + i];
        }
        let mut a_e = vec![0.0_f64; 6 * 6];
        let mut b_e = vec![0.0_f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mut s = 0.0;
                for k in 0..n_te {
                    s += yb[k * 6 + i] * yb[k * 6 + j];
                }
                a_e[i * 6 + j] = s;
            }
            let mut s = 0.0;
            for k in 0..n_te {
                s += yb[k * 6 + i] * fvec[k];
            }
            b_e[i] = s;
        }

        // Scatter: cols/rows = sigma_e, u(e), u(e+1), hu(e), hu(e+1), hs(e).
        let vdofs = [
            off_s + e,
            off_u + e,
            off_u + e + 1,
            off_hu + e,
            off_hu + e + 1,
            off_hs + e,
        ];
        for (li, &gd) in vdofs.iter().enumerate() {
            rhs[gd] += b_e[li];
            for (lj, &gd2) in vdofs.iter().enumerate() {
                a_dense[gd * n_total + gd2] += a_e[li * 6 + lj];
            }
        }
    }

    // Essential BCs: û = 0 at x=0,1; σ̂ left free (natural).
    let ess = [off_hu, off_hu + n_hatu - 1];
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
            for k in 0..n_total {
                a_dense.swap(c * n_total + k, best * n_total + k);
            }
            rhs.swap(c, best);
        }
        let p = a_dense[c * n_total + c];
        for r in (c + 1)..n_total {
            let f = a_dense[r * n_total + c] / p;
            a_dense[r * n_total + c] = f;
            for k in (c + 1)..n_total {
                a_dense[r * n_total + k] -= f * a_dense[c * n_total + k];
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

    // L2 error of u (P1 interpolation of the nodal values).
    let mut err2 = 0.0;
    for e in 0..n_elem {
        for q in 0..2 {
            let x = (e as f64 + QP[q]) * h;
            let xl = QP[q];
            let uh = rhs[off_u + e] * (1.0 - xl) + rhs[off_u + e + 1] * xl;
            let ue = (PI * x).sin();
            err2 += QW[q] * h * (uh - ue) * (uh - ue);
        }
    }
    println!("True 1D ultraweak DPG for -u'' = f (u = sin(pi x))");
    println!("  elements = {n_elem}, L2 error = {:.3e}", err2.sqrt());
}

fn phi_right(i: usize) -> f64 {
    if i == 1 {
        1.0
    } else {
        0.0
    }
}
